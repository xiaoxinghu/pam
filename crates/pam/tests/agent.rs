//! Typed and spoken requests end to end: the core makes a task, prompts it in
//! the meeting's pi session, and the agent's `pam say` reaches the stage. pi
//! is the `stub-pi` binary (`tests/support/stub_pi.rs`), which answers by
//! running this `pam`; the intent check is [`StubIntent`].

use std::path::{Path, PathBuf};
use std::process::Command as Process;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;
use pam_core::llm::{Completion, Request};
use pam_core::models::{Choice, Pick, Source};
use pam_core::question::{Answer, By, End};
use pam_core::usage::{Call, Used};
use pam_core::{
    Command, CoreEvent, Llm, Meeting, MeetingConfig, ModelStatus, TaskId, TaskState, TaskStatus, Ts, Usage,
    UtteranceId,
};
use serde_json::Value;

/// Sets the meeting up with the stub as pi (its `env` set): no scribe, and
/// `stub/model` for the agent, as if from the config file.
fn use_stub(cfg: &mut MeetingConfig, dir: &Path, env: &str) {
    let choice = Choice {
        pi: Some(stub_pi(dir, env)),
        settings: dir.join("pi-settings.json"),
        scribe: Pick::off(Source::Option),
        intent: Pick::Detect,
        agent: Pick::Named { model: "stub/model".into(), source: Source::Config },
    };
    cfg.use_models(&choice, Some(env!("CARGO_BIN_EXE_pam").into()));
}

/// A `pi` that runs the stub with `env` set, e.g. `STUB_PI_VERSION=1.2.0`.
fn stub_pi(dir: &Path, env: &str) -> PathBuf {
    let pi = dir.join("pi");
    std::fs::write(&pi, format!("#!/bin/sh\n{env} exec '{}' \"$@\"\n", env!("CARGO_BIN_EXE_stub-pi")))
        .unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&pi, std::fs::Permissions::from_mode(0o755)).unwrap();
    pi
}

struct Started {
    meeting: Meeting,
    events: Receiver<CoreEvent>,
    _dir: tempfile::TempDir,
}

/// Replays the stand-up fixture to its end with the stub as pi.
fn start(env: &str) -> Started {
    start_with(env, |_| {})
}

/// [`start`], with `edit` applied to the meeting's config first.
fn start_with(env: &str, edit: impl FnOnce(&mut MeetingConfig)) -> Started {
    let dir = tempfile::tempdir().unwrap();
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/standup.fake.jsonl");
    let mut cfg = MeetingConfig::fake(dir.path().join("meetings"), script).unwrap();
    cfg.title = "Stand-up".into();
    use_stub(&mut cfg, dir.path(), env);
    edit(&mut cfg);
    let meeting = Meeting::start(cfg).unwrap();
    let events = meeting.subscribe();
    wait_for(&events, |e| matches!(e, CoreEvent::InputEnded { .. }));
    Started { meeting, events, _dir: dir }
}

/// The intent check, stubbed: someone "from finance" isn't talking to Pam,
/// "I meant" corrects the latest request, "… can you" is unfinished, and
/// anything else is a request. While a question is on the stage, "the first
/// one" and "the second one" reply to it, and "Auckland" answers an open one.
#[derive(Default)]
struct StubIntent {
    said: Mutex<Vec<String>>,
    /// How long a check with a question on the stage takes.
    slow: Duration,
}

/// What each stubbed intent check costs.
const INTENT_CALL: Call = Call { input: 50, output: 5, cache_read: 0, cache_write: 0, cost: Some(0.0005) };

impl pam_core::llm::Model for StubIntent {
    fn complete(&self, req: &Request<'_>) -> Completion {
        let prompt = req.prompt;
        let said = prompt.rsplit("UTTERANCE\n").next().unwrap().trim().to_string();
        self.said.lock().unwrap().push(said.clone());
        let question = prompt.contains("\nQUESTION ON THE STAGE\n");
        let open = prompt.contains("\n(open: answered in words)\n");
        if question {
            std::thread::sleep(self.slow);
        }
        let (kind, answer) = match &said {
            s if open && s.contains("Auckland") => ("reply", r#", "text": "Auckland""#),
            s if question && s.contains("the first one") => ("reply", r#", "option": 1"#),
            s if question && s.contains("the second one") => ("reply", r#", "option": 2"#),
            s if s.contains("from finance") => ("none", ""),
            s if s.contains("I meant") => ("correction", ""),
            s if s.ends_with("can you") => ("unfinished", ""),
            _ => ("request", ""),
        };
        let summary = format!("Summary of: {}", said.split_once(": ").unwrap().1);
        let reply = format!(r#"{{"kind": "{kind}"{answer}, "summary": "{summary}"}}"#);
        Completion { reply: Ok(reply), used: Used::Call(INTENT_CALL) }
    }
}

/// Replays `script` at real-time speed, with the stub as pi (its `env` set)
/// and [`StubIntent`].
fn start_spoken(script: &str, env: &str) -> (Started, Arc<StubIntent>) {
    let intent = Arc::new(StubIntent::default());
    (start_checked_by(script, env, Some(intent.clone())), intent)
}

/// Replays `script` at real-time speed, with the stub as pi (its `env` set)
/// and `intent` as the intent check; with none, as with `--scribe off`.
fn start_checked_by(script: &str, env: &str, intent: Option<Arc<dyn pam_core::llm::Model>>) -> Started {
    let dir = tempfile::tempdir().unwrap();
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures").join(script);
    let mut cfg = MeetingConfig::fake(dir.path().join("meetings"), script).unwrap();
    cfg.realtime = true;
    use_stub(&mut cfg, dir.path(), env);
    cfg.intent = intent.map(|model| Llm { name: "stub/intent".into(), model });
    let meeting = Meeting::start(cfg).unwrap();
    let events = meeting.subscribe();
    Started { meeting, events, _dir: dir }
}

/// Every event up to and including the first that `done` accepts.
fn wait_for(events: &Receiver<CoreEvent>, done: impl Fn(&CoreEvent) -> bool) -> Vec<CoreEvent> {
    let end = Instant::now() + Duration::from_secs(30);
    let mut got = Vec::new();
    loop {
        let left = end.saturating_duration_since(Instant::now());
        let e = events.recv_timeout(left).unwrap_or_else(|_| panic!("timed out; got {got:#?}"));
        if let CoreEvent::Error(e) = &e {
            panic!("{e}");
        }
        let stop = done(&e);
        got.push(e);
        if stop {
            return got;
        }
    }
}

fn updated(id: u32, status: TaskStatus) -> impl Fn(&CoreEvent) -> bool {
    move |e| matches!(e, CoreEvent::TaskUpdated(t) if t.id == TaskId(id) && t.status == status)
}

/// Each task's status as it changed, in order. A task is updated more often
/// than that: when it gets its line, a correction, or its start time.
fn tasks(events: &[CoreEvent]) -> Vec<(TaskId, TaskStatus)> {
    let mut changes: Vec<(TaskId, TaskStatus)> = Vec::new();
    for e in events {
        let CoreEvent::TaskUpdated(TaskState { id, status, .. }) = e else { continue };
        if changes.iter().rev().find(|(t, _)| t == id).is_none_or(|(_, s)| s != status) {
            changes.push((*id, *status));
        }
    }
    changes
}

fn said(events: &[CoreEvent]) -> Vec<(TaskId, &str)> {
    events
        .iter()
        .filter_map(|e| match e {
            CoreEvent::StageMessage { task: Some(id), text, .. } => Some((*id, text.as_str())),
            _ => None,
        })
        .collect()
}

/// The latest `Usage` in `events`.
fn last_usage(events: &[CoreEvent]) -> Usage {
    events
        .iter()
        .rev()
        .find_map(|e| match e {
            CoreEvent::Usage(u) => Some(*u.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("no usage in {events:#?}"))
}

fn json(path: PathBuf) -> Value {
    serde_json::from_str(
        &std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display())),
    )
    .unwrap()
}

/// The prompts the stub pi took, in order, as it recorded them beside the session.
fn prompts(meeting: &Path) -> Vec<Value> {
    let path = meeting.join("agent/stub-pi.jsonl");
    if !path.exists() {
        return Vec::new();
    }
    lines_of(path)
}

/// Waits until the stub pi has taken `n` prompts.
fn prompted(meeting: &Path, n: usize) -> Vec<Value> {
    let end = Instant::now() + Duration::from_secs(10);
    loop {
        let got = prompts(meeting);
        if got.len() >= n {
            return got;
        }
        assert!(Instant::now() < end, "pi took {} prompts, not {n}", got.len());
        std::thread::sleep(Duration::from_millis(20));
    }
}

/// The arguments pi was started with for a prompt.
fn args_of(prompt: &Value) -> Vec<&str> {
    prompt["args"].as_array().unwrap().iter().map(|a| a.as_str().unwrap()).collect()
}

fn pam(meeting: &Path, env: &[(&str, &Path)], args: &[&str]) -> (i32, String, String) {
    pam_in(Path::new(env!("CARGO_MANIFEST_DIR")), meeting, env, args)
}

/// [`pam`], run in `cwd`, as the agent runs it in the meeting folder.
fn pam_in(cwd: &Path, meeting: &Path, env: &[(&str, &Path)], args: &[&str]) -> (i32, String, String) {
    let mut cmd = Process::new(env!("CARGO_BIN_EXE_pam"));
    cmd.current_dir(cwd);
    cmd.args(args).arg("--meeting").arg(meeting).env_remove("PAM_MEETING").env_remove("PI_SESSION_FILE");
    for (k, v) in env {
        cmd.env(k, v);
    }
    let out = cmd.output().unwrap();
    (
        out.status.code().unwrap(),
        String::from_utf8(out.stdout).unwrap(),
        String::from_utf8(out.stderr).unwrap(),
    )
}

#[test]
fn a_typed_request_is_answered_on_the_stage() {
    let s = start("");
    let folder = s.meeting.folder().canonicalize().unwrap();
    s.meeting.send(Command::Ask("  what did we decide?  ".into())).unwrap();
    let got = wait_for(&s.events, updated(1, TaskStatus::Done));
    assert_eq!(tasks(&got), [(TaskId(1), TaskStatus::Running), (TaskId(1), TaskStatus::Done)]);
    assert_eq!(said(&got), [(TaskId(1), "Stub answer to: what did we decide?")]);
    let CoreEvent::TaskUpdated(t) = &got[0] else { panic!("{got:?}") };
    assert_eq!((t.request.as_str(), t.line, t.asked_at), ("what did we decide?", None, 42 * 16_000));

    let task = json(folder.join("tasks/t01/task.json"));
    assert_eq!(task["status"], "done");
    let stub = &prompts(&folder)[0];
    assert_eq!(stub["task"], "t01");
    let args = args_of(stub);
    let session = folder.join("agent/session.jsonl");
    assert_eq!(args[..6], ["--mode", "rpc", "--model", "stub/model", "--session", session.to_str().unwrap()]);
    assert_eq!(args[6], "--append-system-prompt");
    assert!(args[7].contains("pam say"), "{}", args[7]);
    assert_eq!(args[8..], ["-ne", "-ns", "-np", "-nc"]);
    assert_eq!(stub["cwd"], folder.to_str().unwrap());
    assert_eq!(stub["pam_meeting"], folder.to_str().unwrap());
    // context.md, then the last minute word for word, then the request.
    let prompt = stub["prompt"].as_str().unwrap();
    let context = prompt.find("<context.md>\n# Meeting: Stand-up").expect(prompt);
    let recent = prompt.find("<last-60-seconds>\nu0001 00:01 Alice: Okay, let's get started.").expect(prompt);
    let request = prompt
        .find("Request t01 (typed on the meeting laptop at 00:42):\nwhat did we decide?")
        .expect(prompt);
    assert!(context < recent && recent < request, "{prompt}");

    let info = json(folder.join("meeting.json"));
    assert_eq!(info["agent"]["status"], "ready");
    assert_eq!(info["agent"]["version"], "1.1.0");
    assert_eq!(info["agent"]["model"], "stub/model");
    assert!(info["agent"]["pi"].as_str().unwrap().ends_with("/pi"));
    let (code, out, err) = pam(&folder, &[], &["status"]);
    assert_eq!(code, 0, "{err}");
    assert!(
        out.contains("scribe  off\nintent  off\nagent   ready (pi 1.1.0, stub/model, config)\n"),
        "{out}"
    );
    s.meeting.stop().unwrap();
}

#[test]
fn a_second_request_waits_and_stopping_the_meeting_stops_both() {
    let s = start("");
    s.meeting.send(Command::Ask("a slow one".into())).unwrap();
    s.meeting.send(Command::Ask("then this".into())).unwrap();
    let mut got = wait_for(&s.events, updated(2, TaskStatus::Queued));
    // The first stays slow until the meeting stops it; queued_requests_run_in_order checks the order.
    assert_eq!(tasks(&got), [(TaskId(1), TaskStatus::Running), (TaskId(2), TaskStatus::Queued)]);
    let folder = s.meeting.folder().to_path_buf();
    let pid = prompted(&folder, 1)[0]["pid"].as_i64().unwrap() as i32;
    got.extend(s.events.try_iter());
    assert!(said(&got).is_empty(), "{got:?}");

    s.meeting.stop().unwrap();
    // Stopping the meeting stopped the running task's pi, and the queued one never ran.
    assert_eq!(json(folder.join("tasks/t01/task.json"))["status"], "stopped");
    assert_eq!(json(folder.join("tasks/t02/task.json"))["status"], "stopped");
    let asked: Vec<Value> = prompts(&folder).into_iter().map(|p| p["task"].clone()).collect();
    assert_eq!(asked, ["t01"], "t02 was never prompted");
    // SAFETY: signal 0 only asks whether the process exists.
    assert_ne!(unsafe { libc::kill(pid, 0) }, 0, "pi {pid} is still running");
    // pi never reported the call it was making.
    let usage = &json(folder.join("meeting.json"))["usage"];
    assert_eq!((&usage["agent"]["calls"], &usage["partial"]), (&0.into(), &true.into()));
}

#[test]
fn every_change_of_status_is_logged_and_shown_in_the_notes_and_the_context_pack() {
    let s = start("");
    let folder = s.meeting.folder().to_path_buf();
    s.meeting.send(Command::Ask("what did we decide?".into())).unwrap();
    wait_for(&s.events, updated(1, TaskStatus::Done));
    s.meeting.send(Command::Ask("a slow one".into())).unwrap();
    s.meeting.send(Command::Ask("then this".into())).unwrap();
    wait_for(&s.events, updated(3, TaskStatus::Queued));
    let notes = std::fs::read_to_string(folder.join("notes.md")).unwrap();
    assert!(
        notes.ends_with(
            "## Action items\n\n\
             - [x] what did we decide? · done\n\
             - [ ] a slow one · running\n\
             - [ ] then this · queued\n"
        ),
        "{notes}"
    );
    let context = std::fs::read_to_string(folder.join("context.md")).unwrap();
    assert!(
        context.contains(
            "## Tasks\n\n\
             - t01 00:42 typed: what did we decide? · done\n\
             - t02 00:42 typed: a slow one · running\n\
             - t03 00:42 typed: then this · queued\n"
        ),
        "{context}"
    );
    let (_, status, _) = pam(&folder, &[], &["status"]);
    assert!(status.contains("tasks   3 (1 running, 1 queued, 0 ready)\n  t02 "), "{status}");

    s.meeting.stop().unwrap();
    let events: Vec<(String, String, String)> = lines_of(folder.join("events.jsonl"))
        .into_iter()
        .map(|e| {
            let field = |k: &str| e[k].as_str().unwrap().to_string();
            (field("type"), field("task"), field("text"))
        })
        .collect();
    // Each task's in order; a prompt goes out on the agent's thread, so
    // t02's start may come before or after t03's request.
    let of = |task: &str| -> Vec<(String, String)> {
        events.iter().filter(|e| e.1 == task).map(|e| (e.0.clone(), e.2.clone())).collect()
    };
    let row = |kind: &str, text: &str| (kind.to_string(), text.to_string());
    let stopped = "stopped: the meeting stopped first";
    assert_eq!(
        of("t01"),
        [row("request", "what did we decide?"), row("task_update", "started"), row("task_update", "done")]
    );
    assert_eq!(
        of("t02"),
        [row("request", "a slow one"), row("task_update", "started"), row("task_update", stopped)]
    );
    assert_eq!(of("t03"), [row("request", "then this"), row("task_update", stopped)]);
    // When each ran, in meeting time: the replay had ended at 00:42.
    let t = |id: &str| json(folder.join(format!("tasks/{id}/task.json")));
    let (t1, t3) = (t("t01"), t("t03"));
    assert!(t1["started_at"].as_u64().unwrap() <= t1["ended_at"].as_u64().unwrap(), "{t1}");
    assert_eq!((&t3["status"], t3.get("started_at")), (&"stopped".into(), None), "t03 never ran: {t3}");
    assert!(t3["ended_at"].is_u64(), "{t3}");
    let notes = std::fs::read_to_string(folder.join("notes.md")).unwrap();
    assert!(notes.ends_with("- [ ] a slow one · stopped\n- [ ] then this · stopped\n"), "{notes}");
}

#[test]
fn a_task_whose_prompt_never_goes_out_never_started() {
    let s = start("");
    // pi passed its checks, then went away: the task's run can't start it.
    std::fs::remove_file(s._dir.path().join("pi")).unwrap();
    s.meeting.send(Command::Ask("what did we decide?".into())).unwrap();
    wait_for(&s.events, updated(1, TaskStatus::Failed));
    let folder = s.meeting.folder().to_path_buf();
    s.meeting.stop().unwrap();
    let t1 = json(folder.join("tasks/t01/task.json"));
    assert_eq!(t1.get("started_at"), None, "{t1}");
    assert!(t1["ended_at"].is_u64(), "{t1}");
    let updates: Vec<String> = lines_of(folder.join("events.jsonl"))
        .into_iter()
        .filter(|e| e["type"] == "task_update")
        .map(|e| e["text"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(updates.len(), 1, "{updates:?}");
    assert!(updates[0].starts_with("failed: "), "{updates:?}");
}

#[test]
fn a_prompt_that_cannot_be_made_never_starts_its_task() {
    let s = start("");
    let folder = s.meeting.folder().to_path_buf();
    // A folder where context.md should be: the first prompt can't be put
    // together, and nothing is sent. Rendering it fails too, so errors are expected.
    std::fs::remove_file(folder.join("context.md")).unwrap();
    std::fs::create_dir_all(folder.join("context.md/blocked")).unwrap();
    s.meeting.send(Command::Ask("what did we decide?".into())).unwrap();
    let end = Instant::now() + Duration::from_secs(30);
    loop {
        let e = s.events.recv_timeout(end.saturating_duration_since(Instant::now())).expect("never failed");
        if updated(1, TaskStatus::Failed)(&e) {
            break;
        }
    }
    s.meeting.stop().unwrap();
    assert!(prompts(&folder).is_empty(), "pi was never prompted");
    let t1 = json(folder.join("tasks/t01/task.json"));
    assert!(t1["error"].as_str().unwrap().contains("context.md"), "{t1}");
    assert_eq!(t1.get("started_at"), None, "{t1}");
    let updates: Vec<String> = lines_of(folder.join("events.jsonl"))
        .into_iter()
        .filter(|e| e["type"] == "task_update")
        .map(|e| e["text"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(updates.len(), 1, "{updates:?}");
    assert!(updates[0].starts_with("failed: "), "{updates:?}");
}

#[test]
fn queued_requests_run_in_order() {
    let s = start("");
    s.meeting.send(Command::Ask("first".into())).unwrap();
    s.meeting.send(Command::Ask("second".into())).unwrap();
    let got = wait_for(&s.events, updated(2, TaskStatus::Done));
    let order = tasks(&got);
    let done1 = order.iter().position(|t| *t == (TaskId(1), TaskStatus::Done)).unwrap();
    let run2 = order.iter().position(|t| *t == (TaskId(2), TaskStatus::Running)).unwrap();
    assert!(done1 < run2, "{order:?}");
    assert_eq!(said(&got), [(TaskId(1), "Stub answer to: first"), (TaskId(2), "Stub answer to: second")]);
    s.meeting.stop().unwrap();
}

#[test]
fn an_agent_that_only_replies_in_text_has_its_reply_shown() {
    let s = start("");
    s.meeting.send(Command::Ask("stay silent".into())).unwrap();
    let got = wait_for(&s.events, updated(1, TaskStatus::Done));
    assert_eq!(said(&got), [(TaskId(1), "Reply in text to: stay silent")]);
    let saved = std::fs::read_to_string(s.meeting.folder().join("tasks/t01/messages.jsonl")).unwrap();
    assert!(saved.contains("Reply in text to: stay silent"), "{saved}");
    s.meeting.stop().unwrap();
}

#[test]
fn a_failed_run_says_why() {
    let s = start("");
    s.meeting.send(Command::Ask("please fail".into())).unwrap();
    let got = wait_for(&s.events, updated(1, TaskStatus::Failed));
    let Some(CoreEvent::TaskUpdated(t)) = got.last() else { panic!("{got:?}") };
    assert_eq!(t.error.as_deref(), Some("stub model failed"));
    let task = json(s.meeting.folder().join("tasks/t01/task.json"));
    assert_eq!((&task["status"], &task["error"]), (&"failed".into(), &"stub model failed".into()));
    s.meeting.stop().unwrap();
}

/// The scribe's and the agent's models, from the first three events a
/// subscriber gets: those and the intent check's between them.
fn first_two(meeting: &Meeting) -> [CoreEvent; 2] {
    let events = meeting.subscribe();
    let scribe = events.recv().unwrap();
    assert!(matches!(events.recv().unwrap(), CoreEvent::Intent(_)));
    [scribe, events.recv().unwrap()]
}

#[test]
fn an_agent_that_cannot_work_says_why_before_any_request() {
    for (env, why) in [
        ("STUB_PI_VERSION=1.2.0", "pi 1.2.0 is installed, but Pam needs 1.1.0"),
        ("STUB_PI_AUTH=not_ready", "pi has no login for stub/model"),
        ("STUB_PI_AUTH=invalid", "pi can't use the model stub/model"),
    ] {
        let s = start(env);
        let [scribe, agent] = first_two(&s.meeting);
        let (CoreEvent::Scribe(scribe), CoreEvent::Agent(info)) = (&scribe, &agent) else {
            panic!("{env}: {scribe:?}, {agent:?}")
        };
        assert_eq!(scribe.status, ModelStatus::Off, "{env}");
        assert_eq!(info.status, ModelStatus::Unavailable, "{env}");
        assert!(info.reason.as_deref().unwrap().starts_with(why), "{env}: {info:?}");

        let err = s.meeting.send(Command::Ask("anything".into())).unwrap_err().to_string();
        assert!(err.starts_with("the agent is unavailable: ") && err.contains(why), "{err}");
        let folder = s.meeting.folder().to_path_buf();
        assert!(!folder.join("tasks").exists(), "{env}");
        s.meeting.stop().unwrap();
        let (code, out, _) = pam(&folder, &[], &["status"]);
        assert_eq!(code, 0);
        assert!(out.contains(&format!("agent   unavailable: {why}")), "{env}: {out}");
    }
}

#[test]
fn another_patch_release_warns_but_answers() {
    let s = start("STUB_PI_VERSION=1.1.3");
    let [_, CoreEvent::Agent(info)] = first_two(&s.meeting) else { panic!() };
    assert_eq!(info.status, ModelStatus::Warning);
    assert!(info.reason.as_deref().unwrap().starts_with("pi 1.1.3 is not the tested 1.1.0"), "{info:?}");
    s.meeting.send(Command::Ask("hello".into())).unwrap();
    let got = wait_for(&s.events, updated(1, TaskStatus::Done));
    assert_eq!(said(&got), [(TaskId(1), "Stub answer to: hello")]);
    let (_, out, _) = pam(s.meeting.folder(), &[], &["status"]);
    assert!(
        out.contains("agent   ready (pi 1.1.3, stub/model, config), but pi 1.1.3 is not the tested"),
        "{out}"
    );
    s.meeting.stop().unwrap();
}

#[test]
fn pam_say_goes_to_the_running_task_or_the_one_given() {
    let s = start("");
    s.meeting.send(Command::Ask("a slow one".into())).unwrap();
    wait_for(&s.events, updated(1, TaskStatus::Running));
    let folder = s.meeting.folder().to_path_buf();
    prompted(&folder, 1);

    // No --task: the running task, whatever $PI_SESSION_FILE says.
    let session = folder.join("agent/session.jsonl");
    let (code, out, err) = pam(&folder, &[("PI_SESSION_FILE", &session)], &["say", "One", "more", "thing."]);
    assert_eq!((code, out.as_str()), (0, "t01 on the stage: One more thing.\n"), "{err}");
    let got = wait_for(&s.events, |e| matches!(e, CoreEvent::StageMessage { .. }));
    assert_eq!(said(&got), [(TaskId(1), "One more thing.")]);

    s.meeting.send(Command::CancelTask(TaskId(1))).unwrap();
    wait_for(&s.events, updated(1, TaskStatus::Cancelled));
    let (code, _, err) = pam(&folder, &[("PI_SESSION_FILE", &session)], &["say", "hi"]);
    assert_eq!(code, 1);
    assert!(
        err.starts_with("error: which task is this for? None is running. Pass --task; its tasks are t01\n"),
        "{err}"
    );
    let (code, out, err) = pam(&folder, &[], &["say", "--task", "t01", "hi"]);
    assert_eq!((code, out.as_str()), (0, "t01 on the stage: hi\n"), "{err}");
    let (code, _, err) = pam(&folder, &[], &["say", "--task", "t07", "hi"]);
    assert_eq!((code, err.as_str()), (2, "error: no task t07: its tasks are t01\n"));
    let (code, _, err) = pam(&folder, &[], &["say", "--task", "seven", "hi"]);
    assert_eq!(code, 1);
    assert!(err.starts_with("error: `seven` is not a task ID like t02\n"), "{err}");
    s.meeting.stop().unwrap();
}

/// Each report of task `id` as `TaskUpdated` carried it, in order, once per change.
fn reports(events: &[CoreEvent], id: u32) -> Vec<(Option<u8>, Option<&str>)> {
    let mut got: Vec<(Option<u8>, Option<&str>)> = Vec::new();
    for e in events {
        if let CoreEvent::TaskUpdated(t) = e
            && t.id == TaskId(id)
            && got.last() != Some(&(t.progress, t.action.as_deref()))
        {
            got.push((t.progress, t.action.as_deref()));
        }
    }
    got
}

#[test]
fn the_agents_reports_reach_task_json_and_the_room_as_they_come() {
    let s = start("STUB_PI_REPORT_GAP=0.6");
    let folder = s.meeting.folder().to_path_buf();
    s.meeting.send(Command::Ask("report on the launch".into())).unwrap();
    let got = wait_for(&s.events, updated(1, TaskStatus::Done));
    // The stub reports three times, each long enough before the next for the meeting to see it.
    assert_eq!(
        reports(&got, 1),
        [
            (None, None),
            (Some(20), Some("searching")),
            (Some(20), Some("reading source 2 of 3")),
            (Some(100), Some("writing it up")),
        ]
    );
    assert_eq!(tasks(&got), [(TaskId(1), TaskStatus::Running), (TaskId(1), TaskStatus::Done)]);
    let task = json(folder.join("tasks/t01/task.json"));
    assert_eq!((&task["progress"], &task["action"]), (&100.into(), &"writing it up".into()));
    let reported = lines_of(folder.join("tasks/t01/updates.jsonl"));
    assert_eq!(reported.len(), 3);
    assert_eq!(reported[1]["action"], "reading source 2 of 3");
    assert!(reported[1].get("progress").is_none(), "{reported:?}");
    // Reports change no status, so they log no event.
    let logged: Vec<Value> = lines_of(folder.join("events.jsonl"));
    let texts: Vec<&str> =
        logged.iter().filter(|e| e["type"] == "task_update").map(|e| e["text"].as_str().unwrap()).collect();
    assert_eq!(texts, ["started", "done"]);
    s.meeting.stop().unwrap();
}

#[test]
fn pam_task_update_reports_for_a_running_task_and_explains_its_mistakes() {
    let s = start("");
    let folder = s.meeting.folder().to_path_buf();
    let update = |args: &[&str]| pam(&folder, &[], &[&["task", "update"], args].concat());
    let (code, _, err) = update(&["--progress", "5"]);
    assert_eq!(code, 1);
    assert!(
        err.starts_with("error: which task is this for? None is running. Give its ID, e.g. pam task update t01 …; this meeting has no tasks yet\n"),
        "{err}"
    );

    s.meeting.send(Command::Ask("a slow one".into())).unwrap();
    wait_for(&s.events, updated(1, TaskStatus::Running));
    prompted(&folder, 1);
    let (code, out, err) = update(&["--progress", "40%", "--action", "reading\nsources"]);
    assert_eq!((code, out.as_str()), (0, "t01 updated: 40% · reading sources\n"), "{err}");
    let got = wait_for(&s.events, |e| matches!(e, CoreEvent::TaskUpdated(t) if t.progress == Some(40)));
    assert_eq!(reports(&got, 1).last(), Some(&(Some(40), Some("reading sources"))));
    let (code, out, err) = update(&["t01", "--action", "checking", "--json"]);
    assert_eq!(code, 0, "{err}");
    let v: Value = serde_json::from_str(&out).unwrap();
    assert_eq!((&v["task"], &v["update"]["action"]), (&"t01".into(), &"checking".into()));
    assert!(v["update"].get("progress").is_none(), "{v}");

    for (args, why) in [
        (&["--progress", "101"][..], "error: --progress must be 0 to 100, not `101`"),
        (&["--progress", "-5"], "error: --progress must be 0 to 100, not `-5`"),
        (&["--progress", "most"], "error: --progress must be 0 to 100, not `most`"),
        (&["--action", " "], "error: --action is empty"),
        (&[], "error: nothing to report: give --progress, --action or both"),
        (&["t01"], "error: nothing to report"),
        (
            &["--action", "reading", "sources"],
            "error: unexpected `sources`: only a task ID like t02 goes before",
        ),
        (
            &["--progress", "5", "--option", "a"],
            "error: --option is for `pam task ask`, not `pam task update`",
        ),
        (&["--status", "done"], "error: unknown option --status for `pam task`"),
        (&["seven", "--progress", "5"], "error: unexpected `seven`: only a task ID"),
    ] {
        let (code, _, err) = update(args);
        assert_eq!(code, 1, "{args:?}: {err}");
        assert!(err.starts_with(why), "{args:?}: {err}");
        assert!(err.contains("usage: pam task update"), "{err}");
    }
    let (code, _, err) = update(&["t07", "--progress", "5"]);
    assert_eq!((code, err.as_str()), (2, "error: no task t07: its tasks are t01\n"));

    s.meeting.send(Command::CancelTask(TaskId(1))).unwrap();
    wait_for(&s.events, updated(1, TaskStatus::Cancelled));
    let (code, _, err) = update(&["t01", "--progress", "90"]);
    assert_eq!(code, 1);
    assert!(
        err.starts_with("error: t01 is cancelled, not running: only a running task reports progress"),
        "{err}"
    );
    let (code, _, err) = update(&["--progress", "90"]);
    assert_eq!(code, 1);
    assert!(err.starts_with("error: which task is this for? None is running."), "{err}");
    s.meeting.stop().unwrap();
}

#[test]
fn a_result_turns_the_chip_ready_and_reaches_the_notes_the_context_pack_and_get() {
    let s = start("");
    let folder = s.meeting.folder().to_path_buf();
    s.meeting.send(Command::Ask("fact-check it and give a result".into())).unwrap();
    let got = wait_for(&s.events, updated(1, TaskStatus::Done));
    // Attached while the task ran, and sent as it was saved; it shows as ready once done.
    let attached = got.iter().find_map(|e| match e {
        CoreEvent::TaskUpdated(t) if t.result.is_some() => Some(t),
        _ => None,
    });
    let attached = attached.unwrap_or_else(|| panic!("no result: {got:#?}"));
    assert_eq!(attached.status, TaskStatus::Running);
    let r = attached.result.as_ref().unwrap();
    assert_eq!((r.title.as_str(), r.target.as_str()), ("Sources", "tasks/t01/result.md"));
    let CoreEvent::TaskUpdated(done) = got.last().unwrap() else { unreachable!() };
    assert!(done.is_ready());
    assert_eq!(
        said(&got),
        [(TaskId(1), "Partly true: shorter onboarding helps when the steps cut are optional.")]
    );
    let task = json(folder.join("tasks/t01/task.json"));
    assert_eq!(task["result"], serde_json::json!({"title": "Sources", "file": "tasks/t01/result.md"}));
    assert!(
        std::fs::read_to_string(folder.join("tasks/t01/result.md")).unwrap().contains("https://example.com/")
    );

    let notes = std::fs::read_to_string(folder.join("notes.md")).unwrap();
    assert!(
        notes.ends_with("- [x] fact-check it and give a result · ready: [Sources](tasks/t01/result.md)\n"),
        "{notes}"
    );
    let context = std::fs::read_to_string(folder.join("context.md")).unwrap();
    assert!(
        context
            .contains("- t01 00:42 typed: fact-check it and give a result · ready (tasks/t01/result.md)\n"),
        "{context}"
    );
    let (_, out, _) = pam(&folder, &[], &["get", "t01"]);
    assert!(out.contains("\n  result    Sources: tasks/t01/result.md\n"), "{out}");
    let (_, status, _) = pam(&folder, &[], &["status"]);
    assert!(status.contains("tasks   1 (0 running, 0 queued, 1 ready)\n  t01 "), "{status}");
    // Results change no status: started and done are all the events say.
    let texts: Vec<String> = lines_of(folder.join("events.jsonl"))
        .iter()
        .filter(|e| e["type"] == "task_update")
        .map(|e| e["text"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(texts, ["started", "done"]);
    s.meeting.stop().unwrap();
}

#[test]
fn pam_task_result_attaches_a_file_or_a_url_and_explains_its_mistakes() {
    let s = start("");
    let folder = s.meeting.folder().to_path_buf();
    let result = |args: &[&str]| pam_in(&folder, &folder, &[], &[&["task", "result"], args].concat());
    let (code, _, err) = result(&["--url", "https://example.com"]);
    assert_eq!(code, 1);
    assert!(err.starts_with("error: which task is this for? None is running."), "{err}");

    s.meeting.send(Command::Ask("a slow one".into())).unwrap();
    wait_for(&s.events, updated(1, TaskStatus::Running));
    prompted(&folder, 1);
    std::fs::write(folder.join("tasks/t01/result.md"), "# Sources\n").unwrap();
    // A file, from the meeting folder as the agent runs: its name is the title.
    let (code, out, err) = result(&["--file", "tasks/t01/result.md"]);
    assert_eq!((code, out.as_str()), (0, "t01 result: result.md · tasks/t01/result.md\n"), "{err}");
    let has_result = |e: &CoreEvent| matches!(e, CoreEvent::TaskUpdated(t) if t.result.is_some());
    let got = wait_for(&s.events, has_result);
    assert!(
        matches!(got.last(), Some(CoreEvent::TaskUpdated(t)) if t.result.as_ref().unwrap().title == "result.md")
    );
    // From elsewhere, it is still kept relative to the meeting folder.
    let (code, out, err) = pam_in(
        &folder.join("tasks"),
        &folder,
        &[],
        &["task", "result", "t01", "--file", "t01/result.md", "--title", "The  sources"],
    );
    assert_eq!((code, out.as_str()), (0, "t01 result: The sources · tasks/t01/result.md\n"), "{err}");
    // A URL replaces it: one result per task. Its title is the address.
    let (code, out, err) = result(&["--url", "https://example.com/report", "--json"]);
    assert_eq!(code, 0, "{err}");
    let v: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(
        v,
        serde_json::json!({"task": "t01", "result": {"title": "https://example.com/report", "url": "https://example.com/report"}})
    );
    wait_for(
        &s.events,
        |e| matches!(e, CoreEvent::TaskUpdated(t) if t.result.as_ref().is_some_and(|r| r.target.as_str() == "https://example.com/report")),
    );
    let task = json(folder.join("tasks/t01/task.json"));
    assert_eq!(task["result"]["url"], "https://example.com/report");
    // A file outside the meeting folder is kept by its full path.
    let outside = folder.parent().unwrap().join("outside.md");
    std::fs::write(&outside, "x").unwrap();
    let (code, out, err) = result(&["--file", outside.to_str().unwrap()]);
    let full = std::fs::canonicalize(&outside).unwrap();
    assert_eq!((code, out), (0, format!("t01 result: outside.md · {}\n", full.display())), "{err}");

    for (args, why) in [
        (
            &["--file", "tasks/t01/missing.md"][..],
            "error: no file tasks/t01/missing.md: write it first, then attach it",
        ),
        (&[], "error: nothing to attach: give --file <path> or --url <url>"),
        (
            &["--file", "tasks/t01/result.md", "--url", "https://example.com"],
            "error: give --file or --url, not both",
        ),
        (&["--url", "example.com"], "error: --url must be an http:// or https:// address, not `example.com`"),
        (&["--url", "ftp://example.com"], "error: --url must be an http:// or https:// address"),
        (&["--url", "https://example.com/a b"], "error: --url has a space in it"),
        (&["--url", "https://example.com", "--title", " "], "error: --title is empty"),
        (&["--file", ""], "error: --file is empty"),
        (
            &["--title", "Two", "words", "--url", "https://example.com"],
            "error: unexpected `words`: only a task ID like t02 goes before",
        ),
        (
            &["--url", "https://example.com", "--progress", "5"],
            "error: --progress is for `pam task update`, not `pam task result`",
        ),
    ] {
        let (code, _, err) = result(args);
        assert_eq!(code, 1, "{args:?}: {err}");
        assert!(err.starts_with(why), "{args:?}: {err}");
        assert!(err.contains("usage: pam task update"), "{err}");
        assert!(err.contains("pam task result [<id>] (--file <path> | --url <url>)"), "{err}");
    }
    let (code, _, err) = result(&["t07", "--url", "https://example.com"]);
    assert_eq!((code, err.as_str()), (2, "error: no task t07: its tasks are t01\n"));
    let (code, _, err) = pam_in(&folder, &folder, &[], &["task", "update", "--file", "x"]);
    assert_eq!(code, 1);
    assert!(err.starts_with("error: --file is for `pam task result`, not `pam task update`"), "{err}");

    s.meeting.send(Command::CancelTask(TaskId(1))).unwrap();
    wait_for(&s.events, updated(1, TaskStatus::Cancelled));
    let (code, _, err) = result(&["t01", "--url", "https://example.com"]);
    assert_eq!(code, 1);
    assert!(
        err.starts_with("error: t01 is cancelled, not running: only a running task attaches a result"),
        "{err}"
    );
    s.meeting.stop().unwrap();
}

#[test]
fn one_pi_session_takes_every_task_in_turn() {
    let s = start("");
    for request in ["first", "second", "third"] {
        s.meeting.send(Command::Ask(request.into())).unwrap();
    }
    let got = wait_for(&s.events, updated(3, TaskStatus::Done));
    assert_eq!(
        said(&got),
        [
            (TaskId(1), "Stub answer to: first"),
            (TaskId(2), "Stub answer to: second"),
            (TaskId(3), "Stub answer to: third"),
        ]
    );
    let folder = s.meeting.folder().canonicalize().unwrap();
    let all = prompts(&folder);
    let asked: Vec<&Value> = all.iter().map(|p| &p["task"]).collect();
    assert_eq!(asked, ["t01", "t02", "t03"]);
    assert!(all.iter().all(|p| p["pid"] == all[0]["pid"]), "one pi for every task: {all:?}");
    assert_eq!(args_of(&all[0])[5], folder.join("agent/session.jsonl").to_str().unwrap());
    // Each prompt names its task. The first gets the whole brief; the others
    // what was said since the prompt before, nothing here, as the input has ended.
    for (i, (p, (id, request))) in
        all.iter().zip([("t01", "first"), ("t02", "second"), ("t03", "third")]).enumerate()
    {
        let prompt = p["prompt"].as_str().unwrap();
        let lead = if i == 0 {
            "<context.md>\n".to_string()
        } else {
            "Said since the last prompt (00:42), word for word:\n<lines>\n(nobody spoke)\n</lines>\n\
             More: pam transcript --from 00:42\n\n"
                .to_string()
        };
        assert!(prompt.starts_with(&lead), "{prompt}");
        assert!(
            prompt.contains(&format!("Request {id} (typed on the meeting laptop at 00:42):\n{request}\n")),
            "{prompt}"
        );
    }
    assert!(!folder.join("tasks/t01/session.jsonl").exists());
    s.meeting.stop().unwrap();
}

#[test]
fn a_pi_that_exits_fails_its_task_and_the_next_starts_a_new_session() {
    let s = start("");
    s.meeting.send(Command::Ask("crash now".into())).unwrap();
    s.meeting.send(Command::Ask("then this".into())).unwrap();
    let got = wait_for(&s.events, updated(2, TaskStatus::Done));
    let failed = got.iter().find_map(|e| match e {
        CoreEvent::TaskUpdated(t) if t.status == TaskStatus::Failed => Some(t),
        _ => None,
    });
    let error = failed.and_then(|t| t.error.as_deref()).unwrap_or_else(|| panic!("{got:#?}"));
    // pi's last words follow when its stderr is read in time.
    assert!(error.starts_with("pi exited before it finished"), "{error}");
    assert_eq!(said(&got), [(TaskId(2), "Stub answer to: then this")]);

    let folder = s.meeting.folder().canonicalize().unwrap();
    let all = prompts(&folder);
    assert_eq!(all.len(), 2, "{all:?}");
    assert_ne!(all[0]["pid"], all[1]["pid"], "pi started again");
    assert_eq!(args_of(&all[0])[5], folder.join("agent/session.jsonl").to_str().unwrap());
    assert_eq!(args_of(&all[1])[5], folder.join("agent/session-2.jsonl").to_str().unwrap());
    // A new session's first prompt has the whole brief.
    assert!(all[1]["prompt"].as_str().unwrap().starts_with("<context.md>\n"), "{}", all[1]["prompt"]);
    s.meeting.stop().unwrap();
}

/// The meeting time of the last clock before `events[i]`.
fn clock_before(events: &[CoreEvent], i: usize) -> Ts {
    events[..i]
        .iter()
        .rev()
        .find_map(|e| if let CoreEvent::Clock { now } = e { Some(*now) } else { None })
        .unwrap()
}

/// Each spoken request as first announced: its task, its line (`None`
/// before it is final), its summary and the end of its cancel window.
fn invocations(events: &[CoreEvent]) -> Vec<(TaskId, Option<UtteranceId>, &str, Ts)> {
    let mut seen = Vec::new();
    events
        .iter()
        .filter_map(|e| match e {
            CoreEvent::Invocation { task, line, summary, cancel_deadline, .. } if !seen.contains(task) => {
                seen.push(*task);
                Some((*task, *line, summary.as_str(), *cancel_deadline))
            }
            _ => None,
        })
        .collect()
}

fn lines_of(path: PathBuf) -> Vec<Value> {
    std::fs::read_to_string(&path)
        .unwrap_or_else(|e| panic!("{}: {e}", path.display()))
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect()
}

#[test]
fn a_spoken_request_is_answered_after_its_countdown_and_a_cancel_stops_the_next() {
    let (s, intent) = start_spoken("spoken.fake.jsonl", "");
    let folder = s.meeting.folder().canonicalize().unwrap();
    let got = wait_for(&s.events, |e| matches!(e, CoreEvent::InputEnded { .. }));

    // "Pam, what did we decide?" (u0002): announced as soon as the transcriber
    // ended its utterance, before its line was final, then answered once its window closed.
    let inv = invocations(&got);
    assert_eq!(inv.len(), 2, "{inv:?}");
    let (task, line, summary, deadline) = inv[0];
    assert_eq!((task, line, summary), (TaskId(1), None, "Summary of: Pam, what did we decide?"));
    let announced = got.iter().position(|e| matches!(e, CoreEvent::Invocation { .. })).unwrap();
    assert_eq!(deadline, clock_before(&got, announced) + 5 * 16_000, "a 5 s window from the announcement");
    // "decide?" ends at 4.6 s; its utterance is final 0.8 s of silence and 0.3 s later.
    assert!(clock_before(&got, announced) <= 6 * 16_000, "announced at {}", clock_before(&got, announced));
    let line_final =
        got.iter().position(|e| matches!(e, CoreEvent::LineFinal(l) if l.id == UtteranceId(2))).unwrap();
    assert!(announced < line_final, "announced before its line was final");
    // Once the line is final, the task gets it and is announced again with it.
    let again = got
        .iter()
        .position(|e| matches!(e, CoreEvent::Invocation { line: Some(UtteranceId(2)), .. }))
        .unwrap_or_else(|| panic!("no line given: {got:#?}"));
    assert!(again > line_final);
    let answer = got
        .iter()
        .position(|e| matches!(e, CoreEvent::StageMessage { task: Some(TaskId(1)), .. }))
        .unwrap_or_else(|| panic!("no answer: {got:#?}"));
    assert!(clock_before(&got, answer) >= deadline, "the answer waited for the window");
    let CoreEvent::StageMessage { at, .. } = got[answer] else { unreachable!() };
    assert_eq!(at, clock_before(&got, answer), "timed when it went out, not when it was written");
    assert_eq!(said(&got), [(TaskId(1), "Stub answer to: what did we decide?")]);
    let done = got.iter().position(updated(1, TaskStatus::Done)).unwrap();
    assert!(answer < done);

    // "Pam from finance agreed." was checked and made nothing; "Pam, draft an
    // email." was cancelled by "Pam, cancel." in its window, and never showed an answer.
    let checked = intent.said.lock().unwrap().clone();
    assert!(checked.iter().any(|s| s.ends_with("Pam from finance agreed.")), "{checked:?}");
    // Every check counts, the one that made nothing too.
    let usage = last_usage(&got);
    assert_eq!((usage.intent.calls, usage.intent.input), (checked.len() as u64, 50 * checked.len() as u64));
    assert!(!checked.iter().any(|s| s.contains("cancel")), "a cancel needs no model: {checked:?}");
    assert_eq!(inv[1].0, TaskId(2));
    let cancelled = got.iter().position(updated(2, TaskStatus::Cancelled)).unwrap();
    assert!(clock_before(&got, cancelled) < inv[1].3, "cancelled inside its window");
    assert_eq!(pam_core::task::ids(&folder).unwrap(), [TaskId(1), TaskId(2)]);

    let t1 = json(folder.join("tasks/t01/task.json"));
    assert_eq!(
        (&t1["status"], &t1["line"], &t1["request"]),
        (&"done".into(), &"u0002".into(), &"what did we decide?".into())
    );
    assert_eq!(json(folder.join("tasks/t02/task.json"))["status"], "cancelled");
    // The agent got the request line and the minute before it, and who asked.
    let prompt = prompts(&folder)[0]["prompt"].as_str().unwrap().to_string();
    // The request's own line wasn't final yet: it comes without an ID.
    assert!(prompt.contains("<last-60-seconds>\nu0001 00:00 spk_0: Okay, we ship on Friday.\n00:03 spk_1: Pam, what did we decide?\n</last-60-seconds>"), "{prompt}");
    assert!(prompt.contains("Request t01 (spk_1 asked at 00:03):\nwhat did we decide?\n"), "{prompt}");

    // Each request is a `request` event, linked to its line and task.
    let requests: Vec<Value> =
        lines_of(folder.join("events.jsonl")).into_iter().filter(|e| e["type"] == "request").collect();
    let brief: Vec<(&Value, &Value, &Value)> =
        requests.iter().map(|e| (&e["task"], &e["refs"], &e["by"])).collect();
    assert_eq!(
        brief,
        [
            (&"t01".into(), &serde_json::json!(["u0002"]), &serde_json::json!(["spk_1"])),
            (&"t02".into(), &serde_json::json!(["u0004"]), &serde_json::json!(["spk_1"])),
        ]
    );
    // Each change of status is a `task_update`, naming the line once it is final.
    let updates: Vec<(String, String, Value)> = lines_of(folder.join("events.jsonl"))
        .into_iter()
        .filter(|e| e["type"] == "task_update")
        .map(|e| (e["task"].as_str().unwrap().into(), e["text"].as_str().unwrap().into(), e["refs"].clone()))
        .collect();
    let no_refs = serde_json::json!([]);
    assert_eq!(
        updates,
        [
            ("t01".into(), "started".into(), no_refs.clone()),
            ("t01".into(), "done".into(), serde_json::json!(["u0002"])),
            ("t02".into(), "started".into(), no_refs),
            ("t02".into(), "cancelled".into(), serde_json::json!(["u0004"])),
        ]
    );
    // Said, by whom, and when it ran: started once its prompt went out, after
    // the announcement, and ended before its window closed.
    assert_eq!((&t1["by"], &t1["spk"]), (&"voice".into(), &"spk_1".into()));
    let (started, ended) = (t1["started_at"].as_u64().unwrap(), t1["ended_at"].as_u64().unwrap());
    assert!(clock_before(&got, announced) <= started, "started at {started}");
    assert!(started <= ended && ended < deadline, "{started}..{ended}, window to {deadline}");
    // The notes show the answered request as done; the cancelled one not at all.
    let notes = std::fs::read_to_string(folder.join("notes.md")).unwrap();
    assert!(
        notes.contains("## Action items\n\n- [x] Summary of: Pam, what did we decide? · done (spk_1)\n"),
        "{notes}"
    );
    assert!(!notes.contains("draft an email"), "{notes}");
    s.meeting.stop().unwrap();
}

#[test]
fn a_request_said_in_parts_waits_for_its_end_starts_before_its_line_and_takes_what_follows() {
    let (s, intent) = start_spoken("spoken-continued.fake.jsonl", "");
    let folder = s.meeting.folder().canonicalize().unwrap();
    let got = wait_for(&s.events, |e| matches!(e, CoreEvent::InputEnded { .. }));
    let checked = intent.said.lock().unwrap().clone();
    // A voice has a speaker ID once one of its spans is final (at 6 s here):
    // until then, who asked is unknown.
    assert_eq!(
        checked,
        [
            "unknown: hey Pam, can you",
            "unknown: hey Pam, can you uh find the weather",
            "spk_0: hey Pam, can you uh find the weather in Auckland",
        ]
    );
    // Started on "find the weather", 1.1 s after its last word ended (3.58 s)...
    let inv = invocations(&got);
    assert_eq!(inv.len(), 1, "{inv:?}");
    let announced = got.iter().position(|e| matches!(e, CoreEvent::Invocation { .. })).unwrap();
    assert!(clock_before(&got, announced) <= 5 * 16_000, "announced at {}", clock_before(&got, announced));
    // ...and "in Auckland" changed it, in its window, as a correction.
    let t1 = json(folder.join("tasks/t01/task.json"));
    assert_eq!(t1["request"], "can you uh find the weather");
    assert_eq!(t1["corrections"], serde_json::json!(["can you uh find the weather in Auckland"]));
    assert_eq!(t1["line"], "u0001", "the line it was said in, once final");
    let all = prompts(&folder);
    assert!(all.len() >= 2, "{all:?}");
    let again = all.last().unwrap()["prompt"].as_str().unwrap().to_string();
    // A later prompt: what was said since, its own line on its way without an ID, then the correction.
    assert!(
        again.starts_with(
            "Said since the last prompt (00:04), word for word:\n<lines>\n\
             00:00 spk_0: hey Pam, can you uh find the weather in Auckland\n</lines>\n"
        ),
        "{again}"
    );
    assert!(
        again.contains("\n\nCorrection from spk_0 at 00:00:\ncan you uh find the weather in Auckland\n"),
        "{again}"
    );
    // Both `request` events name the line; every check is in intent.jsonl.
    let requests: Vec<Value> =
        lines_of(folder.join("events.jsonl")).into_iter().filter(|e| e["type"] == "request").collect();
    assert_eq!(requests.iter().map(|e| &e["refs"]).collect::<Vec<_>>(), [&serde_json::json!(["u0001"]); 2]);
    let log = lines_of(folder.join("intent.jsonl"));
    let kinds: Vec<&Value> = log.iter().map(|c| &c["kind"]).collect();
    assert_eq!(kinds, ["unfinished", "request", "request"]);
    assert_eq!(log[0]["model"], "stub/intent");
    assert_eq!(log[0]["said"], "hey Pam, can you");
    assert!(
        log[0]["prompt"].as_str().unwrap().ends_with("UTTERANCE\nunknown: hey Pam, can you\n"),
        "{}",
        log[0]
    );
    assert!(log.iter().all(|c| c["ms"].is_u64() && c["sent_at"].as_u64() <= c["at"].as_u64()));
    s.meeting.stop().unwrap();
}

/// An intent check that tells `entered` it was asked, then answers only
/// after the test is over.
struct StuckIntent {
    entered: Mutex<crossbeam_channel::Sender<()>>,
}

impl pam_core::llm::Model for StuckIntent {
    fn complete(&self, _: &Request<'_>) -> Completion {
        let _ = self.entered.lock().unwrap().send(());
        std::thread::sleep(Duration::from_secs(60));
        Completion { reply: Err(anyhow::anyhow!("too late")), used: Used::Unknown }
    }
}

#[test]
fn a_check_still_out_when_the_meeting_stops_is_logged_as_stopped() {
    let (entered, asked) = crossbeam_channel::unbounded();
    let stuck = Arc::new(StuckIntent { entered: Mutex::new(entered) });
    let s = start_checked_by("spoken.fake.jsonl", "", Some(stuck));
    let folder = s.meeting.folder().canonicalize().unwrap();
    asked.recv_timeout(Duration::from_secs(30)).expect("no check was sent");
    s.meeting.stop().unwrap();
    let log = lines_of(folder.join("intent.jsonl"));
    assert!(!log.is_empty());
    for c in &log {
        assert_eq!(c["error"], "the meeting stopped before its reply", "{c}");
        assert!(c["reply"].is_null() && c["ms"].is_u64() && c["said"].is_string(), "{c}");
    }
}

#[test]
fn a_correction_aborts_the_run_and_prompts_the_same_pi_again() {
    let (s, _) = start_spoken("spoken-correction.fake.jsonl", "");
    let folder = s.meeting.folder().canonicalize().unwrap();
    let got = wait_for(&s.events, updated(1, TaskStatus::Done));
    // The first run said something, even as the abort landed; the corrected
    // run replied in text only, and that reply is shown, not dropped because
    // of what the first run said.
    assert_eq!(
        said(&got),
        [
            (TaskId(1), "Working on it."),
            (TaskId(1), "Still on the old request."),
            (TaskId(1), "Reply in text to: I meant the silent report."),
        ]
    );
    assert_eq!(invocations(&got).len(), 1, "a correction is not a new task");
    let t1 = json(folder.join("tasks/t01/task.json"));
    assert_eq!(t1["corrections"], serde_json::json!(["I meant the silent report."]));
    // One pi took both prompts: the run was aborted, not restarted.
    let [first, again] = &prompts(&folder)[..] else { panic!("{:?}", prompts(&folder)) };
    assert_eq!((&first["task"], &again["task"]), (&"t01".into(), &Value::Null));
    assert_eq!(first["pid"], again["pid"]);
    // The correction is a later prompt: it follows what was said since, its own line among it.
    let again = again["prompt"].as_str().unwrap();
    assert!(again.starts_with("Said since the last prompt ("), "{again}");
    assert!(again.contains(" spk_0: Pam, I meant the silent report.\n</lines>\n"), "{again}");
    assert!(again.contains("\n\nCorrection from spk_0 at 00:04:\nI meant the silent report.\n"), "{again}");
    // The correction's `request` event names its line, so it is saved once that is final.
    wait_for(&s.events, |e| matches!(e, CoreEvent::InputEnded { .. }));
    let requests: Vec<Value> =
        lines_of(folder.join("events.jsonl")).into_iter().filter(|e| e["type"] == "request").collect();
    let refs: Vec<&Value> = requests.iter().map(|e| &e["refs"]).collect();
    assert_eq!(refs, [&serde_json::json!(["u0001"]), &serde_json::json!(["u0002"])]);
    // Each message counts once: the aborted one and the reply, though pi
    // repeats messages in `turn_end` and `agent_end`.
    let usage = last_usage(&got);
    assert_eq!((usage.agent.calls, usage.agent.input), (2, 4000));
    assert!((usage.agent.cost - 0.02).abs() < 1e-9, "{usage:?}");
    s.meeting.stop().unwrap();
}

#[test]
fn a_correction_once_the_task_is_answered_is_a_new_request_in_the_same_session() {
    let (s, _) = start_spoken("spoken-followup.fake.jsonl", "");
    let folder = s.meeting.folder().canonicalize().unwrap();
    let got = wait_for(&s.events, updated(2, TaskStatus::Done));
    let inv = invocations(&got);
    assert_eq!(inv.iter().map(|i| i.0).collect::<Vec<_>>(), [TaskId(1), TaskId(2)]);
    let lines: Vec<Value> =
        [1, 2].iter().map(|n| json(folder.join(format!("tasks/t0{n}/task.json")))["line"].clone()).collect();
    assert_eq!(lines, ["u0002", "u0003"], "each task got its line once final");
    assert_eq!(
        said(&got),
        [
            (TaskId(1), "Stub answer to: what did we decide?"),
            (TaskId(2), "Stub answer to: I meant the launch date.")
        ]
    );
    let t1 = json(folder.join("tasks/t01/task.json"));
    assert!(t1.get("corrections").is_none(), "the answered task is left alone: {t1}");
    let all = prompts(&folder);
    let asked: Vec<&Value> = all.iter().map(|p| &p["task"]).collect();
    assert_eq!(asked, ["t01", "t02"]);
    assert_eq!(all[0]["pid"], all[1]["pid"], "it follows the answer in the same session");
    s.meeting.stop().unwrap();
}

#[test]
fn cancelling_a_running_task_aborts_pi_and_the_next_one_runs() {
    let s = start("");
    s.meeting.send(Command::Ask("a slow one".into())).unwrap();
    s.meeting.send(Command::Ask("then this".into())).unwrap();
    wait_for(&s.events, updated(1, TaskStatus::Running));
    prompted(s.meeting.folder(), 1);
    let asked = Instant::now();
    s.meeting.send(Command::CancelTask(TaskId(1))).unwrap();
    let got = wait_for(&s.events, updated(1, TaskStatus::Cancelled));
    assert!(asked.elapsed() < Duration::from_secs(1), "pi confirmed the abort: {:?}", asked.elapsed());
    assert!(said(&got).is_empty(), "{got:?}");
    let got = wait_for(&s.events, updated(2, TaskStatus::Done));
    assert_eq!(said(&got), [(TaskId(2), "Stub answer to: then this")]);
    // The cancel aborted the run, not the session: the same pi took t02.
    let all = prompts(s.meeting.folder());
    assert_eq!((&all[1]["task"], &all[1]["pid"]), (&"t02".into(), &all[0]["pid"]), "{all:?}");
    let err = s.meeting.send(Command::CancelTask(TaskId(2))).unwrap_err().to_string();
    assert_eq!(err, "t02 can't be cancelled: it is answered");
    let err = s.meeting.send(Command::CancelTask(TaskId(9))).unwrap_err().to_string();
    assert_eq!(err, "no task t09");
    s.meeting.stop().unwrap();
}

fn is_question(e: &CoreEvent) -> bool {
    matches!(e, CoreEvent::Question { .. })
}

fn is_closed(e: &CoreEvent) -> bool {
    matches!(e, CoreEvent::QuestionClosed { .. })
}

/// How each question in `events` ended.
fn closed(events: &[CoreEvent]) -> Vec<(TaskId, &End)> {
    events
        .iter()
        .filter_map(|e| match e {
            CoreEvent::QuestionClosed { task, end } => Some((*task, end)),
            _ => None,
        })
        .collect()
}

/// The `pam task ask` waiting for task t01's first question.
fn asker(folder: &Path) -> i32 {
    lines_of(folder.join("tasks/t01/questions.jsonl"))[0]["pid"].as_i64().unwrap() as i32
}

fn alive(pid: i32) -> bool {
    // SAFETY: signal 0 only asks whether the process exists.
    unsafe { libc::kill(pid, 0) == 0 }
}

#[test]
fn a_question_is_answered_by_key_and_the_answer_follows_it() {
    let s = start("");
    let folder = s.meeting.folder().to_path_buf();
    s.meeting.send(Command::Ask("ask which customer to email".into())).unwrap();
    let got = wait_for(&s.events, is_question);
    let Some(CoreEvent::Question { task, q, text, options, deadline }) = got.last() else {
        panic!("{got:?}")
    };
    assert_eq!((*task, text.as_str()), (TaskId(1), "Which customer is the email for?"));
    assert_eq!(options, &["Acme", "Globex"]);
    let left = deadline.saturating_duration_since(Instant::now());
    assert!(left > Duration::from_secs(1) && left <= Duration::from_secs(2), "a 2 s countdown: {left:?}");

    let err = s
        .meeting
        .send(Command::Answer { task: TaskId(1), q: *q, answer: Answer::Option(3) })
        .unwrap_err()
        .to_string();
    assert_eq!(err, "t01's question has options 1–2");
    s.meeting.send(Command::Answer { task: TaskId(1), q: *q, answer: Answer::Option(2) }).unwrap();
    let got = wait_for(&s.events, updated(1, TaskStatus::Done));
    assert_eq!(
        closed(&got),
        [(TaskId(1), &End::Answered { answer: Answer::Option(2), by: By::Key, spk: None })]
    );
    assert_eq!(said(&got), [(TaskId(1), "Drafting the email to Globex.")]);
    let err = s
        .meeting
        .send(Command::Answer { task: TaskId(1), q: *q, answer: Answer::Option(1) })
        .unwrap_err()
        .to_string();
    assert_eq!(err, "t01 isn't asking anything");

    let saved = lines_of(folder.join("tasks/t01/questions.jsonl"));
    assert_eq!(saved.len(), 2, "{saved:?}");
    assert_eq!((&saved[0]["type"], &saved[0]["timeout"]), (&"asked".into(), &2.into()));
    assert_eq!(
        (&saved[1]["type"], &saved[1]["option"], &saved[1]["by"]),
        (&"answered".into(), &2.into(), &"key".into())
    );
    s.meeting.stop().unwrap();
}

#[test]
fn a_question_nobody_answers_times_out_and_the_agent_guesses() {
    let s = start("STUB_PI_ASK_TIMEOUT=1s");
    s.meeting.send(Command::Ask("ask which customer to email".into())).unwrap();
    wait_for(&s.events, is_question);
    let shown = Instant::now();
    let got = wait_for(&s.events, is_closed);
    let waited = shown.elapsed();
    assert!(waited >= Duration::from_millis(700) && waited < Duration::from_secs(2), "{waited:?}");
    assert_eq!(closed(&got), [(TaskId(1), &End::TimedOut)]);
    let got = wait_for(&s.events, updated(1, TaskStatus::Done));
    assert_eq!(said(&got), [(TaskId(1), "No answer in time, so I assumed Acme.")]);
    let saved = lines_of(s.meeting.folder().join("tasks/t01/questions.jsonl"));
    assert_eq!(saved[1]["type"], "timed_out");
    s.meeting.stop().unwrap();
}

#[test]
fn cancelling_a_task_while_it_asks_stops_the_asker_and_drops_the_question() {
    let s = start("STUB_PI_ASK_TIMEOUT=30s");
    let folder = s.meeting.folder().to_path_buf();
    s.meeting.send(Command::Ask("ask which customer to email".into())).unwrap();
    wait_for(&s.events, is_question);
    let pid = asker(&folder);
    assert!(alive(pid));
    s.meeting.send(Command::CancelTask(TaskId(1))).unwrap();
    let got = wait_for(&s.events, updated(1, TaskStatus::Cancelled));
    assert!(
        matches!(closed(&got)[..], [(TaskId(1), End::Dropped(_))]),
        "the question goes with its task: {got:?}"
    );
    assert!(said(&got).is_empty(), "{got:?}");
    let end = Instant::now() + Duration::from_secs(2);
    while alive(pid) {
        assert!(Instant::now() < end, "pam task ask {pid} is still running");
        std::thread::sleep(Duration::from_millis(20));
    }
    let saved = lines_of(folder.join("tasks/t01/questions.jsonl"));
    assert_eq!(saved.last().unwrap()["type"], "dropped");
    s.meeting.stop().unwrap();
}

#[test]
fn an_asker_that_goes_away_takes_its_question_off_the_stage() {
    // pi's own bash timeout could kill `pam task ask` before the countdown ends.
    let s = start("STUB_PI_ASK_TIMEOUT=30s");
    s.meeting.send(Command::Ask("ask which customer to email".into())).unwrap();
    wait_for(&s.events, is_question);
    let killed = Instant::now();
    // SAFETY: kills the stub's own `pam task ask`, which this test started.
    assert_eq!(unsafe { libc::kill(asker(s.meeting.folder()), libc::SIGKILL) }, 0);
    let got = wait_for(&s.events, is_closed);
    assert!(killed.elapsed() < Duration::from_secs(1), "{:?}", killed.elapsed());
    assert_eq!(closed(&got), [(TaskId(1), &End::Dropped("the agent stopped waiting for an answer".into()))]);
    let got = wait_for(&s.events, updated(1, TaskStatus::Done));
    assert_eq!(said(&got), [(TaskId(1), "The question went unanswered.")]);
    s.meeting.stop().unwrap();
}

#[test]
fn option_two_said_in_the_room_answers_the_question_shown_after_the_cancel_window() {
    let (s, intent) = start_spoken("clarify-spoken.fake.jsonl", "STUB_PI_ASK_TIMEOUT=8s");
    let got = wait_for(&s.events, updated(1, TaskStatus::Done));
    let inv = invocations(&got);
    assert_eq!(inv.len(), 1, "{inv:?}");
    // pi asked straight away; the question waited for the window to close.
    let asked = got.iter().position(is_question).unwrap_or_else(|| panic!("no question: {got:#?}"));
    assert!(clock_before(&got, asked) >= inv[0].3, "shown before the window closed");
    let ended = closed(&got);
    let [(TaskId(1), End::Answered { answer: Answer::Option(2), by: By::Voice, .. })] = ended[..] else {
        panic!("{ended:?}")
    };
    assert_eq!(said(&got), [(TaskId(1), "Drafting the email to Globex.")]);
    let checked = intent.said.lock().unwrap().clone();
    assert!(!checked.iter().any(|s| s.contains("option")), "an answer needs no model: {checked:?}");
    s.meeting.stop().unwrap();
}

/// The checks `intent` was asked for, as `speaker: words`.
fn checked(intent: &StubIntent) -> Vec<String> {
    intent.said.lock().unwrap().clone()
}

#[test]
fn a_reply_without_the_name_answers_the_question_and_talk_does_not() {
    let (s, intent) = start_spoken("clarify-reply.fake.jsonl", "STUB_PI_ASK_TIMEOUT=8s");
    let got = wait_for(&s.events, |e| matches!(e, CoreEvent::InputEnded { .. }));
    let ended = closed(&got);
    let [(TaskId(1), End::Answered { answer: Answer::Option(2), by: By::Voice, spk: Some(_) })] = ended[..]
    else {
        panic!("{ended:?}")
    };
    assert_eq!(said(&got), [(TaskId(1), "Drafting the email to Globex.")]);
    let checked = checked(&intent);
    assert!(checked.iter().any(|s| s.ends_with(": Let me think.")), "talk is checked too: {checked:?}");
    assert!(checked.iter().any(|s| s.ends_with(": Yes, the second one.")), "{checked:?}");
    assert!(!checked.iter().any(|s| s.contains("Actually")), "the question had closed: {checked:?}");
    // Neither started a task, though the stub calls everything else a request.
    assert_eq!(pam_core::task::ids(s.meeting.folder()).unwrap(), [TaskId(1)]);
    let saved = lines_of(s.meeting.folder().join("tasks/t01/questions.jsonl"));
    assert_eq!((&saved[1]["type"], &saved[1]["by"]), (&"answered".into(), &"voice".into()));
    s.meeting.stop().unwrap();
}

#[test]
fn a_reply_heard_before_the_deadline_but_checked_after_it_still_counts() {
    // A 4 s countdown ends about 00:13, while "Yes, the second one." (00:12) is still said, and its check takes 4 s.
    let intent = Arc::new(StubIntent { slow: Duration::from_secs(4), ..StubIntent::default() });
    let s = start_checked_by("clarify-reply.fake.jsonl", "STUB_PI_ASK_TIMEOUT=4s", Some(intent.clone()));
    let got = wait_for(&s.events, is_question);
    let Some(CoreEvent::Question { deadline, .. }) = got.last() else { panic!("{got:?}") };
    let got = wait_for(&s.events, is_closed);
    assert!(Instant::now() > *deadline + Duration::from_secs(2), "it waited for the check");
    let ended = closed(&got);
    let [(TaskId(1), End::Answered { answer: Answer::Option(2), by: By::Voice, .. })] = ended[..] else {
        panic!("{ended:?}")
    };
    let got = wait_for(&s.events, updated(1, TaskStatus::Done));
    assert_eq!(said(&got), [(TaskId(1), "Drafting the email to Globex.")]);
    s.meeting.stop().unwrap();
}

#[test]
fn with_the_scribe_off_a_reply_needs_the_name_and_nothing_is_checked() {
    let s = start_checked_by("clarify-reply.fake.jsonl", "STUB_PI_ASK_TIMEOUT=15s", None);
    s.meeting.send(Command::Ask("ask which customer to email".into())).unwrap();
    let got = wait_for(&s.events, is_question);
    let Some(CoreEvent::Question { deadline, .. }) = got.last() else { panic!("{got:?}") };
    // "Yes, the second one." (00:12) goes unheard: the countdown ends on time.
    let got = wait_for(&s.events, is_closed);
    assert!(Instant::now() < *deadline + Duration::from_secs(1), "nothing to wait for");
    assert_eq!(closed(&got), [(TaskId(1), &End::TimedOut)]);
    wait_for(&s.events, |e| matches!(e, CoreEvent::InputEnded { .. }));
    // "Pam, ask us which customer to email." made no task: nothing checked it.
    assert_eq!(pam_core::task::ids(s.meeting.folder()).unwrap(), [TaskId(1)]);
    s.meeting.stop().unwrap();
}

#[test]
fn with_the_scribe_off_option_two_still_answers() {
    let s = start_checked_by("clarify-spoken.fake.jsonl", "STUB_PI_ASK_TIMEOUT=15s", None);
    s.meeting.send(Command::Ask("ask which customer to email".into())).unwrap();
    let got = wait_for(&s.events, is_closed);
    let ended = closed(&got);
    let [(TaskId(1), End::Answered { answer: Answer::Option(2), by: By::Voice, .. })] = ended[..] else {
        panic!("{ended:?}")
    };
    s.meeting.stop().unwrap();
}

#[test]
fn an_open_question_is_answered_by_typing_and_the_agent_gets_the_words() {
    let s = start("STUB_PI_ASK_TIMEOUT=8s");
    let folder = s.meeting.folder().to_path_buf();
    s.meeting.send(Command::Ask("ask which city the weather is for".into())).unwrap();
    let got = wait_for(&s.events, is_question);
    let Some(CoreEvent::Question { task, q, text, options, .. }) = got.last() else { panic!("{got:?}") };
    assert_eq!((*task, *q, text.as_str()), (TaskId(1), 1, "Which city are you in?"));
    assert!(options.is_empty(), "{options:?}");

    let answer = |a| s.meeting.send(Command::Answer { task: TaskId(1), q: 1, answer: a });
    let err = answer(Answer::Option(1)).unwrap_err().to_string();
    assert_eq!(err, "t01's question is open: type the answer");
    let err = answer(Answer::Text(" ".into())).unwrap_err().to_string();
    assert_eq!(err, "the answer is empty");
    answer(Answer::Text(" Auckland ".into())).unwrap();
    let got = wait_for(&s.events, updated(1, TaskStatus::Done));
    let typed = End::Answered { answer: Answer::Text("Auckland".into()), by: By::Typed, spk: None };
    assert_eq!(closed(&got), [(TaskId(1), &typed)]);
    assert_eq!(said(&got), [(TaskId(1), "Tomorrow morning in Auckland: sunny.")]);

    let saved = lines_of(folder.join("tasks/t01/questions.jsonl"));
    assert_eq!(saved.len(), 2, "{saved:?}");
    assert_eq!((&saved[0]["options"], &saved[0]["timeout"]), (&serde_json::json!([]), &8.into()));
    assert_eq!(
        (&saved[1]["type"], &saved[1]["text"], &saved[1]["by"]),
        (&"answered".into(), &"Auckland".into(), &"typed".into())
    );
    assert!(saved[1].get("option").is_none() && saved[1].get("spk").is_none(), "{}", saved[1]);
    s.meeting.stop().unwrap();
}

#[test]
fn an_open_question_nobody_answers_times_out_and_the_agent_guesses() {
    let s = start("STUB_PI_ASK_TIMEOUT=1s");
    s.meeting.send(Command::Ask("ask which city the weather is for".into())).unwrap();
    let got = wait_for(&s.events, updated(1, TaskStatus::Done));
    assert_eq!(closed(&got), [(TaskId(1), &End::TimedOut)]);
    assert_eq!(said(&got), [(TaskId(1), "No answer in time, so I assumed Wellington: sunny.")]);
    let saved = lines_of(s.meeting.folder().join("tasks/t01/questions.jsonl"));
    assert_eq!(saved[1]["type"], "timed_out");
    s.meeting.stop().unwrap();
}

#[test]
fn an_open_question_is_answered_in_words_said_without_the_name() {
    let (s, intent) = start_spoken("open-reply.fake.jsonl", "STUB_PI_ASK_TIMEOUT=8s");
    let got = wait_for(&s.events, |e| matches!(e, CoreEvent::InputEnded { .. }));
    let ended = closed(&got);
    let [(TaskId(1), End::Answered { answer: Answer::Text(text), by: By::Voice, spk: Some(_) })] = ended[..]
    else {
        panic!("{ended:?}")
    };
    assert_eq!(text, "Auckland");
    assert_eq!(said(&got), [(TaskId(1), "Tomorrow morning in Auckland: sunny.")]);
    let checked = checked(&intent);
    assert!(checked.iter().any(|s| s.ends_with(": Let me think.")), "talk is checked too: {checked:?}");
    assert!(checked.iter().any(|s| s.ends_with(": We're in Auckland.")), "{checked:?}");
    assert!(!checked.iter().any(|s| s.contains("Wellington")), "the question had closed: {checked:?}");
    assert_eq!(pam_core::task::ids(s.meeting.folder()).unwrap(), [TaskId(1)]);
    let saved = lines_of(s.meeting.folder().join("tasks/t01/questions.jsonl"));
    assert_eq!(saved[0]["options"], serde_json::json!([]));
    assert_eq!(
        (&saved[1]["type"], &saved[1]["text"], &saved[1]["by"]),
        (&"answered".into(), &"Auckland".into(), &"voice".into())
    );
    assert!(saved[1]["spk"].is_string(), "{}", saved[1]);
    // The intent log keeps the answer's words.
    let log = lines_of(s.meeting.folder().join("intent.jsonl"));
    let reply = log.iter().find(|c| c["kind"] == "reply").unwrap_or_else(|| panic!("{log:?}"));
    assert_eq!(reply["text"], "Auckland");
    s.meeting.stop().unwrap();
}

#[test]
fn with_the_scribe_off_an_open_question_can_only_be_typed() {
    let s = start_checked_by("open-reply.fake.jsonl", "STUB_PI_ASK_TIMEOUT=15s", None);
    s.meeting.send(Command::Ask("ask which city the weather is for".into())).unwrap();
    let got = wait_for(&s.events, is_question);
    let Some(CoreEvent::Question { deadline, .. }) = got.last() else { panic!("{got:?}") };
    // "We're in Auckland." (00:12) goes unheard: the countdown ends on time.
    let got = wait_for(&s.events, is_closed);
    assert!(Instant::now() < *deadline + Duration::from_secs(1), "nothing to wait for");
    assert_eq!(closed(&got), [(TaskId(1), &End::TimedOut)]);
    s.meeting.stop().unwrap();
}

#[test]
fn pam_task_ask_explains_its_mistakes_and_a_stopped_meeting_never_answers() {
    let s = start("");
    s.meeting.send(Command::Ask("hello".into())).unwrap();
    wait_for(&s.events, updated(1, TaskStatus::Done));
    let folder = s.meeting.folder().to_path_buf();
    let ask = |args: &[&str]| {
        let mut all = vec!["task", "ask", "t01", "Which customer?"];
        all.extend(args);
        pam(&folder, &[], &all)
    };
    for (args, why) in [
        (&["--option", "Acme"][..], "error: give at least two --option for the room to choose from, or none"),
        (&["--option", "a", "--option", " "], "error: an --option is empty"),
        (
            &["--option", "a", "--option", "b", "--timeout", "0s"],
            "error: --timeout must be 1s to 60s, not `0s`",
        ),
        (&["--option", "a", "--option", "b", "--timeout", "2m"], "error: --timeout must be 1s to 60s"),
        (&["--option", "a", "--option", "b", "--timeout", "soon"], "error: --timeout needs a duration"),
    ] {
        let (code, _, err) = ask(args);
        assert_eq!(code, 1, "{args:?}: {err}");
        assert!(err.starts_with(why), "{args:?}: {err}");
        assert!(err.contains("\n       pam task ask [<id>] <question>"), "{err}");
    }
    let ten: Vec<String> = (1..=10).flat_map(|n| ["--option".to_string(), n.to_string()]).collect();
    let ten: Vec<&str> = ten.iter().map(String::as_str).collect();
    let (code, _, err) = ask(&ten);
    assert_eq!(code, 1);
    assert!(err.starts_with("error: at most 9 --option: the room answers with the keys 1–9"), "{err}");
    let (code, _, err) = pam(&folder, &[], &["task", "ask", "Which?", "--option", "a", "--option", "b"]);
    assert_eq!(code, 1);
    assert!(err.starts_with("error: which task is this for? None is running. Give its ID first"), "{err}");
    let (code, _, err) = ask(&["--option", "a", "--option", "b", "--progress", "5"]);
    assert_eq!(code, 1);
    assert!(err.starts_with("error: --progress is for `pam task update`, not `pam task ask`"), "{err}");

    s.meeting.stop().unwrap();
    let asked = Instant::now();
    let (code, out, err) = ask(&["--option", "Acme", "--option", "Globex", "--json"]);
    assert_eq!(code, 0, "{err}");
    assert!(asked.elapsed() < Duration::from_secs(5), "nobody is there to answer");
    let v: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(v, serde_json::json!({"task": "t01", "question": 1, "chosen": null}));
    let saved = lines_of(folder.join("tasks/t01/questions.jsonl"));
    let types: Vec<&str> = saved.iter().map(|r| r["type"].as_str().unwrap()).collect();
    assert_eq!(types, ["asked", "timed_out"]);
}

#[test]
fn pam_task_ask_takes_the_meetings_countdown_not_its_own_config() {
    let s = start_with("", |cfg| cfg.question_timeout = 7);
    s.meeting.send(Command::Ask("hello".into())).unwrap();
    wait_for(&s.events, updated(1, TaskStatus::Done));
    let folder = s.meeting.folder().to_path_buf();
    assert_eq!(json(folder.join("meeting.json"))["question_timeout"], 7);
    // A config file the agent's working directory would find instead is ignored.
    let elsewhere = tempfile::tempdir().unwrap();
    let config = elsewhere.path().join("config.toml");
    std::fs::write(&config, "question_timeout = 42\n").unwrap();
    let (code, out, err) = pam(
        &folder,
        &[("PAM_CONFIG", &config)],
        &["task", "ask", "t01", "Which customer?", "--option", "Acme", "--option", "Globex"],
    );
    // t01 has finished, so the meeting drops the question at once.
    assert_eq!((code, out.as_str()), (0, "timeout\n"), "{err}");
    assert_eq!(lines_of(folder.join("tasks/t01/questions.jsonl"))[0]["timeout"], 7);
    // An open question gets twice the meeting's.
    let (code, out, err) = pam(&folder, &[], &["task", "ask", "t01", "Which city?", "--json"]);
    assert_eq!(code, 0, "{err}");
    assert_eq!(serde_json::from_str::<Value>(&out).unwrap()["chosen"], Value::Null);
    let saved = lines_of(folder.join("tasks/t01/questions.jsonl"));
    assert_eq!((&saved[2]["options"], &saved[2]["timeout"]), (&serde_json::json!([]), &14.into()));
    s.meeting.stop().unwrap();
}

#[test]
fn a_tasks_cost_counts_each_call_as_it_ends_and_is_kept_in_meeting_json() {
    let s = start("STUB_PI_OAUTH=stub/model");
    let folder = s.meeting.folder().canonicalize().unwrap();
    // Nothing spent at the start; a subscription login is said once there are calls.
    let usage = json(folder.join("meeting.json"))["usage"].clone();
    assert_eq!(
        usage["agent"],
        serde_json::json!({"calls": 0, "input": 0, "output": 0, "cache_read": 0,
                                                  "cache_write": 0, "cost": 0.0, "login": "oauth"})
    );
    assert_eq!(usage["partial"], false);

    // A compaction is a call, with its own tokens, and then the answer.
    s.meeting.send(Command::Ask("compact first, then answer".into())).unwrap();
    let got = wait_for(&s.events, updated(1, TaskStatus::Done));
    let costs: Vec<(u64, f64)> = got
        .iter()
        .filter_map(|e| match e {
            CoreEvent::Usage(u) => Some((u.total.calls, (u.total.cost * 1000.0).round())),
            _ => None,
        })
        .collect();
    assert_eq!(costs, [(1, 5.0), (2, 15.0)], "one Usage per call, in thousandths of a dollar");
    let usage = last_usage(&got);
    assert_eq!(
        (
            usage.agent.calls,
            usage.agent.input,
            usage.agent.output,
            usage.agent.cache_read,
            usage.agent.cache_write
        ),
        (2, 5000, 350, 1000, 500)
    );
    assert!((usage.agent.cost - 0.015).abs() < 1e-9, "{usage:?}");
    assert_eq!((usage.scribe.calls, usage.intent.calls), (0, 0));
    assert_eq!(usage.total, pam_core::usage::Spend { login: None, ..usage.agent.clone() });
    assert!(!usage.partial);
    // meeting.json has it as it was sent.
    let saved: Usage = serde_json::from_value(json(folder.join("meeting.json"))["usage"].clone()).unwrap();
    assert_eq!(saved, usage);

    s.meeting.stop().unwrap();
    let saved: Usage = serde_json::from_value(json(folder.join("meeting.json"))["usage"].clone()).unwrap();
    assert_eq!(saved, usage, "final at Stop");
    let (code, out, err) = pam(&folder, &[], &["status"]);
    assert_eq!(code, 0, "{err}");
    assert!(out.contains("\ncost    ≈$0.015 (scribe $0.000, intent $0.000, agent ≈$0.015)\n"), "{out}");
    let (_, out, _) = pam(&folder, &[], &["status", "--json"]);
    let status: Value = serde_json::from_str(&out).unwrap();
    assert_eq!(status["usage"]["agent"]["calls"], 2);
    assert_eq!(status["usage"]["agent"]["login"], "oauth");
}

#[test]
fn the_cost_is_partial_only_when_pi_didnt_report_what_a_run_used() {
    // pi exits mid-run: what its call used is never reported.
    let s = start("");
    let folder = s.meeting.folder().canonicalize().unwrap();
    s.meeting.send(Command::Ask("crash now".into())).unwrap();
    let got = wait_for(&s.events, updated(1, TaskStatus::Failed));
    let usage = last_usage(&got);
    assert_eq!((usage.agent.calls, usage.partial), (0, true));
    assert_eq!(usage.label(), "$0.000+");
    s.meeting.stop().unwrap();
    let (_, out, _) = pam(&folder, &[], &["status"]);
    assert!(out.contains("\ncost    $0.000+ (scribe $0.000, intent $0.000, agent $0.000)\n"), "{out}");

    // The model failed, and pi reported what that used (nothing): the
    // task failed, but its cost is known.
    let s = start("");
    s.meeting.send(Command::Ask("fail this one".into())).unwrap();
    let got = wait_for(&s.events, updated(1, TaskStatus::Failed));
    let usage = last_usage(&got);
    assert_eq!((usage.agent.calls, usage.agent.input, usage.partial), (1, 0, false));
    s.meeting.stop().unwrap();

    // pi can't be started for the task, though it passed its checks: nothing ran.
    let s = start("");
    let folder = s.meeting.folder().canonicalize().unwrap();
    let pi = s._dir.path().join("pi");
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&pi, std::fs::Permissions::from_mode(0o644)).unwrap();
    s.meeting.send(Command::Ask("what did we decide?".into())).unwrap();
    let got = wait_for(&s.events, updated(1, TaskStatus::Failed));
    assert!(!got.iter().any(|e| matches!(e, CoreEvent::Usage(_))), "{got:#?}");
    s.meeting.stop().unwrap();
    let usage = &json(folder.join("meeting.json"))["usage"];
    assert_eq!((&usage["agent"]["calls"], &usage["partial"]), (&0.into(), &false.into()));

    // A model missing from pi's catalog: pi prices it with another model's
    // rates and says so, so its tokens count and its cost doesn't.
    let s = start("STUB_PI_UNCATALOGUED=1");
    s.meeting.send(Command::Ask("what did we decide?".into())).unwrap();
    let got = wait_for(&s.events, updated(1, TaskStatus::Done));
    let usage = last_usage(&got);
    assert_eq!((usage.agent.calls, usage.agent.input, usage.agent.cost, usage.partial), (1, 2000, 0.0, true));
    s.meeting.stop().unwrap();
}
