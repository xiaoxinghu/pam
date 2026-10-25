//! A whole meeting on the fake backends: threads, events and the folder.

use std::path::PathBuf;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use crossbeam_channel::Receiver;
use pam_core::folder::EventLog;
use pam_core::llm::{self, Completion, Request};
use pam_core::usage::{Call, Used};
use pam_core::{Command, Config, CoreEvent, Llm, Meeting, MeetingConfig, SpkId, UndoId, Usage};
use serde_json::{Value, json};

fn fixture() -> PathBuf {
    fixture_named("standup")
}

fn fixture_named(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(format!("../../fixtures/{name}.fake.jsonl"))
}

/// Every event until the input ends.
fn until_ended(meeting: &Meeting) -> Vec<CoreEvent> {
    let events = meeting.subscribe();
    let mut got = Vec::new();
    loop {
        let e = events.recv_timeout(Duration::from_secs(10)).expect("meeting stalled");
        if let CoreEvent::Error(e) = &e {
            panic!("{e}");
        }
        let ended = matches!(e, CoreEvent::InputEnded { .. });
        got.push(e);
        if ended {
            return got;
        }
    }
}

fn read_json(path: PathBuf) -> Value {
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

#[test]
fn replaying_the_fixture_writes_the_folder() {
    let root = tempfile::tempdir().unwrap();
    let mut cfg = MeetingConfig::fake(root.path().to_path_buf(), fixture()).unwrap();
    cfg.title = "Stand-up".into();
    let meeting = Meeting::start(cfg).unwrap();
    let events = meeting.subscribe();
    let mut finals = 0;
    loop {
        match events.recv_timeout(Duration::from_secs(10)).expect("meeting stalled") {
            CoreEvent::LineFinal(_) => finals += 1,
            CoreEvent::InputEnded { now } => {
                assert_eq!(now, 42 * 16_000);
                break;
            }
            CoreEvent::Error(e) => panic!("{e}"),
            _ => {}
        }
    }
    let report = meeting.stop().unwrap();
    assert_eq!((report.lines, finals), (10, 10));
    assert!(report.folder.file_name().unwrap().to_string_lossy().ends_with("-stand-up"));

    let transcript = std::fs::read_to_string(report.folder.join("transcript.jsonl")).unwrap();
    let lines: Vec<Value> = transcript.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(lines.len(), 10);
    assert_eq!(
        lines[0],
        serde_json::json!({"id": "u0001", "t0": 17920, "t1": 81920, "slot": 0,
            "text": "Okay, let's get started. Quick intros for the bot. I'm Alice, I run product."})
    );
    assert_eq!(lines[9]["text"], "Sounds good to me.");
    assert_eq!(
        lines.iter().map(|l| l["slot"].as_u64().unwrap()).collect::<Vec<_>>(),
        [0, 1, 2, 0, 0, 1, 2, 1, 0, 2]
    );

    let speakers = read_json(report.folder.join("speakers.json"));
    assert_eq!(speakers["slots"], json!({"0": "spk_0", "1": "spk_1", "2": "spk_2"}));
    // "Marcus, engineering." is left to the scribe.
    assert_eq!(speakers["names"], json!({"spk_0": "Alice", "spk_2": "Jordan"}));

    let info = read_json(report.folder.join("meeting.json"));
    assert_eq!(info["status"], "stopped");
    assert_eq!(info["title"], "Stand-up");
    assert_eq!(info["duration"], 42 * 16_000);
    assert!(info["asr"].as_str().unwrap().starts_with("fake:"));
}

#[test]
fn stopping_early_still_makes_every_line_final() {
    let root = tempfile::tempdir().unwrap();
    let mut cfg = MeetingConfig::fake(root.path().to_path_buf(), fixture()).unwrap();
    cfg.realtime = true;
    let meeting = Meeting::start(cfg).unwrap();
    std::thread::sleep(Duration::from_millis(1500));
    let late = meeting.subscribe();
    let report = meeting.stop().unwrap();
    assert!(report.duration < 3 * 16_000, "heard {}", report.duration);
    // Alice's first words were cut off mid-line, and still became one final line.
    assert_eq!(report.lines, 1);
    let text = std::fs::read_to_string(report.folder.join("transcript.jsonl")).unwrap();
    assert!(text.contains("\"text\":\"Okay, let's\""), "{text}");
    let ended = late.iter().any(|e| matches!(e, CoreEvent::InputEnded { .. }));
    assert!(ended);
}

#[test]
fn dropping_a_meeting_without_stop_still_closes_its_folder() {
    let root = tempfile::tempdir().unwrap();
    let mut cfg = MeetingConfig::fake(root.path().to_path_buf(), fixture()).unwrap();
    cfg.realtime = true;
    let meeting = Meeting::start(cfg).unwrap();
    let folder = meeting.folder().to_path_buf();
    std::thread::sleep(Duration::from_millis(1500));
    drop(meeting);
    let info = read_json(folder.join("meeting.json"));
    assert_eq!(info["status"], "stopped");
    assert!(info["duration"].as_u64().unwrap() < 3 * 16_000);
    let text = std::fs::read_to_string(folder.join("transcript.jsonl")).unwrap();
    assert!(text.contains("\"text\":\"Okay, let's\""), "{text}");
}

#[test]
fn a_backend_failure_fails_stop_after_closing_the_folder() {
    let root = tempfile::tempdir().unwrap();
    let script = root.path().join("failing.fake.jsonl");
    let text = std::fs::read_to_string(fixture()).unwrap();
    let text = text.replace("\"duration\": 42}", "\"duration\": 42, \"fail_after\": 12}");
    std::fs::write(&script, text).unwrap();
    let meeting = Meeting::start(MeetingConfig::fake(root.path().join("meetings"), script).unwrap()).unwrap();
    let folder = meeting.folder().to_path_buf();
    let events = meeting.subscribe();
    let mut errors = Vec::new();
    loop {
        match events.recv_timeout(Duration::from_secs(10)).expect("meeting stalled") {
            CoreEvent::Error(e) => errors.push(e),
            CoreEvent::InputEnded { .. } => break,
            _ => {}
        }
    }
    let err = meeting.stop().expect_err("a backend failed, so stop must fail");
    assert!(format!("{err:#}").contains("fail after"), "{err:#}");
    assert_eq!(errors.len(), 1, "{errors:?}");
    // The folder is still closed, with the lines heard before the failure.
    let info = read_json(folder.join("meeting.json"));
    assert_eq!(info["status"], "stopped");
    assert_eq!(info["duration"], 42 * 16_000);
    let transcript = std::fs::read_to_string(folder.join("transcript.jsonl")).unwrap();
    assert!(transcript.contains("Marcus, engineering."), "{transcript}");
    assert!(!transcript.contains("Then let's prototype"), "{transcript}");
}

#[test]
fn an_event_whose_write_failed_is_never_sent() {
    use std::os::unix::fs::PermissionsExt;
    let root = tempfile::tempdir().unwrap();
    let script = root.path().join("one.fake.jsonl");
    std::fs::write(&script, "{\"config\": {\"duration\": 2}}\n{\"span\": 0, \"start\": 0.5, \"end\": 1.0}\n")
        .unwrap();
    let mut cfg = MeetingConfig::fake(root.path().join("meetings"), script).unwrap();
    cfg.realtime = true;
    let meeting = Meeting::start(cfg).unwrap();
    let folder = meeting.folder().to_path_buf();
    // speakers.json is replaced through a rename, which a read-only folder refuses.
    std::fs::set_permissions(&folder, std::fs::Permissions::from_mode(0o555)).unwrap();
    let events = meeting.subscribe();
    let mut got = Vec::new();
    while let Ok(e) = events.recv_timeout(Duration::from_secs(3)) {
        got.push(e);
    }
    let stopped = meeting.stop();
    std::fs::set_permissions(&folder, std::fs::Permissions::from_mode(0o755)).unwrap();
    assert!(stopped.is_err());
    assert!(got.iter().any(|e| matches!(e, CoreEvent::Error(m) if m.contains("speakers.json"))), "{got:?}");
    assert!(!got.iter().any(|e| matches!(e, CoreEvent::SpeakerAdded(_))), "{got:?}");
}

#[test]
fn introductions_name_speakers_and_undo_reverts_them() {
    let root = tempfile::tempdir().unwrap();
    let mut cfg = MeetingConfig::fake(root.path().to_path_buf(), fixture_named("intros")).unwrap();
    cfg.attendees = vec!["Sam".into(), "Priya".into()];
    let meeting = Meeting::start(cfg).unwrap();
    let events = until_ended(&meeting);

    // Each rename comes right after the line that introduced the speaker.
    let mut renames = Vec::new();
    for (i, e) in events.iter().enumerate() {
        if let CoreEvent::SpeakerRenamed { spk, name, undo } = e {
            let CoreEvent::LineFinal(line) = &events[i - 1] else { panic!("{:?}", events[i - 1]) };
            assert_eq!(line.spk, Some(*spk));
            renames.push((line.id.to_string(), spk.to_string(), name.as_str(), undo.to_string()));
        }
    }
    let r =
        |line: &str, spk: &str, name, undo: &str| (line.to_string(), spk.to_string(), name, undo.to_string());
    assert_eq!(
        renames,
        [
            r("u0001", "spk_0", "Alice", "x1"),
            r("u0002", "spk_1", "Marcus", "x2"),
            r("u0003", "spk_2", "Jordan", "x3"),
            r("u0007", "spk_3", "Sam", "x4"),
        ]
    );

    let folder = meeting.folder().to_path_buf();
    let speakers = read_json(folder.join("speakers.json"));
    assert_eq!(
        speakers["names"],
        json!({"spk_0": "Alice", "spk_1": "Marcus", "spk_2": "Jordan", "spk_3": "Sam"})
    );
    assert_eq!(speakers["log"][3]["line"], "u0007");
    assert_eq!(speakers["log"][3]["by"], "intro");
    let md = std::fs::read_to_string(folder.join("transcript.md")).unwrap();
    assert!(md.contains("`u0005` 00:16 **Sam:** Sorry I'm late. Did I miss the intros?"), "{md}");
    assert!(md.contains("**Marcus:** This is Figma"), "{md}");
    // context.md keeps up with every line and rename, scribe or not.
    let context = std::fs::read_to_string(folder.join("context.md")).unwrap();
    assert!(context.contains("spk_0 Alice · spk_1 Marcus · spk_2 Jordan · spk_3 Sam\n"), "{context}");
    assert!(context.contains("- u0005 00:16 Sam: Sorry I'm late. Did I miss the intros?\n"), "{context}");

    // Undo still works after the input ended, and its event is out before send returns.
    let late = meeting.subscribe();
    meeting.send(Command::Undo(UndoId(4))).unwrap();
    let undone = late.try_iter().find(|e| matches!(e, CoreEvent::Undone { .. })).expect("no Undone event");
    let CoreEvent::Undone { undo, overlay } = undone else { unreachable!() };
    assert_eq!(undo, UndoId(4));
    assert_eq!(overlay.name(SpkId(3)), None);
    assert_eq!(overlay.name(SpkId(0)), Some("Alice"));

    let speakers = read_json(folder.join("speakers.json"));
    assert_eq!(speakers["names"], json!({"spk_0": "Alice", "spk_1": "Marcus", "spk_2": "Jordan"}));
    assert_eq!(speakers["log"].as_array().unwrap().len(), 3);
    let md = std::fs::read_to_string(folder.join("transcript.md")).unwrap();
    assert!(md.contains("`u0005` 00:16 **spk_3:** Sorry I'm late."), "{md}");
    let context = std::fs::read_to_string(folder.join("context.md")).unwrap();
    assert!(context.contains("spk_2 Jordan · spk_3\n"), "{context}");
    assert!(context.contains("- u0005 00:16 spk_3: Sorry I'm late."), "{context}");

    let again = meeting.send(Command::Undo(UndoId(4))).unwrap_err();
    assert_eq!(again.to_string(), "x4 was already undone");

    let report = meeting.stop().unwrap();
    let info = read_json(report.folder.join("meeting.json"));
    assert_eq!(info["attendees"], json!(["Sam", "Priya"]));
}

#[test]
fn a_late_subscriber_sees_renames_and_undos() {
    let root = tempfile::tempdir().unwrap();
    let cfg = MeetingConfig::fake(root.path().to_path_buf(), fixture_named("intros")).unwrap();
    let meeting = Meeting::start(cfg).unwrap();
    until_ended(&meeting);
    meeting.send(Command::Undo(UndoId(2))).unwrap();
    let late: Vec<CoreEvent> = meeting.subscribe().try_iter().collect();
    let renamed = late.iter().filter(|e| matches!(e, CoreEvent::SpeakerRenamed { .. })).count();
    assert_eq!(renamed, 4);
    assert!(late.iter().any(|e| matches!(e, CoreEvent::Undone { undo: UndoId(2), .. })));
    meeting.stop().unwrap();
}

#[test]
fn corrections_fix_a_split_voice_and_a_wrong_line_and_undo_reverts_them() {
    let root = tempfile::tempdir().unwrap();
    let cfg = MeetingConfig::fake(root.path().to_path_buf(), fixture_named("corrections")).unwrap();
    let meeting = Meeting::start(cfg).unwrap();
    let events = until_ended(&meeting);
    let lines: Vec<(String, Option<SpkId>)> = events
        .iter()
        .filter_map(|e| match e {
            CoreEvent::LineFinal(l) => Some((l.id.to_string(), l.spk)),
            _ => None,
        })
        .collect();
    assert_eq!(lines.len(), 9, "{lines:?}");
    assert_eq!(lines[4], ("u0005".into(), Some(SpkId(3))), "the split voice");
    assert_eq!(lines[6], ("u0007".into(), Some(SpkId(0))), "the wrong line");
    let folder = meeting.folder().to_path_buf();
    let md = || std::fs::read_to_string(folder.join("transcript.md")).unwrap();
    assert!(md().contains("**spk_3:** We ship Thursday"), "{}", md());
    assert!(md().contains("**Alice:** Sure, I'll send them tonight."), "{}", md());

    let watch = meeting.subscribe();
    let _ = watch.try_iter().count();
    meeting.send(Command::MergeSpeakers { from: SpkId(3), into: SpkId(1) }).unwrap();
    meeting.send(Command::MoveLine { line: "u0007".parse().unwrap(), to: SpkId(2) }).unwrap();
    assert_eq!(
        watch.try_iter().collect::<Vec<_>>(),
        [
            CoreEvent::SpeakersMerged { from: SpkId(3), into: SpkId(1), undo: UndoId(4) },
            CoreEvent::LineMoved { line: "u0007".parse().unwrap(), to: SpkId(2), undo: UndoId(5) },
        ]
    );
    let md_fixed = md();
    assert!(md_fixed.contains("**Marcus:** We ship Thursday"), "{md_fixed}");
    assert!(md_fixed.contains("**Marcus:** I'll review them"), "{md_fixed}");
    assert!(md_fixed.contains("**Jordan:** Sure, I'll send them tonight."), "{md_fixed}");
    let speakers = read_json(folder.join("speakers.json"));
    assert_eq!(speakers["merges"], json!({"spk_3": "spk_1"}));
    assert_eq!(speakers["moves"], json!({"u0007": "spk_2"}));
    assert_eq!(speakers["log"][4]["op"], "move");

    // Errors say why, and change nothing.
    let e = meeting.send(Command::MoveLine { line: "u0042".parse().unwrap(), to: SpkId(2) }).unwrap_err();
    assert_eq!(e.to_string(), "no line u0042: lines run u0001..u0009");
    let e = meeting.send(Command::MergeSpeakers { from: SpkId(1), into: SpkId(3) }).unwrap_err();
    assert_eq!(e.to_string(), "they are already one speaker, Marcus");
    assert!(watch.try_iter().next().is_none());

    // A late subscriber gets the corrections too.
    let late: Vec<CoreEvent> = meeting.subscribe().try_iter().collect();
    assert!(late.iter().any(|e| matches!(e, CoreEvent::SpeakersMerged { undo: UndoId(4), .. })));
    assert!(late.iter().any(|e| matches!(e, CoreEvent::LineMoved { undo: UndoId(5), .. })));

    meeting.send(Command::Undo(UndoId(5))).unwrap();
    assert!(md().contains("**Alice:** Sure, I'll send them tonight."), "{}", md());
    meeting.send(Command::Undo(UndoId(4))).unwrap();
    assert!(md().contains("**spk_3:** We ship Thursday"), "{}", md());
    let speakers = read_json(folder.join("speakers.json"));
    assert_eq!((&speakers["merges"], &speakers["moves"]), (&json!({}), &json!({})));

    meeting.send(Command::RenameSpeaker { spk: SpkId(3), name: "Marcus".into() }).unwrap();
    assert!(md().contains("**Marcus:** We ship Thursday"), "{}", md());
    meeting.stop().unwrap();
}

/// A stand-in for the cloud model. Every new line becomes a `point` crediting
/// its speaker, and lines that introduce someone become `intro`s. It keeps
/// the prompts it got, fails its first `fail` runs, and waits for `gate`
/// before each reply when there is one.
#[derive(Default)]
struct Stub {
    prompts: Mutex<Vec<String>>,
    fail: AtomicUsize,
    /// Runs left that reply with text that isn't the scribe's JSON.
    unreadable: AtomicUsize,
    gate: Option<Mutex<Receiver<()>>>,
}

/// What each of the stub's successful runs costs.
const STUB_CALL: Call = Call { input: 100, output: 20, cache_read: 0, cache_write: 0, cost: Some(0.0015) };

impl llm::Model for Stub {
    fn complete(&self, req: &Request<'_>) -> Completion {
        let reply = self.reply(req.system, req.prompt);
        // A failed run reported nothing, as when pi crashes mid-call.
        let used = if reply.is_ok() { Used::Call(STUB_CALL) } else { Used::Unknown };
        Completion { reply, used }
    }
}

impl Stub {
    fn reply(&self, _system: &str, prompt: &str) -> anyhow::Result<String> {
        if let Some(gate) = &self.gate {
            gate.lock().unwrap().recv()?;
        }
        self.prompts.lock().unwrap().push(prompt.to_string());
        if self.unreadable.load(Ordering::SeqCst) > 0 {
            self.unreadable.fetch_sub(1, Ordering::SeqCst);
            return Ok("Sorry, I can't help with that.".into());
        }
        if self.fail.load(Ordering::SeqCst) > 0 {
            self.fail.fetch_sub(1, Ordering::SeqCst);
            anyhow::bail!("model unavailable");
        }
        let mut events = Vec::new();
        for line in new_lines(prompt) {
            // "u0002 00:05 spk_1: Marcus, engineering."
            let (head, text) = line.split_once(": ").unwrap();
            let words: Vec<&str> = head.split(' ').collect();
            let (id, spk) = (words[0], words[2]);
            events.push(json!({"type": "point", "by": [spk], "text": text, "refs": [id]}));
            for name in ["Alice", "Marcus"] {
                if text.contains(&format!("I'm {name}")) || text.starts_with(&format!("{name},")) {
                    events.push(json!({"type": "intro", "by": [spk], "name": name, "refs": [id]}));
                }
            }
        }
        let summary = format!("{} runs so far.", self.prompts.lock().unwrap().len());
        Ok(format!("```json\n{}\n```", json!({"events": events, "summary": summary})))
    }
}

fn new_lines(prompt: &str) -> Vec<&str> {
    prompt.split("NEW LINES\n").nth(1).unwrap().lines().collect()
}

fn with_scribe(stub: &Arc<Stub>) -> MeetingConfig {
    let root = tempfile::tempdir().unwrap().keep();
    let mut cfg = MeetingConfig::fake(root, fixture()).unwrap();
    cfg.title = "Stand-up".into();
    cfg.scribe = Some(Llm { name: "stub".into(), model: stub.clone() });
    cfg.scribe_info = Some(
        serde_json::from_value(json!({"status": "ready", "model": "stub", "source": "config"})).unwrap(),
    );
    cfg
}

#[test]
fn the_scribe_turns_lines_into_events_notes_and_names() {
    let stub = Arc::new(Stub::default());
    let meeting = Meeting::start(with_scribe(&stub)).unwrap();
    let events = meeting.subscribe();
    until_ended(&meeting);
    let report = meeting.stop().unwrap();
    let got: Vec<CoreEvent> = events.iter().collect();
    let folder = report.folder;
    let report_failure = report.notes_failure;

    // "Marcus, engineering." is an intro only the scribe catches.
    let renamed: Vec<&CoreEvent> =
        got.iter().filter(|e| matches!(e, CoreEvent::SpeakerRenamed { .. })).collect();
    assert_eq!(renamed.len(), 3, "{renamed:?}");
    assert_eq!(
        renamed.iter().find(|e| matches!(e, CoreEvent::SpeakerRenamed { spk: SpkId(1), .. })),
        Some(&&CoreEvent::SpeakerRenamed { spk: SpkId(1), name: "Marcus".into(), undo: UndoId(3) })
    );
    let speakers = read_json(folder.join("speakers.json"));
    assert_eq!(speakers["names"], json!({"spk_0": "Alice", "spk_1": "Marcus", "spk_2": "Jordan"}));
    assert_eq!(
        speakers["log"][2],
        json!({"undo": "x3", "op": "rename", "spk": "spk_1", "name": "Marcus",
        "by": "scribe", "line": "u0002", "at": speakers["log"][2]["at"]})
    );

    // Every line is read exactly once, across however many runs it took.
    let prompts = stub.prompts.lock().unwrap();
    let read: Vec<&str> = prompts.iter().flat_map(|p| new_lines(p)).map(|l| &l[..5]).collect();
    assert_eq!(read, (1..=10).map(|i| format!("u{i:04}")).collect::<Vec<_>>());

    let saved: Vec<Value> = std::fs::read_to_string(folder.join("events.jsonl"))
        .unwrap()
        .lines()
        .map(|l| serde_json::from_str(l).unwrap())
        .collect();
    assert_eq!(saved.len(), 12);
    assert_eq!(
        saved[0],
        json!({"id": "e001", "type": "point", "t": 17920, "by": ["spk_0"],
        "text": "Okay, let's get started. Quick intros for the bot. I'm Alice, I run product", "refs": ["u0001"]})
    );
    // Alice's intro confirms the fast path's rename; Marcus's made one.
    assert_eq!(
        (&saved[1]["type"], &saved[1]["name"], &saved[1]["rename"]),
        (&json!("intro"), &json!("Alice"), &json!("x1"))
    );
    assert_eq!((&saved[3]["name"], &saved[3]["rename"]), (&json!("Marcus"), &json!("x3")));

    let notes = std::fs::read_to_string(folder.join("notes.md")).unwrap();
    assert!(notes.starts_with("# Notes: Stand-up\n"), "{notes}");
    assert!(notes.contains("- Marcus, engineering (Marcus)\n"), "{notes}");
    assert!(notes.contains("- Sounds good to me (Jordan)\n"), "{notes}");
    let runs = prompts.len();
    assert!(notes.contains(&format!("## Summary\n\n{runs} runs so far.\n")), "{notes}");
    assert_eq!(read_json(folder.join("summary.json"))["through"], "u0010");
    assert_eq!(read_json(folder.join("meeting.json"))["scribe"]["model"], "stub");
    assert_eq!(report_failure, None, "every line made it into the notes");
    let context = std::fs::read_to_string(folder.join("context.md")).unwrap();
    assert!(context.starts_with("# Meeting: Stand-up · 00:42 elapsed · assistant name: Pam\n"), "{context}");
    assert!(
        context.contains("\n- e004 00:06 intro Marcus: Marcus introduced themselves [u0002]\n"),
        "{context}"
    );
    assert!(context.contains("\n- u0010 00:38 Jordan: Sounds good to me.\n"), "{context}");

    let Some(CoreEvent::NotesUpdated(last)) =
        got.iter().rev().find(|e| matches!(e, CoreEvent::NotesUpdated(_)))
    else {
        panic!("no NotesUpdated");
    };
    assert_eq!(last.sections[0].items.len(), 10);
    assert_eq!(last.summary, format!("{runs} runs so far."));

    // Each run's call counts: the meeting starts at nothing, and
    // meeting.json has the last count after Stop.
    let usages: Vec<&Usage> =
        got.iter().filter_map(|e| if let CoreEvent::Usage(u) = e { Some(&**u) } else { None }).collect();
    assert_eq!(usages[0], &Usage::default(), "sent at the start");
    let usage: Usage =
        serde_json::from_value(read_json(folder.join("meeting.json"))["usage"].clone()).unwrap();
    assert_eq!(usages.last().unwrap(), &&usage);
    assert_eq!(
        (usage.scribe.calls, usage.scribe.input, usage.total.calls),
        (runs as u64, 100 * runs as u64, runs as u64)
    );
    assert!((usage.scribe.cost - 0.0015 * runs as f64).abs() < 1e-9, "{usage:?}");
    assert!(!usage.partial);
}

#[test]
fn lines_never_wait_for_the_scribe() {
    let (open, gate) = crossbeam_channel::unbounded();
    let stub = Arc::new(Stub { gate: Some(Mutex::new(gate)), ..Stub::default() });
    let meeting = Meeting::start(with_scribe(&stub)).unwrap();
    // The model hasn't answered once, and still every line is final.
    let events = until_ended(&meeting);
    assert_eq!(events.iter().filter(|e| matches!(e, CoreEvent::LineFinal(_))).count(), 10);
    assert!(!events.iter().any(|e| matches!(e, CoreEvent::NotesUpdated(_))));
    assert!(stub.prompts.lock().unwrap().is_empty());
    for _ in 0..10 {
        open.send(()).unwrap();
    }
    // Stopping waits for the notes to take in the last line.
    let report = meeting.stop().unwrap();
    assert!(stub.prompts.lock().unwrap().len() >= 2, "the first run went out before the end");
    assert_eq!(read_json(report.folder.join("summary.json"))["through"], "u0010");
}

#[test]
fn a_failed_scribe_run_is_reported_and_the_lines_wait_for_the_next() {
    let stub = Arc::new(Stub { fail: AtomicUsize::new(1), ..Stub::default() });
    let meeting = Meeting::start(with_scribe(&stub)).unwrap();
    let events = meeting.subscribe();
    let mut got = Vec::new();
    while !matches!(got.last(), Some(CoreEvent::InputEnded { .. })) {
        got.push(events.recv_timeout(Duration::from_secs(10)).unwrap());
    }
    let report = meeting.stop().unwrap();
    let errors: Vec<String> = got
        .into_iter()
        .chain(events.iter())
        .filter_map(|e| if let CoreEvent::Error(e) = e { Some(e) } else { None })
        .collect();
    assert_eq!(errors, ["scribe: model unavailable"]);
    // The failed run reported nothing, so the cost is partial.
    let usage = &read_json(report.folder.join("meeting.json"))["usage"];
    assert_eq!(usage["partial"], true);
    assert_eq!(usage["scribe"]["calls"].as_u64(), Some(stub.prompts.lock().unwrap().len() as u64 - 1));
    // The last try on stopping covers the lines the failed run had.
    let prompts = stub.prompts.lock().unwrap();
    assert!(new_lines(prompts.last().unwrap())[0].starts_with("u0001"), "{prompts:?}");
    assert_eq!(read_json(report.folder.join("summary.json"))["through"], "u0010");
}

#[test]
fn a_reply_that_cant_be_read_keeps_its_cost_and_isnt_partial() {
    let stub = Arc::new(Stub { unreadable: AtomicUsize::new(1), ..Stub::default() });
    let meeting = Meeting::start(with_scribe(&stub)).unwrap();
    let events = meeting.subscribe();
    until_ended(&meeting);
    let report = meeting.stop().unwrap();
    let errors: Vec<String> =
        events.iter().filter_map(|e| if let CoreEvent::Error(e) = e { Some(e) } else { None }).collect();
    assert_eq!(errors.len(), 1, "{errors:?}");
    // pi reported the unreadable run's call too, so the cost is whole.
    let usage = &read_json(report.folder.join("meeting.json"))["usage"];
    assert_eq!(usage["scribe"]["calls"].as_u64(), Some(stub.prompts.lock().unwrap().len() as u64));
    assert_eq!(usage["partial"], false);
}

#[test]
fn a_note_added_from_outside_reaches_the_subscribers() {
    let root = tempfile::tempdir().unwrap();
    let meeting = Meeting::start(MeetingConfig::fake(root.path().to_path_buf(), fixture()).unwrap()).unwrap();
    until_ended(&meeting);
    let events = meeting.subscribe();
    // What `pam note` does.
    let mut log = EventLog::lock(meeting.folder()).unwrap();
    let note = log.append(vec![note_event("Ship on Friday")]).unwrap();
    drop(log);
    assert_eq!(note[0].id.to_string(), "e001");
    let notes = loop {
        if let CoreEvent::NotesUpdated(n) =
            events.recv_timeout(Duration::from_secs(5)).expect("no NotesUpdated")
        {
            break n;
        }
    };
    assert_eq!(notes.sections[4].title, "Notes");
    assert_eq!(notes.sections[4].items[0].text, "Ship on Friday");
    // A subscriber that joins now still gets the latest notes.
    let late: Vec<CoreEvent> = meeting.subscribe().try_iter().collect();
    assert!(late.iter().any(|e| matches!(e, CoreEvent::NotesUpdated(n) if n == &notes)), "{late:?}");
    meeting.stop().unwrap();
}

fn note_event(text: &str) -> pam_core::record::Event {
    serde_json::from_value(json!({"id": "e000", "type": "note", "t": 0, "text": text})).unwrap()
}

#[test]
fn a_broken_event_from_outside_is_reported_once_and_the_meeting_goes_on() {
    let root = tempfile::tempdir().unwrap();
    let cfg = MeetingConfig::fake(root.path().to_path_buf(), fixture_named("intros")).unwrap();
    let meeting = Meeting::start(cfg).unwrap();
    until_ended(&meeting);
    let events = meeting.subscribe();
    let path = meeting.folder().join("events.jsonl");
    std::fs::write(&path, "{not json\n").unwrap();
    let error = loop {
        if let CoreEvent::Error(e) = events.recv_timeout(Duration::from_secs(5)).expect("no Error") {
            break e;
        }
    };
    assert!(error.contains("events.jsonl"), "{error}");
    std::thread::sleep(Duration::from_millis(600));
    assert!(!events.try_iter().any(|e| matches!(e, CoreEvent::Error(_))), "reported once");
    meeting.send(Command::Undo(UndoId(1))).unwrap();
    meeting.stop().unwrap();
}

#[test]
fn stop_says_which_lines_the_notes_leave_out_when_the_last_run_fails() {
    let stub = Arc::new(Stub { fail: AtomicUsize::new(usize::MAX), ..Stub::default() });
    let meeting = Meeting::start(with_scribe(&stub)).unwrap();
    let events = meeting.subscribe();
    while !matches!(events.recv_timeout(Duration::from_secs(10)).unwrap(), CoreEvent::InputEnded { .. }) {}
    // The meeting is saved all the same; only the notes are short.
    let report = meeting.stop().unwrap();
    assert_eq!(report.lines, 10);
    assert_eq!(
        report.notes_failure.as_deref(),
        Some("notes.md leaves out u0001..u0010: scribe: model unavailable")
    );
}

#[test]
fn stop_renders_the_notes_once_more_and_reports_if_it_cannot() {
    let root = tempfile::tempdir().unwrap();
    let meeting = Meeting::start(MeetingConfig::fake(root.path().to_path_buf(), fixture()).unwrap()).unwrap();
    until_ended(&meeting);
    let notes = meeting.folder().join("notes.md");
    // A directory in its place makes replacing notes.md fail.
    std::fs::remove_file(&notes).unwrap();
    std::fs::create_dir_all(notes.join("blocked")).unwrap();
    let report = meeting.stop().unwrap();
    let why = report.notes_failure.expect("the failed render is reported");
    assert!(why.starts_with("notes.md couldn't be written: writing notes.md"), "{why}");
}

#[test]
fn stop_repairs_notes_md_when_it_can() {
    let root = tempfile::tempdir().unwrap();
    let meeting = Meeting::start(MeetingConfig::fake(root.path().to_path_buf(), fixture()).unwrap()).unwrap();
    until_ended(&meeting);
    let notes = meeting.folder().join("notes.md");
    std::fs::remove_file(&notes).unwrap();
    let report = meeting.stop().unwrap();
    assert_eq!(report.notes_failure, None);
    assert!(std::fs::read_to_string(notes).unwrap().starts_with("# Notes\n"));
}

#[test]
fn stop_renders_context_md_with_the_final_duration_and_reports_if_it_cannot() {
    let root = tempfile::tempdir().unwrap();
    let meeting = Meeting::start(MeetingConfig::fake(root.path().to_path_buf(), fixture()).unwrap()).unwrap();
    until_ended(&meeting);
    let context = meeting.folder().join("context.md");
    std::fs::remove_file(&context).unwrap();
    let report = meeting.stop().unwrap();
    assert_eq!(report.context_failure, None);
    let md = std::fs::read_to_string(&context).unwrap();
    assert!(md.starts_with("# Meeting · 00:42 elapsed · assistant name: Pam\n"), "{md}");

    let meeting = Meeting::start(MeetingConfig::fake(root.path().to_path_buf(), fixture()).unwrap()).unwrap();
    until_ended(&meeting);
    let context = meeting.folder().join("context.md");
    // A directory in its place makes replacing context.md fail.
    std::fs::remove_file(&context).unwrap();
    std::fs::create_dir_all(context.join("blocked")).unwrap();
    let why = meeting.stop().unwrap().context_failure.expect("the failed render is reported");
    assert!(why.starts_with("writing context.md"), "{why}");
}

#[test]
fn stop_ends_even_when_the_last_run_cannot_be_built() {
    let stub = Arc::new(Stub { fail: AtomicUsize::new(usize::MAX), ..Stub::default() });
    let meeting = Meeting::start(with_scribe(&stub)).unwrap();
    let events = meeting.subscribe();
    while !matches!(events.recv_timeout(Duration::from_secs(10)).unwrap(), CoreEvent::InputEnded { .. }) {}
    // Lines are unread, and the last run's prompt can't be built.
    std::fs::write(meeting.folder().join("events.jsonl"), "{not json\n").unwrap();
    let (done, stopped) = crossbeam_channel::bounded(1);
    std::thread::spawn(move || done.send(meeting.stop()).unwrap());
    let report = stopped.recv_timeout(Duration::from_secs(10)).expect("stop hung").unwrap();
    let why = report.notes_failure.unwrap();
    assert!(why.starts_with("notes.md leaves out u0001..u0010: scribe: reading"), "{why}");
    assert!(why.contains("events.jsonl"), "{why}");
}

#[test]
fn a_replayed_recording_goes_to_the_backends_from_the_config_and_into_audio_wav() {
    let root = tempfile::tempdir().unwrap();
    // 42 s of a tone, the stand-up script's length; the fake backends only count samples.
    let wav = root.path().join("standup.wav");
    let spec = hound::WavSpec {
        channels: 2,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(&wav, spec).unwrap();
    let tone = |i: u64| ((i as f32 / 20.0).sin() * 8_000.0) as i16;
    for i in 0..42 * 16_000 {
        w.write_sample(tone(i)).unwrap();
        w.write_sample(tone(i)).unwrap();
    }
    w.finalize().unwrap();
    let backend = format!("fake:{}", fixture().display());
    let config = Config { asr: backend.clone(), diarizer: backend, ..Config::default() };

    let meeting =
        Meeting::start(MeetingConfig::wav(root.path().join("meetings"), wav.clone(), &config)).unwrap();
    let finals = until_ended(&meeting).iter().filter(|e| matches!(e, CoreEvent::LineFinal(_))).count();
    let report = meeting.stop().unwrap();
    assert_eq!((report.lines, finals, report.duration), (10, 10, 42 * 16_000));

    let info = read_json(report.folder.join("meeting.json"));
    assert_eq!(info["input"], wav.display().to_string());
    assert!(info["asr"].as_str().unwrap().starts_with("fake:"));
    let mut audio = hound::WavReader::open(report.folder.join("audio.wav")).unwrap();
    assert_eq!((audio.spec().channels, audio.spec().sample_rate, audio.duration()), (1, 16_000, 42 * 16_000));
    let heard: Vec<i16> = audio.samples::<i16>().map(Result::unwrap).collect();
    assert!(heard.iter().enumerate().all(|(i, s)| (*s - tone(i as u64)).abs() <= 1));
}

#[test]
fn the_speaking_indicator_follows_how_loud_the_recording_is() {
    let root = tempfile::tempdir().unwrap();
    // A quiet room (-80 dBFS) for the stand-up's 42 s, and a voice 20 dB over it only
    // from 21 to 24.5 s, inside Marcus's turn (20.48–25.52 s).
    let wav = root.path().join("room.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(&wav, spec).unwrap();
    let mut seed = 1u32;
    for i in 0..42 * 16_000 {
        seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
        let noise = (seed >> 16) as f32 / 32_768.0 - 1.0;
        let voice = (21 * 16_000..24 * 16_000 + 8_000).contains(&i);
        let amp = if voice { 0.001 } else { 0.000_1 } * 32_768.0 * 3f32.sqrt();
        w.write_sample((noise * amp) as i16).unwrap();
    }
    w.finalize().unwrap();
    let backend = format!("fake:{}", fixture().display());
    let config = Config { asr: backend.clone(), diarizer: backend, ..Config::default() };

    let meeting = Meeting::start(MeetingConfig::wav(root.path().join("meetings"), wav, &config)).unwrap();
    let events = until_ended(&meeting);
    meeting.stop().unwrap();
    let lit: Vec<(SpkId, f32)> = events
        .iter()
        .filter_map(|e| match e {
            CoreEvent::SpeakerActive { spk, level } if *level > 0.0 => Some((*spk, *level)),
            _ => None,
        })
        .collect();
    // Only Marcus lights up, and only while the voice is loud: about 35 chunks.
    assert!(lit.iter().all(|(spk, _)| *spk == SpkId(1)), "{lit:?}");
    assert!((30..=36).contains(&lit.len()), "{} chunks: {lit:?}", lit.len());
    // 20 dB over the room is a little over half level.
    assert!(lit.iter().all(|(_, l)| (0.4..0.7).contains(l)), "{lit:?}");
}

#[test]
fn a_recording_that_isnt_16_khz_is_refused_before_any_folder_is_made() {
    let root = tempfile::tempdir().unwrap();
    let wav = root.path().join("cd.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44_100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    hound::WavWriter::create(&wav, spec).unwrap().finalize().unwrap();
    let meetings = root.path().join("meetings");
    let err = Meeting::start(MeetingConfig::wav(meetings.clone(), wav, &Config::default())).err().unwrap();
    assert!(err.to_string().contains("44100 Hz"), "{err:#}");
    assert!(!meetings.exists());
}
