//! Choosing the scribe's, the intent check's and the agent's models end
//! to end: from the option, the config file or detection, checked with the
//! `stub-pi` binary before the meeting, recorded in `meeting.json` and
//! printed by `pam status`.

use std::path::{Path, PathBuf};
use std::process::Command as Process;
use std::time::{Duration, Instant};

use crossbeam_channel::Receiver;
use pam_core::models::{self, AGENT_CANDIDATES, Choice, LOCAL, Pick, SCRIBE_CANDIDATES, Source, Use};
use pam_core::{Command, CoreEvent, Meeting, MeetingConfig, ModelInfo, ModelStatus, TaskStatus};
use serde_json::Value;

/// A `pi` that runs the stub with `env` set, e.g. `STUB_PI_READY='a/b c/d'`.
fn stub_pi(dir: &Path, env: &str) -> PathBuf {
    let pi = dir.join("pi");
    std::fs::write(&pi, format!("#!/bin/sh\n{env} exec '{}' \"$@\"\n", env!("CARGO_BIN_EXE_stub-pi")))
        .unwrap();
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&pi, std::fs::Permissions::from_mode(0o755)).unwrap();
    pi
}

/// What `pam listen` would pass on, with the stub as pi and `settings` as
/// pi's settings file. The intent check follows the scribe.
fn choice(dir: &Path, env: &str, settings: &str, scribe: Pick, agent: Pick) -> Choice {
    let path = dir.join("settings.json");
    if !settings.is_empty() {
        std::fs::write(&path, settings).unwrap();
    }
    Choice { pi: Some(stub_pi(dir, env)), settings: path, scribe, intent: Pick::Detect, agent }
}

/// The user's pi default, as pi 1.1.0 writes it.
const PI_DEFAULT: &str =
    r#"{"defaultProvider": "openai", "defaultModel": "gpt-6-astra", "defaultThinkingLevel": "high"}"#;

fn named(model: &str, source: Source) -> Pick {
    Pick::Named { model: model.into(), source }
}

fn ready(env_models: &[&str]) -> String {
    format!("STUB_PI_READY='{}'", env_models.join(" "))
}

/// Replays the stand-up fixture with `choice`, to its end.
fn meeting(dir: &Path, choice: &Choice) -> (Meeting, Receiver<CoreEvent>) {
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/standup.fake.jsonl");
    let mut cfg = MeetingConfig::fake(dir.join("meetings"), script).unwrap();
    cfg.title = "Stand-up".into();
    cfg.use_models(choice, Some(env!("CARGO_BIN_EXE_pam").into()));
    let meeting = Meeting::start(cfg).unwrap();
    let events = meeting.subscribe();
    (meeting, events)
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

/// The scribe's, the intent check's and the agent's models, as the meeting
/// sent them first.
fn models_of(events: &Receiver<CoreEvent>) -> (ModelInfo, ModelInfo, ModelInfo) {
    match (events.recv().unwrap(), events.recv().unwrap(), events.recv().unwrap()) {
        (CoreEvent::Scribe(s), CoreEvent::Intent(i), CoreEvent::Agent(a)) => (s, i, a),
        other => panic!("{other:?}"),
    }
}

fn status(folder: &Path, json: bool) -> String {
    let mut cmd = Process::new(env!("CARGO_BIN_EXE_pam"));
    cmd.arg("status").arg("--meeting").arg(folder).env_remove("PAM_MEETING");
    if json {
        cmd.arg("--json");
    }
    let out = cmd.output().unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8(out.stdout).unwrap()
}

fn meeting_json(folder: &Path) -> Value {
    serde_json::from_str(&std::fs::read_to_string(folder.join("meeting.json")).unwrap()).unwrap()
}

#[test]
fn each_model_comes_from_the_option_then_the_config_file_then_detection() {
    let dir = tempfile::tempdir().unwrap();
    // The option beats the config file; the agent's config file beats detection.
    let c = choice(
        dir.path(),
        "",
        PI_DEFAULT,
        Pick::new(Use::Scribe, Some("p/fast"), Some("q/slow")).unwrap(),
        Pick::new(Use::Agent, None, Some("q/big")).unwrap(),
    );
    let (scribe, intent, agent) = models::check(&c);
    assert_eq!(
        (scribe.status, scribe.model.as_deref(), scribe.source),
        (ModelStatus::Ready, Some("p/fast"), Some(Source::Option))
    );
    // Named by nothing, the intent check uses the scribe's model.
    assert_eq!(
        (intent.status, intent.model.as_deref(), intent.source),
        (ModelStatus::Ready, Some("p/fast"), Some(Source::Detected))
    );
    assert_eq!(scribe.provider.as_deref(), Some("p"));
    assert_eq!((agent.model.as_deref(), agent.source), (Some("q/big"), Some(Source::Config)));
    assert_eq!(agent.version.as_deref(), Some("1.1.0"));
    assert!(agent.pi.as_ref().unwrap().ends_with("pi"));

    // Detection: the first ready candidate, in order; the agent's first is pi's default.
    let env =
        ready(&[SCRIBE_CANDIDATES[1], SCRIBE_CANDIDATES[2], "openai/gpt-6-astra:high", AGENT_CANDIDATES[0]]);
    let (scribe, _, agent) = models::check(&choice(dir.path(), &env, PI_DEFAULT, Pick::Detect, Pick::Detect));
    assert_eq!(
        (scribe.model.as_deref(), scribe.source),
        (Some(SCRIBE_CANDIDATES[1]), Some(Source::Detected))
    );
    assert_eq!(scribe.provider.as_deref(), Some("anthropic"));
    assert_eq!(
        (agent.model.as_deref(), agent.source),
        (Some("openai/gpt-6-astra:high"), Some(Source::Detected))
    );
    // Without a pi default, the next.
    std::fs::remove_file(dir.path().join("settings.json")).unwrap();
    let (_, _, agent) = models::check(&choice(dir.path(), &env, "", Pick::Detect, Pick::Detect));
    assert_eq!(agent.model.as_deref(), Some(AGENT_CANDIDATES[0]));
}

#[test]
fn detected_models_take_notes_and_answer_and_status_prints_both() {
    let dir = tempfile::tempdir().unwrap();
    let env = ready(&[SCRIBE_CANDIDATES[0], "openai/gpt-6-astra:high"]);
    let c = choice(dir.path(), &env, PI_DEFAULT, Pick::Detect, Pick::Detect);
    let (meeting, events) = meeting(dir.path(), &c);
    let (scribe, intent, agent) = models_of(&events);
    assert_eq!((scribe.status, scribe.model.as_deref()), (ModelStatus::Ready, Some(SCRIBE_CANDIDATES[0])));
    assert_eq!(intent.model, scribe.model);
    assert_eq!((agent.status, agent.model.as_deref()), (ModelStatus::Ready, Some("openai/gpt-6-astra:high")));
    // The scribe takes notes through the stub, and the agent answers.
    wait_for(&events, |e| matches!(e, CoreEvent::NotesUpdated(n) if n.summary == "Stub summary."));
    meeting.send(Command::Ask("what did we decide?".into())).unwrap();
    wait_for(&events, |e| matches!(e, CoreEvent::TaskUpdated(t) if t.status == TaskStatus::Done));
    let folder = meeting.stop().unwrap().folder;

    let info = meeting_json(&folder);
    assert_eq!(info["scribe"]["model"], SCRIBE_CANDIDATES[0]);
    assert_eq!(
        (&info["scribe"]["source"], &info["scribe"]["provider"]),
        (&"detected".into(), &"openrouter".into())
    );
    assert_eq!((&info["agent"]["source"], &info["agent"]["status"]), (&"detected".into(), &"ready".into()));
    assert_eq!(
        (&info["intent"]["model"], &info["intent"]["source"]),
        (&info["scribe"]["model"], &"detected".into())
    );
    let out = status(&folder, false);
    assert!(
        out.contains(&format!("scribe  ready (pi 1.1.0, {}, detected)\n", SCRIBE_CANDIDATES[0])),
        "{out}"
    );
    assert!(
        out.contains(&format!("intent  ready (pi 1.1.0, {}, detected)\n", SCRIBE_CANDIDATES[0])),
        "{out}"
    );
    assert!(out.contains("agent   ready (pi 1.1.0, openai/gpt-6-astra:high, detected)\n"), "{out}");
    let json: Value = serde_json::from_str(&status(&folder, true)).unwrap();
    assert_eq!(json["scribe"], info["scribe"]);
    assert_eq!(json["intent"], info["intent"]);
    assert_eq!(json["agent"], info["agent"]);

    // Each scribe run is a call, with the usage of pi's final message.
    let usage = &info["usage"];
    let runs = usage["scribe"]["calls"].as_u64().unwrap();
    assert!(runs >= 1, "{usage}");
    assert_eq!(usage["scribe"]["input"].as_u64(), Some(1000 * runs));
    assert!((usage["scribe"]["cost"].as_f64().unwrap() - 0.002 * runs as f64).abs() < 1e-9, "{usage}");
    assert_eq!(usage["scribe"]["login"], "api_key");
    assert_eq!((&usage["agent"]["calls"], &usage["intent"]["calls"]), (&1.into(), &0.into()));
    assert_eq!(usage["total"]["calls"].as_u64(), Some(runs + 1));
    assert_eq!(usage["partial"], false);
    let total = format!("${:.3}", 0.002 * runs as f64 + 0.01);
    let line = format!("cost    {total} (scribe ${:.3}, intent $0.000, agent $0.010)\n", 0.002 * runs as f64);
    assert!(out.contains(&line), "{out}");
    assert_eq!(json["usage"], *usage);
}

#[test]
fn a_named_model_that_isnt_ready_is_reported_not_replaced() {
    let dir = tempfile::tempdir().unwrap();
    // Every candidate is ready, but not the named models.
    let env = ready(&[SCRIBE_CANDIDATES.as_slice(), AGENT_CANDIDATES.as_slice()].concat());
    let c = choice(dir.path(), &env, "", named("p/scribe", Source::Option), named("q/agent", Source::Config));
    let (scribe, _, agent) = models::check(&c);
    assert_eq!((scribe.status, scribe.model.as_deref()), (ModelStatus::Unavailable, Some("p/scribe")));
    assert!(scribe.reason.as_deref().unwrap().ends_with("or pass another model to --scribe"), "{scribe:?}");
    assert_eq!((agent.status, agent.model.as_deref()), (ModelStatus::Unavailable, Some("q/agent")));
    let why = agent.reason.clone().unwrap();
    assert!(
        why.starts_with("pi has no login for q/agent: ")
            && why.ends_with("set `agent_model` in the config file"),
        "{why}"
    );

    // The agent's model has no login: it is refused, and the detected scribe keeps taking notes.
    let c = choice(dir.path(), &env, "", Pick::Detect, named("q/agent", Source::Config));
    let (meeting, events) = meeting(dir.path(), &c);
    let (scribe, _, agent) = models_of(&events);
    assert_eq!((scribe.status, agent.status), (ModelStatus::Ready, ModelStatus::Unavailable));
    wait_for(&events, |e| matches!(e, CoreEvent::NotesUpdated(n) if n.summary == "Stub summary."));
    let err = meeting.send(Command::Ask("anything".into())).unwrap_err().to_string();
    assert_eq!(err, format!("the agent is unavailable: {why}"));
    let folder = meeting.stop().unwrap().folder;
    let out = status(&folder, false);
    assert!(out.contains(&format!("agent   unavailable: {why}\n")), "{out}");
    assert_eq!(meeting_json(&folder)["agent"]["model"], "q/agent");
}

#[test]
fn when_detection_finds_nothing_that_one_is_off_and_says_what_to_do() {
    let dir = tempfile::tempdir().unwrap();
    let c = choice(dir.path(), "STUB_PI_READY=", PI_DEFAULT, Pick::Detect, Pick::Detect);
    let (meeting, events) = meeting(dir.path(), &c);
    let (scribe, _, agent) = models_of(&events);
    for (m, key) in [(&scribe, "scribe_model"), (&agent, "agent_model")] {
        assert_eq!((m.status, m.model.as_deref()), (ModelStatus::Unavailable, None));
        let why = m.reason.as_deref().unwrap();
        assert!(why.contains("log in to a provider with `pi` and /login") && why.contains(key), "{why}");
    }
    assert!(agent.reason.unwrap().contains(
        "(openai/gpt-6-astra:high, openrouter/anthropic/claude-sonnet-5.5, anthropic/claude-sonnet-5-5)"
    ));
    // No notes, and no intent check: the input ends with nothing sent to a model.
    let got = wait_for(&events, |e| matches!(e, CoreEvent::InputEnded { .. }));
    assert!(!got.iter().any(|e| matches!(e, CoreEvent::NotesUpdated(_))), "{got:?}");
    let folder = meeting.stop().unwrap().folder;
    let out = status(&folder, false);
    assert!(
        out.contains("scribe  unavailable: pi has a login for none of the scribe models Pam tries"),
        "{out}"
    );
}

#[test]
fn the_checks_run_side_by_side() {
    let dir = tempfile::tempdir().unwrap();
    // Six candidates at 0.5 s each: 3 s one after another.
    let c =
        choice(dir.path(), "STUB_PI_AUTH_DELAY=0.5 STUB_PI_READY=", PI_DEFAULT, Pick::Detect, Pick::Detect);
    let start = Instant::now();
    let (scribe, _, agent) = models::check(&c);
    let took = start.elapsed();
    assert_eq!((scribe.status, agent.status), (ModelStatus::Unavailable, ModelStatus::Unavailable));
    assert!(took < Duration::from_millis(1500), "took {took:?}");
}

#[test]
fn all_off_needs_no_pi() {
    let c = Choice {
        pi: Some("/nonexistent/pi".into()),
        settings: "/nonexistent/settings.json".into(),
        scribe: Pick::off(Source::Option),
        intent: Pick::Detect,
        agent: Pick::off(Source::Config),
    };
    let (scribe, intent, agent) = models::check(&c);
    assert_eq!((scribe.status, scribe.reason.as_deref()), (ModelStatus::Off, Some("--scribe off")));
    assert_eq!(
        (agent.status, agent.reason.as_deref()),
        (ModelStatus::Off, Some("agent_model = \"off\" in the config file"))
    );
    // Off with the scribe, saying how to check on this Mac instead.
    assert_eq!(intent.status, ModelStatus::Off);
    let why = intent.reason.unwrap();
    assert!(why.contains("the scribe is off (--scribe off)") && why.contains("--intent local"), "{why}");

    // The local model needs no pi either.
    let c = Choice { intent: Pick::new(Use::Intent, Some(LOCAL), None).unwrap(), ..c };
    let (_, intent, _) = models::check(&c);
    assert_eq!(
        (intent.model.as_deref(), intent.provider.as_deref(), intent.source),
        (Some("local/qwen3.5-4b"), Some("local"), Some(Source::Option))
    );
    assert!(intent.pi.is_none());
}

#[test]
fn a_named_intent_model_is_checked_and_recorded_apart_from_the_scribes() {
    let dir = tempfile::tempdir().unwrap();
    let env =
        format!("{} STUB_PI_OAUTH=q/check", ready(&[SCRIBE_CANDIDATES[0], "q/check", AGENT_CANDIDATES[0]]));
    let c = Choice {
        intent: Pick::new(Use::Intent, None, Some("q/check")).unwrap(),
        ..choice(dir.path(), &env, "", Pick::Detect, Pick::Detect)
    };
    let (meeting, events) = meeting(dir.path(), &c);
    let (scribe, intent, _) = models_of(&events);
    assert_eq!(scribe.model.as_deref(), Some(SCRIBE_CANDIDATES[0]));
    assert_eq!(
        (intent.status, intent.model.as_deref(), intent.provider.as_deref(), intent.source),
        (ModelStatus::Ready, Some("q/check"), Some("q"), Some(Source::Config))
    );
    let folder = meeting.stop().unwrap().folder;
    let info = meeting_json(&folder);
    assert_eq!((&info["intent"]["model"], &info["intent"]["source"]), (&"q/check".into(), &"config".into()));
    // Its cost is counted at its own login's prices.
    assert_eq!(
        (&info["usage"]["intent"]["login"], &info["usage"]["scribe"]["login"]),
        (&"oauth".into(), &"api_key".into())
    );
    let out = status(&folder, false);
    assert!(out.contains("intent  ready (pi 1.1.0, q/check, config)\n"), "{out}");

    // Named but not ready: reported, never swapped for the scribe's.
    let c = Choice { intent: Pick::new(Use::Intent, Some("q/none"), None).unwrap(), ..c };
    let (_, intent, _) = models::check(&c);
    assert_eq!((intent.status, intent.model.as_deref()), (ModelStatus::Unavailable, Some("q/none")));
    assert!(intent.reason.unwrap().ends_with("or pass another model to --intent"));
    assert_eq!(
        Pick::new(Use::Intent, Some(" "), None),
        Err("--intent is empty: give it a model as provider/id, local, or off".into())
    );
}

/// Only without llama.cpp: with it built in, the model on this Mac would load.
#[cfg(not(feature = "llama-cpp"))]
#[test]
fn a_local_intent_model_that_cant_run_leaves_spoken_requests_off_and_says_why() {
    let dir = tempfile::tempdir().unwrap();
    let env = ready(&[SCRIBE_CANDIDATES[0], AGENT_CANDIDATES[0]]);
    let c = Choice {
        intent: Pick::new(Use::Intent, Some(LOCAL), None).unwrap(),
        ..choice(dir.path(), &env, "", Pick::Detect, Pick::Detect)
    };
    let (meeting, events) = meeting(dir.path(), &c);
    let (_, intent, _) = models_of(&events);
    assert_eq!(
        (intent.status, intent.model.as_deref()),
        (ModelStatus::Unavailable, Some("local/qwen3.5-4b"))
    );
    let why = intent.reason.clone().unwrap();
    assert!(why.contains("--features llama-cpp"), "{why}");
    let folder = meeting.stop().unwrap().folder;
    assert!(status(&folder, false).contains(&format!("intent  unavailable: {why}\n")));
}
