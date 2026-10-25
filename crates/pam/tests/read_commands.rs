//! The read commands, run as the agent runs them: the `pam` binary against a
//! meeting folder written by replaying the stand-up fixture.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::OnceLock;
use std::time::Duration;

use pam_core::{CoreEvent, Meeting, MeetingConfig};
use serde_json::Value;

struct Run {
    code: i32,
    stdout: String,
    stderr: String,
}

/// Runs `pam` with only the meeting choice we give it in its environment.
fn pam(root: &Path, env_meeting: Option<&Path>, args: &[&str]) -> Run {
    let mut cmd = Command::new(env!("CARGO_BIN_EXE_pam"));
    cmd.args(args).env("PAM_MEETINGS_ROOT", root).env_remove("PAM_MEETING");
    if let Some(m) = env_meeting {
        cmd.env("PAM_MEETING", m);
    }
    let out = cmd.output().unwrap();
    Run {
        code: out.status.code().unwrap(),
        stdout: String::from_utf8(out.stdout).unwrap(),
        stderr: String::from_utf8(out.stderr).unwrap(),
    }
}

/// Replays the fixture into a fresh root and returns (root, meeting folder).
fn replay() -> (tempfile::TempDir, PathBuf) {
    let root = tempfile::tempdir().unwrap();
    let script = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures/standup.fake.jsonl");
    let mut cfg = MeetingConfig::fake(root.path().to_path_buf(), script).unwrap();
    cfg.title = "Stand-up".into();
    let meeting = Meeting::start(cfg).unwrap();
    let events = meeting.subscribe();
    loop {
        match events.recv_timeout(Duration::from_secs(10)).expect("meeting stalled") {
            CoreEvent::InputEnded { .. } => break,
            CoreEvent::Error(e) => panic!("{e}"),
            _ => {}
        }
    }
    let folder = meeting.stop().unwrap().folder;
    (root, folder)
}

/// One replay shared by the tests that only read it.
fn standup() -> &'static (tempfile::TempDir, PathBuf) {
    static STANDUP: OnceLock<(tempfile::TempDir, PathBuf)> = OnceLock::new();
    STANDUP.get_or_init(replay)
}

fn read(args: &[&str]) -> Run {
    let (root, _) = standup();
    pam(root.path(), None, args)
}

fn ids(text: &str) -> Vec<&str> {
    text.lines().filter_map(|l| l.trim_start_matches(['>', ' ']).split(' ').next()).collect()
}

#[test]
fn the_demo_prints_the_right_lines() {
    let last = read(&["transcript", "--last", "5"]);
    assert_eq!(last.code, 0, "{}", last.stderr);
    assert_eq!(ids(&last.stdout), ["u0006", "u0007", "u0008", "u0009", "u0010"]);
    assert!(last.stdout.starts_with("u0006 00:20 spk_1: Single page is faster to build"), "{}", last.stdout);

    let around = read(&["around", "u0003", "--before", "2s", "--after", "3s"]);
    assert_eq!(around.code, 0, "{}", around.stderr);
    assert_eq!(
        around.stdout,
        "  u0002 00:06 spk_1: Marcus, engineering.\n\
         > u0003 00:08 Jordan: And I'm Jordan, design.\n  \
         u0004 00:10 Alice: Today is about the onboarding redesign.\n"
    );

    let search = read(&["search", "mobile"]);
    assert_eq!(search.code, 0, "{}", search.stderr);
    assert_eq!(ids(&search.stdout), ["u0006", "u0007"]);
}

#[test]
fn around_without_a_ref_explains_itself() {
    let r = read(&["around"]);
    assert_eq!(r.code, 1);
    assert_eq!(r.stdout, "");
    assert!(r.stderr.starts_with(
        "error: missing <ref>: which line or event should I show context around?\n\n\
         usage: pam around <ref> [--before <dur>] [--after <dur>] [--json] [--meeting <folder>]\n"
    ));
    assert!(r.stderr.contains("\nexamples:\n  pam around e037"), "{}", r.stderr);
}

#[test]
fn every_usage_error_says_what_went_wrong_the_usage_and_examples() {
    for cmd in [
        "listen",
        "status",
        "context",
        "people",
        "transcript",
        "get",
        "around",
        "search",
        "events",
        "note",
        "say",
        "help",
    ] {
        let r = read(&[cmd, "--bogus"]);
        assert_eq!(r.code, 1, "{cmd}");
        let parts: Vec<&str> = r.stderr.split("\n\n").collect();
        assert!(parts[0].starts_with("error: unknown option --bogus"), "{cmd}: {}", r.stderr);
        assert!(parts[1].starts_with(&format!("usage: pam {cmd}")), "{cmd}: {}", r.stderr);
        assert!(parts[2].starts_with("examples:\n"), "{cmd}: {}", r.stderr);
        let examples = parts[2].lines().filter(|l| l.contains(" # ")).count();
        assert!((2..=3).contains(&examples), "{cmd}: {}", r.stderr);
    }
}

#[test]
fn exit_codes_tell_usage_not_found_and_no_meeting_apart() {
    assert_eq!(read(&["status"]).code, 0);
    assert_eq!(read(&["transcript", "--last", "zero"]).code, 1);
    assert_eq!(read(&["frobnicate"]).code, 1);
    let huge = read(&["transcript", "--since", "18446744073709551615"]);
    assert_eq!(huge.code, 1, "too long a duration is a usage error, not a crash: {}", huge.stderr);
    assert!(huge.stderr.starts_with("error: --since is too long"), "{}", huge.stderr);

    let missing = read(&["get", "u0002", "u0099"]);
    assert_eq!(missing.code, 2);
    assert_eq!(missing.stdout, "u0002 00:06 spk_1: Marcus, engineering.\n", "what exists is still printed");
    assert_eq!(missing.stderr, "error: no line u0099: lines run u0001..u0010\n");
    assert_eq!(read(&["transcript", "--speaker", "Sam"]).code, 2);

    let empty = tempfile::tempdir().unwrap();
    let none = pam(empty.path(), None, &["status"]);
    assert_eq!(none.code, 3);
    assert!(none.stderr.contains("start one with `pam listen`"), "{}", none.stderr);
    let not_a_meeting = pam(empty.path(), None, &["people", "--meeting", empty.path().to_str().unwrap()]);
    assert_eq!(not_a_meeting.code, 3);
    // A usage error is reported as one, meeting or not.
    assert_eq!(pam(empty.path(), None, &["around"]).code, 1);
}

#[test]
fn the_meeting_comes_from_the_flag_then_the_environment_then_the_latest() {
    let (root, folder) = standup();
    let other = tempfile::tempdir().unwrap();
    let r = pam(other.path(), Some(folder), &["status"]);
    assert_eq!(r.code, 0, "{}", r.stderr);
    assert!(r.stdout.starts_with("Stand-up · 00:42 elapsed · stopped\n"), "{}", r.stdout);

    let flag = pam(other.path(), Some(other.path()), &["status", "--meeting", folder.to_str().unwrap()]);
    assert_eq!(flag.code, 0, "the flag wins over $PAM_MEETING: {}", flag.stderr);

    let latest = pam(root.path(), None, &["status", "--json"]);
    let v: Value = serde_json::from_str(&latest.stdout).unwrap();
    assert_eq!(v["folder"], folder.to_str().unwrap());
    assert_eq!((v["lines"].as_u64(), v["elapsed"].as_u64()), (Some(10), Some(42 * 16_000)));
}

#[test]
fn json_output_parses() {
    let v: Value =
        serde_json::from_str(&read(&["transcript", "--from", "00:20", "--to", "00:30", "--json"]).stdout)
            .unwrap();
    let lines = v["lines"].as_array().unwrap();
    assert_eq!(lines.iter().map(|l| l["id"].as_str().unwrap()).collect::<Vec<_>>(), ["u0006", "u0007"]);
    assert_eq!(lines[0]["spk"], "spk_1");

    let v: Value = serde_json::from_str(&read(&["get", "u0001..u0002", "spk_0", "--json"]).stdout).unwrap();
    let kinds: Vec<&str> =
        v["items"].as_array().unwrap().iter().map(|i| i["kind"].as_str().unwrap()).collect();
    assert_eq!(kinds, ["line", "line", "speaker"]);

    let v: Value =
        serde_json::from_str(&read(&["around", "u0005", "--after", "0s", "--json"]).stdout).unwrap();
    let anchors: Vec<bool> =
        v["lines"].as_array().unwrap().iter().map(|l| l["anchor"].as_bool().unwrap()).collect();
    assert_eq!(anchors.last(), Some(&true), "nothing after u0005 with --after 0s");

    let v: Value = serde_json::from_str(&read(&["search", "a", "--limit", "2", "--json"]).stdout).unwrap();
    assert_eq!(v["results"].as_array().unwrap().len(), 2);
    assert!(v["total"].as_u64().unwrap() > 2);
}

#[test]
fn renames_merges_and_events_show_up_without_changes_here() {
    let (root, folder) = replay();
    // What intro naming, corrections and the scribe will write later.
    std::fs::write(
        folder.join("speakers.json"),
        r#"{"slots":{"0":"spk_0","1":"spk_1","2":"spk_2"},
            "names":{"spk_0":"Alice","spk_1":"Marcus"},
            "merges":{"spk_2":"spk_1"},
            "moves":{"u0010":"spk_0"},
            "log":[]}"#,
    )
    .unwrap();
    std::fs::write(
        folder.join("events.jsonl"),
        concat!(
            r#"{"id":"e001","type":"concern","t":332800,"by":["spk_1"],"text":"Single page may lose users on mobile","refs":["u0006"]}"#,
            "\n",
            r#"{"id":"e002","type":"decision","t":506880,"by":["spk_0"],"text":"Prototype both flows","refs":["u0009"]}"#,
            "\n",
        ),
    )
    .unwrap();
    let run = |args: &[&str]| pam(root.path(), None, args);

    let people = run(&["people"]);
    assert_eq!(
        people.stdout,
        "ID     NAME    FIRST HEARD  TALK   LINES\n\
         spk_0  Alice   00:01        00:17  5\n\
         spk_1  Marcus  00:06        00:10  5\n"
    );
    let marcus = run(&["transcript", "--speaker", "marcus"]);
    assert_eq!(ids(&marcus.stdout), ["u0002", "u0003", "u0006", "u0007", "u0008"]);
    assert!(marcus.stdout.contains("u0003 00:08 Marcus: And I'm Jordan, design."));
    assert!(run(&["get", "u0010"]).stdout.starts_with("u0010 00:38 Alice: "));

    let around = run(&["around", "e002", "--before", "2s", "--after", "0s"]);
    assert_eq!(
        around.stdout,
        "e002 00:31 decision Alice: Prototype both flows [u0009]\n  \
         u0008 00:30 Marcus: Right, exactly.\n\
         > u0009 00:32 Alice: Then let's prototype both and look at the numbers next week.\n"
    );
    let decisions = run(&["search", "prototype", "--type", "decision"]);
    assert_eq!(decisions.stdout, "e002 00:31 decision Alice: Prototype both flows [u0009]\n");
    let status = run(&["status"]);
    assert!(status.stdout.contains("people  Alice (spk_0), Marcus (spk_1)\n"), "{}", status.stdout);
    assert!(status.stdout.contains("events  2\nopen    nothing\n"), "{}", status.stdout);
}

#[test]
fn help_has_json_too() {
    let all = read(&["help", "--json"]);
    assert_eq!(all.code, 0, "{}", all.stderr);
    let v: Value = serde_json::from_str(&all.stdout).unwrap();
    let names: Vec<&str> =
        v["commands"].as_array().unwrap().iter().map(|c| c["name"].as_str().unwrap()).collect();
    assert_eq!(
        names,
        [
            "listen",
            "status",
            "context",
            "people",
            "transcript",
            "get",
            "around",
            "search",
            "events",
            "note",
            "say",
            "task",
            "models",
            "help"
        ]
    );
    assert_eq!(v["exit_codes"]["2"], "not found");

    let around: Value = serde_json::from_str(&read(&["help", "around", "--json"]).stdout).unwrap();
    assert_eq!(
        around["usage"],
        "pam around <ref> [--before <dur>] [--after <dur>] [--json] [--meeting <folder>]"
    );
    assert_eq!(around["examples"][0]["command"], "pam around e037");
}

#[test]
fn an_unreadable_file_fails_with_4_not_2_or_3() {
    let (root, folder) = replay();
    let t02 = read(&["get", "t02"]);
    assert_eq!((t02.code, t02.stderr.as_str()), (2, "error: no task t02: the meeting has no tasks yet\n"));

    std::fs::remove_file(folder.join("speakers.json")).unwrap();
    std::fs::create_dir(folder.join("speakers.json")).unwrap();
    let speakers = pam(root.path(), None, &["people"]);
    assert_eq!(speakers.code, 4, "{}", speakers.stderr);
    assert!(speakers.stderr.contains("speakers.json: it should be a file"), "{}", speakers.stderr);

    std::fs::write(folder.join("meeting.json"), "{").unwrap();
    let latest = pam(root.path(), None, &["status"]);
    assert_eq!(latest.code, 4, "{}", latest.stderr);
    assert!(latest.stderr.contains("meeting.json"), "{}", latest.stderr);
}

#[test]
fn a_closed_pipe_is_not_a_failure() {
    let (root, _) = standup();
    let mut child = Command::new(env!("CARGO_BIN_EXE_pam"))
        .args(["transcript"])
        .env("PAM_MEETINGS_ROOT", root.path())
        .env_remove("PAM_MEETING")
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .spawn()
        .unwrap();
    drop(child.stdout.take()); // the reader goes away, as with `| head`
    let out = child.wait_with_output().unwrap();
    assert_eq!(out.status.code(), Some(0), "{}", String::from_utf8_lossy(&out.stderr));
}

#[test]
fn events_lists_and_note_adds_to_the_notes() {
    let (root, folder) = replay();
    std::fs::write(
        folder.join("events.jsonl"),
        concat!(
            r#"{"id":"e001","type":"intro","t":94720,"by":["spk_1"],"text":"Marcus, engineering","refs":["u0002"],"name":"Marcus"}"#,
            "\n",
            r#"{"id":"e002","type":"concern","t":332800,"by":["spk_1"],"text":"Single page may lose users on mobile","refs":["u0006"]}"#,
            "\n",
            r#"{"id":"e003","type":"question","t":332800,"by":["spk_1"],"text":"Is mobile drop-off real?","refs":["u0006"]}"#,
            "\n",
            r#"{"id":"e004","type":"answer","t":384000,"by":["spk_2"],"text":"Most sign-ups start on mobile","refs":["u0007"],"supersedes":"e003"}"#,
            "\n",
        ),
    )
    .unwrap();
    let run = |args: &[&str]| pam(root.path(), None, args);

    assert_eq!(
        run(&["events"]).stdout,
        "e001 00:05 intro spk_1: Marcus, engineering [u0002]\n\
         e002 00:20 concern spk_1: Single page may lose users on mobile [u0006]\n\
         e003 00:20 question spk_1: Is mobile drop-off real? [u0006] [superseded by e004]\n\
         e004 00:24 answer Jordan: Most sign-ups start on mobile [u0007] [supersedes e003]\n"
    );
    assert_eq!(ids(&run(&["events", "--type", "Answer,concern"]).stdout), ["e002", "e004"]);
    // The meeting is 42 s long: the last 20 s start at 00:22.
    assert_eq!(ids(&run(&["events", "--since", "20s"]).stdout), ["e004"]);
    assert_eq!(run(&["events", "--type", "decision"]).stdout, "no events match\n");
    let lines = run(&["events", "--type", "line"]);
    assert_eq!(lines.code, 1);
    assert!(lines.stderr.contains("pam transcript"), "{}", lines.stderr);
    let json: Value = serde_json::from_str(&run(&["events", "--json"]).stdout).unwrap();
    assert_eq!(json["events"][3]["names"], serde_json::json!(["Jordan"]));
    assert_eq!(json["events"][2]["superseded_by"], "e004");

    let note = run(&["note", "Ship", "the", "beta", "on", "Friday"]);
    assert_eq!(
        (note.code, note.stdout.as_str()),
        (0, "e005 00:42 note: Ship the beta on Friday\n"),
        "{}",
        note.stderr
    );
    let json: Value = serde_json::from_str(&run(&["note", "--json", "Book a room"]).stdout).unwrap();
    assert_eq!(json["event"]["id"], "e006");
    assert_eq!(json["event"]["type"], "note");
    assert_eq!(ids(&run(&["events", "--type", "note"]).stdout), ["e005", "e006"]);
    let notes = std::fs::read_to_string(folder.join("notes.md")).unwrap();
    assert!(notes.contains("- Concern: Single page may lose users on mobile (spk_1)\n"), "{notes}");
    assert!(notes.contains("- Answer: Most sign-ups start on mobile (Jordan)\n"), "{notes}");
    assert!(!notes.contains("Is mobile drop-off real?"), "an answered question is no longer open: {notes}");
    assert!(notes.ends_with("## Notes\n\n- Ship the beta on Friday\n- Book a room\n"), "{notes}");

    let empty = run(&["note", "  "]);
    assert_eq!(empty.code, 1);
    assert!(empty.stderr.starts_with("error: missing <text>"), "{}", empty.stderr);
}

#[test]
fn listening_to_a_recording_uses_the_backends_in_the_config_file() {
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("one-second.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(&wav, spec).unwrap();
    for _ in 0..16_000 {
        w.write_sample(0i16).unwrap();
    }
    w.finalize().unwrap();
    let config = dir.path().join("config.toml");
    std::fs::write(
        &config,
        "asr = \"nemo-speech:/nowhere/asr.gguf\"\ndiarizer = \"nemo-speech:/nowhere/diar.gguf\"\n",
    )
    .unwrap();
    let root = dir.path().join("meetings");

    let out = Command::new(env!("CARGO_BIN_EXE_pam"))
        .args(["listen", "--input", wav.to_str().unwrap()])
        .env("PAM_MEETINGS_ROOT", &root)
        .env("PAM_CONFIG", &config)
        .env("PAM_MODELS", dir.path().join("models"))
        .output()
        .unwrap();
    let stderr = String::from_utf8(out.stderr).unwrap();
    assert_ne!(out.status.code(), Some(0));
    // The backend is refused before the dashboard starts or any folder is made.
    let why =
        if cfg!(feature = "nemo-speech") { "no model file /nowhere/" } else { "--features nemo-speech" };
    assert!(stderr.contains(why), "{stderr}");
    assert!(!root.exists());
}

#[test]
fn a_model_missing_from_the_store_stops_listen_before_the_meeting() {
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("one-second.wav");
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 16_000,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };
    let mut w = hound::WavWriter::create(&wav, spec).unwrap();
    for _ in 0..16_000 {
        w.write_sample(0i16).unwrap();
    }
    w.finalize().unwrap();
    let root = dir.path().join("meetings");
    let models = dir.path().join("models");

    let out = Command::new(env!("CARGO_BIN_EXE_pam"))
        .args(["listen", "--input", wav.to_str().unwrap(), "--scribe", "off", "--agent", "off"])
        .env("PAM_MEETINGS_ROOT", &root)
        .env("PAM_CONFIG", dir.path().join("no-config.toml"))
        .env("PAM_MODELS", &models)
        .output()
        .unwrap();
    let stderr = String::from_utf8(out.stderr).unwrap();
    if cfg!(feature = "nemo-speech") {
        assert_eq!(out.status.code(), Some(4), "{stderr}");
        assert!(stderr.contains("model `nemotron-en` isn't in the model store"), "{stderr}");
        assert!(stderr.contains(&models.display().to_string()), "{stderr}");
        assert!(stderr.contains("run `pam models pull nemotron-en`"), "{stderr}");
        assert!(!stderr.contains("Checking pi"), "before anything else: {stderr}");
    } else {
        assert!(stderr.contains("--features nemo-speech"), "{stderr}");
    }
    assert!(!root.exists());
    assert!(!models.exists(), "nothing downloaded");
}

#[test]
fn listen_refuses_realtime_on_the_mic_before_opening_it() {
    let dir = tempfile::tempdir().unwrap();
    let root = dir.path().join("meetings");
    for args in [&["listen", "--realtime"][..], &["listen", "--input", "mic", "--realtime"]] {
        let out = Command::new(env!("CARGO_BIN_EXE_pam"))
            .args(args)
            .env("PAM_MEETINGS_ROOT", &root)
            .output()
            .unwrap();
        let stderr = String::from_utf8(out.stderr).unwrap();
        assert_eq!(out.status.code(), Some(1), "{stderr}");
        assert!(stderr.contains("the mic is always live"), "{stderr}");
        assert!(stderr.contains("usage: pam listen [--input mic"), "{stderr}");
    }
    assert!(!root.exists());
}

#[test]
fn context_prints_the_pack_and_status_lists_open_items() {
    let (root, folder) = replay();
    let run = |args: &[&str]| pam(root.path(), None, args);
    // Stopping the meeting rendered context.md, and `pam context` prints it.
    let saved = std::fs::read_to_string(folder.join("context.md")).unwrap();
    assert!(saved.starts_with("# Meeting: Stand-up · 00:42 elapsed · assistant name: Pam\n"), "{saved}");
    assert!(saved.contains("\n- u0010 00:38 Jordan: Sounds good to me.\n"), "{saved}");
    let context = run(&["context"]);
    assert_eq!((context.code, context.stdout.as_str()), (0, saved.as_str()), "{}", context.stderr);

    std::fs::write(
        folder.join("events.jsonl"),
        concat!(
            r#"{"id":"e001","type":"question","t":332800,"by":["spk_1"],"text":"Is mobile drop-off real?","refs":["u0006"]}"#,
            "\n",
            r#"{"id":"e002","type":"action","t":506880,"by":["spk_0"],"text":"Prototype both flows","refs":["u0009"]}"#,
            "\n",
        ),
    )
    .unwrap();
    let status = run(&["status"]);
    assert!(
        status.stdout.ends_with(
            "events  2\nopen    2\n  \
             e001 00:20 question spk_1: Is mobile drop-off real? [u0006]\n  \
             e002 00:31 action Alice: Prototype both flows [u0009]\n\
             tasks   none yet\n"
        ),
        "{}",
        status.stdout
    );
    let json: Value = serde_json::from_str(&run(&["status", "--json"]).stdout).unwrap();
    assert_eq!(json["open"][1]["text"], "Prototype both flows");

    // A note re-renders it, with the events added from outside.
    assert_eq!(run(&["note", "Book", "a", "room"]).code, 0);
    let context = run(&["context"]).stdout;
    assert!(
        context.contains(
            "## Open\n\n\
             - e001 00:20 question spk_1: Is mobile drop-off real? [u0006]\n\
             - e002 00:31 action Alice: Prototype both flows [u0009]\n"
        ),
        "{context}"
    );
    assert!(context.contains("- e003 00:42 note: Book a room\n\n## Last 3 min"), "{context}");
    let json: Value = serde_json::from_str(&run(&["context", "--json"]).stdout).unwrap();
    assert_eq!(json["text"], context);
    assert_eq!(json["tokens"], (context.len() * 2).div_ceil(5));

    // A meeting without one gets it rendered on the spot; a broken one fails.
    std::fs::remove_file(folder.join("context.md")).unwrap();
    assert_eq!(run(&["context"]).stdout, context);
    std::fs::create_dir(folder.join("context.md")).unwrap();
    let broken = run(&["context"]);
    assert_eq!(broken.code, 4);
    assert!(broken.stderr.contains("context.md: it should be a file"), "{}", broken.stderr);
}

#[test]
fn tasks_show_in_get_status_and_context() {
    use pam_core::{TaskId, TaskState, TaskStatus, UtteranceId, Via, task};
    let (root, folder) = replay();
    let run = |args: &[&str]| pam(root.path(), None, args);
    let s = |secs: u64| secs * 16_000;
    let tasks = [
        TaskState {
            line: Some(UtteranceId(3)),
            spk: Some(pam_core::SpkId(2)),
            corrections: vec!["only the mobile one".into()],
            started_at: Some(s(10)),
            ended_at: Some(s(14)),
            ..TaskState::new(TaskId(1), "put up both flows", s(8), Via::Voice, TaskStatus::Done)
        },
        TaskState {
            error: Some("pi exited".into()),
            started_at: Some(s(20)),
            ended_at: Some(s(21)),
            ..TaskState::new(TaskId(2), "book a room", s(20), Via::Typed, TaskStatus::Failed)
        },
        TaskState {
            progress: Some(40),
            action: Some("reading sources".into()),
            started_at: Some(s(30)),
            ..TaskState::new(TaskId(3), "fact-check the launch date", s(30), Via::Typed, TaskStatus::Running)
        },
        TaskState::new(TaskId(4), "draft the email", s(35), Via::Typed, TaskStatus::Queued),
    ];
    for t in &tasks {
        task::create(&folder, t).unwrap();
    }
    // What t01 said and asked, at wall times 11–13 s into the meeting.
    let info: Value =
        serde_json::from_str(&std::fs::read_to_string(folder.join("meeting.json")).unwrap()).unwrap();
    let started = chrono::DateTime::parse_from_rfc3339(info["started_at"].as_str().unwrap()).unwrap();
    let at = |secs: i64| (started + chrono::Duration::seconds(secs)).to_rfc3339();
    let t01 = folder.join("tasks/t01");
    std::fs::write(
        t01.join("messages.jsonl"),
        format!("{}\n", serde_json::json!({"at": at(12), "text": "Two flows, side by side."})),
    )
    .unwrap();
    let questions = [
        serde_json::json!({"type": "asked", "q": 1, "at": at(11), "text": "Which customer?", "options": ["Acme", "Globex"], "timeout": 15, "pid": 1}),
        serde_json::json!({"type": "answered", "q": 1, "at": at(12), "option": 2, "by": "voice", "spk": "spk_2"}),
        serde_json::json!({"type": "asked", "q": 2, "at": at(13), "text": "Which city?", "options": [], "timeout": 30, "pid": 1}),
        serde_json::json!({"type": "timed_out", "q": 2, "at": at(43)}),
    ];
    std::fs::write(
        t01.join("questions.jsonl"),
        questions.iter().map(|q| format!("{q}\n")).collect::<String>(),
    )
    .unwrap();

    let get = run(&["get", "t01", "t02"]);
    assert_eq!(get.code, 0, "{}", get.stderr);
    assert_eq!(
        get.stdout,
        "t01 00:08 Jordan: put up both flows [u0003] · done\n  \
         asked     00:08, said by Jordan\n  \
         status    done · started 00:10 · ended 00:14 · took 4s\n  \
         then      only the mobile one\n  \
         said      00:12 Two flows, side by side.\n  \
         q1        00:11 Which customer? [1 Acme · 2 Globex] → option 2, Globex, by Jordan\n  \
         q2        00:13 Which city? (open) → timeout\n\
         t02 00:20 typed: book a room · failed\n  \
         asked     00:20, typed on the laptop\n  \
         status    failed · started 00:20 · ended 00:21 · took 1s\n  \
         error     pi exited\n"
    );
    let missing = run(&["get", "t09"]);
    assert_eq!(
        (missing.code, missing.stderr.as_str()),
        (2, "error: no task t09: tasks are t01, t02, t03, t04\n")
    );
    let json: Value = serde_json::from_str(&run(&["get", "t01", "--json"]).stdout).unwrap();
    let item = &json["items"][0];
    assert_eq!(
        (&item["kind"], &item["id"], &item["status"], &item["line"]),
        (&"task".into(), &"t01".into(), &"done".into(), &"u0003".into())
    );
    assert_eq!(item["messages"][0]["text"], "Two flows, side by side.");
    assert_eq!(item["questions"].as_array().unwrap().len(), 4);
    assert_eq!(item["questions"][1]["option"], 2);

    let status = run(&["status"]);
    assert!(
        status.stdout.ends_with(
            "tasks   4 (1 running, 1 queued, 0 ready)\n  \
             t03 00:30 typed: fact-check the launch date · running 40% · reading sources\n  \
             t04 00:35 typed: draft the email · queued\n"
        ),
        "{}",
        status.stdout
    );
    let json: Value = serde_json::from_str(&run(&["status", "--json"]).stdout).unwrap();
    assert_eq!((&json["tasks"]["total"], &json["tasks"]["running"]), (&4.into(), &1.into()));
    assert_eq!(json["tasks"]["listed"][1]["id"], "t04");

    // The meeting has stopped, so nothing re-renders context.md: without one, it is rendered on the spot.
    std::fs::remove_file(folder.join("context.md")).unwrap();
    let context = run(&["context"]).stdout;
    assert!(
        context.contains(
            "## Tasks\n\n\
             - t01 00:08 Jordan: put up both flows [u0003] · done\n\
             - t02 00:20 typed: book a room · failed\n\
             - t03 00:30 typed: fact-check the launch date · running 40% · reading sources\n\
             - t04 00:35 typed: draft the email · queued\n\n## Events"
        ),
        "{context}"
    );
}
