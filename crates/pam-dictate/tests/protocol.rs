//! The piped face, run as a shell runs it: the `dictate` binary with events
//! on stdout and commands on stdin, through the `fake` backend.

use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdout, Command, Stdio};
use std::time::{Duration, Instant};

use serde_json::Value;

/// How much faster than real time scripts play, so a 38 s script takes 1 s.
const SPEED: &str = "40";
/// Far less than the 15 s [`long_script`] takes to play out at [`SPEED`].
const WELL_BEFORE_THE_END: Duration = Duration::from_secs(5);

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures").join(name)
}

/// Where the tests' config file is: nowhere, unless a test writes one, so
/// that the config of whoever runs them stays out of it.
const NO_CONFIG: &str = "/nonexistent/dictate/config.toml";

fn dictate(args: &[&str]) -> Child {
    dictate_with(args, Path::new(NO_CONFIG))
}

fn dictate_with(args: &[&str], config: &Path) -> Child {
    Command::new(env!("CARGO_BIN_EXE_dictate"))
        .args(args)
        .env("DICTATE_REPLAY_SPEED", SPEED)
        .env("DICTATE_CONFIG", config)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap()
}

/// Every event line until stdout closes, and the exit code.
fn finish(mut child: Child, mut lines: Lines) -> (Vec<Value>, i32) {
    let mut events = Vec::new();
    while let Some(e) = lines.next_event() {
        events.push(e);
    }
    (events, child.wait().unwrap().code().unwrap())
}

fn run(args: &[&str]) -> (Vec<Value>, i32) {
    run_with(args, Path::new(NO_CONFIG))
}

fn run_with(args: &[&str], config: &Path) -> (Vec<Value>, i32) {
    let mut child = dictate_with(args, config);
    let lines = Lines::of(&mut child);
    finish(child, lines)
}

struct Lines(BufReader<ChildStdout>);

impl Lines {
    fn of(child: &mut Child) -> Lines {
        Lines(BufReader::new(child.stdout.take().unwrap()))
    }

    fn next_event(&mut self) -> Option<Value> {
        let mut line = String::new();
        (self.0.read_line(&mut line).unwrap() > 0).then(|| {
            assert!(line.ends_with('\n'), "{line:?}");
            serde_json::from_str(&line).unwrap_or_else(|e| panic!("{line:?}: {e}"))
        })
    }

    /// Reads events up to and including the first `final`.
    fn until_final(&mut self) -> Vec<Value> {
        let mut seen = Vec::new();
        while let Some(e) = self.next_event() {
            let done = kind(&e) == "final";
            seen.push(e);
            if done {
                return seen;
            }
        }
        panic!("no final before the end: {seen:?}");
    }
}

fn kind(e: &Value) -> &str {
    e["event"].as_str().unwrap()
}

/// The kinds of `events`, but `level`, which comes ten times a second
/// throughout; [`levels_come_ten_a_second_while_listening`] tests it.
fn kinds(events: &[Value]) -> Vec<&str> {
    events.iter().map(kind).filter(|k| *k != "level").collect()
}

/// A 16 kHz sine of `seconds`, at `amplitude` of full scale.
fn tone(seconds: f64, amplitude: f32) -> Vec<f32> {
    let n = (16_000.0 * seconds) as usize;
    (0..n).map(|i| amplitude * (i as f32 * 300.0 * std::f32::consts::TAU / 16_000.0).sin()).collect()
}

/// The `final` events' texts, joined as `text` joins them.
fn finals(events: &[Value]) -> String {
    let texts: Vec<&str> =
        events.iter().filter(|e| kind(e) == "final").map(|e| e["text"].as_str().unwrap()).collect();
    texts.join(" ")
}

/// A script whose words are all said in the first two seconds, but which
/// lasts ten minutes: time to send a command while it is playing.
fn long_script(dir: &Path) -> PathBuf {
    let path = dir.join("long.fake.jsonl");
    let mut text = String::from("{\"config\": {\"word_delay\": 0.3, \"duration\": 600}}\n");
    for (i, w) in ["one", "two", "three", "four", "five", "six"].iter().enumerate() {
        text.push_str(&format!("{{\"word\": \"{w}\", \"at\": {}}}\n", 0.2 + 0.32 * i as f64));
    }
    std::fs::write(&path, text).unwrap();
    path
}

#[test]
fn a_script_plays_to_its_end_and_gives_its_words() {
    let script = fixture("intros.fake.jsonl");
    let (events, code) = run(&["--input", script.to_str().unwrap()]);
    assert_eq!(code, 0, "{events:?}");
    let k = kinds(&events);
    assert_eq!(k[..2], ["loading", "recording"]);
    assert_eq!(events[1]["device"], "intros.fake.jsonl");
    assert_eq!(k[k.len() - 2..], ["transcribing", "text"]);
    assert!(k[2..k.len() - 2].iter().all(|e| ["preview", "final"].contains(e)), "{k:?}");
    assert!(k.contains(&"preview"));

    let words = pam_speech::fake::Script::load(&script).unwrap().words;
    let said: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
    let text = events.last().unwrap()["text"].as_str().unwrap();
    assert_eq!(text, said.join(" "));
    assert_eq!(finals(&events), text, "the finals arrive in script order and make up the text");
}

#[test]
fn cancel_exits_at_once_with_no_text() {
    let dir = tempfile::tempdir().unwrap();
    let script = long_script(dir.path());
    let mut child = dictate(&["--input", script.to_str().unwrap()]);
    let mut lines = Lines::of(&mut child);
    lines.until_final();
    let sent = Instant::now();
    writeln!(child.stdin.as_mut().unwrap(), "cancel").unwrap();
    let (rest, code) = finish(child, lines);
    assert!(sent.elapsed() < WELL_BEFORE_THE_END, "{:?}", sent.elapsed());
    assert_eq!(code, 3);
    assert!(kinds(&rest).iter().all(|e| ["preview", "final"].contains(e)), "{rest:?}");
}

#[test]
fn stop_gives_the_finals_so_far_and_what_the_transcriber_flushed() {
    let dir = tempfile::tempdir().unwrap();
    let script = long_script(dir.path());
    let mut child = dictate(&["--input", script.to_str().unwrap()]);
    let mut lines = Lines::of(&mut child);
    let mut events = lines.until_final();
    let sent = Instant::now();
    writeln!(child.stdin.as_mut().unwrap(), "stop").unwrap();
    let (rest, code) = finish(child, lines);
    assert!(sent.elapsed() < WELL_BEFORE_THE_END, "{:?}", sent.elapsed());
    events.extend(rest);
    assert_eq!(code, 0, "{events:?}");
    // What the transcriber flushes comes as finals after `transcribing`.
    let k = kinds(&events);
    let flushed = k.iter().position(|e| *e == "transcribing").expect("transcribing");
    assert!(k[flushed + 1..k.len() - 1].iter().all(|e| *e == "final"), "{k:?}");
    assert_eq!(k.last(), Some(&"text"));
    let text = events.last().unwrap()["text"].as_str().unwrap();
    assert_eq!(text, finals(&events));
    assert!(text.starts_with("one"), "{text}");
}

#[test]
fn a_script_with_no_words_is_no_speech() {
    let dir = tempfile::tempdir().unwrap();
    let script = dir.path().join("quiet.fake.jsonl");
    std::fs::write(&script, "{\"config\": {\"duration\": 2}}\n").unwrap();
    let (events, code) = run(&["--input", script.to_str().unwrap()]);
    assert_eq!(code, 2);
    assert_eq!(kinds(&events), ["loading", "recording", "transcribing", "no-speech"]);
}

#[test]
fn a_timeout_is_a_stop() {
    let dir = tempfile::tempdir().unwrap();
    let script = long_script(dir.path());
    let (events, code) = run(&["--input", script.to_str().unwrap(), "--timeout", "1"]);
    assert_eq!(code, 0, "{events:?}");
    // Words begun in the first second: "one" to "three" (0.2, 0.52, 0.84 s).
    assert_eq!(events.last().unwrap()["text"], "one two three");
}

#[test]
fn sigterm_is_a_cancel() {
    let dir = tempfile::tempdir().unwrap();
    let script = long_script(dir.path());
    let mut child = dictate(&["--input", script.to_str().unwrap()]);
    let mut lines = Lines::of(&mut child);
    lines.until_final();
    let sent = Instant::now();
    // SAFETY: signals the child we started.
    unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGTERM) };
    let (rest, code) = finish(child, lines);
    assert!(sent.elapsed() < WELL_BEFORE_THE_END, "{:?}", sent.elapsed());
    assert_eq!(code, 3);
    assert!(!kinds(&rest).contains(&"text"), "{rest:?}");
}

#[test]
fn a_recording_plays_through_the_transcriber_asked_for() {
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("room.wav");
    let mut rec = pam_audio::Recorder::create(&wav).unwrap();
    rec.write(&[0.0; 32_000]).unwrap();
    rec.finish().unwrap();
    let script = long_script(dir.path());
    let asr = format!("fake:{}", script.display());
    let (events, code) = run(&["--input", wav.to_str().unwrap(), "--asr", &asr]);
    assert_eq!(code, 0, "{events:?}");
    assert_eq!(events[1]["device"], "room.wav");
    // The recording's two seconds hold every word.
    assert_eq!(events.last().unwrap()["text"], "one two three four five six");
}

#[test]
fn levels_come_ten_a_second_while_listening() {
    let dir = tempfile::tempdir().unwrap();
    // A second of a quiet room (-69 dBFS), then a second of speech-loud sound (-13 dBFS).
    let wav = dir.path().join("room.wav");
    let mut rec = pam_audio::Recorder::create(&wav).unwrap();
    rec.write(&tone(1.0, 0.000_5)).unwrap();
    rec.write(&tone(1.0, 0.3)).unwrap();
    rec.finish().unwrap();
    let asr = format!("fake:{}", long_script(dir.path()).display());
    let (events, code) = run(&["--input", wav.to_str().unwrap(), "--asr", &asr]);
    assert_eq!(code, 0, "{events:?}");

    let all: Vec<&str> = events.iter().map(kind).collect();
    let first = all.iter().position(|k| *k == "level").unwrap();
    let last = all.iter().rposition(|k| *k == "level").unwrap();
    assert_eq!(all[first - 1], "recording", "{all:?}");
    assert!(last < all.iter().position(|k| *k == "transcribing").unwrap(), "{all:?}");

    let levels: Vec<&Value> = events.iter().filter(|e| kind(e) == "level").map(|e| &e["level"]).collect();
    assert!(
        levels.iter().all(|l| l.to_string().split('.').nth(1).is_none_or(|d| d.len() <= 2)),
        "two decimals at most: {levels:?}"
    );
    let levels: Vec<f64> = levels.iter().map(|l| l.as_f64().unwrap()).collect();
    assert_eq!(levels.len(), 20, "one a chunk, 0.1 s each: {levels:?}");
    assert!(levels.iter().all(|l| (0.0..=1.0).contains(l)), "{levels:?}");
    assert!(levels[..10].iter().all(|l| *l == 0.0), "the room is the floor: {levels:?}");
    assert!(levels[10..].iter().all(|l| *l == 1.0), "56 dB over it is full: {levels:?}");
}

#[test]
fn errors_are_an_event_and_exit_1() {
    let (events, code) = run(&["--input", "nowhere.fake.jsonl"]);
    assert_eq!(code, 1);
    assert_eq!(kinds(&events), ["loading", "error"]);
    assert!(events[1]["message"].as_str().unwrap().contains("nowhere.fake.jsonl"), "{events:?}");

    let (events, code) = run(&["--bogus"]);
    assert_eq!(code, 1);
    assert_eq!(kinds(&events), ["error"]);
    assert!(events[0]["message"].as_str().unwrap().contains("unknown option --bogus"));
}

#[test]
fn the_config_file_sets_what_the_flags_leave_out() {
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("room.wav");
    let mut rec = pam_audio::Recorder::create(&wav).unwrap();
    rec.write(&[0.0; 48_000]).unwrap();
    rec.finish().unwrap();
    let script = long_script(dir.path());
    let other = dir.path().join("other.fake.jsonl");
    std::fs::write(&other, "{\"word\": \"elsewhere\", \"at\": 0.2}\n").unwrap();
    let config = dir.path().join("config.toml");
    std::fs::write(&config, format!("asr = \"fake:{}\"\ntimeout = 1\n", script.display())).unwrap();
    let wav = wav.to_str().unwrap();

    // The config's transcriber and its timeout: the words begun in the first second.
    let (events, code) = run_with(&["--input", wav], &config);
    assert_eq!(code, 0, "{events:?}");
    assert_eq!(events.last().unwrap()["text"], "one two three");

    let (events, code) = run_with(&["--input", wav, "--timeout", "0"], &config);
    assert_eq!(code, 0, "{events:?}");
    assert_eq!(events.last().unwrap()["text"], "one two three four five six");

    let asr = format!("fake:{}", other.display());
    let (events, code) = run_with(&["--input", wav, "--asr", &asr], &config);
    assert_eq!(code, 0, "{events:?}");
    assert_eq!(events.last().unwrap()["text"], "elsewhere");
}

#[test]
fn a_config_file_with_an_unknown_key_fails_before_loading() {
    let dir = tempfile::tempdir().unwrap();
    let config = dir.path().join("config.toml");
    std::fs::write(&config, "mic = \"Yeti\"\nvocabulary = [\"obento\"]\n").unwrap();
    let script = fixture("intros.fake.jsonl");
    let child = dictate_with(&["--input", script.to_str().unwrap()], &config);
    let out = child.wait_with_output().unwrap();
    assert_eq!(out.status.code(), Some(1));
    let events: Vec<Value> =
        String::from_utf8(out.stdout).unwrap().lines().map(|l| serde_json::from_str(l).unwrap()).collect();
    assert_eq!(kinds(&events), ["error"]);
    let message = events[0]["message"].as_str().unwrap();
    assert!(message.contains("vocabulary") && message.contains(config.to_str().unwrap()), "{message}");
    assert!(String::from_utf8(out.stderr).unwrap().contains("vocabulary"));
}

#[test]
fn cancel_and_signals_end_it_even_when_no_one_reads_stdout() {
    let dir = tempfile::tempdir().unwrap();
    // Thousands of words: far more events than a pipe holds unread.
    let script = dir.path().join("chatty.fake.jsonl");
    let mut text = String::from("{\"config\": {\"duration\": 600}}\n");
    for i in 0..5_000 {
        text.push_str(&format!("{{\"word\": \"word{i}\", \"at\": {}}}\n", 0.1 * i as f64));
    }
    std::fs::write(&script, text).unwrap();
    for how in ["cancel", "SIGTERM"] {
        let mut child = Command::new(env!("CARGO_BIN_EXE_dictate"))
            .args(["--input", script.to_str().unwrap()])
            .env("DICTATE_REPLAY_SPEED", "1000")
            .env("DICTATE_CONFIG", NO_CONFIG)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .spawn()
            .unwrap();
        // Unread, stdout fills and the engine blocks writing to it. Were it
        // not blocked, the script would be over (0.6 s) and exit 0.
        std::thread::sleep(Duration::from_secs(1));
        let sent = Instant::now();
        if how == "cancel" {
            writeln!(child.stdin.as_mut().unwrap(), "cancel").unwrap();
        } else {
            // SAFETY: signals the child we started.
            unsafe { libc::kill(child.id() as libc::pid_t, libc::SIGTERM) };
        }
        let code = loop {
            if let Some(status) = child.try_wait().unwrap() {
                break status.code();
            }
            if sent.elapsed() > WELL_BEFORE_THE_END {
                child.kill().unwrap();
                panic!("{how} didn't end it with stdout full");
            }
            std::thread::sleep(Duration::from_millis(20));
        };
        assert_eq!(code, Some(3), "{how}");
    }
}

#[test]
fn the_end_of_stdin_is_a_cancel() {
    let dir = tempfile::tempdir().unwrap();
    let script = long_script(dir.path());
    let mut child = dictate(&["--input", script.to_str().unwrap()]);
    let mut lines = Lines::of(&mut child);
    lines.until_final();
    let sent = Instant::now();
    drop(child.stdin.take());
    let (rest, code) = finish(child, lines);
    assert!(sent.elapsed() < WELL_BEFORE_THE_END, "{:?}", sent.elapsed());
    assert_eq!(code, 3);
    assert!(!kinds(&rest).contains(&"text"), "{rest:?}");
}
