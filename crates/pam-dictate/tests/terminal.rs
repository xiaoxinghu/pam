//! The terminal face, run as a person runs it: the `dictate` binary on a
//! pseudo-terminal, which is its stdin and its `/dev/tty`, with stdout a
//! pipe, as in `dictate | cat`, through the `fake` backend. The screen it
//! draws is read back through a terminal emulator.

use std::fs::File;
use std::io::{Read, Write};
use std::num::NonZeroU16;
use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
use std::os::unix::process::CommandExt;
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

/// How much faster than real time scripts play, as in the protocol tests.
const SPEED: &str = "40";
/// Where the tests' config file is: nowhere, so that the config of whoever
/// runs them stays out of it.
const NO_CONFIG: &str = "/nonexistent/dictate/config.toml";
/// The model store: an empty one, so that what whoever runs the tests has
/// pulled stays out of it.
const NO_MODELS: &str = "/nonexistent/pam/models";
/// How long a test waits for the screen to show something.
const PATIENCE: Duration = Duration::from_secs(10);

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../fixtures").join(name)
}

/// A script whose words are all said in the first two seconds, but which
/// lasts ten minutes: time to press a key while it is playing.
fn long_script(dir: &Path) -> PathBuf {
    script(dir, "long.fake.jsonl", 0.3)
}

/// [`long_script`]'s words, but none final before a stop: a preview until then.
fn pending_script(dir: &Path) -> PathBuf {
    script(dir, "pending.fake.jsonl", 900.0)
}

fn script(dir: &Path, name: &str, word_delay: f64) -> PathBuf {
    let path = dir.join(name);
    let mut text = format!("{{\"config\": {{\"word_delay\": {word_delay}, \"duration\": 600}}}}\n");
    for (i, w) in ["one", "two", "three", "four", "five", "six"].iter().enumerate() {
        text.push_str(&format!("{{\"word\": \"{w}\", \"at\": {}}}\n", 0.2 + 0.32 * i as f64));
    }
    std::fs::write(&path, text).unwrap();
    path
}

fn size(rows: u16, cols: u16) -> libc::winsize {
    libc::winsize { ws_row: rows, ws_col: cols, ws_xpixel: 0, ws_ypixel: 0 }
}

/// `dictate` running on a pseudo-terminal of its own.
struct Run {
    child: Child,
    /// The terminal's side: keys are written to it, and the screen read from it.
    master: File,
    screen: Arc<Mutex<vt100::Parser>>,
    reader: JoinHandle<()>,
}

struct Ended {
    code: i32,
    stdout: String,
    stderr: String,
    /// The screen as the terminal shows it once dictate has exited.
    screen: vt100::Screen,
}

impl Run {
    fn start(args: &[&str], rows: u16, cols: u16) -> Run {
        let (mut master, mut slave) = (0, 0);
        // SAFETY: `openpty` fills the two descriptors; the name and termios are left out.
        let opened = unsafe {
            libc::openpty(
                &mut master,
                &mut slave,
                std::ptr::null_mut(),
                std::ptr::null_mut(),
                &mut size(rows, cols),
            )
        };
        assert_eq!(opened, 0, "{}", std::io::Error::last_os_error());
        // SAFETY: both descriptors were just opened, and are owned here alone.
        let (master, slave) = unsafe { (File::from_raw_fd(master), OwnedFd::from_raw_fd(slave)) };
        let mut cmd = Command::new(env!("CARGO_BIN_EXE_dictate"));
        cmd.args(args)
            .env("DICTATE_REPLAY_SPEED", SPEED)
            .env("DICTATE_CONFIG", NO_CONFIG)
            .env("PAM_MODELS", NO_MODELS)
            .stdin(Stdio::from(slave))
            .stdout(Stdio::piped())
            .stderr(Stdio::piped());
        // SAFETY: only async-signal-safe calls between fork and exec. A
        // session of its own makes the pseudo-terminal its controlling
        // terminal, its /dev/tty.
        unsafe {
            cmd.pre_exec(|| {
                if libc::setsid() < 0 || libc::ioctl(0, libc::TIOCSCTTY as _, 0) < 0 {
                    return Err(std::io::Error::last_os_error());
                }
                Ok(())
            });
        }
        let child = cmd.spawn().unwrap();
        // Closes this side's copy of the slave, so reading ends with the child.
        drop(cmd);
        let screen = Arc::new(Mutex::new(vt100::Parser::new(nz(rows), nz(cols), 0)));
        let (mut from, to) = (master.try_clone().unwrap(), screen.clone());
        let reader = std::thread::spawn(move || {
            let mut buf = [0; 4096];
            while let Ok(n @ 1..) = from.read(&mut buf) {
                to.lock().unwrap().process(&buf[..n]);
            }
        });
        Run { child, master, screen, reader }
    }

    /// Waits until the screen shows what `ok` looks for.
    fn wait_for(&self, what: &str, ok: impl Fn(&vt100::Screen) -> bool) {
        let start = Instant::now();
        loop {
            {
                let parser = self.screen.lock().unwrap();
                if ok(parser.screen()) {
                    return;
                }
                if start.elapsed() > PATIENCE {
                    panic!("the screen never showed {what}:\n{}", parser.screen().contents());
                }
            }
            std::thread::sleep(Duration::from_millis(20));
        }
    }

    fn wait_for_text(&self, text: &str) {
        self.wait_for(&format!("{text:?}"), |s| s.contents().contains(text));
    }

    fn key(&mut self, bytes: &[u8]) {
        self.master.write_all(bytes).unwrap();
    }

    /// Resizes the terminal, which signals dictate as a terminal window does.
    fn resize(&mut self, rows: u16, cols: u16) {
        // SAFETY: sets the window size of the pseudo-terminal opened above.
        assert_eq!(unsafe { libc::ioctl(self.master.as_raw_fd(), libc::TIOCSWINSZ, &size(rows, cols)) }, 0);
        self.screen.lock().unwrap().set_size(nz(rows), nz(cols));
    }

    fn end(self) -> Ended {
        let out = self.child.wait_with_output().unwrap();
        self.reader.join().unwrap();
        let screen = self.screen.lock().unwrap().screen().clone();
        Ended {
            code: out.status.code().unwrap(),
            stdout: String::from_utf8(out.stdout).unwrap(),
            stderr: String::from_utf8(out.stderr).unwrap(),
            screen,
        }
    }
}

/// The box drawn over the whole screen, intact: its four corners in place.
fn framed(s: &vt100::Screen) -> bool {
    let (rows, cols) = s.size();
    let (rows, cols) = (rows.get(), cols.get());
    let at = |r, c| s.cell(r, c).map_or("", |cell| cell.contents()).to_string();
    s.alternate_screen()
        && at(0, 0) == "╭"
        && at(0, cols - 1) == "╮"
        && at(rows - 1, 0) == "╰"
        && at(rows - 1, cols - 1) == "╯"
}

/// Whether the first word inside the box is dim: the preview, not a final.
fn dim_first_word(s: &vt100::Screen) -> bool {
    s.cell(1, 2).is_some_and(|cell| cell.dim())
}

#[test]
fn the_words_show_as_they_come_and_enter_prints_the_text() {
    let dir = tempfile::tempdir().unwrap();
    let script = long_script(dir.path());
    let mut run = Run::start(&["--input", script.to_str().unwrap()], 12, 70);
    run.wait_for("the frame", framed);
    run.wait_for_text("long.fake.jsonl");
    run.wait_for_text("Enter finish");
    run.wait_for_text("one two three");
    run.wait_for("the finals in the normal style", |s| !dim_first_word(s));
    run.key(b"\r");
    let end = run.end();
    assert_eq!(end.code, 0, "{}", end.stderr);
    assert!(end.stdout.starts_with("one two three"), "{:?}", end.stdout);
    assert!(end.stdout.ends_with('\n') && end.stdout.lines().count() == 1, "{:?}", end.stdout);
    assert!(!end.stdout.contains('\x1b'), "nothing of the view goes to stdout: {:?}", end.stdout);
    assert_eq!(end.stderr, "");
    assert!(!end.screen.alternate_screen(), "the view closes");
}

#[test]
fn the_preview_is_dim_and_a_stop_settles_it() {
    let dir = tempfile::tempdir().unwrap();
    let script = pending_script(dir.path());
    let mut run = Run::start(&["--input", script.to_str().unwrap()], 12, 70);
    run.wait_for_text("one two three four five six");
    run.wait_for("a dim preview", dim_first_word);
    run.key(b"\r");
    let end = run.end();
    assert_eq!((end.code, end.stdout.as_str()), (0, "one two three four five six\n"), "{}", end.stderr);
}

#[test]
fn escape_ctrl_c_and_sigterm_discard_it() {
    let dir = tempfile::tempdir().unwrap();
    let script = long_script(dir.path());
    for how in ["Escape", "Ctrl-C", "SIGTERM"] {
        let mut run = Run::start(&["--input", script.to_str().unwrap()], 12, 70);
        run.wait_for_text("one two");
        let sent = Instant::now();
        match how {
            "Escape" => run.key(b"\x1b"),
            "Ctrl-C" => run.key(b"\x03"),
            // SAFETY: signals the child we started.
            _ => assert_eq!(unsafe { libc::kill(run.child.id() as libc::pid_t, libc::SIGTERM) }, 0),
        }
        let end = run.end();
        assert!(sent.elapsed() < Duration::from_secs(5), "{:?}", sent.elapsed());
        assert_eq!((end.code, end.stdout.as_str(), end.stderr.as_str()), (3, "", ""), "{how}");
        assert!(!end.screen.alternate_screen(), "a cancel puts the terminal back too");
    }
}

#[test]
fn a_script_that_plays_out_prints_its_words() {
    let script = fixture("intros.fake.jsonl");
    let run = Run::start(&["--input", script.to_str().unwrap()], 12, 70);
    let end = run.end();
    assert_eq!(end.code, 0, "{}", end.stderr);
    let words = pam_speech::fake::Script::load(&script).unwrap().words;
    let said: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(end.stdout, said.join(" ") + "\n");
    assert!(!end.screen.alternate_screen());
}

#[test]
fn no_speech_and_errors_are_a_sentence_on_stderr() {
    let dir = tempfile::tempdir().unwrap();
    let quiet = dir.path().join("quiet.fake.jsonl");
    std::fs::write(&quiet, "{\"config\": {\"duration\": 2}}\n").unwrap();
    let end = Run::start(&["--input", quiet.to_str().unwrap()], 12, 70).end();
    assert_eq!((end.code, end.stdout.as_str()), (2, ""));
    assert_eq!(end.stderr, "dictate: nothing was heard.\n");

    let end = Run::start(&["--input", "nowhere.fake.jsonl"], 12, 70).end();
    assert_eq!((end.code, end.stdout.as_str()), (1, ""));
    assert!(
        end.stderr.starts_with("dictate: ") && end.stderr.contains("nowhere.fake.jsonl"),
        "{}",
        end.stderr
    );
    assert!(!end.screen.alternate_screen());

    // A usage error comes before any view, and prints no event either.
    let end = Run::start(&["--bogus"], 12, 70).end();
    assert_eq!((end.code, end.stdout.as_str()), (1, ""));
    assert!(end.stderr.contains("unknown option --bogus"), "{}", end.stderr);
}

#[cfg(feature = "nemo-speech")]
#[test]
fn a_missing_model_is_a_sentence_before_any_view() {
    let dir = tempfile::tempdir().unwrap();
    let wav = dir.path().join("room.wav");
    let mut rec = pam_audio::Recorder::create(&wav).unwrap();
    rec.write(&[0.0; 16_000]).unwrap();
    rec.finish().unwrap();
    let end = Run::start(&["--input", wav.to_str().unwrap()], 12, 70).end();
    assert_eq!((end.code, end.stdout.as_str()), (1, ""));
    assert!(
        end.stderr.starts_with("dictate: model `nemotron-en` isn't in the model store"),
        "{}",
        end.stderr
    );
    assert!(end.stderr.contains("run `dictate models pull nemotron-en`"), "{}", end.stderr);
    assert!(!end.screen.alternate_screen());
}

#[test]
fn json_keeps_the_piped_face_on_a_terminal() {
    let dir = tempfile::tempdir().unwrap();
    let quiet = dir.path().join("quiet.fake.jsonl");
    std::fs::write(&quiet, "{\"config\": {\"duration\": 2}}\n").unwrap();
    let end = Run::start(&["--input", quiet.to_str().unwrap(), "--json"], 12, 70).end();
    assert_eq!(end.code, 2);
    assert!(end.stdout.ends_with("{\"event\":\"no-speech\"}\n"), "{:?}", end.stdout);
    assert!(!end.screen.alternate_screen());
}

#[test]
fn resizing_redraws_the_view() {
    let dir = tempfile::tempdir().unwrap();
    let script = long_script(dir.path());
    let mut run = Run::start(&["--input", script.to_str().unwrap()], 12, 70);
    run.wait_for_text("one two three");
    for (rows, cols) in [(6, 30), (20, 100), (4, 16)] {
        run.resize(rows, cols);
        run.wait_for(&format!("the frame at {rows}×{cols}"), framed);
    }
    run.resize(12, 70);
    run.wait_for("the frame again", framed);
    run.wait_for_text("one two three");
    run.key(b"\r");
    let end = run.end();
    assert_eq!(end.code, 0, "{}", end.stderr);
    assert!(end.stdout.starts_with("one two three"), "{:?}", end.stdout);
}

/// Two minutes of dictation at real speed, the terminal resized halfway,
/// then Enter: slow, so run it by hand with `--ignored`.
#[test]
#[ignore]
fn a_two_minute_dictation_at_real_speed() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("two-minutes.fake.jsonl");
    let mut text = String::from("{\"config\": {\"duration\": 600}}\n");
    let mut said = Vec::new();
    // 300 words over two minutes, a sentence every 12.
    for i in 0..300 {
        let w = format!("w{i}{}", if i % 12 == 11 { "." } else { "" });
        text.push_str(&format!("{{\"word\": \"{w}\", \"at\": {}}}\n", 0.4 * i as f64));
        said.push(w);
    }
    std::fs::write(&path, text).unwrap();
    let mut run = Run::start(&["--input", path.to_str().unwrap(), "--realtime"], 10, 60);
    let start = Instant::now();
    run.wait_for_text("w3");
    std::thread::sleep(Duration::from_secs(60));
    run.resize(16, 90);
    // Every second until the last word shows, the frame is whole.
    while !run.screen.lock().unwrap().screen().contents().contains("w299.") {
        run.wait_for("the frame", framed);
        assert!(start.elapsed() < Duration::from_secs(150), "the last word never showed");
        std::thread::sleep(Duration::from_secs(1));
    }
    run.wait_for("the frame at the end", framed);
    run.wait_for_text("2:0");
    let screen = run.screen.lock().unwrap().screen().contents();
    assert!(!screen.contains("w0 "), "the first words have scrolled away: {screen}");
    run.key(b"\r");
    let end = run.end();
    assert_eq!(end.code, 0, "{}", end.stderr);
    assert_eq!(end.stdout, said.join(" ") + "\n");
}

fn nz(n: u16) -> NonZeroU16 {
    NonZeroU16::new(n).unwrap()
}
