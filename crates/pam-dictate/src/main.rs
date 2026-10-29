//! `dictate`: listens to the mic, transcribes live and hands back the text.
//!
//! It has two faces on one engine. Piped, it speaks JSON lines: events on
//! stdout, commands (`stop`, `cancel`) on stdin, one per line; or, with
//! `--input -`, stdin is the audio, and its end the stop. A shell
//! (Portal, Hammerspoon) drives it; the shortcut, panel and pasting are the
//! shell's. On a terminal, a person drives it: the view shows the words and
//! takes Enter or Escape, and only the text goes to stdout.

mod cli;
mod config;
mod engine;
mod models;
mod view;

use std::io::{BufRead, IsTerminal, Write};
use std::process::ExitCode;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::{Arc, Mutex, TryLockError};
use std::time::{Duration, Instant};

use config::Config;
use engine::{Event, Input, Outcome};

/// Exit codes: the text was heard, nothing was, cancelled, failed.
const TEXT: u8 = 0;
const NO_SPEECH: u8 = 2;
const CANCELLED: i32 = 3;
const FAILED: u8 = 1;

fn main() -> ExitCode {
    let args: Vec<String> = std::env::args_os().skip(1).map(|a| a.to_string_lossy().into_owned()).collect();
    if args.first().is_some_and(|a| a == "models") {
        return models::run(&args[1..]);
    }
    let terminal = std::io::stdin().is_terminal();
    let flags = match cli::parse(&args) {
        Ok(cli::Parsed::Run(flags)) => flags,
        Ok(cli::Parsed::Help) => {
            print!("{}", cli::manual());
            return ExitCode::SUCCESS;
        }
        Ok(cli::Parsed::Version) => {
            println!("dictate {}", env!("CARGO_PKG_VERSION"));
            return ExitCode::SUCCESS;
        }
        // Unparsed, `--json` can only have been given bare.
        Err(msg) => return usage_error(msg, !terminal || args.iter().any(|a| a == "--json")),
    };
    // The face follows stdin, never stdout, so stdout stays free for the text.
    let piped = !terminal || flags.json();
    let config = match Config::load() {
        Ok(config) => config,
        Err(e) => {
            let message = format!("{e:#}");
            if piped {
                emit(Event::Error { message: message.clone() });
            }
            eprintln!("error: {message}");
            return ExitCode::from(FAILED);
        }
    };
    let opts = match flags.resolve(&config) {
        Ok(opts) => opts,
        Err(msg) => return usage_error(msg, piped),
    };
    if terminal && matches!(opts.input, Input::Stdin) {
        let msg = "--input - reads audio from stdin, which is a terminal: pipe raw 16 kHz f32le samples in";
        return usage_error(msg.into(), piped);
    }
    // Before the mic opens. Nothing is downloaded on a hotkey's press, where it
    // would read as a hang and fail offline.
    if let Err(e) = models::check(&opts) {
        if !piped {
            return said(Err(e));
        }
        emit(Event::Error { message: format!("{e:#}") });
        return ExitCode::from(FAILED);
    }
    cancel_on_signals();
    let stop = Arc::new(AtomicBool::new(false));
    if !piped {
        return said(view::run(&opts, &stop));
    }
    if !matches!(opts.input, Input::Stdin) {
        read_commands(stop.clone());
    }
    match engine::run(&opts, &stop, &mut emit) {
        Ok(Outcome::Text(_)) => ExitCode::from(TEXT),
        Ok(Outcome::NoSpeech) => ExitCode::from(NO_SPEECH),
        Err(e) => {
            emit(Event::Error { message: format!("{e:#}") });
            ExitCode::from(FAILED)
        }
    }
}

/// Reports a usage error as an event, and on stderr with the manual.
fn usage_error(msg: String, piped: bool) -> ExitCode {
    if piped {
        emit(Event::Error { message: msg.clone() });
    }
    eprint!("error: {msg}\n\n{}", cli::manual());
    ExitCode::from(FAILED)
}

/// The terminal face's ending, once the view has closed: the text on stdout,
/// once; anything else a sentence on stderr.
fn said(ran: anyhow::Result<Outcome>) -> ExitCode {
    match ran {
        Ok(Outcome::Text(text)) => {
            let mut out = std::io::stdout().lock();
            if writeln!(out, "{text}").and_then(|()| out.flush()).is_err() {
                // As in the piped face: with no one to read it, it is a cancel.
                drop(out);
                cancel();
            }
            ExitCode::from(TEXT)
        }
        Ok(Outcome::NoSpeech) => {
            eprintln!("dictate: nothing was heard.");
            ExitCode::from(NO_SPEECH)
        }
        Err(e) => {
            eprintln!("dictate: {e:#}.");
            ExitCode::from(FAILED)
        }
    }
}

/// Held while an event is written or the view drawn, so a cancel waits for
/// the line or the frame to end.
static WRITING: Mutex<()> = Mutex::new(());

/// How long a cancel waits for a line being written: a reader that has
/// stopped reading may never let it end.
const LAST_LINE: Duration = Duration::from_millis(200);

/// Writes one event as a line on stdout and flushes it. With no one left
/// to read it, there is nothing more to do than a cancel.
fn emit(event: Event) {
    let mut line = serde_json::to_string(&event).expect("events serialize");
    line.push('\n');
    let writing = WRITING.lock().unwrap_or_else(|e| e.into_inner());
    let mut out = std::io::stdout().lock();
    if out.write_all(line.as_bytes()).and_then(|()| out.flush()).is_err() {
        drop((out, writing));
        cancel();
    }
}

/// Exits at once with nothing more. It lets a line being written or a frame
/// being drawn end first, for up to [`LAST_LINE`], puts the terminal back if
/// the view is up, and skips the exit handlers, which a speech backend in the
/// middle of a call on another thread may not survive.
fn cancel() -> ! {
    let deadline = Instant::now() + LAST_LINE;
    let _writing = loop {
        match WRITING.try_lock() {
            Ok(guard) => break Some(guard),
            Err(TryLockError::Poisoned(e)) => break Some(e.into_inner()),
            Err(TryLockError::WouldBlock) if Instant::now() < deadline => {
                std::thread::sleep(Duration::from_millis(5))
            }
            Err(TryLockError::WouldBlock) => break None,
        }
    };
    view::close();
    // SAFETY: `_exit` only ends the process.
    unsafe { libc::_exit(CANCELLED) }
}

/// Reads the commands on stdin: `stop` sets `stop`; `cancel` exits. So does
/// the end of stdin: whoever drove dictate is gone, and nobody would get the text.
fn read_commands(stop: Arc<AtomicBool>) {
    std::thread::spawn(move || {
        for line in std::io::stdin().lock().lines() {
            let Ok(line) = line else { break };
            match line.trim() {
                "stop" => stop.store(true, Ordering::Relaxed),
                "cancel" => cancel(),
                "" => {}
                other => eprintln!("dictate: unknown command `{other}`: send stop or cancel"),
            }
        }
        cancel();
    });
}

/// The pipe a signal handler writes to, for the thread that cancels.
static SIGNALLED: AtomicI32 = AtomicI32::new(-1);

/// SIGINT and SIGTERM mean `cancel`. The handler only writes a byte to a
/// pipe; a thread waiting on it does the rest.
fn cancel_on_signals() {
    let mut fds = [0; 2];
    // SAFETY: `fds` holds the two descriptors `pipe` fills.
    if unsafe { libc::pipe(fds.as_mut_ptr()) } != 0 {
        eprintln!(
            "dictate: {}: SIGINT and SIGTERM will end it without a word",
            std::io::Error::last_os_error()
        );
        return;
    }
    SIGNALLED.store(fds[1], Ordering::Relaxed);
    extern "C" fn on_signal(_: libc::c_int) {
        let fd = SIGNALLED.load(Ordering::Relaxed);
        // SAFETY: `write` is async-signal-safe; the byte's value doesn't matter.
        unsafe { libc::write(fd, [0u8].as_ptr().cast(), 1) };
    }
    std::thread::spawn(move || {
        let mut byte = 0u8;
        // SAFETY: reads one byte into `byte` from the pipe's read end.
        while unsafe { libc::read(fds[0], (&raw mut byte).cast(), 1) } < 0 {}
        cancel();
    });
    for sig in [libc::SIGINT, libc::SIGTERM] {
        // SAFETY: the handler only calls async-signal-safe functions.
        unsafe { libc::signal(sig, on_signal as *const () as libc::sighandler_t) };
    }
}
