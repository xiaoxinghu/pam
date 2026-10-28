//! The terminal face: the engine's events drawn on the controlling terminal,
//! and Enter or Escape read from it, while stdout waits for the text.
//!
//! The view is drawn on `/dev/tty`, as fzf draws, so `dictate | pbcopy` and
//! `dictate > note.txt` show it too. While it is up, stderr is held back, so
//! that nothing the mic or a speech library says scribbles on it, and is
//! written once it closes.

use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::os::fd::FromRawFd;
use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::{Arc, Once};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use pam_speech::SAMPLE_RATE;
use ratatui::backend::CrosstermBackend;
use ratatui::crossterm::event::{self as term, KeyCode, KeyEventKind, KeyModifiers};
use ratatui::crossterm::{cursor, execute, terminal};
use ratatui::style::{Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, BorderType, Padding, Paragraph, Wrap};
use ratatui::{Frame, Terminal};

use crate::engine::{self, Event, Options, Outcome};

/// How often the view is drawn, for the clock; a key or a resize wakes it sooner.
const FRAME: Duration = Duration::from_millis(80);

type Tty = Terminal<CrosstermBackend<BufWriter<File>>>;

/// What the engine's thread sends the view.
enum Msg {
    Event(Event),
    Done(Result<Outcome>),
}

/// Runs one dictation with the view up: Enter sets `stop`; Escape and Ctrl-C
/// cancel. Returns once the view has closed and what stderr held is written.
pub fn run(opts: &Options, stop: &Arc<AtomicBool>) -> Result<Outcome> {
    let (mut tty, held) = match open() {
        Ok(up) => up,
        Err(e) => {
            close();
            return Err(e.context("opening the view on /dev/tty"));
        }
    };
    let (tx, rx) = mpsc::channel();
    {
        let (opts, stop) = (opts.clone(), stop.clone());
        std::thread::spawn(move || {
            let events = tx.clone();
            let ran = engine::run(&opts, &stop, &mut |e| drop(events.send(Msg::Event(e))));
            let _ = tx.send(Msg::Done(ran));
        });
    }
    let mut view = View::new(opts);
    let ran = drive(&mut tty, &mut view, &rx, stop);
    drop(tty);
    close();
    held.write();
    ran.context("drawing the view")?
}

/// Draws the view and reads keys until the engine is done.
fn drive(tty: &mut Tty, view: &mut View, rx: &Receiver<Msg>, stop: &AtomicBool) -> Result<Result<Outcome>> {
    loop {
        loop {
            match rx.try_recv() {
                Ok(Msg::Event(e)) => view.apply(e, Instant::now()),
                Ok(Msg::Done(ran)) => return Ok(ran),
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => {
                    return Ok(Err(anyhow!("the engine stopped unexpectedly")));
                }
            }
        }
        {
            let _writing = crate::WRITING.lock().unwrap_or_else(|e| e.into_inner());
            tty.draw(|f| view.draw(f, Instant::now()))?;
        }
        if term::poll(FRAME)?
            && let term::Event::Key(key) = term::read()?
            && key.kind == KeyEventKind::Press
        {
            match key.code {
                KeyCode::Enter => stop.store(true, Ordering::Relaxed),
                KeyCode::Esc => crate::cancel(),
                KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => crate::cancel(),
                _ => {}
            }
        }
    }
}

/// Whether the terminal is in raw mode on the alternate screen, for [`close`].
static UP: AtomicBool = AtomicBool::new(false);
/// stderr as it was before [`hold_stderr`]; -1 when it isn't held.
static STDERR: AtomicI32 = AtomicI32::new(-1);

/// Holds stderr back, puts the terminal in raw mode on the alternate screen,
/// and readies a terminal drawing on `/dev/tty`. Whatever this leaves half
/// done, [`close`] undoes.
fn open() -> Result<(Tty, Held)> {
    let out = OpenOptions::new().write(true).open("/dev/tty").context("opening /dev/tty")?;
    close_on_panic();
    let held = hold_stderr()?;
    UP.store(true, Ordering::SeqCst);
    terminal::enable_raw_mode().context("raw mode")?;
    let mut out = BufWriter::new(out);
    execute!(out, terminal::EnterAlternateScreen, cursor::Hide)?;
    Ok((Terminal::new(CrosstermBackend::new(out))?, held))
}

/// Puts the terminal back and gives stderr back, if they were taken; safe to
/// call from any thread, and more than once. What stderr held is written by
/// [`Held::write`], which a cancel never calls.
pub fn close() {
    if UP.swap(false, Ordering::SeqCst) {
        if let Ok(mut tty) = OpenOptions::new().write(true).open("/dev/tty") {
            let _ = execute!(tty, terminal::LeaveAlternateScreen, cursor::Show);
        }
        let _ = terminal::disable_raw_mode();
    }
    let fd = STDERR.swap(-1, Ordering::SeqCst);
    if fd >= 0 {
        // SAFETY: puts back the descriptor `hold_stderr` saved, and closes its copy.
        unsafe {
            libc::dup2(fd, libc::STDERR_FILENO);
            libc::close(fd);
        }
    }
}

/// A panic on the view's thread closes it first, so its message lands on a
/// working terminal. A panic on the engine's thread leaves the view up; its
/// message is held with the rest of stderr, and the view closes on the error.
fn close_on_panic() {
    static ONCE: Once = Once::new();
    let view = std::thread::current().id();
    ONCE.call_once(|| {
        let previous = std::panic::take_hook();
        std::panic::set_hook(Box::new(move |info| {
            if std::thread::current().id() == view {
                close();
            }
            previous(info)
        }));
    });
}

/// What stderr said while the view was up.
struct Held(JoinHandle<Vec<u8>>);

impl Held {
    /// Writes it on stderr, which [`close`] must have given back.
    fn write(self) {
        if let Ok(said) = self.0.join() {
            let _ = std::io::stderr().write_all(&said);
        }
    }
}

/// Points stderr at a pipe that a thread drains, until [`close`].
fn hold_stderr() -> Result<Held> {
    let mut fds = [0; 2];
    // SAFETY: `fds` holds the two descriptors `pipe` fills; `fcntl` and
    // `dup2` only flag and copy descriptors this process owns. Only stderr
    // itself goes to a program dictate starts, as it would without the view.
    unsafe {
        if libc::pipe(fds.as_mut_ptr()) != 0 {
            return Err(std::io::Error::last_os_error()).context("holding stderr");
        }
        libc::fcntl(fds[0], libc::F_SETFD, libc::FD_CLOEXEC);
        let saved = libc::fcntl(libc::STDERR_FILENO, libc::F_DUPFD_CLOEXEC, 0);
        if saved < 0 || libc::dup2(fds[1], libc::STDERR_FILENO) < 0 {
            let e = std::io::Error::last_os_error();
            libc::close(fds[0]);
            libc::close(fds[1]);
            if saved >= 0 {
                libc::close(saved);
            }
            return Err(e).context("holding stderr");
        }
        libc::close(fds[1]);
        STDERR.store(saved, Ordering::SeqCst);
    }
    // SAFETY: the pipe's read end is ours alone.
    let mut pipe = unsafe { File::from_raw_fd(fds[0]) };
    Ok(Held(std::thread::spawn(move || {
        let mut said = Vec::new();
        let _ = std::io::Read::read_to_end(&mut pipe, &mut said);
        said
    })))
}

/// What the view shows: the engine's events so far, and the clock.
#[derive(Debug)]
struct View {
    /// The mic's or the file's name, once recording.
    device: Option<String>,
    finals: Vec<String>,
    preview: String,
    /// When recording began, and when it stopped.
    began: Option<Instant>,
    stopped: Option<Instant>,
    timeout: Option<Duration>,
}

impl View {
    fn new(opts: &Options) -> View {
        View {
            device: None,
            finals: Vec::new(),
            preview: String::new(),
            began: None,
            stopped: None,
            timeout: opts.timeout.map(|ts| Duration::from_secs_f64(ts as f64 / SAMPLE_RATE as f64)),
        }
    }

    fn apply(&mut self, e: Event, now: Instant) {
        match e {
            Event::Recording { device } => {
                self.device = Some(device);
                self.began = Some(now);
            }
            Event::Preview { text } => self.preview = text,
            Event::Final { text } => {
                self.finals.push(text);
                // After a stop the flushed words come as finals, with no
                // preview after them to take theirs away.
                if self.stopped.is_some() {
                    self.preview.clear();
                }
            }
            Event::Transcribing => self.stopped = Some(now),
            Event::Loading
            | Event::Level { .. }
            | Event::Text { .. }
            | Event::NoSpeech
            | Event::Error { .. }
            | Event::Downloading { .. } => {}
        }
    }

    /// How long it has been recording.
    fn elapsed(&self, now: Instant) -> Duration {
        self.began.map_or(Duration::ZERO, |b| self.stopped.unwrap_or(now).saturating_duration_since(b))
    }

    /// A box over the whole terminal: the device and the clock on top, the
    /// words inside, the latest kept in view, and the keys below.
    fn draw(&self, f: &mut Frame, now: Instant) {
        let dim = Style::new().add_modifier(Modifier::DIM);
        let (top, bottom) = match (&self.device, self.stopped) {
            (None, _) => (" loading… ".to_string(), " Esc discard "),
            (Some(device), None) => (format!(" {device} "), " ● recording · Enter finish · Esc discard "),
            (Some(device), Some(_)) => (format!(" {device} "), " transcribing… · Esc discard "),
        };
        let mut block = Block::bordered()
            .border_type(BorderType::Rounded)
            .padding(Padding::horizontal(1))
            .title(Line::from(vec![" dictate ".bold(), "·".into(), top.into()]))
            .title_bottom(Line::styled(bottom, dim));
        if self.began.is_some() {
            let elapsed = self.elapsed(now);
            let mut clock = format!(" {}", clock(elapsed));
            if let Some(timeout) = self.timeout {
                // Rounded up, so the time gone and the time left add up to the timeout.
                let left = timeout.saturating_sub(elapsed);
                clock += &format!(" · {} left", clock_up(left));
            }
            block = block.title(Line::from(clock + " ").right_aligned());
        }

        let mut words = Vec::new();
        if !self.finals.is_empty() {
            words.push(Span::raw(self.finals.join(" ")));
        }
        if !self.preview.is_empty() {
            if !words.is_empty() {
                words.push(Span::raw(" "));
            }
            words.push(Span::styled(self.preview.as_str(), dim));
        }
        let text = Paragraph::new(Line::from(words)).wrap(Wrap { trim: true });
        let inner = block.inner(f.area());
        let hidden = text.line_count(inner.width).saturating_sub(inner.height as usize);
        f.render_widget(block, f.area());
        f.render_widget(text.scroll((hidden.min(u16::MAX as usize) as u16, 0)), inner);
    }
}

/// `m:ss`, the seconds rounded down.
fn clock(d: Duration) -> String {
    let s = d.as_secs();
    format!("{}:{:02}", s / 60, s % 60)
}

/// `m:ss`, the seconds rounded up.
fn clock_up(d: Duration) -> String {
    clock(Duration::from_secs(d.as_secs() + u64::from(d.subsec_nanos() > 0)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::Input;
    use ratatui::backend::TestBackend;
    use ratatui::buffer::Buffer;

    fn view(timeout: Option<f64>) -> View {
        View::new(&Options {
            input: Input::Mic(None),
            asr: String::new(),
            language: String::new(),
            hints: Vec::new(),
            hints_file: None,
            timeout: timeout.map(|s| (s * SAMPLE_RATE as f64) as u64),
            realtime: false,
        })
    }

    fn draw(v: &View, now: Instant, width: u16, height: u16) -> Buffer {
        let mut term = Terminal::new(TestBackend::new(width, height)).unwrap();
        term.draw(|f| v.draw(f, now)).unwrap();
        term.backend().buffer().clone()
    }

    fn rows(b: &Buffer) -> Vec<String> {
        (0..b.area.height)
            .map(|y| (0..b.area.width).map(|x| b[(x, y)].symbol()).collect::<String>())
            .collect()
    }

    /// Where `text` starts on screen, if it is there.
    fn find(b: &Buffer, text: &str) -> Option<(u16, u16)> {
        rows(b)
            .iter()
            .enumerate()
            .find_map(|(y, row)| row.find(text).map(|i| (row[..i].chars().count() as u16, y as u16)))
    }

    fn dim(b: &Buffer, (x, y): (u16, u16)) -> bool {
        b[(x, y)].modifier.contains(Modifier::DIM)
    }

    #[test]
    fn it_shows_loading_until_recording() {
        let t0 = Instant::now();
        let mut v = view(None);
        v.apply(Event::Loading, t0);
        let b = draw(&v, t0, 60, 6);
        assert!(find(&b, "loading…").is_some(), "{:#?}", rows(&b));
        assert!(find(&b, "0:00").is_none());
        v.apply(Event::Recording { device: "Yeti Stereo Microphone".into() }, t0);
        let b = draw(&v, t0 + Duration::from_secs(75), 60, 6);
        assert!(find(&b, "loading").is_none());
        assert!(find(&b, "Yeti Stereo Microphone").is_some(), "{:#?}", rows(&b));
        assert!(find(&b, "1:15").is_some(), "{:#?}", rows(&b));
        assert!(find(&b, "Enter finish").is_some());
    }

    #[test]
    fn finals_are_normal_and_the_preview_dim() {
        let t0 = Instant::now();
        let mut v = view(None);
        v.apply(Event::Recording { device: "mic".into() }, t0);
        v.apply(Event::Final { text: "So there.".into() }, t0);
        v.apply(Event::Preview { text: "and then".into() }, t0);
        let b = draw(&v, t0, 60, 6);
        assert!(!dim(&b, find(&b, "So there.").unwrap()));
        assert!(dim(&b, find(&b, "and then").unwrap()), "{:#?}", rows(&b));
        assert!(find(&b, "So there. and then").is_some());
    }

    #[test]
    fn the_clock_counts_down_under_a_timeout_and_stops_with_the_recording() {
        let t0 = Instant::now();
        let mut v = view(Some(60.0));
        v.apply(Event::Recording { device: "mic".into() }, t0);
        let b = draw(&v, t0 + Duration::from_millis(12_500), 60, 6);
        assert!(find(&b, "0:12 · 0:48 left").is_some(), "{:#?}", rows(&b));
        v.apply(Event::Transcribing, t0 + Duration::from_secs(20));
        let b = draw(&v, t0 + Duration::from_secs(90), 60, 6);
        assert!(find(&b, "0:20 · 0:40 left").is_some(), "{:#?}", rows(&b));
        assert!(find(&b, "transcribing…").is_some());
    }

    #[test]
    fn the_flushed_finals_take_the_preview_s_place() {
        let t0 = Instant::now();
        let mut v = view(None);
        v.apply(Event::Recording { device: "mic".into() }, t0);
        v.apply(Event::Preview { text: "almost done".into() }, t0);
        v.apply(Event::Transcribing, t0);
        assert_eq!(v.preview, "almost done", "the preview stays until its words settle");
        v.apply(Event::Final { text: "Almost done.".into() }, t0);
        assert_eq!((v.finals.as_slice(), v.preview.as_str()), (&["Almost done.".to_string()][..], ""));
    }

    #[test]
    fn a_long_dictation_keeps_its_latest_words_in_view() {
        let t0 = Instant::now();
        let mut v = view(Some(120.0));
        v.apply(Event::Recording { device: "mic".into() }, t0);
        // Two minutes of speech, about 300 words.
        for i in 0..60 {
            v.apply(Event::Final { text: format!("sentence {i} has five words.") }, t0);
        }
        v.apply(Event::Preview { text: "the very last".into() }, t0);
        let b = draw(&v, t0 + Duration::from_secs(120), 40, 8);
        assert!(find(&b, "the very last").is_some(), "{:#?}", rows(&b));
        assert!(find(&b, "sentence 0 ").is_none());
        assert!(find(&b, "dictate").is_some(), "the frame stays: {:#?}", rows(&b));
        // However small the terminal, it draws.
        for (w, h) in [(1, 1), (5, 2), (12, 3), (200, 3)] {
            draw(&v, t0, w, h);
        }
    }

    #[test]
    fn clocks_round_down_and_up() {
        assert_eq!(clock(Duration::from_millis(59_999)), "0:59");
        assert_eq!(clock_up(Duration::from_millis(59_001)), "1:00");
        assert_eq!(clock_up(Duration::from_secs(60)), "1:00");
        assert_eq!(clock(Duration::from_secs(3_725)), "62:05");
    }
}
