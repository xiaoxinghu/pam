//! The engine: opens the input and the transcriber, pushes audio through it
//! and turns what it hears into events, until a stop.

use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result, bail};
use pam_audio::{Meter, Mic, Pacer, RawSource, Source, WavSource};
use pam_speech::{SAMPLE_RATE, Transcriber, TranscriberConfig, Ts, Update, Word, join_words};
use serde::Serialize;

/// Samples pushed at a time: 0.1 s, as a meeting pushes them.
const CHUNK: usize = 1_600;

/// What the engine reports, one JSON object per line.
#[derive(Debug, Clone, PartialEq, Serialize)]
#[serde(tag = "event", rename_all = "kebab-case")]
pub enum Event {
    /// The transcriber and the input are opening.
    Loading,
    /// Listening to `device`: the mic's name, or the file replayed.
    Recording {
        device: String,
    },
    /// How loud the input is, from 0 to 1 against the room's noise floor,
    /// for each chunk listened to: ten a second, to draw a level meter by.
    Level {
        level: f32,
    },
    /// The words not settled yet; replaces the previous preview.
    Preview {
        text: String,
    },
    /// One settled run of words, appended to those before, and each word
    /// with its times.
    Final {
        text: String,
        words: Vec<TimedWord>,
    },
    /// Stopped; the last words are settling.
    Transcribing,
    /// Every final, joined by single spaces.
    Text {
        text: String,
    },
    /// Stopped, and nothing was heard.
    NoSpeech,
    Error {
        message: String,
    },
    /// `dictate models pull`: how many of `model`'s `total` bytes are in the
    /// store so far.
    Downloading {
        model: String,
        bytes: u64,
        total: u64,
    },
}

/// A word of a `final`, and when it was said: seconds from the start of the
/// input, to the millisecond.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct TimedWord {
    pub text: String,
    pub start: f64,
    pub end: f64,
}

impl From<&Word> for TimedWord {
    fn from(w: &Word) -> TimedWord {
        let secs = |ts: Ts| (ts as f64 / SAMPLE_RATE as f64 * 1_000.0).round() / 1_000.0;
        TimedWord { text: w.text.clone(), start: secs(w.start), end: secs(w.end) }
    }
}

#[derive(Debug, Clone)]
pub enum Input {
    /// A microphone by name, or the default input.
    Mic(Option<String>),
    /// A 16 kHz WAV file.
    Wav(PathBuf),
    /// Raw 16 kHz mono f32le samples on stdin, which then carries no commands.
    Stdin,
    /// A fake script, which is also the transcriber: silence of its length
    /// goes into the `fake` backend.
    Script(PathBuf),
}

#[derive(Debug, Clone)]
pub struct Options {
    pub input: Input,
    /// `<backend>:<model>`; unused for a script.
    pub asr: String,
    pub language: String,
    pub hints: Vec<String>,
    /// More hints, one per line.
    pub hints_file: Option<PathBuf>,
    /// Stop after this much audio; `None` for never.
    pub timeout: Option<Ts>,
    /// Replay a recording, script or stdin at real-time speed, as if spoken,
    /// rather than as fast as the transcriber goes.
    pub realtime: bool,
}

/// How a dictation ended, short of an error or a cancel.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    Text(String),
    NoSpeech,
}

/// Listens until `stop` is set, the timeout passes or the input ends, then
/// flushes the transcriber and reports the text. Every event goes to `emit`,
/// except the error an `Err` stands for.
pub fn run(opts: &Options, stop: &AtomicBool, emit: &mut dyn FnMut(Event)) -> Result<Outcome> {
    emit(Event::Loading);
    let mut hints = opts.hints.clone();
    if let Some(path) = &opts.hints_file {
        hints.extend(read_hints(path)?);
    }
    let (backend, model) = match &opts.input {
        Input::Script(path) => ("fake".to_string(), path.display().to_string()),
        _ => {
            let (b, m) = pam_speech::parse_backend(&opts.asr)?;
            (b.to_string(), m.to_string())
        }
    };
    let cfg = TranscriberConfig { model, language: opts.language.clone(), hints };
    let mut transcriber = pam_speech::open_transcriber(&backend, &cfg)?;
    let (mut source, device) = open(&opts.input)?;
    source.start()?;
    emit(Event::Recording { device });

    let pacer = opts.realtime.then(|| Pacer::start(1.0));
    let mut words = Transcript::default();
    let mut meter = Meter::default();
    let limit = opts.timeout.unwrap_or(Ts::MAX);
    let mut pos: Ts = 0;
    while pos < limit && !stop.load(Ordering::Relaxed) {
        let n = (limit - pos).min(CHUNK as Ts) as usize;
        let chunk = source.read(n, stop).context("reading the input")?;
        if chunk.is_empty() {
            break;
        }
        check(&mut source)?;
        // Before the transcriber's turn, so the meter doesn't wait on it.
        emit(Event::Level { level: hundredths(meter.level(&chunk)) });
        pos += chunk.len() as Ts;
        let update = transcriber.push(&chunk).context("transcriber")?;
        warn(transcriber.as_mut());
        for e in words.update(update) {
            emit(e);
        }
        if let Some(pacer) = &pacer {
            pacer.wait_until(pos);
        }
    }
    drop(source);

    emit(Event::Transcribing);
    let rest = transcriber.finish().context("transcriber")?;
    warn(transcriber.as_mut());
    for e in words.flush(rest) {
        emit(e);
    }
    let text = words.text();
    if text.is_empty() {
        emit(Event::NoSpeech);
        Ok(Outcome::NoSpeech)
    } else {
        emit(Event::Text { text: text.clone() });
        Ok(Outcome::Text(text))
    }
}

/// `x` to two decimals, which a meter can't tell apart anyway, so a `level`
/// line stays short.
fn hundredths(x: f32) -> f32 {
    (x * 100.0).round() / 100.0
}

/// The transcriber's warnings, on stderr: they don't stop it.
fn warn(transcriber: &mut dyn Transcriber) {
    for w in transcriber.warnings() {
        eprintln!("dictate: {w}");
    }
}

/// One hint per line; blank lines and lines starting with `#` are skipped.
fn read_hints(path: &Path) -> Result<Vec<String>> {
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading the hints in {}", path.display()))?;
    Ok(text
        .lines()
        .map(str::trim)
        .filter(|l| !l.is_empty() && !l.starts_with('#'))
        .map(str::to_string)
        .collect())
}

/// Opens `input`, and names it: the mic, the file, or stdin.
fn open(input: &Input) -> Result<(Source, String)> {
    let file_name = |p: &Path| p.file_name().unwrap_or(p.as_os_str()).to_string_lossy().into_owned();
    Ok(match input {
        Input::Mic(name) => {
            let mic = Mic::open(name.as_deref())?;
            let name = mic.name.clone();
            (Source::Mic(Box::new(mic)), name)
        }
        Input::Wav(path) => (Source::Wav(WavSource::open(path)?), file_name(path)),
        Input::Stdin => (Source::Raw(RawSource::new(std::io::stdin())), "stdin".into()),
        Input::Script(path) => {
            let left = pam_speech::fake::Script::load(path)?.duration;
            (Source::Silence { left }, file_name(path))
        }
    })
}

/// Fails on a mic that has sent only zeros, which hears nothing; reports
/// its other problems on stderr, since they don't stop it.
fn check(source: &mut Source) -> Result<()> {
    let Source::Mic(mic) = source else { return Ok(()) };
    if let Some(why) = mic.silence() {
        bail!(why);
    }
    for w in mic.warnings() {
        eprintln!("dictate: {w}");
    }
    Ok(())
}

/// The finals so far and the preview last reported.
#[derive(Debug, Default)]
struct Transcript {
    finals: Vec<String>,
    preview: String,
}

impl Transcript {
    /// A `final` for each utterance that settled, then a `preview` if it changed.
    fn update(&mut self, u: Update<Word>) -> Vec<Event> {
        let mut out = self.flush(u.finalized);
        let preview = join_words(&u.preview);
        if preview != self.preview {
            self.preview = preview.clone();
            out.push(Event::Preview { text: preview });
        }
        out
    }

    /// A `final` for each utterance in `words`; a new one begins at each
    /// word that starts an utterance.
    fn flush(&mut self, words: Vec<Word>) -> Vec<Event> {
        let mut out = Vec::new();
        let mut rest = &words[..];
        while !rest.is_empty() {
            let len = rest.iter().skip(1).position(|w| w.starts_utterance).map_or(rest.len(), |i| i + 1);
            let (said, more) = rest.split_at(len);
            rest = more;
            let text = join_words(said);
            if text.trim().is_empty() {
                continue;
            }
            self.finals.push(text.clone());
            out.push(Event::Final { text, words: said.iter().map(TimedWord::from).collect() });
        }
        out
    }

    fn text(&self) -> String {
        self.finals.join(" ").trim().to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn word(text: &str, starts_utterance: bool) -> Word {
        Word { text: text.into(), starts_utterance, ..Word::default() }
    }

    /// A word said from `start` to `end`, in samples.
    fn said(text: &str, starts_utterance: bool, start: Ts, end: Ts) -> Word {
        Word { start, end, ..word(text, starts_utterance) }
    }

    fn final_(text: &str, words: &[(&str, f64, f64)]) -> Event {
        let words =
            words.iter().map(|&(text, start, end)| TimedWord { text: text.into(), start, end }).collect();
        Event::Final { text: text.into(), words }
    }

    #[test]
    fn events_are_one_json_object_per_line() {
        let line = |e: Event| serde_json::to_string(&e).unwrap();
        assert_eq!(line(Event::Loading), r#"{"event":"loading"}"#);
        assert_eq!(line(Event::NoSpeech), r#"{"event":"no-speech"}"#);
        assert_eq!(
            line(Event::Recording { device: "Yeti".into() }),
            r#"{"event":"recording","device":"Yeti"}"#
        );
        assert_eq!(line(Event::Error { message: "m".into() }), r#"{"event":"error","message":"m"}"#);
        assert_eq!(line(Event::Level { level: hundredths(0.434_567) }), r#"{"event":"level","level":0.43}"#);
        assert_eq!(
            line(Event::Final {
                text: "Hi.".into(),
                words: vec![TimedWord::from(&said("Hi.", true, 16_000, 21_000))]
            }),
            r#"{"event":"final","text":"Hi.","words":[{"text":"Hi.","start":1.0,"end":1.313}]}"#
        );
    }

    #[test]
    fn each_settled_utterance_is_a_final_and_the_preview_is_sent_when_it_changes() {
        let mut t = Transcript::default();
        let u = Update { finalized: vec![], preview: vec![word("so", true), word("let's", false)] };
        assert_eq!(t.update(u.clone()), [Event::Preview { text: "so let's".into() }]);
        assert_eq!(t.update(u), []);
        let u = Update {
            finalized: vec![
                said("So", true, 19_200, 21_760),
                said("there.", false, 21_760, 25_000),
                said("And", true, 40_000, 41_280),
            ],
            preview: vec![],
        };
        assert_eq!(
            t.update(u),
            [
                final_("So there.", &[("So", 1.2, 1.36), ("there.", 1.36, 1.563)]),
                final_("And", &[("And", 2.5, 2.58)]),
                Event::Preview { text: String::new() }
            ]
        );
        assert_eq!(
            t.flush(vec![said("more.", false, 48_000, 49_280)]),
            [final_("more.", &[("more.", 3.0, 3.08)])]
        );
        assert_eq!(t.text(), "So there. And more.");
    }

    #[test]
    fn hints_files_skip_blanks_and_comments() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("hints.txt");
        std::fs::write(&path, "# names\nobento\n\n  Magit \n").unwrap();
        assert_eq!(read_hints(&path).unwrap(), ["obento", "Magit"]);
        assert!(read_hints(&dir.path().join("none")).is_err());
    }
}
