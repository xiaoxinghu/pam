//! The `fake` backend: replays a script of timed words and speaker spans.
//!
//! It ignores what the audio contains and only counts samples. A word becomes
//! final once the pushed audio passes its end plus `word_delay`; a span once the
//! audio passes its end plus `settle_delay`. Items that have started but are not
//! final yet are the preview.
//!
//! # Script format
//!
//! JSON Lines, times in seconds. Blank lines and lines starting with `#` are
//! skipped. Each line is one of:
//!
//! ```text
//! {"config": {"word_delay": 0.9, "settle_delay": 2.88, "duration": 42}}
//! {"span": 0, "start": 1.20, "end": 4.80}
//! {"word": "Hello", "at": 1.36}
//! {"word": "everyone", "start": 1.52, "end": 1.90}
//! {"word": "大家好", "at": 2.40, "lang": "zh-CN", "starts_utterance": true}
//! ```
//!
//! - `config` is optional; every field defaults to 0, and `duration` to the
//!   last end plus 1 s. `"fail_after": 5` makes the transcriber fail once the
//!   audio passes 5 s, to test error handling.
//! - `"at"` is a NeMo-style emission stamp: the word lasts one 80 ms frame.
//! - `"lang"` is optional: the language the transcriber reports for the word.
//! - `"starts_utterance"` is optional: the word begins an utterance.
//! - `"utterance_end": 0.8` in `config` makes the transcriber finalize like
//!   NeMo-Speech: a silence that long ends an utterance, and its words
//!   become final together once the audio passes its last word's end plus
//!   the silence and `word_delay`. The first word of each is marked
//!   `starts_utterance`. Without it, each word is final on its own.
//! - Words are replayed in start order, spans in their own order of ending.

use std::path::Path;

use anyhow::{Context, bail};
use serde::Deserialize;

use crate::{Result, SAMPLE_RATE, SpeakerSpan, SpeakerTracker, SpeakerUpdate, Transcriber, Ts, Update, Word};

/// Length of one NeMo frame, used for `"at"` words.
pub const FRAME: Ts = 1_280;

#[derive(Debug, Clone, PartialEq)]
pub struct Script {
    pub words: Vec<Word>,
    pub spans: Vec<SpeakerSpan>,
    /// How long after its end a word becomes final.
    pub word_delay: Ts,
    /// How long after its end a span becomes final (the tracker's latency).
    pub settle_delay: Ts,
    /// How much audio the script stands for.
    pub duration: Ts,
    /// The transcriber fails once the pushed audio passes this point.
    pub fail_after: Option<Ts>,
    /// The silence that ends an utterance, when words are final an
    /// utterance at a time.
    pub utterance_end: Option<Ts>,
}

#[derive(Deserialize)]
#[serde(untagged, deny_unknown_fields)]
enum Item {
    Config {
        config: Config,
    },
    Span {
        span: u32,
        start: f64,
        end: f64,
    },
    Stamp {
        word: String,
        at: f64,
        lang: Option<String>,
        #[serde(default)]
        starts_utterance: bool,
    },
    Word {
        word: String,
        start: f64,
        end: f64,
        lang: Option<String>,
        #[serde(default)]
        starts_utterance: bool,
    },
}

#[derive(Deserialize, Default)]
#[serde(deny_unknown_fields)]
struct Config {
    #[serde(default)]
    word_delay: f64,
    #[serde(default)]
    settle_delay: f64,
    duration: Option<f64>,
    fail_after: Option<f64>,
    utterance_end: Option<f64>,
}

/// Seconds to samples, rounded to the nearest sample.
pub fn secs(s: f64) -> Ts {
    (s * SAMPLE_RATE as f64).round().max(0.0) as Ts
}

impl Script {
    pub fn load(path: impl AsRef<Path>) -> Result<Script> {
        let path = path.as_ref();
        let text = std::fs::read_to_string(path)
            .with_context(|| format!("reading fake script {}", path.display()))?;
        Script::parse(&text).with_context(|| format!("in fake script {}", path.display()))
    }

    pub fn parse(text: &str) -> Result<Script> {
        let mut config = Config::default();
        let mut words = Vec::new();
        let mut spans = Vec::new();
        for (i, line) in text.lines().enumerate() {
            let line = line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }
            let item: Item = serde_json::from_str(line)
                .with_context(|| format!("line {}: expected a config, span or word object", i + 1))?;
            match item {
                Item::Config { config: c } => config = c,
                Item::Span { span, start, end } => {
                    spans.push(SpeakerSpan { start: secs(start), end: secs(end), slot: span })
                }
                Item::Stamp { word, at, lang, starts_utterance } => words.push(Word {
                    text: word,
                    start: secs(at),
                    end: secs(at) + FRAME,
                    lang,
                    starts_utterance,
                }),
                Item::Word { word, start, end, lang, starts_utterance } => words.push(Word {
                    text: word,
                    start: secs(start),
                    end: secs(end),
                    lang,
                    starts_utterance,
                }),
            }
            if let Some(w) = words.last().filter(|w| w.end < w.start) {
                bail!("line {}: word `{}` ends before it starts", i + 1, w.text);
            }
            if let Some(s) = spans.last().filter(|s| s.end < s.start) {
                bail!("line {}: span for slot {} ends before it starts", i + 1, s.slot);
            }
        }
        words.sort_by_key(|w| w.start);
        let utterance_end = config.utterance_end.map(secs);
        if let Some(gap) = utterance_end {
            for i in 0..words.len() {
                if i == 0 || words[i].start > words[i - 1].end + gap {
                    words[i].starts_utterance = true;
                }
            }
        }
        let last = words.iter().map(|w| w.end).chain(spans.iter().map(|s| s.end)).max();
        let duration = match config.duration {
            Some(d) => secs(d),
            None => last.unwrap_or(0) + SAMPLE_RATE as Ts,
        };
        Ok(Script {
            words,
            spans,
            word_delay: secs(config.word_delay),
            settle_delay: secs(config.settle_delay),
            duration,
            fail_after: config.fail_after.map(secs),
            utterance_end,
        })
    }
}

pub struct FakeTranscriber {
    words: Vec<Word>,
    /// When each word becomes final.
    final_at: Vec<Ts>,
    pushed: Ts,
    /// Words before this index have been returned as final.
    next: usize,
    fail_after: Option<Ts>,
}

impl FakeTranscriber {
    pub fn new(script: Script) -> Self {
        let delay = script.word_delay;
        let mut final_at: Vec<Ts> = script.words.iter().map(|w| w.end + delay).collect();
        if let Some(gap) = script.utterance_end {
            // Each word waits for its utterance's last word, and the silence after it.
            let mut end = Ts::MAX;
            for (i, w) in script.words.iter().enumerate().rev() {
                end = if end == Ts::MAX { w.end + gap + delay } else { end };
                final_at[i] = end;
                if w.starts_utterance {
                    end = Ts::MAX;
                }
            }
        }
        FakeTranscriber { words: script.words, final_at, pushed: 0, next: 0, fail_after: script.fail_after }
    }
}

impl Transcriber for FakeTranscriber {
    fn push(&mut self, audio: &[f32]) -> Result<Update<Word>> {
        self.pushed += audio.len() as Ts;
        if let Some(at) = self.fail_after.filter(|at| self.pushed > *at) {
            bail!("the script asked to fail after {at} samples");
        }
        let first = self.next;
        while self.final_at.get(self.next).is_some_and(|&at| at <= self.pushed) {
            self.next += 1;
        }
        let finalized = self.words[first..self.next].to_vec();
        let preview = self.words[self.next..].iter().take_while(|w| w.start < self.pushed).cloned().collect();
        Ok(Update { finalized, preview })
    }

    /// Flushes the words that started before the end of the pushed audio.
    fn finish(&mut self) -> Result<Vec<Word>> {
        let heard = self.words[self.next..].iter().take_while(|w| w.start < self.pushed).count();
        let rest = self.words[self.next..self.next + heard].to_vec();
        self.next = self.words.len();
        Ok(rest)
    }
}

pub struct FakeTracker {
    spans: Vec<SpeakerSpan>,
    done: Vec<bool>,
    delay: Ts,
    pushed: Ts,
}

impl FakeTracker {
    pub fn new(script: Script) -> Self {
        let done = vec![false; script.spans.len()];
        FakeTracker { spans: script.spans, done, delay: script.settle_delay, pushed: 0 }
    }
}

impl SpeakerTracker for FakeTracker {
    fn push(&mut self, audio: &[f32]) -> Result<SpeakerUpdate> {
        self.pushed += audio.len() as Ts;
        let decided = self.pushed.saturating_sub(self.delay);
        let mut finalized = Vec::new();
        for (s, done) in self.spans.iter().zip(self.done.iter_mut()) {
            if !*done && s.end <= decided {
                *done = true;
                finalized.push(*s);
            }
        }
        finalized.sort_by_key(|s| (s.end, s.start));
        let open = self.spans.iter().zip(&self.done).filter(|(s, done)| !**done && s.start < self.pushed);
        // A span that started before `decided` but hasn't ended will become final later,
        // so nothing is settled past its start.
        let settled_until = open.clone().map(|(s, _)| s.start).fold(decided, Ts::min);
        let preview = open.map(|(s, _)| SpeakerSpan { end: s.end.min(self.pushed), ..*s }).collect();
        Ok(SpeakerUpdate { spans: Update { finalized, preview }, settled_until })
    }

    /// Flushes the spans that started before the end of the pushed audio, cut off there.
    fn finish(&mut self) -> Result<Vec<SpeakerSpan>> {
        let pushed = self.pushed;
        let mut rest: Vec<_> = self
            .spans
            .iter()
            .zip(&mut self.done)
            .filter(|(s, done)| !**done && s.start < pushed)
            .map(|(s, done)| {
                *done = true;
                SpeakerSpan { end: s.end.min(pushed), ..*s }
            })
            .collect();
        rest.sort_by_key(|s| (s.end, s.start));
        Ok(rest)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const SCRIPT: &str = r#"
# two words and a span that settles 1 s late
{"config": {"settle_delay": 1.0}}
{"span": 0, "start": 0.5, "end": 2.0}
{"word": "hello", "at": 0.64}
{"word": "there", "start": 1.0, "end": 1.5}
"#;

    fn chunk(secs: f64) -> Vec<f32> {
        vec![0.0; super::secs(secs) as usize]
    }

    #[test]
    fn parses_stamps_spans_and_config() {
        let s = Script::parse(SCRIPT).unwrap();
        assert_eq!(
            s.words[0],
            Word { text: "hello".into(), start: 10_240, end: 11_520, lang: None, starts_utterance: false }
        );
        assert_eq!(s.words[1].end, 24_000);
        assert_eq!(s.spans, vec![SpeakerSpan { start: 8_000, end: 32_000, slot: 0 }]);
        assert_eq!(s.settle_delay, 16_000);
        assert_eq!(s.duration, 48_000);
    }

    #[test]
    fn rejects_unknown_items() {
        assert!(Script::parse(r#"{"wrod": "x", "at": 1}"#).is_err());
        assert!(Script::parse(r#"{"word": "x", "start": 2, "end": 1}"#).is_err());
    }

    #[test]
    fn words_become_final_when_audio_passes_their_end() {
        let mut t = FakeTranscriber::new(Script::parse(SCRIPT).unwrap());
        let u = t.push(&chunk(0.6)).unwrap();
        assert!(u.finalized.is_empty() && u.preview.is_empty());
        let u = t.push(&chunk(0.5)).unwrap(); // at 1.1 s
        assert_eq!(u.finalized.len(), 1);
        assert_eq!(u.preview.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(), ["there"]);
        let u = t.push(&chunk(0.5)).unwrap();
        assert_eq!(u.finalized[0].text, "there");
        assert!(t.finish().unwrap().is_empty());
    }

    #[test]
    fn with_an_utterance_end_words_are_final_an_utterance_at_a_time() {
        let script = r#"{"config": {"utterance_end": 0.8, "word_delay": 0.1}}
            {"word": "Pam", "at": 0.5}
            {"word": "can", "at": 0.8}
            {"word": "you", "at": 1.1}
            {"word": "find", "at": 2.5}"#;
        let s = Script::parse(script).unwrap();
        let starts: Vec<bool> = s.words.iter().map(|w| w.starts_utterance).collect();
        assert_eq!(starts, [true, false, false, true]);
        let mut t = FakeTranscriber::new(s);
        // "you" ends at 1.18 s: its utterance is final at 1.18 + 0.8 + 0.1 s.
        let u = t.push(&chunk(2.0)).unwrap();
        assert!(u.finalized.is_empty(), "{u:?}");
        let u = t.push(&chunk(0.6)).unwrap();
        assert_eq!(u.finalized.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(), ["Pam", "can", "you"]);
        assert_eq!(u.preview.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(), ["find"]);
    }

    #[test]
    fn finish_flushes_only_what_was_heard() {
        let script = Script::parse(SCRIPT).unwrap();
        let mut t = FakeTranscriber::new(script.clone());
        let mut k = FakeTracker::new(script);
        t.push(&chunk(0.7)).unwrap();
        k.push(&chunk(0.7)).unwrap();
        assert_eq!(t.finish().unwrap().iter().map(|w| w.text.as_str()).collect::<Vec<_>>(), ["hello"]);
        assert_eq!(k.finish().unwrap(), vec![SpeakerSpan { start: 8_000, end: 11_200, slot: 0 }]);
    }

    #[test]
    fn spans_settle_late_and_hold_back_settled_until() {
        let mut k = FakeTracker::new(Script::parse(SCRIPT).unwrap());
        let u = k.push(&chunk(1.6)).unwrap();
        // 0.6 s is decided, but the open span started at 0.5 s.
        assert_eq!(u.settled_until, 8_000);
        assert_eq!(u.spans.preview, vec![SpeakerSpan { start: 8_000, end: 25_600, slot: 0 }]);
        let u = k.push(&chunk(1.0)).unwrap(); // at 2.6 s: span ended, not final yet
        assert!(u.spans.finalized.is_empty());
        assert_eq!(u.settled_until, 8_000);
        let u = k.push(&chunk(0.5)).unwrap(); // at 3.1 s: decided 2.1 s
        assert_eq!(u.spans.finalized.len(), 1);
        assert_eq!(u.settled_until, 33_600);
        assert!(k.finish().unwrap().is_empty());
    }
}
