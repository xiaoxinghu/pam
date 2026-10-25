//! Speech primitives: `Transcriber` (speech → words) and `SpeakerTracker`
//! (speech → who spoke when), and their backends.
//!
//! The primitives know nothing about names, meetings or each other; the
//! core owns all of that.

pub mod clip;
pub mod fake;
pub(crate) mod lang;
pub mod nemo;
pub mod qwen;
pub mod store;

pub use anyhow::Result;

/// Time = sample index at 16 kHz since the stream started. Exact integer; no float drift.
pub type Ts = u64;

/// Samples per second of every stream.
pub const SAMPLE_RATE: u32 = 16_000;

/// Every push returns what just became final, plus a best-guess preview.
#[derive(Debug, Clone, PartialEq)]
pub struct Update<T> {
    /// Never changes again.
    pub finalized: Vec<T>,
    /// Replaces the previous preview entirely; display only, never stored.
    pub preview: Vec<T>,
}

impl<T> Default for Update<T> {
    fn default() -> Self {
        Update { finalized: Vec::new(), preview: Vec::new() }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Default)]
pub struct Word {
    pub text: String,
    pub start: Ts,
    pub end: Ts,
    /// The language of the utterance the word was heard in, as a BCP-47 code
    /// such as `"zh-CN"`; `None` when the backend can't tell.
    pub lang: Option<String>,
    /// Set on the first word of each utterance: the run of words the backend
    /// heard between two pauses, which it reports a language for.
    pub starts_utterance: bool,
}

/// Whether `lang` is a Chinese (Cantonese, `yue`, included: it is written in
/// Han too), Japanese or Korean language code.
pub fn is_cjk_language(lang: &str) -> bool {
    ["zh", "yue", "ja", "ko"].contains(&lang.split('-').next().unwrap_or(lang))
}

/// Han, kana and Hangul: scripts written without spaces between words, each
/// character two columns wide in a terminal.
pub fn is_cjk(c: char) -> bool {
    is_kana(c) || is_hangul(c) || is_han(c)
}

/// Hiragana and katakana.
pub(crate) fn is_kana(c: char) -> bool {
    matches!(c, '\u{3040}'..='\u{30ff}')
}

/// Hangul syllables and jamo.
pub(crate) fn is_hangul(c: char) -> bool {
    matches!(c, '\u{1100}'..='\u{11ff}' | '\u{3130}'..='\u{318f}' | '\u{ac00}'..='\u{d7af}')
}

/// Han characters: the unified ideographs and their extensions, including
/// those outside the Basic Multilingual Plane (𠮷), the compatibility
/// ideographs, and 々, 〆 and 〇.
pub(crate) fn is_han(c: char) -> bool {
    matches!(c,
        '\u{3005}'..='\u{3007}' | '\u{3400}'..='\u{4dbf}' | '\u{4e00}'..='\u{9fff}'
        | '\u{f900}'..='\u{faff}' | '\u{20000}'..='\u{2fa1f}' | '\u{30000}'..='\u{323af}')
}

/// The words as text: separated by spaces, except between two CJK
/// characters, which are written without them.
pub fn join_words<'a>(words: impl IntoIterator<Item = &'a Word>) -> String {
    let mut text = String::new();
    for w in words {
        let joins = text.chars().next_back().is_some_and(is_cjk) && w.text.chars().next().is_some_and(is_cjk);
        if !text.is_empty() && !joins {
            text.push(' ');
        }
        text.push_str(&w.text);
    }
    text
}

pub trait Transcriber: Send {
    fn push(&mut self, audio: &[f32]) -> Result<Update<Word>>;
    /// Flush at end of stream.
    fn finish(&mut self) -> Result<Vec<Word>>;
    /// What went wrong without failing since the last call, for the caller
    /// to log: a transcriber has no log of its own.
    fn warnings(&mut self) -> Vec<String> {
        Vec::new()
    }
}

#[derive(Debug, Clone, Default)]
pub struct TranscriberConfig {
    /// e.g. "nemotron-en"; for `fake`, the script path.
    pub model: String,
    /// A language code such as `"en-US"` or `"zh-CN"`, or `"auto"` (or empty)
    /// for the model to detect each utterance's language. Adapters read
    /// it through [`TranscriberConfig::language`].
    pub language: String,
    /// Names / jargon to favour; set at stream start only, which keeps the
    /// primitive minimal; best-effort.
    pub hints: Vec<String>,
}

impl TranscriberConfig {
    /// The language every utterance is in, or `None` to detect each
    /// utterance's (`"auto"` or empty).
    pub fn language(&self) -> Option<&str> {
        Some(self.language.as_str()).filter(|l| !l.is_empty() && *l != "auto")
    }
}

/// `slot` = order of first appearance (0, 1, 2…), stable only within this stream.
/// Spans may overlap in time: that is cross-talk.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SpeakerSpan {
    pub start: Ts,
    pub end: Ts,
    pub slot: u32,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct SpeakerUpdate {
    pub spans: Update<SpeakerSpan>,
    /// No new final span will start before this point.
    pub settled_until: Ts,
}

pub trait SpeakerTracker: Send {
    fn push(&mut self, audio: &[f32]) -> Result<SpeakerUpdate>;
    fn finish(&mut self) -> Result<Vec<SpeakerSpan>>;
}

#[derive(Debug, Clone, Default)]
pub struct TrackerConfig {
    /// e.g. "nemotron-3-diarization"; for `fake`, the script path.
    pub model: String,
}

/// Splits `<backend>:<model>`, as the config file names a speech backend
/// (`asr = "nemo-speech:nemotron-en"`, `diarizer = "fake:script.jsonl"`).
pub fn parse_backend(spec: &str) -> Result<(&str, &str)> {
    match spec.split_once(':') {
        Some((backend, model)) if !backend.is_empty() && !model.is_empty() => Ok((backend, model)),
        _ => anyhow::bail!("backend `{spec}` should look like <backend>:<model>, e.g. fake:script.jsonl"),
    }
}

/// Whether this build has `backend`: `fake` always, `nemo-speech` and `qwen`
/// with their features.
pub fn built(backend: &str) -> bool {
    backend == "fake"
        || (backend == "nemo-speech" && cfg!(feature = "nemo-speech"))
        || (backend == "qwen" && cfg!(feature = "qwen"))
}

/// Opens a transcriber by backend name: `nemo-speech`, `qwen` or `fake`. A
/// clip adapter's arm wraps its model in the shared [`clip::ClipTranscriber`].
pub fn open_transcriber(backend: &str, cfg: &TranscriberConfig) -> Result<Box<dyn Transcriber>> {
    match backend {
        "fake" => Ok(Box::new(fake::FakeTranscriber::new(fake::Script::load(&cfg.model)?))),
        #[cfg(feature = "nemo-speech")]
        "nemo-speech" => Ok(Box::new(nemo::NemoTranscriber::open(cfg)?)),
        #[cfg(not(feature = "nemo-speech"))]
        "nemo-speech" => anyhow::bail!(NOT_BUILT),
        #[cfg(feature = "qwen")]
        "qwen" => Ok(Box::new(qwen::open(cfg)?)),
        #[cfg(not(feature = "qwen"))]
        "qwen" => anyhow::bail!("this build has no qwen backend: rebuild with `--features qwen`"),
        other => anyhow::bail!("unknown transcriber backend `{other}` (available: nemo-speech, qwen, fake)"),
    }
}

/// Opens a speaker tracker by backend name: `nemo-speech` or `fake`.
pub fn open_tracker(backend: &str, cfg: &TrackerConfig) -> Result<Box<dyn SpeakerTracker>> {
    match backend {
        "fake" => Ok(Box::new(fake::FakeTracker::new(fake::Script::load(&cfg.model)?))),
        #[cfg(feature = "nemo-speech")]
        "nemo-speech" => Ok(Box::new(nemo::NemoTracker::open(cfg)?)),
        #[cfg(not(feature = "nemo-speech"))]
        "nemo-speech" => anyhow::bail!(NOT_BUILT),
        other => anyhow::bail!("unknown speaker tracker backend `{other}` (available: nemo-speech, fake)"),
    }
}

#[cfg(not(feature = "nemo-speech"))]
const NOT_BUILT: &str = "this build has no nemo-speech backend: rebuild with `--features nemo-speech`, \
     which builds NeMo-Speech.cpp v0.2.0 from source unless NEMO_SPEECH_SRC points to a checkout";

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cjk_words_join_without_spaces() {
        let w = |text: &str| Word { text: text.into(), ..Word::default() };
        assert_eq!(join_words(&[w("大家"), w("好"), w("Pam"), w("没问题")]), "大家好 Pam 没问题");
        assert_eq!(join_words(&[]), "");
    }
}
