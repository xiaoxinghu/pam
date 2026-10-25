//! The `qwen` backend: Qwen3-ASR as a clip model, with Qwen3-ForcedAligner
//! giving its finals word times, both run in Pam's process on llama.cpp's
//! one ggml.
//!
//! - **Recognizer:** each clip is transcribed whole and greedily by Qwen3-ASR
//!   through llama.cpp's `mtmd`. Qwen's prompt puts the text after the audio,
//!   so no transcript can be kept between steps: the
//!   shared [`clip::ClipTranscriber`] cuts, previews and finalizes. The text is
//!   split into words ([`pieces`]), each with the clip's span.
//! - **Aligner:** Qwen3-ForcedAligner times each word's units, its letters,
//!   digits and `'`, on finals only. The text is never changed.
//! - **Languages:** with `auto`, Qwen names each clip's language and its
//!   words take it, checked against their script ([`crate::lang::fill`]). A
//!   configured language is forced in the prompt; one Qwen doesn't know
//!   fails at open.
//! - **Hints** are the system prompt, which is where Qwen takes context.
//!
//! The runtimes sit behind [`Asr`] and [`Timer`] and need the `qwen`
//! feature; everything else here is tested without a model.

use std::path::PathBuf;

use anyhow::bail;

use crate::clip::{self, ClipRecognizer};
use crate::store::{self, Store};
use crate::{Result, SAMPLE_RATE, Ts, Word};

#[cfg(feature = "qwen")]
mod native;
#[cfg(test)]
mod tests;

/// A clip's output is cut off after this many tokens, plus [`TOKENS_PER_SECOND`]
/// per second of audio: room for fast Mandarin, so a repetition loop
/// (laughter came out as "哈，哈，…") stops at a cost bounded by the clip.
const BASE_TOKENS: usize = 32;
const TOKENS_PER_SECOND: usize = 12;

/// Qwen3-ASR's languages: its name for each, and the BCP-47 tag Pam gives
/// its words. Chinese, Japanese and Korean take the tags the script helper
/// gives (`crate::lang`), so a clip's words and their script agree.
const LANGUAGES: [(&str, &str); 30] = [
    ("Chinese", "zh-CN"),
    ("English", "en"),
    ("Cantonese", "yue"),
    ("Arabic", "ar"),
    ("German", "de"),
    ("French", "fr"),
    ("Spanish", "es"),
    ("Portuguese", "pt"),
    ("Indonesian", "id"),
    ("Italian", "it"),
    ("Korean", "ko-KR"),
    ("Russian", "ru"),
    ("Thai", "th"),
    ("Vietnamese", "vi"),
    ("Japanese", "ja-JP"),
    ("Turkish", "tr"),
    ("Hindi", "hi"),
    ("Malay", "ms"),
    ("Dutch", "nl"),
    ("Swedish", "sv"),
    ("Danish", "da"),
    ("Finnish", "fi"),
    ("Polish", "pl"),
    ("Czech", "cs"),
    ("Filipino", "fil"),
    ("Persian", "fa"),
    ("Greek", "el"),
    ("Hungarian", "hu"),
    ("Macedonian", "mk"),
    ("Romanian", "ro"),
];

/// Qwen's name for a configured language (`en-US` → `English`), by its
/// primary subtag; `None` for detection. One Qwen doesn't know is refused.
pub fn qwen_language(language: Option<&str>) -> Result<Option<&'static str>> {
    let Some(language) = language else { return Ok(None) };
    let primary = language.split('-').next().unwrap_or(language).to_ascii_lowercase();
    match LANGUAGES.iter().find(|(_, tag)| tag.split('-').next() == Some(primary.as_str())) {
        Some((name, _)) => Ok(Some(name)),
        None => bail!("language `{language}` isn't one Qwen3-ASR transcribes: use auto, or one of {}", {
            let tags: Vec<&str> = LANGUAGES.iter().map(|(_, t)| *t).collect();
            tags.join(", ")
        }),
    }
}

/// The language and text of Qwen's output, `language English<asr_text>…`.
/// A language it names that isn't one of its own, and `None` (no speech),
/// give no tag.
pub fn parse(raw: &str) -> (Option<&'static str>, &str) {
    let Some((head, text)) = raw.split_once("<asr_text>") else { return (None, raw.trim()) };
    let name = head.trim().trim_start_matches("language").trim();
    (LANGUAGES.iter().find(|(n, _)| *n == name).map(|(_, tag)| *tag), text.trim())
}

/// A word as the recognizer gives it, and the unit the aligner times.
#[derive(Debug, Clone, PartialEq)]
pub struct Piece {
    /// The recognized text, unchanged: an English word with its punctuation,
    /// or one Han or kana character with the punctuation after it.
    pub text: String,
    /// What the aligner is given: `text` with only letters, digits and `'`
    /// kept, as Qwen's own processor cleans words. Never empty.
    pub unit: String,
}

/// Splits recognized text at whitespace, and each Han or kana character from
/// what is next to it. Punctuation that leaves nothing to
/// align joins the piece before it (the one after it at the start). A text
/// with nothing to align gives no pieces.
pub fn pieces(text: &str) -> Vec<Piece> {
    let mut out: Vec<Piece> = Vec::new();
    let mut lead = String::new(); // punctuation before the first piece
    // `spaced`: the raw text starts a whitespace-separated token, so text
    // joined to the piece before it keeps a space.
    let mut push = |raw: &str, spaced: bool, out: &mut Vec<Piece>| {
        let unit = clean(raw);
        if !unit.is_empty() {
            let lead = std::mem::take(&mut lead);
            let gap = if !lead.is_empty() && spaced { " " } else { "" };
            out.push(Piece { text: format!("{lead}{gap}{raw}"), unit });
        } else if let Some(last) = out.last_mut() {
            if spaced {
                last.text.push(' ');
            }
            last.text.push_str(raw);
        } else {
            if spaced && !lead.is_empty() {
                lead.push(' ');
            }
            lead.push_str(raw);
        }
    };
    for token in text.split_whitespace() {
        let mut run = String::new();
        let mut first = true;
        // Whether the run follows a Han or kana character in this token: then
        // the punctuation it starts with ends that character ("试。Great").
        let mut after_split = false;
        let mut flush = |run: &str, first: bool, after_split: bool, out: &mut Vec<Piece>| {
            let (lead, rest) = if after_split { split_lead(run) } else { ("", run) };
            if !lead.is_empty() {
                push(lead, false, out);
            }
            if !rest.is_empty() {
                push(rest, first && lead.is_empty(), out);
            }
        };
        for c in token.chars() {
            if crate::is_han(c) || crate::is_kana(c) {
                if !run.is_empty() {
                    flush(&run, std::mem::take(&mut first), after_split, &mut out);
                    run.clear();
                }
                flush(&c.to_string(), std::mem::take(&mut first), false, &mut out);
                after_split = true;
            } else {
                run.push(c);
            }
        }
        if !run.is_empty() {
            flush(&run, first, after_split, &mut out);
        }
    }
    out
}

/// A run's leading punctuation, and the rest from its first letter, digit or
/// opening bracket or quote.
fn split_lead(run: &str) -> (&str, &str) {
    let at = run
        .find(|c: char| c == '\'' || c.is_alphanumeric() || "([{<（【《「『“‘".contains(c))
        .unwrap_or(run.len());
    run.split_at(at)
}

/// Qwen's `clean_token`: keeps `'` and characters whose Unicode category is a
/// letter or a number.
fn clean(s: &str) -> String {
    s.chars().filter(|&c| c == '\'' || c.is_alphanumeric()).collect()
}

/// What the recognizer needs of Qwen3-ASR's runtime.
pub trait Asr: Send {
    /// Qwen's output for one clip, greedy, cut off after `max_tokens`:
    /// `language <Name><asr_text><text>`, or, with a `language` (Qwen's name
    /// for it) the prompt forces, the text alone. `context` is the system
    /// prompt.
    fn generate(
        &mut self,
        audio: &[f32],
        context: &str,
        language: Option<&str>,
        max_tokens: usize,
    ) -> Result<String>;
}

/// What the aligner needs of Qwen3-ForcedAligner's runtime.
pub trait Timer: Send {
    /// Each unit's start and end, in seconds from the clip's start, in order.
    fn time(&mut self, audio: &[f32], units: &[String]) -> Result<Vec<(f32, f32)>>;
}

/// Qwen3-ASR as a [`ClipRecognizer`]: each word gets the clip's span.
pub struct Recognizer<A> {
    asr: A,
    /// Qwen's name for the configured language; `None` to detect it.
    language: Option<&'static str>,
    /// The hints, Qwen's context.
    context: String,
}

impl<A: Asr> Recognizer<A> {
    pub fn new(asr: A, language: Option<&'static str>, hints: &[String]) -> Recognizer<A> {
        Recognizer { asr, language, context: hints.join(", ") }
    }
}

impl<A: Asr> ClipRecognizer for Recognizer<A> {
    fn transcribe(&mut self, audio: &[f32]) -> Result<Vec<Word>> {
        let max_tokens = BASE_TOKENS + TOKENS_PER_SECOND * audio.len().div_ceil(SAMPLE_RATE as usize);
        let raw = self.asr.generate(audio, &self.context, self.language, max_tokens)?;
        // A forced language's words get it from the wrapper.
        let (tag, text) = if self.language.is_some() { (None, raw.trim()) } else { parse(&raw) };
        let len = audio.len() as Ts;
        let mut words: Vec<Word> = pieces(text)
            .into_iter()
            .map(|p| Word { text: p.text, start: 0, end: len, ..Word::default() })
            .collect();
        if self.language.is_none() {
            crate::lang::fill(&mut words, tag);
        }
        Ok(words)
    }
}

/// Qwen3-ForcedAligner as a [`clip::Aligner`]: one aligner for any clip
/// model's words, timed by their units.
pub struct Aligner<T> {
    timer: T,
}

impl<T: Timer> Aligner<T> {
    pub fn new(timer: T) -> Aligner<T> {
        Aligner { timer }
    }
}

impl<T: Timer> clip::Aligner for Aligner<T> {
    /// The aligner hears the clip's speech only, from its first frame the
    /// endpointer counts as speech to the end of its last ([`speech`]). A
    /// word of several units runs from its first unit's start to its last's
    /// end. One with none to time (punctuation alone) takes no time where the
    /// word before it ends, or where the speech starts.
    fn align(&mut self, audio: &[f32], mut words: Vec<Word>) -> Result<Vec<Word>> {
        let units: Vec<(usize, String)> = words
            .iter()
            .enumerate()
            .flat_map(|(i, w)| pieces(&w.text).into_iter().map(move |p| (i, p.unit)))
            .collect();
        if units.is_empty() {
            return Ok(words);
        }
        let texts: Vec<String> = units.iter().map(|(_, u)| u.clone()).collect();
        let (from, to) = speech(audio);
        let times = self.timer.time(&audio[from as usize..to as usize], &texts)?;
        if times.len() != units.len() {
            bail!("the aligner timed {} units of the {} it was given", times.len(), units.len());
        }
        let ts =
            |secs: f32| (from + (f64::from(secs) * f64::from(SAMPLE_RATE)).round().max(0.0) as Ts).min(to);
        let mut spans: Vec<Option<(Ts, Ts)>> = vec![None; words.len()];
        for ((i, _), (start, end)) in units.iter().zip(times) {
            let span = spans[*i].get_or_insert((ts(start), ts(end)));
            span.1 = ts(end).max(span.0);
        }
        let mut end = from;
        for (w, span) in words.iter_mut().zip(&spans) {
            (w.start, w.end) = span.unwrap_or((end, end));
            end = w.end;
        }
        Ok(words)
    }
}

/// Where a clip's speech starts and ends: from its first 20 ms frame at or
/// over the endpointer's fixed −60 dBFS to the end of its last; the whole clip
/// if none is. Given the pre-roll's silence too, the aligner put a clip's
/// first word there, where there was no sound.
fn speech(audio: &[f32]) -> (Ts, Ts) {
    use crate::clip::endpoint::{FRAME, THRESHOLD_DB};
    let loud = |f: &[f32]| {
        10.0 * (f.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>() / f.len() as f64).log10() >= THRESHOLD_DB
    };
    let frames: Vec<bool> = audio.chunks(FRAME as usize).map(loud).collect();
    match (frames.iter().position(|l| *l), frames.iter().rposition(|l| *l)) {
        (Some(first), Some(last)) => (first as Ts * FRAME, ((last as Ts + 1) * FRAME).min(audio.len() as Ts)),
        _ => (0, audio.len() as Ts),
    }
}

/// The files of the model named `model`, from the model store: Qwen3-ASR's
/// decoder and audio projector, and the aligner. Only `qwen3-asr-1.7b` is
/// known. One that isn't there is a [`store::Missing`] error, which says how
/// to make it.
pub fn model_files(model: &str) -> Result<[PathBuf; 3]> {
    model_files_in(model, &Store::open()?)
}

fn model_files_in(model: &str, store: &Store) -> Result<[PathBuf; 3]> {
    if !store::QWEN3_ASR_1_7B.names.contains(&model) {
        bail!("unknown model `{model}`: the qwen backend knows qwen3-asr-1.7b");
    }
    let files = store.locate(&store::QWEN3_ASR_1_7B)?;
    Ok(files.try_into().expect("three files"))
}

/// Opens the `qwen` backend: Qwen3-ASR and its aligner in the shared clip
/// wrapper.
#[cfg(feature = "qwen")]
pub fn open(cfg: &crate::TranscriberConfig) -> Result<clip::ClipTranscriber> {
    let language = qwen_language(cfg.language())?;
    let [model, mmproj, aligner] = model_files(&cfg.model)?;
    let asr = native::Llama::load(&model, &mmproj)?;
    let timer = native::ForcedAligner::load(&aligner)?;
    let model = clip::ClipModel::aligned(Recognizer::new(asr, language, &cfg.hints), Aligner::new(timer));
    Ok(clip::ClipTranscriber::new(model, cfg.language(), clip::PREVIEW_EVERY))
}
