//! The `nemo-speech` backend: NeMo-Speech.cpp v0.2.0 through its C ABI.
//!
//! - **Transcriber:** `asr.h` streaming recognition with endpointing. Interim
//!   results carry text but no word times; finals (one per utterance) carry
//!   words with times. Final words are final; the latest interim's words are
//!   the preview, stamped with when each first appeared.
//! - **Speaker tracker:** `diar.h` streaming diarization, `v3-streaming` with
//!   chunk 32 + right context 4 (2.88 s, which confuses speakers less than
//!   the preset's 1.12 s). Its segments are recomputed
//!   from all labelled frames on every advance, and the part near the labelled
//!   edge can still change. Everything older than [`SETTLE_MARGIN`] behind the
//!   edge is final: segments are cut there, the decided part is a final span
//!   and the rest preview.
//!
//! The parts that don't touch the library ([`Settler`], [`Utterance`], model
//! lookup, languages) are always compiled, so they are tested without it. The backend
//! itself needs the `nemo-speech` feature.

use std::collections::{BTreeMap, VecDeque};
use std::path::{Path, PathBuf};

use anyhow::{Context, bail};

use crate::store::{self, Model, Store};
use crate::{Result, SAMPLE_RATE, SpeakerSpan, SpeakerUpdate, Ts, Update, Word};

#[cfg(feature = "nemo-speech")]
mod backend;
mod gguf;
#[cfg(feature = "nemo-speech")]
mod sys;

#[cfg(feature = "nemo-speech")]
pub use backend::{NemoTracker, NemoTranscriber};

/// How far behind the diarizer's labelled edge its segments are final.
///
/// Segment post-processing (hysteresis, padding, gap filling, dropping short
/// segments) is redone over every labelled frame, so the newest segments move.
/// On the 16 AMI test meetings at chunk 32 + rc 4, with the library's default
/// post-processing, nothing older than 0.93 s behind the edge ever changed,
/// including across the ~20 min compaction (measured with a probe over the
/// C API, kept in pam-private's history). 1.5 s leaves headroom.
pub const SETTLE_MARGIN: Ts = SAMPLE_RATE as Ts * 3 / 2;

/// How long after a word is spoken it first shows in an interim result: the
/// encoder's 160 ms step plus one 80 ms frame of right context. Preview words
/// have no times, so they are stamped this far before they first appeared.
pub const PREVIEW_LAG: Ts = SAMPLE_RATE as Ts * 6 / 25;

/// One 80 ms ASR frame, the length given to a preview word.
const FRAME: Ts = 1_280;

/// Seconds (as the library reports them) to samples.
pub fn ts(secs: f64) -> Ts {
    (secs * SAMPLE_RATE as f64).round().max(0.0) as Ts
}

/// Milliseconds (word times in `asr.h`) to samples.
pub fn ms(ms: i32) -> Ts {
    ms.max(0) as Ts * (SAMPLE_RATE as Ts / 1000)
}

/// A diarizer segment: `speaker` is the library's 1-based speaker number.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Segment {
    pub start: Ts,
    pub end: Ts,
    pub speaker: i32,
}

/// Turns snapshots of the diarizer's segments into final spans, a preview and
/// `settled_until`, assuming nothing more than the margin behind the labelled
/// edge changes.
///
/// Slots are given in order of each speaker's first final span. A speaker seen
/// only in the preview so far gets the next free slot there, provisionally.
#[derive(Debug)]
pub struct Settler {
    margin: Ts,
    settled: Ts,
    slots: BTreeMap<i32, u32>,
}

impl Settler {
    pub fn new(margin: Ts) -> Settler {
        Settler { margin, settled: 0, slots: BTreeMap::new() }
    }

    /// The segments after the diarizer has labelled audio up to `labelled`.
    pub fn update(&mut self, segments: &[Segment], labelled: Ts) -> SpeakerUpdate {
        let settled = self.settled.max(labelled.saturating_sub(self.margin));
        let finalized = self.finalize(segments, settled);
        let mut provisional = self.slots.clone();
        let preview = segments
            .iter()
            .filter(|s| s.end > settled)
            .map(|s| {
                let next = provisional.len() as u32;
                let slot = *provisional.entry(s.speaker).or_insert(next);
                SpeakerSpan { start: s.start.max(settled), end: s.end, slot }
            })
            .collect();
        SpeakerUpdate { spans: Update { finalized, preview }, settled_until: settled }
    }

    /// Where final spans end so far.
    pub fn settled(&self) -> Ts {
        self.settled
    }

    /// The segments after the end of the stream: everything not final yet.
    pub fn finish(&mut self, segments: &[Segment]) -> Vec<SpeakerSpan> {
        self.finalize(segments, Ts::MAX)
    }

    /// The parts of `segments` between the last settled point and `to`.
    fn finalize(&mut self, segments: &[Segment], to: Ts) -> Vec<SpeakerSpan> {
        let from = self.settled;
        let mut pieces: Vec<(Ts, Ts, i32)> = segments
            .iter()
            .map(|s| (s.start.max(from), s.end.min(to), s.speaker))
            .filter(|(a, b, _)| a < b)
            .collect();
        pieces.sort();
        self.settled = to;
        pieces
            .into_iter()
            .map(|(start, end, speaker)| {
                let next = self.slots.len() as u32;
                SpeakerSpan { start, end, slot: *self.slots.entry(speaker).or_insert(next) }
            })
            .collect()
    }
}

/// How much recent audio a transcriber keeps, so that a new stream can start
/// where the last utterance ended. An utterance ends 800 ms after its last word
/// (`END_OF_UTTERANCE_MS` in the backend), so a few seconds is plenty.
pub const TAIL_KEEP: Ts = 5 * SAMPLE_RATE as Ts;

/// The most recent audio pushed to a transcriber, on the meeting clock.
#[derive(Debug, Default)]
pub struct Tail {
    /// When `audio[0]` was heard.
    start: Ts,
    audio: VecDeque<f32>,
}

impl Tail {
    pub fn push(&mut self, audio: &[f32]) {
        self.audio.extend(audio);
        let extra = self.audio.len().saturating_sub(TAIL_KEEP as usize);
        self.audio.drain(..extra);
        self.start += extra as Ts;
    }

    /// Where the audio pushed so far ends.
    pub fn end(&self) -> Ts {
        self.start + self.audio.len() as Ts
    }

    /// The audio from `from` to the end, and where it starts: from `from`, or
    /// from the oldest audio kept if that is later.
    pub fn since(&self, from: Ts) -> (Ts, Vec<f32>) {
        let from = from.clamp(self.start, self.end());
        (from, self.audio.range((from - self.start) as usize..).copied().collect())
    }
}

/// Where the next stream starts after an utterance's final: at the end of its
/// last word, so that whatever came after (the pause, and perhaps the next
/// speaker's first syllables) is heard again by the new stream. A final with
/// no words starts it `pause` before `heard`.
pub fn restart_at(words: &[Word], heard: Ts, pause: Ts) -> Ts {
    words.last().map_or(heard.saturating_sub(pause), |w| w.end).min(heard)
}

/// Keeps the words `finish` flushes within the audio heard. Finishing pads
/// the stream so its last frames are decoded, and NeMo-Speech stamps a word
/// when it is emitted, so the last words can be stamped in the padding,
/// up to about 0.2 s after the audio ends.
pub fn clamp_to(words: &mut [Word], heard: Ts) {
    for w in words {
        w.end = w.end.min(heard);
        w.start = w.start.min(w.end);
    }
}

/// The utterance the recognizer is working on: interim text until its final.
#[derive(Debug, Default)]
pub struct Utterance {
    /// Each interim word and when it first appeared.
    words: Vec<(String, Ts)>,
}

impl Utterance {
    /// An interim result after `heard` samples: the preview words.
    pub fn interim(&mut self, text: &str, heard: Ts) -> Vec<Word> {
        let stamp = heard.saturating_sub(PREVIEW_LAG);
        let texts: Vec<&str> = text.split_whitespace().collect();
        self.words.truncate(texts.len());
        for (i, t) in texts.iter().enumerate() {
            match self.words.get_mut(i) {
                Some(w) => w.0 = t.to_string(),
                None => self.words.push((t.to_string(), stamp)),
            }
        }
        self.preview()
    }

    /// The utterance ended: its final words replace the interim ones.
    pub fn end(&mut self) {
        self.words.clear();
    }

    pub fn preview(&self) -> Vec<Word> {
        self.words
            .iter()
            .map(|(text, at)| Word { text: text.clone(), start: *at, end: at + FRAME, ..Word::default() })
            .collect()
    }
}

/// A model this backend knows: its files are pinned in the model store, at
/// the revision Pam was measured with.
struct Known {
    model: &'static Model,
    /// The one language an ASR model transcribes; `None` for a multilingual one.
    language: Option<&'static str>,
    /// Where its tokenizer is, when its GGUF has none to boost hints with.
    tokenizer: Option<Tokenizer>,
}

impl Known {
    /// Its one GGUF file.
    fn file(&self) -> &'static store::File {
        &self.model.files[0]
    }

    /// Whether `model` names it: by a name, its repository or its file's name.
    fn is(&self, model: &str, file: Option<&str>) -> bool {
        self.model.names.contains(&model) || self.file().repo == model || file == Some(self.file().name)
    }
}

/// A model's SentencePiece tokenizer: `len` bytes at `offset` in a file of
/// the model's repo at the pinned revision, NVIDIA's `.nemo` archive (an
/// uncompressed tar), so a range request fetches it alone.
struct Tokenizer {
    file: &'static str,
    offset: u64,
    len: u64,
    sha256: &'static str,
}

/// Where NeMo-Speech reads the tokenizer it tokenizes hints with, base64.
/// Without it, NeMo-Speech turns word boosting off.
const SPM_KEY: &str = "asr.tokenizer.spm_model";

static ASR_MODELS: &[Known] = &[
    Known {
        model: &store::NEMOTRON_EN,
        language: Some("en-US"),
        tokenizer: Some(Tokenizer {
            file: "nemotron-speech-streaming-en-0.6b.nemo",
            offset: 21_504,
            len: 251_056,
            sha256: "07d4e5a63840a53ab2d4d106d2874768143fb3fbdd47938b3910d2da05bfb0a9",
        }),
    },
    // Its `.nemo` is compressed: no range of it is the tokenizer.
    Known { model: &store::NEMOTRON_3_5, language: None, tokenizer: None },
];

static DIAR_MODELS: &[Known] =
    &[Known { model: &store::NEMOTRON_DIARIZATION, language: None, tokenizer: None }];

/// The one language an ASR model transcribes, when it is a known model that
/// has only one: `en-US` for `nemotron-en`, by name or by its file's name.
pub fn asr_model_language(model: &str) -> Option<&'static str> {
    let file = Path::new(model).file_name().and_then(|f| f.to_str());
    ASR_MODELS.iter().find(|k| k.is(model, file))?.language
}

/// The language codes the ASR model at `path` can be asked for, which a
/// multilingual model lists in its GGUF header (`asr.rnnt.prompt_dictionary`,
/// `"zh-CN:4"`). `None` for a model with one language: it has no list, and
/// can't detect a language.
pub fn model_languages(path: &Path) -> Result<Option<Vec<String>>> {
    let entries = gguf::string_array(path, "asr.rnnt.prompt_dictionary")?;
    Ok(entries.map(|e| e.iter().filter_map(|e| e.split(':').next()).map(String::from).collect()))
}

/// Refuses a `language` that `model`, which lists the codes `listed` (see
/// [`model_languages`]), can't be asked for. NeMo-Speech takes a code its
/// model doesn't know as `auto`, without an error, and Pam
/// would then treat detected speech as one fixed language. A model with one
/// language takes only that one.
pub fn check_language(model: &str, listed: Option<&[String]>, language: &str) -> Result<()> {
    if language.is_empty() || language == "auto" {
        return Ok(());
    }
    if let Some(codes) = listed {
        if !codes.iter().any(|c| c == language) {
            bail!(
                "language `{language}` isn't one `{model}` knows: use auto, or one of {}",
                codes.join(", ")
            );
        }
    } else if let Some(only) = asr_model_language(model)
        && only.split('-').next() != language.split('-').next()
    {
        bail!("`{model}` only transcribes {only}: set language to auto or {only}");
    }
    Ok(())
}

/// The GGUF file for an ASR model: a path, or the name of a model in the
/// store. One that isn't there is a [`store::Missing`] error.
pub fn asr_model(model: &str) -> Result<PathBuf> {
    resolve(model, ASR_MODELS, &Store::open()?)
}

/// The GGUF file for an ASR model that is given hints: one that can boost
/// them. For a pinned model whose GGUF has no tokenizer, that is a
/// copy beside it with the tokenizer added (`….boost.gguf`), made the first
/// time: the tokenizer is fetched from the model's repo and checked against
/// its pinned SHA-256.
pub fn boostable_asr_model(model: &str) -> Result<PathBuf> {
    let path = asr_model(model)?;
    if gguf::has(&path, SPM_KEY)? {
        return Ok(path);
    }
    let file = path.file_name().and_then(|f| f.to_str());
    let Some((k, t)) =
        ASR_MODELS.iter().find(|k| k.is(model, file)).and_then(|k| Some((k, k.tokenizer.as_ref()?)))
    else {
        bail!("`{model}` has no tokenizer to boost hints with, and Pam knows of none for it");
    };
    boost_copy(&path, k, t)
}

/// Makes the copy of `model`'s GGUF that boosts hints unless it is
/// there, as [`boostable_asr_model`] does the first time hints are given, and
/// returns its path: what `models pull` does ahead of time, so that the first
/// dictation with hints needn't. `None` for a model Pam knows no tokenizer
/// for. The model must be in `store`.
pub fn boost(store: &Store, model: &'static Model) -> Result<Option<PathBuf>> {
    let Some((k, t)) = ASR_MODELS
        .iter()
        .find(|k| std::ptr::eq(k.model, model))
        .and_then(|k| Some((k, k.tokenizer.as_ref()?)))
    else {
        return Ok(None);
    };
    let path = store.locate(model)?.remove(0);
    if gguf::has(&path, SPM_KEY)? {
        return Ok(Some(path));
    }
    boost_copy(&path, k, t).map(Some)
}

/// The copy of the GGUF at `path` with `k`'s tokenizer added, beside it,
/// made the first time.
fn boost_copy(path: &Path, k: &Known, t: &Tokenizer) -> Result<PathBuf> {
    let boost = path.with_extension("boost.gguf");
    if boost.is_file() {
        return Ok(boost);
    }
    let spm = fetch_tokenizer(k, t, &boost.with_extension("model.part"))?;
    gguf::copy_with_string(path, &boost, SPM_KEY, &base64(&spm))?;
    Ok(boost)
}

/// Fetches a model's tokenizer into `tmp` and returns it, checked.
fn fetch_tokenizer(k: &Known, t: &Tokenizer, tmp: &Path) -> Result<Vec<u8>> {
    use std::process::Command;
    let url = format!("https://huggingface.co/{}/resolve/{}/{}", k.file().repo, k.file().revision, t.file);
    let range = format!("{}-{}", t.offset, t.offset + t.len - 1);
    let run = |cmd: &mut Command| -> Result<String> {
        let out = cmd.output().with_context(|| format!("running {cmd:?}"))?;
        if !out.status.success() {
            bail!("{cmd:?} failed: {}", String::from_utf8_lossy(&out.stderr).trim());
        }
        Ok(String::from_utf8_lossy(&out.stdout).into_owned())
    };
    let fetched = (|| {
        // Bounded, so an offline or stalled network starts the meeting unboosted, not late.
        let limits = ["--connect-timeout", "5", "--max-time", "30"];
        run(Command::new("curl").args(["-fsSL", "-r", &range]).args(limits).arg("-o").arg(tmp).arg(&url))?;
        let sum = store::sha256(tmp)?;
        if sum != t.sha256 {
            bail!("bytes {range} of {url} have SHA-256 {sum}, expected {}", t.sha256);
        }
        Ok(std::fs::read(tmp)?)
    })();
    let _ = std::fs::remove_file(tmp);
    fetched.with_context(|| format!("fetching the tokenizer of `{}`", k.model.name()))
}

/// Standard base64, with padding.
fn base64(bytes: &[u8]) -> String {
    const ABC: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    for chunk in bytes.chunks(3) {
        let n = chunk.iter().enumerate().fold(0u32, |n, (i, b)| n | u32::from(*b) << (16 - 8 * i));
        for i in 0..4 {
            match i <= chunk.len() {
                true => out.push(ABC[(n >> (18 - 6 * i) & 63) as usize] as char),
                false => out.push('='),
            }
        }
    }
    out
}

/// The GGUF file for a diarizer model: a path, or the name of a model in the
/// store. One that isn't there is a [`store::Missing`] error.
pub fn diar_model(model: &str) -> Result<PathBuf> {
    resolve(model, DIAR_MODELS, &Store::open()?)
}

fn resolve(model: &str, known: &[Known], store: &Store) -> Result<PathBuf> {
    if model.ends_with(".gguf") {
        let path = PathBuf::from(model);
        if !path.is_file() {
            bail!("no model file {model}");
        }
        return Ok(path);
    }
    let Some(k) = known.iter().find(|k| k.is(model, None)) else {
        let names: Vec<&str> = known.iter().flat_map(|k| k.model.names.iter().copied()).collect();
        bail!("unknown model `{model}`: use one of {} or a path to a .gguf file", names.join(", "));
    };
    Ok(store.locate(k.model)?.remove(0))
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: Ts = SAMPLE_RATE as Ts;

    #[test]
    fn base64_pads_to_four_characters() {
        assert_eq!(base64(b"Man"), "TWFu");
        assert_eq!(base64(b"Ma"), "TWE=");
        assert_eq!(base64(b"M"), "TQ==");
        assert_eq!(base64(b""), "");
        assert_eq!(base64(&[0xff, 0xfe, 0x00, 0x3e]), "//4APg==");
    }

    fn seg(start: f64, end: f64, speaker: i32) -> Segment {
        Segment { start: ts(start), end: ts(end), speaker }
    }

    fn span(start: f64, end: f64, slot: u32) -> SpeakerSpan {
        SpeakerSpan { start: ts(start), end: ts(end), slot }
    }

    #[test]
    fn settling_cuts_segments_at_the_margin_and_finalizes_each_part_once() {
        let mut s = Settler::new(S);
        // Labelled to 5 s: settled to 4 s. Speaker 2 is still talking at the edge.
        let u = s.update(&[seg(0.5, 2.0, 2), seg(3.0, 5.0, 2)], ts(5.0));
        assert_eq!(u.settled_until, ts(4.0));
        assert_eq!(u.spans.finalized, [span(0.5, 2.0, 0), span(3.0, 4.0, 0)]);
        assert_eq!(u.spans.preview, [span(4.0, 5.0, 0)]);

        // The open segment grew and closed; a new speaker started.
        let u = s.update(&[seg(0.5, 2.0, 2), seg(3.0, 6.2, 2), seg(6.0, 7.5, 1)], ts(7.5));
        assert_eq!(u.settled_until, ts(6.5));
        assert_eq!(u.spans.finalized, [span(4.0, 6.2, 0), span(6.0, 6.5, 1)]);
        assert_eq!(u.spans.preview, [span(6.5, 7.5, 1)]);

        let rest = s.finish(&[seg(0.5, 2.0, 2), seg(3.0, 6.2, 2), seg(6.0, 8.0, 1)]);
        assert_eq!(rest, [span(6.5, 8.0, 1)]);
    }

    #[test]
    fn a_speaker_first_seen_in_the_preview_gets_a_provisional_slot() {
        let mut s = Settler::new(S);
        let u = s.update(&[seg(0.0, 1.0, 1), seg(2.5, 3.0, 4)], ts(3.0));
        assert_eq!(u.spans.finalized, [span(0.0, 1.0, 0)]);
        assert_eq!(u.spans.preview, [span(2.5, 3.0, 1)]);
        // Speaker 3 is final first, so it takes slot 1; speaker 4 moves to 2.
        let u = s.update(&[seg(0.0, 1.0, 1), seg(2.2, 2.4, 3), seg(2.5, 5.0, 4)], ts(5.0));
        assert_eq!(u.spans.finalized, [span(2.2, 2.4, 1), span(2.5, 4.0, 2)]);
    }

    #[test]
    fn settling_never_goes_back() {
        let mut s = Settler::new(S);
        assert_eq!(s.update(&[], ts(3.0)).settled_until, ts(2.0));
        assert_eq!(s.update(&[], ts(2.5)).settled_until, ts(2.0));
        assert_eq!(s.update(&[], ts(0.5)).settled_until, ts(2.0));
    }

    #[test]
    fn interim_words_keep_the_time_they_first_appeared() {
        let mut u = Utterance::default();
        let p = u.interim("no un", ts(2.0));
        assert_eq!(p.iter().map(|w| w.start).collect::<Vec<_>>(), [ts(2.0) - PREVIEW_LAG; 2]);
        // The last word grows, a new one appears.
        let p = u.interim("no unfortunately cause", ts(3.0));
        let texts: Vec<&str> = p.iter().map(|w| w.text.as_str()).collect();
        assert_eq!(texts, ["no", "unfortunately", "cause"]);
        assert_eq!(p[1].start, ts(2.0) - PREVIEW_LAG);
        assert_eq!(p[2].start, ts(3.0) - PREVIEW_LAG);
        u.end();
        assert!(u.preview().is_empty());
    }

    #[test]
    fn the_tail_keeps_recent_audio_on_the_meeting_clock() {
        let mut t = Tail::default();
        t.push(&vec![0.0; 3 * S as usize]);
        t.push(&(0..4 * S).map(|i| i as f32).collect::<Vec<_>>());
        assert_eq!(t.end(), 7 * S);
        // Only the last 5 s are kept: from 2 s.
        let (from, audio) = t.since(0);
        assert_eq!((from, audio.len()), (2 * S, 5 * S as usize));
        let (from, audio) = t.since(6 * S);
        assert_eq!((from, audio.len()), (6 * S, S as usize));
        assert_eq!(audio[0], (3 * S) as f32);
        assert_eq!(t.since(9 * S), (7 * S, Vec::new()));
    }

    #[test]
    fn a_new_stream_starts_at_the_last_words_end() {
        let w = |start, end| Word { text: "x".into(), start, end, ..Word::default() };
        assert_eq!(restart_at(&[w(S, 2 * S), w(2 * S, 3 * S)], 4 * S, S / 2), 3 * S);
        assert_eq!(restart_at(&[], 4 * S, S / 2), 4 * S - S / 2);
        assert_eq!(restart_at(&[w(S, 5 * S)], 4 * S, S / 2), 4 * S);
    }

    #[test]
    fn flushed_words_end_by_the_end_of_the_audio() {
        let w = |start, end| Word { text: "x".into(), start, end, ..Word::default() };
        let mut words = [w(S, 2 * S), w(3 * S, 5 * S), w(5 * S, 6 * S)];
        clamp_to(&mut words, 4 * S);
        let times: Vec<_> = words.iter().map(|w| (w.start, w.end)).collect();
        assert_eq!(times, [(S, 2 * S), (3 * S, 4 * S), (4 * S, 4 * S)]);
    }

    #[test]
    fn a_language_the_model_doesnt_list_is_refused() {
        let dir = std::env::temp_dir().join(format!("pam-nemo-lang-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let multi = dir.join("multi.gguf");
        gguf::write_string_array(&multi, "asr.rnnt.prompt_dictionary", &["en-US:0", "zh-CN:4", "auto:101"]);
        let listed = model_languages(&multi).unwrap();
        assert_eq!(listed.as_deref().unwrap(), ["en-US", "zh-CN", "auto"]);
        for ok in ["auto", "", "zh-CN", "en-US"] {
            check_language("nemotron-3.5", listed.as_deref(), ok).unwrap();
        }
        let err = check_language("nemotron-3.5", listed.as_deref(), "zh_CN").unwrap_err().to_string();
        assert!(err.contains("zh_CN") && err.contains("en-US, zh-CN, auto"), "{err}");
        // No list: a model with one language takes only that one; any other model, anything.
        let single = dir.join("single.gguf");
        gguf::write_string_array(&single, "general.tags", &[]);
        assert_eq!(model_languages(&single).unwrap(), None);
        check_language("nemotron-en", None, "en-GB").unwrap();
        let err = check_language("nemotron-en", None, "zh-CN").unwrap_err().to_string();
        assert!(err.contains("only transcribes en-US"), "{err}");
        let en_path = "/models/nemotron-speech-streaming-en-0.6b.q8_0.gguf";
        assert!(check_language(en_path, None, "zh-CN").is_err());
        check_language("/models/other.gguf", None, "zh-CN").unwrap();
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn model_names_resolve_in_the_store() {
        let dir = std::env::temp_dir().join(format!("pam-nemo-models-{}", std::process::id()));
        let store = Store::at(&dir);
        let k = &ASR_MODELS[0];
        let file = store.path(k.file());
        let missing = resolve("nemotron-en", ASR_MODELS, &store).unwrap_err();
        assert_eq!(missing.downcast_ref::<store::Missing>().unwrap().path, file);
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, b"").unwrap();
        assert_eq!(resolve("nemotron-en", ASR_MODELS, &store).unwrap(), file);
        assert_eq!(resolve(k.file().repo, ASR_MODELS, &store).unwrap(), file);
        let unknown = resolve("sortformer", DIAR_MODELS, &store).unwrap_err().to_string();
        assert!(unknown.contains("nemotron-3-diarization"), "{unknown}");
        assert!(resolve("/nowhere/x.gguf", ASR_MODELS, &store).is_err());
        assert_eq!(asr_model_language("nemotron-en"), Some("en-US"));
        assert_eq!(asr_model_language("nemotron-3.5"), None);
        assert_eq!(asr_model_language("/models/x.gguf"), None);
        assert_eq!(asr_model_language(file.to_str().unwrap()), Some("en-US"));
        std::fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn boost_makes_the_copy_only_for_a_model_with_a_known_tokenizer() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        assert!(boost(&store, &store::NEMOTRON_EN).unwrap_err().is::<store::Missing>());
        assert!(boost(&store, &store::NEMOTRON_3_5).unwrap().is_none(), "no tokenizer to add");
        let file = store.path(&store::NEMOTRON_EN.files[0]);
        std::fs::create_dir_all(file.parent().unwrap()).unwrap();
        std::fs::write(&file, gguf::tests::header("k")).unwrap();
        // Made already: nothing to fetch.
        let copy = file.with_extension("boost.gguf");
        std::fs::write(&copy, b"").unwrap();
        assert_eq!(boost(&store, &store::NEMOTRON_EN).unwrap(), Some(copy));
    }
}
