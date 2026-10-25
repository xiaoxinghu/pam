//! The library-backed halves of the `nemo-speech` backend.

use std::ffi::{CStr, CString, c_char};
use std::ptr;

use anyhow::{Context, bail};

use super::sys::*;
use super::{SETTLE_MARGIN, Segment, Settler, Tail, Utterance, clamp_to, ms, restart_at, ts};
use crate::{
    Result, SAMPLE_RATE, SpeakerSpan, SpeakerTracker, SpeakerUpdate, TrackerConfig, Transcriber,
    TranscriberConfig, Ts, Update, Word,
};

/// Silence that ends an utterance, as Pam's first measurements set it.
const END_OF_UTTERANCE_MS: i32 = 800;
/// How strongly hints are favoured; NeMo-Speech's docs suggest
/// 2–3 for cache-aware RNNT. The pinned `nemotron-en` GGUF has no tokenizer
/// to boost them with, so a copy with one is used (`boostable_asr_model`).
const HINT_BOOST: f32 = 2.5;

fn check(status: nemo_speech_asr_status::Type, what: &str) -> Result<()> {
    if status == nemo_speech_asr_status::NEMO_SPEECH_ASR_OK {
        return Ok(());
    }
    // SAFETY: the library returns a valid C string for this thread's last error.
    let reason = unsafe { CStr::from_ptr(nemo_speech_asr_last_error()) }.to_string_lossy();
    bail!("{what} failed (status {status}): {reason}")
}

fn cstring(s: &str) -> Result<CString> {
    CString::new(s).with_context(|| format!("{s:?} contains a NUL byte"))
}

/// # Safety
/// `p` is a valid C string owned by a live result.
unsafe fn text(p: *const c_char) -> String {
    if p.is_null() { String::new() } else { unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned() }
}

/// One ASR result, destroyed when dropped.
struct AsrResult(*mut nemo_speech_asr_result);

impl Drop for AsrResult {
    fn drop(&mut self) {
        // SAFETY: we own the result.
        unsafe { nemo_speech_asr_result_destroy(self.0) }
    }
}

impl AsrResult {
    fn is_final(&self) -> bool {
        // SAFETY (here and below): `self.0` is a live result; index 0 exists
        // whenever alternative_count > 0.
        unsafe { nemo_speech_asr_result_is_final(self.0) }
    }

    fn has_text(&self) -> bool {
        unsafe { nemo_speech_asr_result_alternative_count(self.0) > 0 }
    }

    fn transcript(&self) -> String {
        if !self.has_text() {
            return String::new();
        }
        unsafe { text(nemo_speech_asr_result_transcript(self.0, 0)) }
    }

    /// The language the model reports for the result, if any.
    fn language(&self) -> Option<String> {
        if !self.has_text() || unsafe { nemo_speech_asr_result_language_count(self.0, 0) } == 0 {
            return None;
        }
        Some(unsafe { text(nemo_speech_asr_result_language_code(self.0, 0, 0)) }).filter(|l| !l.is_empty())
    }

    /// The words, on the meeting clock for a stream that started at `start`.
    fn words(&self, start: Ts) -> Vec<Word> {
        if !self.has_text() {
            return Vec::new();
        }
        let n = unsafe { nemo_speech_asr_result_word_count(self.0, 0) };
        (0..n)
            .map(|i| unsafe {
                let from = start + ms(nemo_speech_asr_result_word_start_time(self.0, 0, i));
                let end = start + ms(nemo_speech_asr_result_word_end_time(self.0, 0, i));
                Word {
                    text: text(nemo_speech_asr_result_word_text(self.0, 0, i)),
                    start: from,
                    end: end.max(from),
                    lang: None,
                    starts_utterance: i == 0,
                }
            })
            .collect()
    }
}

/// Streaming speech → words with a Nemotron streaming model: Nemotron Speech
/// Streaming EN, or the multilingual Nemotron 3.5.
///
/// **One stream per utterance, when the language is detected.** Within one
/// stream, Nemotron 3.5 is slow to change language: the first half of a
/// Mandarin turn after an English one is lost. So after each
/// final the stream is closed and a new one opened, starting at the end of the
/// final's last word; the audio since then is pushed to it again. A new stream
/// counts word times from its own start, so every time is moved onto the
/// meeting clock before it leaves here.
///
/// A new stream starts cold and misses quiet words in its first seconds: on
/// 8 AMI meetings, `nemotron-en` lost 1,100 words this way (WER 25.0% to
/// 27.8%). With one language, fixed or the model's only one, nothing can
/// change, so the transcriber keeps one stream. Whether the model has only
/// one is read from its file, so a path to `nemotron-en` counts too.
pub struct NemoTranscriber {
    // Field order is drop order: the stream goes before its recognizer.
    stream: *mut nemo_speech_asr_stream,
    recognizer: *mut nemo_speech_asr_recognizer,
    /// The options every stream is opened with, and what they point to.
    options: StreamOptions,
    /// Whether each utterance gets a stream of its own: only when its language
    /// is detected, which takes a model that lists languages.
    per_utterance: bool,
    /// Where the current stream started, on the meeting clock.
    start: Ts,
    utterance: Utterance,
    tail: Tail,
    /// Whether `finish` has flushed the stream. NeMo-Speech hands the last
    /// final out again on a second `stream_finish`, so it is called once.
    finished: bool,
}

// SAFETY: a stream is single-threaded by contract, and the transcriber is used
// from one thread at a time.
unsafe impl Send for NemoTranscriber {}

/// `nemo_speech_asr_recognition_options` and the strings it points to.
struct StreamOptions {
    /// The configured language; empty for detection.
    language: CString,
    /// The language every utterance is in, when that is known without
    /// listening: the configured one, or the only one the model knows.
    fixed: Option<String>,
    hints: Vec<CString>,
}

impl StreamOptions {
    fn new(cfg: &TranscriberConfig) -> Result<StreamOptions> {
        let language = cfg.language();
        let fixed = language.or(super::asr_model_language(&cfg.model));
        Ok(StreamOptions {
            language: cstring(language.unwrap_or(""))?,
            fixed: fixed.map(String::from),
            hints: cfg.hints.iter().map(|h| cstring(h)).collect::<Result<_>>()?,
        })
    }

    /// The language every utterance is in, when known without listening.
    fn fixed_language(&self) -> Option<String> {
        self.fixed.clone()
    }

    fn open(&self, recognizer: *mut nemo_speech_asr_recognizer) -> Result<*mut nemo_speech_asr_stream> {
        let phrases: Vec<*const c_char> = self.hints.iter().map(|h| h.as_ptr()).collect();
        let context = nemo_speech_asr_speech_context {
            size: size_of::<nemo_speech_asr_speech_context>(),
            phrases: phrases.as_ptr(),
            phrase_count: phrases.len(),
            boost: HINT_BOOST,
        };
        // SAFETY: returns a plain struct.
        let mut options = unsafe { nemo_speech_asr_recognition_options_default() };
        options.language_code = self.language.as_ptr();
        options.interim_results = true;
        options.enable_word_time_offsets = true;
        options.enable_automatic_punctuation = true;
        if !phrases.is_empty() {
            options.speech_contexts = &context;
            options.speech_context_count = 1;
        }
        let mut stream = ptr::null_mut();
        // SAFETY: `recognizer` is live; `options` and what it points to outlive the call.
        check(
            unsafe { nemo_speech_asr_streaming_recognize(recognizer, &options, &mut stream) },
            "starting ASR streaming",
        )?;
        Ok(stream)
    }
}

impl NemoTranscriber {
    pub fn open(cfg: &TranscriberConfig) -> Result<NemoTranscriber> {
        let options = StreamOptions::new(cfg)?;
        // Hints need a model that can boost them. Without one, NeMo-Speech
        // says so and transcribes without them, as before.
        let path = match cfg.hints.is_empty() {
            true => super::asr_model(&cfg.model)?,
            false => super::boostable_asr_model(&cfg.model).or_else(|_| super::asr_model(&cfg.model))?,
        };
        let listed = super::model_languages(&path)?;
        super::check_language(&cfg.model, listed.as_deref(), &cfg.language)?;
        let path = cstring(&path.to_string_lossy())?;
        let backend =
            nemo_speech_asr_backend_config { size: size_of::<nemo_speech_asr_backend_config>(), gpu: 0 };
        let model = nemo_speech_asr_model_config {
            size: size_of::<nemo_speech_asr_model_config>(),
            path: path.as_ptr(),
            name: ptr::null(),
        };
        let endpointing = nemo_speech_asr_endpointing_config {
            size: size_of::<nemo_speech_asr_endpointing_config>(),
            enable: true,
            vad_based: false,
            stop_history_eou_ms: END_OF_UTTERANCE_MS,
        };
        let config = nemo_speech_asr_recognizer_config {
            size: size_of::<nemo_speech_asr_recognizer_config>(),
            backend: &backend,
            model: &model,
            endpointing: &endpointing,
            ..Default::default()
        };
        let mut recognizer = ptr::null_mut();
        // SAFETY: every pointer in `config` outlives the call.
        check(unsafe { nemo_speech_asr_create(&config, &mut recognizer) }, "loading the ASR model")?;
        let mut t = NemoTranscriber {
            stream: ptr::null_mut(),
            recognizer,
            per_utterance: options.fixed_language().is_none() && listed.is_some(),
            options,
            start: 0,
            utterance: Utterance::default(),
            tail: Tail::default(),
            finished: false,
        };
        t.stream = t.options.open(recognizer)?;
        Ok(t)
    }

    /// Closes the stream and opens a new one at `at`, pushing it the audio
    /// heard since.
    fn reopen(&mut self, at: Ts) -> Result<()> {
        // SAFETY: the stream is live, and replaced before it is used again.
        unsafe { nemo_speech_asr_stream_close(self.stream) };
        self.stream = ptr::null_mut();
        self.stream = self.options.open(self.recognizer)?;
        let (start, audio) = self.tail.since(at);
        self.start = start;
        self.push_stream(&audio)
    }

    /// The stream, unless opening it failed: then the transcriber is broken.
    fn live(&self) -> Result<*mut nemo_speech_asr_stream> {
        if self.stream.is_null() {
            bail!("the ASR stream failed to reopen earlier");
        }
        Ok(self.stream)
    }

    fn push_stream(&mut self, audio: &[f32]) -> Result<()> {
        let stream = self.live()?;
        // SAFETY: the stream is live and `audio` is a valid slice.
        check(
            unsafe {
                nemo_speech_asr_stream_push_f32(stream, audio.as_ptr(), audio.len(), SAMPLE_RATE as i32)
            },
            "pushing audio to ASR",
        )
    }

    /// The next result, if one is ready.
    fn next(&mut self) -> Result<Option<AsrResult>> {
        let stream = self.live()?;
        let mut out = ptr::null_mut();
        // SAFETY: the stream is live.
        check(unsafe { nemo_speech_asr_stream_next(stream, &mut out) }, "reading ASR results")?;
        Ok((!out.is_null()).then_some(AsrResult(out)))
    }

    /// Ends the stream and returns the last language it reports. Nemotron 3.5
    /// reports an utterance's language on the result after its final, so it
    /// comes with the end-of-stream result. Words in what is flushed are
    /// dropped: the next stream hears that audio again.
    fn finish_language(&mut self) -> Result<Option<String>> {
        let stream = self.live()?;
        // SAFETY: the stream is live.
        check(unsafe { nemo_speech_asr_stream_finish(stream) }, "finishing ASR")?;
        let mut lang = None;
        while let Some(result) = self.next()? {
            lang = result.language().or(lang);
        }
        Ok(lang)
    }

    /// Every result available now: final words, and the latest preview. After
    /// a final, the stream is ended and a new one opened (see the type's docs).
    /// With `last`, the stream has been finished: everything left is final.
    fn drain(&mut self, last: bool) -> Result<Update<Word>> {
        let mut update = Update::default();
        // With `last`, the language reported last.
        let mut last_tag = None;
        while let Some(result) = self.next()? {
            if !result.is_final() {
                self.utterance.interim(&result.transcript(), self.tail.end());
                continue;
            }
            self.utterance.end();
            let mut words = result.words(self.start);
            if last {
                last_tag = result.language().or(last_tag);
                update.finalized.extend(words);
                continue;
            }
            if !self.per_utterance {
                self.label(&mut words, result.language());
                update.finalized.extend(words);
                continue;
            }
            let at = restart_at(&words, self.tail.end(), ms(END_OF_UTTERANCE_MS));
            let tag = self.finish_language()?.or(result.language());
            self.label(&mut words, tag);
            update.finalized.extend(words);
            self.reopen(at)?;
        }
        if last {
            self.label(&mut update.finalized, last_tag);
        }
        update.preview = self.utterance.preview();
        Ok(update)
    }

    /// Gives an utterance's words their language: the model's tag, else the
    /// configured language, checked against the utterance's script and then
    /// word by word (see [`crate::lang::fill`]).
    fn label(&self, words: &mut [Word], tag: Option<String>) {
        crate::lang::fill(words, tag.or_else(|| self.options.fixed_language()).as_deref());
    }
}

impl Drop for NemoTranscriber {
    fn drop(&mut self) {
        // SAFETY: we own both handles; null is never passed.
        unsafe {
            if !self.stream.is_null() {
                nemo_speech_asr_stream_close(self.stream);
            }
            if !self.recognizer.is_null() {
                nemo_speech_asr_destroy(self.recognizer);
            }
        }
    }
}

impl Transcriber for NemoTranscriber {
    fn push(&mut self, audio: &[f32]) -> Result<Update<Word>> {
        self.tail.push(audio);
        self.push_stream(audio)?;
        self.drain(false)
    }

    fn finish(&mut self) -> Result<Vec<Word>> {
        if self.finished {
            return Ok(Vec::new());
        }
        let stream = self.live()?;
        // SAFETY: the stream is live.
        check(unsafe { nemo_speech_asr_stream_finish(stream) }, "finishing ASR")?;
        self.finished = true;
        let mut words = self.drain(true)?.finalized;
        clamp_to(&mut words, self.tail.end());
        Ok(words)
    }
}

/// Streaming speaker diarization with Nemotron 3 Diarization.
pub struct NemoTracker {
    // Field order is drop order: the stream goes before its model.
    stream: *mut nemo_speech_diar_stream,
    model: *mut nemo_speech_diar_model,
    seconds_per_frame: f64,
    frames: i64,
    settler: Settler,
    /// The preview stays until the diarizer labels more audio.
    preview: Vec<SpeakerSpan>,
}

// SAFETY: as for `NemoTranscriber`.
unsafe impl Send for NemoTracker {}

impl NemoTracker {
    pub fn open(cfg: &TrackerConfig) -> Result<NemoTracker> {
        // Chunk 32 + right context 4 on top of the preset (2.88 s of latency instead
        // of its 1.12 s) confuses speakers less, on 16 AMI meetings.
        let (chunk, right_context) = (32, 4);
        let path = super::diar_model(&cfg.model)?;
        let path = cstring(&path.to_string_lossy())?;
        let preset = cstring("v3-streaming")?;
        let config = nemo_speech_diar_model_config {
            size: size_of::<nemo_speech_diar_model_config>(),
            model_path: path.as_ptr(),
            gpu: 0,
            preset: preset.as_ptr(),
            chunk_frames: chunk,
            right_context_frames: right_context,
            left_context_frames: -1,
            ..Default::default()
        };
        let mut model = ptr::null_mut();
        // SAFETY: every pointer in `config` outlives the call.
        check(unsafe { nemo_speech_diar_create(&config, &mut model) }, "loading the diarizer model")?;
        let mut t = NemoTracker {
            stream: ptr::null_mut(),
            model,
            // SAFETY: the model is live.
            seconds_per_frame: unsafe { nemo_speech_diar_seconds_per_frame(model) },
            frames: 0,
            settler: Settler::new(SETTLE_MARGIN),
            preview: Vec::new(),
        };
        // SAFETY: the model is live and outlives the stream (field order).
        check(unsafe { nemo_speech_diar_stream_open(model, &mut t.stream) }, "starting diarization")?;
        Ok(t)
    }

    fn segments(&self) -> Result<Vec<Segment>> {
        let mut count = 0;
        // SAFETY: the stream is live; the first call only counts, the second
        // fills a buffer of `count` segments.
        check(
            unsafe { nemo_speech_diar_segments(self.stream, ptr::null(), ptr::null_mut(), 0, &mut count) },
            "reading speaker segments",
        )?;
        let mut segments = vec![nemo_speech_diar_segment::default(); count];
        check(
            unsafe {
                nemo_speech_diar_segments(self.stream, ptr::null(), segments.as_mut_ptr(), count, &mut count)
            },
            "reading speaker segments",
        )?;
        segments.truncate(count);
        Ok(segments
            .iter()
            .map(|s| Segment { start: ts(s.start_time), end: ts(s.end_time), speaker: s.speaker })
            .collect())
    }
}

impl Drop for NemoTracker {
    fn drop(&mut self) {
        // SAFETY: we own both handles; null is never passed.
        unsafe {
            if !self.stream.is_null() {
                nemo_speech_diar_stream_close(self.stream);
            }
            if !self.model.is_null() {
                nemo_speech_diar_destroy(self.model);
            }
        }
    }
}

impl SpeakerTracker for NemoTracker {
    fn push(&mut self, audio: &[f32]) -> Result<SpeakerUpdate> {
        // SAFETY: the stream is live and `audio` is a valid slice.
        check(
            unsafe {
                nemo_speech_diar_stream_push_f32(self.stream, audio.as_ptr(), audio.len(), SAMPLE_RATE as i32)
            },
            "pushing audio to the diarizer",
        )?;
        // SAFETY: the stream is live.
        let frames = unsafe { nemo_speech_diar_frame_count(self.stream) };
        if frames == self.frames {
            let spans = Update { finalized: Vec::new(), preview: self.preview.clone() };
            return Ok(SpeakerUpdate { spans, settled_until: self.settler.settled() });
        }
        self.frames = frames;
        let labelled = ts(frames as f64 * self.seconds_per_frame);
        let update = self.settler.update(&self.segments()?, labelled);
        self.preview = update.spans.preview.clone();
        Ok(update)
    }

    fn finish(&mut self) -> Result<Vec<SpeakerSpan>> {
        // SAFETY: the stream is live.
        check(unsafe { nemo_speech_diar_stream_finish(self.stream) }, "finishing diarization")?;
        Ok(self.settler.finish(&self.segments()?))
    }
}
