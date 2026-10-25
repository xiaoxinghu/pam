//! A fake clip model, for tests of the clip wrapper and its adapters: it
//! hears the words of a script in whatever clip it is given.
//!
//! [`FakeClip::audio`] writes a script's words into audio: each word is a
//! stretch of one level of its own, well above the endpointer's threshold,
//! and silence is zeros. [`FakeClip::transcribe`] reads the levels back, so a
//! clip holds the words, or the parts of words, whose audio is in it, as a
//! real model's would.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use super::{Aligner, ClipModel, ClipRecognizer};
use crate::{Result, Ts, Word};

/// The level of the script's first word: −24 dBFS. Word `i` is `i / 2^16`
/// above it, exact in `f32`.
const BASE: f32 = 0.0625;
const STEP: f32 = 1.0 / 65_536.0;

/// What the wrapper asked the fake to do.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Calls {
    /// The length of each clip transcribed, final or preview, in order.
    pub transcribed: Vec<usize>,
    /// Each `align` call: the clip's length and the words it got.
    pub aligned: Vec<(usize, Vec<Word>)>,
}

pub struct FakeClip {
    /// The script's words: their text and language.
    words: Vec<Word>,
    word_times: bool,
    slow: Option<Duration>,
    /// Clips this long or longer are heard as no speech.
    deaf_from: Option<usize>,
    calls: Arc<Mutex<Calls>>,
}

impl FakeClip {
    /// A model that hears `script`'s words, with word times.
    pub fn new(script: &[Word]) -> FakeClip {
        FakeClip {
            words: script.to_vec(),
            word_times: true,
            slow: None,
            deaf_from: None,
            calls: Arc::default(),
        }
    }

    /// Without word times: every word gets its clip's span.
    pub fn without_word_times(self) -> FakeClip {
        FakeClip { word_times: false, ..self }
    }

    /// Takes `delay` over each transcription.
    pub fn slow(self, delay: Duration) -> FakeClip {
        FakeClip { slow: Some(delay), ..self }
    }

    /// Hears no words in a clip of `len` samples or more, as Qwen3-ASR says
    /// `None` for a long clip of quiet speech and room noise.
    pub fn deaf_from(self, len: Ts) -> FakeClip {
        FakeClip { deaf_from: Some(len as usize), ..self }
    }

    /// What the wrapper has asked for so far, readable after the fake moved
    /// into it.
    pub fn calls(&self) -> Arc<Mutex<Calls>> {
        self.calls.clone()
    }

    /// An aligner that records its calls beside this model's and leaves the
    /// words unchanged.
    pub fn aligner(&self) -> FakeAligner {
        FakeAligner { calls: self.calls.clone() }
    }

    /// The model with its fake aligner, as a clip adapter with an aligner
    /// gives the wrapper.
    pub fn aligned(self) -> ClipModel {
        let aligner = self.aligner();
        ClipModel::aligned(self, aligner)
    }

    /// `len` samples of audio holding `script`'s words, on the stream's
    /// clock, and zeros between them.
    pub fn audio(script: &[Word], len: Ts) -> Vec<f32> {
        let mut audio = vec![0.0; len as usize];
        for (i, w) in script.iter().enumerate() {
            let end = (w.end.min(len)) as usize;
            let start = (w.start as usize).min(end);
            audio[start..end].fill(BASE + i as f32 * STEP);
        }
        audio
    }
}

impl ClipRecognizer for FakeClip {
    fn transcribe(&mut self, audio: &[f32]) -> Result<Vec<Word>> {
        self.calls.lock().unwrap().transcribed.push(audio.len());
        if let Some(delay) = self.slow {
            std::thread::sleep(delay);
        }
        if self.deaf_from.is_some_and(|len| audio.len() >= len) {
            return Ok(Vec::new());
        }
        let mut words: Vec<Word> = Vec::new();
        let mut run_start = 0;
        for i in 1..=audio.len() {
            if i < audio.len() && audio[i] == audio[run_start] {
                continue;
            }
            let index = ((audio[run_start] - BASE) / STEP).round();
            if audio[run_start] >= BASE
                && let Some(w) = self.words.get(index as usize)
            {
                words.push(Word {
                    start: run_start as Ts,
                    end: i as Ts,
                    starts_utterance: false,
                    ..w.clone()
                });
            }
            run_start = i;
        }
        if !self.word_times {
            for w in &mut words {
                (w.start, w.end) = (0, audio.len() as Ts);
            }
        }
        Ok(words)
    }
}

/// A fake [`Aligner`]: it records each call in the model's [`Calls`] and
/// returns the words as they came.
pub struct FakeAligner {
    calls: Arc<Mutex<Calls>>,
}

impl Aligner for FakeAligner {
    fn align(&mut self, audio: &[f32], words: Vec<Word>) -> Result<Vec<Word>> {
        self.calls.lock().unwrap().aligned.push((audio.len(), words.clone()));
        Ok(words)
    }
}
