//! Clip models: a model that transcribes only finished audio
//! implements [`ClipRecognizer`], and the shared [`ClipTranscriber`] makes it
//! a [`Transcriber`].
//!
//! - **Clips:** the shared [`Endpointer`] cuts the audio into clips, and says
//!   why it cut.
//! - **Finals:** each clip is transcribed when it closes, in the push that
//!   closes it, then passed to the adapter's [`Aligner`], if it has one. Its
//!   words are moved to the stream's clock and are final. The first word
//!   final in a clip that starts an utterance is marked: every clip does,
//!   except one whose speech went on from a cut at the cap
//!   ([`Clip::starts_utterance`]).
//! - **Preview:** the open clip, and only it, is transcribed again every
//!   `preview_every` of audio, on a thread of its own so a push doesn't wait
//!   for it. One due while the last is still running is skipped, not queued.
//!   A final waits for a running preview, since both use the one engine. A
//!   preview with no words leaves the clip's last words showing.
//! - **Words heard stay heard:** a clip whose final comes back with no
//!   words keeps the words of its last preview that had some, aligned on the
//!   audio that preview heard, with a warning; it waits for a preview still
//!   to come first. Qwen3-ASR says `None` for a
//!   whole clip once enough room noise follows quiet speech, though a shorter
//!   preview of it had every word.
//! - **Language:** a word the model gives none gets the configured language,
//!   else its script's ([`crate::lang::fill`]).

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;

use anyhow::{anyhow, bail};

use crate::{Result, SAMPLE_RATE, Transcriber, Ts, Update, Word};

pub mod endpoint;
#[cfg(test)]
mod fake;

pub use endpoint::{Clip, Cut, Endpointer};
#[cfg(test)]
pub(crate) use fake::FakeClip;

/// A model that transcribes a finished piece of audio at once.
pub trait ClipRecognizer: Send {
    /// One clip, 16 kHz mono f32. Word times count from the clip's start.
    /// A model without word times gives every word the clip's span.
    fn transcribe(&mut self, audio: &[f32]) -> Result<Vec<Word>>;
}

/// A forced aligner: a model that gives a finished clip's words their times.
/// It is a part of its own, so one aligner can serve any clip model and be
/// swapped or measured apart from it.
pub trait Aligner: Send {
    /// `words`, with times counted from the clip's start. Called on finished
    /// clips only, never on previews.
    fn align(&mut self, audio: &[f32], words: Vec<Word>) -> Result<Vec<Word>>;
}

/// What a clip adapter gives the wrapper: its model, and an aligner if it
/// has one. A model alone converts with `.into()`.
pub struct ClipModel {
    pub engine: Box<dyn ClipRecognizer>,
    pub aligner: Option<Box<dyn Aligner>>,
}

impl<R: ClipRecognizer + 'static> From<R> for ClipModel {
    fn from(engine: R) -> ClipModel {
        ClipModel { engine: Box::new(engine), aligner: None }
    }
}

impl ClipModel {
    /// `engine` with `aligner` giving its finals their word times.
    pub fn aligned(engine: impl ClipRecognizer + 'static, aligner: impl Aligner + 'static) -> ClipModel {
        ClipModel { engine: Box::new(engine), aligner: Some(Box::new(aligner)) }
    }
}

/// How often the open clip is transcribed again for the preview, by default: 1 s.
pub const PREVIEW_EVERY: Ts = SAMPLE_RATE as Ts;

type Engine = Arc<Mutex<Box<dyn ClipRecognizer>>>;

fn lock(engine: &Engine) -> Result<MutexGuard<'_, Box<dyn ClipRecognizer>>> {
    engine.lock().map_err(|_| anyhow!("the clip model panicked"))
}

/// A [`ClipRecognizer`] as a [`Transcriber`] (see the module's docs).
pub struct ClipTranscriber {
    engine: Engine,
    /// Gives each closed clip's words their times, when the adapter has one.
    aligner: Option<Box<dyn Aligner>>,
    endpointer: Endpointer,
    /// The configured language; `None` for detection.
    language: Option<String>,
    /// How often the open clip is previewed; 0 for never.
    preview_every: Ts,
    /// Where the next preview is due, while a clip is open.
    next_preview: Option<Ts>,
    previews: Option<Previews>,
    /// How many clips have closed: the open clip's number.
    clip: u64,
    /// The latest preview of the open clip with words, on the stream's clock.
    preview: Vec<Word>,
    /// That preview as the model gave it, and the length of audio it heard:
    /// the clip's words if its final has none.
    heard: Option<(usize, Vec<Word>)>,
    /// What went wrong without failing, until the caller takes it.
    warnings: Vec<String>,
    /// Whether the next final word starts an utterance: from a clip that
    /// starts one until a word is final, so a clip with no words passes it on.
    starts_utterance: bool,
    finished: bool,
}

/// The thread that transcribes previews.
struct Previews {
    jobs: Option<Sender<Job>>,
    results: Receiver<(u64, Ts, usize, Result<Vec<Word>>)>,
    /// Set while a preview is waiting or running.
    busy: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
}

/// A preview to transcribe: the open clip so far.
struct Job {
    clip: u64,
    start: Ts,
    audio: Vec<f32>,
}

impl Previews {
    fn start(engine: Engine) -> Previews {
        let (jobs, inbox) = channel::<Job>();
        let (outbox, results) = channel();
        let busy = Arc::new(AtomicBool::new(false));
        let idle = busy.clone();
        let worker = std::thread::spawn(move || {
            for job in inbox {
                let words = lock(&engine).and_then(|mut e| e.transcribe(&job.audio));
                let _ = outbox.send((job.clip, job.start, job.audio.len(), words));
                idle.store(false, Ordering::SeqCst);
            }
        });
        Previews { jobs: Some(jobs), results, busy, worker: Some(worker) }
    }
}

impl Drop for Previews {
    fn drop(&mut self) {
        self.jobs = None;
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

impl ClipTranscriber {
    /// Wraps `model`, for the configured `language` (`None` to detect),
    /// previewing every `preview_every` samples (0 for never).
    pub fn new(model: ClipModel, language: Option<&str>, preview_every: Ts) -> ClipTranscriber {
        let engine = Arc::new(Mutex::new(model.engine));
        let previews = (preview_every > 0).then(|| Previews::start(engine.clone()));
        ClipTranscriber {
            engine,
            aligner: model.aligner,
            endpointer: Endpointer::new(),
            language: language.map(String::from),
            preview_every,
            next_preview: None,
            previews,
            clip: 0,
            preview: Vec::new(),
            heard: None,
            warnings: Vec::new(),
            starts_utterance: true,
            finished: false,
        }
    }

    /// Transcribes and aligns a closed clip: its final words, or, when it
    /// has none, those of its last preview with any.
    fn finalize(&mut self, clip: Clip) -> Result<Vec<Word>> {
        let mut words = lock(&self.engine)?.transcribe(&clip.audio)?;
        let mut audio = &clip.audio[..];
        if words.is_empty() {
            self.collect_previews(true)?;
            if let Some((len, heard)) = self.heard.take() {
                let secs = |at: Ts| at as f64 / f64::from(SAMPLE_RATE);
                self.warnings.push(format!(
                    "the clip from {:.1} s to {:.1} s came back with no words; \
                     it keeps the {} words its preview heard to {:.1} s",
                    secs(clip.start),
                    secs(clip.end()),
                    heard.len(),
                    secs(clip.start + len as Ts),
                ));
                audio = &clip.audio[..len.min(clip.audio.len())];
                words = heard;
            }
        }
        let words = match &mut self.aligner {
            Some(aligner) => aligner.align(audio, words)?,
            None => words,
        };
        let mut words = self.place(words, clip.start, audio.len());
        self.starts_utterance |= clip.starts_utterance;
        if let Some(first) = words.first_mut() {
            first.starts_utterance = self.starts_utterance;
            self.starts_utterance = false;
        }
        self.clip += 1;
        self.preview.clear();
        self.heard = None;
        self.next_preview = None;
        Ok(words)
    }

    /// A clip's words on the stream's clock, inside the clip, with their
    /// languages filled in. None starts an utterance yet.
    fn place(&self, words: Vec<Word>, start: Ts, len: usize) -> Vec<Word> {
        let len = len as Ts;
        let mut words: Vec<Word> = words
            .into_iter()
            .map(|w| Word {
                start: start + w.start.min(len),
                end: start + w.end.max(w.start).min(len),
                starts_utterance: false,
                ..w
            })
            .collect();
        crate::lang::fill(&mut words, self.language.as_deref());
        words
    }

    /// Takes the previews that are done, keeping the open clip's latest
    /// with words; with `wait`, the one waiting or running too.
    fn collect_previews(&mut self, wait: bool) -> Result<()> {
        let Some(p) = &self.previews else { return Ok(()) };
        let mut latest = None;
        let pending = (wait && p.busy.load(Ordering::SeqCst)).then(|| p.results.recv().ok()).flatten();
        for (clip, start, len, words) in
            pending.into_iter().chain(std::iter::from_fn(|| p.results.try_recv().ok()))
        {
            if clip != self.clip {
                continue;
            }
            let words = words?;
            if !words.is_empty() {
                latest = Some((start, len, words));
            }
        }
        if let Some((start, len, words)) = latest {
            let mut placed = self.place(words.clone(), start, len);
            if let Some(first) = placed.first_mut() {
                first.starts_utterance = self.starts_utterance || self.endpointer.open_starts_utterance();
            }
            self.preview = placed;
            self.heard = Some((len, words));
        }
        Ok(())
    }

    /// Starts a preview of the open clip when one is due, unless the last
    /// is still running: then this one is skipped.
    fn start_preview(&mut self) {
        let Some(p) = &self.previews else { return };
        let Some((start, audio)) = self.endpointer.open() else { return };
        let heard = self.endpointer.heard();
        let due = *self.next_preview.get_or_insert(start + self.preview_every);
        if heard < due {
            return;
        }
        self.next_preview = Some(heard + self.preview_every);
        if p.busy.swap(true, Ordering::SeqCst) {
            return;
        }
        let job = Job { clip: self.clip, start, audio: audio.to_vec() };
        if p.jobs.as_ref().is_some_and(|jobs| jobs.send(job).is_err()) {
            p.busy.store(false, Ordering::SeqCst);
        }
    }

    /// Waits until no preview is running, so tests can see each one.
    #[cfg(test)]
    fn settle(&self) {
        while self.previews.as_ref().is_some_and(|p| p.busy.load(Ordering::SeqCst)) {
            std::thread::yield_now();
        }
    }
}

impl Transcriber for ClipTranscriber {
    fn push(&mut self, audio: &[f32]) -> Result<Update<Word>> {
        if self.finished {
            bail!("the clip transcriber has finished");
        }
        let mut finalized = Vec::new();
        for clip in self.endpointer.push(audio) {
            finalized.extend(self.finalize(clip)?);
        }
        self.collect_previews(false)?;
        self.start_preview();
        Ok(Update { finalized, preview: self.preview.clone() })
    }

    /// Closes the open clip and returns its words, and those of any clip the
    /// last part of a frame closed. A second call returns none.
    fn finish(&mut self) -> Result<Vec<Word>> {
        if self.finished {
            return Ok(Vec::new());
        }
        self.finished = true;
        let mut words = Vec::new();
        for clip in self.endpointer.finish() {
            words.extend(self.finalize(clip)?);
        }
        self.preview.clear();
        Ok(words)
    }

    fn warnings(&mut self) -> Vec<String> {
        std::mem::take(&mut self.warnings)
    }
}

#[cfg(test)]
mod tests;
