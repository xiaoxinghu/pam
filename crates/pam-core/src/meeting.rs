//! A running meeting: input → both backends → pipeline → folder and events.
//!
//! ```text
//! input thread ─┬─▶ transcriber thread ─┐
//!   (audio.wav)  └─▶ tracker thread ─────┴─▶ merger thread ─▶ folder, subscribers
//!                              commands ───────┘   ▲   │ new lines
//!                     scribe thread (cloud model) ─────┘◀──┘
//!                     agent thread (pi, one task at a time) ◀─▶ merger
//!                     intent checks (cloud or local model, a thread each) ◀─▶ merger
//! ```
//!
//! The merger thread owns all meeting state. It keeps taking commands after
//! the input ends, until the meeting is stopped. The scribe thread only waits
//! on the model; the merger saves what it reports, so lines never wait for it.
//! The agent thread runs one task's pi at a time; the merger creates tasks,
//! saves their status, and watches what `pam say` and `pam task ask` write for
//! them. Speech that holds the wake name, and all speech while a question is
//! on the stage, is checked as the transcriber ends each utterance, each check
//! on a thread of its own; a spoken request starts at once, before its line is
//! final, but what it shows is held until its cancel window closes. The merger
//! runs the countdown of the agent's questions to the room.
//!
//! The audio channels are bounded: a backend slower than the input holds the
//! input back instead of queueing the whole recording in memory. The mic
//! can't be held back, so it drops audio once the backends are 30 s behind,
//! and says so.

use std::collections::VecDeque;
use std::fmt;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use chrono::Local;
use crossbeam_channel::{Receiver, Sender, bounded, unbounded};
use pam_speech::{
    SpeakerSpan, SpeakerTracker, SpeakerUpdate, TrackerConfig, Transcriber, TranscriberConfig, Update, Word,
};

use crate::Ts;
use crate::agent::{self, AgentConfig, Inbox, Outcome, Report, Runner};
use crate::audio::{self, Mic, Pacer, Recorder, Source, WavSource};
use crate::brief;
use crate::clock::{SECOND, mmss};
use crate::config::Config;
use crate::event::{Command, CoreEvent, Line, SpkId, TaskId, UtteranceId};
use crate::folder::{EventLog, MeetingFolder, MeetingInfo, timestamp, write_info, write_summary};
use crate::identity::{Identity, ScribeIntro};
use crate::invocation::{self, Action, Asked, Invocations, Kind, Verdict};
use crate::llm::{self, Request};
use crate::local;
use crate::models::{self, AgentInfo, IntentInfo, ModelStatus, ScribeInfo};
use crate::notes::Notes;
use crate::pipeline::{Pipeline, Utterance};
use crate::question::{self, Answer, By, End, Question};
use crate::record::{Event, Ref, Summary, read_jsonl};
use crate::scribe::{self, Reply};
use crate::task::{self, TaskState, TaskStatus, Via};
use crate::usage::{Usage, Used, User};

/// Audio is pushed to the backends in chunks of this many samples (100 ms).
const CHUNK: usize = 1_600;
/// Chunks each backend may have waiting (6.4 s of audio).
const QUEUED: usize = 64;
/// How often the meeting looks for events someone else added (`pam note`).
const POLL: Duration = Duration::from_millis(250);
/// How long an intent check through pi may take before it counts as failed
/// (and so as a request). A short reply takes about 2.7 s. A local
/// check takes about 1 s and isn't timed.
const INTENT_TIMEOUT: Duration = Duration::from_secs(15);
/// How long past its deadline a question waits for a reply begun before it
/// that is still being heard or checked: an intent check's own limit.
/// The asker gives up 25 s after its countdown, which may start 5 s before
/// the question shows, in the cancel window, so the meeting ends it first.
const REPLY_WAIT: Duration = INTENT_TIMEOUT;
/// How long after its deadline a question whose replies are checked waits
/// before deciding whether one is on its way: words show in the preview about 0.3 s after they are
/// said, so speech begun just before the deadline may not show at once.
const HEARD_LAG: Duration = Duration::from_secs(1);
/// How long to wait after a failed scribe run before trying again.
const RETRY: Duration = Duration::from_secs(30);

/// Where the audio comes from.
#[derive(Debug, Clone)]
pub enum Input {
    /// Silence of a given length. Drives the `fake` backends, which only count samples.
    Silence { duration: Ts, label: String },
    /// A 16 kHz WAV file, replayed and recorded to `audio.wav`.
    Wav(PathBuf),
    /// A microphone, by name, or the default input; recorded to `audio.wav`.
    Mic(Option<String>),
}

impl Input {
    fn open(&self) -> Result<Source> {
        Ok(match self {
            Input::Silence { duration, .. } => Source::Silence { left: *duration },
            Input::Wav(path) => Source::Wav(WavSource::open(path)?),
            Input::Mic(None) => Source::Mic(Box::new(Mic::open(None)?)),
            Input::Mic(Some(name)) => {
                Source::Mic(Box::new(Mic::open(Some(name)).context("the mic the config file's `mic` names")?))
            }
        })
    }

    /// What `meeting.json` records as the input, once `source` is open.
    fn label(&self, source: &Source) -> String {
        match (self, source) {
            (_, Source::Mic(mic)) => format!("mic: {}", mic.describe()),
            (Input::Silence { label, .. }, _) => label.clone(),
            (Input::Wav(path), _) => path.display().to_string(),
            (Input::Mic(_), _) => "mic".into(),
        }
    }
}

#[derive(Debug, Clone)]
pub struct MeetingConfig {
    /// Where meeting folders go.
    pub root: PathBuf,
    pub title: String,
    pub assistant_name: String,
    pub input: Input,
    /// Replay at real-time speed instead of as fast as possible. The mic is always live.
    pub realtime: bool,
    /// Backend and model, e.g. `nemo-speech:nemotron-en` or `fake:fixtures/standup.fake.jsonl`.
    pub asr: String,
    /// The language the transcriber listens for: a code such as `en-US`, or
    /// `auto` to detect each utterance's language.
    pub language: String,
    /// Backend and model, e.g. `nemo-speech:nemotron-3-diarization` or `fake:fixtures/standup.fake.jsonl`.
    pub diarizer: String,
    /// Names known in advance, passed to the transcriber as hints when the
    /// stream starts.
    pub attendees: Vec<String>,
    /// The scribe's model as chosen and checked, which `meeting.json`
    /// records and front-ends get as `Scribe`; `None` for a meeting that
    /// chose none, such as a replay in a test.
    pub scribe_info: Option<ScribeInfo>,
    /// Who takes the notes; `None` for no notes.
    pub scribe: Option<Llm>,
    /// How to reach the agent; `None` for a meeting that takes no requests.
    pub agent: Option<AgentConfig>,
    /// The intent check's model as chosen and checked, which
    /// `meeting.json` records and front-ends get as `Intent`; `None` for a
    /// meeting that chose none.
    pub intent_info: Option<IntentInfo>,
    /// The model that decides whether what is said to the assistant is a
    /// request; `None` takes typed requests only.
    pub intent: Option<Llm>,
    /// How long the room has to cancel a spoken request, in meeting time.
    pub cancel_window: Ts,
    /// The countdown on the agent's questions, in seconds, when the agent
    /// sets none; kept in `meeting.json`, where `pam task ask` reads it.
    pub question_timeout: u64,
}

/// A text model and its name, as `meeting.json` records it: the scribe's, or
/// the intent check's.
#[derive(Clone)]
pub struct Llm {
    pub name: String,
    pub model: Arc<dyn llm::Model>,
}

impl Llm {
    /// `model` (`provider/id`), reached through the `pi` executable `program`.
    pub fn pi(program: impl Into<PathBuf>, model: &str) -> Llm {
        let pi = llm::Pi { program: program.into(), ..llm::Pi::new(model) };
        Llm { name: model.to_string(), model: Arc::new(pi) }
    }

    /// The intent check through pi, as a meeting asks it: a check that takes
    /// longer than [`INTENT_TIMEOUT`] fails.
    fn intent_pi(program: impl Into<PathBuf>, model: &str) -> Llm {
        let check = llm::Pi { program: program.into(), timeout: INTENT_TIMEOUT, ..llm::Pi::new(model) };
        Llm { name: model.to_string(), model: Arc::new(check) }
    }

    /// The local intent check, the model file at `path`, which starts
    /// loading now with the system prompt for an assistant called `assistant`.
    fn intent_local(path: PathBuf, assistant: &str) -> Llm {
        let check = local::Local::start(path, invocation::system_prompt(assistant));
        Llm { name: local::id(), model: Arc::new(check) }
    }
}

impl fmt::Debug for Llm {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "Llm({})", self.name)
    }
}

impl MeetingConfig {
    /// Replays a fake script: silence of the script's length into the `fake` backends.
    pub fn fake(root: PathBuf, script: PathBuf) -> Result<MeetingConfig> {
        let duration = pam_speech::fake::Script::load(&script)?.duration;
        let backend = format!("fake:{}", script.display());
        Ok(MeetingConfig {
            root,
            title: String::new(),
            assistant_name: "Pam".into(),
            input: Input::Silence { duration, label: script.display().to_string() },
            realtime: false,
            asr: backend.clone(),
            language: "auto".into(),
            diarizer: backend,
            attendees: Vec::new(),
            scribe_info: None,
            scribe: None,
            agent: None,
            intent_info: None,
            intent: None,
            cancel_window: invocation::CANCEL_WINDOW,
            question_timeout: question::DEFAULT_TIMEOUT,
        })
    }

    /// Replays a 16 kHz WAV file through the backends `config` chooses.
    pub fn wav(root: PathBuf, wav: PathBuf, config: &Config) -> MeetingConfig {
        MeetingConfig {
            root,
            title: String::new(),
            assistant_name: "Pam".into(),
            input: Input::Wav(wav),
            realtime: false,
            asr: config.asr.clone(),
            language: config.language.clone(),
            diarizer: config.diarizer.clone(),
            attendees: Vec::new(),
            scribe_info: None,
            scribe: None,
            agent: None,
            intent_info: None,
            intent: None,
            cancel_window: invocation::CANCEL_WINDOW,
            question_timeout: question::DEFAULT_TIMEOUT,
        }
    }

    /// Listens to the mic `config` names, or the default input, through the
    /// backends it chooses.
    pub fn mic(root: PathBuf, config: &Config) -> MeetingConfig {
        MeetingConfig {
            input: Input::Mic(config.mic.clone()),
            ..MeetingConfig::wav(root, PathBuf::new(), config)
        }
    }

    /// Chooses and checks the scribe's, the intent check's and the agent's
    /// models, and sets the meeting up with them: the scribe and
    /// the intent check when their models are ready, and the agent, which
    /// runs `pam`. The assistant's name must be set first: a local intent
    /// check starts loading here, with its system prompt.
    pub fn use_models(&mut self, choice: &models::Choice, pam: Option<PathBuf>) {
        let (scribe, intent, agent) = models::check(choice);
        let through_pi = |m: &models::ModelInfo| m.pi.clone().zip(m.model.clone()).filter(|_| m.usable());
        self.scribe = through_pi(&scribe).map(|(pi, model)| Llm::pi(pi, &model));
        self.intent = match local::path() {
            // Loaded only for an agent to run what it hears: it holds about 3 GB.
            Ok(path) if choice.intent.is_local() && intent.usable() && agent.usable() => {
                Some(Llm::intent_local(path, &self.assistant_name))
            }
            _ if choice.intent.is_local() => None,
            _ => through_pi(&intent).map(|(pi, model)| Llm::intent_pi(pi, &model)),
        };
        self.scribe_info = Some(scribe);
        self.intent_info = Some(intent);
        self.agent = Some(AgentConfig { info: agent, pam });
    }

    /// Whether "Pam, …" can start a task: that needs an agent to run it and
    /// a model to check it, as [`Meeting::start`] sets them up.
    fn takes_spoken_requests(&self) -> bool {
        self.intent.is_some() && self.agent.as_ref().is_some_and(|a| a.info.usable())
    }
}

#[derive(Debug, Clone)]
pub struct StopReport {
    pub folder: PathBuf,
    pub lines: u32,
    /// Audio heard, in samples.
    pub duration: Ts,
    /// Why `notes.md` leaves out the last lines, when the scribe couldn't read
    /// them before the meeting stopped. The meeting itself is saved.
    pub notes_failure: Option<String>,
    /// Why `context.md` couldn't be rendered once the meeting stopped.
    pub context_failure: Option<String>,
}

/// Events every subscriber should see, even one that subscribes late.
#[derive(Default)]
struct Hub {
    subscribers: Vec<Sender<CoreEvent>>,
    history: Vec<CoreEvent>,
    clock: Option<CoreEvent>,
    notes: Option<CoreEvent>,
    usage: Option<CoreEvent>,
}

impl Hub {
    fn send(&mut self, event: CoreEvent) {
        match &event {
            CoreEvent::Clock { .. } => self.clock = Some(event.clone()),
            CoreEvent::NotesUpdated(_) => self.notes = Some(event.clone()),
            CoreEvent::Usage(_) => self.usage = Some(event.clone()),
            CoreEvent::LineFinal(_)
            | CoreEvent::Scribe(_)
            | CoreEvent::Intent(_)
            | CoreEvent::Agent(_)
            | CoreEvent::Invocation { .. }
            | CoreEvent::TaskUpdated(_)
            | CoreEvent::StageMessage { .. }
            | CoreEvent::Question { .. }
            | CoreEvent::QuestionClosed { .. }
            | CoreEvent::SpeakerAdded(_)
            | CoreEvent::SpeakerRenamed { .. }
            | CoreEvent::SpeakersMerged { .. }
            | CoreEvent::LineMoved { .. }
            | CoreEvent::Undone { .. }
            | CoreEvent::InputEnded { .. }
            | CoreEvent::Error(_) => self.history.push(event.clone()),
            _ => {}
        }
        self.subscribers.retain(|s| s.send(event.clone()).is_ok());
    }
}

enum Msg {
    Clock(Ts),
    /// How loud the input was in the chunk that ends here (0–1).
    Level(Ts, f32),
    Words(Update<Word>, Ts),
    Speakers(SpeakerUpdate, Ts),
    WordsDone(Vec<Word>),
    SpeakersDone(Vec<SpeakerSpan>),
    Failed(String),
    /// A problem worth reporting that doesn't stop anything.
    Warning(String),
    Command(Command, Sender<Result<()>>),
    /// The scribe read `lines[..through]` and replied.
    Scribed {
        through: usize,
        reply: Result<Reply>,
        used: Used,
    },
    /// The agent started or ended a task's run.
    Agent(Report),
    /// The intent check `check` came back after `took`: what the model
    /// replied, and what that says.
    Intent {
        check: u64,
        reply: Option<String>,
        verdict: Result<Verdict>,
        used: Used,
        took: Duration,
    },
    /// Time to look for events someone else added, for what `pam say` and
    /// `pam task ask` wrote, and for questions whose time is up.
    Poll,
    /// Every worker is done and the meeting is stopping: finish the notes, then end.
    Stop,
}

/// One scribe run: the prompt, and how many lines it covers.
struct Job {
    prompt: String,
    through: usize,
}

/// One intent check.
struct Check {
    id: u64,
    prompt: String,
    /// What a local model's reply is held to: the answers the question on the stage takes.
    grammar: String,
}

pub struct Meeting {
    folder: PathBuf,
    hub: Arc<Mutex<Hub>>,
    stop: Arc<AtomicBool>,
    /// Into the merger thread; `None` once stopping.
    commands: Option<Sender<Msg>>,
    workers: Vec<JoinHandle<()>>,
    merger: Option<JoinHandle<Result<Merged>>>,
    /// The intent checks' thread, which ends once the merger has.
    intent: Option<JoinHandle<()>>,
    info: MeetingInfo,
}

/// What the transcriber is told to listen for: the attendees, and the
/// assistant's name when spoken requests are taken, which unboosted was
/// heard as "PEM".
fn hints(cfg: &MeetingConfig) -> Vec<String> {
    let name =
        cfg.takes_spoken_requests().then(|| cfg.assistant_name.clone()).filter(|n| !n.trim().is_empty());
    cfg.attendees.iter().cloned().chain(name).collect()
}

impl Meeting {
    /// Creates the meeting folder, opens the backends and starts listening.
    pub fn start(cfg: MeetingConfig) -> Result<Meeting> {
        let source = cfg.input.open()?;
        let (asr, asr_model) = pam_speech::parse_backend(&cfg.asr)?;
        let (diar, diar_model) = pam_speech::parse_backend(&cfg.diarizer)?;
        let transcriber = pam_speech::open_transcriber(
            asr,
            &TranscriberConfig {
                model: asr_model.into(),
                language: cfg.language.clone(),
                hints: hints(&cfg),
            },
        )?;
        let tracker = pam_speech::open_tracker(diar, &TrackerConfig { model: diar_model.into() })?;
        Meeting::begin(cfg, source, transcriber, tracker)
    }

    fn begin(
        mut cfg: MeetingConfig,
        source: Source,
        transcriber: Box<dyn Transcriber>,
        tracker: Box<dyn SpeakerTracker>,
    ) -> Result<Meeting> {
        // A local intent check has been loading meanwhile. One that
        // can't load leaves spoken requests off, and says why.
        if let Some(intent) = &cfg.intent
            && let Err(e) = intent.model.ready()
        {
            cfg.intent = None;
            if let Some(info) = cfg.intent_info.as_mut() {
                info.status = ModelStatus::Unavailable;
                info.reason = Some(format!("the local model didn't load: {e:#}"));
            }
        }
        // Checked before the meeting (`use_models`), so a request never finds out the hard way.
        let agent = cfg.agent.as_ref().map(|a| a.info.clone());

        let started = Local::now();
        let folder = MeetingFolder::create(&cfg.root, &cfg.title, started)?;
        let info = MeetingInfo {
            title: cfg.title.clone(),
            started_at: timestamp(started),
            assistant_name: cfg.assistant_name.clone(),
            input: cfg.input.label(&source),
            asr: cfg.asr.clone(),
            language: cfg.language.clone(),
            diarizer: cfg.diarizer.clone(),
            scribe: cfg.scribe_info.clone(),
            intent: cfg.intent_info.clone(),
            status: "recording".into(),
            attendees: cfg.attendees.clone(),
            stopped_at: None,
            duration: None,
            agent: agent.clone(),
            question_timeout: agent.as_ref().map(|_| cfg.question_timeout),
            usage: Usage::new(
                cfg.scribe_info.as_ref().and_then(|s| s.login.as_deref()),
                cfg.intent_info.as_ref().and_then(|i| i.login.as_deref()),
                agent.as_ref().and_then(|a| a.login.as_deref()),
            ),
        };
        folder.write_info(&info)?;
        EventLog::lock(folder.path())?.write_context()?;
        let recorder = match source.is_audio() {
            true => Some(Recorder::create(&folder.path().join("audio.wav"))?),
            false => None,
        };

        let clock = Arc::new(AtomicU64::new(0));
        let coming = Arc::new(Mutex::new(Vec::new()));
        let runner = match &cfg.agent {
            Some(a) if a.info.usable() => {
                let mut runner = Runner::for_meeting(a, folder.path(), &cfg.assistant_name)?;
                runner.clock = clock.clone();
                runner.coming = coming.clone();
                Some(runner)
            }
            _ => None,
        };

        let hub = Arc::new(Mutex::new(Hub::default()));
        if let Some(s) = &cfg.scribe_info {
            hub.lock().unwrap().send(CoreEvent::Scribe(s.clone()));
        }
        if let Some(i) = &cfg.intent_info {
            hub.lock().unwrap().send(CoreEvent::Intent(i.clone()));
        }
        if let Some(a) = &agent {
            hub.lock().unwrap().send(CoreEvent::Agent(a.clone()));
        }
        let usage = info.usage.clone();
        hub.lock().unwrap().send(CoreEvent::Usage(Box::new(usage.clone())));
        let stop = Arc::new(AtomicBool::new(false));
        let (msg_tx, msg_rx) = unbounded();
        let commands = Some(msg_tx.clone());
        let (asr_tx, asr_rx) = bounded(QUEUED);
        let (diar_tx, diar_rx) = bounded(QUEUED);
        let mut workers = vec![
            spawn("pam-input", {
                let (stop, msg_tx) = (stop.clone(), msg_tx.clone());
                move || feed(source, recorder, cfg.realtime, stop, vec![asr_tx, diar_tx], msg_tx)
            }),
            spawn("pam-asr", {
                let msg_tx = msg_tx.clone();
                move || run_transcriber(transcriber, asr_rx, msg_tx)
            }),
            spawn("pam-diar", {
                let msg_tx = msg_tx.clone();
                move || run_tracker(tracker, diar_rx, msg_tx)
            }),
            spawn("pam-poll", {
                let (stop, msg_tx) = (stop.clone(), msg_tx.clone());
                move || poll(stop, msg_tx)
            }),
        ];
        // Joined: stopping the meeting stops the task running, and so pi.
        let mut tasks = None;
        let mut checks = None;
        let mut intent_thread = None;
        let mut intent_model = String::new();
        if let Some(runner) = runner {
            let (inbox, inbox_rx) = unbounded();
            workers.push(spawn("pam-agent", {
                let (stop, msg_tx) = (stop.clone(), msg_tx.clone());
                move || {
                    agent::serve(runner, inbox_rx, &stop, |r| {
                        let _ = msg_tx.send(Msg::Agent(r));
                    })
                }
            }));
            tasks = Some(inbox);
            // Spoken requests need the agent and a model to check them.
            if let Some(intent) = cfg.intent {
                intent_model = intent.name.clone();
                let (jobs, job_rx) = unbounded();
                let system = invocation::system_prompt(&cfg.assistant_name);
                let msg_tx = msg_tx.clone();
                // Joined after the merger, which holds its queue: a local model
                // must be let go before the process ends (see `run_intent`).
                intent_thread =
                    Some(spawn("pam-intent", move || run_intent(intent.model, system, job_rx, msg_tx)));
                checks = Some(jobs);
            }
        }
        let system = scribe::system_prompt(&cfg.assistant_name);
        // Not joined: the merger waits for its last reply before it ends.
        let jobs = cfg.scribe.map(|s| {
            let (jobs, job_rx) = unbounded();
            spawn("pam-scribe", move || run_scribe(s.model, system, job_rx, msg_tx));
            jobs
        });
        let path = folder.path().to_path_buf();
        let merger = thread::Builder::new()
            .name("pam-merge".into())
            .spawn({
                let hub = hub.clone();
                let mut merger = Merger::new(folder, cfg.assistant_name, hub);
                merger.info = info.clone();
                merger.clock = clock;
                merger.coming = coming;
                merger.usage = usage;
                let unpriced = |m: Option<&models::ModelInfo>| m.is_some_and(|m| m.catalogued == Some(false));
                merger.unpriced = [
                    unpriced(cfg.scribe_info.as_ref()),
                    unpriced(cfg.intent_info.as_ref()),
                    unpriced(agent.as_ref()),
                ];
                merger.jobs = jobs;
                merger.agent = agent;
                merger.tasks_tx = tasks;
                // Without an intent check, "Pam, option two" and "Pam, cancel" still count.
                merger.invocations = merger.tasks_tx.as_ref().map(|_| match &checks {
                    Some(_) => Invocations::new(&merger.assistant),
                    None => Invocations::local(&merger.assistant),
                });
                merger.checks = checks;
                merger.intent_model = intent_model;
                merger.cancel_window = cfg.cancel_window;
                move || merger.run(msg_rx)
            })
            .context("starting the merger thread")?;
        Ok(Meeting {
            folder: path,
            hub,
            stop,
            commands,
            workers,
            merger: Some(merger),
            intent: intent_thread,
            info,
        })
    }

    pub fn folder(&self) -> &std::path::Path {
        &self.folder
    }

    /// A stream of events. A late subscriber first gets every final line,
    /// speaker, identity op, undo and error so far, and the latest clock,
    /// notes and usage.
    pub fn subscribe(&self) -> Receiver<CoreEvent> {
        let (tx, rx) = unbounded();
        let mut hub = self.hub.lock().unwrap();
        for e in hub.history.iter().chain(&hub.clock).chain(&hub.notes).chain(&hub.usage) {
            let _ = tx.send(e.clone());
        }
        hub.subscribers.push(tx);
        rx
    }

    /// Asks the core to do something, and waits until it is done and saved.
    /// The events it causes reach subscribers before this returns.
    pub fn send(&self, cmd: Command) -> Result<()> {
        let (reply, done) = crossbeam_channel::bounded(1);
        let sent = self.commands.as_ref().is_some_and(|c| c.send(Msg::Command(cmd, reply)).is_ok());
        if !sent {
            bail!("the meeting has stopped");
        }
        done.recv().map_err(|_| anyhow!("the meeting has stopped"))?
    }

    /// Stops listening, makes every line final and closes the folder.
    pub fn stop(mut self) -> Result<StopReport> {
        self.shutdown()
    }

    /// Stops and joins every thread, then marks `meeting.json` stopped, even
    /// when a thread failed. Called once, by `stop` or by `drop`.
    fn shutdown(&mut self) -> Result<StopReport> {
        self.stop.store(true, Ordering::Relaxed);
        let mut failed = None;
        for w in self.workers.drain(..) {
            if w.join().is_err() {
                failed.get_or_insert_with(|| anyhow!("a meeting thread panicked"));
            }
        }
        // Every worker has sent its last message: the merger finishes the notes and ends.
        if let Some(c) = self.commands.take() {
            let _ = c.send(Msg::Stop);
        }
        let merged = match self.merger.take() {
            Some(m) => m.join().map_err(|_| anyhow!("the merger thread panicked")).and_then(|r| r),
            None => Err(anyhow!("the meeting was already stopped")),
        };
        // The merger has closed the checks' queue: the checks still out finish.
        if let Some(intent) = self.intent.take()
            && intent.join().is_err()
        {
            failed.get_or_insert_with(|| anyhow!("the intent check's thread panicked"));
        }
        let (lines, duration, notes_failure, usage) = match merged {
            Ok(Merged { lines, heard, failure, notes_failure, usage }) => {
                if let Some(e) = failure {
                    failed.get_or_insert_with(|| anyhow!("{e}"));
                }
                (lines, Some(heard), notes_failure, Some(usage))
            }
            Err(e) => {
                failed.get_or_insert(e);
                (0, None, None, None)
            }
        };
        let usage = final_usage(usage, &self.folder, &self.info.usage);
        let info = MeetingInfo {
            status: "stopped".into(),
            stopped_at: Some(timestamp(Local::now())),
            duration,
            usage,
            ..self.info.clone()
        };
        let written = write_info(&self.folder, &info);
        // context.md shows the time elapsed, so it is rendered once meeting.json is final.
        let context = EventLog::lock(&self.folder).and_then(|log| log.write_context());
        if let Some(e) = failed {
            return Err(e);
        }
        written?;
        Ok(StopReport {
            folder: self.folder.clone(),
            lines,
            duration: duration.unwrap_or(0),
            notes_failure,
            context_failure: context.err().map(|e| format!("{e:#}")),
        })
    }
}

/// A meeting dropped without `stop` (an early return, a panic) still closes its folder.
impl Drop for Meeting {
    fn drop(&mut self) {
        if self.merger.is_some() {
            let _ = self.shutdown();
        }
    }
}

/// The `usage` that `meeting.json` in `dir` holds now.
fn read_usage(dir: &std::path::Path) -> Option<Usage> {
    let bytes = std::fs::read(dir.join("meeting.json")).ok()?;
    Some(serde_json::from_slice::<MeetingInfo>(&bytes).ok()?.usage)
}

/// The cost `meeting.json` keeps at Stop: the merger's final count. A merger
/// that failed stopped counting, so the last count it wrote, else the one
/// at the start, may be low.
fn final_usage(merged: Option<Usage>, dir: &std::path::Path, start: &Usage) -> Usage {
    merged.unwrap_or_else(|| {
        let mut kept = read_usage(dir).unwrap_or_else(|| start.clone());
        kept.lost();
        kept
    })
}

fn spawn(name: &str, f: impl FnOnce() + Send + 'static) -> JoinHandle<()> {
    thread::Builder::new().name(name.into()).spawn(f).expect("starting a meeting thread")
}

/// Reads the input in chunks, records it, and fans each chunk out to both backends.
fn feed(
    mut source: Source,
    mut recorder: Option<Recorder>,
    realtime: bool,
    stop: Arc<AtomicBool>,
    outs: Vec<Sender<Arc<[f32]>>>,
    msg: Sender<Msg>,
) {
    let failed = |e: anyhow::Error| {
        let _ = msg.send(Msg::Failed(format!("{e:#}")));
    };
    if let Err(e) = source.start() {
        failed(e);
        let _ = msg.send(Msg::Clock(0));
        return;
    }
    let pacer = Pacer::start(1.0);
    let mut meter = audio::Meter::default();
    let mut pos: Ts = 0;
    let mut second = 0;
    while !stop.load(Ordering::Relaxed) {
        let chunk = match source.read(CHUNK, &stop) {
            Ok(chunk) if chunk.is_empty() => break,
            Ok(chunk) => chunk,
            Err(e) => {
                failed(e.context("reading the input"));
                break;
            }
        };
        for w in source.warnings() {
            let _ = msg.send(Msg::Warning(w));
        }
        // A recording that fails stops recording, not the meeting.
        if let Some(Err(e)) = recorder.as_mut().map(|r| r.write(&chunk)) {
            failed(e.context("recording audio.wav"));
            recorder = None;
        }
        let chunk: Arc<[f32]> = chunk.into();
        pos += chunk.len() as Ts;
        for out in &outs {
            let _ = out.send(chunk.clone());
        }
        if source.is_audio() {
            let _ = msg.send(Msg::Level(pos, meter.level(&chunk)));
        }
        if pos / SECOND != second {
            second = pos / SECOND;
            let _ = msg.send(Msg::Clock(pos));
            if let Some(Err(e)) = recorder.as_mut().map(Recorder::flush) {
                failed(e.context("recording audio.wav"));
                recorder = None;
            }
        }
        if realtime {
            pacer.wait_until(pos);
        }
    }
    if let Some(Err(e)) = recorder.map(Recorder::finish) {
        failed(e.context("recording audio.wav"));
    }
    let _ = msg.send(Msg::Clock(pos));
}

fn run_transcriber(mut t: Box<dyn Transcriber>, audio: Receiver<Arc<[f32]>>, msg: Sender<Msg>) {
    let warn = |t: &mut Box<dyn Transcriber>| {
        for w in t.warnings() {
            let _ = msg.send(Msg::Warning(format!("transcriber: {w}")));
        }
    };
    let mut heard: Ts = 0;
    for chunk in audio {
        heard += chunk.len() as Ts;
        match t.push(&chunk) {
            Ok(u) => {
                warn(&mut t);
                let _ = msg.send(Msg::Words(u, heard));
            }
            Err(e) => {
                let _ = msg.send(Msg::Failed(format!("transcriber: {e:#}")));
                break;
            }
        }
    }
    let rest = t.finish().unwrap_or_else(|e| {
        let _ = msg.send(Msg::Failed(format!("transcriber: {e:#}")));
        Vec::new()
    });
    warn(&mut t);
    let _ = msg.send(Msg::WordsDone(rest));
}

fn run_tracker(mut k: Box<dyn SpeakerTracker>, audio: Receiver<Arc<[f32]>>, msg: Sender<Msg>) {
    let mut heard: Ts = 0;
    for chunk in audio {
        heard += chunk.len() as Ts;
        match k.push(&chunk) {
            Ok(u) => {
                let _ = msg.send(Msg::Speakers(u, heard));
            }
            Err(e) => {
                let _ = msg.send(Msg::Failed(format!("speaker tracker: {e:#}")));
                break;
            }
        }
    }
    let rest = k.finish().unwrap_or_else(|e| {
        let _ = msg.send(Msg::Failed(format!("speaker tracker: {e:#}")));
        Vec::new()
    });
    let _ = msg.send(Msg::SpeakersDone(rest));
}

/// Asks the merger to look for outside changes until the meeting stops.
fn poll(stop: Arc<AtomicBool>, msg: Sender<Msg>) {
    while !stop.load(Ordering::Relaxed) {
        thread::sleep(POLL);
        if msg.send(Msg::Poll).is_err() {
            break;
        }
    }
}

/// Runs each job past the model and hands the reply to the merger.
fn run_scribe(model: Arc<dyn llm::Model>, system: String, jobs: Receiver<Job>, msg: Sender<Msg>) {
    for job in jobs {
        let done = model.complete(&Request::new(&system, &job.prompt));
        let reply = done.reply.and_then(|r| scribe::parse(&r));
        if msg.send(Msg::Scribed { through: job.through, reply, used: done.used }).is_err() {
            break;
        }
    }
}

/// Runs each intent check past the model, each on a thread of its own, and
/// hands the verdict to the merger. A request is checked again at each
/// utterance final, and while a question is up every line is checked, so
/// a slow check must not hold up the next: a reply queued behind talk could miss
/// the countdown's wait. A local model takes one check at a time,
/// about a second each, so there they wait their turn.
///
/// Once the queue closes, it waits for the checks still out, whose replies go
/// nowhere, so that the model is let go before the meeting ends: a local
/// model's memory must be freed before the process exits, or ggml aborts it.
fn run_intent(model: Arc<dyn llm::Model>, system: String, checks: Receiver<Check>, msg: Sender<Msg>) {
    let system: Arc<str> = system.into();
    let mut out: Vec<JoinHandle<()>> = Vec::new();
    for check in checks {
        out.retain(|c| !c.is_finished());
        let (model, system, msg) = (model.clone(), system.clone(), msg.clone());
        out.push(spawn("pam-intent-check", move || {
            let sent = Instant::now();
            let done = model.complete(&Request::new(&system, &check.prompt).with_grammar(&check.grammar));
            let took = sent.elapsed();
            let reply = done.reply.as_ref().ok().cloned();
            let verdict = done.reply.and_then(|r| invocation::parse(&r));
            let _ = msg.send(Msg::Intent { check: check.id, reply, verdict, used: done.used, took });
        }));
    }
    for c in out {
        let _ = c.join();
    }
}

/// What the merger hands back once both backends are done.
struct Merged {
    lines: u32,
    heard: Ts,
    /// The first backend failure; the lines before it are still in the folder.
    failure: Option<String>,
    /// Why the notes leave out the last lines, if they do.
    notes_failure: Option<String>,
    /// The meeting's model calls, all counted.
    usage: Usage,
}

/// The merger thread's state: the pipeline, who is who, the lines so far,
/// and how far the scribe has got.
struct Merger {
    folder: MeetingFolder,
    assistant: String,
    hub: Arc<Mutex<Hub>>,
    p: Pipeline,
    identity: Identity,
    /// Every final line, for `transcript.md` and the scribe.
    lines: Vec<Line>,
    now: Ts,
    /// `now`, for the agent thread, which reads it as a prompt goes out.
    clock: Arc<AtomicU64>,
    /// The lines on their way to being final, for the agent thread too.
    coming: Arc<Mutex<Vec<brief::Coming>>>,
    /// How much audio the transcriber has been given: a finer clock than
    /// `now`, which moves once a second.
    heard: Ts,
    /// Whether speech is on its way to becoming final lines (the preview isn't empty).
    speaking: bool,
    /// The input has ended: every line is final.
    ended: bool,
    /// Into the scribe thread; `None` when there is no scribe.
    jobs: Option<Sender<Job>>,
    /// The scribe has read `lines[..scribed]`.
    scribed: usize,
    /// A run is out for `lines[..n]`.
    in_flight: Option<usize>,
    /// After a failed run, the next waits until then.
    retry_at: Option<Instant>,
    /// Why the last run failed, until one succeeds.
    last_failure: Option<String>,
    summary: Summary,
    /// The size of `events.jsonl` when the notes were last sent, to notice `pam note`.
    events_seen: u64,
    /// The agent as checked at the start; `None` without one.
    agent: Option<AgentInfo>,
    /// Into the agent thread; `None` when the agent is unavailable.
    tasks_tx: Option<Sender<Inbox>>,
    /// Every task so far, oldest first.
    tasks: Vec<Tracked>,
    /// Spoken requests; `None` without an agent. Without an intent check it
    /// only matches cancels and answers.
    invocations: Option<Invocations>,
    /// Into the intent thread.
    checks: Option<Sender<Check>>,
    /// Intent checks sent that haven't come back: when each was sent, and
    /// what it read, for `intent.jsonl`.
    checking: Vec<Sent>,
    /// The intent check's model, as `meeting.json` names it.
    intent_model: String,
    /// Spoken requests acted on before their line was final.
    asks: Vec<Ask>,
    cancel_window: Ts,
    /// `meeting.json` as written at the start, rewritten with `usage`.
    info: MeetingInfo,
    /// The meeting's model calls so far.
    usage: Usage,
    /// The scribe's, the intent check's and the agent's models are missing
    /// from pi's catalog, so pi's cost for their calls is another model's.
    unpriced: [bool; 3],
}

/// An intent check that is out, sent at `at` in meeting time, as far as
/// the transcriber had heard.
struct Sent {
    id: u64,
    at: Ts,
    /// When it was sent, by the wall clock: what a check still out at the
    /// stop had taken.
    since: Instant,
    /// Where the words checked start and end.
    t0: Ts,
    t1: Ts,
    said: String,
    prompt: String,
}

/// A spoken request acted on before its line was final: the task it made or
/// corrected, and the `request` events it still owes, which name that line.
struct Ask {
    ask: u64,
    task: TaskId,
    /// It made the task, so the line is the task's too.
    made: bool,
    t0: Ts,
    spk: Option<SpkId>,
    /// The summaries of its `request` events still to save.
    unrecorded: Vec<String>,
    /// Its line, once final: what continues it later is saved with it at once.
    line: Option<Line>,
}

/// A task, and how much of its `messages.jsonl` was sent.
struct Tracked {
    state: TaskState,
    read: u64,
    /// How long `messages.jsonl` was when its latest prompt went out, as the
    /// agent thread measured it: the agent said something for that prompt if
    /// it has grown since.
    prompted: u64,
    /// When its cancel window closes. Until then what it says, and how it
    /// ended, are held back, so a cancelled task never shows an answer.
    window: Option<Ts>,
    /// How its run ended, and when, while the window is still open.
    held: Option<(Outcome, Ts)>,
    /// It is running, but its prompt hasn't gone out yet: when it does, it
    /// has started (again, for a correction, if `Some(true)`).
    unannounced: Option<bool>,
    /// The room cancelled it; whatever it says from now on is dropped.
    cancelling: bool,
    /// How much of its `questions.jsonl` was read.
    asked_read: u64,
    /// How much of its `updates.jsonl` was applied.
    updates_read: u64,
    /// The question its agent is asking the room, if any.
    asking: Option<Asking>,
    /// What its `Invocation` said, and its deadline, for a spoken request:
    /// sent again when its line is final.
    invoked: Option<(String, Ts)>,
    /// How its questions ended, by number, where saving that failed: each
    /// is tried again at every look until it saves. A question on the stage
    /// stays open meanwhile, and takes no other answer.
    ending: Vec<(u32, End)>,
    /// Questions asked since, oldest first. The next takes the stage once
    /// the one there has ended and that end is saved.
    backlog: VecDeque<Question>,
}

/// A question a task's agent asks the room (`pam task ask`).
struct Asking {
    question: Question,
    /// When its countdown ends; `None` while the task's cancel window holds it.
    deadline: Option<Instant>,
    /// Once the deadline has passed, the meeting time it stands for: a
    /// reply begun before it is still waited for.
    cut: Option<Ts>,
}

impl Tracked {
    fn new(state: TaskState, window: Option<Ts>) -> Tracked {
        let running = state.status == TaskStatus::Running;
        Tracked {
            state,
            read: 0,
            prompted: 0,
            window,
            held: None,
            unannounced: running.then_some(false),
            cancelling: false,
            asked_read: 0,
            updates_read: 0,
            asking: None,
            invoked: None,
            ending: Vec::new(),
            backlog: VecDeque::new(),
        }
    }

    /// Whether the room can still cancel it: it hasn't finished, or its
    /// window is open.
    fn cancellable(&self) -> bool {
        !self.cancelling && (self.state.status.is_open() || self.window.is_some())
    }

    /// Its `TaskUpdated`, once `task.json` is saved.
    fn saved(&self) -> Vec<CoreEvent> {
        vec![CoreEvent::TaskUpdated(self.state.clone())]
    }
}

impl Merger {
    fn new(folder: MeetingFolder, assistant: String, hub: Arc<Mutex<Hub>>) -> Merger {
        Merger {
            folder,
            assistant,
            hub,
            p: Pipeline::new(),
            identity: Identity::new(),
            lines: Vec::new(),
            now: 0,
            clock: Arc::default(),
            coming: Arc::default(),
            heard: 0,
            speaking: false,
            ended: false,
            jobs: None,
            scribed: 0,
            in_flight: None,
            retry_at: None,
            last_failure: None,
            summary: Summary::default(),
            events_seen: 0,
            agent: None,
            tasks_tx: None,
            tasks: Vec::new(),
            invocations: None,
            checks: None,
            checking: Vec::new(),
            intent_model: String::new(),
            asks: Vec::new(),
            cancel_window: invocation::CANCEL_WINDOW,
            info: MeetingInfo::default(),
            usage: Usage::default(),
            unpriced: [false; 3],
        }
    }

    /// Runs the pipeline, writes the folder, then tells subscribers. After the
    /// input ends, keeps taking commands until the meeting stops, then lets
    /// the scribe read the last lines.
    fn run(mut self, msgs: Receiver<Msg>) -> Result<Merged> {
        let mut failure = None;
        let mut words_done = None;
        let mut spans_done = None;
        for msg in &msgs {
            let events = match msg {
                Msg::Clock(t) => {
                    self.now = t;
                    self.clock.store(t, Ordering::Relaxed);
                    let actions = self.invocations.as_mut().map(|i| i.tick(t)).unwrap_or_default();
                    let mut events = vec![CoreEvent::Clock { now: t }];
                    events.extend(self.act(actions));
                    events
                }
                Msg::Level(at, level) => self.p.level(at, level),
                Msg::Words(u, heard) => {
                    self.heard = self.heard.max(heard);
                    let utterance_end = !u.finalized.is_empty();
                    let events = self.p.words(u, heard);
                    self.publish(events)?;
                    self.share_coming();
                    // The transcriber ended an utterance: check what it heard.
                    match (utterance_end, self.invocations.as_mut()) {
                        (true, Some(i)) => {
                            let actions = i.heard(&self.p.speech(), self.now);
                            self.act(actions)
                        }
                        _ => Vec::new(),
                    }
                }
                Msg::Speakers(u, heard) => {
                    let events = self.p.speakers(u, heard);
                    self.publish(events)?;
                    self.share_coming();
                    Vec::new()
                }
                Msg::WordsDone(w) => {
                    words_done = Some(w);
                    Vec::new()
                }
                Msg::SpeakersDone(s) => {
                    spans_done = Some(s);
                    Vec::new()
                }
                Msg::Failed(e) => {
                    failure.get_or_insert_with(|| e.clone());
                    vec![CoreEvent::Error(e)]
                }
                Msg::Warning(e) => vec![CoreEvent::Error(e)],
                Msg::Command(cmd, reply) => {
                    let (done, saved) = match self.command(cmd) {
                        Ok(events) => (Ok(()), self.publish(events)),
                        Err(e) => (Err(e), Ok(())),
                    };
                    let _ = reply.send(match &saved {
                        Ok(()) => done,
                        Err(e) => Err(anyhow!("{e:#}")),
                    });
                    saved?;
                    Vec::new()
                }
                Msg::Scribed { through, reply, used } => {
                    let used = self.spend(User::Scribe, used);
                    self.publish(used)?;
                    if let Err(e) = self.scribed(through, reply) {
                        // A source file wasn't written: say so, and stop.
                        self.hub.lock().unwrap().send(CoreEvent::Error(format!("{e:#}")));
                        return Err(e);
                    }
                    Vec::new()
                }
                Msg::Agent(Report::Started(id)) => self.task_started(id),
                Msg::Agent(Report::Prompted(id, lengths)) => {
                    if let Some(t) = self.tasks.iter_mut().find(|t| t.state.id == id) {
                        t.prompted = lengths.messages;
                    }
                    let mut events = self.prompted(id, lengths.updates);
                    // A new prompt: the run before it was aborted, and its question with it.
                    events.extend(self.close_question(id, End::Dropped("the run was aborted".into())));
                    events
                }
                Msg::Agent(Report::Used(call)) => self.spend(User::Agent, Used::Call(call)),
                Msg::Agent(Report::Lost) => self.spend(User::Agent, Used::Unknown),
                Msg::Agent(Report::Ended(id, outcome)) => {
                    let mut events = self.close_question(id, End::Dropped("the run ended".into()));
                    events.extend(self.task_ended(id, outcome, self.now));
                    events
                }
                Msg::Intent { check, reply, verdict, used, took } => {
                    let used = self.spend(User::Intent, used);
                    self.publish(used)?;
                    self.log_check(check, reply, &verdict, took);
                    let now = self.now;
                    let actions = match (self.invocations.as_mut(), verdict) {
                        (Some(i), Ok(v)) => i.verdict(check, v, now),
                        (Some(i), Err(e)) => {
                            self.hub.lock().unwrap().send(CoreEvent::Error(format!("intent check: {e:#}")));
                            i.failed(check, now)
                        }
                        (None, _) => Vec::new(),
                    };
                    self.act(actions)
                }
                Msg::Poll => {
                    self.look_for_notes();
                    // Reports first, as at a run's end: a result attached before
                    // its verdict was said is known when the verdict shows.
                    let mut events = self.look_for_updates();
                    events.extend(self.look_for_messages());
                    events.extend(self.look_for_questions());
                    events
                }
                Msg::Stop => {
                    // Checks still out finish on their own thread, their replies
                    // going nowhere; only the scribe's reply is waited for.
                    self.checks = None;
                    self.record_unattached();
                    if !self.checking.is_empty() {
                        // Their replies go nowhere, and so does what they used.
                        // They are logged all the same, as stopped.
                        let out: Vec<(u64, Duration)> =
                            self.checking.iter().map(|c| (c.id, c.since.elapsed())).collect();
                        for (id, took) in out {
                            self.log_check(
                                id,
                                None,
                                &Err(anyhow!("the meeting stopped before its reply")),
                                took,
                            );
                        }
                        let used = self.spend(User::Intent, Used::Unknown);
                        self.publish(used)?;
                    }
                    break;
                }
            };
            self.publish(events)?;
            if words_done.is_some() && spans_done.is_some() {
                let mut events = self.p.finish(words_done.take().unwrap(), spans_done.take().unwrap());
                events.push(CoreEvent::InputEnded { now: self.now });
                self.publish(events)?;
                self.share_coming();
                self.ended = true;
                let actions = self.invocations.as_mut().map(|i| i.end(self.now)).unwrap_or_default();
                let events = self.act(actions);
                self.publish(events)?;
            }
            let closed = self.close_windows();
            self.publish(closed)?;
            self.scribe_if_due();
        }
        let notes_failure = self.finish(&msgs)?;
        Ok(Merged { lines: self.p.line_count(), heard: self.now, failure, notes_failure, usage: self.usage })
    }

    /// The end of the meeting, in order: the run in flight comes back, one
    /// last run reads the lines left, and `notes.md` is rendered once more,
    /// which repairs an earlier failed render. Nobody reads events any more,
    /// so what went wrong is returned for the stop report. Every wait is for
    /// the reply to a job that was sent, so this always ends.
    fn finish(&mut self, msgs: &Receiver<Msg>) -> Result<Option<String>> {
        self.await_reply(msgs)?;
        if self.jobs.is_some() && self.scribed < self.lines.len() {
            match self.send_job() {
                Ok(()) => self.await_reply(msgs)?,
                Err(e) => self.failed(format!("{e:#}")),
            }
        }
        let mut problems = Vec::new();
        if let (Some(_), [first, .., last] | [first @ last]) = (&self.jobs, &self.lines[self.scribed..]) {
            let why = self.last_failure.as_deref().unwrap_or("the scribe didn't finish");
            problems.push(format!("notes.md leaves out {}..{}: {why}", first.id, last.id));
        }
        let rendered =
            EventLog::lock(self.folder.path()).and_then(|log| self.write_notes(&log, &log.events()?));
        if let Err(e) = rendered {
            problems.push(format!("notes.md couldn't be written: {e:#}"));
        }
        Ok((!problems.is_empty()).then(|| problems.join("; ")))
    }

    /// Waits for the reply to the run in flight, if there is one, and saves it.
    fn await_reply(&mut self, msgs: &Receiver<Msg>) -> Result<()> {
        if self.in_flight.is_none() {
            return Ok(());
        }
        // Every other sender is gone by now; only the scribe thread can still send.
        for msg in msgs {
            if let Msg::Scribed { through, reply, used } = msg {
                // Nobody reads events any more; meeting.json is what counts.
                let _ = self.spend(User::Scribe, used);
                return self.scribed(through, reply);
            }
        }
        self.in_flight = None;
        self.last_failure.get_or_insert_with(|| "the scribe stopped".into());
        self.usage.lost();
        Ok(())
    }

    /// Counts what a run used, rewrites `meeting.json` with
    /// the new totals, and returns the `Usage` to send. The totals live here,
    /// so a failed write is only reported: the next one repairs it.
    fn spend(&mut self, who: User, used: Used) -> Vec<CoreEvent> {
        let unpriced = match who {
            User::Scribe => self.unpriced[0],
            User::Intent => self.unpriced[1],
            User::Agent => self.unpriced[2],
        };
        match used {
            Used::Call(call) if unpriced => self.usage.add(who, &call.unpriced()),
            Used::Call(call) => self.usage.add(who, &call),
            Used::Nothing => {}
            Used::Unknown => self.usage.lost(),
        }
        let info = MeetingInfo { usage: self.usage.clone(), ..self.info.clone() };
        let mut events = Vec::new();
        if let Err(e) = self.folder.write_info(&info) {
            events.push(CoreEvent::Error(format!("{e:#}")));
        }
        events.push(CoreEvent::Usage(Box::new(self.usage.clone())));
        events
    }

    fn command(&mut self, cmd: Command) -> Result<Vec<CoreEvent>> {
        let at = timestamp(Local::now());
        let event = match cmd {
            Command::RenameSpeaker { spk, name } => self.identity.rename(spk, &name, at),
            Command::MergeSpeakers { from, into } => self.identity.merge(from, into, at),
            Command::MoveLine { line, to } => {
                let Ok(i) = self.lines.binary_search_by_key(&line, |l| l.id) else {
                    match (self.lines.first(), self.lines.last()) {
                        (Some(a), Some(b)) => bail!("no line {line}: lines run {}..{}", a.id, b.id),
                        _ => bail!("no line {line}: there are no final lines yet"),
                    }
                };
                self.identity.move_line(&self.lines[i], to, at)
            }
            Command::Undo(undo) => self.identity.undo(undo),
            Command::Ask(request) => return self.ask(request),
            Command::Answer { task, q, answer } => {
                // An option comes from the keys, words from the composer.
                let (answer, by) = match answer {
                    Answer::Option(n) => (Answer::Option(n), By::Key),
                    Answer::Text(text) => (Answer::Text(text.trim().to_string()), By::Typed),
                };
                return self.answer(task, q, answer, by, None);
            }
            Command::CancelTask(id) => {
                let Some(t) = self.tasks.iter().find(|t| t.state.id == id) else {
                    bail!("no task {id}");
                };
                if !t.cancellable() {
                    bail!("{id} can't be cancelled: it is {}", status_word(t.state.status, t.cancelling));
                }
                return Ok(self.cancel(id));
            }
        };
        event.map(|e| vec![e])
    }

    /// Makes a typed request a task, and hands it to the agent. With a task
    /// still open, it waits its turn.
    fn ask(&mut self, request: String) -> Result<Vec<CoreEvent>> {
        let request = request.trim();
        if request.is_empty() {
            bail!("nothing to ask: type a request first");
        }
        let Some(info) = &self.agent else { bail!("this meeting has no agent") };
        if !info.usable() {
            bail!("the agent is unavailable: {}", info.reason.as_deref().unwrap_or("unknown reason"));
        }
        let state = self.new_task(request, None, self.now, None)?;
        let mut events = vec![CoreEvent::TaskUpdated(state.clone())];
        events.extend(self.record_request(state.id, state.asked_at, None, None, request));
        Ok(events)
    }

    /// Makes a task of `request`, asked at `asked_at`, saves it and hands it
    /// to the agent. A spoken request names who asked, `by`, and gets a
    /// cancel window from now; its `line` is `None` until that is final.
    fn new_task(
        &mut self,
        request: &str,
        line: Option<&Line>,
        asked_at: Ts,
        by: Option<(&str, Option<SpkId>)>,
    ) -> Result<TaskState> {
        let dir = self.folder.path();
        let busy = self.tasks.iter().any(|t| t.state.status.is_open());
        let status = if busy { TaskStatus::Queued } else { TaskStatus::Running };
        let via = if by.is_some() { Via::Voice } else { Via::Typed };
        let state = TaskState {
            line: line.map(|l| l.id),
            spk: line.map_or(by.and_then(|(_, spk)| spk), |l| self.identity.speaker_of(l)),
            ..TaskState::new(task::next_id(dir)?, request, asked_at, via, status)
        };
        let by = by.map(|(who, _)| who);
        let prompt = agent::request_prompt(state.id, &state.request, asked_at, by);
        task::create(dir, &state)?;
        let job = agent::Job { id: state.id, asked_at, prompt };
        let sent = self.tasks_tx.as_ref().is_some_and(|t| t.send(Inbox::Job(job)).is_ok());
        if !sent {
            let failed = TaskState {
                status: TaskStatus::Failed,
                error: Some("the agent has stopped".into()),
                ended_at: Some(self.now),
                ..state
            };
            task::write(dir, &failed)?;
            bail!("the agent has stopped");
        }
        let window = by.map(|_| self.now + self.cancel_window);
        self.tasks.push(Tracked::new(state.clone(), window));
        Ok(state)
    }

    /// Saves a `request` event for task `id`: what was asked at `at`,
    /// by whom, and from which line, and returns the notes to send. Who
    /// asked is `line`'s speaker, or `spk` while the line isn't final; a
    /// typed request has neither. A failure is only reported: the task goes on.
    fn record_request(
        &mut self,
        id: TaskId,
        at: Ts,
        line: Option<&Line>,
        spk: Option<SpkId>,
        text: &str,
    ) -> Vec<CoreEvent> {
        let by = match line {
            Some(l) => self.identity.speaker_of(l),
            None => spk,
        };
        let event = Event {
            id: crate::event::EventId(0),
            kind: "request".into(),
            t: at,
            by: by.into_iter().collect(),
            text: text.to_string(),
            refs: line.map(|l| l.id.to_string()).into_iter().collect(),
            task: Some(id.to_string()),
            supersedes: None,
            name: None,
            rename: None,
        };
        self.save_event(event)
    }

    /// Saves a `task_update` event for task `id`, saying what its status,
    /// reached `at`, means, and returns the notes to send. Only a task that
    /// has started or ended has one: a queued task has nothing to say yet.
    /// A failure is only reported: the task goes on.
    fn record_update(&mut self, id: TaskId, at: Ts, again: bool) -> Vec<CoreEvent> {
        let Some(t) = self.tasks.iter().find(|t| t.state.id == id) else { return Vec::new() };
        let state = &t.state;
        let text = match state.status {
            TaskStatus::Queued => return Vec::new(),
            TaskStatus::Running if again => "started again, for a correction".to_string(),
            TaskStatus::Running => "started".to_string(),
            TaskStatus::Done => "done".to_string(),
            TaskStatus::Failed => format!("failed: {}", state.error.as_deref().unwrap_or("unknown reason")),
            TaskStatus::Stopped => "stopped: the meeting stopped first".to_string(),
            TaskStatus::Cancelled => "cancelled".to_string(),
        };
        let event = Event {
            id: crate::event::EventId(0),
            kind: "task_update".into(),
            t: at,
            by: Vec::new(),
            text,
            refs: state.line.map(|l| l.to_string()).into_iter().collect(),
            task: Some(id.to_string()),
            supersedes: None,
            name: None,
            rename: None,
        };
        self.save_event(event)
    }

    /// Appends `event` to `events.jsonl`, re-renders the notes and
    /// `context.md`, and returns the notes to send, or the failure.
    fn save_event(&mut self, event: Event) -> Vec<CoreEvent> {
        let saved = EventLog::lock(self.folder.path()).and_then(|mut log| {
            log.append(vec![event])?;
            self.notes_updated(&log)
        });
        saved.unwrap_or_else(|e| vec![CoreEvent::Error(format!("{e:#}"))])
    }

    /// The speech on its way to final lines, as `pam transcript`
    /// prints lines but with no ID: each line on its way that reaches past
    /// `from` and starts by `until`, its utterances from `until` on left
    /// out when `cut`.
    fn speech_lines(&self, from: Ts, until: Ts, cut: bool) -> Vec<String> {
        self.speech_coming(from, until, cut).into_iter().map(|c| c.text).collect()
    }

    /// As [`Merger::speech_lines`], with where each starts and ends.
    fn speech_coming(&self, from: Ts, until: Ts, cut: bool) -> Vec<brief::Coming> {
        let mut out = Vec::new();
        for s in self.p.speech() {
            let us: Vec<&Utterance> =
                s.utterances.iter().filter(|u| u.t1 > from && (!cut || u.t0 < until)).collect();
            if us.first().is_none_or(|u| u.t0 > until) {
                continue;
            }
            let text = us.iter().map(|u| u.text.as_str()).collect::<Vec<_>>().join(" ");
            out.push(brief::Coming {
                t0: us[0].t0,
                t1: us[us.len() - 1].t1,
                text: format!("{} {}: {text}", mmss(us[0].t0), self.identity.overlay().label(s.spk)),
            });
        }
        out
    }

    /// Hands the agent thread the lines on their way as they are now, once
    /// the lines made final with them are in the meeting folder: a prompt
    /// may go out at any time, and is put together from both.
    fn share_coming(&self) {
        *self.coming.lock().unwrap() = self.speech_coming(0, Ts::MAX, false);
    }

    /// A line as `pam transcript` prints it, corrections applied.
    fn transcript_line(&self, l: &Line) -> String {
        format!("{} {} {}: {}", l.id, mmss(l.t0), self.speaker_label(l), l.text)
    }

    /// Who said `l`, by name once known.
    fn speaker_label(&self, l: &Line) -> String {
        self.identity.overlay().label(self.identity.speaker_of(l))
    }

    /// Who said a spoken request: its line's speaker, or the tracker's
    /// speaker so far while its line isn't final.
    fn asker_label(&self, line: Option<&Line>, spk: Option<SpkId>) -> String {
        match line {
            Some(l) => self.speaker_label(l),
            None => self.identity.overlay().label(spk),
        }
    }

    /// Carries out what the spoken requests decided.
    fn act(&mut self, actions: Vec<Action>) -> Vec<CoreEvent> {
        let mut events = Vec::new();
        for a in actions {
            match a {
                Action::Check { id, said, spk, t0, t1, line, question, .. } => {
                    let prompt = self.check_prompt(&said, spk, t0, line, question.as_ref());
                    let grammar = invocation::grammar(question.as_ref());
                    let sent = self
                        .checks
                        .as_ref()
                        .is_some_and(|c| c.send(Check { id, prompt: prompt.clone(), grammar }).is_ok());
                    if sent {
                        let at = self.now.max(self.heard);
                        self.checking.push(Sent { id, at, since: Instant::now(), t0, t1, said, prompt });
                    } else {
                        let now = self.now;
                        let more = self.invocations.as_mut().map(|i| i.failed(id, now)).unwrap_or_default();
                        events.extend(self.act(more));
                    }
                }
                Action::Request { ask, t0, spk, line, text, verdict } => {
                    events.extend(self.spoken(ask, t0, spk, line.as_ref(), &text, verdict))
                }
                Action::Continue { ask, t0, spk, text, verdict } => {
                    events.extend(self.continued(ask, t0, spk, &text, verdict))
                }
                Action::Attach { ask, line } => events.extend(self.attach(ask, &line)),
                Action::Cancel { before } => {
                    let newest = self
                        .tasks
                        .iter()
                        .rev()
                        .find(|t| t.state.asked_at < before && t.cancellable())
                        .map(|t| t.state.id);
                    if let Some(id) = newest {
                        events.extend(self.cancel(id));
                    }
                }
                Action::Answer { answer, spk } => {
                    // Tasks run one at a time, so at most one question is on the stage.
                    let asking = self
                        .tasks
                        .iter()
                        .rev()
                        .find(|t| t.asking.as_ref().is_some_and(|a| a.deadline.is_some()))
                        .map(|t| t.state.id);
                    // An answer the question doesn't take answers nothing; one
                    // that can't be saved is reported and the question stays open.
                    if let Some(id) = asking
                        && self.answerable(id, None, &answer).is_ok()
                    {
                        events.extend(self.close_question(id, End::Answered { answer, by: By::Voice, spk }));
                    }
                }
            }
        }
        events
    }

    /// What the intent check reads: the lines just before, the latest
    /// request, the question on the stage, and the utterance. The lines
    /// before include speech not in a final line yet.
    fn check_prompt(
        &self,
        said: &str,
        spk: Option<SpkId>,
        t0: Ts,
        line: Option<UtteranceId>,
        question: Option<&Asked>,
    ) -> String {
        let mut before: Vec<String> = self
            .lines
            .iter()
            .filter(|l| l.t0 < t0 && Some(l.id) != line)
            .map(|l| self.transcript_line(l))
            .collect();
        before.extend(self.speech_lines(0, t0, true));
        let recent = &before[before.len().saturating_sub(4)..];
        let by = self.asker_label(line.and_then(|id| self.lines.iter().find(|l| l.id == id)), spk);
        // A request said just before may not be a task yet; it is the latest all the same.
        let asked = self.invocations.as_ref().and_then(|i| i.asked_before(t0));
        let latest = match asked {
            Some(text) => Some(("new request".to_string(), "being checked", text)),
            None => self.latest().map(|t| {
                let status = match (t.window, t.state.status) {
                    (Some(_), _) => "starting",
                    (None, TaskStatus::Queued) => "queued",
                    (None, _) => "answering",
                };
                (t.state.id.to_string(), status, t.state.request.as_str())
            }),
        };
        invocation::check_prompt(&invocation::CheckInput { recent, latest, question, by: &by, said })
    }

    /// Appends intent check `id` to `intent.jsonl`: what it read, what the
    /// model said, how long it took. Real meetings so grow the evaluation set.
    /// A failure is only reported.
    fn log_check(&mut self, id: u64, reply: Option<String>, verdict: &Result<Verdict>, took: Duration) {
        let Some(i) = self.checking.iter().position(|c| c.id == id) else { return };
        let sent = self.checking.remove(i);
        let mut record = serde_json::json!({
            "check": id,
            "sent_at": sent.at,
            "at": self.now.max(self.heard),
            "ms": took.as_millis() as u64,
            "model": self.intent_model,
            "t0": sent.t0,
            "t1": sent.t1,
            "said": sent.said,
            "prompt": sent.prompt,
            "reply": reply,
        });
        match verdict {
            Ok(v) => {
                let kind = match &v.kind {
                    Kind::Reply(Answer::Option(n)) => {
                        record["option"] = (*n).into();
                        "reply"
                    }
                    Kind::Reply(Answer::Text(text)) => {
                        record["text"] = text.clone().into();
                        "reply"
                    }
                    Kind::Request => "request",
                    Kind::Correction => "correction",
                    Kind::Unfinished => "unfinished",
                    Kind::None => "none",
                };
                record["kind"] = kind.into();
                record["summary"] = v.summary.clone().into();
            }
            Err(e) => record["error"] = format!("{e:#}").into(),
        }
        let saved = std::fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.folder.path().join("intent.jsonl"))
            .and_then(|mut f| crate::folder::append_to(&mut f, format!("{record}\n").as_bytes()));
        if let Err(e) = saved {
            self.hub.lock().unwrap().send(CoreEvent::Error(format!("writing intent.jsonl: {e:#}")));
        }
    }

    /// The task a correction would change: the newest that is still
    /// going, running, queued or in its cancel window. An answered one is
    /// left alone: its answer is in the session, so a new task follows it.
    fn latest(&self) -> Option<&Tracked> {
        self.tasks.iter().rev().find(|t| t.cancellable())
    }

    /// Spoken request `ask`, said by `spk` from `t0`, passed the intent
    /// check: a correction changes the latest task, if one is still going;
    /// anything else, and a correction with none going, becomes a task with
    /// a cancel window. With no final `line` yet, its `request` event waits
    /// for it.
    fn spoken(
        &mut self,
        ask: u64,
        t0: Ts,
        spk: Option<SpkId>,
        line: Option<&Line>,
        text: &str,
        verdict: Verdict,
    ) -> Vec<CoreEvent> {
        let summary = if verdict.summary.is_empty() { text.to_string() } else { verdict.summary };
        let by = self.asker_label(line, spk);
        if verdict.kind == Kind::Correction
            && let Some(id) = self.latest().map(|t| t.state.id)
        {
            let mut events = self.correct(id, &by, t0, text);
            events.extend(self.record_or_owe(ask, id, false, t0, line, spk, summary));
            return events;
        }
        let state = match self.new_task(text, line, t0, Some((&by, spk))) {
            Ok(state) => state,
            Err(e) => return vec![CoreEvent::Error(format!("{e:#}"))],
        };
        let deadline = self.tasks.last().and_then(|t| t.window).unwrap_or(self.now);
        if let Some(t) = self.tasks.last_mut() {
            t.invoked = Some((summary.clone(), deadline));
        }
        let mut events = vec![
            CoreEvent::TaskUpdated(state.clone()),
            CoreEvent::Invocation {
                task: state.id,
                line: line.map(|l| l.id),
                spk: line.map_or(spk, |l| l.spk),
                summary: summary.clone(),
                cancel_deadline: deadline,
            },
        ];
        events.extend(self.record_or_owe(ask, state.id, true, t0, line, spk, summary));
        events
    }

    /// Saves the `request` event of `ask` for task `id`, or, while its line
    /// isn't final, keeps it until it is.
    #[allow(clippy::too_many_arguments)]
    fn record_or_owe(
        &mut self,
        ask: u64,
        id: TaskId,
        made: bool,
        t0: Ts,
        line: Option<&Line>,
        spk: Option<SpkId>,
        summary: String,
    ) -> Vec<CoreEvent> {
        let attached = self.asks.iter().find(|a| a.ask == ask).and_then(|a| a.line.clone());
        if let Some(line) = line.or(attached.as_ref()) {
            return self.record_request(id, t0, Some(line), spk, &summary);
        }
        match self.asks.iter_mut().find(|a| a.ask == ask) {
            Some(a) => a.unrecorded.push(summary),
            None => {
                self.asks.push(Ask { ask, task: id, made, t0, spk, unrecorded: vec![summary], line: None })
            }
        }
        Vec::new()
    }

    /// The speaker of `ask` went on after it started: all it asks now
    /// changes the task it made or corrected, while that is still going.
    /// Its line isn't final yet, so the `request` event waits too.
    fn continued(
        &mut self,
        ask: u64,
        t0: Ts,
        spk: Option<SpkId>,
        text: &str,
        verdict: Verdict,
    ) -> Vec<CoreEvent> {
        let Some(id) = self.asks.iter().find(|a| a.ask == ask).map(|a| a.task) else { return Vec::new() };
        if !self.tasks.iter().any(|t| t.state.id == id && t.cancellable()) {
            return Vec::new();
        }
        let summary = if verdict.summary.is_empty() { text.to_string() } else { verdict.summary };
        let by = self.asker_label(None, spk);
        let mut events = self.correct(id, &by, t0, text);
        events.extend(self.record_or_owe(ask, id, false, t0, None, spk, summary));
        events
    }

    /// The line of `ask` is final: the task it made gets it, and its
    /// `request` events are saved with it. It is kept for what may still
    /// continue it: a check of its last words can come back after its line.
    fn attach(&mut self, ask: u64, line: &Line) -> Vec<CoreEvent> {
        let Some(a) = self.asks.iter_mut().find(|a| a.ask == ask) else { return Vec::new() };
        a.line = Some(line.clone());
        let a = Ask { unrecorded: std::mem::take(&mut a.unrecorded), line: None, ..*a };
        let mut events = Vec::new();
        if a.made
            && let Some(t) = self.tasks.iter_mut().find(|t| t.state.id == a.task)
        {
            let state = TaskState { line: Some(line.id), ..t.state.clone() };
            match task::write(self.folder.path(), &state) {
                Ok(()) => {
                    t.state = state;
                    events.extend(t.saved());
                }
                Err(e) => events.push(CoreEvent::Error(format!("{e:#}"))),
            }
            if let Some((summary, deadline)) = t.invoked.clone() {
                events.push(CoreEvent::Invocation {
                    task: a.task,
                    line: Some(line.id),
                    spk: line.spk,
                    summary,
                    cancel_deadline: deadline,
                });
            }
        }
        for summary in a.unrecorded {
            events.extend(self.record_request(a.task, a.t0, Some(line), a.spk, &summary));
        }
        events
    }

    /// The meeting stops with requests whose line never came: their
    /// `request` events are saved without it. Nobody reads events any more.
    fn record_unattached(&mut self) {
        for a in std::mem::take(&mut self.asks) {
            for summary in a.unrecorded {
                let _ = self.record_request(a.task, a.t0, None, a.spk, &summary);
            }
        }
    }

    /// Tells the agent what the room changed about task `id`: `by` said
    /// `text` at `at`.
    fn correct(&mut self, id: TaskId, by: &str, at: Ts, text: &str) -> Vec<CoreEvent> {
        let prompt = agent::correction_prompt(by, at, text);
        if self.tasks_tx.as_ref().is_none_or(|t| t.send(Inbox::Correct(id, at, prompt)).is_err()) {
            return vec![CoreEvent::Error("the agent has stopped".into())];
        }
        let Some(t) = self.tasks.iter_mut().find(|t| t.state.id == id) else { return Vec::new() };
        let state = TaskState {
            corrections: [&t.state.corrections[..], &[text.to_string()]].concat(),
            ..t.state.clone()
        };
        match task::write(self.folder.path(), &state) {
            Ok(()) => {
                t.state = state;
                t.saved()
            }
            Err(e) => vec![CoreEvent::Error(format!("{e:#}"))],
        }
    }

    /// Cancels task `id`. Whatever it said and hasn't shown is dropped. A
    /// run that ended in its window is cancelled at once; otherwise the agent
    /// aborts it, and it counts as cancelled once pi confirms.
    fn cancel(&mut self, id: TaskId) -> Vec<CoreEvent> {
        // The abort stops whoever asked it.
        let mut events = self.close_question(id, End::Dropped("the task was cancelled".into()));
        let path = task::messages_file(self.folder.path(), id);
        let Some(t) = self.tasks.iter_mut().find(|t| t.state.id == id) else { return events };
        t.cancelling = true;
        t.window = None;
        t.read = std::fs::metadata(&path).map_or(t.read, |m| m.len());
        let ended = t.held.take().is_some();
        if ended || self.tasks_tx.as_ref().is_none_or(|tx| tx.send(Inbox::Cancel(id)).is_err()) {
            events.extend(self.set_status(id, TaskStatus::Cancelled, None, self.now));
        }
        events
    }

    /// Closes the cancel windows whose time is up, or all once the input
    /// ended, and shows what their tasks held back. While the input lasts,
    /// none closes while a cancel is being decided.
    fn close_windows(&mut self) -> Vec<CoreEvent> {
        if !self.ended && self.invocations.as_ref().is_some_and(Invocations::holding) {
            return Vec::new();
        }
        let (now, ended) = (self.now, self.ended);
        let due: Vec<TaskId> = self
            .tasks
            .iter_mut()
            .filter(|t| t.window.is_some_and(|w| ended || now >= w))
            .map(|t| {
                t.window = None;
                t.state.id
            })
            .collect();
        let mut events = Vec::new();
        for id in due {
            events.extend(self.updates(id));
            events.extend(self.messages(id));
            events.extend(self.show_question(id));
            let held = self.tasks.iter_mut().find(|t| t.state.id == id).and_then(|t| t.held.take());
            if let Some((outcome, at)) = held {
                events.extend(self.task_ended(id, outcome, at));
            }
        }
        events
    }

    fn task_started(&mut self, id: TaskId) -> Vec<CoreEvent> {
        if let Some(t) = self.tasks.iter_mut().find(|t| t.state.id == id) {
            // A run again, for a correction: how it ends starts afresh.
            t.held = None;
        }
        self.set_status(id, TaskStatus::Running, None, self.now)
    }

    /// Sends what the agent reported and said, then the task's new status.
    /// An agent that finished without `pam say` has its last reply shown
    /// instead. In its cancel window, all wait for the window to close. The
    /// run ended `at`.
    fn task_ended(&mut self, id: TaskId, outcome: Outcome, at: Ts) -> Vec<CoreEvent> {
        let Some(t) = self.tasks.iter_mut().find(|t| t.state.id == id) else { return Vec::new() };
        if t.cancelling {
            return self.set_status(id, TaskStatus::Cancelled, None, at);
        }
        if t.window.is_some() {
            t.held = Some((outcome, at));
            return Vec::new();
        }
        let mut events = self.updates(id);
        events.extend(self.messages(id));
        let (status, error) = match outcome {
            Outcome::Done { reply } => {
                // Said with `pam say` since its latest prompt went out?
                let size = messages_size(self.folder.path(), id);
                let said = self.tasks.iter().any(|t| t.state.id == id && size > t.prompted);
                if let Some(reply) = reply.filter(|_| !said) {
                    match task::say(self.folder.path(), id, &reply) {
                        Ok(_) => events.extend(self.messages(id)),
                        Err(e) => events.push(CoreEvent::Error(format!("{e:#}"))),
                    }
                }
                (TaskStatus::Done, None)
            }
            Outcome::Failed(why) => (TaskStatus::Failed, Some(why)),
            Outcome::Cancelled => (TaskStatus::Cancelled, None),
            Outcome::Stopped => (TaskStatus::Stopped, None),
        };
        events.extend(self.set_status(id, status, error, at));
        events
    }

    /// Saves a task's new status, reached `at`, in `task.json`, and returns
    /// its `TaskUpdated`, then the notes once its `task_update` event is
    /// saved. The core makes every change of status, so `task.json` and the
    /// events keep in step with the run. A failed save is sent as an error
    /// instead; the meeting goes on.
    fn set_status(
        &mut self,
        id: TaskId,
        status: TaskStatus,
        error: Option<String>,
        at: Ts,
    ) -> Vec<CoreEvent> {
        let Some(t) = self.tasks.iter_mut().find(|t| t.state.id == id) else { return Vec::new() };
        if t.state.status == status && t.state.error == error {
            return Vec::new();
        }
        let state =
            TaskState { status, error, ended_at: (!status.is_open()).then_some(at), ..t.state.clone() };
        let changed = state.status != t.state.status;
        match task::write(self.folder.path(), &state) {
            Ok(()) => {
                let again = t.state.ended_at.is_some();
                t.state = state;
                let mut events = t.saved();
                // A run starts once its prompt goes out (`prompted`); an end is logged now.
                match status {
                    TaskStatus::Running => t.unannounced = Some(again),
                    _ => t.unannounced = None,
                }
                if changed && status != TaskStatus::Running {
                    events.extend(self.record_update(id, at, false));
                }
                events
            }
            Err(e) => vec![CoreEvent::Error(format!("{e:#}"))],
        }
    }

    /// Task `id`'s prompt went out, when its `updates.jsonl` was `updates`
    /// bytes long. Every prompt starts the agent afresh, a correction's too:
    /// what it reported before is over, so its progress, action and result
    /// are cleared and those reports are never read; what it reported since is
    /// applied again, though a look may have applied it before this was
    /// heard. The first prompt for a run starts it: the first run sets
    /// `started_at`, and each is logged as a `task_update`. A prompt for a
    /// correction to a running task starts nothing.
    fn prompted(&mut self, id: TaskId, updates: u64) -> Vec<CoreEvent> {
        let mut events = self.prompt_went_out(id, updates);
        events.extend(self.updates(id));
        events
    }

    fn prompt_went_out(&mut self, id: TaskId, updates: u64) -> Vec<CoreEvent> {
        let now = self.now;
        let Some(t) = self.tasks.iter_mut().find(|t| t.state.id == id) else { return Vec::new() };
        if t.state.status != TaskStatus::Running {
            return Vec::new();
        }
        t.updates_read = updates;
        let mut state = TaskState { progress: None, action: None, result: None, ..t.state.clone() };
        let starts = t.unannounced;
        if starts.is_some() && state.started_at.is_none() {
            state.started_at = Some(now);
        }
        let mut events = Vec::new();
        if state != t.state {
            match task::write(self.folder.path(), &state) {
                Ok(()) => {
                    t.state = state;
                    events.extend(t.saved());
                }
                Err(e) => {
                    // The cleared report is kept and saved with the next change.
                    (t.state.progress, t.state.action, t.state.result) = (None, None, None);
                    events.push(CoreEvent::Error(format!("{e:#}")));
                    if t.state.started_at.is_none() {
                        // Not started as far as task.json says: the next prompt tries again.
                        return events;
                    }
                }
            }
        }
        let Some(again) = starts else { return events };
        t.unannounced = None;
        events.extend(self.record_update(id, now, again));
        events
    }

    /// What `pam say` wrote for any task since the last look.
    fn look_for_messages(&mut self) -> Vec<CoreEvent> {
        let ids: Vec<TaskId> = self.tasks.iter().map(|t| t.state.id).collect();
        ids.into_iter().flat_map(|id| self.messages(id)).collect()
    }

    /// What `pam task update` reported for any task since the last look.
    fn look_for_updates(&mut self) -> Vec<CoreEvent> {
        let ids: Vec<TaskId> = self.tasks.iter().map(|t| t.state.id).collect();
        ids.into_iter().flat_map(|id| self.updates(id)).collect()
    }

    /// What task `id`'s agent reported since the last look, saved in
    /// `task.json` and sent as one `TaskUpdated`: nothing while its cancel window is open, and
    /// only while it runs, so a cancelled task takes none. A failed save is
    /// sent as an error; the report stays with the task and is saved with
    /// its next change.
    fn updates(&mut self, id: TaskId) -> Vec<CoreEvent> {
        let path = task::updates_file(self.folder.path(), id);
        let Some(t) = self.tasks.iter_mut().find(|t| t.state.id == id) else { return Vec::new() };
        let size = std::fs::metadata(&path).map_or(0, |m| m.len());
        if t.window.is_some() || size == t.updates_read {
            return Vec::new();
        }
        if t.cancelling || t.state.status != TaskStatus::Running {
            t.updates_read = size;
            return Vec::new();
        }
        let updates = match task::lines_from::<task::Update>(&path, t.updates_read) {
            Ok((updates, read)) => {
                t.updates_read = read;
                updates
            }
            Err(e) => {
                t.updates_read = size;
                return vec![CoreEvent::Error(format!("{e:#}"))];
            }
        };
        let mut state = t.state.clone();
        if !updates.iter().fold(false, |changed, u| state.apply(u) || changed) {
            return Vec::new();
        }
        t.state = state.clone();
        match task::write(self.folder.path(), &state) {
            Ok(()) => t.saved(),
            Err(e) => vec![CoreEvent::Error(format!("{e:#}"))],
        }
    }

    /// What `pam say` wrote for task `id` since the last look: nothing while
    /// its cancel window is open, and never anything once it is cancelled.
    /// Only reads, so a failure is sent as an error and the meeting goes on.
    fn messages(&mut self, id: TaskId) -> Vec<CoreEvent> {
        let path = task::messages_file(self.folder.path(), id);
        let now = self.now;
        let Some(t) = self.tasks.iter_mut().find(|t| t.state.id == id) else { return Vec::new() };
        let size = std::fs::metadata(&path).map_or(0, |m| m.len());
        if t.window.is_some() || size == t.read {
            return Vec::new();
        }
        if t.cancelling {
            t.read = size;
            return Vec::new();
        }
        match task::messages_from(&path, t.read) {
            Ok((messages, read)) => {
                t.read = read;
                messages
                    .into_iter()
                    .map(|m| CoreEvent::StageMessage { task: Some(id), text: m.text, at: now })
                    .collect()
            }
            Err(e) => {
                t.read = size;
                vec![CoreEvent::Error(format!("{e:#}"))]
            }
        }
    }

    /// What the agents asked the room since the last look, and the questions
    /// whose countdown ran out or whose asker went away.
    fn look_for_questions(&mut self) -> Vec<CoreEvent> {
        let ids: Vec<TaskId> = self.tasks.iter().map(|t| t.state.id).collect();
        let mut events = Vec::new();
        for id in ids {
            events.extend(self.questions(id));
            events.extend(self.check_question(id));
        }
        events
    }

    /// What task `id`'s `questions.jsonl` gained since the last look. A new
    /// question replaces an open one, and is shown now or when the task's
    /// cancel window closes; one asked by a task that isn't running, or by
    /// an asker already gone, is dropped unseen. An end written by someone
    /// else (the asker giving up on its own) takes the question off the stage.
    fn questions(&mut self, id: TaskId) -> Vec<CoreEvent> {
        let path = task::questions_file(self.folder.path(), id);
        let Some(t) = self.tasks.iter_mut().find(|t| t.state.id == id) else { return Vec::new() };
        let size = std::fs::metadata(&path).map_or(0, |m| m.len());
        if size == t.asked_read {
            return Vec::new();
        }
        let records = match task::lines_from::<question::Record>(&path, t.asked_read) {
            Ok((records, read)) => {
                t.asked_read = read;
                records
            }
            Err(e) => {
                t.asked_read = size;
                return vec![CoreEvent::Error(format!("{e:#}"))];
            }
        };
        let mut events = Vec::new();
        for r in records {
            let Some(t) = self.tasks.iter_mut().find(|t| t.state.id == id) else { break };
            if let Some(asked) = r.question() {
                t.backlog.push_back(asked);
            } else if let Some(end) = r.end() {
                // The asker ended it itself: nothing is left to save.
                t.ending.retain(|(q, _)| *q != r.q());
                t.backlog.retain(|b| b.q != r.q());
                events.extend(self.take_off(id, r.q(), end));
            }
        }
        events.extend(self.next_question(id));
        events
    }

    /// Takes task `id`'s next questions from its backlog, while no question
    /// on the stage, this task's or another's, has an end still being saved:
    /// a newer question replaces this task's. One whose task isn't running, or whose asker has gone, is dropped
    /// unseen; the last that is neither is shown, now or when the task's
    /// cancel window closes.
    fn next_question(&mut self, id: TaskId) -> Vec<CoreEvent> {
        let mut events = Vec::new();
        while self.tasks.iter().any(|t| t.state.id == id && !t.backlog.is_empty()) && !self.stage_saving(id) {
            events.extend(self.close_question(id, End::Dropped("a newer question replaced it".into())));
            if self.stage_saving(id) {
                break;
            }
            let Some(t) = self.tasks.iter_mut().find(|t| t.state.id == id) else { break };
            let asked = t.backlog.pop_front().unwrap();
            let why = if t.state.status != TaskStatus::Running || t.cancelling {
                Some(format!("{id} isn't running"))
            } else if !question::alive(asked.pid) {
                Some("the agent stopped waiting for an answer".to_string())
            } else {
                None
            };
            t.asking = Some(Asking { question: asked, deadline: None, cut: None });
            match why {
                Some(why) => events.extend(self.close_question(id, End::Dropped(why))),
                None => events.extend(self.show_question(id)),
            }
        }
        events
    }

    /// Whether a question has ended but saving how failed, so it stays up:
    /// task `id`'s own, or one on the stage for another task. Until it saves
    /// no other question takes the stage, so `QuestionClosed` always closes
    /// the one the dashboard shows.
    fn stage_saving(&self, id: TaskId) -> bool {
        self.tasks.iter().any(|t| {
            t.asking.as_ref().is_some_and(|a| {
                (t.state.id == id || a.deadline.is_some()) && t.ending.iter().any(|(q, _)| *q == a.question.q)
            })
        })
    }

    /// Shows task `id`'s question, unless its cancel window is still open:
    /// the countdown starts now.
    fn show_question(&mut self, id: TaskId) -> Vec<CoreEvent> {
        let Some(t) = self.tasks.iter_mut().find(|t| t.state.id == id) else { return Vec::new() };
        if t.window.is_some() {
            return Vec::new();
        }
        let Some(a) = t.asking.as_mut().filter(|a| a.deadline.is_none()) else { return Vec::new() };
        let deadline = Instant::now() + a.question.timeout;
        a.deadline = Some(deadline);
        vec![CoreEvent::Question {
            task: id,
            q: a.question.q,
            text: a.question.text.clone(),
            options: a.question.options.clone(),
            deadline,
        }]
    }

    /// Saves again the ends that failed to save, and takes the next question
    /// if one waits; then ends task `id`'s question once its countdown has
    /// run out, and drops it once its asker has gone (pi's own `bash`
    /// timeout, say).
    fn check_question(&mut self, id: TaskId) -> Vec<CoreEvent> {
        let Some(t) = self.tasks.iter().find(|t| t.state.id == id) else { return Vec::new() };
        let mut events = Vec::new();
        for (q, end) in t.ending.clone() {
            if let (Some(saved), more) = self.save_end(id, q, end) {
                events.extend(more);
                events.extend(self.take_off(id, q, saved));
            }
        }
        events.extend(self.next_question(id));
        let now = self.now;
        let Some(t) = self.tasks.iter_mut().find(|t| t.state.id == id) else { return events };
        let Some(a) = t.asking.as_mut().filter(|a| !t.ending.iter().any(|(q, _)| *q == a.question.q)) else {
            return events;
        };
        // Past the deadline, a reply begun before it is waited for, for a while.
        let hearing = self.invocations.as_ref().is_some_and(Invocations::hears_replies);
        let lag = if hearing { HEARD_LAG } else { Duration::ZERO };
        let over = a.deadline.filter(|d| Instant::now() >= *d + lag).map(|d| {
            // The clock counts whole seconds heard, and the preview may stamp a word when
            // it shows: speech that begins before this may have begun before the deadline.
            let cut = *a.cut.get_or_insert(now + SECOND);
            Instant::now() >= d + REPLY_WAIT
                || self.invocations.as_ref().is_none_or(|i| !i.replying(cut, now))
        });
        if over == Some(true) {
            events.extend(self.close_question(id, End::TimedOut));
        } else if !question::alive(a.question.pid) {
            events.extend(
                self.close_question(id, End::Dropped("the agent stopped waiting for an answer".into())),
            );
        }
        events
    }

    /// The room answered question `q` of task `id` on the laptop: an option
    /// from the keys, or words typed for an open question. An answer that
    /// can't be saved is refused, and the question stays open.
    fn answer(
        &mut self,
        id: TaskId,
        q: u32,
        answer: Answer,
        by: By,
        spk: Option<SpkId>,
    ) -> Result<Vec<CoreEvent>> {
        self.answerable(id, Some(q), &answer)?;
        let saved = question::end(self.folder.path(), id, q, End::Answered { answer, by, spk })?;
        Ok(self.take_off(id, q, saved))
    }

    /// Whether task `id`'s question is on the stage, still open, is question
    /// `q` when one is named, and takes `answer`: an option it has, or words
    /// for an open question.
    fn answerable(&self, id: TaskId, q: Option<u32>, answer: &Answer) -> Result<()> {
        let Some(t) = self.tasks.iter().find(|t| t.state.id == id) else { bail!("no task {id}") };
        let Some(a) = t.asking.as_ref().filter(|a| a.deadline.is_some()) else {
            bail!("{id} isn't asking anything");
        };
        if t.ending.iter().any(|(q, _)| *q == a.question.q) {
            bail!("{id}'s question has ended; saving how failed, and is being tried again");
        }
        // A task asks one question after another, under the same ID.
        if q.is_some_and(|q| q != a.question.q) {
            bail!("{id} is asking something else now; that answer was for its earlier question");
        }
        let n = a.question.options.len();
        match answer {
            Answer::Option(_) if n == 0 => bail!("{id}'s question is open: type the answer"),
            Answer::Option(option) if !(1..=n).contains(option) => bail!("{id}'s question has options 1–{n}"),
            Answer::Text(_) if n > 0 => bail!("{id}'s question has options: answer 1–{n}"),
            Answer::Text(text) if text.trim().is_empty() => bail!("the answer is empty"),
            _ => Ok(()),
        }
    }

    /// The number of the question task `id` is asking.
    fn question_of(&self, id: TaskId) -> Option<u32> {
        self.tasks.iter().find(|t| t.state.id == id)?.asking.as_ref().map(|a| a.question.q)
    }

    /// Ends task `id`'s question as `end`, and takes it off the stage once
    /// that is saved. Nothing shows an end the asker can't see.
    fn close_question(&mut self, id: TaskId, end: End) -> Vec<CoreEvent> {
        let Some(q) = self.question_of(id) else { return Vec::new() };
        let (saved, mut events) = self.save_end(id, q, end);
        if let Some(saved) = saved {
            events.extend(self.take_off(id, q, saved));
        }
        events
    }

    /// Saves `end` for question `q` of task `id`, and returns the end that
    /// counts: the first one saved, maybe the asker's own. A failed save is
    /// kept in `ending`, tried again at every look, and reported once.
    fn save_end(&mut self, id: TaskId, q: u32, end: End) -> (Option<End>, Vec<CoreEvent>) {
        let saved = question::end(self.folder.path(), id, q, end.clone());
        let Some(t) = self.tasks.iter_mut().find(|t| t.state.id == id) else { return (None, Vec::new()) };
        match saved {
            Ok(saved) => {
                t.ending.retain(|(p, _)| *p != q);
                (Some(saved), Vec::new())
            }
            Err(_) if t.ending.iter().any(|(p, _)| *p == q) => (None, Vec::new()),
            Err(e) => {
                t.ending.push((q, end));
                (None, vec![CoreEvent::Error(format!("{e:#}"))])
            }
        }
    }

    /// Takes question `q` of task `id` off the stage, if it is the one there,
    /// now that `end` is saved.
    fn take_off(&mut self, id: TaskId, q: u32, end: End) -> Vec<CoreEvent> {
        let Some(t) = self.tasks.iter_mut().find(|t| t.state.id == id) else { return Vec::new() };
        if t.asking.as_ref().is_none_or(|a| a.question.q != q) {
            return Vec::new();
        }
        let shown = t.asking.take().is_some_and(|a| a.deadline.is_some());
        if shown { vec![CoreEvent::QuestionClosed { task: id, end }] } else { Vec::new() }
    }

    /// Saves each event's change to the folder, then sends it, in order. A
    /// final line is followed by the rename its introduction causes.
    ///
    /// The source files (`transcript.jsonl`, `speakers.json`) decide: on a
    /// failed write, sends the error instead and stops, so subscribers never
    /// get an event whose change isn't on disk. Rendered files are rebuilt
    /// from the sources on every change, so a failed render is sent as an
    /// error after the event and the meeting goes on; the next render repairs it.
    fn publish(&mut self, events: Vec<CoreEvent>) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }
        let hub = self.hub.clone();
        let mut hub = hub.lock().unwrap();
        let mut send = |m: &mut Merger, e: CoreEvent| {
            if let Err(err) = m.persist(&e) {
                hub.send(CoreEvent::Error(format!("{err:#}")));
                return Err(err);
            }
            let rendered = m.render(&e);
            hub.send(e);
            if let Err(err) = rendered {
                hub.send(CoreEvent::Error(format!("{err:#}")));
            }
            Ok(())
        };
        // What spoken requests make of the lines, carried out once these are sent.
        let mut actions = Vec::new();
        for e in events {
            let now = self.now;
            let line = match &e {
                CoreEvent::LineFinal(line) => Some(line.clone()),
                CoreEvent::LinePreview(p) => {
                    self.speaking = !p.is_empty();
                    actions.extend(self.invocations.as_mut().map(|i| i.preview(p, now)).unwrap_or_default());
                    None
                }
                // While a question is on the stage, what is said may answer it.
                CoreEvent::Question { text, options, .. } => {
                    let asked = Asked { text: text.clone(), options: options.clone() };
                    if let Some(i) = self.invocations.as_mut() {
                        i.question(Some(asked));
                    }
                    None
                }
                CoreEvent::QuestionClosed { .. } => {
                    if let Some(i) = self.invocations.as_mut() {
                        i.question(None);
                    }
                    None
                }
                _ => None,
            };
            send(self, e)?;
            if let Some(line) = line {
                if let Some(renamed) = self.identity.intro(&line, &self.assistant, timestamp(Local::now())) {
                    send(self, renamed)?;
                }
                actions.extend(self.invocations.as_mut().map(|i| i.line(&line, now)).unwrap_or_default());
            }
        }
        drop(hub);
        let events = self.act(actions);
        self.publish(events)
    }

    /// Saves exactly what `event` says to the source files: the pipeline may
    /// already be further on.
    fn persist(&mut self, event: &CoreEvent) -> Result<()> {
        match event {
            CoreEvent::LineFinal(line) => {
                self.folder.append_line(line)?;
                self.lines.push(line.clone());
                Ok(())
            }
            CoreEvent::SpeakerAdded(spk) => {
                self.identity.set_slots(self.p.slots().up_to(*spk));
                self.folder.write_speakers(&self.identity)
            }
            CoreEvent::SpeakerRenamed { .. }
            | CoreEvent::SpeakersMerged { .. }
            | CoreEvent::LineMoved { .. }
            | CoreEvent::Undone { .. } => self.folder.write_speakers(&self.identity),
            _ => Ok(()),
        }
    }

    /// Re-renders the files `event` changes, once its sources are saved. Each
    /// is tried even if another fails; the first failure is returned.
    fn render(&self, event: &CoreEvent) -> Result<()> {
        let notes = match event {
            // context.md lists the tasks. A new status also comes with a
            // `task_update` event, whose save renders the notes too.
            CoreEvent::TaskUpdated(_) => {
                return EventLog::lock(self.folder.path()).and_then(|log| log.write_context());
            }
            CoreEvent::LineFinal(_) => false,
            CoreEvent::SpeakerRenamed { .. }
            | CoreEvent::SpeakersMerged { .. }
            | CoreEvent::LineMoved { .. }
            | CoreEvent::Undone { .. } => true,
            _ => return Ok(()),
        };
        let transcript = self.folder.write_transcript_md(&self.lines, &self.identity);
        let rest = EventLog::lock(self.folder.path()).and_then(|log| match notes {
            // An unreadable events.jsonl fails context.md too, so nothing is left to try.
            true => log.events().and_then(|e| self.write_notes_and_context(&log, &e)),
            false => log.write_context(),
        });
        transcript.and(rest)
    }

    fn notes(&self, events: &[Event]) -> Notes {
        let tasks: Vec<TaskState> = self.tasks.iter().map(|t| t.state.clone()).collect();
        Notes::build(&self.summary.text, events, &tasks)
    }

    fn write_notes(&self, log: &EventLog, events: &[Event]) -> Result<()> {
        log.write_notes(self.folder.title(), &self.notes(events), self.identity.overlay())
    }

    fn write_notes_and_context(&self, log: &EventLog, events: &[Event]) -> Result<()> {
        log.write_notes_and_context(self.folder.title(), &self.notes(events), self.identity.overlay())
    }

    /// Sends the next lines to the scribe if a run is due: after enough
    /// speech, at a pause, or once the input ended. The last run is `finish`'s.
    fn scribe_if_due(&mut self) {
        if self.jobs.is_none() || self.in_flight.is_some() || self.scribed == self.lines.len() {
            return;
        }
        let due = self.retry_at.is_none_or(|t| Instant::now() >= t)
            && (self.ended || scribe::due(&self.lines[self.scribed..], self.now, self.speaking));
        if due && let Err(e) = self.send_job() {
            self.failed(format!("{e:#}"));
        }
    }

    /// Sends the scribe every line it hasn't read.
    fn send_job(&mut self) -> Result<()> {
        let events = read_jsonl::<Event>(&self.folder.path().join("events.jsonl"))?;
        let overlay = self.identity.overlay();
        let mut people: Vec<SpkId> = self.speakers().into_iter().map(|s| overlay.follow(s)).collect();
        people.sort();
        people.dedup();
        let people = people.into_iter().map(|s| (s, overlay.name(s).map(str::to_string))).collect();
        let lines = self.corrected(self.lines.len());
        let prompt = scribe::prompt(&scribe::Input {
            title: self.folder.title(),
            people,
            summary: &self.summary.text,
            events: &events,
            earlier: &lines[..self.scribed],
            new: &lines[self.scribed..],
        });
        let through = self.lines.len();
        let sent = self.jobs.as_ref().is_some_and(|jobs| jobs.send(Job { prompt, through }).is_ok());
        if !sent {
            bail!("the scribe has stopped");
        }
        self.in_flight = Some(through);
        Ok(())
    }

    /// Every speaker ID given out so far, merged-away ones included.
    fn speakers(&self) -> Vec<SpkId> {
        self.identity.speakers().slots.values().copied().collect()
    }

    /// `lines[..n]` with corrections applied: as the scribe should read them.
    fn corrected(&self, n: usize) -> Vec<Line> {
        self.lines[..n].iter().map(|l| Line { spk: self.identity.speaker_of(l), ..l.clone() }).collect()
    }

    /// A run failed: say so, and try again later with these lines and any new ones.
    fn failed(&mut self, why: String) {
        self.retry_at = Some(Instant::now() + RETRY);
        let why = format!("scribe: {why}");
        self.last_failure = Some(why.clone());
        self.hub.lock().unwrap().send(CoreEvent::Error(why));
    }

    /// Saves what the scribe made of `lines[..through]`: renames from
    /// introductions first, then the events and summary, then the notes.
    fn scribed(&mut self, through: usize, reply: Result<Reply>) -> Result<()> {
        self.in_flight = None;
        let reply = match reply {
            Ok(r) => r,
            Err(e) => {
                self.failed(format!("{e:#}"));
                return Ok(());
            }
        };
        self.retry_at = None;
        self.last_failure = None;
        let log = EventLog::lock(self.folder.path())?;
        let speakers = self.speakers().into_iter().collect();
        let first = self.lines[self.scribed].id;
        let mut new = scribe::check(&reply, &self.corrected(through), first, &speakers, &log.events()?);
        drop(log);

        let mut renames = Vec::new();
        for e in new.iter_mut().filter(|e| e.kind == "intro") {
            let (Some(name), &[spk]) = (e.name.clone(), &e.by[..]) else { continue };
            let Some(line) = e.refs.first().and_then(|r| match r.parse() {
                Ok(Ref::Line(id) | Ref::Lines(id, _)) => Some(id),
                _ => None,
            }) else {
                continue;
            };
            match self.identity.scribe_intro(spk, &name, line, timestamp(Local::now())) {
                ScribeIntro::Confirmed(undo) => e.rename = Some(undo),
                ScribeIntro::Renamed(renamed) => {
                    if let CoreEvent::SpeakerRenamed { undo, .. } = &*renamed {
                        e.rename = Some(*undo);
                    }
                    renames.push(*renamed);
                }
                ScribeIntro::Skipped => {}
            }
        }
        self.publish(renames)?;

        let mut log = EventLog::lock(self.folder.path())?;
        log.append(new)?;
        if let Some(text) = reply.summary {
            self.summary = Summary {
                through: through.checked_sub(1).and_then(|i| self.lines.get(i)).map(|l| l.id),
                text,
            };
            write_summary(self.folder.path(), &self.summary)?;
        }
        self.scribed = through;
        self.send_notes(&log)
    }

    /// Re-renders `notes.md` and `context.md`, and sends the notes.
    fn send_notes(&mut self, log: &EventLog) -> Result<()> {
        let events = self.notes_updated(log)?;
        let mut hub = self.hub.lock().unwrap();
        for e in events {
            hub.send(e);
        }
        Ok(())
    }

    /// Re-renders `notes.md` and `context.md`, and returns the notes to send,
    /// then the render's failure, if any.
    fn notes_updated(&mut self, log: &EventLog) -> Result<Vec<CoreEvent>> {
        let events = log.events()?;
        let rendered = self.write_notes_and_context(log, &events);
        self.events_seen = events_size(self.folder.path());
        let mut out = vec![CoreEvent::NotesUpdated(self.notes(&events))];
        if let Err(err) = rendered {
            out.push(CoreEvent::Error(format!("{err:#}")));
        }
        Ok(out)
    }

    /// Sends the notes again if someone else (`pam note`) added events. Only
    /// reads, so a failure is reported once and the meeting goes on.
    fn look_for_notes(&mut self) {
        let size = events_size(self.folder.path());
        if size == self.events_seen {
            return;
        }
        if let Err(e) = EventLog::lock(self.folder.path()).and_then(|log| self.send_notes(&log)) {
            self.events_seen = size;
            self.hub.lock().unwrap().send(CoreEvent::Error(format!("{e:#}")));
        }
    }
}

/// A task's status in words, for an error.
fn status_word(status: TaskStatus, cancelling: bool) -> &'static str {
    match status {
        _ if cancelling => "being cancelled",
        TaskStatus::Queued => "queued",
        TaskStatus::Running => "running",
        TaskStatus::Done => "answered",
        TaskStatus::Failed => "failed",
        TaskStatus::Stopped => "stopped",
        TaskStatus::Cancelled => "cancelled",
    }
}

/// The size of task `id`'s `messages.jsonl`, or 0 if it can't be read.
fn messages_size(dir: &std::path::Path, id: TaskId) -> u64 {
    std::fs::metadata(task::messages_file(dir, id)).map_or(0, |m| m.len())
}

/// The size of `events.jsonl`, or 0 if it can't be read.
fn events_size(dir: &std::path::Path) -> u64 {
    std::fs::metadata(dir.join("events.jsonl")).map_or(0, |m| m.len())
}

#[cfg(test)]
mod tests {
    use crate::task::{Target, TaskResult};
    use pam_speech::{SpeakerSpan, SpeakerUpdate, Update};

    use super::*;

    #[test]
    fn a_failed_mergers_last_count_is_partial_at_stop() {
        let dir = tempfile::tempdir().unwrap();
        let mut counted = Usage::default();
        let call =
            crate::usage::Call { input: 5, output: 1, cache_read: 0, cache_write: 0, cost: Some(0.01) };
        counted.add(User::Scribe, &call);
        // The merger's own count is final.
        assert_eq!(final_usage(Some(counted.clone()), dir.path(), &Usage::default()), counted);
        // Failed: the one it last wrote, partial.
        write_info(dir.path(), &MeetingInfo { usage: counted.clone(), ..MeetingInfo::default() }).unwrap();
        let kept = final_usage(None, dir.path(), &Usage::default());
        assert_eq!((kept.total.calls, kept.partial), (1, true));
        // Nothing written: the start's, partial.
        let empty = tempfile::tempdir().unwrap();
        let kept = final_usage(None, empty.path(), &Usage::default());
        assert_eq!((kept.total.calls, kept.partial), (0, true));
    }

    /// Holds the check "talk" until the test lets it go.
    struct Stuck(Mutex<Option<Receiver<()>>>);

    impl llm::Model for Stuck {
        fn complete(&self, req: &Request<'_>) -> llm::Completion {
            if req.prompt == "talk" {
                let held = self.0.lock().unwrap().take().unwrap();
                let _ = held.recv_timeout(Duration::from_secs(5));
            }
            llm::Completion { reply: Ok(r#"{"kind":"none"}"#.into()), used: Used::Unknown }
        }
    }

    #[test]
    fn a_check_does_not_wait_behind_a_slow_one() {
        let (release, held) = bounded(1);
        let model = Arc::new(Stuck(Mutex::new(Some(held))));
        let (checks, check_rx) = unbounded();
        let (msg_tx, msg_rx) = unbounded();
        spawn("pam-intent", move || run_intent(model, String::new(), check_rx, msg_tx));
        checks.send(Check { id: 1, prompt: "talk".into(), grammar: String::new() }).unwrap();
        checks.send(Check { id: 2, prompt: "reply".into(), grammar: String::new() }).unwrap();
        let next = || match msg_rx.recv_timeout(Duration::from_secs(10)).unwrap() {
            Msg::Intent { check, .. } => check,
            _ => panic!("not a verdict"),
        };
        // Only let the talk's check go once the reply's has come back, so the
        // order can't come from the threads' scheduling.
        assert_eq!(next(), 2, "the reply's check ran while the talk's was stuck");
        release.send(()).unwrap();
        assert_eq!(next(), 1);
    }

    #[test]
    fn the_checks_still_out_finish_before_the_model_is_let_go() {
        let (release, held) = bounded(1);
        let model = Arc::new(Stuck(Mutex::new(Some(held))));
        let (checks, check_rx) = unbounded();
        let (msg_tx, msg_rx) = unbounded();
        let intent = spawn("pam-intent", {
            let model = model.clone();
            move || run_intent(model, String::new(), check_rx, msg_tx)
        });
        checks.send(Check { id: 1, prompt: "talk".into(), grammar: String::new() }).unwrap();
        // The meeting stops: the merger closes the queue.
        drop(checks);
        thread::sleep(Duration::from_millis(200));
        assert!(!intent.is_finished(), "it waits for the check still out");
        release.send(()).unwrap();
        intent.join().unwrap();
        assert_eq!(Arc::strong_count(&model), 1, "every check has let the model go");
        assert!(matches!(msg_rx.try_recv(), Ok(Msg::Intent { check: 1, .. })));
    }

    #[test]
    fn the_name_is_a_hint_when_spoken_requests_are_taken() {
        let mut cfg = MeetingConfig::wav(PathBuf::new(), PathBuf::new(), &Config::default());
        cfg.attendees = vec!["Jordan".into()];
        assert_eq!(hints(&cfg), ["Jordan"], "no intent check, no spoken requests");
        cfg.intent = Some(Llm::pi("pi", "stub/intent"));
        assert_eq!(hints(&cfg), ["Jordan"], "no agent to run them");
        let ready = models::ModelInfo {
            status: models::ModelStatus::Ready,
            reason: None,
            pi: None,
            version: None,
            model: None,
            provider: None,
            source: None,
            login: None,
            catalogued: None,
        };
        cfg.agent = Some(AgentConfig { info: ready, pam: None });
        assert_eq!(hints(&cfg), ["Jordan", "Pam"]);
        cfg.assistant_name = " ".into();
        assert_eq!(hints(&cfg), ["Jordan"]);
    }

    #[test]
    fn a_failed_render_still_sends_the_saved_line_and_the_next_render_repairs_it() {
        let root = tempfile::tempdir().unwrap();
        let folder = MeetingFolder::create(root.path(), "", Local::now()).unwrap();
        let started_at = timestamp(Local::now());
        folder
            .write_info(&MeetingInfo { started_at, status: "recording".into(), ..MeetingInfo::default() })
            .unwrap();
        let dir = folder.path().to_path_buf();
        let mut m = Merger::new(folder, "Pam".into(), Arc::default());
        let (tx, events) = unbounded();
        m.hub.lock().unwrap().subscribers.push(tx);
        let line = |id, text: &str| {
            CoreEvent::LineFinal(crate::event::Line {
                id: crate::event::UtteranceId(id),
                t0: 0,
                t1: 0,
                slot: None,
                spk: None,
                lang: None,
                text: text.into(),
            })
        };
        // A directory in its place makes replacing transcript.md fail.
        std::fs::remove_file(dir.join("transcript.md")).unwrap();
        std::fs::create_dir_all(dir.join("transcript.md/blocked")).unwrap();

        m.publish(vec![line(1, "First.")]).unwrap();
        let got: Vec<CoreEvent> = events.try_iter().collect();
        assert_eq!(got[0], line(1, "First."));
        assert!(matches!(&got[1], CoreEvent::Error(e) if e.contains("transcript.md")), "{got:?}");
        assert!(std::fs::read_to_string(dir.join("transcript.jsonl")).unwrap().contains("First."));

        std::fs::remove_dir_all(dir.join("transcript.md")).unwrap();
        m.publish(vec![line(2, "Second.")]).unwrap();
        assert_eq!(events.try_iter().collect::<Vec<_>>(), [line(2, "Second.")]);
        let md = std::fs::read_to_string(dir.join("transcript.md")).unwrap();
        assert!(md.contains("First.") && md.contains("Second."), "{md}");
    }

    #[test]
    fn each_speaker_added_saves_only_the_slots_up_to_it() {
        let root = tempfile::tempdir().unwrap();
        let folder = MeetingFolder::create(root.path(), "", Local::now()).unwrap();
        let mut m = Merger::new(folder, "Pam".into(), Arc::default());
        let spans =
            vec![SpeakerSpan { start: 0, end: 10, slot: 0 }, SpeakerSpan { start: 0, end: 10, slot: 1 }];
        let update =
            SpeakerUpdate { spans: Update { finalized: spans, preview: Vec::new() }, settled_until: 10 };
        let added: Vec<CoreEvent> =
            m.p.speakers(update, 10)
                .into_iter()
                .filter(|e| matches!(e, CoreEvent::SpeakerAdded(_)))
                .collect();
        assert_eq!(added.len(), 2);
        m.persist(&added[0]).unwrap();
        let saved = std::fs::read_to_string(m.folder.path().join("speakers.json")).unwrap();
        let saved: serde_json::Value = serde_json::from_str(&saved).unwrap();
        assert_eq!(saved["slots"], serde_json::json!({"0": "spk_0"}));
    }

    #[test]
    fn new_events_reach_context_md_even_when_notes_md_cannot_be_written() {
        let root = tempfile::tempdir().unwrap();
        let folder = MeetingFolder::create(root.path(), "", Local::now()).unwrap();
        let started_at = timestamp(Local::now());
        folder
            .write_info(&MeetingInfo { started_at, status: "recording".into(), ..MeetingInfo::default() })
            .unwrap();
        let dir = folder.path().to_path_buf();
        let mut m = Merger::new(folder, "Pam".into(), Arc::default());
        let (tx, events) = unbounded();
        m.hub.lock().unwrap().subscribers.push(tx);
        // A directory in its place makes replacing notes.md fail.
        std::fs::remove_file(dir.join("notes.md")).unwrap();
        std::fs::create_dir_all(dir.join("notes.md/blocked")).unwrap();

        let mut log = EventLog::lock(&dir).unwrap();
        let note = Event {
            id: crate::event::EventId(0),
            kind: "note".into(),
            t: 0,
            by: Vec::new(),
            text: "Book a room".into(),
            refs: Vec::new(),
            task: None,
            supersedes: None,
            name: None,
            rename: None,
        };
        log.append(vec![note]).unwrap();
        m.send_notes(&log).unwrap();
        let got: Vec<CoreEvent> = events.try_iter().collect();
        assert!(
            matches!(&got[..], [CoreEvent::NotesUpdated(_), CoreEvent::Error(e)] if e.contains("notes.md")),
            "{got:?}"
        );
        let context = std::fs::read_to_string(dir.join("context.md")).unwrap();
        assert!(context.contains("- e001 00:00 note: Book a room\n"), "{context}");
    }

    #[test]
    fn a_start_whose_time_cannot_be_saved_is_not_logged_until_it_is() {
        let root = tempfile::tempdir().unwrap();
        let folder = MeetingFolder::create(root.path(), "", Local::now()).unwrap();
        let mut m = Merger::new(folder, "Pam".into(), Arc::default());
        let state = TaskState::new(TaskId(1), "what did we decide?", 0, Via::Typed, TaskStatus::Running);
        task::create(m.folder.path(), &state).unwrap();
        m.tasks.push(Tracked::new(state, None));
        let updates = |m: &Merger| -> Vec<String> {
            read_jsonl::<Event>(&m.folder.path().join("events.jsonl"))
                .unwrap()
                .into_iter()
                .filter(|e| e.kind == "task_update")
                .map(|e| e.text)
                .collect()
        };
        // A folder in place of the temporary file makes saving task.json fail.
        let blocked = task::dir(m.folder.path(), TaskId(1)).join(".task.json.tmp/x");
        std::fs::create_dir_all(&blocked).unwrap();
        m.now = 16_000;
        let events = m.prompted(TaskId(1), 0);
        assert!(matches!(&events[..], [CoreEvent::Error(_)]), "{events:?}");
        assert!(updates(&m).is_empty(), "no start is logged that task.json doesn't have");
        assert_eq!(task::read(m.folder.path(), TaskId(1)).unwrap().started_at, None);

        // The next prompt tries again, and then it has started.
        std::fs::remove_dir_all(blocked.parent().unwrap()).unwrap();
        m.now = 32_000;
        m.prompted(TaskId(1), 0);
        assert_eq!(updates(&m), ["started"]);
        assert_eq!(task::read(m.folder.path(), TaskId(1)).unwrap().started_at, Some(32_000));
        assert!(m.prompted(TaskId(1), 0).is_empty(), "a correction's prompt starts nothing");
        assert_eq!(updates(&m), ["started"]);
    }

    #[test]
    fn reports_reach_task_json_after_the_cancel_window_and_only_while_the_task_runs() {
        let root = tempfile::tempdir().unwrap();
        let folder = MeetingFolder::create(root.path(), "", Local::now()).unwrap();
        let mut m = Merger::new(folder, "Pam".into(), Arc::default());
        let state = TaskState::new(TaskId(1), "fact-check that", 0, Via::Voice, TaskStatus::Running);
        task::create(m.folder.path(), &state).unwrap();
        m.tasks.push(Tracked::new(state, Some(5 * SECOND)));
        let dir = m.folder.path().to_path_buf();
        let saved = || task::read(&dir, TaskId(1)).unwrap();
        let reported = |events: &[CoreEvent]| -> Vec<(Option<u8>, Option<String>)> {
            events
                .iter()
                .filter_map(|e| match e {
                    CoreEvent::TaskUpdated(t) => Some((t.progress, t.action.clone())),
                    _ => None,
                })
                .collect()
        };

        task::update(&dir, TaskId(1), Some(20), Some("searching")).unwrap();
        assert!(m.look_for_updates().is_empty(), "held while its cancel window is open");
        m.now = 5 * SECOND;
        let events = m.close_windows();
        assert_eq!(reported(&events), [(Some(20), Some("searching".into()))]);
        assert_eq!((saved().progress, saved().action.as_deref()), (Some(20), Some("searching")));

        // Several reports between two looks are one update; one that changes nothing is none.
        task::update(&dir, TaskId(1), Some(40), None).unwrap();
        task::update(&dir, TaskId(1), None, Some("reading source 2 of 5")).unwrap();
        assert_eq!(reported(&m.look_for_updates()), [(Some(40), Some("reading source 2 of 5".into()))]);
        task::update(&dir, TaskId(1), Some(40), None).unwrap();
        assert!(m.look_for_updates().is_empty());

        // A report that can't be saved stays with the task, and is saved with its next change.
        let blocked = task::dir(&dir, TaskId(1)).join(".task.json.tmp/x");
        std::fs::create_dir_all(&blocked).unwrap();
        task::update(&dir, TaskId(1), Some(80), None).unwrap();
        assert!(matches!(&m.look_for_updates()[..], [CoreEvent::Error(_)]));
        assert_eq!(saved().progress, Some(40));
        std::fs::remove_dir_all(blocked.parent().unwrap()).unwrap();
        let events = m.task_ended(TaskId(1), Outcome::Done { reply: None }, 6 * SECOND);
        assert_eq!(reported(&events), [(Some(80), Some("reading source 2 of 5".into()))]);
        assert_eq!((saved().status, saved().progress), (TaskStatus::Done, Some(80)));

        // Once it has ended, a report changes nothing.
        task::update(&dir, TaskId(1), Some(90), None).unwrap();
        assert!(m.look_for_updates().is_empty());
        // A prompt starts the agent afresh: run again for a correction, and a
        // correction to the run still going. Reports from before it are the old attempt's.
        let length = || std::fs::metadata(task::updates_file(&dir, TaskId(1))).unwrap().len();
        m.task_started(TaskId(1));
        task::update(&dir, TaskId(1), Some(95), Some("still the old run")).unwrap();
        assert_eq!(reported(&m.prompted(TaskId(1), length())), [(None, None)]);
        assert!(m.look_for_updates().is_empty(), "never read");
        assert_eq!((saved().progress, &saved().action), (None, &None));
        task::update(&dir, TaskId(1), Some(50), Some("writing it up")).unwrap();
        assert_eq!(reported(&m.look_for_updates()), [(Some(50), Some("writing it up".into()))]);
        task::update(&dir, TaskId(1), Some(70), None).unwrap();
        assert_eq!(
            reported(&m.prompted(TaskId(1), length())),
            [(None, None)],
            "a correction to a running task"
        );
        assert!(m.look_for_updates().is_empty());
        assert_eq!((saved().status, saved().progress, &saved().action), (TaskStatus::Running, None, &None));
        // A quick new attempt reports before its prompt is heard, and a look applies
        // it first: hearing the prompt keeps what came after it.
        task::update(&dir, TaskId(1), Some(25), Some("the old attempt's last")).unwrap();
        let boundary = length();
        task::update(&dir, TaskId(1), Some(5), Some("starting over")).unwrap();
        assert_eq!(reported(&m.look_for_updates()), [(Some(5), Some("starting over".into()))]);
        let events = m.prompted(TaskId(1), boundary);
        assert_eq!(reported(&events).last(), Some(&(Some(5), Some("starting over".into()))), "{events:?}");
        assert_eq!((saved().progress, saved().action.as_deref()), (Some(5), Some("starting over")));
        task::update(&dir, TaskId(1), Some(10), None).unwrap();
        m.tasks[0].cancelling = true;
        assert!(m.look_for_updates().is_empty(), "a task being cancelled takes none");
    }

    #[test]
    fn a_result_is_held_in_the_cancel_window_replaced_cleared_by_a_prompt_and_dropped_when_cancelled() {
        let root = tempfile::tempdir().unwrap();
        let folder = MeetingFolder::create(root.path(), "", Local::now()).unwrap();
        let mut m = Merger::new(folder, "Pam".into(), Arc::default());
        let state = TaskState::new(TaskId(1), "fact-check that", 0, Via::Voice, TaskStatus::Running);
        task::create(m.folder.path(), &state).unwrap();
        m.tasks.push(Tracked::new(state, Some(5 * SECOND)));
        let dir = m.folder.path().to_path_buf();
        let saved = || task::read(&dir, TaskId(1)).unwrap().result;
        let result =
            |title: &str| TaskResult { title: title.into(), target: Target::File("tasks/t01/r.md".into()) };
        // The results the `TaskUpdated`s in `events` carry.
        let results = |events: &[CoreEvent]| -> Vec<Option<String>> {
            events
                .iter()
                .filter_map(|e| match e {
                    CoreEvent::TaskUpdated(t) => Some(t.result.as_ref().map(|r| r.title.clone())),
                    _ => None,
                })
                .collect()
        };

        task::attach(&dir, TaskId(1), result("First")).unwrap();
        assert!(m.look_for_updates().is_empty(), "held while its cancel window is open");
        assert_eq!(saved(), None);
        m.now = 5 * SECOND;
        let events = m.close_windows();
        assert_eq!(results(&events), [Some("First".into())]);
        assert_eq!(saved(), Some(result("First")));

        // Attaching again replaces it; the same one again changes nothing.
        task::attach(&dir, TaskId(1), result("Second")).unwrap();
        assert_eq!(results(&m.look_for_updates()), [Some("Second".into())]);
        task::attach(&dir, TaskId(1), result("Second")).unwrap();
        assert!(m.look_for_updates().is_empty());

        // One that can't be saved stays with the task, and is sent with its next save.
        let blocked = task::dir(&dir, TaskId(1)).join(".task.json.tmp/x");
        std::fs::create_dir_all(&blocked).unwrap();
        task::attach(&dir, TaskId(1), result("Third")).unwrap();
        assert!(matches!(&m.look_for_updates()[..], [CoreEvent::Error(_)]));
        std::fs::remove_dir_all(blocked.parent().unwrap()).unwrap();
        let events = m.task_ended(TaskId(1), Outcome::Done { reply: None }, 6 * SECOND);
        assert_eq!(results(&events), [Some("Third".into())]);
        assert_eq!(saved(), Some(result("Third")));
        assert!(task::read(&dir, TaskId(1)).unwrap().is_ready());

        // Run again for a correction: the agent starts afresh, and its result goes.
        let length = || std::fs::metadata(task::updates_file(&dir, TaskId(1))).unwrap().len();
        m.task_started(TaskId(1));
        let events = m.prompted(TaskId(1), length());
        assert_eq!(results(&events)[0], None, "{events:?}");
        assert_eq!(saved(), None);
        task::attach(&dir, TaskId(1), result("Third")).unwrap();
        assert_eq!(results(&m.look_for_updates()), [Some("Third".into())], "attached anew");

        // A task being cancelled takes none.
        task::attach(&dir, TaskId(1), result("Fourth")).unwrap();
        m.tasks[0].cancelling = true;
        assert!(m.look_for_updates().is_empty());
        assert_eq!(saved(), Some(result("Third")));
    }

    #[test]
    fn a_correction_goes_to_the_newest_task_still_going() {
        let root = tempfile::tempdir().unwrap();
        let folder = MeetingFolder::create(root.path(), "", Local::now()).unwrap();
        let mut m = Merger::new(folder, "Pam".into(), Arc::default());
        let task = |id, status| TaskState::new(TaskId(id), &format!("question {id}"), 0, Via::Typed, status);
        m.tasks.push(Tracked::new(task(1, TaskStatus::Running), None));
        m.tasks.push(Tracked::new(task(2, TaskStatus::Cancelled), None));
        assert_eq!(m.latest().map(|t| t.state.id), Some(TaskId(1)), "a cancelled newer task is skipped");

        // Answered in its window: still going until the window closes.
        m.tasks[0].state.status = TaskStatus::Done;
        m.tasks[0].window = Some(5 * SECOND);
        assert_eq!(m.latest().map(|t| t.state.id), Some(TaskId(1)));
        // Answered: a correction now is a new request.
        m.tasks[0].window = None;
        assert!(m.latest().is_none(), "an answered task is not corrected");
        let p = m.check_prompt("Pam, I meant the launch email.", None, 4 * SECOND, None, None);
        assert!(p.contains("LATEST REQUEST\n(none)\n"), "{p}");
    }

    /// A merger whose task t01 asks "Which customer?", due at `deadline`,
    /// with spoken requests checked, and the question on the stage.
    fn asking_merger(root: &std::path::Path, deadline: Instant) -> Merger {
        let folder = MeetingFolder::create(root, "", Local::now()).unwrap();
        let mut m = Merger::new(folder, "Pam".into(), Arc::default());
        let state = TaskState::new(TaskId(1), "draft the email", 0, Via::Typed, TaskStatus::Running);
        task::create(m.folder.path(), &state).unwrap();
        let mut t = Tracked::new(state, None);
        let options = vec!["Acme".to_string(), "Globex".to_string()];
        t.asking = Some(Asking {
            question: Question {
                q: 1,
                text: "Which customer?".into(),
                options: options.clone(),
                timeout: Duration::from_secs(15),
                pid: std::process::id(),
            },
            deadline: Some(deadline),
            cut: None,
        });
        m.tasks.push(t);
        let mut inv = Invocations::new("Pam");
        inv.question(Some(Asked { text: "Which customer?".into(), options }));
        m.invocations = Some(inv);
        m
    }

    #[test]
    fn past_its_deadline_a_question_waits_for_speech_that_shows_late() {
        let root = tempfile::tempdir().unwrap();
        let mut m = asking_merger(root.path(), Instant::now() - Duration::from_millis(300));
        m.now = 20 * SECOND;
        assert!(m.check_question(TaskId(1)).is_empty(), "words show in the preview a little late");
        // Speech begun just before the deadline shows now.
        let p =
            [crate::event::PreviewLine { spk: None, text: "yes the".into(), t0: 20 * SECOND + SECOND / 2 }];
        m.invocations.as_mut().unwrap().preview(&p, 21 * SECOND);
        m.tasks[0].asking.as_mut().unwrap().deadline = Some(Instant::now() - HEARD_LAG);
        assert!(m.check_question(TaskId(1)).is_empty(), "a reply may be on its way");
        // It became a line that wasn't a reply: the question times out.
        m.invocations.as_mut().unwrap().preview(&[], 23 * SECOND);
        let closed = m.check_question(TaskId(1));
        assert_eq!(closed, [CoreEvent::QuestionClosed { task: TaskId(1), end: End::TimedOut }]);
    }

    #[test]
    fn past_its_deadline_a_question_ignores_speech_begun_well_after_it() {
        let root = tempfile::tempdir().unwrap();
        let mut m = asking_merger(root.path(), Instant::now() - HEARD_LAG);
        m.now = 20 * SECOND;
        let p = [crate::event::PreviewLine { spk: None, text: "so anyway".into(), t0: 23 * SECOND }];
        m.invocations.as_mut().unwrap().preview(&p, 23 * SECOND);
        let closed = m.check_question(TaskId(1));
        assert_eq!(closed, [CoreEvent::QuestionClosed { task: TaskId(1), end: End::TimedOut }]);
    }

    #[test]
    fn a_question_whose_end_cannot_be_saved_stays_open_until_it_can() {
        let root = tempfile::tempdir().unwrap();
        let folder = MeetingFolder::create(root.path(), "", Local::now()).unwrap();
        let dir = folder.path().to_path_buf();
        let mut m = Merger::new(folder, "Pam".into(), Arc::default());
        let state = TaskState::new(TaskId(1), "draft the email", 0, Via::Typed, TaskStatus::Running);
        task::create(&dir, &state).unwrap();
        let mut t = Tracked::new(state, None);
        t.asking = Some(Asking {
            question: Question {
                q: 1,
                text: "Which customer?".into(),
                options: vec!["Acme".into(), "Globex".into()],
                timeout: Duration::from_secs(15),
                pid: std::process::id(),
            },
            deadline: Some(Instant::now() + Duration::from_secs(15)),
            cut: None,
        });
        m.tasks.push(t);
        // A directory in its place makes appending to questions.jsonl fail.
        let file = task::questions_file(&dir, TaskId(1));
        std::fs::create_dir_all(file.join("blocked")).unwrap();

        // An answer for the task's earlier question: refused, not taken by this one.
        let err = m.answer(TaskId(1), 0, Answer::Option(2), By::Key, None).unwrap_err().to_string();
        assert_eq!(err, "t01 is asking something else now; that answer was for its earlier question");
        // By key: refused, so the footer says why.
        let err = m.answer(TaskId(1), 1, Answer::Option(2), By::Key, None).unwrap_err();
        assert!(format!("{err:#}").contains("questions.jsonl"), "{err:#}");
        // By voice: kept, and tried again as it was.
        let voice = End::Answered { answer: Answer::Option(1), by: By::Voice, spk: Some(SpkId(2)) };
        let failed = m.close_question(TaskId(1), voice.clone());
        assert!(matches!(&failed[..], [CoreEvent::Error(_)]), "{failed:?}");
        assert!(m.check_question(TaskId(1)).is_empty(), "reported once");
        assert!(m.tasks[0].asking.is_some(), "still open: the asker hasn't seen an end");
        let err = m.answer(TaskId(1), 1, Answer::Option(2), By::Key, None).unwrap_err().to_string();
        assert!(err.contains("has ended"), "no other answer: {err}");

        std::fs::remove_dir_all(&file).unwrap();
        let closed = m.check_question(TaskId(1));
        assert_eq!(closed, [CoreEvent::QuestionClosed { task: TaskId(1), end: voice }]);
        assert!(m.tasks[0].asking.is_none() && m.tasks[0].ending.is_empty());
    }

    #[test]
    fn a_newer_question_waits_until_the_shown_one_has_saved_its_end() {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir().unwrap();
        let folder = MeetingFolder::create(root.path(), "", Local::now()).unwrap();
        let dir = folder.path().to_path_buf();
        let mut m = Merger::new(folder, "Pam".into(), Arc::default());
        let state = TaskState::new(TaskId(1), "draft the email", 0, Via::Typed, TaskStatus::Running);
        task::create(&dir, &state).unwrap();
        m.tasks.push(Tracked::new(state, None));
        let options = ["Acme".to_string(), "Globex".to_string()];
        question::ask(&dir, TaskId(1), "Which customer?", &options, 15).unwrap();
        assert!(matches!(&m.questions(TaskId(1))[..], [CoreEvent::Question { .. }]));

        // Its timeout can't be saved; then a newer question comes.
        let file = task::questions_file(&dir, TaskId(1));
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o444)).unwrap();
        assert!(matches!(&m.close_question(TaskId(1), End::TimedOut)[..], [CoreEvent::Error(_)]));
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o644)).unwrap();
        question::ask(&dir, TaskId(1), "Formal or casual?", &options, 15).unwrap();
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o444)).unwrap();
        assert!(m.questions(TaskId(1)).is_empty(), "the first is still on the stage");
        assert_eq!(m.question_of(TaskId(1)), Some(1));

        // Once its end saves, the first closes and the second shows.
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o644)).unwrap();
        let got = m.check_question(TaskId(1));
        assert!(
            matches!(
                &got[..],
                [CoreEvent::QuestionClosed { end: End::TimedOut, .. }, CoreEvent::Question { text, .. }]
                    if text == "Formal or casual?"
            ),
            "{got:?}"
        );
    }

    #[test]
    fn another_tasks_question_waits_while_one_on_the_stage_saves_its_end() {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir().unwrap();
        let folder = MeetingFolder::create(root.path(), "", Local::now()).unwrap();
        let dir = folder.path().to_path_buf();
        let mut m = Merger::new(folder, "Pam".into(), Arc::default());
        let options = ["Acme".to_string(), "Globex".to_string()];
        for id in [1, 2] {
            let state =
                TaskState::new(TaskId(id), &format!("request {id}"), 0, Via::Typed, TaskStatus::Running);
            task::create(&dir, &state).unwrap();
            m.tasks.push(Tracked::new(state, None));
        }
        question::ask(&dir, TaskId(1), "Which customer?", &options, 15).unwrap();
        assert!(matches!(&m.look_for_questions()[..], [CoreEvent::Question { task: TaskId(1), .. }]));

        // t01's drop can't be saved as its run ends; t02 runs next and asks.
        let t01 = task::questions_file(&dir, TaskId(1));
        std::fs::set_permissions(&t01, std::fs::Permissions::from_mode(0o444)).unwrap();
        let got = m.close_question(TaskId(1), End::Dropped("the run ended".into()));
        assert!(matches!(&got[..], [CoreEvent::Error(_)]), "{got:?}");
        m.tasks[0].state.status = TaskStatus::Done;
        question::ask(&dir, TaskId(2), "Formal or casual?", &options, 15).unwrap();
        assert!(m.look_for_questions().is_empty(), "t01's question is still on the stage");

        std::fs::set_permissions(&t01, std::fs::Permissions::from_mode(0o644)).unwrap();
        let got = m.look_for_questions();
        assert!(
            matches!(
                &got[..],
                [
                    CoreEvent::QuestionClosed { task: TaskId(1), .. },
                    CoreEvent::Question { task: TaskId(2), .. }
                ]
            ),
            "{got:?}"
        );
    }

    #[test]
    fn a_drop_that_cannot_be_saved_before_the_question_shows_is_tried_again() {
        use std::os::unix::fs::PermissionsExt;
        let root = tempfile::tempdir().unwrap();
        let folder = MeetingFolder::create(root.path(), "", Local::now()).unwrap();
        let dir = folder.path().to_path_buf();
        let mut m = Merger::new(folder, "Pam".into(), Arc::default());
        // Asked by a task that has finished: it is dropped without being shown.
        let state = TaskState::new(TaskId(1), "draft the email", 0, Via::Typed, TaskStatus::Done);
        task::create(&dir, &state).unwrap();
        m.tasks.push(Tracked::new(state, None));
        question::ask(&dir, TaskId(1), "Which customer?", &["Acme".into(), "Globex".into()], 15).unwrap();
        // Readable, but appending fails.
        let file = task::questions_file(&dir, TaskId(1));
        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o444)).unwrap();

        let got = m.questions(TaskId(1));
        assert!(matches!(&got[..], [CoreEvent::Error(_)]), "{got:?}");
        assert!(m.check_question(TaskId(1)).is_empty(), "reported once, never shown");

        std::fs::set_permissions(&file, std::fs::Permissions::from_mode(0o644)).unwrap();
        assert!(m.check_question(TaskId(1)).is_empty(), "never shown, so nothing to close");
        assert!(m.tasks[0].ending.is_empty() && m.tasks[0].asking.is_none());
        let end = question::ended(&dir, TaskId(1), 1).unwrap();
        assert_eq!(end, Some(End::Dropped("t01 isn't running".into())), "the waiting asker sees it");
    }

    #[test]
    fn a_correction_is_checked_against_a_request_not_yet_a_task() {
        let root = tempfile::tempdir().unwrap();
        let folder = MeetingFolder::create(root.path(), "", Local::now()).unwrap();
        let mut m = Merger::new(folder, "Pam".into(), Arc::default());
        let asked = Line {
            id: UtteranceId(1),
            t0: SECOND,
            t1: 3 * SECOND,
            slot: None,
            spk: None,
            lang: None,
            text: "Pam, draft the email.".into(),
        };
        m.invocations = Some(Invocations::new("Pam"));
        m.invocations.as_mut().unwrap().line(&asked, 4 * SECOND);
        let p = m.check_prompt("Pam, I meant the launch email.", None, 4 * SECOND, None, None);
        assert!(p.contains("LATEST REQUEST\nnew request (being checked): draft the email.\n"), "{p}");
    }
}
