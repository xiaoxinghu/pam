//! What the core tells front-ends, and what they ask of it.

use std::fmt;
use std::str::FromStr;
use std::time::Instant;

use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::Ts;
use crate::identity::Overlay;
use crate::models::{AgentInfo, IntentInfo, ScribeInfo};
use crate::notes::Notes;
use crate::question::{Answer, End};
use crate::task::TaskState;
use crate::usage::Usage;

/// A permanent speaker ID within one meeting, shown as `spk_N`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SpkId(pub u32);

impl fmt::Display for SpkId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "spk_{}", self.0)
    }
}

impl Serialize for SpkId {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl FromStr for SpkId {
    type Err = String;
    fn from_str(s: &str) -> Result<SpkId, String> {
        number_after(s, "spk_").map(SpkId).ok_or_else(|| format!("`{s}` is not a speaker ID like spk_3"))
    }
}

impl<'de> Deserialize<'de> for SpkId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<SpkId, D::Error> {
        parse(d)
    }
}

/// A final line's stable ID, shown as `u0001`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UtteranceId(pub u32);

impl fmt::Display for UtteranceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "u{:04}", self.0)
    }
}

impl Serialize for UtteranceId {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

/// Identifies one identity op in `speakers.json`, so it can be undone. Shown as `x1`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct UndoId(pub u32);

impl fmt::Display for UndoId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "x{}", self.0)
    }
}

impl Serialize for UndoId {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl FromStr for UndoId {
    type Err = String;
    fn from_str(s: &str) -> Result<UndoId, String> {
        number_after(s, "x").map(UndoId).ok_or_else(|| format!("`{s}` is not an undo ID like x4"))
    }
}

impl<'de> Deserialize<'de> for UndoId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<UndoId, D::Error> {
        parse(d)
    }
}

impl FromStr for UtteranceId {
    type Err = String;
    fn from_str(s: &str) -> Result<UtteranceId, String> {
        number_after(s, "u").map(UtteranceId).ok_or_else(|| format!("`{s}` is not a line ID like u0412"))
    }
}

impl<'de> Deserialize<'de> for UtteranceId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<UtteranceId, D::Error> {
        parse(d)
    }
}

/// An event's stable ID in `events.jsonl`, shown as `e037`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct EventId(pub u32);

impl fmt::Display for EventId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "e{:03}", self.0)
    }
}

impl Serialize for EventId {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl FromStr for EventId {
    type Err = String;
    fn from_str(s: &str) -> Result<EventId, String> {
        number_after(s, "e").map(EventId).ok_or_else(|| format!("`{s}` is not an event ID like e037"))
    }
}

impl<'de> Deserialize<'de> for EventId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<EventId, D::Error> {
        parse(d)
    }
}

/// A task's ID, shown as `t02`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct TaskId(pub u32);

impl fmt::Display for TaskId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "t{:02}", self.0)
    }
}

impl Serialize for TaskId {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.collect_str(self)
    }
}

impl FromStr for TaskId {
    type Err = String;
    fn from_str(s: &str) -> Result<TaskId, String> {
        match s.strip_prefix('t') {
            Some(n) if !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit()) => {
                n.parse().map(TaskId).map_err(|_| format!("`{s}` is too big for a task ID"))
            }
            _ => Err(format!("`{s}` is not a task ID like t02")),
        }
    }
}

impl<'de> Deserialize<'de> for TaskId {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<TaskId, D::Error> {
        parse(d)
    }
}

/// The digits after `prefix`, e.g. 412 for `u0412` with prefix `u`.
fn number_after(s: &str, prefix: &str) -> Option<u32> {
    let digits = s.strip_prefix(prefix)?;
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    digits.parse().ok()
}

fn parse<'de, D: Deserializer<'de>, T: FromStr<Err = String>>(d: D) -> Result<T, D::Error> {
    String::deserialize(d)?.parse().map_err(serde::de::Error::custom)
}

/// A final line: consecutive words from one speaker.
#[derive(Debug, Clone, PartialEq)]
pub struct Line {
    pub id: UtteranceId,
    pub t0: Ts,
    pub t1: Ts,
    /// The tracker slot, or `None` when no speaker could be assigned.
    pub slot: Option<u32>,
    pub spk: Option<SpkId>,
    /// The language the line was spoken in, as the transcriber reported it
    /// (e.g. `"zh-CN"`); `None` when it reported none.
    pub lang: Option<String>,
    pub text: String,
}

/// Speech that isn't a final line yet. Display only, never stored.
#[derive(Debug, Clone, PartialEq)]
pub struct PreviewLine {
    /// `None` for a voice that has no ID yet, or no speaker at all.
    pub spk: Option<SpkId>,
    pub text: String,
    pub t0: Ts,
}

#[derive(Debug, Clone, PartialEq)]
pub enum CoreEvent {
    /// The meeting clock moved on; sent about once a second of audio.
    Clock {
        now: Ts,
    },
    /// Everything not yet final, oldest first; replaces the previous preview.
    LinePreview(Vec<PreviewLine>),
    /// Has a stable `UtteranceId`; already appended to `transcript.jsonl`.
    LineFinal(Line),
    /// Drives the speaking indicator. Level 0 means the speaker stopped.
    SpeakerActive {
        spk: SpkId,
        level: f32,
    },
    SpeakerAdded(SpkId),
    /// `spk` is now called `name`, for example from an introduction. Lines
    /// already shown under `spk` take the name too.
    SpeakerRenamed {
        spk: SpkId,
        name: String,
        undo: UndoId,
    },
    /// `from`'s lines now read as `into`'s, and `from` is no longer shown.
    /// Both are speakers that weren't merged away yet.
    SpeakersMerged {
        from: SpkId,
        into: SpkId,
        undo: UndoId,
    },
    /// `line` now reads as `to`'s.
    LineMoved {
        line: UtteranceId,
        to: SpkId,
        undo: UndoId,
    },
    /// The op `undo` was undone. `overlay` is every name, merge and moved
    /// line now.
    Undone {
        undo: UndoId,
        overlay: Overlay,
    },
    /// `notes.md` changed: the scribe added events or a new summary, or
    /// `pam note` added a note. Carries the whole notes, already sorted into
    /// sections; front-ends apply speaker names when drawing.
    NotesUpdated(Notes),
    /// The scribe's model as the meeting chose and checked it: which
    /// model, how it was chosen, and if it is off or not ready, why. Sent
    /// once, before anything else.
    Scribe(ScribeInfo),
    /// The intent check's model the same way: the scribe's unless one
    /// was named. Sent once, after `Scribe`.
    Intent(IntentInfo),
    /// What the meeting found out about its agent when it started: whether it
    /// takes requests, its model and how it was chosen, and if not, why.
    /// Sent once, after `Intent`.
    Agent(AgentInfo),
    /// The meeting's tokens and cost so far, as `meeting.json`'s `usage`:
    /// sent at the start, and after each model call once it is saved.
    Usage(Box<Usage>),
    /// A spoken request passed the intent check and became `task`. The room
    /// can cancel it until the meeting time reaches `cancel_deadline`; what
    /// the agent says for it is held back until then. It starts before its
    /// line is final, to start sooner: `line` is `None` and `spk` is the tracker's
    /// speaker so far; it is sent again, with the same deadline, once the
    /// line is final.
    Invocation {
        task: TaskId,
        line: Option<UtteranceId>,
        spk: Option<SpkId>,
        summary: String,
        cancel_deadline: Ts,
    },
    /// A task was created, its status changed, or the agent reported for it
    /// (`pam task update`, `pam task result`); already saved in `task.json`.
    /// A result never takes the stage: the chip turns ready once the task is
    /// done, so a result never grabs the room's attention mid-discussion.
    TaskUpdated(TaskState),
    /// The agent asks the room (`pam task ask`): `options`, numbered
    /// from 1, or none for an open question, until `deadline`. `q` is
    /// its number among the task's questions, which an `Answer` names. Like
    /// what the task says, it is held until the task's cancel window
    /// closes, and the countdown starts then.
    Question {
        task: TaskId,
        q: u32,
        text: String,
        options: Vec<String>,
        deadline: Instant,
    },
    /// The task's question is off the stage: the room answered it, the
    /// countdown ran out, or nobody waits for the answer any more. Already
    /// saved in its `questions.jsonl`.
    QuestionClosed {
        task: TaskId,
        end: End,
    },
    /// Something to show on the stage, from `pam say`. It replaces what the
    /// same task showed before, so the agent can correct itself. `at` is when
    /// the room could first see it, in meeting time, so a subscriber that
    /// joins later and gets it replayed still knows how old it is.
    StageMessage {
        task: Option<TaskId>,
        text: String,
        at: Ts,
    },
    /// The input ran out or the meeting was stopped. Every line is final now.
    InputEnded {
        now: Ts,
    },
    /// Something failed; the meeting may have stopped listening.
    Error(String),
}

/// What a front-end asks the core to do, through `Meeting::send`.
#[derive(Debug, Clone, PartialEq)]
pub enum Command {
    /// Name a speaker, replacing any name they have.
    RenameSpeaker { spk: SpkId, name: String },
    /// Make `from` part of `into`, when the tracker split one voice in two.
    MergeSpeakers { from: SpkId, into: SpkId },
    /// Give one line to another speaker.
    MoveLine { line: UtteranceId, to: SpkId },
    /// Undo an identity op: a rename, merge or move.
    Undo(UndoId),
    /// A typed request, always treated as an instruction: it becomes a task.
    Ask(String),
    /// Stop a task that hasn't finished, or one whose cancel window is open.
    CancelTask(TaskId),
    /// Answer question `q` of `task`, the one on the stage: with an option,
    /// numbered from 1 as on the stage, by key; or an open question with the
    /// words typed in the composer. Refused once the task asks another.
    Answer { task: TaskId, q: u32, answer: Answer },
}
