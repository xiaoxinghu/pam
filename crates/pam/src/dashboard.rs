//! The terminal dashboard: header, speaker strip, then transcript, stage and
//! notes side by side.
//!
//! It only keeps what it needs to draw the core's events. Its own state is
//! whether the transcript is shown, which line or speaker is selected, an open
//! correction prompt or composer, which task is on the stage, which card or
//! ready chip is focused, the result opened with `Enter`, and a
//! one-line notice in the footer. A spoken request's cancel window pops up
//! over the stage until its deadline, and so does a question the agent asks
//! the room, answered with the keys 1–9. Keys become [`Action`]s;
//! corrections, typed requests and answers go to the core as [`Command`]s,
//! and `o` asks the caller to open a result.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use pam_core::clock::{SECOND, mmss};
use pam_core::question::{Answer, By, End};
use pam_core::{
    AgentInfo, Command, CoreEvent, IntentInfo, Line, ModelInfo, ModelStatus, Notes, Overlay, PreviewLine,
    ScribeInfo, Source, SpkId, Target, TaskId, TaskState, TaskStatus, Ts, UndoId, Usage, UtteranceId, task,
};
use ratatui::Frame;
use ratatui::crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use ratatui::layout::{Constraint, Layout, Rect};
use ratatui::style::{Color, Modifier, Style};
use ratatui::text::{Line as TextLine, Span};
use ratatui::widgets::{Block, BorderType, Clear, Paragraph};
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

/// The dashboard's dark terminal palette.
mod palette {
    use ratatui::style::Color;
    pub const FG: Color = Color::Rgb(0xD9, 0xDB, 0xE0);
    pub const DIM: Color = Color::Rgb(0x8D, 0x93, 0xA0);
    pub const FAINT: Color = Color::Rgb(0x5A, 0x60, 0x6C);
    pub const BORDER: Color = Color::Rgb(0x34, 0x39, 0x44);
    pub const HEADER: Color = Color::Rgb(0x1F, 0x22, 0x29);
    pub const ACCENT: Color = Color::Rgb(0x8E, 0xA6, 0xFF);
    pub const OK: Color = Color::Rgb(0x4C, 0xC3, 0x8A);
    pub const WARN: Color = Color::Rgb(0xE7, 0xB2, 0x4C);
    pub const BAD: Color = Color::Rgb(0xF0, 0x64, 0x5A);
    pub const ON: Color = Color::Rgb(0x15, 0x17, 0x1C);
    pub const SPEAKERS: [Color; 6] = [
        Color::Rgb(0xEE, 0x8A, 0x55),
        Color::Rgb(0x55, 0xAE, 0xE3),
        Color::Rgb(0xB8, 0x92, 0xF2),
        Color::Rgb(0x43, 0xC2, 0xB0),
        Color::Rgb(0xE0, 0x6C, 0x9F),
        Color::Rgb(0x9B, 0xC5, 0x3D),
    ];
}
use palette::*;

const LEVELS: [char; 8] = ['▁', '▂', '▃', '▄', '▅', '▆', '▇', '█'];
const TRANSCRIPT_WIDTH: u16 = 42;
const NOTES_WIDTH: u16 = 40;

/// One entry in the transcript pane.
enum Row {
    Line(Line),
    /// An identity op, such as "spk_3 is now Sam" right after the line that
    /// introduced Sam. `note` is dropped once the op is undone.
    Op {
        undo: UndoId,
        text: String,
        note: Option<&'static str>,
        undone: bool,
    },
    /// Something that happened to a task, such as "t02 · option 2 chosen by Jordan".
    Note(String),
}

/// What a key asks of the caller.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    None,
    Quit,
    Send(Command),
    /// Open a result, or what a line of it points at, with the system's
    /// default app: an absolute path or a URL.
    Open(String),
}

/// A correction being made, shown in the footer.
#[derive(Debug, Clone, PartialEq)]
enum Prompt {
    /// Typing a new name for `spk`.
    Rename { spk: SpkId, name: String },
    /// Choosing which speaker `from` joins.
    Merge { from: SpkId, into: SpkId },
    /// Choosing who said `line`.
    Move { line: UtteranceId, to: SpkId },
    /// Typing in the composer: a request, or the answer to the open question
    /// `answering` names (task, question number), as it was when the composer
    /// opened. An answer whose question has closed meanwhile is refused,
    /// never sent as a request, nor taken by the task's next question.
    Ask { text: String, answering: Option<(TaskId, u32)> },
}

pub struct Dashboard {
    title: String,
    date: String,
    now: Ts,
    ended: bool,
    error: Option<String>,
    rows: Vec<Row>,
    lines: usize,
    preview: Vec<PreviewLine>,
    /// Every ID given out, merged-away ones included.
    speakers: Vec<SpkId>,
    /// Names, merges and moved lines, kept in step with the core.
    overlay: Overlay,
    /// Identity ops not undone yet, oldest first: `u` undoes the last.
    undoable: Vec<UndoId>,
    /// The line picked with ↑↓, if any.
    selected_line: Option<UtteranceId>,
    /// The speaker picked with ←→. A selected line's speaker wins over it.
    selected_spk: Option<SpkId>,
    prompt: Option<Prompt>,
    notice: Option<String>,
    /// Speakers talking now, with their level.
    active: BTreeMap<SpkId, f32>,
    /// Who spoke last, for "Listening to …".
    last_active: Option<SpkId>,
    notes: Notes,
    /// The scribe's model as the meeting chose it; `None` until it says (a
    /// meeting that chose none takes notes if it can).
    scribe: Option<ScribeInfo>,
    /// The intent check's model the same way; `None` until it says (a
    /// meeting that chose none checks on the scribe's model).
    intent: Option<IntentInfo>,
    show_transcript: bool,
    frame: u64,
    /// The agent as the meeting found it; `None` for a meeting without one.
    agent: Option<AgentInfo>,
    /// The meeting's cost so far; `None` until the meeting says.
    usage: Option<Usage>,
    /// Every task, oldest first.
    tasks: Vec<TaskState>,
    /// What the agent said last on the stage, per task: saying again replaces it.
    said: BTreeMap<TaskId, String>,
    /// The answer on the stage, and the meeting time it was given: a task's
    /// `pam say`, its failure or its cancel. `Esc` takes it off; the Working
    /// cards take over from it once it is stale.
    staged: Option<(TaskId, Ts)>,
    /// The Working card or ready chip `Tab` focused, while it can be.
    focus: Option<TaskId>,
    /// The result `Enter` opened, until `Esc` or a new answer.
    opened: Option<Opened>,
    /// The meeting folder, where a result's file is.
    folder: Option<PathBuf>,
    /// Spoken requests, by task: the line that asked, what the intent check
    /// made of it, and when its cancel window closes.
    invocations: BTreeMap<TaskId, Invoked>,
    /// The assistant's name, which is also its wake word.
    name: String,
    /// The questions the agent is asking the room, by task.
    questions: BTreeMap<TaskId, Asked>,
    /// The time the countdowns are drawn at, set by [`Dashboard::tick`].
    wall: Instant,
}

/// A question the agent asks the room, from its `Question`.
struct Asked {
    /// Its number among the task's questions, which an answer names.
    q: u32,
    text: String,
    options: Vec<String>,
    /// When it was shown, and when its countdown ends.
    opened: Instant,
    deadline: Instant,
}

/// A result opened on the stage.
struct Opened {
    task: TaskId,
    /// A Markdown or plain-text file's lines, read when it was opened.
    text: Option<Vec<String>>,
    /// Why its text couldn't be read.
    unread: Option<String>,
    /// The line of `text` selected with ↑↓; `None` for the result itself.
    line: Option<usize>,
}

/// The most lines of a result shown.
const RESULT_LINES: usize = 5000;

/// A spoken request, from its `Invocation`.
struct Invoked {
    /// Its line, once final; until then, who the tracker says asked.
    line: Option<UtteranceId>,
    spk: Option<SpkId>,
    summary: String,
    /// The meeting time when the window opened, and when it closes.
    opened: Ts,
    deadline: Ts,
}

/// What the stage's body shows ([`Dashboard::stage`]).
#[derive(Debug, Clone, Copy, PartialEq)]
enum Stage {
    /// A result someone opened.
    Result(TaskId),
    /// A task's answer.
    Answer(TaskId),
    /// A card per task running or waiting.
    Working,
    Listening,
}

/// How long an answer keeps the stage from the Working cards: enough to
/// read a few plain sentences.
const FRESH: Ts = 30 * SECOND;

/// The running chip's and the "Thinking…" spinner's frames.
const SPIN: [char; 10] = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏'];

impl Dashboard {
    pub fn new(title: String, date: String) -> Dashboard {
        Dashboard {
            title,
            date,
            now: 0,
            ended: false,
            error: None,
            rows: Vec::new(),
            lines: 0,
            preview: Vec::new(),
            speakers: Vec::new(),
            overlay: Overlay::default(),
            undoable: Vec::new(),
            selected_line: None,
            selected_spk: None,
            prompt: None,
            notice: None,
            active: BTreeMap::new(),
            last_active: None,
            notes: Notes::default(),
            scribe: None,
            intent: None,
            show_transcript: true,
            frame: 0,
            agent: None,
            usage: None,
            tasks: Vec::new(),
            said: BTreeMap::new(),
            staged: None,
            focus: None,
            opened: None,
            folder: None,
            invocations: BTreeMap::new(),
            name: "Pam".into(),
            questions: BTreeMap::new(),
            wall: Instant::now(),
        }
    }

    /// The assistant's name, as the meeting uses it.
    pub fn set_name(&mut self, name: String) {
        self.name = name;
    }

    /// The meeting folder, where a result's file is.
    pub fn set_folder(&mut self, folder: PathBuf) {
        self.folder = Some(folder);
    }

    pub fn apply(&mut self, event: CoreEvent) {
        let before = self.overlay.clone();
        self.overlay.apply(&event);
        match event {
            CoreEvent::Clock { now } => self.now = self.now.max(now),
            CoreEvent::LinePreview(p) => self.preview = p,
            CoreEvent::LineFinal(l) => {
                self.lines += 1;
                self.rows.push(Row::Line(l));
            }
            CoreEvent::SpeakerActive { spk, level } => {
                if level > 0.0 {
                    self.active.insert(spk, level);
                    self.last_active = Some(spk);
                } else {
                    self.active.remove(&spk);
                }
            }
            CoreEvent::SpeakerAdded(spk) => self.speakers.push(spk),
            CoreEvent::SpeakerRenamed { spk, name, undo } => {
                let said = self.lines_by(spk);
                let note = (said > 1).then_some("earlier lines updated");
                self.op(undo, format!("{} is now {name}", before.label(Some(spk))), note);
            }
            CoreEvent::SpeakersMerged { from, into, undo } => {
                let text = format!("{} merged into {}", before.label(Some(from)), before.label(Some(into)));
                self.op(undo, text, Some("lines updated"));
            }
            CoreEvent::LineMoved { line, to, undo } => {
                let was = self.line(line).map(|l| before.label(before.speaker(l.id, l.spk)));
                let was = was.map_or_else(String::new, |w| format!(" from {w}"));
                self.op(undo, format!("{line} moved{was} to {}", before.label(Some(to))), None);
            }
            CoreEvent::Undone { undo, .. } => {
                self.undoable.retain(|u| *u != undo);
                for r in &mut self.rows {
                    if let Row::Op { undo: u, undone, .. } = r
                        && *u == undo
                    {
                        *undone = true;
                    }
                }
            }
            CoreEvent::InputEnded { now } => {
                self.now = self.now.max(now);
                self.ended = true;
            }
            CoreEvent::NotesUpdated(n) => self.notes = n,
            CoreEvent::Scribe(info) => self.scribe = Some(info),
            CoreEvent::Intent(info) => self.intent = Some(info),
            CoreEvent::Agent(info) => self.agent = Some(info),
            CoreEvent::Usage(usage) => self.usage = Some(*usage),
            CoreEvent::Invocation { task, line, spk, summary, cancel_deadline } => {
                // Sent again once its line is final: the window it opened stays.
                let opened = self.invocations.get(&task).map_or(self.now.min(cancel_deadline), |i| i.opened);
                self.invocations
                    .insert(task, Invoked { line, spk, summary, opened, deadline: cancel_deadline });
            }
            CoreEvent::Question { task, q, text, options, deadline } => {
                let opened = self.wall.min(deadline);
                self.questions.insert(task, Asked { q, text, options, opened, deadline });
            }
            CoreEvent::QuestionClosed { task, end } => {
                let Some(asked) = self.questions.remove(&task) else { return };
                let chosen = |option: usize, how: String| {
                    let text = asked.options.get(option.wrapping_sub(1)).map_or("", String::as_str);
                    format!("{task} · option {option} chosen {how}: {text}")
                };
                let how = |by: By, spk: Option<SpkId>| match (by, spk) {
                    (By::Key | By::Typed, _) => "on the laptop".to_string(),
                    (By::Voice, Some(s)) => format!("by {}", self.label(s)),
                    (By::Voice, None) => "by voice".to_string(),
                };
                let note = match end {
                    End::Answered { answer: Answer::Option(option), by, spk } => {
                        Some(chosen(option, how(by, spk)))
                    }
                    End::Answered { answer: Answer::Text(text), by, spk } => {
                        Some(format!("{task} · answered {}: {text}", how(by, spk)))
                    }
                    End::TimedOut => {
                        Some(format!("{task} · no answer in time: {} goes with its best guess", self.name))
                    }
                    End::Dropped(_) => None,
                };
                if let Some(note) = note {
                    self.rows.push(Row::Note(note));
                }
            }
            CoreEvent::TaskUpdated(t) => {
                // A failure or a cancel is the task's answer; a start, with
                // nothing to read yet, isn't.
                // Timed by the core, as a replay may come before the clock.
                let was = self.tasks.iter().find(|o| o.id == t.id);
                let (was_status, was_ready) = (was.map(|o| o.status), was.is_some_and(TaskState::is_ready));
                if matches!(t.status, TaskStatus::Failed | TaskStatus::Cancelled)
                    && was_status != Some(t.status)
                {
                    self.staged = Some((t.id, t.ended_at.unwrap_or(self.now)));
                    self.opened = None;
                }
                // Ready: said in the transcript, never shown on the stage, so
                // a result doesn't take over mid-discussion.
                if let Some(r) = t.result.as_ref().filter(|_| t.is_ready() && !was_ready) {
                    self.rows.push(Row::Note(format!("{} · ready: {}", t.id, r.title)));
                }
                match self.tasks.iter_mut().find(|o| o.id == t.id) {
                    Some(old) => *old = t,
                    None => self.tasks.push(t),
                }
            }
            CoreEvent::StageMessage { task, text, at } => {
                if let Some(id) = task {
                    self.said.insert(id, text);
                    self.staged = Some((id, at));
                    // A direct answer shows at once; a result's verdict waits in it.
                    if !self.quiet(id) {
                        self.opened = None;
                    }
                }
            }
            CoreEvent::Error(e) => self.error = Some(e),
        }
    }

    fn op(&mut self, undo: UndoId, text: String, note: Option<&'static str>) {
        self.undoable.push(undo);
        self.rows.push(Row::Op { undo, text, note, undone: false });
    }

    /// How many final lines `spk` has, corrections applied.
    fn lines_by(&self, spk: SpkId) -> usize {
        self.final_lines().filter(|l| self.speaker_of(l) == Some(spk)).count()
    }

    fn final_lines(&self) -> impl Iterator<Item = &Line> {
        self.rows.iter().filter_map(|r| match r {
            Row::Line(l) => Some(l),
            Row::Op { .. } | Row::Note(_) => None,
        })
    }

    fn line(&self, id: UtteranceId) -> Option<&Line> {
        self.final_lines().find(|l| l.id == id)
    }

    fn speaker_of(&self, l: &Line) -> Option<SpkId> {
        self.overlay.speaker(l.id, l.spk)
    }

    /// The op `u` undoes: the latest one still in force.
    pub fn last_undo(&self) -> Option<UndoId> {
        self.undoable.last().copied()
    }

    /// Turns a key into what the caller should do. Corrections act on the
    /// selection: ↑↓ picks a line, ←→ a speaker.
    pub fn key(&mut self, key: KeyEvent) -> Action {
        self.notice = None;
        if key.code == KeyCode::Char('c') && key.modifiers.contains(KeyModifiers::CONTROL) {
            return Action::Quit;
        }
        if let Some(prompt) = self.prompt.take() {
            return self.prompt_key(prompt, key.code);
        }
        if let KeyCode::Char(c @ '1'..='9') = key.code
            && let Some((id, asked)) = self.question()
        {
            let option = c as usize - '0' as usize;
            let n = asked.options.len();
            if n == 0 {
                self.notice(format!("{id}'s question is open: type the answer with /."));
                return Action::None;
            }
            if option > n {
                self.notice(format!("{id}'s question has options 1–{n}."));
                return Action::None;
            }
            return Action::Send(Command::Answer { task: id, q: asked.q, answer: Answer::Option(option) });
        }
        let result = match self.stage() {
            Stage::Result(id) => Some(id),
            _ => None,
        };
        match key.code {
            KeyCode::Char('q') => return Action::Quit,
            KeyCode::Char('t') => self.toggle_transcript(),
            KeyCode::Up if result.is_some() => self.step_result_line(-1),
            KeyCode::Down if result.is_some() => self.step_result_line(1),
            KeyCode::Up => self.step_line(-1),
            KeyCode::Down => self.step_line(1),
            KeyCode::Left => self.step_speaker(-1),
            KeyCode::Right => self.step_speaker(1),
            KeyCode::Char('/') => {
                self.prompt = Some(Prompt::Ask { text: String::new(), answering: self.open_question() })
            }
            KeyCode::Esc if self.window().is_some() => {
                let id = self.window().map(|(t, _)| t.id).unwrap();
                return Action::Send(Command::CancelTask(id));
            }
            KeyCode::Esc if result.is_some() => self.opened = None,
            KeyCode::Esc if self.selected_line.is_none() && self.selected_spk.is_none() => self.staged = None,
            KeyCode::Esc => (self.selected_line, self.selected_spk) = (None, None),
            KeyCode::Tab => self.next_focus(),
            KeyCode::Enter if result.is_none() => match self.focused().filter(|id| self.is_ready(*id)) {
                Some(id) => self.open_result(id),
                None => self.notice("No result to show: Tab to a ready chip, then press Enter."),
            },
            KeyCode::Char('o') => match result.or(self.focused().filter(|id| self.is_ready(*id))) {
                Some(id) => match self.target(id) {
                    Some(target) => return Action::Open(target),
                    None => self.notice(format!("{id} has no result to open.")),
                },
                None => self.notice("No result to open: Tab to a ready chip, then press o."),
            },
            KeyCode::Char('r') => match self.selected() {
                Some(spk) => self.prompt = Some(Prompt::Rename { spk, name: String::new() }),
                None => self.notice("Pick a speaker with ←→, or a line with ↑↓, then press r."),
            },
            KeyCode::Char('m') => self.start_merge_or_move(),
            KeyCode::Char('u') => match self.last_undo() {
                Some(undo) => return Action::Send(Command::Undo(undo)),
                None => self.notice("Nothing to undo."),
            },
            _ => {}
        }
        Action::None
    }

    fn prompt_key(&mut self, prompt: Prompt, code: KeyCode) -> Action {
        if code == KeyCode::Esc {
            return Action::None;
        }
        let step = match code {
            KeyCode::Left => -1,
            KeyCode::Right => 1,
            _ => 0,
        };
        let cmd = match (prompt, code) {
            (Prompt::Rename { spk, name }, KeyCode::Enter) => Command::RenameSpeaker { spk, name },
            (Prompt::Rename { spk, mut name }, KeyCode::Backspace) => {
                name.pop();
                self.prompt = Some(Prompt::Rename { spk, name });
                return Action::None;
            }
            (Prompt::Rename { spk, mut name }, KeyCode::Char(c)) => {
                name.push(c);
                self.prompt = Some(Prompt::Rename { spk, name });
                return Action::None;
            }
            (Prompt::Ask { text, .. }, KeyCode::Enter) if text.trim().is_empty() => return Action::None,
            // Opened while an open question was up, what is typed answers it.
            (Prompt::Ask { text, answering }, KeyCode::Enter) => match answering {
                Some((task, q)) => Command::Answer { task, q, answer: Answer::Text(text) },
                None => Command::Ask(text),
            },
            (Prompt::Ask { mut text, answering }, KeyCode::Backspace) => {
                text.pop();
                self.prompt = Some(Prompt::Ask { text, answering });
                return Action::None;
            }
            (Prompt::Ask { mut text, answering }, KeyCode::Char(c)) => {
                text.push(c);
                self.prompt = Some(Prompt::Ask { text, answering });
                return Action::None;
            }
            (Prompt::Merge { from, into }, KeyCode::Enter) => {
                self.selected_spk = Some(into);
                Command::MergeSpeakers { from, into }
            }
            (Prompt::Move { line, to }, KeyCode::Enter) => Command::MoveLine { line, to },
            (Prompt::Merge { from, into }, _) => {
                let into = self.other_speaker(into, Some(from), step).unwrap_or(into);
                self.prompt = Some(Prompt::Merge { from, into });
                return Action::None;
            }
            (Prompt::Move { line, to }, _) => {
                let current = self.line(line).and_then(|l| self.speaker_of(l));
                let to = self.other_speaker(to, current, step).unwrap_or(to);
                self.prompt = Some(Prompt::Move { line, to });
                return Action::None;
            }
            (prompt, _) => {
                self.prompt = Some(prompt);
                return Action::None;
            }
        };
        Action::Send(cmd)
    }

    /// `m` moves the selected line, or else merges the selected speaker.
    fn start_merge_or_move(&mut self) {
        if let Some(line) = self.selected_line {
            let current = self.line(line).and_then(|l| self.speaker_of(l));
            let to = match current {
                Some(c) => self.other_speaker(c, Some(c), 1),
                None => self.visible().first().copied(),
            };
            match to {
                Some(to) => self.prompt = Some(Prompt::Move { line, to }),
                None => self.notice("There is no other speaker to move it to."),
            }
            return;
        }
        let Some(from) = self.selected() else {
            return self.notice("Pick a speaker with ←→ to merge, or a line with ↑↓ to move, then press m.");
        };
        match self.other_speaker(from, Some(from), 1) {
            Some(into) => self.prompt = Some(Prompt::Merge { from, into }),
            None => self.notice("There is no other speaker to merge into."),
        }
    }

    /// Speakers shown in the strip: everyone not merged away, in ID order.
    fn visible(&self) -> Vec<SpkId> {
        self.speakers.iter().copied().filter(|&s| self.overlay.follow(s) == s).collect()
    }

    /// The selected line's speaker as it reads now, else the picked
    /// speaker, after merges.
    fn selected(&self) -> Option<SpkId> {
        match self.selected_line.and_then(|id| self.line(id)) {
            Some(l) => self.speaker_of(l),
            None => self.selected_spk.map(|s| self.overlay.follow(s)),
        }
    }

    /// The visible speaker `step` places from `from` in ID order, wrapping
    /// around and never `not`. A `from` that isn't a choice sits between its
    /// neighbours.
    fn other_speaker(&self, from: SpkId, not: Option<SpkId>, step: isize) -> Option<SpkId> {
        let all: Vec<SpkId> = self.visible().into_iter().filter(|&s| Some(s) != not).collect();
        if all.is_empty() {
            return None;
        }
        let i = match all.binary_search(&from) {
            Ok(i) => i as isize + step,
            Err(i) if step > 0 => i as isize + step - 1,
            Err(i) => i as isize + step,
        };
        Some(all[i.rem_euclid(all.len() as isize) as usize])
    }

    /// Moves the line selection; ↓ past the last line goes back to following the meeting.
    fn step_line(&mut self, step: isize) {
        let ids: Vec<UtteranceId> = self.final_lines().map(|l| l.id).collect();
        let Some(&last) = ids.last() else { return };
        let next = match self.selected_line.and_then(|s| ids.iter().position(|&i| i == s)) {
            None if step < 0 => Some(last),
            None => None,
            Some(i) => {
                let j = i as isize + step;
                if j < 0 { Some(ids[0]) } else { ids.get(j as usize).copied() }
            }
        };
        // Leaving the lines keeps their last speaker picked.
        self.selected_spk = self.selected();
        self.selected_line = next;
    }

    fn step_speaker(&mut self, step: isize) {
        let current = self.selected();
        self.selected_line = None;
        let all = self.visible();
        if all.is_empty() {
            return;
        }
        self.selected_spk = match current {
            Some(s) => self.other_speaker(s, None, step),
            None if step < 0 => all.last().copied(),
            None => all.first().copied(),
        };
    }

    /// Says in the footer why `cmd` failed.
    pub fn failed(&mut self, cmd: &Command, e: &anyhow::Error) {
        let what = match cmd {
            Command::RenameSpeaker { .. } => "rename",
            Command::MergeSpeakers { .. } => "merge",
            Command::MoveLine { .. } => "move the line",
            Command::Undo(_) => "undo",
            Command::Ask(_) => "ask",
            Command::CancelTask(_) => "cancel",
            Command::Answer { .. } => "answer",
        };
        self.notice(format!("Can't {what}: {e:#}"));
    }

    /// `o` couldn't open `target`: says why in the footer.
    pub fn open_failed(&mut self, target: &str, why: &str) {
        self.notice(format!("Can't open {target}: {why}"));
    }

    /// Shows a one-line message in the footer until the next key.
    fn notice(&mut self, text: impl Into<String>) {
        self.notice = Some(text.into());
    }

    fn label(&self, spk: SpkId) -> String {
        self.overlay.label(Some(spk))
    }

    /// The spoken request whose cancel window is open, newest first.
    fn window(&self) -> Option<(&TaskState, &Invoked)> {
        self.invocations.keys().rev().find_map(|id| {
            let t = self.tasks.iter().find(|t| t.id == *id)?;
            self.window_of(t).map(|inv| (t, inv))
        })
    }

    /// `t`'s cancel window, while it is open: `t` was spoken, hasn't
    /// finished, and the meeting hasn't reached its deadline.
    fn window_of(&self, t: &TaskState) -> Option<&Invoked> {
        let inv = self.invocations.get(&t.id)?;
        (!self.ended && t.status.is_open() && self.now < inv.deadline).then_some(inv)
    }

    /// The question on the stage: the newest one asked.
    fn question(&self) -> Option<(TaskId, &Asked)> {
        self.questions.iter().next_back().map(|(id, q)| (*id, q))
    }

    /// The task whose open question is on the stage, and not behind
    /// a cancel window: the composer answers it.
    fn open_question(&self) -> Option<(TaskId, u32)> {
        if self.window().is_some() {
            return None;
        }
        self.question().filter(|(_, a)| a.options.is_empty()).map(|(id, a)| (id, a.q))
    }

    /// What the stage's body shows, under a cancel window or a question: a
    /// result someone opened, then a fresh answer, then the Working cards
    /// while a task runs or waits, then a stale answer until `Esc`, then
    /// Listening. A task with a result has no answer of its own: what it
    /// said is its verdict, shown with the result.
    fn stage(&self) -> Stage {
        if let Some(o) = &self.opened
            && self.tasks.iter().any(|t| t.id == o.task && t.result.is_some())
        {
            return Stage::Result(o.task);
        }
        let working = self.tasks.iter().any(|t| t.status.is_open());
        match self.staged.filter(|(id, _)| !self.quiet(*id)) {
            Some((id, at)) if !working || self.now < at + FRESH => Stage::Answer(id),
            _ if working => Stage::Working,
            _ => Stage::Listening,
        }
    }

    /// Whether what task `id` said waits in its result rather than taking
    /// the stage: it has a result, and hasn't failed or been cancelled.
    fn quiet(&self, id: TaskId) -> bool {
        self.tasks.iter().any(|t| {
            t.id == id
                && t.result.is_some()
                && !matches!(t.status, TaskStatus::Failed | TaskStatus::Cancelled)
        })
    }

    fn is_ready(&self, id: TaskId) -> bool {
        self.tasks.iter().any(|t| t.id == id && t.is_ready())
    }

    /// The tasks with a Working card: running, then queued, oldest first.
    fn cards(&self) -> Vec<&TaskState> {
        let running = self.tasks.iter().filter(|t| t.status == TaskStatus::Running);
        running.chain(self.tasks.iter().filter(|t| t.status == TaskStatus::Queued)).collect()
    }

    /// What `Tab` moves through: the Working cards while they are on the
    /// stage, then the ready chips, oldest first.
    fn focusable(&self) -> Vec<TaskId> {
        let cards = match self.stage() {
            Stage::Working => self.cards().iter().map(|t| t.id).collect(),
            _ => Vec::new(),
        };
        let ready = self.tasks.iter().filter(|t| t.is_ready()).map(|t| t.id);
        cards.into_iter().chain(ready).collect()
    }

    /// The focused card or chip: the one `Tab` chose, else the first card,
    /// else the newest ready chip.
    fn focused(&self) -> Option<TaskId> {
        let ids = self.focusable();
        let first = match self.stage() {
            Stage::Working => self.cards().first().map(|t| t.id),
            _ => ids.last().copied(),
        };
        self.focus.filter(|f| ids.contains(f)).or(first)
    }

    /// `Tab`: focuses the next Working card or ready chip.
    fn next_focus(&mut self) {
        let ids = self.focusable();
        let at = self.focused().and_then(|f| ids.iter().position(|&id| id == f)).unwrap_or(0);
        self.focus = ids.get((at + 1) % ids.len().max(1)).copied();
    }

    /// Opens task `id`'s result on the stage, reading a Markdown
    /// or plain-text file's lines to show under it.
    fn open_result(&mut self, id: TaskId) {
        let Some(r) = self.tasks.iter().find(|t| t.id == id).and_then(|t| t.result.clone()) else { return };
        let (mut text, mut unread) = (None, None);
        if let Target::File(f) = &r.target
            && is_text(f)
        {
            let path = task::file_path(self.folder.as_deref().unwrap_or(Path::new("")), f);
            match std::fs::read(&path) {
                Ok(bytes) => {
                    let all = String::from_utf8_lossy(&bytes);
                    text = Some(all.lines().take(RESULT_LINES).map(|l| l.replace('\t', "    ")).collect());
                }
                Err(e) => unread = Some(format!("Can't read {f}: {e}")),
            }
        }
        self.opened = Some(Opened { task: id, text, unread, line: None });
    }

    /// ↑↓ in an open result: selects the next or previous line of its
    /// text; above the first is the result itself.
    fn step_result_line(&mut self, step: isize) {
        let Some(o) = self.opened.as_mut() else { return };
        let n = o.text.as_ref().map_or(0, Vec::len);
        o.line = match (o.line, step) {
            (None, 1) if n > 0 => Some(0),
            (None, _) => None,
            (Some(0), -1) => None,
            (Some(i), -1) => Some(i - 1),
            (Some(i), _) => Some((i + 1).min(n.saturating_sub(1))),
        };
    }

    /// What `o` opens for task `id`: in its open result, what the selected
    /// line points at, if anything; else the result itself. A file is
    /// given as an absolute path; a relative link is from the result's
    /// folder, else the meeting folder.
    fn target(&self, id: TaskId) -> Option<String> {
        let r = self.tasks.iter().find(|t| t.id == id)?.result.as_ref()?;
        let folder = self.folder.as_deref().unwrap_or(Path::new(""));
        let link = self.opened.as_ref().filter(|o| o.task == id).and_then(|o| {
            let line = o.text.as_ref()?.get(o.line?)?;
            task::link_on(line)
        });
        let Some(link) = link else {
            return Some(match &r.target {
                Target::Url(u) => u.clone(),
                Target::File(f) => task::file_path(folder, f).display().to_string(),
            });
        };
        let Some(link) = task::link_file(&link) else {
            return Some(link);
        };
        let link = link.as_str();
        if link.is_empty() {
            return Some(match &r.target {
                Target::Url(u) => u.clone(),
                Target::File(f) => task::file_path(folder, f).display().to_string(),
            });
        }
        let beside = match &r.target {
            Target::File(f) => task::file_path(folder, f).parent().map(|d| d.join(link)),
            Target::Url(_) => None,
        };
        let path = beside.filter(|p| p.exists()).unwrap_or_else(|| task::file_path(folder, link));
        Some(path.display().to_string())
    }

    fn unnamed(&self) -> usize {
        self.visible().into_iter().filter(|&s| self.overlay.name(s).is_none()).count()
    }

    pub fn toggle_transcript(&mut self) {
        self.show_transcript = !self.show_transcript;
    }

    /// Advances animations by one frame, and countdowns to now.
    pub fn tick(&mut self) {
        self.frame += 1;
        self.wall = Instant::now();
    }

    pub fn draw(&self, f: &mut Frame) {
        let [header, _, strip, _, panes, composer, footer] = Layout::vertical([
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Length(1),
            Constraint::Fill(1),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .areas(f.area());
        self.draw_header(f, header);
        self.draw_strip(f, strip.inner(ratatui::layout::Margin::new(1, 0)));

        let (mut tw, mut nw) = (if self.show_transcript { TRANSCRIPT_WIDTH } else { 0 }, NOTES_WIDTH);
        if panes.width < tw + nw + 36 {
            tw = if self.show_transcript { panes.width * 28 / 100 } else { 0 };
            nw = panes.width * 26 / 100;
        }
        let [transcript, stage, notes] =
            Layout::horizontal([Constraint::Length(tw), Constraint::Fill(1), Constraint::Length(nw)])
                .areas(panes);
        if self.show_transcript {
            self.draw_transcript(f, transcript);
        }
        self.draw_stage(f, stage);
        self.draw_notes(f, notes);
        if let Some((t, inv)) = self.window() {
            self.draw_cancel_window(f, stage, t, inv);
        } else if let Some((id, asked)) = self.question() {
            self.draw_question(f, stage, id, asked);
        }
        f.render_widget(Paragraph::new(self.composer()), composer.inner(ratatui::layout::Margin::new(1, 0)));
        let mut footer_line = self.footer();
        if let Some(n) = &self.notice {
            footer_line.push(Span::styled(format!("     {n}"), Style::new().fg(WARN)));
        }
        f.render_widget(
            Paragraph::new(TextLine::from(footer_line)),
            footer.inner(ratatui::layout::Margin::new(1, 0)),
        );
    }

    /// The composer: `/ Ask Pam…`, `/ Type the answer…` while an open
    /// question is up, or what is being typed.
    fn composer(&self) -> TextLine<'static> {
        let slash = Span::styled("/ ", Style::new().fg(ACCENT).bold());
        match &self.prompt {
            Some(Prompt::Ask { text, .. }) => TextLine::from(vec![
                slash,
                Span::styled(text.clone(), Style::new().fg(FG).bold()),
                Span::styled("▏", Style::new().fg(ACCENT)),
            ]),
            _ => {
                let hint = match self.open_question() {
                    Some((id, _)) => format!("Type the answer to {id}…"),
                    None => format!("Ask {}…", self.name),
                };
                TextLine::from(vec![slash, Span::styled(hint, Style::new().fg(FAINT))])
            }
        }
    }

    /// The keys that work now, or the open prompt.
    fn footer(&self) -> Vec<Span<'static>> {
        let ask = |text: String, k: &[(&str, &str)]| {
            let mut line = vec![Span::styled(text, Style::new().fg(FG).bold()), Span::raw("   ")];
            line.extend(keys(k));
            line
        };
        match &self.prompt {
            Some(Prompt::Rename { spk, name }) => {
                let mut line = vec![
                    Span::styled(format!("Rename {} to: ", self.label(*spk)), Style::new().fg(FG).bold()),
                    Span::styled(name.clone(), Style::new().fg(ACCENT).bold()),
                    Span::styled("▏", Style::new().fg(ACCENT)),
                    Span::raw("   "),
                ];
                line.extend(keys(&[("Enter", "rename"), ("Esc", "cancel")]));
                line
            }
            Some(Prompt::Merge { from, into }) => ask(
                format!("Merge {} into {}?", self.label(*from), self.label(*into)),
                &[("←→", "other speaker"), ("Enter", "merge"), ("Esc", "cancel")],
            ),
            Some(Prompt::Move { line, to }) => ask(
                format!("Move {line} to {}?", self.label(*to)),
                &[("←→", "other speaker"), ("Enter", "move"), ("Esc", "cancel")],
            ),
            Some(Prompt::Ask { answering, .. }) => {
                let send = match answering {
                    Some((id, _)) => format!("answer {id}"),
                    None => format!("ask {}", self.name),
                };
                keys(&[("Enter", &send), ("Esc", "cancel")])
            }
            None => {
                if let Some((t, _)) = self.window() {
                    return keys(&[("Esc", &format!("cancel {}", t.id)), ("q", "stop")]);
                }
                if let Some((_, asked)) = self.question() {
                    let answer = match asked.options.len() {
                        0 => "/".to_string(),
                        n => format!("1–{n}"),
                    };
                    let mut line = keys(&[(&answer, "answer"), ("q", "stop")]);
                    // With an intent check, it hears a reply without the name;
                    // without one, an open question can only be typed.
                    let say = match (self.spoken_on(), asked.options.is_empty()) {
                        (true, _) => Some("   or just say it".to_string()),
                        (false, false) => Some(format!("   or say “{}, option two”", self.name)),
                        (false, true) => None,
                    };
                    line.extend(say.map(|say| Span::styled(say, Style::new().fg(FAINT))));
                    return line;
                }
                if let Stage::Result(_) = self.stage() {
                    return keys(&[
                        ("↑↓", "line"),
                        ("o", "open"),
                        ("Esc", "back"),
                        ("/", "ask"),
                        ("q", "stop"),
                    ]);
                }
                let m = if self.selected_line.is_some() { "move line" } else { "merge" };
                let mut line = Vec::new();
                if self.focusable().len() > 1 {
                    let next = if self.stage() == Stage::Working { "next card" } else { "next result" };
                    line.extend(keys(&[("Tab", next)]));
                    line.push(Span::raw("   "));
                }
                if let Some(id) = self.focused().filter(|id| self.is_ready(*id)) {
                    line.extend(keys(&[("Enter", &format!("show {id}")), ("o", "open")]));
                    line.push(Span::raw("   "));
                }
                line.extend(keys(&[
                    ("/", "ask"),
                    ("t", "transcript"),
                    ("↑↓", "line"),
                    ("←→", "speaker"),
                    ("r", "rename"),
                    ("m", m),
                    ("u", "undo"),
                    ("q", "stop"),
                ]));
                line
            }
        }
    }

    /// The chip to highlight: the prompt's choice, else the selected speaker.
    fn picked(&self) -> Option<SpkId> {
        match &self.prompt {
            Some(Prompt::Merge { into, .. }) => Some(*into),
            Some(Prompt::Move { to, .. }) => Some(*to),
            Some(Prompt::Rename { spk, .. }) => Some(self.overlay.follow(*spk)),
            Some(Prompt::Ask { .. }) | None => self.selected(),
        }
    }

    fn draw_header(&self, f: &mut Frame, area: Rect) {
        let bar = Style::new().bg(HEADER).fg(FG);
        f.render_widget(Block::new().style(bar), area);
        let mut left = vec![
            Span::styled(" pam ", Style::new().bg(ACCENT).fg(ON).bold()),
            Span::styled(format!("  {}", self.title), Style::new().bold()),
            Span::styled(format!("  ·  {}", self.date), Style::new().fg(DIM)),
        ];
        let toggle = if self.show_transcript { "hide transcript" } else { "show transcript" };
        let mut right = keys(&[("t", toggle)]);
        right.push(Span::raw("    "));
        if self.ended {
            right.push(Span::styled("■ ENDED ", Style::new().fg(WARN).bold()));
        } else {
            let dot = if self.frame % 12 < 7 { "●" } else { " " };
            right.push(Span::styled(dot, Style::new().fg(BAD)));
            right.push(Span::styled(" REC ", Style::new().fg(BAD).bold()));
        }
        right.push(Span::styled(format!("{} ", mmss(self.now)), Style::new().bold()));
        // When the header is narrow, the model IDs go first; the providers stay.
        let width = |spans: &[Span]| spans.iter().map(Span::width).sum::<usize>();
        let room = (area.width as usize).saturating_sub(width(&right) + 1);
        let mut models = self.models(true);
        if width(&left) + width(&models) > room {
            models = self.models(false);
        }
        left.extend(models);
        f.render_widget(Paragraph::new(TextLine::from(right).right_aligned()).style(bar), area);
        // Narrower still, the left is cut short rather than run into the timer.
        f.render_widget(Paragraph::new(TextLine::from(left)).style(bar), Rect { width: room as u16, ..area });
    }

    /// The header's models, so the room sees who gets the transcript:
    /// `scribe ● claude-haiku-5.5 · openrouter auto`, the dot's colour giving
    /// its state, `auto` marking a detected one, and `off` with no dot.
    /// Without `ids`, only the providers. The intent check's comes between
    /// them when one was named; otherwise it is the scribe's. Then the
    /// meeting's cost so far.
    fn models(&self, ids: bool) -> Vec<Span<'static>> {
        let mut spans = Vec::new();
        let intent = self.intent.as_ref().filter(|i| i.source != Some(Source::Detected)).cloned();
        for (what, m) in [("scribe", &self.scribe), ("intent", &intent), ("agent", &self.agent)] {
            let Some(m) = m else { continue };
            spans.push(Span::styled(format!("   {what} "), Style::new().fg(DIM)));
            let dot = match m.status {
                ModelStatus::Off => {
                    spans.push(Span::styled("off", Style::new().fg(DIM)));
                    continue;
                }
                ModelStatus::Ready => OK,
                ModelStatus::Warning => WARN,
                ModelStatus::Unavailable => BAD,
            };
            spans.push(Span::styled("● ", Style::new().fg(dot)));
            let name = match (m.short_model(), m.provider()) {
                (None, _) => "none".to_string(),
                (Some(id), Some(provider)) if ids => format!("{id} · {provider}"),
                (Some(id), None) => id.to_string(),
                (Some(_), Some(provider)) => provider.to_string(),
            };
            spans.push(Span::styled(name, Style::new().fg(FG)));
            if m.source == Some(Source::Detected) && m.model.is_some() {
                spans.push(Span::styled(" auto", Style::new().fg(FAINT)));
            }
        }
        if let Some(u) = &self.usage {
            spans.push(Span::styled(format!("   {}", u.label()), Style::new().fg(FG)));
        }
        spans
    }

    fn draw_strip(&self, f: &mut Frame, area: Rect) {
        let mut chips = Vec::new();
        let visible = self.visible();
        let picked = self.picked();
        for &spk in &visible {
            let c = color(spk);
            // A merged-away voice still lights its speaker's chip.
            let level = self
                .active
                .iter()
                .filter(|&(&a, _)| self.overlay.follow(a) == spk)
                .map(|(_, &l)| l)
                .reduce(f32::max);
            let on = level.is_some();
            let mut emphasis = if on { Modifier::BOLD } else { Modifier::empty() };
            if picked == Some(spk) {
                emphasis |= Modifier::REVERSED;
            }
            match self.overlay.name(spk) {
                Some(name) => {
                    chips.push(Span::styled(
                        format!(" {} ", initials(name)),
                        Style::new().bg(c).fg(ON).bold(),
                    ));
                    chips.push(Span::styled(format!(" {name} "), Style::new().fg(FG).add_modifier(emphasis)));
                }
                None => {
                    chips.push(Span::styled(" ·· ", Style::new().bg(c).fg(ON).bold()));
                    chips.push(Span::styled(format!(" {spk} "), Style::new().fg(c).add_modifier(emphasis)));
                    chips.push(Span::styled("not introduced", Style::new().fg(FAINT).italic()));
                }
            }
            chips.push(Span::styled(
                match level {
                    Some(level) => format!(" {}", meter(self.frame, spk.0, level)),
                    None => "       ".into(),
                },
                Style::new().fg(c),
            ));
            chips.push(Span::raw("   "));
        }
        if visible.is_empty() {
            chips.push(Span::styled("No voices yet", Style::new().fg(FAINT).italic()));
        }
        let n = visible.len();
        let mut summary =
            vec![Span::styled(format!("{n} voice{}", if n == 1 { "" } else { "s" }), Style::new().fg(DIM))];
        let unnamed = self.unnamed();
        if unnamed > 0 {
            summary.push(Span::styled(" · ", Style::new().fg(DIM)));
            summary.push(Span::styled(format!("{unnamed} unnamed"), Style::new().fg(WARN)));
        }
        let summary = TextLine::from(summary).right_aligned();
        let sw = summary.width() as u16 + 2;
        let chip_area = Rect { width: area.width.saturating_sub(sw), ..area };
        f.render_widget(Paragraph::new(TextLine::from(chips)), chip_area);
        f.render_widget(Paragraph::new(summary), area);
    }

    fn draw_transcript(&self, f: &mut Frame, area: Rect) {
        let block =
            pane("Transcript", BORDER).title_top(TextLine::from(keys(&[("t", "hide")])).right_aligned());
        let inner = block.inner(area).inner(ratatui::layout::Margin::new(1, 0));
        f.render_widget(block, area);
        let w = inner.width as usize;
        let mut rows: Vec<TextLine> = Vec::new();
        // Where the selected line starts, to keep it in view.
        let mut selected_rows = None;
        for r in &self.rows {
            if !rows.is_empty() {
                rows.push(TextLine::default());
            }
            match r {
                Row::Line(l) => {
                    let selected = self.selected_line == Some(l.id);
                    if selected {
                        selected_rows = Some(rows.len());
                    }
                    let mut head = TextLine::from(vec![
                        Span::styled(mmss(l.t0), Style::new().fg(FAINT)),
                        Span::raw(" "),
                        self.speaker_name(self.speaker_of(l), true),
                    ]);
                    if selected {
                        head = head.patch_style(Modifier::REVERSED);
                        head.push_span(Span::styled(format!("  {}", l.id), Style::new().fg(DIM)));
                    }
                    // A line that asked something leads to its task.
                    if let Some(id) =
                        self.invocations.iter().find(|(_, i)| i.line == Some(l.id)).map(|(id, _)| id)
                    {
                        head.push_span(Span::styled(format!("  → {}", self.name), Style::new().fg(ACCENT)));
                        head.push_span(Span::styled(format!(" {id}"), Style::new().fg(FAINT)));
                    }
                    rows.push(head);
                    for part in wrap(&l.text, w.saturating_sub(2)) {
                        rows.push(TextLine::from(format!("  {part}")));
                    }
                }
                Row::Note(text) => {
                    for (i, part) in wrap(&format!("· {text}"), w.saturating_sub(2)).into_iter().enumerate()
                    {
                        let indent = if i == 0 { "" } else { "  " };
                        rows.push(TextLine::from(Span::styled(
                            format!("{indent}{part}"),
                            Style::new().fg(ACCENT).italic(),
                        )));
                    }
                }
                Row::Op { undo, text, note, undone } => {
                    let mut text = format!("· {text}");
                    if let Some(note) = note.filter(|_| !undone) {
                        text.push_str(&format!(" · {note}"));
                    }
                    let style = if *undone {
                        Style::new().fg(FAINT).italic().crossed_out()
                    } else {
                        Style::new().fg(ACCENT).italic()
                    };
                    for (i, part) in wrap(&text, w.saturating_sub(2)).into_iter().enumerate() {
                        let indent = if i == 0 { "" } else { "  " };
                        rows.push(TextLine::from(Span::styled(format!("{indent}{part}"), style)));
                    }
                    let hint = if *undone {
                        vec![Span::styled("  undone", Style::new().fg(FAINT))]
                    } else if self.last_undo() == Some(*undo) {
                        vec![
                            Span::styled("  u", Style::new().fg(ACCENT).bold()),
                            Span::styled(" undo", Style::new().fg(DIM)),
                        ]
                    } else {
                        Vec::new()
                    };
                    // The hint goes on a line of its own when it doesn't fit.
                    let hint_width: usize = hint.iter().map(Span::width).sum();
                    if rows.last().unwrap().width() + hint_width > w {
                        rows.push(TextLine::default());
                    }
                    let last = rows.last_mut().unwrap();
                    for span in hint {
                        last.push_span(span);
                    }
                }
            }
        }
        for p in &self.preview {
            if !rows.is_empty() {
                rows.push(TextLine::default());
            }
            rows.push(TextLine::from(vec![
                Span::styled(mmss(p.t0), Style::new().fg(FAINT)),
                Span::raw(" "),
                self.speaker_name(p.spk, false),
                Span::styled("  …", Style::new().fg(FAINT)),
            ]));
            for part in wrap(&format!("{}…", p.text), w.saturating_sub(2)) {
                rows.push(TextLine::from(Span::styled(format!("  {part}"), Style::new().fg(FAINT).italic())));
            }
        }
        if rows.is_empty() {
            rows.push(TextLine::from(Span::styled("Nothing said yet.", Style::new().fg(FAINT).italic())));
        }
        let h = inner.height as usize;
        let mut skip = rows.len().saturating_sub(h);
        if let Some(at) = selected_rows {
            skip = skip.min(at);
        }
        f.render_widget(Paragraph::new(rows.split_off(skip)), inner);
    }

    fn draw_stage(&self, f: &mut Frame, area: Rect) {
        let block = pane("Stage", ACCENT);
        let inner = block.inner(area).inner(ratatui::layout::Margin::new(2, 0));
        f.render_widget(block, area);
        let w = inner.width as usize;

        let mut rows = vec![
            self.activity_strip(w),
            TextLine::from(Span::styled("─".repeat(w), Style::new().fg(BORDER))),
        ];
        match self.stage() {
            Stage::Result(id) => {
                if let Some(t) = self.tasks.iter().find(|t| t.id == id) {
                    self.draw_result(f, inner, rows, t);
                    return;
                }
            }
            Stage::Answer(id) => {
                if let Some(t) = self.tasks.iter().find(|t| t.id == id) {
                    rows.push(TextLine::default());
                    rows.extend(self.responding(t, w));
                    f.render_widget(Paragraph::new(rows), inner);
                    return;
                }
            }
            Stage::Working => {
                let [top, body] = Layout::vertical([Constraint::Length(2), Constraint::Fill(1)]).areas(inner);
                f.render_widget(Paragraph::new(rows), top);
                self.draw_working(f, body);
                return;
            }
            Stage::Listening => {}
        }
        let lead = (inner.height as usize).saturating_sub(16).min(3);
        rows.extend(std::iter::repeat_n(TextLine::default(), lead + 1));

        let speaking = self.active.keys().next_back().copied();
        let (c, amp) = match speaking {
            Some(spk) if !self.ended => (color(spk), self.active[&spk].max(0.15) as f64),
            _ => (FAINT, 0.15),
        };
        let n = w.min(60);
        let pad = " ".repeat((w - n) / 2);
        for r in 0..3 {
            rows.push(TextLine::from(Span::styled(
                format!("{pad}{}", wave_row(n, r, self.frame, amp)),
                Style::new().fg(c),
            )));
        }
        rows.push(TextLine::default());

        if self.ended {
            rows.push(TextLine::from(Span::styled(
                "The input has ended. Every line is final.",
                Style::new().fg(DIM),
            )));
            rows.push(TextLine::from(keys(&[("q", "close the meeting")])));
        } else {
            let who = speaking.or(self.last_active);
            rows.push(match who {
                Some(spk) if speaking.is_some() => TextLine::from(vec![
                    Span::styled("Listening to ", Style::new().fg(DIM)),
                    Span::styled(self.label(spk), Style::new().fg(color(spk)).bold()),
                ]),
                _ => TextLine::from(Span::styled("Listening", Style::new().fg(DIM))),
            });
            // Turned off on purpose: said in passing. Not ready: said below, with what to do.
            let off = match &self.scribe {
                Some(ModelInfo { status: ModelStatus::Off, reason: Some(why), .. }) => format!(" ({why})"),
                _ => String::new(),
            };
            let spoken = match &self.intent {
                Some(ModelInfo { status: ModelStatus::Off, reason: Some(why), .. }) => {
                    Some(format!(" ({why})"))
                }
                _ if self.spoken_on() => None,
                _ => Some(String::new()),
            };
            let hint = match (self.notes_on(), spoken, self.agent_down().is_none()) {
                (true, None, true) => {
                    format!("Taking notes quietly. Say “{0}, …” or press / to ask {0} something.", self.name)
                }
                (false, None, true) => {
                    format!("Not taking notes{off}. Say “{0}, …” or press / to ask {0} something.", self.name)
                }
                (true, Some(why), true) => format!(
                    "Taking notes quietly. Not listening for “{0}, …”{why}. Typed requests still work: press / to ask {0} something.",
                    self.name
                ),
                (false, Some(_), true) => format!(
                    "Not taking notes or listening for “{0}, …”{off}. Typed requests still work: press / to ask {0} something.",
                    self.name
                ),
                (true, _, false) => "Taking notes quietly.".into(),
                (false, _, false) => format!("Not taking notes{off}."),
            };
            for part in wrap(&hint, w) {
                rows.push(TextLine::from(Span::styled(part, Style::new().fg(DIM))));
            }
        }
        let mut said = Vec::new();
        let unavailable = |m: &Option<ModelInfo>| match m {
            Some(ModelInfo { status: ModelStatus::Unavailable, reason, .. }) => {
                Some(reason.clone().unwrap_or_else(|| "unknown reason".into()))
            }
            _ => None,
        };
        match (unavailable(&self.scribe), unavailable(&self.intent)) {
            (Some(a), Some(b)) if a == b => said.push((format!("No notes or spoken requests: {a}"), WARN)),
            (notes, spoken) => {
                if let Some(why) = notes {
                    let what = if self.intent.is_none() { "No notes or spoken requests" } else { "No notes" };
                    said.push((format!("{what}: {why}"), WARN));
                }
                if let Some(why) = spoken {
                    said.push((format!("No spoken requests: {why}"), WARN));
                }
            }
        }
        match &self.agent {
            Some(AgentInfo { status: ModelStatus::Unavailable, reason, .. }) => {
                let why = reason.as_deref().unwrap_or("unknown reason");
                said.push((format!("{} can't take requests: {why}", self.name), WARN));
            }
            Some(AgentInfo { status: ModelStatus::Off, reason, .. }) => {
                let why = reason.as_deref().map_or_else(String::new, |r| format!(" ({r})"));
                said.push((format!("{} takes no requests{why}.", self.name), DIM));
            }
            _ => {}
        }
        // The same pi warns for each: once is enough.
        for m in [&self.scribe, &self.intent, &self.agent].into_iter().flatten() {
            if let ModelInfo { status: ModelStatus::Warning, reason: Some(why), .. } = m
                && !said.iter().any(|(s, _)| s == why)
            {
                said.push((why.clone(), FAINT));
            }
        }
        for (text, fg) in said {
            for part in wrap(&text, w) {
                rows.push(TextLine::from(Span::styled(part, Style::new().fg(fg))));
            }
        }
        if let Some((section, item)) = self.notes.newest() {
            rows.push(TextLine::default());
            rows.push(TextLine::from(vec![
                Span::styled("JUST NOTED", Style::new().fg(DIM).bold()),
                Span::styled(format!(" · {}", mmss(item.t)), Style::new().fg(FAINT)),
            ]));
            for part in wrap(&item.label(), w) {
                rows.push(TextLine::from(part));
            }
            rows.push(TextLine::from(Span::styled(format!("→ {section}"), Style::new().fg(ACCENT))));
        }
        rows.push(TextLine::default());
        let n = self.visible().len();
        let unnamed = self.unnamed();
        rows.push(TextLine::from(vec![
            Span::styled(n.to_string(), Style::new().bold()),
            Span::styled(if n == 1 { " voice · " } else { " voices · " }, Style::new().fg(DIM)),
            Span::styled(unnamed.to_string(), Style::new().fg(if unnamed > 0 { WARN } else { FG }).bold()),
            Span::styled(" unnamed     ", Style::new().fg(DIM)),
            Span::styled(self.lines.to_string(), Style::new().bold()),
            Span::styled(" lines     ", Style::new().fg(DIM)),
            Span::styled(self.notes.len().to_string(), Style::new().bold()),
            Span::styled(" notes     ", Style::new().fg(DIM)),
            Span::styled(self.tasks.len().to_string(), Style::new().bold()),
            Span::styled(if self.tasks.len() == 1 { " request" } else { " requests" }, Style::new().fg(DIM)),
        ]));
        if let Some(e) = &self.error {
            rows.push(TextLine::default());
            for part in wrap(&format!("Error: {e}"), w) {
                rows.push(TextLine::from(Span::styled(part, Style::new().fg(BAD))));
            }
        }
        f.render_widget(Paragraph::new(rows), inner);
    }

    /// The activity strip, the stage's top line: "Listening", then a chip
    /// per task but the cancelled ones, which stay until the meeting ends.
    /// When they don't fit, the chips drop the request's words, then the
    /// oldest finished ones fold into `+N earlier`.
    fn activity_strip(&self, w: usize) -> TextLine<'static> {
        let dot = if self.ended || self.frame % 12 < 7 { "●" } else { "○" };
        let mut head = vec![
            Span::styled(dot, Style::new().fg(if self.ended { DIM } else { OK })),
            Span::raw(if self.ended { " Stopped listening" } else { " Listening" }),
        ];
        let mut chips: Vec<&TaskState> =
            self.tasks.iter().filter(|t| t.status != TaskStatus::Cancelled).collect();
        if chips.is_empty() {
            if !self.ended {
                let how = match self.spoken_on() {
                    true => format!("say “{}, …” to start one", self.name),
                    false => format!("/ to ask {}", self.name),
                };
                head.push(Span::styled(format!("    no tasks · {how}"), Style::new().fg(FAINT)));
            }
            return TextLine::from(head);
        }
        let line = |chips: &[&TaskState], words: bool, folded: usize| {
            let mut spans = head.clone();
            if folded > 0 {
                spans.push(Span::styled(format!("   +{folded} earlier"), Style::new().fg(FAINT)));
            }
            for t in chips {
                spans.push(Span::raw("   "));
                spans.extend(self.chip(t, words));
            }
            TextLine::from(spans)
        };
        for words in [true, false] {
            let all = line(&chips, words, 0);
            if all.width() <= w {
                return all;
            }
        }
        let mut folded = 0;
        while let Some(i) = chips.iter().position(|t| !t.status.is_open()) {
            chips.remove(i);
            folded += 1;
            let fewer = line(&chips, false, folded);
            if fewer.width() <= w {
                return fewer;
            }
        }
        line(&chips, false, folded)
    }

    /// A task's chip on the strip: an icon, its ID, the request's first words
    /// with `words`, and where it stands: the seconds left in its cancel
    /// window, `asking`, its progress while it runs (`3/5` or `60%`),
    /// `queued`, `ready`, `failed`. A finished task with nothing to show is dimmed.
    fn chip(&self, t: &TaskState, words: bool) -> Vec<Span<'static>> {
        let starts = self.window_of(t).map(|inv| inv.deadline.saturating_sub(self.now).div_ceil(SECOND));
        let spin = SPIN[(self.frame / 2) as usize % SPIN.len()].to_string();
        let (icon, badge, c) = match (t.status, starts) {
            (_, Some(secs)) => ("…".to_string(), format!("in {secs}s"), WARN),
            (TaskStatus::Running, _) if self.questions.contains_key(&t.id) => {
                ("?".into(), "asking".into(), WARN)
            }
            (TaskStatus::Running, _) => (spin, t.progress_label().unwrap_or_default(), ACCENT),
            (TaskStatus::Queued, _) => ("○".into(), "queued".into(), FAINT),
            (TaskStatus::Done, _) if t.is_ready() => ("✓".into(), "ready".into(), OK),
            (TaskStatus::Done, _) => ("✓".into(), "done".into(), FAINT),
            (TaskStatus::Failed, _) => ("✗".into(), "failed".into(), BAD),
            (TaskStatus::Stopped, _) => ("■".into(), "stopped".into(), FAINT),
            (TaskStatus::Cancelled, _) => ("×".into(), "cancelled".into(), FAINT),
        };
        let dimmed = matches!(t.status, TaskStatus::Done | TaskStatus::Stopped) && !t.is_ready();
        let mut text = Style::new().fg(if dimmed { FAINT } else { FG });
        // The focused ready chip, which Enter shows (a focused card has its border).
        if t.is_ready() && self.focused() == Some(t.id) {
            text = text.bold().underlined();
        }
        let mut spans =
            vec![Span::styled(icon, Style::new().fg(c)), Span::styled(format!(" {}", t.id), text)];
        if words {
            spans.push(Span::styled(format!(" {}", clip(&t.request, 18)), text));
        }
        if !badge.is_empty() {
            spans.push(Span::styled(format!(" {badge}"), Style::new().fg(c)));
        }
        spans
    }

    /// The Working stage: a summary, then a card per task running or
    /// waiting, the focused one with the accent border.
    fn draw_working(&self, f: &mut Frame, area: Rect) {
        let cards = self.cards();
        let running = cards.iter().filter(|t| t.status == TaskStatus::Running).count();
        let queued = cards.len() - running;
        let dim = |s: &str| Span::styled(s.to_string(), Style::new().fg(DIM));
        let bold = |s: String| Span::styled(s, Style::new().fg(FG).bold());
        let mut summary = Vec::new();
        if running > 0 {
            summary.push(dim("Running "));
            summary.push(bold(format!("{running} task{}", if running == 1 { "" } else { "s" })));
        }
        if queued > 0 {
            summary.push(dim(if running > 0 { ", " } else { "" }));
            summary.push(bold(format!("{queued} queued")));
        }
        summary.push(dim(if self.ended { ". The input has ended." } else { ". Still listening." }));
        f.render_widget(Paragraph::new(TextLine::from(summary)), Rect { height: 1, ..area });

        let focused = self.focused();
        let mut y = area.y + 2;
        for (i, t) in cards.iter().enumerate() {
            let body = self.card_rows(t, area.width.saturating_sub(4) as usize);
            let h = body.len() as u16 + 2;
            if y + h > area.bottom() {
                if y < area.bottom() {
                    let more = format!("… {} more", cards.len() - i);
                    f.render_widget(
                        Paragraph::new(Span::styled(more, Style::new().fg(FAINT))),
                        Rect { y, height: 1, ..area },
                    );
                }
                return;
            }
            let label = match t.status {
                TaskStatus::Queued => Span::styled(" queued ", Style::new().fg(FAINT)),
                _ if self.questions.contains_key(&t.id) => {
                    Span::styled(" asking ", Style::new().fg(WARN).bold())
                }
                _ => Span::styled(
                    t.progress_label().map_or_else(String::new, |l| format!(" {l} ")),
                    Style::new().fg(ACCENT).bold(),
                ),
            };
            let room = (area.width as usize).saturating_sub(label.width() + 10);
            let block = Block::bordered()
                .border_type(BorderType::Rounded)
                .border_style(Style::new().fg(if focused == Some(t.id) { ACCENT } else { BORDER }))
                .title(TextLine::from(vec![
                    Span::styled(format!(" {} ", t.id), Style::new().fg(FAINT)),
                    Span::styled(format!("{} ", clip(&t.request, room)), Style::new().fg(FG).bold()),
                ]))
                .title_top(TextLine::from(label).right_aligned());
            let rect = Rect { y, height: h, ..area };
            f.render_widget(
                Paragraph::new(body).block(block.padding(ratatui::widgets::Padding::horizontal(1))),
                rect,
            );
            y += h;
        }
        if cards.len() > 1 && y < area.bottom() {
            f.render_widget(
                Paragraph::new(TextLine::from(keys(&[("Tab", "next card")]))),
                Rect { y, height: 1, ..area },
            );
        }
    }

    /// A Working card's rows: who asked and when, and how long it has run
    /// (or what it waits for); then its progress bar and current action.
    fn card_rows(&self, t: &TaskState, w: usize) -> Vec<TextLine<'static>> {
        let when = match (t.status, t.started_at) {
            (TaskStatus::Queued, _) => match self.tasks.iter().find(|o| o.status == TaskStatus::Running) {
                Some(o) => format!("waits for {}", o.id),
                None => "waits its turn".into(),
            },
            (_, Some(start)) => format!("running {}", elapsed(self.now.saturating_sub(start))),
            (_, None) => "starting".into(),
        };
        let mut rows = vec![TextLine::from(vec![
            self.asker(t),
            Span::styled(format!(" · {} · {when}", mmss(t.asked_at)), Style::new().fg(DIM)),
        ])];
        for c in &t.corrections {
            let then = clip(&format!("then: “{c}”"), w);
            rows.push(TextLine::from(Span::styled(then, Style::new().fg(DIM).italic())));
        }
        if t.status == TaskStatus::Queued {
            return rows;
        }
        if let Some(share) = t.progress_share() {
            let full = ((w as f64 * share) as usize).min(w);
            rows.push(TextLine::from(vec![
                Span::styled("█".repeat(full), Style::new().fg(ACCENT)),
                Span::styled("░".repeat(w - full), Style::new().fg(FAINT)),
            ]));
        }
        let spin = SPIN[(self.frame / 2) as usize % SPIN.len()];
        rows.push(match (&t.action, self.questions.contains_key(&t.id)) {
            (_, true) => TextLine::from(vec![
                Span::styled("? ", Style::new().fg(WARN)),
                Span::styled("Asking the room…", Style::new().fg(DIM)),
            ]),
            (Some(action), false) => TextLine::from(vec![
                Span::styled(format!("{spin} "), Style::new().fg(ACCENT)),
                Span::styled(clip(action, w.saturating_sub(2)), Style::new().fg(FG)),
            ]),
            (None, false) => TextLine::from(vec![
                Span::styled(format!("{spin} "), Style::new().fg(ACCENT)),
                Span::styled("Thinking…", Style::new().fg(DIM)),
            ]),
        });
        rows
    }

    /// Who asked for task `t`: the speaker of its line, or the one the
    /// tracker had while its line isn't final; `typed` for the composer.
    fn asker(&self, t: &TaskState) -> Span<'static> {
        match (t.line.and_then(|id| self.line(id)), self.invocations.get(&t.id)) {
            (Some(l), _) => self.speaker_name(self.speaker_of(l), true),
            (None, Some(inv)) => self.speaker_name(inv.spk, true),
            (None, None) => Span::styled("typed", Style::new().fg(FG).bold()),
        }
    }

    /// The cancel window: who asked and when, what Pam is about to do, and a
    /// bar that empties as the deadline nears.
    fn draw_cancel_window(&self, f: &mut Frame, stage: Rect, t: &TaskState, inv: &Invoked) {
        let w = stage.width.saturating_sub(4).min(58);
        let h = 9.min(stage.height);
        let area = Rect {
            x: stage.x + (stage.width - w) / 2,
            y: stage.y + (stage.height - h).min(6),
            width: w,
            height: h,
        };
        let block = Block::bordered()
            .border_type(BorderType::Thick)
            .border_style(Style::new().fg(WARN))
            .title(Span::styled(format!(" Starting {} ", t.id), Style::new().fg(WARN).bold()))
            .title_top(TextLine::from(keys(&[("Esc", "cancel")])).right_aligned());
        let inner = block.inner(area).inner(ratatui::layout::Margin::new(2, 1));
        f.render_widget(Clear, area);
        f.render_widget(block, area);
        let iw = inner.width as usize;
        let who = match inv.line.and_then(|id| self.line(id)) {
            Some(l) => self.speaker_name(self.speaker_of(l), true),
            None if inv.spk.is_some() => self.speaker_name(inv.spk, true),
            None => Span::styled("someone", Style::new().fg(FG).bold()),
        };
        let mut rows = vec![TextLine::from(vec![
            who,
            Span::styled(format!(" · {}", mmss(t.asked_at)), Style::new().fg(FAINT)),
        ])];
        for part in wrap(&inv.summary, iw) {
            rows.push(TextLine::from(Span::styled(part, Style::new().fg(FG))));
        }
        rows.push(TextLine::default());
        let left = inv.deadline.saturating_sub(self.now);
        let secs = left.div_ceil(pam_core::clock::SECOND);
        let label = format!("  starts in {secs}s");
        let n = iw.saturating_sub(label.chars().count());
        let length = inv.deadline.saturating_sub(inv.opened).max(1);
        let full = (n as u64 * left / length).min(n as u64) as usize;
        rows.push(TextLine::from(vec![
            Span::styled("█".repeat(full), Style::new().fg(WARN)),
            Span::styled("░".repeat(n - full), Style::new().fg(FAINT)),
            Span::styled(label, Style::new().fg(WARN).bold()),
        ]));
        rows.push(TextLine::from(Span::styled(
            format!("Say “{}, cancel” to stop it.", self.name),
            Style::new().fg(DIM),
        )));
        f.render_widget(Paragraph::new(rows), inner);
    }

    /// A question the agent asks the room: the question, what was asked of
    /// the agent, the numbered options and a bar that empties as the
    /// countdown runs out.
    fn draw_question(&self, f: &mut Frame, stage: Rect, id: TaskId, asked: &Asked) {
        let w = stage.width.saturating_sub(4).min(64);
        let iw = w.saturating_sub(6) as usize;
        let mut rows = Vec::new();
        for part in wrap(&asked.text, iw) {
            rows.push(TextLine::from(Span::styled(part, Style::new().fg(FG).bold())));
        }
        if let Some(t) = self.tasks.iter().find(|t| t.id == id) {
            let who = match t.line.and_then(|l| self.line(l)) {
                Some(l) => Some(self.label_of(l)),
                // Spoken, its line not final yet.
                None => {
                    self.invocations.get(&t.id).map(|i| i.spk.map_or("Someone".into(), |s| self.label(s)))
                }
            };
            let asked_by = match who {
                Some(who) => format!("{who} asked: “{}”", t.request),
                None => format!("Typed: “{}”", t.request),
            };
            for part in wrap(&asked_by, iw) {
                rows.push(TextLine::from(Span::styled(part, Style::new().fg(DIM))));
            }
        }
        rows.push(TextLine::default());
        if asked.options.is_empty() {
            // An open question: without an intent check, nothing hears a spoken answer.
            let how = match self.spoken_on() {
                true => "Say your answer, or type it with /",
                false => "Type your answer with /",
            };
            rows.push(TextLine::from(Span::styled(how, Style::new().fg(FG))));
        }
        for (i, option) in asked.options.iter().enumerate() {
            for (j, part) in wrap(option, iw.saturating_sub(6)).into_iter().enumerate() {
                let key = if j == 0 {
                    Span::styled(format!(" {} ", i + 1), Style::new().bg(ACCENT).fg(ON).bold())
                } else {
                    Span::raw("   ")
                };
                rows.push(TextLine::from(vec![key, Span::styled(format!("  {part}"), Style::new().fg(FG))]));
            }
        }
        rows.push(TextLine::default());
        let left = asked.deadline.saturating_duration_since(self.wall);
        let length =
            asked.deadline.saturating_duration_since(asked.opened).max(std::time::Duration::from_millis(1));
        let n = iw.min(48);
        let full = ((n as f64 * left.as_secs_f64() / length.as_secs_f64()).ceil() as usize).min(n);
        rows.push(TextLine::from(vec![
            Span::styled("█".repeat(full), Style::new().fg(WARN)),
            Span::styled("░".repeat(n - full), Style::new().fg(FAINT)),
        ]));
        let guess = format!("If nobody answers, {} goes with its best guess and says so.", self.name);
        for part in wrap(&guess, iw) {
            rows.push(TextLine::from(Span::styled(part, Style::new().fg(DIM))));
        }

        let h = (rows.len() as u16 + 4).min(stage.height);
        let area = Rect {
            x: stage.x + (stage.width - w) / 2,
            y: stage.y + (stage.height - h).min(4),
            width: w,
            height: h,
        };
        // Past its deadline the core still waits for a reply being heard or checked.
        let left = match left.as_millis().div_ceil(1000) {
            0 => " hearing a reply… ".to_string(),
            secs => format!(" {secs}s left "),
        };
        let block = Block::bordered()
            .border_type(BorderType::Thick)
            .border_style(Style::new().fg(WARN))
            .title(Span::styled(format!(" {} is asking · {id} ", self.name), Style::new().fg(WARN).bold()))
            .title_top(TextLine::from(Span::styled(left, Style::new().fg(WARN).bold())).right_aligned());
        let inner = block.inner(area).inner(ratatui::layout::Margin::new(2, 1));
        f.render_widget(Clear, area);
        f.render_widget(block, area);
        f.render_widget(Paragraph::new(rows), inner);
    }

    /// Who said `l`, by name once known.
    fn label_of(&self, l: &Line) -> String {
        self.overlay.label(self.speaker_of(l))
    }

    /// Why the agent takes no requests, if it doesn't: it is off or not ready.
    fn agent_down(&self) -> Option<&str> {
        self.agent.as_ref().and_then(ModelInfo::why_not)
    }

    /// Whether the meeting takes notes: a meeting that hasn't said otherwise does.
    fn notes_on(&self) -> bool {
        self.scribe.as_ref().is_none_or(ModelInfo::usable)
    }

    /// Whether the meeting checks spoken requests: on its intent check's
    /// model, else on the scribe's.
    fn spoken_on(&self) -> bool {
        self.intent.as_ref().map_or_else(|| self.notes_on(), ModelInfo::usable)
    }

    /// A result opened on the stage: its title, who asked and when, how long
    /// the task took, what it said (its verdict), and the path or URL; then a
    /// text file's lines, scrolled to keep the one selected with ↑↓ in sight.
    /// `top` is the strip.
    fn draw_result(&self, f: &mut Frame, area: Rect, mut rows: Vec<TextLine<'static>>, t: &TaskState) {
        let (Some(r), Some(o)) = (&t.result, &self.opened) else { return };
        let w = area.width as usize;
        let took = match (t.status, t.started_at, t.ended_at) {
            (_, Some(a), Some(b)) => format!(" · took {}", elapsed(b.saturating_sub(a))),
            (TaskStatus::Running, Some(a), None) => {
                format!(" · running {}", elapsed(self.now.saturating_sub(a)))
            }
            _ => String::new(),
        };
        let (badge, c) = match t.status {
            TaskStatus::Done => ("✓ ready".to_string(), OK),
            s => (s.as_str().to_string(), DIM),
        };
        let left = vec![
            Span::styled(format!("{} ", t.id), Style::new().fg(FAINT)),
            Span::styled("RESULT", Style::new().fg(ACCENT).bold()),
        ];
        let right = vec![Span::styled(badge, Style::new().fg(c)), Span::styled(took, Style::new().fg(DIM))];
        let gap =
            w.saturating_sub(TextLine::from(left.clone()).width() + TextLine::from(right.clone()).width());
        rows.push(TextLine::default());
        rows.push(TextLine::from([left, vec![Span::raw(" ".repeat(gap.max(1)))], right].concat()));
        for part in wrap(&r.title, w) {
            rows.push(TextLine::from(Span::styled(part, Style::new().fg(FG).bold())));
        }
        rows.push(TextLine::from(vec![
            self.asker(t),
            Span::styled(format!(" asked at {}", mmss(t.asked_at)), Style::new().fg(DIM)),
        ]));
        for part in wrap(&format!("“{}”", t.request), w) {
            rows.push(TextLine::from(Span::styled(part, Style::new().fg(DIM).italic())));
        }
        if let Some(said) = self.said.get(&t.id) {
            rows.push(TextLine::default());
            for para in said.lines().filter(|p| !p.trim().is_empty()) {
                for part in wrap(para, w) {
                    rows.push(TextLine::from(Span::styled(part, Style::new().fg(FG))));
                }
            }
        }
        rows.push(TextLine::default());
        let mut here = Style::new().fg(ACCENT).underlined();
        if o.line.is_none() {
            here = here.add_modifier(Modifier::REVERSED);
        }
        rows.push(TextLine::from(vec![
            Span::styled("→ ", Style::new().fg(ACCENT)),
            Span::styled(clip(r.target.as_str(), w.saturating_sub(2)), here),
        ]));
        let note = |text: String, fg: Color| TextLine::from(Span::styled(text, Style::new().fg(fg)));
        let Some(text) = &o.text else {
            rows.push(match (&o.unread, &r.target) {
                (Some(why), _) => note(why.clone(), WARN),
                (None, Target::Url(_)) => note("o opens it in the browser".into(), DIM),
                (None, Target::File(_)) => note("o opens it".into(), DIM),
            });
            f.render_widget(Paragraph::new(rows), area);
            return;
        };
        rows.push(TextLine::from(Span::styled("─".repeat(w), Style::new().fg(BORDER))));
        let head = rows.len() as u16;
        f.render_widget(Paragraph::new(rows), Rect { height: head.min(area.height), ..area });
        let body = Rect { y: area.y + head, height: area.height.saturating_sub(head), ..area };
        if body.height == 0 {
            return;
        }
        // Each line of the text, wrapped, with the line it came from.
        let mut lines: Vec<(usize, String)> = Vec::new();
        for (i, l) in text.iter().enumerate() {
            let parts = wrap(l, w);
            if parts.is_empty() {
                lines.push((i, String::new()));
            }
            lines.extend(parts.into_iter().map(|p| (i, p)));
        }
        let h = body.height as usize;
        let skip = match o.line.and_then(|sel| lines.iter().rposition(|(i, _)| *i == sel)) {
            Some(last) if last >= h => last + 1 - h,
            _ => 0,
        };
        let shown: Vec<TextLine<'static>> = lines
            .into_iter()
            .skip(skip)
            .take(h)
            .map(|(i, part)| {
                let mut style = Style::new().fg(FG);
                if o.line == Some(i) {
                    style = style.add_modifier(Modifier::REVERSED);
                }
                TextLine::from(Span::styled(part, style))
            })
            .collect();
        f.render_widget(Paragraph::new(shown), body);
    }

    /// An answer on the stage: the request, who asked and when, then the answer.
    fn responding(&self, t: &TaskState, w: usize) -> Vec<TextLine<'static>> {
        let state = match t.status {
            TaskStatus::Queued => "QUEUED",
            TaskStatus::Running => "ANSWERING",
            TaskStatus::Done => "ANSWERED",
            TaskStatus::Failed => "NOT ANSWERED",
            TaskStatus::Stopped => "STOPPED",
            TaskStatus::Cancelled => "CANCELLED",
        };
        let mut rows = vec![TextLine::from(vec![
            Span::styled(format!("{state}  "), Style::new().fg(DIM).bold()),
            self.asker(t),
            Span::styled(format!("  · {} · asked at {}", t.id, mmss(t.asked_at)), Style::new().fg(FAINT)),
        ])];
        for part in wrap(&format!("“{}”", t.request), w) {
            rows.push(TextLine::from(Span::styled(part, Style::new().fg(DIM).italic())));
        }
        for c in &t.corrections {
            for part in wrap(&format!("then: “{c}”"), w) {
                rows.push(TextLine::from(Span::styled(part, Style::new().fg(DIM).italic())));
            }
        }
        rows.push(TextLine::default());
        let said = self.said.get(&t.id);
        for para in said.iter().flat_map(|text| text.lines()).filter(|p| !p.trim().is_empty()) {
            for part in wrap(para, w.saturating_sub(4)) {
                rows.push(TextLine::from(Span::styled(format!("  {part}"), Style::new().fg(FG))));
            }
        }
        let status = |text: String, color: Color| {
            wrap(&text, w).into_iter().map(move |p| TextLine::from(Span::styled(p, Style::new().fg(color))))
        };
        match t.status {
            TaskStatus::Running if said.is_none() => {
                let spin = SPIN[(self.frame / 2) as usize % SPIN.len()];
                let doing =
                    if self.questions.contains_key(&t.id) { "Asking the room…" } else { "Thinking…" };
                rows.push(TextLine::from(vec![
                    Span::styled(format!("  {spin} "), Style::new().fg(ACCENT)),
                    Span::styled(doing, Style::new().fg(DIM)),
                ]));
            }
            TaskStatus::Running => {
                rows.push(TextLine::from(Span::styled("  ▍", Style::new().fg(ACCENT))));
            }
            TaskStatus::Queued => {
                let ahead = self.tasks.iter().find(|o| o.status == TaskStatus::Running).map(|o| o.id);
                let text = match ahead {
                    Some(id) => format!("Waits for {id} to finish."),
                    None => "Waits for the task before it.".into(),
                };
                rows.extend(status(text, DIM));
            }
            TaskStatus::Done if said.is_none() => {
                rows.extend(status("Finished without an answer.".into(), DIM))
            }
            TaskStatus::Done => {}
            TaskStatus::Failed => {
                let why = t.error.as_deref().unwrap_or("unknown reason");
                rows.extend(status(format!("Couldn't answer: {why}"), BAD));
            }
            TaskStatus::Stopped => rows.extend(status("Stopped when the meeting ended.".into(), DIM)),
            TaskStatus::Cancelled => rows.extend(status("Cancelled.".into(), DIM)),
        }
        let next: Vec<&TaskState> =
            self.tasks.iter().filter(|o| o.status == TaskStatus::Queued && o.id != t.id).collect();
        if !next.is_empty() {
            rows.push(TextLine::default());
            for o in next {
                for part in wrap(&format!("Next: {} “{}”", o.id, o.request), w) {
                    rows.push(TextLine::from(Span::styled(part, Style::new().fg(FAINT))));
                }
            }
        }
        rows.push(TextLine::default());
        rows.push(TextLine::from(keys(&[("Esc", "back to listening"), ("/", "ask again")])));
        rows
    }

    fn draw_notes(&self, f: &mut Frame, area: Rect) {
        let block = pane(&format!("{}’s notes", self.name), BORDER)
            .title_top(TextLine::from(Span::styled(" notes.md ", Style::new().fg(FAINT))).right_aligned());
        let inner = block.inner(area).inner(ratatui::layout::Margin::new(1, 0));
        f.render_widget(block, area);
        if self.notes.is_empty() {
            let none = match &self.scribe {
                Some(ModelInfo { status: ModelStatus::Off, reason: Some(why), .. }) => {
                    format!("Notes are off ({why}).")
                }
                Some(s) if !s.usable() => "Notes are off: the scribe's model isn't ready.".into(),
                _ => "No notes yet.".into(),
            };
            let rows: Vec<TextLine> = wrap(&none, inner.width as usize)
                .into_iter()
                .map(|part| TextLine::from(Span::styled(part, Style::new().fg(FAINT).italic())))
                .collect();
            f.render_widget(Paragraph::new(rows), inner);
            return;
        }
        // When the notes don't fit, the oldest discussion items give way first.
        let (w, h) = (inner.width as usize, inner.height as usize);
        let discussion = self.notes.sections.first().map_or(0, |s| s.items.len());
        let mut hidden = 0;
        let mut rows = self.note_rows(w, hidden);
        while rows.len() > h && hidden < discussion {
            hidden += 1;
            rows = self.note_rows(w, hidden);
        }
        f.render_widget(Paragraph::new(rows), inner);
    }

    /// The notes pane, `w` wide, leaving out the first
    /// `hidden` discussion items.
    fn note_rows(&self, w: usize, hidden: usize) -> Vec<TextLine<'static>> {
        let label = |t: &str| TextLine::from(Span::styled(t.to_uppercase(), Style::new().fg(DIM).bold()));
        let mut rows = vec![label("Summary")];
        let summary = if self.notes.summary.is_empty() { "Nothing yet." } else { &self.notes.summary };
        rows.extend(wrap(summary, w).into_iter().map(TextLine::from));
        for (i, s) in self.notes.sections.iter().enumerate() {
            if s.items.is_empty() && matches!(s.title, "Open questions" | "Notes") {
                continue;
            }
            rows.push(TextLine::default());
            rows.push(label(s.title));
            if s.items.is_empty() {
                rows.push(TextLine::from(Span::styled("None yet.", Style::new().fg(FAINT))));
            }
            let skip = if i == 0 { hidden } else { 0 };
            if skip > 0 {
                rows.push(TextLine::from(Span::styled(format!("… {skip} earlier"), Style::new().fg(FAINT))));
            }
            for (n, item) in s.items.iter().skip(skip).enumerate() {
                if item.kind == "topic" {
                    // A gap sets a topic off from the items of the one before.
                    if n > 0 || skip > 0 {
                        rows.push(TextLine::default());
                    }
                    rows.extend(
                        wrap(&item.text, w)
                            .into_iter()
                            .map(|p| TextLine::from(Span::styled(p, Style::new().bold()))),
                    );
                    continue;
                }
                let (bullet, style) = match item.kind.as_str() {
                    "action" | "request" => ("○", Style::new().fg(DIM)),
                    "question" => ("?", Style::new().fg(WARN)),
                    _ => ("•", Style::new().fg(FAINT)),
                };
                for (n, part) in wrap(&item.label(), w.saturating_sub(2)).into_iter().enumerate() {
                    let lead =
                        if n == 0 { Span::styled(format!("{bullet} "), style) } else { Span::raw("  ") };
                    rows.push(TextLine::from(vec![lead, Span::raw(part)]));
                }
                let credits = item.credits(&self.overlay);
                let Some(&first) = credits.first() else { continue };
                let tag: Vec<String> = credits.iter().map(|&s| self.label(s)).collect();
                let tag = Span::styled(tag.join(", "), Style::new().fg(color(first)));
                let last = rows.last_mut().unwrap();
                let room = w.saturating_sub(last.width());
                if tag.width() < room {
                    last.push_span(Span::raw(" ".repeat(room - tag.width())));
                    last.push_span(tag);
                } else {
                    rows.push(TextLine::from(tag).right_aligned());
                }
            }
        }
        rows
    }

    fn speaker_name(&self, spk: Option<SpkId>, unknown: bool) -> Span<'static> {
        match spk {
            Some(spk) => Span::styled(self.label(spk), Style::new().fg(color(spk)).bold()),
            None if unknown => Span::styled("unknown", Style::new().fg(DIM).bold()),
            None => Span::styled("new voice", Style::new().fg(DIM).bold()),
        }
    }
}

/// Initials for a chip, two columns wide: e.g. `A `, `AC`, or `王` for a
/// name in Chinese, whose characters are two columns each.
fn initials(name: &str) -> String {
    let mut s = String::new();
    for c in name.split_whitespace().filter_map(|w| w.chars().next()).flat_map(char::to_uppercase) {
        if s.width() + c.width().unwrap_or(0) > 2 {
            break;
        }
        s.push(c);
    }
    let pad = 2usize.saturating_sub(s.width());
    s + &" ".repeat(pad)
}

fn pane(title: &str, border: Color) -> Block<'static> {
    Block::bordered()
        .border_type(BorderType::Rounded)
        .border_style(Style::new().fg(border))
        .title(Span::styled(format!(" {title} "), Style::new().fg(FG).bold()))
}

fn color(spk: SpkId) -> Color {
    SPEAKERS[spk.0 as usize % SPEAKERS.len()]
}

/// `key label   key label`, keys in the accent colour.
fn keys(list: &[(&str, &str)]) -> Vec<Span<'static>> {
    let mut out = Vec::new();
    for (i, (k, label)) in list.iter().enumerate() {
        if i > 0 {
            out.push(Span::raw("   "));
        }
        out.push(Span::styled(k.to_string(), Style::new().fg(ACCENT).bold()));
        out.push(Span::styled(format!(" {label}"), Style::new().fg(DIM)));
    }
    out
}

/// A small level meter for a speaking chip: six bars that flicker around
/// `level` (0–1), the input level the core sends.
fn meter(frame: u64, seed: u32, level: f32) -> String {
    let f = frame as f64 + seed as f64 * 3.0;
    (0..6)
        .map(|i| {
            let i = i as f64;
            let wobble = 0.75 + 0.25 * (f * 0.9 + i * 1.7).sin() * (f * 0.37 + i).cos();
            let v = level as f64 * wobble;
            LEVELS[(v * 7.0).round().clamp(0.0, 7.0) as usize]
        })
        .collect()
}

/// One of the three rows of the listening wave, `r` = 0 at the top.
fn wave_row(n: usize, r: usize, frame: u64, amp: f64) -> String {
    let f = frame as f64;
    (0..n)
        .map(|i| {
            let x = i as f64;
            let env = (std::f64::consts::PI * (x + 0.5) / n as f64).sin();
            let v = (0.55 + 0.45 * (x * 0.52 - f * 0.55).sin())
                * (0.6 + 0.4 * (x * 0.19 + f * 0.21).sin())
                * env
                * amp;
            let seg = (v * 24.0 - (2 - r) as f64 * 8.0).round().clamp(0.0, 8.0) as usize;
            if seg > 0 {
                LEVELS[seg - 1]
            } else if r == 2 {
                '▁'
            } else {
                ' '
            }
        })
        .collect()
}

/// Whether a result's file is shown as text: Markdown or plain text.
fn is_text(file: &str) -> bool {
    let ext = Path::new(file).extension().and_then(|e| e.to_str()).unwrap_or_default().to_ascii_lowercase();
    matches!(ext.as_str(), "md" | "markdown" | "txt" | "text")
}

/// `text` on one line of at most `w` terminal columns, cut short with `…`.
fn clip(text: &str, w: usize) -> String {
    let text = text.split_whitespace().collect::<Vec<_>>().join(" ");
    if text.width() <= w {
        return text;
    }
    let mut out = String::new();
    for c in text.chars() {
        if out.width() + c.width().unwrap_or(0) + 1 > w {
            break;
        }
        out.push(c);
    }
    format!("{}…", out.trim_end())
}

/// A stretch of meeting time as a card shows it: `48s`, `1m 10s`, `1h 02m`.
fn elapsed(t: Ts) -> String {
    let s = t / SECOND;
    match s {
        0..60 => format!("{s}s"),
        60..3600 => format!("{}m {:02}s", s / 60, s % 60),
        _ => format!("{}h {:02}m", s / 3600, s % 3600 / 60),
    }
}

/// Greedy word wrap to `w` terminal columns, in which a CJK character takes
/// two. CJK text has no spaces, so a line may also break before or after any
/// CJK character, though not before the punctuation that follows one. Other
/// words longer than `w` are cut.
fn wrap(text: &str, w: usize) -> Vec<String> {
    let w = w.max(4);
    let mut out = Vec::new();
    let mut cur = String::new();
    for word in text.split_whitespace() {
        for (i, piece) in pieces(word).into_iter().enumerate() {
            let space = usize::from(i == 0 && !cur.is_empty());
            if cur.width() + space + piece.width() <= w {
                cur.push_str(&" ".repeat(space));
                cur.push_str(piece);
            } else {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                }
                cur.push_str(piece);
            }
            while cur.width() > w {
                let mut head = String::new();
                for c in cur.chars() {
                    if head.width() + c.width().unwrap_or(0) > w {
                        break;
                    }
                    head.push(c);
                }
                cur = cur[head.len()..].to_string();
                out.push(head);
            }
        }
    }
    if !cur.is_empty() {
        out.push(cur);
    }
    out
}

/// A word cut where a line may break inside it: each CJK character with the
/// punctuation after it, and the runs of other characters between them.
fn pieces(word: &str) -> Vec<&str> {
    let mut out = Vec::new();
    let mut start = 0;
    let mut prev_cjk = false;
    for (i, c) in word.char_indices() {
        let cjk = pam_speech::is_cjk(c);
        let attaches = prev_cjk && !cjk && !c.is_alphanumeric();
        if i > start && (cjk || (prev_cjk && !attaches)) {
            out.push(&word[start..i]);
            start = i;
        }
        if !attaches {
            prev_cjk = cjk;
        }
    }
    out.push(&word[start..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn screen(dash: &Dashboard) -> String {
        screen_at(dash, 150)
    }

    fn screen_at(dash: &Dashboard, width: u16) -> String {
        let mut term = ratatui::Terminal::new(ratatui::backend::TestBackend::new(width, 30)).unwrap();
        term.draw(|f| dash.draw(f)).unwrap();
        let buf = term.backend().buffer();
        (0..buf.area.height)
            .map(|y| (0..buf.area.width).map(|x| buf[(x, y)].symbol()).collect::<String>() + "\n")
            .collect()
    }

    #[test]
    fn a_rename_relabels_lines_and_undo_reverts_it() {
        let mut dash = Dashboard::new("Sync".into(), "11 Oct 2026".into());
        let line = |id, text: &str| Line {
            id: pam_core::UtteranceId(id),
            t0: 0,
            t1: 0,
            slot: Some(3),
            spk: Some(SpkId(3)),
            lang: None,
            text: text.into(),
        };
        dash.apply(CoreEvent::SpeakerAdded(SpkId(3)));
        dash.apply(CoreEvent::LineFinal(line(1, "Sorry I'm late.")));
        dash.apply(CoreEvent::LineFinal(line(2, "I'm Sam, from sales.")));
        assert!(screen(&dash).contains("spk_3 not introduced"));
        dash.apply(CoreEvent::SpeakerRenamed { spk: SpkId(3), name: "Sam".into(), undo: UndoId(4) });
        let s = screen(&dash);
        assert!(s.contains("· spk_3 is now Sam · earlier lines"), "{s}");
        assert!(s.contains("  updated  u undo"), "{s}");
        assert!(!s.contains("not introduced"), "{s}");
        assert_eq!(s.matches("00:00 Sam").count(), 2, "{s}");
        assert_eq!(dash.last_undo(), Some(UndoId(4)));

        dash.apply(CoreEvent::Undone { undo: UndoId(4), overlay: Overlay::default() });
        let s = screen(&dash);
        assert!(s.contains("· spk_3 is now Sam  undone"), "{s}");
        assert_eq!(s.matches("00:00 spk_3").count(), 2, "{s}");
        assert_eq!(dash.last_undo(), None);
    }

    fn press(dash: &mut Dashboard, code: KeyCode) -> Action {
        dash.key(KeyEvent::from(code))
    }

    /// Alice (spk_0), Marcus split over spk_1 and spk_3, Jordan (spk_2), and
    /// Jordan's u0004 on Alice's slot.
    fn split_meeting() -> Dashboard {
        let mut dash = Dashboard::new("Sync".into(), "11 Oct 2026".into());
        let line = |id, spk: u32, text: &str| {
            CoreEvent::LineFinal(Line {
                id: UtteranceId(id),
                t0: 0,
                t1: 0,
                slot: Some(spk),
                spk: Some(SpkId(spk)),
                lang: None,
                text: text.into(),
            })
        };
        for spk in 0..4 {
            dash.apply(CoreEvent::SpeakerAdded(SpkId(spk)));
        }
        dash.apply(line(1, 1, "Marcus here."));
        dash.apply(CoreEvent::SpeakerRenamed { spk: SpkId(1), name: "Marcus".into(), undo: UndoId(1) });
        dash.apply(line(2, 3, "We ship Thursday."));
        dash.apply(line(3, 0, "Can you send the mockups?"));
        dash.apply(line(4, 0, "Sure, tonight."));
        for (spk, name, undo) in [(0, "Alice", 2), (2, "Jordan", 3)] {
            dash.apply(CoreEvent::SpeakerRenamed { spk: SpkId(spk), name: name.into(), undo: UndoId(undo) });
        }
        dash
    }

    #[test]
    fn keys_merge_the_selected_speaker_and_move_the_selected_line() {
        let mut dash = split_meeting();
        assert_eq!(press(&mut dash, KeyCode::Char('m')), Action::None);
        assert!(dash.notice.as_deref().unwrap().starts_with("Pick a speaker"));

        // ← from nothing picks the last chip, spk_3; m offers the next one round, Alice.
        press(&mut dash, KeyCode::Left);
        assert_eq!(dash.selected(), Some(SpkId(3)));
        press(&mut dash, KeyCode::Char('m'));
        assert!(screen(&dash).contains("Merge spk_3 into Alice?"));
        press(&mut dash, KeyCode::Right);
        assert!(screen(&dash).contains("Merge spk_3 into Marcus?"));
        assert_eq!(
            press(&mut dash, KeyCode::Enter),
            Action::Send(Command::MergeSpeakers { from: SpkId(3), into: SpkId(1) })
        );
        dash.apply(CoreEvent::SpeakersMerged { from: SpkId(3), into: SpkId(1), undo: UndoId(4) });
        let s = screen(&dash);
        assert!(s.contains("· spk_3 merged into Marcus · lines") && s.contains("updated  u undo"), "{s}");
        assert_eq!(s.matches("00:00 Marcus").count(), 2, "{s}");
        assert!(!s.contains("spk_3 not introduced"), "{s}");
        assert!(s.contains("3 voices"), "{s}");

        // ↑ picks the latest line; m moves it, offering the speakers after Alice.
        press(&mut dash, KeyCode::Up);
        assert!(screen(&dash).contains("m move line"));
        press(&mut dash, KeyCode::Char('m'));
        assert!(screen(&dash).contains("Move u0004 to Marcus?"));
        press(&mut dash, KeyCode::Right);
        assert_eq!(
            press(&mut dash, KeyCode::Enter),
            Action::Send(Command::MoveLine { line: UtteranceId(4), to: SpkId(2) })
        );
        dash.apply(CoreEvent::LineMoved { line: UtteranceId(4), to: SpkId(2), undo: UndoId(5) });
        let s = screen(&dash);
        assert!(s.contains("· u0004 moved from Alice to Jordan") && s.contains("│   u undo "), "{s}");
        assert!(s.contains("00:00 Jordan"), "{s}");

        // u undoes the latest correction, then the one before.
        assert_eq!(press(&mut dash, KeyCode::Char('u')), Action::Send(Command::Undo(UndoId(5))));
        let mut overlay = dash.overlay.clone();
        overlay.moves.clear();
        dash.apply(CoreEvent::Undone { undo: UndoId(5), overlay: overlay.clone() });
        assert_eq!(press(&mut dash, KeyCode::Char('u')), Action::Send(Command::Undo(UndoId(4))));
        overlay.merges.clear();
        dash.apply(CoreEvent::Undone { undo: UndoId(4), overlay });
        let s = screen(&dash);
        assert!(s.contains("· u0004 moved from Alice to Jordan") && s.contains("│   undone "), "{s}");
        assert!(s.contains("· spk_3 merged into Marcus  undone"), "{s}");
        assert!(s.contains("spk_3 not introduced"), "{s}");
    }

    #[test]
    fn r_types_a_new_name_for_the_selected_lines_speaker() {
        let mut dash = split_meeting();
        press(&mut dash, KeyCode::Up);
        press(&mut dash, KeyCode::Up);
        press(&mut dash, KeyCode::Up);
        assert_eq!(dash.selected_line, Some(UtteranceId(2)));
        press(&mut dash, KeyCode::Char('r'));
        for c in "Mqx".chars() {
            press(&mut dash, KeyCode::Char(c));
        }
        press(&mut dash, KeyCode::Backspace);
        assert!(screen(&dash).contains("Rename spk_3 to: Mq"));
        assert_eq!(
            press(&mut dash, KeyCode::Enter),
            Action::Send(Command::RenameSpeaker { spk: SpkId(3), name: "Mq".into() })
        );
        // A selected line that moves takes its new speaker with it.
        dash.apply(CoreEvent::LineMoved { line: UtteranceId(2), to: SpkId(0), undo: UndoId(4) });
        press(&mut dash, KeyCode::Char('r'));
        assert!(screen(&dash).contains("Rename Alice to:"));
        press(&mut dash, KeyCode::Esc);
        // ←→ steps on from the line's speaker.
        press(&mut dash, KeyCode::Right);
        assert_eq!((dash.selected_line, dash.selected()), (None, Some(SpkId(1))));
        // Esc leaves a prompt without sending anything.
        press(&mut dash, KeyCode::Char('r'));
        assert_eq!(press(&mut dash, KeyCode::Esc), Action::None);
        assert_eq!(press(&mut dash, KeyCode::Char('q')), Action::Quit);
    }

    #[test]
    fn the_notes_pane_shows_the_notes_with_names_and_the_stage_what_was_just_noted() {
        let mut dash = Dashboard::new("Sync".into(), "11 Oct 2026".into());
        assert!(screen(&dash).contains("No notes yet."));
        let event = |id: u32, kind: &str, spk: u32, text: &str| -> pam_core::record::Event {
            serde_json::from_value(serde_json::json!({
                "id": format!("e{id:03}"), "type": kind, "t": 16_000 * 65, "by": [format!("spk_{spk}")],
                "text": text, "refs": ["u0001"]
            }))
            .unwrap()
        };
        let events = [
            event(1, "topic", 0, "Onboarding flow"),
            event(2, "concern", 1, "Single page may lose users on mobile"),
            event(3, "decision", 0, "Prototype both flows"),
        ];
        dash.apply(CoreEvent::SpeakerAdded(SpkId(0)));
        dash.apply(CoreEvent::SpeakerRenamed { spk: SpkId(0), name: "Alice".into(), undo: UndoId(1) });
        dash.apply(CoreEvent::NotesUpdated(Notes::build("Choosing a signup flow.", &events, &[])));
        let s = screen(&dash);
        for want in
            ["SUMMARY", "Choosing a signup flow.", "Onboarding flow", "• Concern: Single page", "DECISIONS"]
        {
            assert!(s.contains(want), "{want}: {s}");
        }
        assert!(s.contains("ACTION ITEMS") && s.contains("None yet."), "{s}");
        // Speakers are credited by name once they have one.
        assert!(s.contains("• Prototype both flows") && s.contains("   Alice │"), "{s}");
        assert!(s.contains("spk_1 │"), "{s}");
        assert!(s.contains("JUST NOTED · 01:05") && s.contains("→ Decisions"), "{s}");
        assert!(s.contains("3 notes"), "{s}");
    }

    /// A model as the core reports it.
    fn model(status: ModelStatus, model: Option<&str>, provider: &str, source: Source) -> ModelInfo {
        ModelInfo {
            status,
            reason: None,
            pi: None,
            version: None,
            model: model.map(str::to_string),
            provider: Some(provider.into()).filter(|p: &String| !p.is_empty()),
            source: Some(source),
            login: None,
            catalogued: None,
        }
    }

    #[test]
    fn with_the_scribe_off_neither_the_stage_nor_the_notes_claim_notes() {
        let mut dash = Dashboard::new("Sync".into(), "11 Oct 2026".into());
        assert!(screen(&dash).contains("Taking notes quietly"));
        dash.apply(CoreEvent::Scribe(ModelInfo {
            reason: Some("--scribe off".into()),
            ..model(ModelStatus::Off, None, "", Source::Option)
        }));
        let s = screen(&dash);
        assert!(s.contains("Not taking notes or listening for “Pam, …” (--scribe off)."), "{s}");
        assert!(s.contains("Typed requests still work: press / to ask Pam something."), "{s}");
        assert!(s.contains("Notes are off (--scribe off)."), "{s}");
        assert!(!s.contains("Taking notes quietly"), "{s}");
        assert!(s.contains("11 Oct 2026   scribe off "), "{s}");
    }

    #[test]
    fn a_named_intent_check_shows_in_the_header_and_can_listen_without_the_scribe() {
        let mut dash = Dashboard::new("Sync".into(), "11 Oct 2026".into());
        let haiku = "openrouter/anthropic/claude-haiku-5.5";
        dash.apply(CoreEvent::Scribe(model(ModelStatus::Ready, Some(haiku), "openrouter", Source::Detected)));
        // Following the scribe: the scribe's entry already says where utterances go.
        dash.apply(CoreEvent::Intent(model(ModelStatus::Ready, Some(haiku), "openrouter", Source::Detected)));
        let s = screen(&dash);
        assert!(s.contains("scribe ● claude-haiku-5.5 · openrouter auto ") && !s.contains("intent"), "{s}");

        // Named, it shows; with the scribe off, spoken requests still work.
        dash.apply(CoreEvent::Scribe(ModelInfo {
            reason: Some("--scribe off".into()),
            ..model(ModelStatus::Off, None, "", Source::Option)
        }));
        dash.apply(CoreEvent::Intent(model(
            ModelStatus::Ready,
            Some("local/qwen3.5-4b"),
            "local",
            Source::Option,
        )));
        let s = screen(&dash);
        assert!(s.contains("scribe off   intent ● qwen3.5-4b · local "), "{s}");
        assert!(s.contains("Not taking notes (--scribe off). Say “Pam, …” or press /"), "{s}");

        // One that couldn't load says why, apart from the notes.
        dash.apply(CoreEvent::Scribe(model(ModelStatus::Ready, Some(haiku), "openrouter", Source::Detected)));
        dash.apply(CoreEvent::Intent(ModelInfo {
            reason: Some("the local model didn't load: no memory".into()),
            ..model(ModelStatus::Unavailable, Some("local/qwen3.5-4b"), "local", Source::Config)
        }));
        let s = screen(&dash);
        assert!(s.contains("Taking notes quietly. Not listening for “Pam, …”. Typed"), "{s}");
        assert!(s.contains("No spoken requests: the local model didn't load:"), "{s}");
        assert!(!s.contains("No notes:") && !s.contains("No notes or"), "{s}");
    }

    #[test]
    fn a_scribe_with_no_model_says_why_and_that_typed_requests_still_work() {
        let mut dash = Dashboard::new("Sync".into(), "11 Oct 2026".into());
        dash.apply(CoreEvent::Scribe(ModelInfo {
            reason: Some("pi has a login for none of the scribe models Pam tries".into()),
            ..model(ModelStatus::Unavailable, None, "", Source::Detected)
        }));
        dash.apply(CoreEvent::Agent(model(
            ModelStatus::Ready,
            Some("openai/gpt-6-astra"),
            "openai",
            Source::Config,
        )));
        let s = screen(&dash);
        assert!(s.contains("Not taking notes or listening for “Pam, …”. Typed requests"), "{s}");
        assert!(s.contains("No notes or spoken requests: pi has a login for none of the"), "{s}");
        assert!(s.contains("Notes are off: the scribe's model"), "{s}");
        assert!(s.contains("scribe ● none   agent ● gpt-6-astra · openai "), "{s}");

        // The agent off too: nothing to ask.
        dash.apply(CoreEvent::Agent(ModelInfo {
            reason: Some("agent_model = \"off\" in the config file".into()),
            ..model(ModelStatus::Off, None, "", Source::Config)
        }));
        let s = screen(&dash);
        assert!(s.contains("Not taking notes.") && !s.contains("Typed requests"), "{s}");
        assert!(s.contains("Pam takes no requests (agent_model = \"off\" in the config"), "{s}");
        assert!(s.contains("agent off "), "{s}");
    }

    #[test]
    fn the_header_shows_the_cost_so_far_after_the_models() {
        let mut dash = Dashboard::new("Onboarding sync".into(), "11 Oct 2026".into());
        let haiku = "openrouter/anthropic/claude-haiku-5.5";
        dash.apply(CoreEvent::Scribe(model(ModelStatus::Ready, Some(haiku), "openrouter", Source::Detected)));
        dash.apply(CoreEvent::Agent(model(
            ModelStatus::Ready,
            Some("openai/gpt-6-astra"),
            "openai",
            Source::Config,
        )));
        let header = |dash: &Dashboard, w| screen_at(dash, w).lines().next().unwrap().to_string();
        assert!(!header(&dash, 180).contains('$'), "none until the meeting says");

        let mut usage = Usage::new(Some("api_key"), Some("api_key"), Some("oauth"));
        let call = |cost| pam_core::usage::Call { input: 1, output: 1, cache_read: 0, cache_write: 0, cost };
        usage.add(pam_core::usage::User::Scribe, &call(Some(0.12)));
        dash.apply(CoreEvent::Usage(Box::new(usage.clone())));
        assert!(
            header(&dash, 180).contains("agent ● gpt-6-astra · openai   $0.120 "),
            "{}",
            header(&dash, 180)
        );
        // At API prices for a subscription login, and partial.
        usage.add(pam_core::usage::User::Agent, &call(Some(0.08)));
        usage.add(pam_core::usage::User::Agent, &call(None));
        dash.apply(CoreEvent::Usage(Box::new(usage)));
        assert!(header(&dash, 180).contains("openai   ≈$0.200+ "), "{}", header(&dash, 180));
        // Narrow: the IDs go, the cost stays.
        let narrow = header(&dash, 130);
        assert!(narrow.contains("scribe ● openrouter auto   agent ● openai   ≈$0.200+ "), "{narrow}");
    }

    #[test]
    fn the_header_shows_both_models_and_drops_their_ids_when_narrow() {
        let mut dash = Dashboard::new("Onboarding sync".into(), "11 Oct 2026".into());
        let haiku = "openrouter/anthropic/claude-haiku-5.5";
        dash.apply(CoreEvent::Scribe(model(ModelStatus::Ready, Some(haiku), "openrouter", Source::Detected)));
        dash.apply(CoreEvent::Agent(model(
            ModelStatus::Ready,
            Some("openai/gpt-6-astra:high"),
            "openai",
            Source::Detected,
        )));
        let full = "scribe ● claude-haiku-5.5 · openrouter auto   agent ● gpt-6-astra:high · openai auto";
        let header = |dash: &Dashboard, w| screen_at(dash, w).lines().next().unwrap().to_string();
        assert!(header(&dash, 180).contains(full), "{}", header(&dash, 180));
        let narrow = header(&dash, 130);
        assert!(narrow.contains("scribe ● openrouter auto   agent ● openai auto "), "{narrow}");
        assert!(narrow.contains("Onboarding sync") && narrow.contains("00:00"), "{narrow}");
        assert!(!narrow.contains("claude"), "{narrow}");

        // A named model with no login: a red dot, no `auto`, and the stage says why.
        dash.apply(CoreEvent::Agent(ModelInfo {
            reason: Some("pi has no login for anthropic/claude-sonnet-5-5".into()),
            ..model(
                ModelStatus::Unavailable,
                Some("anthropic/claude-sonnet-5-5"),
                "anthropic",
                Source::Config,
            )
        }));
        assert!(
            header(&dash, 180).contains("agent ● claude-sonnet-5-5 · anthropic "),
            "{}",
            header(&dash, 180)
        );
        assert!(!header(&dash, 180).contains("anthropic auto"), "{}", header(&dash, 180));
        let mut term = ratatui::Terminal::new(ratatui::backend::TestBackend::new(180, 30)).unwrap();
        term.draw(|f| dash.draw(f)).unwrap();
        let buf = term.backend().buffer();
        let dots: Vec<Color> =
            (0..180).filter(|&x| buf[(x, 0)].symbol() == "●").map(|x| buf[(x, 0)].fg).collect();
        assert_eq!(dots[..2], [OK, BAD], "the REC dot comes after");
        assert!(screen_at(&dash, 180).contains("Pam can't take requests: pi has no login"));
    }

    #[test]
    fn a_note_is_credited_again_to_its_speaker_when_a_merge_is_undone() {
        let mut dash = Dashboard::new("Sync".into(), "11 Oct 2026".into());
        let point: pam_core::record::Event = serde_json::from_value(serde_json::json!({
            "id": "e001", "type": "point", "t": 0, "by": ["spk_0"], "text": "Ship on Friday", "refs": ["u0001"]
        }))
        .unwrap();
        dash.apply(CoreEvent::SpeakerAdded(SpkId(0)));
        dash.apply(CoreEvent::SpeakerAdded(SpkId(1)));
        dash.apply(CoreEvent::SpeakerRenamed { spk: SpkId(1), name: "Marcus".into(), undo: UndoId(1) });
        dash.apply(CoreEvent::SpeakersMerged { from: SpkId(0), into: SpkId(1), undo: UndoId(2) });
        dash.apply(CoreEvent::NotesUpdated(Notes::build("", &[point], &[])));
        assert!(screen(&dash).contains("Marcus │"), "{}", screen(&dash));
        // Only Undone follows: the notes the dashboard has must still say spk_0.
        let overlay = Overlay { names: [(SpkId(1), "Marcus".into())].into(), ..Overlay::default() };
        dash.apply(CoreEvent::Undone { undo: UndoId(2), overlay });
        assert!(screen(&dash).contains("spk_0 │"), "{}", screen(&dash));
    }

    #[test]
    fn wraps_words() {
        assert_eq!(wrap("one two three four", 9), ["one two", "three", "four"]);
        assert_eq!(wrap("abcdefghij", 4), ["abcd", "efgh", "ij"]);
        assert!(wrap("", 10).is_empty());
    }

    #[test]
    fn wraps_cjk_text_by_columns_between_characters() {
        // Each character is two columns; punctuation stays with the one before it.
        let got = wrap("大家好，我是王磊。我们这边", 10);
        assert_eq!(got, ["大家好，我", "是王磊。我", "们这边"]);
        assert!(got.iter().all(|l| l.width() <= 10), "{got:?}");
        assert_eq!(wrap("OK 那就周五吧 Friday then", 12), ["OK 那就周五", "吧 Friday", "then"]);
        assert_eq!(wrap("没问题, 我周三之前", 8), ["没问题,", "我周三之", "前"]);
    }

    #[test]
    fn initials_are_two_columns_wide() {
        assert_eq!(initials("Alice"), "A ");
        assert_eq!(initials("Alice Chen"), "AC");
        assert_eq!(initials("王磊"), "王");
        assert_eq!(initials("王 磊"), "王");
    }

    fn task(id: u32, status: TaskStatus) -> TaskState {
        TaskState::new(TaskId(id), &format!("question {id}?"), 16_000 * 62, pam_core::Via::Typed, status)
    }

    #[test]
    fn slash_types_a_request_and_its_answer_takes_the_stage() {
        let mut dash = Dashboard::new("Pricing".into(), "11 Oct 2026".into());
        assert!(screen(&dash).contains("/ Ask Pam…"));
        press(&mut dash, KeyCode::Char('/'));
        assert_eq!(press(&mut dash, KeyCode::Enter), Action::None, "nothing typed, nothing sent");
        press(&mut dash, KeyCode::Char('/'));
        for c in "pricing??".chars() {
            press(&mut dash, KeyCode::Char(c));
        }
        press(&mut dash, KeyCode::Backspace);
        let s = screen(&dash);
        assert!(s.contains("/ pricing?▏") && s.contains("Enter ask Pam   Esc cancel"), "{s}");
        assert_eq!(press(&mut dash, KeyCode::Enter), Action::Send(Command::Ask("pricing?".into())));

        dash.apply(CoreEvent::Clock { now: 16_000 * 62 });
        dash.apply(CoreEvent::TaskUpdated(task(1, TaskStatus::Running)));
        dash.apply(CoreEvent::TaskUpdated(task(2, TaskStatus::Queued)));
        // A task shows its card until it says something.
        let s = screen(&dash);
        assert!(s.contains("● Listening   ⠋ t01 question 1?   ○ t02 question 2? queued"), "{s}");
        assert!(s.contains("Running 1 task, 1 queued. Still listening."), "{s}");
        assert!(s.contains("│ typed · 01:02 · starting") && s.contains("│ ⠋ Thinking…"), "{s}");
        assert!(s.contains("│ typed · 01:02 · waits for t01"), "{s}");

        // Its reports show on its chip and its card as they come.
        dash.apply(CoreEvent::Clock { now: 16_000 * 132 });
        dash.apply(CoreEvent::TaskUpdated(TaskState {
            progress: Some(60),
            action: Some("reading source 3 of 5".into()),
            started_at: Some(16_000 * 62),
            ..task(1, TaskStatus::Running)
        }));
        let s = screen(&dash);
        assert!(s.contains("⠋ t01 question 1? 3/5   ○ t02"), "{s}");
        assert!(s.contains("─ 3/5 ╮") && s.contains("│ ████████████"), "{s}");
        assert!(
            s.contains("│ typed · 01:02 · running 1m 10s") && s.contains("│ ⠋ reading source 3 of 5"),
            "{s}"
        );

        let say = |text: &str| CoreEvent::StageMessage {
            task: Some(TaskId(1)),
            text: text.into(),
            at: 16_000 * 132,
        };
        dash.apply(say("We settled on 0 a seat."));
        let s = screen(&dash);
        assert!(
            s.contains("ANSWERING  typed  · t01 · asked at 01:02") && s.contains("  We settled on 0 a seat.")
        );
        assert!(s.contains("“question 1?”") && s.contains("Next: t02 “question 2?”"), "{s}");
        // Saying again replaces it, so the agent can correct itself.
        dash.apply(say("We settled on 40 a seat."));
        dash.apply(CoreEvent::TaskUpdated(task(1, TaskStatus::Done)));
        let s = screen(&dash);
        assert!(s.contains("ANSWERED  typed") && s.contains("  We settled on 40 a seat."), "{s}");
        assert!(!s.contains("Thinking…") && !s.contains("on 0 a seat"), "{s}");
        assert!(s.contains("● Listening   ✓ t01 done   ○ t02 queued "), "too long for their words: {s}");

        // The answer keeps the stage while it is fresh, then gives way to the next task's card.
        dash.apply(CoreEvent::TaskUpdated(task(2, TaskStatus::Running)));
        assert!(screen(&dash).contains("ANSWERED  typed"));
        dash.apply(CoreEvent::Clock { now: 16_000 * 162 });
        let s = screen(&dash);
        assert!(s.contains("Running 1 task. Still listening.") && s.contains("╭ t02 question 2? ─"), "{s}");
        // A failure is the task's answer.
        dash.apply(CoreEvent::TaskUpdated(TaskState {
            error: Some("no login".into()),
            ..task(2, TaskStatus::Failed)
        }));
        let s = screen(&dash);
        assert!(s.contains("NOT ANSWERED  typed") && s.contains("Couldn't answer: no login"), "{s}");
        assert!(s.contains("✓ t01 done   ✗ t02 failed"), "{s}");
        // Esc takes it off the stage, for the card of the task that runs, then listening.
        dash.apply(CoreEvent::TaskUpdated(task(3, TaskStatus::Running)));
        assert!(screen(&dash).contains("NOT ANSWERED"), "fresh still");
        press(&mut dash, KeyCode::Esc);
        assert!(screen(&dash).contains("╭ t03 question 3? ─"));
        dash.apply(CoreEvent::TaskUpdated(task(3, TaskStatus::Done)));
        let s = screen(&dash);
        assert!(s.contains("Say “Pam, …” or press / to ask") && s.contains("3 requests"), "{s}");
        assert!(s.contains("✗ t02 failed   ✓ t03 done "), "a task that said nothing stays on the strip: {s}");
    }

    #[test]
    fn a_replayed_answer_is_as_old_as_the_core_says_not_as_the_clock_it_came_before() {
        // A late subscriber gets the history first and the clock after it.
        let mut dash = Dashboard::new("Pricing".into(), "11 Oct 2026".into());
        dash.apply(CoreEvent::TaskUpdated(task(1, TaskStatus::Done)));
        dash.apply(CoreEvent::StageMessage {
            task: Some(TaskId(1)),
            text: "Forty a seat.".into(),
            at: 16_000 * 100,
        });
        dash.apply(CoreEvent::TaskUpdated(task(3, TaskStatus::Running)));
        dash.apply(CoreEvent::Clock { now: 16_000 * 110 });
        let s = screen(&dash);
        assert!(s.contains("ANSWERED  typed") && s.contains("Forty a seat."), "fresh, given at 01:40: {s}");
        dash.apply(CoreEvent::Clock { now: 16_000 * 130 });
        assert!(screen(&dash).contains("╭ t03 question 3? ─"), "stale 30 s after it was given");
        // A failure counts from when it ended.
        let mut dash = Dashboard::new("Pricing".into(), "11 Oct 2026".into());
        dash.apply(CoreEvent::TaskUpdated(TaskState {
            ended_at: Some(16_000 * 100),
            ..task(2, TaskStatus::Failed)
        }));
        dash.apply(CoreEvent::TaskUpdated(task(3, TaskStatus::Running)));
        dash.apply(CoreEvent::Clock { now: 16_000 * 110 });
        assert!(screen(&dash).contains("NOT ANSWERED"));
        dash.apply(CoreEvent::Clock { now: 16_000 * 130 });
        assert!(screen(&dash).contains("╭ t03 question 3? ─"));
    }

    #[test]
    fn the_strip_drops_words_then_folds_finished_tasks_and_tab_moves_the_focus() {
        let mut dash = Dashboard::new("Pricing".into(), "11 Oct 2026".into());
        let s = screen(&dash);
        assert!(s.contains("● Listening    no tasks · say “Pam, …” to start one"), "{s}");
        for id in 1..=6 {
            dash.apply(CoreEvent::TaskUpdated(task(id, TaskStatus::Done)));
        }
        dash.apply(CoreEvent::TaskUpdated(task(7, TaskStatus::Running)));
        let s = screen_at(&dash, 300);
        assert!(s.contains("✓ t06 question 6? done   ⠋ t07 question 7? "), "{s}");
        dash.apply(CoreEvent::TaskUpdated(task(7, TaskStatus::Cancelled)));
        press(&mut dash, KeyCode::Esc);
        dash.apply(CoreEvent::TaskUpdated(task(8, TaskStatus::Running)));
        dash.apply(CoreEvent::TaskUpdated(task(9, TaskStatus::Queued)));
        let s = screen_at(&dash, 220);
        assert!(s.contains("● Listening   ✓ t01 done   ✓ t02 done"), "too many for their words: {s}");
        assert!(!s.contains("t07"), "a cancelled task's chip goes at once: {s}");
        let s = screen(&dash);
        assert!(
            s.contains("● Listening   +5 earlier   ✓ t06 done   ⠋ t08   ○ t09 queued "),
            "the oldest finished fold: {s}"
        );

        // Tab moves the accent border from the first card to the next, and round.
        let focused = |dash: &Dashboard| dash.focused();
        assert_eq!(focused(&dash), Some(TaskId(8)));
        assert!(screen(&dash).contains("Tab next card"));
        press(&mut dash, KeyCode::Tab);
        assert_eq!(focused(&dash), Some(TaskId(9)));
        press(&mut dash, KeyCode::Tab);
        assert_eq!(focused(&dash), Some(TaskId(8)));
        dash.apply(CoreEvent::TaskUpdated(task(8, TaskStatus::Done)));
        dash.apply(CoreEvent::TaskUpdated(task(9, TaskStatus::Running)));
        assert_eq!(focused(&dash), Some(TaskId(9)), "a card that went falls back to the first");
    }

    #[test]
    fn an_unavailable_agent_says_why_on_the_stage() {
        let mut dash = Dashboard::new("Pricing".into(), "11 Oct 2026".into());
        dash.apply(CoreEvent::Agent(AgentInfo {
            reason: Some("pi has no login for stub/model".into()),
            ..model(ModelStatus::Unavailable, Some("stub/model"), "stub", Source::Config)
        }));
        let s = screen(&dash);
        assert!(s.contains("Pam can't take requests: pi has no login for stub/model"), "{s}");
        assert!(!s.contains("press / to ask"), "{s}");
        dash.failed(&Command::Ask("x".into()), &anyhow::anyhow!("the agent is unavailable"));
        assert!(screen(&dash).contains("Can't ask: the agent is unavailable"));
    }

    #[test]
    fn a_spoken_request_counts_down_and_esc_cancels_it() {
        let mut dash = Dashboard::new("Launch".into(), "11 Oct 2026".into());
        dash.set_name("Jarvis".into());
        dash.apply(CoreEvent::SpeakerAdded(SpkId(2)));
        dash.apply(CoreEvent::SpeakerRenamed { spk: SpkId(2), name: "Jordan".into(), undo: UndoId(1) });
        let asked = Line {
            id: UtteranceId(7),
            t0: 16_000 * 62,
            t1: 16_000 * 64,
            slot: Some(2),
            spk: Some(SpkId(2)),
            lang: None,
            text: "Jarvis, mock up both flows.".into(),
        };
        dash.apply(CoreEvent::LineFinal(asked));
        dash.apply(CoreEvent::Clock { now: 16_000 * 66 });
        dash.apply(CoreEvent::TaskUpdated(TaskState {
            request: "mock up both flows.".into(),
            line: Some(UtteranceId(7)),
            ..task(2, TaskStatus::Running)
        }));
        dash.apply(CoreEvent::Invocation {
            task: TaskId(2),
            line: Some(UtteranceId(7)),
            spk: None,
            summary: "Mock up both onboarding flows side by side".into(),
            cancel_deadline: 16_000 * 71,
        });
        let s = screen(&dash);
        for want in [
            "Starting t02",
            "Jordan · 01:02",
            "Mock up both onboarding flows side by side",
            "starts in 5s",
            "Say “Jarvis, cancel” to stop it.",
            "Esc cancel t02",
            "… t02 mock up both flow… in 5s",
            "→ Jarvis t02",
        ] {
            assert!(s.contains(want), "{want}: {s}");
        }
        dash.apply(CoreEvent::Clock { now: 16_000 * 69 });
        assert!(screen(&dash).contains("starts in 2s"));
        assert_eq!(press(&mut dash, KeyCode::Esc), Action::Send(Command::CancelTask(TaskId(2))));

        // Cancelled: the pop-up goes, and the stage says so.
        dash.apply(CoreEvent::TaskUpdated(TaskState {
            request: "mock up both flows.".into(),
            line: Some(UtteranceId(7)),
            ..task(2, TaskStatus::Cancelled)
        }));
        let s = screen(&dash);
        assert!(!s.contains("Starting t02"), "{s}");
        assert!(s.contains("CANCELLED  Jordan") && s.contains("Cancelled."), "{s}");
        // Esc no longer cancels: it takes the task off the stage.
        assert_eq!(press(&mut dash, KeyCode::Esc), Action::None);
        let s = screen(&dash);
        assert!(s.contains("Say “Jarvis, …” or press / to ask"), "{s}");

        // Two windows open at once: each chip counts down its own.
        dash.apply(CoreEvent::Clock { now: 16_000 * 80 });
        for (id, deadline) in [(3, 86), (4, 88)] {
            dash.apply(CoreEvent::TaskUpdated(task(id, TaskStatus::Running)));
            dash.apply(CoreEvent::Invocation {
                task: TaskId(id),
                line: None,
                spk: None,
                summary: format!("question {id}"),
                cancel_deadline: 16_000 * deadline,
            });
        }
        let s = screen(&dash);
        assert!(s.contains("t03 in 6s") && s.contains("t04 in 8s") && s.contains("Starting t04"), "{s}");
    }

    #[test]
    fn a_question_pops_up_with_a_countdown_and_number_keys_answer_it() {
        let mut dash = Dashboard::new("Launch".into(), "11 Oct 2026".into());
        dash.apply(CoreEvent::TaskUpdated(TaskState {
            request: "draft the email to the customer".into(),
            ..task(2, TaskStatus::Running)
        }));
        let question = |dash: &Dashboard| CoreEvent::Question {
            task: TaskId(2),
            q: 1,
            text: "Which customer is the email for?".into(),
            options: vec!["Acme".into(), "Globex".into()],
            deadline: dash.wall + std::time::Duration::from_secs(15),
        };
        dash.apply(question(&dash));
        let s = screen(&dash);
        for want in [
            "Pam is asking · t02",
            "15s left",
            "Which customer is the email for?",
            "Typed: “draft the email to the customer”",
            " 1   Acme",
            " 2   Globex",
            "If nobody answers, Pam goes with its best guess and says",
            "1–2 answer",
            "or just say it",
            "? t02 draft the email t… asking",
        ] {
            assert!(s.contains(want), "{want}: {s}");
        }
        dash.wall += std::time::Duration::from_secs(10);
        assert!(screen(&dash).contains(" 5s left "));
        dash.wall += std::time::Duration::from_secs(5);
        assert!(screen(&dash).contains(" hearing a reply… "), "past the deadline, until the core closes it");
        dash.wall -= std::time::Duration::from_secs(5);
        // The core decides when a question ends, not the task's status.
        let done =
            TaskState { request: "draft the email to the customer".into(), ..task(2, TaskStatus::Done) };
        dash.apply(CoreEvent::TaskUpdated(done));
        assert!(screen(&dash).contains("Pam is asking · t02"));

        assert_eq!(press(&mut dash, KeyCode::Char('3')), Action::None);
        assert!(screen(&dash).contains("t02's question has options 1–2."));
        assert_eq!(
            press(&mut dash, KeyCode::Char('2')),
            Action::Send(Command::Answer { task: TaskId(2), q: 1, answer: Answer::Option(2) })
        );
        dash.apply(CoreEvent::QuestionClosed {
            task: TaskId(2),
            end: End::Answered { answer: Answer::Option(2), by: By::Key, spk: None },
        });
        let s = screen(&dash);
        assert!(!s.contains("Pam is asking"), "{s}");
        assert!(s.contains("· t02 · option 2 chosen on the"), "{s}");
        assert_eq!(press(&mut dash, KeyCode::Char('1')), Action::None, "no question, no answer");

        // A digit typed in the composer is text, not an answer.
        dash.apply(question(&dash));
        press(&mut dash, KeyCode::Char('/'));
        assert_eq!(press(&mut dash, KeyCode::Char('1')), Action::None);
        assert!(screen(&dash).contains("/ 1▏"));
        press(&mut dash, KeyCode::Esc);
        dash.apply(CoreEvent::QuestionClosed { task: TaskId(2), end: End::TimedOut });
        assert!(screen(&dash).contains("· t02 · no answer in time: Pam"));

        // With the scribe off nothing hears a reply without the name.
        dash.apply(CoreEvent::Scribe(ModelInfo {
            reason: Some("--scribe off".into()),
            ..model(ModelStatus::Off, None, "", Source::Option)
        }));
        dash.apply(question(&dash));
        let s = screen(&dash);
        assert!(s.contains("1–2 answer   q stop   or say “Pam, option two”"), "{s}");
        assert!(!s.contains("just say it"), "{s}");

        // The intent check hears replies, not the scribe: on this Mac with no notes…
        dash.apply(CoreEvent::Intent(model(
            ModelStatus::Ready,
            Some("local/qwen3.5-4b"),
            "local",
            Source::Option,
        )));
        assert!(screen(&dash).contains("1–2 answer   q stop   or just say it"));
        // …and none with notes.
        let haiku = "openrouter/anthropic/claude-haiku-5.5";
        dash.apply(CoreEvent::Scribe(model(ModelStatus::Ready, Some(haiku), "openrouter", Source::Detected)));
        dash.apply(CoreEvent::Intent(ModelInfo {
            reason: Some("--intent off".into()),
            ..model(ModelStatus::Off, None, "", Source::Option)
        }));
        let s = screen(&dash);
        assert!(s.contains("or say “Pam, option two”") && !s.contains("just say it"), "{s}");
    }

    #[test]
    fn an_open_question_is_answered_by_typing_and_its_answer_goes_in_the_transcript() {
        let mut dash = Dashboard::new("Weather".into(), "14 Oct 2026".into());
        dash.apply(CoreEvent::TaskUpdated(TaskState {
            request: "get me the weather for tomorrow morning".into(),
            ..task(1, TaskStatus::Running)
        }));
        let question = |dash: &Dashboard, q| CoreEvent::Question {
            task: TaskId(1),
            q,
            text: "Which city are you in?".into(),
            options: Vec::new(),
            deadline: dash.wall + std::time::Duration::from_secs(30),
        };
        dash.apply(question(&dash, 1));
        let s = screen(&dash);
        for want in [
            "Pam is asking · t01",
            "30s left",
            "Which city are you in?",
            "Say your answer, or type it with /",
            "If nobody answers, Pam goes with its best guess and says",
            "/ answer   q stop   or just say it",
            "/ Type the answer to t01…",
        ] {
            assert!(s.contains(want), "{want}: {s}");
        }
        assert!(!s.contains(" 1   "), "no options: {s}");
        // Keys 1–9 don't answer it.
        assert_eq!(press(&mut dash, KeyCode::Char('2')), Action::None);
        assert!(screen(&dash).contains("t01's question is open: type the answer with /."));

        // What is typed answers it, instead of making a request.
        press(&mut dash, KeyCode::Char('/'));
        for c in "Auckland".chars() {
            press(&mut dash, KeyCode::Char(c));
        }
        assert!(screen(&dash).contains("Enter answer t01"));
        assert_eq!(
            press(&mut dash, KeyCode::Enter),
            Action::Send(Command::Answer { task: TaskId(1), q: 1, answer: Answer::Text("Auckland".into()) })
        );
        dash.apply(CoreEvent::QuestionClosed {
            task: TaskId(1),
            end: End::Answered { answer: Answer::Text("Auckland".into()), by: By::Typed, spk: None },
        });
        let s = screen(&dash);
        assert!(!s.contains("Pam is asking"), "{s}");
        assert!(s.contains("· t01 · answered on the laptop:") && s.contains("│   Auckland"), "{s}");
        assert!(s.contains("/ Ask Pam…"), "{s}");
        // With no question up, the composer asks again.
        press(&mut dash, KeyCode::Char('/'));
        press(&mut dash, KeyCode::Char('x'));
        assert_eq!(press(&mut dash, KeyCode::Enter), Action::Send(Command::Ask("x".into())));

        // An answer still being typed when its question closes stays an answer, which the
        // core refuses: it never becomes a request.
        dash.apply(question(&dash, 1));
        press(&mut dash, KeyCode::Char('/'));
        press(&mut dash, KeyCode::Char('y'));
        dash.apply(CoreEvent::QuestionClosed { task: TaskId(1), end: End::TimedOut });
        assert!(screen(&dash).contains("Enter answer t01"));
        assert_eq!(
            press(&mut dash, KeyCode::Enter),
            Action::Send(Command::Answer { task: TaskId(1), q: 1, answer: Answer::Text("y".into()) })
        );
        // Nor does the task's next question take it: the answer names the question it was for.
        dash.apply(question(&dash, 1));
        press(&mut dash, KeyCode::Char('/'));
        press(&mut dash, KeyCode::Char('w'));
        dash.apply(CoreEvent::QuestionClosed { task: TaskId(1), end: End::TimedOut });
        dash.apply(question(&dash, 2));
        assert_eq!(
            press(&mut dash, KeyCode::Enter),
            Action::Send(Command::Answer { task: TaskId(1), q: 1, answer: Answer::Text("w".into()) })
        );
        dash.apply(CoreEvent::QuestionClosed { task: TaskId(1), end: End::TimedOut });
        // And a request begun before a question shows stays a request.
        press(&mut dash, KeyCode::Char('/'));
        press(&mut dash, KeyCode::Char('z'));
        dash.apply(question(&dash, 1));
        assert!(screen(&dash).contains("Enter ask Pam"));
        assert_eq!(press(&mut dash, KeyCode::Enter), Action::Send(Command::Ask("z".into())));
        dash.apply(CoreEvent::QuestionClosed { task: TaskId(1), end: End::TimedOut });

        // An answer by voice says who.
        dash.apply(question(&dash, 1));
        dash.apply(CoreEvent::QuestionClosed {
            task: TaskId(1),
            end: End::Answered {
                answer: Answer::Text("Wellington".into()),
                by: By::Voice,
                spk: Some(SpkId(2)),
            },
        });
        let s = screen(&dash);
        assert!(s.contains("· t01 · answered by spk_2:") && s.contains("Wellington"), "{s}");

        // With the scribe off, nothing hears a spoken answer: it can only be typed.
        dash.apply(CoreEvent::Scribe(ModelInfo {
            reason: Some("--scribe off".into()),
            ..model(ModelStatus::Off, None, "", Source::Option)
        }));
        dash.apply(question(&dash, 1));
        let s = screen(&dash);
        assert!(s.contains("Type your answer with /") && !s.contains("Say your answer"), "{s}");
        assert!(s.contains("/ answer   q stop") && !s.contains("say it") && !s.contains("option two"), "{s}");
    }

    #[test]
    fn the_window_closes_at_its_deadline_and_corrections_show_under_the_request() {
        let mut dash = Dashboard::new("Launch".into(), "11 Oct 2026".into());
        dash.apply(CoreEvent::Clock { now: 16_000 * 10 });
        dash.apply(CoreEvent::TaskUpdated(TaskState {
            line: Some(UtteranceId(1)),
            ..task(1, TaskStatus::Running)
        }));
        dash.apply(CoreEvent::Invocation {
            task: TaskId(1),
            line: Some(UtteranceId(1)),
            spk: None,
            summary: "Find the price".into(),
            cancel_deadline: 16_000 * 15,
        });
        assert!(screen(&dash).contains("Starting t01"));
        dash.apply(CoreEvent::Clock { now: 16_000 * 15 });
        let s = screen(&dash);
        assert!(!s.contains("Starting t01") && s.contains("● Listening   ⠋ t01 question 1? "), "{s}");
        assert!(s.contains("Running 1 task. Still listening.") && s.contains("╭ t01 question 1? ─"), "{s}");
        dash.apply(CoreEvent::TaskUpdated(TaskState {
            line: Some(UtteranceId(1)),
            corrections: vec!["I meant the annual price.".into()],
            ..task(1, TaskStatus::Running)
        }));
        assert!(screen(&dash).contains("then: “I meant the annual price.”"));
    }

    #[test]
    fn a_request_announced_before_its_line_names_the_speaker_so_far_and_keeps_its_countdown() {
        let mut dash = Dashboard::new("Launch".into(), "11 Oct 2026".into());
        dash.apply(CoreEvent::SpeakerAdded(SpkId(1)));
        dash.apply(CoreEvent::Clock { now: 16_000 * 10 });
        dash.apply(CoreEvent::TaskUpdated(task(1, TaskStatus::Running)));
        let invocation = |line| CoreEvent::Invocation {
            task: TaskId(1),
            line,
            spk: Some(SpkId(1)),
            summary: "Find the price".into(),
            cancel_deadline: 16_000 * 15,
        };
        dash.apply(invocation(None));
        let s = screen(&dash);
        assert!(s.contains("Starting t01") && s.contains("spk_1 · 01:02"), "{s}");
        dash.apply(CoreEvent::Clock { now: 16_000 * 15 });
        assert!(screen(&dash).contains("│ spk_1 · 01:02 · starting"), "its card names them too");
        dash.apply(CoreEvent::Clock { now: 16_000 * 12 });
        dash.apply(CoreEvent::LineFinal(Line {
            id: UtteranceId(4),
            t0: 16_000 * 9,
            t1: 16_000 * 10,
            slot: Some(1),
            spk: Some(SpkId(1)),
            lang: None,
            text: "Pam, find the price.".into(),
        }));
        dash.apply(invocation(Some(UtteranceId(4))));
        // The window it opened at 00:10 stays: 3 s of 5 are left.
        let inv = &dash.invocations[&TaskId(1)];
        assert_eq!((inv.opened, inv.line), (16_000 * 10, Some(UtteranceId(4))));
        assert!(screen(&dash).contains("→ Pam t01"));
    }

    fn with_result(t: TaskState, title: &str, target: Target) -> TaskState {
        TaskState { result: Some(pam_core::TaskResult { title: title.into(), target }), ..t }
    }

    #[test]
    fn a_result_waits_on_its_chip_until_enter_shows_it_and_o_opens_it() {
        let dir = tempfile::tempdir().unwrap();
        let folder = dir.path().to_path_buf();
        std::fs::create_dir_all(folder.join("tasks/t01")).unwrap();
        std::fs::write(
            folder.join("tasks/t01/result.md"),
            "# Sources\nPartly true.\n1. [Study](https://example.com/study)\n2. Notes: [interviews](../../notes.md)\n",
        )
        .unwrap();
        std::fs::write(folder.join("notes.md"), "notes").unwrap();
        let mut dash = Dashboard::new("Pricing".into(), "11 Oct 2026".into());
        dash.set_folder(folder.clone());
        dash.apply(CoreEvent::Clock { now: 16_000 * 62 });
        let running = TaskState { started_at: Some(16_000 * 62), ..task(1, TaskStatus::Running) };
        dash.apply(CoreEvent::TaskUpdated(running.clone()));
        // Its verdict, said just before the result came: the result keeps it.
        let verdict = "Partly true: it helps when the steps cut are optional.";
        dash.apply(CoreEvent::StageMessage { task: Some(TaskId(1)), text: verdict.into(), at: 16_000 * 70 });
        assert!(screen(&dash).contains("ANSWERING"));
        let file = || Target::File("tasks/t01/result.md".into());
        dash.apply(CoreEvent::TaskUpdated(with_result(running.clone(), "Sources", file())));
        let s = screen(&dash);
        assert!(!s.contains("ANSWERING") && s.contains("╭ t01 question 1? ─"), "back to its card: {s}");
        dash.apply(CoreEvent::TaskUpdated(with_result(
            TaskState {
                ended_at: Some(16_000 * 187),
                ..TaskState { status: TaskStatus::Done, ..running.clone() }
            },
            "Sources",
            file(),
        )));
        dash.apply(CoreEvent::Clock { now: 16_000 * 190 });
        // It never takes the stage: the chip turns ready.
        let s = screen(&dash);
        assert!(s.contains("Taking notes quietly.") || s.contains("Listening"), "{s}");
        assert!(!s.contains(verdict), "{s}");
        assert!(s.contains("✓ t01 question 1? ready"), "{s}");
        assert!(s.contains("· t01 · ready: Sources"), "said in the transcript: {s}");
        assert!(s.contains("Enter show t01   o open"), "{s}");
        let path = |p: &str| folder.join(p).display().to_string();
        assert_eq!(press(&mut dash, KeyCode::Char('o')), Action::Open(path("tasks/t01/result.md")));

        // Enter shows it: title, who asked and when, how long it took, the verdict, the path, the text.
        press(&mut dash, KeyCode::Enter);
        let s = screen(&dash);
        assert!(s.contains("t01 RESULT") && s.contains("✓ ready · took 2m 05s"), "{s}");
        assert!(s.contains("│  Sources ") && s.contains("typed asked at 01:02"), "{s}");
        assert!(s.contains("“question 1?”") && s.contains(verdict), "{s}");
        assert!(
            s.contains("→ tasks/t01/result.md") && s.contains("1. [Study](https://example.com/study)"),
            "{s}"
        );
        assert!(s.contains("↑↓ line   o open   Esc back"), "{s}");
        // ↑↓ pick a line; o opens what it points at, else the result.
        press(&mut dash, KeyCode::Down);
        assert_eq!(press(&mut dash, KeyCode::Char('o')), Action::Open(path("tasks/t01/result.md")));
        press(&mut dash, KeyCode::Down);
        press(&mut dash, KeyCode::Down);
        assert_eq!(press(&mut dash, KeyCode::Char('o')), Action::Open("https://example.com/study".into()));
        press(&mut dash, KeyCode::Down);
        press(&mut dash, KeyCode::Down);
        let beside = folder.join("tasks/t01").join("../../notes.md").display().to_string();
        assert_eq!(press(&mut dash, KeyCode::Char('o')), Action::Open(beside), "from the result's folder");
        for _ in 0..5 {
            press(&mut dash, KeyCode::Up);
        }
        assert_eq!(press(&mut dash, KeyCode::Char('o')), Action::Open(path("tasks/t01/result.md")));
        assert_eq!(dash.selected_line, None, "↑↓ move in the result, not the transcript");
        press(&mut dash, KeyCode::Esc);
        let s = screen(&dash);
        assert!(!s.contains("t01 RESULT") && s.contains("✓ t01 question 1? ready"), "back: {s}");

        // A URL result, and one whose file is gone. Tab moves between the ready chips,
        // the newest focused first.
        let url = Target::Url("https://example.com/report".into());
        dash.apply(CoreEvent::TaskUpdated(with_result(task(2, TaskStatus::Done), "Report", url)));
        let gone = Target::File("tasks/t03/result.md".into());
        dash.apply(CoreEvent::TaskUpdated(with_result(task(3, TaskStatus::Done), "Gone", gone)));
        assert!(screen(&dash).contains("Tab next result   Enter show t03"));
        press(&mut dash, KeyCode::Enter);
        let s = screen(&dash);
        assert!(s.contains("Can't read tasks/t03/result.md:"), "{s}");
        press(&mut dash, KeyCode::Esc);
        press(&mut dash, KeyCode::Tab);
        assert!(screen(&dash).contains("Enter show t01"));
        press(&mut dash, KeyCode::Tab);
        press(&mut dash, KeyCode::Enter);
        let s = screen(&dash);
        assert!(s.contains("→ https://example.com/report") && s.contains("o opens it in the browser"), "{s}");
        assert_eq!(press(&mut dash, KeyCode::Char('o')), Action::Open("https://example.com/report".into()));

        // A direct answer still shows at once, over the result.
        dash.apply(CoreEvent::TaskUpdated(task(4, TaskStatus::Running)));
        dash.apply(CoreEvent::StageMessage {
            task: Some(TaskId(4)),
            text: "Forty a seat.".into(),
            at: 16_000 * 190,
        });
        let s = screen(&dash);
        assert!(s.contains("ANSWERING  typed") && s.contains("Forty a seat."), "{s}");
        // With nothing ready to show, Enter and o say so.
        let mut dash = Dashboard::new("Pricing".into(), "11 Oct 2026".into());
        press(&mut dash, KeyCode::Enter);
        assert!(screen(&dash).contains("No result to show: Tab to a ready chip"));
        assert_eq!(press(&mut dash, KeyCode::Char('o')), Action::None);
    }

    #[test]
    fn a_long_result_scrolls_to_keep_the_selected_line_in_sight() {
        let dir = tempfile::tempdir().unwrap();
        std::fs::create_dir_all(dir.path().join("tasks/t01")).unwrap();
        let text: String = (1..=100).map(|i| format!("line {i}\n")).collect();
        std::fs::write(dir.path().join("tasks/t01/result.txt"), text).unwrap();
        let mut dash = Dashboard::new("Pricing".into(), "11 Oct 2026".into());
        dash.set_folder(dir.path().to_path_buf());
        let file = Target::File("tasks/t01/result.txt".into());
        dash.apply(CoreEvent::TaskUpdated(with_result(task(1, TaskStatus::Done), "Notes", file)));
        press(&mut dash, KeyCode::Enter);
        let s = screen(&dash);
        assert!(s.contains("line 1 ") && !s.contains("line 60 "), "{s}");
        for _ in 0..60 {
            press(&mut dash, KeyCode::Down);
        }
        let s = screen(&dash);
        assert!(s.contains("line 60 ") && !s.contains("line 1 "), "{s}");
        for _ in 0..200 {
            press(&mut dash, KeyCode::Down);
        }
        assert!(screen(&dash).contains("line 100 "), "stops at the last line");
        // A file that isn't Markdown or text shows only its path.
        let pdf = Target::File("tasks/t01/slides.pdf".into());
        dash.apply(CoreEvent::TaskUpdated(with_result(task(2, TaskStatus::Done), "Slides", pdf)));
        press(&mut dash, KeyCode::Esc);
        press(&mut dash, KeyCode::Enter);
        let s = screen(&dash);
        assert!(s.contains("→ tasks/t01/slides.pdf") && s.contains("o opens it"), "{s}");
    }
}
