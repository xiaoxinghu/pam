//! A meetig folder read back: what the agent commands show.
//!
//! Lines get their speakers through `speakers.json` each time the folder is
//! read, so renames, merges and moved lines show up without the transcript
//! changing.

use std::collections::BTreeSet;
use std::fmt;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};
use std::str::FromStr;

use anyhow::{Context, Result, bail};
use chrono::{DateTime, FixedOffset, Local};
use serde::{Deserialize, Serialize};

use crate::Ts;
use crate::clock::{SECOND, mmss};
pub use crate::event::TaskId;
use crate::event::{EventId, Line, SpkId, UndoId, UtteranceId};
use crate::folder::MeetingInfo;
use crate::identity::Speakers;
use crate::question;
use crate::task::{self, Message, TaskState, Via};

/// The event types of `events.jsonl`.
pub const EVENT_TYPES: [&str; 12] = [
    "topic",
    "point",
    "proposal",
    "concern",
    "question",
    "answer",
    "decision",
    "action",
    "request",
    "task_update",
    "note",
    "intro",
];

/// One line of `events.jsonl` (L1).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Event {
    pub id: EventId,
    #[serde(rename = "type")]
    pub kind: String,
    pub t: Ts,
    /// As written; [`MeetingRecord::event_speakers`] applies merges.
    #[serde(default)]
    pub by: Vec<SpkId>,
    pub text: String,
    /// Line IDs or ranges, e.g. `u0405..u0410`.
    #[serde(default)]
    pub refs: Vec<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub task: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supersedes: Option<EventId>,
    /// `intro` only: the name the speaker gave.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    /// `intro` only: the rename in `speakers.json` it made or confirmed.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rename: Option<UndoId>,
}

/// `summary.json`: the scribe's latest summary of the meeting.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    /// The last line the scribe had read when it wrote this.
    pub through: Option<UtteranceId>,
    pub text: String,
}

/// Anything a command can point at.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Ref {
    Line(UtteranceId),
    /// Both ends included.
    Lines(UtteranceId, UtteranceId),
    Event(EventId),
    Task(TaskId),
    Speaker(SpkId),
}

impl fmt::Display for Ref {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Ref::Line(u) => write!(f, "{u}"),
            Ref::Lines(a, b) => write!(f, "{a}..{b}"),
            Ref::Event(e) => write!(f, "{e}"),
            Ref::Task(t) => write!(f, "{t}"),
            Ref::Speaker(s) => write!(f, "{s}"),
        }
    }
}

impl FromStr for Ref {
    type Err = String;
    fn from_str(s: &str) -> Result<Ref, String> {
        if let Some((a, b)) = s.split_once("..") {
            let (a, b): (UtteranceId, UtteranceId) = (a.parse()?, b.parse()?);
            if a > b {
                return Err(format!("`{s}` runs backwards: put the earlier line first, e.g. {b}..{a}"));
            }
            return Ok(Ref::Lines(a, b));
        }
        if let Ok(u) = s.parse() {
            return Ok(Ref::Line(u));
        }
        if let Ok(e) = s.parse() {
            return Ok(Ref::Event(e));
        }
        if let Ok(spk) = s.parse() {
            return Ok(Ref::Speaker(spk));
        }
        if s.strip_prefix('t').is_some_and(|n| !n.is_empty() && n.bytes().all(|b| b.is_ascii_digit())) {
            return s.parse().map(Ref::Task);
        }
        Err(format!(
            "`{s}` is not an ID: use a line (u0412), line range (u0400..u0412), event (e037), task (t02) or speaker (spk_3)"
        ))
    }
}

/// A speaker as `pam people` shows them.
#[derive(Debug, Clone, PartialEq, Serialize)]
pub struct Person {
    pub spk: SpkId,
    pub name: Option<String>,
    /// Start of their first line, or `None` before they have one.
    pub first_heard: Option<Ts>,
    /// The length of all their lines added up.
    pub talk_time: Ts,
    pub lines: usize,
}

/// Which part of the transcript to show.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Window {
    All,
    /// The last N lines.
    Last(usize),
    /// Lines that reach into the last stretch of this length.
    Since(Ts),
    /// Lines that overlap `from..to`; a missing end is open.
    Between {
        from: Option<Ts>,
        to: Option<Ts>,
    },
}

/// What a ref points at, found.
#[derive(Debug, Clone)]
pub enum Item<'a> {
    Line(&'a Line),
    Event(&'a Event),
    Speaker(Person),
    Task(&'a TaskState),
}

/// The stretch `pam around` shows context around.
#[derive(Debug, Clone)]
pub struct Anchor<'a> {
    pub t0: Ts,
    pub t1: Ts,
    /// The lines the ref names: the line, the range, or the event's refs.
    pub lines: Vec<UtteranceId>,
    pub event: Option<&'a Event>,
}

/// A keyword search.
#[derive(Debug, Clone, Default)]
pub struct Query {
    /// Every word must appear, ignoring case. A word may be part of a longer one.
    pub words: Vec<String>,
    /// Only lines and events by these speakers.
    pub speakers: Option<Vec<SpkId>>,
    /// Only these event types; `line` stands for transcript lines.
    pub types: Option<Vec<String>>,
}

impl Query {
    /// Splits `text` into lowercase words, dropping punctuation around them.
    pub fn words(text: &str) -> Vec<String> {
        text.split_whitespace()
            .map(|w| fold(w.trim_matches(|c: char| !c.is_alphanumeric())))
            .filter(|w| !w.is_empty())
            .collect()
    }

    fn matches(&self, text: &str) -> bool {
        let text = fold(text);
        self.words.iter().all(|w| text.contains(w.as_str()))
    }

    fn wants(&self, kind: &str) -> bool {
        self.types.as_ref().is_none_or(|t| t.iter().any(|t| t == kind))
    }

    fn by(&self, spks: &[SpkId]) -> bool {
        self.speakers.as_ref().is_none_or(|want| spks.iter().any(|s| want.contains(s)))
    }
}

#[derive(Debug, Clone, Copy)]
pub enum Hit<'a> {
    Line(&'a Line),
    Event(&'a Event),
}

/// Why a ref found nothing; the message says what does exist.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct NotFound(pub String);

impl fmt::Display for NotFound {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.0)
    }
}

/// Which meeting to read, in the spec's order: the `--meeting` flag, then
/// `$PAM_MEETING`, then the most recent meeting under the meetings root,
/// which `root` finds only when it comes to that. The inner error says why
/// there is none and how to get one; the outer one is a folder or file that
/// could not be read.
pub fn choose_meeting(
    flag: Option<&Path>,
    env: Option<&Path>,
    root: impl FnOnce() -> Result<PathBuf>,
) -> Result<Result<PathBuf, String>> {
    if let Some(dir) = flag {
        return Ok(read_info(dir)?.map(|_| dir.to_path_buf()).ok_or_else(|| {
            format!("--meeting {} is not a meeting folder: it has no meeting.json", dir.display())
        }));
    }
    if let Some(dir) = env.filter(|d| !d.as_os_str().is_empty()) {
        return Ok(read_info(dir)?.map(|_| dir.to_path_buf()).ok_or_else(|| {
            format!(
                "$PAM_MEETING is {}, which is not a meeting folder: it has no meeting.json",
                dir.display()
            )
        }));
    }
    let root = root()?;
    Ok(latest_meeting(&root)?.ok_or_else(|| {
        format!(
            "no meetings under {} yet: start one with `pam listen`, or pass --meeting <folder>",
            root.display()
        )
    }))
}

/// The meeting under `root` that started last. Folder names break ties.
fn latest_meeting(root: &Path) -> Result<Option<PathBuf>> {
    let mut latest = None;
    for dir in list_if_present(root)? {
        if let Some(info) = read_info(&dir)? {
            latest = latest.max(Some((info.started, dir)));
        }
    }
    Ok(latest.map(|(_, d)| d))
}

/// `meeting.json`, checked: the only place it is read, so a broken one fails
/// the same way however the meeting was chosen.
struct CheckedInfo {
    info: MeetingInfo,
    started: DateTime<FixedOffset>,
    stopped: Option<DateTime<FixedOffset>>,
}

/// `dir/meeting.json`, or `None` if `dir` is not a meeting folder.
fn read_info(dir: &Path) -> Result<Option<CheckedInfo>> {
    let path = dir.join("meeting.json");
    let Some(bytes) = read_if_present(&path)? else { return Ok(None) };
    let info: MeetingInfo = parse_json(&path, &bytes)?;
    let time = |field: &str, t: &str| {
        DateTime::parse_from_rfc3339(t)
            .with_context(|| format!("reading {}: {field} `{t}` is not an RFC 3339 time", path.display()))
    };
    let started = time("started_at", &info.started_at)?;
    let stopped = info.stopped_at.as_deref().map(|t| time("stopped_at", t)).transpose()?;
    if let Some(stop) = stopped.filter(|&stop| stop < started) {
        bail!("reading {}: stopped_at {stop} is before started_at {started}", path.display());
    }
    if !matches!(info.status.as_str(), "recording" | "stopped") {
        bail!("reading {}: status `{}` is neither recording nor stopped", path.display(), info.status);
    }
    Ok(Some(CheckedInfo { info, started, stopped }))
}

/// A meeting folder, read once.
#[derive(Debug, Clone)]
pub struct MeetingRecord {
    pub dir: PathBuf,
    pub info: MeetingInfo,
    /// `info.started_at` and `info.stopped_at`, parsed.
    pub started: DateTime<FixedOffset>,
    pub stopped: Option<DateTime<FixedOffset>>,
    pub speakers: Speakers,
    /// Final lines in ID order, speakers resolved.
    pub lines: Vec<Line>,
    /// `events.jsonl`, empty until the scribe writes it.
    pub events: Vec<Event>,
    /// `summary.json`, empty until the scribe writes it.
    pub summary: Summary,
    /// Every `tasks/<id>/task.json`, in ID order. A task folder without
    /// one yet (it is being made) is left out.
    pub tasks: Vec<TaskState>,
}

/// One row of `transcript.jsonl`.
#[derive(Deserialize)]
struct SavedLine {
    id: UtteranceId,
    t0: Ts,
    t1: Ts,
    slot: Option<u32>,
    #[serde(default)]
    lang: Option<String>,
    text: String,
}

impl MeetingRecord {
    pub fn load(dir: &Path) -> Result<MeetingRecord> {
        let CheckedInfo { info, started, stopped } = read_info(dir)?
            .with_context(|| format!("{} is not a meeting folder: it has no meeting.json", dir.display()))?;
        let path = dir.join("speakers.json");
        let speakers: Speakers =
            read_if_present(&path)?.map(|bytes| parse_json(&path, &bytes)).transpose()?.unwrap_or_default();
        let lines = read_jsonl::<SavedLine>(&dir.join("transcript.jsonl"))?
            .into_iter()
            .map(|l| Line {
                spk: speakers.resolve(l.id, l.slot),
                id: l.id,
                t0: l.t0,
                t1: l.t1,
                slot: l.slot,
                lang: l.lang,
                text: l.text,
            })
            .collect();
        let events = read_jsonl::<Event>(&dir.join("events.jsonl"))?;
        let summary = read_summary(dir)?;
        let mut tasks = Vec::new();
        for id in task::ids(dir)? {
            let path = task::dir(dir, id).join("task.json");
            if let Some(bytes) = read_if_present(&path)? {
                tasks.push(parse_json(&path, &bytes)?);
            }
        }
        Ok(MeetingRecord {
            dir: dir.to_path_buf(),
            info,
            started,
            stopped,
            speakers,
            lines,
            events,
            summary,
            tasks,
        })
    }

    pub fn name(&self, spk: SpkId) -> Option<&str> {
        self.speakers.overlay.name(spk)
    }

    /// How far into the meeting it is: the audio heard once stopped, otherwise
    /// the time from its start to now, or to `stopped_at` if it stopped without
    /// a duration (a failed meeting). Never before the end of the last line.
    pub fn now(&self) -> Ts {
        let last = self.lines.last().map_or(0, |l| l.t1);
        if let Some(d) = self.info.duration {
            return last.max(d);
        }
        let end = match self.stopped {
            Some(t) => Some(t),
            None if self.info.status == "stopped" => None,
            None => Some(Local::now().fixed_offset()),
        };
        let wall = end
            .and_then(|end| (end - self.started).to_std().ok())
            .map_or(0, |d| (d.as_secs_f64() * SECOND as f64) as Ts);
        last.max(wall)
    }

    /// Everyone with an ID, merged speakers folded in, in ID order.
    pub fn people(&self) -> Vec<Person> {
        let mut ids: BTreeSet<SpkId> = self
            .speakers
            .slots
            .values()
            .chain(self.speakers.overlay.moves.values())
            .map(|&s| self.speakers.overlay.follow(s))
            .collect();
        ids.extend(self.lines.iter().filter_map(|l| l.spk));
        ids.into_iter().map(|spk| self.person(spk)).collect()
    }

    fn person(&self, spk: SpkId) -> Person {
        let theirs = self.lines.iter().filter(|l| l.spk == Some(spk));
        let (mut first_heard, mut talk_time, mut lines) = (None, 0, 0);
        for l in theirs {
            first_heard.get_or_insert(l.t0);
            talk_time += l.t1.saturating_sub(l.t0);
            lines += 1;
        }
        Person { spk, name: self.name(spk).map(str::to_string), first_heard, talk_time, lines }
    }

    /// The speakers `who` means: an ID (followed through merges) or a name,
    /// ignoring case. Empty when nobody matches.
    pub fn speakers_matching(&self, who: &str) -> Vec<SpkId> {
        let people = self.people();
        if let Ok(spk) = who.parse::<SpkId>() {
            let spk = self.speakers.overlay.follow(spk);
            return people.iter().filter(|p| p.spk == spk).map(|p| p.spk).collect();
        }
        people
            .iter()
            .filter(|p| p.name.as_deref().is_some_and(|n| fold(n) == fold(who.trim())))
            .map(|p| p.spk)
            .collect()
    }

    /// Lines in `window`, only by `speakers` if given.
    pub fn transcript(&self, window: Window, speakers: Option<&[SpkId]>) -> Vec<&Line> {
        let by = |l: &&Line| speakers.is_none_or(|s| l.spk.is_some_and(|spk| s.contains(&spk)));
        let lines = self.lines.iter().filter(by);
        match window {
            Window::All => lines.collect(),
            Window::Last(n) => {
                let all: Vec<&Line> = lines.collect();
                all[all.len().saturating_sub(n)..].to_vec()
            }
            Window::Since(d) => {
                let from = self.now().saturating_sub(d);
                lines.filter(|l| l.t1 > from).collect()
            }
            Window::Between { from, to } => {
                lines.filter(|l| from.is_none_or(|f| l.t1 > f) && to.is_none_or(|t| l.t0 < t)).collect()
            }
        }
    }

    pub fn line(&self, id: UtteranceId) -> Option<&Line> {
        self.lines.binary_search_by_key(&id, |l| l.id).ok().map(|i| &self.lines[i])
    }

    /// Lines `a..b`, both ends included.
    fn line_range(&self, a: UtteranceId, b: UtteranceId) -> &[Line] {
        let start = self.lines.partition_point(|l| l.id < a);
        let end = self.lines.partition_point(|l| l.id <= b);
        &self.lines[start..end.max(start)]
    }

    pub fn event(&self, id: EventId) -> Option<&Event> {
        self.events.iter().find(|e| e.id == id)
    }

    /// Who an event is by, with merges applied.
    pub fn event_speakers(&self, e: &Event) -> Vec<SpkId> {
        let mut by: Vec<SpkId> = Vec::new();
        for spk in e.by.iter().map(|&s| self.speakers.overlay.follow(s)) {
            if !by.contains(&spk) {
                by.push(spk);
            }
        }
        by
    }

    /// Open items: questions and actions that no later event supersedes.
    pub fn open_items(&self) -> Vec<&Event> {
        self.events
            .iter()
            .filter(|e| {
                matches!(e.kind.as_str(), "question" | "action") && self.superseded_by(e.id).is_none()
            })
            .collect()
    }

    /// The event that supersedes `id`, if any.
    pub fn superseded_by(&self, id: EventId) -> Option<EventId> {
        self.events.iter().rev().find(|e| e.supersedes == Some(id)).map(|e| e.id)
    }

    /// A line as the commands and `context.md` show it: `u0412 03:10 Alice: text`.
    pub fn line_text(&self, l: &Line) -> String {
        format!("{} {} {}: {}", l.id, mmss(l.t0), self.speakers.overlay.label(l.spk), l.text)
    }

    /// An event as the commands and `context.md` show it:
    /// `e037 04:03 request Jordan: text [u0412] [task t02] [superseded by e040]`.
    pub fn event_text(&self, e: &Event) -> String {
        let mut out = format!("{} {} {}", e.id, mmss(e.t), e.kind);
        let by: Vec<String> =
            self.event_speakers(e).into_iter().map(|s| self.speakers.overlay.label(Some(s))).collect();
        if !by.is_empty() {
            out += &format!(" {}", by.join(", "));
        }
        out += &format!(": {}", e.text);
        if !e.refs.is_empty() {
            out += &format!(" [{}]", e.refs.join(", "));
        }
        if let Some(t) = &e.task {
            out += &format!(" [task {t}]");
        }
        if let Some(s) = e.supersedes {
            out += &format!(" [supersedes {s}]");
        }
        if let Some(s) = self.superseded_by(e.id) {
            out += &format!(" [superseded by {s}]");
        }
        out
    }

    /// What `r` points at. A range gives every line in it that exists.
    pub fn get(&self, r: Ref) -> Result<Vec<Item<'_>>, NotFound> {
        match r {
            Ref::Line(id) => self.line(id).map(|l| vec![Item::Line(l)]).ok_or_else(|| self.no_line(&r)),
            Ref::Lines(a, b) => match self.line_range(a, b) {
                [] => Err(self.no_line(&r)),
                lines => Ok(lines.iter().map(Item::Line).collect()),
            },
            Ref::Event(id) => self.event(id).map(|e| vec![Item::Event(e)]).ok_or_else(|| self.no_event(id)),
            Ref::Speaker(spk) => match &self.speakers_matching(&spk.to_string())[..] {
                [] => Err(self.no_speaker(&spk.to_string())),
                found => Ok(found.iter().map(|&s| Item::Speaker(self.person(s))).collect()),
            },
            Ref::Task(id) => self.task(id).map(|t| vec![Item::Task(t)]).ok_or_else(|| self.no_task(id)),
        }
    }

    /// The stretch of meeting `r` covers. Only lines, ranges and events have one.
    pub fn anchor(&self, r: Ref) -> Result<Anchor<'_>, NotFound> {
        let (lines, event): (Vec<&Line>, _) = match r {
            Ref::Line(id) => (vec![self.line(id).ok_or_else(|| self.no_line(&r))?], None),
            Ref::Lines(a, b) => {
                let lines: Vec<&Line> = self.line_range(a, b).iter().collect();
                if lines.is_empty() {
                    return Err(self.no_line(&r));
                }
                (lines, None)
            }
            Ref::Event(id) => {
                let e = self.event(id).ok_or_else(|| self.no_event(id))?;
                (self.event_lines(e), Some(e))
            }
            Ref::Task(_) | Ref::Speaker(_) => {
                return Err(NotFound(format!(
                    "{r} is not a line or event, so it has no place in the transcript"
                )));
            }
        };
        let t0 = lines.iter().map(|l| l.t0).chain(event.map(|e| e.t)).min().unwrap_or(0);
        let t1 = lines.iter().map(|l| l.t1).chain(event.map(|e| e.t)).max().unwrap_or(0);
        Ok(Anchor { t0, t1, lines: lines.iter().map(|l| l.id).collect(), event })
    }

    /// The lines an event's refs name.
    fn event_lines(&self, e: &Event) -> Vec<&Line> {
        let mut out: Vec<&Line> = Vec::new();
        for r in &e.refs {
            match r.parse() {
                Ok(Ref::Line(id)) => out.extend(self.line(id)),
                Ok(Ref::Lines(a, b)) => out.extend(self.line_range(a, b)),
                _ => {}
            }
        }
        out.sort_by_key(|l| l.id);
        out.dedup_by_key(|l| l.id);
        out
    }

    /// The anchor's lines and every line that overlaps the stretch from
    /// `before` ahead of it to `after` past it.
    pub fn around(&self, anchor: &Anchor, before: Ts, after: Ts) -> Vec<&Line> {
        let (from, to) = (anchor.t0.saturating_sub(before), anchor.t1.saturating_add(after));
        self.lines.iter().filter(|l| anchor.lines.contains(&l.id) || (l.t1 > from && l.t0 < to)).collect()
    }

    /// Lines and events that match, oldest first.
    pub fn search(&self, q: &Query) -> Vec<Hit<'_>> {
        let mut hits: Vec<(Ts, Hit)> = Vec::new();
        if q.wants("line") {
            hits.extend(
                self.lines
                    .iter()
                    .filter(|l| q.by(&l.spk.into_iter().collect::<Vec<_>>()) && q.matches(&l.text))
                    .map(|l| (l.t0, Hit::Line(l))),
            );
        }
        for e in &self.events {
            if q.wants(&e.kind) && q.by(&self.event_speakers(e)) && q.matches(&e.text) {
                hits.push((e.t, Hit::Event(e)));
            }
        }
        // Stable: at the same time, lines come before the events about them.
        hits.sort_by_key(|(t, _)| *t);
        hits.into_iter().map(|(_, h)| h).collect()
    }

    pub fn task(&self, id: TaskId) -> Option<&TaskState> {
        self.tasks.iter().find(|t| t.id == id)
    }

    /// Who asked for a task: its line's speaker, else the speaker the
    /// tracker had when it was asked, by name once known; `typed` for the composer.
    pub fn asker(&self, t: &TaskState) -> String {
        if t.by == Via::Typed {
            return "typed".into();
        }
        let spk = match t.line.and_then(|id| self.line(id)) {
            Some(l) => l.spk,
            None => t.spk.map(|s| self.speakers.overlay.follow(s)),
        };
        self.speakers.overlay.label(spk)
    }

    /// A task as `pam status` and `context.md` show it, like a line:
    /// `t02 04:10 Jordan: mock up both flows [u0412] · running 60% · reading sources`,
    /// with `ready (tasks/t02/result.md)` once it has a result.
    pub fn task_text(&self, t: &TaskState) -> String {
        let mut out = format!("{} {} {}: {}", t.id, mmss(t.asked_at), self.asker(t), t.request);
        if let Some(line) = t.line {
            out += &format!(" [{line}]");
        }
        out += &format!(" · {}", t.shown_status());
        // Progress and the current action say how a task is going, until it ends.
        let open = t.status.is_open();
        if let Some(p) = t.progress.filter(|_| open) {
            out += &format!(" {p}%");
        }
        if let Some(r) = &t.result {
            out += &format!(" ({})", r.target.as_str());
        }
        if let Some(a) = t.action.as_ref().filter(|_| open) {
            out += &format!(" · {a}");
        }
        out
    }

    /// What the agent showed on the stage for task `id` (`messages.jsonl`).
    pub fn task_messages(&self, id: TaskId) -> Result<Vec<Message>> {
        read_jsonl(&task::messages_file(&self.dir, id))
    }

    /// What the agent asked the room for task `id`, and how each question
    /// ended (`questions.jsonl`), as written.
    pub fn task_questions(&self, id: TaskId) -> Result<Vec<question::Record>> {
        read_jsonl(&task::questions_file(&self.dir, id))
    }

    /// A wall time from a task's files (RFC 3339) as meeting time, if it parses.
    pub fn meeting_time(&self, at: &str) -> Option<Ts> {
        let at = DateTime::parse_from_rfc3339(at).ok()?;
        let d = (at - self.started).to_std().ok()?;
        Some((d.as_secs_f64() * SECOND as f64) as Ts)
    }

    fn no_task(&self, id: TaskId) -> NotFound {
        NotFound(match &self.tasks[..] {
            [] => format!("no task {id}: the meeting has no tasks yet"),
            tasks => format!(
                "no task {id}: tasks are {}",
                tasks.iter().map(|t| t.id.to_string()).collect::<Vec<_>>().join(", ")
            ),
        })
    }

    fn no_line(&self, r: &Ref) -> NotFound {
        NotFound(match (self.lines.first(), self.lines.last()) {
            (Some(a), Some(b)) => format!("no line {r}: lines run {}..{}", a.id, b.id),
            _ => format!("no line {r}: the meeting has no lines yet"),
        })
    }

    fn no_event(&self, id: EventId) -> NotFound {
        NotFound(match (self.events.first(), self.events.last()) {
            (Some(a), Some(b)) => format!("no event {id}: events run {}..{}", a.id, b.id),
            _ => format!("no event {id}: the meeting has no events yet"),
        })
    }

    /// For a speaker filter or ref that matched nobody.
    pub fn no_speaker(&self, who: &str) -> NotFound {
        let people: Vec<String> = self
            .people()
            .iter()
            .map(|p| match &p.name {
                Some(n) => format!("{} ({n})", p.spk),
                None => p.spk.to_string(),
            })
            .collect();
        NotFound(if people.is_empty() {
            format!("no speaker `{who}`: nobody has spoken yet")
        } else {
            format!("no speaker `{who}`: speakers are {}", people.join(", "))
        })
    }
}

/// `dir/summary.json`, or an empty summary if the scribe hasn't written one.
fn read_summary(dir: &Path) -> Result<Summary> {
    let path = dir.join("summary.json");
    Ok(read_if_present(&path)?.map(|bytes| parse_json(&path, &bytes)).transpose()?.unwrap_or_default())
}

/// `s` with case removed, for "ignoring case" comparisons. Uppercasing first
/// approximates Unicode case folding with std alone: `ΟΣ`, `οσ` and `ος` all
/// become `ος`, and `Straße` and `STRASSE` both become `strasse`. It is not
/// full case folding (dotless `ı` folds with `i`); the default transcriber
/// is English, so a case-folding crate isn't worth adding for that.
fn fold(s: &str) -> String {
    s.to_uppercase().to_lowercase()
}

/// Reads a JSON Lines file. A missing file is empty, and a last line without
/// its newline is skipped: the meeting may still be writing it.
pub(crate) fn read_jsonl<T: for<'de> Deserialize<'de>>(path: &Path) -> Result<Vec<T>> {
    let Some(bytes) = read_if_present(path)? else { return Ok(Vec::new()) };
    let text = String::from_utf8(bytes).with_context(|| format!("reading {}", path.display()))?;
    let complete = match text.rfind('\n') {
        Some(i) => &text[..i],
        None => "",
    };
    complete
        .lines()
        .enumerate()
        .filter(|(_, l)| !l.trim().is_empty())
        .map(|(i, l)| {
            serde_json::from_str(l).with_context(|| format!("reading {}, line {}", path.display(), i + 1))
        })
        .collect()
}

// What "not there" means, decided once for every file and folder this
// module reads. Only a missing entry is "not there". Anything else that stops
// a read, including an entry of the wrong kind, is an error naming the path,
// which the commands report with exit 4. Symlinks are followed.
//
// One case reads a file inside something that may not be a folder: a
// candidate meeting under the root can be a plain file, and `--meeting` can
// name one. There, a file where the folder should be means that path isn't
// a meeting folder, so `read_if_present` counts it as "not there" too.

/// The file's bytes, or `None` if it isn't there.
pub(crate) fn read_if_present(path: &Path) -> Result<Option<Vec<u8>>> {
    match fs::metadata(path) {
        Err(e) if matches!(e.kind(), ErrorKind::NotFound | ErrorKind::NotADirectory) => return Ok(None),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        Ok(m) if !m.is_file() => bail!("reading {}: it should be a file", path.display()),
        Ok(_) => {}
    }
    fs::read(path).map(Some).with_context(|| format!("reading {}", path.display()))
}

/// The paths in a folder, or none if it isn't there.
fn list_if_present(dir: &Path) -> Result<Vec<PathBuf>> {
    match fs::metadata(dir) {
        Err(e) if e.kind() == ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e).with_context(|| format!("reading {}", dir.display())),
        Ok(m) if !m.is_dir() => bail!("reading {}: it should be a folder", dir.display()),
        Ok(_) => {}
    }
    let entries = fs::read_dir(dir).with_context(|| format!("reading {}", dir.display()))?;
    entries.map(|e| Ok(e.with_context(|| format!("reading {}", dir.display()))?.path())).collect()
}

pub(crate) fn parse_json<T: for<'de> Deserialize<'de>>(path: &Path, bytes: &[u8]) -> Result<T> {
    serde_json::from_slice(bytes).with_context(|| format!("reading {}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::*;

    const S: Ts = SECOND;

    /// A stopped 60 s meeting: four slots, spk_3 merged into spk_1, u0004
    /// moved to spk_2, and Alice and Marcus named.
    fn meeting() -> (tempfile::TempDir, MeetingRecord) {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        fs::write(
            p.join("meeting.json"),
            r#"{"title":"Sync","started_at":"2026-10-10T14:02:00+11:00","status":"stopped","duration":960000}"#,
        )
        .unwrap();
        let lines = [
            (1, 0, 5, Some(0), "Hi, I'm Alice and I run product."),
            (2, 6, 10, Some(1), "Marcus here, backend."),
            (3, 11, 20, Some(3), "The signup flow loses users on mobile."),
            (4, 21, 25, Some(0), "Should we prototype both flows?"),
            (5, 26, 30, None, "Mm."),
            (6, 31, 40, Some(1), "Let's keep the team-invite step."),
        ];
        let mut text = String::new();
        for (id, t0, t1, slot, words) in lines {
            let slot = slot.map_or("null".into(), |s: u32| s.to_string());
            text += &format!(
                "{{\"id\":\"u{id:04}\",\"t0\":{},\"t1\":{},\"slot\":{slot},\"text\":\"{words}\"}}\n",
                t0 * S,
                t1 * S
            );
        }
        text += r#"{"id":"u0007","t0":"#; // being written
        fs::write(p.join("transcript.jsonl"), text).unwrap();
        fs::write(
            p.join("speakers.json"),
            r#"{"slots":{"0":"spk_0","1":"spk_1","2":"spk_2","3":"spk_3"},
                "names":{"spk_0":"Alice","spk_1":"Marcus"},
                "merges":{"spk_3":"spk_1"},
                "moves":{"u0004":"spk_2"},
                "log":[]}"#,
        )
        .unwrap();
        fs::write(
            p.join("events.jsonl"),
            format!(
                "{}\n{}\n{}\n",
                format_args!(
                    r#"{{"id":"e001","type":"concern","t":{},"by":["spk_3"],"text":"Signup may lose mobile users","refs":["u0003"]}}"#,
                    11 * S
                ),
                format_args!(
                    r#"{{"id":"e002","type":"decision","t":{},"by":["spk_1"],"text":"Drop the team-invite step","refs":["u0006"]}}"#,
                    31 * S
                ),
                format_args!(
                    r#"{{"id":"e003","type":"decision","t":{},"by":["spk_1"],"text":"Keep the team-invite step","refs":["u0006"],"supersedes":"e002"}}"#,
                    31 * S
                ),
            ),
        )
        .unwrap();
        fs::write(p.join("summary.json"), r#"{"through":"u0006","text":"Choosing a signup flow."}"#).unwrap();
        let rec = MeetingRecord::load(p).unwrap();
        assert_eq!(rec.summary.through, Some(UtteranceId(6)));
        (dir, rec)
    }

    fn ids(lines: &[&Line]) -> Vec<u32> {
        lines.iter().map(|l| l.id.0).collect()
    }

    #[test]
    fn resolves_speakers_in_the_spec_order() {
        let (_d, m) = meeting();
        let spk: Vec<Option<u32>> = m.lines.iter().map(|l| l.spk.map(|s| s.0)).collect();
        // u0003 is slot 3 → spk_3 → merged into spk_1; u0004 is moved to spk_2.
        assert_eq!(spk, [Some(0), Some(1), Some(1), Some(2), None, Some(1)]);
        assert_eq!(m.lines.len(), 6, "the half-written last line is skipped");
    }

    #[test]
    fn merges_are_followed_through_chains_and_cycles() {
        let s: Speakers = serde_json::from_str(r#"{"merges":{"spk_3":"spk_2","spk_2":"spk_1"}}"#).unwrap();
        assert_eq!(s.overlay.follow(SpkId(3)), SpkId(1));
        let s: Speakers = serde_json::from_str(r#"{"merges":{"spk_1":"spk_2","spk_2":"spk_1"}}"#).unwrap();
        s.overlay.follow(SpkId(1)); // returns
    }

    #[test]
    fn people_fold_in_merged_speakers() {
        let (_d, m) = meeting();
        let people = m.people();
        assert_eq!(people.iter().map(|p| p.spk.0).collect::<Vec<_>>(), [0, 1, 2]);
        let marcus = &people[1];
        assert_eq!(marcus.name.as_deref(), Some("Marcus"));
        assert_eq!((marcus.first_heard, marcus.talk_time, marcus.lines), (Some(6 * S), 22 * S, 3));
        assert_eq!(people[2].name, None);
    }

    #[test]
    fn finds_speakers_by_id_or_name() {
        let (_d, m) = meeting();
        assert_eq!(m.speakers_matching("marcus"), [SpkId(1)]);
        assert_eq!(m.speakers_matching("spk_3"), [SpkId(1)]);
        assert!(m.speakers_matching("spk_9").is_empty());
        assert!(m.speakers_matching("Sam").is_empty());
        assert!(m.no_speaker("Sam").0.contains("spk_0 (Alice), spk_1 (Marcus), spk_2"));
        let mut m = m;
        m.speakers.overlay.names.insert(SpkId(2), "Élodie".into());
        assert_eq!(m.speakers_matching("élodie"), [SpkId(2)], "any case, beyond ASCII");
        assert_eq!(m.speakers_matching("ÉLODIE"), [SpkId(2)]);
        m.speakers.overlay.names.insert(SpkId(0), "ΟΣ".into());
        assert_eq!(m.speakers_matching("οσ"), [SpkId(0)], "case folding, not just lowercasing");
        assert_eq!(m.speakers_matching("ος"), [SpkId(0)]);
    }

    #[test]
    fn a_meeting_that_stopped_without_a_duration_stops_its_clock() {
        let (_d, mut m) = meeting();
        m.info.duration = Some(30 * S); // shorter than the speech: the last line wins
        assert_eq!(m.now(), 40 * S);
        m.info.duration = None;
        m.stopped = Some(DateTime::parse_from_rfc3339("2026-10-10T14:03:30+11:00").unwrap()); // 90 s after the start
        assert_eq!(m.now(), 90 * S);
        m.stopped = None; // and no stop time either: the last line is all there is
        assert_eq!(m.now(), 40 * S);
        m.info.status = "recording".into();
        assert!(m.now() > 3600 * S, "a recording meeting runs on the wall clock");
    }

    #[test]
    fn open_items_are_questions_and_actions_nothing_supersedes() {
        let (_d, mut m) = meeting();
        assert!(m.open_items().is_empty());
        let event = |id, kind: &str, supersedes| Event {
            id: EventId(id),
            kind: kind.into(),
            t: 0,
            by: Vec::new(),
            text: String::new(),
            refs: Vec::new(),
            task: None,
            supersedes,
            name: None,
            rename: None,
        };
        m.events.push(event(4, "question", None));
        m.events.push(event(5, "action", None));
        m.events.push(event(6, "action", Some(EventId(5))));
        let open: Vec<u32> = m.open_items().iter().map(|e| e.id.0).collect();
        assert_eq!(open, [4, 6]);
    }

    #[test]
    fn windows_the_transcript() {
        let (_d, m) = meeting();
        assert_eq!(ids(&m.transcript(Window::Last(2), None)), [5, 6]);
        assert_eq!(ids(&m.transcript(Window::Last(9), Some(&[SpkId(1)]))), [2, 3, 6]);
        // now is 60 s: the last 25 s reach lines ending after 35 s.
        assert_eq!(ids(&m.transcript(Window::Since(25 * S), None)), [6]);
        let between = Window::Between { from: Some(10 * S), to: Some(21 * S) };
        assert_eq!(ids(&m.transcript(between, None)), [3]);
        let from = Window::Between { from: Some(29 * S), to: None };
        assert_eq!(ids(&m.transcript(from, None)), [5, 6]);
    }

    #[test]
    fn parses_refs() {
        assert_eq!("u0412".parse(), Ok(Ref::Line(UtteranceId(412))));
        assert_eq!("u0400..u0412".parse(), Ok(Ref::Lines(UtteranceId(400), UtteranceId(412))));
        assert_eq!("e037".parse(), Ok(Ref::Event(EventId(37))));
        assert_eq!("t02".parse(), Ok(Ref::Task(TaskId(2))));
        assert_eq!("spk_3".parse(), Ok(Ref::Speaker(SpkId(3))));
        assert!("u0412..u0400".parse::<Ref>().unwrap_err().contains("u0400..u0412"));
        assert!("hello".parse::<Ref>().unwrap_err().contains("not an ID"));
        assert!("u".parse::<Ref>().is_err());
    }

    #[test]
    fn gets_what_refs_point_at() {
        let (_d, m) = meeting();
        let get = |r: &str| m.get(r.parse().unwrap());
        assert_eq!(get("u0002..u0004").unwrap().len(), 3);
        assert!(matches!(get("e001").unwrap()[..], [Item::Event(e)] if e.kind == "concern"));
        assert!(matches!(&get("spk_3").unwrap()[..], [Item::Speaker(p)] if p.spk == SpkId(1)));
        assert_eq!(get("u0099").unwrap_err().0, "no line u0099: lines run u0001..u0006");
        assert_eq!(get("u0090..u0099").unwrap_err().0, "no line u0090..u0099: lines run u0001..u0006");
        assert_eq!(get("e009").unwrap_err().0, "no event e009: events run e001..e003");
        assert_eq!(get("t02").unwrap_err().0, "no task t02: the meeting has no tasks yet");
        assert_eq!(m.superseded_by(EventId(2)), Some(EventId(3)));
        assert_eq!(m.event_speakers(m.event(EventId(1)).unwrap()), [SpkId(1)], "spk_3 is merged into spk_1");
    }

    /// meeting.json is checked when it's read, not where its fields are used.
    #[test]
    fn a_meeting_json_with_bad_fields_fails_to_load() {
        let cases = [
            (r#""started_at":"yesterday""#, "started_at `yesterday`"),
            (r#""started_at":"2026-10-10T14:02:00+11:00","stopped_at":"later""#, "stopped_at `later`"),
            (r#""started_at":"2026-10-10T14:02:00+11:00","status":"paused""#, "status `paused`"),
            (
                r#""started_at":"2026-10-10T14:02:00+11:00","stopped_at":"2026-10-10T14:01:00+11:00""#,
                "is before started_at",
            ),
        ];
        for (fields, says) in cases {
            let (d, _) = meeting();
            let mut json: serde_json::Value =
                serde_json::from_str(&fs::read_to_string(d.path().join("meeting.json")).unwrap()).unwrap();
            let patch: serde_json::Value = serde_json::from_str(&format!("{{{fields}}}")).unwrap();
            for (k, v) in patch.as_object().unwrap() {
                json[k] = v.clone();
            }
            fs::write(d.path().join("meeting.json"), json.to_string()).unwrap();
            let err = format!("{:#}", MeetingRecord::load(d.path()).unwrap_err());
            assert!(err.contains("meeting.json") && err.contains(says), "{err}");
            let err = format!("{:#}", choose_meeting(Some(d.path()), None, || unreachable!()).unwrap_err());
            assert!(err.contains(says), "an explicit choice checks it too: {err}");
        }
    }

    /// The one rule for every file this module reads: missing is "not there",
    /// anything else that stops the read fails and names the file.
    #[test]
    fn every_file_is_there_missing_or_broken() {
        for name in ["speakers.json", "transcript.jsonl", "events.jsonl", "summary.json"] {
            let (d, _) = meeting();
            fs::remove_file(d.path().join(name)).unwrap();
            assert!(MeetingRecord::load(d.path()).is_ok(), "a missing {name} is empty");
        }
        for name in ["meeting.json", "speakers.json", "transcript.jsonl", "events.jsonl", "summary.json"] {
            for broken in ["a folder", "not json"] {
                let (d, _) = meeting();
                let path = d.path().join(name);
                fs::remove_file(&path).unwrap();
                if broken == "a folder" {
                    fs::create_dir(&path).unwrap();
                } else {
                    fs::write(&path, "{not json\n").unwrap();
                }
                let err = format!("{:#}", MeetingRecord::load(d.path()).unwrap_err());
                assert!(err.contains(&format!("{}", path.display())), "{name} as {broken}: {err}");
            }
        }
    }

    #[test]
    fn shows_lines_around_a_line_or_event() {
        let (_d, m) = meeting();
        let a = m.anchor("u0003".parse().unwrap()).unwrap();
        assert_eq!(ids(&m.around(&a, 5 * S, 5 * S)), [2, 3, 4]);
        assert_eq!(ids(&m.around(&a, 0, 0)), [3]);
        assert_eq!(ids(&m.around(&a, Ts::MAX, Ts::MAX)), [1, 2, 3, 4, 5, 6]);
        let e = m.anchor("e003".parse().unwrap()).unwrap();
        assert_eq!((e.t0, e.t1, e.lines.clone()), (31 * S, 40 * S, vec![UtteranceId(6)]));
        assert_eq!(ids(&m.around(&e, 10 * S, 30 * S)), [4, 5, 6]);
        assert!(m.anchor("spk_1".parse().unwrap()).is_err());
    }

    #[test]
    fn searches_lines_and_events() {
        let (_d, m) = meeting();
        let q = |text: &str| Query { words: Query::words(text), ..Query::default() };
        let found = |q: &Query| -> Vec<String> {
            m.search(q)
                .iter()
                .map(|h| match h {
                    Hit::Line(l) => l.id.to_string(),
                    Hit::Event(e) => e.id.to_string(),
                })
                .collect()
        };
        assert_eq!(found(&q("Team-Invite")), ["u0006", "e002", "e003"]);
        assert_eq!(found(&q("mobile, users")), ["u0003", "e001"]);
        let mut decisions = q("invite");
        decisions.types = Some(vec!["decision".into()]);
        assert_eq!(found(&decisions), ["e002", "e003"]);
        let mut by_marcus = q("mobile");
        by_marcus.speakers = Some(vec![SpkId(1)]);
        assert_eq!(found(&by_marcus), ["u0003", "e001"], "spk_3's line and event count as Marcus's");
        assert_eq!(found(&q("prototype")), ["u0004"]);
        assert!(q("STRASSE").matches("Take the Straße exit"), "search ignores case beyond ASCII too");
        assert!(q("οσ").matches("ΟΣ"));
    }

    #[test]
    fn a_lines_language_is_kept_and_chinese_is_found_inside_a_line() {
        let root = tempfile::tempdir().unwrap();
        let mut folder =
            crate::folder::MeetingFolder::create(root.path(), "Mixed", chrono::Local::now()).unwrap();
        folder
            .write_info(&MeetingInfo {
                started_at: "2026-10-11T19:00:00+11:00".into(),
                status: "stopped".into(),
                ..MeetingInfo::default()
            })
            .unwrap();
        let line = |id, lang: Option<&str>, text: &str| Line {
            id: UtteranceId(id),
            t0: u64::from(id) * S,
            t1: u64::from(id) * S + S,
            slot: Some(0),
            spk: None,
            lang: lang.map(String::from),
            text: text.into(),
        };
        folder.append_line(&line(1, Some("zh-CN"), "大家好, 我是王磊, 但是还需要做一些性能测试")).unwrap();
        folder.append_line(&line(2, None, "Great!")).unwrap();
        let raw = fs::read_to_string(folder.path().join("transcript.jsonl")).unwrap();
        assert!(raw.contains(r#""slot":0,"lang":"zh-CN","text""#), "{raw}");
        assert!(!raw.lines().nth(1).unwrap().contains("lang"), "{raw}");

        let m = MeetingRecord::load(folder.path()).unwrap();
        assert_eq!(m.lines[0].lang.as_deref(), Some("zh-CN"));
        assert_eq!(m.lines[1].lang, None);
        let q = |text: &str| Query { words: Query::words(text), ..Query::default() };
        assert_eq!(m.search(&q("王磊")).len(), 1);
        assert_eq!(m.search(&q("性能测试，")).len(), 1, "full-width punctuation is dropped from the query");
        assert!(m.search(&q("王雷")).is_empty());
    }

    #[test]
    fn chooses_the_meeting() {
        fn at(root: &Path) -> impl FnOnce() -> Result<PathBuf> + '_ {
            move || Ok(root.to_path_buf())
        }
        let root = tempfile::tempdir().unwrap();
        let make = |name: &str, started: &str| {
            let d = root.path().join(name);
            fs::create_dir(&d).unwrap();
            fs::write(d.join("meeting.json"), format!(r#"{{"started_at":"{started}","status":"stopped"}}"#))
                .unwrap();
            d
        };
        let choose =
            |flag: Option<&Path>, env: Option<&Path>| choose_meeting(flag, env, at(root.path())).unwrap();
        let err = choose(None, None).unwrap_err();
        assert!(err.contains("no meetings under") && err.contains("pam listen"), "{err}");
        let old = make("2026-10-10-0900-b", "2026-10-10T09:00:00+11:00");
        let new = make("2026-10-10-0900-a", "2026-10-10T09:00:30+11:00");
        fs::create_dir(root.path().join("not-a-meeting")).unwrap();
        assert_eq!(choose(None, None).unwrap(), new);
        assert_eq!(choose(None, Some(&old)).unwrap(), old);
        assert_eq!(choose(Some(&new), Some(&old)).unwrap(), new);
        assert_eq!(choose(None, Some(Path::new(""))).unwrap(), new);
        let bad = root.path().join("not-a-meeting");
        assert!(choose(Some(&bad), None).unwrap_err().starts_with("--meeting"));
        assert!(choose(None, Some(&bad)).unwrap_err().starts_with("$PAM_MEETING"));
        // A meeting.json that can't be read is a failure, not a missing meeting.
        let broken = make("2026-10-10-0800-broken", "");
        fs::write(broken.join("meeting.json"), "{").unwrap();
        let err = choose_meeting(None, None, at(root.path())).unwrap_err();
        assert!(format!("{err:#}").contains("broken/meeting.json"), "{err:#}");
        assert_eq!(choose(Some(&new), None).unwrap(), new, "an explicit choice doesn't scan the root");
        // So is a meeting.json that isn't a file, chosen or found.
        fs::write(
            broken.join("meeting.json"),
            r#"{"started_at":"2026-10-10T08:00:00+11:00","status":"stopped"}"#,
        )
        .unwrap();
        let odd = root.path().join("2026-10-10-0700-odd");
        fs::create_dir_all(odd.join("meeting.json")).unwrap();
        let err = choose_meeting(Some(&odd), None, at(root.path())).unwrap_err();
        assert!(format!("{err:#}").contains("odd/meeting.json: it should be a file"), "{err:#}");
        let err = choose_meeting(None, None, at(root.path())).unwrap_err();
        assert!(format!("{err:#}").contains("odd/meeting.json: it should be a file"), "{err:#}");

        // The root is only looked for when nothing else chose the meeting.
        let no_root = || -> Result<PathBuf> { anyhow::bail!("HOME is not set") };
        assert_eq!(choose_meeting(Some(&new), None, no_root).unwrap(), Ok(new.clone()));
        assert!(choose_meeting(None, None, no_root).is_err());
        // A missing root has no meetings yet; a file there is broken.
        let missing = root.path().join("nowhere");
        assert!(
            choose_meeting(None, None, at(&missing)).unwrap().unwrap_err().starts_with("no meetings under")
        );
        let file = root.path().join("a-file");
        fs::write(&file, "").unwrap();
        let err = choose_meeting(None, None, at(&file)).unwrap_err();
        assert!(format!("{err:#}").contains("a-file: it should be a folder"), "{err:#}");
    }
}
