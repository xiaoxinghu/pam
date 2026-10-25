//! A meeting's tasks: `tasks/<id>/task.json`, what the agent says on the
//! stage, `tasks/<id>/messages.jsonl`, how far along it says it is and what
//! it produced, `tasks/<id>/updates.jsonl`, and the questions it asks the
//! room ([`crate::question`]). The agent's pi session is the meeting's, in `agent/`.
//!
//! The core owns `task.json` and every change of status: the agent
//! reports through files the meeting watches, so `task.json` and the events
//! keep in step with the run the core can see.

use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use chrono::Local;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::Ts;
use crate::event::{SpkId, TaskId, UtteranceId};
use crate::folder::{append_to, replace, timestamp};

/// Where a task stands.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TaskStatus {
    /// Held until the task before it finishes.
    Queued,
    Running,
    /// The agent finished.
    Done,
    /// The agent couldn't finish; `error` says why.
    Failed,
    /// The meeting stopped before the agent finished.
    Stopped,
    /// The room cancelled it ("Pam, cancel", or Esc in its cancel window).
    Cancelled,
}

impl TaskStatus {
    /// Still to finish: queued or running.
    pub fn is_open(self) -> bool {
        matches!(self, TaskStatus::Queued | TaskStatus::Running)
    }

    pub fn as_str(self) -> &'static str {
        match self {
            TaskStatus::Queued => "queued",
            TaskStatus::Running => "running",
            TaskStatus::Done => "done",
            TaskStatus::Failed => "failed",
            TaskStatus::Stopped => "stopped",
            TaskStatus::Cancelled => "cancelled",
        }
    }
}

/// How a request reached Pam.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Via {
    /// Said in the room ("Pam, …").
    Voice,
    /// Typed in the composer.
    Typed,
}

/// What a task produced: a file or a page, with a title.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskResult {
    pub title: String,
    #[serde(flatten)]
    pub target: Target,
}

/// Where a result is: `{"file": …}` or `{"url": …}` in `task.json`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Target {
    /// A path, relative to the meeting folder when it is inside it.
    File(String),
    Url(String),
}

impl Target {
    pub fn as_str(&self) -> &str {
        match self {
            Target::File(s) | Target::Url(s) => s,
        }
    }
}

/// `task.json`, and what `TaskUpdated` carries.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TaskState {
    pub id: TaskId,
    /// What was asked, as typed or said.
    pub request: String,
    /// The line that asked; `None` for a typed request.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line: Option<UtteranceId>,
    /// When it was asked, in meeting time.
    pub asked_at: Ts,
    /// Said or typed.
    pub by: Via,
    /// Who said it, as the speaker tracker had it when the request started;
    /// none when it didn't know yet. Once `line` is final, its speaker counts.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub spk: Option<SpkId>,
    pub status: TaskStatus,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
    /// What the room said since to change the request, oldest first.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub corrections: Vec<String>,
    /// How far along the agent says it is, 0–100; none until it reports.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress: Option<u8>,
    /// What the agent says it is doing now, in one line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
    /// What it produced, once attached.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<TaskResult>,
    /// When it first ran, in meeting time; none while it waits its turn.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub started_at: Option<Ts>,
    /// When it ended, in meeting time; none while it is queued or running.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ended_at: Option<Ts>,
}

impl TaskState {
    /// A new task: nothing reported, started or corrected yet.
    pub fn new(id: TaskId, request: &str, asked_at: Ts, by: Via, status: TaskStatus) -> TaskState {
        TaskState {
            id,
            request: request.to_string(),
            line: None,
            asked_at,
            by,
            spk: None,
            status,
            error: None,
            corrections: Vec::new(),
            progress: None,
            action: None,
            result: None,
            started_at: None,
            ended_at: None,
        }
    }

    /// Done with a result attached: how the room and the agent see it, as
    /// the chip, `pam status` and `context.md` show it. It is
    /// never a status of its own.
    pub fn is_ready(&self) -> bool {
        self.status == TaskStatus::Done && self.result.is_some()
    }

    /// The status as it is shown: `ready` for a done task with a result.
    pub fn shown_status(&self) -> &'static str {
        if self.is_ready() { "ready" } else { self.status.as_str() }
    }

    /// Takes the agent's report: a progress, action or result it gives
    /// replaces the one before. Returns whether anything changed.
    pub fn apply(&mut self, update: &Update) -> bool {
        let before = (self.progress, self.action.clone(), self.result.clone());
        if let Some(p) = update.progress {
            self.progress = Some(p.min(100));
        }
        if let Some(a) = update.action.as_deref().map(one_line).filter(|a| !a.is_empty()) {
            self.action = Some(a);
        }
        if let Some(r) = &update.result {
            self.result = Some(r.clone());
        }
        before != (self.progress, self.action.clone(), self.result.clone())
    }

    /// How far along it is, as its chip and card show it: `3/5` when its
    /// action counts ("reading source 3 of 5"), else `60%`; none until the
    /// agent reports either.
    pub fn progress_label(&self) -> Option<String> {
        match (self.action.as_deref().and_then(count), self.progress) {
            (Some((n, of)), _) => Some(format!("{n}/{of}")),
            (None, Some(p)) => Some(format!("{p}%")),
            (None, None) => None,
        }
    }

    /// The share done, 0–1, that the card's bar fills, in step with
    /// [`TaskState::progress_label`]: the action's count, else the progress.
    pub fn progress_share(&self) -> Option<f64> {
        match (self.action.as_deref().and_then(count), self.progress) {
            (Some((n, of)), _) => Some(n as f64 / of as f64),
            (None, Some(p)) => Some(p as f64 / 100.0),
            (None, None) => None,
        }
    }
}

/// The first count in `action`, as `3 of 5` or `3/5`: at least one, and no
/// more than it is of.
fn count(action: &str) -> Option<(u32, u32)> {
    let words: Vec<&str> = action
        .split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_ascii_alphanumeric() && c != '/'))
        .collect();
    let fits = |n: &str, of: &str| match (n.parse::<u32>(), of.parse::<u32>()) {
        (Ok(n), Ok(of)) if 1 <= n && n <= of => Some((n, of)),
        _ => None,
    };
    words.iter().enumerate().find_map(|(i, w)| match w.split_once('/') {
        Some((n, of)) => fits(n, of),
        None if words.get(i + 1) == Some(&"of") => fits(w, words.get(i + 2)?),
        None => None,
    })
}

/// `text` with its whitespace, line breaks included, made single spaces.
fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// One line of `messages.jsonl`: something shown on the stage for the task.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Message {
    /// Wall time, RFC 3339.
    pub at: String,
    pub text: String,
}

pub fn dir(meeting: &Path, id: TaskId) -> PathBuf {
    meeting.join("tasks").join(id.to_string())
}

pub fn messages_file(meeting: &Path, id: TaskId) -> PathBuf {
    dir(meeting, id).join("messages.jsonl")
}

pub fn questions_file(meeting: &Path, id: TaskId) -> PathBuf {
    dir(meeting, id).join("questions.jsonl")
}

pub fn updates_file(meeting: &Path, id: TaskId) -> PathBuf {
    dir(meeting, id).join("updates.jsonl")
}

/// Every task in the meeting, by ID. A folder under `tasks/` named like a task counts.
pub fn ids(meeting: &Path) -> Result<Vec<TaskId>> {
    let tasks = meeting.join("tasks");
    let entries = match fs::read_dir(&tasks) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e).with_context(|| format!("reading {}", tasks.display())),
    };
    let mut ids = Vec::new();
    for entry in entries {
        let entry = entry.with_context(|| format!("reading {}", tasks.display()))?;
        if entry.path().is_dir()
            && let Some(Ok(id)) = entry.file_name().to_str().map(str::parse::<TaskId>)
        {
            ids.push(id);
        }
    }
    ids.sort();
    Ok(ids)
}

/// The ID after the highest one so far, starting at `t01`.
pub fn next_id(meeting: &Path) -> Result<TaskId> {
    Ok(TaskId(ids(meeting)?.last().map_or(1, |t| t.0 + 1)))
}

/// Creates the task's folder and writes its `task.json`. Fails if it exists.
pub fn create(meeting: &Path, task: &TaskState) -> Result<()> {
    let dir = dir(meeting, task.id);
    fs::create_dir_all(dir.parent().unwrap()).with_context(|| format!("creating {}", dir.display()))?;
    fs::create_dir(&dir).with_context(|| format!("creating {}", dir.display()))?;
    write(meeting, task)
}

pub fn write(meeting: &Path, task: &TaskState) -> Result<()> {
    replace(&dir(meeting, task.id), "task.json", &serde_json::to_vec_pretty(task)?)
        .with_context(|| format!("saving task {}", task.id))
}

pub fn read(meeting: &Path, id: TaskId) -> Result<TaskState> {
    let path = dir(meeting, id).join("task.json");
    let bytes = fs::read(&path).with_context(|| format!("reading {}", path.display()))?;
    serde_json::from_slice(&bytes).with_context(|| format!("reading {}", path.display()))
}

/// One line of `updates.jsonl`: what the agent reported with `pam task
/// update` or attached with `pam task result`. The meeting applies it to
/// `task.json`, which only the core writes.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Update {
    /// Wall time, RFC 3339.
    pub at: String,
    /// 0–100.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub progress: Option<u8>,
    /// What it is doing now, in one line.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub action: Option<String>,
    /// What it produced: replaces the one before (one result per task).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub result: Option<TaskResult>,
}

/// Shows `text` on the stage for task `id`: appends it to `messages.jsonl`,
/// which the running meeting watches.
pub fn say(meeting: &Path, id: TaskId, text: &str) -> Result<Message> {
    let message = Message { at: timestamp(Local::now()), text: text.to_string() };
    append(meeting, id, &messages_file(meeting, id), &message)?;
    Ok(message)
}

/// Reports how far along task `id` is, what it is doing, or both: appends
/// it to `updates.jsonl`, which the running meeting watches.
pub fn update(meeting: &Path, id: TaskId, progress: Option<u8>, action: Option<&str>) -> Result<Update> {
    let update = Update { at: timestamp(Local::now()), progress, action: action.map(one_line), result: None };
    append(meeting, id, &updates_file(meeting, id), &update)?;
    Ok(update)
}

/// Attaches `result` to task `id`, replacing the one before: appends it to
/// `updates.jsonl`, which the running meeting watches.
pub fn attach(meeting: &Path, id: TaskId, result: TaskResult) -> Result<Update> {
    let update = Update { at: timestamp(Local::now()), progress: None, action: None, result: Some(result) };
    append(meeting, id, &updates_file(meeting, id), &update)?;
    Ok(update)
}

/// A file result's target: `path`, which must exist, relative to the
/// meeting folder when it is inside it, else absolute. A relative `path`
/// is taken from `cwd`.
pub fn file_target(meeting: &Path, cwd: &Path, path: &Path) -> Result<Target> {
    let full = cwd.join(path);
    let real = fs::canonicalize(&full).with_context(|| format!("no file {}", path.display()))?;
    let home = fs::canonicalize(meeting).with_context(|| format!("reading {}", meeting.display()))?;
    let shown = match real.strip_prefix(&home) {
        Ok(inside) if !inside.as_os_str().is_empty() => inside.to_path_buf(),
        _ => real,
    };
    Ok(Target::File(shown.to_string_lossy().into_owned()))
}

/// Where a file target is on disk: relative ones are in the meeting folder.
pub fn file_path(meeting: &Path, file: &str) -> PathBuf {
    meeting.join(file)
}

/// What a result's line points at: the target of its first Markdown link,
/// `[title](target)`, else its first `http://` or `https://` address. A
/// target may hold balanced parentheses, as `Function_(mathematics)` does.
pub fn link_on(line: &str) -> Option<String> {
    let markdown = line.match_indices("](").find_map(|(i, _)| destination(&line[i + 2..]));
    markdown.map(unescape).or_else(|| {
        let at = ["https://", "http://"].iter().filter_map(|s| line.find(s)).min()?;
        let mut url = line[at..].split(|c: char| c.is_whitespace() || "<>\"'`".contains(c)).next()?;
        // Punctuation after it is the sentence's, and so is a `)` it doesn't open.
        loop {
            let trimmed = url.trim_end_matches(['.', ',', ';', ':', ']', '!', '?']);
            let open = trimmed.matches('(').count() < trimmed.matches(')').count();
            url = if open { &trimmed[..trimmed.len() - 1] } else { trimmed };
            if url.len() == trimmed.len() {
                return Some(url.to_string());
            }
        }
    })
}

/// A Markdown link's destination as written, from just after its `(`: up
/// to the `)` that closes it, past balanced or escaped ones, or `<…>` up to
/// its `>`. None when it is empty, never closes, or has a space outside `<…>`.
fn destination(rest: &str) -> Option<&str> {
    let rest = rest.trim_start();
    if let Some(inner) = rest.strip_prefix('<') {
        let mut escaped = false;
        let end = inner.char_indices().find_map(|(i, c)| match c {
            _ if escaped => {
                escaped = false;
                None
            }
            '\\' => {
                escaped = true;
                None
            }
            '>' => Some(i),
            _ => None,
        })?;
        return (end > 0).then_some(&inner[..end]);
    }
    let mut depth = 0;
    let mut escaped = false;
    for (i, c) in rest.char_indices() {
        match c {
            _ if escaped => escaped = false,
            '\\' => escaped = true,
            '(' => depth += 1,
            ')' if depth == 0 => {
                let target = rest[..i].trim_end();
                return (!target.is_empty() && !target.contains(char::is_whitespace)).then_some(target);
            }
            ')' => depth -= 1,
            _ => {}
        }
    }
    None
}

/// `text` with Markdown's backslash escapes of ASCII punctuation decoded:
/// `foo\)bar.md` is `foo)bar.md`.
fn unescape(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut chars = text.chars().peekable();
    while let Some(c) = chars.next() {
        match chars.peek() {
            Some(&next) if c == '\\' && next.is_ascii_punctuation() => {
                out.push(next);
                chars.next();
            }
            _ => out.push(c),
        }
    }
    out
}

/// `text` escaped for a Markdown link's text: `[`, `]` and `\` get a backslash.
pub fn markdown_text(text: &str) -> String {
    text.chars().fold(String::new(), |mut out, c| {
        if matches!(c, '[' | ']' | '\\') {
            out.push('\\');
        }
        out.push(c);
        out
    })
}

impl Target {
    /// The target as a Markdown link's destination. A file's path is
    /// percent-encoded, every byte but `A-Z a-z 0-9 - . _ ~ /`, so a `#`,
    /// `?`, `%`, space or parenthesis in it still links to the file; a
    /// URL is already one, and goes in `<…>` when it holds a parenthesis.
    pub fn markdown(&self) -> String {
        match self {
            Target::File(f) => f.bytes().fold(String::new(), |mut out, b| {
                if b.is_ascii_alphanumeric() || b"-._~/".contains(&b) {
                    out.push(b as char);
                } else {
                    out.push_str(&format!("%{b:02X}"));
                }
                out
            }),
            Target::Url(u) if u.contains(|c: char| c.is_whitespace() || "()<>".contains(c)) => {
                let mut out = String::from("<");
                for c in u.chars() {
                    if matches!(c, '<' | '>' | '\\') {
                        out.push('\\');
                    }
                    out.push(c);
                }
                out.push('>');
                out
            }
            Target::Url(u) => u.clone(),
        }
    }
}

/// The file a link from [`link_on`] names, or None when it is an address
/// (`scheme://…` or `mailto:`): the link without its `#…`, percent-decoded.
pub fn link_file(link: &str) -> Option<String> {
    if link.contains("://") || link.starts_with("mailto:") {
        return None;
    }
    let link = link.split('#').next().unwrap_or_default();
    let bytes = link.as_bytes();
    let mut out = Vec::with_capacity(bytes.len());
    let mut i = 0;
    while i < bytes.len() {
        let hex = bytes.get(i + 1..i + 3).filter(|h| h.iter().all(u8::is_ascii_hexdigit));
        match hex {
            Some(h) if bytes[i] == b'%' => {
                let b = u8::from_str_radix(std::str::from_utf8(h).unwrap(), 16).unwrap();
                out.push(b);
                i += 3;
            }
            _ => {
                out.push(bytes[i]);
                i += 1;
            }
        }
    }
    Some(String::from_utf8(out).unwrap_or_else(|_| link.to_string()))
}

/// Appends `record` to task `id`'s JSON-lines file `path`, as one whole line.
fn append(meeting: &Path, id: TaskId, path: &Path, record: &impl Serialize) -> Result<()> {
    if !dir(meeting, id).join("task.json").is_file() {
        bail!("no task {id} in {}", meeting.display());
    }
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(path)
        .with_context(|| format!("opening {}", path.display()))?;
    let mut json = serde_json::to_vec(record)?;
    json.push(b'\n');
    append_to(&mut file, &json).with_context(|| format!("appending to {}", path.display()))
}

/// The messages written to `path` from byte `from` on, and where the next
/// read starts: after the last whole line. A missing file has none.
pub fn messages_from(path: &Path, from: u64) -> Result<(Vec<Message>, u64)> {
    lines_from(path, from)
}

/// The records appended to the JSON-lines file `path` from byte `from` on,
/// and where the next read starts: after the last whole line, so a line still
/// being written waits for the next read. A missing file has none.
pub fn lines_from<T: DeserializeOwned>(path: &Path, from: u64) -> Result<(Vec<T>, u64)> {
    let mut file = match fs::File::open(path) {
        Ok(f) => f,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok((Vec::new(), from)),
        Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
    };
    let mut bytes = Vec::new();
    file.seek(SeekFrom::Start(from))
        .and_then(|_| file.read_to_end(&mut bytes))
        .with_context(|| format!("reading {}", path.display()))?;
    let whole = bytes.iter().rposition(|&b| b == b'\n').map_or(0, |i| i + 1);
    let mut records = Vec::new();
    for line in bytes[..whole].split(|&b| b == b'\n').filter(|l| !l.is_empty()) {
        records.push(serde_json::from_slice(line).with_context(|| format!("reading {}", path.display()))?);
    }
    Ok((records, from + whole as u64))
}

/// `agent/running.json`: the task the agent is working on, written by the
/// agent thread before it prompts the task and removed once the run ends.
fn running_file(meeting: &Path) -> PathBuf {
    meeting.join("agent").join("running.json")
}

#[derive(Serialize, Deserialize)]
struct Running {
    task: TaskId,
}

/// Records that the agent is working on `id`, or, with `None`, on nothing.
pub fn set_running(meeting: &Path, id: Option<TaskId>) -> Result<()> {
    let path = running_file(meeting);
    match id {
        Some(task) => {
            let dir = path.parent().unwrap();
            fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
            replace(dir, "running.json", &serde_json::to_vec(&Running { task })?)
        }
        None => match fs::remove_file(&path) {
            Err(e) if e.kind() != std::io::ErrorKind::NotFound => {
                Err(e).with_context(|| format!("removing {}", path.display()))
            }
            _ => Ok(()),
        },
    }
}

/// The meeting's running task, the one the agent is working on (one runs
/// at a time), from `agent/running.json`; none when it is missing.
pub fn running(meeting: &Path) -> Result<Option<TaskId>> {
    let path = running_file(meeting);
    match fs::read(&path) {
        Ok(bytes) => {
            let r: Running =
                serde_json::from_slice(&bytes).with_context(|| format!("reading {}", path.display()))?;
            Ok(Some(r.task))
        }
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => Ok(None),
        Err(e) => Err(e).with_context(|| format!("reading {}", path.display())),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn task(id: u32) -> TaskState {
        TaskState::new(TaskId(id), "what did we decide?", 16_000, Via::Typed, TaskStatus::Running)
    }

    #[test]
    fn ids_count_up_from_the_highest_task_folder() {
        let m = tempfile::tempdir().unwrap();
        assert_eq!(next_id(m.path()).unwrap(), TaskId(1));
        create(m.path(), &task(1)).unwrap();
        create(m.path(), &task(7)).unwrap();
        fs::create_dir_all(m.path().join("tasks/notes")).unwrap();
        fs::write(m.path().join("tasks/t09"), "a file, not a task").unwrap();
        assert_eq!(ids(m.path()).unwrap(), [TaskId(1), TaskId(7)]);
        assert_eq!(next_id(m.path()).unwrap(), TaskId(8));
        assert!(create(m.path(), &task(7)).is_err(), "a task is never created twice");
    }

    #[test]
    fn task_json_round_trips_and_leaves_out_what_is_unset() {
        let m = tempfile::tempdir().unwrap();
        create(m.path(), &task(2)).unwrap();
        let json: serde_json::Value =
            serde_json::from_slice(&fs::read(m.path().join("tasks/t02/task.json")).unwrap()).unwrap();
        assert_eq!(
            json,
            serde_json::json!({"id": "t02", "request": "what did we decide?", "asked_at": 16000, "by": "typed", "status": "running"})
        );
        let failed = TaskState { status: TaskStatus::Failed, error: Some("no login".into()), ..task(2) };
        write(m.path(), &failed).unwrap();
        assert_eq!(read(m.path(), TaskId(2)).unwrap(), failed);
        let ready = TaskState {
            status: TaskStatus::Done,
            progress: Some(100),
            action: Some("writing it up".into()),
            result: Some(TaskResult {
                title: "Sources".into(),
                target: Target::File("tasks/t02/result.md".into()),
            }),
            started_at: Some(16_000),
            ended_at: Some(32_000),
            ..task(2)
        };
        write(m.path(), &ready).unwrap();
        let json: serde_json::Value =
            serde_json::from_slice(&fs::read(m.path().join("tasks/t02/task.json")).unwrap()).unwrap();
        assert_eq!(json["result"], serde_json::json!({"title": "Sources", "file": "tasks/t02/result.md"}));
        assert_eq!(
            (&json["progress"], &json["started_at"], &json["ended_at"]),
            (&100.into(), &16000.into(), &32000.into())
        );
        assert_eq!(read(m.path(), TaskId(2)).unwrap(), ready);
        assert_eq!((ready.shown_status(), failed.shown_status()), ("ready", "failed"));
        let url = TaskResult { title: "Docs".into(), target: Target::Url("https://example.com".into()) };
        assert_eq!(
            serde_json::to_value(&url).unwrap(),
            serde_json::json!({"title": "Docs", "url": "https://example.com"})
        );
    }

    #[test]
    fn messages_are_read_a_whole_line_at_a_time() {
        let m = tempfile::tempdir().unwrap();
        assert!(say(m.path(), TaskId(1), "hello").unwrap_err().to_string().starts_with("no task t01"));
        create(m.path(), &task(1)).unwrap();
        let path = messages_file(m.path(), TaskId(1));
        assert_eq!(messages_from(&path, 0).unwrap(), (Vec::new(), 0));
        say(m.path(), TaskId(1), "first").unwrap();
        let (got, at) = messages_from(&path, 0).unwrap();
        assert_eq!(got.iter().map(|m| m.text.as_str()).collect::<Vec<_>>(), ["first"]);
        // A line still being written waits for the next read.
        let mut f = OpenOptions::new().append(true).open(&path).unwrap();
        std::io::Write::write_all(&mut f, br#"{"at":"x","text":"sec"#).unwrap();
        assert_eq!(messages_from(&path, at).unwrap(), (Vec::new(), at));
        std::io::Write::write_all(&mut f, b"ond\"}\n").unwrap();
        let (got, _) = messages_from(&path, at).unwrap();
        assert_eq!(got, [Message { at: "x".into(), text: "second".into() }]);
    }

    #[test]
    fn updates_are_appended_whole_and_applied_field_by_field() {
        let m = tempfile::tempdir().unwrap();
        assert!(
            update(m.path(), TaskId(1), Some(5), None).unwrap_err().to_string().starts_with("no task t01")
        );
        create(m.path(), &task(1)).unwrap();
        update(m.path(), TaskId(1), Some(40), Some("reading\n  sources")).unwrap();
        update(m.path(), TaskId(1), None, Some("reading source 3 of 5")).unwrap();
        let path = updates_file(m.path(), TaskId(1));
        let raw = fs::read_to_string(&path).unwrap();
        assert_eq!(raw.lines().count(), 2);
        assert!(
            raw.lines().next().unwrap().ends_with(r#","progress":40,"action":"reading sources"}"#),
            "{raw}"
        );
        let (got, at) = lines_from::<Update>(&path, 0).unwrap();
        assert_eq!(at, raw.len() as u64);

        let mut t = task(1);
        assert_eq!((t.progress_label(), t.progress_share()), (None, None));
        assert!(t.apply(&got[0]));
        assert_eq!((t.progress, t.action.as_deref()), (Some(40), Some("reading sources")));
        assert_eq!((t.progress_label().as_deref(), t.progress_share()), (Some("40%"), Some(0.4)));
        // A report without a progress keeps the one before.
        assert!(t.apply(&got[1]));
        assert_eq!((t.progress, t.action.as_deref()), (Some(40), Some("reading source 3 of 5")));
        assert_eq!(t.progress_label().as_deref(), Some("3/5"), "the action counts");
        assert_eq!(t.progress_share(), Some(0.6), "and the bar follows it");
        assert!(!t.apply(&got[1]), "the same report again changes nothing");
        let wild = Update { at: "x".into(), progress: Some(250), action: Some("  ".into()), result: None };
        assert!(t.apply(&wild));
        assert_eq!((t.progress, t.action.as_deref()), (Some(100), Some("reading source 3 of 5")));
    }

    #[test]
    fn a_result_is_appended_as_a_report_and_replaces_the_one_before() {
        let m = tempfile::tempdir().unwrap();
        create(m.path(), &task(1)).unwrap();
        let file = |f: &str| TaskResult { title: "Sources".into(), target: Target::File(f.into()) };
        attach(m.path(), TaskId(1), file("tasks/t01/result.md")).unwrap();
        let url = TaskResult { title: "Docs".into(), target: Target::Url("https://example.com".into()) };
        attach(m.path(), TaskId(1), url.clone()).unwrap();
        let path = updates_file(m.path(), TaskId(1));
        let raw = fs::read_to_string(&path).unwrap();
        assert!(
            raw.lines()
                .next()
                .unwrap()
                .ends_with(r#","result":{"title":"Sources","file":"tasks/t01/result.md"}}"#),
            "{raw}"
        );
        let (got, _) = lines_from::<Update>(&path, 0).unwrap();
        let mut t = TaskState { progress: Some(60), ..task(1) };
        assert!(t.apply(&got[0]));
        assert_eq!((t.result.as_ref(), t.progress), (Some(&file("tasks/t01/result.md")), Some(60)));
        assert!(t.apply(&got[1]), "one result per task: the next replaces it");
        assert_eq!(t.result.as_ref(), Some(&url));
        assert!(!t.apply(&got[1]));
        // A progress report keeps the result.
        assert!(t.apply(&Update { at: "x".into(), progress: Some(70), action: None, result: None }));
        assert_eq!(t.result.as_ref(), Some(&url));
    }

    #[test]
    fn a_file_result_is_kept_relative_to_the_meeting_folder_when_it_is_in_it() {
        let root = tempfile::tempdir().unwrap();
        let m = root.path().join("meeting");
        fs::create_dir_all(m.join("tasks/t01")).unwrap();
        fs::write(m.join("tasks/t01/result.md"), "x").unwrap();
        fs::write(root.path().join("elsewhere.md"), "x").unwrap();
        let target = |cwd: &Path, f: &str| file_target(&m, cwd, Path::new(f)).map(|t| t.as_str().to_string());
        assert_eq!(target(&m, "tasks/t01/result.md").unwrap(), "tasks/t01/result.md");
        assert_eq!(target(&m.join("tasks"), "t01/result.md").unwrap(), "tasks/t01/result.md");
        assert_eq!(
            target(root.path(), "meeting/tasks/../tasks/t01/result.md").unwrap(),
            "tasks/t01/result.md"
        );
        let outside = fs::canonicalize(root.path().join("elsewhere.md")).unwrap();
        assert_eq!(target(&m, "../elsewhere.md").unwrap(), outside.to_str().unwrap());
        assert_eq!(target(&m, outside.to_str().unwrap()).unwrap(), outside.to_str().unwrap());
        assert!(target(&m, "tasks/t01/missing.md").is_err());
        assert_eq!(file_path(&m, "tasks/t01/result.md"), m.join("tasks/t01/result.md"));
        assert_eq!(file_path(&m, outside.to_str().unwrap()), outside);
    }

    #[test]
    fn a_line_points_at_its_first_link_or_address() {
        let link = |line: &str| link_on(line);
        let some = |s: &str| Some(s.to_string());
        assert_eq!(link("1. [Study](https://example.com/a) and more"), some("https://example.com/a"));
        assert_eq!(link("- notes: [interviews](../../notes.md)"), some("../../notes.md"));
        assert_eq!(link("see https://example.com/b, then (http://x.org/c)."), some("https://example.com/b"));
        assert_eq!(link("(at http://x.org/c)."), some("http://x.org/c"));
        assert_eq!(link("<https://example.com/d>"), some("https://example.com/d"));
        assert_eq!(link("[a]( ) and https://e.org"), some("https://e.org"));
        assert_eq!(link("no link here"), None);
        // Parentheses that belong to the address.
        let wiki = "https://en.wikipedia.org/wiki/Function_(mathematics)";
        assert_eq!(link(&format!("1. [spec]({wiki}) and more")), some(wiki));
        assert_eq!(link(&format!("see {wiki}.")), some(wiki));
        assert_eq!(link(&format!("(see {wiki})")), some(wiki));
        assert_eq!(link("[a](<docs/my notes.md>)"), some("docs/my notes.md"));
        assert_eq!(link("[a](https://x.org/(unclosed"), some("https://x.org/(unclosed"), "the address, then");
        assert_eq!(link("[a](docs/(unclosed"), None);
        // Escapes are decoded: what is opened is the destination, not how it was written.
        assert_eq!(link(r"[a](https://x.org/a\)b)"), some("https://x.org/a)b"));
        assert_eq!(link(r"[report](foo\)bar.md)"), some("foo)bar.md"));
        assert_eq!(link(r"[a](<x \> y.md>)"), some("x > y.md"));
        assert_eq!(link(r"[a](a\b.md)"), some(r"a\b.md"), "a backslash before a letter stays");
    }

    #[test]
    fn a_file_target_links_percent_encoded_and_reads_back_as_it_was() {
        let file = |f: &str| Target::File(f.into()).markdown();
        assert_eq!(file("tasks/t01/result.md"), "tasks/t01/result.md");
        assert_eq!(file("/Users/a/final report.md"), "/Users/a/final%20report.md");
        assert_eq!(file("tasks/t01/report#final.md"), "tasks/t01/report%23final.md");
        assert_eq!(file("a (b).md"), "a%20%28b%29.md");
        assert_eq!(file("été.md"), "%C3%A9t%C3%A9.md");
        assert_eq!(
            Target::Url("https://x.org/Function_(m)#Notes".into()).markdown(),
            "<https://x.org/Function_(m)#Notes>"
        );
        assert_eq!(Target::Url("https://x.org/a?b=1#c".into()).markdown(), "https://x.org/a?b=1#c");
        assert_eq!(markdown_text(r"Sources [2] a\b"), r"Sources \[2\] a\\b");
        // What is written is read back as it was, whatever the path holds.
        for t in [
            "tasks/t01/final report.md",
            "report#final.md",
            "a?b.md",
            "100%.md",
            "a (b).md",
            r"x<y>\z.md",
            "été.md",
        ] {
            let line = format!("[{}]({})", markdown_text("T [1]"), file(t));
            assert_eq!(link_on(&line).and_then(|l| link_file(&l)).as_deref(), Some(t), "{line}");
        }
        // A link the agent wrote by hand: its `#…` is a heading, a stray `%` stays.
        assert_eq!(link_file("../../notes.md#decisions").as_deref(), Some("../../notes.md"));
        assert_eq!(link_file("docs/my%20notes.md").as_deref(), Some("docs/my notes.md"));
        assert_eq!(link_file("100%.md").as_deref(), Some("100%.md"));
        assert_eq!(link_file("a%+1.md").as_deref(), Some("a%+1.md"));
        assert_eq!(link_file("https://x.org/a%20b"), None);
        assert_eq!(link_file("mailto:a@b.org"), None);
    }

    #[test]
    fn an_action_counts_as_n_of_m_or_n_slash_m() {
        let label = |action: &str| {
            let mut t = task(1);
            t.progress = Some(60);
            t.action = Some(action.into());
            t.progress_label().unwrap()
        };
        assert_eq!(label("reading source 3 of 5"), "3/5");
        assert_eq!(label("step 2/4: building the page"), "2/4");
        assert_eq!(label("checked (4/4)."), "4/4");
        assert_eq!(label("reading 2 of the sources"), "60%");
        assert_eq!(label("6 of 5 done"), "60%", "never more than it is of");
        assert_eq!(label("0 of 5 done"), "60%");
        assert_eq!(label("writing it up"), "60%");
    }

    #[test]
    fn the_running_task_is_the_one_the_agent_named() {
        let m = tempfile::tempdir().unwrap();
        assert_eq!(running(m.path()).unwrap(), None);
        set_running(m.path(), Some(TaskId(3))).unwrap();
        assert_eq!(running(m.path()).unwrap(), Some(TaskId(3)));
        assert_eq!(fs::read_to_string(running_file(m.path())).unwrap(), r#"{"task":"t03"}"#);
        set_running(m.path(), Some(TaskId(4))).unwrap();
        assert_eq!(running(m.path()).unwrap(), Some(TaskId(4)));
        set_running(m.path(), None).unwrap();
        assert_eq!(running(m.path()).unwrap(), None);
        set_running(m.path(), None).unwrap();
        fs::write(running_file(m.path()), "t04").unwrap();
        assert!(running(m.path()).unwrap_err().to_string().starts_with("reading "));
    }
}
