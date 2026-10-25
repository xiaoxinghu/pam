//! The agent's questions to the room: `tasks/<id>/questions.jsonl`.
//!
//! `pam task ask` appends a question and waits. The running meeting shows it
//! with a countdown and appends how it ended: the option the room chose, by
//! key or by voice, or for an open question the words of the answer, said or
//! typed; the countdown running out; or the question dropped,
//! because nobody waits for its answer any more. Every append happens under
//! the file's lock, and the first end written for a question is the one that
//! counts, so the asker and the meeting always agree on it.

use std::fs::{File, OpenOptions};
use std::path::Path;
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use chrono::Local;
use serde::{Deserialize, Serialize};

use crate::event::{SpkId, TaskId};
use crate::folder::{append_to, timestamp};
use crate::task;

/// The countdown, in seconds, when neither `--timeout` nor the config file
/// sets one. An open question gets twice the meeting's: a spoken answer
/// takes longer than a number.
pub const DEFAULT_TIMEOUT: u64 = 15;
/// The longest countdown, in seconds: a question never stalls the meeting for long.
pub const MAX_TIMEOUT: u64 = 60;
/// Options are answered with the keys 1–9.
pub const MAX_OPTIONS: usize = 9;
/// How long past its countdown the asker waits for the meeting before it
/// gives up on its own: the meeting holds a question through its task's
/// cancel window first, and it may have stopped.
const GRACE: Duration = Duration::from_secs(25);
/// The longest `pam task ask` can wait, in seconds: the longest countdown
/// plus [`GRACE`]. The agent is told, so that its own `bash` timeout doesn't
/// cut a question short.
pub const LONGEST_WAIT: u64 = MAX_TIMEOUT + GRACE.as_secs();
/// How often the asker looks for an answer.
const POLL: Duration = Duration::from_millis(100);

/// How the room answered.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum By {
    Key,
    Voice,
    /// In the composer, for an open question.
    Typed,
}

/// What the room answered. In `questions.jsonl` it is the record's
/// `option` or `text`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Answer {
    /// The option chosen, numbered from 1 as on the stage.
    Option(usize),
    /// The words of the answer to an open question.
    Text(String),
}

/// How a question ended.
#[derive(Debug, Clone, PartialEq)]
pub enum End {
    /// The room answered. `spk` said it, for an answer by voice.
    Answered { answer: Answer, by: By, spk: Option<SpkId> },
    /// The countdown ran out.
    TimedOut,
    /// Nobody waits for the answer any more, and why: the run ended, or the
    /// asker went away.
    Dropped(String),
}

/// A question as the meeting shows it.
#[derive(Debug, Clone, PartialEq)]
pub struct Question {
    /// Numbered from 1 within the task.
    pub q: u32,
    pub text: String,
    /// None for an open question, answered in words.
    pub options: Vec<String>,
    pub timeout: Duration,
    /// The `pam task ask` waiting for the answer.
    pub pid: u32,
}

/// One line of `questions.jsonl`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Record {
    /// The agent asked. `timeout` is in seconds; `pid` is the `pam task ask` waiting.
    Asked {
        q: u32,
        at: String,
        text: String,
        options: Vec<String>,
        timeout: u64,
        pid: u32,
    },
    Answered {
        q: u32,
        at: String,
        #[serde(flatten)]
        answer: Answer,
        by: By,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        spk: Option<SpkId>,
    },
    TimedOut {
        q: u32,
        at: String,
    },
    Dropped {
        q: u32,
        at: String,
        why: String,
    },
}

impl Record {
    pub fn q(&self) -> u32 {
        match self {
            Record::Asked { q, .. }
            | Record::Answered { q, .. }
            | Record::TimedOut { q, .. }
            | Record::Dropped { q, .. } => *q,
        }
    }

    /// The question, for an `asked` record.
    pub fn question(&self) -> Option<Question> {
        match self {
            Record::Asked { q, text, options, timeout, pid, .. } => Some(Question {
                q: *q,
                text: text.clone(),
                options: options.clone(),
                timeout: Duration::from_secs(*timeout),
                pid: *pid,
            }),
            _ => None,
        }
    }

    /// How the question ended, for any other record.
    pub fn end(&self) -> Option<End> {
        match self {
            Record::Asked { .. } => None,
            Record::Answered { answer, by, spk, .. } => {
                Some(End::Answered { answer: answer.clone(), by: *by, spk: *spk })
            }
            Record::TimedOut { .. } => Some(End::TimedOut),
            Record::Dropped { why, .. } => Some(End::Dropped(why.clone())),
        }
    }

    fn ending(q: u32, end: &End) -> Record {
        let at = timestamp(Local::now());
        match end {
            End::Answered { answer, by, spk } => {
                Record::Answered { q, at, answer: answer.clone(), by: *by, spk: *spk }
            }
            End::TimedOut => Record::TimedOut { q, at },
            End::Dropped(why) => Record::Dropped { q, at, why: why.clone() },
        }
    }
}

/// Asks the room `text`, with `options` (none for an open question) and a
/// countdown of `timeout` seconds, for task `id`, on behalf of this process.
/// Returns the question's number.
pub fn ask(meeting: &Path, id: TaskId, text: &str, options: &[String], timeout: u64) -> Result<u32> {
    let (mut file, records) = lock(meeting, id)?;
    let q = records.iter().filter(|r| r.question().is_some()).map(Record::q).max().unwrap_or(0) + 1;
    let asked = Record::Asked {
        q,
        at: timestamp(Local::now()),
        text: text.to_string(),
        options: options.to_vec(),
        timeout,
        pid: std::process::id(),
    };
    append(&mut file, meeting, id, &asked)?;
    Ok(q)
}

/// Ends question `q` of task `id` as `end`, unless it has ended already.
/// Returns the end that counts: the first one written.
pub fn end(meeting: &Path, id: TaskId, q: u32, end: End) -> Result<End> {
    let (mut file, records) = lock(meeting, id)?;
    if let Some(first) = records.iter().filter(|r| r.q() == q).find_map(Record::end) {
        return Ok(first);
    }
    append(&mut file, meeting, id, &Record::ending(q, &end))?;
    Ok(end)
}

/// How question `q` of task `id` ended, if it has.
pub fn ended(meeting: &Path, id: TaskId, q: u32) -> Result<Option<End>> {
    let (records, _) = task::lines_from::<Record>(&task::questions_file(meeting, id), 0)?;
    Ok(records.iter().filter(|r| r.q() == q).find_map(Record::end))
}

/// Waits for question `q` of task `id` to end. A meeting that hasn't ended
/// it [`GRACE`] after its countdown of `timeout` is gone: the question then
/// ends as timed out, here.
pub fn wait(meeting: &Path, id: TaskId, q: u32, timeout: Duration) -> Result<End> {
    let give_up = Instant::now() + timeout + GRACE;
    loop {
        if let Some(end) = ended(meeting, id, q)? {
            return Ok(end);
        }
        if Instant::now() >= give_up {
            return end(meeting, id, q, End::TimedOut);
        }
        thread::sleep(POLL);
    }
}

/// Whether process `pid` still runs: a question's asker, still waiting.
pub fn alive(pid: u32) -> bool {
    // 0 and negative pids would ask about process groups.
    let Ok(pid) = libc::pid_t::try_from(pid) else { return false };
    if pid <= 0 {
        return false;
    }
    // SAFETY: signal 0 only asks whether the process exists.
    let found = unsafe { libc::kill(pid, 0) } == 0;
    found || std::io::Error::last_os_error().raw_os_error() == Some(libc::EPERM)
}

/// Opens task `id`'s `questions.jsonl` and waits for its lock; returns it
/// and what it holds. The task must exist.
fn lock(meeting: &Path, id: TaskId) -> Result<(File, Vec<Record>)> {
    if !task::dir(meeting, id).join("task.json").is_file() {
        bail!("no task {id} in {}", meeting.display());
    }
    let path = task::questions_file(meeting, id);
    let file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
        .with_context(|| format!("opening {}", path.display()))?;
    file.lock().with_context(|| format!("locking {}", path.display()))?;
    let (records, _) = task::lines_from(&path, 0)?;
    Ok((file, records))
}

fn append(file: &mut File, meeting: &Path, id: TaskId, record: &Record) -> Result<()> {
    let mut json = serde_json::to_vec(record)?;
    json.push(b'\n');
    append_to(file, &json)
        .with_context(|| format!("appending to {}", task::questions_file(meeting, id).display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::task::{TaskState, TaskStatus, Via};

    fn meeting_with_task() -> tempfile::TempDir {
        let m = tempfile::tempdir().unwrap();
        let state = TaskState::new(TaskId(1), "draft the email", 0, Via::Typed, TaskStatus::Running);
        task::create(m.path(), &state).unwrap();
        m
    }

    fn options() -> Vec<String> {
        vec!["Acme".into(), "Globex".into()]
    }

    #[test]
    fn questions_are_numbered_and_the_first_end_counts() {
        let m = meeting_with_task();
        let dir = m.path();
        assert!(
            ask(dir, TaskId(2), "Which?", &options(), 15).unwrap_err().to_string().starts_with("no task t02")
        );
        assert_eq!(ask(dir, TaskId(1), "Which customer?", &options(), 15).unwrap(), 1);
        assert_eq!(ended(dir, TaskId(1), 1).unwrap(), None);
        let key = End::Answered { answer: Answer::Option(2), by: By::Key, spk: None };
        assert_eq!(end(dir, TaskId(1), 1, key.clone()).unwrap(), key);
        // The countdown ran out too late: the answer stands.
        assert_eq!(end(dir, TaskId(1), 1, End::TimedOut).unwrap(), key);
        assert_eq!(ended(dir, TaskId(1), 1).unwrap(), Some(key));
        assert_eq!(ask(dir, TaskId(1), "Formal or casual?", &options(), 15).unwrap(), 2);

        let text = std::fs::read_to_string(task::questions_file(dir, TaskId(1))).unwrap();
        let lines: Vec<serde_json::Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        assert_eq!(lines.len(), 3);
        assert_eq!(lines[0]["type"], "asked");
        assert_eq!(lines[0]["options"], serde_json::json!(["Acme", "Globex"]));
        assert_eq!(lines[0]["pid"], std::process::id());
        assert_eq!(lines[1]["type"], "answered");
        assert_eq!((&lines[1]["option"], &lines[1]["by"]), (&2.into(), &"key".into()));
        assert!(lines[1].get("spk").is_none(), "{}", lines[1]);
        assert!(lines[1].get("text").is_none(), "{}", lines[1]);
    }

    #[test]
    fn an_open_question_has_no_options_and_ends_with_the_words_of_its_answer() {
        let m = meeting_with_task();
        let dir = m.path();
        assert_eq!(ask(dir, TaskId(1), "Which city are you in?", &[], 30).unwrap(), 1);
        let said =
            End::Answered { answer: Answer::Text("Auckland".into()), by: By::Voice, spk: Some(SpkId(3)) };
        assert_eq!(end(dir, TaskId(1), 1, said.clone()).unwrap(), said);
        assert_eq!(ask(dir, TaskId(1), "Which day?", &[], 30).unwrap(), 2);
        let typed = End::Answered { answer: Answer::Text("Friday".into()), by: By::Typed, spk: None };
        assert_eq!(end(dir, TaskId(1), 2, typed.clone()).unwrap(), typed);
        assert_eq!(ended(dir, TaskId(1), 1).unwrap(), Some(said));
        assert_eq!(ended(dir, TaskId(1), 2).unwrap(), Some(typed));

        let text = std::fs::read_to_string(task::questions_file(dir, TaskId(1))).unwrap();
        let lines: Vec<serde_json::Value> = text.lines().map(|l| serde_json::from_str(l).unwrap()).collect();
        let keys = |v: &serde_json::Value| v.as_object().unwrap().keys().cloned().collect::<Vec<_>>();
        assert_eq!(lines[0]["options"], serde_json::json!([]));
        assert_eq!(keys(&lines[1]), ["at", "by", "q", "spk", "text", "type"]);
        assert_eq!(
            (&lines[1]["text"], &lines[1]["by"], &lines[1]["spk"]),
            (&"Auckland".into(), &"voice".into(), &"spk_3".into())
        );
        assert_eq!(keys(&lines[3]), ["at", "by", "q", "text", "type"]);
        assert_eq!((&lines[3]["text"], &lines[3]["by"]), (&"Friday".into(), &"typed".into()));
    }

    #[test]
    fn the_asker_gets_the_answer_written_while_it_waits() {
        let m = meeting_with_task();
        let dir = m.path().to_path_buf();
        let q = ask(&dir, TaskId(1), "Which customer?", &options(), 15).unwrap();
        let answering = {
            let dir = dir.clone();
            thread::spawn(move || {
                thread::sleep(Duration::from_millis(150));
                let voice = End::Answered { answer: Answer::Option(1), by: By::Voice, spk: Some(SpkId(2)) };
                end(&dir, TaskId(1), q, voice).unwrap()
            })
        };
        let got = wait(&dir, TaskId(1), q, Duration::from_secs(15)).unwrap();
        assert_eq!(got, answering.join().unwrap());
        assert_eq!(got, End::Answered { answer: Answer::Option(1), by: By::Voice, spk: Some(SpkId(2)) });
    }

    #[test]
    fn this_process_is_alive_and_pid_zero_never_is() {
        assert!(alive(std::process::id()));
        assert!(!alive(0));
        assert!(!alive(u32::MAX));
    }
}
