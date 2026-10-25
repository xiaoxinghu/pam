//! `pam task`: commands for the agent's own task. `pam task update` reports
//! how far along it is and `pam task result` attaches what it produced,
//! which the meeting applies to `task.json`, its only writer, and `pam task
//! ask` asks the room a question on the stage and waits for the answer.

use std::fmt::Write as _;
use std::path::Path;
use std::time::Duration;

use pam_core::clock::SECOND;
use pam_core::question::{self, Answer, By, DEFAULT_TIMEOUT, End, MAX_OPTIONS, MAX_TIMEOUT};
use pam_core::{Target, TaskId, TaskResult, TaskStatus, task};
use serde_json::json;

use crate::cli::{Command, Fail, Parsed};
use crate::read;

/// Each task command, with the options only it takes.
const OPTIONS: [(&str, &[&str]); 3] = [
    ("update", &["--progress", "--action"]),
    ("ask", &["--option", "--timeout"]),
    ("result", &["--file", "--url", "--title"]),
];

pub fn run(cmd: &'static Command, p: Parsed, out: &mut String) -> Result<(), Fail> {
    let sub = p.positionals.first().map(String::as_str);
    if let Some(sub) = sub.filter(|s| OPTIONS.iter().any(|(name, _)| name == s)) {
        for (other, theirs) in OPTIONS.iter().filter(|(name, _)| *name != sub) {
            if let Some(name) = theirs.iter().find(|o| p.has(o)) {
                return Err(Fail::usage(
                    cmd,
                    format!("{name} is for `pam task {other}`, not `pam task {sub}`"),
                ));
            }
        }
    }
    match sub {
        Some("update") => update(cmd, &p, out),
        Some("ask") => ask(cmd, &p, out),
        Some("result") => result(cmd, &p, out),
        Some(other) => Err(Fail::usage(
            cmd,
            format!("unknown task command `{other}`: the ones there are are update, result and ask"),
        )),
        None => Err(Fail::usage(cmd, "missing what to do with the task: update, result or ask")),
    }
}

/// `pam task update [<id>] --progress <0-100> --action <text>`: reports how
/// far along a running task is. It appends the report to the task's
/// `updates.jsonl`, and the meeting saves it in `task.json` and shows it.
fn update(cmd: &'static Command, p: &Parsed, out: &mut String) -> Result<(), Fail> {
    let given = given_id(cmd, p, "an action of several words, e.g. --action \"reading sources\"")?;
    let progress = match p.value("--progress") {
        Some(v) => match v.strip_suffix('%').unwrap_or(v).parse::<u8>() {
            Ok(n) if n <= 100 => Some(n),
            _ => return Err(Fail::usage(cmd, format!("--progress must be 0 to 100, not `{v}`"))),
        },
        None => None,
    };
    let action = p.value("--action");
    if action.is_some_and(|a| a.trim().is_empty()) {
        return Err(Fail::usage(cmd, "--action is empty: say what the task is doing now"));
    }
    if progress.is_none() && action.is_none() {
        return Err(Fail::usage(cmd, "nothing to report: give --progress, --action or both"));
    }
    let rec = read::meeting(p)?;
    let id = which_task(cmd, given, "Give its ID, e.g. pam task update t01 …", &rec.dir)?;
    let status = task::read(&rec.dir, id)?.status;
    if status != TaskStatus::Running {
        return Err(Fail::usage(
            cmd,
            format!("{id} is {}, not running: only a running task reports progress", status.as_str()),
        ));
    }
    let update = task::update(&rec.dir, id, progress, action)?;
    if p.has("--json") {
        let v = json!({ "task": id, "update": update });
        writeln!(out, "{}", serde_json::to_string_pretty(&v).unwrap()).unwrap();
    } else {
        let what: Vec<String> =
            update.progress.map(|n| format!("{n}%")).into_iter().chain(update.action.clone()).collect();
        writeln!(out, "{id} updated: {}", what.join(" · ")).unwrap();
    }
    Ok(())
}

/// `pam task result [<id>] (--file <path> | --url <url>) [--title <t>]`:
/// attaches what a running task produced, replacing the one before. It
/// appends it to the task's `updates.jsonl`, and the meeting saves it in
/// `task.json`; the task's chip turns ready once it is done.
fn result(cmd: &'static Command, p: &Parsed, out: &mut String) -> Result<(), Fail> {
    let given = given_id(cmd, p, "a title of several words, e.g. --title \"Sources\"")?;
    let title = p.value("--title");
    if title.is_some_and(|t| t.trim().is_empty()) {
        return Err(Fail::usage(cmd, "--title is empty: name the result, or leave --title out"));
    }
    let (file, url) = (p.value("--file"), p.value("--url"));
    if file.is_some() && url.is_some() {
        return Err(Fail::usage(cmd, "give --file or --url, not both: a task has one result"));
    }
    if let Some(u) = url {
        let scheme = u.split_once("://").map(|(s, rest)| (s.to_ascii_lowercase(), rest));
        if !matches!(&scheme, Some((s, rest)) if (s == "http" || s == "https") && !rest.trim().is_empty()) {
            return Err(Fail::usage(cmd, format!("--url must be an http:// or https:// address, not `{u}`")));
        }
        if u.contains(char::is_whitespace) {
            return Err(Fail::usage(cmd, format!("--url has a space in it: `{u}`")));
        }
    }
    if file.is_some_and(|f| f.trim().is_empty()) {
        return Err(Fail::usage(cmd, "--file is empty: give the path of the file to attach"));
    }
    let rec = read::meeting(p)?;
    let id = which_task(cmd, given, "Give its ID, e.g. pam task result t01 …", &rec.dir)?;
    let status = task::read(&rec.dir, id)?.status;
    if status != TaskStatus::Running {
        return Err(Fail::usage(
            cmd,
            format!("{id} is {}, not running: only a running task attaches a result", status.as_str()),
        ));
    }
    let target = match (file, url) {
        (Some(f), _) => {
            let cwd = std::env::current_dir().map_err(anyhow::Error::from)?;
            if !cwd.join(f).exists() {
                return Err(Fail::usage(cmd, format!("no file {f}: write it first, then attach it")));
            }
            task::file_target(&rec.dir, &cwd, Path::new(f))?
        }
        (None, Some(u)) => Target::Url(u.to_string()),
        (None, None) => {
            return Err(Fail::usage(cmd, "nothing to attach: give --file <path> or --url <url>"));
        }
    };
    let title = match (title, &target) {
        (Some(t), _) => t.split_whitespace().collect::<Vec<_>>().join(" "),
        (None, Target::File(f)) => {
            Path::new(f).file_name().map_or(f.clone(), |n| n.to_string_lossy().into_owned())
        }
        (None, Target::Url(u)) => u.clone(),
    };
    let update = task::attach(&rec.dir, id, TaskResult { title, target })?;
    let result = update.result.as_ref().unwrap();
    if p.has("--json") {
        let v = json!({ "task": id, "result": result });
        writeln!(out, "{}", serde_json::to_string_pretty(&v).unwrap()).unwrap();
    } else {
        writeln!(out, "{id} result: {} · {}", result.title, result.target.as_str()).unwrap();
    }
    Ok(())
}

/// `pam task ask [<id>] <question> --option <a> --option <b>…`: prints the
/// option the room chose, or `timeout`. With no options the question is
/// open, and it prints the words of the answer.
fn ask(cmd: &'static Command, p: &Parsed, out: &mut String) -> Result<(), Fail> {
    let args = &p.positionals[1..];
    // The ID is optional, so a first word only counts as one when a question follows it.
    let (given, words) = match args {
        [first, rest @ ..] if !rest.is_empty() && first.parse::<TaskId>().is_ok() => {
            (first.parse::<TaskId>().ok(), rest)
        }
        _ => (None, args),
    };
    let text = words.join(" ");
    if text.trim().is_empty() {
        return Err(Fail::usage(cmd, "missing <question>: what should I ask the room?"));
    }
    let options = p.values("--option");
    if options.len() == 1 {
        return Err(Fail::usage(
            cmd,
            "give at least two --option for the room to choose from, or none for an open question",
        ));
    }
    if options.len() > MAX_OPTIONS {
        return Err(Fail::usage(
            cmd,
            format!("at most {MAX_OPTIONS} --option: the room answers with the keys 1–{MAX_OPTIONS}"),
        ));
    }
    if options.iter().any(|o| o.trim().is_empty()) {
        return Err(Fail::usage(cmd, "an --option is empty: say what the room would choose"));
    }
    let asked_for = match p.value("--timeout") {
        Some(t) => {
            let secs = read::duration(t).map_err(|m| Fail::usage(cmd, format!("--timeout {m}")))? / SECOND;
            if !(1..=MAX_TIMEOUT).contains(&secs) {
                return Err(Fail::usage(cmd, format!("--timeout must be 1s to {MAX_TIMEOUT}s, not `{t}`")));
            }
            Some(secs)
        }
        None => None,
    };
    let rec = read::meeting(p)?;
    // The meeting's own, as it resolved the config file when it started;
    // twice that for an open question, which takes longer to answer.
    let open = if options.is_empty() { 2 } else { 1 };
    let meetings = rec.info.question_timeout.unwrap_or(DEFAULT_TIMEOUT) * open;
    let timeout = asked_for.unwrap_or(meetings).clamp(1, MAX_TIMEOUT);
    let id = which_task(cmd, given, "Give its ID first, e.g. pam task ask t01 …", &rec.dir)?;
    let q = question::ask(&rec.dir, id, text.trim(), &options, timeout)?;
    // A stopped meeting has nobody to answer.
    let end = if rec.info.status == "stopped" {
        question::end(&rec.dir, id, q, End::TimedOut)?
    } else {
        question::wait(&rec.dir, id, q, Duration::from_secs(timeout))?
    };
    // The option's number, if it was one, and the words: the option's, or the answer's.
    let chosen = match end {
        End::Answered { answer: Answer::Option(option), by, .. } => {
            options.get(option.wrapping_sub(1)).map(|text| (Some(option), text.clone(), by))
        }
        End::Answered { answer: Answer::Text(text), by, .. } => Some((None, text, by)),
        End::TimedOut | End::Dropped(_) => None,
    };
    if p.has("--json") {
        let chosen = chosen.map(|(option, text, by)| {
            let by = match by {
                By::Key => "key",
                By::Voice => "voice",
                By::Typed => "typed",
            };
            json!({ "option": option, "text": text, "by": by })
        });
        let v = json!({ "task": id, "question": q, "chosen": chosen });
        writeln!(out, "{}", serde_json::to_string_pretty(&v).unwrap()).unwrap();
    } else {
        writeln!(out, "{}", chosen.as_ref().map_or("timeout", |(_, text, _)| text.as_str())).unwrap();
    }
    Ok(())
}

/// The task ID given before the options, if any (`pam task update t02 …`).
/// A second word is a mistake: words of an `--action` or `--title` that
/// weren't quoted come apart into positionals, so the error says to `quote`.
fn given_id(cmd: &'static Command, p: &Parsed, quote: &str) -> Result<Option<TaskId>, Fail> {
    match &p.positionals[1..] {
        [] => Ok(None),
        [id] if id.parse::<TaskId>().is_ok() => Ok(id.parse::<TaskId>().ok()),
        [id, rest @ ..] => {
            let extra = if id.parse::<TaskId>().is_ok() { &rest[0] } else { id };
            Err(Fail::usage(
                cmd,
                format!(
                    "unexpected `{extra}`: only a task ID like t02 goes before the options; quote {quote}"
                ),
            ))
        }
    }
}

/// The task a command is for: `given`, else the meeting's running task,
/// the one the agent is working on (one runs at a time). It must
/// be one of the meeting's tasks. `how` says how to give one.
pub fn which_task(
    cmd: &'static Command,
    given: Option<TaskId>,
    how: &str,
    meeting: &Path,
) -> Result<TaskId, Fail> {
    let ids = task::ids(meeting)?;
    let listed = || match &ids[..] {
        [] => "this meeting has no tasks yet".to_string(),
        ids => format!("its tasks are {}", ids.iter().map(TaskId::to_string).collect::<Vec<_>>().join(", ")),
    };
    let id = match given {
        Some(id) => id,
        None => match task::running(meeting)? {
            Some(id) => id,
            None => {
                return Err(Fail::usage(
                    cmd,
                    format!("which task is this for? None is running. {how}; {}", listed()),
                ));
            }
        },
    };
    if !ids.contains(&id) {
        return Err(Fail::NotFound(format!("no task {id}: {}", listed())));
    }
    Ok(id)
}
