//! Arguments, help and errors for every `pam` command.
//!
//! Hand-written rather than clap, so that every usage error can say what went
//! wrong, show that command's full usage and give worked examples.

use std::fmt::Write as _;
use std::io::Write as _;
use std::process::ExitCode;

use serde_json::{Value, json};

use crate::{listen, models, note, read, say, task};

/// One option or positional argument, as the usage block lists it.
pub struct Arg {
    pub name: &'static str,
    /// Options only: whether it takes a value (`--last 20`) or is a switch.
    pub value: bool,
    pub help: &'static str,
}

pub struct Command {
    pub name: &'static str,
    pub summary: &'static str,
    pub usage: &'static str,
    pub args: &'static [Arg],
    pub examples: &'static [(&'static str, &'static str)],
}

const fn pos(name: &'static str, help: &'static str) -> Arg {
    Arg { name, value: false, help }
}
const fn opt(name: &'static str, help: &'static str) -> Arg {
    Arg { name, value: true, help }
}
const fn switch(name: &'static str, help: &'static str) -> Arg {
    Arg { name, value: false, help }
}

const JSON: Arg = switch("--json", "stable machine-readable output");
const MEETING: Arg =
    opt("--meeting", "the meeting folder to read; default $PAM_MEETING, else the latest meeting");
const SPEAKER: Arg = opt("--speaker", "only this speaker: a name (Alice) or ID (spk_3)");

pub const COMMANDS: &[Command] = &[
    Command {
        name: "listen",
        summary: "start a meeting and show the terminal dashboard",
        usage: "pam listen [--input mic | --input <file.wav> [--realtime] | --input <script.fake.jsonl> [--realtime]] [--title <t>] [--name <assistant-name>] [--attendee <name>]... [--scribe <model|off>] [--intent <model|local|off>] [--agent <model|off>]",
        args: &[
            opt(
                "--input",
                "what to listen to: mic, the default (the config file's `mic`, or the system's default input), a 16 kHz recording (*.wav), or a fake script (*.fake.jsonl). The mic and recordings go through the speech backends in the config file",
            ),
            switch(
                "--realtime",
                "replay a recording or script at real-time speed instead of as fast as possible",
            ),
            opt("--title", "the meeting's title"),
            opt(
                "--name",
                "the assistant's name, which is also its wake word (default: `name` in the config file, else Pam)",
            ),
            opt(
                "--attendee",
                "someone expected at the meeting; repeat for each. Helps the transcriber spell their name",
            ),
            opt(
                "--scribe",
                "the model that takes the notes, through pi, as provider/id[:thinking], or off (default: `scribe_model` in the config file, else detected)",
            ),
            opt(
                "--intent",
                "the model that checks spoken requests: one through pi, as provider/id[:thinking]; local, a small model on this Mac, downloaded once (2.7 GB); or off (default: `intent_model` in the config file, else the scribe's model)",
            ),
            opt(
                "--agent",
                "the model that answers requests, through pi, as provider/id[:thinking], or off (default: `agent_model` in the config file, else detected)",
            ),
        ],
        examples: &[
            ("pam listen --title \"Stand-up\"", "listen to the room through the default mic"),
            (
                "pam listen --input standup.wav --realtime --title \"Stand-up\"",
                "transcribe a recording live, as if in the room",
            ),
            (
                "pam listen --scribe off --intent local",
                "no notes, and spoken requests checked on this Mac: only what the agent is asked leaves it",
            ),
        ],
    },
    Command {
        name: "status",
        summary: "the meeting at a glance: elapsed time, people, lines, open items",
        usage: "pam status [--json] [--meeting <folder>]",
        args: &[JSON, MEETING],
        examples: &[
            ("pam status", "the most recent meeting"),
            ("pam status --meeting ~/Pam/meetings/2026-10-10-1402-sync", "a given meeting"),
        ],
    },
    Command {
        name: "context",
        summary: "what's going on: people, open items, events and the last 3 minutes word for word",
        usage: "pam context [--json] [--meeting <folder>]",
        args: &[JSON, MEETING],
        examples: &[
            ("pam context", "read this first: the meeting's context pack, context.md"),
            ("pam context --json", "the same text, with its size in estimated tokens"),
        ],
    },
    Command {
        name: "people",
        summary: "speakers: ID, name, first heard, talk time, line count",
        usage: "pam people [--json] [--meeting <folder>]",
        args: &[JSON, MEETING],
        examples: &[("pam people", "who has spoken so far"), ("pam people --json", "the same, as JSON")],
    },
    Command {
        name: "transcript",
        summary: "lines with IDs, names applied",
        usage: "pam transcript [--last <n> | --since <dur> | --from <time> --to <time>] [--speaker <name|id>] [--json] [--meeting <folder>]",
        args: &[
            opt("--last", "only the last <n> lines"),
            opt("--since", "only lines from the last <dur> of the meeting, e.g. 5m or 90s"),
            opt("--from", "only lines from this meeting time on, as mm:ss or h:mm:ss"),
            opt("--to", "only lines before this meeting time"),
            SPEAKER,
            JSON,
            MEETING,
        ],
        examples: &[
            ("pam transcript --last 20", "the last 20 lines"),
            ("pam transcript --since 5m --speaker Alice", "what Alice said in the last 5 minutes"),
            ("pam transcript --from 03:10 --to 04:00", "a stretch of the meeting"),
        ],
    },
    Command {
        name: "get",
        summary: "exact lines, events, speakers or tasks by ID",
        usage: "pam get <ref>... [--json] [--meeting <folder>]",
        args: &[
            pos(
                "<ref>",
                "a line (u0412), line range (u0400..u0412), event (e037), task (t02) or speaker (spk_3); give several to get them all",
            ),
            JSON,
            MEETING,
        ],
        examples: &[
            ("pam get u0412", "one line, word for word"),
            ("pam get u0400..u0412", "a run of lines"),
            ("pam get e037 spk_2", "an event and a speaker"),
        ],
    },
    Command {
        name: "around",
        summary: "lines surrounding a line or event",
        usage: "pam around <ref> [--before <dur>] [--after <dur>] [--json] [--meeting <folder>]",
        args: &[
            pos("<ref>", "a line (u0412), line range (u0400..u0412) or event (e037)"),
            opt("--before", "how much earlier speech to include (default 60s)"),
            opt("--after", "how much later speech to include (default 30s)"),
            JSON,
            MEETING,
        ],
        examples: &[
            ("pam around e037", "what led up to the prototype request"),
            ("pam around u0412 --before 2m", "two minutes before line u0412"),
            ("pam around e052 --after 0s", "only what came before decision e052"),
        ],
    },
    Command {
        name: "search",
        summary: "keyword search over lines and events",
        usage: "pam search <query> [--speaker <name|id>] [--type <types>] [--limit <n>] [--json] [--meeting <folder>]",
        args: &[
            pos(
                "<query>",
                "words that must all appear, ignoring case; a word also matches inside longer words",
            ),
            SPEAKER,
            opt(
                "--type",
                "only these kinds, comma-separated: an event type (decision, action, question, …) or line",
            ),
            opt("--limit", "at most <n> results, oldest first (default 20)"),
            JSON,
            MEETING,
        ],
        examples: &[
            ("pam search \"single page\"", "where the single-page flow came up"),
            ("pam search invite --type decision", "decisions about invites"),
            ("pam search mobile --speaker Marcus --limit 5", "Marcus on mobile, first five"),
        ],
    },
    Command {
        name: "events",
        summary: "the scribe's events: topics, points, decisions, actions, questions, notes…",
        usage: "pam events [--type <types>] [--since <dur>] [--json] [--meeting <folder>]",
        args: &[
            opt("--type", "only these types, comma-separated, e.g. decision,action"),
            opt("--since", "only events from the last <dur> of the meeting, e.g. 10m"),
            JSON,
            MEETING,
        ],
        examples: &[
            ("pam events --type decision,action", "what was decided and who does what"),
            ("pam events --since 10m", "the last ten minutes"),
            ("pam events --type question --json", "questions, as JSON"),
        ],
    },
    Command {
        name: "note",
        summary: "add a note to the meeting's notes",
        usage: "pam note <text> [--json] [--meeting <folder>]",
        args: &[pos("<text>", "what to note, in a sentence"), JSON, MEETING],
        examples: &[
            ("pam note \"Prototype review moved to Friday\"", "add a note"),
            ("pam note Ship the beta on Monday --json", "the saved event, as JSON"),
        ],
    },
    Command {
        name: "say",
        summary: "show a message on the stage, for a task",
        usage: "pam say <text> [--task <id>] [--json] [--meeting <folder>]",
        args: &[
            pos("<text>", "what to show: the answer, in a few plain sentences"),
            opt("--task", "the task it is for, e.g. t01; default the meeting's running task"),
            JSON,
            MEETING,
        ],
        examples: &[
            ("pam say \"We settled on 40 a seat (Marcus, 01:20).\"", "answer your own task"),
            ("pam say --task t02 \"Still looking; two sources so far.\"", "a message for a given task"),
        ],
    },
    Command {
        name: "task",
        summary: "for the agent's own task: report how far along it is, attach its result, or ask the room a question and wait for the answer",
        usage: "pam task update [<id>] [--progress <0-100>] [--action <text>] [--json] [--meeting <folder>]\n       pam task result [<id>] (--file <path> | --url <url>) [--title <t>] [--json] [--meeting <folder>]\n       pam task ask [<id>] <question> [--option <a> --option <b>...] [--timeout <dur>] [--json] [--meeting <folder>]",
        args: &[
            pos(
                "update",
                "show how far along the task is on its chip and card: --progress, --action or both. Only a running task reports; the core sets its status",
            ),
            pos(
                "result",
                "attach what the task produced, a file or a page, replacing the one before. It never takes the stage: the task's chip turns ready, and the room opens it",
            ),
            pos(
                "ask",
                "show <question> on the stage with numbered options and a countdown, then print the option the room chose, or `timeout`; with no options it is an open question, and prints what the room answered",
            ),
            pos("<id>", "the task, e.g. t02; default the meeting's running task"),
            opt("--progress", "update: how far along it is, 0 to 100 (percent)"),
            opt(
                "--action",
                "update: what it is doing now, in one short line, e.g. \"reading source 2 of 5\" (a count shows as 2/5)",
            ),
            opt(
                "--file",
                "result: a file the task wrote, e.g. tasks/t02/result.md; kept relative to the meeting folder when it is in it",
            ),
            opt("--url", "result: a web page, an http:// or https:// address"),
            opt("--title", "result: what to call it (default the file's name, or the address)"),
            pos("<question>", "ask: what to ask, in one short sentence"),
            opt(
                "--option",
                "ask: an answer the room can choose; give 2 to 9, best guess first, or none for an open question",
            ),
            opt(
                "--timeout",
                "ask: how long the room has, e.g. 20s (default the meeting's, from `question_timeout` in the config file, else 15s; twice that for an open question; at most 60s)",
            ),
            JSON,
            MEETING,
        ],
        examples: &[
            (
                "pam task update --progress 40 --action \"reading source 2 of 5\"",
                "report how far along your own task is",
            ),
            (
                "pam task result --file tasks/t02/result.md --title \"Sources\"",
                "attach a longer answer the room can open",
            ),
            (
                "pam task ask t02 \"Which customer is the email for?\" --option Acme --option Globex",
                "ask the room; prints Acme, Globex or timeout",
            ),
        ],
    },
    Command {
        name: "models",
        summary: "the speech and intent models on this Mac: list, pull, remove them, or print where they are",
        usage: "pam models list\n       pam models pull [<name>...]\n       pam models rm <name>...\n       pam models path",
        args: &[
            pos(
                "list",
                "every model the store knows: its size, whether it is pulled, and `used` for those pam listen loads with the config file's choices",
            ),
            pos(
                "pull",
                "download models into the store, checked against their pinned SHA-256; with no name, what pam listen needs: `asr`, `diarizer`, and the local intent model when `intent_model` is local. pam listen never downloads: a missing model stops it before the meeting",
            ),
            pos(
                "rm",
                "remove models from the store. The store is shared with dictate, which may still use them",
            ),
            pos(
                "path",
                "print the store's folder: $PAM_MODELS, else ~/Library/Application Support/pam/models",
            ),
            pos("<name>", "a model, as `pam models list` names it, e.g. nemotron-en"),
        ],
        examples: &[
            ("pam models pull", "get what pam listen needs, once"),
            ("pam models list", "what is on this Mac, and what pam uses"),
            ("pam models rm qwen3.5-4b", "free the local intent model's 2.7 GB"),
        ],
    },
    Command {
        name: "help",
        summary: "this manual, or one command's usage and examples",
        usage: "pam help [<command>] [--json]",
        args: &[pos("<command>", "a command name, e.g. transcript"), JSON],
        examples: &[
            ("pam help", "every command"),
            ("pam help around", "one command"),
            ("pam help --json", "every command's usage, arguments and examples, as JSON"),
        ],
    },
];

/// The general manual, shown by `pam help` and with top-level usage errors.
const OVERVIEW: &str = "\
which meeting: --meeting <folder>, else $PAM_MEETING, else the most recent
  meeting under $PAM_MEETINGS_ROOT (default ~/Pam/meetings)
IDs: u0412 line, u0400..u0412 line range, e037 event, t02 task, spk_3 speaker
times: meeting time as mm:ss or h:mm:ss; durations like 90s, 5m, 1h or 1m30s
output: short text with IDs; --json gives machine-readable output
exit codes: 0 ok, 1 usage error, 2 not found, 3 no meeting, 4 failed";

/// The top-level usage line.
const USAGE: &str = "pam <command> [<args>] [--meeting <folder>] [--json]";

const EXIT_CODES: &[(u8, &str)] =
    &[(0, "ok"), (1, "usage error"), (2, "not found"), (3, "no meeting"), (4, "failed")];

const OVERVIEW_EXAMPLES: &[(&str, &str)] = &[
    ("pam transcript --last 20", "the last 20 lines"),
    ("pam around u0412 --before 2m", "what led up to line u0412"),
    ("pam search \"single page\"", "where something was discussed"),
    ("pam help around", "one command's usage and examples"),
];

/// Why a command failed. Each kind has its own exit code.
pub enum Fail {
    /// The command line is wrong. `None` for a missing or unknown command.
    Usage(Option<&'static Command>, String),
    NotFound(String),
    NoMeeting(String),
    Failed(anyhow::Error),
}

impl Fail {
    pub fn usage(cmd: &'static Command, msg: impl Into<String>) -> Fail {
        Fail::Usage(Some(cmd), msg.into())
    }

    fn code(&self) -> u8 {
        match self {
            Fail::Usage(..) => 1,
            Fail::NotFound(_) => 2,
            Fail::NoMeeting(_) => 3,
            Fail::Failed(_) => 4,
        }
    }

    pub fn message(&self) -> String {
        match self {
            Fail::Usage(Some(cmd), msg) => format!("error: {msg}\n\n{}", manual(cmd)),
            Fail::Usage(None, msg) => format!("error: {msg}\n\n{}", overview()),
            Fail::NotFound(msg) | Fail::NoMeeting(msg) => format!("error: {msg}"),
            Fail::Failed(e) => format!("error: {e:#}"),
        }
    }
}

impl From<anyhow::Error> for Fail {
    fn from(e: anyhow::Error) -> Fail {
        Fail::Failed(e)
    }
}

/// A command line split into positionals and options.
pub struct Parsed {
    pub positionals: Vec<String>,
    options: Vec<(&'static str, Option<String>)>,
}

impl Parsed {
    pub fn has(&self, name: &str) -> bool {
        self.options.iter().any(|(n, _)| *n == name)
    }

    /// Every value given for a repeatable option, in order.
    pub fn values(&self, name: &str) -> Vec<String> {
        self.options.iter().filter(|(n, _)| *n == name).filter_map(|(_, v)| v.clone()).collect()
    }

    /// The option's value; the last one wins if it is given twice.
    pub fn value(&self, name: &str) -> Option<&str> {
        self.options.iter().rev().find(|(n, _)| *n == name).and_then(|(_, v)| v.as_deref())
    }
}

/// Runs `pam` with `args` (without the program name) and returns its exit code.
pub fn run(args: Vec<String>) -> ExitCode {
    let mut out = String::new();
    let result = dispatch(args, &mut out);
    let result = emit(&out, &mut std::io::stdout().lock(), result);
    match result {
        Ok(()) => ExitCode::SUCCESS,
        Err(fail) => {
            let mut msg = fail.message();
            if !msg.ends_with('\n') {
                msg.push('\n');
            }
            let _ = std::io::stderr().lock().write_all(msg.as_bytes());
            ExitCode::from(fail.code())
        }
    }
}

/// Writes the output. Failing to write it fails the command, unless the
/// reader went away (`pam transcript | head`), which is not worth reporting.
fn emit(out: &str, w: &mut impl std::io::Write, result: Result<(), Fail>) -> Result<(), Fail> {
    match w.write_all(out.as_bytes()).and_then(|()| w.flush()) {
        Err(e) if e.kind() != std::io::ErrorKind::BrokenPipe && result.is_ok() => {
            Err(Fail::Failed(anyhow::Error::new(e).context("writing the output")))
        }
        _ => result,
    }
}

fn dispatch(args: Vec<String>, out: &mut String) -> Result<(), Fail> {
    let Some(name) = args.first().cloned() else {
        return Err(Fail::Usage(None, "missing command: what should pam do?".into()));
    };
    match name.as_str() {
        "-h" | "--help" => {
            out.push_str(&overview());
            return Ok(());
        }
        "-V" | "--version" => {
            writeln!(out, "pam {}", env!("CARGO_PKG_VERSION")).unwrap();
            return Ok(());
        }
        // `pam --meeting X status`: options belong to commands.
        n if n.starts_with('-') => {
            return Err(Fail::Usage(
                None,
                format!("`{n}` before the command: options go after it, e.g. pam status --meeting <folder>"),
            ));
        }
        _ => {}
    }
    let Some(cmd) = find(&name) else {
        return Err(Fail::Usage(None, format!("unknown command `{name}`")));
    };
    let Some(parsed) = parse(cmd, &args[1..])? else {
        writeln!(out, "pam {}: {}\n\n{}", cmd.name, cmd.summary, manual(cmd)).unwrap();
        return Ok(());
    };
    match cmd.name {
        "listen" => listen_command(cmd, parsed),
        "models" => models::run(cmd, parsed, out),
        "help" => help(cmd, parsed, out),
        "note" => note::run(cmd, parsed, out),
        "say" => say::run(cmd, parsed, out),
        "task" => task::run(cmd, parsed, out),
        _ => read::run(cmd, parsed, out),
    }
}

fn find(name: &str) -> Option<&'static Command> {
    COMMANDS.iter().find(|c| c.name == name)
}

/// Splits `args` by `cmd`'s options. `None` means help was asked for.
pub fn parse(cmd: &'static Command, args: &[String]) -> Result<Option<Parsed>, Fail> {
    let mut parsed = Parsed { positionals: Vec::new(), options: Vec::new() };
    let mut args = args.iter();
    while let Some(arg) = args.next() {
        if arg == "--" {
            parsed.positionals.extend(args.cloned());
            break;
        }
        if arg == "-h" || arg == "--help" {
            return Ok(None);
        }
        if !arg.starts_with('-') || arg == "-" {
            parsed.positionals.push(arg.clone());
            continue;
        }
        let (name, inline) = match arg.split_once('=') {
            Some((n, v)) => (n, Some(v.to_string())),
            None => (arg.as_str(), None),
        };
        let Some(spec) = cmd.args.iter().find(|a| a.name == name && a.name.starts_with("--")) else {
            let known: Vec<&str> =
                cmd.args.iter().filter(|a| a.name.starts_with("--")).map(|a| a.name).collect();
            let msg = if known.is_empty() {
                format!("unknown option {name}: `pam {}` takes no options", cmd.name)
            } else {
                format!("unknown option {name} for `pam {}`; it takes {}", cmd.name, known.join(", "))
            };
            return Err(Fail::usage(cmd, msg));
        };
        let value = match (spec.value, inline) {
            (true, Some(v)) => Some(v),
            (true, None) => match args.next() {
                Some(v) if !v.starts_with("--") => Some(v.clone()),
                _ => return Err(Fail::usage(cmd, format!("{name} needs a value"))),
            },
            (false, Some(_)) => {
                return Err(Fail::usage(cmd, format!("{name} is a switch and takes no value")));
            }
            (false, None) => None,
        };
        parsed.options.push((spec.name, value));
    }
    Ok(Some(parsed))
}

fn help(cmd: &'static Command, parsed: Parsed, out: &mut String) -> Result<(), Fail> {
    let json = parsed.has("--json");
    match &parsed.positionals[..] {
        [] if json => {
            let v = json!({
                "usage": USAGE,
                "commands": COMMANDS.iter().map(command_json).collect::<Vec<_>>(),
                "exit_codes": EXIT_CODES.iter().map(|(c, m)| (c.to_string(), Value::from(*m))).collect::<serde_json::Map<_, _>>(),
                "examples": examples_json(OVERVIEW_EXAMPLES),
            });
            writeln!(out, "{}", serde_json::to_string_pretty(&v).unwrap()).unwrap();
        }
        [] => out.push_str(&overview()),
        [name] => {
            let Some(c) = find(name) else {
                return Err(Fail::usage(cmd, format!("no command `{name}`; run `pam help` to list them")));
            };
            if json {
                writeln!(out, "{}", serde_json::to_string_pretty(&command_json(c)).unwrap()).unwrap();
            } else {
                writeln!(out, "pam {}: {}\n\n{}", c.name, c.summary, manual(c)).unwrap();
            }
        }
        more => {
            return Err(Fail::usage(cmd, format!("help takes one command name; got {}", more.join(" "))));
        }
    }
    Ok(())
}

fn command_json(c: &Command) -> Value {
    json!({
        "name": c.name,
        "summary": c.summary,
        "usage": c.usage,
        "args": c.args.iter().map(|a| json!({ "name": a.name, "value": a.value, "help": a.help })).collect::<Vec<_>>(),
        "examples": examples_json(c.examples),
    })
}

fn examples_json(list: &[(&str, &str)]) -> Value {
    list.iter().map(|(e, why)| json!({ "command": e, "why": why })).collect()
}

fn listen_command(cmd: &'static Command, p: Parsed) -> Result<(), Fail> {
    if let Some(extra) = p.positionals.first() {
        return Err(Fail::usage(cmd, format!("unexpected `{extra}`: pass the input with --input")));
    }
    let input = p.value("--input").unwrap_or("mic");
    if input == "mic" && p.has("--realtime") {
        return Err(Fail::usage(
            cmd,
            "--realtime is for replaying a recording or script; the mic is always live",
        ));
    }
    listen::run(listen::Options {
        input: input.into(),
        realtime: p.has("--realtime"),
        title: p.value("--title").unwrap_or_default().to_string(),
        name: p.value("--name").map(str::to_string),
        attendees: p.values("--attendee"),
        scribe: p.value("--scribe").map(str::to_string),
        intent: p.value("--intent").map(str::to_string),
        agent: p.value("--agent").map(str::to_string),
    })?;
    Ok(())
}

/// A command's usage, arguments and examples: what every usage error shows.
pub fn manual(cmd: &Command) -> String {
    let mut s = format!("usage: {}\n", cmd.usage);
    let width = cmd.args.iter().map(|a| a.name.len()).max().unwrap_or(0).max(10) + 2;
    for a in cmd.args {
        writeln!(s, "  {:width$}{}", a.name, a.help).unwrap();
    }
    s.push('\n');
    s.push_str(&examples(cmd.examples));
    s
}

fn overview() -> String {
    let mut s = format!(
        "pam: an in-room meeting assistant. `pam listen` runs a meeting; the other\ncommands read it.\n\n\
         usage: {USAGE}\n\ncommands:\n"
    );
    for c in COMMANDS {
        writeln!(s, "  {:12}{}", c.name, c.summary).unwrap();
    }
    write!(s, "\n{OVERVIEW}\n\n{}", examples(OVERVIEW_EXAMPLES)).unwrap();
    s
}

fn examples(list: &[(&str, &str)]) -> String {
    let width = list.iter().map(|(e, _)| e.len()).max().unwrap_or(0) + 2;
    let mut s = String::from("examples:\n");
    for (e, why) in list {
        writeln!(s, "  {e:width$} # {why}").unwrap();
    }
    s
}

#[cfg(test)]
mod tests {
    use super::*;

    fn args(s: &str) -> Vec<String> {
        s.split_whitespace().map(str::to_string).collect()
    }

    #[test]
    fn every_command_has_two_or_three_examples() {
        for c in COMMANDS {
            assert!((2..=3).contains(&c.examples.len()), "{}", c.name);
            assert!(c.examples.iter().all(|(e, _)| e.starts_with(&format!("pam {}", c.name))), "{}", c.name);
        }
    }

    #[test]
    fn output_that_cannot_be_written_fails_unless_the_pipe_closed() {
        struct Failing(std::io::ErrorKind);
        impl std::io::Write for Failing {
            fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
                Err(self.0.into())
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        let full = emit("u0001 …\n", &mut Failing(std::io::ErrorKind::StorageFull), Ok(()));
        assert!(matches!(&full, Err(f @ Fail::Failed(_)) if f.code() == 4));
        assert!(emit("u0001 …\n", &mut Failing(std::io::ErrorKind::BrokenPipe), Ok(())).is_ok());
        let not_found =
            emit("u0001 …\n", &mut Failing(std::io::ErrorKind::StorageFull), Err(Fail::NotFound("x".into())));
        assert!(matches!(not_found, Err(Fail::NotFound(_))), "the command's own failure comes first");
    }

    #[test]
    fn the_meeting_is_an_option_like_any_other() {
        let status = find("status").unwrap();
        let p = parse(status, &args("--meeting=/m --json")).ok().flatten().unwrap();
        assert_eq!(p.value("--meeting"), Some("/m"));
        assert!(
            matches!(parse(status, &args("--meeting --json")), Err(Fail::Usage(Some(_), m)) if m == "--meeting needs a value")
        );
        let mut out = String::new();
        let before = dispatch(args("--meeting /m status"), &mut out);
        assert!(matches!(before, Err(Fail::Usage(None, m)) if m.contains("options go after it")));
        assert!(
            matches!(dispatch(args("listen --meeting /m"), &mut out), Err(Fail::Usage(Some(_), m)) if m.starts_with("unknown option --meeting"))
        );
    }

    #[test]
    fn parses_options_and_positionals() {
        let cmd = find("search").unwrap();
        let p = parse(cmd, &args("single --limit=5 page --type decision --json")).ok().flatten().unwrap();
        assert_eq!(p.positionals, ["single", "page"]);
        assert_eq!(
            (p.value("--limit"), p.value("--type"), p.has("--json")),
            (Some("5"), Some("decision"), true)
        );
        assert!(
            matches!(parse(cmd, &args("x --limit")), Err(Fail::Usage(Some(_), m)) if m == "--limit needs a value")
        );
        assert!(matches!(parse(cmd, &args("x --json=1")), Err(Fail::Usage(..))));
        assert!(
            matches!(parse(cmd, &args("x --bogus")), Err(Fail::Usage(Some(_), m)) if m.contains("--speaker"))
        );
        assert!(matches!(parse(cmd, &args("x -h")), Ok(None)));
        let p = parse(cmd, &args("-- --json")).ok().flatten().unwrap();
        assert_eq!(p.positionals, ["--json"]);
    }
}
