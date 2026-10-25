//! A stand-in for pi in the agent tests: `--version`, `auth check`, `-p` and
//! `--mode rpc`, as pi 1.1.0 behaves. Built as the `stub-pi`
//! binary, so the tests find it as `CARGO_BIN_EXE_stub-pi`.
//!
//! STUB_PI_VERSION (default 1.1.0) and STUB_PI_AUTH (default ready) change
//! what the checks say. STUB_PI_READY, when set, lists the models (separated
//! by spaces) that are ready; the others are not_ready. STUB_PI_OAUTH lists
//! the models whose login is OAuth; the others' is an API key.
//! STUB_PI_AUTH_DELAY makes each `auth check` take that many seconds. The
//! provider is the model's part before the first `/`. `--list-models
//! <model>` finds the model in pi's catalog, unless STUB_PI_UNCATALOGUED is
//! set: then it finds nothing, as for a model pi can't price.
//!
//! In print mode (`-p`, the scribe and the intent check) it answers with no
//! events, a summary and no request; with `--mode json`, as pi's JSON event
//! stream whose final assistant message used [`PRINT_USAGE`].
//!
//! In RPC mode it is one session, as pi is: it takes one prompt after
//! another until its stdin closes. It records each prompt in `stub-pi.jsonl`
//! beside its `--session` file, one JSON object per line: its arguments,
//! working directory, PAM_MEETING, pid, the task the prompt names (none for
//! a correction) and the prompt. It answers by running `pam say`, as the
//! agent would, with PI_SESSION_FILE set as pi sets it. Each assistant
//! message it ends used [`RPC_USAGE`], except a failed one, which used
//! nothing; `turn_end` and `agent_end` repeat it, as pi's do. The request
//! decides the rest:
//!   "slow"   waits up to 30 s first for `abort`, which it confirms as pi
//!            does, and exits as soon as its stdin closes;
//!   "busy"   says "Working on it." with `pam say`, then waits as "slow" does,
//!            and says "Still on the old request." as the abort lands;
//!   "silent" replies in text only, without `pam say`;
//!   "fail"   ends with a model error;
//!   "crash"  exits mid-run, as a pi that dies does;
//!   "compact" compacts the session first, which used [`COMPACTION_USAGE`];
//!   "ask"    runs `pam task ask` (countdown STUB_PI_ASK_TIMEOUT, default 2s)
//!            in its own process group, as pi runs a tool, and says the
//!            choice; `abort` or stdin closing kills it, as pi does. With
//!            "city" too, it asks an open question, "Which city are you in?",
//!            and says the weather there;
//!   "report" reports three times with `pam task update` before it answers:
//!            20% "searching", "reading source 2 of 3" and 100% "writing it
//!            up", STUB_PI_REPORT_GAP seconds apart (default 0.3);
//!   "result" writes `tasks/<id>/result.md`, with a source by URL and one
//!            by path, attaches it with `pam task result --title Sources`,
//!            then says its one-line verdict with `pam say`, as the
//!            cheat-sheet asks.

use std::io::{BufRead, Read, Write};
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::time::Duration;
use std::{env, fs, thread};

use serde_json::{Value, json};

/// What a print-mode run used: tokens, then cost in US dollars.
const PRINT_USAGE: (u64, u64, u64, u64, f64) = (1000, 100, 0, 0, 0.002);
/// What each assistant message in RPC mode used.
const RPC_USAGE: (u64, u64, u64, u64, f64) = (2000, 50, 1000, 500, 0.01);
/// What a compaction used.
const COMPACTION_USAGE: (u64, u64, u64, u64, f64) = (3000, 300, 0, 0, 0.005);

/// pi's `usage`, as an assistant message or a compaction carries it.
fn usage((input, output, cache_read, cache_write, cost): (u64, u64, u64, u64, f64)) -> Value {
    json!({
        "input": input, "output": output, "cacheRead": cache_read, "cacheWrite": cache_write,
        "totalTokens": input + output + cache_read + cache_write,
        "cost": {"input": 0, "output": 0, "cacheRead": 0, "cacheWrite": 0, "total": cost},
    })
}

fn main() {
    let args: Vec<String> = env::args().skip(1).collect();
    if args.iter().any(|a| a == "--version") {
        println!("{}", env::var("STUB_PI_VERSION").unwrap_or_else(|_| "1.1.0".into()));
        return;
    }
    if let Some(search) = option(&args, "--list-models") {
        list_models(&search);
        return;
    }
    if args.len() >= 2 && args[0] == "auth" && args[1] == "check" {
        std::process::exit(auth_check(&args));
    }
    if args.iter().any(|a| a == "-p") {
        print_mode(&args);
        return;
    }
    if option(&args, "--mode").as_deref() != Some("rpc") {
        die("only --mode rpc is stubbed");
    }
    let session = option(&args, "--session").unwrap_or_else(|| die("no --session"));
    Rpc::new(args, session).serve();
}

fn die(why: &str) -> ! {
    eprintln!("stub-pi: {why}");
    std::process::exit(1);
}

/// The value after `name`.
fn option(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}

fn auth_check(args: &[String]) -> i32 {
    let model = option(args, "--model").unwrap_or_default();
    if let Some(secs) = env::var("STUB_PI_AUTH_DELAY").ok().and_then(|d| d.parse::<f64>().ok()) {
        thread::sleep(Duration::from_secs_f64(secs));
    }
    let mut status = env::var("STUB_PI_AUTH").unwrap_or_else(|_| "ready".into());
    if let Ok(ready) = env::var("STUB_PI_READY") {
        status = if ready.split_whitespace().any(|m| m == model) { "ready" } else { "not_ready" }.into();
    }
    let provider = model.split_once('/').map_or("stub", |(p, _)| p);
    let oauth = env::var("STUB_PI_OAUTH").is_ok_and(|l| l.split_whitespace().any(|m| m == model));
    let login = if oauth { "oauth" } else { "api_key" };
    println!("{}", json!({"status": status, "provider": provider, "authType": login}));
    if status == "invalid" { 2 } else { 0 }
}

/// pi's table of the models matching `search`.
fn list_models(search: &str) {
    if env::var("STUB_PI_UNCATALOGUED").is_ok() {
        println!("No models matching \"{search}\"");
        return;
    }
    let (provider, id) = search.split_once('/').unwrap_or(("stub", search));
    println!("provider  model  context  max-out  thinking  images");
    println!("{provider}  {id}  200K  64K  yes  no");
}

fn print_mode(args: &[String]) {
    let mut prompt = String::new();
    let _ = std::io::stdin().read_to_string(&mut prompt);
    let reply = json!({"events": [], "summary": "Stub summary.", "kind": "none"}).to_string();
    if option(args, "--mode").as_deref() != Some("json") {
        println!("{reply}");
        return;
    }
    let message = json!({"role": "assistant", "content": [{"type": "text", "text": reply}],
                         "stopReason": "stop", "usage": usage(PRINT_USAGE)});
    for v in [
        json!({"type": "session", "version": 3, "id": "stub", "cwd": "."}),
        json!({"type": "agent_start"}),
        json!({"type": "message_end", "message": message}),
        json!({"type": "agent_end", "messages": [message]}),
        json!({"type": "agent_settled", "aborted": false}),
    ] {
        println!("{v}");
    }
}

/// What the stub hears while it works.
enum Heard {
    Command(Value),
    /// stdin closed: pi exits at once, even mid-run.
    Closed,
    /// The `pam task ask` it ran printed this and exited.
    Asked(String),
}

struct Rpc {
    args: Vec<String>,
    session: String,
    heard: Receiver<Heard>,
    tx: Sender<Heard>,
}

impl Rpc {
    fn new(args: Vec<String>, session: String) -> Rpc {
        let (tx, heard) = channel();
        let stdin = tx.clone();
        thread::spawn(move || {
            for line in std::io::stdin().lock().lines() {
                let Ok(line) = line else { break };
                let cmd = serde_json::from_str(&line).unwrap_or_else(|_| die("a command that isn't JSON"));
                if stdin.send(Heard::Command(cmd)).is_err() {
                    return;
                }
            }
            let _ = stdin.send(Heard::Closed);
        });
        Rpc { args, session, heard, tx }
    }

    fn emit(&self, v: Value) {
        let mut out = std::io::stdout().lock();
        let _ = writeln!(out, "{v}");
        let _ = out.flush();
    }

    /// The next thing heard, waiting at most `wait`.
    fn next(&self, wait: Option<Duration>) -> Option<Heard> {
        match wait {
            None => self.heard.recv().ok(),
            Some(w) => match self.heard.recv_timeout(w) {
                Ok(h) => Some(h),
                Err(RecvTimeoutError::Timeout) => None,
                Err(RecvTimeoutError::Disconnected) => Some(Heard::Closed),
            },
        }
    }

    fn serve(&self) {
        loop {
            match self.next(None) {
                None | Some(Heard::Closed) => return,
                Some(Heard::Asked(_)) => {}
                Some(Heard::Command(cmd)) => match cmd["type"].as_str() {
                    // Idle: pi answers and does nothing else.
                    Some("abort") => self.respond(&cmd, None),
                    Some("prompt") => self.prompt(&cmd),
                    _ => {}
                },
            }
        }
    }

    fn respond(&self, cmd: &Value, data: Option<Value>) {
        let mut r = json!({"id": cmd["id"], "type": "response", "command": cmd["type"], "success": true});
        if let Some(d) = data {
            r["data"] = d;
        }
        self.emit(r);
    }

    /// `pam say text`, as the agent runs it.
    fn say(&self, text: &str) {
        self.pam(&["say", text]);
    }

    /// Runs `pam` with `args`, as the agent runs it.
    fn pam(&self, args: &[&str]) {
        let ok = Command::new("pam")
            .args(args)
            .env("PI_SESSION_FILE", &self.session)
            .status()
            .is_ok_and(|s| s.success());
        if !ok {
            die(&format!("pam {} failed", args[..2].join(" ")));
        }
    }

    /// Reports three times with `pam task update`, each followed by a gap.
    /// Returns the `abort` command if one came in a gap; exits if stdin closes.
    fn report(&self) -> Option<Value> {
        let gap = env::var("STUB_PI_REPORT_GAP").ok().and_then(|g| g.parse::<f64>().ok()).unwrap_or(0.3);
        let reports: [&[&str]; 3] = [
            &["--progress", "20", "--action", "searching"],
            &["--action", "reading source 2 of 3"],
            &["--progress", "100", "--action", "writing it up"],
        ];
        for args in reports {
            self.pam(&[&["task", "update"], args].concat());
            match self.next(Some(Duration::from_secs_f64(gap))) {
                Some(Heard::Closed) => std::process::exit(0),
                Some(Heard::Command(next)) if next["type"] == "abort" => return Some(next),
                _ => {}
            }
        }
        None
    }

    /// Writes a result for task `id` (the running task's, for a correction)
    /// and attaches it, as the agent would.
    fn write_result(&self, id: Option<&str>) {
        let running = || {
            let r: Value = serde_json::from_str(&fs::read_to_string("agent/running.json").ok()?).ok()?;
            r["task"].as_str().map(str::to_string)
        };
        let Some(id) = id.map(str::to_string).or_else(running) else { die("no task for the result") };
        let file = format!("tasks/{id}/result.md");
        let text = "# Does shorter onboarding improve activation?\n\n\
                    Partly true. It helps when the steps cut are optional.\n\n\
                    Sources:\n\
                    1. [Onboarding study](https://example.com/onboarding-study)\n\
                    2. Team notes: [interviews](../../notes.md)\n";
        fs::write(&file, text).unwrap_or_else(|e| die(&e.to_string()));
        self.pam(&["task", "result", "--file", &file, "--title", "Sources"]);
    }

    fn record(&self, task: Option<&str>, prompt: &str) {
        let dir = std::path::Path::new(&self.session).parent().unwrap();
        fs::create_dir_all(dir).unwrap_or_else(|e| die(&e.to_string()));
        let line = json!({
            "args": self.args,
            "cwd": env::current_dir().unwrap(),
            "pam_meeting": env::var("PAM_MEETING").ok(),
            "pid": std::process::id(),
            "task": task,
            "prompt": prompt,
        });
        let mut f = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(dir.join("stub-pi.jsonl"))
            .unwrap_or_else(|e| die(&e.to_string()));
        // In one write, so a test reading the file never sees half a line.
        f.write_all(format!("{line}\n").as_bytes()).unwrap_or_else(|e| die(&e.to_string()));
    }

    /// Ends the run with `message`. pi repeats it in `turn_end` and `agent_end`.
    fn ended(&self, message: Value, aborted: bool) {
        self.emit(json!({"type": "message_end", "message": message}));
        self.emit(json!({"type": "turn_end", "message": message, "toolResults": []}));
        self.emit(json!({"type": "agent_end", "messages": [message], "willRetry": false}));
        self.emit(json!({"type": "agent_settled", "aborted": aborted}));
    }

    /// Ends the run as pi does on `abort`: `agent_settled` before the abort's
    /// response. The cut-short message keeps what it used so far.
    fn aborted(&self, abort: &Value) {
        let message =
            json!({"role": "assistant", "content": [], "stopReason": "aborted", "usage": usage(RPC_USAGE)});
        self.ended(message, true);
        self.respond(abort, None);
    }

    fn prompt(&self, cmd: &Value) {
        let message = cmd["message"].as_str().unwrap_or_default();
        let (task, request) = parse(message);
        self.record(task.as_deref(), message);
        self.respond(cmd, Some(json!({"disposition": "started"})));
        self.emit(json!({"type": "agent_start"}));
        if request.contains("crash") {
            die("crashed");
        }
        if request.contains("compact") {
            self.emit(json!({"type": "compaction_start", "reason": "threshold"}));
            self.emit(
                json!({"type": "compaction_end", "reason": "threshold", "aborted": false, "willRetry": false,
                             "result": {"summary": "Stub summary.", "tokensBefore": 150000,
                                        "usage": usage(COMPACTION_USAGE)}}),
            );
        }
        if request.contains("busy") {
            self.say("Working on it.");
        }
        if request.split_whitespace().any(|w| w == "report")
            && let Some(abort) = self.report()
        {
            return self.aborted(&abort);
        }
        let result = request.split_whitespace().any(|w| w == "result");
        if result {
            self.write_result(task.as_deref());
        }
        let asking = request.split_whitespace().any(|w| w == "ask");
        if asking && let Some(abort) = self.ask_the_room(request.contains("city")) {
            return self.aborted(&abort);
        }
        if request.contains("slow") || request.contains("busy") {
            match self.next(Some(Duration::from_secs(30))) {
                Some(Heard::Closed) => std::process::exit(0),
                Some(Heard::Command(next)) if next["type"] == "abort" => {
                    if request.contains("busy") {
                        // A `pam say` still in flight when the abort lands.
                        self.say("Still on the old request.");
                    }
                    return self.aborted(&next);
                }
                _ => {}
            }
        }
        let mut reply = json!({"role": "assistant", "content": [{"type": "text", "text": "Done."}],
                               "stopReason": "stop", "usage": usage(RPC_USAGE)});
        if request.contains("fail") {
            reply = json!({"role": "assistant", "content": [], "stopReason": "error",
                           "errorMessage": "stub model failed", "usage": usage((0, 0, 0, 0, 0.0))});
        } else if asking {
            // Said already.
        } else if result {
            self.say("Partly true: shorter onboarding helps when the steps cut are optional.");
        } else if request.contains("silent") {
            reply["content"][0]["text"] = json!(format!("Reply in text to: {request}"));
        } else {
            self.say(&format!("Stub answer to: {request}"));
        }
        self.ended(reply, false);
    }

    /// Runs `pam task ask` as pi runs a bash tool, then says what the room
    /// chose. Returns the `abort` command if one came first; exits if stdin closes.
    fn ask_the_room(&self, open: bool) -> Option<Value> {
        let timeout = env::var("STUB_PI_ASK_TIMEOUT").unwrap_or_else(|_| "2s".into());
        let question: &[&str] = match open {
            true => &["Which city are you in?"],
            false => &["Which customer is the email for?", "--option", "Acme", "--option", "Globex"],
        };
        let mut child = Command::new("pam")
            .args(["task", "ask"])
            .args(question)
            .args(["--timeout", &timeout])
            .env("PI_SESSION_FILE", &self.session)
            .stdout(Stdio::piped())
            .process_group(0)
            .spawn()
            .unwrap_or_else(|e| die(&e.to_string()));
        let pid = child.id() as libc::pid_t;
        let mut out = child.stdout.take().unwrap();
        let tx = self.tx.clone();
        thread::spawn(move || {
            let mut answer = String::new();
            let _ = out.read_to_string(&mut answer);
            let _ = tx.send(Heard::Asked(answer));
        });
        // SAFETY: signals the process group of the asker this stub started.
        let kill = || unsafe { libc::kill(-pid, libc::SIGTERM) };
        let answer = loop {
            match self.next(None) {
                Some(Heard::Asked(answer)) => break answer,
                None | Some(Heard::Closed) => {
                    kill();
                    std::process::exit(0);
                }
                Some(Heard::Command(cmd)) if cmd["type"] == "abort" => {
                    kill();
                    let _ = child.wait();
                    return Some(cmd);
                }
                Some(Heard::Command(_)) => {}
            }
        };
        let _ = child.wait();
        let answer = answer.trim();
        // Killed before it printed anything: a model takes a while to go on.
        if answer.is_empty() {
            thread::sleep(Duration::from_millis(1500));
        }
        self.say(&match (answer, open) {
            ("timeout", false) => "No answer in time, so I assumed Acme.".to_string(),
            ("timeout", true) => "No answer in time, so I assumed Wellington: sunny.".to_string(),
            ("", _) => "The question went unanswered.".to_string(),
            (chosen, false) => format!("Drafting the email to {chosen}."),
            (city, true) => format!("Tomorrow morning in {city}: sunny."),
        });
        None
    }
}

/// The task a prompt names and its request: `Request t01 (…):\n<request>\n\n`,
/// or a correction's `Correction from …:\n<text>\n\n`, which names none.
/// Either may follow a catch-up on what was said.
fn parse(message: &str) -> (Option<String>, String) {
    let body = |rest: &str| rest.split("\n\n").next().unwrap_or_default().to_string();
    if let Some(i) = message.find("Request t") {
        let rest = &message[i + "Request ".len()..];
        let task = rest.split_whitespace().next().unwrap_or_default().to_string();
        if let Some(j) = rest.find("):\n") {
            return (Some(task), body(&rest[j + 3..]));
        }
    }
    let correction = message.find("Correction from ").map(|i| &message[i + "Correction from ".len()..]);
    match correction.and_then(|m| m.split_once(":\n")) {
        Some((_, rest)) => (None, body(rest)),
        None => (None, String::new()),
    }
}
