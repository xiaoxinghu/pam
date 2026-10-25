//! The agent: pi in RPC mode, one session for the whole meeting, so a
//! follow-up sees what Pam was asked and answered before.
//!
//! The meeting checks the agent's model and pi before it starts ([`crate::models`]),
//! so an agent that can't work says why on the stage and in `pam status`
//! instead of failing on the first request. The first request starts pi
//! isolated from the user's pi setup but with their login, in the
//! meeting folder with `PAM_MEETING` set, its session in `agent/session.jsonl`.
//! Each task is then a prompt in that session, one at a time, until the
//! meeting stops. If pi goes away, the next task starts a new session.

use std::collections::VecDeque;
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use crossbeam_channel::Receiver;
use serde_json::{Value, json};

use crate::Ts;
use crate::brief::{self, Coming, Mark};
use crate::clock::mmss;
use crate::event::TaskId;
use crate::llm::text_of;
use crate::models::AgentInfo;
use crate::question;
use crate::record::MeetingRecord;
use crate::rpc::{Next, Rpc};
use crate::task;
use crate::usage::Call;

/// How often a running task looks at whether the meeting is stopping.
const STOP_POLL: Duration = Duration::from_millis(100);
/// How often a running task looks for a cancel or a correction.
const INBOX_POLL: Duration = Duration::from_millis(20);
/// How long pi gets to confirm an abort before it is stopped anyway.
const ABORT_WAIT: Duration = Duration::from_secs(2);

/// How the meeting reaches its agent.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// The agent's model and pi, as checked before the meeting.
    pub info: AgentInfo,
    /// The `pam` executable the agent runs. Its folder goes first on the
    /// agent's `PATH`, so the agent reaches this meeting's `pam` even when it
    /// isn't installed.
    pub pam: Option<PathBuf>,
}

/// The pam-cli cheat-sheet, appended to pi's own system prompt. Appending
/// keeps pi's rules for its tools, which `--system-prompt` would drop.
pub fn system_prompt(assistant: &str) -> String {
    format!(
        r#"You are {assistant}, an assistant in an in-person meeting. People ask you things by typing on the meeting laptop or by saying "{assistant}, ...". Use the `pam` command to read the meeting and to answer.
Answer on the stage, the shared screen everyone in the room reads, with `pam say '<answer>'`. Use single quotes, so the shell leaves `$` alone. Saying again replaces what you said. Keep it short: a few plain sentences, no Markdown. Say who said what, with the time from the transcript, e.g. "Marcus (01:20) suggested...". If the meeting doesn't say, say so; don't guess.
If a request could mean clearly different things and the meeting doesn't say which, ask the room with `pam task ask`, best guess first. It shows numbered options with a countdown, and prints the option chosen, or `timeout`. If you need something only the room knows, such as a place or a name, ask it with `pam task ask` and no options, rather than ending your answer with a question; it prints what they said. On `timeout`, go with your best guess and say what you assumed. It can wait up to {wait} seconds: give that bash call no timeout, or a longer one.
When a request takes more than a few seconds, show the room what you are doing as you go with `pam task update --action '<what you are doing now>'`, and add `--progress <0-100>` when you know how much of the work is done. A count in the action, such as "reading source 2 of 5", shows as 2/5.
When an answer has sources or runs past a few sentences, write it to `tasks/<id>/result.md` (the request names its ID), attach it with `pam task result --file tasks/<id>/result.md --title '<title>'`, and put the one-line verdict on the stage with `pam say`. The room opens the result from its chip.
This is one conversation for the whole meeting: each request starts with what was said since the last one. Answer the newest request; use the earlier ones as context.
Common commands (run `pam help` for everything):
  pam context                      # what's going on: people, open items, recent lines
  pam around <id>                  # exact words around a line/event
  pam search "<words>"             # find where something was discussed
  pam transcript --since 5m        # the last five minutes, word for word
  pam say '<text>'                 # show your answer on the stage
  pam task update --progress 40 --action 'reading source 2 of 5'   # show how far along you are
  pam task result --file tasks/t03/result.md --title 'Sources'   # attach a longer answer the room can open
  pam task ask '<question>' --option '<a>' --option '<b>'   # ask the room; no options for an open question"#,
        wait = question::LONGEST_WAIT
    )
}

/// What a task is asked, naming it and who asked (`None` when typed on the
/// laptop). [`brief::lead`] goes before it when the prompt goes out.
pub fn request_prompt(id: TaskId, request: &str, asked_at: Ts, by: Option<&str>) -> String {
    let how = match by {
        Some(who) => format!("{who} asked at {}", mmss(asked_at)),
        None => format!("typed on the meeting laptop at {}", mmss(asked_at)),
    };
    format!(
        "Request {id} ({how}):\n{request}\n\n\
         Answer it on the stage with `pam say`. Use `pam` to look further into the meeting if you need to."
    )
}

/// What the agent is told when the room changes a request: who said what, and when.
pub fn correction_prompt(by: &str, at: Ts, text: &str) -> String {
    format!(
        "Correction from {by} at {}:\n{text}\n\n\
         Answer again on the stage with `pam say`, taking it into account.",
        mmss(at)
    )
}

/// One task for the agent.
#[derive(Debug, Clone)]
pub struct Job {
    pub id: TaskId,
    /// When it was asked: the catch-up's last minute is the one before this.
    pub asked_at: Ts,
    /// What it is asked, without the lead [`brief::lead`] puts before it.
    pub prompt: String,
}

/// What the meeting tells the agent thread.
#[derive(Debug, Clone)]
pub enum Inbox {
    /// A new task; it waits for the ones before it, so its
    /// answer isn't put under another task's question.
    Job(Job),
    /// Drop the task if it waits; abort it if it runs. A finished task stays finished.
    Cancel(TaskId),
    /// A correction ([`correction_prompt`]) of a task still going. A
    /// waiting task gets it in its prompt; the running one is aborted and
    /// prompted again with it, since a `steer` would wait out the running
    /// tool call. One whose run ended in its cancel window is
    /// prompted again with it. The time is when the correction was said.
    Correct(TaskId, Ts, String),
}

/// What the agent thread tells the meeting.
#[derive(Debug, Clone, PartialEq)]
pub enum Report {
    Started(TaskId),
    /// A prompt went out for the task, when its files were this long: what
    /// the agent says and reports for that prompt comes after. Taken once an
    /// aborted run has settled, so nothing the old run says or reports counts.
    Prompted(TaskId, Lengths),
    /// pi finished a model call: an assistant message or a compaction.
    Used(Call),
    /// A run ended before pi reported all it used: it was stopped, or pi
    /// exited, mid-call.
    Lost,
    Ended(TaskId, Outcome),
}

/// How long a task's `messages.jsonl` and `updates.jsonl` were.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Lengths {
    pub messages: u64,
    pub updates: u64,
}

/// What happens during a run that the agent thread passes on.
pub enum Aside {
    /// A message about another task.
    Inbox(Inbox),
    /// A prompt went out; the task's files were this long just before.
    Prompted(Lengths),
    /// pi finished a model call.
    Used(Call),
    /// The run ended before pi reported all it used.
    Lost,
}

/// How a task's run ended.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// The agent settled. `reply` is the text of its last message, if any.
    Done {
        reply: Option<String>,
    },
    Failed(String),
    /// The room cancelled it: it never started, or its run was aborted.
    Cancelled,
    /// The meeting stopped first; pi was stopped.
    Stopped,
}

/// What the room asked of the running task since its prompt went out.
enum Change {
    Cancel,
    /// Corrections to prompt with once pi has stopped the run, and when the last was said.
    Correct(Vec<String>, Ts),
}

/// The file of the meeting's `n`th agent session, from 1: `agent/session.jsonl`,
/// then `agent/session-2.jsonl` and so on.
fn session_file(meeting: &Path, n: u32) -> PathBuf {
    let name = if n <= 1 { "session.jsonl".to_string() } else { format!("session-{n}.jsonl") };
    meeting.join("agent").join(name)
}

/// Runs the meeting's tasks one at a time, as prompts in one pi session.
pub struct Runner {
    pub pi: PathBuf,
    pub model: String,
    /// The meeting folder, as an absolute path.
    pub meeting: PathBuf,
    pub assistant: String,
    /// The `pam` to put first on the agent's `PATH`.
    pub pam: Option<PathBuf>,
    /// The meeting time, which the meeting keeps up to date.
    pub clock: Arc<AtomicU64>,
    /// The lines on their way to being final, which the meeting keeps up to date.
    pub coming: Arc<Mutex<Vec<Coming>>>,
    /// The pi the tasks are prompted in, from the first task on.
    session: Option<Rpc>,
    /// How many sessions pi was started with.
    sessions: u32,
    /// Where the session's last prompt left the agent; `None` before its first.
    mark: Option<Mark>,
}

impl Runner {
    pub fn new(
        pi: PathBuf,
        model: String,
        meeting: PathBuf,
        assistant: String,
        pam: Option<PathBuf>,
    ) -> Runner {
        Runner {
            pi,
            model,
            meeting,
            assistant,
            pam,
            clock: Arc::default(),
            coming: Arc::default(),
            session: None,
            sessions: 0,
            mark: None,
        }
    }

    /// The command for the session in `file`.
    pub fn command(&self, file: &Path) -> Command {
        let mut cmd = Command::new(&self.pi);
        cmd.args(["--mode", "rpc", "--model", &self.model])
            .arg("--session")
            .arg(file)
            .arg("--append-system-prompt")
            .arg(system_prompt(&self.assistant))
            .args(["-ne", "-ns", "-np", "-nc"])
            .current_dir(&self.meeting)
            .env("PAM_MEETING", &self.meeting);
        if let Some(dir) = self.pam.as_deref().and_then(Path::parent) {
            let mut path = OsString::from(dir);
            if let Some(rest) = std::env::var_os("PATH").filter(|p| !p.is_empty()) {
                path.push(":");
                path.push(rest);
            }
            cmd.env("PATH", path);
        }
        cmd
    }

    /// Prompts `job` in the meeting's session and waits until the agent
    /// settles, or until `stop` is set or the room cancels it. The session
    /// is started first if there is none, or a new one if pi has gone. pi
    /// stays for the next task unless the meeting stops, or pi exits, fails
    /// to take the prompt or doesn't stop a run it was told to. While it
    /// runs, `agent/running.json` names the task. What `inbox` says about
    /// other tasks, and each prompt sent, go to `aside`.
    pub fn run(
        &mut self,
        job: &Job,
        inbox: &Receiver<Inbox>,
        stop: &AtomicBool,
        aside: &mut dyn FnMut(Aside),
    ) -> Outcome {
        let mut session = self.session.take();
        if session.as_mut().is_some_and(|rpc| !rpc.ready()) {
            // pi went away since the last task: a new session.
            session = None;
        }
        let mut rpc = match session {
            Some(rpc) => rpc,
            None => match self.start() {
                Ok(rpc) => rpc,
                // Nothing ran, so nothing was spent.
                Err(e) => return Outcome::Failed(format!("{e:#}")),
            },
        };
        // Before the prompt goes out, so the agent's `pam say` finds its task.
        if let Err(e) = task::set_running(&self.meeting, Some(job.id)) {
            self.session = Some(rpc);
            return Outcome::Failed(format!("{e:#}"));
        }
        let mut running = false;
        let (outcome, idle) = self
            .drive(&mut rpc, job, inbox, stop, &mut running, aside)
            .unwrap_or_else(|e| (Outcome::Failed(format!("{e:#}")), false));
        if idle {
            self.session = Some(rpc);
        } else {
            let _ = rpc.stop();
        }
        // Stopped, or pi gone, before the run settled: what its call in
        // flight used is never reported.
        if running {
            aside(Aside::Lost);
        }
        // If this fails, a late `pam say` still reaches the finished task; nothing else is wrong.
        let _ = task::set_running(&self.meeting, None);
        outcome
    }

    /// Starts pi with a new session.
    fn start(&mut self) -> Result<Rpc> {
        let n = self.sessions + 1;
        let rpc = Rpc::spawn(self.command(&session_file(&self.meeting, n)))?;
        self.sessions = n;
        self.mark = None;
        Ok(rpc)
    }

    /// Stops pi, if it runs.
    pub fn close(&mut self) {
        if let Some(mut rpc) = self.session.take() {
            let _ = rpc.stop();
        }
    }

    /// `text` led by what the agent hasn't been told yet, for a
    /// prompt about a request made at `asked_at`; and the mark to keep once it is sent.
    fn message(&self, text: &str, asked_at: Ts) -> Result<(String, Mark)> {
        let rec = MeetingRecord::load(&self.meeting)?;
        let now = self.clock.load(Ordering::Relaxed);
        let coming = self.coming.lock().unwrap().clone();
        let (lead, mark) = brief::lead(&rec, self.mark.as_ref(), &coming, asked_at, now)?;
        Ok((format!("{lead}{text}"), mark))
    }

    /// Sends `text` as the session's next prompt, led by [`Runner::message`],
    /// and returns the prompt's request ID.
    fn prompt(&mut self, rpc: &mut Rpc, text: &str, asked_at: Ts) -> Result<String> {
        let (message, mark) = self.message(text, asked_at)?;
        let id = rpc.send(json!({"type": "prompt", "message": message}))?;
        self.mark = Some(mark);
        Ok(id)
    }

    /// Prompts `job` and follows its run to the end. Says, with the
    /// outcome, whether pi is idle and can take the next prompt.
    fn drive(
        &mut self,
        rpc: &mut Rpc,
        job: &Job,
        inbox: &Receiver<Inbox>,
        stop: &AtomicBool,
        running: &mut bool,
        aside: &mut dyn FnMut(Aside),
    ) -> Result<(Outcome, bool)> {
        let (messages, updates) =
            (task::messages_file(&self.meeting, job.id), task::updates_file(&self.meeting, job.id));
        let len = |path: &Path| std::fs::metadata(path).map_or(0, |m| m.len());
        let size = || Lengths { messages: len(&messages), updates: len(&updates) };
        // Measured before the prompt goes out, so that whatever the agent
        // says for it counts; reported once it has gone, so that a prompt
        // that couldn't be sent never starts the task.
        let before = size();
        let mut prompt = self.prompt(rpc, &job.prompt, job.asked_at)?;
        aside(Aside::Prompted(before));
        // Whether a run is going that pi hasn't settled.
        *running = true;
        let mut accepted = false;
        // agent_settled can come before a response (it does for abort), so
        // the run is over once both have arrived, in either order.
        let mut settled = None;
        let mut reply = None;
        let mut error = None;
        let mut change = None;
        let mut aborted_at: Option<Instant> = None;
        loop {
            if stop.load(Ordering::Relaxed) {
                return Ok((Outcome::Stopped, false));
            }
            for m in inbox.try_iter() {
                match m {
                    Inbox::Cancel(id) if id == job.id => change = Some(Change::Cancel),
                    Inbox::Correct(id, at, text) if id == job.id => match &mut change {
                        Some(Change::Cancel) => {}
                        Some(Change::Correct(texts, last)) => {
                            texts.push(text);
                            *last = at;
                        }
                        None => change = Some(Change::Correct(vec![text], at)),
                    },
                    m => aside(Aside::Inbox(m)),
                }
            }
            if accepted && settled.is_some() && change.is_none() {
                break;
            }
            if change.is_some() && settled.is_some() {
                // The run is over: the cancel takes, or the corrections go in.
                match change.take().unwrap() {
                    Change::Cancel => return Ok((Outcome::Cancelled, true)),
                    Change::Correct(texts, at) => {
                        let before = size();
                        prompt = self.prompt(rpc, &texts.join("\n\n"), at)?;
                        aside(Aside::Prompted(before));
                        *running = true;
                        (accepted, settled, reply, error, aborted_at) = (false, None, None, None, None);
                        continue;
                    }
                }
            }
            if let Some(c) = &change {
                match aborted_at {
                    None => {
                        rpc.send(json!({"type": "abort"}))?;
                        aborted_at = Some(Instant::now());
                    }
                    // It took 1–8 ms as measured. A pi still in the run can't take the next prompt.
                    Some(at) if at.elapsed() > ABORT_WAIT => match c {
                        Change::Cancel => return Ok((Outcome::Cancelled, false)),
                        Change::Correct(..) => bail!("pi didn't stop the run to take a correction"),
                    },
                    Some(_) => {}
                }
            }
            let v = match rpc.next(INBOX_POLL) {
                Next::Record(v) => v,
                Next::Nothing => continue,
                Next::Closed => {
                    let why = rpc.stderr_tail(3);
                    bail!(
                        "pi exited before it finished{}",
                        if why.is_empty() { String::new() } else { format!(": {why}") }
                    );
                }
            };
            match v["type"].as_str() {
                Some("response") if v["id"] == prompt.as_str() => {
                    if v["success"] != true {
                        *running = false;
                        bail!("pi refused the request: {}", v["error"].as_str().unwrap_or("no reason given"));
                    }
                    if v["data"]["disposition"] != "started" {
                        *running = false;
                        bail!("pi didn't start the request: {}", v["data"]);
                    }
                    accepted = true;
                }
                Some("message_end") if v["message"]["role"] == "assistant" => {
                    let m = &v["message"];
                    used(&m["usage"], aside);
                    if m["stopReason"] == "error" {
                        error = Some(m["errorMessage"].as_str().unwrap_or("the model failed").to_string());
                    } else {
                        error = None;
                        reply = Some(text_of(m)).filter(|t| !t.trim().is_empty()).or(reply);
                    }
                }
                // A compaction is a model call too; one that failed or was aborted has no result.
                Some("compaction_end") if v["result"].is_object() => {
                    used(&v["result"]["usage"], aside);
                }
                // Only `agent_settled.aborted` says reliably that an abort took.
                Some("agent_settled") => {
                    settled = Some(v["aborted"] == true);
                    *running = false;
                }
                _ => {}
            }
        }
        let outcome = match (settled, error) {
            (Some(true), _) => Outcome::Failed("the run was aborted".into()),
            (_, Some(e)) => Outcome::Failed(e),
            _ => Outcome::Done { reply },
        };
        Ok((outcome, true))
    }
}

/// Passes on what one model call used, as pi reported it in `usage`, and
/// as lost when there is none.
fn used(usage: &Value, aside: &mut dyn FnMut(Aside)) {
    match Call::from_pi(usage) {
        Some(call) => aside(Aside::Used(call)),
        None => aside(Aside::Lost),
    }
}

/// Runs each task in turn, in the order they came, until `stop` is set;
/// tasks still waiting then are reported as stopped, and pi is stopped.
pub fn serve(mut runner: Runner, inbox: Receiver<Inbox>, stop: &AtomicBool, mut report: impl FnMut(Report)) {
    let mut waiting: VecDeque<Job> = VecDeque::new();
    loop {
        if let Some(job) = waiting.pop_front() {
            if stop.load(Ordering::Relaxed) {
                report(Report::Ended(job.id, Outcome::Stopped));
                continue;
            }
            report(Report::Started(job.id));
            let id = job.id;
            let outcome = runner.run(&job, &inbox, stop, &mut |a| match a {
                Aside::Inbox(m) => take(m, &mut waiting, &mut report),
                Aside::Prompted(lengths) => report(Report::Prompted(id, lengths)),
                Aside::Used(call) => report(Report::Used(call)),
                Aside::Lost => report(Report::Lost),
            });
            report(Report::Ended(job.id, outcome));
            continue;
        }
        match inbox.recv_timeout(STOP_POLL) {
            Ok(m) => take(m, &mut waiting, &mut report),
            Err(crossbeam_channel::RecvTimeoutError::Timeout) if !stop.load(Ordering::Relaxed) => {}
            Err(_) => break,
        }
        if stop.load(Ordering::Relaxed) && waiting.is_empty() {
            break;
        }
    }
    for m in inbox.try_iter() {
        take(m, &mut waiting, &mut report);
    }
    for job in waiting {
        report(Report::Ended(job.id, Outcome::Stopped));
    }
    runner.close();
}

/// What the inbox says about a task that isn't running.
fn take(m: Inbox, waiting: &mut VecDeque<Job>, report: &mut impl FnMut(Report)) {
    match m {
        Inbox::Job(job) => waiting.push_back(job),
        Inbox::Cancel(id) => {
            if let Some(i) = waiting.iter().position(|j| j.id == id) {
                waiting.remove(i);
                report(Report::Ended(id, Outcome::Cancelled));
            }
        }
        Inbox::Correct(id, at, text) => match waiting.iter_mut().find(|j| j.id == id) {
            Some(job) => {
                job.prompt.push_str("\n\n");
                job.prompt.push_str(&text);
            }
            // Its run ended in its cancel window before the correction came:
            // it is prompted again, in the session.
            None => waiting.push_back(Job { id, asked_at: at, prompt: text }),
        },
    }
}

impl Runner {
    /// A runner for the agent's pi, if the agent is usable.
    pub fn for_meeting(cfg: &AgentConfig, meeting: &Path, assistant: &str) -> Result<Runner> {
        let info = &cfg.info;
        let (Some(pi), Some(model)) = (info.pi.clone().filter(|_| info.usable()), info.model.clone()) else {
            bail!("the agent is unavailable");
        };
        let meeting = meeting.canonicalize().with_context(|| format!("finding {}", meeting.display()))?;
        Ok(Runner::new(pi, model, meeting, assistant.into(), cfg.pam.clone()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::clock::SECOND;

    #[test]
    fn the_command_isolates_pi_and_points_it_at_the_meetings_session() {
        let runner = Runner::new(
            "/opt/pi".into(),
            "p/m:low".into(),
            "/m".into(),
            "Pam".into(),
            Some("/build/pam".into()),
        );
        let cmd = runner.command(&session_file(Path::new("/m"), 1));
        let args: Vec<String> = cmd.get_args().map(|a| a.to_string_lossy().into_owned()).collect();
        assert_eq!(args[..6], ["--mode", "rpc", "--model", "p/m:low", "--session", "/m/agent/session.jsonl"]);
        assert_eq!(args[6], "--append-system-prompt");
        assert!(args[7].starts_with("You are Pam,"));
        assert_eq!(args[8..], ["-ne", "-ns", "-np", "-nc"]);
        assert!(!args.iter().any(|a| a == "--system-prompt"));
        assert_eq!(cmd.get_current_dir(), Some(Path::new("/m")));
        let env: Vec<_> = cmd.get_envs().collect();
        assert!(env.contains(&(std::ffi::OsStr::new("PAM_MEETING"), Some(std::ffi::OsStr::new("/m")))));
        let path = env.iter().find(|(k, _)| *k == "PATH").unwrap().1.unwrap().to_string_lossy();
        assert!(path.starts_with("/build:"), "{path}");
        assert_eq!(session_file(Path::new("/m"), 2), Path::new("/m/agent/session-2.jsonl"));
    }

    /// A meeting folder, as an absolute path, with `meeting.json` and `context.md`.
    fn meeting_folder() -> (tempfile::TempDir, PathBuf) {
        let m = tempfile::tempdir().unwrap();
        let meeting = m.path().canonicalize().unwrap();
        std::fs::write(
            meeting.join("meeting.json"),
            r#"{"title":"","started_at":"2026-10-10T14:02:00+11:00","assistant_name":"Pam","status":"recording"}"#,
        )
        .unwrap();
        std::fs::write(meeting.join("context.md"), "# Meeting\n").unwrap();
        (m, meeting)
    }

    /// A pi that logs each command to `prompts.jsonl` and answers each prompt
    /// with what agent/running.json said when it came. A prompt with "crash" makes it exit.
    fn fake_pi(meeting: &Path) -> PathBuf {
        let pi = meeting.join("pi");
        std::fs::write(
            &pi,
            "#!/bin/sh\nwhile read -r line; do\n  printf '%s\\n' \"$line\" >> prompts.jsonl\n  \
             case \"$line\" in *crash*) exit 1;; esac\n  \
             id=$(printf '%s' \"$line\" | sed 's/.*\"id\":\"\\([^\"]*\\)\".*/\\1/')\n  \
             said=$(tr -d '\"{}' < agent/running.json 2>/dev/null)\n  \
             printf '{\"type\":\"response\",\"id\":\"%s\",\"success\":true,\"data\":{\"disposition\":\"started\"}}\\n' \"$id\"\n  \
             printf '{\"type\":\"message_end\",\"message\":{\"role\":\"assistant\",\"content\":[{\"type\":\"text\",\"text\":\"%s\"}],\"stopReason\":\"stop\"}}\\n' \"$said\"\n  \
             printf '{\"type\":\"agent_settled\",\"aborted\":false}\\n'\ndone\n",
        )
        .unwrap();
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&pi, std::fs::Permissions::from_mode(0o755)).unwrap();
        pi
    }

    #[test]
    fn the_running_task_is_named_before_its_prompt_goes_out_and_cleared_after() {
        let (_m, meeting) = meeting_folder();
        let mut runner = Runner::new(fake_pi(&meeting), "p/m".into(), meeting.clone(), "Pam".into(), None);
        let (_tx, inbox) = crossbeam_channel::unbounded();
        let stop = AtomicBool::new(false);
        for id in [3, 4] {
            let job = Job { id: TaskId(id), asked_at: 0, prompt: "Request".into() };
            let outcome = runner.run(&job, &inbox, &stop, &mut |_| {});
            assert_eq!(outcome, Outcome::Done { reply: Some(format!("task:t0{id}")) });
            assert_eq!(task::running(&meeting).unwrap(), None, "cleared once the run ended");
        }
        runner.close();
    }

    #[test]
    fn later_prompts_catch_up_and_a_new_session_gets_the_whole_brief_again() {
        let (_m, meeting) = meeting_folder();
        let say = |id: u32, t0_s: u64, text: &str| {
            let (t0, t1) = (t0_s * SECOND, (t0_s + 4) * SECOND);
            let line = format!(
                "{{\"id\":\"u{id:04}\",\"t0\":{t0},\"t1\":{t1},\"slot\":null,\"text\":\"{text}\"}}\n"
            );
            let mut f =
                std::fs::OpenOptions::new().create(true).append(true).open(meeting.join("transcript.jsonl"));
            std::io::Write::write_all(f.as_mut().unwrap(), line.as_bytes()).unwrap();
        };
        let mut runner = Runner::new(fake_pi(&meeting), "p/m".into(), meeting.clone(), "Pam".into(), None);
        let (_tx, inbox) = crossbeam_channel::unbounded();
        let stop = AtomicBool::new(false);
        let mut ask = |id: u32, request: &str, at_s: u64, now_s: u64| {
            runner.clock.store(now_s * SECOND, Ordering::Relaxed);
            let job =
                Job { id: TaskId(id), asked_at: at_s * SECOND, prompt: format!("Request t0{id}: {request}") };
            runner.run(&job, &inbox, &stop, &mut |_| {})
        };
        say(1, 60, "Pam, what did we decide?");
        ask(1, "what did we decide?", 60, 62);
        say(2, 100, "Forty a seat.");
        say(3, 300, "Pam, and the launch?");
        ask(2, "and the launch?", 300, 302);
        say(4, 310, "Friday.");
        ask(3, "anything else?", 310, 312);
        assert!(matches!(ask(4, "crash", 320, 320), Outcome::Failed(_)));
        ask(5, "still there?", 330, 330);
        runner.close();

        let sent: Vec<String> = std::fs::read_to_string(meeting.join("prompts.jsonl"))
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str::<Value>(l).unwrap()["message"].as_str().unwrap().to_string())
            .collect();
        assert_eq!(sent.len(), 5, "{sent:#?}");
        assert!(
            sent[0].starts_with("<context.md>\n# Meeting\n</context.md>\n\n<last-60-seconds>\nu0001 01:00 "),
            "{}",
            sent[0]
        );
        assert!(sent[0].ends_with("Request t01: what did we decide?"), "{}", sent[0]);
        // Almost four minutes on: the long gap's catch-up, with the new lines from a minute before the request.
        assert!(sent[1].starts_with("Since the last prompt (01:02), in short:\n"), "{}", sent[1]);
        assert!(sent[1].contains("<last-60-seconds>\nu0003 05:00 "), "{}", sent[1]);
        assert!(!sent[1].contains("u0002"), "{}", sent[1]);
        assert!(
            sent[1].ends_with("More: pam transcript --from 01:02\n\nRequest t02: and the launch?"),
            "{}",
            sent[1]
        );
        // Ten seconds on: the line since, word for word.
        assert!(
            sent[2].starts_with("Said since the last prompt (05:02), word for word:\n<lines>\nu0004 05:10 "),
            "{}",
            sent[2]
        );
        assert!(
            sent[3]
                .starts_with("Said since the last prompt (05:12), word for word:\n<lines>\n(nobody spoke)\n"),
            "{}",
            sent[3]
        );
        // pi exited on the crash: the next prompt starts a new session with the whole brief.
        assert!(
            sent[4].starts_with("<context.md>\n# Meeting\n</context.md>\n\n<last-60-seconds>\nu0003 05:00 "),
            "{}",
            sent[4]
        );
    }

    #[test]
    fn a_request_names_its_task_and_who_asked() {
        let typed = request_prompt(TaskId(1), "what did we decide about pricing?", 62 * SECOND, None);
        assert!(
            typed.starts_with(
                "Request t01 (typed on the meeting laptop at 01:02):\nwhat did we decide about pricing?\n\n"
            ),
            "{typed}"
        );
        let spoken = request_prompt(TaskId(2), "the launch date?", 65 * SECOND, Some("Jordan"));
        assert!(spoken.starts_with("Request t02 (Jordan asked at 01:05):\nthe launch date?\n\n"), "{spoken}");
    }
}
