//! The one seam every text model is reached through: the scribe's, the
//! intent check's, and any later one. A [`Model`] answers a
//! [`Request`] with a [`Completion`]; today's implementations are [`Pi`], a
//! cloud model through pi in print mode, and [`crate::local::Local`], a model
//! run in Pam's process. Tests use stubs; nothing else here touches the
//! network.

use std::io::{Read as _, Write as _};
use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use serde_json::Value;

use crate::usage::{Call, Used};

/// A model the scribe or the intent check can ask.
pub trait Model: Send + Sync {
    /// Waits until the model can answer; `Err` says why it never will. A
    /// local model is loading until then.
    fn ready(&self) -> Result<()> {
        Ok(())
    }

    /// The model's reply to `req`, and what the call used.
    fn complete(&self, req: &Request<'_>) -> Completion;
}

/// One call to a [`Model`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Request<'a> {
    /// The instructions the model reads first. Between calls with the same
    /// system prompt a local model keeps it in memory.
    pub system: &'a str,
    /// What it answers.
    pub prompt: &'a str,
    /// A GBNF grammar the reply must fit, when the caller needs its shape
    /// held. A local model holds the reply to it; pi has no way to, so the
    /// prompt must also say what shape to answer in, and the caller checks
    /// the reply either way.
    pub grammar: Option<&'a str>,
}

impl<'a> Request<'a> {
    pub fn new(system: &'a str, prompt: &'a str) -> Request<'a> {
        Request { system, prompt, grammar: None }
    }

    pub fn with_grammar(self, grammar: &'a str) -> Request<'a> {
        Request { grammar: Some(grammar), ..self }
    }
}

/// What one call to a [`Model`] came back with.
#[derive(Debug)]
pub struct Completion {
    /// The reply's text, or why the call failed.
    pub reply: Result<String>,
    /// What the call used, as pi reported it.
    pub used: Used,
}

/// A model reached through pi in print mode, with the user's own pi login
/// (as in step 2): no tools, no session, no extensions or context files.
pub struct Pi {
    /// The `pi` executable.
    pub program: PathBuf,
    /// `provider/id[:thinking]`, as pi's `--model` takes it.
    pub model: String,
    /// A run that takes longer is killed and counts as failed.
    pub timeout: Duration,
}

impl Pi {
    pub fn new(model: impl Into<String>) -> Pi {
        Pi { program: "pi".into(), model: model.into(), timeout: Duration::from_secs(90) }
    }
}

impl Model for Pi {
    /// A request's grammar isn't enforced: pi's print mode takes none.
    fn complete(&self, req: &Request<'_>) -> Completion {
        match self.run(req.system, req.prompt) {
            Ok(done) => done,
            // Killed for taking too long, or lost track of: the call may have been made.
            Err(e) => Completion { reply: Err(e), used: Used::Unknown },
        }
    }
}

impl Pi {
    /// Runs pi in print mode with its JSON event stream (`--mode json`), so
    /// the reply comes with its usage: both are the final assistant message's.
    fn run(&self, system: &str, prompt: &str) -> Result<Completion> {
        let spawned = Command::new(&self.program)
            .args(["-p", "--mode", "json", "--no-session", "-ne", "-ns", "-np", "-nc", "-nt", "--no-mcp"])
            .args(["--offline", "--model", &self.model, "--system-prompt", system])
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .map_err(|e| match e.kind() {
                std::io::ErrorKind::NotFound => anyhow!(
                    "`{}` isn't installed or isn't on PATH; install pi, or run with --scribe off",
                    self.program.display()
                ),
                _ => anyhow!(e).context(format!("starting {}", self.program.display())),
            });
        let mut child = match spawned {
            Ok(child) => child,
            Err(e) => return Ok(Completion { reply: Err(e), used: Used::Nothing }),
        };
        // The prompt goes in on stdin: it can be longer than an argument may be.
        let mut stdin = child.stdin.take().unwrap();
        let prompt = prompt.to_string();
        let writer = std::thread::spawn(move || stdin.write_all(prompt.as_bytes()));
        let mut stdout = child.stdout.take().unwrap();
        let reader = std::thread::spawn(move || {
            let mut s = String::new();
            stdout.read_to_string(&mut s).map(|_| s)
        });
        let mut stderr = child.stderr.take().unwrap();
        let errors = std::thread::spawn(move || {
            let mut s = String::new();
            let _ = stderr.read_to_string(&mut s);
            s
        });
        let deadline = Instant::now() + self.timeout;
        let status = loop {
            if let Some(status) = child.try_wait().context("waiting for pi")? {
                break status;
            }
            if Instant::now() >= deadline {
                let _ = child.kill();
                let _ = child.wait();
                bail!("{} took over {} s; stopped it", self.model, self.timeout.as_secs());
            }
            std::thread::sleep(Duration::from_millis(50));
        };
        let _ = writer.join();
        let out =
            reader.join().map_err(|_| anyhow!("reading pi's output"))?.context("reading pi's output")?;
        let err = errors.join().unwrap_or_default();
        let last = last_reply(&out);
        let call = last.as_ref().and_then(|m| Call::from_pi(&m["usage"]));
        let used = call.map_or(Used::Unknown, Used::Call);
        let reply = match last {
            _ if !status.success() => {
                let err = err.trim();
                let why = if err.is_empty() { out.trim() } else { err };
                Err(anyhow!("pi --model {} failed ({status}): {}", self.model, last_lines(why, 3)))
            }
            None => Err(anyhow!("pi --model {} gave no reply: {}", self.model, last_lines(err.trim(), 3))),
            // pi exits 0 when the model fails; the message says why.
            Some(m) if matches!(m["stopReason"].as_str(), Some("error" | "aborted")) => Err(anyhow!(
                "pi --model {} failed: {}",
                self.model,
                m["errorMessage"].as_str().unwrap_or("the model gave no reason")
            )),
            Some(m) => Ok(text_of(&m)),
        };
        Ok(Completion { reply, used })
    }
}

/// The last assistant message in pi's JSON event stream: the final one, of
/// a print-mode run.
fn last_reply(stream: &str) -> Option<Value> {
    stream
        .lines()
        .rev()
        .filter_map(|l| serde_json::from_str::<Value>(l).ok())
        .find(|v| v["type"] == "message_end" && v["message"]["role"] == "assistant")
        .map(|mut v| v["message"].take())
}

/// The text parts of a pi message, joined.
pub(crate) fn text_of(message: &Value) -> String {
    let parts = message["content"].as_array().map(Vec::as_slice).unwrap_or_default();
    parts
        .iter()
        .filter(|p| p["type"] == "text")
        .filter_map(|p| p["text"].as_str())
        .collect::<Vec<_>>()
        .join("\n")
}

fn last_lines(s: &str, n: usize) -> String {
    let lines: Vec<&str> = s.lines().collect();
    lines[lines.len().saturating_sub(n)..].join(" / ")
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `pi` that prints `stdout` and `stderr` and exits with `code`.
    fn fake_pi(dir: &std::path::Path, stdout: &str, stderr: &str, code: i32) -> Pi {
        use std::os::unix::fs::PermissionsExt;
        let (out, err) = (dir.join("out"), dir.join("err"));
        std::fs::write(&out, stdout).unwrap();
        std::fs::write(&err, stderr).unwrap();
        let pi = dir.join("pi");
        let script = format!(
            "#!/bin/sh\ncat >/dev/null\ncat '{}'\ncat '{}' >&2\nexit {code}\n",
            out.display(),
            err.display()
        );
        std::fs::write(&pi, script).unwrap();
        std::fs::set_permissions(&pi, std::fs::Permissions::from_mode(0o755)).unwrap();
        Pi { program: pi, ..Pi::new("openrouter/anthropic/claude-haiku-5.5") }
    }

    const USAGE: &str =
        r#""usage":{"input":97,"output":4,"cacheRead":5,"cacheWrite":6,"cost":{"total":0.0000117}}"#;

    #[test]
    fn the_reply_and_its_usage_come_from_the_final_assistant_message() {
        let dir = tempfile::tempdir().unwrap();
        let stream = [
            r#"{"type":"session","version":3,"id":"x","cwd":"/"}"#.to_string(),
            r#"{"type":"message_end","message":{"role":"user","content":[{"type":"text","text":"hi"}]}}"#.into(),
            format!(r#"{{"type":"message_update",{USAGE},"assistantMessageEvent":{{"type":"text_delta"}}}}"#),
            format!(
                r#"{{"type":"message_end","message":{{"role":"assistant","content":[{{"type":"thinking","thinking":"hm"}},{{"type":"text","text":"{{\"a\": 1}}"}},{{"type":"text","text":"done"}}],"stopReason":"stop",{USAGE}}}}}"#
            ),
            r#"{"type":"agent_settled","aborted":false}"#.into(),
        ]
        .join("\n");
        let done = fake_pi(dir.path(), &stream, "", 0).complete(&Request::new("s", "p"));
        assert_eq!(done.reply.unwrap(), "{\"a\": 1}\ndone");
        assert_eq!(
            done.used,
            Used::Call(Call { input: 97, output: 4, cache_read: 5, cache_write: 6, cost: Some(0.0000117) })
        );
    }

    #[test]
    fn a_failed_model_call_is_an_error_even_when_pi_exits_0() {
        let dir = tempfile::tempdir().unwrap();
        let failed = format!(
            r#"{{"type":"message_end","message":{{"role":"assistant","content":[],"stopReason":"error","errorMessage":"400: not a valid model ID",{USAGE}}}}}"#
        );
        let req = Request::new("s", "p").with_grammar("root ::= \"x\"");
        let done = fake_pi(dir.path(), &failed, "", 0).complete(&req);
        assert!(done.reply.unwrap_err().to_string().ends_with("failed: 400: not a valid model ID"));
        assert!(matches!(done.used, Used::Call(c) if c.input == 97), "what pi reported still counts");

        // No reply at all, and pi failing.
        let done = fake_pi(dir.path(), "", "Error: no model\n", 0).complete(&req);
        assert!(done.reply.unwrap_err().to_string().contains("gave no reply: Error: no model"));
        assert_eq!(done.used, Used::Unknown);
        let done = fake_pi(dir.path(), "", "Error: Model \"nope/x\" not found.\n", 1).complete(&req);
        assert!(done.reply.unwrap_err().to_string().contains("failed (exit status: 1): Error: Model"));
        assert_eq!(done.used, Used::Unknown);
    }

    #[test]
    fn pi_that_isnt_there_says_how_to_go_on() {
        let pi = Pi { program: "/nonexistent/pi".into(), ..Pi::new("openrouter/anthropic/claude-haiku-5.5") };
        let done = pi.complete(&Request::new("s", "p"));
        let err = done.reply.unwrap_err().to_string();
        assert!(err.contains("--scribe off"), "{err}");
        assert_eq!(done.used, Used::Nothing, "pi never started, so nothing was spent");
    }
}
