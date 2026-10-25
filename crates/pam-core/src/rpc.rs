//! A client for `pi --mode rpc`: JSON lines over the child's stdin and stdout.
//!
//! It follows how pi 1.1.0 was measured to behave:
//! - stdout is split on LF only, never on U+2028/U+2029, which pi writes raw
//!   inside JSON strings;
//! - stdout is read all the time, on a thread of its own, since pi stalls
//!   when nobody reads it;
//! - pi is stopped by closing its stdin, then SIGTERM if it hasn't exited.
//!   Never SIGKILL alone: pi runs each tool in its own process group, so the
//!   tool would keep running.

use std::io::{BufRead, BufReader, Read, Write};
use std::process::{Child, ChildStdin, Command, ExitStatus, Stdio};
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow};
use crossbeam_channel::{Receiver, RecvTimeoutError, TryRecvError, unbounded};
use serde_json::Value;

/// How long pi gets to exit once its stdin is closed. It took 4–6 ms as measured.
const CLOSE_GRACE: Duration = Duration::from_millis(500);
/// How long pi gets to exit after SIGTERM, before the last resort.
const TERM_GRACE: Duration = Duration::from_secs(2);
/// How much of pi's stderr is kept, from the end, to explain a failure.
const STDERR_KEPT: usize = 4096;

/// What the next record from pi turned out to be.
#[derive(Debug)]
pub enum Next {
    Record(Value),
    /// Nothing within the wait.
    Nothing,
    /// pi closed its stdout: it has exited or is about to.
    Closed,
}

/// A running `pi --mode rpc`.
pub struct Rpc {
    child: Child,
    stdin: Option<ChildStdin>,
    records: Receiver<Value>,
    stderr: Arc<Mutex<Vec<u8>>>,
    next_id: u64,
}

impl Rpc {
    /// Starts `cmd`, which should run pi in RPC mode. Its stdio are taken over.
    pub fn spawn(mut cmd: Command) -> Result<Rpc> {
        let program = cmd.get_program().to_string_lossy().into_owned();
        let mut child = cmd
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .with_context(|| format!("starting {program}"))?;
        let stdout = child.stdout.take().unwrap();
        let (tx, records) = unbounded();
        thread::Builder::new()
            .name("pam-pi-out".into())
            .spawn(move || {
                let mut r = BufReader::new(stdout);
                let mut buf = Vec::new();
                loop {
                    buf.clear();
                    match r.read_until(b'\n', &mut buf) {
                        Ok(0) | Err(_) => break,
                        Ok(_) => {}
                    }
                    let line = buf.strip_suffix(b"\n").unwrap_or(&buf);
                    let line = line.strip_suffix(b"\r").unwrap_or(line);
                    // pi only writes JSON; anything else is skipped.
                    if let Ok(v) = serde_json::from_slice::<Value>(line)
                        && tx.send(v).is_err()
                    {
                        break;
                    }
                }
            })
            .context("starting a thread to read pi")?;
        let stderr = Arc::new(Mutex::new(Vec::new()));
        let mut err = child.stderr.take().unwrap();
        thread::Builder::new()
            .name("pam-pi-err".into())
            .spawn({
                let stderr = stderr.clone();
                move || {
                    let mut chunk = [0; 1024];
                    while let Ok(n @ 1..) = err.read(&mut chunk) {
                        let mut kept = stderr.lock().unwrap();
                        kept.extend_from_slice(&chunk[..n]);
                        let over = kept.len().saturating_sub(STDERR_KEPT);
                        kept.drain(..over);
                    }
                }
            })
            .context("starting a thread to read pi's errors")?;
        let stdin = child.stdin.take();
        Ok(Rpc { child, stdin, records, stderr, next_id: 0 })
    }

    /// Sends one command, giving it an `id` if it has none, and returns the id.
    pub fn send(&mut self, mut command: Value) -> Result<String> {
        if command.get("id").is_none() {
            self.next_id += 1;
            command["id"] = Value::from(format!("pam{}", self.next_id));
        }
        let id = command["id"].as_str().unwrap_or_default().to_string();
        let mut line = serde_json::to_vec(&command)?;
        line.push(b'\n');
        let stdin = self.stdin.as_mut().ok_or_else(|| anyhow!("pi's stdin is closed"))?;
        stdin.write_all(&line).and_then(|()| stdin.flush()).context("writing to pi")?;
        Ok(id)
    }

    /// The next record, waiting at most `wait`.
    pub fn next(&self, wait: Duration) -> Next {
        match self.records.recv_timeout(wait) {
            Ok(v) => Next::Record(v),
            Err(RecvTimeoutError::Timeout) => Next::Nothing,
            Err(RecvTimeoutError::Disconnected) => Next::Closed,
        }
    }

    /// Whether pi is there to take a command: it hasn't exited or closed its
    /// stdout. Records not read yet, from a run that is over, are dropped.
    pub fn ready(&mut self) -> bool {
        loop {
            match self.records.try_recv() {
                Ok(_) => {}
                Err(TryRecvError::Empty) => break,
                Err(TryRecvError::Disconnected) => return false,
            }
        }
        matches!(self.child.try_wait(), Ok(None))
    }

    /// The last lines pi wrote to stderr, joined, for an error message.
    pub fn stderr_tail(&self, lines: usize) -> String {
        let kept = self.stderr.lock().unwrap();
        let text = String::from_utf8_lossy(&kept);
        let all: Vec<&str> = text.lines().map(str::trim).filter(|l| !l.is_empty()).collect();
        all[all.len().saturating_sub(lines)..].join(" / ")
    }

    /// Stops pi: closes its stdin, then sends SIGTERM if it hasn't exited,
    /// and only as a last resort SIGKILL.
    pub fn stop(&mut self) -> Result<ExitStatus> {
        self.stdin.take();
        if let Some(status) = self.wait_for(CLOSE_GRACE)? {
            return Ok(status);
        }
        // SAFETY: kill(2) on our own child, which hasn't been reaped yet.
        unsafe { libc::kill(self.child.id() as libc::pid_t, libc::SIGTERM) };
        if let Some(status) = self.wait_for(TERM_GRACE)? {
            return Ok(status);
        }
        self.child.kill().context("killing pi")?;
        self.child.wait().context("waiting for pi")
    }

    fn wait_for(&mut self, grace: Duration) -> Result<Option<ExitStatus>> {
        let end = Instant::now() + grace;
        loop {
            if let Some(status) = self.child.try_wait().context("waiting for pi")? {
                return Ok(Some(status));
            }
            if Instant::now() >= end {
                return Ok(None);
            }
            thread::sleep(Duration::from_millis(2));
        }
    }
}

/// pi never outlives its client.
impl Drop for Rpc {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(None)) {
            let _ = self.stop();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sh(script: &str) -> Command {
        let mut cmd = Command::new("/bin/sh");
        cmd.args(["-c", script]);
        cmd
    }

    fn record(rpc: &Rpc) -> Value {
        match rpc.next(Duration::from_secs(5)) {
            Next::Record(v) => v,
            other => panic!("expected a record, got {other:?}"),
        }
    }

    #[test]
    fn splits_on_lf_only_and_skips_what_isnt_json() {
        // U+2028 and U+2029 written raw inside a JSON string, a CRLF line, and noise.
        let rpc = Rpc::spawn(sh(
            "printf '{\"text\":\"a\\342\\200\\250b\\342\\200\\251c\"}\\nnoise\\n{\"n\":2}\\r\\n'",
        ))
        .unwrap();
        assert_eq!(record(&rpc)["text"], "a\u{2028}b\u{2029}c");
        assert_eq!(record(&rpc)["n"], 2);
        assert!(matches!(rpc.next(Duration::from_secs(5)), Next::Closed));
    }

    #[test]
    fn commands_get_ids_and_go_out_as_lines() {
        let mut rpc = Rpc::spawn(sh("cat")).unwrap();
        let id = rpc.send(serde_json::json!({"type": "get_state"})).unwrap();
        assert_eq!(id, "pam1");
        assert_eq!(record(&rpc), serde_json::json!({"type": "get_state", "id": "pam1"}));
        assert!(rpc.stop().unwrap().success());
    }

    #[test]
    fn ready_drops_what_is_left_unread_until_pi_has_gone() {
        let mut rpc = Rpc::spawn(sh("echo '{\"type\":\"late\"}'; read line; exit 0")).unwrap();
        while rpc.records.is_empty() {
            thread::sleep(Duration::from_millis(5));
        }
        assert!(rpc.ready());
        assert!(matches!(rpc.next(Duration::from_millis(50)), Next::Nothing), "the late record is gone");
        rpc.send(serde_json::json!({"type": "prompt"})).unwrap();
        let end = Instant::now() + Duration::from_secs(5);
        while rpc.ready() {
            assert!(Instant::now() < end, "pi never went away");
            thread::sleep(Duration::from_millis(5));
        }
    }

    #[test]
    fn stop_closes_stdin_first() {
        // Exits 3 once its stdin closes, as pi does.
        let mut rpc = Rpc::spawn(sh("cat >/dev/null; exit 3")).unwrap();
        assert_eq!(rpc.stop().unwrap().code(), Some(3));
    }

    #[test]
    fn stop_sends_sigterm_to_a_child_that_ignores_its_stdin() {
        let mut rpc = Rpc::spawn(sh("echo started >&2; exec sleep 30")).unwrap();
        while rpc.stderr_tail(1).is_empty() {
            thread::sleep(Duration::from_millis(5));
        }
        let started = Instant::now();
        let status = rpc.stop().unwrap();
        use std::os::unix::process::ExitStatusExt;
        assert_eq!(status.signal(), Some(libc::SIGTERM));
        assert!(started.elapsed() < TERM_GRACE, "{:?}", started.elapsed());
        assert_eq!(rpc.stderr_tail(3), "started");
    }
}
