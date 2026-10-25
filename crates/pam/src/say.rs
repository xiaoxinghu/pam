//! `pam say`: shows a message on the stage for a task.

use std::fmt::Write as _;

use pam_core::TaskId;
use pam_core::task;
use serde_json::json;

use crate::cli::{Command, Fail, Parsed};
use crate::read;

pub fn run(cmd: &'static Command, p: Parsed, out: &mut String) -> Result<(), Fail> {
    let text = p.positionals.join(" ");
    if text.trim().is_empty() {
        return Err(Fail::usage(cmd, "missing <text>: what should I show on the stage?"));
    }
    let given =
        p.value("--task").map(|t| t.parse::<TaskId>().map_err(|e| Fail::usage(cmd, e))).transpose()?;
    let rec = read::meeting(&p)?;
    let id = crate::task::which_task(cmd, given, "Pass --task", &rec.dir)?;
    let message = task::say(&rec.dir, id, &text)?;
    if p.has("--json") {
        let v = json!({ "task": id, "message": message });
        writeln!(out, "{}", serde_json::to_string_pretty(&v).unwrap()).unwrap();
    } else {
        writeln!(out, "{id} on the stage: {}", message.text).unwrap();
    }
    Ok(())
}
