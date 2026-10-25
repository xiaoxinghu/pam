//! `pam note`: adds a `note` event to the meeting, and its notes.

use std::fmt::Write as _;

use pam_core::clock::mmss;
use serde_json::json;

use crate::cli::{Command, Fail, Parsed};
use crate::read;

pub fn run(cmd: &'static Command, p: Parsed, out: &mut String) -> Result<(), Fail> {
    let text = p.positionals.join(" ");
    if text.trim().is_empty() {
        return Err(Fail::usage(cmd, "missing <text>: what should I note?"));
    }
    let rec = read::meeting(&p)?;
    let note = pam_core::notes::add_note(&rec.dir, &text)?;
    if p.has("--json") {
        writeln!(out, "{}", serde_json::to_string_pretty(&json!({ "event": note })).unwrap()).unwrap();
    } else {
        writeln!(out, "{} {} note: {}", note.id, mmss(note.t), note.text).unwrap();
    }
    Ok(())
}
