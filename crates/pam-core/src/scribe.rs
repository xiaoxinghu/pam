//! The scribe: turns new final lines into typed events and a summary.
//!
//! This module decides when a run is due, writes the model's prompt, and
//! checks its reply. The meeting runs it on its own thread, so lines never
//! wait for the model; the core then saves the events and renders the notes.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::Ts;
use crate::clock::{SECOND, mmss};
use crate::event::{EventId, Line, SpkId, UtteranceId};
use crate::record::{EVENT_TYPES, Event, Ref};

/// A run is due once the lines waiting for it span this much speech…
const SPEECH: Ts = 25 * SECOND;
/// …or once nobody has spoken for this long.
pub const PAUSE: Ts = 3 * SECOND;
/// Earlier lines sent along as context, so a run can follow on from the last.
const CONTEXT_LINES: usize = 8;
/// Events sent along so the model doesn't repeat them.
const RECENT_EVENTS: usize = 30;

/// Whether `kind` is an event type the scribe writes: every type but the
/// two that come from tasks.
fn scribe_type(kind: &str) -> bool {
    EVENT_TYPES.contains(&kind) && !matches!(kind, "request" | "task_update")
}

/// Whether the lines waiting for the scribe should go now. `speaking` is
/// whether anything is still on its way to becoming a final line.
pub fn due(waiting: &[Line], now: Ts, speaking: bool) -> bool {
    let (Some(first), Some(last)) = (waiting.first(), waiting.last()) else { return false };
    last.t1.saturating_sub(first.t0) >= SPEECH || (!speaking && now >= last.t1 + PAUSE)
}

/// What the scribe is shown on one run.
pub struct Input<'a> {
    pub title: &'a str,
    /// Everyone with an ID, and their name if they have one.
    pub people: Vec<(SpkId, Option<String>)>,
    pub summary: &'a str,
    /// Every event so far; the prompt shows open items and the recent ones.
    pub events: &'a [Event],
    /// Lines the scribe has already read, oldest first.
    pub earlier: &'a [Line],
    /// The lines this run is about.
    pub new: &'a [Line],
}

/// The instructions, the same on every run.
pub fn system_prompt(assistant: &str) -> String {
    format!(
        r#"You are the scribe for a live, in-person meeting. Each time, you get the newest lines of the transcript. You record what matters as typed events, and you update a short summary. Your events become the meeting notes, read live on a shared screen.

The transcript comes from speech recognition: expect misheard words, missing punctuation and filler. Each line has an ID (u0012), a time, and its speaker as an ID (spk_1), with a name once known.

Event types:
- topic: a new subject starts. text: the subject in 2-6 words.
- point: a claim, fact or piece of information worth keeping.
- proposal: a suggestion for what to do.
- concern: a worry, risk or objection.
- question: an open question the meeting should answer. Not rhetorical or small-talk questions, and not one already answered in the NEW LINES: record the answer instead.
- answer: an answer to an open question. Set "supersedes" to the question's event ID, and make the text stand on its own.
- decision: something the group agreed or settled, including a proposal the others went along with.
- action: a task someone will do. "by" is the owner.
- note: someone asked {assistant} to note something ("{assistant}, note that...").
- intro: someone said who they are ("I'm Sam, from sales"), or someone else introduced them. "by" is the speaker who is that person; "name" is their name, one or two words.

Rules:
- Lines that ask {assistant} to do or answer something ("{assistant}, what did we decide about...?") are its tasks, recorded elsewhere: no events for them, except a note.
- Record what a good note-taker would: the substance, not the chatter. Skip greetings, filler, false starts and anything already recorded. Several lines can make one event, and many lines make none. Keep the notes short: about one event for every 30 seconds of discussion, merging related remarks into one.
- text: a short note of at most 15 words, in plain English, without the speaker's name and without a final full stop. Fix obvious recognition errors.
- refs: the IDs of the lines the event comes from, or a range (u0012..u0015). Use the NEW LINES.
- by: speaker IDs, e.g. ["spk_1"]: who said it, or for an action, who will do it.
- To change an earlier event (a decision reversed, an action reassigned), add a new event with "supersedes" set to its event ID.
- summary: at most 2 sentences and 40 words on where the whole meeting stands now: what it is about, what is settled and what is open. Use names where known, else leave people out. Keep what still holds from the summary so far.

Reply with one JSON object and nothing else:
{{"events": [{{"type": "point", "by": ["spk_1"], "text": "...", "refs": ["u0012"]}}], "summary": "..."}}"#
    )
}

/// The prompt for one run.
pub fn prompt(input: &Input) -> String {
    let who = |spk: Option<SpkId>| match spk {
        Some(s) => match input.people.iter().find(|(p, _)| *p == s).and_then(|(_, n)| n.as_deref()) {
            Some(name) => format!("{s} {name}"),
            None => s.to_string(),
        },
        None => "unknown".into(),
    };
    let mut out = String::new();
    let title = if input.title.is_empty() { "(untitled)" } else { input.title };
    writeln!(out, "MEETING: {title}").unwrap();
    out.push_str("\nPEOPLE\n");
    if input.people.is_empty() {
        out.push_str("nobody yet\n");
    }
    for (spk, name) in &input.people {
        writeln!(out, "{spk} {}", name.as_deref().unwrap_or("(no name yet)")).unwrap();
    }
    writeln!(out, "\nSUMMARY SO FAR\n{}", if input.summary.is_empty() { "(none)" } else { input.summary })
        .unwrap();

    let superseded: BTreeSet<EventId> = input.events.iter().filter_map(|e| e.supersedes).collect();
    let open: Vec<&Event> = input
        .events
        .iter()
        .filter(|e| matches!(e.kind.as_str(), "question" | "action") && !superseded.contains(&e.id))
        .collect();
    let event_line = |e: &Event| {
        let by: Vec<String> = e.by.iter().map(|&s| who(Some(s))).collect();
        format!("{} {} [{}]: {}\n", e.id, e.kind, by.join(", "), e.text)
    };
    out.push_str("\nOPEN QUESTIONS AND ACTIONS\n");
    if open.is_empty() {
        out.push_str("(none)\n");
    }
    for e in &open {
        out.push_str(&event_line(e));
    }
    out.push_str("\nRECENT EVENTS\n");
    let recent = &input.events[input.events.len().saturating_sub(RECENT_EVENTS)..];
    if recent.is_empty() {
        out.push_str("(none)\n");
    }
    for e in recent {
        out.push_str(&event_line(e));
    }
    let line = |l: &Line| format!("{} {} {}: {}\n", l.id, mmss(l.t0), who(l.spk), l.text);
    let earlier = &input.earlier[input.earlier.len().saturating_sub(CONTEXT_LINES)..];
    if !earlier.is_empty() {
        out.push_str("\nEARLIER LINES (already recorded, for context)\n");
        for l in earlier {
            out.push_str(&line(l));
        }
    }
    out.push_str("\nNEW LINES\n");
    for l in input.new {
        out.push_str(&line(l));
    }
    out
}

/// The model's reply, read leniently: anything that doesn't fit is dropped later.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Reply {
    pub events: Vec<Value>,
    pub summary: Option<String>,
}

/// Finds the JSON object in the model's reply, even inside a code fence or
/// after a sentence of preamble.
pub fn parse(reply: &str) -> Result<Reply> {
    let (Some(start), Some(end)) = (reply.find('{'), reply.rfind('}')) else {
        bail!("the scribe's reply has no JSON object: {}", snippet(reply));
    };
    let v: Value = serde_json::from_str(&reply[start..=end])
        .with_context(|| format!("the scribe's reply isn't valid JSON: {}", snippet(reply)))?;
    let events = match v.get("events") {
        Some(Value::Array(a)) => a.clone(),
        None | Some(Value::Null) => Vec::new(),
        Some(_) => bail!("the scribe's \"events\" isn't a list: {}", snippet(reply)),
    };
    let summary =
        v.get("summary").and_then(Value::as_str).map(|s| s.trim().to_string()).filter(|s| !s.is_empty());
    Ok(Reply { events, summary })
}

fn snippet(s: &str) -> String {
    let s = s.trim();
    match s.char_indices().nth(120) {
        Some((i, _)) => format!("{}…", &s[..i]),
        None => s.to_string(),
    }
}

/// The reply's events that meet the spec, ready to save (IDs are given when
/// they are appended). Only what the spec allows gets in:
/// - `type` is a scribe type ([`scribe_type`]);
/// - `refs` are lines that exist, a range only if both ends do (one past the
///   last line would grow), and at least one is a line of this run (`new`
///   on): the earlier lines were context, already recorded;
/// - `t` is the first cited line's start;
/// - `by` is known speakers, else the cited lines' speakers;
/// - an `intro` names one speaker and a `name` that looks like one;
/// - `text` isn't empty;
/// - `supersedes` names an earlier event of the same type, or a question
///   for an answer; otherwise it is dropped and the event stands alone.
pub fn check(
    reply: &Reply,
    lines: &[Line],
    new: UtteranceId,
    speakers: &BTreeSet<SpkId>,
    events: &[Event],
) -> Vec<Event> {
    let line = |id| lines.binary_search_by_key(&id, |l: &Line| l.id).ok().map(|i| &lines[i]);
    let mut out = Vec::new();
    for v in &reply.events {
        let text = |k: &str| v.get(k).and_then(Value::as_str).map(str::trim).unwrap_or("");
        let kind = text("type").to_lowercase();
        if !scribe_type(&kind) {
            continue;
        }
        let mut refs = Vec::new();
        let mut cited: Vec<&Line> = Vec::new();
        for r in strings(v.get("refs")) {
            match r.parse::<Ref>() {
                Ok(Ref::Line(id)) if line(id).is_some() => {
                    cited.extend(line(id));
                    refs.push(r);
                }
                // Both ends must be lines: a range past the last one would grow
                // to take in lines that come later.
                Ok(Ref::Lines(a, b)) if line(a).is_some() && line(b).is_some() => {
                    cited.extend(lines.iter().filter(|l| l.id >= a && l.id <= b));
                    refs.push(r);
                }
                _ => {}
            }
        }
        if !cited.iter().any(|l| l.id >= new) {
            continue;
        }
        let Some(t) = cited.iter().map(|l| l.t0).min() else { continue };
        let mut by: Vec<SpkId> = Vec::new();
        for spk in strings(v.get("by")).iter().filter_map(|s| s.parse::<SpkId>().ok()) {
            if speakers.contains(&spk) && !by.contains(&spk) {
                by.push(spk);
            }
        }
        if by.is_empty() {
            for spk in cited.iter().filter_map(|l| l.spk) {
                if !by.contains(&spk) {
                    by.push(spk);
                }
            }
        }
        let mut name = None;
        if kind == "intro" {
            match (&by[..], person_name(text("name"))) {
                ([_], Some(n)) => name = Some(n),
                _ => continue,
            }
        }
        let body = match (text("text"), &name) {
            ("", Some(n)) => format!("{n} introduced themselves"),
            ("", None) => continue,
            // Notes read as phrases: one final full stop is dropped, an ellipsis kept.
            (t, _) => t.strip_suffix('.').filter(|s| !s.ends_with('.')).unwrap_or(t).to_string(),
        };
        let revises = |old: &Event| old.kind == kind || (kind == "answer" && old.kind == "question");
        let supersedes = text("supersedes")
            .parse::<EventId>()
            .ok()
            .filter(|id| events.iter().any(|e| e.id == *id && revises(e)));
        out.push(Event {
            id: EventId(0),
            kind,
            t,
            by,
            text: body,
            refs,
            task: None,
            supersedes,
            name,
            rename: None,
        });
    }
    out
}

/// A list of strings, or one string, as models write either.
fn strings(v: Option<&Value>) -> Vec<String> {
    match v {
        Some(Value::String(s)) => vec![s.trim().to_string()],
        Some(Value::Array(a)) => a.iter().filter_map(Value::as_str).map(|s| s.trim().to_string()).collect(),
        _ => Vec::new(),
    }
}

/// A name as an introduction gives it: one to three words, each starting
/// with a capital letter.
fn person_name(s: &str) -> Option<String> {
    let words: Vec<&str> = s.split_whitespace().collect();
    let ok = (1..=3).contains(&words.len())
        && words.iter().all(|w| {
            w.chars().next().is_some_and(char::is_uppercase)
                && w.chars().all(|c| c.is_alphabetic() || c == '-' || c == '\'')
        });
    ok.then(|| words.join(" "))
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn line(id: u32, spk: u32, secs: (u64, u64), text: &str) -> Line {
        Line {
            id: UtteranceId(id),
            t0: secs.0 * SECOND,
            t1: secs.1 * SECOND,
            slot: Some(spk),
            spk: Some(SpkId(spk)),
            lang: None,
            text: text.into(),
        }
    }

    #[test]
    fn a_run_is_due_after_enough_speech_or_at_a_pause() {
        let lines = [line(1, 0, (0, 8), "a"), line(2, 1, (9, 20), "b"), line(3, 0, (21, 26), "c")];
        assert!(!due(&[], 100 * SECOND, false));
        assert!(!due(&lines[..2], 21 * SECOND, true), "20 s of speech, still talking");
        assert!(due(&lines, 27 * SECOND, true), "26 s of speech");
        assert!(!due(&lines[..1], 10 * SECOND, false), "only 2 s quiet");
        assert!(due(&lines[..1], 11 * SECOND, false), "3 s quiet");
        assert!(!due(&lines[..1], 30 * SECOND, true), "someone is speaking: no pause");
    }

    #[test]
    fn finds_the_json_in_a_reply() {
        let r =
            parse("Here you go:\n```json\n{\"events\": [{\"type\": \"point\"}], \"summary\": \" A. \"}\n```")
                .unwrap();
        assert_eq!(r, Reply { events: vec![json!({"type": "point"})], summary: Some("A.".into()) });
        assert_eq!(parse("{\"summary\": \"\"}").unwrap(), Reply::default());
        assert!(parse("Sorry, I can't.").unwrap_err().to_string().contains("no JSON"));
        assert!(parse("{\"events\": 3}").is_err());
    }

    #[test]
    fn keeps_only_events_that_hold_up() {
        let lines = [
            line(1, 0, (0, 4), "I'm Alice, I run product."),
            line(2, 1, (5, 9), "And this is Marcus, he does backend."),
            line(3, 2, (10, 14), "Single page is faster to build."),
        ];
        let speakers = BTreeSet::from([SpkId(0), SpkId(1), SpkId(2)]);
        let earlier = Event {
            id: EventId(4),
            kind: "question".into(),
            t: 0,
            by: vec![],
            text: "Which flow?".into(),
            refs: vec!["u0001".into()],
            task: None,
            supersedes: None,
            name: None,
            rename: None,
        };
        let reply = Reply {
            events: vec![
                json!({"type": "Point", "text": "Single page is faster.", "refs": ["u0003"]}),
                json!({"type": "answer", "by": "spk_2", "text": "Single page", "refs": ["u0002..u0003"], "supersedes": "e004"}),
                json!({"type": "intro", "by": ["spk_1"], "name": "Marcus", "refs": ["u0002"]}),
                json!({"type": "intro", "by": ["spk_1"], "name": "the backend guy", "refs": ["u0002"]}),
                json!({"type": "decision", "text": "Ship it", "refs": ["u0099"]}),
                json!({"type": "decision", "text": "Ship it", "refs": ["u0002..u0099"]}),
                json!({"type": "request", "text": "Mock it up", "refs": ["u0003"]}),
                json!({"type": "point", "by": ["spk_7"], "text": "Who?", "refs": ["u0001"], "supersedes": "e009"}),
                json!({"type": "point", "text": "", "refs": ["u0001"]}),
                json!({"type": "decision", "text": "Single page it is", "refs": ["u0003"], "supersedes": "e004"}),
            ],
            summary: None,
        };
        let got = check(&reply, &lines, UtteranceId(1), &speakers, std::slice::from_ref(&earlier));
        let brief: Vec<(&str, &str, Vec<u32>, Ts)> = got
            .iter()
            .map(|e| (e.kind.as_str(), e.text.as_str(), e.by.iter().map(|s| s.0).collect(), e.t))
            .collect();
        assert_eq!(
            brief,
            [
                ("point", "Single page is faster", vec![2], 10 * SECOND),
                ("answer", "Single page", vec![2], 5 * SECOND),
                ("intro", "Marcus introduced themselves", vec![1], 5 * SECOND),
                ("point", "Who?", vec![0], 0),
                ("decision", "Single page it is", vec![2], 10 * SECOND),
            ]
        );
        assert_eq!(got[1].supersedes, Some(EventId(4)), "an answer closes a question");
        assert_eq!(got[2].name.as_deref(), Some("Marcus"));
        assert_eq!(got[3].supersedes, None, "e009 doesn't exist");
        assert_eq!(got[4].supersedes, None, "a decision can't take a question's place");
        // With u0003 the only new line, events backed by earlier lines alone are context, not news.
        let kinds: Vec<String> = check(&reply, &lines, UtteranceId(3), &speakers, &[earlier])
            .into_iter()
            .map(|e| e.kind)
            .collect();
        assert_eq!(kinds, ["point", "answer", "decision"]);
    }

    #[test]
    fn the_prompt_shows_people_open_items_and_the_new_lines() {
        let lines = [line(1, 0, (0, 4), "I'm Alice."), line(2, 1, (65, 70), "Do we have the numbers?")];
        let q = Event {
            id: EventId(1),
            kind: "question".into(),
            t: 65 * SECOND,
            by: vec![SpkId(1)],
            text: "Do we have mobile numbers?".into(),
            refs: vec!["u0002".into()],
            task: None,
            supersedes: None,
            name: None,
            rename: None,
        };
        let p = prompt(&Input {
            title: "Sync",
            people: vec![(SpkId(0), Some("Alice".into())), (SpkId(1), None)],
            summary: "",
            events: &[q],
            earlier: &lines[..1],
            new: &lines[1..],
        });
        assert!(p.contains("PEOPLE\nspk_0 Alice\nspk_1 (no name yet)\n"), "{p}");
        assert!(
            p.contains("OPEN QUESTIONS AND ACTIONS\ne001 question [spk_1]: Do we have mobile numbers?\n"),
            "{p}"
        );
        assert!(
            p.contains(
                "EARLIER LINES (already recorded, for context)\nu0001 00:00 spk_0 Alice: I'm Alice.\n"
            ),
            "{p}"
        );
        assert!(p.ends_with("NEW LINES\nu0002 01:05 spk_1: Do we have the numbers?\n"), "{p}");
        assert!(system_prompt("Jarvis").contains("\"Jarvis, note that...\""));
    }
}
