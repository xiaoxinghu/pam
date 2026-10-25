//! The agent's read commands: `status`, `context`, `people`, `transcript`,
//! `get`, `around`, `search` and `events`. The core reads and queries the folder;
//! this file checks arguments and prints the result as text or `--json`.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::Context;
use pam_core::clock::{SECOND, mmss};
use pam_core::folder::meetings_root;
use pam_core::question::{Answer, By, Record};
use pam_core::record::{self, EVENT_TYPES, Event, Hit, Item, MeetingRecord, Person, Query, Ref, Window};
use pam_core::{Line, ModelInfo, ModelStatus, SpkId, TaskState, TaskStatus, Ts, Usage, Via};
use serde_json::{Value, json};

use crate::cli::{Command, Fail, Parsed};

/// `pam around` defaults.
const BEFORE: Ts = 60 * SECOND;
const AFTER: Ts = 30 * SECOND;
/// `pam search` default.
const LIMIT: usize = 20;

pub fn run(cmd: &'static Command, p: Parsed, out: &mut String) -> Result<(), Fail> {
    // Check the command line before looking for a meeting, so a usage error
    // is reported as one even when there is no meeting.
    let read = Read::check(cmd, &p)?;
    let rec = meeting(&p)?;
    let json = p.has("--json");
    match read {
        Read::Status => status(&rec, json, out),
        Read::Context => context(&rec, json, out),
        Read::People => people(&rec, json, out),
        Read::Transcript { window, speaker } => transcript(&rec, window, speaker.as_deref(), json, out),
        Read::Get(refs) => get(&rec, &refs, json, out),
        Read::Around { at, before, after } => around(&rec, at, before, after, json, out),
        Read::Search { query, speaker, types, limit } => {
            search(&rec, &query, speaker.as_deref(), types, limit, json, out)
        }
        Read::Events { types, since } => events(&rec, types, since, json, out),
    }
}

/// A read command with its arguments checked.
enum Read {
    Status,
    Context,
    People,
    Transcript { window: Window, speaker: Option<String> },
    Get(Vec<Ref>),
    Around { at: Ref, before: Ts, after: Ts },
    Search { query: String, speaker: Option<String>, types: Option<Vec<String>>, limit: usize },
    Events { types: Option<Vec<String>>, since: Option<Ts> },
}

impl Read {
    fn check(cmd: &'static Command, p: &Parsed) -> Result<Read, Fail> {
        let usage = |msg: String| Fail::usage(cmd, msg);
        let no_positionals = || match p.positionals.first() {
            Some(extra) => Err(usage(format!("unexpected `{extra}`: `pam {}` takes no arguments", cmd.name))),
            None => Ok(()),
        };
        Ok(match cmd.name {
            "status" => no_positionals().map(|()| Read::Status)?,
            "context" => no_positionals().map(|()| Read::Context)?,
            "people" => no_positionals().map(|()| Read::People)?,
            "transcript" => {
                if let Some(extra) = p.positionals.first() {
                    return Err(usage(format!(
                        "unexpected `{extra}`: to find lines by their words, use `pam search {extra}`"
                    )));
                }
                let window = match (p.value("--last"), p.value("--since"), p.value("--from"), p.value("--to"))
                {
                    (None, None, None, None) => Window::All,
                    (Some(n), None, None, None) => {
                        Window::Last(count(n).map_err(|m| usage(format!("--last {m}")))?)
                    }
                    (None, Some(d), None, None) => {
                        Window::Since(duration(d).map_err(|m| usage(format!("--since {m}")))?)
                    }
                    (None, None, from, to) => {
                        let at = |name: &str, t: Option<&str>| {
                            t.map(|t| time(t).map_err(|m| usage(format!("{name} {m}")))).transpose()
                        };
                        let (from, to) = (at("--from", from)?, at("--to", to)?);
                        if let (Some(f), Some(t)) = (from, to)
                            && f >= t
                        {
                            return Err(usage(format!("--from {} is not before --to {}", mmss(f), mmss(t))));
                        }
                        Window::Between { from, to }
                    }
                    _ => return Err(usage("use one of --last, --since or --from/--to, not several".into())),
                };
                Read::Transcript { window, speaker: p.value("--speaker").map(str::to_string) }
            }
            "get" => {
                if p.positionals.is_empty() {
                    return Err(usage(
                        "missing <ref>: which lines, events, speakers or tasks should I get?".into(),
                    ));
                }
                Read::Get(p.positionals.iter().map(|r| r.parse().map_err(usage)).collect::<Result<_, _>>()?)
            }
            "around" => {
                let at: Ref = match &p.positionals[..] {
                    [] => {
                        return Err(usage(
                            "missing <ref>: which line or event should I show context around?".into(),
                        ));
                    }
                    [r] => r.parse().map_err(usage)?,
                    more => {
                        return Err(usage(format!(
                            "around takes one ref, not {} ({}); `pam get` takes several",
                            more.len(),
                            more.join(" ")
                        )));
                    }
                };
                match at {
                    Ref::Speaker(s) => {
                        return Err(usage(format!(
                            "{s} is a speaker, not a moment: try `pam transcript --speaker {s}`"
                        )));
                    }
                    Ref::Task(t) => {
                        return Err(usage(format!("{t} is a task, not a moment: try `pam get {t}`")));
                    }
                    _ => {}
                }
                let dur = |name: &str, default: Ts| {
                    p.value(name)
                        .map_or(Ok(default), |d| duration(d).map_err(|m| usage(format!("{name} {m}"))))
                };
                Read::Around { at, before: dur("--before", BEFORE)?, after: dur("--after", AFTER)? }
            }
            "search" => {
                let query = p.positionals.join(" ");
                if Query::words(&query).is_empty() {
                    let msg = if query.trim().is_empty() {
                        "missing <query>: what words should I look for?".to_string()
                    } else {
                        format!("`{query}` has no words to look for")
                    };
                    return Err(usage(msg));
                }
                let types = p.value("--type").map(|t| types(t).map_err(usage)).transpose()?;
                let limit =
                    p.value("--limit").map_or(Ok(LIMIT), count).map_err(|m| usage(format!("--limit {m}")))?;
                Read::Search { query, speaker: p.value("--speaker").map(str::to_string), types, limit }
            }
            "events" => {
                no_positionals()?;
                let types =
                    p.value("--type").map(|t| types(t).and_then(no_lines).map_err(usage)).transpose()?;
                let since = p
                    .value("--since")
                    .map(|d| duration(d).map_err(|m| usage(format!("--since {m}"))))
                    .transpose()?;
                Read::Events { types, since }
            }
            other => unreachable!("`{other}` is not a read command"),
        })
    }
}

fn status(rec: &MeetingRecord, json: bool, out: &mut String) -> Result<(), Fail> {
    let people = rec.people();
    let open = rec.open_items();
    let span = rec.lines.first().zip(rec.lines.last());
    // The tasks still going, and the ones with a result to look at.
    let listed: Vec<&TaskState> = rec.tasks.iter().filter(|t| t.status.is_open() || t.is_ready()).collect();
    let count = |status: TaskStatus| rec.tasks.iter().filter(|t| t.status == status).count();
    let (running, queued) = (count(TaskStatus::Running), count(TaskStatus::Queued));
    let ready = rec.tasks.iter().filter(|t| t.is_ready()).count();
    if json {
        return print_json(
            out,
            json!({
                "title": rec.info.title,
                "folder": rec.dir,
                "status": rec.info.status,
                "started_at": rec.info.started_at,
                "elapsed": rec.now(),
                "people": people.iter().map(person_json).collect::<Vec<_>>(),
                "lines": rec.lines.len(),
                "first_line": span.map(|(a, _)| a.id),
                "last_line": span.map(|(_, b)| b.id),
                "events": rec.events.len(),
                "open": open.iter().map(|e| event_json(rec, e)).collect::<Vec<_>>(),
                "tasks": {
                    "total": rec.tasks.len(),
                    "running": running,
                    "queued": queued,
                    "ready": ready,
                    "listed": listed,
                },
                "scribe": rec.info.scribe,
                "intent": rec.info.intent,
                "agent": rec.info.agent,
                "usage": rec.info.usage,
            }),
        );
    }
    let title = if rec.info.title.is_empty() { "Meeting" } else { &rec.info.title };
    writeln!(out, "{title} · {} elapsed · {}", mmss(rec.now()), rec.info.status).unwrap();
    writeln!(out, "folder  {}", rec.dir.display()).unwrap();
    let models = [("scribe", &rec.info.scribe), ("intent", &rec.info.intent), ("agent ", &rec.info.agent)];
    for (what, model) in models {
        if let Some(m) = model {
            writeln!(out, "{what}  {}", model_status(m)).unwrap();
        }
    }
    writeln!(out, "cost    {}", cost(&rec.info.usage)).unwrap();
    let who: Vec<String> = people.iter().map(who_with_id).collect();
    writeln!(out, "people  {}", if who.is_empty() { "nobody yet".into() } else { who.join(", ") }).unwrap();
    match span {
        Some((a, b)) => writeln!(out, "lines   {} ({}..{})", rec.lines.len(), a.id, b.id),
        None => writeln!(out, "lines   none yet"),
    }
    .unwrap();
    if rec.events.is_empty() {
        writeln!(out, "events  none yet").unwrap();
    } else {
        writeln!(out, "events  {}", rec.events.len()).unwrap();
        if open.is_empty() {
            writeln!(out, "open    nothing").unwrap();
        } else {
            writeln!(out, "open    {}", open.len()).unwrap();
        }
        for e in open {
            writeln!(out, "  {}", rec.event_text(e)).unwrap();
        }
    }
    if rec.tasks.is_empty() {
        writeln!(out, "tasks   none yet").unwrap();
    } else {
        writeln!(out, "tasks   {} ({running} running, {queued} queued, {ready} ready)", rec.tasks.len())
            .unwrap();
    }
    for t in listed {
        writeln!(out, "  {}", rec.task_text(t)).unwrap();
    }
    Ok(())
}

/// The meeting's cost so far, in total and per use: `$0.20 (scribe
/// $0.12, intent $0.00, agent $0.08)`, with `≈` for API prices on a
/// subscription login and `+` when partial.
fn cost(u: &Usage) -> String {
    format!(
        "{} (scribe {}, intent {}, agent {})",
        u.label(),
        u.scribe.dollars(),
        u.intent.dollars(),
        u.agent.dollars()
    )
}

/// Whether the scribe, the intent check or the agent works, as
/// `meeting.json` recorded it at the start: `ready (pi 1.1.0, <model>,
/// <source>)`, `ready (…), but <warning>`, `off`, or `unavailable: <why>`.
/// The local model runs in pam's process, with no pi.
fn model_status(m: &ModelInfo) -> String {
    let pi = m.version.as_ref().map(|v| format!("pi {v}"));
    let source = m.source.map(|s| s.as_str().to_string());
    let about = [pi, m.model.clone(), source].into_iter().flatten().collect::<Vec<_>>().join(", ");
    match (m.status, m.reason.as_deref()) {
        (ModelStatus::Off, _) => "off".into(),
        (ModelStatus::Unavailable, why) => format!("unavailable: {}", why.unwrap_or("unknown reason")),
        (ModelStatus::Ready | ModelStatus::Warning, Some(warning)) => {
            format!("ready ({about}), but {warning}")
        }
        (ModelStatus::Ready | ModelStatus::Warning, None) => format!("ready ({about})"),
    }
}

fn context(rec: &MeetingRecord, json: bool, out: &mut String) -> Result<(), Fail> {
    let text = pam_core::context::read(rec)?;
    if json {
        return print_json(out, json!({ "text": text, "tokens": pam_core::context::tokens(&text) }));
    }
    out.push_str(&text);
    Ok(())
}

fn people(rec: &MeetingRecord, json: bool, out: &mut String) -> Result<(), Fail> {
    let people = rec.people();
    if json {
        return print_json(out, json!({ "people": people.iter().map(person_json).collect::<Vec<_>>() }));
    }
    if people.is_empty() {
        out.push_str("nobody has spoken yet\n");
        return Ok(());
    }
    let rows: Vec<[String; 5]> = people
        .iter()
        .map(|p| {
            [
                p.spk.to_string(),
                p.name.clone().unwrap_or_else(|| "-".into()),
                p.first_heard.map_or("-".into(), mmss),
                mmss(p.talk_time),
                p.lines.to_string(),
            ]
        })
        .collect();
    table(out, ["ID", "NAME", "FIRST HEARD", "TALK", "LINES"], &rows);
    Ok(())
}

fn transcript(
    rec: &MeetingRecord,
    window: Window,
    speaker: Option<&str>,
    json: bool,
    out: &mut String,
) -> Result<(), Fail> {
    let speakers = speaker.map(|s| find_speakers(rec, s)).transpose()?;
    let lines = rec.transcript(window, speakers.as_deref());
    if json {
        return print_json(
            out,
            json!({ "lines": lines.iter().map(|l| line_json(rec, l)).collect::<Vec<_>>() }),
        );
    }
    if lines.is_empty() {
        out.push_str(if rec.lines.is_empty() { "no lines yet\n" } else { "no lines match\n" });
    }
    for l in lines {
        write_line(out, rec, l);
    }
    Ok(())
}

fn get(rec: &MeetingRecord, refs: &[Ref], json: bool, out: &mut String) -> Result<(), Fail> {
    let mut items = Vec::new();
    let mut missing = Vec::new();
    for &r in refs {
        match rec.get(r) {
            Ok(found) => items.extend(found),
            Err(e) => missing.push(e.0),
        }
    }
    if json {
        let mut shown = Vec::new();
        for item in &items {
            let mut v = item_json(rec, item);
            // A task comes with what it said and asked.
            if let Item::Task(t) = item {
                v["messages"] = json!(rec.task_messages(t.id)?);
                v["questions"] = json!(rec.task_questions(t.id)?);
            }
            shown.push(v);
        }
        print_json(out, json!({ "items": shown }))?;
    } else {
        for item in &items {
            match item {
                Item::Line(l) => write_line(out, rec, l),
                Item::Event(e) => write_event(out, rec, e),
                Item::Speaker(p) => write_person(out, p),
                Item::Task(t) => write_task(out, rec, t)?,
            }
        }
    }
    match missing.len() {
        0 => Ok(()),
        _ => Err(Fail::NotFound(missing.join("\nerror: "))),
    }
}

fn around(
    rec: &MeetingRecord,
    at: Ref,
    before: Ts,
    after: Ts,
    json: bool,
    out: &mut String,
) -> Result<(), Fail> {
    let anchor = rec.anchor(at).map_err(|e| Fail::NotFound(e.0))?;
    let lines = rec.around(&anchor, before, after);
    if json {
        let lines: Vec<Value> = lines
            .iter()
            .map(|l| {
                let mut v = line_json(rec, l);
                v["anchor"] = anchor.lines.contains(&l.id).into();
                v
            })
            .collect();
        return print_json(
            out,
            json!({
                "ref": at.to_string(),
                "t0": anchor.t0,
                "t1": anchor.t1,
                "event": anchor.event.map(|e| event_json(rec, e)),
                "lines": lines,
            }),
        );
    }
    if let Some(e) = anchor.event {
        write_event(out, rec, e);
    }
    if lines.is_empty() {
        writeln!(out, "no lines within {} before and {} after", secs(before), secs(after)).unwrap();
    }
    for l in lines {
        out.push_str(if anchor.lines.contains(&l.id) { "> " } else { "  " });
        write_line(out, rec, l);
    }
    Ok(())
}

fn search(
    rec: &MeetingRecord,
    query: &str,
    speaker: Option<&str>,
    types: Option<Vec<String>>,
    limit: usize,
    json: bool,
    out: &mut String,
) -> Result<(), Fail> {
    let speakers = speaker.map(|s| find_speakers(rec, s)).transpose()?;
    let hits = rec.search(&Query { words: Query::words(query), speakers, types });
    let shown = &hits[..hits.len().min(limit)];
    if json {
        let results: Vec<Value> = shown
            .iter()
            .map(|h| match h {
                Hit::Line(l) => item_json(rec, &Item::Line(l)),
                Hit::Event(e) => item_json(rec, &Item::Event(e)),
            })
            .collect();
        return print_json(out, json!({ "query": query, "total": hits.len(), "results": results }));
    }
    if hits.is_empty() {
        writeln!(out, "nothing matches `{query}`").unwrap();
    }
    for h in shown {
        match h {
            Hit::Line(l) => write_line(out, rec, l),
            Hit::Event(e) => write_event(out, rec, e),
        }
    }
    if shown.len() < hits.len() {
        writeln!(out, "({} of {} shown; raise --limit for more)", shown.len(), hits.len()).unwrap();
    }
    Ok(())
}

fn events(
    rec: &MeetingRecord,
    types: Option<Vec<String>>,
    since: Option<Ts>,
    json: bool,
    out: &mut String,
) -> Result<(), Fail> {
    let from = since.map(|d| rec.now().saturating_sub(d));
    let shown: Vec<&Event> = rec
        .events
        .iter()
        .filter(|e| types.as_ref().is_none_or(|t| t.contains(&e.kind)) && from.is_none_or(|f| e.t >= f))
        .collect();
    if json {
        return print_json(
            out,
            json!({ "events": shown.iter().map(|e| event_json(rec, e)).collect::<Vec<_>>() }),
        );
    }
    if shown.is_empty() {
        out.push_str(if rec.events.is_empty() { "no events yet\n" } else { "no events match\n" });
    }
    for e in shown {
        write_event(out, rec, e);
    }
    Ok(())
}

/// The meeting a command reads: `--meeting`, else `$PAM_MEETING`, else the latest.
pub fn meeting(p: &Parsed) -> Result<MeetingRecord, Fail> {
    let env = std::env::var_os("PAM_MEETING").map(std::path::PathBuf::from);
    let dir = record::choose_meeting(p.value("--meeting").map(Path::new), env.as_deref(), meetings_root)?
        .map_err(Fail::NoMeeting)?;
    Ok(MeetingRecord::load(&dir).with_context(|| format!("reading the meeting in {}", dir.display()))?)
}

/// The speakers a `--speaker` value means, or a not-found error listing who exists.
fn find_speakers(rec: &MeetingRecord, who: &str) -> Result<Vec<SpkId>, Fail> {
    let found = rec.speakers_matching(who);
    if found.is_empty() {
        return Err(Fail::NotFound(rec.no_speaker(who).0));
    }
    Ok(found)
}

fn who_with_id(p: &Person) -> String {
    match &p.name {
        Some(n) => format!("{n} ({})", p.spk),
        None => p.spk.to_string(),
    }
}

fn write_line(out: &mut String, rec: &MeetingRecord, l: &Line) {
    writeln!(out, "{}", rec.line_text(l)).unwrap();
}

fn write_event(out: &mut String, rec: &MeetingRecord, e: &Event) {
    writeln!(out, "{}", rec.event_text(e)).unwrap();
}

/// `spk_0 Alice · first heard 00:01 · talked 00:12 · 4 lines`
fn write_person(out: &mut String, p: &Person) {
    write!(out, "{}", p.spk).unwrap();
    if let Some(n) = &p.name {
        write!(out, " {n}").unwrap();
    }
    match p.first_heard {
        Some(t) => {
            writeln!(out, " · first heard {} · talked {} · {} lines", mmss(t), mmss(p.talk_time), p.lines)
        }
        None => writeln!(out, " · no lines yet"),
    }
    .unwrap();
}

/// A task in full, as `pam get t02` prints it: the line `pam status` gives,
/// then who asked and when, how it went, what the room corrected, what it
/// said on the stage, what it asked the room and how each question ended,
/// and its result.
fn write_task(out: &mut String, rec: &MeetingRecord, t: &TaskState) -> Result<(), Fail> {
    writeln!(out, "{}", rec.task_text(t)).unwrap();
    let how = match t.by {
        Via::Voice => "said",
        Via::Typed => "typed on the laptop",
    };
    let who = match t.by {
        Via::Voice => format!(" by {}", rec.asker(t)),
        Via::Typed => String::new(),
    };
    writeln!(out, "  asked     {}, {how}{who}", mmss(t.asked_at)).unwrap();
    let mut ran = Vec::new();
    if let Some(at) = t.started_at {
        ran.push(format!("started {}", mmss(at)));
    }
    if let Some(at) = t.ended_at {
        ran.push(format!("ended {}", mmss(at)));
    }
    if let (Some(a), Some(b)) = (t.started_at, t.ended_at) {
        ran.push(format!("took {}", secs(b.saturating_sub(a))));
    }
    writeln!(out, "  status    {}", [vec![t.shown_status().to_string()], ran].concat().join(" · ")).unwrap();
    if let Some(e) = &t.error {
        writeln!(out, "  error     {e}").unwrap();
    }
    if t.progress.is_some() || t.action.is_some() {
        let progress = t.progress.map(|p| format!("{p}%"));
        let shown: Vec<String> = progress.into_iter().chain(t.action.clone()).collect();
        writeln!(out, "  progress  {}", shown.join(" · ")).unwrap();
    }
    for c in &t.corrections {
        writeln!(out, "  then      {c}").unwrap();
    }
    for m in rec.task_messages(t.id)? {
        let at = rec.meeting_time(&m.at).map_or_else(|| m.at.clone(), mmss);
        writeln!(out, "  said      {at} {}", m.text).unwrap();
    }
    let records = rec.task_questions(t.id)?;
    for r in &records {
        let Record::Asked { q, at, text, options, .. } = r else { continue };
        let at = rec.meeting_time(at).map_or_else(|| at.clone(), mmss);
        let choices = if options.is_empty() {
            " (open)".to_string()
        } else {
            let numbered: Vec<String> =
                options.iter().enumerate().map(|(i, o)| format!("{} {o}", i + 1)).collect();
            format!(" [{}]", numbered.join(" · "))
        };
        let end = records.iter().find(|e| e.q() == *q && !matches!(e, Record::Asked { .. }));
        let ended = match end {
            None => "waiting".to_string(),
            Some(Record::Answered { answer, by, spk, .. }) => {
                let by = match (by, spk) {
                    (By::Key, _) => "on the laptop".to_string(),
                    (By::Typed, _) => "typed on the laptop".to_string(),
                    (By::Voice, Some(s)) => format!("by {}", rec.speakers.overlay.label(Some(*s))),
                    (By::Voice, None) => "by voice".to_string(),
                };
                match answer {
                    Answer::Option(n) => {
                        let text = options.get(n.wrapping_sub(1)).map_or("", String::as_str);
                        format!("option {n}, {text}, {by}")
                    }
                    Answer::Text(words) => format!("“{words}”, {by}"),
                }
            }
            Some(Record::TimedOut { .. }) => "timeout".to_string(),
            Some(Record::Dropped { why, .. }) => format!("dropped: {why}"),
            Some(Record::Asked { .. }) => unreachable!(),
        };
        writeln!(out, "  q{q}        {at} {text}{choices} → {ended}").unwrap();
    }
    if let Some(r) = &t.result {
        writeln!(out, "  result    {}: {}", r.title, r.target.as_str()).unwrap();
    }
    Ok(())
}

fn table<const N: usize>(out: &mut String, head: [&str; N], rows: &[[String; N]]) {
    let mut widths = head.map(str::len);
    for row in rows {
        for (w, cell) in widths.iter_mut().zip(row) {
            *w = (*w).max(cell.chars().count());
        }
    }
    let mut row_out = |cells: &mut dyn Iterator<Item = &str>| {
        let line: Vec<String> = cells.zip(widths).map(|(c, w)| format!("{c:w$}")).collect();
        writeln!(out, "{}", line.join("  ").trim_end()).unwrap();
    };
    row_out(&mut head.iter().copied());
    for row in rows {
        row_out(&mut row.iter().map(String::as_str));
    }
}

fn line_json(rec: &MeetingRecord, l: &Line) -> Value {
    json!({
        "id": l.id,
        "t0": l.t0,
        "t1": l.t1,
        "slot": l.slot,
        "spk": l.spk,
        "name": l.spk.and_then(|s| rec.name(s)),
        "lang": l.lang,
        "text": l.text,
    })
}

fn event_json(rec: &MeetingRecord, e: &Event) -> Value {
    let mut v = serde_json::to_value(e).unwrap();
    let by = rec.event_speakers(e);
    v["by"] = json!(by);
    v["names"] = json!(by.iter().map(|&s| rec.name(s)).collect::<Vec<_>>());
    v["superseded_by"] = json!(rec.superseded_by(e.id));
    v
}

fn person_json(p: &Person) -> Value {
    serde_json::to_value(p).unwrap()
}

fn item_json(rec: &MeetingRecord, item: &Item) -> Value {
    let (kind, mut v) = match item {
        Item::Line(l) => ("line", line_json(rec, l)),
        Item::Event(e) => ("event", event_json(rec, e)),
        Item::Speaker(p) => ("speaker", person_json(p)),
        Item::Task(t) => ("task", serde_json::to_value(t).unwrap()),
    };
    v["kind"] = kind.into();
    v
}

fn print_json(out: &mut String, v: Value) -> Result<(), Fail> {
    out.push_str(&serde_json::to_string_pretty(&v).unwrap());
    out.push('\n');
    Ok(())
}

/// A positive whole number.
fn count(s: &str) -> Result<usize, String> {
    match s.parse::<usize>() {
        Ok(n) if n > 0 => Ok(n),
        _ => Err(format!("needs a positive whole number, not `{s}`")),
    }
}

/// `90s`, `5m`, `1h`, `1m30s`, or plain seconds.
pub fn duration(s: &str) -> Result<Ts, String> {
    let bad = || format!("needs a duration like 90s, 5m, 1h or 1m30s, not `{s}`");
    if !s.is_empty() && s.bytes().all(|b| b.is_ascii_digit()) {
        return to_ts(number(s, s).ok_or_else(bad)??, s);
    }
    let (mut total, mut num) = (0u64, String::new());
    for c in s.chars() {
        if c.is_ascii_digit() {
            num.push(c);
            continue;
        }
        let unit = match c {
            'h' => 3600,
            'm' => 60,
            's' => 1,
            _ => return Err(bad()),
        };
        let n = number(&num, s).ok_or_else(bad)??;
        total = n.checked_mul(unit).and_then(|n| n.checked_add(total)).ok_or_else(|| too_long(s))?;
        num.clear();
    }
    if !num.is_empty() || s.is_empty() {
        return Err(bad());
    }
    to_ts(total, s)
}

/// A meeting time: `mm:ss`, `h:mm:ss`, or a duration from the start.
fn time(s: &str) -> Result<Ts, String> {
    let bad = || format!("needs a meeting time like 03:10 or 1:02:05, not `{s}`");
    if !s.contains(':') {
        return duration(s).map_err(|e| if e == too_long(s) { e } else { bad() });
    }
    let parts: Option<Vec<u64>> =
        s.split(':').map(|p| number(p, s)).collect::<Option<Result<_, _>>>().transpose()?;
    let secs = match parts.as_deref() {
        Some(&[m, sec]) if sec < 60 => m.checked_mul(60).and_then(|m| m.checked_add(sec)),
        Some(&[h, m, sec]) if m < 60 && sec < 60 => {
            h.checked_mul(3600).and_then(|h| h.checked_add(m * 60 + sec))
        }
        _ => return Err(bad()),
    };
    to_ts(secs.ok_or_else(|| too_long(s))?, s)
}

/// `digits` as a number: `None` unless it is one or more ASCII digits, and
/// too long (for the whole value `s`) if it doesn't fit a `u64`.
fn number(digits: &str, s: &str) -> Option<Result<u64, String>> {
    if digits.is_empty() || !digits.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    Some(digits.parse().map_err(|_| too_long(s)))
}

/// Whole seconds to samples, refusing what a `Ts` can't hold.
fn to_ts(secs: u64, s: &str) -> Result<Ts, String> {
    secs.checked_mul(SECOND).ok_or_else(|| too_long(s))
}

fn too_long(s: &str) -> String {
    format!("is too long: `{s}` is longer than any meeting")
}

/// Comma-separated event types, or `line`.
fn types(s: &str) -> Result<Vec<String>, String> {
    s.split(',')
        .map(|t| t.trim().to_lowercase())
        .filter(|t| !t.is_empty())
        .map(|t| {
            if t == "line" || EVENT_TYPES.contains(&t.as_str()) {
                Ok(t)
            } else {
                Err(format!("unknown type `{t}`: types are {}, or line", EVENT_TYPES.join(", ")))
            }
        })
        .collect::<Result<Vec<_>, _>>()
        .and_then(|t| if t.is_empty() { Err("--type needs at least one type".into()) } else { Ok(t) })
}

/// Event types only: `pam events` lists no transcript lines.
fn no_lines(types: Vec<String>) -> Result<Vec<String>, String> {
    if types.iter().any(|t| t == "line") {
        return Err("`line` isn't an event type; for lines use `pam transcript` or `pam search`".into());
    }
    Ok(types)
}

/// A duration for display: `60s`, `2m`, `1m30s`.
fn secs(ts: Ts) -> String {
    let s = ts / SECOND;
    match (s / 60, s % 60) {
        (0, s) => format!("{s}s"),
        (m, 0) => format!("{m}m"),
        (m, s) => format!("{m}m{s}s"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_durations_and_times() {
        assert_eq!(duration("90s"), Ok(90 * SECOND));
        assert_eq!(duration("1m30s"), Ok(90 * SECOND));
        assert_eq!(duration("2m"), Ok(120 * SECOND));
        assert_eq!(duration("1h"), Ok(3600 * SECOND));
        assert_eq!(duration("0s"), Ok(0));
        assert_eq!(duration("45"), Ok(45 * SECOND));
        assert!(duration("5 minutes").is_err());
        assert!(duration("m").is_err());
        assert!(duration("5m3").is_err());
        assert_eq!(time("03:10"), Ok(190 * SECOND));
        assert_eq!(time("1:02:05"), Ok(3725 * SECOND));
        assert_eq!(time("90s"), Ok(90 * SECOND));
        assert!(time("03:75").is_err());
        assert!(time("3:x").is_err());
        // Too long for a 16 kHz u64 sample index: a usage error, never an overflow.
        let max = u64::MAX.to_string();
        assert_eq!(duration(&max), Err(too_long(&max)));
        assert_eq!(duration("1152921504606846976s"), Err(too_long("1152921504606846976s")));
        assert_eq!(duration("9999999999999999999h"), Err(too_long("9999999999999999999h")));
        assert_eq!(duration("18446744073709551615s1s"), Err(too_long("18446744073709551615s1s")));
        assert_eq!(time(&max), Err(too_long(&max)));
        assert_eq!(time(&format!("{max}:00")), Err(too_long(&format!("{max}:00"))));
        assert_eq!(time("9999999999999999999:00:00"), Err(too_long("9999999999999999999:00:00")));
        // Digits past u64 are too long too, not malformed.
        for s in ["18446744073709551616", "18446744073709551616s", "1m18446744073709551616s"] {
            assert_eq!(duration(s), Err(too_long(s)), "{s}");
        }
        assert_eq!(time("18446744073709551616:00"), Err(too_long("18446744073709551616:00")));
        assert!(duration("+5").is_err() && time("+5:00").is_err());
    }

    #[test]
    fn checks_types() {
        assert_eq!(types("decision, action"), Ok(vec!["decision".into(), "action".into()]));
        assert_eq!(types("line"), Ok(vec!["line".into()]));
        assert!(types("decisions").unwrap_err().contains("decision, action"));
        assert!(types(",").is_err());
    }
}
