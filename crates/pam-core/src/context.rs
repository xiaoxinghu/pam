//! `context.md`, the context pack: what an agent reads first about a
//! meeting, in about 6k tokens, so you can read exactly what Pam knows.
//!
//! Rendered from the meeting folder as it is, like the other rendered files.
//! Each section has a share of the budget. The events take what the others
//! leave, and when they don't fit, the oldest collapse first: down to their
//! topics and decisions, then into one line.

use std::collections::BTreeSet;
use std::fmt::Write as _;

use anyhow::{Context, Result};

use crate::Ts;
use crate::clock::{SECOND, mmss};
use crate::event::SpkId;
use crate::record::{Event, MeetingRecord, Window, read_if_present};
use crate::task::TaskStatus;

/// The whole pack, in estimated tokens.
const BUDGET: usize = 6_000;
/// The stretch at the end that is shown word for word.
pub const RECENT: Ts = 3 * 60 * SECOND;
/// Shares of the budget for the sections that can grow, besides the events.
const OPEN_SHARE: usize = 1_000;
/// The recent lines' share, which a short gap between prompts also gets.
pub const RECENT_SHARE: usize = 2_500;
/// The newest events never collapse.
const NEWEST: usize = 10;

const DIG: &str = "\
## How to dig

- `pam around <id>`: the exact words around a line or event
- `pam search <words>`: where something was said
- `pam get <ref>`: lines, events, speakers or tasks by ID, collapsed events included
- `pam events --type topic,decision`: every topic and decision
- `pam transcript --since 10m`: more of the transcript
";

/// About how many tokens `text` is: one per 2.5 bytes. Claude Haiku 5.5's
/// tokenizer gave 2.56–2.69 bytes per token on event lines, transcript lines
/// and prose, well under the usual rule of four. Pam has no
/// tokenizer, so the budget is approximate.
pub fn tokens(text: &str) -> usize {
    (text.len() * 2).div_ceil(5)
}

/// What `pam context` prints: `context.md` as the meeting last rendered it,
/// or rendered now when the file is not there.
pub fn read(rec: &MeetingRecord) -> Result<String> {
    let path = rec.dir.join("context.md");
    match read_if_present(&path)? {
        Some(bytes) => String::from_utf8(bytes).with_context(|| format!("reading {}", path.display())),
        None => Ok(render(rec)),
    }
}

/// `context.md` for the meeting as `rec` has it.
pub fn render(rec: &MeetingRecord) -> String {
    let now = rec.now();
    let mut head = String::new();
    match rec.info.title.as_str() {
        "" => write!(head, "# Meeting"),
        title => write!(head, "# Meeting: {title}"),
    }
    .unwrap();
    writeln!(head, " · {} elapsed · assistant name: {}", mmss(now), rec.info.assistant_name).unwrap();
    head.push_str("\n<!-- Rendered from the meeting folder on every change. Do not edit. -->\n");
    write!(head, "\n## People\n\n{}\n", people(rec)).unwrap();
    write!(head, "\n## Open\n\n{}", open(rec)).unwrap();
    write!(head, "\n## Tasks\n\n{}", tasks(rec)).unwrap();
    let recent = format!("\n## Last 3 min\n\n{}", recent(rec));
    let events_head = "\n## Events\n\n";
    let used = tokens(&head) + tokens(events_head) + tokens(&recent) + tokens(DIG) + 1;
    let events = events(rec, &rec.events, BUDGET.saturating_sub(used));
    format!("{head}{events_head}{events}{recent}\n{DIG}")
}

/// `spk_0 Alice · spk_1 Marcus (also spk_4) · spk_2`: everyone, with the IDs
/// merged into them.
fn people(rec: &MeetingRecord) -> String {
    let overlay = &rec.speakers.overlay;
    let people = rec.people();
    if people.is_empty() {
        return "Nobody has spoken yet.".into();
    }
    let shown: Vec<String> = people
        .iter()
        .map(|p| {
            let mut out = p.spk.to_string();
            if let Some(name) = &p.name {
                write!(out, " {name}").unwrap();
            }
            let also: BTreeSet<SpkId> = rec
                .speakers
                .slots
                .values()
                .copied()
                .filter(|&s| s != p.spk && overlay.follow(s) == p.spk)
                .collect();
            let also: Vec<String> = also.iter().map(SpkId::to_string).collect();
            if !also.is_empty() {
                write!(out, " (also {})", also.join(", ")).unwrap();
            }
            out
        })
        .collect();
    shown.join(" · ")
}

/// Every task but the cancelled ones, one line each, oldest first, as
/// `pam status` lists them. The section is small and never collapses: one
/// task runs at a time, and each takes a line.
fn tasks(rec: &MeetingRecord) -> String {
    let shown: Vec<String> = rec
        .tasks
        .iter()
        .filter(|t| t.status != TaskStatus::Cancelled)
        .map(|t| format!("- {}\n", rec.task_text(t)))
        .collect();
    if shown.is_empty() {
        return "None yet.\n".into();
    }
    shown.concat()
}

/// Open questions and actions, oldest first. Over their share, the oldest
/// give way to a list of their IDs.
fn open(rec: &MeetingRecord) -> String {
    let open = rec.open_items();
    if open.is_empty() {
        return "Nothing open.\n".into();
    }
    let lines: Vec<String> = open.iter().map(|e| format!("- {}\n", rec.event_text(e))).collect();
    for dropped in 0..open.len() {
        let mut out = String::new();
        if dropped > 0 {
            let ids: Vec<String> = open[..dropped].iter().map(|e| e.id.to_string()).collect();
            writeln!(out, "- … {dropped} older open items: {}", ids.join(", ")).unwrap();
        }
        out.extend(lines[dropped..].iter().cloned());
        if tokens(&out) <= OPEN_SHARE {
            return out;
        }
    }
    let ids: Vec<String> = open.iter().map(|e| e.id.to_string()).collect();
    format!("- {} open items: {}\n", open.len(), ids.join(", "))
}

/// The final lines of the last [`RECENT`], word for word. Over their share,
/// the oldest give way to a pointer at the transcript.
fn recent(rec: &MeetingRecord) -> String {
    let lines = rec.transcript(Window::Since(RECENT), None);
    if lines.is_empty() {
        return "Nothing said in the last 3 minutes.\n".into();
    }
    let texts: Vec<String> = lines.iter().map(|l| format!("- {}\n", rec.line_text(l))).collect();
    let mut dropped = 0;
    loop {
        let mut out = String::new();
        if dropped > 0 {
            writeln!(out, "- … {dropped} earlier lines: `pam transcript --since 3m`").unwrap();
        }
        out.extend(texts[dropped..].iter().cloned());
        if tokens(&out) <= RECENT_SHARE || dropped + 1 == texts.len() {
            return out;
        }
        dropped += 1;
    }
}

/// `events`, one line each, in about `share` tokens. Events before `c1`
/// show only topics and decisions in force, with a marker for each run of
/// others; events before `c0` collapse into one line. Each cutoff is pushed
/// only as far as it must, `c1` first, and never into the [`NEWEST`] events.
pub(crate) fn events(rec: &MeetingRecord, events: &[Event], share: usize) -> String {
    if events.is_empty() {
        return "None yet.\n".into();
    }
    let texts: Vec<String> = events.iter().map(|e| rec.event_text(e)).collect();
    let kept: Vec<bool> = events
        .iter()
        .map(|e| match e.kind.as_str() {
            "topic" => true,
            "decision" => rec.superseded_by(e.id).is_none(),
            _ => false,
        })
        .collect();
    let section = |c0: usize, c1: usize| render_events(events, &texts, &kept, c0, c1);
    let fits = |s: &String| tokens(s) <= share;
    let full = section(0, 0);
    // The newest events stay whole however long the meeting.
    let limit = events.len().saturating_sub(NEWEST);
    if fits(&full) || limit == 0 {
        return full;
    }
    if fits(&section(0, limit)) {
        let c1 = smallest(limit, |c1| fits(&section(0, c1)));
        return section(0, c1);
    }
    let c0 = smallest(limit, |c0| fits(&section(c0, limit)));
    section(c0, limit)
}

/// The smallest `i` in `1..=n` for which `ok(i)` holds, assuming `ok(n)` does
/// and `ok(0)` doesn't. A binary search: if `ok` isn't monotonic, the answer
/// may not be the smallest, but `ok` still holds for it.
fn smallest(n: usize, ok: impl Fn(usize) -> bool) -> usize {
    let (mut lo, mut hi) = (0, n);
    while hi - lo > 1 {
        let mid = lo + (hi - lo) / 2;
        if ok(mid) {
            hi = mid;
        } else {
            lo = mid;
        }
    }
    hi
}

fn render_events(events: &[Event], texts: &[String], kept: &[bool], c0: usize, c1: usize) -> String {
    let mut out = String::new();
    if c0 > 0 {
        let (first, last) = (&events[0], &events[c0 - 1]);
        writeln!(
            out,
            "- … {c0} earlier events collapsed: {}..{}, {}–{}",
            first.id,
            last.id,
            mmss(first.t),
            mmss(last.t)
        )
        .unwrap();
    }
    let mut run: Vec<&Event> = Vec::new();
    let flush = |out: &mut String, run: &mut Vec<&Event>| {
        match &run[..] {
            [] => {}
            [one] => writeln!(out, "- … 1 event collapsed: {}", one.id).unwrap(),
            [first, .., last] => {
                writeln!(out, "- … {} events collapsed: {}..{}", run.len(), first.id, last.id).unwrap()
            }
        }
        run.clear();
    };
    for i in c0..events.len() {
        if i < c1 && !kept[i] {
            run.push(&events[i]);
            continue;
        }
        flush(&mut out, &mut run);
        writeln!(out, "- {}", texts[i]).unwrap();
    }
    flush(&mut out, &mut run);
    out
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    /// A stopped meeting in a temporary folder, from the files' contents.
    fn meeting(
        duration_s: u64,
        speakers: &str,
        lines: &[String],
        events: &[String],
    ) -> (tempfile::TempDir, MeetingRecord) {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        fs::write(
            p.join("meeting.json"),
            format!(
                r#"{{"title":"Sync","started_at":"2026-10-10T14:02:00+11:00","assistant_name":"Pam","status":"stopped","duration":{}}}"#,
                duration_s * SECOND
            ),
        )
        .unwrap();
        fs::write(p.join("speakers.json"), speakers).unwrap();
        fs::write(p.join("transcript.jsonl"), lines.iter().map(|l| format!("{l}\n")).collect::<String>())
            .unwrap();
        fs::write(p.join("events.jsonl"), events.iter().map(|e| format!("{e}\n")).collect::<String>())
            .unwrap();
        let rec = MeetingRecord::load(p).unwrap();
        (dir, rec)
    }

    fn line(id: u32, t0_s: u64, slot: u32, text: &str) -> String {
        format!(
            r#"{{"id":"u{id:04}","t0":{},"t1":{},"slot":{slot},"text":"{text}"}}"#,
            t0_s * SECOND,
            (t0_s + 4) * SECOND
        )
    }

    fn event(
        id: u32,
        kind: &str,
        t_s: u64,
        spk: u32,
        text: &str,
        line: u32,
        supersedes: Option<u32>,
    ) -> String {
        let sup = supersedes.map_or(String::new(), |s| format!(r#","supersedes":"e{s:03}""#));
        format!(
            r#"{{"id":"e{id:03}","type":"{kind}","t":{},"by":["spk_{spk}"],"text":"{text}","refs":["u{line:04}"]{sup}}}"#,
            t_s * SECOND
        )
    }

    const SPEAKERS: &str = r#"{"slots":{"0":"spk_0","1":"spk_1","2":"spk_2","3":"spk_3"},
        "names":{"spk_0":"Alice","spk_1":"Marcus"},"merges":{"spk_3":"spk_1"}}"#;

    #[test]
    fn renders_every_section_in_the_spec_order() {
        let lines = [
            line(1, 2, 0, "Hi, I'm Alice and I run product."),
            line(2, 10, 1, "Marcus here, backend."),
            line(3, 200, 3, "The signup flow loses users on mobile."),
            line(4, 230, 2, "Should we prototype both flows?"),
        ];
        let events = [
            event(1, "topic", 200, 3, "Signup flow", 3, None),
            event(2, "concern", 200, 3, "Mobile users drop off", 3, None),
            event(3, "question", 230, 2, "Prototype both flows?", 4, None),
            event(4, "action", 230, 0, "Alice drafts the brief", 4, None),
            event(5, "question", 10, 1, "Who owns backend?", 2, None),
            event(6, "answer", 12, 1, "Marcus does", 2, Some(5)),
        ];
        let (_dir, rec) = meeting(240, SPEAKERS, &lines, &events);
        assert_eq!(
            render(&rec),
            "# Meeting: Sync · 04:00 elapsed · assistant name: Pam\n\n\
             <!-- Rendered from the meeting folder on every change. Do not edit. -->\n\n\
             ## People\n\nspk_0 Alice · spk_1 Marcus (also spk_3) · spk_2\n\n\
             ## Open\n\n\
             - e003 03:50 question spk_2: Prototype both flows? [u0004]\n\
             - e004 03:50 action Alice: Alice drafts the brief [u0004]\n\n\
             ## Tasks\n\nNone yet.\n\n\
             ## Events\n\n\
             - e001 03:20 topic Marcus: Signup flow [u0003]\n\
             - e002 03:20 concern Marcus: Mobile users drop off [u0003]\n\
             - e003 03:50 question spk_2: Prototype both flows? [u0004]\n\
             - e004 03:50 action Alice: Alice drafts the brief [u0004]\n\
             - e005 00:10 question Marcus: Who owns backend? [u0002] [superseded by e006]\n\
             - e006 00:12 answer Marcus: Marcus does [u0002] [supersedes e005]\n\n\
             ## Last 3 min\n\n\
             - u0003 03:20 Marcus: The signup flow loses users on mobile.\n\
             - u0004 03:50 spk_2: Should we prototype both flows?\n\n\
             ## How to dig\n\n\
             - `pam around <id>`: the exact words around a line or event\n\
             - `pam search <words>`: where something was said\n\
             - `pam get <ref>`: lines, events, speakers or tasks by ID, collapsed events included\n\
             - `pam events --type topic,decision`: every topic and decision\n\
             - `pam transcript --since 10m`: more of the transcript\n"
        );
    }

    #[test]
    fn tasks_show_one_line_each_with_their_status_and_who_asked() {
        use crate::event::{TaskId, UtteranceId};
        use crate::task::{self, Target, TaskResult, TaskState, Via};
        let lines = [line(1, 2, 0, "Hi, I'm Alice."), line(2, 10, 1, "Pam, mock up both flows.")];
        let (dir, _) = meeting(240, SPEAKERS, &lines, &[]);
        let at = |s: u64| s * SECOND;
        let tasks = [
            TaskState {
                line: Some(UtteranceId(2)),
                result: Some(TaskResult {
                    title: "Flows".into(),
                    target: Target::File("tasks/t01/flows.md".into()),
                }),
                progress: Some(100),
                action: Some("writing it up".into()),
                ..TaskState::new(TaskId(1), "mock up both flows", at(10), Via::Voice, TaskStatus::Done)
            },
            TaskState::new(TaskId(2), "what did we decide?", at(30), Via::Typed, TaskStatus::Done),
            TaskState {
                error: Some("pi exited".into()),
                ..TaskState::new(TaskId(3), "book a room", at(40), Via::Typed, TaskStatus::Failed)
            },
            TaskState::new(TaskId(4), "never mind", at(50), Via::Typed, TaskStatus::Cancelled),
            TaskState {
                spk: Some(SpkId(3)),
                progress: Some(60),
                action: Some("reading sources".into()),
                ..TaskState::new(TaskId(5), "fact-check that", at(60), Via::Voice, TaskStatus::Running)
            },
            TaskState::new(TaskId(6), "draft the email", at(70), Via::Typed, TaskStatus::Queued),
        ];
        for t in &tasks {
            task::create(dir.path(), t).unwrap();
        }
        // A task folder still being made has no task.json yet.
        fs::create_dir_all(dir.path().join("tasks/t07")).unwrap();
        let rec = MeetingRecord::load(dir.path()).unwrap();
        let md = render(&rec);
        let section = md.split("## Tasks\n\n").nth(1).unwrap().split("\n## ").next().unwrap();
        assert_eq!(
            section,
            "- t01 00:10 Marcus: mock up both flows [u0002] · ready (tasks/t01/flows.md)\n\
             - t02 00:30 typed: what did we decide? · done\n\
             - t03 00:40 typed: book a room · failed\n\
             - t05 01:00 Marcus: fact-check that · running 60% · reading sources\n\
             - t06 01:10 typed: draft the email · queued\n"
        );
        assert!(tokens(&md) <= BUDGET);
    }

    #[test]
    fn an_empty_meeting_says_so_in_each_section() {
        let (_dir, rec) = meeting(0, "{}", &[], &[]);
        let md = render(&rec);
        for part in [
            "## People\n\nNobody has spoken yet.\n",
            "## Open\n\nNothing open.\n",
            "## Tasks\n\nNone yet.\n",
            "## Events\n\nNone yet.\n",
            "## Last 3 min\n\nNothing said in the last 3 minutes.\n",
        ] {
            assert!(md.contains(part), "{part:?} in {md}");
        }
    }

    /// A meeting of `minutes`, a line every 5 s, `events_per_line` events
    /// for each, with a topic every 40 events and a decision every 25.
    fn long_meeting(minutes: u32, events_per_line: usize) -> (tempfile::TempDir, MeetingRecord) {
        let n = minutes * 60 / 5;
        let lines: Vec<String> = (1..=n)
            .map(|i| line(i, (i as u64 - 1) * 5, i % 4, "We could make the remote control smaller and put the buttons on the side so it fits the hand."))
            .collect();
        let mut events = Vec::new();
        for i in 1..=n {
            for _ in 0..events_per_line {
                let id = events.len() as u32 + 1;
                let kind = match id {
                    _ if id % 40 == 1 => "topic",
                    _ if id.is_multiple_of(25) => "decision",
                    _ if id.is_multiple_of(7) => "question",
                    _ => "point",
                };
                events.push(event(
                    id,
                    kind,
                    (i as u64 - 1) * 5,
                    i % 4,
                    "The remote should fit in one hand and have fewer buttons",
                    i,
                    None,
                ));
            }
        }
        meeting(minutes as u64 * 60, SPEAKERS, &lines, &events)
    }

    #[test]
    fn a_long_meeting_stays_within_budget_and_collapses_the_oldest_events_first() {
        let (_dir, rec) = long_meeting(30, 1);
        let md = render(&rec);
        assert!(tokens(&md) <= BUDGET, "{} tokens", tokens(&md));
        let events = md.split("## Events\n\n").nth(1).unwrap().split("\n## ").next().unwrap();
        // The newest events are all there; the oldest show only topics and decisions.
        let last = rec.events.last().unwrap();
        assert!(events.contains(&format!("- {}\n", rec.event_text(last))), "{events}");
        assert!(events.starts_with("- e001 00:00 topic "), "{events}");
        assert!(events.contains("- … 23 events collapsed: e002..e024\n- e025 "), "{events}");
        // Lines of the last 3 minutes are all there, word for word.
        let recent = md.split("## Last 3 min\n\n").nth(1).unwrap();
        let lines = rec.transcript(Window::Since(RECENT), None);
        assert_eq!(lines.len(), 36);
        for l in lines {
            assert!(recent.contains(&rec.line_text(l)), "{} missing", l.id);
        }
        // Open questions give way to their IDs past their share.
        let open = md.split("## Open\n\n").nth(1).unwrap().split("\n## ").next().unwrap();
        assert!(open.starts_with("- … "), "{open}");
        assert!(tokens(open) <= OPEN_SHARE);
    }

    #[test]
    fn when_even_topics_and_decisions_dont_fit_the_oldest_fold_into_one_line() {
        let (_dir, rec) = long_meeting(90, 1);
        let md = render(&rec);
        assert!(tokens(&md) <= BUDGET, "{} tokens", tokens(&md));
        let events = md.split("## Events\n\n").nth(1).unwrap();
        assert!(events.starts_with("- … "), "{events}");
        assert!(events.contains(" earlier events collapsed: e001.."), "{events}");
        let last = rec.events.last().unwrap();
        assert!(events.contains(&format!("- {}\n", rec.event_text(last))), "{events}");
    }
}
