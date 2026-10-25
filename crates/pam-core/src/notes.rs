//! Pam's notes: the summary plus the events, sorted into the fixed sections of
//! `notes.md`. Only the summary is rewritten: text that keeps shifting is
//! distracting to read live.
//!
//! Notes are rebuilt from `events.jsonl`, `summary.json` and the tasks every
//! time, so they never drift from their sources. Items keep the order of
//! their events, and an event that something supersedes is left out. A
//! request to Pam is Pam's action item, with its task's status.

use std::fmt::Write as _;

use std::path::Path;

use anyhow::{Context, Result};

use crate::Ts;
use crate::event::{EventId, SpkId, TaskId};
use crate::folder::EventLog;
use crate::identity::Overlay;
use crate::record::{Event, MeetingRecord};
use crate::task::{TaskResult, TaskState, TaskStatus};

/// The sections of `notes.md` after the summary, in order.
const SECTIONS: [&str; 5] = ["Discussion", "Decisions", "Action items", "Open questions", "Notes"];

#[derive(Debug, Clone, PartialEq)]
pub struct Notes {
    pub summary: String,
    /// One per entry of [`SECTIONS`], in that order.
    pub sections: Vec<Section>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Section {
    pub title: &'static str,
    pub items: Vec<NoteItem>,
}

/// One item: an event that nothing supersedes.
#[derive(Debug, Clone, PartialEq)]
pub struct NoteItem {
    pub event: EventId,
    /// The event type: `topic`, `point`, `decision`…
    pub kind: String,
    pub text: String,
    /// Who it credits, as the event says. Merges and names are applied when
    /// drawn ([`NoteItem::credits`]), so undoing a merge needs no new notes.
    pub by: Vec<SpkId>,
    pub t: Ts,
    /// A request's task: its status as shown (`ready` once it has a
    /// result), and the result. `None` for other items, and for a request
    /// whose task can't be read.
    pub task: Option<(&'static str, Option<TaskResult>)>,
}

impl NoteItem {
    /// The text as a reader sees it, e.g. "Concern: single page loses mobile users".
    pub fn label(&self) -> String {
        match self.kind.as_str() {
            "proposal" => format!("Proposal: {}", self.text),
            "concern" => format!("Concern: {}", self.text),
            "answer" => format!("Answer: {}", self.text),
            "request" => match &self.task {
                Some((status, _)) => format!("{} · {status}", self.text),
                None => self.text.clone(),
            },
            _ => self.text.clone(),
        }
    }

    /// Ticked in `notes.md`: Pam's item once its task is done.
    fn is_done(&self) -> bool {
        matches!(self.task, Some(("done" | "ready", _)))
    }

    /// Who it credits now: merges followed, each speaker once.
    pub fn credits(&self, overlay: &Overlay) -> Vec<SpkId> {
        let mut out: Vec<SpkId> = Vec::new();
        for spk in self.by.iter().map(|&s| overlay.follow(s)) {
            if !out.contains(&spk) {
                out.push(spk);
            }
        }
        out
    }
}

/// The section an event type goes in, if any. Intros name speakers rather
/// than make notes; a task's updates show as its request's status.
fn section_of(kind: &str) -> Option<usize> {
    Some(match kind {
        "topic" | "point" | "proposal" | "concern" | "answer" => 0,
        "decision" => 1,
        "action" | "request" => 2,
        "question" => 3,
        "note" => 4,
        _ => return None,
    })
}

/// No summary and no items, but every section, so `notes.md` always has the template.
impl Default for Notes {
    fn default() -> Notes {
        Notes::build("", &[], &[])
    }
}

impl Notes {
    /// Sorts `events` into sections. A task's first `request` is its item,
    /// with the task's status from `tasks`; its corrections, its updates
    /// and a cancelled task make none.
    pub fn build(summary: &str, events: &[Event], tasks: &[TaskState]) -> Notes {
        let mut sections: Vec<Section> =
            SECTIONS.iter().map(|&title| Section { title, items: Vec::new() }).collect();
        let mut requested: Vec<&str> = Vec::new();
        for e in events {
            let Some(s) = section_of(&e.kind) else { continue };
            if events.iter().any(|later| later.supersedes == Some(e.id)) {
                continue;
            }
            let mut task = None;
            if e.kind == "request" {
                let Some(id) = e.task.as_deref() else { continue };
                if requested.contains(&id) {
                    continue;
                }
                requested.push(id);
                let state = id.parse::<TaskId>().ok().and_then(|id| tasks.iter().find(|t| t.id == id));
                if state.is_some_and(|t| t.status == TaskStatus::Cancelled) {
                    continue;
                }
                task = state.map(|t| (t.shown_status(), t.result.clone()));
            }
            sections[s].items.push(NoteItem {
                event: e.id,
                kind: e.kind.clone(),
                text: e.text.clone(),
                by: e.by.clone(),
                t: e.t,
                task,
            });
        }
        Notes { summary: summary.to_string(), sections }
    }

    /// Items in every section.
    pub fn len(&self) -> usize {
        self.sections.iter().map(|s| s.items.len()).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.summary.is_empty() && self.len() == 0
    }

    /// The item added last, with the title of its section.
    pub fn newest(&self) -> Option<(&'static str, &NoteItem)> {
        self.sections.iter().flat_map(|s| s.items.iter().map(|i| (s.title, i))).max_by_key(|(_, i)| i.event)
    }

    /// `notes.md`, with the corrections in `overlay` applied.
    pub fn render(&self, title: &str, overlay: &Overlay) -> String {
        let mut out = if title.is_empty() { "# Notes\n".to_string() } else { format!("# Notes: {title}\n") };
        out.push_str(
            "\n<!-- Rendered from events.jsonl, summary.json and speakers.json on every change. Do not edit. -->\n",
        );
        let summary = if self.summary.is_empty() { "Nothing yet." } else { &self.summary };
        write!(out, "\n## Summary\n\n{summary}\n").unwrap();
        for s in &self.sections {
            // Open questions and notes appear once there are some.
            if s.items.is_empty() && matches!(s.title, "Open questions" | "Notes") {
                continue;
            }
            write!(out, "\n## {}\n\n", s.title).unwrap();
            if s.items.is_empty() {
                out.push_str("None yet.\n");
            }
            for i in &s.items {
                if i.kind == "topic" {
                    if !out.ends_with("\n\n") {
                        out.push('\n');
                    }
                    writeln!(out, "**{}**\n", i.text).unwrap();
                    continue;
                }
                let box_ = match i.kind.as_str() {
                    "request" if i.is_done() => "[x] ",
                    "action" | "request" => "[ ] ",
                    _ => "",
                };
                write!(out, "- {box_}{}", i.label()).unwrap();
                if let Some((_, Some(result))) = &i.task {
                    let title = crate::task::markdown_text(&result.title);
                    write!(out, ": [{title}]({})", result.target.markdown()).unwrap();
                }
                let by: Vec<String> =
                    i.credits(overlay).into_iter().map(|s| overlay.label(Some(s))).collect();
                if !by.is_empty() {
                    write!(out, " ({})", by.join(", ")).unwrap();
                }
                out.push('\n');
            }
        }
        out
    }
}

/// Adds a `note` event at the meeting's current time, as `pam note` does,
/// and re-renders `notes.md` and `context.md`. A running meeting sees the new event and sends
/// the notes to its front-ends.
pub fn add_note(dir: &Path, text: &str) -> Result<Event> {
    let mut log = EventLog::lock(dir)?;
    // Read under the lock: the scribe may have changed the summary or names
    // while this waited for it.
    let rec = MeetingRecord::load(dir)?;
    let note = Event {
        id: EventId(0),
        kind: "note".into(),
        t: rec.now(),
        by: Vec::new(),
        text: text.split_whitespace().collect::<Vec<_>>().join(" "),
        refs: Vec::new(),
        task: None,
        supersedes: None,
        name: None,
        rename: None,
    };
    let note = log.append(vec![note])?.remove(0);
    let notes = Notes::build(&rec.summary.text, &log.events()?, &rec.tasks);
    log.write_notes_and_context(&rec.info.title, &notes, &rec.speakers.overlay)
        .with_context(|| format!("{} is saved, but not every rendered file was updated", note.id))?;
    Ok(note)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn event(id: u32, kind: &str, by: &[u32], text: &str, supersedes: Option<u32>) -> Event {
        Event {
            id: EventId(id),
            kind: kind.into(),
            t: id as Ts,
            by: by.iter().map(|&s| SpkId(s)).collect(),
            text: text.into(),
            refs: vec![format!("u{id:04}")],
            task: None,
            supersedes: supersedes.map(EventId),
            name: None,
            rename: None,
        }
    }

    #[test]
    fn sorts_events_into_sections_and_leaves_out_superseded_ones() {
        let events = [
            event(1, "intro", &[0], "Alice runs product", None),
            event(2, "topic", &[0], "Onboarding flow", None),
            event(3, "point", &[1], "Single page is faster to build", None),
            event(4, "concern", &[3], "Single page may lose mobile users", None),
            event(5, "question", &[0], "Do we have mobile drop-off data?", None),
            event(6, "decision", &[0], "Drop the team-invite step", None),
            event(7, "decision", &[0], "Keep the team-invite step", Some(6)),
            event(8, "action", &[2], "Prototype both flows", None),
            event(9, "question", &[1], "Who owns the welcome email?", None),
            event(10, "answer", &[2], "Jordan owns the welcome email", Some(9)),
        ];
        let notes = Notes::build("Choosing an onboarding flow.", &events, &[]);
        // spk_3 was merged into spk_1; Alice and Marcus are named.
        let overlay = Overlay {
            names: [(SpkId(0), "Alice".into()), (SpkId(1), "Marcus".into())].into(),
            merges: [(SpkId(3), SpkId(1))].into(),
            ..Overlay::default()
        };
        let ids = |s: usize| notes.sections[s].items.iter().map(|i| i.event.0).collect::<Vec<_>>();
        assert_eq!(ids(0), [2, 3, 4, 10]);
        assert_eq!(ids(1), [7]);
        assert_eq!(ids(2), [8]);
        assert_eq!(ids(3), [5]);
        assert!(ids(4).is_empty());
        // Notes keep the event's IDs; credits follow the merge, so undoing it needs no new notes.
        assert_eq!(notes.sections[0].items[2].by, [SpkId(3)]);
        assert_eq!(notes.sections[0].items[2].credits(&overlay), [SpkId(1)]);
        assert_eq!(notes.sections[0].items[2].credits(&Overlay::default()), [SpkId(3)]);
        assert_eq!(notes.len(), 7);
        assert_eq!(notes.newest().map(|(s, i)| (s, i.event.0)), Some(("Discussion", 10)));

        let md = notes.render("Sync", &overlay);
        assert_eq!(
            md,
            "# Notes: Sync\n\n\
             <!-- Rendered from events.jsonl, summary.json and speakers.json on every change. Do not edit. -->\n\n\
             ## Summary\n\nChoosing an onboarding flow.\n\n\
             ## Discussion\n\n\
             **Onboarding flow**\n\n\
             - Single page is faster to build (Marcus)\n\
             - Concern: Single page may lose mobile users (Marcus)\n\
             - Answer: Jordan owns the welcome email (spk_2)\n\n\
             ## Decisions\n\n- Keep the team-invite step (Alice)\n\n\
             ## Action items\n\n- [ ] Prototype both flows (spk_2)\n\n\
             ## Open questions\n\n- Do we have mobile drop-off data? (Alice)\n"
        );
    }

    #[test]
    fn a_request_is_pams_action_item_with_its_tasks_status() {
        use crate::task::{Target, Via};
        let request = |id: u32, task: u32, by: &[u32], text: &str| Event {
            task: Some(format!("t{task:02}")),
            ..event(id, "request", by, text, None)
        };
        let update = |id: u32, task: u32, text: &str| Event {
            by: Vec::new(),
            task: Some(format!("t{task:02}")),
            ..event(id, "task_update", &[], text, None)
        };
        let events = [
            event(1, "action", &[0], "Alice drafts the brief", None),
            request(2, 1, &[2], "Prototype both flows"),
            update(3, 1, "started"),
            request(4, 1, &[2], "Only the mobile one"),
            request(5, 2, &[], "Fact-check the activation claim"),
            request(6, 3, &[1], "Draft the follow-up email"),
            request(7, 4, &[0], "Cancel me"),
            request(8, 5, &[0], "Book a room"),
            request(9, 6, &[0], "A task that can't be read"),
        ];
        let task = |id: u32, status| TaskState::new(TaskId(id), "x", 0, Via::Voice, status);
        let tasks = [
            task(1, TaskStatus::Running),
            TaskState {
                result: Some(TaskResult {
                    // A space in its path and brackets in its title still make a link.
                    title: "Sources [2]".into(),
                    target: Target::File("tasks/t02/final report.md".into()),
                }),
                ..task(2, TaskStatus::Done)
            },
            task(3, TaskStatus::Queued),
            task(4, TaskStatus::Cancelled),
            task(5, TaskStatus::Done),
        ];
        let notes = Notes::build("", &events, &tasks);
        let items = &notes.sections[2].items;
        assert_eq!(items.iter().map(|i| i.event.0).collect::<Vec<_>>(), [1, 2, 5, 6, 8, 9]);
        assert_eq!(items[1].label(), "Prototype both flows · running");
        let overlay = Overlay { names: [(SpkId(2), "Jordan".into())].into(), ..Overlay::default() };
        let md = notes.render("", &overlay);
        assert!(
            md.contains(
                "## Action items\n\n\
                 - [ ] Alice drafts the brief (spk_0)\n\
                 - [ ] Prototype both flows · running (Jordan)\n\
                 - [x] Fact-check the activation claim · ready: [Sources \\[2\\]](tasks/t02/final%20report.md)\n\
                 - [ ] Draft the follow-up email · queued (spk_1)\n\
                 - [x] Book a room · done (spk_0)\n\
                 - [ ] A task that can't be read (spk_0)\n"
            ),
            "{md}"
        );
    }

    #[test]
    fn empty_notes_say_so() {
        let md = Notes::build("", &[], &[]).render("", &Overlay::default());
        assert_eq!(
            md,
            "# Notes\n\n\
             <!-- Rendered from events.jsonl, summary.json and speakers.json on every change. Do not edit. -->\n\n\
             ## Summary\n\nNothing yet.\n\n\
             ## Discussion\n\nNone yet.\n\n\
             ## Decisions\n\nNone yet.\n\n\
             ## Action items\n\nNone yet.\n"
        );
    }

    #[test]
    fn a_note_renders_what_the_scribe_wrote_while_it_waited_for_the_lock() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::write(
            p.join("meeting.json"),
            r#"{"started_at":"2026-10-11T10:00:00+13:00","status":"stopped"}"#,
        )
        .unwrap();
        std::fs::write(p.join("summary.json"), r#"{"through":null,"text":"Old summary."}"#).unwrap();
        let rec = MeetingRecord::load(p).unwrap();
        // The scribe finishes a run after `pam note` read the meeting.
        std::fs::write(p.join("summary.json"), r#"{"through":"u0009","text":"New summary."}"#).unwrap();
        let note = add_note(&rec.dir, "  Ship   on Friday ").unwrap();
        assert_eq!((note.id, note.text.as_str()), (EventId(1), "Ship on Friday"));
        let md = std::fs::read_to_string(p.join("notes.md")).unwrap();
        assert!(
            md.contains("## Summary\n\nNew summary.\n") && md.ends_with("## Notes\n\n- Ship on Friday\n"),
            "{md}"
        );
    }

    #[test]
    fn a_note_reaches_context_md_even_when_notes_md_cannot_be_written() {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        std::fs::write(
            p.join("meeting.json"),
            r#"{"started_at":"2026-10-11T10:00:00+13:00","status":"stopped"}"#,
        )
        .unwrap();
        // A directory in its place makes replacing notes.md fail.
        std::fs::create_dir_all(p.join("notes.md/blocked")).unwrap();
        let err = add_note(p, "Ship on Friday").unwrap_err();
        assert!(format!("{err:#}").contains("writing notes.md"), "{err:#}");
        let context = std::fs::read_to_string(p.join("context.md")).unwrap();
        assert!(context.contains("- e001 00:00 note: Ship on Friday\n"), "{context}");
    }
}
