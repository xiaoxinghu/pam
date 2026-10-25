//! What the agent is told before each prompt's own text, read from the
//! meeting folder as the prompt goes out. A session's first prompt gets the
//! context pack and the minute before the request; every later one, what
//! happened since the prompt before it, so the agent needs no tool calls to
//! catch up.

use std::fmt::Write as _;

use anyhow::Result;

use crate::Ts;
use crate::clock::{SECOND, mmss};
use crate::context::{self, RECENT_SHARE, tokens};
use crate::event::{EventId, UtteranceId};
use crate::record::MeetingRecord;

/// How much speech before a request the agent gets word for word.
pub const RECENT: Ts = 60 * SECOND;
/// The longest span of new lines that come word for word.
const SHORT_GAP: Ts = 3 * 60 * SECOND;
/// What the events added in a long gap get, in estimated tokens.
const EVENTS_SHARE: usize = 1_500;

/// Where a prompt left the agent: when it went out, and the newest line and
/// event the meeting folder had then.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Mark {
    pub at: Ts,
    pub line: Option<UtteranceId>,
    pub event: Option<EventId>,
}

impl Mark {
    fn of(rec: &MeetingRecord, at: Ts) -> Mark {
        Mark { at, line: rec.lines.last().map(|l| l.id), event: rec.events.last().map(|e| e.id) }
    }
}

/// A line on its way to being final, as the meeting last heard it:
/// `text` is as `pam transcript` prints a line, with no ID
/// (`00:27 Jordan: Pam, what did we decide?`). A spoken request starts
/// before its line is final, so its own words are among these.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Coming {
    pub t0: Ts,
    pub t1: Ts,
    pub text: String,
}

/// What goes before a prompt sent at `now` for a request made at
/// `asked_at`, and the mark it leaves for the next. With no `since`, it is
/// the session's first prompt: `context.md` and the [`RECENT`] before the
/// request. Otherwise it is what happened since that mark. Either way, the
/// lines still `coming` that started by the request follow the final ones.
pub fn lead(
    rec: &MeetingRecord,
    since: Option<&Mark>,
    coming: &[Coming],
    asked_at: Ts,
    now: Ts,
) -> Result<(String, Mark)> {
    let coming: Vec<&Coming> = coming.iter().filter(|c| c.t0 <= asked_at).collect();
    let lead = match since {
        None => first(rec, &coming, asked_at)?,
        Some(mark) => catch_up(rec, mark, &coming, asked_at),
    };
    Ok((lead, Mark::of(rec, now)))
}

/// `context.md` and the lines of the minute before the request.
fn first(rec: &MeetingRecord, coming: &[&Coming], asked_at: Ts) -> Result<String> {
    let from = asked_at.saturating_sub(RECENT);
    let mut recent: Vec<String> =
        rec.lines.iter().filter(|l| l.t1 > from && l.t0 <= asked_at).map(|l| rec.line_text(l)).collect();
    recent.extend(coming.iter().filter(|c| c.t1 > from).map(|c| c.text.clone()));
    Ok(format!(
        "<context.md>\n{}\n</context.md>\n\n<last-60-seconds>\n{}\n</last-60-seconds>\n\n",
        context::read(rec)?.trim_end(),
        or_nobody(&recent)
    ))
}

/// The lines since `mark` word for word when they span at most
/// [`SHORT_GAP`], from the first's start to the last's end, and fit
/// [`RECENT_SHARE`]; otherwise the summary, the
/// events added since, and the lines from [`RECENT`] before the request on.
/// Either way, where to read the rest.
fn catch_up(rec: &MeetingRecord, mark: &Mark, coming: &[&Coming], asked_at: Ts) -> String {
    let new = rec.lines.iter().filter(|l| mark.line.is_none_or(|m| l.id > m));
    let since = mmss(mark.at);
    let mut lines: Vec<String> = new.clone().map(|l| rec.line_text(l)).collect();
    let start = new.clone().map(|l| l.t0).min();
    let span = start.map_or(0, |t0| new.clone().map(|l| l.t1).max().unwrap_or(t0).saturating_sub(t0));
    let from = asked_at.saturating_sub(RECENT);
    let mut out = String::new();
    if span <= SHORT_GAP && tokens(&lines.join("\n")) <= RECENT_SHARE {
        lines.extend(coming.iter().map(|c| c.text.clone()));
        write!(
            out,
            "Said since the last prompt ({since}), word for word:\n<lines>\n{}\n</lines>\n",
            or_nobody(&lines)
        )
        .unwrap();
    } else {
        let summary = match rec.summary.text.trim() {
            "" => "(none yet)",
            text => text,
        };
        let added: Vec<_> =
            rec.events.iter().filter(|e| mark.event.is_none_or(|m| e.id > m)).cloned().collect();
        let events = match added.is_empty() {
            true => "(none)".to_string(),
            false => context::events(rec, &added, EVENTS_SHARE).trim_end().to_string(),
        };
        let mut recent: Vec<String> = new.filter(|l| l.t1 > from).map(|l| rec.line_text(l)).collect();
        recent.extend(coming.iter().filter(|c| c.t1 > from).map(|c| c.text.clone()));
        write!(
            out,
            "Since the last prompt ({since}), in short:\n<summary>\n{summary}\n</summary>\n\
             <new-events>\n{events}\n</new-events>\n<last-60-seconds>\n{}\n</last-60-seconds>\n",
            or_nobody(&recent)
        )
        .unwrap();
    }
    writeln!(out, "More: pam transcript --from {since}\n").unwrap();
    out
}

fn or_nobody(lines: &[String]) -> String {
    if lines.is_empty() { "(nobody spoke)".to_string() } else { lines.join("\n") }
}

#[cfg(test)]
mod tests {
    use std::fs;

    use super::*;

    const SPEAKERS: &str =
        r#"{"slots":{"0":"spk_0","1":"spk_1"},"names":{"spk_0":"Alice","spk_1":"Marcus"}}"#;

    /// A meeting under way, with `context.md`, the lines `(t0 in s, speaker slot, text)`, 4 s each,
    /// the events `(type, t in s, text)` and a summary.
    fn meeting(
        lines: &[(u64, u32, &str)],
        events: &[(&str, u64, &str)],
        summary: &str,
    ) -> (tempfile::TempDir, MeetingRecord) {
        let dir = tempfile::tempdir().unwrap();
        let p = dir.path();
        fs::write(
            p.join("meeting.json"),
            r#"{"title":"Sync","started_at":"2026-10-10T14:02:00+11:00","assistant_name":"Pam","status":"recording"}"#,
        )
        .unwrap();
        fs::write(p.join("speakers.json"), SPEAKERS).unwrap();
        fs::write(p.join("context.md"), "# Meeting: Sync\n\n## People\n").unwrap();
        let lines: String = lines
            .iter()
            .enumerate()
            .map(|(i, (t0, slot, text))| {
                let (t0, t1) = (t0 * SECOND, (t0 + 4) * SECOND);
                format!(
                    "{{\"id\":\"u{:04}\",\"t0\":{t0},\"t1\":{t1},\"slot\":{slot},\"text\":\"{text}\"}}\n",
                    i + 1
                )
            })
            .collect();
        fs::write(p.join("transcript.jsonl"), lines).unwrap();
        let events: String = events
            .iter()
            .enumerate()
            .map(|(i, (kind, t, text))| {
                format!(
                    "{{\"id\":\"e{:03}\",\"type\":\"{kind}\",\"t\":{},\"text\":\"{text}\"}}\n",
                    i + 1,
                    t * SECOND
                )
            })
            .collect();
        fs::write(p.join("events.jsonl"), events).unwrap();
        fs::write(p.join("summary.json"), serde_json::json!({"through": null, "text": summary}).to_string())
            .unwrap();
        let rec = MeetingRecord::load(p).unwrap();
        (dir, rec)
    }

    const LINES: [(u64, u32, &str); 5] = [
        (10, 0, "Hi, I'm Alice."),
        (70, 1, "Pam, what did we decide?"),
        (90, 0, "Pricing is forty a seat."),
        (150, 1, "And the launch is Friday."),
        (160, 0, "Pam, when do we launch?"),
    ];

    /// [`LINES`] with the launch talk later, so the lines after u0001 span over 3 min.
    const LATER: [(u64, u32, &str); 5] = [
        (10, 0, "Hi, I'm Alice."),
        (70, 1, "Pam, what did we decide?"),
        (90, 0, "Pricing is forty a seat."),
        (250, 1, "And the launch is Friday."),
        (260, 0, "Pam, when do we launch?"),
    ];

    fn mark(at_s: u64, line: u32, event: Option<u32>) -> Mark {
        Mark { at: at_s * SECOND, line: Some(UtteranceId(line)), event: event.map(EventId) }
    }

    #[test]
    fn lines_on_their_way_that_started_by_the_request_follow_the_final_ones() {
        // The request, u0005 in LINES, isn't final yet; Marcus answers after it.
        let (_dir, rec) = meeting(&LINES[..4], &[], "");
        let line =
            |t0: u64, t1: u64, text: &str| Coming { t0: t0 * SECOND, t1: t1 * SECOND, text: text.into() };
        let coming = [
            line(160, 163, "02:40 Alice: Pam, when do we launch?"),
            line(166, 168, "02:46 Marcus: Next week."),
        ];
        let first = super::lead(&rec, None, &coming, 160 * SECOND, 164 * SECOND).unwrap().0;
        assert!(
            first.ends_with(
                "u0004 02:30 Marcus: And the launch is Friday.\n02:40 Alice: Pam, when do we launch?\n\
                 </last-60-seconds>\n\n"
            ),
            "{first}"
        );
        let (later, left) =
            super::lead(&rec, Some(&mark(155, 4, None)), &coming, 160 * SECOND, 164 * SECOND).unwrap();
        assert_eq!(
            later,
            "Said since the last prompt (02:35), word for word:\n<lines>\n\
             02:40 Alice: Pam, when do we launch?\n</lines>\n\
             More: pam transcript --from 02:35\n\n"
        );
        // The mark is the final lines': the request's comes again once final, with its ID.
        assert_eq!(left.line, Some(UtteranceId(4)));
    }

    #[test]
    fn a_sessions_first_prompt_gets_the_context_pack_and_the_minute_before_the_request() {
        let (_dir, rec) = meeting(&LINES, &[("topic", 70, "Pricing")], "");
        let (lead, left) = super::lead(&rec, None, &[], 150 * SECOND, 152 * SECOND).unwrap();
        assert_eq!(
            lead,
            "<context.md>\n# Meeting: Sync\n\n## People\n</context.md>\n\n<last-60-seconds>\n\
             u0003 01:30 Alice: Pricing is forty a seat.\nu0004 02:30 Marcus: And the launch is Friday.\n\
             </last-60-seconds>\n\n"
        );
        assert_eq!(left, Mark { at: 152 * SECOND, line: Some(UtteranceId(5)), event: Some(EventId(1)) });
    }

    #[test]
    fn a_short_gap_comes_word_for_word() {
        let (_dir, rec) = meeting(&LINES, &[("topic", 70, "Pricing")], "They priced it.");
        let (lead, left) =
            super::lead(&rec, Some(&mark(72, 2, None)), &[], 160 * SECOND, 165 * SECOND).unwrap();
        assert_eq!(
            lead,
            "Said since the last prompt (01:12), word for word:\n<lines>\n\
             u0003 01:30 Alice: Pricing is forty a seat.\n\
             u0004 02:30 Marcus: And the launch is Friday.\n\
             u0005 02:40 Alice: Pam, when do we launch?\n</lines>\n\
             More: pam transcript --from 01:12\n\n"
        );
        assert_eq!(left, Mark { at: 165 * SECOND, line: Some(UtteranceId(5)), event: Some(EventId(1)) });
        // Nothing said since: still a catch-up, not the whole brief.
        let (lead, _) =
            super::lead(&rec, Some(&mark(165, 5, Some(1))), &[], 170 * SECOND, 170 * SECOND).unwrap();
        assert_eq!(
            lead,
            "Said since the last prompt (02:45), word for word:\n<lines>\n(nobody spoke)\n</lines>\n\
             More: pam transcript --from 02:45\n\n"
        );
    }

    #[test]
    fn a_long_gap_gets_the_summary_the_new_events_and_the_minute_before_the_request() {
        let events =
            [("topic", 10, "Intros"), ("decision", 90, "Forty a seat"), ("point", 150, "Launch Friday")];
        let (_dir, rec) = meeting(&LATER, &events, "They priced it at forty a seat.");
        let (lead, left) =
            super::lead(&rec, Some(&mark(12, 1, Some(1))), &[], 260 * SECOND, 265 * SECOND).unwrap();
        assert_eq!(
            lead,
            "Since the last prompt (00:12), in short:\n\
             <summary>\nThey priced it at forty a seat.\n</summary>\n\
             <new-events>\n- e002 01:30 decision: Forty a seat\n- e003 02:30 point: Launch Friday\n</new-events>\n\
             <last-60-seconds>\n\
             u0004 04:10 Marcus: And the launch is Friday.\n\
             u0005 04:20 Alice: Pam, when do we launch?\n</last-60-seconds>\n\
             More: pam transcript --from 00:12\n\n"
        );
        assert_eq!(left, Mark { at: 265 * SECOND, line: Some(UtteranceId(5)), event: Some(EventId(3)) });
        // Without a scribe there is no summary and there are no events.
        let (_dir, rec) = meeting(&LATER, &[], "");
        let (lead, _) = super::lead(&rec, Some(&mark(12, 1, None)), &[], 260 * SECOND, 265 * SECOND).unwrap();
        assert!(
            lead.contains("<summary>\n(none yet)\n</summary>\n<new-events>\n(none)\n</new-events>\n"),
            "{lead}"
        );
    }

    #[test]
    fn a_gap_is_short_while_its_lines_span_three_minutes_or_less() {
        // Each line lasts 4 s: these span 03:00, then 03:01.
        for (last, short) in [(176, true), (177, false)] {
            let (_dir, rec) = meeting(&[(0, 0, "Hi."), (100, 1, "So."), (last, 0, "Done.")], &[], "");
            let since = Mark { at: 0, line: None, event: None };
            let lead = super::lead(&rec, Some(&since), &[], last * SECOND, 600 * SECOND).unwrap().0;
            assert_eq!(lead.starts_with("Said since"), short, "{lead}");
        }
    }

    #[test]
    fn a_short_talk_after_a_long_silence_still_comes_whole() {
        // Ten minutes since the last prompt, but the lines since span 02:20: all of them, word for word.
        let (_dir, rec) =
            meeting(&[(0, 0, "Hi."), (480, 1, "Back to pricing."), (616, 0, "Pam, sum up?")], &[], "");
        let lead = super::lead(&rec, Some(&mark(4, 1, None)), &[], 616 * SECOND, 620 * SECOND).unwrap().0;
        assert!(
            lead.starts_with("Said since the last prompt (00:04), word for word:\n<lines>\nu0002 08:00 Marcus: Back to pricing.\n"),
            "{lead}"
        );
    }

    #[test]
    fn a_gap_is_short_while_its_lines_fit_the_recent_lines_share() {
        // `u0001 00:00 Alice: ` is 19 bytes, and 2,500 tokens are 6,250 bytes.
        let fits = "a".repeat(6_250 - 19);
        let over = "a".repeat(6_250 - 19 + 1);
        for (text, short) in [(&fits, true), (&over, false)] {
            let (_dir, rec) = meeting(&[(0, 0, text)], &[], "");
            let lead = super::lead(&rec, Some(&Mark { at: 0, line: None, event: None }), &[], 0, 10 * SECOND)
                .unwrap()
                .0;
            assert_eq!(lead.starts_with("Said since"), short, "{} bytes", text.len());
        }
    }

    #[test]
    fn many_new_events_collapse_the_oldest_first() {
        let events: Vec<(&str, u64, &str)> = (0..200)
            .map(|i| {
                (
                    if i % 40 == 0 { "topic" } else { "point" },
                    i,
                    "The remote should fit in one hand and have fewer buttons",
                )
            })
            .collect();
        let (_dir, rec) = meeting(&LATER, &events, "Remote design.");
        let lead =
            super::lead(&rec, Some(&mark(0, 0, Some(100))), &[], 260 * SECOND, 265 * SECOND).unwrap().0;
        let new = lead.split("<new-events>\n").nth(1).unwrap().split("</new-events>").next().unwrap();
        assert!(tokens(new) <= EVENTS_SHARE, "{} tokens", tokens(new));
        // Only events after the mark, the oldest down to their topics.
        assert!(new.starts_with("- … 20 events collapsed: e101..e120\n- e121 02:00 topic: "), "{new}");
        assert!(
            new.ends_with("- e200 03:19 point: The remote should fit in one hand and have fewer buttons\n"),
            "{new}"
        );
    }
}
