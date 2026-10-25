//! The meeting folder: `meeting.json`, `transcript.jsonl`, `speakers.json`,
//! `events.jsonl`, `summary.json` and the files rendered from them: `transcript.md`,
//! `notes.md` and `context.md`.

use std::fs::{self, File, OpenOptions};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use chrono::{DateTime, Local, SecondsFormat};
use serde::{Deserialize, Serialize};

use crate::Ts;
use crate::clock::mmss;
use crate::context;
use crate::event::{EventId, Line, UtteranceId};
use crate::identity::{Identity, Overlay};
use crate::models::{AgentInfo, IntentInfo, ScribeInfo};
use crate::notes::Notes;
use crate::record::{Event, MeetingRecord, Summary, read_jsonl};
use crate::usage::Usage;

/// `meeting.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default)]
pub struct MeetingInfo {
    pub title: String,
    pub started_at: String,
    pub assistant_name: String,
    /// What was listened to, e.g. `mic` or a file path.
    pub input: String,
    pub asr: String,
    /// The language the transcriber listened for: a code, or `auto` for
    /// detection.
    pub language: String,
    pub diarizer: String,
    /// The scribe's model as chosen and checked at the start; none
    /// without one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub scribe: Option<ScribeInfo>,
    /// The intent check's model as chosen and checked at the start:
    /// the scribe's unless one was named; none without one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub intent: Option<IntentInfo>,
    /// `recording` or `stopped`.
    pub status: String,
    /// Names known in advance, passed to the transcriber as hints.
    #[serde(skip_serializing_if = "Vec::is_empty")]
    pub attendees: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stopped_at: Option<String>,
    /// Audio heard, in samples.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub duration: Option<Ts>,
    /// The agent as the meeting found it at the start; none without one.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub agent: Option<AgentInfo>,
    /// The countdown on the agent's questions, in seconds, when `pam task
    /// ask --timeout` doesn't set one; none without an agent.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub question_timeout: Option<u64>,
    /// The tokens and cost of the meeting's model calls so far.
    pub usage: Usage,
}

/// Where meeting folders go: `$PAM_MEETINGS_ROOT`, or `~/Pam/meetings`.
pub fn meetings_root() -> Result<PathBuf> {
    if let Some(root) = std::env::var_os("PAM_MEETINGS_ROOT").filter(|r| !r.is_empty()) {
        return Ok(PathBuf::from(root));
    }
    let home = std::env::var_os("HOME").context("HOME is not set; set PAM_MEETINGS_ROOT")?;
    Ok(PathBuf::from(home).join("Pam").join("meetings"))
}

pub fn timestamp(t: DateTime<Local>) -> String {
    t.to_rfc3339_opts(SecondsFormat::Secs, false)
}

/// One line of `transcript.jsonl`. The speaker is resolved through `speakers.json`.
#[derive(Serialize)]
struct TranscriptLine<'a> {
    id: UtteranceId,
    t0: Ts,
    t1: Ts,
    slot: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    lang: Option<&'a str>,
    text: &'a str,
}

pub struct MeetingFolder {
    path: PathBuf,
    title: String,
    transcript: File,
}

impl MeetingFolder {
    /// Creates `<root>/<yyyy-mm-dd-hhmm>[-<title>]`, adding `-2`, `-3`… if it exists.
    pub fn create(root: &Path, title: &str, started: DateTime<Local>) -> Result<MeetingFolder> {
        fs::create_dir_all(root).with_context(|| format!("creating {}", root.display()))?;
        let mut base = started.format("%Y-%m-%d-%H%M").to_string();
        let slug = slug(title);
        if !slug.is_empty() {
            base = format!("{base}-{slug}");
        }
        let mut path = root.join(&base);
        let mut n = 1;
        loop {
            match fs::create_dir(&path) {
                Ok(()) => break,
                Err(e) if e.kind() == std::io::ErrorKind::AlreadyExists => {
                    n += 1;
                    path = root.join(format!("{base}-{n}"));
                }
                Err(e) => return Err(e).with_context(|| format!("creating {}", path.display())),
            }
        }
        let transcript = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path.join("transcript.jsonl"))
            .context("creating transcript.jsonl")?;
        let folder = MeetingFolder { path, title: title.to_string(), transcript };
        folder.write_speakers(&Identity::new())?;
        folder.write_transcript_md(&[], &Identity::new())?;
        let log = EventLog::lock(&folder.path)?;
        log.write_notes(&folder.title, &Notes::default(), &Overlay::default())?;
        Ok(folder)
    }

    pub fn path(&self) -> &Path {
        &self.path
    }

    pub fn title(&self) -> &str {
        &self.title
    }

    pub fn write_info(&self, info: &MeetingInfo) -> Result<()> {
        write_info(&self.path, info)
    }

    pub fn append_line(&mut self, line: &Line) -> Result<()> {
        let mut json = serde_json::to_vec(&TranscriptLine {
            id: line.id,
            t0: line.t0,
            t1: line.t1,
            slot: line.slot,
            lang: line.lang.as_deref(),
            text: &line.text,
        })?;
        json.push(b'\n');
        append_whole(&mut self.transcript, &json).context("appending to transcript.jsonl")
    }

    pub fn write_speakers(&self, identity: &Identity) -> Result<()> {
        self.replace("speakers.json", &serde_json::to_vec_pretty(identity.speakers())?)
    }

    /// Re-renders `transcript.md` from every final line so far, corrections applied.
    pub fn write_transcript_md(&self, lines: &[Line], identity: &Identity) -> Result<()> {
        self.replace("transcript.md", render_transcript(&self.title, lines, identity).as_bytes())
    }

    fn replace(&self, name: &str, bytes: &[u8]) -> Result<()> {
        replace(&self.path, name, bytes)
    }
}

/// Writes `summary.json`.
pub fn write_summary(dir: &Path, summary: &Summary) -> Result<()> {
    replace(dir, "summary.json", &serde_json::to_vec_pretty(summary)?)
}

/// `events.jsonl`, held by this process until dropped. Whoever adds events or
/// renders `notes.md` or `context.md` holds it, so the meeting's scribe and `pam note` never
/// give out the same event ID or write a rendered file at once.
pub struct EventLog {
    dir: PathBuf,
    file: File,
}

impl EventLog {
    /// Waits for the lock on `dir/events.jsonl`, creating the file if needed.
    pub fn lock(dir: &Path) -> Result<EventLog> {
        let path = dir.join("events.jsonl");
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(&path)
            .with_context(|| format!("opening {}", path.display()))?;
        file.lock().with_context(|| format!("locking {}", path.display()))?;
        Ok(EventLog { dir: dir.to_path_buf(), file })
    }

    pub fn events(&self) -> Result<Vec<Event>> {
        read_jsonl(&self.dir.join("events.jsonl"))
    }

    /// Appends `new`, numbered after the highest ID so far, and returns them as saved.
    pub fn append(&mut self, new: Vec<Event>) -> Result<Vec<Event>> {
        let last = self.events()?.iter().map(|e| e.id.0).max().unwrap_or(0);
        let mut saved = Vec::new();
        let mut bytes = Vec::new();
        for (n, mut e) in (last + 1..).zip(new) {
            e.id = EventId(n);
            serde_json::to_writer(&mut bytes, &e)?;
            bytes.push(b'\n');
            saved.push(e);
        }
        append_whole(&mut self.file, &bytes).context("appending to events.jsonl")?;
        Ok(saved)
    }

    /// Re-renders `notes.md`, with the corrections in `overlay` applied.
    pub fn write_notes(&self, title: &str, notes: &Notes, overlay: &Overlay) -> Result<()> {
        replace(&self.dir, "notes.md", notes.render(title, overlay).as_bytes())
    }

    /// Re-renders `context.md` from the folder as it is now.
    pub fn write_context(&self) -> Result<()> {
        let rec = MeetingRecord::load(&self.dir).context("rendering context.md")?;
        replace(&self.dir, "context.md", context::render(&rec).as_bytes())
    }

    /// Re-renders `notes.md` and `context.md` after the events changed. They
    /// don't depend on each other, so each is written even if the other
    /// fails; the first failure is returned.
    pub fn write_notes_and_context(&self, title: &str, notes: &Notes, overlay: &Overlay) -> Result<()> {
        let written = self.write_notes(title, notes, overlay);
        let context = self.write_context();
        written.and(context)
    }
}

/// The readable transcript: one paragraph per line, led by its ID, with
/// names, merges and moved lines applied.
fn render_transcript(title: &str, lines: &[Line], identity: &Identity) -> String {
    let mut out =
        if title.is_empty() { "# Transcript\n".to_string() } else { format!("# Transcript: {title}\n") };
    out.push_str(
        "\n<!-- Rendered from transcript.jsonl and speakers.json on every change. Do not edit. -->\n",
    );
    for l in lines {
        out.push_str(&format!(
            "\n`{}` {} **{}:** {}\n",
            l.id,
            mmss(l.t0),
            identity.overlay().label(identity.speaker_of(l)),
            l.text
        ));
    }
    out
}

/// Writes `meeting.json` in the meeting folder `dir`.
pub fn write_info(dir: &Path, info: &MeetingInfo) -> Result<()> {
    replace(dir, "meeting.json", &serde_json::to_vec_pretty(info)?)
}

/// Writes a whole file through a temporary file, so readers never see half of it.
pub fn replace(dir: &Path, name: &str, bytes: &[u8]) -> Result<()> {
    let tmp = dir.join(format!(".{name}.tmp"));
    fs::write(&tmp, bytes).with_context(|| format!("writing {name}"))?;
    fs::rename(&tmp, dir.join(name)).with_context(|| format!("writing {name}"))
}

/// An append-only file that can be cut back to an earlier length.
trait Log: Write {
    fn size(&self) -> io::Result<u64>;
    fn cut(&mut self, len: u64) -> io::Result<()>;
}

impl Log for File {
    fn size(&self) -> io::Result<u64> {
        Ok(self.metadata()?.len())
    }
    fn cut(&mut self, len: u64) -> io::Result<()> {
        self.set_len(len)
    }
}

/// [`append_whole`] for a file.
pub fn append_to(file: &mut File, bytes: &[u8]) -> io::Result<()> {
    append_whole(file, bytes)
}

/// Appends `bytes` whole or not at all: a failed write (a full disk) is cut
/// back, so the file never ends in half a line.
fn append_whole(log: &mut impl Log, bytes: &[u8]) -> io::Result<()> {
    let len = log.size()?;
    match log.write_all(bytes) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = log.cut(len);
            Err(e)
        }
    }
}

/// Lowercase letters and digits joined by dashes, at most 40 characters.
fn slug(title: &str) -> String {
    let words: Vec<String> = title
        .split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect();
    let mut out = String::new();
    for w in words {
        if out.len() + w.len() + 1 > 40 {
            break;
        }
        if !out.is_empty() {
            out.push('-');
        }
        out.push_str(&w);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::SpkId;

    #[test]
    fn a_failed_append_leaves_no_half_line() {
        /// A disk with room for `room` more bytes.
        struct Disk {
            data: Vec<u8>,
            room: usize,
        }
        impl Write for Disk {
            fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
                let n = buf.len().min(self.room);
                if n == 0 {
                    return Err(io::Error::new(io::ErrorKind::StorageFull, "disk full"));
                }
                self.data.extend_from_slice(&buf[..n]);
                self.room -= n;
                Ok(n)
            }
            fn flush(&mut self) -> io::Result<()> {
                Ok(())
            }
        }
        impl Log for Disk {
            fn size(&self) -> io::Result<u64> {
                Ok(self.data.len() as u64)
            }
            fn cut(&mut self, len: u64) -> io::Result<()> {
                self.data.truncate(len as usize);
                Ok(())
            }
        }
        let mut disk = Disk { data: Vec::new(), room: 12 };
        append_whole(&mut disk, b"{\"id\":1}\n").unwrap();
        assert!(append_whole(&mut disk, b"{\"id\":2}\n").is_err());
        assert_eq!(disk.data, b"{\"id\":1}\n");
    }

    #[test]
    fn slugs_titles() {
        assert_eq!(slug("Onboarding sync: v2!"), "onboarding-sync-v2");
        assert_eq!(slug("   "), "");
    }

    #[test]
    fn folder_names_never_collide() {
        let root = tempfile::tempdir().unwrap();
        let t = Local::now();
        let a = MeetingFolder::create(root.path(), "Sync", t).unwrap();
        let b = MeetingFolder::create(root.path(), "Sync", t).unwrap();
        assert_ne!(a.path(), b.path());
        assert!(b.path().to_string_lossy().ends_with("-sync-2"));
        let speakers: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(a.path().join("speakers.json")).unwrap()).unwrap();
        assert_eq!(
            speakers,
            serde_json::json!({"slots": {}, "names": {}, "merges": {}, "moves": {}, "log": []})
        );
        assert!(
            fs::read_to_string(a.path().join("transcript.md")).unwrap().starts_with("# Transcript: Sync\n")
        );
        // notes.md has the whole template from the start, scribe or not.
        let notes = fs::read_to_string(a.path().join("notes.md")).unwrap();
        assert_eq!(notes, Notes::build("", &[], &[]).render("Sync", &Overlay::default()));
        assert!(
            notes.contains("## Discussion\n\nNone yet.\n")
                && notes.contains("## Action items\n\nNone yet.\n"),
            "{notes}"
        );
    }

    #[test]
    fn renders_the_transcript_with_names_applied() {
        let mut identity = Identity::new();
        let line = |id, spk: Option<u32>, t0, text: &str| Line {
            id: UtteranceId(id),
            t0,
            t1: t0,
            slot: spk,
            spk: spk.map(SpkId),
            lang: None,
            text: text.into(),
        };
        let lines = [
            line(1, Some(0), 16_000, "Sorry I'm late."),
            line(2, None, 32_000, "Mm."),
            line(3, Some(0), 75 * 16_000, "I'm Sam, from sales."),
        ];
        identity.intro(&lines[2], "Pam", "2026-10-11T10:00:00+00:00".into()).unwrap();
        assert_eq!(
            render_transcript("", &lines, &identity),
            "# Transcript\n\n<!-- Rendered from transcript.jsonl and speakers.json on every change. Do not edit. -->\n\n\
             `u0001` 00:01 **Sam:** Sorry I'm late.\n\n\
             `u0002` 00:02 **unknown:** Mm.\n\n\
             `u0003` 01:15 **Sam:** I'm Sam, from sales.\n"
        );
    }
}
