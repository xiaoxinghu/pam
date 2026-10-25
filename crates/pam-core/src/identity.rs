//! Who is who: the slot → `spk_N` map as published, the corrections overlay
//! (names, merges, moved lines), and the log of identity ops behind it, each
//! with an undo ID. Saved as `speakers.json`.
//!
//! The overlay is never edited in place: it is rebuilt from the log, so
//! undoing any op, not only the last, leaves the state as if it never happened.

use std::collections::BTreeMap;

use anyhow::{Result, bail};
use serde::{Deserialize, Serialize};

use crate::event::{CoreEvent, Line, SpkId, UndoId, UtteranceId};
use crate::intro::introduced_name;
use crate::pipeline::SlotMap;

/// `speakers.json`: slots → `spk_N`, the overlay on top, and the ops behind it.
/// Written by the core and read back by the agent commands.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Speakers {
    pub slots: BTreeMap<u32, SpkId>,
    #[serde(flatten)]
    pub overlay: Overlay,
    /// The ops in force, oldest first.
    pub log: Vec<Op>,
}

impl Speakers {
    /// A line's speaker: `moves[line]`, else `slots[slot]`, then merges followed.
    pub fn resolve(&self, line: UtteranceId, slot: Option<u32>) -> Option<SpkId> {
        self.overlay.speaker(line, slot.and_then(|s| self.slots.get(&s).copied()))
    }
}

/// The corrections on top of the slot map. Front-ends keep a copy in step
/// with [`Overlay::apply`] to label lines as the core does.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Overlay {
    pub names: BTreeMap<SpkId, String>,
    /// `from` → `into`.
    pub merges: BTreeMap<SpkId, SpkId>,
    pub moves: BTreeMap<UtteranceId, SpkId>,
}

impl Overlay {
    /// The speaker of line `line`, whose slot maps to `spk`: `moves[line]`,
    /// else `spk`, then merges followed.
    pub fn speaker(&self, line: UtteranceId, spk: Option<SpkId>) -> Option<SpkId> {
        let spk = self.moves.get(&line).copied().or(spk)?;
        Some(self.follow(spk))
    }

    /// The speaker `spk` now belongs to, after merges.
    pub fn follow(&self, mut spk: SpkId) -> SpkId {
        // A merge cycle would be a bug elsewhere; stop rather than loop.
        for _ in 0..=self.merges.len() {
            match self.merges.get(&spk) {
                Some(&into) => spk = into,
                None => break,
            }
        }
        spk
    }

    /// `spk`'s name, after merges.
    pub fn name(&self, spk: SpkId) -> Option<&str> {
        self.names.get(&self.follow(spk)).map(String::as_str)
    }

    /// How a speaker reads, after merges: a name, `spk_N`, or `unknown`.
    pub fn label(&self, spk: Option<SpkId>) -> String {
        match spk {
            Some(spk) => self.name(spk).map_or_else(|| self.follow(spk).to_string(), str::to_string),
            None => "unknown".into(),
        }
    }

    /// Follows the core: applies an identity event, ignores the rest.
    pub fn apply(&mut self, event: &CoreEvent) {
        match event {
            CoreEvent::SpeakerRenamed { spk, name, .. } => self.rename(*spk, name),
            CoreEvent::SpeakersMerged { from, into, .. } => {
                self.merge(*from, *into);
            }
            CoreEvent::LineMoved { line, to, .. } => self.move_line(*line, *to),
            CoreEvent::Undone { overlay, .. } => *self = overlay.clone(),
            _ => {}
        }
    }

    fn rename(&mut self, spk: SpkId, name: &str) {
        self.names.insert(spk, name.to_string());
    }

    /// A speaker merged into an unnamed one gives it their name. Says whether
    /// it did, so the core can tell where that name came from.
    fn merge(&mut self, from: SpkId, into: SpkId) -> bool {
        self.merges.insert(from, into);
        if !self.names.contains_key(&into)
            && let Some(name) = self.names.get(&from).cloned()
        {
            self.names.insert(into, name);
            return true;
        }
        false
    }

    fn move_line(&mut self, line: UtteranceId, to: SpkId) {
        self.moves.insert(line, to);
    }
}

/// One identity op in the log.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Op {
    pub undo: UndoId,
    #[serde(flatten)]
    pub kind: OpKind,
    pub by: By,
    /// Wall time, RFC 3339.
    pub at: String,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "op", rename_all = "lowercase")]
pub enum OpKind {
    Rename {
        spk: SpkId,
        name: String,
        /// The line that caused it, such as an introduction.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        line: Option<UtteranceId>,
    },
    Merge {
        from: SpkId,
        into: SpkId,
    },
    Move {
        line: UtteranceId,
        to: SpkId,
    },
}

/// What made an op.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum By {
    /// The intro fast path matched a line.
    Intro,
    /// Someone corrected it by hand, through a front-end.
    Manual,
    /// The scribe reported an introduction the fast path missed or got wrong.
    Scribe,
}

/// What the scribe's report of an introduction did.
#[derive(Debug, Clone, PartialEq)]
pub enum ScribeIntro {
    /// The speaker already had that name, from this op.
    Confirmed(UndoId),
    /// The speaker was unnamed, or named otherwise by an introduction: renamed.
    Renamed(Box<CoreEvent>),
    /// Left alone: someone named them by hand, or undid this very name.
    Skipped,
}

#[derive(Debug, Default, Clone)]
pub struct Identity {
    speakers: Speakers,
    /// For each named speaker, the op their name comes from, through merges.
    named_by: BTreeMap<SpkId, UndoId>,
    /// Ops undone so far, so the scribe doesn't bring back a name someone took away.
    undone: Vec<Op>,
    /// The last undo ID given out. IDs are never reused.
    last_undo: u32,
}

impl Identity {
    pub fn new() -> Identity {
        Identity::default()
    }

    /// Everything `speakers.json` holds.
    pub fn speakers(&self) -> &Speakers {
        &self.speakers
    }

    pub fn overlay(&self) -> &Overlay {
        &self.speakers.overlay
    }

    pub fn set_slots(&mut self, slots: SlotMap) {
        self.speakers.slots = slots.iter().collect();
    }

    /// The ops in force, oldest first.
    pub fn log(&self) -> &[Op] {
        &self.speakers.log
    }

    /// Who said `line`, corrections applied.
    pub fn speaker_of(&self, line: &Line) -> Option<SpkId> {
        self.overlay().speaker(line.id, line.spk)
    }

    /// Runs the intro fast path on a final line. Renames the line's speaker if
    /// the line introduces them and they have no name yet.
    pub fn intro(&mut self, line: &Line, assistant: &str, at: String) -> Option<CoreEvent> {
        let spk = self.speaker_of(line)?;
        if self.overlay().names.contains_key(&spk) {
            return None;
        }
        let name = introduced_name(&line.text, assistant)?;
        let undo = self.apply(OpKind::Rename { spk, name: name.clone(), line: Some(line.id) }, By::Intro, at);
        Some(CoreEvent::SpeakerRenamed { spk, name, undo })
    }

    /// Acts on the scribe's report that `spk` introduced themselves as `name`
    /// in `line`. It agrees with the name they have, or renames them: the
    /// scribe reads more context than the fast path, so it wins over it. A
    /// name given by hand wins over the scribe.
    pub fn scribe_intro(&mut self, spk: SpkId, name: &str, line: UtteranceId, at: String) -> ScribeIntro {
        let spk = self.overlay().follow(spk);
        let same = |a: &str| a.to_lowercase() == name.to_lowercase();
        if let Some(op) = self.naming(spk) {
            match &op.kind {
                OpKind::Rename { name: n, .. } if same(n) => return ScribeIntro::Confirmed(op.undo),
                _ if op.by == By::Manual => return ScribeIntro::Skipped,
                _ => {}
            }
        }
        // Undone ops keep the ID they named: follow it through merges made since.
        let overlay = self.overlay();
        let undone = |op: &Op| matches!(&op.kind, OpKind::Rename { spk: s, name: n, .. } if overlay.follow(*s) == spk && same(n));
        if self.undone.iter().any(undone) {
            return ScribeIntro::Skipped;
        }
        let name = name.to_string();
        let undo = self.apply(OpKind::Rename { spk, name: name.clone(), line: Some(line) }, By::Scribe, at);
        ScribeIntro::Renamed(Box::new(CoreEvent::SpeakerRenamed { spk, name, undo }))
    }

    /// The op in force that gave `spk` their name, if any, including a name a
    /// merge passed on.
    pub fn naming(&self, spk: SpkId) -> Option<&Op> {
        let undo = self.named_by.get(&self.overlay().follow(spk))?;
        self.log().iter().find(|op| op.undo == *undo)
    }

    /// Names `spk`, or the speaker it was merged into.
    pub fn rename(&mut self, spk: SpkId, name: &str, at: String) -> Result<CoreEvent> {
        self.known(spk)?;
        let name = name.split_whitespace().collect::<Vec<_>>().join(" ");
        if name.is_empty() {
            bail!("a name can't be empty");
        }
        let spk = self.overlay().follow(spk);
        if self.overlay().names.get(&spk) == Some(&name) {
            bail!("{spk} is already called {name}");
        }
        let undo = self.apply(OpKind::Rename { spk, name: name.clone(), line: None }, By::Manual, at);
        Ok(CoreEvent::SpeakerRenamed { spk, name, undo })
    }

    /// Makes `from` part of `into`: their lines read as `into`'s.
    pub fn merge(&mut self, from: SpkId, into: SpkId, at: String) -> Result<CoreEvent> {
        self.known(from)?;
        self.known(into)?;
        if from == into {
            bail!("can't merge {from} into itself");
        }
        let (from, into) = (self.overlay().follow(from), self.overlay().follow(into));
        if from == into {
            bail!("they are already one speaker, {}", self.overlay().label(Some(into)));
        }
        let undo = self.apply(OpKind::Merge { from, into }, By::Manual, at);
        Ok(CoreEvent::SpeakersMerged { from, into, undo })
    }

    /// Gives one line to another speaker.
    pub fn move_line(&mut self, line: &Line, to: SpkId, at: String) -> Result<CoreEvent> {
        self.known(to)?;
        let to = self.overlay().follow(to);
        if self.speaker_of(line) == Some(to) {
            bail!("{} is already {}'s", line.id, self.overlay().label(Some(to)));
        }
        let undo = self.apply(OpKind::Move { line: line.id, to }, By::Manual, at);
        Ok(CoreEvent::LineMoved { line: line.id, to, undo })
    }

    /// Takes `undo` out of the log and rebuilds the overlay without it.
    pub fn undo(&mut self, undo: UndoId) -> Result<CoreEvent> {
        let Some(i) = self.speakers.log.iter().position(|op| op.undo == undo) else {
            if (1..=self.last_undo).contains(&undo.0) {
                bail!("{undo} was already undone");
            }
            bail!("there is no change {undo} to undo");
        };
        self.undone.push(self.speakers.log.remove(i));
        self.rebuild();
        Ok(CoreEvent::Undone { undo, overlay: self.overlay().clone() })
    }

    /// Only speakers the slot map has given out can be corrected.
    fn known(&self, spk: SpkId) -> Result<()> {
        if self.speakers.slots.values().any(|&s| s == spk) {
            return Ok(());
        }
        let all: Vec<String> = self.speakers.slots.values().map(SpkId::to_string).collect();
        if all.is_empty() {
            bail!("no speaker {spk}: nobody has spoken yet");
        }
        bail!("no speaker {spk}: speakers are {}", all.join(", "));
    }

    fn apply(&mut self, kind: OpKind, by: By, at: String) -> UndoId {
        self.last_undo += 1;
        let undo = UndoId(self.last_undo);
        self.speakers.log.push(Op { undo, kind, by, at });
        self.rebuild();
        undo
    }

    /// Replays the log: the overlay, and where each name came from. The only
    /// place either is worked out, after a new op and after an undo alike.
    fn rebuild(&mut self) {
        let mut overlay = Overlay::default();
        let mut named_by = BTreeMap::new();
        for op in &self.speakers.log {
            match &op.kind {
                OpKind::Rename { spk, name, .. } => {
                    overlay.rename(*spk, name);
                    named_by.insert(*spk, op.undo);
                }
                OpKind::Merge { from, into } => {
                    if overlay.merge(*from, *into)
                        && let Some(&undo) = named_by.get(from)
                    {
                        named_by.insert(*into, undo);
                    }
                }
                OpKind::Move { line, to } => overlay.move_line(*line, *to),
            }
        }
        self.speakers.overlay = overlay;
        self.named_by = named_by;
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    const AT: &str = "2026-10-11T10:00:00+00:00";

    fn line(id: u32, spk: Option<u32>, text: &str) -> Line {
        Line {
            id: UtteranceId(id),
            t0: 0,
            t1: 0,
            slot: spk,
            spk: spk.map(SpkId),
            lang: None,
            text: text.into(),
        }
    }

    fn intro(identity: &mut Identity, l: &Line) -> Option<CoreEvent> {
        identity.intro(l, "Pam", AT.into())
    }

    /// An identity with speakers spk_0..spk_{n-1}.
    fn with_speakers(n: u32) -> Identity {
        let mut slots = SlotMap::default();
        for s in 0..n {
            slots.add(s);
        }
        let mut id = Identity::new();
        id.set_slots(slots);
        id
    }

    fn label(id: &Identity, l: &Line) -> String {
        id.overlay().label(id.speaker_of(l))
    }

    fn err(r: Result<CoreEvent>) -> String {
        r.unwrap_err().to_string()
    }

    #[test]
    fn an_intro_names_the_lines_speaker() {
        let mut id = Identity::new();
        assert_eq!(intro(&mut id, &line(1, Some(0), "Hi, how are you?")), None);
        let e = intro(&mut id, &line(2, Some(0), "I'm Alice, I run product.")).unwrap();
        assert_eq!(e, CoreEvent::SpeakerRenamed { spk: SpkId(0), name: "Alice".into(), undo: UndoId(1) });
        assert_eq!(id.overlay().label(Some(SpkId(0))), "Alice");
        assert_eq!(id.overlay().label(Some(SpkId(1))), "spk_1");
        assert_eq!(id.overlay().label(None), "unknown");
        assert!(matches!(id.log()[0].kind, OpKind::Rename { line: Some(UtteranceId(2)), .. }));
    }

    #[test]
    fn a_named_speaker_or_a_line_without_one_is_skipped() {
        let mut id = Identity::new();
        intro(&mut id, &line(1, Some(0), "I'm Alice.")).unwrap();
        assert_eq!(intro(&mut id, &line(2, Some(0), "This is Sam, from sales.")), None);
        assert_eq!(intro(&mut id, &line(3, None, "I'm Sam.")), None);
        assert_eq!(id.log().len(), 1);
    }

    #[test]
    fn undo_rebuilds_the_names_from_what_is_left() {
        let mut id = Identity::new();
        intro(&mut id, &line(1, Some(0), "I'm Alice.")).unwrap();
        intro(&mut id, &line(2, Some(1), "I'm Sam.")).unwrap();
        let e = id.undo(UndoId(1)).unwrap();
        let names = BTreeMap::from([(SpkId(1), "Sam".to_string())]);
        assert_eq!(
            e,
            CoreEvent::Undone { undo: UndoId(1), overlay: Overlay { names, ..Overlay::default() } }
        );
        // Once unnamed, the speaker can be named by a later intro, with a new ID.
        let e = intro(&mut id, &line(3, Some(0), "Sorry, I'm Alicia actually.")).unwrap();
        assert_eq!(e, CoreEvent::SpeakerRenamed { spk: SpkId(0), name: "Alicia".into(), undo: UndoId(3) });
    }

    #[test]
    fn undo_says_why_it_cannot() {
        let mut id = Identity::new();
        intro(&mut id, &line(1, Some(0), "I'm Alice.")).unwrap();
        id.undo(UndoId(1)).unwrap();
        assert_eq!(err(id.undo(UndoId(1))), "x1 was already undone");
        assert_eq!(err(id.undo(UndoId(7))), "there is no change x7 to undo");
    }

    #[test]
    fn a_manual_rename_replaces_a_name_and_undo_brings_it_back() {
        let mut id = with_speakers(2);
        intro(&mut id, &line(1, Some(1), "Marcus here.")).unwrap();
        let e = id.rename(SpkId(1), "  Marc   Smith ", AT.into()).unwrap();
        assert_eq!(
            e,
            CoreEvent::SpeakerRenamed { spk: SpkId(1), name: "Marc Smith".into(), undo: UndoId(2) }
        );
        assert_eq!(id.log()[1].by, By::Manual);
        assert_eq!(err(id.rename(SpkId(1), "Marc Smith", AT.into())), "spk_1 is already called Marc Smith");
        assert_eq!(err(id.rename(SpkId(0), " ", AT.into())), "a name can't be empty");
        assert_eq!(err(id.rename(SpkId(5), "Sam", AT.into())), "no speaker spk_5: speakers are spk_0, spk_1");
        assert_eq!(
            err(Identity::new().rename(SpkId(0), "Sam", AT.into())),
            "no speaker spk_0: nobody has spoken yet"
        );
        id.undo(UndoId(2)).unwrap();
        assert_eq!(id.overlay().label(Some(SpkId(1))), "Marcus");
    }

    #[test]
    fn a_merge_relabels_the_lines_and_keeps_a_name() {
        let mut id = with_speakers(4);
        let (u1, u2) = (line(1, Some(1), "Marcus here."), line(2, Some(3), "We ship Thursday."));
        intro(&mut id, &u1).unwrap();
        // Merging the named voice into the unnamed one keeps the name too.
        let e = id.merge(SpkId(1), SpkId(3), AT.into()).unwrap();
        assert_eq!(e, CoreEvent::SpeakersMerged { from: SpkId(1), into: SpkId(3), undo: UndoId(2) });
        assert_eq!((label(&id, &u1), label(&id, &u2)), ("Marcus".into(), "Marcus".into()));
        // A merged-away ID stands for the speaker it joined.
        let e = id.merge(SpkId(2), SpkId(1), AT.into()).unwrap();
        assert_eq!(e, CoreEvent::SpeakersMerged { from: SpkId(2), into: SpkId(3), undo: UndoId(3) });
        assert_eq!(err(id.merge(SpkId(1), SpkId(2), AT.into())), "they are already one speaker, Marcus");
        assert_eq!(err(id.merge(SpkId(0), SpkId(0), AT.into())), "can't merge spk_0 into itself");
        // An intro on a merged voice doesn't rename the speaker it joined.
        assert_eq!(intro(&mut id, &line(3, Some(2), "I'm Jordan.")), None);

        id.undo(UndoId(2)).unwrap();
        assert_eq!((label(&id, &u1), label(&id, &u2)), ("Marcus".into(), "spk_3".into()));
        assert_eq!(id.overlay().follow(SpkId(2)), SpkId(3), "x3 stays in force");
    }

    #[test]
    fn a_moved_line_reads_as_its_new_speaker_until_undone() {
        let mut id = with_speakers(3);
        let wrong = line(7, Some(0), "Sure, I'll send them tonight.");
        intro(&mut id, &line(1, Some(2), "My name is Jordan.")).unwrap();
        let e = id.move_line(&wrong, SpkId(2), AT.into()).unwrap();
        assert_eq!(e, CoreEvent::LineMoved { line: UtteranceId(7), to: SpkId(2), undo: UndoId(2) });
        assert_eq!(label(&id, &wrong), "Jordan");
        assert_eq!(err(id.move_line(&wrong, SpkId(2), AT.into())), "u0007 is already Jordan's");
        // Moving again replaces the move; undoing that brings the first one back.
        id.move_line(&wrong, SpkId(1), AT.into()).unwrap();
        assert_eq!(label(&id, &wrong), "spk_1");
        id.undo(UndoId(3)).unwrap();
        assert_eq!(label(&id, &wrong), "Jordan");
        id.undo(UndoId(2)).unwrap();
        assert_eq!(label(&id, &wrong), "spk_0");
        // A line with no speaker can be given one.
        id.move_line(&line(8, None, "Mm."), SpkId(0), AT.into()).unwrap();
    }

    #[test]
    fn a_front_end_overlay_follows_the_events() {
        let mut id = with_speakers(3);
        let events = [
            intro(&mut id, &line(1, Some(0), "I'm Alice.")).unwrap(),
            id.merge(SpkId(0), SpkId(1), AT.into()).unwrap(),
            id.move_line(&line(2, Some(1), "Hi."), SpkId(2), AT.into()).unwrap(),
            id.rename(SpkId(2), "Jordan", AT.into()).unwrap(),
        ];
        let mut mirror = Overlay::default();
        for e in &events {
            mirror.apply(e);
        }
        assert_eq!(&mirror, id.overlay());
        mirror.apply(&id.undo(UndoId(2)).unwrap());
        assert_eq!(&mirror, id.overlay());
    }

    #[test]
    fn the_scribe_confirms_or_corrects_names_but_not_a_persons() {
        let mut id = with_speakers(4);
        intro(&mut id, &line(1, Some(0), "I'm Alice.")).unwrap();
        // Same name, any case: confirmed, pointing at the fast path's op.
        let confirmed = id.scribe_intro(SpkId(0), "alice", UtteranceId(1), AT.into());
        assert_eq!(confirmed, ScribeIntro::Confirmed(UndoId(1)));
        // Missed by the fast path: named, by the scribe.
        let e = id.scribe_intro(SpkId(1), "Marcus", UtteranceId(2), AT.into());
        let renamed = CoreEvent::SpeakerRenamed { spk: SpkId(1), name: "Marcus".into(), undo: UndoId(2) };
        assert_eq!(e, ScribeIntro::Renamed(Box::new(renamed)));
        assert_eq!(id.log()[1].by, By::Scribe);
        assert!(matches!(id.log()[1].kind, OpKind::Rename { line: Some(UtteranceId(2)), .. }));
        // "This is Jordan" was said by spk_2 about someone else: the scribe corrects it.
        intro(&mut id, &line(3, Some(2), "This is Jordan, design.")).unwrap();
        let e = id.scribe_intro(SpkId(2), "Priya", UtteranceId(4), AT.into());
        assert!(
            matches!(e, ScribeIntro::Renamed(b) if matches!(*b, CoreEvent::SpeakerRenamed { undo: UndoId(4), .. }))
        );
        assert_eq!(id.overlay().label(Some(SpkId(2))), "Priya");
        // Undoing the correction brings back the fast path's name, and the
        // scribe doesn't redo what a person undid.
        id.undo(UndoId(4)).unwrap();
        assert_eq!(id.overlay().label(Some(SpkId(2))), "Jordan");
        assert_eq!(id.scribe_intro(SpkId(2), "Priya", UtteranceId(4), AT.into()), ScribeIntro::Skipped);
        // A name given by hand wins; a merged-away ID stands for its speaker.
        id.rename(SpkId(3), "Sam", AT.into()).unwrap();
        assert_eq!(id.scribe_intro(SpkId(3), "Samuel", UtteranceId(5), AT.into()), ScribeIntro::Skipped);
        id.merge(SpkId(1), SpkId(3), AT.into()).unwrap();
        let confirmed = id.scribe_intro(SpkId(1), "sam", UtteranceId(6), AT.into());
        assert_eq!(confirmed, ScribeIntro::Confirmed(UndoId(5)));
    }

    #[test]
    fn a_name_given_by_hand_still_wins_after_a_merge_passes_it_on() {
        let mut id = with_speakers(2);
        id.rename(SpkId(0), "Sam", AT.into()).unwrap();
        // spk_1 had no name, so it takes Sam's.
        id.merge(SpkId(0), SpkId(1), AT.into()).unwrap();
        assert_eq!(id.naming(SpkId(1)).map(|op| op.undo), Some(UndoId(1)));
        for spk in [SpkId(0), SpkId(1)] {
            assert_eq!(id.scribe_intro(spk, "Samuel", UtteranceId(3), AT.into()), ScribeIntro::Skipped);
        }
        assert_eq!(id.overlay().label(Some(SpkId(1))), "Sam");
    }

    #[test]
    fn an_undone_name_stays_undone_after_a_merge() {
        let mut id = with_speakers(2);
        let ScribeIntro::Renamed(renamed) = id.scribe_intro(SpkId(0), "Sam", UtteranceId(1), AT.into())
        else {
            panic!("not renamed");
        };
        let CoreEvent::SpeakerRenamed { undo, .. } = *renamed else { panic!("not a rename") };
        id.undo(undo).unwrap();
        id.merge(SpkId(0), SpkId(1), AT.into()).unwrap();
        for spk in [SpkId(0), SpkId(1)] {
            assert_eq!(id.scribe_intro(spk, "sam", UtteranceId(2), AT.into()), ScribeIntro::Skipped);
        }
        assert_eq!(id.overlay().label(Some(SpkId(1))), "spk_1");
    }

    #[test]
    fn speakers_json_reads_back_what_was_written() {
        let mut id = with_speakers(4);
        intro(&mut id, &line(1, Some(0), "I'm Alice.")).unwrap();
        id.merge(SpkId(3), SpkId(1), AT.into()).unwrap();
        id.move_line(&line(2, Some(0), "Yes."), SpkId(2), AT.into()).unwrap();
        let saved = serde_json::to_value(id.speakers()).unwrap();
        assert_eq!(saved["slots"], json!({"0": "spk_0", "1": "spk_1", "2": "spk_2", "3": "spk_3"}));
        assert_eq!(saved["merges"], json!({"spk_3": "spk_1"}));
        assert_eq!(saved["moves"], json!({"u0002": "spk_2"}));
        assert_eq!(
            saved["log"],
            json!([
                {"undo": "x1", "op": "rename", "spk": "spk_0", "name": "Alice", "line": "u0001", "by": "intro", "at": AT},
                {"undo": "x2", "op": "merge", "from": "spk_3", "into": "spk_1", "by": "manual", "at": AT},
                {"undo": "x3", "op": "move", "line": "u0002", "to": "spk_2", "by": "manual", "at": AT},
            ])
        );
        let back: Speakers = serde_json::from_value(saved).unwrap();
        assert_eq!(&back, id.speakers());
        assert_eq!(back.resolve(UtteranceId(9), Some(3)), Some(SpkId(1)));
        assert_eq!(back.resolve(UtteranceId(2), Some(0)), Some(SpkId(2)));
    }
}
