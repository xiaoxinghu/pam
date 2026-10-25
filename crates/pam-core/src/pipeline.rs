//! From backend updates to lines: slot → `spk_N` mapping, merging words to
//! speakers, line building, the preview and the speaking indicator.
//!
//! [`Pipeline`] is pure: it takes backend updates and returns events, with no
//! threads, clock or files, so the rules can be tested on fake scripts.

use std::collections::{BTreeMap, VecDeque};

use pam_speech::{SpeakerSpan, SpeakerUpdate, Update, Word, join_words};

use crate::Ts;
use crate::clock::SECOND;
use crate::event::{CoreEvent, Line, PreviewLine, SpkId, UtteranceId};

/// A word with no overlapping span takes the nearest span within this distance.
const NEAREST: Ts = SECOND;
/// The merge rule looks at each word from this long before its start, because
/// a streaming transcriber stamps a word after it was said: NeMo-Speech gives
/// most words one 80 ms frame at emission.
const LEAD: Ts = SECOND / 5;
/// A silence longer than this ends a line whose last word ends a sentence.
const PAUSE: Ts = SECOND;
/// A silence longer than this ends a line in the middle of a sentence:
/// people pause that long to think, and the sentence goes on.
const MID_SENTENCE_PAUSE: Ts = SECOND * 5 / 2;
/// A line that would grow longer than this is closed first.
const MAX_LINE: Ts = 20 * SECOND;
/// A line closed at [`MAX_LINE`] ends at the best place in this last stretch:
/// after its last sentence end there, else at its longest pause, rather than at
/// the word that crosses it, which would split a sentence with no pause.
const CAP_WINDOW: Ts = 5 * SECOND;
/// How long after its end a final word may still arrive. A line is closed on a
/// pause only once the transcriber has heard this far past the pause.
const WORD_GRACE: Ts = 2 * SECOND;
/// The speakers whose spans end this close to the newest span's end are the
/// ones speaking. Without an input level, the newest span must also reach this
/// close to now.
const ACTIVE_WINDOW: Ts = SECOND * 3 / 10;
/// With an input level, the newest span may end this long before now and
/// still say who is speaking: the tracker labels audio seconds after it is
/// heard (NeMo-Speech: up to 2.88 s plus one 2.56 s chunk).
const LABEL_LAG: Ts = 6 * SECOND;

/// Speech on its way to a final line: final words not in one yet, as the
/// line they would make so far. Its speaker is the tracker's so far, which
/// may still change, and so may where the line ends.
#[derive(Debug, Clone, PartialEq)]
pub struct Speech {
    pub spk: Option<SpkId>,
    /// In order; each is one utterance the transcriber made final.
    pub utterances: Vec<Utterance>,
}

/// The words the transcriber heard between two pauses (or the first part of
/// an utterance that began in a line before).
#[derive(Debug, Clone, PartialEq)]
pub struct Utterance {
    pub t0: Ts,
    pub t1: Ts,
    pub text: String,
}

/// Backend slots → permanent meeting speaker IDs, allocated in order of each
/// slot's first final span.
#[derive(Debug, Default, Clone)]
pub struct SlotMap {
    slots: BTreeMap<u32, SpkId>,
}

impl SlotMap {
    /// Maps `slot`, returning the new ID if it had none yet.
    pub fn add(&mut self, slot: u32) -> Option<SpkId> {
        if self.slots.contains_key(&slot) {
            return None;
        }
        let spk = SpkId(self.slots.len() as u32);
        self.slots.insert(slot, spk);
        Some(spk)
    }

    pub fn get(&self, slot: u32) -> Option<SpkId> {
        self.slots.get(&slot).copied()
    }

    pub fn iter(&self) -> impl Iterator<Item = (u32, SpkId)> + '_ {
        self.slots.iter().map(|(s, k)| (*s, *k))
    }

    /// The map as it was right after `spk` was added. IDs are given out in order.
    pub fn up_to(&self, spk: SpkId) -> SlotMap {
        SlotMap { slots: self.iter().filter(|(_, k)| *k <= spk).collect() }
    }
}

struct OpenLine {
    slot: Option<u32>,
    words: Vec<Word>,
    /// Where the utterance of the last word begins in `words`.
    utterance: usize,
}

impl OpenLine {
    fn t0(&self) -> Ts {
        self.words[0].start
    }
    fn last_end(&self) -> Ts {
        self.words.last().map_or(0, |w| w.end)
    }
    /// The silence that ends the line: short after a sentence, longer within one.
    fn pause(&self) -> Ts {
        match self.words.last().is_some_and(|w| ends_sentence(&w.text)) {
            true => PAUSE,
            false => MID_SENTENCE_PAUSE,
        }
    }

    /// Splits the line before word `k`: the words from `k` on become a line of
    /// their own, returned.
    fn split_before(&mut self, k: usize) -> OpenLine {
        let words = self.words.split_off(k);
        let utterance = self.utterance.saturating_sub(k);
        if self.utterance >= k {
            self.utterance = 0;
        }
        OpenLine { slot: self.slot, words, utterance }
    }

    /// The first language reported for the line's words. A line holds one
    /// language: a word reported in another one starts a new line.
    fn lang(&self) -> Option<&str> {
        lang(&self.words)
    }

    /// Splits off the last utterance when its language doesn't fit the
    /// line's: then it is a turn of its own, maybe one the backend couldn't
    /// name a language for. Words without a language can't end a line by
    /// themselves, as they may be a name inside a sentence.
    fn split(&mut self) -> Option<OpenLine> {
        let (before, last) = self.words.split_at(self.utterance);
        if before.is_empty() || last.is_empty() || fits(lang(before), lang(last)) {
            return None;
        }
        let words = self.words.split_off(self.utterance);
        self.utterance = 0;
        Some(OpenLine { slot: self.slot, words, utterance: 0 })
    }

    /// Takes the words of the last utterance if none of them has a language
    /// yet, and words before them stay: a word in another language has just
    /// come, and they are its utterance's start, such as "Friday" opening a
    /// Mandarin sentence after an English line.
    fn take_untagged_utterance(&mut self) -> Vec<Word> {
        if self.utterance == 0 || self.words[self.utterance..].iter().any(|w| w.lang.is_some()) {
            return Vec::new();
        }
        let words = self.words.split_off(self.utterance);
        self.utterance = 0;
        words
    }
}

/// Where to end a line of `words` when the next word, from `start` to `end`,
/// would make it too long: the index of the first word left for the next
/// line, at least 1. Of the places between words that leave the next line
/// starting within `window` of `end`, the place before the next word
/// included: the last after a sentence end, else the longest pause, the
/// latest of equals.
fn cap_at(words: &[&Word], start: Ts, end: Ts, window: Ts) -> usize {
    let from = end.saturating_sub(window);
    let places: Vec<(usize, Ts, bool)> = (1..=words.len())
        .filter_map(|i| {
            let next = words.get(i).map_or(start, |w| w.start);
            let before = words[i - 1];
            (next >= from).then(|| (i, next.saturating_sub(before.end), ends_sentence(&before.text)))
        })
        .collect();
    let sentence = places.iter().rev().find(|p| p.2);
    let longest = places.iter().max_by_key(|p| (p.1, p.0));
    sentence.or(longest).map_or(words.len(), |p| p.0)
}

/// Whether a word ends a sentence: the transcriber's punctuation, in letters
/// or in CJK, perhaps before a closing quote or bracket.
fn ends_sentence(word: &str) -> bool {
    word.trim_end_matches(['"', '\'', '”', '’', ')', '」', '』'])
        .ends_with(['.', '?', '!', '…', '。', '？', '！'])
}

/// The first language reported for the words.
fn lang<'a>(words: impl IntoIterator<Item = &'a Word>) -> Option<&'a str> {
    words.into_iter().find_map(|w| w.lang.as_deref())
}

/// Whether runs of words in languages `a` and `b` can share a line: the same
/// language, or none next to one written in letters. The transcriber names
/// Chinese, Japanese and Korean from their script, so words it can't name
/// are in letters.
fn fits(a: Option<&str>, b: Option<&str>) -> bool {
    match (a, b) {
        (Some(x), None) | (None, Some(x)) => !pam_speech::is_cjk_language(x),
        _ => a == b,
    }
}

#[derive(Default)]
pub struct Pipeline {
    /// Final words waiting for the tracker to settle past their end.
    pending: VecDeque<Word>,
    preview_words: Vec<Word>,
    /// Final spans still close enough to matter.
    spans: Vec<SpeakerSpan>,
    preview_spans: Vec<SpeakerSpan>,
    settled_until: Ts,
    /// How much audio the transcriber has been given.
    heard: Ts,
    /// The previous word's slot: `None` before the first word, `Some(None)` for unknown.
    prev: Option<Option<u32>>,
    open: Option<OpenLine>,
    lines: u32,
    slots: SlotMap,
    /// How loud the input is now, when it is audio.
    level: Option<f32>,
    /// Speakers shown as speaking, with their level.
    active: BTreeMap<SpkId, f32>,
    shown: Vec<PreviewLine>,
}

impl Pipeline {
    pub fn new() -> Pipeline {
        Pipeline::default()
    }

    pub fn slots(&self) -> &SlotMap {
        &self.slots
    }

    /// Final lines emitted so far.
    pub fn line_count(&self) -> u32 {
        self.lines
    }

    /// A transcriber update, after `heard` samples were pushed to it.
    pub fn words(&mut self, update: Update<Word>, heard: Ts) -> Vec<CoreEvent> {
        self.pending.extend(update.finalized);
        self.preview_words = update.preview;
        self.heard = self.heard.max(heard);
        let mut out = Vec::new();
        self.step(&mut out);
        out
    }

    /// A tracker update, after `now` samples were pushed to it.
    pub fn speakers(&mut self, update: SpeakerUpdate, now: Ts) -> Vec<CoreEvent> {
        let mut out = Vec::new();
        self.add_spans(update.spans.finalized, &mut out);
        self.preview_spans = update.spans.preview;
        self.settled_until = self.settled_until.max(update.settled_until);
        self.activity(now, &mut out);
        self.step(&mut out);
        out
    }

    /// How loud the input was in the chunk ending at `now`, from 0 to 1.
    pub fn level(&mut self, now: Ts, level: f32) -> Vec<CoreEvent> {
        self.level = Some(level);
        let mut out = Vec::new();
        self.activity(now, &mut out);
        out
    }

    /// End of stream: what both backends flushed. Every word gets a line.
    pub fn finish(&mut self, words: Vec<Word>, spans: Vec<SpeakerSpan>) -> Vec<CoreEvent> {
        let mut out = Vec::new();
        self.pending.extend(words);
        self.preview_words.clear();
        self.add_spans(spans, &mut out);
        self.preview_spans.clear();
        self.settled_until = Ts::MAX;
        self.step(&mut out);
        self.close(&mut out);
        self.update_preview(&mut out);
        for spk in std::mem::take(&mut self.active).into_keys() {
            out.push(CoreEvent::SpeakerActive { spk, level: 0.0 });
        }
        out
    }

    fn add_spans(&mut self, spans: Vec<SpeakerSpan>, out: &mut Vec<CoreEvent>) {
        for s in spans {
            if let Some(spk) = self.slots.add(s.slot) {
                out.push(CoreEvent::SpeakerAdded(spk));
            }
            self.spans.push(s);
        }
    }

    fn step(&mut self, out: &mut Vec<CoreEvent>) {
        while let Some(w) = self.pending.front() {
            if self.settled_until < w.end {
                break;
            }
            let w = self.pending.pop_front().unwrap();
            let slot = self.assign(&w);
            self.place(w, slot, out);
        }
        self.close_on_pause(out);
        self.prune();
        self.update_preview(out);
    }

    /// The spec's merge rule, for a word the tracker has settled past.
    fn assign(&mut self, w: &Word) -> Option<u32> {
        let prev = self.prev.flatten();
        let w = &Word { start: w.start.saturating_sub(LEAD), end: w.end, ..Word::default() };
        let slot = best_overlap(&self.spans, w, prev).or_else(|| nearest(&self.spans, w, prev)).or(prev);
        self.prev = Some(slot);
        slot
    }

    fn place(&mut self, w: Word, slot: Option<u32>, out: &mut Vec<CoreEvent>) {
        if w.starts_utterance {
            self.end_utterance(out);
        }
        let mut carried = Vec::new();
        if let Some(open) = &mut self.open {
            let other_lang = open.lang().zip(w.lang.as_deref()).is_some_and(|(a, b)| a != b);
            if other_lang && open.slot == slot && !w.starts_utterance {
                carried = open.take_untagged_utterance();
            }
            if other_lang || open.slot != slot || w.start > open.last_end() + open.pause() {
                self.close(out);
            } else if w.end.saturating_sub(open.t0()) > MAX_LINE {
                let words: Vec<&Word> = open.words.iter().collect();
                let rest = open.split_before(cap_at(&words, w.start, w.end, CAP_WINDOW));
                self.close(out);
                self.open = (!rest.words.is_empty()).then_some(rest);
            }
        }
        let open = self.open.get_or_insert(OpenLine { slot, words: carried, utterance: 0 });
        if w.starts_utterance {
            open.utterance = open.words.len();
        }
        open.words.push(w);
    }

    /// The open line's last utterance is complete: it gets a line of its own
    /// if its language doesn't fit the line's (see [`OpenLine::split`]).
    fn end_utterance(&mut self, out: &mut Vec<CoreEvent>) {
        if let Some(rest) = self.open.as_mut().and_then(OpenLine::split) {
            self.emit(out);
            self.open = Some(rest);
        }
    }

    /// Closes the open line once the pause after it is certain: the tracker has
    /// settled past it and no word that could continue the line can still arrive.
    /// A word still in the preview may, unless the tracker already puts it on
    /// someone else: NeMo-Speech makes an utterance's words final only when it
    /// ends, which for a long one is well after [`WORD_GRACE`].
    fn close_on_pause(&mut self, out: &mut Vec<CoreEvent>) {
        let Some(open) = &self.open else { return };
        let gap_ends = open.last_end() + open.pause();
        let heard = self.preview_words.first().filter(|w| {
            let spans = self.spans.iter().chain(&self.preview_spans);
            best_overlap(spans, w, open.slot).is_none_or(|slot| Some(slot) == open.slot)
        });
        let continues = self.pending.front().or(heard).is_some_and(|w| w.start <= gap_ends);
        let known = self.settled_until.min(self.heard.saturating_sub(WORD_GRACE));
        if !continues && known > gap_ends {
            self.close(out);
        }
    }

    fn close(&mut self, out: &mut Vec<CoreEvent>) {
        let rest = self.open.as_mut().and_then(OpenLine::split);
        self.emit(out);
        self.open = rest;
        self.emit(out);
    }

    fn emit(&mut self, out: &mut Vec<CoreEvent>) {
        let Some(open) = self.open.take() else { return };
        self.lines += 1;
        out.push(CoreEvent::LineFinal(Line {
            id: UtteranceId(self.lines),
            t0: open.t0(),
            t1: open.last_end(),
            slot: open.slot,
            spk: open.slot.and_then(|s| self.slots.get(s)),
            lang: open.lang().map(String::from),
            text: join_words(&open.words),
        }));
    }

    /// Drops final spans too old to touch any word still to come.
    fn prune(&mut self) {
        let Some(from) = self.pending.front().map(|w| w.start).or(self.open.as_ref().map(|o| o.last_end()))
        else {
            return;
        };
        let keep_after = from.saturating_sub(NEAREST + 10 * SECOND);
        self.spans.retain(|s| s.end >= keep_after);
    }

    /// The final words not in a final line yet: the open line's, then those
    /// waiting for the tracker, with the speaker its spans give so far. They
    /// are split into lines by the rules a final line follows (a change of
    /// speaker, a pause, the maximum length, a change of language), and each
    /// line into utterances.
    pub fn speech(&self) -> Vec<Speech> {
        let mut runs: Vec<(Option<u32>, Vec<&Word>)> = Vec::new();
        let mut last = self.prev.flatten();
        if let Some(open) = &self.open {
            runs.push((open.slot, open.words.iter().collect()));
            last = open.slot;
        }
        for w in &self.pending {
            let slot = best_overlap(self.spans.iter().chain(&self.preview_spans), w, last).or(last);
            let mut carried = Vec::new();
            let joins = runs.last_mut().is_some_and(|(s, words)| {
                let other_lang =
                    lang(words.iter().copied()).zip(w.lang.as_deref()).is_some_and(|(a, b)| a != b);
                if other_lang && *s == slot && !w.starts_utterance {
                    // As [`OpenLine::take_untagged_utterance`].
                    let from = words.iter().rposition(|w| w.starts_utterance).unwrap_or(0);
                    if from > 0 && words[from..].iter().all(|w| w.lang.is_none()) {
                        carried = words.split_off(from);
                    }
                }
                let (first, prev) = (words[0], words[words.len() - 1]);
                let pause = match ends_sentence(&prev.text) {
                    true => PAUSE,
                    false => MID_SENTENCE_PAUSE,
                };
                let continues = !other_lang && *s == slot && w.start <= prev.end + pause;
                if continues && w.end.saturating_sub(first.start) > MAX_LINE {
                    // As [`Pipeline::place`] ends a line too long.
                    carried = words.split_off(cap_at(words, w.start, w.end, CAP_WINDOW));
                    return false;
                }
                continues
            });
            match runs.last_mut() {
                Some((_, words)) if joins => words.push(w),
                _ => {
                    carried.push(w);
                    runs.push((slot, carried));
                }
            }
            last = slot;
        }
        let mut speech = Vec::new();
        for (slot, words) in runs {
            let mut utterances: Vec<Vec<&Word>> = Vec::new();
            for w in words {
                match utterances.last_mut() {
                    Some(u) if !w.starts_utterance => u.push(w),
                    _ => utterances.push(vec![w]),
                }
            }
            // An utterance whose language doesn't fit the line's before it is
            // a line of its own, as [`OpenLine::split`] makes it at its end.
            let mut lines: Vec<Vec<Vec<&Word>>> = Vec::new();
            for u in utterances {
                match lines.last_mut() {
                    Some(l) if fits(lang(l.iter().flatten().copied()), lang(u.iter().copied())) => l.push(u),
                    _ => lines.push(vec![u]),
                }
            }
            speech.extend(lines.into_iter().map(|l| {
                Speech {
                    spk: slot.and_then(|s| self.slots.get(s)),
                    utterances: l
                        .into_iter()
                        .map(|u| Utterance {
                            t0: u[0].start,
                            t1: u[u.len() - 1].end,
                            text: join_words(u.iter().copied()),
                        })
                        .collect(),
                }
            }));
        }
        speech
    }

    /// The open line, then words still waiting for a speaker, then preview
    /// words. Words not yet merged take the slot of an overlapping span
    /// (final or preview), or the speaker of the word before.
    fn update_preview(&mut self, out: &mut Vec<CoreEvent>) {
        let mut runs: Vec<(Option<u32>, Vec<&Word>)> = Vec::new();
        let mut last = self.prev.flatten();
        if let Some(open) = &self.open {
            runs.push((open.slot, open.words.iter().collect()));
            last = open.slot;
        }
        let waiting = self.pending.iter().chain(self.preview_words.iter());
        for w in waiting {
            let slot = best_overlap(self.spans.iter().chain(&self.preview_spans), w, last).or(last);
            match runs.last_mut() {
                Some((s, words)) if *s == slot => words.push(w),
                _ => runs.push((slot, vec![w])),
            }
            last = slot;
        }
        let lines: Vec<PreviewLine> = runs
            .into_iter()
            .map(|(slot, words)| PreviewLine {
                spk: slot.and_then(|s| self.slots.get(s)),
                text: join_words(words.iter().copied()),
                t0: words[0].start,
            })
            .collect();
        if lines != self.shown {
            self.shown = lines.clone();
            out.push(CoreEvent::LinePreview(lines));
        }
    }

    /// Who is speaking, and how loud, at `now`. The tracker's newest spans
    /// say who; the input level says how loud, and whether anyone is speaking
    /// at all, since the tracker labels audio seconds late. Without a level
    /// (the fake backends' silence), someone speaks at full level while their
    /// span reaches now.
    fn activity(&mut self, now: Ts, out: &mut Vec<CoreEvent>) {
        let spans = || self.preview_spans.iter().chain(&self.spans);
        let newest = spans().map(|s| s.end).max().unwrap_or(0);
        let level = match self.level {
            Some(level) if newest + LABEL_LAG >= now => level,
            None if newest + ACTIVE_WINDOW >= now => 1.0,
            _ => 0.0,
        };
        let active: BTreeMap<SpkId, f32> = match level > 0.0 {
            true => spans()
                .filter(|s| s.end + ACTIVE_WINDOW >= newest)
                .filter_map(|s| self.slots.get(s.slot))
                .map(|spk| (spk, level))
                .collect(),
            false => BTreeMap::new(),
        };
        for (&spk, &level) in &active {
            if self.active.get(&spk) != Some(&level) {
                out.push(CoreEvent::SpeakerActive { spk, level });
            }
        }
        for &spk in self.active.keys().filter(|spk| !active.contains_key(spk)) {
            out.push(CoreEvent::SpeakerActive { spk, level: 0.0 });
        }
        self.active = active;
    }
}

/// Picks the highest-scoring slot.
/// Ties go to the previous word's slot, then to the lowest slot.
fn pick(scores: BTreeMap<u32, i64>, prev: Option<u32>) -> Option<u32> {
    let best = *scores.values().max()?;
    let tied: Vec<u32> = scores.iter().filter(|(_, v)| **v == best).map(|(s, _)| *s).collect();
    Some(prev.filter(|p| tied.contains(p)).unwrap_or(tied[0]))
}

/// The slot with the greatest total time overlap with `w`, if any overlaps.
fn best_overlap<'a>(
    spans: impl IntoIterator<Item = &'a SpeakerSpan>,
    w: &Word,
    prev: Option<u32>,
) -> Option<u32> {
    let mut scores = BTreeMap::new();
    for s in spans {
        let overlap = w.end.min(s.end).saturating_sub(w.start.max(s.start));
        if overlap > 0 {
            *scores.entry(s.slot).or_insert(0) += overlap as i64;
        }
    }
    pick(scores, prev)
}

/// The slot of the nearest span within [`NEAREST`] of `w`.
fn nearest(spans: &[SpeakerSpan], w: &Word, prev: Option<u32>) -> Option<u32> {
    let mut scores = BTreeMap::new();
    for s in spans {
        let gap = w.start.saturating_sub(s.end).max(s.start.saturating_sub(w.end));
        if gap <= NEAREST {
            let score = scores.entry(s.slot).or_insert(i64::MIN);
            *score = (*score).max(-(gap as i64));
        }
    }
    pick(scores, prev)
}

#[cfg(test)]
mod tests {
    use pam_speech::fake::{FakeTracker, FakeTranscriber, Script, secs};
    use pam_speech::{SpeakerTracker, Transcriber};

    use super::*;

    const STANDUP: &str = include_str!("../../../fixtures/standup.fake.jsonl");

    /// Every event, with how much audio had been pushed when it came out.
    fn replay(script: &str, chunk_secs: f64) -> Vec<(Ts, CoreEvent)> {
        let script = Script::parse(script).unwrap();
        let duration = script.duration;
        let mut t = FakeTranscriber::new(script.clone());
        let mut k = FakeTracker::new(script);
        let mut p = Pipeline::new();
        let mut out = Vec::new();
        let mut now = 0;
        while now < duration {
            let chunk = vec![0.0; secs(chunk_secs).min(duration - now) as usize];
            now += chunk.len() as Ts;
            let words = p.words(t.push(&chunk).unwrap(), now);
            let spans = p.speakers(k.push(&chunk).unwrap(), now);
            out.extend(words.into_iter().chain(spans).map(|e| (now, e)));
        }
        let words = t.finish().unwrap();
        let spans = k.finish().unwrap();
        out.extend(p.finish(words, spans).into_iter().map(|e| (now, e)));
        out
    }

    fn lines(events: &[(Ts, CoreEvent)]) -> Vec<Line> {
        events
            .iter()
            .filter_map(|(_, e)| match e {
                CoreEvent::LineFinal(l) => Some(l.clone()),
                _ => None,
            })
            .collect()
    }

    /// `(slot, text)` of every final line.
    fn said(script: &str, chunk_secs: f64) -> Vec<(Option<u32>, String)> {
        lines(&replay(script, chunk_secs)).into_iter().map(|l| (l.slot, l.text)).collect()
    }

    fn s(slot: u32, text: &str) -> (Option<u32>, String) {
        (Some(slot), text.to_string())
    }

    #[test]
    fn standup_fixture_gives_one_line_per_turn() {
        let got = lines(&replay(STANDUP, 0.1));
        let want = [
            (0, "Okay, let's get started. Quick intros for the bot. I'm Alice, I run product."),
            (1, "Marcus, engineering."),
            (2, "And I'm Jordan, design."),
            (0, "Today is about the onboarding redesign."),
            (0, "We're deciding between the three-step flow and a single page."),
            (1, "Single page is faster to build, but I worry about drop-off on mobile."),
            (2, "Mobile is where most sign-ups start, so that matters."),
            (1, "Right, exactly."),
            (0, "Then let's prototype both and look at the numbers next week."),
            (2, "Sounds good to me."),
        ];
        assert_eq!(got.iter().map(|l| (l.slot.unwrap(), l.text.as_str())).collect::<Vec<_>>(), want);
        assert_eq!(got.iter().map(|l| l.id.to_string()).next_back().unwrap(), "u0010");
        assert_eq!(got[1].spk, Some(SpkId(1)));
        assert_eq!((got[0].t0, got[0].t1), (secs(1.12), secs(5.04) + 1_280));
    }

    #[test]
    fn late_settling_and_chunk_size_do_not_change_the_lines() {
        let prompt = STANDUP.replace("\"settle_delay\": 2.88", "\"settle_delay\": 0");
        let base = said(&prompt, 0.1);
        assert_eq!(said(STANDUP, 0.1), base);
        assert_eq!(said(STANDUP, 0.016), base);
        assert_eq!(said(STANDUP, 1.0), base);
    }

    #[test]
    fn lines_wait_for_the_tracker_to_settle() {
        let events = replay(STANDUP, 0.1);
        let (at, first) = events
            .iter()
            .find_map(|(now, e)| match e {
                CoreEvent::LineFinal(l) => Some((*now, l.clone())),
                _ => None,
            })
            .unwrap();
        // Alice's span settles at 5.36 + 2.88 s, but Marcus's span is open from 5.84 s,
        // so settled_until stays there and the pause after her line is not certain.
        // Her line closes when Marcus's first word is merged: his span ends at
        // 7.36 s and settles 2.88 s later.
        assert!(at >= secs(7.36 + 2.88), "first line final at {at}");
        assert!(at < secs(7.36 + 2.88 + 0.2), "first line final at {at}");
        assert_eq!(first.slot, Some(0));
        // Before that, her words were already showing as a preview.
        let early = events.iter().find_map(|(now, e)| match e {
            CoreEvent::LinePreview(p) if p.iter().any(|l| l.text.contains("Alice")) => Some(*now),
            _ => None,
        });
        assert!(early.unwrap() < secs(4.3));
    }

    #[test]
    fn a_word_without_overlap_takes_the_nearest_span_within_a_second() {
        let got = said(
            r#"{"span": 0, "start": 0, "end": 1}
               {"word": "a", "at": 0.5}
               {"word": "b", "at": 1.5}"#,
            0.1,
        );
        // b starts 0.5 s after slot 0's span ends.
        assert_eq!(got, [s(0, "a b")]);
    }

    #[test]
    fn nearest_span_only_sees_spans_already_final() {
        // "c" is stamped 0.3 s before its speaker's span starts, as an emission stamp
        // can be. It is merged as soon as the tracker settles past its end; whether
        // slot 1's span is final by then depends on how far the tracker had got.
        let script = r#"{"config": {"settle_delay": 0}}
               {"span": 0, "start": 0, "end": 1}
               {"word": "a.", "at": 0.5}
               {"span": 1, "start": 2.5, "end": 3.5}
               {"word": "c", "at": 2.12}
               {"word": "d", "at": 3.0}"#;
        // Small steps: only slot 0's span is known, 1.12 s away: previous word's speaker.
        // (The 1.54 s pause after the sentence "a." still ends the line.)
        assert_eq!(said(script, 0.1), [s(0, "a."), s(0, "c"), s(1, "d")]);
        // One step past the whole span: slot 1 is known and 0.3 s away.
        assert_eq!(said(script, 4.0), [s(0, "a."), s(1, "c d")]);
    }

    #[test]
    fn a_word_stamped_just_after_its_speakers_span_stays_with_them() {
        // "done" is stamped 50 ms after slot 0's span ends, as the next speaker
        // starts. On the stamp alone it only touches slot 1; looking from LEAD
        // before it, it overlaps slot 0 more.
        let got = said(
            r#"{"span": 0, "start": 0, "end": 2.0}
               {"word": "nearly", "at": 1.6}
               {"word": "done", "at": 2.05}
               {"span": 1, "start": 2.1, "end": 4.0}
               {"word": "great", "at": 2.8}"#,
            0.1,
        );
        assert_eq!(got, [s(0, "nearly done"), s(1, "great")]);
    }

    #[test]
    fn a_word_far_from_any_span_takes_the_previous_speaker() {
        let got = said(
            r#"{"span": 0, "start": 0, "end": 1}
               {"word": "a.", "at": 0.5}
               {"word": "b", "at": 2.5}
               {"word": "c", "at": 2.9}"#,
            0.1,
        );
        // b is 1.5 s past the span: previous word's speaker. The 2 s pause after
        // a sentence ends the line.
        assert_eq!(got, [s(0, "a."), s(0, "b c")]);
    }

    #[test]
    fn a_word_with_no_speaker_at_all_is_unknown() {
        let got = said(
            r#"{"word": "hello", "at": 0.5}
               {"word": "there", "at": 0.8}
               {"span": 0, "start": 3, "end": 4}
               {"word": "hi", "at": 3.5}"#,
            0.1,
        );
        assert_eq!(got, [(None, "hello there".into()), s(0, "hi")]);
    }

    #[test]
    fn cross_talk_goes_to_the_greater_overlap() {
        let got = said(
            r#"{"span": 0, "start": 0, "end": 2}
               {"span": 1, "start": 1.5, "end": 3}
               {"word": "one", "start": 0.5, "end": 1.0}
               {"word": "two", "start": 1.6, "end": 1.9}
               {"word": "three", "start": 1.8, "end": 2.4}"#,
            0.1,
        );
        // "two" lies inside both spans (a tie) and stays with slot 0; "three"
        // overlaps slot 1 for 0.4 s against 0.2 s.
        assert_eq!(got, [s(0, "one two"), s(1, "three")]);
    }

    #[test]
    fn an_emission_stamp_inside_cross_talk_stays_with_the_previous_speaker() {
        let script = r#"{"span": 0, "start": 0, "end": 2}
               {"span": 1, "start": 1.2, "end": 3}
               {"word": "mine", "at": 0.6}
               {"word": "yours", "at": 1.6}
               {"word": "ours", "at": 2.4}"#;
        // "yours" (really slot 1's) lies inside both spans: a tie, so it stays with
        // slot 0. This is how 80 ms stamps behave under the greatest-overlap rule.
        assert_eq!(said(script, 0.1), [s(0, "mine yours"), s(1, "ours")]);
        // With no previous word, a tie goes to the lowest slot.
        let alone = r#"{"span": 1, "start": 0, "end": 2}
               {"span": 0, "start": 0, "end": 2}
               {"word": "both", "at": 1}"#;
        assert_eq!(said(alone, 0.1), [s(0, "both")]);
    }

    #[test]
    fn after_a_sentence_a_pause_over_a_second_ends_the_line() {
        let got = said(
            r#"{"span": 0, "start": 0, "end": 6}
               {"word": "a", "at": 0.5}
               {"word": "b.", "at": 1.4}
               {"word": "c?", "at": 2.6}
               {"word": "d", "at": 4.0}"#,
            0.1,
        );
        // a→b: 0.82 s gap. b→c: 1.12 s. c→d: 1.32 s.
        assert_eq!(got, [s(0, "a b."), s(0, "c?"), s(0, "d")]);
    }

    #[test]
    fn mid_sentence_it_takes_a_pause_over_two_and_a_half_seconds() {
        let got = said(
            r#"{"span": 0, "start": 0, "end": 12}
               {"word": "remind", "at": 0.5}
               {"word": "me", "at": 0.8}
               {"word": "to", "at": 1.1}
               {"word": "uh", "at": 3.3}
               {"word": "buy", "at": 3.6}
               {"word": "milk", "at": 3.9}
               {"word": "next", "at": 6.6}
               {"word": "week", "at": 6.9}"#,
            0.1,
        );
        // to→uh: 2.12 s gap. milk→next: 2.62 s.
        assert_eq!(got, [s(0, "remind me to uh buy milk"), s(0, "next week")]);
    }

    #[test]
    fn a_word_ends_a_sentence_before_a_closing_quote_and_in_cjk() {
        assert!(ends_sentence("done."));
        assert!(ends_sentence("really?\""));
        assert!(ends_sentence("好。"));
        assert!(!ends_sentence("Pam,"));
        assert!(!ends_sentence("U.S"));
    }

    #[test]
    fn a_line_closes_on_a_pause_before_the_next_word_arrives() {
        let script = r#"{"config": {"word_delay": 0.9, "settle_delay": 2.88, "duration": 20}}
               {"span": 0, "start": 0, "end": 1}
               {"word": "a.", "at": 0.5}
               {"span": 0, "start": 15, "end": 16}
               {"word": "b", "at": 15.5}"#;
        let closed = |script: &str| {
            let events = replay(script, 0.1);
            events.iter().find(|(_, e)| matches!(e, CoreEvent::LineFinal(_))).unwrap().0
        };
        // Settled past 0.58 + 1 s pause, and the transcriber has heard 2 s past that.
        let at = closed(script);
        assert!(at > secs(1.58 + 2.0) && at < secs(5.0), "closed at {at}");
        // Mid-sentence, the pause is 2.5 s.
        let at = closed(&script.replace("\"a.\"", "\"a\""));
        assert!(at > secs(3.08 + 2.0) && at < secs(6.5), "closed at {at}");
    }

    #[test]
    fn a_short_pause_stays_in_the_line_while_the_words_after_it_are_still_heard() {
        // Finals come 5 s after each word, as NeMo-Speech gives a long
        // utterance's words only when it ends; until then they are the preview.
        let script = r#"{"config": {"word_delay": 5.0, "settle_delay": 2.88, "duration": 30}}
               {"span": 0, "start": 0, "end": 1}
               {"span": 0, "start": 1.7, "end": 2.3}
               {"word": "as", "at": 0.5}
               {"word": "the", "at": 0.8}
               {"word": "previous", "at": 1.8}
               {"word": "one", "at": 2.1}"#;
        assert_eq!(said(script, 0.1), [s(0, "as the previous one")]);
        // After a pause longer than the line's, it still ends.
        let later = script
            .replace("1.8}", "4.0}")
            .replace("2.1}", "4.3}")
            .replace("1.7, \"end\": 2.3", "3.9, \"end\": 4.5");
        assert_eq!(said(&later, 0.1), [s(0, "as the"), s(0, "previous one")]);
    }

    #[test]
    fn a_line_does_not_wait_for_someone_else_still_being_heard() {
        // Slot 1 answers 1 s after slot 0; its words are final 5 s later.
        let script = r#"{"config": {"word_delay": 5.0, "settle_delay": 0.5, "duration": 30}}
               {"span": 0, "start": 0, "end": 1}
               {"span": 1, "start": 1.9, "end": 3.0}
               {"word": "a", "at": 0.5}
               {"word": "b", "at": 0.8}
               {"word": "c", "at": 2.0}
               {"word": "d", "at": 2.5}"#;
        let events = replay(script, 0.1);
        let (at, first) = events
            .iter()
            .find_map(|(now, e)| match e {
                CoreEvent::LineFinal(l) => Some((*now, l.clone())),
                _ => None,
            })
            .unwrap();
        assert_eq!(first.text, "a b");
        // Closed after its 2.5 s pause and the 2 s grace, not when "c" became final (7.08 s).
        assert!(at < secs(6.0), "closed at {at}");
    }

    #[test]
    fn speech_on_its_way_is_split_like_lines_and_into_utterances() {
        // Utterances are final 0.8 s after they end; speakers settle 2.88 s late.
        let script = Script::parse(
            r#"{"config": {"utterance_end": 0.8, "settle_delay": 2.88, "duration": 20}}
               {"span": 0, "start": 0, "end": 1.5}
               {"word": "nice.", "at": 0.5}
               {"span": 1, "start": 2, "end": 8}
               {"word": "hey", "at": 2.2}
               {"word": "Pam", "at": 2.5}
               {"word": "can", "at": 2.8}
               {"word": "you", "at": 3.1}
               {"word": "uh", "at": 4.4}
               {"word": "find", "at": 4.7}
               {"word": "it", "at": 7.6}"#,
        )
        .unwrap();
        let (mut t, mut k, mut p) =
            (FakeTranscriber::new(script.clone()), FakeTracker::new(script), Pipeline::new());
        let mut pushed = 0;
        let mut at = |p: &mut Pipeline, until: f64| {
            let chunk = vec![0.0; (secs(until) - pushed) as usize];
            pushed = secs(until);
            p.words(t.push(&chunk).unwrap(), pushed);
            p.speakers(k.push(&chunk).unwrap(), pushed);
            p.speech()
        };
        let u = |t0: f64, t1: f64, text: &str| Utterance {
            t0: secs(t0),
            t1: secs(t1) + pam_speech::fake::FRAME,
            text: text.into(),
        };
        assert_eq!(at(&mut p, 1.0), []);
        // "nice." is final but its speaker's span isn't: it is still on its way.
        assert_eq!(at(&mut p, 2.0), [Speech { spk: None, utterances: vec![u(0.5, 0.5, "nice.")] }]);
        // "nice." is a final line by now. Two utterances of slot 1, 1.2 s
        // apart, are one line on its way; slot 1 has no speaker ID yet.
        assert_eq!(
            at(&mut p, 5.6),
            [Speech { spk: None, utterances: vec![u(2.2, 3.1, "hey Pam can you"), u(4.4, 4.7, "uh find")] }]
        );
        // "it" comes 2.82 s after "find": a pause that ends the line.
        let got = at(&mut p, 8.5);
        let texts: Vec<Vec<&str>> =
            got.iter().map(|s| s.utterances.iter().map(|u| u.text.as_str()).collect()).collect();
        assert_eq!(texts, [vec!["hey Pam can you", "uh find"], vec!["it"]]);
    }

    #[test]
    fn speech_on_its_way_is_split_at_a_change_of_language_like_lines() {
        // The scripts of the two language tests below, one speaker in two
        // spans: the first settles while the rest is still on its way, so
        // some words are in the open line and some wait for the tracker.
        for words in [
            r#"{"word": "OK", "at": 0.5, "lang": "en-US"}
               {"word": "那就", "at": 1.0, "lang": "zh-CN"}
               {"word": "周五吧", "at": 1.5, "lang": "zh-CN"}
               {"word": "Friday", "at": 2.0}
               {"word": "then", "at": 2.5, "lang": "en-US"}
               {"word": "sure", "at": 4.0, "lang": "en-US"}
               {"word": "我们", "at": 4.5, "lang": "zh-CN"}"#,
            r#"{"word": "OK", "at": 0.5, "starts_utterance": true}
               {"word": "我们", "at": 1.0, "lang": "zh-CN"}
               {"word": "Friday", "at": 1.5}
               {"word": "见", "at": 2.0, "lang": "zh-CN"}
               {"word": "sounds", "at": 2.9, "starts_utterance": true}
               {"word": "good", "at": 3.4}
               {"word": "那就", "at": 4.3, "lang": "zh-CN", "starts_utterance": true}
               {"word": "这样", "at": 4.8, "lang": "zh-CN"}
               {"word": "great", "at": 5.7, "lang": "en-US", "starts_utterance": true}
               {"word": "thanks", "at": 6.2, "starts_utterance": true}
               {"word": "Friday", "at": 7.1, "starts_utterance": true}
               {"word": "我们见", "at": 7.6, "lang": "zh-CN"}"#,
        ] {
            let script = format!(
                r#"{{"config": {{"settle_delay": 2.88, "duration": 14}}}}
                   {{"span": 0, "start": 0, "end": 3.7}}
                   {{"span": 0, "start": 3.7, "end": 8}}
                   {words}"#
            );
            let span = |l: Line| (l.t0, l.t1);
            let want: Vec<(Ts, Ts)> = lines(&replay(&script, 0.1)).into_iter().map(span).collect();
            let script = Script::parse(&script).unwrap();
            let (mut t, mut k, mut p) =
                (FakeTranscriber::new(script.clone()), FakeTracker::new(script), Pipeline::new());
            let (mut now, mut done) = (0, Vec::new());
            while now < secs(14.0) {
                let chunk = vec![0.0; secs(0.1) as usize];
                now += chunk.len() as Ts;
                let mut events: Vec<(Ts, CoreEvent)> =
                    p.words(t.push(&chunk).unwrap(), now).into_iter().map(|e| (now, e)).collect();
                events.extend(p.speakers(k.push(&chunk).unwrap(), now).into_iter().map(|e| (now, e)));
                done.extend(lines(&events).into_iter().map(span));
                if now < secs(8.0) {
                    continue; // words still to become final
                }
                // The lines so far and the speech on its way are the lines to come.
                let on_its_way =
                    p.speech().into_iter().map(|s| (s.utterances[0].t0, s.utterances.last().unwrap().t1));
                let got: Vec<(Ts, Ts)> = done.iter().copied().chain(on_its_way).collect();
                assert_eq!(got, want, "at {now}");
            }
        }
    }

    #[test]
    fn a_long_monologue_is_split_at_the_maximum_length() {
        let mut script = String::from(r#"{"span": 0, "start": 0, "end": 31}"#);
        for i in 0..60 {
            script += &format!("\n{{\"word\": \"w{i}\", \"at\": {}}}", i as f64 * 0.5);
        }
        let got = lines(&replay(&script, 0.1));
        assert_eq!(got.len(), 2);
        assert!(got.iter().all(|l| l.t1 - l.t0 <= MAX_LINE));
        assert!(got[1].text.starts_with("w40"));
    }

    /// A 30 s monologue, a word every 0.5 s, with a 1.32 s pause before word
    /// `pause_before`, and word `sentence` ending a sentence.
    fn monologue(pause_before: usize, sentence: Option<usize>) -> Vec<String> {
        let mut script = String::from(r#"{"span": 0, "start": 0, "end": 32}"#);
        for i in 0..60 {
            let at = i as f64 * 0.5 + if i >= pause_before { 0.9 } else { 0.0 };
            let dot = if Some(i) == sentence { "." } else { "" };
            script += &format!("\n{{\"word\": \"w{i}{dot}\", \"at\": {at}}}");
        }
        let got = lines(&replay(&script, 0.1));
        assert!(got.iter().all(|l| l.t1 - l.t0 <= MAX_LINE));
        got.into_iter().map(|l| l.text).collect()
    }

    #[test]
    fn at_the_maximum_length_a_line_ends_at_its_longest_pause_in_the_last_seconds() {
        // w39 (20.4 s) would cross 20 s; the pause before w35 (18.4 s) is in the window.
        let got = monologue(35, None);
        assert!(got[0].ends_with("w34") && got[1].starts_with("w35"), "{got:?}");
        // A pause further back than the window doesn't count: cut at the maximum.
        let got = monologue(20, None);
        assert!(got[0].ends_with("w38") && got[1].starts_with("w39"), "{got:?}");
    }

    #[test]
    fn speech_on_its_way_is_split_at_the_maximum_length_like_lines() {
        // The monologue with a pause before w35, its speaker not settled yet.
        let mut script = String::from(
            r#"{"config": {"settle_delay": 40, "duration": 32}}
               {"span": 0, "start": 0, "end": 32}"#,
        );
        for i in 0..60 {
            let at = i as f64 * 0.5 + if i >= 35 { 0.9 } else { 0.0 };
            script += &format!("\n{{\"word\": \"w{i}\", \"at\": {at}}}");
        }
        let script = Script::parse(&script).unwrap();
        let (mut t, mut k, mut p) =
            (FakeTranscriber::new(script.clone()), FakeTracker::new(script), Pipeline::new());
        let chunk = vec![0.0; secs(32.0) as usize];
        p.words(t.push(&chunk).unwrap(), secs(32.0));
        p.speakers(k.push(&chunk).unwrap(), secs(32.0));
        let got: Vec<String> = p
            .speech()
            .iter()
            .map(|s| s.utterances.iter().map(|u| u.text.as_str()).collect::<Vec<_>>().join(" "))
            .collect();
        assert!(got.len() > 1 && got[0].ends_with("w34") && got[1].starts_with("w35"), "{got:?}");
    }

    #[test]
    fn at_the_maximum_length_a_line_ends_after_its_last_sentence_in_the_last_seconds() {
        let got = monologue(35, Some(37));
        assert!(got[0].ends_with("w37.") && got[1].starts_with("w38"), "{got:?}");
    }

    #[test]
    fn a_change_of_language_ends_a_line_and_each_line_keeps_its_language() {
        let got = lines(&replay(
            r#"{"span": 0, "start": 0, "end": 4}
               {"word": "OK", "at": 0.5, "lang": "en-US"}
               {"word": "那就", "at": 1.0, "lang": "zh-CN"}
               {"word": "周五吧", "at": 1.5, "lang": "zh-CN"}
               {"word": "Friday", "at": 2.0}
               {"word": "then", "at": 2.5, "lang": "en-US"}"#,
            0.1,
        ));
        let got: Vec<(Option<&str>, &str)> =
            got.iter().map(|l| (l.lang.as_deref(), l.text.as_str())).collect();
        // A word with no language stays in the line it falls in.
        assert_eq!(
            got,
            [(Some("en-US"), "OK"), (Some("zh-CN"), "那就周五吧 Friday"), (Some("en-US"), "then")]
        );
    }

    #[test]
    fn an_utterance_whose_language_doesnt_fit_the_line_is_a_line_of_its_own() {
        // One speaker, under a second between turns: too short a pause to end a line.
        let got = lines(&replay(
            r#"{"span": 0, "start": 0, "end": 8}
               {"word": "OK", "at": 0.5, "starts_utterance": true}
               {"word": "我们", "at": 1.0, "lang": "zh-CN"}
               {"word": "Friday", "at": 1.5}
               {"word": "见", "at": 2.0, "lang": "zh-CN"}
               {"word": "sounds", "at": 2.9, "starts_utterance": true}
               {"word": "good", "at": 3.4}
               {"word": "那就", "at": 4.3, "lang": "zh-CN", "starts_utterance": true}
               {"word": "这样", "at": 4.8, "lang": "zh-CN"}
               {"word": "great", "at": 5.7, "lang": "en-US", "starts_utterance": true}
               {"word": "thanks", "at": 6.2, "starts_utterance": true}
               {"word": "Friday", "at": 7.1, "starts_utterance": true}
               {"word": "我们见", "at": 7.6, "lang": "zh-CN"}"#,
            0.1,
        ));
        let got: Vec<(Option<&str>, &str)> =
            got.iter().map(|l| (l.lang.as_deref(), l.text.as_str())).collect();
        // Words in letters inside a Mandarin utterance stay in its line, also
        // when they open it, and an utterance in letters with no language
        // joins a line in en-US.
        assert_eq!(
            got,
            [
                (Some("zh-CN"), "OK 我们 Friday 见"),
                (None, "sounds good"),
                (Some("zh-CN"), "那就这样"),
                (Some("en-US"), "great thanks"),
                (Some("zh-CN"), "Friday 我们见"),
            ]
        );
    }

    #[test]
    fn slots_get_ids_in_order_of_their_first_final_span() {
        let events = replay(
            r#"{"span": 3, "start": 0, "end": 1}
               {"span": 1, "start": 2, "end": 3}
               {"span": 3, "start": 4, "end": 5}
               {"word": "hi", "at": 4.5}"#,
            0.1,
        );
        let added: Vec<SpkId> = events
            .iter()
            .filter_map(|(_, e)| match e {
                CoreEvent::SpeakerAdded(k) => Some(*k),
                _ => None,
            })
            .collect();
        assert_eq!(added, [SpkId(0), SpkId(1)]);
        let line = &lines(&events)[0];
        assert_eq!((line.slot, line.spk), (Some(3), Some(SpkId(0))));
    }

    #[test]
    fn preview_labels_follow_preview_spans_and_clear_at_the_end() {
        let events = replay(STANDUP, 0.1);
        // At 9 s Alice's line is open (her span is final), and Marcus's span is still
        // a preview, so his slot has no ID yet.
        let at_9 = events
            .iter()
            .filter(|(now, _)| *now <= secs(9.0))
            .filter_map(|(_, e)| match e {
                CoreEvent::LinePreview(p) => Some(p.clone()),
                _ => None,
            })
            .next_back()
            .unwrap();
        let marcus = at_9.iter().find(|l| l.text.starts_with("Marcus")).unwrap();
        assert_eq!(marcus.spk, None);
        let alice = at_9.iter().find(|l| l.text.starts_with("Okay")).unwrap();
        assert_eq!(alice.spk, Some(SpkId(0)));
        let last = events.iter().rev().find_map(|(_, e)| match e {
            CoreEvent::LinePreview(p) => Some(p),
            _ => None,
        });
        assert_eq!(last.unwrap(), &Vec::<PreviewLine>::new());
    }

    #[test]
    fn speakers_are_active_while_their_spans_reach_now() {
        let events = replay(STANDUP, 0.1);
        let marcus: Vec<(Ts, f32)> = events
            .iter()
            .filter_map(|(now, e)| match e {
                CoreEvent::SpeakerActive { spk: SpkId(1), level } => Some((*now, *level)),
                _ => None,
            })
            .collect();
        // Marcus has an ID from 7.36 + 2.88 s; he next speaks from 20.48 to 25.52 s.
        assert_eq!(marcus[0].1, 1.0);
        assert!(marcus[0].0 >= secs(20.48) && marcus[0].0 < secs(20.6));
        assert_eq!(marcus[1].1, 0.0);
        assert!(marcus[1].0 > secs(25.52 + 0.3) && marcus[1].0 < secs(26.0));
    }

    #[test]
    fn with_an_input_level_the_newest_spans_say_who_and_the_level_how_loud() {
        let span = |slot, start: f64, end: f64| SpeakerSpan { start: secs(start), end: secs(end), slot };
        let mut p = Pipeline::new();
        // Who is lit, as a front-end following the events sees it.
        let mut lit = BTreeMap::new();
        let mut follow = |events: Vec<CoreEvent>| {
            for e in events {
                if let CoreEvent::SpeakerActive { spk, level } = e {
                    match level > 0.0 {
                        true => lit.insert(spk.0, level),
                        false => lit.remove(&spk.0),
                    };
                }
            }
            lit.clone().into_iter().collect::<Vec<_>>()
        };
        // The tracker labels 3 s behind now (settled 1.5 s behind that).
        let labelled = |p: &mut Pipeline, finalized, preview, now: f64| {
            let update =
                SpeakerUpdate { spans: Update { finalized, preview }, settled_until: secs(now - 4.5) };
            p.speakers(update, secs(now))
        };

        // Slot 0 is talking at the tracker's edge.
        follow(labelled(&mut p, vec![span(0, 0.0, 1.0)], vec![span(0, 1.0, 2.0)], 5.0));
        assert_eq!(follow(p.level(secs(5.1), 0.6)), [(0, 0.6)]);
        assert_eq!(follow(p.level(secs(5.2), 0.4)), [(0, 0.4)]);
        // Quiet: nobody is speaking, whatever the spans say.
        assert_eq!(follow(p.level(secs(5.3), 0.0)), []);
        // Slot 1 takes over at the edge; slot 0's span ended well before.
        follow(p.level(secs(7.0), 0.5));
        let (finalized, preview) = (vec![span(0, 1.0, 2.5), span(1, 2.6, 2.8)], vec![span(1, 2.8, 4.0)]);
        assert_eq!(follow(labelled(&mut p, finalized, preview, 7.0)), [(1, 0.5)]);
        // Cross-talk at the edge lights both.
        let (finalized, preview) = (vec![span(1, 2.8, 3.0)], vec![span(0, 3.5, 4.5), span(1, 3.0, 4.6)]);
        assert_eq!(follow(labelled(&mut p, finalized, preview, 7.6)), [(0, 0.5), (1, 0.5)]);
        // A voice only in the preview has no speaker ID yet, so it lights nobody.
        let (finalized, preview) = (vec![], vec![span(2, 4.6, 5.0)]);
        assert_eq!(follow(labelled(&mut p, finalized, preview, 8.0)), []);
        let (finalized, preview) = (vec![], vec![span(0, 4.6, 5.0)]);
        assert_eq!(follow(labelled(&mut p, finalized, preview, 8.0)), [(0, 0.5)]);
        // Over 6 s after the newest span, the voice is someone the tracker hasn't placed yet.
        assert_eq!(follow(p.level(secs(11.0), 0.5)), [(0, 0.5)]);
        assert_eq!(follow(p.level(secs(11.1), 0.5)), []);
    }
}
