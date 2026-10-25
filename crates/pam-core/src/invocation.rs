//! Spoken requests: the wake name in what someone says, an intent check by a
//! model, and "Pam, cancel".
//!
//! This module only decides. [`Invocations`] is fed the live preview, the
//! speech the transcriber has made final, final lines and the intent checks'
//! verdicts, and answers with [`Action`]s; the meeting runs the checks, makes
//! the tasks and cancels them.
//!
//! - **A wide gate, the model decides**: it judges intent better than a text
//!   rule, and a start-of-line rule failed every spoken request in a demo
//!   meeting ("PEM can you find…", "Pam" alone). An utterance is checked when
//!   the name opens it, perhaps after a filler or misheard as a word that
//!   sounds like it, or when the name itself is anywhere in it (see
//!   [`address`]). A line that is only the name is checked with its
//!   speaker's next line.
//! - **Checked as the transcriber ends each utterance**, seconds before its
//!   line is final (an utterance ends about 0.9 s after its last word, a line
//!   5.6 s after): the words from the utterance holding the name on,
//!   again each time the speaker goes on. The check may say `unfinished`
//!   ("Pam, can you"), and the next utterance is checked with it. A request
//!   starts at once and gets its line when that is final; what its speaker
//!   adds after that changes it. A cancel in the preview counts at once,
//!   since the cancel window is short.
//! - **A cancel waits for the requests said before it** to be checked, then
//!   takes back the newest of them; with none, the meeting cancels its newest
//!   open task. So "Pam, draft an email. … Pam, cancel." cancels the email
//!   even when the cancel is heard before the email's check comes back.
//! - **"Pam, option two"** answers the question on the stage. Like a
//!   cancel it needs no model and counts at once, in the preview too.
//! - **Replies without the name**, since people answer a question the way
//!   they answer each other. While a question is on the stage, everything
//!   said is checked with it at utterance finals, name or no name, and the
//!   check may find it answers the question: "yes, the first one", or, for
//!   an open question, the words of the answer: "we're in Auckland".
//!   Without the name, nothing else counts. The countdown waits for a reply
//!   still being heard or checked ([`Invocations::replying`]).
//! - **Without an intent check** (`--intent off`, or `--scribe off` with no
//!   intent model named) nothing is checked: only the cancels and answers
//!   matched without a model count.

use std::fmt::Write as _;

use anyhow::{Context, Result, bail};
use serde_json::Value;

use crate::Ts;
use crate::clock::SECOND;
use crate::event::{Line, PreviewLine, SpkId, UtteranceId};
use crate::pipeline::{Speech, Utterance};
use crate::question::Answer;

/// How long the room has to say "Pam, cancel" once a request is announced.
pub const CANCEL_WINDOW: Ts = 5 * SECOND;
/// How long a cancel waits for speech said before it to become a final line.
/// A final line being checked is always waited for.
const CANCEL_WAIT: Ts = 8 * SECOND;
/// A cancel this soon after the last is the same one, heard again or repeated.
const CANCEL_REPEAT: Ts = 5 * SECOND;
/// How long a match in a preview or an utterance waits for its line to
/// become final.
const PREVIEW_KEPT: Ts = 30 * SECOND;

/// How long after a line that is only the name the same speaker's next line
/// may start and still go with it: "Pam … can you find the weather?".
const NAME_WAIT: Ts = 5 * SECOND;

/// Words that may come before the name: "uh, Pam", "hey Pam", "okay Pam".
/// Speech recognition hears "uh" as "a" often enough that it counts too.
const BEFORE_NAME: [&str; 18] = [
    "a", "ah", "and", "er", "erm", "hey", "hi", "hm", "oh", "ok", "okay", "right", "so", "uh", "uhm", "um",
    "well", "yo",
];

/// How an utterance may speak to the assistant. The intent check
/// decides whether it does.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Address<'a> {
    /// The name opens the utterance, perhaps after a word or two from
    /// [`BEFORE_NAME`] ("uh, Pam, …"), or a word that sounds like it does
    /// ("PEM can you…"). `rest` is what follows; `exact` when it is the name
    /// itself.
    Opens { rest: &'a str, exact: bool },
    /// The name is elsewhere in it: "can you find the weather, Pam?".
    Within,
}

/// How `text` may speak to `name`, if it may: the name in any case, as
/// whole words ("Pamela" and "Pam's" don't count). A name with no words
/// matches nothing.
fn address<'a>(text: &'a str, name: &str) -> Option<Address<'a>> {
    let name: Vec<String> = name.split_whitespace().map(str::to_lowercase).collect();
    // A name with no words would match every line.
    if name.is_empty() {
        return None;
    }
    let words = words(text);
    // The name from word `i` on: where it ends, and whether every word is the name's own.
    let at = |i: usize, sound_alike: bool| -> Option<(usize, bool)> {
        let heard = words.get(i..i + name.len())?;
        let mut exact = true;
        for ((_, w), n) in heard.iter().zip(&name) {
            let w = w.to_lowercase();
            if w != *n {
                exact = false;
                if !sound_alike || !sounds_like(&w, n) {
                    return None;
                }
            }
        }
        let (start, last) = heard[heard.len() - 1];
        Some((start + last.len(), exact))
    };
    let mut lead = 0;
    loop {
        if let Some((end, exact)) = at(lead, true) {
            let rest = text[end..].trim_start_matches(|c: char| !c.is_alphanumeric()).trim_end();
            return Some(Address::Opens { rest, exact });
        }
        let filler = words.get(lead).is_some_and(|(_, w)| BEFORE_NAME.contains(&w.to_lowercase().as_str()));
        if lead == 2 || !filler {
            break;
        }
        lead += 1;
    }
    (0..words.len()).any(|i| at(i, false).is_some()).then_some(Address::Within)
}

/// The words in `text` and where each starts: letters, digits and
/// apostrophes, so "Pam's" is one word.
fn words(text: &str) -> Vec<(usize, &str)> {
    let is_word = |c: char| c.is_alphanumeric() || c == '\'' || c == '’';
    let mut out = Vec::new();
    let mut start = None;
    for (i, c) in text.char_indices().chain([(text.len(), ' ')]) {
        match (is_word(c), start) {
            (true, None) => start = Some(i),
            (false, Some(s)) => {
                out.push((s, &text[s..i]));
                start = None;
            }
            _ => {}
        }
    }
    out
}

/// Whether `heard` may be the name word `name` misheard, both in lower case:
/// the same first letter and at most one letter different ("pem" for
/// "pam"). Only for names of three letters or more, and words with a vowel
/// ("pm" is not "pam").
fn sounds_like(heard: &str, name: &str) -> bool {
    let (h, n): (Vec<char>, Vec<char>) = (heard.chars().collect(), name.chars().collect());
    let vowel = h.iter().any(|c| "aeiouy".contains(*c));
    if n.len() < 3 || h.first() != n.first() || !vowel || !h.iter().all(|c| c.is_alphabetic()) {
        return false;
    }
    // At most one substitution, insertion or deletion.
    let (short, long) = if h.len() <= n.len() { (&h, &n) } else { (&n, &h) };
    if long.len() - short.len() > 1 {
        return false;
    }
    let same = short.iter().zip(long.iter()).take_while(|(a, b)| a == b).count();
    let skip = if short.len() == long.len() { 1 } else { 0 };
    short.get(same + skip..) == long.get(same + 1..)
}

/// Whether what followed the name is empty or only fillers: "Pam", "Pam, uh…".
fn only_name(rest: &str) -> bool {
    key(rest).split(' ').all(|w| w.is_empty() || BEFORE_NAME.contains(&w))
}

/// Whether what followed the name asks to cancel: "cancel", "cancel that",
/// "never mind", "stop". Anything longer is a request ("cancel the lunch").
fn is_cancel(rest: &str) -> bool {
    const CANCELS: [&str; 10] = [
        "cancel",
        "cancel that",
        "cancel it",
        "cancel this",
        "cancel please",
        "please cancel",
        "never mind",
        "nevermind",
        "stop",
        "stop that",
    ];
    CANCELS.contains(&key(rest).as_str())
}

/// Which option what followed the name chooses, numbered from 1: "option
/// two", "number 2", "the second one", "two". Speech recognition hears
/// "option two" as "option to" often enough that after "option" or
/// "number" it counts too. Anything longer, such as "two more things", is not
/// an answer.
fn chosen_option(rest: &str) -> Option<usize> {
    const NUMBERS: [&str; 9] = ["one", "two", "three", "four", "five", "six", "seven", "eight", "nine"];
    const ORDINALS: [&str; 9] =
        ["first", "second", "third", "fourth", "fifth", "sixth", "seventh", "eighth", "ninth"];
    let key = key(rest);
    let mut words: Vec<&str> = key.split(' ').filter(|w| !w.is_empty()).collect();
    for lead in [&["let's", "go", "with"][..], &["lets", "go", "with"], &["go", "with"], &["pick"], &["take"]]
    {
        if words.starts_with(lead) {
            words.drain(..lead.len());
            break;
        }
    }
    if words.last() == Some(&"please") {
        words.pop();
    }
    if words.first() == Some(&"the") {
        words.remove(0);
    }
    let number = |w: &str| match w.parse::<usize>() {
        Ok(n) if (1..=9).contains(&n) => Some(n),
        Ok(_) => None,
        Err(_) => NUMBERS.iter().position(|n| *n == w).map(|i| i + 1),
    };
    let ordinal = |w: &str| ORDINALS.iter().position(|n| *n == w).map(|i| i + 1);
    let misheard = |w: &str| match w {
        "to" | "too" => Some(2),
        "for" => Some(4),
        _ => None,
    };
    match words[..] {
        ["option" | "number", w] | ["option", "number", w] => number(w).or_else(|| misheard(w)),
        [w] => number(w).or_else(|| ordinal(w)),
        [w, "one" | "option"] => ordinal(w),
        _ => None,
    }
}

/// The words, in lower case, without punctuation: preview and final line
/// compare equal when they hold the same words.
fn key(text: &str) -> String {
    text.split(|c: char| !c.is_alphanumeric() && c != '\'')
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect::<Vec<_>>()
        .join(" ")
}

/// What an utterance asks, and whether the name itself opened it: what
/// followed the name, or all of it when the name is within it or isn't there.
fn asked<'a>(said: &'a str, addr: Option<Address<'a>>) -> (&'a str, bool) {
    match addr {
        Some(Address::Opens { rest, exact }) => (rest, exact),
        Some(Address::Within) | None => (said.trim(), false),
    }
}

fn ends_sentence(text: &str) -> bool {
    text.trim_end().ends_with(['.', '?', '!'])
}

/// Whether `a` and `b` may be the same speaker: a speaker not known yet, as
/// at the start of a meeting, may be anyone.
fn same(a: Option<SpkId>, b: Option<SpkId>) -> bool {
    a.is_none() || b.is_none() || a == b
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Kind {
    /// Asks the assistant to do or answer something.
    Request,
    /// Changes or adds to the latest request.
    Correction,
    /// Answers the question on the stage: with an option, numbered from 1,
    /// or an open question with the words of the answer.
    Reply(Answer),
    /// Talks to the assistant but stopped before saying what it wants, and
    /// will go on: "Pam, can you".
    Unfinished,
    /// Not for the assistant: someone else called Pam, talk about it, or a note.
    None,
}

impl Kind {
    /// Whether it touches a task: a request or a correction.
    fn acts(&self) -> bool {
        matches!(self, Kind::Request | Kind::Correction)
    }
}

/// What the intent check made of an utterance.
#[derive(Debug, Clone, PartialEq)]
pub struct Verdict {
    pub kind: Kind,
    /// What is asked, as an instruction; empty unless it acts.
    pub summary: String,
}

impl Verdict {
    fn unfinished() -> Verdict {
        Verdict { kind: Kind::Unfinished, summary: String::new() }
    }

    fn none() -> Verdict {
        Verdict { kind: Kind::None, summary: String::new() }
    }
}

/// The question on the stage, as the intent check is shown it.
#[derive(Debug, Clone, PartialEq)]
pub struct Asked {
    pub text: String,
    /// None for an open question.
    pub options: Vec<String>,
}

impl Asked {
    /// Whether `answer` answers it: an option it has, or, for an open
    /// question, some words.
    pub fn takes(&self, answer: &Answer) -> bool {
        match answer {
            Answer::Option(n) => (1..=self.options.len()).contains(n),
            Answer::Text(text) => self.options.is_empty() && !text.trim().is_empty(),
        }
    }
}

/// What the meeting should do. A spoken request is known by its `ask`, from
/// its first check until its line is final.
#[derive(Debug, Clone, PartialEq)]
pub enum Action {
    /// Run intent check `id` on `said`, by `spk`, which runs from `t0` to
    /// `t1`; `text` is what it asks: what followed the name, or all of it
    /// when the name is within it or isn't there. `line` is its final line,
    /// once that is known. `question` is the one on the stage, which it may
    /// answer.
    Check {
        id: u64,
        said: String,
        text: String,
        spk: Option<SpkId>,
        t0: Ts,
        t1: Ts,
        line: Option<UtteranceId>,
        question: Option<Asked>,
    },
    /// `ask`, said by `spk` from `t0`, asks something: `text` is what it asks,
    /// as for [`Action::Check`]. `line` is `None` while its line isn't final;
    /// [`Action::Attach`] brings it.
    Request { ask: u64, t0: Ts, spk: Option<SpkId>, line: Option<Line>, text: String, verdict: Verdict },
    /// Its speaker went on after `ask` started: `text` is all it asks now, and
    /// changes what was asked.
    Continue { ask: u64, t0: Ts, spk: Option<SpkId>, text: String, verdict: Verdict },
    /// The final line of `ask`, which was acted on before its line was final.
    Attach { ask: u64, line: Line },
    /// Cancel the newest task asked before `before` that can still be cancelled.
    Cancel { before: Ts },
    /// Answer the question on the stage with `answer`; `spk` said it.
    Answer { answer: Answer, spk: Option<SpkId> },
}

/// Speech being checked, until it is a request with a final line, a reply,
/// or nothing: from the utterance holding the name on, or, while a question
/// is on the stage, anything said.
#[derive(Debug)]
struct Candidate {
    /// Its `ask`.
    id: u64,
    /// Where the utterance holding the name starts, or the speech without it.
    t0: Ts,
    /// Where the words checked so far end.
    t1: Ts,
    /// Where its speaker's speech heard with it ends, a cancel or an answer
    /// matched in it included: its final line has more only if words never
    /// came as speech on its way.
    heard: Ts,
    spk: Option<SpkId>,
    /// What was said from `t0` on, the name included.
    said: String,
    /// What it asks: what followed the name, or all of `said` when the name
    /// is within it or isn't there.
    text: String,
    /// `text`'s words: it is checked again only when they change.
    key: String,
    /// Whether the name is in it. Without it, only a reply counts.
    addressed: bool,
    /// Whether the name itself opened it: only then does a failed check
    /// count as a request.
    exact: bool,
    /// The question on the stage when its words were last checked, by
    /// number: a reply counts only while that question is still there.
    question: Option<u64>,
    /// The check of `key` still out.
    check: Option<u64>,
    /// What the check of `key` said.
    verdict: Option<Verdict>,
    /// It was acted on: what its speaker adds now continues it.
    started: bool,
    /// Its final line, once it came.
    line: Option<Line>,
}

impl Candidate {
    /// A request or correction ready to act on: checked, and not acted on yet.
    fn ready(&self) -> bool {
        !self.started && self.check.is_none() && self.verdict.as_ref().is_some_and(|v| v.kind.acts())
    }

    /// Its answer, when the check of its latest words found it answers the
    /// question.
    fn reply(&self) -> Option<&Answer> {
        match (&self.check, &self.verdict) {
            (None, Some(Verdict { kind: Kind::Reply(answer), .. })) => Some(answer),
            _ => None,
        }
    }

    /// Not acted on, and its speaker hasn't said what they want of the
    /// assistant yet.
    fn unfinished(&self) -> bool {
        self.addressed && !self.started && self.verdict.as_ref().is_some_and(|v| v.kind == Kind::Unfinished)
    }

    /// Whether it may still answer question `q`: being checked with it, or
    /// not finished.
    fn may_reply(&self, q: u64) -> bool {
        !self.started
            && self.question == Some(q)
            && (self.check.is_some() || self.verdict.as_ref().is_some_and(|v| v.kind == Kind::Unfinished))
    }
}

/// Speech that was only the name, or unfinished, with no more words yet:
/// its speaker's next words go with it.
struct Waiting {
    spk: Option<SpkId>,
    t0: Ts,
    t1: Ts,
    said: String,
}

/// How long an acted-on request waits for its line before it is forgotten.
/// Every final word is in a line, so only a line split unlike its speech is late.
const LINE_KEPT: Ts = 60 * SECOND;

/// The spoken requests in flight, for one meeting.
#[derive(Debug)]
pub struct Invocations {
    name: String,
    /// Whether speech goes to the intent check; without one, only cancels
    /// and answers count.
    checks: bool,
    candidates: Vec<Candidate>,
    next: u64,
    next_check: u64,
    /// The question on the stage, numbered so a reply checked against one
    /// question never answers the next.
    asking: Option<(u64, Asked)>,
    asked: u64,
    /// Where each line of the latest preview starts: speech being heard.
    shown: Vec<Ts>,
    /// When a cancel was said that waits for the requests before it.
    cancel: Option<Ts>,
    /// When the last cancel was said.
    last_cancel: Option<Ts>,
    /// Where the utterances and preview sentences start that were matched
    /// without a model, as answers or cancels. Each is that, whatever it
    /// grows into: its later previews and its final line are neither
    /// answers nor requests.
    matched: Vec<Ts>,
    /// Where the speech starts that a check found answers the question on
    /// the stage. Like a match, it is that answer whatever it grows into.
    answered: Vec<Ts>,
    /// Where the latest preview has speech that starts with the name, isn't
    /// a cancel and isn't heard as an utterance yet: requests on their way,
    /// which a cancel waits for.
    coming: Vec<Ts>,
    /// Final lines that were only the name, or unfinished, each waiting for
    /// its speaker's next line: requests on their way, which a cancel waits for too.
    named: Vec<Line>,
}

impl Invocations {
    /// Spoken requests checked by the intent check.
    pub fn new(name: &str) -> Invocations {
        Invocations {
            name: name.to_string(),
            checks: true,
            candidates: Vec::new(),
            next: 1,
            next_check: 1,
            asking: None,
            asked: 0,
            shown: Vec::new(),
            cancel: None,
            last_cancel: None,
            matched: Vec::new(),
            answered: Vec::new(),
            coming: Vec::new(),
            named: Vec::new(),
        }
    }

    /// Without an intent check (`--intent off`, or `--scribe off` with no
    /// intent model named): nothing is checked, so
    /// only "Pam, cancel" and "Pam, option two" count.
    pub fn local(name: &str) -> Invocations {
        Invocations { checks: false, ..Invocations::new(name) }
    }

    /// Whether a cancel is waiting to be decided. Cancel windows stay open
    /// meanwhile, so a cancel said in time is never too late.
    pub fn holding(&self) -> bool {
        self.cancel.is_some()
    }

    /// The question on the stage changed: `asked` is now there, or none is.
    /// While one is, everything said is checked with it, when there is
    /// an intent check.
    pub fn question(&mut self, asked: Option<Asked>) {
        // What was being checked only as a reply answers nothing now.
        self.candidates.retain(|c| c.addressed);
        self.asking = asked.map(|a| {
            self.asked += 1;
            (self.asked, a)
        });
    }

    /// Whether what is said is checked as a reply to a question on the stage.
    pub fn hears_replies(&self) -> bool {
        self.checks && self.asking.is_some()
    }

    /// Whether "Pam, option `n`" may answer: the question on the stage has
    /// that option, or there is none, and the answer is nothing. An option
    /// the question lacks takes nothing, so the utterance may go on to say
    /// another ("option three. No, option two.").
    fn offers(&self, n: usize) -> bool {
        self.asking.as_ref().is_none_or(|(_, a)| a.takes(&Answer::Option(n)))
    }

    /// Which option what followed the name chooses, as [`chosen_option`]
    /// reads it, unless the question on the stage is open: "Pam, two" may
    /// then be its answer, which only the intent check can tell.
    fn option_in(&self, rest: &str) -> Option<usize> {
        let open = self.asking.as_ref().is_some_and(|(_, a)| a.options.is_empty());
        chosen_option(rest).filter(|_| !open)
    }

    /// Whether speech begun before `cut` may still answer the question on
    /// the stage at `now`: it waits for its next line after the name alone
    /// ("Pam." / "Option two."), or, with an intent check, it is being heard
    /// or checked.
    pub fn replying(&self, cut: Ts, now: Ts) -> bool {
        let Some((q, _)) = self.asking.as_ref() else { return false };
        let decided = |t: Ts| self.matched.contains(&t) || self.answered.contains(&t);
        self.named.iter().any(|n| n.t0 < cut && now < n.t1 + NAME_WAIT)
            || self.checks
                && (self.shown.iter().any(|&t| t < cut && !decided(t))
                    || self.candidates.iter().any(|c| c.t0 < cut && c.may_reply(*q)))
    }

    /// A new live preview. A cancel or an answer in a sentence that starts
    /// with the name counts at once; other speech that starts with it is on
    /// its way, until the transcriber makes it final.
    pub fn preview(&mut self, preview: &[PreviewLine], now: Ts) -> Vec<Action> {
        let mut out = Vec::new();
        self.coming.clear();
        self.shown = preview.iter().map(|p| p.t0).collect();
        for (i, p) in preview.iter().enumerate() {
            let Some(addr) = address(&p.text, &self.name) else { continue };
            if self.matched.contains(&p.t0) || self.answered.contains(&p.t0) {
                continue;
            }
            if let Address::Opens { rest, .. } = addr {
                if is_cancel(rest) {
                    if ends_sentence(&p.text) {
                        self.matched.push(p.t0);
                        self.heard_cancel(p.t0);
                    }
                    continue;
                }
                if let Some(option) = self.option_in(rest) {
                    if ends_sentence(&p.text) && self.offers(option) {
                        self.matched.push(p.t0);
                        out.push(Action::Answer { answer: Answer::Option(option), spk: p.spk });
                    }
                    continue;
                }
            }
            if !self.checks {
                continue;
            }
            // Speech already heard as utterances is checked; it is no longer only on its way.
            let until = preview.get(i + 1).map_or(Ts::MAX, |n| n.t0);
            if !self.candidates.iter().any(|c| c.t0 >= p.t0 && c.t0 < until) {
                self.coming.push(p.t0);
            }
        }
        out.extend(self.settle(now));
        out
    }

    /// The transcriber made an utterance final; `speech` is every final word
    /// not in a final line yet. A line on its way that holds the name
    /// is checked from the utterance holding it on, each time it grows, and,
    /// while a question is on the stage, so is one without the name; a
    /// cancel or an answer opening an utterance counts at once.
    pub fn heard(&mut self, speech: &[Speech], now: Ts) -> Vec<Action> {
        let mut out = Vec::new();
        let mut waiting: Vec<Waiting> = self
            .named
            .iter()
            .map(|n| Waiting { spk: n.spk, t0: n.t0, t1: n.t1, said: n.text.clone() })
            .collect();
        let question = self.asking.as_ref().map(|(q, a)| (*q, a.clone()));
        for s in speech {
            // A reply: whatever its speaker adds to it is that answer.
            if s.utterances.iter().any(|u| self.answered.contains(&u.t0)) {
                continue;
            }
            for u in &s.utterances {
                let Some(Address::Opens { rest, .. }) = address(&u.text, &self.name) else { continue };
                if self.matched.contains(&u.t0) {
                    continue;
                }
                if is_cancel(rest) {
                    self.matched.push(u.t0);
                    self.heard_cancel(u.t0);
                } else if let Some(option) = chosen_option(rest).filter(|&o| self.offers(o)) {
                    self.matched.push(u.t0);
                    out.push(Action::Answer { answer: Answer::Option(option), spk: s.spk });
                }
            }
            let kept: Vec<&Utterance> =
                s.utterances.iter().filter(|u| !self.matched.contains(&u.t0)).collect();
            let Some(last) = kept.last() else { continue };
            let joined = |u: &[&Utterance]| u.iter().map(|u| u.text.as_str()).collect::<Vec<_>>().join(" ");
            // The gate is the line's; what is checked starts at the utterance holding the name.
            let line = joined(&kept);
            let (t0, said) = match address(&line, &self.name) {
                Some(Address::Opens { .. }) => (kept[0].t0, line),
                Some(Address::Within) => {
                    let named = |u: &&&Utterance| {
                        matches!(
                            address(&u.text, &self.name),
                            Some(Address::Within | Address::Opens { exact: true, .. })
                        )
                    };
                    let i = kept.iter().position(|u| named(&u)).unwrap_or(0);
                    (kept[i].t0, joined(&kept[i..]))
                }
                None => {
                    // The next words of a speaker who said only the name, or stopped short.
                    let w = waiting.iter().position(|w| {
                        same(w.spk, s.spk) && w.t1 <= kept[0].t0 && kept[0].t0 <= w.t1 + NAME_WAIT
                    });
                    match w {
                        Some(w) => {
                            let w = waiting.remove(w);
                            (w.t0, format!("{} {line}", w.said))
                        }
                        // While a question is on the stage, anything said may answer it.
                        None if self.hears_replies() => (kept[0].t0, line),
                        None => continue,
                    }
                }
            };
            let addr = address(&said, &self.name);
            if let Some(Address::Opens { rest, .. }) = addr {
                // The name, then a cancel or an answer in its speaker's next words: "Pam." / "Option two."
                let option = chosen_option(rest).filter(|&o| self.offers(o));
                if is_cancel(rest) || option.is_some() {
                    self.matched.push(t0);
                    self.matched.extend(kept.iter().map(|u| u.t0));
                    self.named.retain(|n| n.t0 != t0);
                    self.candidates.retain(|c| c.t0 != t0 || c.started);
                    match option {
                        Some(option) => {
                            out.push(Action::Answer { answer: Answer::Option(option), spk: s.spk })
                        }
                        None => self.heard_cancel(t0),
                    }
                    continue;
                }
            }
            if !self.checks {
                continue;
            }
            let (text, exact) = asked(&said, addr);
            let key = key(text);
            let (spk, t1) = (s.spk, last.t1);
            let (i, new) = match self.candidates.iter().position(|c| c.t0 == t0) {
                Some(i) => (i, false),
                None => {
                    self.candidates.push(Candidate {
                        id: self.next,
                        t0,
                        t1,
                        heard: t1,
                        spk,
                        said: String::new(),
                        text: String::new(),
                        key: String::new(),
                        addressed: addr.is_some(),
                        exact,
                        question: None,
                        check: None,
                        verdict: None,
                        started: false,
                        line: None,
                    });
                    self.next += 1;
                    (self.candidates.len() - 1, true)
                }
            };
            let c = &mut self.candidates[i];
            c.spk = spk;
            c.t1 = t1;
            c.heard = s.utterances[s.utterances.len() - 1].t1;
            if (new || c.key != key) && c.line.is_none() {
                c.said = said.clone();
                c.text = text.to_string();
                c.key = key;
                if !c.started {
                    c.addressed = addr.is_some();
                    c.exact = exact;
                }
                c.question = question.as_ref().map(|(q, _)| *q);
                if c.addressed && only_name(text) {
                    // "Pam" alone: unfinished, without asking a model.
                    c.check = None;
                    c.verdict = Some(Verdict::unfinished());
                } else {
                    let id = self.next_check;
                    self.next_check += 1;
                    c.check = Some(id);
                    c.verdict = None;
                    out.push(Action::Check {
                        id,
                        said,
                        text: c.text.clone(),
                        spk,
                        t0,
                        t1,
                        line: None,
                        question: question.as_ref().map(|(_, a)| a.clone()),
                    });
                }
            }
            let c = &self.candidates[i];
            if c.unfinished() {
                waiting.push(Waiting { spk, t0: c.t0, t1, said: c.said.clone() });
            }
        }
        out.extend(self.settle(now));
        out
    }

    /// A final line. One heard on its way gives its requests their line;
    /// otherwise one that holds the name, or any while a question is on the
    /// stage, is checked now.
    pub fn line(&mut self, line: &Line, now: Ts) -> Vec<Action> {
        // A line that was only the name, or unfinished, and this one, its
        // speaker's next, are one utterance. Final lines come seconds after
        // their speech, so a wait ends with a line that starts after it, not
        // with the clock.
        self.named.retain(|n| line.t0 <= n.t1 + NAME_WAIT);
        let joined;
        let line = match self.named.iter().position(|n| same(n.spk, line.spk)) {
            Some(i) => {
                let named = self.named.remove(i);
                joined = Line { t0: named.t0, text: format!("{} {}", named.text, line.text), ..line.clone() };
                &joined
            }
            None => line,
        };
        let within = |t: Ts| t >= line.t0 && t <= line.t1;
        self.coming.retain(|&t| !within(t));
        self.shown.retain(|&t| !within(t));
        // What in it answered the question or was matched without a model was
        // acted on already: it is that, and no request.
        let answered = self.answered.iter().any(|&t| within(t));
        self.answered.retain(|&t| !within(t));
        if answered {
            self.candidates.retain(|c| c.addressed || !within(c.t0));
        }
        let matched = answered || self.matched.iter().any(|&t| within(t));
        self.matched.retain(|&t| !within(t));
        let question = self.asking.as_ref().map(|(q, a)| (*q, a.clone()));
        let heard: Vec<u64> = self.candidates.iter().filter(|c| within(c.t0)).map(|c| c.id).collect();
        if !heard.is_empty() {
            let mut out = Vec::new();
            for id in heard {
                let i = self.candidates.iter().position(|c| c.id == id).unwrap();
                let c = &mut self.candidates[i];
                if c.t1 > line.t1 {
                    // The line only opens it, as a line that is only the name does:
                    // the request's line is its speaker's next.
                    if c.addressed {
                        self.named.retain(|n| n.spk != line.spk);
                        self.named.push(line.clone());
                    }
                    continue;
                }
                if c.started {
                    out.push(Action::Attach { ask: c.id, line: line.clone() });
                }
                c.line = Some(line.clone());
                if c.heard < line.t1 {
                    // Words its speech didn't have, such as a preview the input's
                    // end made final: check what the line says. For a started
                    // request they continue it.
                    let addr = address(&line.text, &self.name);
                    let (text, exact) = asked(&line.text, addr);
                    if (addr.is_some() || question.is_some()) && key(text) != c.key {
                        let id = self.next_check;
                        self.next_check += 1;
                        (c.said, c.text, c.key, c.t1, c.heard) =
                            (line.text.clone(), text.to_string(), key(text), line.t1, line.t1);
                        if !c.started {
                            (c.addressed, c.exact) = (addr.is_some(), exact);
                        }
                        c.question = question.as_ref().map(|(q, _)| *q);
                        c.check = Some(id);
                        c.verdict = None;
                        out.push(Action::Check {
                            id,
                            said: line.text.clone(),
                            text: c.text.clone(),
                            spk: line.spk,
                            t0: c.t0,
                            t1: line.t1,
                            line: Some(line.id),
                            question: question.as_ref().map(|(_, a)| a.clone()),
                        });
                    }
                }
                match c.started {
                    // A check of what its speaker added may still be out: it continues it all the same.
                    true if c.check.is_none() => drop(self.candidates.remove(i)),
                    true => {}
                    false => self.line_ends(i),
                }
            }
            out.extend(self.settle(now));
            return out;
        }
        if matched {
            return self.settle(now);
        }
        let addr = address(&line.text, &self.name);
        if let Some(Address::Opens { rest, .. }) = addr {
            if is_cancel(rest) {
                self.heard_cancel(line.t0);
                return self.settle(now);
            }
            if let Some(option) = self.option_in(rest) {
                let mut out = Vec::new();
                if self.offers(option) {
                    out.push(Action::Answer { answer: Answer::Option(option), spk: line.spk });
                }
                out.extend(self.settle(now));
                return out;
            }
        }
        let (text, exact) = asked(&line.text, addr);
        if addr.is_some() && only_name(text) {
            // Its speaker says what they want in their next line, checked or not ("Pam" / "cancel").
            self.named.retain(|n| n.spk != line.spk);
            self.named.push(line.clone());
            return self.settle(now);
        }
        if !self.checks || (addr.is_none() && question.is_none()) {
            return self.settle(now);
        }
        let id = self.next_check;
        self.next_check += 1;
        self.candidates.push(Candidate {
            id: self.next,
            t0: line.t0,
            t1: line.t1,
            heard: line.t1,
            spk: line.spk,
            said: line.text.clone(),
            text: text.to_string(),
            key: key(text),
            addressed: addr.is_some(),
            exact,
            question: question.as_ref().map(|(q, _)| *q),
            check: Some(id),
            verdict: None,
            started: false,
            line: Some(line.clone()),
        });
        self.next += 1;
        let mut out = vec![Action::Check {
            id,
            said: line.text.clone(),
            text: text.to_string(),
            spk: line.spk,
            t0: line.t0,
            t1: line.t1,
            line: Some(line.id),
            question: question.map(|(_, a)| a),
        }];
        out.extend(self.settle(now));
        out
    }

    /// Candidate `i`'s line is final and it wasn't acted on: once checked,
    /// unfinished waits for its speaker's next line, and nothing is done with.
    fn line_ends(&mut self, i: usize) {
        let c = &self.candidates[i];
        if c.check.is_some() {
            return;
        }
        match c.verdict.as_ref().map(|v| &v.kind) {
            Some(Kind::Unfinished) if c.addressed => {
                let c = self.candidates.remove(i);
                let line = c.line.unwrap();
                self.named.retain(|n| n.spk != line.spk);
                self.named.push(Line { t0: c.t0, text: c.said, ..line });
            }
            Some(Kind::Unfinished | Kind::None) => {
                self.candidates.remove(i);
            }
            _ => {}
        }
    }

    /// The intent check `id` came back. Only the check of a candidate's
    /// latest words counts. Without the name only a reply counts, and a reply
    /// only while the question it was checked with is on the stage, with an
    /// answer that question takes: an option it has, or words for an open
    /// one. Anything else is nothing, so its speaker's next words may still
    /// be checked.
    pub fn verdict(&mut self, id: u64, verdict: Verdict, now: Ts) -> Vec<Action> {
        let mut out = Vec::new();
        if let Some(i) = self.candidates.iter().position(|c| c.check == Some(id)) {
            let c = &mut self.candidates[i];
            c.check = None;
            if c.started {
                if verdict.kind.acts() {
                    out.push(Action::Continue {
                        ask: c.id,
                        t0: c.t0,
                        spk: c.spk,
                        text: c.text.clone(),
                        verdict: verdict.clone(),
                    });
                }
                c.verdict = Some(verdict);
                // Its line came while the check was out: nothing more can be added now.
                if c.line.is_some() {
                    self.candidates.remove(i);
                }
            } else {
                let counts = match &verdict.kind {
                    Kind::Reply(answer) => {
                        self.asking.as_ref().is_some_and(|(q, a)| c.question == Some(*q) && a.takes(answer))
                    }
                    Kind::Request | Kind::Correction => c.addressed,
                    Kind::Unfinished | Kind::None => true,
                };
                c.verdict = Some(if counts { verdict } else { Verdict::none() });
                if c.line.is_some() {
                    self.line_ends(i);
                }
            }
        }
        out.extend(self.settle(now));
        out
    }

    /// The intent check `id` failed. When the name itself opened the
    /// utterance it counts as a request: a cancel takes back a false start,
    /// but a lost question stays lost. Otherwise it counts as nothing, so a
    /// name said in passing never starts a task. Words added to a
    /// started request change nothing.
    pub fn failed(&mut self, id: u64, now: Ts) -> Vec<Action> {
        let Some(c) = self.candidates.iter().find(|c| c.check == Some(id)) else { return self.settle(now) };
        let verdict = match c.exact && !c.started {
            true => Verdict { kind: Kind::Request, summary: c.text.clone() },
            false => Verdict::none(),
        };
        self.verdict(id, verdict, now)
    }

    /// The meeting clock moved on.
    pub fn tick(&mut self, now: Ts) -> Vec<Action> {
        self.matched.retain(|&t| now < t + PREVIEW_KEPT);
        self.answered.retain(|&t| now < t + PREVIEW_KEPT);
        self.candidates.retain(|c| c.line.is_some() || now < c.t1 + LINE_KEPT);
        self.settle(now)
    }

    /// The input ended: every line is final, so previews will never be.
    pub fn end(&mut self, now: Ts) -> Vec<Action> {
        self.candidates.retain(|c| c.line.is_some());
        self.coming.clear();
        self.shown.clear();
        self.matched.clear();
        self.answered.clear();
        self.named.clear();
        self.settle(now)
    }

    /// What the newest request said before `t0` asked, while it isn't a
    /// task yet: being checked, or waiting for a cancel or the requests
    /// before it. A correction may be about it.
    pub fn asked_before(&self, t0: Ts) -> Option<&str> {
        self.candidates
            .iter()
            .filter(|c| {
                c.addressed
                    && c.t0 < t0
                    && !c.started
                    && c.verdict.as_ref().is_none_or(|v| v.kind.acts())
                    && !only_name(&c.text)
            })
            .max_by_key(|c| c.t0)
            .map(|c| c.text.as_str())
    }

    fn heard_cancel(&mut self, t0: Ts) {
        if self.last_cancel.is_some_and(|last| t0.abs_diff(last) < CANCEL_REPEAT) {
            return;
        }
        self.last_cancel = Some(t0);
        self.cancel.get_or_insert(t0);
    }

    /// Sends on what is decided: replies at once, requests in the order they
    /// were said, and a cancel once nothing said before it is still undecided.
    fn settle(&mut self, now: Ts) -> Vec<Action> {
        let mut out = Vec::new();
        // A reply answers now, before its line is final: like "Pam, option
        // two", the speech is that answer, whatever it grows into.
        for c in self.candidates.extract_if(.., |c| !c.started && c.reply().is_some()) {
            if c.line.is_none() {
                self.answered.push(c.t0);
            }
            out.extend(c.reply().map(|answer| Action::Answer { answer: answer.clone(), spk: c.spk }));
        }
        if let Some(at) = self.cancel {
            // Only what has the name in it may be a request to take back.
            let before = |c: &Candidate| c.addressed && c.t0 < at;
            // A check always gets a verdict, or fails at its own time limit:
            // the cancel waits for it.
            let checking = self.candidates.iter().any(|c| before(c) && !c.started && c.check.is_some());
            // Speech whose speaker hasn't said what they want may never: that wait has a limit.
            let unfinished = self.coming.iter().any(|&t| t < at)
                || self.named.iter().any(|n| n.t0 < at)
                || self.candidates.iter().any(|c| before(c) && c.unfinished());
            if !checking && (!unfinished || now >= at + CANCEL_WAIT) {
                self.cancel = None;
                // The cancel takes back the newest request said before it, or else a task.
                let newest = self.candidates.iter().filter(|c| before(c) && c.ready()).map(|c| c.t0).max();
                match newest {
                    Some(t0) => self.candidates.retain(|c| !(c.t0 == t0 && c.ready())),
                    None => out.push(Action::Cancel { before: at }),
                }
                // Whatever before it is still unfinished now never will be in time.
                self.named.retain(|n| n.t0 >= at);
                self.candidates.retain(|c| !before(c) || !c.unfinished());
            }
        }
        // In the order they were said: a request waits for any request
        // before it that is still being checked or held by a cancel, so a
        // correction never overtakes the request it corrects. Speech without
        // the name holds nothing up.
        self.candidates.sort_by_key(|c| c.t0);
        let mut i = 0;
        while i < self.candidates.len() {
            let c = &mut self.candidates[i];
            let decided_not = c.check.is_none() && c.verdict.as_ref().is_some_and(|v| !v.kind.acts());
            if !c.addressed || c.started || decided_not {
                i += 1;
                continue;
            }
            if c.check.is_some() || self.cancel.is_some_and(|at| c.t0 < at) {
                break;
            }
            c.started = true;
            out.push(Action::Request {
                ask: c.id,
                t0: c.t0,
                spk: c.spk,
                line: c.line.clone(),
                text: c.text.clone(),
                verdict: c.verdict.clone().unwrap(),
            });
            // Its line is final already: nothing more can be added to it.
            if c.line.is_some() {
                self.candidates.remove(i);
            } else {
                i += 1;
            }
        }
        out
    }
}

/// The intent check's instructions.
pub fn system_prompt(name: &str) -> String {
    format!(
        r#"You decide whether something said in an in-person meeting is an instruction to {name}, the meeting's AI assistant. People talk to {name} by saying its name, usually first: "{name}, what did we decide about pricing?", "uh, {name}, can you find…", or at the end: "what's the weather tomorrow, {name}?". Someone may say the name, pause, and go on. But the name can also come up for other reasons: someone else called {name} ("{name} from finance said..."), or talk about the assistant ("{name} got that wrong earlier", "I asked {name} yesterday").

The transcript comes from speech recognition, so expect misheard words, fillers and missing punctuation. The name itself may be misheard as a word that sounds like it: the utterance may open with that word instead of "{name}". Judge by what the speaker means.

Reply with one JSON object and nothing else:
{{"kind": "request", "summary": "..."}}

kind is one of:
- "request": the speaker asks {name} to answer, find, do or show something.
- "correction": the speaker changes or adds to {name}'s latest request, shown below, such as "{name}, I meant the annual price" or "{name}, only for Europe". Only when a latest request is shown.
- "reply": the speaker answers {name}'s question, shown below with its numbered options, by choosing one: "yes, the first one", "let's do Globex", "number two I think". People answer it as they answer each other, usually without the name. Add "option": the number chosen, as in {{"kind": "reply", "option": 2, "summary": ""}}. Only when a question is shown, and only when it is clear which option they chose. An open question has no options: add "text", the answer in a few words, as in {{"kind": "reply", "text": "Auckland", "summary": ""}}.
- "unfinished": the speaker is talking to {name} but stopped before saying what they want, and will go on: "{name}, can you", "hey {name} do you", "{name}, find me the". A request that makes sense as it stands is a request.
- "none": anything else, including a request to note something ("{name}, note that..."), which the meeting notes take care of, and talk among the people in the room while a question is shown.

summary: for a request or a correction, what is asked, as a short instruction of at most 12 words ("Find what was decided about the launch date"). Empty otherwise."#
    )
}

/// What the intent check is shown about one utterance.
pub struct CheckInput<'a> {
    /// The lines just before it, as `pam transcript` prints them.
    pub recent: &'a [String],
    /// The latest request, if one is still going: its ID, where it stands
    /// and what was asked.
    pub latest: Option<(String, &'a str, &'a str)>,
    /// The question on the stage, if one is: the utterance may answer it.
    pub question: Option<&'a Asked>,
    /// Who said it, e.g. `Jordan` or `spk_2`.
    pub by: &'a str,
    /// What was said, the name included.
    pub said: &'a str,
}

pub fn check_prompt(input: &CheckInput) -> String {
    let mut out = String::from("RECENT LINES\n");
    if input.recent.is_empty() {
        out.push_str("(none)\n");
    }
    for l in input.recent {
        writeln!(out, "{l}").unwrap();
    }
    out.push_str("\nLATEST REQUEST\n");
    match &input.latest {
        Some((id, status, request)) => writeln!(out, "{id} ({status}): {request}").unwrap(),
        None => out.push_str("(none)\n"),
    }
    if let Some(q) = input.question {
        writeln!(out, "\nQUESTION ON THE STAGE\n{}", q.text).unwrap();
        if q.options.is_empty() {
            out.push_str("(open: answered in words)\n");
        }
        for (i, o) in q.options.iter().enumerate() {
            writeln!(out, "{}. {o}", i + 1).unwrap();
        }
    }
    write!(out, "\nUTTERANCE\n{}: {}\n", input.by, input.said).unwrap();
    out
}

/// The verdict's JSON as a GBNF grammar, which a local model's reply is held
/// to, so every reply parses: `kind` first, then an optional `summary`
/// and, for a reply, the answer `question` takes, in either order, as the
/// prompt's examples put it first: an `option` from 1 to 9 for a question
/// with options, a `text` for an open one, and none with no question
/// on the stage. Held to one shape, a small model can't put an open
/// question's answer in an option or its summary.
pub fn grammar(question: Option<&Asked>) -> String {
    let answer = match question {
        None => None,
        Some(q) if q.options.is_empty() => Some(r#""\"text\"" ws ":" ws string"#),
        Some(_) => Some(r#""\"option\"" ws ":" ws [1-9]"#),
    };
    let fields = match answer {
        None => "fields ::= ( ws \",\" ws summary )?\n".to_string(),
        Some(answer) => format!(
            "fields ::= ( ws \",\" ws summary )? ( ws \",\" ws answer )? | ws \",\" ws answer ws \",\" ws summary\nanswer ::= {answer}\n"
        ),
    };
    format!(
        r#"root ::= "{{" ws "\"kind\"" ws ":" ws kind fields ws "}}"
kind ::= "\"request\"" | "\"correction\"" | "\"reply\"" | "\"unfinished\"" | "\"none\""
{fields}summary ::= "\"summary\"" ws ":" ws string
string ::= "\"" char* "\""
char ::= [^"\\\x7F\x00-\x1F] | "\\" ( ["\\/bfnrt] | "u" [0-9a-fA-F]{{4}} )
ws ::= | " " | "\n" [ \t]{{0,20}}
"#
    )
}

/// Reads the check's reply: the JSON object in it, even inside a code fence.
pub fn parse(reply: &str) -> Result<Verdict> {
    let (Some(start), Some(end)) = (reply.find('{'), reply.rfind('}')) else {
        bail!("the intent check's reply has no JSON object: {}", reply.trim());
    };
    let v: Value = serde_json::from_str(&reply[start..=end])
        .with_context(|| format!("the intent check's reply isn't valid JSON: {}", reply.trim()))?;
    let kind = match v["kind"].as_str().map(str::to_lowercase).as_deref() {
        Some("request") => Kind::Request,
        Some("correction") => Kind::Correction,
        Some("unfinished") => Kind::Unfinished,
        Some("none") => Kind::None,
        // A reply that gives no option and no words answers nothing.
        Some("reply") => match (v["option"].as_u64(), v["text"].as_str().map(str::trim)) {
            (Some(n @ 1..), _) => Kind::Reply(Answer::Option(n as usize)),
            (_, Some(text)) if !text.is_empty() => Kind::Reply(Answer::Text(text.to_string())),
            _ => Kind::None,
        },
        _ => bail!("the intent check's reply has no kind: {}", reply.trim()),
    };
    let summary = v["summary"].as_str().unwrap_or_default().trim().to_string();
    Ok(Verdict { kind, summary })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::event::UtteranceId;

    fn line(id: u32, secs: (f64, f64), text: &str) -> Line {
        Line {
            id: UtteranceId(id),
            t0: (secs.0 * SECOND as f64) as Ts,
            t1: (secs.1 * SECOND as f64) as Ts,
            slot: Some(1),
            spk: Some(SpkId(1)),
            lang: None,
            text: text.into(),
        }
    }

    fn preview(at: f64, text: &str) -> Vec<PreviewLine> {
        vec![PreviewLine { spk: Some(SpkId(1)), text: text.into(), t0: (at * SECOND as f64) as Ts }]
    }

    fn s(secs: u64) -> Ts {
        secs * SECOND
    }

    fn request(summary: &str) -> Verdict {
        Verdict { kind: Kind::Request, summary: summary.into() }
    }

    fn none() -> Verdict {
        Verdict { kind: Kind::None, summary: String::new() }
    }

    fn checks(actions: &[Action]) -> Vec<(u64, &str, Option<UtteranceId>)> {
        actions
            .iter()
            .filter_map(|a| match a {
                Action::Check { id, text, line, .. } => Some((*id, text.as_str(), *line)),
                _ => None,
            })
            .collect()
    }

    fn requests(actions: &[Action]) -> Vec<(UtteranceId, &str)> {
        actions
            .iter()
            .filter_map(|a| match a {
                Action::Request { line: Some(line), text, .. } => Some((line.id, text.as_str())),
                _ => None,
            })
            .collect()
    }

    fn heard(spk: u32, utterances: &[(f64, f64, &str)]) -> Speech {
        Speech {
            spk: Some(SpkId(spk)),
            utterances: utterances
                .iter()
                .map(|&(t0, t1, text)| Utterance {
                    t0: (t0 * SECOND as f64) as Ts,
                    t1: (t1 * SECOND as f64) as Ts,
                    text: text.into(),
                })
                .collect(),
        }
    }

    fn unfinished() -> Verdict {
        Verdict::unfinished()
    }

    fn reply(option: usize) -> Verdict {
        Verdict { kind: Kind::Reply(Answer::Option(option)), summary: String::new() }
    }

    fn said(text: &str) -> Verdict {
        Verdict { kind: Kind::Reply(Answer::Text(text.into())), summary: String::new() }
    }

    fn open() -> Asked {
        Asked { text: "Which city are you in?".into(), options: Vec::new() }
    }

    fn answered(text: &str) -> Action {
        Action::Answer { answer: Answer::Text(text.into()), spk: Some(SpkId(1)) }
    }

    fn asked() -> Asked {
        Asked {
            text: "Which customer is the email for?".into(),
            options: vec!["Acme".into(), "Globex".into()],
        }
    }

    fn answer(option: usize) -> Action {
        Action::Answer { answer: Answer::Option(option), spk: Some(SpkId(1)) }
    }

    /// The question each check was shown.
    fn shown(actions: &[Action]) -> Vec<Option<&Asked>> {
        actions
            .iter()
            .filter_map(|a| match a {
                Action::Check { question, .. } => Some(question.as_ref()),
                _ => None,
            })
            .collect()
    }

    /// `(ask, line, text)` of every request, its line or not.
    fn started(actions: &[Action]) -> Vec<(u64, Option<UtteranceId>, &str)> {
        actions
            .iter()
            .filter_map(|a| match a {
                Action::Request { ask, line, text, .. } => {
                    Some((*ask, line.as_ref().map(|l| l.id), text.as_str()))
                }
                _ => None,
            })
            .collect()
    }

    fn opens(rest: &str, exact: bool) -> Option<Address<'_>> {
        Some(Address::Opens { rest, exact })
    }

    #[test]
    fn the_name_opens_the_utterance_as_a_whole_word() {
        assert_eq!(address("Pam, what did we decide?", "Pam"), opens("what did we decide?", true));
        assert_eq!(address("pam. What did we decide", "Pam"), opens("What did we decide", true));
        assert_eq!(address("…Pam: cancel.", "Pam"), opens("cancel.", true));
        assert_eq!(address("Pam", "Pam"), opens("", true));
        assert_eq!(address("Pam from finance said", "Pam"), opens("from finance said", true));
        assert_eq!(address("jarvis  two, go", "Jarvis Two"), opens("go", true));
        assert_eq!(address("Émile, go", "émile"), opens("go", true));
        assert_eq!(address("Pamela said", "Pam"), None);
        assert_eq!(address("Pam's idea", "Pam"), None);
        assert_eq!(address("Pam’s idea", "Pam"), None);
        assert_eq!(address("We ship Friday.", ""), None, "an empty name matches nothing");
        assert_eq!(address("We ship Friday.", "  "), None);
    }

    #[test]
    fn a_filler_or_two_may_come_before_the_name() {
        assert_eq!(
            address("a Pam, uh can you find the weather", "Pam"),
            opens("uh can you find the weather", true)
        );
        assert_eq!(address("So Pam, what now?", "Pam"), opens("what now?", true));
        assert_eq!(address("Okay, hey Pam, go", "Pam"), opens("go", true));
        assert_eq!(address("Hey Jarvis Two, go", "Jarvis Two"), opens("go", true));
        // Three fillers, or a word that isn't one: the name is only within it.
        assert_eq!(address("so um okay Pam, go", "Pam"), Some(Address::Within));
        assert_eq!(address("Ask Pam what she thinks", "Pam"), Some(Address::Within));
    }

    #[test]
    fn a_word_that_sounds_like_the_name_may_open_the_utterance() {
        assert_eq!(address("PEM can you find", "Pam"), opens("can you find", false));
        assert_eq!(address("uh Palm, book a room", "Pam"), opens("book a room", false));
        assert_eq!(address("Pan cancel", "Pam"), opens("cancel", false));
        assert_eq!(address("Jarvis to, go", "Jarvis Two"), opens("go", false), "each word may sound alike");
        assert_eq!(address("Jervis Two, go", "Jarvis Two"), opens("go", false));
        // A different first letter, two letters off, no vowel, or a short name.
        for no in ["Ham, go", "Pump it", "PM go", "Pamela said", "Bo go"] {
            assert_eq!(address(no, if no.starts_with("Bo") { "Al" } else { "Pam" }), None, "{no}");
        }
        // Not within: "pay" is no call to Pam.
        assert_eq!(address("we need to pay them", "Pam"), None);
    }

    #[test]
    fn the_name_anywhere_else_counts_only_as_itself() {
        assert_eq!(address("can you find the weather, Pam?", "Pam"), Some(Address::Within));
        assert_eq!(address("I told Pam about it", "Pam"), Some(Address::Within));
        assert_eq!(address("the idea of Pam's", "Pam"), None);
        assert_eq!(address("can you find the weather, PEM?", "Pam"), None);
    }

    #[test]
    fn sound_alikes_are_one_letter_off() {
        for (heard, yes) in
            [("pem", true), ("pam", true), ("pams", true), ("pa", true), ("palm", true), ("pram", true)]
        {
            assert_eq!(sounds_like(heard, "pam"), yes, "{heard}");
        }
        for heard in ["pm", "ham", "pump", "pamela", "p4m", "spam"] {
            assert!(!sounds_like(heard, "pam"), "{heard}");
        }
    }

    #[test]
    fn only_a_short_cancel_is_a_cancel() {
        for yes in ["cancel", "Cancel.", "cancel that!", "never mind", "Stop."] {
            assert!(is_cancel(yes), "{yes}");
        }
        for no in ["cancel the lunch order", "stop the build and restart it", "what did we cancel?", ""] {
            assert!(!is_cancel(no), "{no}");
        }
    }

    #[test]
    fn option_two_chooses_the_second_option() {
        for (said, n) in [
            ("option two.", 2),
            ("Option 2", 2),
            ("option to", 2),
            ("option number three", 3),
            ("number four, please", 4),
            ("the second one", 2),
            ("first", 1),
            ("let's go with option one", 1),
            ("two.", 2),
            ("ninth option", 9),
        ] {
            assert_eq!(chosen_option(said), Some(n), "{said}");
        }
        for not in ["two more things", "to the point", "option ten", "0", "option", "for the record", ""] {
            assert_eq!(chosen_option(not), None, "{not}");
        }
    }

    #[test]
    fn an_answer_needs_no_check_and_counts_once() {
        let mut inv = Invocations::new("Pam");
        assert!(inv.preview(&preview(10.0, "Pam, option two"), s(10)).is_empty(), "may still grow");
        let out = inv.preview(&preview(10.0, "Pam, option two."), s(11));
        assert_eq!(out, [Action::Answer { answer: Answer::Option(2), spk: Some(SpkId(1)) }]);
        assert!(inv.coming.is_empty(), "not a request on its way: a cancel needn't wait for it");
        assert!(inv.line(&line(5, (10.0, 11.0), "Pam, option two."), s(13)).is_empty(), "the same answer");
        let out = inv.line(&line(6, (30.0, 31.0), "Pam, the first one."), s(32));
        assert_eq!(out, [Action::Answer { answer: Answer::Option(1), spk: Some(SpkId(1)) }]);
        assert!(inv.candidates.is_empty());
    }

    #[test]
    fn another_option_said_straight_after_still_counts() {
        // "option three" for a question with two options answers nothing, so
        // the correction a moment later must get through.
        let mut inv = Invocations::new("Pam");
        let out = inv.preview(&preview(10.0, "Pam, option three."), s(11));
        assert_eq!(out, [Action::Answer { answer: Answer::Option(3), spk: Some(SpkId(1)) }]);
        let out = inv.preview(&preview(12.0, "Pam, option two."), s(13));
        assert_eq!(out, [Action::Answer { answer: Answer::Option(2), spk: Some(SpkId(1)) }]);
        assert!(inv.line(&line(6, (12.0, 13.0), "Pam, option two."), s(14)).is_empty(), "the same answer");
    }

    #[test]
    fn an_answer_that_grows_is_still_only_that_answer() {
        let mut inv = Invocations::new("Pam");
        let out = inv.preview(&preview(10.0, "Pam, option two."), s(11));
        assert_eq!(out, [Action::Answer { answer: Answer::Option(2), spk: Some(SpkId(1)) }]);
        assert!(inv.preview(&preview(10.0, "Pam, option two is fine."), s(12)).is_empty(), "no check");
        assert!(inv.coming.is_empty());
        let out = inv.line(&line(5, (10.0, 12.0), "Pam, option two is fine."), s(13));
        assert!(out.is_empty(), "neither an answer again nor a request: {out:?}");
        assert!(inv.candidates.is_empty() && inv.matched.is_empty());
    }

    #[test]
    fn the_same_option_said_again_answers_again() {
        // A follow-up question can have the same answer as the one before it.
        let mut inv = Invocations::new("Pam");
        let out = inv.line(&line(5, (10.0, 11.0), "Pam, option two."), s(12));
        assert_eq!(out, [Action::Answer { answer: Answer::Option(2), spk: Some(SpkId(1)) }]);
        let out = inv.preview(&preview(13.0, "Pam, option two."), s(14));
        assert_eq!(out, [Action::Answer { answer: Answer::Option(2), spk: Some(SpkId(1)) }]);
        assert!(inv.preview(&preview(13.0, "Pam, option two."), s(14)).is_empty(), "the same preview");
    }

    #[test]
    fn an_utterance_is_checked_once_final_and_its_request_starts_before_its_line() {
        let mut inv = Invocations::new("Pam");
        let out = inv.heard(&[heard(1, &[(3.0, 5.0, "Pam, what did we decide?")])], s(6));
        assert_eq!(checks(&out), [(1, "what did we decide?", None)]);
        assert!(
            inv.heard(&[heard(1, &[(3.0, 5.0, "Pam, what did we decide?")])], s(6)).is_empty(),
            "checked once"
        );
        let out = inv.verdict(1, request("Recall the decision"), s(7));
        assert_eq!(started(&out), [(1, None, "what did we decide?")]);
        let [Action::Request { t0, spk, .. }] = &out[..] else { panic!("{out:?}") };
        assert_eq!((*t0, *spk), (s(3), Some(SpkId(1))), "who asked, from the tracker so far");
        // Its line comes seconds later, and is given to the request.
        let l = line(2, (3.0, 5.0), "Pam, what did we decide?");
        assert_eq!(inv.line(&l, s(9)), [Action::Attach { ask: 1, line: l }]);
        assert!(inv.candidates.is_empty());
    }

    #[test]
    fn unfinished_waits_for_the_next_utterance_and_is_checked_with_it() {
        let mut inv = Invocations::new("Pam");
        let first = (2.2, 3.1, "hey Pam, can you");
        let out = inv.heard(&[heard(1, &[first])], s(4));
        assert_eq!(checks(&out), [(1, "can you", None)]);
        assert!(inv.verdict(1, unfinished(), s(5)).is_empty());
        let out = inv.heard(&[heard(1, &[first, (4.4, 6.0, "uh find the weather")])], s(7));
        assert_eq!(checks(&out), [(2, "can you uh find the weather", None)]);
        let out = inv.verdict(2, request("Find the weather"), s(9));
        assert_eq!(started(&out), [(1, None, "can you uh find the weather")]);
    }

    #[test]
    fn only_the_check_of_the_latest_words_counts() {
        let mut inv = Invocations::new("Pam");
        let first = (1.0, 2.0, "Pam, find the price");
        inv.heard(&[heard(1, &[first])], s(3));
        inv.heard(&[heard(1, &[first, (3.0, 4.0, "of milk")])], s(5));
        assert!(inv.verdict(1, request("Find the price"), s(6)).is_empty(), "words were added since");
        let out = inv.verdict(2, request("Find the price of milk"), s(7));
        assert_eq!(started(&out), [(1, None, "find the price of milk")]);
    }

    #[test]
    fn what_its_speaker_adds_after_it_started_continues_it() {
        let mut inv = Invocations::new("Pam");
        let first = (1.0, 4.0, "hi Pam, can you get me a pricing list for milk");
        inv.heard(&[heard(1, &[first])], s(5));
        assert_eq!(started(&inv.verdict(1, request("Get a milk pricing list"), s(6))).len(), 1);
        let out = inv.heard(&[heard(1, &[first, (5.0, 5.8, "in Auckland")])], s(7));
        assert_eq!(checks(&out), [(2, "can you get me a pricing list for milk in Auckland", None)]);
        let out = inv.verdict(2, request("Get a milk pricing list for Auckland"), s(9));
        let [Action::Continue { ask: 1, text, verdict, .. }] = &out[..] else { panic!("{out:?}") };
        assert_eq!(text, "can you get me a pricing list for milk in Auckland");
        assert_eq!(verdict.summary, "Get a milk pricing list for Auckland");
        // What adds nothing to it changes nothing; a failed check neither.
        inv.heard(&[heard(1, &[first, (5.0, 5.8, "in Auckland"), (6.5, 7.0, "thanks")])], s(8));
        assert!(inv.verdict(3, none(), s(9)).is_empty());
        inv.heard(
            &[heard(1, &[first, (5.0, 5.8, "in Auckland"), (6.5, 7.0, "thanks"), (7.5, 8.0, "Pam")])],
            s(9),
        );
        let l = line(4, (1.0, 8.0), "hi Pam, can you get me a pricing list for milk in Auckland thanks Pam");
        let out = inv.line(&l, s(12));
        assert_eq!(out, [Action::Attach { ask: 1, line: l }]);
    }

    #[test]
    fn what_its_speaker_added_still_continues_it_when_its_line_comes_first() {
        let mut inv = Invocations::new("Pam");
        let first = (1.0, 2.0, "Pam, find the price of milk");
        inv.heard(&[heard(1, &[first])], s(3));
        assert_eq!(started(&inv.verdict(1, request("Find the milk price"), s(4))).len(), 1);
        inv.heard(&[heard(1, &[first, (3.0, 3.5, "in Auckland")])], s(5));
        let l = line(1, (1.0, 3.5), "Pam, find the price of milk in Auckland");
        assert_eq!(inv.line(&l, s(7)), [Action::Attach { ask: 1, line: l }]);
        let out = inv.verdict(2, request("Find the milk price in Auckland"), s(8));
        assert!(matches!(&out[..], [Action::Continue { ask: 1, .. }]), "{out:?}");
        assert!(inv.candidates.is_empty());
    }

    #[test]
    fn words_a_started_request_first_has_in_its_line_continue_it() {
        // "in Auckland" was still a preview when the input ended, so it was
        // never speech on its way: the final line is the first to hold it.
        let mut inv = Invocations::new("Pam");
        inv.heard(&[heard(1, &[(1.0, 2.0, "Pam, find the price of milk")])], s(3));
        assert_eq!(started(&inv.verdict(1, request("Find the milk price"), s(4))).len(), 1);
        let l = line(1, (1.0, 3.5), "Pam, find the price of milk in Auckland");
        let out = inv.line(&l, s(5));
        assert_eq!(out[0], Action::Attach { ask: 1, line: l.clone() });
        assert_eq!(checks(&out), [(2, "find the price of milk in Auckland", Some(l.id))]);
        let out = inv.verdict(2, request("Find the milk price in Auckland"), s(6));
        assert!(matches!(&out[..], [Action::Continue { ask: 1, .. }]), "{out:?}");
        assert!(inv.candidates.is_empty());
    }

    #[test]
    fn a_cancel_or_an_answer_opening_an_utterance_counts_at_once_and_is_no_continuation() {
        let mut inv = Invocations::new("Pam");
        let first = (1.0, 2.0, "Pam, draft an email");
        inv.heard(&[heard(1, &[first])], s(3));
        assert_eq!(started(&inv.verdict(1, request("Draft an email"), s(4))).len(), 1);
        let out = inv.heard(&[heard(1, &[first, (3.0, 3.5, "Pam cancel")])], s(4));
        assert_eq!(out, [Action::Cancel { before: s(3) }], "no check: the cancel isn't part of the request");
        let out =
            inv.heard(&[heard(1, &[first, (3.0, 3.5, "Pam cancel"), (4.0, 4.5, "Pam option two")])], s(5));
        assert_eq!(out, [Action::Answer { answer: Answer::Option(2), spk: Some(SpkId(1)) }]);
        // The final line, which holds all three, is no request or cancel again.
        let l = line(1, (1.0, 4.5), "Pam, draft an email Pam cancel Pam option two");
        assert_eq!(inv.line(&l, s(8)), [Action::Attach { ask: 1, line: l }]);
        assert!(inv.matched.is_empty());
    }

    #[test]
    fn a_cancel_heard_in_a_line_with_no_request_is_only_a_cancel() {
        let mut inv = Invocations::new("Pam");
        let out = inv.heard(&[heard(1, &[(1.0, 2.0, "so yeah"), (3.0, 3.5, "Pam, never mind")])], s(4));
        assert_eq!(out, [Action::Cancel { before: s(3) }]);
        assert!(inv.line(&line(1, (1.0, 3.5), "so yeah Pam, never mind"), s(7)).is_empty(), "not checked");
    }

    #[test]
    fn someone_else_called_pam_triggers_nothing() {
        let mut inv = Invocations::new("Pam");
        let out = inv.heard(&[heard(1, &[(1.0, 3.0, "Pam from finance said it's fine.")])], s(4));
        assert_eq!(checks(&out).len(), 1);
        assert!(inv.verdict(1, none(), s(5)).is_empty());
        let out = inv.line(&line(1, (1.0, 3.0), "Pam from finance said it's fine."), s(7));
        assert!(out.is_empty(), "{out:?}");
        assert!(inv.candidates.is_empty());
    }

    #[test]
    fn the_check_starts_at_the_utterance_holding_the_name() {
        let mut inv = Invocations::new("Pam");
        let out = inv.heard(
            &[heard(
                1,
                &[(1.0, 4.0, "triggered by keyword so"), (5.0, 7.0, "oh maybe hi Pam, can you get me milk")],
            )],
            s(8),
        );
        let [Action::Check { said, text, t0, .. }] = &out[..] else { panic!("{out:?}") };
        assert_eq!(
            (said.as_str(), text.as_str()),
            ("oh maybe hi Pam, can you get me milk", "oh maybe hi Pam, can you get me milk")
        );
        assert_eq!(*t0, s(5));
        // A word that only sounds like the name, opening a later utterance, isn't the gate:
        // "pay" and "pan" are common words.
        assert!(
            inv.heard(&[heard(2, &[(10.0, 11.0, "we should"), (12.0, 13.0, "pan the logs back")])], s(14))
                .is_empty()
        );
        assert_eq!(inv.candidates.len(), 1);
    }

    #[test]
    fn the_name_alone_is_unfinished_without_a_check() {
        let mut inv = Invocations::new("Pam");
        assert!(inv.heard(&[heard(1, &[(1.0, 1.4, "Pam")])], s(2)).is_empty());
        let out = inv.heard(&[heard(1, &[(1.0, 1.4, "Pam"), (2.5, 4.0, "K find the weather")])], s(5));
        assert_eq!(checks(&out), [(1, "K find the weather", None)]);
    }

    #[test]
    fn the_name_alone_on_a_line_of_its_own_goes_with_its_speakers_next_words() {
        let mut inv = Invocations::new("Pam");
        let pam = (1.0, 1.4, "Pam");
        inv.heard(&[heard(1, &[pam])], s(2));
        // 3 s later: a line of its own, still on its way.
        let next = (4.4, 6.0, "can you find the weather");
        let out = inv.heard(&[heard(1, &[pam]), heard(1, &[next])], s(7));
        let [Action::Check { said, t0, .. }] = &out[..] else { panic!("{out:?}") };
        assert_eq!((said.as_str(), *t0), ("Pam can you find the weather", s(1)));
        assert_eq!(
            started(&inv.verdict(1, request("Find the weather"), s(8))),
            [(1, None, "can you find the weather")]
        );
        // "Pam" is final, then the line after it: the request's line is the second.
        assert!(inv.line(&line(1, (1.0, 1.4), "Pam"), s(9)).is_empty());
        let out = inv.line(&line(2, (4.4, 6.0), "can you find the weather"), s(11));
        let [Action::Attach { ask: 1, line }] = &out[..] else { panic!("{out:?}") };
        assert_eq!(
            (line.id, line.t0, line.text.as_str()),
            (UtteranceId(2), s(1), "Pam can you find the weather")
        );
    }

    #[test]
    fn the_name_alone_already_final_goes_with_the_next_words_heard() {
        let mut inv = Invocations::new("Pam");
        assert!(inv.line(&line(1, (1.0, 1.4), "Pam"), s(5)).is_empty());
        let out = inv.heard(&[heard(1, &[(4.4, 6.0, "can you find the weather")])], s(7));
        assert_eq!(checks(&out), [(1, "can you find the weather", None)]);
        // Someone else's words don't.
        let mut inv = Invocations::new("Pam");
        inv.line(&line(1, (1.0, 1.4), "Pam"), s(5));
        assert!(inv.heard(&[heard(2, &[(4.4, 6.0, "can you find the weather")])], s(7)).is_empty());
    }

    #[test]
    fn a_line_that_ends_unfinished_goes_with_its_speakers_next_line() {
        let mut inv = Invocations::new("Pam");
        inv.heard(&[heard(1, &[(1.0, 2.0, "Pam can you")])], s(3));
        inv.verdict(1, unfinished(), s(4));
        assert!(inv.line(&line(1, (1.0, 2.0), "Pam can you"), s(7)).is_empty());
        assert!(inv.candidates.is_empty() && inv.named.len() == 1);
        let out = inv.line(&line(2, (5.0, 6.0), "book a room"), s(9));
        assert_eq!(checks(&out), [(2, "can you book a room", Some(UtteranceId(2)))]);
    }

    #[test]
    fn a_line_with_words_its_speech_lacked_is_checked_again() {
        let mut inv = Invocations::new("Pam");
        inv.heard(&[heard(1, &[(3.0, 5.0, "Pam, what did we decide?")])], s(6));
        let out = inv.line(&line(2, (3.0, 7.0), "Pam, what did we decide? And by when?"), s(8));
        assert_eq!(checks(&out), [(2, "what did we decide? And by when?", Some(UtteranceId(2)))]);
        assert!(inv.verdict(1, request("stale"), s(8)).is_empty(), "the first check is of no use now");
        assert_eq!(
            started(&inv.verdict(2, request("Recall"), s(9))),
            [(1, Some(UtteranceId(2)), "what did we decide? And by when?")]
        );
        assert!(inv.candidates.is_empty());
    }

    #[test]
    fn a_failed_check_of_an_utterance_counts_as_a_request_only_for_the_name_itself() {
        let mut inv = Invocations::new("Pam");
        inv.heard(&[heard(1, &[(1.0, 2.0, "uh Pam, book a room")])], s(3));
        assert_eq!(started(&inv.failed(1, s(4))), [(1, None, "book a room")]);
        inv.heard(&[heard(2, &[(5.0, 6.0, "PEM book a room")])], s(7));
        assert!(inv.failed(2, s(8)).is_empty(), "a word that only sounds like it");
        // A failed check of what was added to a started request changes nothing.
        inv.heard(&[heard(1, &[(1.0, 2.0, "uh Pam, book a room"), (2.5, 3.0, "for four")])], s(9));
        assert!(inv.failed(3, s(10)).is_empty());
    }

    #[test]
    fn a_request_waits_for_one_said_before_it_still_being_checked() {
        let mut inv = Invocations::new("Pam");
        inv.heard(&[heard(1, &[(1.0, 2.0, "Pam, draft the email")])], s(3));
        let out = inv.heard(
            &[
                heard(1, &[(1.0, 2.0, "Pam, draft the email")]),
                heard(2, &[(3.0, 4.0, "Pam, I meant the launch email")]),
            ],
            s(5),
        );
        assert_eq!(checks(&out).len(), 1);
        assert_eq!(inv.asked_before(s(3)), Some("draft the email"), "the correction's context");
        let correction = Verdict { kind: Kind::Correction, summary: "The launch email".into() };
        assert!(inv.verdict(2, correction, s(6)).is_empty(), "it waits for the request it corrects");
        let out = inv.verdict(1, request("Draft the email"), s(7));
        assert_eq!(started(&out), [(1, None, "draft the email"), (2, None, "I meant the launch email")]);
        // Unfinished holds nothing up.
        inv.heard(
            &[heard(1, &[(10.0, 11.0, "Pam can you")]), heard(2, &[(12.0, 13.0, "Pam, book a room")])],
            s(14),
        );
        inv.verdict(3, unfinished(), s(15));
        assert_eq!(started(&inv.verdict(4, request("Book a room"), s(16))), [(4, None, "book a room")]);
    }

    #[test]
    fn a_cancel_takes_back_a_heard_request_still_being_checked() {
        let mut inv = Invocations::new("Pam");
        inv.heard(&[heard(1, &[(1.0, 2.0, "Pam, draft an email")])], s(3));
        assert!(
            inv.heard(&[heard(1, &[(1.0, 2.0, "Pam, draft an email"), (3.0, 3.5, "Pam, cancel")])], s(4))
                .is_empty()
        );
        assert!(inv.holding());
        assert!(inv.verdict(1, request("Draft an email"), s(5)).is_empty(), "taken back");
        assert!(!inv.holding() && inv.candidates.is_empty());
    }

    #[test]
    fn a_cancel_waits_a_while_for_an_unfinished_request_before_it() {
        let mut inv = Invocations::new("Pam");
        inv.heard(&[heard(1, &[(1.0, 2.0, "Pam, can you")])], s(3));
        inv.verdict(1, unfinished(), s(3));
        inv.heard(&[heard(1, &[(1.0, 2.0, "Pam, can you")]), heard(2, &[(3.0, 3.5, "Pam, cancel")])], s(4));
        assert!(inv.tick(s(10)).is_empty() && inv.holding());
        assert_eq!(inv.tick(s(11)), [Action::Cancel { before: s(3) }]);
        assert!(inv.candidates.is_empty(), "it never will be in time");
    }

    #[test]
    fn a_final_line_is_checked_and_becomes_a_request() {
        let mut inv = Invocations::new("Pam");
        assert!(inv.line(&line(1, (0.0, 2.0), "We ship Friday."), s(3)).is_empty());
        let l = line(2, (3.0, 5.0), "Pam, what did we decide?");
        let out = inv.line(&l, s(6));
        assert_eq!(checks(&out), [(1, "what did we decide?", Some(UtteranceId(2)))]);
        let out = inv.verdict(1, request("Recall the decision"), s(8));
        assert_eq!(requests(&out), [(UtteranceId(2), "what did we decide?")]);
        assert!(inv.candidates.is_empty());
    }

    #[test]
    fn a_failed_check_counts_as_a_request() {
        let mut inv = Invocations::new("Pam");
        inv.line(&line(1, (1.0, 3.0), "Pam, book a room."), s(4));
        let out = inv.failed(1, s(5));
        let [Action::Request { verdict, .. }] = &out[..] else { panic!("{out:?}") };
        assert_eq!(verdict, &request("book a room."));
    }

    #[test]
    fn a_misheard_or_later_name_is_checked_and_the_model_decides() {
        let mut inv = Invocations::new("Pam");
        let out = inv.line(&line(1, (1.0, 3.0), "PEM can you find the weather"), s(4));
        assert_eq!(checks(&out), [(1, "can you find the weather", Some(UtteranceId(1)))]);
        let out = inv.verdict(1, request("Find the weather"), s(5));
        assert_eq!(requests(&out), [(UtteranceId(1), "can you find the weather")]);
        // Within: the whole utterance is what is asked.
        let out = inv.line(&line(2, (6.0, 8.0), "what's the weather tomorrow, Pam?"), s(9));
        assert_eq!(checks(&out), [(2, "what's the weather tomorrow, Pam?", Some(UtteranceId(2)))]);
        assert!(inv.verdict(2, none(), s(10)).is_empty());
        assert!(inv.candidates.is_empty());
    }

    #[test]
    fn a_failed_check_is_a_request_only_when_the_name_itself_opened_it() {
        let mut inv = Invocations::new("Pam");
        inv.line(&line(1, (1.0, 3.0), "I told Pam about it."), s(4));
        assert!(inv.failed(1, s(5)).is_empty(), "the name in passing");
        inv.line(&line(2, (6.0, 8.0), "PEM can you book a room"), s(9));
        assert!(inv.failed(2, s(10)).is_empty(), "a word that only sounds like it");
        inv.line(&line(3, (11.0, 13.0), "uh Pam, book a room."), s(14));
        assert_eq!(requests(&inv.failed(3, s(15))), [(UtteranceId(3), "book a room.")]);
        assert!(inv.candidates.is_empty());
    }

    #[test]
    fn the_name_alone_waits_for_its_speakers_next_line() {
        let mut inv = Invocations::new("Pam");
        assert!(inv.line(&line(1, (54.0, 54.5), "Pam"), s(55)).is_empty());
        assert!(inv.tick(s(56)).is_empty());
        // Another speaker in between doesn't take it.
        let other = Line { spk: Some(SpkId(2)), slot: Some(2), ..line(2, (55.0, 55.5), "yes") };
        assert!(inv.line(&other, s(56)).is_empty());
        let out = inv.line(&line(3, (56.0, 58.0), "KF find the weather for tomorrow"), s(59));
        let [Action::Check { id, said, text, t0, line: Some(id3), .. }] = &out[..] else { panic!("{out:?}") };
        assert_eq!(
            (said.as_str(), text.as_str()),
            ("Pam KF find the weather for tomorrow", "KF find the weather for tomorrow")
        );
        assert_eq!((*t0, *id3), (s(54), UtteranceId(3)));
        let out = inv.verdict(*id, request("Find the weather"), s(60));
        assert_eq!(requests(&out), [(UtteranceId(3), "KF find the weather for tomorrow")]);
    }

    #[test]
    fn the_name_alone_from_a_speaker_not_known_yet_joins_the_next_line() {
        let mut inv = Invocations::new("Pam");
        let first = Line { spk: None, slot: None, ..line(1, (0.5, 1.0), "Pam") };
        assert!(inv.line(&first, s(2)).is_empty());
        let out = inv.line(&line(2, (2.0, 4.0), "find the weather"), s(5));
        assert_eq!(checks(&out), [(1, "find the weather", Some(UtteranceId(2)))]);
    }

    #[test]
    fn two_speakers_saying_the_name_alone_each_keep_their_request() {
        let mut inv = Invocations::new("Pam");
        let jo = |id, secs, text| Line { spk: Some(SpkId(2)), slot: Some(2), ..line(id, secs, text) };
        inv.line(&line(1, (10.0, 10.5), "Pam"), s(11));
        inv.line(&jo(2, (11.0, 11.5), "Pam"), s(12));
        let out = inv.line(&line(3, (12.0, 15.0), "find the weather"), s(17));
        assert_eq!(checks(&out), [(1, "find the weather", Some(UtteranceId(3)))]);
        let out = inv.line(&jo(4, (13.0, 16.0), "book a room"), s(18));
        assert_eq!(checks(&out), [(2, "book a room", Some(UtteranceId(4)))]);
        // A cancel waits for a request said before it, whatever is said after.
        inv.line(&line(5, (30.0, 30.5), "Pam"), s(31));
        inv.line(&jo(6, (32.0, 33.0), "Pam, cancel."), s(34));
        inv.line(&jo(7, (34.0, 34.5), "Pam"), s(35));
        assert!(inv.tick(s(36)).is_empty(), "still waiting for the first line after the name");
    }

    #[test]
    fn a_cancel_waits_for_the_line_after_the_name_alone() {
        let mut inv = Invocations::new("Pam");
        inv.line(&line(1, (10.0, 10.5), "Pam"), s(11));
        let other = Line { spk: Some(SpkId(2)), slot: Some(2), ..line(2, (13.0, 14.0), "Pam, cancel.") };
        assert!(inv.line(&other, s(15)).is_empty(), "the first speaker is still asking");
        let out = inv.line(&line(3, (11.0, 16.0), "can you find the weather"), s(18));
        assert!(checks(&out).len() == 1 && requests(&out).is_empty(), "{out:?}");
        assert!(inv.verdict(1, request("Find the weather"), s(19)).is_empty(), "the cancel takes it back");
        // With no line after the name, the cancel goes on once its wait is over.
        inv.line(&line(4, (30.0, 30.5), "Pam"), s(31));
        let other = Line { spk: Some(SpkId(2)), slot: Some(2), ..line(5, (32.0, 33.0), "Pam, cancel.") };
        assert!(inv.line(&other, s(34)).is_empty());
        assert_eq!(inv.tick(s(32) + CANCEL_WAIT), [Action::Cancel { before: s(32) }]);
        assert!(inv.named.is_empty());
    }

    #[test]
    fn the_name_alone_stops_waiting_for_a_line_that_starts_too_late() {
        let mut inv = Invocations::new("Pam");
        assert!(inv.line(&line(1, (10.0, 10.5), "Pam, uh"), s(11)).is_empty(), "fillers are no request");
        assert!(inv.line(&line(2, (16.0, 18.0), "the weather is nice"), s(19)).is_empty());
        assert!(inv.named.is_empty());
    }

    #[test]
    fn the_name_alone_waits_for_a_line_that_becomes_final_late() {
        let mut inv = Invocations::new("Pam");
        inv.line(&line(1, (10.0, 10.5), "Pam"), s(11));
        assert!(inv.tick(s(20)).is_empty());
        let out = inv.line(&line(2, (12.0, 19.0), "can you find the weather for tomorrow morning"), s(25));
        assert_eq!(
            checks(&out),
            [(1, "can you find the weather for tomorrow morning", Some(UtteranceId(2)))]
        );
    }

    #[test]
    fn the_name_alone_then_cancel_is_a_cancel() {
        let mut inv = Invocations::new("Pam");
        inv.line(&line(1, (10.0, 10.5), "Pam."), s(11));
        let out = inv.line(&line(2, (12.0, 12.5), "cancel"), s(13));
        assert_eq!(out, [Action::Cancel { before: s(10) }]);
    }

    #[test]
    fn a_cancel_or_an_answer_may_follow_a_filler_or_a_sound_alike() {
        let mut inv = Invocations::new("Pam");
        assert_eq!(
            inv.line(&line(1, (10.0, 11.0), "uh Pam, cancel."), s(12)),
            [Action::Cancel { before: s(10) }]
        );
        let out = inv.preview(&preview(20.0, "PEM, option two."), s(21));
        assert_eq!(out, [Action::Answer { answer: Answer::Option(2), spk: Some(SpkId(1)) }]);
        // Within, "cancel" is for the model: "we should cancel, Pam" isn't one.
        let out = inv.line(&line(3, (30.0, 31.0), "cancel, Pam"), s(32));
        assert_eq!(checks(&out).len(), 1);
    }

    #[test]
    fn a_previewed_cancel_counts_at_once_and_only_once() {
        let mut inv = Invocations::new("Pam");
        assert!(inv.preview(&preview(10.0, "Pam, cancel"), s(10)).is_empty(), "may still grow");
        let out = inv.preview(&preview(10.0, "Pam, cancel."), s(11));
        assert_eq!(out, [Action::Cancel { before: s(10) }]);
        assert!(inv.line(&line(5, (10.0, 11.0), "Pam, cancel."), s(13)).is_empty(), "the same cancel");
        // One said a while later counts again.
        let out = inv.line(&line(6, (20.0, 21.0), "Pam, never mind"), s(23));
        assert_eq!(out, [Action::Cancel { before: s(20) }]);
    }

    #[test]
    fn a_cancel_takes_back_a_request_still_being_checked() {
        let mut inv = Invocations::new("Pam");
        inv.line(&line(1, (1.0, 3.0), "Pam, draft an email."), s(4));
        // Heard before the check came back: it waits, and holds the request.
        assert!(inv.preview(&preview(4.5, "Pam, cancel."), s(5)).is_empty());
        assert!(inv.holding());
        assert!(inv.verdict(1, request("Draft an email"), s(6)).is_empty(), "taken back");
        assert!(!inv.holding());
        assert!(inv.candidates.is_empty());
    }

    #[test]
    fn a_cancel_waits_for_a_request_still_in_the_preview() {
        let mut inv = Invocations::new("Pam");
        let both = vec![
            PreviewLine { spk: Some(SpkId(1)), text: "Pam, draft an email".into(), t0: s(1) },
            PreviewLine { spk: Some(SpkId(2)), text: "Pam, cancel.".into(), t0: s(4) },
        ];
        assert!(inv.preview(&both, s(5)).is_empty());
        let out = inv.line(&line(1, (1.0, 3.0), "Pam, draft an email"), s(6));
        assert_eq!(checks(&out).len(), 1);
        assert!(inv.verdict(1, request("Draft an email"), s(7)).is_empty(), "taken back");
        assert!(!inv.holding());
    }

    #[test]
    fn a_cancel_with_nothing_undecided_cancels_a_task_and_later_requests_go_on() {
        let mut inv = Invocations::new("Pam");
        inv.line(&line(1, (1.0, 3.0), "Pam, draft an email."), s(4));
        assert_eq!(requests(&inv.verdict(1, request("Draft"), s(5))).len(), 1);
        // A request said after the cancel isn't held by it.
        inv.preview(&preview(9.0, "Pam, book a room."), s(10));
        let out = inv.line(&line(2, (6.0, 7.0), "Pam, cancel."), s(10));
        assert_eq!(out, [Action::Cancel { before: s(6) }]);
        let out = inv.line(&line(3, (9.0, 10.0), "Pam, book a room."), s(11));
        assert_eq!(checks(&out), [(2, "book a room.", Some(UtteranceId(3)))]);
        assert_eq!(requests(&inv.verdict(2, request("Book"), s(12))), [(UtteranceId(3), "book a room.")]);
    }

    #[test]
    fn a_cancel_stops_waiting_after_a_while() {
        let mut inv = Invocations::new("Pam");
        let both = vec![
            PreviewLine { spk: Some(SpkId(1)), text: "Pam, draft an email".into(), t0: s(1) },
            PreviewLine { spk: Some(SpkId(2)), text: "Pam, cancel.".into(), t0: s(3) },
        ];
        assert!(inv.preview(&both, s(4)).is_empty());
        assert!(inv.tick(s(10)).is_empty());
        // The email's speech never came: the cancel goes to the tasks instead.
        assert_eq!(inv.tick(s(11)), [Action::Cancel { before: s(3) }]);
        assert!(inv.candidates.is_empty());
    }

    #[test]
    fn a_cancel_waits_for_a_slow_check_however_long_it_takes() {
        let mut inv = Invocations::new("Pam");
        inv.line(&line(1, (1.0, 3.0), "Pam, draft an email."), s(4));
        inv.preview(&preview(4.0, "Pam, cancel."), s(5));
        assert!(inv.tick(s(20)).is_empty(), "the check is still out");
        assert!(inv.holding());
        assert!(inv.verdict(1, request("Draft an email"), s(21)).is_empty(), "taken back, no task cancelled");
        assert!(!inv.holding());
    }

    #[test]
    fn requests_go_out_in_the_order_they_were_said() {
        let mut inv = Invocations::new("Pam");
        inv.line(&line(1, (1.0, 3.0), "Pam, draft the email."), s(4));
        let out = inv.line(&line(2, (4.0, 6.0), "Pam, I meant the launch email."), s(7));
        assert_eq!(checks(&out).len(), 1);
        assert_eq!(inv.asked_before(s(4)), Some("draft the email."), "the correction's context");
        // The correction's check comes back first: it waits for the request it corrects.
        let correction = Verdict { kind: Kind::Correction, summary: "The launch email".into() };
        assert!(inv.verdict(2, correction, s(8)).is_empty());
        let out = inv.verdict(1, request("Draft the email"), s(9));
        assert_eq!(
            requests(&out),
            [(UtteranceId(1), "draft the email."), (UtteranceId(2), "I meant the launch email.")]
        );
        assert_eq!(inv.asked_before(s(10)), None);
    }

    #[test]
    fn a_plain_reply_while_a_question_is_up_answers_it() {
        let mut inv = Invocations::new("Pam");
        inv.question(Some(asked()));
        let out = inv.line(&line(1, (10.0, 12.0), "Yes, the first one."), s(13));
        assert_eq!(checks(&out), [(1, "Yes, the first one.", Some(UtteranceId(1)))]);
        assert_eq!(shown(&out), [Some(&asked())]);
        assert_eq!(inv.verdict(1, reply(1), s(15)), [answer(1)]);
        assert!(inv.candidates.is_empty());
    }

    #[test]
    fn talk_that_is_not_a_reply_answers_nothing_and_starts_nothing() {
        let mut inv = Invocations::new("Pam");
        inv.question(Some(asked()));
        let out = inv.line(&line(1, (10.0, 12.0), "Let me look at the calendar."), s(13));
        assert_eq!(checks(&out).len(), 1);
        assert!(inv.verdict(1, none(), s(14)).is_empty());
        // Without the name, a request is none of the assistant's business, and neither is a failed check.
        inv.line(&line(2, (14.0, 16.0), "can you book a room for us"), s(17));
        assert!(inv.verdict(2, request("Book a room"), s(18)).is_empty());
        inv.line(&line(3, (18.0, 19.0), "Sure."), s(20));
        assert!(inv.failed(3, s(21)).is_empty());
        assert!(inv.candidates.is_empty());
    }

    #[test]
    fn without_a_question_on_the_stage_talk_is_not_checked() {
        let mut inv = Invocations::new("Pam");
        assert!(inv.line(&line(1, (10.0, 12.0), "Yes, the first one."), s(13)).is_empty());
        inv.question(Some(asked()));
        inv.question(None);
        assert!(inv.heard(&[heard(1, &[(14.0, 15.0, "Yes, the first one.")])], s(15)).is_empty());
        assert!(inv.line(&line(2, (14.0, 15.0), "Yes, the first one."), s(16)).is_empty());
        assert!(!inv.replying(s(20), s(20)));
    }

    #[test]
    fn a_reply_checked_after_its_question_has_gone_answers_nothing() {
        let mut inv = Invocations::new("Pam");
        inv.question(Some(asked()));
        inv.line(&line(1, (10.0, 12.0), "Yes, the first one."), s(13));
        inv.question(None);
        assert!(inv.verdict(1, reply(1), s(14)).is_empty());
        // A newer question isn't answered by a reply to the one before it.
        inv.question(Some(asked()));
        inv.line(&line(2, (15.0, 17.0), "The second one."), s(18));
        inv.question(Some(Asked { text: "Formal or casual?".into(), ..asked() }));
        assert!(inv.verdict(2, reply(2), s(19)).is_empty());
        assert!(inv.candidates.is_empty());
    }

    #[test]
    fn a_reply_heard_on_its_way_answers_at_once_and_its_line_is_that_answer() {
        let mut inv = Invocations::new("Pam");
        inv.question(Some(asked()));
        let first = (10.0, 11.0, "Yes compare.");
        let out = inv.heard(&[heard(1, &[first])], s(11));
        assert_eq!(checks(&out), [(1, "Yes compare.", None)]);
        assert_eq!(shown(&out), [Some(&asked())]);
        assert!(inv.verdict(1, none(), s(12)).is_empty());
        let both = [first, (11.5, 12.0, "Yes, number one.")];
        let out = inv.heard(&[heard(1, &both)], s(12));
        assert_eq!(checks(&out), [(2, "Yes compare. Yes, number one.", None)]);
        assert_eq!(inv.verdict(2, reply(1), s(13)), [answer(1)]);
        let all = [first, (11.5, 12.0, "Yes, number one."), (12.5, 13.0, "Great.")];
        assert!(inv.heard(&[heard(1, &all)], s(13)).is_empty(), "no check: it is that answer");
        let out = inv.line(&line(1, (10.0, 13.0), "Yes compare. Yes, number one. Great."), s(15));
        assert!(out.is_empty(), "neither checked again nor answered again: {out:?}");
        assert!(inv.candidates.is_empty() && inv.answered.is_empty());
    }

    #[test]
    fn a_reply_with_an_option_the_question_lacks_is_no_reply() {
        let mut inv = Invocations::new("Pam");
        inv.question(Some(asked()));
        inv.heard(&[heard(1, &[(10.0, 11.0, "The third one.")])], s(11));
        assert!(inv.verdict(1, reply(3), s(12)).is_empty(), "two options only");
        assert!(inv.answered.is_empty(), "the speech isn't taken");
        // Its line, grown, is checked again and may still answer.
        let out = inv.line(&line(1, (10.0, 13.0), "The third one. No, the second one."), s(14));
        assert_eq!(checks(&out).len(), 1, "{out:?}");
        assert_eq!(inv.verdict(2, reply(2), s(15)), [answer(2)]);
    }

    #[test]
    fn option_three_for_two_options_takes_nothing_so_the_utterance_can_go_on() {
        for checked in [true, false] {
            let mut inv = if checked { Invocations::new("Pam") } else { Invocations::local("Pam") };
            inv.question(Some(asked()));
            assert!(inv.preview(&preview(10.0, "Pam, option three."), s(11)).is_empty(), "no such option");
            let said = [(10.0, 11.0, "Pam, option three."), (11.5, 12.5, "No, option two.")];
            let out = inv.heard(&[heard(1, &said)], s(13));
            if checked {
                assert_eq!(checks(&out), [(1, "option three. No, option two.", None)]);
                assert_eq!(inv.verdict(1, reply(2), s(14)), [answer(2)]);
            } else {
                assert!(out.is_empty(), "too long to match without a model: {out:?}");
            }
            // Said on its own, it is nothing.
            assert!(inv.line(&line(5, (20.0, 21.0), "Pam, option three."), s(22)).is_empty());
            assert_eq!(inv.line(&line(6, (23.0, 24.0), "Pam, option two."), s(25)), [answer(2)]);
        }
    }

    #[test]
    fn a_line_with_the_name_may_reply_too() {
        let mut inv = Invocations::new("Pam");
        inv.question(Some(asked()));
        let out = inv.line(&line(1, (10.0, 12.0), "Pam, Globex please."), s(13));
        assert_eq!(shown(&out), [Some(&asked())]);
        assert_eq!(inv.verdict(1, reply(2), s(14)), [answer(2)]);
        // With no question up, a reply is nothing, and not a request either.
        inv.question(None);
        let out = inv.line(&line(2, (20.0, 22.0), "Pam, the first one sounds good."), s(23));
        assert_eq!(shown(&out), [None]);
        assert!(inv.verdict(2, reply(1), s(24)).is_empty());
        assert!(inv.candidates.is_empty());
    }

    #[test]
    fn the_name_alone_then_an_answer_in_its_speakers_next_words_is_that_answer() {
        for checked in [true, false] {
            let mut inv = if checked { Invocations::new("Pam") } else { Invocations::local("Pam") };
            inv.question(Some(asked()));
            let name = (10.0, 10.5, "Pam.");
            assert!(
                checks(&inv.heard(&[heard(1, &[name])], s(11))).is_empty(),
                "the name alone needs no model"
            );
            let out = inv.heard(&[heard(1, &[name, (11.0, 12.0, "Option two.")])], s(13));
            assert_eq!(out, [answer(2)]);
            assert!(
                inv.line(&line(1, (10.0, 12.0), "Pam. Option two."), s(15)).is_empty(),
                "the same answer"
            );
            assert!(inv.candidates.is_empty() && inv.named.is_empty());
        }
    }

    #[test]
    fn the_countdown_waits_for_a_reply_being_heard_or_checked() {
        let mut inv = Invocations::new("Pam");
        assert!(!inv.replying(s(20), s(20)), "no question");
        inv.question(Some(asked()));
        assert!(!inv.replying(s(20), s(20)), "nothing said");
        inv.preview(&preview(10.0, "yes compare yes"), s(11));
        assert!(inv.replying(s(12), s(12)), "being heard");
        assert!(!inv.replying(s(10), s(12)), "begun after the deadline");
        inv.heard(&[heard(1, &[(10.0, 12.0, "yes compare yes number one")])], s(13));
        inv.preview(&[], s(13));
        assert!(inv.replying(s(12), s(13)), "being checked");
        assert!(inv.verdict(1, none(), s(14)).is_empty());
        assert!(!inv.replying(s(12), s(14)), "it wasn't a reply");
        // The name alone waits for its speaker's next line, which may be the reply,
        // but only as long as that line could still join it.
        inv.line(&line(2, (15.0, 15.5), "Pam"), s(16));
        assert!(inv.replying(s(16), s(17)));
        assert!(inv.replying(s(16), s(20)));
        assert!(!inv.replying(s(16), s(21)), "nobody went on");
    }

    #[test]
    fn a_reply_being_checked_holds_up_no_request_and_no_cancel() {
        let mut inv = Invocations::new("Pam");
        inv.question(Some(asked()));
        inv.line(&line(1, (10.0, 12.0), "yes the first one"), s(13));
        inv.line(&line(2, (13.0, 15.0), "Pam, book a room."), s(16));
        assert_eq!(inv.asked_before(s(13)), None, "talk is no request a correction may be about");
        assert_eq!(
            requests(&inv.verdict(2, request("Book a room"), s(17))),
            [(UtteranceId(2), "book a room.")]
        );
        inv.line(&line(3, (17.0, 18.0), "or the second"), s(19));
        assert_eq!(
            inv.line(&line(4, (19.0, 20.0), "Pam, cancel."), s(21)),
            [Action::Cancel { before: s(19) }]
        );
        assert!(!inv.holding());
        assert_eq!(inv.verdict(1, reply(1), s(22)), [answer(1)]);
    }

    #[test]
    fn without_an_intent_check_only_cancels_and_answers_count() {
        let mut inv = Invocations::local("Pam");
        inv.question(Some(asked()));
        assert!(inv.preview(&preview(10.0, "Yes, the first one."), s(11)).is_empty());
        assert!(inv.heard(&[heard(1, &[(10.0, 11.0, "Yes, the first one.")])], s(11)).is_empty());
        assert!(inv.line(&line(1, (10.0, 11.0), "Yes, the first one."), s(12)).is_empty());
        assert!(!inv.replying(s(12), s(12)), "nothing to wait for");
        assert!(
            inv.heard(&[heard(1, &[(13.0, 15.0, "Pam, book a room.")])], s(15)).is_empty(),
            "not checked"
        );
        assert!(inv.line(&line(2, (13.0, 15.0), "Pam, book a room."), s(16)).is_empty());
        assert!(inv.preview(&preview(17.0, "Pam, book a room."), s(18)).is_empty());
        assert_eq!(inv.line(&line(3, (20.0, 21.0), "Pam, option two."), s(22)), [answer(2)]);
        assert_eq!(
            inv.line(&line(4, (23.0, 24.0), "Pam, cancel."), s(25)),
            [Action::Cancel { before: s(23) }]
        );
        // The name alone, then the command in the speaker's next line.
        assert!(inv.line(&line(5, (27.0, 27.5), "Pam."), s(28)).is_empty());
        assert!(inv.replying(s(29), s(29)), "the countdown waits for what the name's speaker says next");
        assert!(!inv.replying(s(29), s(33)), "nobody went on");
        assert_eq!(inv.line(&line(6, (29.0, 30.0), "Option two."), s(31)), [answer(2)]);
        assert!(inv.line(&line(7, (32.0, 32.5), "Pam."), s(33)).is_empty());
        assert_eq!(inv.line(&line(8, (34.0, 35.0), "Cancel."), s(36)), [Action::Cancel { before: s(32) }]);
        assert!(inv.candidates.is_empty() && inv.coming.is_empty() && inv.named.is_empty());
    }

    #[test]
    fn the_local_grammar_allows_every_kind_the_prompt_names() {
        let prompt = system_prompt("Pam");
        let kinds: Vec<&str> =
            prompt.lines().filter_map(|l| l.strip_prefix("- \"")?.split('"').next()).collect();
        assert_eq!(kinds, ["request", "correction", "reply", "unfinished", "none"]);
        let (none, options, open) = (grammar(None), grammar(Some(&asked())), grammar(Some(&open())));
        for g in [&none, &options, &open] {
            for kind in &kinds {
                assert!(g.contains(&format!("\"\\\"{kind}\\\"\"")), "{kind} isn't in the grammar");
            }
        }
        // The answer the question takes, and only that.
        let (option, text) = ("\"\\\"option\\\"\"", "\"\\\"text\\\"\"");
        assert!(options.contains(option) && !options.contains(text), "{options}");
        assert!(options.contains("[1-9]\n"), "options are numbered 1 to 9: {options}");
        assert!(open.contains(text) && !open.contains(option), "{open}");
        assert!(!none.contains(option) && !none.contains(text) && !none.contains("answer"), "{none}");
    }

    #[test]
    fn reads_the_verdict() {
        assert_eq!(
            parse("```json\n{\"kind\": \"Request\", \"summary\": \" Find X \"}\n```").unwrap(),
            request("Find X")
        );
        assert_eq!(parse("{\"kind\": \"none\"}").unwrap(), none());
        assert_eq!(parse("{\"kind\": \"unfinished\", \"summary\": \"\"}").unwrap(), Verdict::unfinished());
        assert_eq!(parse(r#"{"kind": "reply", "option": 2, "summary": ""}"#).unwrap(), reply(2));
        assert_eq!(parse(r#"{"kind": "reply", "option": 0}"#).unwrap(), none(), "no option chosen");
        assert_eq!(parse(r#"{"kind": "reply", "text": " Auckland "}"#).unwrap(), said("Auckland"));
        assert_eq!(parse(r#"{"kind": "reply", "text": " "}"#).unwrap(), none(), "no words");
        assert_eq!(parse(r#"{"kind": "reply", "summary": "", "text": "x", "option": 1}"#).unwrap(), reply(1));
        assert!(parse("{\"kind\": \"maybe\"}").is_err());
        assert!(parse("no").is_err());
    }

    #[test]
    fn the_prompt_ends_with_the_utterance() {
        let recent = ["u0001 00:01 Alice: We ship Friday.".to_string()];
        let p = check_prompt(&CheckInput {
            recent: &recent,
            latest: Some(("t01".into(), "answering", "what did we decide?")),
            question: None,
            by: "Jordan",
            said: "Pam, I meant the date.",
        });
        assert!(p.starts_with("RECENT LINES\nu0001 00:01 Alice: We ship Friday.\n"), "{p}");
        assert!(p.contains("LATEST REQUEST\nt01 (answering): what did we decide?\n"), "{p}");
        assert!(!p.contains("QUESTION"), "{p}");
        assert!(p.ends_with("UTTERANCE\nJordan: Pam, I meant the date.\n"), "{p}");
        assert!(system_prompt("Jarvis").contains("\"Jarvis, note that...\""));
        assert!(system_prompt("Jarvis").contains("- \"unfinished\": the speaker is talking to Jarvis"));
    }

    #[test]
    fn the_prompt_shows_the_question_on_the_stage_and_its_options() {
        let q = asked();
        let p = check_prompt(&CheckInput {
            recent: &[],
            latest: Some(("t01".into(), "answering", "draft the email")),
            question: Some(&q),
            by: "Jordan",
            said: "yes, the first one",
        });
        assert!(
            p.contains(
                "\nQUESTION ON THE STAGE\nWhich customer is the email for?\n1. Acme\n2. Globex\n\nUTTERANCE\n"
            ),
            "{p}"
        );
        assert!(system_prompt("Pam").contains(r#"{"kind": "reply", "option": 2, "summary": ""}"#));
    }

    #[test]
    fn the_prompt_says_an_open_question_is_answered_in_words() {
        let q = open();
        let p = check_prompt(&CheckInput {
            recent: &[],
            latest: Some(("t01".into(), "answering", "get me the weather for tomorrow morning")),
            question: Some(&q),
            by: "Jordan",
            said: "We're in Auckland.",
        });
        assert!(
            p.contains(
                "\nQUESTION ON THE STAGE\nWhich city are you in?\n(open: answered in words)\n\nUTTERANCE\n"
            ),
            "{p}"
        );
        assert!(system_prompt("Pam").contains(r#"{"kind": "reply", "text": "Auckland", "summary": ""}"#));
    }

    #[test]
    fn an_open_question_is_answered_with_the_words_of_a_reply() {
        let mut inv = Invocations::new("Pam");
        inv.question(Some(open()));
        let out = inv.heard(&[heard(1, &[(10.0, 11.0, "Let me think.")])], s(11));
        assert_eq!(shown(&out), [Some(&open())]);
        assert!(inv.verdict(1, none(), s(12)).is_empty());
        let out = inv
            .heard(&[heard(1, &[(10.0, 11.0, "Let me think."), (11.5, 13.0, "We're in Auckland.")])], s(13));
        assert_eq!(checks(&out), [(2, "Let me think. We're in Auckland.", None)]);
        assert_eq!(inv.verdict(2, said("Auckland"), s(14)), [answered("Auckland")]);
        let out = inv.line(&line(1, (10.0, 14.0), "Let me think. We're in Auckland. Right."), s(16));
        assert!(out.is_empty(), "it is that answer: {out:?}");
    }

    #[test]
    fn an_answer_the_question_does_not_take_is_no_reply() {
        // Words don't answer a question with options, and an option doesn't answer an open one.
        let mut inv = Invocations::new("Pam");
        inv.question(Some(asked()));
        inv.line(&line(1, (10.0, 12.0), "Globex, I think."), s(13));
        assert!(inv.verdict(1, said("Globex"), s(14)).is_empty());
        inv.question(Some(open()));
        inv.line(&line(2, (15.0, 17.0), "The first one."), s(18));
        assert!(inv.verdict(2, reply(1), s(19)).is_empty());
        assert!(inv.candidates.is_empty());
        // Nor does a reply checked against the question before.
        inv.question(Some(asked()));
        inv.line(&line(3, (20.0, 22.0), "Auckland."), s(23));
        inv.question(Some(open()));
        assert!(inv.verdict(3, said("Auckland"), s(24)).is_empty());
    }

    #[test]
    fn option_two_for_an_open_question_goes_to_the_check() {
        let mut inv = Invocations::new("Pam");
        inv.question(Some(open()));
        assert!(inv.preview(&preview(10.0, "Pam, two."), s(11)).is_empty(), "no options to match");
        assert_eq!(inv.coming, [s(10)], "on its way to the check");
        // "How many are coming?" / "Pam, two.": only the check can tell.
        let out = inv.line(&line(1, (10.0, 11.0), "Pam, two."), s(12));
        assert_eq!(checks(&out), [(1, "two.", Some(UtteranceId(1)))]);
        assert_eq!(inv.verdict(1, said("two"), s(13)), [answered("two")]);
        // Without an intent check nothing answers an open question by voice.
        let mut inv = Invocations::local("Pam");
        inv.question(Some(open()));
        assert!(inv.line(&line(1, (10.0, 11.0), "Pam, option two."), s(12)).is_empty());
        assert!(inv.line(&line(2, (12.0, 14.0), "We're in Auckland."), s(15)).is_empty());
    }
}
