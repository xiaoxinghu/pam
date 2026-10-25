use std::time::Duration;

use super::endpoint::{CAP, END_SILENCE, PRE_ROLL};
use super::*;

const S: Ts = SAMPLE_RATE as Ts;

fn at(secs: f64) -> Ts {
    (secs * S as f64).round() as Ts
}

fn word(text: &str, start: f64, end: f64) -> Word {
    Word { text: text.into(), start: at(start), end: at(end), ..Word::default() }
}

/// Words of `each` seconds, one after another with no gap, from `from` to `to`.
fn unbroken(from: f64, to: f64, each: f64) -> Vec<Word> {
    let n = ((to - from) / each).ceil() as usize;
    (0..n)
        .map(|i| word(&format!("w{i}"), from + i as f64 * each, (from + (i + 1) as f64 * each).min(to)))
        .collect()
}

/// Speech at a steady −20 dBFS from `from` to `to`. A stream that starts with
/// it starts a frame in: the endpointer hears the room before it can tell
/// speech from it, and a steady level from the first frame is the room.
fn speech(audio: &mut [f32], from: f64, to: f64) {
    audio[at(from) as usize..at(to) as usize].fill(0.1);
}

fn clips(e: &mut Endpointer, audio: &[f32], push: usize) -> Vec<(Ts, Ts, Cut)> {
    let mut out: Vec<Clip> = audio.chunks(push).flat_map(|c| e.push(c)).collect();
    out.extend(e.finish());
    out.iter().map(|c| (c.start, c.end(), c.cut)).collect()
}

#[test]
fn a_clip_ends_after_800_ms_of_silence_and_starts_with_200_ms_of_pre_roll() {
    let mut audio = vec![0.0; 4 * S as usize];
    speech(&mut audio, 1.0, 2.0);
    assert_eq!(
        clips(&mut Endpointer::new(), &audio, 1_600),
        [(S - PRE_ROLL, 2 * S + END_SILENCE, Cut::Silence)]
    );
}

#[test]
fn the_cap_cuts_speech_that_goes_on_and_the_next_clip_follows_on() {
    let mut audio = vec![0.0; 47 * S as usize];
    speech(&mut audio, 0.02, 45.0);
    let cuts = clips(&mut Endpointer::new(), &audio, 1_600);
    assert_eq!(
        cuts,
        [(0, CAP, Cut::Cap), (CAP, 2 * CAP, Cut::Cap), (2 * CAP, 45 * S + END_SILENCE, Cut::Silence)]
    );
}

#[test]
fn a_silence_that_begins_just_before_the_cap_still_ends_the_utterance() {
    let starts = |audio: &[f32]| {
        let mut e = Endpointer::new();
        let mut out: Vec<Clip> = e.push(audio);
        out.extend(e.finish());
        out.iter().map(|c| (c.start, c.cut, c.starts_utterance)).collect::<Vec<_>>()
    };
    // Speech to 19.5 s, cut at the cap at 20 s; the next speech, at 25 s, is
    // a new utterance.
    let mut audio = vec![0.0; 27 * S as usize];
    speech(&mut audio, 0.02, 19.5);
    speech(&mut audio, 25.0, 26.0);
    assert_eq!(starts(&audio), [(0, Cut::Cap, true), (at(24.8), Cut::Silence, true)]);
    // Back within 800 ms of the last speech: the same utterance goes on.
    let mut audio = vec![0.0; 27 * S as usize];
    speech(&mut audio, 0.02, 19.5);
    speech(&mut audio, 20.2, 26.0);
    assert_eq!(starts(&audio), [(0, Cut::Cap, true), (at(20.0), Cut::Silence, false)]);

    // Through the wrapper: two utterance starts, the second at 25 s.
    let mut script = unbroken(0.02, 19.5, 0.5);
    script.extend(unbroken(25.0, 26.0, 0.5).into_iter().map(|w| Word { text: format!("x{}", w.text), ..w }));
    let audio = FakeClip::audio(&script, 27 * S);
    let finals = replay(&mut wrap(FakeClip::new(&script), "en-US"), &audio, false).finals();
    let starts: Vec<&str> = finals.iter().filter(|w| w.starts_utterance).map(|w| w.text.as_str()).collect();
    assert_eq!(starts, ["w0", "xw0"]);
}

/// Noise at `dbfs` RMS from `from` to `to`, in place of what is there: the
/// room, or speech louder than it. The same every run.
fn noise(audio: &mut [f32], from: f64, to: f64, dbfs: f64) {
    // Uniform in [−a, a] has an RMS of a / √3.
    let a = 10f64.powf(dbfs / 20.0) * 3f64.sqrt();
    let mut x: u32 = 0x9E37_79B9;
    for s in &mut audio[at(from) as usize..at(to) as usize] {
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        *s = ((f64::from(x) / f64::from(u32::MAX) * 2.0 - 1.0) * a) as f32;
    }
}

#[test]
fn a_clip_ends_after_800_ms_of_room_noise() {
    // A room at −40 dBFS is never speech.
    let mut audio = vec![0.0; 10 * S as usize];
    noise(&mut audio, 0.0, 10.0, -40.0);
    assert_eq!(clips(&mut Endpointer::new(), &audio, 1_600), []);
    // Speech at −20 dBFS over it ends 800 ms after it stops.
    noise(&mut audio, 2.0, 3.0, -20.0);
    assert_eq!(clips(&mut Endpointer::new(), &audio, 1_600), [(at(1.8), at(3.8), Cut::Silence)]);
}

#[test]
fn quiet_speech_in_a_quiet_room_still_counts() {
    // At −55 dBFS in a room at −80: over the fixed −60 dBFS, which the room
    // is too quiet to raise.
    let mut audio = vec![0.0; 4 * S as usize];
    noise(&mut audio, 0.0, 4.0, -80.0);
    noise(&mut audio, 1.0, 2.0, -55.0);
    assert_eq!(clips(&mut Endpointer::new(), &audio, 1_600), [(at(0.8), at(2.8), Cut::Silence)]);
    // 15 dB over a room at −60.
    noise(&mut audio, 0.0, 4.0, -60.0);
    noise(&mut audio, 1.0, 2.0, -45.0);
    assert_eq!(clips(&mut Endpointer::new(), &audio, 1_600), [(at(0.8), at(2.8), Cut::Silence)]);
}

#[test]
fn a_room_that_gets_louder_stops_counting_as_speech() {
    // The room rises 15 dB, from −55 to −40 dBFS: it is speech at first, and
    // stops being speech once the floor has risen 5 dB, in about 10 s.
    let mut audio = vec![0.0; 30 * S as usize];
    noise(&mut audio, 0.0, 2.0, -55.0);
    noise(&mut audio, 2.0, 30.0, -40.0);
    let cuts = clips(&mut Endpointer::new(), &audio, 1_600);
    assert!(
        matches!(cuts[..], [(start, end, Cut::Silence)] if start == at(1.8) && end < at(14.0)),
        "{cuts:?}"
    );
}

#[test]
fn pre_roll_never_reaches_back_into_the_last_clip() {
    let mut audio = vec![0.0; 5 * S as usize];
    speech(&mut audio, 1.0, 2.0);
    speech(&mut audio, 2.9, 3.5);
    let cuts = clips(&mut Endpointer::new(), &audio, 1_600);
    assert_eq!(cuts, [(at(0.8), at(2.8), Cut::Silence), (at(2.8), at(4.3), Cut::Silence)]);
}

#[test]
fn push_sizes_dont_move_the_cuts_and_finish_closes_the_open_clip() {
    let mut audio = vec![0.0; at(30.01) as usize];
    speech(&mut audio, 0.5, 1.5);
    speech(&mut audio, 3.0, 30.01);
    let whole = clips(&mut Endpointer::new(), &audio, audio.len());
    assert_eq!(
        whole,
        [(at(0.3), at(2.3), Cut::Silence), (at(2.8), at(22.8), Cut::Cap), (at(22.8), at(30.01), Cut::End)]
    );
    for push in [1, 333, 1_600, 16_000] {
        assert_eq!(clips(&mut Endpointer::new(), &audio, push), whole, "{push}-sample pushes");
    }
    let mut e = Endpointer::new();
    e.push(&audio[..at(10.0) as usize]);
    assert_eq!(e.open().map(|(start, a)| (start, a.len() as Ts)), Some((at(2.8), at(7.2))));
}

/// Everything a transcriber gave: each push's update, then what `finish` did.
struct Replay {
    updates: Vec<Update<Word>>,
    flush: Vec<Word>,
}

impl Replay {
    fn finals(&self) -> Vec<Word> {
        self.updates.iter().flat_map(|u| u.finalized.iter()).chain(&self.flush).cloned().collect()
    }
}

/// Pushes `audio` 100 ms at a time; with `settle`, each preview started is
/// done before the next push.
fn replay(t: &mut ClipTranscriber, audio: &[f32], settle: bool) -> Replay {
    let mut updates = Vec::new();
    for chunk in audio.chunks(1_600) {
        updates.push(t.push(chunk).unwrap());
        if settle {
            t.settle();
        }
    }
    let flush = t.finish().unwrap();
    Replay { updates, flush }
}

/// The fake with its fake aligner, for `language` (`"auto"` to detect).
fn wrap(fake: FakeClip, language: &str) -> ClipTranscriber {
    ClipTranscriber::new(fake.aligned(), lang(language), PREVIEW_EVERY)
}

fn lang(language: &str) -> Option<&str> {
    Some(language).filter(|l| *l != "auto")
}

/// 45 s of unbroken speech from a frame in, in words of 0.7 s so the cap
/// cuts one in two, then 2 s of silence.
fn forty_five_seconds() -> (Vec<Word>, Vec<f32>) {
    let script = unbroken(0.02, 45.0, 0.7);
    let audio = FakeClip::audio(&script, 47 * S);
    (script, audio)
}

#[test]
fn forty_five_seconds_of_speech_give_three_clips_one_utterance_and_bounded_previews() {
    let (script, audio) = forty_five_seconds();
    let fake = FakeClip::new(&script);
    let calls = fake.calls();
    let mut t = wrap(fake, "en-US");
    let r = replay(&mut t, &audio, true);

    let calls = calls.lock().unwrap().clone();
    let lengths: Vec<Ts> = calls.aligned.iter().map(|(len, _)| *len as Ts).collect();
    assert_eq!(lengths, [CAP, CAP, 5 * S + END_SILENCE], "three clips, each aligned once");
    let finals = r.finals();
    assert_eq!(finals.iter().filter(|w| w.starts_utterance).count(), 1);
    assert!(finals[0].starts_utterance);
    // Every word is final: the two the cap cut in two, in both halves.
    let texts: Vec<&str> = finals.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(texts.len(), script.len() + 2);
    assert_eq!(
        finals.iter().filter(|w| w.text == "w28").map(|w| (w.start, w.end)).collect::<Vec<_>>(),
        [(at(19.62), CAP), (CAP, at(20.32))]
    );

    // A preview a second, of the open clip only: never longer than the cap,
    // never audio already final.
    let previews = calls.transcribed.len() - calls.aligned.len();
    assert!((40..=46).contains(&previews), "{previews} previews");
    assert!(calls.transcribed.iter().all(|len| *len as Ts <= CAP), "{:?}", calls.transcribed);
    let mut committed = 0;
    let mut shown = 0;
    for u in &r.updates {
        committed = u.finalized.iter().map(|w| w.end).max().unwrap_or(committed);
        for w in &u.preview {
            assert!(
                w.start >= committed && w.end > committed,
                "{w:?} previewed with audio final to {committed}"
            );
        }
        shown += usize::from(!u.preview.is_empty());
    }
    assert!(shown > 400, "a preview showed after {shown} pushes of 470");
}

#[test]
fn without_word_times_each_word_gets_its_own_clips_span() {
    let (script, audio) = forty_five_seconds();
    let mut t = wrap(FakeClip::new(&script).without_word_times(), "en-US");
    let r = replay(&mut t, &audio, true);
    let spans = [(0, CAP), (CAP, 2 * CAP), (2 * CAP, 45 * S + END_SILENCE)];
    let finals = r.finals();
    for w in &finals {
        let i: usize = w.text[1..].parse().unwrap();
        let heard = &script[i];
        // A clip's span, and a clip it was heard in: a word the cap cut in two
        // is in both, with each one's span.
        assert!(spans.contains(&(w.start, w.end)) && heard.start < w.end && heard.end > w.start, "{w:?}");
    }
    let mut seen: Vec<(Ts, Ts)> = finals.iter().map(|w| (w.start, w.end)).collect();
    seen.dedup();
    assert_eq!(seen, spans);
    for u in &r.updates {
        for w in &u.preview {
            assert!(spans.iter().any(|(a, b)| w.start == *a && w.end <= *b), "{w:?}");
        }
    }
}

#[test]
fn a_slow_model_skips_previews_rather_than_queue_them() {
    let script = unbroken(0.5, 10.5, 0.5);
    let audio = FakeClip::audio(&script, 12 * S);
    let fake = FakeClip::new(&script).slow(Duration::from_millis(300));
    let calls = fake.calls();
    let mut t = wrap(fake, "en-US");
    let r = replay(&mut t, &audio, false);
    let calls = calls.lock().unwrap().clone();
    // Ten previews were due; the first was still running for all the others.
    let previews = calls.transcribed.len() - calls.aligned.len();
    assert!((1..=2).contains(&previews), "{previews} previews ran");
    // Nothing was dropped: the final waited for the preview.
    assert_eq!(
        r.finals().iter().map(|w| w.text.as_str()).collect::<Vec<_>>(),
        script.iter().map(|w| w.text.as_str()).collect::<Vec<_>>()
    );
}

#[test]
fn the_aligner_is_called_on_finals_only_and_a_model_without_one_keeps_its_words() {
    let script = vec![word("hello", 0.5, 1.0), word("there", 1.0, 1.4), word("again", 3.0, 3.5)];
    let audio = FakeClip::audio(&script, 5 * S);
    let fake = FakeClip::new(&script);
    let calls = fake.calls();
    let mut t = wrap(fake, "en-US");
    replay(&mut t, &audio, true);
    let calls = calls.lock().unwrap().clone();
    assert_eq!(calls.aligned.len(), 2);
    let aligned: Vec<&str> =
        calls.aligned.iter().flat_map(|(_, ws)| ws.iter().map(|w| w.text.as_str())).collect();
    assert_eq!(aligned, ["hello", "there", "again"]);
    assert!(calls.transcribed.len() > calls.aligned.len(), "previews ran, unaligned");

    // A model with no aligner: the finals are its words, moved to the stream's clock.
    let plain = FakeClip::new(&script);
    let calls = plain.calls();
    let mut t = ClipTranscriber::new(plain.into(), Some("en-US"), 0);
    let finals = replay(&mut t, &audio, false).finals();
    let times: Vec<(Ts, Ts)> = finals.iter().map(|w| (w.start, w.end)).collect();
    assert_eq!(times, script.iter().map(|w| (w.start, w.end)).collect::<Vec<_>>());
    let starts: Vec<bool> = finals.iter().map(|w| w.starts_utterance).collect();
    assert_eq!(starts, [true, false, true], "the pause before `again` ends an utterance");
    assert!(calls.lock().unwrap().aligned.is_empty(), "nothing to align with");
}

#[test]
fn words_without_a_language_get_the_configured_one_else_their_scripts() {
    let mut script = vec![word("Pam", 0.5, 0.9), word("没问题", 1.0, 1.5), word("hola", 1.6, 2.0)];
    script[2].lang = Some("es-ES".into());
    let audio = FakeClip::audio(&script, 4 * S);
    for (language, expected) in [
        ("en-US", [Some("en-US"), Some("zh-CN"), Some("es-ES")]),
        ("auto", [None, Some("zh-CN"), Some("es-ES")]),
    ] {
        let mut t = wrap(FakeClip::new(&script), language);
        let r = replay(&mut t, &audio, true);
        let langs: Vec<Option<String>> = r.finals().iter().map(|w| w.lang.clone()).collect();
        assert_eq!(langs, expected.map(|l| l.map(String::from)), "{language}");
        let preview =
            r.updates.iter().map(|u| &u.preview).find(|p| p.len() == 3).expect("a preview of all three");
        assert_eq!(preview.iter().map(|w| w.lang.clone()).collect::<Vec<_>>(), langs, "{language} preview");
    }
}

#[test]
fn a_final_with_no_words_keeps_its_clips_last_preview_with_words() {
    // One clip, 0.3 s to 3.8 s; previews of 1, 2 and 3 s of it.
    let script = unbroken(0.5, 3.0, 0.5);
    let audio = FakeClip::audio(&script, 6 * S);
    let texts = |words: &[Word]| words.iter().map(|w| w.text.clone()).collect::<Vec<_>>();

    // Every preview heard the words and the final none: the last preview's
    // words are final, aligned on the 3 s it heard, and a warning says so.
    let fake = FakeClip::new(&script).deaf_from(at(3.2));
    let calls = fake.calls();
    let mut t = wrap(fake, "en-US");
    let r = replay(&mut t, &audio, true);
    let finals = r.finals();
    assert_eq!(texts(&finals), texts(&script));
    assert_eq!(finals.iter().map(|w| (w.start, w.end)).collect::<Vec<_>>(), spans(&script));
    assert!(finals[0].starts_utterance);
    let calls = calls.lock().unwrap().clone();
    assert_eq!(calls.aligned.iter().map(|(len, _)| *len as Ts).collect::<Vec<_>>(), [3 * S]);
    let warnings = t.warnings();
    assert_eq!(warnings.len(), 1, "{warnings:?}");
    assert!(warnings[0].contains("0.3 s to 3.8 s came back with no words"), "{}", warnings[0]);
    assert!(t.warnings().is_empty(), "taken once");

    // The last preview heard nothing too: the one before it shows until the
    // clip closes, and its words are final.
    let fake = FakeClip::new(&script).deaf_from(at(2.5));
    let calls = fake.calls();
    let mut t = wrap(fake, "en-US");
    let r = replay(&mut t, &audio, true);
    let first = r.updates.iter().position(|u| !u.preview.is_empty()).unwrap();
    let close = r.updates.iter().position(|u| !u.finalized.is_empty()).unwrap();
    assert!(r.updates[first..close].iter().all(|u| !u.preview.is_empty()), "a preview went blank");
    assert_eq!(texts(&r.finals()), ["w0", "w1", "w2", "w3"], "heard to 2.3 s");
    assert_eq!(r.finals().last().map(|w| w.end), Some(at(2.3)));
    assert_eq!(calls.lock().unwrap().aligned[0].0 as Ts, 2 * S);
    assert_eq!(t.warnings().len(), 1);

    // A model that heard nothing in the clip at all: no words, no warning.
    let mut t = wrap(FakeClip::new(&script).deaf_from(0), "en-US");
    assert!(replay(&mut t, &audio, true).finals().is_empty());
    assert!(t.warnings().is_empty());
    // Nor when the final has its words.
    let mut t = wrap(FakeClip::new(&script), "en-US");
    assert_eq!(texts(&replay(&mut t, &audio, true).finals()), texts(&script));
    assert!(t.warnings().is_empty());
}

fn spans(words: &[Word]) -> Vec<(Ts, Ts)> {
    words.iter().map(|w| (w.start, w.end)).collect()
}

#[test]
fn finish_flushes_the_open_clip_once() {
    let script = unbroken(0.2, 3.0, 0.4);
    let audio = FakeClip::audio(&script, 3 * S);
    let mut t = wrap(FakeClip::new(&script), "en-US");
    let mut finals = Vec::new();
    for chunk in audio.chunks(1_600) {
        finals.extend(t.push(chunk).unwrap().finalized);
    }
    assert!(finals.is_empty());
    let flush = t.finish().unwrap();
    assert_eq!(flush.len(), script.len());
    assert_eq!(flush.last().map(|w| w.end), Some(3 * S));
    assert!(t.finish().unwrap().is_empty());
    assert!(t.push(&[0.0; 160]).is_err());
}
