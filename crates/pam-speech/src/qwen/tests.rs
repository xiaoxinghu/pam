use std::sync::{Arc, Mutex};
use std::time::Duration;

use anyhow::Context;

use super::*;
use crate::Transcriber;
use crate::clip::{Aligner as _, ClipModel, ClipTranscriber, FakeClip, PREVIEW_EVERY};

const S: Ts = SAMPLE_RATE as Ts;

fn texts(s: &str) -> Vec<(String, String)> {
    pieces(s).into_iter().map(|p| (p.text, p.unit)).collect()
}

fn pairs(v: &[(&str, &str)]) -> Vec<(String, String)> {
    v.iter().map(|(a, b)| (a.to_string(), b.to_string())).collect()
}

#[test]
fn english_keeps_punctuation_on_the_word() {
    assert_eq!(
        texts("Hello, we'll go — twenty-five U.S. minutes."),
        pairs(&[
            ("Hello,", "Hello"),
            ("we'll", "we'll"),
            ("go —", "go"),
            ("twenty-five", "twentyfive"),
            ("U.S.", "US"),
            ("minutes.", "minutes"),
        ])
    );
}

#[test]
fn han_characters_one_by_one_with_the_punctuation_after_them() {
    assert_eq!(
        texts("大家好，我是王磊。"),
        pairs(&[
            ("大", "大"),
            ("家", "家"),
            ("好，", "好"),
            ("我", "我"),
            ("是", "是"),
            ("王", "王"),
            ("磊。", "磊")
        ])
    );
    assert_eq!(
        texts("这个API的latency太高了"),
        pairs(&[
            ("这", "这"),
            ("个", "个"),
            ("API", "API"),
            ("的", "的"),
            ("latency", "latency"),
            ("太", "太"),
            ("高", "高"),
            ("了", "了"),
        ])
    );
    assert_eq!(texts("测试。Great."), pairs(&[("测", "测"), ("试。", "试"), ("Great.", "Great")]));
    assert_eq!(texts("2026年 3.5"), pairs(&[("2026", "2026"), ("年", "年"), ("3.5", "35")]));
}

#[test]
fn punctuation_with_nothing_to_align_joins_a_neighbour() {
    assert_eq!(texts("“好” 的"), pairs(&[("“好”", "好"), ("的", "的")]));
    assert_eq!(texts("“ 好"), pairs(&[("“ 好", "好")]));
    assert!(pieces("。。 ...").is_empty());
}

#[test]
fn the_pieces_give_back_the_text_but_for_whitespace() {
    for s in ["这个API的latency太高了。", "Hello, world!", "“好” 的", "a — b", "OK，那就周五吧。Friday then."]
    {
        let squash = |t: &str| t.chars().filter(|c| !c.is_whitespace()).collect::<String>();
        let joined: String = pieces(s).iter().map(|p| p.text.as_str()).collect();
        assert_eq!(squash(&joined), squash(s), "{s}");
    }
}

#[test]
fn qwens_language_names_become_tags_and_configured_tags_become_names() {
    assert_eq!(parse("language Chinese<asr_text>你好。"), (Some("zh-CN"), "你好。"));
    assert_eq!(parse("language English<asr_text> Hi there."), (Some("en"), "Hi there."));
    assert_eq!(parse("language None<asr_text>"), (None, ""));
    assert_eq!(parse("language Klingon<asr_text>Qapla'"), (None, "Qapla'"));
    assert_eq!(parse("no tag at all"), (None, "no tag at all"));
    assert_eq!(qwen_language(None).unwrap(), None);
    assert_eq!(qwen_language(Some("en-US")).unwrap(), Some("English"));
    assert_eq!(qwen_language(Some("zh-CN")).unwrap(), Some("Chinese"));
    assert_eq!(qwen_language(Some("yue")).unwrap(), Some("Cantonese"));
    let err = qwen_language(Some("tlh")).unwrap_err().to_string();
    assert!(err.contains("`tlh` isn't one Qwen3-ASR transcribes"), "{err}");
}

/// What the fake runtimes were asked.
#[derive(Debug, Default)]
struct Asked {
    /// Each `generate`: the clip's length, the context, the language and the token limit.
    generated: Vec<(usize, String, Option<String>, usize)>,
    /// Each `time`: the clip's length and the units.
    timed: Vec<(usize, Vec<String>)>,
}

/// Qwen3-ASR as a fake: it hears the words a script wrote into the audio
/// (`FakeClip::audio`), and says them in `language`. Unless the prompt
/// forces a language, it names it, as Qwen does. From `none_from` samples
/// on, it names none and hears nothing, as Qwen did once room noise followed
/// quiet speech long enough.
struct FakeAsr {
    script: Vec<Word>,
    language: &'static str,
    slow: Option<Duration>,
    none_from: Option<usize>,
    asked: Arc<Mutex<Asked>>,
}

impl Asr for FakeAsr {
    fn generate(&mut self, audio: &[f32], context: &str, forced: Option<&str>, max: usize) -> Result<String> {
        self.asked.lock().unwrap().generated.push((
            audio.len(),
            context.into(),
            forced.map(String::from),
            max,
        ));
        if let Some(delay) = self.slow {
            std::thread::sleep(delay);
        }
        if self.none_from.is_some_and(|len| audio.len() >= len) {
            return Ok("language None<asr_text>".into());
        }
        let mut heard = FakeClip::new(&self.script);
        let words: Vec<String> = heard.transcribe(audio)?.into_iter().map(|w| w.text).collect();
        let text = words.join(" ");
        Ok(if forced.is_some() { text } else { format!("language {}<asr_text>{text}", self.language) })
    }
}

/// Qwen3-ForcedAligner as a fake: each unit's time is where the script's
/// word with that text is in the audio, found by its level; `fail` makes it fail.
struct FakeTimer {
    script: Vec<Word>,
    fail: bool,
    asked: Arc<Mutex<Asked>>,
}

impl Timer for FakeTimer {
    fn time(&mut self, audio: &[f32], units: &[String]) -> Result<Vec<(f32, f32)>> {
        self.asked.lock().unwrap().timed.push((audio.len(), units.to_vec()));
        if self.fail {
            bail!("the fake aligner fails");
        }
        let mut heard = FakeClip::new(&self.script);
        let words = heard.transcribe(audio)?;
        let secs = |t: Ts| t as f32 / SAMPLE_RATE as f32;
        units
            .iter()
            .map(|u| {
                let w =
                    words.iter().find(|w| clean(&w.text) == *u).context("a unit the audio doesn't hold")?;
                Ok((secs(w.start), secs(w.end)))
            })
            .collect()
    }
}

fn word(text: &str, start: f64, end: f64) -> Word {
    Word {
        text: text.into(),
        start: (start * S as f64) as Ts,
        end: (end * S as f64) as Ts,
        ..Word::default()
    }
}

/// `n` words of 0.4 s with 0.1 s between them, from `from` seconds on.
fn talk(prefix: &str, n: usize, from: f64) -> Vec<Word> {
    (0..n)
        .map(|i| word(&format!("{prefix}{i}"), from + i as f64 * 0.5, from + i as f64 * 0.5 + 0.4))
        .collect()
}

struct Setup {
    language: Option<&'static str>,
    hints: Vec<String>,
    slow: Option<Duration>,
    none_from: Option<usize>,
    fail: bool,
}

impl Default for Setup {
    fn default() -> Setup {
        Setup { language: None, hints: vec!["Pam".into()], slow: None, none_from: None, fail: false }
    }
}

/// The adapter over fake runtimes in the shared wrapper, as `open` builds it.
fn adapter(script: &[Word], setup: Setup, preview_every: Ts) -> (ClipTranscriber, Arc<Mutex<Asked>>) {
    let asked = Arc::new(Mutex::new(Asked::default()));
    let asr = FakeAsr {
        script: script.to_vec(),
        language: "English",
        slow: setup.slow,
        none_from: setup.none_from,
        asked: asked.clone(),
    };
    let timer = FakeTimer { script: script.to_vec(), fail: setup.fail, asked: asked.clone() };
    let model = ClipModel::aligned(Recognizer::new(asr, setup.language, &setup.hints), Aligner::new(timer));
    let configured = setup.language.map(|_| "en-US");
    (ClipTranscriber::new(model, configured, preview_every), asked)
}

/// Pushes `audio` in pushes of `push` samples, then finishes: the final
/// words in order, and the previews seen.
fn replay(t: &mut ClipTranscriber, audio: &[f32], push: usize) -> (Vec<Word>, Vec<Vec<Word>>) {
    let (mut finals, mut previews) = (Vec::new(), Vec::new());
    for chunk in audio.chunks(push) {
        let u = t.push(chunk).unwrap();
        finals.extend(u.finalized);
        if !u.preview.is_empty() {
            previews.push(u.preview);
        }
    }
    finals.extend(t.finish().unwrap());
    (finals, previews)
}

#[test]
fn finals_get_the_aligners_times() {
    let script = talk("w", 6, 0.5);
    let audio = FakeClip::audio(&script, 6 * S);
    let (mut t, asked) = adapter(&script, Setup::default(), S);
    let (finals, previews) = replay(&mut t, &audio, 1_600);
    let texts: Vec<&str> = finals.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(texts, ["w0", "w1", "w2", "w3", "w4", "w5"]);
    for (got, want) in finals.iter().zip(&script) {
        assert_eq!((got.start, got.end), (want.start, want.end), "{}", got.text);
        assert_eq!(got.lang.as_deref(), Some("en"));
    }
    assert!(finals[0].starts_utterance && finals[1..].iter().all(|w| !w.starts_utterance));
    // Whether a preview finishes before its clip closes is up to the threads;
    // one that did has the clip's span, unaligned (the recognizer's own).
    for p in &previews {
        assert!(p.iter().all(|w| (w.start, w.end) == (p[0].start, p[0].end)), "{p:?}");
    }
    let asked = asked.lock().unwrap();
    assert_eq!(asked.timed.len(), 1, "one clip, aligned once, previews never");
    assert_eq!(asked.timed[0].1, ["w0", "w1", "w2", "w3", "w4", "w5"]);
    assert!(asked.generated.iter().all(|g| g.1 == "Pam" && g.2.is_none()), "hints are the context");
}

#[test]
fn a_configured_language_is_forced_and_the_wrapper_tags_its_words() {
    let script = talk("w", 3, 0.5);
    let audio = FakeClip::audio(&script, 3 * S);
    let setup =
        Setup { language: Some("English"), hints: vec!["Sam".into(), "Pam".into()], ..Setup::default() };
    let (mut t, asked) = adapter(&script, setup, 0);
    let (finals, _) = replay(&mut t, &audio, 1_600);
    assert!(finals.iter().all(|w| w.lang.as_deref() == Some("en-US")), "{finals:?}");
    let asked = asked.lock().unwrap();
    assert_eq!(asked.generated.len(), 1);
    let (len, context, language, max) = &asked.generated[0];
    assert_eq!((context.as_str(), language.as_deref()), ("Sam, Pam", Some("English")));
    // A clip of 2.4 s, from its pre-roll to 0.8 s after the speech: 3 s of tokens.
    assert_eq!(*max, BASE_TOKENS + 3 * TOKENS_PER_SECOND, "clip of {len} samples");
}

#[test]
fn a_mixed_clip_tags_each_word_by_its_language_and_script() {
    struct Mixed;
    impl Asr for Mixed {
        fn generate(&mut self, _: &[f32], _: &str, _: Option<&str>, _: usize) -> Result<String> {
            Ok("language Chinese<asr_text>OK，这个API的latency太高了。".into())
        }
    }
    let words = Recognizer::new(Mixed, None, &[]).transcribe(&vec![0.0; 2 * S as usize]).unwrap();
    let got: Vec<(&str, Option<&str>)> = words.iter().map(|w| (w.text.as_str(), w.lang.as_deref())).collect();
    assert_eq!(
        got,
        [
            ("OK，", None),
            ("这", Some("zh-CN")),
            ("个", Some("zh-CN")),
            ("API", None),
            ("的", Some("zh-CN")),
            ("latency", None),
            ("太", Some("zh-CN")),
            ("高", Some("zh-CN")),
            ("了。", Some("zh-CN")),
        ],
        "a word in letters in a Chinese clip may be English or a name: unknown"
    );
    assert!(words.iter().all(|w| (w.start, w.end) == (0, 2 * S)));
}

#[test]
fn a_cantonese_clip_keeps_its_tag_on_han_and_none_on_letters() {
    struct Cantonese;
    impl Asr for Cantonese {
        fn generate(&mut self, _: &[f32], _: &str, _: Option<&str>, _: usize) -> Result<String> {
            Ok("language Cantonese<asr_text>我哋聽日去飲茶，OK？".into())
        }
    }
    let words = Recognizer::new(Cantonese, None, &[]).transcribe(&vec![0.0; S as usize]).unwrap();
    let got: Vec<(&str, Option<&str>)> = words.iter().map(|w| (w.text.as_str(), w.lang.as_deref())).collect();
    let han: Vec<(&str, Option<&str>)> =
        ["我", "哋", "聽", "日", "去", "飲", "茶，"].iter().map(|t| (*t, Some("yue"))).collect();
    assert_eq!(got[..7], han[..]);
    assert_eq!(got[7], ("OK？", None));
}

#[test]
fn a_cut_at_the_cap_continues_the_utterance_and_a_pause_starts_one() {
    // 45 s of talk, a 1.5 s pause, then 3 s more: clips cut at 20 s and 40 s
    // only bound the compute, so one utterance runs to the pause.
    let mut script = talk("a", 90, 0.5);
    script.extend(talk("b", 6, 47.0));
    let audio = FakeClip::audio(&script, 52 * S);
    let (mut t, asked) = adapter(&script, Setup::default(), 0);
    let (finals, _) = replay(&mut t, &audio, 1_600);
    let starts: Vec<&str> = finals.iter().filter(|w| w.starts_utterance).map(|w| w.text.as_str()).collect();
    assert_eq!(starts, ["a0", "b0"]);
    let timed = &asked.lock().unwrap().timed;
    assert_eq!(timed.len(), 4, "three clips of talk, each aligned, and the clip after the pause");
    assert!(timed.iter().all(|(len, _)| *len as Ts <= crate::clip::endpoint::CAP));
    // Every word is final, in order: nothing is dropped at a cut. A word cut
    // in two at the cap is heard in both clips, as a model may hear it.
    let heard: Vec<&str> = finals.iter().map(|w| w.text.as_str()).collect();
    let said: Vec<&str> = script.iter().map(|w| w.text.as_str()).collect();
    let mut once = heard.clone();
    once.dedup();
    assert_eq!(once, said);
    assert!(heard.len() - once.len() <= 2, "{heard:?}");
}

#[test]
fn finish_aligns_the_open_clip_once() {
    // Speech up to the end: no silence closes the clip, finish does.
    let script = talk("w", 4, 0.2);
    let audio = FakeClip::audio(&script, script.last().unwrap().end);
    let (mut t, asked) = adapter(&script, Setup::default(), 0);
    let mut finals = Vec::new();
    for chunk in audio.chunks(1_600) {
        finals.extend(t.push(chunk).unwrap().finalized);
    }
    assert!(finals.is_empty() && asked.lock().unwrap().timed.is_empty(), "nothing waits to be aligned");
    let flushed = t.finish().unwrap();
    assert_eq!(
        flushed.iter().map(|w| w.end).collect::<Vec<_>>(),
        script.iter().map(|w| w.end).collect::<Vec<_>>()
    );
    assert_eq!(asked.lock().unwrap().timed.len(), 1);
    assert!(t.finish().unwrap().is_empty());
    assert_eq!(asked.lock().unwrap().timed.len(), 1, "a second finish aligns nothing");
}

#[test]
fn silence_holds_nothing_back() {
    // A clip is aligned in the push that closes it, 800 ms into the silence:
    // nothing is queued for a later push, so silence never holds words back.
    let script = talk("w", 2, 0.5);
    let audio = FakeClip::audio(&script, 10 * S);
    let (mut t, asked) = adapter(&script, Setup::default(), 0);
    let mut closed_at = None;
    for (i, chunk) in audio.chunks(1_600).enumerate() {
        if !t.push(chunk).unwrap().finalized.is_empty() {
            closed_at.get_or_insert((i + 1) * 1_600);
        }
    }
    let speech_end = script.last().unwrap().end as usize;
    let closed_at = closed_at.expect("the clip closed in the silence");
    assert!(closed_at <= speech_end + crate::clip::endpoint::END_SILENCE as usize + 1_600, "{closed_at}");
    assert_eq!(asked.lock().unwrap().timed.len(), 1);
    assert!(t.finish().unwrap().is_empty());
}

#[test]
fn a_clip_qwen_calls_none_keeps_the_words_its_preview_heard() {
    // One clip, 0.3 s to 2.2 s, previewed once at 1 s: the preview hears
    // both words, the final hears `None`.
    let script = talk("w", 2, 0.5);
    let audio = FakeClip::audio(&script, 4 * S);
    let setup = Setup { none_from: Some(3 * S as usize / 2), ..Setup::default() };
    let (mut t, asked) = adapter(&script, setup, PREVIEW_EVERY);
    let (finals, _) = replay(&mut t, &audio, 1_600);
    let texts: Vec<&str> = finals.iter().map(|w| w.text.as_str()).collect();
    assert_eq!(texts, ["w0", "w1"]);
    assert_eq!(finals[0].lang.as_deref(), Some("en"), "the preview's language");
    assert!(finals[1].end <= (1.3 * S as f64) as Ts, "aligned on the audio the preview heard");
    assert_eq!(asked.lock().unwrap().timed.len(), 1);
    assert_eq!(t.warnings().len(), 1);
}

#[test]
fn an_overloaded_model_slows_pushes_down_and_drops_no_audio() {
    // Each transcription takes 50 ms, slower than 100 ms pushes of 20 ms
    // previews could keep up with: previews are skipped, finals are made in
    // the push that closes their clip, and every word becomes final.
    let mut script = talk("a", 8, 0.5);
    script.extend(talk("b", 8, 6.0));
    let audio = FakeClip::audio(&script, 12 * S);
    let setup = Setup { slow: Some(Duration::from_millis(50)), ..Setup::default() };
    let (mut t, asked) = adapter(&script, setup, S / 50);
    let (finals, _) = replay(&mut t, &audio, 1_600);
    assert_eq!(finals.len(), script.len());
    let pushes = audio.len().div_ceil(1_600);
    let asked = asked.lock().unwrap();
    assert!(asked.generated.len() < pushes, "previews due every 20 ms were skipped, not queued");
    assert_eq!(asked.timed.len(), 2);
}

#[test]
fn an_aligner_that_fails_fails_the_push_that_closes_the_clip() {
    let script = talk("w", 2, 0.5);
    let audio = FakeClip::audio(&script, 4 * S);
    let (mut t, _) = adapter(&script, Setup { fail: true, ..Setup::default() }, 0);
    let err = audio.chunks(1_600).find_map(|chunk| t.push(chunk).err()).expect("a push failed");
    assert!(format!("{err:#}").contains("the fake aligner fails"), "{err:#}");
}

/// A timer that gives each unit the times it is told.
struct Fixed(Vec<(f32, f32)>);

impl Timer for Fixed {
    fn time(&mut self, _: &[f32], units: &[String]) -> Result<Vec<(f32, f32)>> {
        Ok(self.0.iter().copied().take(units.len()).collect())
    }
}

fn spans(words: &[Word]) -> Vec<(f64, f64)> {
    words.iter().map(|w| (w.start as f64 / S as f64, w.end as f64 / S as f64)).collect()
}

#[test]
fn any_models_words_are_timed_by_their_units() {
    // Another model's words: one holding two Han characters, and a dash
    // with nothing to align.
    let words = vec![word("这个", 0.0, 3.0), word("—", 0.0, 3.0), word("API.", 0.0, 3.0)];
    let mut a = Aligner::new(Fixed(vec![(0.25, 0.5), (0.5, 0.75), (1.0, 9.0)]));
    let got = a.align(&vec![0.0; 3 * S as usize], words).unwrap();
    assert_eq!(
        got.iter().map(|w| w.text.as_str()).collect::<Vec<_>>(),
        ["这个", "—", "API."],
        "text unchanged"
    );
    assert_eq!(spans(&got), [(0.25, 0.75), (0.75, 0.75), (1.0, 3.0)], "kept inside the clip");
    let mut short = Aligner::new(Fixed(vec![(0.0, 0.5)]));
    let err =
        short.align(&vec![0.0; S as usize], vec![word("a", 0.0, 1.0), word("b", 0.0, 1.0)]).unwrap_err();
    assert!(err.to_string().contains("timed 1 units of the 2"), "{err}");
}

#[test]
fn the_aligner_hears_the_clips_speech_not_its_pre_roll_or_closing_silence() {
    // 0.2 s of pre-roll, 1 s of speech, 0.8 s of the silence that closed the clip.
    let mut audio = vec![0.0; 2 * S as usize];
    audio[(S / 5) as usize..(S * 6 / 5) as usize].fill(0.1);
    struct Heard(Arc<Mutex<usize>>);
    impl Timer for Heard {
        fn time(&mut self, audio: &[f32], _: &[String]) -> Result<Vec<(f32, f32)>> {
            *self.0.lock().unwrap() = audio.len();
            Ok(vec![(0.0, 0.25), (0.5, 2.0)])
        }
    }
    let heard = Arc::new(Mutex::new(0));
    let words = vec![word("我", 0.0, 2.0), word("好", 0.0, 2.0)];
    let got = Aligner::new(Heard(heard.clone())).align(&audio, words).unwrap();
    assert_eq!(*heard.lock().unwrap(), S as usize);
    assert_eq!(spans(&got), [(0.2, 0.45), (0.7, 1.2)], "back on the clip's clock, inside the speech");
}

#[test]
fn only_the_pinned_model_is_known_and_a_missing_file_says_where_to_look() {
    let err = model_files("qwen3-asr-0.6b").unwrap_err().to_string();
    assert!(err.contains("knows qwen3-asr-1.7b"), "{err}");
    let dir = std::env::temp_dir().join(format!("pam-qwen-models-{}", std::process::id()));
    let err = model_files_in("qwen3-asr-1.7b", &Store::at(&dir)).unwrap_err().to_string();
    assert!(err.contains("Qwen3-ASR-1.7B-Q8_0.gguf") && err.contains("tools/qwen-gguf/make.sh"), "{err}");
}

#[cfg(not(feature = "qwen"))]
#[test]
fn without_the_feature_opening_says_how_to_get_it() {
    let err = crate::open_transcriber("qwen", &crate::TranscriberConfig::default()).err().unwrap();
    assert!(err.to_string().contains("--features qwen"), "{err}");
}
