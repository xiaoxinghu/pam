//! The real NeMo-Speech backend, end to end on synthetic audio. Needs the
//! `nemo-speech` feature and both models in the model store
//! (`pam_speech::store`); without the models it says so and passes.

#![cfg(feature = "nemo-speech")]

use pam_speech::{TrackerConfig, TranscriberConfig, Word, nemo};

fn models_present() -> bool {
    match nemo::asr_model("nemotron-en").and_then(|_| nemo::diar_model("nemotron-3-diarization")) {
        Ok(_) => true,
        Err(e) => {
            eprintln!("skipped: {e:#}");
            false
        }
    }
}

#[test]
fn silence_gives_no_words_and_no_speakers_and_settles_behind_the_audio() {
    if !models_present() {
        return;
    }
    let mut t = pam_speech::open_transcriber(
        "nemo-speech",
        &TranscriberConfig {
            model: "nemotron-en".into(),
            language: "en".into(),
            hints: vec!["Siobhan".into()],
        },
    )
    .unwrap();
    let mut k =
        pam_speech::open_tracker("nemo-speech", &TrackerConfig { model: "nemotron-3-diarization".into() })
            .unwrap();
    let chunk = vec![0.0f32; 1_600];
    let mut settled = 0;
    for _ in 0..90 {
        assert!(t.push(&chunk).unwrap().finalized.is_empty());
        let u = k.push(&chunk).unwrap();
        assert!(u.spans.finalized.is_empty());
        settled = settled.max(u.settled_until);
    }
    // 9 s pushed: labelled to 7.68 s (three 2.56 s chunks, each with 0.32 s of right
    // context after it), settled 1.5 s behind that.
    assert_eq!(settled, nemo::ts(7.68) - nemo::SETTLE_MARGIN);
    assert!(t.finish().unwrap().is_empty());
    assert!(k.finish().unwrap().is_empty());
}

/// `text` spoken by the macOS voice `voice`, as 16 kHz samples; `None` when
/// `say` or the voice isn't there.
fn say(voice: &str, text: &str) -> Option<Vec<f32>> {
    // Tests run in parallel: each call gets a file of its own.
    static CALLS: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
    let n = CALLS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
    let path = std::env::temp_dir().join(format!("pam-nemo-say-{}-{n}.wav", std::process::id()));
    let ok = std::process::Command::new("say")
        .args(["-v", voice, "--file-format=WAVE", "--data-format=LEI16@16000", "-o"])
        .arg(&path)
        .arg(text)
        .status()
        .is_ok_and(|s| s.success());
    if !ok {
        eprintln!("skipped: `say -v {voice}` didn't work");
        return None;
    }
    let samples =
        hound::WavReader::open(&path).ok()?.samples::<i16>().map(|s| s.unwrap() as f32 / 32768.0).collect();
    std::fs::remove_file(&path).ok();
    Some(samples)
}

/// Pushes `audio` in 100 ms chunks, then finishes: every final word.
fn transcribe(model: &str, audio: &[f32]) -> Vec<Word> {
    let cfg =
        TranscriberConfig { model: model.into(), language: "auto".into(), ..TranscriberConfig::default() };
    let mut t = pam_speech::open_transcriber("nemo-speech", &cfg).unwrap();
    let mut words = Vec::new();
    for chunk in audio.chunks(1_600) {
        words.extend(t.push(chunk).unwrap().finalized);
    }
    words.extend(t.finish().unwrap());
    words
}

const SILENCE: usize = 2 * 16_000;

#[test]
fn after_a_new_stream_words_keep_the_meeting_clock() {
    if !models_present() {
        return;
    }
    let (Some(first), Some(second)) = (
        say("Samantha", "Hi everyone, let's go through the release plan."),
        say("Samantha", "Okay, then let's keep Friday as the backup date."),
    ) else {
        return;
    };
    // 2 s of silence, the first sentence, 2 s, the second, 2 s: the first
    // utterance ends in the middle, so with the language detected the second
    // is heard by a new stream. An English-only model keeps one.
    let mut audio = vec![0.0; SILENCE];
    audio.extend(&first);
    audio.extend(vec![0.0; SILENCE]);
    let second_at = audio.len() as u64;
    audio.extend(&second);
    let second_end = audio.len() as u64;
    audio.extend(vec![0.0; SILENCE]);

    let models: &[&str] = if nemo::asr_model("nemotron-3.5").is_ok() {
        &["nemotron-3.5", "nemotron-en"]
    } else {
        &["nemotron-en"]
    };
    for model in models {
        let words = transcribe(model, &audio);
        let text: Vec<&str> = words.iter().map(|w| w.text.as_str()).collect();
        let friday = words
            .iter()
            .find(|w| w.text.starts_with("Friday"))
            .unwrap_or_else(|| panic!("{model}: {text:?}"));
        // Stamped at emission, within the second sentence, not 6 s earlier.
        assert!(friday.start > second_at && friday.end <= second_end + 16_000, "{model}: {friday:?}");
        assert!(words.windows(2).all(|w| w[0].start <= w[1].start), "{model}: {words:?}");
        let hi = &words[0];
        assert!(hi.start >= SILENCE as u64 && hi.start < SILENCE as u64 + 16_000, "{model}: {hi:?}");
    }
    // An English-only model's words are English, named or given as a path.
    let path = nemo::asr_model("nemotron-en").unwrap();
    for model in ["nemotron-en", path.to_str().unwrap()] {
        let words = transcribe(model, &audio);
        assert!(words.iter().all(|w| w.lang.as_deref() == Some("en-US")), "{model}: {words:?}");
    }
}

#[test]
fn each_turn_is_heard_whole_in_its_own_language() {
    if let Err(e) = nemo::asr_model("nemotron-3.5") {
        eprintln!("skipped: {e:#}");
        return;
    }
    let (Some(en), Some(zh)) = (
        say("Samantha", "Great. How long do you think the performance testing will take?"),
        say("Tingting", "大概需要三天。如果发现问题的话，可能要推迟到下周五再发布。"),
    ) else {
        return;
    };
    // In one stream, the start of the Mandarin turn after the English one was lost.
    let mut audio = vec![0.0; SILENCE];
    audio.extend(&en);
    audio.extend(vec![0.0; SILENCE]);
    let zh_at = audio.len() as u64;
    audio.extend(&zh);
    audio.extend(vec![0.0; SILENCE]);

    let words = transcribe("nemotron-3.5", &audio);
    let (en, zh): (Vec<&Word>, Vec<&Word>) = words.iter().partition(|w| w.start < zh_at);
    let zh_text: String = zh.iter().map(|w| w.text.as_str()).collect();
    assert!(zh_text.starts_with("大概需要三天"), "{zh_text}");
    assert!(zh.iter().all(|w| w.lang.as_deref() == Some("zh-CN")), "{zh:?}");
    assert!(en.iter().any(|w| w.text.starts_with("performance")), "{en:?}");
}

#[test]
fn nemotron_3_5_lists_its_languages_and_refuses_others() {
    let path = match nemo::asr_model("nemotron-3.5") {
        Ok(path) => path,
        Err(e) => {
            eprintln!("skipped: {e:#}");
            return;
        }
    };
    let listed = nemo::model_languages(&path).unwrap();
    nemo::check_language("nemotron-3.5", listed.as_deref(), "zh-CN").unwrap();
    let err = nemo::check_language("nemotron-3.5", listed.as_deref(), "zh_CN").unwrap_err().to_string();
    assert!(err.contains("zh-CN") && err.contains("en-US"), "{err}");
    let cfg = TranscriberConfig {
        model: "nemotron-3.5".into(),
        language: "xx-YY".into(),
        ..TranscriberConfig::default()
    };
    assert!(pam_speech::open_transcriber("nemo-speech", &cfg).is_err());
}
