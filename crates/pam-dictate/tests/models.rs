//! `dictate models` and the model store, through the `dictate` binary with a
//! store of its own. Nothing here downloads: a pull is tested in
//! `src/models.rs`, with a fake download.

use std::path::Path;
use std::process::{Command, Output, Stdio};

use serde_json::Value;

/// Runs `dictate` piped, with `models` as its store and no config file.
/// Its stdin stays open until it exits: its end would be a cancel.
fn dictate(models: &Path, args: &[&str]) -> Output {
    let mut child = Command::new(env!("CARGO_BIN_EXE_dictate"))
        .args(args)
        .env("DICTATE_CONFIG", "/nonexistent/dictate/config.toml")
        .env("PAM_MODELS", models)
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .unwrap();
    let _stdin = child.stdin.take();
    child.wait_with_output().unwrap()
}

fn events(out: &Output) -> Vec<Value> {
    String::from_utf8_lossy(&out.stdout).lines().map(|l| serde_json::from_str(l).unwrap()).collect()
}

#[test]
fn a_missing_model_stops_it_before_the_mic_opens() {
    let dir = tempfile::tempdir().unwrap();
    let models = dir.path().join("models");
    let out = dictate(&models, &["--input", "mic"]);
    assert_eq!(out.status.code(), Some(1));
    let events = events(&out);
    let kinds: Vec<&str> = events.iter().map(|e| e["event"].as_str().unwrap()).collect();
    let message = events.last().unwrap()["message"].as_str().unwrap();
    if cfg!(feature = "nemo-speech") {
        assert_eq!(kinds, ["error"], "before anything loads");
        assert!(message.starts_with("model `nemotron-en` isn't in the model store"), "{message}");
        assert!(message.contains(&models.display().to_string()), "{message}");
        assert!(message.ends_with("run `dictate models pull nemotron-en`"), "{message}");
    } else {
        // Left to the transcriber, which says how to build it.
        assert_eq!(kinds, ["loading", "error"]);
        assert!(message.contains("--features nemo-speech"), "{message}");
    }
    assert!(!models.exists(), "nothing downloaded");
}

#[test]
fn list_and_path_show_the_store() {
    let dir = tempfile::tempdir().unwrap();
    let models = dir.path().join("models");
    let out = dictate(&models, &["models", "path"]);
    assert_eq!(out.status.code(), Some(0));
    assert_eq!(String::from_utf8_lossy(&out.stdout), format!("{}\n", models.display()));

    let out = dictate(&models, &["models", "list"]);
    assert_eq!(out.status.code(), Some(0));
    let list = String::from_utf8_lossy(&out.stdout).into_owned();
    let first: Vec<&str> = list.lines().next().unwrap().split_whitespace().collect();
    assert_eq!(first, ["nemotron-en", "0.70", "GB", "-", "used"], "{list}");
    assert!(list.contains("qwen3.5-4b"), "{list}");
}

#[test]
fn an_unknown_model_or_verb_is_an_error_event() {
    let dir = tempfile::tempdir().unwrap();
    let out = dictate(dir.path(), &["models", "rm", "whisper"]);
    assert_eq!(out.status.code(), Some(1));
    let said = events(&out);
    assert_eq!(said[0]["event"], "error");
    assert!(said[0]["message"].as_str().unwrap().contains("unknown model `whisper`"), "{said:?}");

    let out = dictate(dir.path(), &["models", "rm", "nemotron-en"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(events(&out)[0]["message"].as_str().unwrap().contains("isn't in the store"));

    let out = dictate(dir.path(), &["models"]);
    assert_eq!(out.status.code(), Some(1));
    assert!(String::from_utf8_lossy(&out.stderr).contains("dictate models list | pull"));
}

#[test]
fn help_says_the_store_is_shared_and_documents_downloading() {
    let dir = tempfile::tempdir().unwrap();
    let help = String::from_utf8(dictate(dir.path(), &["models", "--help"]).stdout).unwrap();
    assert!(help.contains("The store is shared with pam"), "{help}");
    assert!(help.contains("downloading (model, bytes, total)"), "{help}");
    assert!(help.contains("dictate models pull [<name>…]"), "{help}");
}
