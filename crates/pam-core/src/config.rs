//! Pam's config file: the assistant's name; which speech backends,
//! microphone, pi and models to use; and how long the room has to answer
//! the agent's questions.
//!
//! `$PAM_CONFIG`, or `~/.config/pam/config.toml`. A missing file means the
//! defaults.
//!
//! ```toml
//! name = "Pam"
//! asr = "nemo-speech:nemotron-en"
//! language = "auto"
//! diarizer = "nemo-speech:nemotron-3-diarization"
//! mic = "MacBook Pro Microphone"
//! pi = "/opt/pi/pi"
//! scribe_model = "openrouter/anthropic/claude-haiku-5.5"
//! intent_model = "local"
//! agent_model = "openrouter/anthropic/claude-sonnet-5.5"
//! question_timeout = 15
//! ```

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// The assistant's name, which is also its wake word; `None` for
    /// Pam. `pam listen --name` overrides it.
    pub name: Option<String>,
    /// The transcriber, as `<backend>:<model>`.
    pub asr: String,
    /// The language the transcriber listens for: a code its model knows, such
    /// as `en-US` or `zh-CN`, or `auto` to detect each utterance's language.
    pub language: String,
    /// The speaker tracker, as `<backend>:<model>`.
    pub diarizer: String,
    /// The input device `--input mic` listens to, by name; `None` for the
    /// system's default input.
    pub mic: Option<String>,
    /// The `pi` executable for the agent and the scribe; `None` for `pi` on `PATH`.
    pub pi: Option<PathBuf>,
    /// The scribe's model, as pi's `--model` takes it, or `off`; `None` to
    /// detect one. `pam listen --scribe` overrides it.
    pub scribe_model: Option<String>,
    /// The intent check's model, the same way, or `local`; `None` for
    /// the scribe's. `pam listen --intent` overrides it.
    pub intent_model: Option<String>,
    /// The agent's model, the same way. `pam listen --agent` overrides it.
    pub agent_model: Option<String>,
    /// The countdown on the agent's questions, in seconds, when `pam task
    /// ask --timeout` doesn't set one.
    pub question_timeout: u64,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            name: None,
            asr: "nemo-speech:nemotron-en".into(),
            language: "auto".into(),
            diarizer: "nemo-speech:nemotron-3-diarization".into(),
            mic: None,
            pi: None,
            scribe_model: None,
            intent_model: None,
            agent_model: None,
            question_timeout: crate::question::DEFAULT_TIMEOUT,
        }
    }
}

impl Config {
    /// Reads the config file, if there is one.
    pub fn load() -> Result<Config> {
        match config_path()? {
            Some(path) => Config::read(&path),
            None => Ok(Config::default()),
        }
    }

    /// Reads `path`; a missing file gives the defaults.
    pub fn read(path: &Path) -> Result<Config> {
        let text = match std::fs::read_to_string(path) {
            Ok(text) => text,
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Config::default()),
            Err(e) => return Err(e).with_context(|| format!("reading {}", path.display())),
        };
        toml::from_str(&text).with_context(|| format!("in {}", path.display()))
    }
}

fn config_path() -> Result<Option<PathBuf>> {
    if let Some(path) = std::env::var_os("PAM_CONFIG").filter(|p| !p.is_empty()) {
        return Ok(Some(PathBuf::from(path)));
    }
    Ok(std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config/pam/config.toml")))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_file_or_key_keeps_the_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        assert_eq!(Config::read(&path).unwrap(), Config::default());
        std::fs::write(&path, "asr = \"fake:script.jsonl\"\n").unwrap();
        let cfg = Config::read(&path).unwrap();
        assert_eq!(cfg.asr, "fake:script.jsonl");
        assert_eq!(cfg.diarizer, Config::default().diarizer);
    }

    #[test]
    fn an_unknown_key_names_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "asr_model = \"x\"\n").unwrap();
        let err = format!("{:#}", Config::read(&path).unwrap_err());
        assert!(err.contains("config.toml") && err.contains("asr_model"), "{err}");
    }
}
