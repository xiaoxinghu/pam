//! `dictate`'s config file: the mic, the transcriber, the language, the
//! timeout and the hints, so that a person need not repeat them on every
//! launch. A flag overrides its key; `--hint` and `--hints` add to `hints`.
//!
//! `$DICTATE_CONFIG`, or `~/.config/dictate/config.toml`. A missing file
//! means the defaults; every key is optional.
//!
//! ```toml
//! mic = "Yeti Stereo Microphone"
//! asr = "nemo-speech:nemotron-en"
//! language = "en-US"
//! timeout = 120
//! hints = ["obento", "Magit", "gptel", "Ghostty"]
//! ```

use std::ffi::OsString;
use std::path::{Path, PathBuf};

use anyhow::{Context, Result, bail};
use serde::Deserialize;

#[derive(Debug, Clone, PartialEq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    /// The input device `--input mic` listens to, by name; `None` for the
    /// system's default input. `--device` overrides it.
    pub mic: Option<String>,
    /// The transcriber, as `<backend>:<model>`. `--asr` overrides it.
    pub asr: String,
    /// What is spoken, as a language code, or `auto`. `--language` overrides it.
    pub language: String,
    /// Stop after this many seconds; 0 for never. `--timeout` overrides it.
    pub timeout: f64,
    /// Names and words the transcriber should favour. `--hint` and
    /// `--hints` add to them.
    pub hints: Vec<String>,
}

impl Default for Config {
    fn default() -> Config {
        Config {
            mic: None,
            asr: "nemo-speech:nemotron-en".into(),
            language: "en-US".into(),
            timeout: 0.0,
            hints: Vec::new(),
        }
    }
}

impl Config {
    /// Reads the config file, if there is one.
    pub fn load() -> Result<Config> {
        Config::load_with(|name| std::env::var_os(name))
    }

    /// [`Config::load`], with the environment read through `var`.
    pub(crate) fn load_with(var: impl Fn(&str) -> Option<OsString>) -> Result<Config> {
        match config_path(var) {
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
        let config: Config = toml::from_str(&text).with_context(|| format!("in {}", path.display()))?;
        if !(config.timeout >= 0.0 && config.timeout.is_finite()) {
            bail!(
                "in {}: timeout = {}: give a number of seconds, e.g. 120; 0 means never",
                path.display(),
                config.timeout
            );
        }
        Ok(config)
    }
}

fn config_path(var: impl Fn(&str) -> Option<OsString>) -> Option<PathBuf> {
    if let Some(path) = var("DICTATE_CONFIG").filter(|p| !p.is_empty()) {
        return Some(PathBuf::from(path));
    }
    var("HOME").map(|home| PathBuf::from(home).join(".config/dictate/config.toml"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_file_or_key_keeps_the_defaults() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        assert_eq!(Config::read(&path).unwrap(), Config::default());
        std::fs::write(&path, "timeout = 120\n").unwrap();
        let cfg = Config::read(&path).unwrap();
        assert_eq!(cfg.timeout, 120.0);
        assert_eq!(cfg.asr, Config::default().asr);
    }

    #[test]
    fn an_unknown_key_names_itself_and_the_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "device = \"Yeti\"\n").unwrap();
        let err = format!("{:#}", Config::read(&path).unwrap_err());
        assert!(err.contains("config.toml") && err.contains("device"), "{err}");
    }

    #[test]
    fn a_bad_value_names_its_key() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.toml");
        std::fs::write(&path, "timeout = -1\n").unwrap();
        let err = format!("{:#}", Config::read(&path).unwrap_err());
        assert!(err.contains("config.toml") && err.contains("timeout = -1"), "{err}");
        std::fs::write(&path, "hints = \"obento\"\n").unwrap();
        let err = format!("{:#}", Config::read(&path).unwrap_err());
        assert!(err.contains("hints"), "{err}");
    }

    #[test]
    fn dictate_config_names_the_file_else_home_does() {
        let env = |pairs: &'static [(&str, &str)]| {
            move |name: &str| pairs.iter().find(|(n, _)| *n == name).map(|(_, v)| OsString::from(v))
        };
        assert_eq!(
            config_path(env(&[("DICTATE_CONFIG", "/a/b.toml"), ("HOME", "/h")])),
            Some("/a/b.toml".into())
        );
        assert_eq!(
            config_path(env(&[("DICTATE_CONFIG", ""), ("HOME", "/h")])),
            Some("/h/.config/dictate/config.toml".into())
        );
        assert_eq!(config_path(env(&[])), None);
    }
}
