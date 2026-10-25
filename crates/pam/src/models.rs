//! `pam models`: the model store, as pam uses it. A thin
//! wrapper over `pam_speech::store`: what pam's config needs comes from
//! `pam_core::models::stored`, and the store is shared with `dictate`.

use std::fmt::Write as _;

use pam_core::Config;
use pam_core::models::{self, Choice, Pick, Use};
use pam_speech::nemo;
use pam_speech::store::{self, Curl, Download, Model, Store};

use crate::cli::{Command, Fail, Parsed};

pub fn run(cmd: &'static Command, p: Parsed, out: &mut String) -> Result<(), Fail> {
    run_in(cmd, p, out, &Store::open()?, &Curl { progress: false }, &Config::load()?)
}

fn run_in(
    cmd: &'static Command,
    p: Parsed,
    out: &mut String,
    store: &Store,
    via: &dyn Download,
    config: &Config,
) -> Result<(), Fail> {
    let Some((verb, names)) = p.positionals.split_first() else {
        return Err(Fail::usage(cmd, "missing what to do: list, pull, rm or path"));
    };
    let models = names
        .iter()
        .map(|name| {
            store.model(name).ok_or_else(|| {
                Fail::usage(
                    cmd,
                    format!("unknown model `{name}`; the store knows {}", store.names().join(", ")),
                )
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    match verb.as_str() {
        "list" if models.is_empty() => list(store, config, out),
        "pull" => pull(store, via, config, models),
        "rm" if !models.is_empty() => rm(store, models, out),
        "path" if models.is_empty() => {
            writeln!(out, "{}", store.root().display()).unwrap();
            Ok(())
        }
        "list" | "path" => Err(Fail::usage(cmd, format!("`pam models {verb}` takes no model names"))),
        "rm" => Err(Fail::usage(cmd, "missing the model to remove, e.g. pam models rm qwen3.5-4b")),
        other => Err(Fail::usage(cmd, format!("unknown `{other}`: give list, pull, rm or path"))),
    }
}

/// What `pam listen` loads from the store with the config file's choices
/// and no options.
fn used(store: &Store, config: &Config) -> Result<Vec<&'static Model>, Fail> {
    let pick = |u, named: &Option<String>| Pick::new(u, None, named.as_deref()).map_err(anyhow::Error::msg);
    let choice = Choice {
        pi: None,
        settings: Default::default(),
        scribe: pick(Use::Scribe, &config.scribe_model)?,
        intent: pick(Use::Intent, &config.intent_model)?,
        agent: pick(Use::Agent, &config.agent_model)?,
    };
    Ok(models::stored(store, &config.asr, &config.diarizer, choice.loads_local())?)
}

fn list(store: &Store, config: &Config, out: &mut String) -> Result<(), Fail> {
    // A config naming a model the store doesn't know still lists the store.
    let used = used(store, config).unwrap_or_default();
    for row in store.list_rows(&used)? {
        writeln!(out, "{row}").unwrap();
    }
    Ok(())
}

/// Fetches `models`, or with none, what `pam listen` needs, and the copy of
/// its transcriber's model that boosts hints: the assistant's name is
/// one whenever spoken requests are on.
fn pull(store: &Store, via: &dyn Download, config: &Config, models: Vec<&'static Model>) -> Result<(), Fail> {
    let boost = if models.is_empty() { store.model_for(&config.asr)? } else { None };
    let models = if models.is_empty() { used(store, config)? } else { models };
    for model in models {
        store.pull(model, via, &mut store::draw_progress(model))?;
    }
    if let Some(model) = boost
        && nemo::boost(store, model)?.is_some()
    {
        eprintln!("{}: ready to boost hints", model.name());
    }
    Ok(())
}

fn rm(store: &Store, models: Vec<&'static Model>, out: &mut String) -> Result<(), Fail> {
    for model in models {
        match store.remove_if_present(model)? {
            Some(freed) => writeln!(out, "removed {} ({})", model.name(), store::gb(freed)).unwrap(),
            None => return Err(Fail::NotFound(format!("model `{}` isn't in the store", model.name()))),
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::cli;

    /// A model of one file holding `bytes`, as tiny as a test needs.
    fn tiny(name: &'static str, bytes: &'static [u8]) -> &'static Model {
        use sha2::{Digest, Sha256};
        let file = store::File {
            repo: "org/tiny",
            revision: "abc",
            name: format!("{name}.gguf").leak(),
            sha256: format!("{:x}", Sha256::digest(bytes)).leak(),
            bytes: bytes.len() as u64,
            made: None,
        };
        Box::leak(Box::new(Model { names: Box::leak(Box::new([name])), files: Box::leak(Box::new([file])) }))
    }

    /// Serves each tiny model's bytes by its file's name; never the network.
    struct Fake;

    impl Download for Fake {
        fn download(&self, url: &str, to: &Path) -> anyhow::Result<()> {
            let body: &[u8] = match url.rsplit('/').next().unwrap() {
                "asr.gguf" => b"asr weights",
                "diar.gguf" => b"diarizer",
                "other.gguf" => b"x",
                other => anyhow::bail!("no such file {other}"),
            };
            Ok(std::fs::write(to, body)?)
        }
    }

    fn store(dir: &Path) -> Store {
        let models: &'static [&'static Model] = Box::leak(Box::new([
            tiny("asr", b"asr weights"),
            tiny("diar", b"diarizer"),
            tiny("other", b"x"),
        ]));
        Store::at(dir).with_models(models)
    }

    fn config() -> Config {
        Config { asr: "nemo-speech:asr".into(), diarizer: "nemo-speech:diar".into(), ..Config::default() }
    }

    fn models(store: &Store, line: &str) -> Result<String, Fail> {
        let cmd = cli::COMMANDS.iter().find(|c| c.name == "models").unwrap();
        let args: Vec<String> = line.split_whitespace().map(str::to_string).collect();
        let p = cli::parse(cmd, &args).ok().flatten().unwrap();
        let mut out = String::new();
        run_in(cmd, p, &mut out, store, &Fake, &config()).map(|()| out)
    }

    fn ok(store: &Store, line: &str) -> String {
        models(store, line).unwrap_or_else(|f| panic!("pam models {line}: {}", f.message()))
    }

    #[test]
    fn pull_fetches_what_listen_needs_and_list_shows_it() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        let before = ok(&store, "list");
        assert_eq!(
            before.lines().next().unwrap().split_whitespace().collect::<Vec<_>>(),
            ["asr", "0.00", "GB", "-", "used"]
        );
        assert!(before.lines().nth(2).unwrap().ends_with("GB  -"), "no trailing space: {before:?}");

        ok(&store, "pull");
        let after = ok(&store, "list");
        let rows: Vec<Vec<&str>> = after.lines().map(|l| l.split_whitespace().collect()).collect();
        assert_eq!(rows[0], ["asr", "0.00", "GB", "pulled", "used"]);
        assert_eq!(rows[1], ["diar", "0.00", "GB", "pulled", "used"]);
        assert_eq!(rows[2], ["other", "0.00", "GB", "-"], "not needed, not pulled");

        ok(&store, "pull other");
        assert!(ok(&store, "list").lines().nth(2).unwrap().contains("pulled"));
        assert_eq!(ok(&store, "path"), format!("{}\n", dir.path().display()));
    }

    #[test]
    fn rm_removes_a_model_and_only_one_that_is_there() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        ok(&store, "pull asr");
        assert_eq!(ok(&store, "rm asr"), "removed asr (0.00 GB)\n");
        assert!(store.locate(store.model("asr").unwrap()).is_err());
        assert!(
            matches!(models(&store, "rm asr"), Err(Fail::NotFound(m)) if m.contains("isn't in the store"))
        );
        assert!(
            matches!(models(&store, "rm whisper"), Err(Fail::Usage(_, m)) if m.contains("asr, diar, other"))
        );
        assert!(matches!(models(&store, "rm"), Err(Fail::Usage(..))));
        assert!(matches!(models(&store, ""), Err(Fail::Usage(_, m)) if m.contains("list, pull, rm or path")));
    }
}
