//! `dictate models`: the model store, as dictate uses it. A thin wrapper over
//! `pam_speech::store`, which pam shares: what
//! dictate needs is its config's transcriber, and the copy of its model that
//! boosts hints when the config has hints.

use std::fmt::Write as _;
use std::io::IsTerminal;
use std::process::ExitCode;

use anyhow::Result;
use pam_speech::nemo;
use pam_speech::store::{self, Curl, Download, Model, Store};

use crate::config::Config;
use crate::engine::{Event, Input, Options};
use crate::{FAILED, cli, emit, usage_error};

/// The command that fetches a model, as a missing model's error names it.
const PULL: &str = "dictate models pull";

/// Why `dictate models` failed.
enum Fail {
    Usage(String),
    Failed(anyhow::Error),
}

impl From<anyhow::Error> for Fail {
    fn from(e: anyhow::Error) -> Fail {
        Fail::Failed(e)
    }
}

/// Runs `dictate models <args>` (the arguments after `models`). Piped, or
/// with `--json`, a pull's progress is `downloading` events on stdout, and a
/// failure an `error` event; on a terminal it is drawn on stderr.
pub fn run(args: &[String]) -> ExitCode {
    let json = args.iter().any(|a| a == "--json");
    let piped = !std::io::stdin().is_terminal() || json;
    if args.iter().any(|a| a == "-h" || a == "--help") {
        print!("{}", cli::manual());
        return ExitCode::SUCCESS;
    }
    let args: Vec<&str> = args.iter().map(String::as_str).filter(|a| *a != "--json").collect();
    let mut out = String::new();
    let ran = (|| {
        let (config, store) = (Config::load()?, Store::open()?);
        run_in(&args, &store, &Curl { progress: false }, &config, piped, &mut emit, &mut out)
    })();
    print!("{out}");
    match ran {
        Ok(()) => ExitCode::SUCCESS,
        Err(Fail::Usage(msg)) => usage_error(msg, piped),
        Err(Fail::Failed(e)) => {
            let message = format!("{e:#}");
            if piped {
                emit(Event::Error { message: message.clone() });
            }
            eprintln!("error: {message}");
            ExitCode::from(FAILED)
        }
    }
}

fn run_in(
    args: &[&str],
    store: &Store,
    via: &dyn Download,
    config: &Config,
    piped: bool,
    emit: &mut dyn FnMut(Event),
    out: &mut String,
) -> Result<(), Fail> {
    if let Some(opt) = args.iter().find(|a| a.starts_with('-')) {
        return Err(Fail::Usage(format!("unknown option {opt} for `dictate models`; it takes --json")));
    }
    let Some((verb, names)) = args.split_first() else {
        return Err(Fail::Usage("missing what `dictate models` should do: list, pull, rm or path".into()));
    };
    let models = names
        .iter()
        .map(|name| {
            store.model(name).ok_or_else(|| {
                Fail::Usage(format!("unknown model `{name}`; the store knows {}", store.names().join(", ")))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    match *verb {
        "list" if models.is_empty() => list(store, config, out),
        "pull" => pull(store, via, config, models, piped, emit),
        "rm" if !models.is_empty() => rm(store, models, out),
        "path" if models.is_empty() => {
            writeln!(out, "{}", store.root().display()).unwrap();
            Ok(())
        }
        "list" | "path" => Err(Fail::Usage(format!("`dictate models {verb}` takes no model names"))),
        "rm" => Err(Fail::Usage("missing the model to remove, e.g. dictate models rm nemotron-3.5".into())),
        other => Err(Fail::Usage(format!("unknown `dictate models {other}`: give list, pull, rm or path"))),
    }
}

fn list(store: &Store, config: &Config, out: &mut String) -> Result<(), Fail> {
    // A config naming a model the store doesn't know still lists the store.
    let used: Vec<&'static Model> = store.model_for(&config.asr).ok().flatten().into_iter().collect();
    for row in store.list_rows(&used)? {
        writeln!(out, "{row}").unwrap();
    }
    Ok(())
}

/// Fetches `models`, or with none, the config's transcriber and, when the
/// config has hints, the copy of its model that boosts them.
fn pull(
    store: &Store,
    via: &dyn Download,
    config: &Config,
    models: Vec<&'static Model>,
    piped: bool,
    emit: &mut dyn FnMut(Event),
) -> Result<(), Fail> {
    let named = !models.is_empty();
    let models = if named { models } else { store.model_for(&config.asr)?.into_iter().collect() };
    for &model in &models {
        let mut told = None;
        let mut events = |bytes, total| {
            if told != Some(bytes) {
                told = Some(bytes);
                emit(Event::Downloading { model: model.name().into(), bytes, total });
            }
        };
        let mut drawn = store::draw_progress(model);
        let progress: &mut dyn FnMut(u64, u64) = if piped { &mut events } else { &mut drawn };
        store.pull(model, via, progress)?;
    }
    if !named && !config.hints.is_empty() {
        for &model in &models {
            if nemo::boost(store, model)?.is_some() {
                eprintln!("{}: ready to boost hints", model.name());
            }
        }
    }
    Ok(())
}

fn rm(store: &Store, models: Vec<&'static Model>, out: &mut String) -> Result<(), Fail> {
    for model in models {
        match store.remove_if_present(model)? {
            Some(freed) => writeln!(out, "removed {} ({})", model.name(), store::gb(freed)).unwrap(),
            None => return Err(Fail::Failed(anyhow::anyhow!("model `{}` isn't in the store", model.name()))),
        }
    }
    Ok(())
}

/// Checks that the store has the model `opts` would load, before anything
/// opens: a missing one is an error that names it, the path looked at and
/// `dictate models pull <model>`. dictate never downloads one itself.
/// A script needs none; a backend this build lacks, or a model the store
/// doesn't know, is left to opening the transcriber, which says why.
pub fn check(opts: &Options) -> Result<()> {
    if matches!(opts.input, Input::Script(_)) {
        return Ok(());
    }
    let built = pam_speech::parse_backend(&opts.asr).is_ok_and(|(b, _)| pam_speech::built(b));
    let store = Store::open()?;
    match store.model_for(&opts.asr) {
        Ok(Some(model)) if built => store.require(model, PULL),
        _ => Ok(()),
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;

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
                "other.gguf" => b"x",
                other => anyhow::bail!("no such file {other}"),
            };
            Ok(std::fs::write(to, body)?)
        }
    }

    fn store(dir: &Path) -> Store {
        let models: &'static [&'static Model] =
            Box::leak(Box::new([tiny("asr", b"asr weights"), tiny("other", b"x")]));
        Store::at(dir).with_models(models)
    }

    /// Runs `dictate models <line>` piped, returning stdout and the events.
    fn models(store: &Store, line: &str) -> Result<(String, Vec<Event>), Fail> {
        let config = Config { asr: "nemo-speech:asr".into(), ..Config::default() };
        let args: Vec<&str> = line.split_whitespace().collect();
        let (mut out, mut events) = (String::new(), Vec::new());
        run_in(&args, store, &Fake, &config, true, &mut |e| events.push(e), &mut out)?;
        Ok((out, events))
    }

    fn ok(store: &Store, line: &str) -> (String, Vec<Event>) {
        models(store, line).unwrap_or_else(|f| match f {
            Fail::Usage(m) => panic!("dictate models {line}: usage: {m}"),
            Fail::Failed(e) => panic!("dictate models {line}: {e:#}"),
        })
    }

    fn rows(out: &str) -> Vec<Vec<&str>> {
        out.lines().map(|l| l.split_whitespace().collect()).collect()
    }

    #[test]
    fn pull_fetches_the_configs_model_and_tells_how_far_it_has_got() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        assert_eq!(
            rows(&ok(&store, "list").0),
            [vec!["asr", "0.00", "GB", "-", "used"], vec!["other", "0.00", "GB", "-"]]
        );

        let (out, events) = ok(&store, "pull");
        assert_eq!(out, "", "stdout is for events");
        assert_eq!(events.first(), Some(&Event::Downloading { model: "asr".into(), bytes: 0, total: 11 }));
        assert_eq!(events.last(), Some(&Event::Downloading { model: "asr".into(), bytes: 11, total: 11 }));
        assert_eq!(rows(&ok(&store, "list").0)[0], ["asr", "0.00", "GB", "pulled", "used"]);
        assert_eq!(rows(&ok(&store, "list").0)[1], ["other", "0.00", "GB", "-"], "not needed, not pulled");

        assert!(ok(&store, "pull").1.is_empty(), "there already");
        ok(&store, "pull other");
        assert_eq!(rows(&ok(&store, "list").0)[1], ["other", "0.00", "GB", "pulled"]);
        assert_eq!(ok(&store, "path").0, format!("{}\n", dir.path().display()));
    }

    #[test]
    fn rm_removes_a_model_and_only_one_that_is_there() {
        let dir = tempfile::tempdir().unwrap();
        let store = store(dir.path());
        ok(&store, "pull asr");
        assert_eq!(ok(&store, "rm asr").0, "removed asr (0.00 GB)\n");
        assert!(store.locate(store.model("asr").unwrap()).is_err());
        assert!(
            matches!(models(&store, "rm asr"), Err(Fail::Failed(e)) if e.to_string().contains("isn't in the store"))
        );
        assert!(matches!(models(&store, "rm whisper"), Err(Fail::Usage(m)) if m.contains("asr, other")));
        assert!(matches!(models(&store, "rm"), Err(Fail::Usage(_))));
        assert!(matches!(models(&store, "list --all"), Err(Fail::Usage(m)) if m.contains("--all")));
        assert!(matches!(models(&store, ""), Err(Fail::Usage(m)) if m.contains("list, pull, rm or path")));
    }
}
