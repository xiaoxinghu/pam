//! Choosing and checking the scribe's, the intent check's and the agent's
//! models.
//!
//! Each comes from the `pam listen` option, else the config file, else
//! detection: the first candidate that `pi auth check` says is ready. The
//! intent check isn't detected: unless one is named, it uses the scribe's
//! model, and `local` names the local model. Before
//! the meeting the core finds pi, checks its version against the pinned one
//! and checks the login for each model, so a model that can't work says why on
//! the stage and in `pam status` from the start. A model that was named but
//! isn't ready is reported, never swapped for a detected one.

use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use pam_speech::store::{self, Store};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// The pi version Pam is tested with. To bump it, re-run step 2's pi harness
/// (in `pam-private`'s history, `bench/step-2/`) against the new version and
/// record the results first.
const PINNED: &str = "1.1.0";

/// The scribe's candidates, in order: fast, cheap models, since it runs
/// every 20–30 s.
pub const SCRIBE_CANDIDATES: [&str; 3] =
    ["openrouter/anthropic/claude-haiku-5.5", "anthropic/claude-haiku-5-5", "openai/gpt-5.6-luna"];

/// The agent's candidates after the default model in the user's pi settings:
/// stronger than the scribe's, on the same providers, so one login serves both.
pub const AGENT_CANDIDATES: [&str; 2] =
    ["openrouter/anthropic/claude-sonnet-5.5", "anthropic/claude-sonnet-5-5"];

/// How long `pi --version` and `pi auth check` may take.
const CHECK_TIMEOUT: Duration = Duration::from_secs(20);

/// What `--intent` and `intent_model` name the local model by.
pub const LOCAL: &str = "local";

/// What chose a model.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Source {
    /// `pam listen --scribe` or `--agent`.
    Option,
    /// `scribe_model`, `intent_model` or `agent_model` in the config file.
    Config,
    /// The first candidate pi had a login for; for the intent check, the
    /// scribe's model.
    Detected,
}

impl Source {
    pub fn as_str(self) -> &'static str {
        match self {
            Source::Option => "option",
            Source::Config => "config",
            Source::Detected => "detected",
        }
    }
}

/// Which use a model is for.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Use {
    Scribe,
    Intent,
    Agent,
}

impl Use {
    fn name(self) -> &'static str {
        match self {
            Use::Scribe => "scribe",
            Use::Intent => "intent check",
            Use::Agent => "agent",
        }
    }

    /// The `pam listen` option and the config key that name its model.
    fn option(self) -> &'static str {
        match self {
            Use::Scribe => "--scribe",
            Use::Intent => "--intent",
            Use::Agent => "--agent",
        }
    }

    fn key(self) -> &'static str {
        match self {
            Use::Scribe => "scribe_model",
            Use::Intent => "intent_model",
            Use::Agent => "agent_model",
        }
    }
}

/// What the option and the config file asked for.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Pick {
    /// A model as pi's `--model` takes it (`provider/id[:thinking]`), or
    /// `off`; or for the intent check, `local`.
    Named { model: String, source: Source },
    /// Neither named one.
    Detect,
}

impl Pick {
    /// The option's model for `what` if given, else the config file's, else
    /// detection. Only the one that counts must name a model: an empty
    /// config value under an option is never read.
    pub fn new(what: Use, option: Option<&str>, config: Option<&str>) -> Result<Pick, String> {
        let (model, source) = match (option, config) {
            (Some(m), _) => (m.trim(), Source::Option),
            (None, Some(m)) => (m.trim(), Source::Config),
            (None, None) => return Ok(Pick::Detect),
        };
        if model.is_empty() {
            let given = match source {
                Source::Config => format!("`{}` in the config file", what.key()),
                _ => what.option().to_string(),
            };
            let local = if what == Use::Intent { ", local" } else { "" };
            return Err(format!("{given} is empty: give it a model as provider/id{local}, or off"));
        }
        Ok(Pick::Named { model: model.to_string(), source })
    }

    pub fn off(source: Source) -> Pick {
        Pick::Named { model: "off".into(), source }
    }

    fn is_off(&self) -> bool {
        matches!(self, Pick::Named { model, .. } if model == "off")
    }

    /// Whether it names the local model.
    pub fn is_local(&self) -> bool {
        matches!(self, Pick::Named { model, .. } if model == LOCAL)
    }
}

/// Whether a model can be used.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum ModelStatus {
    Ready,
    /// Ready, but something is worth knowing (a pi patch release other than
    /// the pinned one).
    Warning,
    /// Named or detected, but it can't be used; the reason says why.
    Unavailable,
    /// Turned off with the option or the config file.
    Off,
}

/// What the meeting found out about the scribe's, the intent check's or the
/// agent's model when it started. Kept in `meeting.json` as `scribe`,
/// `intent` and `agent`, and sent to front-ends.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ModelInfo {
    pub status: ModelStatus,
    /// Why it is unavailable or off, or the warning.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// The pi that runs it, as an absolute path, once found; none for the
    /// local model.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub pi: Option<PathBuf>,
    /// What `pi --version` said.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub version: Option<String>,
    /// As pi's `--model` takes it; none when it is off or detection found nothing.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// The pi provider that gets the calls, as `pi auth check` named it;
    /// `local` for the local model, which sends nothing anywhere.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider: Option<String>,
    /// How it was chosen; none when it is off or unavailable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<Source>,
    /// The kind of login pi has for it, as `pi auth check` named it
    /// (`oauth`, `api_key`). Kept in `meeting.json`'s `usage`, not here.
    #[serde(skip)]
    pub login: Option<String>,
    /// Whether pi's model catalog has it (`pi --list-models`), so pi can
    /// price its calls; none when the check failed. A model it lacks
    /// still runs, priced at the rates of its provider's default model.
    #[serde(skip)]
    pub catalogued: Option<bool>,
}

/// The scribe's model.
pub type ScribeInfo = ModelInfo;
/// The intent check's model: the scribe's unless one is named.
pub type IntentInfo = ModelInfo;
/// The agent's model.
pub type AgentInfo = ModelInfo;

impl ModelInfo {
    fn new(status: ModelStatus) -> ModelInfo {
        ModelInfo {
            status,
            reason: None,
            pi: None,
            version: None,
            model: None,
            provider: None,
            source: None,
            login: None,
            catalogued: None,
        }
    }

    /// Whether calls can go to it.
    pub fn usable(&self) -> bool {
        matches!(self.status, ModelStatus::Ready | ModelStatus::Warning)
    }

    /// The provider the calls go to: pi's, else the model's part before the
    /// first `/`.
    pub fn provider(&self) -> Option<&str> {
        self.provider.as_deref().or_else(|| self.model.as_deref()?.split_once('/').map(|(p, _)| p))
    }

    /// The model's ID, after the last `/`, as the dashboard header shows it.
    pub fn short_model(&self) -> Option<&str> {
        self.model.as_deref().map(|m| m.rsplit('/').next().unwrap_or(m))
    }

    /// Why it can't be used, if it can't.
    pub fn why_not(&self) -> Option<&str> {
        match self.status {
            ModelStatus::Unavailable | ModelStatus::Off => {
                Some(self.reason.as_deref().unwrap_or("unknown reason"))
            }
            _ => None,
        }
    }
}

/// What `pam listen` passes on: the pi to use and what the option and the
/// config file picked.
#[derive(Debug, Clone)]
pub struct Choice {
    /// The `pi` executable from the config file; `None` for `pi` on `PATH`.
    pub pi: Option<PathBuf>,
    /// pi's `settings.json`, where the agent's first candidate comes from.
    pub settings: PathBuf,
    pub scribe: Pick,
    pub intent: Pick,
    pub agent: Pick,
}

impl Choice {
    /// Whether the meeting loads the local intent model: `local` was
    /// chosen and the agent isn't turned off, since without an agent nothing
    /// is done with what the check hears.
    pub fn loads_local(&self) -> bool {
        self.intent.is_local() && !self.agent.is_off()
    }
}

/// The models a meeting loads from the model store: its transcriber's and
/// its speaker tracker's, given as `<backend>:<model>`, then the local
/// intent model if `local`. `pam models pull` fetches them, and `pam listen`
/// stops before the meeting when one is missing, rather than download it at
/// launch, which would read as a hang and fail offline. A model given as a
/// path, or a fake script, loads nothing from the store.
pub fn stored(store: &Store, asr: &str, diarizer: &str, local: bool) -> Result<Vec<&'static store::Model>> {
    let mut models = Vec::new();
    for spec in [asr, diarizer] {
        models.extend(store.model_for(spec)?);
    }
    if local {
        models.push(crate::local::MODEL);
    }
    Ok(models)
}

/// Checks that the store has what a meeting loads ([`stored`]), before it
/// starts. A missing model is an error that names it, the path looked at and
/// `pam models pull <model>`. A backend this build lacks, or a model name the
/// store doesn't know, is left to opening the backend, which says why; the
/// local model without llama.cpp, to the intent check, which says how to
/// build it.
pub fn check_stored(store: &Store, asr: &str, diarizer: &str, local: bool) -> Result<()> {
    let built = |spec: &&str| pam_speech::parse_backend(spec).is_ok_and(|(b, _)| pam_speech::built(b));
    let speech = [asr, diarizer].into_iter().filter(built).filter_map(|s| store.model_for(s).ok().flatten());
    let local = local && cfg!(feature = "llama-cpp");
    for model in speech.chain(local.then_some(crate::local::MODEL)) {
        store.require(model, "pam models pull")?;
    }
    Ok(())
}

/// pi's `settings.json` as pi 1.1.0 finds it: under `$PI_CODING_AGENT_DIR`,
/// else `~/.pi/agent/`.
pub fn pi_settings() -> PathBuf {
    let home = || std::env::var_os("HOME").map(PathBuf::from).unwrap_or_default();
    let dir = match std::env::var_os("PI_CODING_AGENT_DIR").filter(|d| !d.is_empty()) {
        Some(d) => match Path::new(&d).strip_prefix("~") {
            Ok(rest) => home().join(rest),
            Err(_) => PathBuf::from(d),
        },
        None => home().join(".pi/agent"),
    };
    dir.join("settings.json")
}

/// The default model in pi's settings, as `provider/id[:thinking]`: its
/// `defaultProvider` and `defaultModel`, with the thinking level pi would
/// start it with (`modelThinkingLevels` for that model, else
/// `defaultThinkingLevel`). `None` when the file or either key is missing.
fn pi_default(settings: &Path) -> Option<String> {
    let text = std::fs::read_to_string(settings).ok()?;
    let s: Value = serde_json::from_str(&text).ok()?;
    let (provider, id) = (s["defaultProvider"].as_str()?.trim(), s["defaultModel"].as_str()?.trim());
    if provider.is_empty() || id.is_empty() {
        return None;
    }
    let model = format!("{provider}/{id}");
    let thinking = s["modelThinkingLevels"][&model].as_str().or(s["defaultThinkingLevel"].as_str());
    Some(match thinking.map(str::trim).filter(|t| !t.is_empty()) {
        Some(t) => format!("{model}:{t}"),
        None => model,
    })
}

/// The models detection tries for `what`, in order.
pub fn candidates(what: Use, settings: &Path) -> Vec<String> {
    let mut list: Vec<String> = match what {
        Use::Scribe => SCRIBE_CANDIDATES.iter().map(|m| m.to_string()).collect(),
        Use::Agent => pi_default(settings).into_iter().chain(AGENT_CANDIDATES.map(String::from)).collect(),
        // Never detected: it follows the scribe.
        Use::Intent => Vec::new(),
    };
    let mut seen = std::collections::BTreeSet::new();
    list.retain(|m| seen.insert(m.clone()));
    list
}

/// Finds pi, checks its version against [`PINNED`] and picks and checks the
/// scribe's, the intent check's and the agent's models. Every check runs at
/// once, so this takes about as long as one (~0.2 s). Never fails: a model
/// that can't be used is `Unavailable`, with the reason.
pub fn check(choice: &Choice) -> (ScribeInfo, IntentInfo, AgentInfo) {
    let tries = |what: Use, pick: &Pick| match pick {
        _ if pick.is_off() || pick.is_local() => Vec::new(),
        Pick::Named { model, .. } => vec![model.clone()],
        // The intent check follows the scribe.
        Pick::Detect if what == Use::Intent => Vec::new(),
        Pick::Detect => candidates(what, &choice.settings),
    };
    let scribe_tries = tries(Use::Scribe, &choice.scribe);
    let intent_tries = tries(Use::Intent, &choice.intent);
    let agent_tries = tries(Use::Agent, &choice.agent);
    // None goes through pi: pi isn't needed, or even looked for.
    let pi = match scribe_tries.is_empty() && intent_tries.is_empty() && agent_tries.is_empty() {
        true => Err("none of them uses pi".to_string()),
        false => find(choice.pi.as_deref()),
    };
    let checked = pi.map(|pi| {
        let models: std::collections::BTreeSet<&String> =
            scribe_tries.iter().chain(&intent_tries).chain(&agent_tries).collect();
        thread::scope(|s| {
            let logins: Vec<_> = models.iter().map(|&m| (m.clone(), s.spawn(|| login(&pi, m)))).collect();
            let listings: Vec<_> = models.iter().map(|&m| (m.clone(), s.spawn(|| listing(&pi, m)))).collect();
            let version = version(&pi);
            let logins = logins
                .into_iter()
                .map(|(m, h)| {
                    (m, h.join().unwrap_or_else(|_| Auth::Failed("checking the login failed".into())))
                })
                .collect();
            let listings = listings.into_iter().map(|(m, h)| (m, h.join().ok().flatten())).collect();
            Checked { pi: pi.clone(), version, logins, listings }
        })
    });
    let pi = checked.as_ref().map_err(String::as_str);
    let scribe = decide(Use::Scribe, &choice.scribe, pi, &scribe_tries);
    let intent = match &choice.intent {
        Pick::Detect => follow(&scribe),
        Pick::Named { source, .. } if choice.intent.is_local() => local(*source),
        named => decide(Use::Intent, named, pi, &intent_tries),
    };
    (scribe, intent, decide(Use::Agent, &choice.agent, pi, &agent_tries))
}

/// The intent check when nothing names its model: the scribe's, off
/// with it.
fn follow(scribe: &ScribeInfo) -> IntentInfo {
    let mut info = ModelInfo { source: Some(Source::Detected), ..scribe.clone() };
    if info.status == ModelStatus::Off {
        info.reason = Some(format!(
            "it uses the scribe's model, and the scribe is off ({}); `--intent local` checks on this Mac",
            scribe.reason.as_deref().unwrap_or("off")
        ));
    }
    info
}

/// The local model, ready when this pam was built with llama.cpp and
/// the model is downloaded. Whether it loads is found out as the meeting
/// starts.
fn local(source: Source) -> IntentInfo {
    let mut info = ModelInfo {
        model: Some(crate::local::id()),
        provider: Some(LOCAL.into()),
        source: Some(source),
        ..ModelInfo::new(ModelStatus::Ready)
    };
    let why = match crate::local::path() {
        _ if !cfg!(feature = "llama-cpp") => {
            Some("this pam was built without llama.cpp: build it with `--features llama-cpp`".to_string())
        }
        Ok(path) if path.is_file() => None,
        Ok(path) => Some(format!(
            "the local model isn't in the model store ({}): run `pam models pull {}`",
            path.display(),
            crate::local::MODEL.name()
        )),
        Err(e) => Some(format!("{e:#}")),
    };
    if let Some(why) = why {
        info.status = ModelStatus::Unavailable;
        info.reason = Some(why);
    }
    info
}

/// What the checks found out.
struct Checked {
    pi: PathBuf,
    version: Result<String, String>,
    logins: BTreeMap<String, Auth>,
    /// What `pi --list-models` found for each model; none where it failed.
    listings: BTreeMap<String, Option<Vec<Listed>>>,
}

/// One model's info, from the pick, what the checks found and the models
/// it tried, in order.
fn decide(what: Use, pick: &Pick, pi: Result<&Checked, &str>, tried: &[String]) -> ModelInfo {
    let source = match pick {
        Pick::Named { source, .. } => *source,
        Pick::Detect => Source::Detected,
    };
    let mut info = ModelInfo { source: Some(source), ..ModelInfo::new(ModelStatus::Ready) };
    if pick.is_off() {
        info.status = ModelStatus::Off;
        info.reason = Some(match source {
            Source::Config => format!("{} = \"off\" in the config file", what.key()),
            _ => format!("{} off", what.option()),
        });
        return info;
    }
    if let Pick::Named { model, .. } = pick {
        info.model = Some(model.clone());
    }
    let unavailable = |mut info: ModelInfo, why: String| {
        info.status = ModelStatus::Unavailable;
        info.reason = Some(why);
        info
    };
    let checked = match pi {
        Ok(c) => c,
        Err(why) => return unavailable(info, why.to_string()),
    };
    info.pi = Some(checked.pi.clone());
    let version = match &checked.version {
        Ok(v) => v.clone(),
        Err(why) => return unavailable(info, why.clone()),
    };
    info.version = Some(version.clone());
    match pin(&version) {
        Pin::Same => {}
        Pin::Patch => {
            info.status = ModelStatus::Warning;
            info.reason = Some(format!(
                "pi {version} is not the tested {PINNED}, but the same minor release; it should work"
            ));
        }
        Pin::Other => {
            return unavailable(
                info,
                format!(
                    "pi {version} is installed, but Pam needs {PINNED}: pi's minor releases change how it works. \
                     Install pi {PINNED}, or set `pi` in the config file to one"
                ),
            );
        }
    }
    let auth = |m: &String| checked.logins.get(m).cloned().unwrap_or(Auth::Failed("not checked".into()));
    let catalogued = |m: &String, provider: Option<&str>| {
        checked.listings.get(m).and_then(Option::as_deref).map(|rows| in_catalog(rows, m, provider))
    };
    match pick {
        Pick::Named { model, source } => match auth(model) {
            Auth::Ready { provider, login } => {
                info.catalogued = catalogued(model, provider.as_deref());
                info.provider = provider;
                info.login = login;
                info
            }
            other => unavailable(info, other.why(model, what, *source)),
        },
        Pick::Detect => match tried.iter().find_map(|m| match auth(m) {
            Auth::Ready { provider, login } => Some((m, provider, login)),
            _ => None,
        }) {
            Some((model, provider, login)) => {
                info.model = Some(model.clone());
                info.catalogued = catalogued(model, provider.as_deref());
                info.provider = provider;
                info.login = login;
                info
            }
            None => unavailable(
                info,
                format!(
                    "pi has a login for none of the {} models Pam tries ({}): log in to a provider with `pi` and /login, \
                     or set `{}` in the config file",
                    what.name(),
                    tried.join(", "),
                    what.key()
                ),
            ),
        },
    }
}

/// The `pi` to run, as an absolute path: `configured` if set, else `pi` on `PATH`.
pub fn find(configured: Option<&Path>) -> Result<PathBuf, String> {
    let found = match configured {
        Some(p) if is_executable(p) => p.to_path_buf(),
        Some(p) => return Err(format!("there is no pi at {}, the `pi` in the config file", p.display())),
        None => std::env::var_os("PATH")
            .iter()
            .flat_map(std::env::split_paths)
            .map(|dir| dir.join("pi"))
            .find(|p| is_executable(p))
            .ok_or_else(|| {
                format!("pi isn't installed, or isn't on PATH: install pi {PINNED}, or set `pi` in the config file")
            })?,
    };
    std::path::absolute(&found).map_err(|e| format!("finding {}: {e}", found.display()))
}

fn is_executable(p: &Path) -> bool {
    use std::os::unix::fs::PermissionsExt;
    p.metadata().is_ok_and(|m| m.is_file() && m.permissions().mode() & 0o111 != 0)
}

/// How a pi version relates to [`PINNED`].
#[derive(Debug, PartialEq, Eq)]
enum Pin {
    Same,
    /// The same major.minor, another patch.
    Patch,
    /// Another major or minor release, or a version that can't be read.
    Other,
}

fn pin(found: &str) -> Pin {
    let parts = |v: &str| -> Option<(u32, u32, u32)> {
        let mut it = v.trim().trim_start_matches('v').splitn(3, '.');
        let (a, b, c) = (it.next()?, it.next()?, it.next()?);
        let c = c.split(|ch: char| !ch.is_ascii_digit()).next()?;
        Some((a.parse().ok()?, b.parse().ok()?, c.parse().ok()?))
    };
    match (parts(found), parts(PINNED)) {
        (Some(f), Some(p)) if f == p => Pin::Same,
        (Some(f), Some(p)) if (f.0, f.1) == (p.0, p.1) => Pin::Patch,
        _ => Pin::Other,
    }
}

/// What `<pi> --version` prints, e.g. `1.1.0`.
fn version(pi: &Path) -> Result<String, String> {
    let out = run_quick(Command::new(pi).arg("--version")).map_err(|e| format!("{e:#}"))?;
    let v = out.stdout.split_whitespace().last().unwrap_or_default().to_string();
    if !out.ok || v.is_empty() {
        return Err(format!("`{} --version` failed: {}", pi.display(), out.why()));
    }
    Ok(v)
}

/// What `pi auth check` said about a model.
#[derive(Debug, Clone)]
enum Auth {
    Ready { provider: Option<String>, login: Option<String> },
    NotReady,
    Invalid(String),
    Failed(String),
}

impl Auth {
    /// Why a model named by `source` can't be used, and what to do.
    fn why(&self, model: &str, what: Use, source: Source) -> String {
        let instead = match source {
            Source::Option => format!("pass another model to {}", what.option()),
            _ => format!("set `{}` in the config file", what.key()),
        };
        match self {
            Auth::Ready { .. } => String::new(),
            Auth::NotReady => format!(
                "pi has no login for {model}: run `pi` and log in with /login, or set the provider's API key, or {instead}"
            ),
            Auth::Invalid(why) => format!("pi can't use the model {model} ({why}): {instead}"),
            Auth::Failed(why) => format!("`pi auth check` failed: {why}"),
        }
    }
}

/// One row of `pi --list-models`: a model in pi's catalog, by provider.
#[derive(Debug, Clone, PartialEq)]
struct Listed {
    pub provider: String,
    pub id: String,
}

/// The models in pi's catalog that `pi --list-models <model>` finds (a fuzzy
/// search, so more than the model); `None` when it fails. pi lists only the
/// providers it has a login for.
fn listing(pi: &Path, model: &str) -> Option<Vec<Listed>> {
    let out = run_quick(Command::new(pi).args(["--list-models", without_thinking(model)])).ok()?;
    out.ok.then(|| parse_listing(&out.stdout))
}

/// The rows of `pi --list-models`' table, after its header; none for "No
/// models matching".
fn parse_listing(table: &str) -> Vec<Listed> {
    let mut lines = table.lines();
    if !lines.next().is_some_and(|h| h.split_whitespace().take(2).eq(["provider", "model"])) {
        return Vec::new();
    }
    lines
        .filter_map(|l| {
            let mut cols = l.split_whitespace();
            Some(Listed { provider: cols.next()?.into(), id: cols.next()?.into() })
        })
        .collect()
}

/// Whether `rows` have `model` (`provider/id[:thinking]`), as pi resolves
/// it: its whole name as the ID, or the part after the provider. pi takes
/// `anthropic/claude-haiku-5.5` as OpenRouter's model of that ID, so the
/// row's provider must be the one `pi auth check` named, when it named one.
fn in_catalog(rows: &[Listed], model: &str, provider: Option<&str>) -> bool {
    let model = without_thinking(model);
    rows.iter().filter(|r| provider.is_none_or(|p| r.provider == p)).any(|r| {
        r.id == model
            || model.strip_prefix(r.provider.as_str()).and_then(|m| m.strip_prefix('/')) == Some(&r.id)
    })
}

/// `model` without a `:thinking` suffix, which pi's `--model` takes but its
/// catalog doesn't name. Other suffixes, such as OpenRouter's `:free`, are
/// part of the ID.
fn without_thinking(model: &str) -> &str {
    const LEVELS: [&str; 7] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];
    match model.rsplit_once(':') {
        Some((m, level)) if LEVELS.contains(&level) => m,
        _ => model,
    }
}

/// Checks that pi has a login for `model`, with `pi auth check`.
fn login(pi: &Path, model: &str) -> Auth {
    let out = match run_quick(Command::new(pi).args(["auth", "check", "--model", model, "--json"])) {
        Ok(out) => out,
        Err(e) => return Auth::Failed(format!("{e:#}")),
    };
    let reply: Value = serde_json::from_str(out.stdout.trim()).unwrap_or(Value::Null);
    match reply["status"].as_str() {
        Some("ready") => Auth::Ready {
            provider: reply["provider"].as_str().map(str::to_string),
            login: reply["authType"].as_str().map(str::to_string),
        },
        Some("not_ready") => Auth::NotReady,
        Some("invalid") => Auth::Invalid(reply["reason"].as_str().unwrap_or("invalid").to_string()),
        _ => Auth::Failed(out.why()),
    }
}

struct Quick {
    ok: bool,
    stdout: String,
    stderr: String,
}

impl Quick {
    fn why(&self) -> String {
        let text = if self.stderr.trim().is_empty() { &self.stdout } else { &self.stderr };
        text.trim().lines().last().unwrap_or("no output").to_string()
    }
}

/// Runs a short command to the end, or kills it after [`CHECK_TIMEOUT`].
fn run_quick(cmd: &mut Command) -> Result<Quick> {
    let program = cmd.get_program().to_string_lossy().into_owned();
    let mut child = cmd
        .stdin(Stdio::null())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped())
        .spawn()
        .with_context(|| format!("starting {program}"))?;
    let read = |mut r: Box<dyn Read + Send>| {
        thread::spawn(move || {
            let mut s = String::new();
            let _ = r.read_to_string(&mut s);
            s
        })
    };
    let stdout = read(Box::new(child.stdout.take().unwrap()));
    let stderr = read(Box::new(child.stderr.take().unwrap()));
    let end = Instant::now() + CHECK_TIMEOUT;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if Instant::now() >= end {
            let _ = child.kill();
            let _ = child.wait();
            bail!("{program} took over {} s; stopped it", CHECK_TIMEOUT.as_secs());
        }
        thread::sleep(Duration::from_millis(10));
    };
    Ok(Quick {
        ok: status.success(),
        stdout: stdout.join().unwrap_or_default(),
        stderr: stderr.join().unwrap_or_default(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn choice(intent: Pick) -> Choice {
        let pick = Pick::new(Use::Scribe, Some("off"), None).unwrap();
        Choice { pi: None, settings: PathBuf::new(), scribe: pick.clone(), intent, agent: pick }
    }

    #[test]
    fn only_a_local_intent_check_with_an_agent_loads_the_local_model() {
        let named = |m| Pick::new(Use::Intent, Some(m), None).unwrap();
        for pick in [Pick::Detect, named("off"), named("anthropic/claude-haiku-5-5")] {
            assert!(!choice(pick).loads_local());
        }
        assert!(Choice { agent: Pick::Detect, ..choice(named("local")) }.loads_local());
        assert!(
            !choice(named("local")).loads_local(),
            "with the agent off, nothing would act on what it hears"
        );
    }

    #[test]
    fn a_patch_release_warns_and_a_minor_one_disables() {
        assert_eq!(pin("1.1.0"), Pin::Same);
        assert_eq!(pin("v1.1.0\n"), Pin::Same);
        assert_eq!(pin("1.1.3"), Pin::Patch);
        assert_eq!(pin("1.1.4-beta.1"), Pin::Patch);
        assert_eq!(pin("1.2.0"), Pin::Other);
        assert_eq!(pin("2.1.0"), Pin::Other);
        assert_eq!(pin("1.0.9"), Pin::Other);
        assert_eq!(pin("pi"), Pin::Other);
    }

    #[test]
    fn a_configured_pi_must_exist() {
        let dir = tempfile::tempdir().unwrap();
        let missing = dir.path().join("pi");
        assert!(find(Some(&missing)).unwrap_err().contains("the `pi` in the config file"));
        std::fs::write(&missing, "#!/bin/sh\n").unwrap();
        assert!(find(Some(&missing)).is_err(), "not executable yet");
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&missing, std::fs::Permissions::from_mode(0o755)).unwrap();
        assert_eq!(find(Some(&missing)).unwrap(), missing);
    }

    #[test]
    fn the_option_beats_the_config_file_which_beats_detection() {
        let named = |model: &str, source| Pick::Named { model: model.into(), source };
        let pick = |option, config| Pick::new(Use::Scribe, option, config);
        assert_eq!(pick(Some("a/b"), Some("c/d")), Ok(named("a/b", Source::Option)));
        assert_eq!(pick(None, Some(" c/d ")), Ok(named("c/d", Source::Config)));
        assert_eq!(pick(Some("off"), None), Ok(Pick::off(Source::Option)));
        assert_eq!(pick(None, None), Ok(Pick::Detect));
        // Only the value that counts must name a model.
        assert_eq!(pick(Some("a/b"), Some("")), Ok(named("a/b", Source::Option)));
        assert_eq!(
            Pick::new(Use::Agent, None, Some(" ")),
            Err("`agent_model` in the config file is empty: give it a model as provider/id, or off".into())
        );
        assert_eq!(
            pick(Some(""), Some("c/d")),
            Err("--scribe is empty: give it a model as provider/id, or off".into())
        );
    }

    #[test]
    fn the_agent_tries_the_pi_default_first_with_its_thinking_level() {
        let dir = tempfile::tempdir().unwrap();
        let settings = dir.path().join("settings.json");
        assert_eq!(pi_default(&settings), None);
        assert_eq!(candidates(Use::Agent, &settings), AGENT_CANDIDATES);
        std::fs::write(&settings, r#"{"defaultProvider": "openai", "defaultModel": "gpt-6-astra"}"#).unwrap();
        assert_eq!(pi_default(&settings).as_deref(), Some("openai/gpt-6-astra"));
        std::fs::write(
            &settings,
            r#"{"defaultProvider": "openai", "defaultModel": "gpt-6-astra", "defaultThinkingLevel": "high"}"#,
        )
        .unwrap();
        assert_eq!(candidates(Use::Agent, &settings)[0], "openai/gpt-6-astra:high");
        std::fs::write(
            &settings,
            r#"{"defaultProvider": "openai", "defaultModel": "gpt-6-astra", "defaultThinkingLevel": "high",
                "modelThinkingLevels": {"openai/gpt-6-astra": "low"}}"#,
        )
        .unwrap();
        assert_eq!(pi_default(&settings).as_deref(), Some("openai/gpt-6-astra:low"));
        // Already a candidate: tried once, first.
        std::fs::write(&settings, r#"{"defaultProvider": "anthropic", "defaultModel": "claude-sonnet-5-5"}"#)
            .unwrap();
        assert_eq!(candidates(Use::Agent, &settings), ["anthropic/claude-sonnet-5-5", AGENT_CANDIDATES[0]]);
        std::fs::write(&settings, r#"{"defaultModel": "gpt-6-astra"}"#).unwrap();
        assert_eq!(pi_default(&settings), None, "no provider");
        assert_eq!(candidates(Use::Scribe, &settings), SCRIBE_CANDIDATES);
    }

    #[test]
    fn finds_a_model_in_pis_catalog_as_pi_resolves_it() {
        let table = "provider      model                         context  max-out  thinking  images\n\
                     openai        gpt-6-astra                   272K     128K     yes       yes   \n\
                     openrouter    anthropic/claude-haiku-5.5    1M       128K     yes       yes   \n\
                     openrouter    google/gemma-4-31b-it:free    262.1K   32.8K    yes       yes   \n";
        let rows = parse_listing(table);
        assert_eq!(rows.len(), 3);
        assert_eq!(rows[0], Listed { provider: "openai".into(), id: "gpt-6-astra".into() });
        assert!(in_catalog(&rows, "openai/gpt-6-astra:high", Some("openai")));
        assert!(in_catalog(&rows, "openrouter/anthropic/claude-haiku-5.5", Some("openrouter")));
        // pi resolves this one through OpenRouter.
        assert!(in_catalog(&rows, "anthropic/claude-haiku-5.5", Some("openrouter")));
        assert!(!in_catalog(&rows, "anthropic/claude-haiku-5.5", Some("anthropic")));
        // `:free` is part of the ID, not a thinking level.
        assert!(in_catalog(&rows, "openrouter/google/gemma-4-31b-it:free", Some("openrouter")));
        assert!(!in_catalog(&rows, "openrouter/google/gemma-4-31b-it", Some("openrouter")));
        // The fuzzy search found a model, but not this one.
        assert!(!in_catalog(&rows, "openai/gpt-6", Some("openai")));
        assert!(parse_listing("No models matching \"openrouter/gryphe/mythomax-l2-13b\"\n").is_empty());
    }

    #[test]
    fn a_meeting_needs_its_speech_models_and_the_local_one_from_the_store() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let asr = "nemo-speech:nemotron-en";
        let diar = "nemo-speech:nemotron-3-diarization";
        let names = |local| -> Vec<&str> {
            stored(&store, asr, diar, local).unwrap().iter().map(|m| m.name()).collect()
        };
        assert_eq!(names(false), ["nemotron-en", "nemotron-3-diarization"]);
        assert_eq!(names(true), ["nemotron-en", "nemotron-3-diarization", "qwen3.5-4b"]);
        assert!(stored(&store, "fake:a.fake.jsonl", "nemo-speech:/m/d.gguf", false).unwrap().is_empty());

        // Missing: named with the command to run, when this build has the backend.
        let checked = check_stored(&store, asr, "fake:x", false);
        if pam_speech::built("nemo-speech") {
            let err = checked.unwrap_err().to_string();
            assert!(err.ends_with("run `pam models pull nemotron-en`"), "{err}");
        } else {
            checked.expect("left to opening the backend");
        }
        let checked = check_stored(&store, "fake:x", "fake:x", true);
        if cfg!(feature = "llama-cpp") {
            let err = checked.unwrap_err().to_string();
            assert!(err.ends_with("run `pam models pull qwen3.5-4b`"), "{err}");
        } else {
            checked.expect("left to the intent check");
        }
        check_stored(&store, "fake:x", "nemo-speech:no-such-model", false).unwrap();
    }
}
