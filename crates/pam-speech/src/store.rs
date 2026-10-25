//! The model store: one folder for the models of every product
//! built on these crates, `~/Library/Application Support/pam/models`, or
//! `$PAM_MODELS` when set and non-empty. Files keep their Hugging Face
//! layout, `<org>/<repo>/<revision>/<file>`.
//!
//! Every model is pinned here ([`MODELS`]) by repository, revision, SHA-256
//! and size, and fetched by [`Store::fetch`]: into `<file>.part`, renamed into
//! place only once its size and checksum match. A file that has to be made
//! here rather than downloaded as is says how instead.
//!
//! A missing model is a [`Missing`] error, which names the model and the path
//! looked at. What to run about it is the caller's to say.

use std::ffi::OsString;
use std::fmt;
use std::fs;
use std::io::IsTerminal;
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};
use std::thread;
use std::time::Duration;

use anyhow::{Context, bail};
use sha2::{Digest, Sha256};

use crate::Result;

/// One file of a model, pinned.
#[derive(Debug)]
pub struct File {
    /// The Hugging Face repository, `<org>/<repo>`.
    pub repo: &'static str,
    pub revision: &'static str,
    pub name: &'static str,
    pub sha256: &'static str,
    pub bytes: u64,
    /// How to make it, for a file that is converted here from the repository
    /// rather than downloaded as is, because the published one was made with
    /// Python.
    pub made: Option<&'static str>,
}

impl File {
    /// Where it is under the store's root.
    pub fn rel(&self) -> PathBuf {
        Path::new(self.repo).join(self.revision).join(self.name)
    }

    pub fn url(&self) -> String {
        format!("https://huggingface.co/{}/resolve/{}/{}", self.repo, self.revision, self.name)
    }
}

/// A model: the files a product loads together, under one name.
#[derive(Debug)]
pub struct Model {
    /// Its name, as configs give it, then any other names it answers to.
    pub names: &'static [&'static str],
    pub files: &'static [File],
}

impl Model {
    pub fn name(&self) -> &'static str {
        self.names[0]
    }

    /// What its files weigh.
    pub fn bytes(&self) -> u64 {
        self.files.iter().map(|f| f.bytes).sum()
    }
}

/// How the `qwen` backend's files are made, without Python.
const QWEN_MADE: &str = "run tools/qwen-gguf/make.sh in pam's repository, which converts \
     Qwen's checkpoints into the store";

pub static NEMOTRON_EN: Model = Model {
    names: &["nemotron-en", "nemotron-speech-streaming-en-0.6b"],
    files: &[File {
        repo: "nvidia/nemotron-speech-streaming-en-0.6b",
        revision: "ebe59e5a817142986528bbbee5dba8db7b38ed50",
        name: "nemotron-speech-streaming-en-0.6b.q8_0.gguf",
        sha256: "d9a01898d2a611c8764e23a1c2f45e70bbd5a425dc4de93692ac951dd603812d",
        bytes: 699_872_960,
        made: None,
    }],
};

pub static NEMOTRON_3_5: Model = Model {
    names: &["nemotron-3.5", "nemotron-3.5-asr-streaming-0.6b"],
    files: &[File {
        repo: "nvidia/nemotron-3.5-asr-streaming-0.6b",
        revision: "1c8deaecc64b91f034d73e08dd8b64625eb3395d",
        name: "nemotron-3.5-asr-streaming-0.6b.q8_0.gguf",
        sha256: "a5c435f294eea8f88ce68dd27b8c3bfea7f777cb2fbba04fcd30eaa555f429ae",
        bytes: 741_548_352,
        made: None,
    }],
};

pub static NEMOTRON_DIARIZATION: Model = Model {
    names: &["nemotron-3-diarization", "nemotron-diar"],
    files: &[File {
        repo: "nvidia/Nemotron-3-Diarization",
        revision: "f667ed73aee57d40cc39428eb768b4fd87a0a29e",
        name: "Nemotron-3-Diarization.q8_0.gguf",
        sha256: "08456d9e22cd9a323c0364d98375f3746d6e68507ebb705cd46438c534c7a3a1",
        bytes: 107_012_128,
        made: None,
    }],
};

/// Qwen3-ASR's decoder and audio projector, and Qwen3-ForcedAligner, in the
/// order the `qwen` backend loads them.
pub static QWEN3_ASR_1_7B: Model = Model {
    names: &["qwen3-asr-1.7b"],
    files: &[
        File {
            repo: "Qwen/Qwen3-ASR-1.7B",
            revision: "7278e1e70fe206f11671096ffdd38061171dd6e5",
            name: "Qwen3-ASR-1.7B-Q8_0.gguf",
            sha256: "58e22d0532d4eacaf034cfac17a6fed159f37c41390c710186783be439d1fc57",
            bytes: 2_165_034_944,
            made: Some(QWEN_MADE),
        },
        File {
            repo: "Qwen/Qwen3-ASR-1.7B",
            revision: "7278e1e70fe206f11671096ffdd38061171dd6e5",
            name: "mmproj-Qwen3-ASR-1.7B-Q8_0.gguf",
            sha256: "b43f64ac7ef3df4d7c5055e478085035e32ee6db6217b05e4f0d3112b405aab1",
            bytes: 355_709_344,
            made: Some(QWEN_MADE),
        },
        File {
            repo: "Qwen/Qwen3-ForcedAligner-0.6B",
            revision: "c7cbfc2048c462b0d63a45797104fc9db3ad62b7",
            name: "qwen3-forced-aligner-0.6b-q8_0.gguf",
            sha256: "2e383c048f11731fc65931e2adf2587377186b4695198e7a0542e42b8cab6373",
            bytes: 1_191_933_344,
            made: Some(QWEN_MADE),
        },
    ],
};

/// Qwen3.5 4B, Q4_K_M: Pam's local intent check.
pub static QWEN3_5_4B: Model = Model {
    names: &["qwen3.5-4b"],
    files: &[File {
        repo: "unsloth/Qwen3.5-4B-GGUF",
        revision: "e87f176479d0",
        name: "Qwen3.5-4B-Q4_K_M.gguf",
        sha256: "00fe7986ff5f6b463e62455821146049db6f9313603938a70800d1fb69ef11a4",
        bytes: 2_740_937_888,
        made: None,
    }],
};

/// Every model the store knows.
pub static MODELS: [&Model; 5] =
    [&NEMOTRON_EN, &NEMOTRON_3_5, &NEMOTRON_DIARIZATION, &QWEN3_ASR_1_7B, &QWEN3_5_4B];

/// A model, or one of its files, isn't in the store.
#[derive(Debug)]
pub struct Missing {
    pub model: &'static str,
    /// Where it was looked for in the store.
    pub path: PathBuf,
    /// How to make it, when it can't be downloaded.
    pub made: Option<&'static str>,
}

impl fmt::Display for Missing {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "model `{}` isn't in the model store (no {})", self.model, self.path.display())?;
        match self.made {
            Some(how) => write!(f, "; it is made, not downloaded: {how}"),
            None => Ok(()),
        }
    }
}

impl std::error::Error for Missing {}

impl Missing {
    /// The error, with the command that fetches the model, `<pull> <model>`
    /// (`pull` is the product's own, e.g. `dictate models pull`), unless it is
    /// made, not downloaded, which the error already says how.
    pub fn suggest(&self, pull: &str) -> String {
        match self.made {
            Some(_) => self.to_string(),
            None => format!("{self}: run `{pull} {}`", self.model),
        }
    }
}

/// What the store holds of a model.
#[derive(Debug)]
pub struct Entry {
    pub model: &'static Model,
    /// How many of its files are there.
    pub files: usize,
    /// What it takes on disk, with anything made beside its files (such as
    /// the `.boost.gguf` copy that boosts hints).
    pub bytes: u64,
}

impl Entry {
    pub fn complete(&self) -> bool {
        self.files == self.model.files.len()
    }
}

/// Gets a file from a URL: [`Curl`], or a fake in tests. It runs on a
/// thread of its own while the store watches the file grow.
pub trait Download: Sync {
    /// Writes what `url` holds to `to`.
    fn download(&self, url: &str, to: &Path) -> Result<()>;
}

/// Downloads with `curl`, showing its progress bar on stderr if `progress`.
pub struct Curl {
    pub progress: bool,
}

impl Download for Curl {
    fn download(&self, url: &str, to: &Path) -> Result<()> {
        let quiet = if self.progress { "-#" } else { "-s" };
        let status = Command::new("curl")
            .args(["-fSL", quiet, "--connect-timeout", "10", "-o"])
            .arg(to)
            .arg(url)
            .stdin(Stdio::null())
            .stdout(Stdio::null())
            .status()
            .context("running curl")?;
        if !status.success() {
            bail!("downloading {url} failed ({status})");
        }
        Ok(())
    }
}

/// The store.
#[derive(Debug)]
pub struct Store {
    root: PathBuf,
    /// The models it knows: [`MODELS`], or others in tests.
    models: &'static [&'static Model],
}

impl Store {
    /// The store the environment names ([`root`]).
    pub fn open() -> Result<Store> {
        Store::from_env(|key| std::env::var_os(key))
    }

    /// The store at `root`.
    pub fn at(root: impl Into<PathBuf>) -> Store {
        Store { root: root.into(), models: &MODELS }
    }

    /// This store, knowing `models` instead of [`MODELS`]: for tests, whose
    /// models are small enough to fetch from a fake.
    pub fn with_models(self, models: &'static [&'static Model]) -> Store {
        Store { models, ..self }
    }

    /// The model a name means, by any of its names.
    pub fn model(&self, name: &str) -> Option<&'static Model> {
        self.models.iter().copied().find(|m| m.names.contains(&name))
    }

    /// The model `<backend>:<model>` loads from the store: `None` for the
    /// fake backend and for a model given as a path to its file. A name the
    /// store doesn't know is an error that lists the names it knows.
    pub fn model_for(&self, spec: &str) -> Result<Option<&'static Model>> {
        let (backend, model) = crate::parse_backend(spec)?;
        if backend == "fake" || model.ends_with(".gguf") {
            return Ok(None);
        }
        match self.model(model) {
            Some(m) => Ok(Some(m)),
            None => bail!("unknown model `{model}`: the model store knows {}", self.names().join(", ")),
        }
    }

    /// Each model's name.
    pub fn names(&self) -> Vec<&'static str> {
        self.models.iter().map(|m| m.name()).collect()
    }

    fn from_env(var: impl Fn(&str) -> Option<OsString>) -> Result<Store> {
        let var = |key| var(key).filter(|v| !v.is_empty());
        if let Some(root) = var("PAM_MODELS") {
            return Ok(Store::at(root));
        }
        let home = PathBuf::from(
            var("HOME").context("can't find the model store: neither PAM_MODELS nor HOME is set")?,
        );
        Ok(Store::at(home.join("Library/Application Support/pam/models")))
    }

    pub fn root(&self) -> &Path {
        &self.root
    }

    /// Where `file` goes in the store.
    pub fn path(&self, file: &File) -> PathBuf {
        self.root.join(file.rel())
    }

    /// Where `file` is, if the store has it.
    pub fn find(&self, file: &File) -> Option<PathBuf> {
        Some(self.path(file)).filter(|p| p.is_file())
    }

    /// The lock a pinned file's download holds, under `.locks/` in the store
    /// and never removed, so that everyone waiting for a download of the file
    /// waits on the same lock.
    fn lock(&self, f: &File) -> Result<fs::File> {
        let dir = self.root.join(".locks");
        fs::create_dir_all(&dir).with_context(|| format!("creating {}", dir.display()))?;
        let path = dir.join(format!("{}.lock", f.sha256));
        fs::File::create(&path).with_context(|| format!("creating {}", path.display()))
    }

    /// The paths of `model`'s files, in order, or [`Missing`] for the first
    /// one that isn't there.
    pub fn locate(&self, model: &'static Model) -> Result<Vec<PathBuf>> {
        model
            .files
            .iter()
            .map(|f| {
                self.find(f)
                    .ok_or_else(|| Missing { model: model.name(), path: self.path(f), made: f.made }.into())
            })
            .collect()
    }

    /// Checks that every file of `model` is there, before anything loads it.
    /// A missing one is [`Missing`], with the command that fetches it,
    /// `<pull> <model>` ([`Missing::suggest`]).
    pub fn require(&self, model: &'static Model, pull: &str) -> Result<()> {
        match self.locate(model) {
            Ok(_) => Ok(()),
            Err(e) => match e.downcast_ref::<Missing>() {
                Some(missing) => bail!(missing.suggest(pull)),
                None => Err(e),
            },
        }
    }

    /// Gets each of `model`'s files that isn't there yet through `via`, and
    /// returns their paths as [`Store::locate`] does. A file is downloaded to
    /// `<file>.part` and takes its place only when its size and SHA-256 are
    /// the pinned ones; otherwise nothing is left behind. Two fetches of a
    /// file at once take turns, by its lock: the second finds it there when
    /// its turn comes. A file that is made here, not downloaded, fails saying
    /// how to make it.
    pub fn fetch(&self, model: &'static Model, via: &dyn Download) -> Result<Vec<PathBuf>> {
        self.fetch_with(model, via, &mut |_, _| {})
    }

    /// [`Store::fetch`], telling `progress` how many of the model's bytes
    /// are there and how many it has in all, as they come.
    pub fn fetch_with(
        &self,
        model: &'static Model,
        via: &dyn Download,
        progress: &mut dyn FnMut(u64, u64),
    ) -> Result<Vec<PathBuf>> {
        let total = model.bytes();
        let mut done: u64 = model.files.iter().filter(|f| self.find(f).is_some()).map(|f| f.bytes).sum();
        for f in model.files {
            if self.find(f).is_some() {
                continue;
            }
            if let Some(how) = f.made {
                bail!("`{}` of model `{}` can't be downloaded: {how}", f.name, model.name());
            }
            progress(done, total);
            self.fetch_file(f, via, &mut |bytes| progress(done + bytes, total))
                .with_context(|| format!("fetching model `{}`", model.name()))?;
            done += f.bytes;
            progress(done, total);
        }
        self.locate(model)
    }

    fn fetch_file(&self, f: &File, via: &dyn Download, progress: &mut dyn FnMut(u64)) -> Result<()> {
        let path = self.path(f);
        let lock = self.lock(f)?;
        lock.lock().context("waiting for another download of the same file")?;
        // Only now: a remove that held the lock may have taken the folder.
        let dir = path.parent().unwrap();
        fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        if path.is_file() {
            return Ok(());
        }
        let part = with_suffix(&path, ".part");
        let checked = (|| {
            download(via, &f.url(), &part, f.bytes, progress)?;
            let bytes = fs::metadata(&part).with_context(|| format!("reading {}", part.display()))?.len();
            if bytes != f.bytes {
                bail!("{} has {bytes} bytes, expected {}", f.url(), f.bytes);
            }
            let sum = sha256(&part)?;
            if sum != f.sha256 {
                bail!("{} has SHA-256 {sum}, expected {}", f.url(), f.sha256);
            }
            fs::rename(&part, &path).with_context(|| format!("moving it to {}", path.display()))
        })();
        if checked.is_err() {
            let _ = fs::remove_file(&part);
        }
        drop(lock);
        checked
    }

    /// What the store holds of each model it knows.
    pub fn list(&self) -> Result<Vec<Entry>> {
        self.models
            .iter()
            .copied()
            .map(|model| {
                let mut entry = Entry { model, files: 0, bytes: 0 };
                for f in model.files {
                    let path = self.path(f);
                    entry.files += usize::from(path.is_file());
                    for p in beside(&path)? {
                        entry.bytes += fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
                    }
                }
                Ok(entry)
            })
            .collect()
    }

    /// The rows of a product's `models list`: every model the store knows,
    /// what it weighs (on disk once pulled, else as pinned), whether it is
    /// pulled, and `used` for those in `used`.
    pub fn list_rows(&self, used: &[&'static Model]) -> Result<Vec<String>> {
        let width = self.names().iter().map(|n| n.len()).max().unwrap_or(0) + 2;
        let mut rows = Vec::new();
        for e in self.list()? {
            let state = match e.files {
                0 => "-",
                _ if e.complete() => "pulled",
                _ => "partial",
            };
            let size = gb(if e.bytes > 0 { e.bytes } else { e.model.bytes() });
            let used = if used.iter().any(|m| std::ptr::eq(*m, e.model)) { "used" } else { "" };
            rows.push(format!("{:width$}{size:>9}  {state:8}{used}", e.model.name()).trim_end().to_string());
        }
        Ok(rows)
    }

    /// Pulls `model` unless it is there, as a product's `models pull` does:
    /// says on stderr what it downloads, tells `progress` how far it has got
    /// as [`Store::fetch_with`] does, and says when it is done.
    pub fn pull(
        &self,
        model: &'static Model,
        via: &dyn Download,
        progress: &mut dyn FnMut(u64, u64),
    ) -> Result<()> {
        let missing: u64 = model.files.iter().filter(|f| self.find(f).is_none()).map(|f| f.bytes).sum();
        if missing == 0 {
            eprintln!("{}: in the store", model.name());
            return Ok(());
        }
        eprintln!("{}: downloading {}…", model.name(), gb(missing));
        self.fetch_with(model, via, progress)?;
        eprintln!("{}: done", model.name());
        Ok(())
    }

    /// Removes `model` as a product's `models rm` does: the bytes freed, or
    /// `None` when the store holds nothing of it.
    pub fn remove_if_present(&self, model: &Model) -> Result<Option<u64>> {
        let there = self.list()?.iter().any(|e| std::ptr::eq(e.model, model) && e.bytes > 0);
        if !there {
            return Ok(None);
        }
        self.remove(model).map(Some)
    }

    /// Removes `model`'s files from the store, with anything made beside
    /// them, and the folders that leaves empty. Returns the bytes freed. A
    /// model with a file being downloaded is refused, and kept whole.
    pub fn remove(&self, model: &Model) -> Result<u64> {
        let mut locks = Vec::new();
        for f in model.files {
            let lock = self.lock(f)?;
            match lock.try_lock() {
                Ok(()) => locks.push(lock),
                Err(fs::TryLockError::WouldBlock) => {
                    bail!("model `{}` is being downloaded: remove it once that is done", model.name())
                }
                Err(fs::TryLockError::Error(e)) => return Err(e).context("locking the model's files"),
            }
        }
        let mut freed = 0;
        for f in model.files {
            let path = self.path(f);
            for p in beside(&path)? {
                freed += fs::metadata(&p).map(|m| m.len()).unwrap_or(0);
                fs::remove_file(&p).with_context(|| format!("removing {}", p.display()))?;
            }
            let mut dir = path.parent();
            while let Some(d) = dir.filter(|d| d.starts_with(&self.root) && *d != self.root) {
                if fs::remove_dir(d).is_err() {
                    break;
                }
                dir = d.parent();
            }
        }
        Ok(freed)
    }
}

/// How often a download's progress is told.
const PROGRESS_EVERY: Duration = Duration::from_millis(100);

/// Downloads `url` to `to` through `via` on a thread of its own, telling
/// `progress` how much of `to` is there, up to `bytes`, as it grows.
fn download(
    via: &dyn Download,
    url: &str,
    to: &Path,
    bytes: u64,
    progress: &mut dyn FnMut(u64),
) -> Result<()> {
    thread::scope(|s| {
        let job = s.spawn(|| via.download(url, to));
        while !job.is_finished() {
            thread::sleep(PROGRESS_EVERY);
            if let Ok(m) = fs::metadata(to) {
                progress(m.len().min(bytes));
            }
        }
        job.join().unwrap_or_else(|_| bail!("downloading {url} panicked"))
    })
}

/// A model's size, as people read it: `0.70 GB`.
pub fn gb(bytes: u64) -> String {
    format!("{:.2} GB", bytes as f64 / 1e9)
}

/// Draws a model's download on stderr, in place, when stderr is a terminal:
/// `nemotron-en  0.31 / 0.70 GB  45%`. A [`Store::fetch_with`] `progress`.
pub fn draw_progress(model: &Model) -> impl FnMut(u64, u64) + use<> {
    let name = model.name();
    let terminal = std::io::stderr().is_terminal();
    let mut drawn = None;
    move |done, total| {
        let pct = (done * 100).checked_div(total).unwrap_or(100);
        if !terminal || drawn == Some(pct) {
            return;
        }
        drawn = Some(pct);
        let gb = |b: u64| b as f64 / 1e9;
        eprint!("\r\x1b[K{name}  {:.2} / {:.2} GB  {pct}%", gb(done), gb(total));
        if done >= total {
            eprintln!();
        }
    }
}

/// `path` with `suffix` added to its file name: `m.gguf` → `m.gguf.part`.
fn with_suffix(path: &Path, suffix: &str) -> PathBuf {
    let mut name = path.file_name().unwrap_or_default().to_os_string();
    name.push(suffix);
    path.with_file_name(name)
}

/// The file at `path`, if there, and what is made beside it: the files in its
/// folder whose names start with its own without the extension, then a dot
/// (`m.gguf.part`, `m.boost.gguf`).
fn beside(path: &Path) -> Result<Vec<PathBuf>> {
    let (Some(dir), Some(stem)) = (path.parent(), path.file_stem().and_then(|s| s.to_str())) else {
        return Ok(Vec::new());
    };
    let entries = match fs::read_dir(dir) {
        Ok(entries) => entries,
        Err(e) if e.kind() == std::io::ErrorKind::NotFound => return Ok(Vec::new()),
        Err(e) => return Err(e).with_context(|| format!("reading {}", dir.display())),
    };
    let prefix = format!("{stem}.");
    let mut found = Vec::new();
    for entry in entries {
        let entry = entry?;
        if entry.file_name().to_str().is_some_and(|n| n.starts_with(&prefix)) && entry.file_type()?.is_file()
        {
            found.push(entry.path());
        }
    }
    found.sort();
    Ok(found)
}

/// The SHA-256 of the file at `path`, in hex.
pub fn sha256(path: &Path) -> Result<String> {
    let mut f = fs::File::open(path).with_context(|| format!("opening {}", path.display()))?;
    let mut h = Sha256::new();
    std::io::copy(&mut f, &mut h).with_context(|| format!("reading {}", path.display()))?;
    Ok(h.finalize().iter().map(|b| format!("{b:02x}")).collect())
}

#[cfg(test)]
mod tests {
    use std::collections::BTreeMap;
    use std::sync::Mutex;
    use std::thread;

    use super::*;

    fn env(pairs: &[(&str, &str)]) -> impl Fn(&str) -> Option<OsString> {
        let vars: BTreeMap<String, OsString> = pairs.iter().map(|(k, v)| (k.to_string(), v.into())).collect();
        move |key| vars.get(key).cloned()
    }

    #[test]
    fn pam_models_chooses_the_store() {
        let s = Store::from_env(env(&[("PAM_MODELS", "/m"), ("HOME", "/h")])).unwrap();
        assert_eq!(s.root(), Path::new("/m"));

        let s = Store::from_env(env(&[("PAM_MODELS", ""), ("HOME", "/h")])).unwrap();
        assert_eq!(s.root(), Path::new("/h/Library/Application Support/pam/models"));

        let err = Store::from_env(env(&[])).unwrap_err().to_string();
        assert!(err.contains("PAM_MODELS"), "{err}");
    }

    #[test]
    fn a_missing_model_in_the_default_store_names_its_folder() {
        let store = Store::from_env(env(&[("HOME", "/nowhere")])).unwrap();
        let err = store.locate(&NEMOTRON_EN).unwrap_err().to_string();
        assert!(err.contains("/nowhere/Library/Application Support/pam/models/nvidia/"), "{err}");
    }

    /// A model of one small file, pinned to `bytes`.
    fn tiny(bytes: &'static [u8]) -> &'static Model {
        let sha256 = format!("{:x}", Sha256::digest(bytes)).leak();
        let file = File {
            repo: "org/tiny",
            revision: "abc",
            name: "tiny.gguf",
            sha256,
            bytes: bytes.len() as u64,
            made: None,
        };
        Box::leak(Box::new(Model { names: &["tiny"], files: Box::leak(Box::new([file])) }))
    }

    /// Writes the same bytes whatever is asked for, and counts the calls.
    struct Fake(&'static [u8], Mutex<u32>);

    impl Download for Fake {
        fn download(&self, _url: &str, to: &Path) -> Result<()> {
            *self.1.lock().unwrap() += 1;
            Ok(fs::write(to, self.0)?)
        }
    }

    #[test]
    fn a_missing_model_names_itself_and_the_path_looked_at() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let err = store.locate(&NEMOTRON_EN).unwrap_err();
        let missing = err.downcast_ref::<Missing>().expect("a Missing");
        assert_eq!(missing.model, "nemotron-en");
        assert_eq!(missing.path, dir.path().join(NEMOTRON_EN.files[0].rel()));
        let text = err.to_string();
        assert!(
            text.contains("`nemotron-en`") && text.contains(&missing.path.display().to_string()),
            "{text}"
        );
    }

    #[test]
    fn a_missing_model_is_required_with_the_command_that_fetches_it() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let err = store.require(&NEMOTRON_EN, "dictate models pull").unwrap_err().to_string();
        assert!(err.contains("isn't in the model store"), "{err}");
        assert!(err.ends_with(": run `dictate models pull nemotron-en`"), "{err}");
        let err = store.require(&QWEN3_ASR_1_7B, "dictate models pull").unwrap_err().to_string();
        assert!(err.contains("made, not downloaded") && !err.contains("models pull"), "{err}");
        store.fetch(tiny(b"weights"), &Fake(b"weights", Mutex::new(0))).unwrap();
        store.require(tiny(b"weights"), "pam models pull").unwrap();
    }

    #[test]
    fn a_fetched_file_is_checked_and_put_in_place() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let model = tiny(b"weights");
        let fake = Fake(b"weights", Mutex::new(0));
        let paths = store.fetch(model, &fake).unwrap();
        assert_eq!(paths, [dir.path().join("org/tiny/abc/tiny.gguf")]);
        assert_eq!(fs::read(&paths[0]).unwrap(), b"weights");
        let left: Vec<_> =
            fs::read_dir(paths[0].parent().unwrap()).unwrap().map(|e| e.unwrap().file_name()).collect();
        assert_eq!(left, ["tiny.gguf"], "nothing but the file beside it");
        // There: not downloaded again.
        store.fetch(model, &fake).unwrap();
        assert_eq!(*fake.1.lock().unwrap(), 1);
    }

    #[test]
    fn a_fetch_tells_how_far_it_has_got() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let model = tiny(b"weights");
        let mut told = Vec::new();
        store
            .fetch_with(model, &Fake(b"weights", Mutex::new(0)), &mut |done, total| told.push((done, total)))
            .unwrap();
        assert_eq!(told.first(), Some(&(0, 7)));
        assert_eq!(told.last(), Some(&(7, 7)));
        told.clear();
        store
            .fetch_with(model, &Fake(b"weights", Mutex::new(0)), &mut |done, total| told.push((done, total)))
            .unwrap();
        assert!(told.is_empty(), "nothing to fetch, nothing to tell: {told:?}");
    }

    #[test]
    fn a_backend_spec_names_the_model_it_loads() {
        let store = Store::at("/m");
        assert!(std::ptr::eq(store.model_for("nemo-speech:nemotron-en").unwrap().unwrap(), &NEMOTRON_EN));
        assert!(std::ptr::eq(store.model_for("qwen:qwen3-asr-1.7b").unwrap().unwrap(), &QWEN3_ASR_1_7B));
        assert!(store.model_for("fake:a.fake.jsonl").unwrap().is_none());
        assert!(store.model_for("nemo-speech:/models/asr.gguf").unwrap().is_none());
        let err = store.model_for("nemo-speech:whisper").unwrap_err().to_string();
        assert!(err.contains("`whisper`") && err.contains("nemotron-3-diarization"), "{err}");
        let store = store.with_models(Box::leak(Box::new([tiny(b"x")])));
        assert_eq!(store.names(), ["tiny"]);
        assert!(store.model_for("nemo-speech:nemotron-en").is_err(), "only the models it was given");
        assert_eq!(store.list().unwrap().len(), 1);
    }

    #[test]
    fn a_bad_checksum_leaves_no_file_behind() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let model = tiny(b"weights");
        // The right size, the wrong bytes.
        let err = store.fetch(model, &Fake(b"wrights", Mutex::new(0))).unwrap_err();
        assert!(format!("{err:#}").contains("SHA-256"), "{err:#}");
        let path = store.path(&model.files[0]);
        assert!(!path.exists() && !with_suffix(&path, ".part").exists());
        let err = store.fetch(model, &Fake(b"short", Mutex::new(0))).unwrap_err();
        assert!(format!("{err:#}").contains("bytes"), "{err:#}");
        assert!(!path.exists() && !with_suffix(&path, ".part").exists());
        assert!(store.locate(model).unwrap_err().is::<Missing>());
    }

    #[test]
    fn a_second_fetch_waits_its_turn_and_finds_the_file_there() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let model = tiny(b"weights");
        let path = store.path(&model.files[0]);
        fs::create_dir_all(path.parent().unwrap()).unwrap();
        // Another fetch, downloading: it holds the lock.
        let lock = store.lock(&model.files[0]).unwrap();
        lock.lock().unwrap();
        let fake: &'static Fake = Box::leak(Box::new(Fake(b"weights", Mutex::new(0))));
        let waiting =
            thread::spawn(move || Store::at(store.root).fetch_file(&model.files[0], fake, &mut |_| {}));
        thread::sleep(std::time::Duration::from_millis(200));
        assert!(!waiting.is_finished(), "it waits for the lock");
        fs::write(&path, "weights").unwrap();
        drop(lock);
        waiting.join().unwrap().unwrap();
        assert_eq!(*fake.1.lock().unwrap(), 0, "not downloaded again");
    }

    #[test]
    fn a_file_made_here_says_how_to_make_it() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let err = store.fetch(&QWEN3_ASR_1_7B, &Fake(b"", Mutex::new(0))).unwrap_err().to_string();
        assert!(err.contains("tools/qwen-gguf/make.sh"), "{err}");
        let err = store.locate(&QWEN3_ASR_1_7B).unwrap_err().to_string();
        assert!(err.contains("qwen3-asr-1.7b") && err.contains("made, not downloaded"), "{err}");
    }

    #[test]
    fn list_counts_what_is_made_beside_a_file_and_remove_takes_it_all() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let file = store.path(&NEMOTRON_EN.files[0]);
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        fs::write(&file, b"model").unwrap();
        fs::write(file.with_extension("boost.gguf"), b"boosted").unwrap();
        fs::write(with_suffix(&file, ".lock"), b"").unwrap();
        // Another model's file in the same folder isn't counted.
        fs::write(file.with_file_name("other.gguf"), b"x").unwrap();
        let listed = store.list().unwrap();
        let en = listed.iter().find(|e| e.model.name() == "nemotron-en").unwrap();
        assert!(en.complete());
        assert_eq!(en.bytes, 12);
        let qwen = listed.iter().find(|e| e.model.name() == "qwen3-asr-1.7b").unwrap();
        assert_eq!((qwen.files, qwen.bytes, qwen.complete()), (0, 0, false));

        assert_eq!(store.remove(&NEMOTRON_EN).unwrap(), 12);
        assert_eq!(fs::read_dir(file.parent().unwrap()).unwrap().count(), 1, "only other.gguf is left");
        fs::remove_file(file.with_file_name("other.gguf")).unwrap();
        fs::write(&file, b"model").unwrap();
        store.remove(&NEMOTRON_EN).unwrap();
        assert!(!dir.path().join("nvidia").exists(), "empty folders go too");
        assert!(dir.path().exists(), "the store stays");
    }

    #[test]
    fn a_fetch_waiting_while_its_folder_is_removed_makes_it_again() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let model = tiny(b"weights");
        // A remove, under way: it holds the lock.
        let lock = store.lock(&model.files[0]).unwrap();
        lock.lock().unwrap();
        let fake: &'static Fake = Box::leak(Box::new(Fake(b"weights", Mutex::new(0))));
        let root = store.root.clone();
        let waiting = thread::spawn(move || Store::at(root).fetch(model, fake));
        thread::sleep(std::time::Duration::from_millis(200));
        // What it removes, the model's folders included.
        let _ = fs::remove_dir_all(dir.path().join("org"));
        drop(lock);
        let paths = waiting.join().unwrap().unwrap();
        assert_eq!(fs::read(&paths[0]).unwrap(), b"weights");
    }

    #[test]
    fn a_model_being_downloaded_is_not_removed() {
        let dir = tempfile::tempdir().unwrap();
        let store = Store::at(dir.path());
        let file = store.path(&NEMOTRON_EN.files[0]);
        fs::create_dir_all(file.parent().unwrap()).unwrap();
        let part = with_suffix(&file, ".part");
        fs::write(&part, b"half").unwrap();
        // A fetch, downloading: it holds the lock.
        let lock = store.lock(&NEMOTRON_EN.files[0]).unwrap();
        lock.lock().unwrap();
        let err = store.remove(&NEMOTRON_EN).unwrap_err().to_string();
        assert!(err.contains("being downloaded"), "{err}");
        assert!(part.exists(), "the download is left alone");
        drop(lock);
        assert_eq!(store.remove(&NEMOTRON_EN).unwrap(), 4, "a stale part goes once no one downloads it");
    }

    #[test]
    fn every_file_is_pinned_by_a_whole_checksum() {
        for model in MODELS {
            for f in model.files {
                assert_eq!(f.sha256.len(), 64, "{}", f.name);
            }
        }
    }
}
