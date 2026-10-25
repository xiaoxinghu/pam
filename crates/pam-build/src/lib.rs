//! What pam-llama's and pam-speech's build scripts share: checkouts of the
//! native libraries' sources at pinned revisions, and their CMake builds.
//!
//! Both are kept in `~/Library/Caches/pam`, where every checkout and worktree
//! of pam finds them. Only developers need that folder: the libraries built
//! there are linked into the binaries statically.

use std::fs::File;
use std::path::{Path, PathBuf};
use std::process::Command;

/// `~/Library/Caches/pam`.
pub fn cache() -> PathBuf {
    let home = std::env::var_os("HOME").expect("HOME is not set");
    Path::new(&home).join("Library/Caches/pam")
}

/// The macOS release the native libraries are built for: the one rustc links
/// for, `MACOSX_DEPLOYMENT_TARGET` if set, else its default.
pub fn deployment_target() -> String {
    println!("cargo:rerun-if-env-changed=MACOSX_DEPLOYMENT_TARGET");
    if let Some(version) = std::env::var("MACOSX_DEPLOYMENT_TARGET").ok().filter(|v| !v.is_empty()) {
        return version;
    }
    let rustc = std::env::var("RUSTC").unwrap_or_else(|_| "rustc".into());
    let target = std::env::var("TARGET").expect("cargo sets TARGET");
    let out = run(Command::new(rustc).args(["--print", "deployment-target", "--target", &target]));
    out.trim().strip_prefix("MACOSX_DEPLOYMENT_TARGET=").expect("rustc prints the deployment target").into()
}

/// A source tree: `$var` if non-empty, else a checkout of `rev` from `repo`
/// in the cache, fetched by git the first time it is needed. `probe` is a
/// file the tree must hold.
pub fn source(var: &str, name: &str, repo: &str, rev: &str, probe: &str) -> PathBuf {
    println!("cargo:rerun-if-env-changed={var}");
    // Empty counts as unset.
    if let Some(dir) = std::env::var_os(var).filter(|d| !d.is_empty()) {
        let dir = PathBuf::from(dir);
        assert!(dir.join(probe).exists(), "{var} has no {probe}");
        // Your own checkout: an edit to it is rebuilt.
        println!("cargo:rerun-if-changed={}", dir.display());
        return dir;
    }
    let cache = cache();
    let dir = cache.join(format!("{name}-{}", &rev[..12]));
    println!("cargo:rerun-if-changed={}", dir.join(probe).display());
    if dir.join(probe).exists() {
        return dir;
    }
    println!("cargo:warning=fetching {name} {} into {}", &rev[..12], cache.display());
    // Fetched beside it under a unique name, then renamed into place, so a
    // failed or concurrent build never leaves half a checkout.
    let tmp = cache.join(format!(".{name}.{}", std::process::id()));
    std::fs::create_dir_all(&tmp).unwrap_or_else(|e| panic!("creating {}: {e}", tmp.display()));
    let git = |args: &[&str]| run(Command::new("git").arg("-C").arg(&tmp).args(args));
    git(&["init", "-q"]);
    git(&["fetch", "-q", "--depth", "1", repo, rev]);
    git(&["checkout", "-q", "FETCH_HEAD"]);
    // Git names a revision by its content: this is the pinned source.
    assert_eq!(git(&["rev-parse", "HEAD"]).trim(), rev);
    // Another build may have finished first; its copy is as good as ours.
    if std::fs::rename(&tmp, &dir).is_err() && !dir.join(probe).exists() {
        panic!("could not move {name} into {}", dir.display());
    }
    let _ = std::fs::remove_dir_all(&tmp);
    dir
}

/// A static Release build with CMake and Ninja, for arm64 at
/// [`deployment_target`], in a folder of the cache held locked while this
/// lives, so two builds never run in it at once.
pub struct CMake {
    dir: PathBuf,
    _lock: File,
}

impl CMake {
    /// Configures `src` with `args`, unless its folder already is. The folder
    /// is `build/<name>-<hash>`, hashed from the source's path and the
    /// arguments, so a change to either gets a folder of its own.
    pub fn configure(name: &str, src: &Path, args: &[&str]) -> CMake {
        let target = deployment_target();
        let mut args: Vec<String> = args.iter().map(|a| a.to_string()).collect();
        args.extend([
            "-DCMAKE_BUILD_TYPE=Release".into(),
            "-DBUILD_SHARED_LIBS=OFF".into(),
            "-DCMAKE_OSX_ARCHITECTURES=arm64".into(),
            format!("-DCMAKE_OSX_DEPLOYMENT_TARGET={target}"),
        ]);
        let key = fnv1a(std::iter::once(src.to_string_lossy().into_owned()).chain(args.iter().cloned()));
        let builds = cache().join("build");
        let dir = builds.join(format!("{name}-{key:016x}"));
        let lock = lock(&dir);
        if !dir.join("build.ninja").exists() {
            run(Command::new("cmake").args(["-G", "Ninja", "-S"]).arg(src).arg("-B").arg(&dir).args(&args));
        }
        CMake { dir, _lock: lock }
    }

    /// Builds `targets`; Ninja skips what is up to date.
    pub fn build(&self, targets: &[&str]) {
        let jobs = std::env::var("NUM_JOBS").unwrap_or_else(|_| "8".into());
        run(Command::new("cmake")
            .arg("--build")
            .arg(&self.dir)
            .args(["-j", &jobs, "--target"])
            .args(targets));
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }
}

/// Locks `<path>.lock` until the file returned is dropped, so two builds
/// never write `path` at once.
pub fn lock(path: &Path) -> File {
    let lock = PathBuf::from(format!("{}.lock", path.display()));
    if let Some(parent) = lock.parent() {
        std::fs::create_dir_all(parent).unwrap_or_else(|e| panic!("creating {}: {e}", parent.display()));
    }
    let file = File::create(&lock).unwrap_or_else(|e| panic!("creating {}: {e}", lock.display()));
    file.lock().unwrap_or_else(|e| panic!("locking {}: {e}", lock.display()));
    file
}

/// Runs a command and returns its stdout, failing the build with its output
/// if it fails.
pub fn run(cmd: &mut Command) -> String {
    let out = cmd.output().unwrap_or_else(|e| panic!("running {cmd:?}: {e}"));
    assert!(
        out.status.success(),
        "{cmd:?} failed:\n{}{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// FNV-1a over `parts`, each ended by a NUL: a key that stays the same
/// across Rust releases, unlike std's hasher.
fn fnv1a(parts: impl IntoIterator<Item = String>) -> u64 {
    let mut hash = 0xcbf29ce484222325u64;
    for part in parts {
        for byte in part.bytes().chain([0]) {
            hash = (hash ^ byte as u64).wrapping_mul(0x100000001b3);
        }
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_key_tells_the_parts_apart() {
        let key = |parts: &[&str]| fnv1a(parts.iter().map(|p| p.to_string()));
        assert_eq!(key(&["a", "-DX=1"]), key(&["a", "-DX=1"]));
        assert_ne!(key(&["a", "-DX=1"]), key(&["a", "-DX=0"]));
        assert_ne!(key(&["ab", "c"]), key(&["a", "bc"]));
    }
}
