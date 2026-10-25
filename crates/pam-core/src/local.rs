//! The local intent check: a small model, pinned, run in-process by
//! llama.cpp (`pam-llama`) behind the same [`Model`] seam as the cloud one.
//!
//! The model file is kept in the model store (`pam_speech::store`), which
//! `pam models pull` fetches it into, checked against its SHA-256. [`Local::start`] loads it on a thread of its own, so it loads while
//! the speech models do, and reads the system prompt into the model's memory
//! there, so the first check needn't.

use std::path::PathBuf;
use std::sync::Mutex;
use std::thread::{self, JoinHandle};

use anyhow::{Result, anyhow};
use pam_llama::{Llm, Settings};
use pam_speech::store::{self, Store};

use crate::llm::{Completion, Model, Request};
use crate::usage::{Call, Used};

/// Qwen3.5 4B, Q4_K_M: the smallest model found that does the check,
/// pinned in the model store.
pub static MODEL: &store::Model = &store::QWEN3_5_4B;

/// As pi's `--model` names a cloud model: what `meeting.json` records.
pub fn id() -> String {
    format!("local/{}", MODEL.name())
}

/// Where the model is in the model store, or would be.
pub fn path() -> Result<PathBuf> {
    let store = Store::open()?;
    let file = &MODEL.files[0];
    Ok(store.find(file).unwrap_or_else(|| store.path(file)))
}

/// A local model as a [`Model`]: one call at a time, its reply held to the
/// request's grammar. A call counts its tokens, at no cost.
pub struct Local {
    state: Mutex<State>,
}

enum State {
    Loading(JoinHandle<Result<Llm>>),
    Ready(Llm),
    Failed(String),
}

impl Local {
    /// Starts loading the model at `path` on a thread of its own, and reads
    /// `system` into its memory once loaded, so the first call needn't.
    pub fn start(path: PathBuf, system: String) -> Local {
        let load = thread::Builder::new().name("pam-local-load".into()).spawn(move || {
            let mut llm = Llm::load(&path, Settings::default())?;
            llm.remember(&system)?;
            Ok(llm)
        });
        let state = match load {
            Ok(handle) => State::Loading(handle),
            Err(e) => State::Failed(format!("starting the model's loader: {e}")),
        };
        Local { state: Mutex::new(state) }
    }

    /// Waits for the model to load; `Err` says why it didn't.
    fn loaded(state: &mut State) -> Result<&mut Llm> {
        if let State::Loading(_) = state {
            let State::Loading(handle) = std::mem::replace(state, State::Failed(String::new())) else {
                unreachable!()
            };
            *state = match handle.join() {
                Ok(Ok(llm)) => State::Ready(llm),
                Ok(Err(e)) => State::Failed(format!("{e:#}")),
                Err(_) => State::Failed("loading the model crashed".into()),
            };
        }
        match state {
            State::Ready(llm) => Ok(llm),
            State::Failed(why) => Err(anyhow!("{why}")),
            State::Loading(_) => unreachable!(),
        }
    }
}

impl Model for Local {
    fn ready(&self) -> Result<()> {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        Local::loaded(&mut state).map(|_| ())
    }

    fn complete(&self, req: &Request<'_>) -> Completion {
        let mut state = self.state.lock().unwrap_or_else(|p| p.into_inner());
        let reply =
            Local::loaded(&mut state).and_then(|llm| llm.complete(req.system, req.prompt, req.grammar));
        match reply {
            Ok(r) => Completion {
                reply: Ok(r.text),
                used: Used::Call(Call {
                    input: r.read as u64,
                    output: r.written as u64,
                    cache_read: r.cached as u64,
                    cache_write: 0,
                    cost: Some(0.0),
                }),
            },
            Err(e) => Completion { reply: Err(e), used: Used::Nothing },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_model_is_kept_by_repository_and_revision() {
        let file = &MODEL.files[0];
        assert_eq!(
            file.rel(),
            std::path::Path::new("unsloth/Qwen3.5-4B-GGUF/e87f176479d0/Qwen3.5-4B-Q4_K_M.gguf")
        );
        assert_eq!(id(), "local/qwen3.5-4b");
    }

    #[cfg(not(feature = "llama-cpp"))]
    #[test]
    fn without_llama_cpp_a_check_fails_saying_how_to_build_it() {
        let local = Local::start("model.gguf".into(), "system".into());
        let e = local.ready().unwrap_err();
        assert!(e.to_string().contains("--features llama-cpp"), "{e}");
        let done = local.complete(&Request::new("system", "prompt").with_grammar("root ::= \"x\""));
        assert!(done.reply.is_err());
        assert_eq!(done.used, Used::Nothing);
    }
}
