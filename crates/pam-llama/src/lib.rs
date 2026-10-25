//! A small language model run in-process through llama.cpp's C API, for the
//! local intent check.
//!
//! [`Llm`] loads one GGUF model and answers one prompt at a time, greedily,
//! with the reply held to a GBNF grammar. The system prompt stays in the
//! model's memory between calls, so each call only reads its own part.
//! It knows nothing about meetings: `pam-core` gives it the prompts and the
//! grammar. Without the `llama-cpp` feature, [`Llm::load`] says how to build
//! with it.

use std::path::Path;

pub use anyhow::Result;

#[cfg(feature = "llama-cpp")]
pub mod backend;
#[cfg(feature = "llama-cpp")]
mod llm;
#[cfg(feature = "llama-cpp")]
mod sys;

/// How the model is run.
#[derive(Debug, Clone)]
pub struct Settings {
    /// Tokens the context holds: the system prompt, one call's prompt and its reply.
    pub context: u32,
    /// CPU threads, for what the GPU doesn't do.
    pub threads: i32,
    /// A reply stops after this many tokens.
    pub max_reply: usize,
}

impl Default for Settings {
    /// As the intent check was measured with llama-server (`-c 4096`, 96 tokens a
    /// reply), with llama.cpp's default threads; every layer on the GPU.
    fn default() -> Settings {
        Settings { context: 4096, threads: 4, max_reply: 96 }
    }
}

/// What one call came back with.
#[derive(Debug, Clone, PartialEq)]
pub struct Reply {
    pub text: String,
    /// Prompt tokens the model read for this call.
    pub read: usize,
    /// Prompt tokens it already had: the system prompt, after the first call.
    pub cached: usize,
    /// Tokens in the reply.
    pub written: usize,
}

/// The prompt as the pinned Qwen3.5 model's chat template renders a system
/// and a user message with thinking off, split where the system part ends:
/// the first part is what stays in memory between calls.
pub fn chatml(system: &str, user: &str) -> (String, String) {
    let head = format!("<|im_start|>system\n{}<|im_end|>\n<|im_start|>user\n", system.trim());
    let tail = format!("{}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", user.trim());
    (head, tail)
}

/// A loaded model and its context.
pub struct Llm {
    #[cfg(feature = "llama-cpp")]
    inner: llm::Inner,
}

impl Llm {
    /// Loads the GGUF model at `path`, every layer on the GPU.
    pub fn load(path: &Path, settings: Settings) -> Result<Llm> {
        #[cfg(feature = "llama-cpp")]
        return Ok(Llm { inner: llm::Inner::load(path, settings)? });
        #[cfg(not(feature = "llama-cpp"))]
        {
            let _ = (path, settings);
            anyhow::bail!("this pam was built without llama.cpp: build it with `--features llama-cpp`")
        }
    }

    /// Reads `system` now, so the first call needn't. Calls with another
    /// system prompt replace it.
    pub fn remember(&mut self, system: &str) -> Result<()> {
        #[cfg(feature = "llama-cpp")]
        return self.inner.remember(system).map(|_| ());
        #[cfg(not(feature = "llama-cpp"))]
        {
            let _ = system;
            unreachable!("an Llm is never loaded without llama-cpp")
        }
    }

    /// The model's reply to `user` after `system`, held to `grammar` (GBNF,
    /// from its rule `root`) when one is given, else the likeliest tokens.
    pub fn complete(&mut self, system: &str, user: &str, grammar: Option<&str>) -> Result<Reply> {
        #[cfg(feature = "llama-cpp")]
        return self.inner.complete(system, user, grammar);
        #[cfg(not(feature = "llama-cpp"))]
        {
            let _ = (system, user, grammar);
            unreachable!("an Llm is never loaded without llama-cpp")
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_system_prompt_is_a_prefix_of_every_prompt() {
        let (head, tail) = chatml(" Be brief.\n", "RECENT LINES\n(none)\n");
        assert_eq!(head, "<|im_start|>system\nBe brief.<|im_end|>\n<|im_start|>user\n");
        assert_eq!(tail, "RECENT LINES\n(none)<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    }

    #[cfg(not(feature = "llama-cpp"))]
    #[test]
    fn without_the_feature_loading_says_how_to_get_it() {
        let e = Llm::load(Path::new("model.gguf"), Settings::default()).err().unwrap();
        assert!(e.to_string().contains("--features llama-cpp"), "{e}");
    }
}
