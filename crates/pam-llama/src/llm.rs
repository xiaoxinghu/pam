//! The llama.cpp side of [`Llm`](crate::Llm).

use std::ffi::CString;
use std::os::unix::ffi::OsStrExt;
use std::path::Path;

use anyhow::{Context, Result, bail};

use crate::backend::{self, last_error};
use crate::sys::{self, llama_seq_id, llama_token};
use crate::{Reply, Settings, chatml};

/// The sequence that keeps the system prompt between calls…
const SYSTEM: llama_seq_id = 0;
/// …and the one each call runs in: a copy of it, removed afterwards.
const CALL: llama_seq_id = 1;

pub struct Inner {
    model: *mut sys::llama_model,
    ctx: *mut sys::llama_context,
    vocab: *const sys::llama_vocab,
    batch: sys::llama_batch,
    batch_size: usize,
    settings: Settings,
    /// The tokens sequence [`SYSTEM`] holds.
    system: Vec<llama_token>,
}

// The pointers are only used through `&mut self`, by one thread at a time.
unsafe impl Send for Inner {}

impl Inner {
    pub fn load(path: &Path, settings: Settings) -> Result<Inner> {
        backend::init();
        let file = CString::new(path.as_os_str().as_bytes()).context("the model's path holds a NUL")?;
        let mut mp = unsafe { sys::llama_model_default_params() };
        mp.n_gpu_layers = -1;
        let model = unsafe { sys::llama_model_load_from_file(file.as_ptr(), mp) };
        if model.is_null() {
            bail!("llama.cpp couldn't load {}: {}", path.display(), last_error());
        }
        let mut cp = unsafe { sys::llama_context_default_params() };
        cp.n_ctx = settings.context;
        cp.n_seq_max = 2;
        cp.n_threads = settings.threads;
        cp.n_threads_batch = settings.threads;
        cp.flash_attn_type = sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED;
        // One buffer for both sequences, which share the system prompt.
        cp.kv_unified = true;
        cp.no_perf = true;
        let ctx = unsafe { sys::llama_init_from_model(model, cp) };
        if ctx.is_null() {
            unsafe { sys::llama_model_free(model) };
            bail!("llama.cpp couldn't make a context for {}: {}", path.display(), last_error());
        }
        let batch_size = unsafe { sys::llama_n_batch(ctx) } as usize;
        Ok(Inner {
            model,
            ctx,
            vocab: unsafe { sys::llama_model_get_vocab(model) },
            batch: unsafe { sys::llama_batch_init(batch_size as i32, 0, 1) },
            batch_size,
            settings,
            system: Vec::new(),
        })
    }

    /// Reads `system` into sequence [`SYSTEM`], unless it holds it already;
    /// says whether it did.
    pub fn remember(&mut self, system: &str) -> Result<bool> {
        let (head, _) = chatml(system, "");
        let head = self.tokenize(&head, true)?;
        if head == self.system {
            return Ok(true);
        }
        self.system.clear();
        unsafe { sys::llama_memory_clear(self.memory(), true) };
        self.decode(&head, 0, SYSTEM, false)?;
        self.system = head;
        Ok(false)
    }

    pub fn complete(&mut self, system: &str, user: &str, grammar: Option<&str>) -> Result<Reply> {
        let grammar = grammar.map(|g| Grammar::new(self.vocab, g)).transpose()?;
        let (_, tail) = chatml(system, user);
        let tail = self.tokenize(&tail, false)?;
        let start = self.system.len();
        let cached = if self.remember(system)? { start } else { 0 };
        let start = self.system.len();
        if start + tail.len() + self.settings.max_reply > self.settings.context as usize {
            bail!(
                "the prompt is {} tokens: with a reply, over the {} the context holds",
                start + tail.len(),
                self.settings.context
            );
        }
        unsafe { sys::llama_memory_seq_cp(self.memory(), SYSTEM, CALL, -1, -1) };
        let reply = self.reply(&tail, start, grammar);
        unsafe { sys::llama_memory_seq_rm(self.memory(), CALL, -1, -1) };
        let (text, written) = reply?;
        Ok(Reply { text, read: start - cached + tail.len(), cached, written })
    }

    /// Reads `prompt` after the system prompt, then writes the reply.
    fn reply(
        &mut self,
        prompt: &[llama_token],
        start: usize,
        mut grammar: Option<Grammar>,
    ) -> Result<(String, usize)> {
        self.decode(prompt, start, CALL, true)?;
        let n_vocab = unsafe { sys::llama_vocab_n_tokens(self.vocab) } as usize;
        let mut pos = start + prompt.len();
        let (mut bytes, mut written) = (Vec::new(), 0);
        while written < self.settings.max_reply {
            let logits = unsafe { sys::llama_get_logits_ith(self.ctx, -1) };
            if logits.is_null() {
                bail!("llama.cpp has no logits: {}", last_error());
            }
            let logits = unsafe { std::slice::from_raw_parts(logits, n_vocab) };
            let token = match &mut grammar {
                Some(grammar) => grammar.pick(logits)?,
                None => argmax(logits.iter().copied()).context("the model gave no logits")? as llama_token,
            };
            if unsafe { sys::llama_vocab_is_eog(self.vocab, token) } {
                break;
            }
            bytes.extend(self.piece(token));
            written += 1;
            self.decode(&[token], pos, CALL, true)?;
            pos += 1;
        }
        Ok((String::from_utf8_lossy(&bytes).into_owned(), written))
    }

    fn memory(&self) -> sys::llama_memory_t {
        unsafe { sys::llama_get_memory(self.ctx) }
    }

    /// `text`'s tokens, with the chat template's markers as the special
    /// tokens they are; `first` adds what the model puts at the start.
    fn tokenize(&self, text: &str, first: bool) -> Result<Vec<llama_token>> {
        let len = i32::try_from(text.len()).context("the prompt is too long")?;
        let mut tokens = vec![0; text.len() + 2];
        let mut n = unsafe {
            let max = tokens.len() as i32;
            sys::llama_tokenize(self.vocab, text.as_ptr().cast(), len, tokens.as_mut_ptr(), max, first, true)
        };
        if n < 0 {
            tokens.resize(n.unsigned_abs() as usize, 0);
            n = unsafe {
                let max = tokens.len() as i32;
                sys::llama_tokenize(
                    self.vocab,
                    text.as_ptr().cast(),
                    len,
                    tokens.as_mut_ptr(),
                    max,
                    first,
                    true,
                )
            };
        }
        if n < 0 {
            bail!("llama.cpp couldn't tokenize the prompt");
        }
        tokens.truncate(n as usize);
        Ok(tokens)
    }

    /// The bytes a token stands for; a character may span two tokens.
    fn piece(&self, token: llama_token) -> Vec<u8> {
        let mut buf = vec![0u8; 64];
        loop {
            let n = unsafe {
                sys::llama_token_to_piece(
                    self.vocab,
                    token,
                    buf.as_mut_ptr().cast(),
                    buf.len() as i32,
                    0,
                    false,
                )
            };
            if n >= 0 {
                buf.truncate(n as usize);
                return buf;
            }
            buf.resize(n.unsigned_abs() as usize, 0);
        }
    }

    /// Reads `tokens` into sequence `seq` from position `start`, keeping the
    /// last token's logits if `logits`.
    fn decode(
        &mut self,
        tokens: &[llama_token],
        start: usize,
        seq: llama_seq_id,
        logits: bool,
    ) -> Result<()> {
        let chunks = tokens.len().div_ceil(self.batch_size);
        for (c, chunk) in tokens.chunks(self.batch_size).enumerate() {
            let b = &mut self.batch;
            b.n_tokens = chunk.len() as i32;
            for (j, &t) in chunk.iter().enumerate() {
                unsafe {
                    *b.token.add(j) = t;
                    *b.pos.add(j) = (start + c * self.batch_size + j) as i32;
                    *b.n_seq_id.add(j) = 1;
                    **b.seq_id.add(j) = seq;
                    *b.logits.add(j) = (logits && c + 1 == chunks && j + 1 == chunk.len()) as i8;
                }
            }
            let r = unsafe { sys::llama_decode(self.ctx, self.batch) };
            if r != 0 {
                bail!("llama.cpp couldn't read the prompt ({r}): {}", last_error());
            }
        }
        Ok(())
    }
}

impl Drop for Inner {
    fn drop(&mut self) {
        unsafe {
            sys::llama_batch_free(self.batch);
            sys::llama_free(self.ctx);
            sys::llama_model_free(self.model);
        }
    }
}

/// A GBNF grammar the reply is held to.
struct Grammar(*mut sys::llama_sampler);

impl Grammar {
    fn new(vocab: *const sys::llama_vocab, gbnf: &str) -> Result<Grammar> {
        let text = CString::new(gbnf).context("the grammar holds a NUL")?;
        let sampler = unsafe { sys::llama_sampler_init_grammar(vocab, text.as_ptr(), c"root".as_ptr()) };
        if sampler.is_null() {
            bail!("the reply's grammar doesn't parse: {}", last_error());
        }
        Ok(Grammar(sampler))
    }

    /// The likeliest token the grammar allows next, taken into the grammar.
    /// Like llama.cpp's own sampler, it asks the grammar about the likeliest
    /// token first, and filters the whole vocabulary only when that one is
    /// refused: the same token, much faster.
    fn pick(&mut self, logits: &[f32]) -> Result<llama_token> {
        let best = argmax(logits.iter().copied()).context("the model gave no logits")?;
        let mut one = [sys::llama_token_data { id: best as llama_token, logit: logits[best], p: 0.0 }];
        self.apply(&mut one);
        let token = if one[0].logit.is_finite() {
            best as llama_token
        } else {
            let mut all: Vec<sys::llama_token_data> = (0..logits.len())
                .map(|i| sys::llama_token_data { id: i as llama_token, logit: logits[i], p: 0.0 })
                .collect();
            self.apply(&mut all);
            let i = argmax(all.iter().map(|t| t.logit)).filter(|&i| all[i].logit.is_finite());
            all[i.context("the grammar allows no token here")?].id
        };
        unsafe { sys::llama_sampler_accept(self.0, token) };
        Ok(token)
    }

    /// Sets the logit of each token the grammar refuses to minus infinity.
    fn apply(&mut self, tokens: &mut [sys::llama_token_data]) {
        let mut array = sys::llama_token_data_array {
            data: tokens.as_mut_ptr(),
            size: tokens.len(),
            selected: -1,
            sorted: false,
        };
        unsafe { sys::llama_sampler_apply(self.0, &mut array) };
    }
}

impl Drop for Grammar {
    fn drop(&mut self) {
        unsafe { sys::llama_sampler_free(self.0) };
    }
}

fn argmax(values: impl Iterator<Item = f32>) -> Option<usize> {
    values
        .enumerate()
        .fold(None, |best, (i, v)| match best {
            Some((_, b)) if b >= v => best,
            _ => Some((i, v)),
        })
        .map(|(i, _)| i)
}
