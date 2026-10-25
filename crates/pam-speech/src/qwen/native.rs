//! The runtimes: Qwen3-ASR through llama.cpp's C API (`llama.h`, `mtmd.h`),
//! and Qwen3-ForcedAligner through a C shim over qwen3-asr.cpp
//! (`shim/shim.cpp`). Both log through
//! ggml's logger, which pam-llama sets up for the process and which keeps
//! only each thread's last error, to say why a call failed: anything else
//! would draw over the dashboard.

use std::ffi::{CStr, CString, c_char, c_int};
use std::os::unix::ffi::OsStrExt;
use std::path::Path;
use std::ptr;
use std::sync::Once;

use anyhow::{Context, bail};
use pam_llama::backend::{self, last_error};

use super::{Asr, Timer};
use crate::Result;

#[allow(non_camel_case_types, non_upper_case_globals, non_snake_case, dead_code, unsafe_op_in_unsafe_fn)]
#[allow(clippy::upper_case_acronyms)]
mod sys {
    include!(concat!(env!("OUT_DIR"), "/qwen_llama.rs"));
}

/// Threads for what runs on the CPU, as 01 and 02 measured.
const ASR_THREADS: i32 = 8;
const ALIGNER_THREADS: c_int = 4;
/// Tokens the context holds: the prompt, a 20 s clip's ~250 audio tokens and the output.
const CONTEXT: u32 = 4096;

static INIT: Once = Once::new();

/// llama.cpp's set-up, through pam-llama, which owns it for the process
/// (the local intent check uses it too); then `mtmd`'s two loggers, which
/// only this backend uses, to the same logger.
fn init() {
    backend::init();
    INIT.call_once(|| unsafe {
        sys::mtmd_log_set(Some(backend::log), ptr::null_mut());
        sys::mtmd_helper_log_set(Some(backend::log), ptr::null_mut());
    });
}

fn c_path(path: &Path) -> Result<CString> {
    CString::new(path.as_os_str().as_bytes()).context("a model's path holds a NUL")
}

/// Qwen3-ASR's decoder and audio projector, loaded once.
pub struct Llama {
    model: *mut sys::llama_model,
    ctx: *mut sys::llama_context,
    mctx: *mut sys::mtmd_context,
    vocab: *const sys::llama_vocab,
    batch: *mut sys::llama_batch_ext,
    n_batch: i32,
    n_vocab: usize,
}

// The pointers are only used through `&mut self`, by one thread at a time.
unsafe impl Send for Llama {}

impl Llama {
    /// Loads the decoder `model` and its audio projector `mmproj`, every layer on the GPU.
    pub fn load(model: &Path, mmproj: &Path) -> Result<Llama> {
        init();
        let path = c_path(model)?;
        let mut mp = unsafe { sys::llama_model_default_params() };
        mp.n_gpu_layers = -1;
        let m = unsafe { sys::llama_model_load_from_file(path.as_ptr(), mp) };
        if m.is_null() {
            bail!("llama.cpp couldn't load {}: {}", model.display(), last_error());
        }
        let mut cp = unsafe { sys::llama_context_default_params() };
        cp.n_ctx = CONTEXT;
        cp.n_batch = 2048;
        cp.n_threads = ASR_THREADS;
        cp.n_threads_batch = ASR_THREADS;
        cp.flash_attn_type = sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED;
        cp.no_perf = true;
        let ctx = unsafe { sys::llama_init_from_model(m, cp) };
        if ctx.is_null() {
            unsafe { sys::llama_model_free(m) };
            bail!("llama.cpp couldn't make a context for {}: {}", model.display(), last_error());
        }
        let proj = c_path(mmproj)?;
        let mut pp = unsafe { sys::mtmd_context_params_default() };
        pp.use_gpu = true;
        pp.n_threads = ASR_THREADS;
        pp.print_timings = false;
        pp.warmup = true;
        pp.flash_attn_type = sys::llama_flash_attn_type::LLAMA_FLASH_ATTN_TYPE_ENABLED;
        let mctx = unsafe { sys::mtmd_init_from_file(proj.as_ptr(), m, pp) };
        if mctx.is_null() {
            unsafe {
                sys::llama_free(ctx);
                sys::llama_model_free(m);
            }
            bail!("llama.cpp couldn't load the audio projector {}: {}", mmproj.display(), last_error());
        }
        let vocab = unsafe { sys::llama_model_get_vocab(m) };
        Ok(Llama {
            model: m,
            ctx,
            mctx,
            vocab,
            batch: unsafe { sys::llama_batch_ext_init(ctx) },
            n_batch: unsafe { sys::llama_n_batch(ctx) } as i32,
            n_vocab: unsafe { sys::llama_vocab_n_tokens(vocab) } as usize,
        })
    }

    fn tokenize(&self, text: &str) -> Result<Vec<i32>> {
        let mut out = vec![0i32; text.len() + 8];
        let n = unsafe {
            sys::llama_tokenize(
                self.vocab,
                text.as_ptr().cast(),
                text.len() as i32,
                out.as_mut_ptr(),
                out.len() as i32,
                false,
                true,
            )
        };
        if n < 0 {
            bail!("llama.cpp couldn't tokenize `{text}`");
        }
        out.truncate(n as usize);
        Ok(out)
    }

    /// The text of `tokens`, special tokens such as `<asr_text>` included.
    fn detokenize(&self, tokens: &[i32]) -> String {
        let mut bytes = Vec::new();
        let mut buf = vec![0u8; 64];
        for &t in tokens {
            loop {
                let n = unsafe {
                    sys::llama_token_to_piece(
                        self.vocab,
                        t,
                        buf.as_mut_ptr().cast(),
                        buf.len() as i32,
                        0,
                        true,
                    )
                };
                if n >= 0 {
                    bytes.extend_from_slice(&buf[..n as usize]);
                    break;
                }
                buf.resize(n.unsigned_abs() as usize, 0);
            }
        }
        String::from_utf8_lossy(&bytes).into_owned()
    }

    /// Reads `tokens` from `pos` on, keeping the last one's logits.
    fn decode(&self, tokens: &[i32], pos: &mut i32) -> Result<()> {
        for (c, part) in tokens.chunks(self.n_batch as usize).enumerate() {
            let last_part = (c + 1) * self.n_batch as usize >= tokens.len();
            unsafe { sys::llama_batch_ext_clear(self.batch) };
            let mut idx = -1;
            for &t in part {
                idx = unsafe { sys::llama_batch_ext_add_token(self.batch, 0, t) };
                let p = *pos;
                unsafe { sys::llama_batch_ext_set_pos(self.batch, idx, &p) };
                *pos += 1;
            }
            if last_part {
                unsafe { sys::llama_batch_ext_set_output_logits(self.batch, idx, true) };
            }
            let r = unsafe {
                sys::llama_process(self.ctx, sys::llama_process_type::LLAMA_PROCESS_TYPE_DECODE, self.batch)
            };
            if r != 0 {
                bail!("llama.cpp couldn't decode ({r}): {}", last_error());
            }
        }
        Ok(())
    }

    /// The likeliest next token.
    fn argmax(&self) -> Result<i32> {
        let p = unsafe { sys::llama_get_logits_ith(self.ctx, -1) };
        if p.is_null() {
            bail!("llama.cpp gave no logits");
        }
        let logits = unsafe { std::slice::from_raw_parts(p, self.n_vocab) };
        let best = (0..logits.len()).fold(0, |best, i| if logits[i] > logits[best] { i } else { best });
        Ok(best as i32)
    }

    /// Qwen's chat prompt with `audio` as its user message, read into a cleared memory.
    fn prompt(&self, audio: &[f32], context: &str, logits: bool) -> Result<i32> {
        unsafe { sys::llama_memory_clear(sys::llama_get_memory(self.ctx), true) };
        let marker = unsafe { CStr::from_ptr(sys::mtmd_default_marker()) }.to_str()?;
        let prompt = format!(
            "<|im_start|>system\n{context}<|im_end|>\n<|im_start|>user\n{marker}<|im_end|>\n<|im_start|>assistant\n"
        );
        let text = CString::new(prompt).context("a hint holds a NUL")?;
        let input = sys::mtmd_input_text {
            text: text.as_ptr(),
            text_len: text.as_bytes().len(),
            add_special: false,
            parse_special: true,
        };
        let bitmap = unsafe { sys::mtmd_bitmap_init_from_audio(audio.len(), audio.as_ptr()) };
        let chunks = unsafe { sys::mtmd_input_chunks_init() };
        let bitmaps = [bitmap as *const sys::mtmd_bitmap];
        let mut r = unsafe { sys::mtmd_tokenize(self.mctx, chunks, &input, bitmaps.as_ptr(), 1) };
        unsafe { sys::mtmd_bitmap_free(bitmap) };
        let mut pos = 0;
        if r == 0 {
            r = unsafe {
                sys::mtmd_helper_eval_chunks(
                    self.mctx,
                    self.ctx,
                    chunks,
                    0,
                    0,
                    self.n_batch,
                    logits,
                    &mut pos,
                )
            };
        }
        unsafe { sys::mtmd_input_chunks_free(chunks) };
        if r != 0 {
            bail!("llama.cpp couldn't read a clip ({r}): {}", last_error());
        }
        Ok(pos)
    }
}

impl Asr for Llama {
    fn generate(
        &mut self,
        audio: &[f32],
        context: &str,
        language: Option<&str>,
        max_tokens: usize,
    ) -> Result<String> {
        let forced = language.map(|l| self.tokenize(&format!("language {l}<asr_text>"))).transpose()?;
        let mut pos = self.prompt(audio, context, forced.is_none())?;
        if let Some(forced) = &forced {
            self.decode(forced, &mut pos)?;
        }
        let mut out = Vec::new();
        while out.len() < max_tokens {
            let token = self.argmax()?;
            if unsafe { sys::llama_vocab_is_eog(self.vocab, token) } {
                break;
            }
            out.push(token);
            self.decode(&[token], &mut pos)?;
        }
        Ok(self.detokenize(&out))
    }
}

impl Drop for Llama {
    fn drop(&mut self) {
        // Freed before exit: ggml's Metal device asserts if buffers outlive it.
        unsafe {
            sys::llama_batch_ext_free(self.batch);
            sys::mtmd_free(self.mctx);
            sys::llama_free(self.ctx);
            sys::llama_model_free(self.model);
        }
    }
}

#[repr(C)]
struct FaUnit {
    text: *const c_char,
    start: f32,
    end: f32,
}

#[repr(C)]
struct Fa {
    _p: [u8; 0],
}

unsafe extern "C" {
    fn pam_fa_open(model: *const c_char, n_threads: c_int) -> *mut Fa;
    fn pam_fa_align(
        a: *mut Fa,
        samples: *const f32,
        n: c_int,
        units: *const c_char,
        out: *mut *const FaUnit,
    ) -> c_int;
    fn pam_fa_error(a: *const Fa) -> *const c_char;
    fn pam_fa_close(a: *mut Fa);
}

/// Qwen3-ForcedAligner, loaded once.
pub struct ForcedAligner(*mut Fa);

// Only used through `&mut self`, by one thread at a time.
unsafe impl Send for ForcedAligner {}

impl ForcedAligner {
    pub fn load(model: &Path) -> Result<ForcedAligner> {
        init();
        let path = c_path(model)?;
        let fa = unsafe { pam_fa_open(path.as_ptr(), ALIGNER_THREADS) };
        if fa.is_null() {
            bail!("qwen3-asr.cpp couldn't load the aligner {}: {}", model.display(), last_error());
        }
        Ok(ForcedAligner(fa))
    }
}

impl Timer for ForcedAligner {
    fn time(&mut self, audio: &[f32], units: &[String]) -> Result<Vec<(f32, f32)>> {
        let text = CString::new(units.join(" ")).context("a word holds a NUL")?;
        let mut out: *const FaUnit = ptr::null();
        let n =
            unsafe { pam_fa_align(self.0, audio.as_ptr(), audio.len() as c_int, text.as_ptr(), &mut out) };
        if n < 0 {
            let why = unsafe { CStr::from_ptr(pam_fa_error(self.0)) }.to_string_lossy();
            bail!("the aligner failed: {why}");
        }
        let got = if n == 0 { &[][..] } else { unsafe { std::slice::from_raw_parts(out, n as usize) } };
        let same = got.len() == units.len()
            && got
                .iter()
                .zip(units)
                .all(|(g, u)| unsafe { CStr::from_ptr(g.text) }.to_bytes() == u.as_bytes());
        if !same {
            bail!("the aligner timed other units than the {} it was given", units.len());
        }
        Ok(got.iter().map(|g| (g.start, g.end)).collect())
    }
}

impl Drop for ForcedAligner {
    fn drop(&mut self) {
        unsafe { pam_fa_close(self.0) };
    }
}
