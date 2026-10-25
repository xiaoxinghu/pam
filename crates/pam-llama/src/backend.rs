//! llama.cpp's process-wide set-up, owned here for every user of the
//! framework in Pam: [`Llm`](crate::Llm) for the local intent check, and
//! pam-speech's `qwen` backend. llama.cpp asks for `llama_backend_init` once
//! per program, and its logger is global, so both go through [`init`].

use std::cell::RefCell;
use std::ffi::{CStr, c_char, c_void};
use std::ptr;
use std::sync::Once;

use crate::sys;

thread_local! {
    /// The last error llama.cpp logged on this thread, to say why a call
    /// failed. Per thread, so models loading side by side don't read each
    /// other's. Its log goes nowhere else: it would draw over the dashboard.
    static LAST_ERROR: RefCell<String> = const { RefCell::new(String::new()) };
}

static INIT: Once = Once::new();

/// Sets llama.cpp's (and ggml's) logger to [`log`] and initializes its
/// backend, once per process, whoever asks first.
pub fn init() {
    INIT.call_once(|| unsafe {
        sys::llama_log_set(Some(log), ptr::null_mut());
        sys::llama_backend_init();
    });
}

/// The logger: keeps errors for [`last_error`], drops the rest. Also for
/// loggers of libraries in the framework that set their own, such as `mtmd`.
///
/// # Safety
///
/// `text` is null or a NUL-terminated string, as llama.cpp passes it.
pub unsafe extern "C" fn log(level: sys::ggml_log_level::Type, text: *const c_char, _: *mut c_void) {
    if level == sys::ggml_log_level::GGML_LOG_LEVEL_ERROR && !text.is_null() {
        let text = unsafe { CStr::from_ptr(text) }.to_string_lossy();
        LAST_ERROR.with(|last| *last.borrow_mut() = text.trim().to_string());
    }
}

/// The last error llama.cpp logged on this thread, taken.
pub fn last_error() -> String {
    let last = LAST_ERROR.with(|last| std::mem::take(&mut *last.borrow_mut()));
    if last.is_empty() { "llama.cpp gave no reason".into() } else { last }
}
