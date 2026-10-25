//! Builds the native speech libraries from source and links them statically,
//! all on the one ggml pam-llama builds from llama.cpp b11553, so a binary
//! needs no library beside it.
//!
//! With `nemo-speech`, see the `nemo` module below; with `qwen`, the `qwen`
//! module.

fn main() {
    #[cfg(feature = "nemo-speech")]
    nemo::build();
    #[cfg(feature = "qwen")]
    qwen::build();
}

/// The headers' folders pam-llama gives: llama.cpp's, ggml's and mtmd's.
#[cfg(any(feature = "nemo-speech", feature = "qwen"))]
fn llama_include() -> Vec<std::path::PathBuf> {
    let include = std::env::var_os("DEP_LLAMA_INCLUDE").expect("pam-llama gives its headers");
    std::env::split_paths(&include).collect()
}

/// With the `qwen` feature: bindings for llama.cpp's `llama.h` and `mtmd.h`,
/// whose libraries pam-llama links, and qwen3-asr.cpp's forced aligner
/// compiled against the same headers, so the process holds one ggml for both
/// models. The aligner's source is `QWEN3_ASR_DIR` if non-empty, else a
/// checkout of the pinned revision in `~/Library/Caches/pam/`, fetched by git
/// the first time it is needed.
#[cfg(feature = "qwen")]
mod qwen {
    use std::path::PathBuf;

    const REPO: &str = "https://github.com/predict-woo/qwen3-asr.cpp";
    /// The revision Qwen streaming was measured with.
    const REV: &str = "6dcc586e5073fd6e85ee5728e75f0903d6c70c6c";
    /// Pam's patch: the decoder's attention keys are cast to
    /// F16 like its values, or Metal can't run `flash_attn_ext` and the
    /// attention of all 28 layers runs on the CPU.
    const K_LINE: &str = "ggml_tensor * Kfa = ggml_permute(ctx0, Kcur, 0, 2, 1, 3);";
    const K_F16: &str =
        "ggml_tensor * Kfa = ggml_cast(ctx0, ggml_permute(ctx0, Kcur, 0, 2, 1, 3), GGML_TYPE_F16);";

    pub fn build() {
        let include = super::llama_include();
        let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        println!("cargo:rerun-if-changed=src/qwen/shim");
        let src = pam_build::source("QWEN3_ASR_DIR", "qwen3-asr.cpp", REPO, REV, "src/forced_aligner.cpp");

        let aligner =
            std::fs::read_to_string(src.join("src/forced_aligner.cpp")).expect("reading forced_aligner.cpp");
        assert_eq!(aligner.matches(K_LINE).count(), 1, "forced_aligner.cpp isn't the pinned revision's");
        let patched = out.join("forced_aligner.cpp");
        std::fs::write(&patched, aligner.replace(K_LINE, K_F16)).expect("writing the patched aligner");
        cc::Build::new()
            .cpp(true)
            .std("c++20")
            .opt_level(3)
            .warnings(false)
            .flag("-w")
            .files([patched, src.join("src/mel_spectrogram.cpp"), "src/qwen/shim/shim.cpp".into()])
            // shim/ holds the part of ggml-impl.h the aligner uses.
            .includes(include.iter().cloned().chain([
                "src/qwen/shim".into(),
                src.join("include"),
                src.join("src"),
            ]))
            .compile("qwen_aligner");

        let bindings = bindgen::Builder::default()
            .header_contents(
                "wrapper.h",
                "#include \"llama.h\"\n#include \"mtmd.h\"\n#include \"mtmd-helper.h\"\n",
            )
            .clang_args(include.iter().map(|dir| format!("-I{}", dir.display())))
            .allowlist_function("(llama|mtmd)_.*")
            .allowlist_type("(llama|mtmd)_.*")
            .default_enum_style(bindgen::EnumVariation::ModuleConsts)
            .derive_default(true)
            .layout_tests(false)
            .generate()
            .expect("generating llama.cpp bindings");
        bindings.write_to_file(out.join("qwen_llama.rs")).expect("writing llama.cpp bindings");
    }
}

/// With the `nemo-speech` feature: NeMo-Speech.cpp v0.2.0, built from source
/// and linked statically, and bindings for its C ABI (`asr.h`, `diar.h`).
///
/// The source is `NEMO_SPEECH_SRC` if non-empty, else a checkout of v0.2.0
/// in `~/Library/Caches/pam/`, fetched by git the first time it is needed.
/// It is built there, with its `metal-asr` preset, on b11553's ggml rather
/// than the llama.cpp its own checkout pins, and with its two ASR libraries
/// static (`nemo/static.cmake`). Its libraries and SentencePiece are then
/// pre-linked into one object that exports only the C ABI: NeMo-Speech has
/// its own `llama_file` and `format`, which would clash with llama.cpp's in
/// one link.
#[cfg(feature = "nemo-speech")]
mod nemo {
    use std::path::{Path, PathBuf};
    use std::process::Command;

    use pam_build::run;

    const REPO: &str = "https://github.com/NVIDIA/NeMo-Speech.cpp";
    /// Release v0.2.0.
    const REV: &str = "6a3ca369370782acee0dd155e34394ba60b920c4";
    /// NeMo-Speech's libraries, the C ABI first.
    const LIBS: &[(&str, &str)] = &[
        ("src/asr", "nemo_speech_asr_c"),
        ("src/asr", "nemo_speech_asr"),
        ("src/runtime/ggml", "nemo_speech_runtime_ggml"),
        ("src/common", "nemo_speech_common"),
    ];

    pub fn build() {
        let src =
            pam_build::source("NEMO_SPEECH_SRC", "NeMo-Speech.cpp", REPO, REV, "include/nemo_speech/asr.h");
        let llama = std::env::var("DEP_LLAMA_SRC").expect("pam-llama gives its source");
        let ggml = std::env::var("DEP_LLAMA_GGML_OPTIONS").expect("pam-llama gives ggml's options");
        let manifest = PathBuf::from(std::env::var("CARGO_MANIFEST_DIR").unwrap());
        println!("cargo:rerun-if-changed=nemo");

        let deps = sentencepiece(&src);
        let mut args = vec![
            "--preset".to_string(),
            "metal-asr".into(),
            format!("-DNEMO_SPEECH_LLAMA_CPP_SOURCE_DIR={llama}"),
            format!("-DNEMO_SPEECH_DEPENDENCY_PREFIX={}", deps.display()),
            format!("-DCMAKE_PROJECT_nemo_speech_INCLUDE={}", manifest.join("nemo/static.cmake").display()),
            "-DNEMO_SPEECH_BUILD_CLI=OFF".into(),
            "-DNEMO_SPEECH_BUILD_MIC_CAPTURE=OFF".into(),
            "-DNEMO_SPEECH_WITH_NORM=OFF".into(),
            // ggml is pam-llama's: its targets are configured here, for
            // their headers and definitions, and only ggml-metal's archive is
            // built (CMake keeps it for its shader embedding), never linked.
            "-DCMAKE_OPTIMIZE_DEPENDENCIES=ON".into(),
        ];
        args.extend(ggml.split(' ').map(String::from));
        let args: Vec<&str> = args.iter().map(String::as_str).collect();
        let cmake = pam_build::CMake::configure("NeMo-Speech.cpp", &src, &args);
        cmake.build(&LIBS.iter().map(|(_, name)| *name).collect::<Vec<_>>());

        let out = PathBuf::from(std::env::var("OUT_DIR").unwrap());
        let object = out.join("nemo_speech.o");
        let target = pam_build::deployment_target();
        let mut ld = Command::new("ld");
        ld.args(["-r", "-arch", "arm64", "-platform_version", "macos", &target, &target])
            .arg("-exported_symbols_list")
            .arg(manifest.join("nemo/exports.txt"))
            .arg("-all_load");
        for (dir, name) in LIBS {
            ld.arg(cmake.dir().join(dir).join(format!("lib{name}.a")));
        }
        ld.arg(deps.join("sentencepiece/lib/libsentencepiece.a")).arg("-o").arg(&object);
        run(&mut ld);
        let archive = out.join("libnemo_speech.a");
        let _ = std::fs::remove_file(&archive);
        run(Command::new("libtool").args(["-static", "-o"]).arg(&archive).arg(&object));
        println!("cargo:rustc-link-search=native={}", out.display());
        println!("cargo:rustc-link-lib=static=nemo_speech");
        println!("cargo:rustc-link-lib=c++");

        let include = src.join("include");
        let bindings = bindgen::Builder::default()
            .header_contents("wrapper.h", "#include \"nemo_speech/asr.h\"\n#include \"nemo_speech/diar.h\"\n")
            .clang_arg(format!("-I{}", include.display()))
            .allowlist_function("nemo_speech_(asr|diar)_.*")
            .allowlist_type("nemo_speech_(asr|diar)_.*")
            .allowlist_var("NEMO_SPEECH_.*")
            .default_enum_style(bindgen::EnumVariation::ModuleConsts)
            .derive_default(true)
            .layout_tests(false)
            .generate()
            .expect("generating NeMo-Speech bindings");
        bindings.write_to_file(out.join("nemo_speech.rs")).expect("writing NeMo-Speech bindings");
    }

    /// SentencePiece built static by NeMo-Speech's own script, at the commit
    /// that script pins, in the cache: the prefix NeMo-Speech's build reads
    /// `sentencepiece/` from.
    fn sentencepiece(src: &Path) -> PathBuf {
        let script = src.join("scripts/build_sentencepiece_static.sh");
        let text = std::fs::read_to_string(&script).expect("reading build_sentencepiece_static.sh");
        let commit = text
            .lines()
            .find_map(|line| line.strip_prefix("COMMIT="))
            .expect("build_sentencepiece_static.sh pins no commit");
        let target = pam_build::deployment_target();
        let deps = pam_build::cache().join(format!("build/sentencepiece-{}-macos{target}", &commit[..12]));
        let _lock = pam_build::lock(&deps);
        if !deps.join("sentencepiece/lib/libsentencepiece.a").exists() {
            run(Command::new("bash")
                .arg(&script)
                .arg(deps.join("work"))
                .env("PREFIX", deps.join("sentencepiece"))
                .env("MACOSX_DEPLOYMENT_TARGET", &target)
                .env("JOBS", std::env::var("NUM_JOBS").unwrap_or_else(|_| "4".into())));
        }
        deps
    }
}
