//! Builds llama.cpp release b11553 from source and links it statically. With
//! `ggml`, its ggml: the one ggml every model in the process runs on,
//! pam-speech's NeMo-Speech backend and aligner included. With `llama-cpp`,
//! libllama too, and bindings for `llama.h`; with `mtmd`, libmtmd, for
//! pam-speech's `qwen`.
//!
//! The source is `LLAMA_CPP_SRC` if non-empty, else a checkout of b11553 in
//! `~/Library/Caches/pam/`, fetched by git the first time it is needed, and it
//! is built in that cache (see pam-build). Dependents get the source folder as
//! `DEP_LLAMA_SRC`, the headers' folders as `DEP_LLAMA_INCLUDE` (a path list),
//! and ggml's CMake options as `DEP_LLAMA_GGML_OPTIONS`, so a library built
//! against this ggml sees it as it was built.

fn main() {
    #[cfg(feature = "ggml")]
    llama::build();
}

#[cfg(feature = "ggml")]
mod llama {
    const REPO: &str = "https://github.com/ggml-org/llama.cpp";
    /// Release b11553, which the local intent check and qwen were measured with.
    const REV: &str = "62a6f74bab748b1755f4cd8dfc0871caf865c6c8";
    /// ggml as the release's xcframework builds it, with llamafile's CPU
    /// matmuls, which NeMo-Speech asks for too.
    const GGML: &[&str] = &[
        "-DGGML_METAL=ON",
        "-DGGML_METAL_EMBED_LIBRARY=ON",
        "-DGGML_BLAS=ON",
        "-DGGML_NATIVE=OFF",
        "-DGGML_OPENMP=OFF",
        "-DGGML_LLAMAFILE=ON",
    ];
    /// llama.cpp's libraries alone: no tools, no server, no video in mtmd.
    const LLAMA: &[&str] = &[
        "-DLLAMA_BUILD_COMMON=OFF",
        "-DLLAMA_BUILD_TOOLS=OFF",
        "-DLLAMA_BUILD_TESTS=OFF",
        "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_SERVER=OFF",
        "-DLLAMA_BUILD_APP=OFF",
        "-DLLAMA_BUILD_MTMD=ON",
        "-DMTMD_VIDEO=OFF",
    ];

    pub fn build() {
        let src = pam_build::source("LLAMA_CPP_SRC", "llama.cpp", REPO, REV, "include/llama.h");
        let args: Vec<&str> = GGML.iter().chain(LLAMA).copied().collect();
        let cmake = pam_build::CMake::configure("llama.cpp", &src, &args);

        // Each library with what it needs, the dependents first.
        let mut libs: Vec<(&str, &str)> = Vec::new();
        if cfg!(feature = "mtmd") {
            libs.extend([("tools/mtmd", "mtmd"), ("vendor/hash", "vendor-hash")]);
        }
        if cfg!(feature = "llama-cpp") {
            libs.push(("src", "llama"));
        }
        libs.extend([
            ("ggml/src", "ggml"),
            ("ggml/src/ggml-metal", "ggml-metal"),
            ("ggml/src/ggml-blas", "ggml-blas"),
            ("ggml/src", "ggml-cpu"),
            ("ggml/src", "ggml-base"),
        ]);
        let targets: Vec<&str> = libs.iter().map(|(_, name)| *name).collect();
        cmake.build(&targets);
        for (dir, name) in &libs {
            println!("cargo:rustc-link-search=native={}", cmake.dir().join(dir).display());
            println!("cargo:rustc-link-lib=static={name}");
        }
        for framework in ["Foundation", "Metal", "MetalKit", "Accelerate"] {
            println!("cargo:rustc-link-lib=framework={framework}");
        }
        println!("cargo:rustc-link-lib=c++");

        let include = [src.join("include"), src.join("ggml/include"), src.join("tools/mtmd")];
        println!("cargo:src={}", src.display());
        println!("cargo:include={}", std::env::join_paths(&include).unwrap().to_string_lossy());
        println!("cargo:ggml_options={}", GGML.join(" "));

        #[cfg(feature = "llama-cpp")]
        bindings(&include);
    }

    #[cfg(feature = "llama-cpp")]
    fn bindings(include: &[std::path::PathBuf]) {
        let bindings = bindgen::Builder::default()
            .header(include[0].join("llama.h").display().to_string())
            .clang_args(include.iter().map(|dir| format!("-I{}", dir.display())))
            .allowlist_function("llama_.*")
            .allowlist_type("llama_.*")
            .default_enum_style(bindgen::EnumVariation::ModuleConsts)
            .derive_default(true)
            .layout_tests(false)
            .generate()
            .expect("generating llama.cpp bindings");
        let out = std::path::PathBuf::from(std::env::var("OUT_DIR").unwrap()).join("llama.rs");
        bindings.write_to_file(out).expect("writing llama.cpp bindings");
    }
}
