use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for source in [
        "gpu_runtime.h",
        "gpu_runtime.cpp",
        "gpu_build.cpp",
        "clip_decode.h",
        "clip_decode.cpp",
        "Makefile",
        "media_runtime.h",
        "media_internal.h",
        "media_runtime.cpp",
        "media_metadata.cpp",
        "media_recording.cpp",
    ] {
        println!("cargo:rerun-if-changed=../native/{source}");
    }
    link_native("SEEON_GPU_LIB_DIR", "seeon_gpu");
    link_native("SEEON_MEDIA_LIB_DIR", "seeon_media");
    link_native("SEEON_CLIPDEC_LIB_DIR", "seeon_clipdec");
    // Deployment owns LD_LIBRARY_PATH. Do not embed an rpath or compile a substitute.
}

fn link_native(variable: &str, name: &str) {
    println!("cargo:rerun-if-env-changed={variable}");
    let supplied =
        PathBuf::from(env::var_os(variable).unwrap_or_else(|| {
            panic!("{variable} must name the canonical native library directory")
        }));
    let canonical = fs::canonicalize(&supplied)
        .unwrap_or_else(|_| panic!("{variable} must name an existing canonical directory"));
    assert!(
        canonical.is_dir() && supplied.as_os_str() == canonical.as_os_str(),
        "{variable} must be the exact absolute canonical directory"
    );
    let directory = canonical
        .to_str()
        .unwrap_or_else(|| panic!("{variable} must be UTF-8 for Cargo's link directives"));
    assert!(
        !directory.contains(['\r', '\n']),
        "{variable} cannot contain Cargo directive separators"
    );
    let library = canonical.join(format!("lib{name}.so"));
    println!("cargo:rerun-if-changed={directory}/lib{name}.so");
    assert!(
        fs::metadata(&library).is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0),
        "{variable} must contain a nonempty lib{name}.so; no fallback is available"
    );
    println!("cargo:rustc-link-search=native={directory}");
    println!("cargo:rustc-link-lib=dylib={name}");
}
