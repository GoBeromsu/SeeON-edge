use std::{env, fs, path::PathBuf};

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    for source in [
        "ort_runtime.h",
        "ort_internal.h",
        "ort_runtime.cpp",
        "ort_tensor.cpp",
        "Makefile",
    ] {
        println!("cargo:rerun-if-changed=../native/{source}");
    }
    println!("cargo:rerun-if-env-changed=SEEON_ORT_LIB_DIR");
    let supplied = PathBuf::from(env::var_os("SEEON_ORT_LIB_DIR").unwrap_or_else(|| {
        panic!("SEEON_ORT_LIB_DIR must name the canonical native library directory")
    }));
    let canonical = fs::canonicalize(&supplied)
        .unwrap_or_else(|_| panic!("SEEON_ORT_LIB_DIR must name an existing canonical directory"));
    assert!(
        canonical.is_dir() && supplied.as_os_str() == canonical.as_os_str(),
        "SEEON_ORT_LIB_DIR must be the exact absolute canonical directory"
    );
    let directory = canonical
        .to_str()
        .unwrap_or_else(|| panic!("SEEON_ORT_LIB_DIR must be UTF-8 for Cargo's link directives"));
    assert!(
        !directory.contains(['\r', '\n']),
        "SEEON_ORT_LIB_DIR cannot contain Cargo directive separators"
    );
    let library = canonical.join("libseeon_ort.so");
    println!("cargo:rerun-if-changed={directory}/libseeon_ort.so");
    assert!(
        fs::metadata(&library).is_ok_and(|metadata| metadata.is_file() && metadata.len() > 0),
        "SEEON_ORT_LIB_DIR must contain a nonempty libseeon_ort.so; no fallback is available"
    );
    println!("cargo:rustc-link-search=native={directory}");
    println!("cargo:rustc-link-lib=dylib=seeon_ort");
    // Deployment owns the loader search path; no rpath or substitute is embedded.
}
