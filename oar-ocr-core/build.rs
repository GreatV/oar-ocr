//! Build script for oar-ocr-core.
//!
//! Only does work when the `rknpu` cargo feature is on AND
//! `target_arch = "aarch64"`. In every other configuration it short-circuits
//! immediately. FFI bindings to librknnrt are pre-generated and checked in
//! at `vendor/rknn/rknn_bindings.rs`, so we don't need bindgen at build time
//! and aarch64 build hosts (e.g. RK3588 itself) don't need libclang.

fn main() {
    println!("cargo:rerun-if-env-changed=CARGO_FEATURE_RKNPU");
    println!("cargo:rerun-if-env-changed=CARGO_CFG_TARGET_ARCH");
    println!("cargo:rerun-if-env-changed=RKNN_LIB_DIR");
    println!("cargo:rerun-if-changed=vendor/rknn/rknn_bindings.rs");

    let feature_rknpu = std::env::var_os("CARGO_FEATURE_RKNPU").is_some();
    let target_arch = std::env::var("CARGO_CFG_TARGET_ARCH").unwrap_or_default();

    if !feature_rknpu || target_arch != "aarch64" {
        return;
    }

    // Link librknnrt. Override the search path with RKNN_LIB_DIR if needed.
    let lib_dir =
        std::env::var("RKNN_LIB_DIR").unwrap_or_else(|_| "/usr/lib/aarch64-linux-gnu".to_string());
    println!("cargo:rustc-link-search=native={}", lib_dir);
    println!("cargo:rustc-link-lib=rknnrt");
}
