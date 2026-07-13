fn main() {
    println!("cargo:rerun-if-changed=src/hunyuanocr/dynamic_kv.cu");
    let metal_enabled = std::env::var_os("CARGO_FEATURE_METAL").is_some();
    let cuda_enabled = std::env::var_os("CARGO_FEATURE_CUDA").is_some();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if metal_enabled && target_os != "macos" {
        panic!("oar-ocr-vl feature `metal` is only supported on macOS targets");
    }

    if cuda_enabled {
        let out_dir = std::path::PathBuf::from(
            std::env::var_os("OUT_DIR").expect("Cargo always sets OUT_DIR"),
        );
        let output = std::process::Command::new("nvcc")
            .args([
                "--ptx",
                "--std=c++17",
                "-O3",
                "--gpu-architecture=compute_80",
                "-o",
            ])
            .arg(out_dir.join("hunyuan_dynamic_kv.ptx"))
            .arg("src/hunyuanocr/dynamic_kv.cu")
            .output()
            .expect("failed to invoke nvcc for HunyuanOCR dynamic KV kernel");
        if !output.status.success() {
            panic!(
                "nvcc failed for HunyuanOCR dynamic KV kernel:\n{}",
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}
