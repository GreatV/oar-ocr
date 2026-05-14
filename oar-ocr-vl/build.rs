fn main() {
    let metal_enabled = std::env::var_os("CARGO_FEATURE_METAL").is_some();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if metal_enabled && target_os != "macos" {
        panic!("oar-ocr-vl feature `metal` is only supported on macOS targets");
    }
}
