use std::process::Command;

const MIN_CUDA_COMPUTE_CAP: u32 = 80;

fn parse_compute_cap(value: &str) -> Option<(String, u32)> {
    let value = value.trim().to_ascii_lowercase();
    let value = value
        .strip_prefix("compute_")
        .or_else(|| value.strip_prefix("sm_"))
        .unwrap_or(&value)
        .replace('.', "");
    let digit_count = value.bytes().take_while(u8::is_ascii_digit).count();
    if digit_count == 0 {
        return None;
    }
    let (digits, suffix) = value.split_at(digit_count);
    if !matches!(suffix, "" | "a" | "f") {
        return None;
    }
    let mut base = digits.parse::<u32>().ok()?;
    if base < 20 {
        base *= 10;
    }
    Some((format!("{base}{suffix}"), base))
}

fn detect_local_compute_cap() -> Option<u32> {
    let output = Command::new("nvidia-smi")
        .args(["--query-gpu=compute_cap", "--format=csv,noheader"])
        .output()
        .ok()?;
    if !output.status.success() {
        return None;
    }
    String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| parse_compute_cap(line).map(|(_, base)| base))
        // PTX compiled for the oldest GPU reported by nvidia-smi remains
        // loadable on newer GPUs in a heterogeneous machine.
        .min()
}

fn cuda_compute_arch() -> String {
    if let Ok(value) = std::env::var("CUDA_COMPUTE_CAP") {
        let (arch, base) = parse_compute_cap(&value).unwrap_or_else(|| {
            panic!(
                "invalid CUDA_COMPUTE_CAP={value:?}; expected values such as 89, 8.9, sm_89, or compute_89"
            )
        });
        assert!(
            base >= MIN_CUDA_COMPUTE_CAP,
            "oar-ocr-vl CUDA kernels require compute capability 8.0 or newer; got CUDA_COMPUTE_CAP={value:?}"
        );
        return format!("compute_{arch}");
    }

    match detect_local_compute_cap() {
        Some(base) if base >= MIN_CUDA_COMPUTE_CAP => format!("compute_{base}"),
        Some(base) => {
            println!(
                "cargo:warning=detected GPU compute capability {base} is below the oar-ocr-vl CUDA kernel minimum; compiling forward-compatible compute_{MIN_CUDA_COMPUTE_CAP} PTX"
            );
            format!("compute_{MIN_CUDA_COMPUTE_CAP}")
        }
        None => {
            println!(
                "cargo:warning=could not detect a CUDA GPU; compiling compute_{MIN_CUDA_COMPUTE_CAP} PTX (set CUDA_COMPUTE_CAP to override for cross/headless builds)"
            );
            format!("compute_{MIN_CUDA_COMPUTE_CAP}")
        }
    }
}

fn main() {
    println!("cargo:rerun-if-changed=src/hunyuanocr/dynamic_kv.cu");
    println!("cargo:rerun-if-env-changed=CUDA_COMPUTE_CAP");
    println!("cargo:rerun-if-env-changed=NVCC");
    let metal_enabled = std::env::var_os("CARGO_FEATURE_METAL").is_some();
    let cuda_enabled = std::env::var_os("CARGO_FEATURE_CUDA").is_some();
    let target_os = std::env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();

    if metal_enabled && target_os != "macos" {
        panic!("oar-ocr-vl feature `metal` is only supported on macOS targets");
    }

    if cuda_enabled {
        let cuda_arch = cuda_compute_arch();
        let nvcc = std::env::var_os("NVCC").unwrap_or_else(|| "nvcc".into());
        let out_dir = std::path::PathBuf::from(
            std::env::var_os("OUT_DIR").expect("Cargo always sets OUT_DIR"),
        );
        let output = Command::new(&nvcc)
            .args(["--ptx", "--std=c++17", "-O3"])
            .arg(format!("--gpu-architecture={cuda_arch}"))
            .arg("-o")
            .arg(out_dir.join("oar_vl_kernels.ptx"))
            .arg("src/hunyuanocr/dynamic_kv.cu")
            .output()
            .unwrap_or_else(|error| {
                panic!(
                    "failed to invoke {:?} for oar-ocr-vl CUDA kernels; install the CUDA toolkit or set NVCC to the compiler path: {error}",
                    nvcc
                )
            });
        if !output.status.success() {
            panic!(
                "{:?} failed for oar-ocr-vl CUDA kernels ({cuda_arch}):\n{}",
                nvcc,
                String::from_utf8_lossy(&output.stderr)
            );
        }
    }
}
