use crate::{
    manifest::{Case, Inputs},
    memory::GpuMeasurement,
};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
    process::Command,
};

include!(concat!(env!("OUT_DIR"), "/build_info.rs"));

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct Statistics {
    pub(crate) mean: f64,
    pub(crate) p50: f64,
    pub(crate) p95: f64,
    pub(crate) min: f64,
    pub(crate) max: f64,
}
impl Statistics {
    pub(crate) fn calculate(values: &[f64]) -> Result<Self> {
        ensure!(
            !values.is_empty() && values.iter().all(|v| v.is_finite() && *v >= 0.0),
            "statistics require finite nonnegative samples"
        );
        let mut sorted = values.to_vec();
        sorted.sort_by(f64::total_cmp);
        let percentile = |p: f64| {
            let index = p * (sorted.len() - 1) as f64;
            let low = index.floor() as usize;
            let high = index.ceil() as usize;
            sorted[low] + (sorted[high] - sorted[low]) * (index - low as f64)
        };
        Ok(Self {
            mean: sorted.iter().sum::<f64>() / sorted.len() as f64,
            p50: percentile(0.5),
            p95: percentile(0.95),
            min: sorted[0],
            max: sorted[sorted.len() - 1],
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct PageSample {
    pub(crate) page_id: String,
    pub(crate) input_sha256: String,
    pub(crate) repetition: usize,
    pub(crate) latency_ms: f64,
    pub(crate) output_sha256: String,
    pub(crate) document_sha256: Option<String>,
    pub(crate) output_characters: usize,
    pub(crate) diagnostics: usize,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Measurement {
    pub(crate) model_load_ms: f64,
    pub(crate) measured_seconds: f64,
    pub(crate) latency_ms: Statistics,
    pub(crate) pages_per_second: f64,
    pub(crate) tokens_per_second: Option<f64>,
    pub(crate) output_characters_per_second: Option<f64>,
    pub(crate) rate_basis: String,
    pub(crate) latency_basis: String,
    pub(crate) host_peak_bytes: Option<u64>,
    pub(crate) gpu: Option<GpuMeasurement>,
    pub(crate) warnings: Vec<String>,
    pub(crate) samples: Vec<PageSample>,
    pub(crate) output_stable: bool,
    pub(crate) valid: bool,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CaseResult {
    pub(crate) config: Case,
    pub(crate) measurement: Option<Measurement>,
    pub(crate) error: Option<String>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Environment {
    pub(crate) git_commit: Option<String>,
    pub(crate) dirty: Option<bool>,
    pub(crate) build_git_commit: Option<String>,
    pub(crate) build_dirty: Option<bool>,
    pub(crate) rustc: Option<String>,
    pub(crate) enabled_features: Vec<String>,
    pub(crate) cpu_model: Option<String>,
    pub(crate) os: String,
    pub(crate) architecture: String,
    pub(crate) release_build: bool,
    pub(crate) gpu_devices: Option<Vec<crate::memory::GpuIdentity>>,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RunResult {
    pub(crate) schema_version: u32,
    pub(crate) timestamp_unix_ms: u128,
    pub(crate) environment: Environment,
    pub(crate) manifest_sha256: String,
    pub(crate) manifest_content: String,
    pub(crate) inputs: Inputs,
    pub(crate) cases: Vec<CaseResult>,
}

fn command(root: &Path, program: &str, args: &[&str]) -> Option<String> {
    let out = Command::new(program)
        .args(args)
        .current_dir(root)
        .output()
        .ok()?;
    out.status
        .success()
        .then(|| String::from_utf8_lossy(&out.stdout).trim().to_string())
}
impl Environment {
    pub(crate) fn collect(root: &Path) -> Self {
        let cpu_model = if cfg!(target_os = "linux") {
            std::fs::read_to_string("/proc/cpuinfo")
                .ok()
                .and_then(|text| {
                    text.lines().find_map(|line| {
                        let (key, value) = line.split_once(':')?;
                        matches!(key.trim(), "model name" | "Hardware")
                            .then(|| value.trim().to_string())
                    })
                })
        } else if cfg!(target_os = "macos") {
            command(root, "sysctl", &["-n", "machdep.cpu.brand_string"])
        } else if cfg!(target_os = "windows") {
            command(
                root,
                "powershell",
                &[
                    "-NoProfile",
                    "-Command",
                    "(Get-CimInstance Win32_Processor).Name",
                ],
            )
        } else {
            None
        };
        let mut features = Vec::new();
        if cfg!(feature = "cuda") {
            features.push("cuda".into());
        }
        if cfg!(feature = "metal") {
            features.push("metal".into());
        }
        if cfg!(feature = "nvml") {
            features.push("nvml".into());
        }
        Self {
            git_commit: command(root, "git", &["rev-parse", "HEAD"]),
            dirty: command(root, "git", &["status", "--porcelain"]).map(|s| !s.is_empty()),
            build_git_commit: BUILD_COMMIT.map(str::to_string),
            build_dirty: BUILD_DIRTY,
            rustc: BUILD_RUSTC.map(str::to_string),
            enabled_features: features,
            cpu_model,
            os: std::env::consts::OS.into(),
            architecture: std::env::consts::ARCH.into(),
            release_build: !cfg!(debug_assertions),
            gpu_devices: None,
        }
    }
}

pub(crate) fn fingerprints(
    samples: &[PageSample],
) -> BTreeMap<String, BTreeSet<(String, Option<String>)>> {
    let mut result = BTreeMap::new();
    for sample in samples {
        result
            .entry(sample.page_id.clone())
            .or_insert_with(BTreeSet::new)
            .insert((sample.output_sha256.clone(), sample.document_sha256.clone()));
    }
    result
}

pub(crate) fn table(results: &[CaseResult]) -> String {
    let mut text = "| Case | Load ms | Mean ms/page | p50 | p95 | Min | Max | Pages/s | Output chars/s | Host peak MiB | GPU peak/delta MiB | Status |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|\n".to_string();
    for row in results {
        let name = row.config.name.replace('|', "\\|").replace('\n', " ");
        if let Some(m) = &row.measurement {
            let host = m
                .host_peak_bytes
                .map(|n| format!("{:.2}", n as f64 / 1_048_576.0))
                .unwrap_or_else(|| "—".into());
            let gpu = m
                .gpu
                .as_ref()
                .map(|g| {
                    format!(
                        "{:.2}/{:.2}",
                        g.peak_bytes as f64 / 1_048_576.0,
                        g.delta_bytes as f64 / 1_048_576.0
                    )
                })
                .unwrap_or_else(|| "—".into());
            let chars = m
                .output_characters_per_second
                .map(|n| format!("{n:.2}"))
                .unwrap_or_else(|| "—".into());
            let status = if !m.valid {
                "INVALID"
            } else if !m.output_stable {
                "OUTPUT VARIES"
            } else {
                "OK"
            };
            text.push_str(&format!("| {name} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {:.2} | {chars} | {host} | {gpu} | {status} |\n", m.model_load_ms, m.latency_ms.mean, m.latency_ms.p50, m.latency_ms.p95, m.latency_ms.min, m.latency_ms.max, m.pages_per_second));
        } else {
            text.push_str(&format!(
                "| {name} | — | — | — | — | — | — | — | — | — | — | FAILED |\n"
            ));
        }
    }
    text
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn statistics_use_linear_interpolation() {
        let stats = Statistics::calculate(&[4.0, 1.0, 3.0, 2.0]).unwrap();
        assert_eq!(stats.mean, 2.5);
        assert_eq!(stats.p50, 2.5);
        assert!((stats.p95 - 3.85).abs() < 1e-10);
        assert_eq!(stats.min, 1.0);
        assert_eq!(stats.max, 4.0);
        assert_eq!(Statistics::calculate(&[9.0]).unwrap().p95, 9.0);
    }
    #[test]
    fn statistics_reject_bad_samples() {
        assert!(Statistics::calculate(&[]).is_err());
        assert!(Statistics::calculate(&[f64::NAN]).is_err());
        assert!(Statistics::calculate(&[-1.0]).is_err());
    }
}
