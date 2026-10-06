use crate::{manifest::Case, memory::GpuMemory};
use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};
use std::{path::Path, process::Command};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub(crate) struct Statistics {
    pub(crate) mean: f64,
    pub(crate) p50: f64,
    pub(crate) p95: f64,
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
            let (low, high) = (index.floor() as usize, index.ceil() as usize);
            sorted[low] + (sorted[high] - sorted[low]) * (index - low as f64)
        };
        Ok(Self {
            mean: sorted.iter().sum::<f64>() / sorted.len() as f64,
            p50: percentile(0.5),
            p95: percentile(0.95),
        })
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct Measurement {
    /// Device actually used, e.g. `cuda:0` for an `auto` case on a GPU machine.
    pub(crate) device: String,
    pub(crate) pages: Vec<String>,
    pub(crate) load_ms: f64,
    pub(crate) latency_ms: Statistics,
    pub(crate) pages_per_second: f64,
    /// VL only: PageParser exposes no generated-token count.
    pub(crate) output_chars_per_second: Option<f64>,
    pub(crate) host_peak_bytes: Option<u64>,
    pub(crate) gpu: Option<GpuMemory>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct CaseResult {
    pub(crate) case: Case,
    pub(crate) measurement: Option<Measurement>,
    pub(crate) error: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub(crate) struct Environment {
    pub(crate) git_commit: Option<String>,
    pub(crate) dirty: Option<bool>,
    pub(crate) cpu: Option<String>,
    pub(crate) features: Vec<String>,
    pub(crate) release_build: bool,
    /// `OAR_*` runtime overrides, such as a forced VL dtype.
    pub(crate) overrides: std::collections::BTreeMap<String, String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct RunResult {
    pub(crate) timestamp_unix_ms: u128,
    pub(crate) environment: Environment,
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
        let cpu = std::fs::read_to_string("/proc/cpuinfo")
            .ok()
            .and_then(|text| {
                text.lines().find_map(|line| {
                    let (key, value) = line.split_once(':')?;
                    (key.trim() == "model name").then(|| value.trim().to_string())
                })
            })
            .or_else(|| command(root, "sysctl", &["-n", "machdep.cpu.brand_string"]));
        let features = [
            ("cuda", cfg!(feature = "cuda")),
            ("metal", cfg!(feature = "metal")),
            ("nvml", cfg!(feature = "nvml")),
        ]
        .into_iter()
        .filter(|(_, enabled)| *enabled)
        .map(|(name, _)| name.to_string())
        .collect();
        Self {
            git_commit: command(root, "git", &["rev-parse", "HEAD"]),
            dirty: command(root, "git", &["status", "--porcelain"]).map(|s| !s.is_empty()),
            cpu,
            features,
            release_build: !cfg!(debug_assertions),
            overrides: std::env::vars()
                .filter(|(key, _)| key.starts_with("OAR_"))
                .collect(),
        }
    }
}

fn mib(bytes: u64) -> String {
    format!("{:.0}", bytes as f64 / 1_048_576.0)
}

pub(crate) fn table(results: &[CaseResult]) -> String {
    let mut text = "| Case | Device | Load ms | Mean ms/page | p50 | p95 | Pages/s | Chars/s | Host peak MiB | GPU Δ MiB |\n|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n".to_string();
    for row in results {
        let name = &row.case.name;
        let Some(m) = &row.measurement else {
            text.push_str(&format!("| {name} | FAILED | | | | | | | | |\n"));
            continue;
        };
        let or_dash = |value: Option<String>| value.unwrap_or_else(|| "—".into());
        text.push_str(&format!(
            "| {name} | {} | {:.0} | {:.1} | {:.1} | {:.1} | {:.2} | {} | {} | {} |\n",
            m.device,
            m.load_ms,
            m.latency_ms.mean,
            m.latency_ms.p50,
            m.latency_ms.p95,
            m.pages_per_second,
            or_dash(m.output_chars_per_second.map(|v| format!("{v:.0}"))),
            or_dash(m.host_peak_bytes.map(mib)),
            or_dash(m.gpu.as_ref().map(|g| mib(g.delta_bytes()))),
        ));
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
        assert!(Statistics::calculate(&[]).is_err());
        assert!(Statistics::calculate(&[f64::NAN]).is_err());
    }
}
