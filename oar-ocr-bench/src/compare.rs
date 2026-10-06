use crate::result::{CaseResult, Measurement, RunResult, fingerprints};
use anyhow::{Result, ensure};
use std::collections::{BTreeMap, BTreeSet};

pub(crate) struct Comparison {
    pub(crate) markdown: String,
    pub(crate) failed: bool,
}

pub(crate) fn threshold(text: &str) -> Result<f64> {
    let number: f64 = text.trim().trim_end_matches('%').parse()?;
    ensure!(
        number.is_finite() && number >= 0.0,
        "threshold must be a finite nonnegative percentage"
    );
    Ok(number)
}

fn regression(base: f64, new: f64, higher_is_better: bool, threshold: f64) -> bool {
    if base == 0.0 {
        return !higher_is_better && new > 0.0;
    }
    let delta = (new / base - 1.0) * 100.0;
    let worse = if higher_is_better { -delta } else { delta };
    worse > threshold + 1e-9
}

fn metrics(m: &Measurement) -> Vec<(&'static str, Option<f64>, Option<bool>)> {
    vec![
        ("load_ms", Some(m.model_load_ms), Some(false)),
        ("mean_ms", Some(m.latency_ms.mean), Some(false)),
        ("p50_ms", Some(m.latency_ms.p50), Some(false)),
        ("p95_ms", Some(m.latency_ms.p95), Some(false)),
        ("min_ms", Some(m.latency_ms.min), Some(false)),
        ("max_ms", Some(m.latency_ms.max), Some(false)),
        ("pages/s", Some(m.pages_per_second), Some(true)),
        ("tokens/s", m.tokens_per_second, Some(true)),
        ("output_chars/s", m.output_characters_per_second, Some(true)),
        (
            "host_peak_bytes",
            m.host_peak_bytes.map(|v| v as f64),
            Some(false),
        ),
        (
            "gpu_baseline_bytes",
            m.gpu.as_ref().map(|g| g.baseline_bytes as f64),
            None,
        ),
        (
            "gpu_peak_bytes",
            m.gpu.as_ref().map(|g| g.peak_bytes as f64),
            Some(false),
        ),
        (
            "gpu_delta_bytes",
            m.gpu.as_ref().map(|g| g.delta_bytes as f64),
            Some(false),
        ),
    ]
}
fn corpus(m: &Measurement) -> BTreeSet<(&str, &str)> {
    m.samples
        .iter()
        .map(|s| (s.page_id.as_str(), s.input_sha256.as_str()))
        .collect()
}
fn compatible(a: &CaseResult, b: &CaseResult) -> bool {
    let mut a = a.config.clone();
    let mut b = b.config.clone();
    // Different sample counts are comparable, but decoding and batching must match.
    a.repetitions = 0;
    b.repetitions = 0;
    a == b
}

pub(crate) fn compare(base: &RunResult, new: &RunResult, limit: f64) -> Result<Comparison> {
    ensure!(
        base.schema_version == 1 && new.schema_version == 1,
        "unsupported result schema"
    );
    let old: BTreeMap<_, _> = base
        .cases
        .iter()
        .map(|c| (c.config.name.as_str(), c))
        .collect();
    let next: BTreeMap<_, _> = new
        .cases
        .iter()
        .map(|c| (c.config.name.as_str(), c))
        .collect();
    ensure!(
        old.len() == base.cases.len() && next.len() == new.cases.len(),
        "duplicate case names in results"
    );
    let mut text =
        "| Case | Metric | Base | New | Change | Status |\n|---|---|---:|---:|---:|---|\n"
            .to_string();
    let mut failed = false;
    let hardware_changed = base.environment.cpu_model != new.environment.cpu_model
        || base.environment.os != new.environment.os
        || base.environment.architecture != new.environment.architecture
        || base.environment.release_build != new.environment.release_build
        || base.environment.enabled_features != new.environment.enabled_features
        || base.environment.rustc != new.environment.rustc;
    if hardware_changed {
        text.push_str("| All | environment | — | — | — | INCOMPATIBLE HARDWARE/PROFILE |\n");
        failed = true;
    }
    for name in old
        .keys()
        .chain(next.keys())
        .copied()
        .collect::<BTreeSet<_>>()
    {
        let label = name.replace('|', "\\|").replace('\n', " ");
        let (Some(a), Some(b)) = (old.get(name), next.get(name)) else {
            text.push_str(&format!("| {label} | case | — | — | — | ADDED/REMOVED |\n"));
            failed = true;
            continue;
        };
        let (Some(x), Some(y)) = (&a.measurement, &b.measurement) else {
            text.push_str(&format!("| {label} | case | — | — | — | FAILED RUN |\n"));
            failed = true;
            continue;
        };
        let same_gpu = match (&x.gpu, &y.gpu) {
            (Some(a), Some(b)) => a.device == b.device,
            _ => true,
        };
        let same_inputs = corpus(x) == corpus(y);
        let comparable = compatible(a, b)
            && same_inputs
            && x.rate_basis == y.rate_basis
            && x.latency_basis == y.latency_basis
            && same_gpu
            && !hardware_changed;
        if !same_inputs {
            text.push_str(&format!(
                "| {label} | inputs | — | — | — | inputs differ / not comparable |\n"
            ));
            failed = true;
        } else if !comparable {
            text.push_str(&format!(
                "| {label} | inputs/configuration | — | — | — | INCOMPATIBLE |\n"
            ));
            failed = true;
        }
        let valid = x.valid && y.valid && x.output_stable && y.output_stable;
        if !valid {
            text.push_str(&format!(
                "| {label} | validity | — | — | — | INVALID/UNSTABLE |\n"
            ));
            failed = true;
        }
        let output_changed = same_inputs && fingerprints(&x.samples) != fingerprints(&y.samples);
        text.push_str(&format!(
            "| {label} | output fingerprints | — | — | — | {} |\n",
            if !same_inputs {
                "NOT COMPARED (inputs differ)"
            } else if output_changed {
                "CHANGED"
            } else {
                "SAME"
            }
        ));
        failed |= output_changed;
        for ((metric, left, direction), (_, right, _)) in metrics(x).into_iter().zip(metrics(y)) {
            let format_value = |value: Option<f64>| {
                value
                    .map(|v| format!("{v:.3}"))
                    .unwrap_or_else(|| "—".into())
            };
            let (delta, worse) = match (left, right) {
                (Some(left), Some(right)) if left > 0.0 => (
                    format!("{:+.2}%", (right / left - 1.0) * 100.0),
                    direction.is_some_and(|higher| regression(left, right, higher, limit)),
                ),
                (Some(left), Some(right)) => (
                    format!("{:+.3} (zero base)", right - left),
                    direction.is_some_and(|higher| regression(left, right, higher, limit)),
                ),
                _ => ("—".into(), false),
            };
            let status = if !comparable || !valid {
                "NOT COMPARABLE"
            } else if left.is_none() || right.is_none() {
                "UNAVAILABLE"
            } else if worse {
                "REGRESSION"
            } else {
                "OK"
            };
            failed |= comparable && valid && worse;
            text.push_str(&format!(
                "| {label} | {metric} | {} | {} | {delta} | {status} |\n",
                format_value(left),
                format_value(right)
            ));
        }
    }
    Ok(Comparison {
        markdown: text,
        failed,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        manifest::Manifest,
        result::{Environment, PageSample, Statistics},
    };
    fn run(latency: f64, output: &str) -> RunResult {
        let manifest = Manifest::parse("[inputs]\nimages=['page.png']\n[[cases]]\nname='test'\nkind='ocr'\n[cases.models]\ndetector='a'\nrecognizer='b'\ndictionary='c'", None, None).unwrap();
        RunResult {
            schema_version: 1,
            timestamp_unix_ms: 0,
            environment: Environment::collect(std::path::Path::new(".")),
            manifest_sha256: "manifest".into(),
            manifest_content: "".into(),
            inputs: manifest.inputs,
            cases: vec![CaseResult {
                config: manifest.cases[0].clone(),
                error: None,
                measurement: Some(Measurement {
                    model_load_ms: 1.0,
                    measured_seconds: latency / 1000.0,
                    latency_ms: Statistics::calculate(&[latency]).unwrap(),
                    pages_per_second: 1000.0 / latency,
                    tokens_per_second: None,
                    output_characters_per_second: Some(10.0),
                    rate_basis: "characters".into(),
                    latency_basis: "single page".into(),
                    host_peak_bytes: None,
                    gpu: None,
                    warnings: Vec::new(),
                    output_stable: true,
                    valid: true,
                    samples: vec![PageSample {
                        page_id: "page.png".into(),
                        input_sha256: "input".into(),
                        repetition: 0,
                        latency_ms: latency,
                        output_sha256: output.into(),
                        document_sha256: None,
                        output_characters: 1,
                        diagnostics: 0,
                    }],
                }),
            }],
        }
    }
    #[test]
    fn threshold_is_strict_and_direction_aware() {
        assert!(!regression(100.0, 105.0, false, 5.0));
        assert!(regression(100.0, 105.1, false, 5.0));
        assert!(regression(100.0, 94.9, true, 5.0));
        assert!(!regression(100.0, 110.0, true, 5.0));
        assert!(regression(0.0, 1.0, false, 5.0));
        assert_eq!(threshold("5%").unwrap(), 5.0);
        assert!(threshold("NaN").is_err());
        assert!(threshold("-5%").is_err());
    }
    #[test]
    fn regressions_and_fingerprints_are_reported() {
        let base = run(100.0, "same");
        let comparison = compare(&base, &run(106.0, "same"), 5.0).unwrap();
        assert!(comparison.failed && comparison.markdown.contains("REGRESSION"));
        let changed = compare(&base, &run(99.0, "different"), 5.0).unwrap();
        assert!(changed.failed && changed.markdown.contains("CHANGED"));
        assert!(!compare(&base, &run(100.0, "same"), 5.0).unwrap().failed);
    }
    #[test]
    fn input_changes_and_invalid_runs_are_not_comparable() {
        let base = run(100.0, "same");
        let mut next = base.clone();
        next.cases[0].measurement.as_mut().unwrap().samples[0].input_sha256 = "different".into();
        next.cases[0].measurement.as_mut().unwrap().samples[0].output_sha256 =
            "also different".into();
        let comparison = compare(&base, &next, 5.0).unwrap();
        assert!(comparison.failed);
        assert!(
            comparison
                .markdown
                .contains("inputs differ / not comparable")
        );
        assert!(comparison.markdown.contains("NOT COMPARED (inputs differ)"));
        assert!(!comparison.markdown.contains("| CHANGED |"));
        next = base.clone();
        next.cases[0].measurement.as_mut().unwrap().valid = false;
        assert!(compare(&base, &next, 5.0).unwrap().failed);
    }
}
