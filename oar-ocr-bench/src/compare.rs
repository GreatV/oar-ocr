use crate::{
    manifest::Case,
    result::{CaseResult, Measurement, RunResult},
};
use anyhow::{Result, ensure};
use std::collections::{BTreeMap, BTreeSet};

fn cases(run: &RunResult) -> BTreeMap<&str, &CaseResult> {
    run.cases
        .iter()
        .map(|c| (c.case.name.as_str(), c))
        .collect()
}

/// Whether two cases run the same workload. The requested device is checked
/// through the resolved device instead, and sample counts may differ.
fn same_workload(a: &Case, b: &Case) -> bool {
    let normalize = |case: &Case| Case {
        device: String::new(),
        warmup: 0,
        repetitions: 0,
        ..case.clone()
    };
    normalize(a) == normalize(b)
}

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

/// Percentage change and whether it is worse than `limit` percent.
fn change(base: f64, new: f64, higher_is_better: bool, limit: f64) -> (f64, bool) {
    if base <= 0.0 {
        return (0.0, false);
    }
    let delta = (new / base - 1.0) * 100.0;
    let worse = if higher_is_better { -delta } else { delta };
    (delta, worse > limit + 1e-9)
}

/// Compared metrics and their direction (`Some(true)` when higher is better).
/// Memory peaks vary between identical runs, so they are shown without gating.
fn metrics(m: &Measurement) -> [(&'static str, Option<f64>, Option<bool>); 7] {
    let mib = |bytes: u64| bytes as f64 / 1_048_576.0;
    [
        ("mean_ms", Some(m.latency_ms.mean), Some(false)),
        ("p50_ms", Some(m.latency_ms.p50), Some(false)),
        ("p95_ms", Some(m.latency_ms.p95), Some(false)),
        ("pages/s", Some(m.pages_per_second), Some(true)),
        ("chars/s", m.output_chars_per_second, Some(true)),
        ("host_peak_mib", m.host_peak_bytes.map(mib), None),
        (
            "gpu_delta_mib",
            m.gpu.as_ref().map(|g| mib(g.delta_bytes())),
            None,
        ),
    ]
}

pub(crate) fn compare(base: &RunResult, new: &RunResult, limit: f64) -> Result<Comparison> {
    let mut text = String::new();
    if base.environment != new.environment {
        text.push_str(&format!(
            "Note: environments differ\n- base: {:?}\n- new:  {:?}\n\n",
            base.environment, new.environment
        ));
    }
    text.push_str(
        "| Case | Metric | Base | New | Change | Status |\n|---|---|---:|---:|---:|---|\n",
    );
    let (old, next) = (cases(base), cases(new));
    let mut failed = false;
    for name in old.keys().chain(next.keys()).collect::<BTreeSet<_>>() {
        let (Some(a), Some(b)) = (old.get(name), next.get(name)) else {
            text.push_str(&format!("| {name} | — | | | | MISSING OR FAILED |\n"));
            failed = true;
            continue;
        };
        let (Some(x), Some(y)) = (&a.measurement, &b.measurement) else {
            text.push_str(&format!("| {name} | — | | | | MISSING OR FAILED |\n"));
            failed = true;
            continue;
        };
        if !same_workload(&a.case, &b.case) {
            text.push_str(&format!("| {name} | — | | | | CONFIGURATION DIFFERS |\n"));
            failed = true;
            continue;
        }
        if x.pages != y.pages {
            text.push_str(&format!("| {name} | — | | | | INPUTS DIFFER |\n"));
            failed = true;
            continue;
        }
        // A logical ordinal can name different hardware; the GPU model is known
        // when both runs sampled memory with `nvml`.
        let hardware = |m: &Measurement| m.gpu.as_ref().map(|g| g.name.clone());
        let different_gpu = matches!(
            (hardware(x), hardware(y)),
            (Some(a), Some(b)) if a != b
        );
        if x.device != y.device || different_gpu {
            let describe = |m: &Measurement| match hardware(m) {
                Some(gpu) => format!("{} ({gpu})", m.device),
                None => m.device.clone(),
            };
            text.push_str(&format!(
                "| {name} | device | {} | {} | | DEVICES DIFFER |\n",
                describe(x),
                describe(y)
            ));
            failed = true;
            continue;
        }
        for ((metric, left, higher), (_, right, _)) in metrics(x).into_iter().zip(metrics(y)) {
            let (Some(left), Some(right)) = (left, right) else {
                continue;
            };
            let (delta, _) = change(left, right, false, limit);
            let worse = higher.is_some_and(|higher| change(left, right, higher, limit).1);
            failed |= worse;
            let status = match (higher, worse) {
                (None, _) => "INFO",
                (_, true) => "REGRESSION",
                _ => "OK",
            };
            text.push_str(&format!(
                "| {name} | {metric} | {left:.2} | {right:.2} | {delta:+.1}% | {status} |\n"
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
        memory::GpuMemory,
        result::{CaseResult, Environment, Statistics},
    };

    fn run(latency: f64, device: &str) -> RunResult {
        let manifest = Manifest::parse("[inputs]\nimages=['page.png']\n[[cases]]\nname='test'\nkind='ocr'\n[cases.models]\ndetector='a'\nrecognizer='b'\ndictionary='c'", None, None).unwrap();
        RunResult {
            timestamp_unix_ms: 0,
            environment: Environment::collect(std::path::Path::new(".")),
            cases: vec![CaseResult {
                case: manifest.cases[0].clone(),
                error: None,
                measurement: Some(Measurement {
                    device: device.into(),
                    pages: vec!["page.png".into()],
                    load_ms: 1.0,
                    latency_ms: Statistics::calculate(&[latency]).unwrap(),
                    pages_per_second: 1000.0 / latency,
                    output_chars_per_second: None,
                    host_peak_bytes: None,
                    gpu: None,
                }),
            }],
        }
    }

    #[test]
    fn regressions_respect_threshold_and_direction() {
        assert!(!change(100.0, 105.0, false, 5.0).1);
        assert!(change(100.0, 105.1, false, 5.0).1);
        assert!(change(100.0, 94.9, true, 5.0).1);
        assert!(threshold("-5%").is_err());
        let base = run(100.0, "cpu");
        assert!(!compare(&base, &run(104.0, "cpu"), 5.0).unwrap().failed);
        let slower = compare(&base, &run(106.0, "cpu"), 5.0).unwrap();
        assert!(slower.failed && slower.markdown.contains("REGRESSION"));
    }

    #[test]
    fn different_devices_or_inputs_are_not_compared() {
        let base = run(100.0, "cpu");
        let gpu = compare(&base, &run(10.0, "cuda:0"), 5.0).unwrap();
        assert!(gpu.failed && gpu.markdown.contains("DEVICES DIFFER"));
        let on_gpu = |name: &str| {
            let mut result = run(100.0, "cuda:0");
            result.cases[0].measurement.as_mut().unwrap().gpu = Some(GpuMemory {
                name: name.into(),
                baseline_bytes: 0,
                peak_bytes: 0,
            });
            result
        };
        let hardware = compare(&on_gpu("RTX 4090"), &on_gpu("A100"), 5.0).unwrap();
        assert!(hardware.failed && hardware.markdown.contains("DEVICES DIFFER"));
        assert!(
            !compare(&on_gpu("RTX 4090"), &on_gpu("RTX 4090"), 5.0)
                .unwrap()
                .failed
        );
        let mut other = run(100.0, "cpu");
        other.cases[0].measurement.as_mut().unwrap().pages = vec!["other.png".into()];
        let inputs = compare(&base, &other, 5.0).unwrap();
        assert!(inputs.failed && inputs.markdown.contains("INPUTS DIFFER"));
    }

    #[test]
    fn cases_missing_from_either_report_fail() {
        let base = run(100.0, "cpu");
        let mut empty = base.clone();
        empty.cases.clear();
        for (a, b) in [(&base, &empty), (&empty, &base)] {
            let comparison = compare(a, b, 5.0).unwrap();
            assert!(comparison.failed && comparison.markdown.contains("MISSING OR FAILED"));
        }
    }

    #[test]
    fn changed_case_configuration_is_not_compared() {
        let base = run(100.0, "cpu");
        let mut more_samples = run(100.0, "cpu");
        more_samples.cases[0].case.repetitions += 5;
        assert!(!compare(&base, &more_samples, 5.0).unwrap().failed);
        let mut batched = run(100.0, "cpu");
        batched.cases[0].case.options.batch_size = Some(4);
        let comparison = compare(&base, &batched, 5.0).unwrap();
        assert!(comparison.failed && comparison.markdown.contains("CONFIGURATION DIFFERS"));
    }
}
