use crate::result::{Measurement, RunResult};
use anyhow::{Result, ensure};
use std::collections::BTreeMap;

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
fn metrics(m: &Measurement) -> [(&'static str, Option<f64>, Option<bool>); 6] {
    let mib = |bytes: u64| bytes as f64 / 1_048_576.0;
    [
        ("mean_ms", Some(m.latency_ms.mean), Some(false)),
        ("p50_ms", Some(m.latency_ms.p50), Some(false)),
        ("p95_ms", Some(m.latency_ms.p95), Some(false)),
        ("pages/s", Some(m.pages_per_second), Some(true)),
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
    let old: BTreeMap<_, _> = base.cases.iter().map(|c| (&c.case.name, c)).collect();
    let mut failed = false;
    for case in &new.cases {
        let name = &case.case.name;
        let (Some(x), Some(y)) = (
            old.get(name).and_then(|c| c.measurement.as_ref()),
            case.measurement.as_ref(),
        ) else {
            text.push_str(&format!("| {name} | — | | | | MISSING OR FAILED |\n"));
            failed = true;
            continue;
        };
        if x.pages != y.pages {
            text.push_str(&format!("| {name} | — | | | | INPUTS DIFFER |\n"));
            failed = true;
            continue;
        }
        if x.device != y.device {
            text.push_str(&format!(
                "| {name} | device | {} | {} | | DEVICES DIFFER |\n",
                x.device, y.device
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
        let mut other = run(100.0, "cpu");
        other.cases[0].measurement.as_mut().unwrap().pages = vec!["other.png".into()];
        let inputs = compare(&base, &other, 5.0).unwrap();
        assert!(inputs.failed && inputs.markdown.contains("INPUTS DIFFER"));
    }
}
