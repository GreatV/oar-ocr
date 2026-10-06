mod compare;
mod input;
mod manifest;
mod memory;
mod pipeline;
mod result;

use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use manifest::{Case, Inputs, Kind, Manifest};
use result::{CaseResult, Environment, Measurement, PageSample, RunResult, Statistics};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    collections::BTreeSet,
    fs,
    io::Write,
    path::{Path, PathBuf},
    process::{Command, ExitCode, Stdio},
    time::{Instant, SystemTime, UNIX_EPOCH},
};

#[derive(Parser)]
#[command(
    name = "oar-bench",
    about = "Isolated end-to-end OCR and document parsing benchmarks"
)]
struct Args {
    #[command(subcommand)]
    command: Action,
}
#[derive(Subcommand)]
enum Action {
    /// Run every selected case in a fresh subprocess and write one JSON report.
    Run {
        #[arg(long, default_value = "oar-ocr-bench/manifests/default.toml")]
        manifest: PathBuf,
        #[arg(long, default_value = ".")]
        root: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
        /// Restrict execution to named cases (repeat this flag to select several).
        #[arg(long = "case")]
        cases: Vec<String>,
        /// Override all selected cases with an explicit cpu, cuda:N, or metal device.
        #[arg(long)]
        device: Option<String>,
    },
    /// Compare metrics and output fingerprints; exit nonzero on regressions or invalid comparisons.
    Compare {
        base: PathBuf,
        new: PathBuf,
        #[arg(long, default_value = "5%")]
        threshold: String,
    },
    /// Internal worker protocol; normally invoked by `run`.
    #[command(hide = true)]
    RunCase {
        #[arg(long)]
        request: PathBuf,
        #[arg(long)]
        output: PathBuf,
    },
}
#[derive(Serialize, Deserialize)]
struct Request {
    case: Case,
    inputs: Inputs,
    root: PathBuf,
}

pub(crate) fn hash(data: &[u8]) -> String {
    hash_parts(&[data])
}
pub(crate) fn hash_parts(parts: &[&[u8]]) -> String {
    let mut digest = Sha256::new();
    for part in parts {
        digest.update(part);
    }
    digest
        .finalize()
        .iter()
        .map(|byte| format!("{byte:02x}"))
        .collect()
}

fn main() -> ExitCode {
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_writer(std::io::stderr)
        .init();
    match execute(Args::parse()) {
        Ok(true) => ExitCode::SUCCESS,
        Ok(false) => ExitCode::from(1),
        Err(error) => {
            eprintln!("{error:#}");
            ExitCode::from(2)
        }
    }
}
fn execute(args: Args) -> Result<bool> {
    match args.command {
        Action::Run {
            manifest,
            root,
            output,
            cases,
            device,
        } => run(
            &manifest,
            &root,
            output.as_deref(),
            &cases,
            device.as_deref(),
        ),
        Action::Compare {
            base,
            new,
            threshold,
        } => {
            let base: RunResult = serde_json::from_slice(&fs::read(base)?)?;
            let new: RunResult = serde_json::from_slice(&fs::read(new)?)?;
            let comparison = compare::compare(&base, &new, compare::threshold(&threshold)?)?;
            print!("{}", comparison.markdown);
            Ok(!comparison.failed)
        }
        Action::RunCase { request, output } => {
            let request: Request = serde_json::from_slice(&fs::read(request)?)?;
            let measurement = run_case(&request)?;
            fs::write(output, serde_json::to_vec(&measurement)?)?;
            Ok(true)
        }
    }
}

fn run(
    path: &Path,
    root: &Path,
    output: Option<&Path>,
    names: &[String],
    device: Option<&str>,
) -> Result<bool> {
    let root = fs::canonicalize(root).context("resolve benchmark root")?;
    let raw = fs::read_to_string(path).context("read manifest")?;
    let manifest = Manifest::parse(&raw, device)?;
    for name in names {
        ensure!(
            manifest.cases.iter().any(|case| &case.name == name),
            "unknown case {name}"
        );
    }
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let output = output.map(Path::to_owned).unwrap_or_else(|| {
        PathBuf::from(format!(
            "benchmark-results/run-{timestamp}-{}.json",
            std::process::id()
        ))
    });
    ensure!(
        !output.exists(),
        "output {} already exists; choose a new filename",
        output.display()
    );
    let environment = Environment::collect(&root);
    let temp = tempfile::tempdir()?;
    let executable = std::env::current_exe()?;
    let mut results = Vec::new();
    for (index, case) in manifest
        .cases
        .iter()
        .enumerate()
        .filter(|(_, case)| names.is_empty() || names.contains(&case.name))
    {
        eprintln!("Running {} ({:?}, {})", case.name, case.kind, case.device);
        let request = temp.path().join(format!("request-{index}.json"));
        let destination = temp.path().join(format!("result-{index}.json"));
        fs::write(
            &request,
            serde_json::to_vec(&Request {
                case: case.clone(),
                inputs: manifest.inputs.clone(),
                root: root.clone(),
            })?,
        )?;
        let child = Command::new(&executable)
            .arg("run-case")
            .arg("--request")
            .arg(&request)
            .arg("--output")
            .arg(&destination)
            .current_dir(&root)
            .stdout(Stdio::null())
            .stderr(Stdio::piped())
            .output()?;
        let measurement = if child.status.success() {
            serde_json::from_slice::<Measurement>(&fs::read(&destination)?)
                .context("read child measurement")
        } else {
            Err(anyhow::anyhow!(
                "worker exited {}: {}",
                child.status,
                String::from_utf8_lossy(&child.stderr).trim()
            ))
        };
        let (measurement, error) = match measurement {
            Ok(measurement) => (Some(measurement), None),
            Err(error) => {
                eprintln!("{}: {error:#}", case.name);
                (None, Some(format!("{error:#}")))
            }
        };
        results.push(CaseResult {
            config: case.clone(),
            measurement,
            error,
        });
    }
    let valid = results.iter().all(|case| {
        case.measurement
            .as_ref()
            .is_some_and(|m| m.valid && m.output_stable)
    });
    let mut result = RunResult {
        schema_version: 1,
        timestamp_unix_ms: timestamp,
        environment,
        manifest_sha256: hash(raw.as_bytes()),
        manifest_content: raw,
        inputs: manifest.inputs,
        cases: results,
    };
    let mut devices = Vec::new();
    for gpu in result
        .cases
        .iter()
        .filter_map(|c| c.measurement.as_ref()?.gpu.as_ref())
    {
        if !devices.contains(&gpu.device) {
            devices.push(gpu.device.clone());
        }
    }
    if !devices.is_empty() {
        result.environment.gpu_devices = Some(devices);
    }
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let mut file = fs::OpenOptions::new()
        .create_new(true)
        .write(true)
        .open(&output)?;
    file.write_all(&serde_json::to_vec_pretty(&result)?)?;
    println!("{}", result::table(&result.cases));
    eprintln!("Saved {}", output.display());
    Ok(valid)
}

fn run_case(request: &Request) -> Result<Measurement> {
    request.case.validate()?;
    let case = &request.case;
    rayon::ThreadPoolBuilder::new()
        .num_threads(case.options.cpu_threads())
        .build_global()?;
    let pages = input::load(&request.root, &request.inputs)?;
    let monitor = memory::Monitor::start(
        &case.device,
        case.options.nvml_device.as_deref(),
        case.options.interval_ms(),
    );
    let load_start = Instant::now();
    let model = pipeline::Pipeline::load(&request.root, case)?;
    let model_load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    for _ in 0..case.warmup {
        for batch in pages.chunks(case.options.batch_size()) {
            let images: Vec<_> = batch.iter().map(|page| &page.image).collect();
            model.infer(&images, case)?;
        }
    }
    let mut samples = Vec::new();
    let mut measured_seconds = 0.0;
    for repetition in 0..case.repetitions {
        for batch in pages.chunks(case.options.batch_size()) {
            let images: Vec<_> = batch.iter().map(|page| &page.image).collect();
            let start = Instant::now();
            let outputs = model.infer(&images, case)?;
            let elapsed = start.elapsed().as_secs_f64();
            measured_seconds += elapsed;
            ensure!(outputs.len() == batch.len(), "output count mismatch");
            for (page, output) in batch.iter().zip(outputs) {
                samples.push(PageSample {
                    page_id: page.id.clone(),
                    input_sha256: page.sha256.clone(),
                    repetition,
                    latency_ms: elapsed * 1000.0 / batch.len() as f64,
                    output_sha256: hash(output.text.as_bytes()),
                    document_sha256: output
                        .document
                        .as_ref()
                        .map(serde_json::to_vec)
                        .transpose()?
                        .map(|bytes| hash(&bytes)),
                    output_characters: output.text.chars().count(),
                    diagnostics: output.diagnostics,
                });
            }
        }
    }
    ensure!(measured_seconds > 0.0, "measurement timer returned zero");
    let latency_ms =
        Statistics::calculate(&samples.iter().map(|s| s.latency_ms).collect::<Vec<_>>())?;
    let chars = samples
        .iter()
        .map(|s| s.output_characters as f64)
        .sum::<f64>();
    let output_stable = result::fingerprints(&samples)
        .values()
        .all(|set| set.len() == 1);
    let (gpu, gpu_warning) = monitor.finish();
    let mut warnings = Vec::new();
    if let Some(warning) = gpu_warning {
        warnings.push(warning);
    }
    let gpu_valid = gpu.as_ref().is_none_or(|gpu| gpu.valid);
    let valid = gpu_valid && samples.iter().all(|s| s.diagnostics == 0);
    if samples.iter().any(|s| s.diagnostics > 0) {
        warnings
            .push("PageParser reported diagnostics; inspect page sample diagnostic counts".into());
    }
    if !output_stable {
        warnings.push("output varies across repetitions".into());
    }
    if !gpu_valid {
        warnings.push("GPU isolation/sampling validation failed; measurement is invalid".into());
    }
    let diagnostic_pages: BTreeSet<_> = samples
        .iter()
        .filter(|s| s.diagnostics > 0)
        .map(|s| &s.page_id)
        .collect();
    if !diagnostic_pages.is_empty() {
        warnings.push(format!(
            "{} pages had non-fatal diagnostics",
            diagnostic_pages.len()
        ));
    }
    Ok(Measurement {
        model_load_ms,
        measured_seconds,
        latency_ms,
        pages_per_second: samples.len() as f64 / measured_seconds,
        tokens_per_second: None,
        output_characters_per_second: (case.kind == Kind::Vl).then_some(chars / measured_seconds),
        rate_basis: if case.kind == Kind::Vl {
            "Unicode output characters; PageParser exposes no generated-token count"
        } else {
            "pages"
        }
        .into(),
        latency_basis: if case.options.batch_size() > 1 {
            "amortized batch wall time per page"
        } else {
            "single-page wall time"
        }
        .into(),
        host_peak_bytes: memory::host_peak_bytes(),
        gpu,
        warnings,
        samples,
        output_stable,
        valid,
    })
}
