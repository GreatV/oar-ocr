mod compare;
mod input;
mod manifest;
mod memory;
mod pipeline;
mod result;

use anyhow::{Context, Result, ensure};
use clap::{Parser, Subcommand};
use manifest::{Case, Inputs, Kind, Manifest};
use result::{CaseResult, Environment, Measurement, RunResult, Statistics};
use serde::{Deserialize, Serialize};
use std::{
    fs,
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
    /// Run each selected case in a fresh subprocess and write one JSON report.
    Run {
        #[arg(long, default_value = "oar-ocr-bench/manifests/default.toml")]
        manifest: PathBuf,
        #[arg(long, default_value = ".")]
        root: PathBuf,
        #[arg(long)]
        output: Option<PathBuf>,
        /// Restrict execution to named cases (repeatable).
        #[arg(long = "case")]
        cases: Vec<String>,
        /// Override every case's device: auto, cpu, cuda:N, or metal.
        #[arg(long)]
        device: Option<String>,
        /// Replace the manifest inputs with image files or directories (repeatable).
        #[arg(long = "input")]
        inputs: Vec<PathBuf>,
    },
    /// Compare two reports; exit nonzero on regressions or incomparable cases.
    Compare {
        base: PathBuf,
        new: PathBuf,
        #[arg(long, default_value = "5%")]
        threshold: String,
    },
    /// Internal worker protocol used by `run`.
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
            inputs,
        } => run(&manifest, &root, output, &cases, device.as_deref(), &inputs),
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
            fs::write(output, serde_json::to_vec(&run_case(&request)?)?)?;
            Ok(true)
        }
    }
}

fn run(
    path: &Path,
    root: &Path,
    output: Option<PathBuf>,
    names: &[String],
    device: Option<&str>,
    inputs: &[PathBuf],
) -> Result<bool> {
    let root = fs::canonicalize(root).context("resolve benchmark root")?;
    let input_override = if inputs.is_empty() {
        None
    } else {
        Some(input::from_paths(&root, inputs)?)
    };
    let manifest = Manifest::parse(
        &fs::read_to_string(path).context("read manifest")?,
        device,
        input_override,
    )?;
    for name in names {
        ensure!(
            manifest.cases.iter().any(|case| &case.name == name),
            "unknown case {name}"
        );
    }
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_millis();
    let output = output.unwrap_or_else(|| format!("benchmark-results/run-{timestamp}.json").into());
    ensure!(!output.exists(), "{} already exists", output.display());
    let temp = tempfile::tempdir()?;
    let executable = std::env::current_exe()?;
    let mut results = Vec::new();
    for case in manifest
        .cases
        .iter()
        .filter(|case| names.is_empty() || names.contains(&case.name))
    {
        eprintln!("Running {} ({})", case.name, case.device);
        let request = temp.path().join("request.json");
        let destination = temp.path().join("result.json");
        fs::write(
            &request,
            serde_json::to_vec(&Request {
                case: case.clone(),
                inputs: manifest.inputs.clone(),
                root: root.clone(),
            })?,
        )?;
        // Worker warnings go straight to the terminal; failures are kept in the report.
        let status = Command::new(&executable)
            .args(["run-case", "--request"])
            .arg(&request)
            .arg("--output")
            .arg(&destination)
            .current_dir(&root)
            .stdout(Stdio::null())
            .status()?;
        let (measurement, error) = if status.success() {
            (
                Some(serde_json::from_slice(&fs::read(&destination)?)?),
                None,
            )
        } else {
            eprintln!("{}: worker exited with {status}", case.name);
            (None, Some(format!("worker exited with {status}")))
        };
        results.push(CaseResult {
            case: case.clone(),
            measurement,
            error,
        });
    }
    let all_succeeded = results.iter().all(|case| case.measurement.is_some());
    let result = RunResult {
        timestamp_unix_ms: timestamp,
        environment: Environment::collect(&root),
        cases: results,
    };
    if let Some(parent) = output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    fs::write(&output, serde_json::to_vec_pretty(&result)?)?;
    println!("{}", result::table(&result.cases));
    eprintln!("Saved {}", output.display());
    Ok(all_succeeded)
}

fn run_case(request: &Request) -> Result<Measurement> {
    let case = &request.case;
    rayon::ThreadPoolBuilder::new()
        .num_threads(case.options.cpu_threads())
        .build_global()?;
    let pages = input::load(&request.root, &request.inputs)?;
    let load_start = Instant::now();
    let device = pipeline::DeviceSelection::resolve(case)?;
    let device_name = device.name()?;
    let sampler = memory::GpuSampler::start(&device_name);
    let model = pipeline::Pipeline::load(&request.root, case, device)?;
    let load_ms = load_start.elapsed().as_secs_f64() * 1000.0;
    let batch_size = case.options.batch_size();
    for _ in 0..case.warmup {
        for batch in pages.chunks(batch_size) {
            let images: Vec<_> = batch.iter().map(|page| &page.image).collect();
            model.infer(&images, case)?;
        }
    }
    let mut latencies = Vec::new();
    let mut chars = 0usize;
    let mut measured = 0.0;
    for _ in 0..case.repetitions {
        for batch in pages.chunks(batch_size) {
            let images: Vec<_> = batch.iter().map(|page| &page.image).collect();
            let start = Instant::now();
            let outputs = model.infer(&images, case)?;
            let elapsed = start.elapsed().as_secs_f64();
            ensure!(outputs.len() == batch.len(), "output count mismatch");
            measured += elapsed;
            chars += outputs
                .iter()
                .map(|text| text.chars().count())
                .sum::<usize>();
            // Batched pages share the batch wall time.
            latencies.extend(std::iter::repeat_n(
                elapsed * 1000.0 / batch.len() as f64,
                batch.len(),
            ));
        }
    }
    ensure!(measured > 0.0, "measurement timer returned zero");
    Ok(Measurement {
        pages: pages.iter().map(|page| page.id.clone()).collect(),
        load_ms,
        latency_ms: Statistics::calculate(&latencies)?,
        pages_per_second: latencies.len() as f64 / measured,
        output_chars_per_second: (case.kind == Kind::Vl).then_some(chars as f64 / measured),
        host_peak_bytes: memory::host_peak_bytes(),
        gpu: sampler.finish(),
        device: device_name,
    })
}
