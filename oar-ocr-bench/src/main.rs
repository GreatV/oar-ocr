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
        /// Save each page's final-repeat text as `<DIR>/<case>/<image-stem>.md`.
        #[arg(long)]
        save_outputs: Option<PathBuf>,
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
    #[serde(default)]
    save_outputs: Option<PathBuf>,
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
            save_outputs,
        } => run(
            &manifest,
            &root,
            output,
            &cases,
            device.as_deref(),
            &inputs,
            save_outputs,
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
    save_outputs: Option<PathBuf>,
) -> Result<bool> {
    let root = fs::canonicalize(root).context("resolve benchmark root")?;
    let save_outputs = save_outputs.map(std::path::absolute).transpose()?;
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
    // Record the checkout before saved outputs can make it look dirty.
    let environment = Environment::collect(&root);
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
                save_outputs: save_outputs.clone(),
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
        environment,
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
    let output_paths = request
        .save_outputs
        .as_deref()
        .map(|directory| page_output_paths(directory, &case.name, &pages))
        .transpose()?;
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
    for repeat in 0..case.repetitions {
        for (batch_index, batch) in pages.chunks(batch_size).enumerate() {
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
            // Every page in a batch completes when the whole batch does.
            latencies.extend(std::iter::repeat_n(elapsed * 1000.0, batch.len()));
            if repeat + 1 == case.repetitions
                && let Some(paths) = &output_paths
            {
                for (path, text) in paths[batch_index * batch_size..].iter().zip(&outputs) {
                    fs::write(path, text)
                        .with_context(|| format!("save page output {}", path.display()))?;
                }
            }
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

fn page_output_paths(directory: &Path, case: &str, pages: &[input::Page]) -> Result<Vec<PathBuf>> {
    // Check the raw name: `Path` normalization would accept aliases like `ocr/.`.
    ensure!(
        !case.is_empty() && case != "." && case != ".." && !case.contains(['/', '\\']),
        "saving outputs requires a case name without path components"
    );
    let directory = directory.join(case);
    let paths = pages
        .iter()
        .map(|page| {
            let name = Path::new(&page.id)
                .file_name()
                .context("page has no filename")?;
            Ok(directory.join(name).with_extension("md"))
        })
        .collect::<Result<Vec<_>>>()?;
    // Compare case-folded names so case-insensitive filesystems cannot merge two pages.
    ensure!(
        paths
            .iter()
            .map(|path| path.to_string_lossy().to_lowercase())
            .collect::<std::collections::BTreeSet<_>>()
            .len()
            == paths.len(),
        "saving outputs requires image stems that are unique, ignoring case, within each case"
    );
    // Leftover files from an earlier run would be scored alongside this one.
    ensure!(
        fs::read_dir(&directory).map_or(true, |mut entries| entries.next().is_none()),
        "output directory {} is not empty; choose a fresh --save-outputs directory",
        directory.display()
    );
    fs::create_dir_all(directory)?;
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pages(ids: &[&str]) -> Vec<input::Page> {
        ids.iter()
            .map(|id| input::Page {
                id: (*id).into(),
                image: image::RgbImage::new(1, 1),
            })
            .collect()
    }

    #[test]
    fn saved_pages_use_case_directories_and_preserve_stems() {
        let directory = tempfile::tempdir().unwrap();
        let pages = pages(&[
            "images/document.pdf_4.JPG",
            "elsewhere/report.v2.png",
            "图像/页面.tiff",
            "paper_pg1_repeat1.png",
        ]);
        let paths = page_output_paths(directory.path(), "ocr-tiny", &pages).unwrap();
        let case = directory.path().join("ocr-tiny");
        assert!(case.is_dir());
        assert_eq!(
            paths,
            [
                "document.pdf_4.md",
                "report.v2.md",
                "页面.md",
                "paper_pg1_repeat1.md"
            ]
            .map(|name| case.join(name))
        );
        let other = page_output_paths(directory.path(), "structure-v3", &pages).unwrap();
        assert_ne!(paths, other);
        assert!(directory.path().join("structure-v3").is_dir());
        // A case directory holding an earlier run's output is refused.
        std::fs::write(&paths[0], "old").unwrap();
        assert!(page_output_paths(directory.path(), "ocr-tiny", &pages).is_err());
    }

    #[test]
    fn saving_rejects_colliding_names_and_case_path_components() {
        let directory = tempfile::tempdir().unwrap();
        let collision = pages(&["first/page.png", "second/page.jpg"]);
        assert!(page_output_paths(directory.path(), "ocr-tiny", &collision).is_err());
        let case_alias = pages(&["first/Page.png", "second/page.png"]);
        assert!(page_output_paths(directory.path(), "ocr-tiny", &case_alias).is_err());
        assert!(!directory.path().join("ocr-tiny").exists());
        for case in [
            "../escape",
            ".",
            "nested/case",
            "",
            "ocr/.",
            "ocr/",
            r"ocr\x",
        ] {
            assert!(page_output_paths(directory.path(), case, &pages(&["page.png"])).is_err());
        }
    }
}
