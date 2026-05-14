//! Hierarchical Speculative Decoding demo / harness, shared across backends.
//!
//! Two passes per run:
//!
//! 1. **Correctness check** (skip with `--skip-check`).
//!    Use the baseline `generate(...)` output as a perfect draft and run HSD
//!    with τ=1.0. The output must match the baseline exactly. Any divergence
//!    indicates a bug somewhere in the HSD pipeline.
//!
//! 2. **Performance measurement.**
//!    Run HSD with the user-supplied draft (or the baseline output if no
//!    `--draft-text` / `--draft-file` is given) at τ=0.75 and report SR_e2e
//!    along with Average Acceptance Length, fallback steps, and the per-stage
//!    breakdown.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p oar-ocr-vl --release --features hsd,download-binaries --example hsd_demo -- \
//!     --backend hunyuanocr \
//!     --model-dir models/HunyuanOCR \
//!     --device cuda:0 \
//!     --image document.jpg \
//!     --max-tokens 4096
//! ```
//!
//! Pass `--draft-text "..."` or `--draft-file path` to use a real drafter's
//! output as the speculative draft instead of the baseline (which would
//! otherwise produce an artificially high AAL).

mod utils;

#[cfg(not(feature = "hsd"))]
fn main() {
    eprintln!("This example requires the `hsd` feature. Re-run with `--features hsd`.");
    std::process::exit(1);
}

#[cfg(feature = "hsd")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    imp::run()
}

#[cfg(feature = "hsd")]
mod imp {

use super::utils;

use clap::{Parser, ValueEnum};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use image::RgbImage;
use oar_ocr_core::utils::load_image;
use oar_ocr_core::{
    domain::structure::{LayoutElement, LayoutElementType},
    processors::BoundingBox,
};
use oar_ocr_vl::hsd::types::{Draft, HsdStats};
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::{GlmOcr, HunyuanOcr, MinerU, PaddleOcrVl, PaddleOcrVlTask};
use utils::{
    DEMO_DEFAULT_MAX_CANDIDATES, DEMO_DEFAULT_MAX_SUFFIX_LEN, auto_tune_hsd_oracle, make_hsd_cfg,
    print_diff, print_hsd_stats, print_preview,
};

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Backend {
    #[value(name = "hunyuanocr")]
    HunyuanOcr,
    #[value(name = "paddleocr_vl")]
    PaddleOcrVl,
    #[value(name = "mineru")]
    MinerU,
    #[value(name = "glmocr")]
    GlmOcr,
}

#[derive(Copy, Clone, Debug, ValueEnum)]
enum Task {
    Ocr,
    Table,
    Chart,
    Formula,
    Spotting,
    Seal,
}

impl Task {
    fn to_native(self) -> PaddleOcrVlTask {
        match self {
            Task::Ocr => PaddleOcrVlTask::Ocr,
            Task::Table => PaddleOcrVlTask::Table,
            Task::Chart => PaddleOcrVlTask::Chart,
            Task::Formula => PaddleOcrVlTask::Formula,
            Task::Spotting => PaddleOcrVlTask::Spotting,
            Task::Seal => PaddleOcrVlTask::Seal,
        }
    }
}

#[derive(Parser)]
#[command(name = "hsd_demo")]
#[command(about = "HSD correctness check + performance demo across VL backends")]
struct Args {
    /// Target backend.
    #[arg(long, value_enum, default_value_t = Backend::HunyuanOcr)]
    backend: Backend,

    /// Path to the backend model directory.
    #[arg(long)]
    model_dir: PathBuf,

    /// Path to a single page / region image.
    #[arg(long)]
    image: PathBuf,

    /// Device: cpu, cuda, cuda:N, or metal.
    #[arg(long, default_value = "cuda:0")]
    device: String,

    /// Maximum tokens to generate.
    #[arg(long, default_value_t = 4096)]
    max_tokens: usize,

    /// Instruction prompt. Defaults depend on `--backend`:
    /// - hunyuanocr: full-page spotting prompt
    /// - glmocr:     "Read the text in this image."
    /// - mineru:     "Read the text in this image:"
    ///
    /// The default for paddleocr_vl is encoded by `--task`, not by this flag.
    #[arg(long)]
    instruction: Option<String>,

    /// PaddleOCR-VL task. Ignored for other backends.
    #[arg(long, value_enum, default_value_t = Task::Ocr)]
    task: Task,

    /// Optional pre-computed draft text from a real drafter pipeline. If
    /// omitted, the baseline output is reused as an oracle draft (upper-bound
    /// AAL, still useful for wall-clock validation).
    #[arg(long)]
    draft_text: Option<String>,

    /// File whose contents are used as the draft. Mutually exclusive with
    /// `--draft-text`.
    #[arg(long)]
    draft_file: Option<PathBuf>,

    /// Skip the τ=1.0 correctness check (saves one HSD run).
    #[arg(long)]
    skip_check: bool,

    /// Acceptance threshold for the perf pass.
    #[arg(long, default_value_t = 0.75)]
    tau: f32,

    /// HunyuanOCR-only: exercise `generate_hsd_full` with a single full-page
    /// region draft. Minimal real Stage-1 path; dataset region benchmarks
    /// should use `hsd_omnidocbench`.
    #[arg(long)]
    stage1_full: bool,

    /// Reference window length n (paper §3.2).
    #[arg(long, default_value_t = 3)]
    window_len: usize,

    /// Maximum prefix-tree candidate count per verification step.
    #[arg(long, default_value_t = DEMO_DEFAULT_MAX_CANDIDATES)]
    max_candidates: usize,

    /// Maximum candidate suffix length.
    #[arg(long, default_value_t = DEMO_DEFAULT_MAX_SUFFIX_LEN)]
    max_suffix_len: usize,
}

enum DraftSource {
    Oracle,
    Cli(String),
    File(String),
}

impl DraftSource {
    fn label(&self) -> &'static str {
        match self {
            DraftSource::Oracle => "oracle (baseline output)",
            DraftSource::Cli(_) => "--draft-text",
            DraftSource::File(_) => "--draft-file",
        }
    }

    fn text(&self) -> Option<&str> {
        match self {
            DraftSource::Oracle => None,
            DraftSource::Cli(s) | DraftSource::File(s) => Some(s.as_str()),
        }
    }
}

fn default_instruction(backend: Backend) -> &'static str {
    match backend {
        Backend::HunyuanOcr => {
            "Detect and recognize text in the image, and output the text coordinates in a formatted manner."
        }
        Backend::GlmOcr => "Read the text in this image.",
        Backend::MinerU => "Read the text in this image:",
        // PaddleOCR-VL builds its prompt from --task; instruction is unused.
        Backend::PaddleOcrVl => "",
    }
}

pub fn run() -> Result<(), Box<dyn std::error::Error>> {
    utils::init_tracing();
    let args = Args::parse();
    let device = parse_device(&args.device)?;
    let image = load_image(&args.image)?;
    let instruction = args
        .instruction
        .clone()
        .unwrap_or_else(|| default_instruction(args.backend).to_string());

    let draft_source = match (&args.draft_text, &args.draft_file) {
        (Some(t), None) => DraftSource::Cli(t.clone()),
        (None, Some(p)) => DraftSource::File(fs::read_to_string(p)?),
        (None, None) => DraftSource::Oracle,
        (Some(_), Some(_)) => {
            return Err("--draft-text and --draft-file are mutually exclusive".into());
        }
    };

    if args.stage1_full && !matches!(args.backend, Backend::HunyuanOcr) {
        return Err("--stage1-full is only supported with --backend hunyuanocr".into());
    }

    match args.backend {
        Backend::HunyuanOcr => run_hunyuanocr(&args, device, &image, &instruction, &draft_source),
        Backend::GlmOcr => run_glmocr(&args, device, &image, &instruction, &draft_source),
        Backend::MinerU => run_mineru(&args, device, &image, &instruction, &draft_source),
        Backend::PaddleOcrVl => run_paddleocr_vl(&args, device, &image, &draft_source),
    }
}

fn run_hunyuanocr(
    args: &Args,
    device: candle_core::Device,
    image: &RgbImage,
    instruction: &str,
    draft_source: &DraftSource,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading HunyuanOCR from {}", args.model_dir.display());
    let t_load = Instant::now();
    let model = HunyuanOcr::from_dir(&args.model_dir, device)?;
    println!("Model load: {:?}", t_load.elapsed());
    println!("Loaded image: {}x{}", image.width(), image.height());

    let (baseline_tokens, baseline_text, baseline_dur) =
        baseline_pass("HunyuanOCR", || {
            let t = Instant::now();
            let res = model.generate_tokens(std::slice::from_ref(image), &[instruction], args.max_tokens);
            let dur = t.elapsed();
            let tokens = first_result(res, "HunyuanOCR baseline")?;
            let text = model.decode_tokens(&tokens)?;
            Ok::<_, Box<dyn std::error::Error>>((tokens, text, dur))
        })?;
    print_preview("BASELINE", &baseline_text);

    if !args.skip_check {
        println!("\n[2/3] HSD τ=1.0 correctness check (oracle draft = baseline)...");
        let cfg = make_hsd_cfg(
            args.max_tokens,
            1.0,
            args.window_len,
            args.max_candidates,
            args.max_suffix_len,
            false,
        );
        let t = Instant::now();
        let token_drafts = vec![Draft::new(baseline_tokens.clone())];
        let (hsd_tokens, _stats) =
            model.generate_hsd_tokens_with_token_drafts(image, instruction, &token_drafts, &cfg)?;
        let dur = t.elapsed();
        if hsd_tokens == baseline_tokens {
            println!("      ✓ τ=1.0 HSD output matches baseline ({:?})", dur);
        } else {
            let hsd_text = model.decode_tokens(&hsd_tokens)?.trim().to_string();
            eprintln!("      ✗ τ=1.0 HSD output diverges from baseline.");
            print_diff(&baseline_text, &hsd_text);
            return Err("HSD τ=1.0 mismatch".into());
        }
    } else {
        println!("\n[2/3] correctness check skipped");
    }

    println!(
        "\n[3/3] HSD τ={:.2} performance pass (draft source: {})...",
        args.tau,
        draft_source.label()
    );
    let (eff_max_candidates, eff_max_suffix_len, note) = auto_tune_hsd_oracle(
        matches!(draft_source, DraftSource::Oracle),
        args.max_candidates,
        args.max_suffix_len,
        args.max_tokens,
    );
    if let Some(n) = note {
        println!("{n}");
    }
    let cfg = make_hsd_cfg(
        args.max_tokens,
        args.tau,
        args.window_len,
        eff_max_candidates,
        eff_max_suffix_len,
        args.stage1_full,
    );
    let t_hsd = Instant::now();
    let (hsd_text, stats): (String, HsdStats) = if args.stage1_full {
        let text = match draft_source {
            DraftSource::Oracle => baseline_text.as_str(),
            DraftSource::Cli(s) | DraftSource::File(s) => s.as_str(),
        };
        let element = full_page_text_element(image, text);
        model.generate_hsd_full(
            image,
            oar_ocr_vl::HunyuanHsdPrompts {
                page: instruction,
                region: instruction,
            },
            std::slice::from_ref(&element),
            &[],
            |elem| elem.text.iter().cloned().collect(),
            &cfg,
        )?
    } else {
        match draft_source {
            DraftSource::Oracle => {
                let token_drafts = vec![Draft::new(baseline_tokens.clone())];
                model.generate_hsd_with_token_drafts(image, instruction, &token_drafts, &cfg)?
            }
            DraftSource::Cli(s) | DraftSource::File(s) => {
                model.generate_hsd(image, instruction, std::slice::from_ref(s), &cfg)?
            }
        }
    };
    let hsd_dur = t_hsd.elapsed();
    print_preview("HSD OUTPUT", &hsd_text);
    print_hsd_stats(baseline_dur, hsd_dur, &stats, args.stage1_full);
    oracle_note(draft_source);
    Ok(())
}

fn run_glmocr(
    args: &Args,
    device: candle_core::Device,
    image: &RgbImage,
    instruction: &str,
    draft_source: &DraftSource,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading GLM-OCR from {}", args.model_dir.display());
    let model = GlmOcr::from_dir(&args.model_dir, device)?;

    let (baseline_tokens, baseline_text, baseline_dur) =
        baseline_pass("GLM-OCR", || {
            let t = Instant::now();
            let res = model.generate_tokens(std::slice::from_ref(image), &[instruction], args.max_tokens);
            let dur = t.elapsed();
            let tokens = first_result(res, "GLM-OCR baseline")?;
            let text = model.decode_tokens(&tokens)?;
            Ok::<_, Box<dyn std::error::Error>>((tokens, text, dur))
        })?;

    if !args.skip_check {
        println!("\n[2/3] HSD τ=1.0 correctness check (oracle draft = baseline)...");
        let cfg = make_hsd_cfg(
            args.max_tokens,
            1.0,
            args.window_len,
            args.max_candidates,
            args.max_suffix_len,
            false,
        );
        let t = Instant::now();
        let token_drafts = vec![Draft::new(baseline_tokens.clone())];
        let (hsd_text, _) =
            model.generate_hsd_with_token_drafts(image, instruction, &token_drafts, &cfg)?;
        let dur = t.elapsed();
        if hsd_text == baseline_text {
            println!("      ✓ matches baseline ({:?})", dur);
        } else {
            eprintln!("      ✗ diverges from baseline.");
            print_diff(&baseline_text, &hsd_text);
            return Err("τ=1.0 mismatch".into());
        }
    } else {
        println!("\n[2/3] correctness check skipped");
    }

    println!("\n[3/3] HSD τ={:.2} performance pass...", args.tau);
    let (eff_max_candidates, eff_max_suffix_len, note) = auto_tune_hsd_oracle(
        matches!(draft_source, DraftSource::Oracle),
        args.max_candidates,
        args.max_suffix_len,
        args.max_tokens,
    );
    if let Some(n) = note {
        println!("{n}");
    }
    let cfg = make_hsd_cfg(
        args.max_tokens,
        args.tau,
        args.window_len,
        eff_max_candidates,
        eff_max_suffix_len,
        false,
    );
    let t = Instant::now();
    let (hsd_text, stats) = match draft_source {
        DraftSource::Oracle => {
            let token_drafts = vec![Draft::new(baseline_tokens.clone())];
            model.generate_hsd_with_token_drafts(image, instruction, &token_drafts, &cfg)?
        }
        DraftSource::Cli(s) | DraftSource::File(s) => {
            model.generate_hsd(image, instruction, std::slice::from_ref(s), &cfg)?
        }
    };
    let hsd_dur = t.elapsed();
    print_preview("HSD OUTPUT", &hsd_text);
    print_hsd_stats(baseline_dur, hsd_dur, &stats, false);
    oracle_note(draft_source);
    Ok(())
}

fn run_mineru(
    args: &Args,
    device: candle_core::Device,
    image: &RgbImage,
    instruction: &str,
    draft_source: &DraftSource,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("Loading MinerU2.5 from {}", args.model_dir.display());
    let model = MinerU::from_dir(&args.model_dir, device)?;

    let (baseline_tokens, baseline_text, baseline_dur) =
        baseline_pass("MinerU2.5", || {
            let t = Instant::now();
            let res = model.generate_tokens(std::slice::from_ref(image), &[instruction], args.max_tokens);
            let dur = t.elapsed();
            let tokens = first_result(res, "MinerU2.5 baseline")?;
            let text = model.decode_tokens(&tokens)?;
            Ok::<_, Box<dyn std::error::Error>>((tokens, text, dur))
        })?;

    if !args.skip_check {
        println!("\n[2/3] HSD τ=1.0 correctness check (oracle draft = baseline)...");
        let cfg = make_hsd_cfg(
            args.max_tokens,
            1.0,
            args.window_len,
            args.max_candidates,
            args.max_suffix_len,
            false,
        );
        let t = Instant::now();
        let token_drafts = vec![Draft::new(baseline_tokens.clone())];
        let (hsd_text, _) =
            model.generate_hsd_with_token_drafts(image, instruction, &token_drafts, &cfg)?;
        let dur = t.elapsed();
        if hsd_text == baseline_text {
            println!("      ✓ matches baseline ({:?})", dur);
        } else {
            eprintln!("      ✗ diverges from baseline.");
            print_diff(&baseline_text, &hsd_text);
            return Err("τ=1.0 mismatch".into());
        }
    } else {
        println!("\n[2/3] correctness check skipped");
    }

    println!("\n[3/3] HSD τ={:.2} performance pass...", args.tau);
    let (eff_max_candidates, eff_max_suffix_len, note) = auto_tune_hsd_oracle(
        matches!(draft_source, DraftSource::Oracle),
        args.max_candidates,
        args.max_suffix_len,
        args.max_tokens,
    );
    if let Some(n) = note {
        println!("{n}");
    }
    let cfg = make_hsd_cfg(
        args.max_tokens,
        args.tau,
        args.window_len,
        eff_max_candidates,
        eff_max_suffix_len,
        false,
    );
    let t = Instant::now();
    let (hsd_text, stats) = match draft_source {
        DraftSource::Oracle => {
            let token_drafts = vec![Draft::new(baseline_tokens.clone())];
            model.generate_hsd_with_token_drafts(image, instruction, &token_drafts, &cfg)?
        }
        DraftSource::Cli(s) | DraftSource::File(s) => {
            model.generate_hsd(image, instruction, std::slice::from_ref(s), &cfg)?
        }
    };
    let hsd_dur = t.elapsed();
    print_preview("HSD OUTPUT", &hsd_text);
    print_hsd_stats(baseline_dur, hsd_dur, &stats, false);
    oracle_note(draft_source);
    Ok(())
}

fn run_paddleocr_vl(
    args: &Args,
    device: candle_core::Device,
    image: &RgbImage,
    draft_source: &DraftSource,
) -> Result<(), Box<dyn std::error::Error>> {
    let task = args.task.to_native();
    println!("Loading PaddleOCR-VL from {}", args.model_dir.display());
    let model = PaddleOcrVl::from_dir(&args.model_dir, device)?;

    let (baseline_tokens, baseline_text, baseline_dur) =
        baseline_pass("PaddleOCR-VL", || {
            let t = Instant::now();
            let res = model.generate_tokens(std::slice::from_ref(image), &[task], args.max_tokens);
            let dur = t.elapsed();
            let tokens = first_result(res, "PaddleOCR-VL baseline")?;
            let (_, pp) = model.decode_tokens(&tokens, task)?;
            Ok::<_, Box<dyn std::error::Error>>((tokens, pp, dur))
        })?;

    if !args.skip_check {
        println!("\n[2/3] HSD τ=1.0 correctness check (oracle draft = baseline)...");
        let cfg = make_hsd_cfg(
            args.max_tokens,
            1.0,
            args.window_len,
            args.max_candidates,
            args.max_suffix_len,
            false,
        );
        let t = Instant::now();
        let token_drafts = vec![Draft::new(baseline_tokens.clone())];
        let (hsd_text, _) = model.generate_hsd_with_token_drafts(image, task, &token_drafts, &cfg)?;
        let dur = t.elapsed();
        if hsd_text == baseline_text {
            println!("      ✓ matches baseline ({:?})", dur);
        } else {
            eprintln!("      ✗ diverges from baseline.");
            print_diff(&baseline_text, &hsd_text);
            return Err("τ=1.0 mismatch".into());
        }
    } else {
        println!("\n[2/3] correctness check skipped");
    }

    println!("\n[3/3] HSD τ={:.2} performance pass...", args.tau);
    let (eff_max_candidates, eff_max_suffix_len, note) = auto_tune_hsd_oracle(
        matches!(draft_source, DraftSource::Oracle),
        args.max_candidates,
        args.max_suffix_len,
        args.max_tokens,
    );
    if let Some(n) = note {
        println!("{n}");
    }
    let cfg = make_hsd_cfg(
        args.max_tokens,
        args.tau,
        args.window_len,
        eff_max_candidates,
        eff_max_suffix_len,
        false,
    );
    let t = Instant::now();
    let (hsd_text, stats) = match draft_source {
        DraftSource::Oracle => {
            let token_drafts = vec![Draft::new(baseline_tokens.clone())];
            model.generate_hsd_with_token_drafts(image, task, &token_drafts, &cfg)?
        }
        DraftSource::Cli(s) | DraftSource::File(s) => {
            model.generate_hsd(image, task, std::slice::from_ref(s), &cfg)?
        }
    };
    let hsd_dur = t.elapsed();
    print_preview("HSD OUTPUT", &hsd_text);
    print_hsd_stats(baseline_dur, hsd_dur, &stats, false);
    oracle_note(draft_source);
    Ok(())
}

fn baseline_pass<F, T>(label: &str, f: F) -> Result<T, Box<dyn std::error::Error>>
where
    F: FnOnce() -> Result<T, Box<dyn std::error::Error>>,
{
    println!("\n[1/3] {label} baseline generate...");
    f()
}

fn first_result<T, E: std::fmt::Display>(
    results: Vec<Result<T, E>>,
    label: &str,
) -> Result<T, Box<dyn std::error::Error>> {
    match results.into_iter().next() {
        Some(Ok(t)) => Ok(t),
        Some(Err(e)) => Err(format!("{label} failed: {e}").into()),
        None => Err(format!("{label} returned no results").into()),
    }
}

fn full_page_text_element(image: &RgbImage, text: &str) -> LayoutElement {
    LayoutElement::new(
        BoundingBox::from_coords(0.0, 0.0, image.width() as f32, image.height() as f32),
        LayoutElementType::Text,
        1.0,
    )
    .with_text(text.to_string())
}

fn oracle_note(draft_source: &DraftSource) {
    if draft_source.text().is_none() {
        println!(
            "(Oracle draft = baseline output — AAL is an upper bound; \
             realistic drafters give lower numbers.)"
        );
    }
}

} // mod imp
