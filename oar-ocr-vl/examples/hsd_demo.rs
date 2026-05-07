//! Hierarchical Speculative Decoding demo / harness for HunyuanOCR.
//!
//! Two passes per run:
//!
//! 1. **Correctness check** (skip with `--skip-check`).
//!    Use the baseline `generate(...)` output as a perfect draft and run HSD
//!    with τ=1.0. The output must match the baseline exactly. Any divergence
//!    indicates a bug somewhere in the HSD pipeline (KV gather, tree mask,
//!    position ids, log-softmax precision).
//!
//! 2. **Performance measurement.**
//!    Run HSD with the user-supplied draft (or the baseline output if
//!    `--draft-text` is omitted) at τ=0.75 and report SR_e2e along with
//!    Average Acceptance Length, fallback steps, and the per-stage
//!    breakdown.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p oar-ocr-vl --release --features cuda --example hsd_demo -- \
//!     --model-dir models/HunyuanOCR \
//!     --device cuda:0 \
//!     --image document.jpg \
//!     --max-tokens 4096
//! ```
//!
//! Pass `--draft-text "..."` to use a real drafter's output as the speculative
//! draft instead of the baseline (which would otherwise produce an
//! artificially high AAL).

mod utils;

use clap::Parser;
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use oar_ocr_core::utils::load_image;
use oar_ocr_core::{
    domain::structure::{LayoutElement, LayoutElementType},
    processors::BoundingBox,
};
use oar_ocr_vl::HunyuanOcr;
use oar_ocr_vl::hsd::types::Draft;
use oar_ocr_vl::utils::parse_device;
use utils::{make_hsd_cfg, print_diff, print_hsd_stats, print_preview};

#[derive(Parser)]
#[command(name = "hsd_demo")]
#[command(about = "HunyuanOCR HSD correctness check + performance demo")]
struct Args {
    /// Path to the HunyuanOCR model directory.
    #[arg(long)]
    model_dir: PathBuf,

    /// Path to a single page / region image.
    #[arg(long)]
    image: PathBuf,

    /// Device to run on: cpu, cuda, cuda:N, or metal.
    #[arg(long, default_value = "cuda:0")]
    device: String,

    /// Maximum tokens to generate.
    #[arg(long, default_value_t = 4096)]
    max_tokens: usize,

    /// Instruction prompt (matches the existing hunyuanocr example default).
    #[arg(
        long,
        default_value = "Detect and recognize text in the image, and output the text coordinates in a formatted manner."
    )]
    instruction: String,

    /// Optional pre-computed draft text. Use this to feed in a real pipeline
    /// drafter's output (e.g. PaddleOCR-VL or a PP-StructureV3-style
    /// doc_parser pipeline). If omitted, the
    /// baseline output is reused as the draft, which gives an upper-bound AAL
    /// (oracle scenario) but is still useful for validating wall-clock
    /// improvements.
    #[arg(long)]
    draft_text: Option<String>,

    /// Path to a file whose contents will be loaded as the draft. Mutually
    /// exclusive with `--draft-text`.
    #[arg(long)]
    draft_file: Option<PathBuf>,

    /// Skip the τ=1.0 correctness check (saves one HSD run).
    #[arg(long)]
    skip_check: bool,

    /// Acceptance threshold for the τ-tolerance test in the perf pass.
    #[arg(long, default_value_t = 0.75)]
    tau: f32,

    /// Exercise `generate_hsd_full` with a single full-page region draft.
    /// This is a minimal real Stage-1 path for regression/profiling; dataset
    /// region benchmarks should use `hsd_omnidocbench`.
    #[arg(long)]
    stage1_full: bool,

    /// Reference window length n (paper §3.2).
    #[arg(long, default_value_t = 3)]
    window_len: usize,

    /// Maximum prefix-tree candidate count per verification step.
    #[arg(long, default_value_t = 32)]
    max_candidates: usize,

    /// Maximum candidate suffix length.
    #[arg(long, default_value_t = 64)]
    max_suffix_len: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    utils::init_tracing();
    let args = Args::parse();

    let device = parse_device(&args.device)?;
    println!("Loading HunyuanOCR from {}", args.model_dir.display());
    let t_load = Instant::now();
    let model = HunyuanOcr::from_dir(&args.model_dir, device)?;
    println!("Model load: {:?}", t_load.elapsed());

    let image = load_image(&args.image)?;
    println!("Loaded image: {}x{}", image.width(), image.height());

    // ── Baseline ────────────────────────────────────────────────────────
    println!("\n[1/3] Baseline generate...");
    let t_base = Instant::now();
    let baseline_results = model.generate_tokens(
        &[image.clone()],
        &[args.instruction.as_str()],
        args.max_tokens,
    );
    let baseline_dur = t_base.elapsed();
    let baseline_tokens = match baseline_results.into_iter().next() {
        Some(Ok(tokens)) => tokens,
        Some(Err(e)) => return Err(format!("baseline generate failed: {e}").into()),
        None => return Err("baseline generate returned no results".into()),
    };
    let baseline = model.decode_tokens(&baseline_tokens)?;
    println!(
        "      Baseline: {:?} | {} chars",
        baseline_dur,
        baseline.len()
    );
    print_preview("BASELINE", &baseline);

    // ── Correctness check ───────────────────────────────────────────────
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
        let (hsd_tokens, _stats) = model.generate_hsd_tokens_with_token_drafts(
            &image,
            &args.instruction,
            &token_drafts,
            &cfg,
        )?;
        let hsd_text = model.decode_tokens(&hsd_tokens)?.trim().to_string();
        let dur = t.elapsed();
        if hsd_tokens == baseline_tokens {
            println!("      ✓ τ=1.0 HSD output matches baseline ({:?})", dur);
        } else {
            eprintln!("      ✗ τ=1.0 HSD output diverges from baseline.");
            print_token_diff(&baseline_tokens, &hsd_tokens);
            print_diff(&baseline, &hsd_text);
            return Err("HSD τ=1.0 mismatch — see diff above".into());
        }
    } else {
        println!("\n[2/3] Correctness check skipped (--skip-check).");
    }

    // ── Performance measurement ─────────────────────────────────────────
    let draft_source = match (&args.draft_text, &args.draft_file) {
        (Some(t), None) => DraftSource::Cli(t.clone()),
        (None, Some(p)) => DraftSource::File(fs::read_to_string(p)?),
        (None, None) => DraftSource::Oracle,
        (Some(_), Some(_)) => {
            return Err("--draft-text and --draft-file are mutually exclusive".into());
        }
    };
    println!(
        "\n[3/3] HSD τ={:.2} performance pass (draft source: {})...",
        args.tau,
        draft_source.label()
    );
    let cfg = make_hsd_cfg(
        args.max_tokens,
        args.tau,
        args.window_len,
        args.max_candidates,
        args.max_suffix_len,
        args.stage1_full,
    );
    let t_hsd = Instant::now();
    let (hsd_text, stats) = match &draft_source {
        DraftSource::Oracle if args.stage1_full => {
            let element = full_page_text_element(&image, &baseline);
            model.generate_hsd_full(
                &image,
                &args.instruction,
                std::slice::from_ref(&element),
                &[],
                &cfg,
            )?
        }
        DraftSource::Cli(draft) | DraftSource::File(draft) if args.stage1_full => {
            let element = full_page_text_element(&image, draft);
            model.generate_hsd_full(
                &image,
                &args.instruction,
                std::slice::from_ref(&element),
                &[],
                &cfg,
            )?
        }
        DraftSource::Oracle => {
            let token_drafts = vec![Draft::new(baseline_tokens.clone())];
            model.generate_hsd_with_token_drafts(&image, &args.instruction, &token_drafts, &cfg)?
        }
        DraftSource::Cli(draft) | DraftSource::File(draft) => {
            model.generate_hsd(&image, &args.instruction, std::slice::from_ref(draft), &cfg)?
        }
    };
    let hsd_dur = t_hsd.elapsed();
    print_preview("HSD OUTPUT", &hsd_text);

    print_hsd_stats(baseline_dur, hsd_dur, &stats, args.stage1_full);
    if matches!(draft_source, DraftSource::Oracle) {
        println!(
            "(Oracle draft = baseline output — AAL is an upper bound; \
             realistic drafters give lower numbers.)"
        );
    }
    Ok(())
}

fn full_page_text_element(image: &image::RgbImage, text: &str) -> LayoutElement {
    LayoutElement::new(
        BoundingBox::from_coords(0.0, 0.0, image.width() as f32, image.height() as f32),
        LayoutElementType::Text,
        1.0,
    )
    .with_text(text.to_string())
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
}

fn print_token_diff(baseline: &[u32], hsd: &[u32]) {
    let common = baseline
        .iter()
        .zip(hsd.iter())
        .take_while(|(a, b)| a == b)
        .count();
    eprintln!(
        "Token lengths: baseline={}, hsd={}, common prefix={}",
        baseline.len(),
        hsd.len(),
        common
    );
    let start = common.saturating_sub(8);
    let baseline_end = (common + 12).min(baseline.len());
    let hsd_end = (common + 12).min(hsd.len());
    eprintln!(
        "baseline tokens[{start}..{baseline_end}]: {:?}",
        &baseline[start..baseline_end]
    );
    eprintln!(
        "hsd tokens[{start}..{hsd_end}]:      {:?}",
        &hsd[start..hsd_end]
    );
}
