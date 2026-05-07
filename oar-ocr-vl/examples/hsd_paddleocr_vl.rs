//! PaddleOCR-VL (1.0 / 1.5) whole-image HSD demo.
//!
//! Same structure as `hsd_demo`: τ=1.0 correctness pass + τ-tunable perf pass.
//! The image is treated as a single recognition region (caller is responsible
//! for cropping). Use `--task` to pick OCR / Table / Formula / Chart /
//! Spotting / Seal — Spotting / Seal require PaddleOCR-VL-1.5 weights.
//!
//! ```bash
//! cargo run -p oar-ocr-vl --release --features cuda --example hsd_paddleocr_vl -- \
//!     --model-dir models/PaddleOCR-VL-1.5 \
//!     --device cuda:0 \
//!     --image region.jpg \
//!     --task ocr
//! ```

mod utils;

use clap::{Parser, ValueEnum};
use std::fs;
use std::path::PathBuf;
use std::time::Instant;

use oar_ocr_core::utils::load_image;
use oar_ocr_vl::hsd::types::Draft;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::{PaddleOcrVl, PaddleOcrVlTask};
use utils::{make_hsd_cfg, print_diff, print_hsd_stats, print_preview};

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
#[command(name = "hsd_paddleocr_vl")]
struct Args {
    #[arg(long)]
    model_dir: PathBuf,
    #[arg(long)]
    image: PathBuf,
    #[arg(long, default_value = "cuda:0")]
    device: String,
    #[arg(long, default_value_t = 4096)]
    max_tokens: usize,
    #[arg(long, value_enum, default_value_t = Task::Ocr)]
    task: Task,
    #[arg(long)]
    draft_text: Option<String>,
    #[arg(long)]
    draft_file: Option<PathBuf>,
    #[arg(long)]
    skip_check: bool,
    #[arg(long, default_value_t = 0.75)]
    tau: f32,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    utils::init_tracing();
    let args = Args::parse();
    let device = parse_device(&args.device)?;
    println!("Loading PaddleOCR-VL from {}", args.model_dir.display());
    let model = PaddleOcrVl::from_dir(&args.model_dir, device)?;
    let image = load_image(&args.image)?;
    let task = args.task.to_native();

    println!(
        "\n[1/3] Baseline generate (task={:?}, image {}×{})",
        task,
        image.width(),
        image.height()
    );
    let t = Instant::now();
    let res = model.generate_tokens(&[image.clone()], &[task], args.max_tokens);
    let baseline_dur = t.elapsed();
    let baseline_tokens = match res.into_iter().next() {
        Some(Ok(tokens)) => tokens,
        Some(Err(e)) => return Err(format!("baseline failed: {e}").into()),
        None => return Err("baseline returned no results".into()),
    };
    let (_, baseline_pp) = model.decode_tokens(&baseline_tokens, task)?;
    println!(
        "      baseline: {:?} | postprocessed {} chars",
        baseline_dur,
        baseline_pp.len()
    );

    if !args.skip_check {
        println!("\n[2/3] HSD τ=1.0 correctness check (oracle draft = baseline)...");
        let cfg = make_hsd_cfg(args.max_tokens, 1.0, 3, 32, 64, false);
        let t = Instant::now();
        let token_drafts = vec![Draft::new(baseline_tokens.clone())];
        let (hsd_text, _) =
            model.generate_hsd_with_token_drafts(&image, task, &token_drafts, &cfg)?;
        let dur = t.elapsed();
        if hsd_text == baseline_pp {
            println!("      ✓ matches baseline ({:?})", dur);
        } else {
            eprintln!("      ✗ diverges from baseline.");
            print_diff(&baseline_pp, &hsd_text);
            return Err("τ=1.0 mismatch".into());
        }
    } else {
        println!("\n[2/3] correctness check skipped");
    }

    let draft_source = match (&args.draft_text, &args.draft_file) {
        (Some(s), None) => DraftSource::Text(s.clone()),
        (None, Some(p)) => DraftSource::Text(fs::read_to_string(p)?),
        (None, None) => DraftSource::Oracle,
        (Some(_), Some(_)) => return Err("--draft-text and --draft-file are exclusive".into()),
    };

    println!("\n[3/3] HSD τ={:.2} performance pass...", args.tau);
    let cfg = make_hsd_cfg(args.max_tokens, args.tau, 3, 32, 64, false);
    let t = Instant::now();
    let (hsd_text, stats) = match &draft_source {
        DraftSource::Oracle => {
            let token_drafts = vec![Draft::new(baseline_tokens.clone())];
            model.generate_hsd_with_token_drafts(&image, task, &token_drafts, &cfg)?
        }
        DraftSource::Text(draft) => {
            model.generate_hsd(&image, task, std::slice::from_ref(draft), &cfg)?
        }
    };
    let hsd_dur = t.elapsed();
    print_preview("HSD OUTPUT", &hsd_text);
    print_hsd_stats(baseline_dur, hsd_dur, &stats, false);
    Ok(())
}

enum DraftSource {
    Oracle,
    Text(String),
}
