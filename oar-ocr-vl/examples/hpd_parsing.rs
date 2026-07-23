//! HPD-Parsing native Candle inference.
//!
//! ```bash
//! cargo run --release -p oar-ocr-vl --features cuda,download-binaries \
//!   --example hpd_parsing -- \
//!   --model-dir PaddlePaddle/HPD-Parsing --device cuda:0 document.jpg
//! ```

mod utils;

use clap::Parser;
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::HpdParsing;
use oar_ocr_vl::hpd_parsing::{
    DEFAULT_MAX_NEW_TOKENS, DEFAULT_PROMPT, DEFAULT_SPECULATIVE_TOKENS, HpdGenerationConfig,
};
use oar_ocr_vl::utils::parse_device;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info};

#[derive(Parser)]
#[command(name = "hpd-parsing")]
#[command(about = "PaddlePaddle HPD-Parsing hierarchical document parsing")]
struct Args {
    /// Path to the PaddlePaddle/HPD-Parsing model directory
    #[arg(short, long)]
    model_dir: PathBuf,

    /// Paths to one or more page images
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Device: cpu, cuda, cuda:N, or metal
    #[arg(short, long, default_value = "cpu")]
    device: String,

    /// Maximum generated tokens for each parent or child branch
    #[arg(long, default_value_t = DEFAULT_MAX_NEW_TOKENS)]
    max_tokens: usize,

    /// Disable the P-MTP speculative head and use ordinary greedy decoding
    #[arg(long)]
    no_mtp: bool,

    /// Number of P-MTP future tokens drafted per verification step
    #[arg(long, default_value_t = DEFAULT_SPECULATIVE_TOKENS)]
    speculative_tokens: usize,

    /// Maximum parent/content branches in each continuous decode batch
    #[arg(long, default_value_t = 64)]
    max_active_branches: usize,

    /// Override the official fork-enabled document parsing prompt
    #[arg(long, default_value = DEFAULT_PROMPT)]
    prompt: String,

    /// Log parent/forked-child token counts
    #[arg(long)]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    utils::init_tracing();
    let args = Args::parse();
    if !args.model_dir.exists() {
        return Err(format!("Model directory not found: {}", args.model_dir.display()).into());
    }
    let mut paths = Vec::new();
    let mut images = Vec::new();
    for path in args.images {
        if !path.exists() {
            error!("Image not found: {}", path.display());
            continue;
        }
        match load_image(&path) {
            Ok(image) => {
                paths.push(path);
                images.push(image);
            }
            Err(err) => error!("Failed to load {}: {err}", path.display()),
        }
    }
    if images.is_empty() {
        return Err("No valid input images".into());
    }

    let device = parse_device(&args.device)?;
    info!("Loading {}", args.model_dir.display());
    let started = Instant::now();
    let model = HpdParsing::from_dir(&args.model_dir, device)?;
    info!("Model loaded in {:.2}s", started.elapsed().as_secs_f64());
    let generation = HpdGenerationConfig {
        max_new_tokens: args.max_tokens,
        use_mtp: !args.no_mtp,
        num_speculative_tokens: args.speculative_tokens,
        max_active_branches: args.max_active_branches,
    };

    let started = Instant::now();
    let mut failed = false;
    for (path, image) in paths.iter().zip(&images) {
        match model.generate_one(image, &args.prompt, &generation) {
            Ok(output) => {
                println!("<!-- HPD-Parsing source: {} -->", path.display());
                println!("{}", output.text);
                if args.verbose {
                    info!(
                        "{}: parent_tokens={}, child_tokens={:?}, rounds={}, peak_active={}, forks={}, shared_prefix_tokens={}, mtp={}/{}",
                        path.display(),
                        output.parent_token_count,
                        output.child_token_counts,
                        output.runtime.scheduler_rounds,
                        output.runtime.peak_active_branches,
                        output.runtime.forked_branches,
                        output.runtime.shared_prefix_tokens,
                        output.runtime.mtp_accepted_tokens,
                        output.runtime.mtp_drafted_tokens,
                    );
                }
            }
            Err(err) => {
                error!("Inference failed for {}: {err:?}", path.display());
                failed = true;
            }
        }
    }
    info!(
        "Processed {} image(s) in {:.2}s",
        images.len(),
        started.elapsed().as_secs_f64()
    );
    if failed {
        Err("One or more HPD-Parsing inferences failed".into())
    } else {
        Ok(())
    }
}
