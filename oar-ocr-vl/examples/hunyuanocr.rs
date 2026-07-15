//! HunyuanOCR 1.5 / 1.0 Recognition Example (Candle-based)
//!
//! This example demonstrates how to run `tencent/HunyuanOCR` (1.5 at the model
//! repository root, or archived 1.0 under `v1.0/`) in Rust.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p oar-ocr-vl --features download-binaries --example hunyuanocr -- [OPTIONS] <IMAGES>...
//! ```
//!
//! # Examples
//!
//! ```bash
//! cargo run -p oar-ocr-vl --features download-binaries --example hunyuanocr -- \\
//!     --model-dir models/HunyuanOCR \\
//!     --prompt "Detect and recognize text in the image, and output the text coordinates in a formatted manner." \\
//!     document.jpg
//! ```

mod utils;

use clap::Parser;
use std::path::PathBuf;
use std::time::Duration;
use std::time::Instant;
use tracing::{error, info};

use oar_ocr_core::utils::load_image;
use oar_ocr_vl::HunyuanOcr;
use oar_ocr_vl::utils::parse_device;
use utils::token_fingerprint;

#[derive(Parser)]
#[command(name = "hunyuanocr")]
#[command(about = "HunyuanOCR 1.5/1.0 - image-to-text using Candle")]
struct Args {
    /// Path to the HunyuanOCR model directory
    #[arg(short, long)]
    model_dir: PathBuf,

    /// Optional DFlash draft directory (official checkpoint: <model-dir>/dflash)
    #[arg(long)]
    dflash_dir: Option<PathBuf>,

    /// Paths to input images to process
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Device to run on: cpu, cuda, cuda:N, or metal (default: cpu)
    #[arg(short, long, default_value = "cpu")]
    device: String,

    /// Maximum number of tokens to generate (default: 4096)
    #[arg(long, default_value = "4096")]
    max_tokens: usize,

    /// Override repetition penalty (1.0 matches the official speed benchmark)
    #[arg(long)]
    repetition_penalty: Option<f64>,

    /// Instruction prompt (default: text spotting)
    #[arg(
        long,
        default_value = "Detect and recognize text in the image, and output the text coordinates in a formatted manner."
    )]
    prompt: String,

    /// Suppress generated text and print aggregate timing/token statistics
    #[arg(long)]
    benchmark: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    utils::init_tracing();
    let args = Args::parse();

    if !args.model_dir.exists() {
        error!("Model directory not found: {}", args.model_dir.display());
        return Err("Model directory not found".into());
    }

    let existing_images: Vec<PathBuf> = args
        .images
        .into_iter()
        .filter(|path| {
            if path.exists() {
                true
            } else {
                error!("Image file not found: {}", path.display());
                false
            }
        })
        .collect();
    if existing_images.is_empty() {
        return Err("No valid image files found".into());
    }

    let device = parse_device(&args.device)?;
    info!("Using device: {:?}", device);

    info!(
        "Loading HunyuanOCR model from: {}",
        args.model_dir.display()
    );
    let load_start = Instant::now();
    let mut model = match &args.dflash_dir {
        Some(dflash_dir) => {
            if !dflash_dir.exists() {
                return Err(format!("DFlash directory not found: {}", dflash_dir.display()).into());
            }
            HunyuanOcr::from_dirs(&args.model_dir, dflash_dir, device)?
        }
        None => HunyuanOcr::from_dir(&args.model_dir, device)?,
    };
    if let Some(penalty) = args.repetition_penalty {
        model.set_repetition_penalty(penalty)?;
    }
    info!(
        "HunyuanOCR {} loaded in {:.2}ms{}, repetition penalty {:.3}",
        model.version(),
        load_start.elapsed().as_secs_f64() * 1000.0,
        model
            .dflash_num_speculative_tokens()
            .map(|n| format!(", DFlash enabled ({n} speculative tokens)"))
            .unwrap_or_default(),
        model.repetition_penalty(),
    );

    info!("\n=== Processing {} images ===", existing_images.len());
    let mut total_inference = Duration::ZERO;
    let mut total_tokens = 0usize;
    let mut succeeded = 0usize;
    for image_path in &existing_images {
        info!("\nProcessing: {}", image_path.display());
        let rgb_img = match load_image(image_path) {
            Ok(img) => img,
            Err(e) => {
                error!("  Failed to load image: {}", e);
                continue;
            }
        };

        let infer_start = Instant::now();
        match model
            .generate_tokens(&[rgb_img], &[args.prompt.as_str()], args.max_tokens)
            .pop()
        {
            Some(Ok(tokens)) => {
                let elapsed = infer_start.elapsed();
                total_inference += elapsed;
                total_tokens += tokens.len();
                succeeded += 1;
                info!(
                    "  Inference time: {:.2}ms, tokens: {}, fingerprint: {:016x}",
                    elapsed.as_secs_f64() * 1000.0,
                    tokens.len(),
                    token_fingerprint(&tokens)
                );
                if !args.benchmark {
                    println!("{}", model.decode_tokens(&tokens)?);
                }
            }
            Some(Err(e)) => error!("  Inference failed: {}", e),
            None => error!("  No result returned from model"),
        }
    }

    if succeeded > 0 {
        info!(
            "Benchmark summary: pages={}, total={:.2}ms, avg={:.2}ms/page, tokens={}, throughput={:.2} tokens/s",
            succeeded,
            total_inference.as_secs_f64() * 1000.0,
            total_inference.as_secs_f64() * 1000.0 / succeeded as f64,
            total_tokens,
            total_tokens as f64 / total_inference.as_secs_f64()
        );
    }

    Ok(())
}
