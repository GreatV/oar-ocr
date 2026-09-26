//! jina-ocr-v1 full-page document parsing example (Candle-based).
//!
//! jina-ocr-v1 uses its official OCR instruction and SAM+CLIP image
//! preprocessing internally and returns one Markdown document per input page.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p oar-ocr-vl --example jina_ocr -- \
//!     --model-dir jinaai/jina-ocr-v1 \
//!     --device cpu \
//!     document-1.jpg document-2.png
//! ```

mod utils;

use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info};

use oar_ocr_vl::jina_ocr::DEFAULT_MAX_NEW_TOKENS;
use oar_ocr_vl::utils::image::load_image;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::{JinaOcr, JinaOcrLoadOptions, RuntimeConfig};

#[derive(Parser)]
#[command(name = "jina_ocr")]
#[command(about = "jina-ocr-v1 model-native full-page document-to-Markdown parsing")]
struct Args {
    /// Path to the jina-ocr-v1 model directory
    #[arg(short, long)]
    model_dir: PathBuf,

    /// Paths to one or more page images
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Device to run on: cpu, cuda, cuda:N, or metal
    #[arg(short, long, default_value = "cpu")]
    device: String,

    /// Maximum number of new tokens to generate per page
    #[arg(long, default_value_t = DEFAULT_MAX_NEW_TOKENS)]
    max_tokens: usize,

    /// Opt in to FastMTP speculative decoding (CUDA only; off by default
    /// because graphed plain decoding is faster on typical pages)
    #[arg(long)]
    mtp: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    utils::init_tracing();
    let args = Args::parse();
    let mut had_errors = false;

    if !args.model_dir.exists() {
        error!("Model directory not found: {}", args.model_dir.display());
        return Err("Model directory not found".into());
    }

    let mut loaded_paths = Vec::new();
    let mut images = Vec::new();
    for image_path in &args.images {
        if !image_path.exists() {
            error!("Image file not found: {}", image_path.display());
            had_errors = true;
            continue;
        }
        match load_image(image_path) {
            Ok(image) => {
                loaded_paths.push(image_path.clone());
                images.push(image);
            }
            Err(err) => {
                error!("Failed to load {}: {err}", image_path.display());
                had_errors = true;
            }
        }
    }
    if images.is_empty() {
        return Err("No valid images to process".into());
    }

    let device = parse_device(&args.device)?;
    info!("Device: {device:?}");

    info!("Loading jina-ocr-v1 model...");
    let load_start = Instant::now();
    let model = if args.mtp {
        JinaOcr::from_dir_with_options(
            &args.model_dir,
            RuntimeConfig::new(device),
            JinaOcrLoadOptions::default().with_mtp(true),
        )?
    } else {
        JinaOcr::from_dir(&args.model_dir, device)?
    };
    info!(
        "jina-ocr-v1 loaded in {:.2}ms",
        load_start.elapsed().as_secs_f64() * 1000.0
    );

    let generate_start = Instant::now();
    let results = model.generate(&images, args.max_tokens);
    info!(
        "Generated in {:.2}s",
        generate_start.elapsed().as_secs_f64()
    );

    // BatchResult is Vec<Result<String>>: one entry per image.
    match results {
        Ok(markdowns) => {
            for (image_path, markdown) in loaded_paths.iter().zip(markdowns) {
                match markdown {
                    Ok(markdown) => {
                        println!("=== {} ===\n{markdown}\n", image_path.display());
                    }
                    Err(err) => {
                        error!("Failed to parse {}: {err}", image_path.display());
                        had_errors = true;
                    }
                }
            }
        }
        Err(err) => {
            error!("Failed to parse: {err}");
            return Err("Generation failed".into());
        }
    }
    if had_errors {
        return Err("One or more images failed".into());
    }
    Ok(())
}
