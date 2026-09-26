//! WeVisDoc full-page document parsing example (Candle-based).
//!
//! WeVisDoc uses its official prompt (WeDocKit system prompt + "Convert this
//! document image to Markdown.") and Qwen2-VL image preprocessing internally
//! and returns one Markdown document per input page.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p oar-ocr-vl --example wevisdoc -- \
//!     --model-dir Tencent/WeVisDoc-2B \
//!     --device cpu \
//!     document-1.jpg document-2.png
//! ```

mod utils;

use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info};

use oar_ocr_vl::WeVisDoc;
use oar_ocr_vl::utils::image::load_image;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::wevisdoc::DEFAULT_MAX_NEW_TOKENS;

#[derive(Parser)]
#[command(name = "wevisdoc")]
#[command(about = "WeVisDoc model-native full-page document-to-Markdown parsing")]
struct Args {
    /// Path to the WeVisDoc model directory
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

    info!("Loading WeVisDoc model...");
    let load_start = Instant::now();
    let model = WeVisDoc::from_dir(&args.model_dir, device)?;
    info!(
        "WeVisDoc loaded in {:.2}ms",
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
