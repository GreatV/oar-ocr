//! OvisOCR2 full-page document parsing example (Candle-based).
//!
//! OvisOCR2 uses its official prompt and image preprocessing internally and
//! returns one Markdown document per input page.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p oar-ocr-vl --features download-binaries --example ovisocr2 -- \
//!     --model-dir models/OvisOCR2 \
//!     --device cpu \
//!     document-1.jpg document-2.png
//! ```

mod utils;

use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info};

use oar_ocr_core::utils::load_image;
use oar_ocr_vl::OvisOcr2;
use oar_ocr_vl::ovisocr2::DEFAULT_MAX_NEW_TOKENS;
use oar_ocr_vl::utils::parse_device;

#[derive(Parser)]
#[command(name = "ovisocr2")]
#[command(about = "OvisOCR2 model-native full-page document-to-Markdown parsing")]
struct Args {
    /// Path to the OvisOCR2 model directory
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

    /// Retain visual-region <img> tags in the generated Markdown
    #[arg(long)]
    keep_image_tags: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    utils::init_tracing();
    let args = Args::parse();
    let mut had_errors = false;

    if !args.model_dir.exists() {
        error!("Model directory not found: {}", args.model_dir.display());
        return Err("Model directory not found".into());
    }

    let mut image_paths = Vec::new();
    let mut images = Vec::new();
    for image_path in args.images {
        if !image_path.exists() {
            error!("Image file not found: {}", image_path.display());
            had_errors = true;
            continue;
        }
        match load_image(&image_path) {
            Ok(image) => {
                image_paths.push(image_path);
                images.push(image);
            }
            Err(err) => {
                error!("Failed to load {}: {err}", image_path.display());
                had_errors = true;
            }
        }
    }
    if images.is_empty() {
        return Err("No valid image files found".into());
    }

    let device = parse_device(&args.device)?;
    info!("Using device: {:?}", device);

    info!("Loading OvisOCR2 model from: {}", args.model_dir.display());
    let load_start = Instant::now();
    let model = OvisOcr2::from_dir(&args.model_dir, device)?;
    info!(
        "Model loaded in {:.2}ms",
        load_start.elapsed().as_secs_f64() * 1000.0
    );

    let page_count = images.len();
    info!("Processing {page_count} page(s)");
    let infer_start = Instant::now();
    let results = model.parse_with_image_tags(&images, args.max_tokens, args.keep_image_tags);
    let infer_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
    info!(
        "Inference completed for {page_count} page(s) in {infer_ms:.2}ms ({:.2}ms/page)",
        infer_ms / page_count as f64
    );

    for (index, (image_path, result)) in image_paths.iter().zip(results).enumerate() {
        match result {
            Ok(markdown) => {
                if index > 0 {
                    println!();
                }
                println!("<!-- OvisOCR2 source: {} -->", image_path.display());
                println!("{markdown}");
            }
            Err(err) => {
                error!("Inference failed for {}: {err}", image_path.display());
                had_errors = true;
            }
        }
    }

    if had_errors {
        Err("One or more OvisOCR2 pages failed".into())
    } else {
        Ok(())
    }
}
