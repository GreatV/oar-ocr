//! GLM-OCR Recognition Example (Candle-based)
//!
//! This example demonstrates how to run `zai-org/GLM-OCR` in Rust.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p oar-ocr-vl --example glmocr -- [OPTIONS] <IMAGES>...
//! ```
//!
//! # Examples
//!
//! ```bash
//! cargo run -p oar-ocr-vl --example glmocr -- \
//!     --model-dir /path/to/GLM-OCR \
//!     --prompt "Text Recognition:" \
//!     document.jpg
//! ```

mod utils;

use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info};

use oar_ocr_core::utils::load_image;
use oar_ocr_vl::GlmOcr;
use oar_ocr_vl::utils::parse_device;

#[derive(Parser)]
#[command(name = "glmocr")]
#[command(about = "GLM-OCR Recognition Example - image-to-text using Candle")]
struct Args {
    /// Path to the GLM-OCR model directory
    #[arg(short, long)]
    model_dir: PathBuf,

    /// Paths to input images to process
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Device to run on: cpu, cuda, or cuda:N (default: cpu)
    #[arg(short, long, default_value = "cpu")]
    device: String,

    /// Maximum number of tokens to generate (default: 4096)
    #[arg(long, default_value = "4096")]
    max_tokens: usize,

    /// Instruction prompt (default: Text Recognition)
    #[arg(long, default_value = "Text Recognition:")]
    prompt: String,
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

    info!("Loading GLM-OCR model from: {}", args.model_dir.display());
    let load_start = Instant::now();
    let model = GlmOcr::from_dir(&args.model_dir, device)?;
    info!(
        "Model loaded in {:.2}ms",
        load_start.elapsed().as_secs_f64() * 1000.0
    );

    info!("\n=== Processing {} images ===", existing_images.len());
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
            .generate(&[rgb_img], &[args.prompt.as_str()], args.max_tokens)
            .into_iter()
            .next()
        {
            Some(Ok(result)) => {
                info!(
                    "  Inference time: {:.2}ms",
                    infer_start.elapsed().as_secs_f64() * 1000.0
                );
                println!("{}", result);
            }
            Some(Err(e)) => error!("  Inference failed: {}", e),
            None => error!("  No result returned from model"),
        }
    }

    Ok(())
}
