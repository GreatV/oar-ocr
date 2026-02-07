//! UniRec Unified Recognition Example (Candle-based)
//!
//! This example demonstrates how to use the UniRec model for unified recognition
//! of text, formulas, tables, and more using the Candle ML framework for native Rust inference.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p oar-ocr-vl --example unirec -- [OPTIONS] <IMAGES>...
//! ```
//!
//! # Arguments
//!
//! * `-m, --model-dir` - Path to the UniRec model directory (containing model.safetensors, config.json, tokenizer.json)
//! * `-d, --device` - Device to run on: cpu, cuda, cuda:N, or metal (default: cpu)
//! * `--max-tokens` - Maximum number of tokens to generate (default: 512)
//! * `-v, --verbose` - Enable verbose output
//! * `<IMAGES>...` - Paths to input images to process
//!
//! # Examples
//!
//! ```bash
//! # Run on CPU
//! cargo run -p oar-ocr-vl --example unirec -- \
//!     -m models/unirec-0.1b \
//!     formula.jpg text.jpg
//!
//! # Run on CUDA GPU
//! cargo run -p oar-ocr-vl --features cuda --example unirec -- \
//!     -m models/unirec-0.1b -d cuda \
//!     formula.jpg text.jpg
//! ```

mod utils;

use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

use tracing::{error, info};

use oar_ocr_core::utils::load_image;
use oar_ocr_vl::UniRec;
use oar_ocr_vl::utils::parse_device;

/// Command-line arguments for the UniRec example
#[derive(Parser)]
#[command(name = "unirec")]
#[command(
    about = "UniRec Unified Recognition Example - recognizes text, formulas, tables using Candle for native Rust inference"
)]
struct Args {
    /// Path to the UniRec model directory (containing model.safetensors, config.json, tokenizer.json)
    #[arg(short, long)]
    model_dir: PathBuf,

    /// Paths to input images to process
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Device to run on: cpu, cuda, cuda:N, or metal (default: cpu)
    #[arg(short, long, default_value = "cpu")]
    device: String,

    /// Maximum number of tokens to generate (default: 512)
    #[arg(long, default_value = "512")]
    max_tokens: usize,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    utils::init_tracing();

    // Parse command-line arguments
    let args = Args::parse();

    info!("UniRec Unified Recognition Example (Candle-based)");

    // Verify that the model directory exists
    if !args.model_dir.exists() {
        error!("Model directory not found: {}", args.model_dir.display());
        return Err("Model directory not found".into());
    }

    // Check for required files
    let model_file = args.model_dir.join("model.safetensors");
    let config_file = args.model_dir.join("config.json");
    let tokenizer_file = args.model_dir.join("tokenizer.json");

    if !model_file.exists() {
        error!("Model file not found: {}", model_file.display());
        return Err("model.safetensors not found in model directory".into());
    }
    if !config_file.exists() {
        error!("Config file not found: {}", config_file.display());
        return Err("config.json not found in model directory".into());
    }
    if !tokenizer_file.exists() {
        error!("Tokenizer file not found: {}", tokenizer_file.display());
        return Err("tokenizer.json not found in model directory".into());
    }

    // Filter out non-existent image files
    let existing_images: Vec<PathBuf> = args
        .images
        .iter()
        .filter(|path| {
            let exists = path.exists();
            if !exists {
                error!("Image file not found: {}", path.display());
            }
            exists
        })
        .cloned()
        .collect();

    if existing_images.is_empty() {
        error!("No valid image files found");
        return Err("No valid image files found".into());
    }

    // Determine device
    let device = parse_device(&args.device)?;
    info!("Using device: {:?}", device);

    // Load the UniRec model
    info!("Loading UniRec model from: {}", args.model_dir.display());
    let load_start = Instant::now();
    let model = UniRec::from_dir(&args.model_dir, device)?;
    let load_duration = load_start.elapsed();
    info!(
        "Model loaded in {:.2}ms",
        load_duration.as_secs_f64() * 1000.0
    );

    if args.verbose {
        let cfg = model.config();
        info!("Model configuration:");
        info!("  d_model: {}", cfg.d_model);
        info!("  vocab_size: {}", cfg.vocab_size);
        info!("  decoder_layers: {}", cfg.decoder_layers);
        info!("  decoder_attention_heads: {}", cfg.decoder_attention_heads);
        info!("  input_size: {}x{}", cfg.input_width, cfg.input_height);
    }

    // Process each image
    info!("\n=== Processing {} images ===", existing_images.len());

    for image_path in &existing_images {
        info!("\nProcessing: {}", image_path.display());

        // Load image
        let rgb_img = match load_image(image_path) {
            Ok(img) => {
                if args.verbose {
                    info!("  Loaded image: {}x{}", img.width(), img.height());
                }
                img
            }
            Err(e) => {
                error!("  Failed to load image: {}", e);
                continue;
            }
        };

        // Run inference
        let infer_start = Instant::now();
        match model.generate(&[rgb_img], args.max_tokens).pop().unwrap() {
            Ok(result) => {
                let infer_duration = infer_start.elapsed();
                info!(
                    "  Inference time: {:.2}ms",
                    infer_duration.as_secs_f64() * 1000.0
                );
                info!("  Result: {}", result);
            }
            Err(e) => {
                error!("  Inference failed: {}", e);
            }
        }
    }
    Ok(())
}
