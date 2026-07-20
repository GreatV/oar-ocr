//! PaddleOCR-VL Recognition Example (Candle-based)
//!
//! This example runs PaddleOCR-VL, PaddleOCR-VL-1.5, or PaddleOCR-VL-1.6 for
//! task-specific recognition using the Candle ML framework. All variants
//! support text, formula, table, and chart recognition. Versions 1.5 and 1.6
//! also support text spotting and seal recognition.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p oar-ocr-vl --features download-binaries --example paddleocr_vl -- [OPTIONS] <IMAGES>...
//! ```
//!
//! # Arguments
//!
//! * `-m, --model-dir` - Path to the PaddleOCR-VL model directory
//! * `-t, --task` - Recognition task: ocr, table, formula, chart, spotting, seal (default: ocr)
//! * `-d, --device` - Device to run on: cpu, cuda, cuda:N, or metal (default: cpu)
//! * `--max-tokens` - Maximum number of tokens to generate (default: 512)
//! * `<IMAGES>...` - Paths to input images to process
//!
//! # Examples
//!
//! ```bash
//! # OCR text recognition
//! cargo run -p oar-ocr-vl --features download-binaries --example paddleocr_vl -- \
//!     -m PaddleOCR-VL --task ocr document.jpg
//!
//! # Table recognition
//! cargo run -p oar-ocr-vl --features download-binaries --example paddleocr_vl -- \
//!     -m PaddleOCR-VL --task table table.jpg
//!
//! # Formula recognition
//! cargo run -p oar-ocr-vl --features download-binaries --example paddleocr_vl -- \
//!     -m PaddleOCR-VL --task formula formula.jpg
//!
//! # Chart recognition
//! cargo run -p oar-ocr-vl --features download-binaries --example paddleocr_vl -- \
//!     -m PaddleOCR-VL --task chart chart.jpg
//!
//! # Text spotting with PaddleOCR-VL-1.5 or 1.6
//! cargo run -p oar-ocr-vl --features download-binaries --example paddleocr_vl -- \
//!     -m PaddleOCR-VL-1.5 --task spotting spotting.jpg
//!
//! # Seal recognition with PaddleOCR-VL-1.5 or 1.6
//! cargo run -p oar-ocr-vl --features download-binaries --example paddleocr_vl -- \
//!     -m PaddleOCR-VL-1.6 --task seal seal.jpg
//!
//! # Run on CUDA GPU
//! cargo run -p oar-ocr-vl --features cuda,download-binaries --example paddleocr_vl -- \
//!     -m PaddleOCR-VL -d cuda --task ocr document.jpg
//! ```

mod utils;

use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;

use tracing::{error, info};

use oar_ocr_core::utils::load_image;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::{PaddleOcrVl, PaddleOcrVlTask};
use utils::token_fingerprint;

/// Command-line arguments for the PaddleOCR-VL example
#[derive(Parser)]
#[command(name = "paddleocr_vl")]
#[command(about = "PaddleOCR-VL Recognition Example - task-specific recognition using Candle")]
struct Args {
    /// Path to the PaddleOCR-VL model directory
    #[arg(short, long, default_value = "PaddleOCR-VL")]
    model_dir: PathBuf,

    /// Paths to input images to process
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Recognition task: ocr, table, formula, chart, spotting, seal
    #[arg(short, long, default_value = "ocr")]
    task: String,

    /// Device to run on: cpu, cuda, cuda:N, or metal (default: cpu)
    #[arg(short, long, default_value = "cpu")]
    device: String,

    /// Maximum number of tokens to generate (default: 512)
    #[arg(long, default_value = "512")]
    max_tokens: usize,

    /// Repeat inference to expose Metal warm-up and steady-state latency.
    #[arg(long, default_value_t = 1, value_parser = clap::value_parser!(u32).range(1..))]
    repeat: u32,
}

fn parse_task(task_str: &str) -> Result<PaddleOcrVlTask, Box<dyn std::error::Error>> {
    match task_str.to_lowercase().as_str() {
        "ocr" => Ok(PaddleOcrVlTask::Ocr),
        "table" => Ok(PaddleOcrVlTask::Table),
        "formula" => Ok(PaddleOcrVlTask::Formula),
        "chart" => Ok(PaddleOcrVlTask::Chart),
        "spotting" => Ok(PaddleOcrVlTask::Spotting),
        "seal" => Ok(PaddleOcrVlTask::Seal),
        _ => Err(format!(
            "Unknown task: {}. Use 'ocr', 'table', 'formula', 'chart', 'spotting', or 'seal'",
            task_str
        )
        .into()),
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    utils::init_tracing();

    let args = Args::parse();

    info!("PaddleOCR-VL Recognition Example (Candle-based)");

    // Parse task
    let task = parse_task(&args.task)?;
    info!("Task: {:?}", task);

    // Verify model directory exists
    if !args.model_dir.exists() {
        error!("Model directory not found: {}", args.model_dir.display());
        return Err("Model directory not found".into());
    }

    // Filter valid images
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

    // Load model
    info!(
        "Loading PaddleOCR-VL model from: {}",
        args.model_dir.display()
    );
    let load_start = Instant::now();
    let model = PaddleOcrVl::from_dir(&args.model_dir, device)?;
    let load_duration = load_start.elapsed();
    info!(
        "Model loaded in {:.2}ms",
        load_duration.as_secs_f64() * 1000.0
    );

    // Process each image
    info!("\n=== Processing {} images ===", existing_images.len());

    for image_path in &existing_images {
        info!("\nProcessing: {}", image_path.display());
        let rgb_img = match load_image(image_path) {
            Ok(image) => image,
            Err(e) => {
                error!("  Failed to load image: {}", e);
                continue;
            }
        };

        let mut last_tokens = None;
        for iteration in 1..=args.repeat {
            let infer_start = Instant::now();
            let result = model
                .generate_tokens(std::slice::from_ref(&rgb_img), &[task], args.max_tokens)
                .pop()
                .expect("single-image request returns one result");
            let infer_duration = infer_start.elapsed();
            match result {
                Ok(tokens) => {
                    info!(
                        "  Inference time (run {}/{}): {:.2}ms, tokens: {}, {:.2} tokens/s, fingerprint: {:016x}",
                        iteration,
                        args.repeat,
                        infer_duration.as_secs_f64() * 1000.0,
                        tokens.len(),
                        tokens.len() as f64 / infer_duration.as_secs_f64(),
                        token_fingerprint(&tokens)
                    );
                    last_tokens = Some(tokens);
                }
                Err(e) => {
                    error!("  Inference failed: {}", e);
                    break;
                }
            }
        }
        if let Some(tokens) = last_tokens {
            println!("{}", model.decode_tokens(&tokens, task)?.1);
        }
    }
    Ok(())
}
