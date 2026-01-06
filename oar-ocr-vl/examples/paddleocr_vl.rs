//! PaddleOCR-VL Recognition Example (Candle-based)
//!
//! This example demonstrates how to use PaddleOCR-VL for task-specific recognition
//! of text, formulas, tables, and charts using the Candle ML framework.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p oar-ocr-vl --example paddleocr_vl -- [OPTIONS] <IMAGES>...
//! ```
//!
//! # Arguments
//!
//! * `-m, --model-dir` - Path to the PaddleOCR-VL model directory
//! * `-t, --task` - Recognition task: ocr, table, formula, chart (default: ocr)
//! * `-d, --device` - Device to run on: cpu, cuda, or cuda:N (default: cpu)
//! * `--max-tokens` - Maximum number of tokens to generate (default: 512)
//! * `<IMAGES>...` - Paths to input images to process
//!
//! # Examples
//!
//! ```bash
//! # OCR (text recognition)
//! cargo run -p oar-ocr-vl --example paddleocr_vl -- \
//!     -m PaddleOCR-VL --task ocr document.jpg
//!
//! # Table recognition
//! cargo run -p oar-ocr-vl --example paddleocr_vl -- \
//!     -m PaddleOCR-VL --task table table.jpg
//!
//! # Formula recognition
//! cargo run -p oar-ocr-vl --example paddleocr_vl -- \
//!     -m PaddleOCR-VL --task formula formula.jpg
//!
//! # Chart recognition
//! cargo run -p oar-ocr-vl --example paddleocr_vl -- \
//!     -m PaddleOCR-VL --task chart chart.jpg
//!
//! # Run on CUDA GPU
//! cargo run -p oar-ocr-vl --features cuda --example paddleocr_vl -- \
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

    /// Recognition task: ocr, table, formula, chart
    #[arg(short, long, default_value = "ocr")]
    task: String,

    /// Device to run on: cpu, cuda, or cuda:N (default: cpu)
    #[arg(short, long, default_value = "cpu")]
    device: String,

    /// Maximum number of tokens to generate (default: 512)
    #[arg(long, default_value = "512")]
    max_tokens: usize,
}

fn parse_task(task_str: &str) -> Result<PaddleOcrVlTask, Box<dyn std::error::Error>> {
    match task_str.to_lowercase().as_str() {
        "ocr" => Ok(PaddleOcrVlTask::Ocr),
        "table" => Ok(PaddleOcrVlTask::Table),
        "formula" => Ok(PaddleOcrVlTask::Formula),
        "chart" => Ok(PaddleOcrVlTask::Chart),
        _ => Err(format!(
            "Unknown task: {}. Use 'ocr', 'table', 'formula', or 'chart'",
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

        // Load image
        let rgb_img = match load_image(image_path) {
            Ok(img) => img,
            Err(e) => {
                error!("  Failed to load image: {}", e);
                continue;
            }
        };

        // Run inference
        let infer_start = Instant::now();
        match model.generate(rgb_img, task, args.max_tokens) {
            Ok(result) => {
                let infer_duration = infer_start.elapsed();
                info!(
                    "  Inference time: {:.2}ms",
                    infer_duration.as_secs_f64() * 1000.0
                );
                println!("{}", result);
            }
            Err(e) => {
                error!("  Inference failed: {}", e);
            }
        }
    }
    Ok(())
}
