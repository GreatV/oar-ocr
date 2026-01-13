//! Formula Recognition Example
//!
//! This example demonstrates how to use the OCR pipeline to recognize mathematical formulas
//! in images and convert them to LaTeX strings. It supports various formula recognition models
//! from formula recognition models, including UniMERNet and PP-FormulaNet.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example formula_recognition -- [OPTIONS] <IMAGES>...
//! ```
//!
//! # Arguments
//!
//! * `-m, --model-path` - Path to the formula recognition model file (ONNX)
//! * `-t, --tokenizer-path` - Path to the tokenizer file (tokenizer.json)
//! * `-o, --output-dir` - Directory to save output results
//! * `--vis` - Enable visualization output
//! * `--device` - Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
//! * `--model-name` - Model name to explicitly specify the model type (required for correct model detection).
//!   Supported names:
//!   - `UniMERNet` - UniMERNet formula recognition model
//!   - `PP-FormulaNet-S` - PP-FormulaNet Small variant
//!   - `PP-FormulaNet-L` - PP-FormulaNet Large variant
//!   - `PP-FormulaNet_plus-S` - PP-FormulaNet Plus Small variant
//!   - `PP-FormulaNet_plus-M` - PP-FormulaNet Plus Medium variant
//!   - `PP-FormulaNet_plus-L` - PP-FormulaNet Plus Large variant
//! * `--score-thresh` - Score threshold for recognition (default: 0.0)
//! * `--target-width` - Target image width (default: auto)
//! * `--target-height` - Target image height (default: auto)
//! * `--max-length` - Maximum formula length in tokens (default: 1536)
//! * `-v, --verbose` - Enable verbose output
//! * `<IMAGES>...` - Paths to input formula images to process
//!
//! # Examples
//!
//! Basic usage:
//! ```bash
//! cargo run --example formula_recognition -- \
//!     -m models/PP-FormulaNet_plus-M/inference.onnx \
//!     -t models/PP-FormulaNet_plus-M/tokenizer.json \
//!     --model-name "PP-FormulaNet_plus-M" \
//!     formula1.jpg formula2.jpg
//! ```
//!
//! With visualization:
//! ```bash
//! cargo run --release --example formula_recognition -- \
//!     -m models/unimernet.onnx \
//!     -t models/unimernet_tokenizer.json \
//!     --model-name UniMERNet \
//!     -o output/ --vis \
//!     formula1.jpg formula2.jpg
//! ```

mod utils;

use clap::Parser;
use oar_ocr::predictors::FormulaRecognitionPredictor;
use oar_ocr::utils::load_image;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info, warn};
use utils::device_config::parse_device_config;
use utils::visualization::{ClassificationVisConfig, save_rgb_image, visualize_classification};

/// Command-line arguments for the formula recognition example
#[derive(Parser)]
#[command(name = "formula_recognition")]
#[command(about = "Formula Recognition Example - recognizes mathematical formulas in images")]
struct Args {
    /// Path to the formula recognition model file (ONNX)
    #[arg(short, long)]
    model_path: PathBuf,

    /// Path to the tokenizer file (tokenizer.json)
    #[arg(short, long)]
    tokenizer_path: PathBuf,

    /// Paths to input formula images to process
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Directory to save output results
    #[arg(short, long)]
    output_dir: Option<PathBuf>,

    /// Enable visualization output
    #[arg(long)]
    vis: bool,

    /// Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Score threshold for recognition (default: 0.0)
    #[arg(long, default_value = "0.0")]
    score_thresh: f32,

    /// Target image width (default: auto)
    #[arg(long, default_value = "0")]
    target_width: u32,

    /// Target image height (default: auto)
    #[arg(long, default_value = "0")]
    target_height: u32,

    /// Maximum formula length in tokens (default: 1536)
    #[arg(long, default_value = "1536")]
    max_length: usize,

    /// Model name to explicitly specify the model type (required for correct model detection).
    /// Supported: UniMERNet, PP-FormulaNet-S, PP-FormulaNet-L, PP-FormulaNet_plus-S, PP-FormulaNet_plus-M, PP-FormulaNet_plus-L
    #[arg(long, default_value = "FormulaRecognition")]
    model_name: String,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    utils::init_tracing();

    // Parse command-line arguments
    let args = Args::parse();

    info!("Formula Recognition Example");

    // Verify that the model file exists
    if !args.model_path.exists() {
        error!("Model file not found: {}", args.model_path.display());
        return Err("Model file not found".into());
    }

    // Filter out non-existent image files and log errors for missing files
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

    // Exit early if no valid images were provided
    if existing_images.is_empty() {
        error!("No valid image files found");
        return Err("No valid image files found".into());
    }

    // Log device configuration
    info!("Using device: {}", args.device);
    let ort_config = parse_device_config(&args.device)?.unwrap_or_default();

    if ort_config.execution_providers.is_some() {
        info!("CUDA execution provider configured successfully");
    }

    if args.verbose {
        info!("Formula Recognition Configuration:");
        info!("  Score threshold: {}", args.score_thresh);
        info!("  Max formula length: {}", args.max_length);
        if args.target_width > 0 && args.target_height > 0 {
            info!(
                "  Target size override: {}x{}",
                args.target_width, args.target_height
            );
        } else {
            info!("  Target size: auto-detect from model input");
        }
    }

    // Build the formula recognition predictor
    if args.verbose {
        info!("Building formula recognition predictor...");
        info!("  Model: {}", args.model_path.display());
        info!("  Tokenizer: {}", args.tokenizer_path.display());
    }

    let predictor = FormulaRecognitionPredictor::builder()
        .score_threshold(args.score_thresh)
        .model_name(&args.model_name)
        .tokenizer_path(&args.tokenizer_path)
        .with_ort_config(ort_config)
        .build(&args.model_path)?;

    info!("Formula recognition predictor built successfully");

    // Load all images into memory
    info!("Processing {} images...", existing_images.len());
    let mut images = Vec::new();

    for image_path in &existing_images {
        match load_image(image_path) {
            Ok(rgb_img) => {
                if args.verbose {
                    info!(
                        "Loaded image: {} ({}x{})",
                        image_path.display(),
                        rgb_img.width(),
                        rgb_img.height()
                    );
                }
                images.push(rgb_img);
            }
            Err(e) => {
                error!("Failed to load image {}: {}", image_path.display(), e);
                continue;
            }
        }
    }

    if images.is_empty() {
        error!("No images could be loaded for processing");
        return Err("No images could be loaded".into());
    }

    // Run formula recognition
    info!("Running formula recognition...");
    let start = Instant::now();
    let output = predictor.predict(images.clone())?;
    let duration = start.elapsed();

    info!(
        "Recognition completed in {:.2}ms",
        duration.as_secs_f64() * 1000.0
    );

    // Display results for each image
    info!("\n=== Formula Recognition Results ===");
    for (idx, (image_path, formula, score)) in existing_images
        .iter()
        .zip(output.formulas.iter())
        .zip(output.scores.iter())
        .map(|((path, formula), score)| (path, formula, score))
        .enumerate()
    {
        info!("\nImage {}: {}", idx + 1, image_path.display());
        if formula.is_empty() {
            warn!("  No formula recognized (below threshold or invalid)");
        } else {
            info!("  LaTeX: {}", formula);
            if let Some(s) = score {
                info!("  Confidence: {:.2}%", s * 100.0);
            }
        }
    }

    // Save visualization if --vis is enabled
    if args.vis {
        let output_dir = args
            .output_dir
            .as_ref()
            .ok_or("--output-dir is required when --vis is enabled")?;

        // Create output directory if it doesn't exist
        std::fs::create_dir_all(output_dir)?;

        info!("\nSaving visualizations to: {}", output_dir.display());

        let vis_config = ClassificationVisConfig::default();

        for (image_path, rgb_img, formula, score) in existing_images
            .iter()
            .zip(images.iter())
            .zip(output.formulas.iter())
            .zip(output.scores.iter())
            .map(|(((path, img), formula), score)| (path, img, formula, score))
        {
            if !formula.is_empty() {
                // Use the original filename for output
                let output_filename = image_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown.jpg");
                let output_path = output_dir.join(output_filename);

                // Truncate formula for display if too long (use chars to avoid UTF-8 panic)
                let display_formula = if formula.chars().count() > 50 {
                    format!("{}...", formula.chars().take(50).collect::<String>())
                } else {
                    formula.clone()
                };

                let confidence = score.unwrap_or(1.0);
                let visualized = visualize_classification(
                    rgb_img,
                    &display_formula,
                    confidence,
                    "LaTeX",
                    &vis_config,
                );
                save_rgb_image(&visualized, &output_path)
                    .map_err(|e| format!("Failed to save visualization: {}", e))?;
                info!("  Saved: {}", output_path.display());
            } else {
                warn!(
                    "  Skipping visualization for {} (no formula recognized)",
                    image_path.display()
                );
            }
        }
    }

    Ok(())
}
