//! Text Line Orientation Classification Example
//!
//! This example demonstrates how to use the OCR pipeline to classify text line orientation.
//! It loads a text line orientation model, processes input text images (typically cropped text regions),
//! and predicts whether the text is upright (0) or upside-down (180) with confidence scores.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example text_line_orientation -- [OPTIONS] <IMAGES>...
//! ```
//!
//! # Arguments
//!
//! * `-m, --model-path` - Path to the text line orientation model file
//! * `-o, --output-dir` - Directory to save output results
//! * `--vis` - Enable visualization output
//! * `--device` - Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
//! * `<IMAGES>...` - Paths to input text line images to process
//!
//! # Example
//!
//! ```bash
//! cargo run --example text_line_orientation -- \
//!     -m models/pplcnet_x1_0_textline_ori.onnx \
//!     -o output/ --vis \
//!     text_line1.jpg text_line2.jpg
//! ```

mod utils;

use clap::Parser;
use oar_ocr::predictors::TextLineOrientationPredictor;
use oar_ocr::utils::load_image;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info, warn};
use utils::device_config::parse_device_config;
use utils::visualization::{ClassificationVisConfig, save_rgb_image, visualize_classification};

/// Command-line arguments for the text line orientation example
#[derive(Parser)]
#[command(name = "text_line_orientation")]
#[command(about = "Text Line Orientation Classification Example - detects text line rotation")]
struct Args {
    /// Path to the text line orientation model file
    #[arg(short, long)]
    model_path: PathBuf,

    /// Paths to input text line images to process
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

    /// Score threshold for classification (default: 0.5)
    #[arg(long, default_value = "0.5")]
    score_thresh: f32,

    /// Number of top predictions to return (default: 2)
    #[arg(long, default_value = "2")]
    topk: usize,

    /// Model input height (default: 80)
    #[arg(long, default_value = "80")]
    input_height: u32,

    /// Model input width (default: 160)
    #[arg(long, default_value = "160")]
    input_width: u32,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    utils::init_tracing();

    // Parse command-line arguments
    let args = Args::parse();

    info!("Text Line Orientation Classification Example");

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
        info!("Classification Configuration:");
        info!("  Score threshold: {}", args.score_thresh);
        info!("  Top-k: {}", args.topk);
        info!(
            "  Input shape: ({}, {})",
            args.input_height, args.input_width
        );
    }

    // Build the text line orientation classifier predictor
    if args.verbose {
        info!("Building text line orientation classifier predictor...");
        info!("  Model: {}", args.model_path.display());
    }

    let predictor = TextLineOrientationPredictor::builder()
        .score_threshold(args.score_thresh)
        .topk(args.topk)
        .input_shape((args.input_height, args.input_width))
        .with_ort_config(ort_config)
        .build(&args.model_path)?;

    info!("Text line orientation classifier predictor built successfully");

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

    // Run text line orientation classification
    info!("Running text line orientation classification...");
    let start = Instant::now();
    let output = predictor.predict(images.clone())?;
    let duration = start.elapsed();

    info!(
        "Classification completed in {:.2}ms",
        duration.as_secs_f64() * 1000.0
    );

    // Display results for each image
    info!("\n=== Classification Results ===");
    for (idx, (image_path, classifications)) in existing_images
        .iter()
        .zip(output.orientations.iter())
        .enumerate()
    {
        info!("\nImage {}: {}", idx + 1, image_path.display());

        if classifications.is_empty() {
            warn!("  No predictions available");
        } else {
            // Show top prediction prominently
            let top = &classifications[0];

            info!("  Detected orientation: {}", top.label);
            info!("  Confidence: {:.2}%", top.score * 100.0);

            // Show all predictions if verbose
            if args.verbose && classifications.len() > 1 {
                info!("  All predictions:");
                for (rank, c) in classifications.iter().enumerate() {
                    info!("    [{}] {} - {:.2}%", rank + 1, c.label, c.score * 100.0);
                }
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

        for (image_path, rgb_img, classifications) in existing_images
            .iter()
            .zip(images.iter())
            .zip(output.orientations.iter())
            .map(|((path, img), classifications)| (path, img, classifications))
        {
            if !classifications.is_empty() {
                // Get top prediction
                let top = &classifications[0];
                let label = top.label.to_string();

                // Use the original filename for output
                let output_filename = image_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown.jpg");
                let output_path = output_dir.join(output_filename);

                let visualized = visualize_classification(
                    rgb_img,
                    &label,
                    top.score,
                    "Text Line Orientation",
                    &vis_config,
                );
                save_rgb_image(&visualized, &output_path)
                    .map_err(|e| format!("Failed to save visualization: {}", e))?;
                info!("  Saved: {}", output_path.display());
            } else {
                warn!(
                    "  Skipping visualization for {} (no predictions)",
                    image_path.display()
                );
            }
        }
    }

    Ok(())
}
