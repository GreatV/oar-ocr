//! Text Detection Example
//!
//! This example demonstrates how to use the text detection predictor to detect text regions in images.
//! It loads a text detection model, processes input images, and optionally visualizes the detected text regions.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example text_detection -- [OPTIONS] <IMAGES>...
//! ```
//!
//! # Arguments
//!
//! * `-m, --model-path` - Path to the text detection model file
//! * `-o, --output-dir` - Directory to save output results (visualizations, etc.)
//! * `--vis` - Enable visualization output
//! * `-d, --device` - Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
//! * `<IMAGES>...` - Paths to input images to process
//!
//! # Example
//!
//! ```bash
//! cargo run --example text_detection -- -m model.onnx -o output/ --vis -d cpu image1.jpg image2.jpg
//! ```

mod utils;

use clap::Parser;
use oar_ocr::predictors::TextDetectionPredictor;
use oar_ocr::utils::load_image;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info, warn};
use utils::device_config::parse_device_config;
use utils::visualization::{Detection, DetectionVisConfig, save_rgb_image, visualize_detections};

/// Command-line arguments for the text detection example
#[derive(Parser)]
#[command(name = "text_detection")]
#[command(about = "Text Detection Example - detects text regions in images")]
struct Args {
    /// Path to the text detection model file
    #[arg(short, long)]
    model_path: PathBuf,

    /// Paths to input images to process
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Directory to save output results (visualizations, etc.)
    #[arg(short, long)]
    output_dir: Option<PathBuf>,

    /// Enable visualization output
    #[arg(long)]
    vis: bool,

    /// Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
    #[arg(short, long, default_value = "cpu")]
    device: String,

    /// Score threshold for detection (default: 0.3)
    #[arg(long, default_value = "0.3")]
    thresh: f32,

    /// Box threshold for filtering (default: 0.6)
    #[arg(long, default_value = "0.6")]
    box_thresh: f32,

    /// Unclip ratio for expanding detected regions (default: 1.5)
    #[arg(long, default_value = "1.5")]
    unclip_ratio: f32,

    /// Maximum candidates to consider (default: 1000)
    #[arg(long, default_value = "1000")]
    max_candidates: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    utils::init_tracing();

    // Parse command-line arguments
    let args = Args::parse();

    info!("Text Detection Example");

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

    // Parse device configuration
    info!("Using device: {}", args.device);
    let ort_config = parse_device_config(&args.device)?.unwrap_or_default();

    if ort_config.execution_providers.is_some() {
        info!("CUDA execution provider configured successfully");
    }

    // Build the text detection predictor
    let predictor = TextDetectionPredictor::builder()
        .score_threshold(args.thresh)
        .box_threshold(args.box_thresh)
        .unclip_ratio(args.unclip_ratio)
        .max_candidates(args.max_candidates)
        .with_ort_config(ort_config)
        .build(&args.model_path)?;

    info!("Detection predictor built successfully");

    // Load all images into memory
    info!("Processing {} images...", existing_images.len());
    let mut images = Vec::new();

    for image_path in &existing_images {
        match load_image(image_path) {
            Ok(rgb_img) => {
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

    // Run detection
    info!("Running text detection...");
    let start = Instant::now();
    let result = predictor.predict(images.clone())?;
    let duration = start.elapsed();

    info!(
        "Detection completed in {:.2}ms",
        duration.as_secs_f64() * 1000.0
    );

    // Display results for each image
    for (idx, (image_path, detections)) in existing_images
        .iter()
        .zip(result.detections.iter())
        .enumerate()
    {
        info!("\n=== Results for image {} ===", idx + 1);
        info!("Image: {}", image_path.display());
        info!("Total text regions detected: {}", detections.len());

        if detections.is_empty() {
            warn!("No text regions found in this image");
        } else {
            // Log bounding box details
            for (i, detection) in detections.iter().enumerate() {
                let bbox = &detection.bbox;
                let score = detection.score;
                // Calculate bounding box rectangle for display
                let (min_x, max_x, min_y, max_y) = bbox.points.iter().fold(
                    (
                        f32::INFINITY,
                        f32::NEG_INFINITY,
                        f32::INFINITY,
                        f32::NEG_INFINITY,
                    ),
                    |(min_x, max_x, min_y, max_y), p| {
                        (
                            min_x.min(p.x),
                            max_x.max(p.x),
                            min_y.min(p.y),
                            max_y.max(p.y),
                        )
                    },
                );

                info!(
                    "  Box #{}: [{:.0}, {:.0}, {:.0}, {:.0}] confidence {:.2}%",
                    i + 1,
                    min_x,
                    min_y,
                    max_x,
                    max_y,
                    score * 100.0
                );
            }
        }
    }

    // Save visualization if --vis is enabled and output directory is provided
    if args.vis {
        let output_dir = args
            .output_dir
            .as_ref()
            .ok_or("--output-dir is required when --vis is enabled")?;

        // Create output directory if it doesn't exist
        std::fs::create_dir_all(output_dir)?;

        info!("\nSaving visualizations to: {}", output_dir.display());

        let vis_config = DetectionVisConfig::default();

        for (image_path, rgb_img, detections) in existing_images
            .iter()
            .zip(images.iter())
            .zip(result.detections.iter())
            .map(|((path, img), detections)| (path, img, detections))
        {
            if !detections.is_empty() {
                // Build detection list for visualization
                let vis_detections: Vec<Detection> = detections
                    .iter()
                    .map(|d| Detection::new(&d.bbox, d.score))
                    .collect();

                // Use the original filename for output
                let output_filename = image_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown.jpg");
                let output_path = output_dir.join(output_filename);

                let visualized = visualize_detections(rgb_img, &vis_detections, &vis_config);
                save_rgb_image(&visualized, &output_path)
                    .map_err(|e| format!("Failed to save visualization: {}", e))?;
                info!("  Saved: {}", output_path.display());
            } else {
                warn!(
                    "  Skipping visualization for {} (no detections)",
                    image_path.display()
                );
            }
        }
    }
    Ok(())
}
