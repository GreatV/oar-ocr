//! Example demonstrating seal text detection using PP-OCR models.
//!
//! This example shows how to detect text in seal/stamp images where text
//! often follows curved paths along circular borders.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example seal_text_detection -- [OPTIONS] <IMAGES>...
//! ```
//!
//! # Arguments
//!
//! * `-m, --model-path` - Path to the seal detection model file
//! * `-o, --output-dir` - Directory to save output results
//! * `--vis` - Enable visualization output
//! * `--server-model-path` - Path to the server model for higher accuracy
//! * `--score-threshold` - Pixel-level threshold for text detection
//! * `--box-threshold` - Box-level threshold for filtering detections
//! * `--unclip-ratio` - Expansion ratio for detected regions
//! * `--device` - Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
//! * `<IMAGES>...` - Paths to input images to process

mod utils;

use clap::Parser;
use oar_ocr::predictors::SealTextDetectionPredictor;
use oar_ocr::utils::load_image;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info, warn};
use utils::device_config::parse_device_config;
use utils::visualization::{Detection, DetectionVisConfig, save_rgb_image, visualize_detections};

/// Command-line arguments for seal text detection example.
#[derive(Parser, Debug)]
#[command(name = "seal_text_detection")]
#[command(about = "Seal Text Detection - detects text in seal/stamp images")]
struct Args {
    /// Path to the seal detection model file
    #[arg(short, long)]
    model_path: PathBuf,

    /// Paths to input images to process
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Directory to save output results
    #[arg(short, long)]
    output_dir: Option<PathBuf>,

    /// Enable visualization output
    #[arg(long)]
    vis: bool,

    /// Path to the server model for higher accuracy (overrides model_path if provided)
    #[arg(long)]
    server_model_path: Option<PathBuf>,

    /// Pixel-level threshold for text detection
    #[arg(long, default_value = "0.2")]
    score_threshold: f32,

    /// Box-level threshold for filtering detections
    #[arg(long, default_value = "0.6")]
    box_threshold: f32,

    /// Expansion ratio for detected regions
    #[arg(long, default_value = "0.5")]
    unclip_ratio: f32,

    /// Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
    #[arg(long, default_value = "cpu")]
    device: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    utils::init_tracing();

    let args = Args::parse();

    // Determine which model to use
    let model_path = if let Some(server_path) = args.server_model_path {
        server_path
    } else {
        args.model_path.clone()
    };

    info!("Loading seal text detection model from: {:?}", model_path);

    // Check if model exists
    if !model_path.exists() {
        error!("Model file not found: {:?}", model_path);
        error!("Please download the model using the provided scripts");
        std::process::exit(1);
    }

    // Log device configuration
    info!("Using device: {}", args.device);
    let ort_config = parse_device_config(&args.device)?.unwrap_or_default();

    if ort_config.execution_providers.is_some() {
        info!("CUDA execution provider configured successfully");
    }

    // Build predictor
    let predictor = match SealTextDetectionPredictor::builder()
        .score_threshold(args.score_threshold)
        .with_ort_config(ort_config)
        .build(&model_path)
    {
        Ok(predictor) => predictor,
        Err(e) => {
            error!("Failed to build seal detection predictor: {}", e);
            return Err(e.into());
        }
    };

    info!("Processing {} images", args.images.len());
    info!("Configuration:");
    info!("  Score threshold: {}", args.score_threshold);
    info!("  Box threshold: {}", args.box_threshold);
    info!("  Unclip ratio: {}", args.unclip_ratio);

    // Load all images into memory
    let mut images = Vec::new();
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

    for image_path in &existing_images {
        match load_image(image_path) {
            Ok(img) => {
                info!(
                    "Loaded image: {} ({}x{})",
                    image_path.display(),
                    img.width(),
                    img.height()
                );
                images.push(img);
            }
            Err(e) => {
                error!("Failed to load image {:?}: {}", image_path, e);
                continue;
            }
        }
    }

    if images.is_empty() {
        error!("No images could be loaded for processing");
        return Err("No images could be loaded".into());
    }

    // Run detection
    info!("Running seal text detection...");
    let start = Instant::now();
    let output = match predictor.predict(images.clone()) {
        Ok(output) => output,
        Err(e) => {
            error!("Detection failed: {}", e);
            return Err(e.into());
        }
    };
    let duration = start.elapsed();

    info!(
        "Detection completed in {:.2}ms",
        duration.as_secs_f64() * 1000.0
    );

    // Display results for each image
    for (idx, (image_path, detections)) in existing_images
        .iter()
        .zip(output.detections.iter())
        .enumerate()
    {
        info!("\n=== Results for image {} ===", idx + 1);
        info!("Image: {}", image_path.display());
        info!("Detected {} seal text regions", detections.len());

        if detections.is_empty() {
            warn!("No text regions found in this image");
        } else {
            for (i, detection) in detections.iter().enumerate() {
                let bbox = &detection.bbox;
                let score = detection.score;
                // Calculate bounding box statistics
                let (min_x, max_x, min_y, max_y) = bbox.points.iter().fold(
                    (f32::MAX, f32::MIN, f32::MAX, f32::MIN),
                    |(min_x, max_x, min_y, max_y), point| {
                        (
                            min_x.min(point.x),
                            max_x.max(point.x),
                            min_y.min(point.y),
                            max_y.max(point.y),
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

                // Display polygon points for curved text regions
                if bbox.points.len() > 4 {
                    info!(
                        "    Polygon with {} points (curved text)",
                        bbox.points.len()
                    );
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

        // Use polygon mode for seal text detection (curved text)
        let vis_config = DetectionVisConfig::default().with_polygon(true);

        for (image_path, rgb_img, detections) in existing_images
            .iter()
            .zip(images.iter())
            .zip(output.detections.iter())
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
                    .unwrap_or("unknown.png");
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
