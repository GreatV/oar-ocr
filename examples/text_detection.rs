//! Text Detection Example
//!
//! This example demonstrates how to use the text detection predictor to detect text regions in images.
//! It loads a text detection model, processes input images, and visualizes the detected text regions.
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
//! * `-o, --output-dir` - Directory to save visualization results
//! * `-d, --device` - Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
//! * `<IMAGES>...` - Paths to input images to process
//!
//! # Example
//!
//! ```bash
//! cargo run --example text_detection -- -m model.onnx -o output/ -d cpu image1.jpg image2.jpg
//! ```

mod utils;

use clap::Parser;
use oar_ocr::predictors::TextDetectionPredictor;
use oar_ocr::utils::load_image;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info, warn};
use utils::parse_device_config;

#[cfg(feature = "visualization")]
use image::RgbImage;

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

    /// Directory to save visualization results
    #[arg(short, long)]
    output_dir: Option<PathBuf>,

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

    // Save visualization if output directory is provided
    #[cfg(feature = "visualization")]
    if let Some(output_dir) = args.output_dir {
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&output_dir)?;

        info!("\nSaving visualizations to: {}", output_dir.display());

        for (image_path, rgb_img, detections) in existing_images
            .iter()
            .zip(images.iter())
            .zip(result.detections.iter())
            .map(|((path, img), detections)| (path, img, detections))
        {
            if !detections.is_empty() {
                // Extract boxes and scores from detections
                let boxes: Vec<_> = detections.iter().map(|d| d.bbox.clone()).collect();
                let scores: Vec<_> = detections.iter().map(|d| d.score).collect();
                // Use the original filename for output
                let output_filename = image_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown.jpg");
                let output_path = output_dir.join(output_filename);

                let visualized = visualize_detections(rgb_img, &boxes, &scores);
                visualized.save(&output_path)?;
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

/// Visualizes detected text regions by drawing bounding boxes on the image
#[cfg(feature = "visualization")]
fn visualize_detections(
    img: &RgbImage,
    boxes: &[oar_ocr::processors::BoundingBox],
    scores: &[f32],
) -> RgbImage {
    use image::Rgb;
    use imageproc::drawing::{draw_filled_circle_mut, draw_hollow_rect_mut, draw_text_mut};
    use imageproc::rect::Rect;

    let mut output = img.clone();
    let box_color = Rgb([0u8, 255u8, 0u8]); // Green
    let text_color = Rgb([255u8, 0u8, 0u8]); // Red for text labels
    let img_bounds = (output.width() as i32, output.height() as i32);

    // Try to load a font for text rendering
    let font = load_font();

    for (idx, (bbox, score)) in boxes.iter().zip(scores.iter()).enumerate() {
        // Convert polygon to rectangle
        if let Some(rect) = bbox_to_rect(bbox) {
            // Draw thick rectangle (thickness = 2)
            for t in 0..2 {
                let thick_rect = Rect::at(rect.0 - t, rect.1 - t)
                    .of_size(rect.2 + (2 * t) as u32, rect.3 + (2 * t) as u32);

                if is_rect_in_bounds(&thick_rect, img_bounds) {
                    draw_hollow_rect_mut(&mut output, thick_rect, box_color);
                }
            }

            // Draw corner points
            for point in &bbox.points {
                let x = point.x as i32;
                let y = point.y as i32;
                if is_point_in_bounds(x, y, img_bounds) {
                    draw_filled_circle_mut(&mut output, (x, y), 3, box_color);
                }
            }

            // Draw label with index and confidence score
            if let Some(ref font) = font {
                let label = format!("#{} {:.1}%", idx + 1, score * 100.0);
                let label_x = rect.0.max(0);
                let label_y = (rect.1 - 20).max(0);

                if is_point_in_bounds(label_x, label_y, img_bounds) {
                    draw_text_mut(
                        &mut output,
                        text_color,
                        label_x,
                        label_y,
                        20.0,
                        font,
                        &label,
                    );
                }
            }
        }
    }

    output
}

#[cfg(feature = "visualization")]
fn load_font() -> Option<ab_glyph::FontVec> {
    use ab_glyph::FontVec;

    // Try common font paths
    let font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ];

    for path in &font_paths {
        if let Ok(font_data) = std::fs::read(path)
            && let Ok(font) = FontVec::try_from_vec(font_data)
        {
            return Some(font);
        }
    }

    None
}

#[cfg(feature = "visualization")]
fn bbox_to_rect(bbox: &oar_ocr::processors::BoundingBox) -> Option<(i32, i32, u32, u32)> {
    if bbox.points.is_empty() {
        return None;
    }

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

    let x = min_x as i32;
    let y = min_y as i32;
    let width = (max_x - min_x).max(0.0).round() as u32;
    let height = (max_y - min_y).max(0.0).round() as u32;

    (width > 0 && height > 0).then_some((x, y, width, height))
}

#[cfg(feature = "visualization")]
fn is_rect_in_bounds(rect: &imageproc::rect::Rect, img_bounds: (i32, i32)) -> bool {
    let (img_width, img_height) = img_bounds;
    rect.left() >= 0 && rect.top() >= 0 && rect.right() < img_width && rect.bottom() < img_height
}

#[cfg(feature = "visualization")]
fn is_point_in_bounds(x: i32, y: i32, img_bounds: (i32, i32)) -> bool {
    let (img_width, img_height) = img_bounds;
    x >= 0 && y >= 0 && x < img_width && y < img_height
}
