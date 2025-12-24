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
//! * `-o, --output-dir` - Directory to save visualization results
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
use tracing::{error, info};
use utils::parse_device_config;

#[cfg(feature = "visualization")]
use image::RgbImage;
#[cfg(feature = "visualization")]
use std::fs;

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

    /// Directory to save visualization results (if visualization feature is enabled)
    #[arg(short, long)]
    output_dir: Option<PathBuf>,

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
            return Err(e);
        }
    };

    info!("Processing {} images", args.images.len());
    info!("Configuration:");
    info!("  Score threshold: {}", args.score_threshold);
    info!("  Box threshold: {}", args.box_threshold);
    info!("  Unclip ratio: {}", args.unclip_ratio);

    // Create output directory if specified
    #[cfg(feature = "visualization")]
    if let Some(output_dir) = args.output_dir.as_ref().filter(|dir| !dir.exists()) {
        fs::create_dir_all(output_dir)?;
        info!("Created output directory: {:?}", output_dir);
    }

    // Process each image
    for image_path in &args.images {
        info!("Processing: {:?}", image_path);

        // Load input image
        let image = match load_image(image_path) {
            Ok(img) => img,
            Err(e) => {
                error!("Failed to load image {:?}: {}", image_path, e);
                continue;
            }
        };

        let (width, height) = (image.width(), image.height());
        info!("  Image dimensions: {}x{}", width, height);

        #[cfg(feature = "visualization")]
        let image_for_vis = image.clone();

        // Run detection
        let start = Instant::now();
        let output = match predictor.predict(vec![image]) {
            Ok(output) => output,
            Err(e) => {
                error!("  Detection failed: {}", e);
                continue;
            }
        };
        let duration = start.elapsed();

        // Display results
        if let Some(detections) = output.detections.first() {
            info!(
                "  Detected {} seal text regions in {:?}",
                detections.len(),
                duration
            );

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
                    "    Box #{}: [{:.0}, {:.0}, {:.0}, {:.0}] confidence {:.2}%",
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
                        "      Polygon with {} points (curved text)",
                        bbox.points.len()
                    );
                }
            }

            // Save visualization if output directory is specified
            #[cfg(feature = "visualization")]
            if let Some(ref output_dir) = args.output_dir {
                // Use the original filename for output
                let output_filename = image_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown.png");
                let output_path = output_dir.join(output_filename);

                // Extract boxes and scores from detections
                let boxes: Vec<_> = detections.iter().map(|d| d.bbox.clone()).collect();
                let scores: Vec<_> = detections.iter().map(|d| d.score).collect();

                // Draw bounding boxes on the image
                let vis_image = visualize_detections(&image_for_vis, &boxes, &scores);

                if let Err(e) = vis_image.save(&output_path) {
                    error!("    Failed to save visualization: {}", e);
                } else {
                    info!("    Saved visualization to: {:?}", output_path);
                }
            }
        } else {
            info!("  No text regions detected");
        }
    }

    Ok(())
}

/// Visualizes detection results by drawing polygons on the image.
///
/// For seal text detection, this function draws the actual polygon shapes
/// that follow curved text paths, rather than simple bounding boxes.
///
/// - Red lines: The polygon outline connecting all detection points
/// - Green dots: Vertex markers shown for complex polygons (>8 points)
/// - White text: Confidence scores displayed at polygon centers
#[cfg(feature = "visualization")]
fn visualize_detections(
    img: &RgbImage,
    boxes: &[oar_ocr::processors::BoundingBox],
    scores: &[f32],
) -> RgbImage {
    use image::Rgb;
    use imageproc::drawing::{draw_filled_circle_mut, draw_line_segment_mut, draw_text_mut};

    let mut output = img.clone();

    // Define colors for visualization
    let seal_color = Rgb([255, 0, 0]); // Red for seal text
    let vertex_color = Rgb([0, 255, 0]); // Green for vertices
    let text_color = Rgb([255, 255, 255]); // White for score text
    let font = load_font();
    let font_scale = 12.0;

    let img_bounds = (img.width() as i32, img.height() as i32);

    for (bbox, &score) in boxes.iter().zip(scores.iter()) {
        // Draw polygon by connecting adjacent points
        if !bbox.points.is_empty() {
            // Draw lines connecting all adjacent points
            for i in 0..bbox.points.len() {
                let p1 = &bbox.points[i];
                let p2 = &bbox.points[(i + 1) % bbox.points.len()]; // Connect last to first

                let x1 = p1.x;
                let y1 = p1.y;
                let x2 = p2.x;
                let y2 = p2.y;

                // Draw line segment
                draw_line_segment_mut(&mut output, (x1, y1), (x2, y2), seal_color);

                // Draw thicker lines for better visibility
                draw_line_segment_mut(&mut output, (x1 + 1.0, y1), (x2 + 1.0, y2), seal_color);
                draw_line_segment_mut(&mut output, (x1, y1 + 1.0), (x2, y2 + 1.0), seal_color);
            }

            // For polygons with many points (curved text), optionally draw vertices
            if bbox.points.len() > 8 {
                // Draw small circles at vertices for curved regions
                for (i, point) in bbox.points.iter().enumerate() {
                    // Draw every Nth vertex to avoid cluttering
                    if i % 3 == 0 {
                        let x = point.x as i32;
                        let y = point.y as i32;
                        if is_point_in_bounds(x, y, img_bounds) {
                            draw_filled_circle_mut(&mut output, (x, y), 2, vertex_color);
                        }
                    }
                }
            }

            // Find a good position for the score text (center of polygon)
            let (center_x, center_y) = bbox
                .points
                .iter()
                .fold((0.0, 0.0), |(sx, sy), p| (sx + p.x, sy + p.y));
            let center_x = (center_x / bbox.points.len() as f32) as i32;
            let center_y = (center_y / bbox.points.len() as f32) as i32;

            // Draw score text at polygon center
            if let Some(ref font) = font {
                let score_text = format!("{:.1}%", score * 100.0);
                if is_point_in_bounds(center_x, center_y - 10, img_bounds) {
                    // Draw text background for better visibility
                    for dx in -1..=1 {
                        for dy in -1..=1 {
                            draw_text_mut(
                                &mut output,
                                Rgb([0, 0, 0]), // Black background
                                center_x + dx,
                                center_y - 10 + dy,
                                font_scale,
                                font,
                                &score_text,
                            );
                        }
                    }
                    draw_text_mut(
                        &mut output,
                        text_color,
                        center_x,
                        center_y - 10,
                        font_scale,
                        font,
                        &score_text,
                    );
                }
            }
        }
    }

    output
}

/// Loads a font for visualization.
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

/// Checks if a point is within image bounds.
#[cfg(feature = "visualization")]
fn is_point_in_bounds(x: i32, y: i32, img_bounds: (i32, i32)) -> bool {
    let (img_width, img_height) = img_bounds;
    x >= 0 && y >= 0 && x < img_width && y < img_height
}
