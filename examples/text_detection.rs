//! Text Detection Example
//!
//! This example demonstrates how to use the OCR pipeline to detect text regions in images.
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
//! * `-b, --batch` - Enable batch processing mode
//! * `<IMAGES>...` - Paths to input images to process
//!
//! # Example
//!
//! ```bash
//! cargo run --example text_detection -- -m model.onnx -o output/ image1.jpg image2.jpg
//! ```

use clap::Parser;
use oar_ocr::core::{Predictor, init_tracing};
use oar_ocr::predictor::TextDetPredictorBuilder;
use oar_ocr::predictor::db_detector::TextDetResult;
use std::path::Path;
use std::sync::Arc;
use tracing::{error, info};

/// Command-line arguments for the text detection example
#[derive(Parser)]
#[command(name = "text_detection")]
#[command(about = "Text Detection Example - detects text regions in images")]
struct Args {
    /// Path to the text detection model file
    #[arg(short, long)]
    model_path: String,

    /// Paths to input images to process
    #[arg(required = true)]
    images: Vec<String>,

    /// Directory to save visualization results
    #[arg(short, long)]
    output_dir: String,

    /// Enable batch processing mode
    #[arg(short, long)]
    batch: bool,
}

use image::{Rgb, RgbImage};
use imageproc::drawing::{draw_filled_circle_mut, draw_hollow_rect_mut};
use imageproc::rect::Rect;

/// Color for high confidence text detection results (green)
const HIGH_CONFIDENCE_COLOR: Rgb<u8> = Rgb([0, 255, 0]);
/// Color for medium confidence text detection results (orange)
const MEDIUM_CONFIDENCE_COLOR: Rgb<u8> = Rgb([255, 165, 0]);
/// Color for low confidence text detection results (red)
const LOW_CONFIDENCE_COLOR: Rgb<u8> = Rgb([255, 0, 0]);

/// Visualizes text detection results on an image
///
/// This function takes an image and text detection results, then draws bounding boxes
/// around detected text regions with different colors based on confidence scores.
/// High confidence regions are drawn in green, medium in orange, and low in red.
///
/// # Arguments
///
/// * `image` - The original image to draw on
/// * `result` - The text detection results containing polygons and scores
/// * `output_path` - The path where the visualization will be saved
///
/// # Returns
///
/// A Result indicating success or failure of the visualization operation
fn visualize_detection_results(
    image: &RgbImage,
    result: &TextDetResult,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Create a copy of the image for drawing visualization
    let mut vis_image = image.clone();

    // Iterate through detection results for each image in the batch
    for (polys, scores) in result.dt_polys.iter().zip(result.dt_scores.iter()) {
        // Process each detected text region
        for (poly, &score) in polys.iter().zip(scores.iter()) {
            // Determine color based on confidence score
            let color = if score > 0.8 {
                HIGH_CONFIDENCE_COLOR
            } else if score > 0.5 {
                MEDIUM_CONFIDENCE_COLOR
            } else {
                LOW_CONFIDENCE_COLOR
            };

            // Only process polygons with at least 4 points (quadrilaterals)
            if poly.points.len() >= 4 {
                // Calculate bounding box coordinates
                let min_x = poly.points.iter().map(|p| p.x as i32).min().unwrap_or(0);
                let min_y = poly.points.iter().map(|p| p.y as i32).min().unwrap_or(0);
                let max_x = poly.points.iter().map(|p| p.x as i32).max().unwrap_or(0);
                let max_y = poly.points.iter().map(|p| p.y as i32).max().unwrap_or(0);

                // Calculate width and height of the bounding box
                let width = (max_x - min_x).max(1) as u32;
                let height = (max_y - min_y).max(1) as u32;

                // Check if the bounding box is within image boundaries
                if min_x >= 0
                    && min_y >= 0
                    && min_x < vis_image.width() as i32
                    && min_y < vis_image.height() as i32
                {
                    // Draw a thick hollow rectangle around the text region
                    for thickness in 0..2 {
                        let thick_rect = Rect::at(min_x - thickness, min_y - thickness)
                            .of_size(width + 2 * thickness as u32, height + 2 * thickness as u32);
                        draw_hollow_rect_mut(&mut vis_image, thick_rect, color);
                    }

                    // Draw filled circles at each corner of the polygon
                    for point in &poly.points {
                        let x = point.x as i32;
                        let y = point.y as i32;
                        // Check if the point is within image boundaries
                        if x >= 0
                            && y >= 0
                            && x < vis_image.width() as i32
                            && y < vis_image.height() as i32
                        {
                            draw_filled_circle_mut(&mut vis_image, (x, y), 3, color);
                        }
                    }

                    // Draw a small indicator square near the top-left corner
                    let indicator_size = 10;
                    let indicator_x = min_x - 15;
                    let indicator_y = min_y - 15;

                    // Check if the indicator square is within image boundaries
                    if indicator_x >= 0 && indicator_y >= 0 {
                        for dx in 0..indicator_size {
                            for dy in 0..indicator_size {
                                let px = indicator_x + dx;
                                let py = indicator_y + dy;
                                // Check if the pixel is within image boundaries
                                if px >= 0
                                    && py >= 0
                                    && px < vis_image.width() as i32
                                    && py < vis_image.height() as i32
                                {
                                    vis_image.put_pixel(px as u32, py as u32, color);
                                }
                            }
                        }
                    }
                }
            }
        }
    }

    // Save the visualization to the specified output path
    vis_image.save(output_path)?;
    info!("Visualization saved to: {}", output_path);

    Ok(())
}

/// Main function for the text detection example
///
/// This function initializes the OCR pipeline, loads the text detection model,
/// processes input images, and visualizes the results. It supports both
/// single image processing and batch processing modes.
///
/// # Returns
///
/// A Result indicating success or failure of the entire operation
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    init_tracing();

    // Parse command-line arguments
    let args = Args::parse();

    info!("Text Detection Example");

    // Get the model path from arguments
    let model_path = &args.model_path;

    // Verify that the model file exists
    if !Path::new(model_path).exists() {
        error!("Model file not found: {}", model_path);
        return Err("Model file not found".into());
    }

    // Filter out non-existent image files and log errors for missing files
    let existing_images: Vec<String> = args
        .images
        .iter()
        .filter(|path| {
            let exists = Path::new(path).exists();
            if !exists {
                error!("Image file not found: {}", path);
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

    // Create a text detection predictor with specified parameters
    let mut predictor = TextDetPredictorBuilder::new()
        .thresh(0.3) // Binarization threshold
        .box_thresh(0.6) // Box score threshold
        .unclip_ratio(2.0) // Unclip ratio for text boxes
        .limit_side_len(960) // Limit side length for image resizing
        .limit_type(oar_ocr::processors::LimitType::Max) // Limit type for resizing
        .max_side_limit(4000) // Maximum side limit for images
        .model_name("PP-OCRv5_mobile_det") // Model name
        .build(Path::new(model_path))?;

    // Process images in batch mode if requested and multiple images are provided
    if args.batch && existing_images.len() > 1 {
        info!("Batch detection for {} images...", existing_images.len());
        // Convert image paths to Path references for batch processing
        let batch_paths: Vec<&Path> = existing_images.iter().map(Path::new).collect();
        // Perform batch detection
        let batch_results = predictor.predict_batch(&batch_paths)?;

        // Process and log results for each batch
        for (batch_idx, batch_result) in batch_results.iter().enumerate() {
            // Extract detection polygons and scores from the result
            let (batch_dt_polys, _) = match batch_result {
                oar_ocr::core::PredictionResult::Detection {
                    dt_polys,
                    dt_scores,
                    ..
                } => (dt_polys, dt_scores),
                _ => continue,
            };

            // Log the number of text regions detected in each image of the batch
            for (i, polys) in batch_dt_polys.iter().enumerate() {
                info!(
                    "Batch {}, Image {}: {} text regions detected",
                    batch_idx + 1,
                    i + 1,
                    polys.len()
                );
            }
        }
    } else {
        // Process images individually
        for (i, image_path) in existing_images.iter().enumerate() {
            info!(
                "Processing image {} of {}: {}",
                i + 1,
                existing_images.len(),
                image_path
            );

            // Perform text detection on a single image
            let result = predictor.predict_single(Path::new(image_path))?;

            // Extract detection polygons and scores from the result
            let (dt_polys, dt_scores) = match &result {
                oar_ocr::core::PredictionResult::Detection {
                    dt_polys,
                    dt_scores,
                    ..
                } => (dt_polys, dt_scores),
                _ => return Err("Unexpected result type".into()),
            };

            // Log the number of text regions detected
            let region_count = dt_polys[0].len();
            info!("Detected {} text regions", region_count);

            // Load the original image for visualization
            let original_image = image::open(image_path)?.to_rgb8();
            // Create a TextDetResult for visualization
            let text_det_result = TextDetResult {
                input_path: result
                    .input_paths()
                    .iter()
                    .map(|p| Arc::from(p.as_ref()))
                    .collect(),
                index: result.indices().to_vec(),
                input_img: result.input_images().to_vec(),
                dt_polys: dt_polys.clone(),
                dt_scores: dt_scores.clone(),
            };

            // Generate output filename and path for visualization
            let output_filename = format!("detection_result_{}.jpg", i + 1);
            let output_path = Path::new(&args.output_dir).join(&output_filename);

            // Visualize the detection results
            if let Err(e) = visualize_detection_results(
                &original_image,
                &text_det_result,
                output_path.to_str().unwrap(),
            ) {
                error!("Visualization failed for {}: {}", image_path, e);
            }
        }
    }

    // Test with adjusted parameters on the first image to show the effect of parameter tuning
    if !existing_images.is_empty() {
        info!("Testing adjusted parameters on first image...");
        // Adjust parameters for potentially better results
        predictor.set_thresh(0.4); // Increase binarization threshold
        predictor.set_box_thresh(0.7); // Increase box score threshold
        predictor.set_unclip_ratio(1.8); // Decrease unclip ratio

        // Process the first image with adjusted parameters
        let first_image = &existing_images[0];
        let adjusted_result = predictor.predict_single(Path::new(first_image))?;
        // Extract detection polygons and scores from the adjusted result
        let (adjusted_dt_polys, _) = match &adjusted_result {
            oar_ocr::core::PredictionResult::Detection {
                dt_polys,
                dt_scores,
                ..
            } => (dt_polys, dt_scores),
            _ => return Err("Unexpected result type".into()),
        };

        // Log the number of text regions detected with adjusted parameters
        for (i, polys) in adjusted_dt_polys.iter().enumerate() {
            info!(
                "Adjusted parameters - Image {}: {} regions detected",
                i + 1,
                polys.len()
            );
        }
    }

    info!("Example completed!");
    Ok(())
}
