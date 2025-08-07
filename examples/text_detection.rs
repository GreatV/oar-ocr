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
//! * `<IMAGES>...` - Paths to input images to process
//!
//! # Example
//!
//! ```bash
//! cargo run --example text_detection -- -m model.onnx -o output/ image1.jpg image2.jpg
//! ```

use clap::Parser;
use oar_ocr::core::traits::StandardPredictor;
use oar_ocr::predictor::TextDetPredictorBuilder;
use oar_ocr::utils::init_tracing;
use oar_ocr::utils::load_image;
use std::path::Path;
use tracing::{error, info};

// Visualization-specific imports
#[cfg(feature = "visualization")]
use oar_ocr::utils::visualization::visualize_detection_results;

#[cfg(not(feature = "visualization"))]
use tracing::warn;

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
}

/// Main function for the text detection example
///
/// This function initializes the OCR pipeline, loads the text detection model,
/// processes input images, and visualizes the results. It automatically handles
/// both single and multiple images efficiently.
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
    let predictor = TextDetPredictorBuilder::new()
        .thresh(0.3) // Binarization threshold
        .box_thresh(0.6) // Box score threshold
        .unclip_ratio(2.0) // Unclip ratio for text boxes
        .limit_side_len(960) // Limit side length for image resizing
        .limit_type(oar_ocr::processors::LimitType::Max) // Limit type for resizing
        .max_side_limit(4000) // Maximum side limit for images
        .model_name("PP-OCRv5_mobile_det") // Model name
        .build(Path::new(model_path))?;

    // Load all images into memory
    info!("Processing {} images...", existing_images.len());
    let mut images = Vec::new();

    for image_path in &existing_images {
        match load_image(Path::new(image_path)) {
            Ok(img) => {
                images.push(img);
            }
            Err(e) => {
                error!("Failed to load image {}: {}", image_path, e);
                continue;
            }
        }
    }

    if images.is_empty() {
        error!("No images could be loaded for processing");
        return Err("No images could be loaded".into());
    }

    // Perform detection using the predict API (handles both single and batch automatically)
    let text_det_result = match predictor.predict(images, None) {
        Ok(result) => result,
        Err(e) => {
            error!("Detection failed: {}", e);
            return Err("Detection failed".into());
        }
    };

    // Display results
    info!("{}", text_det_result);

    // Save visualization if feature is enabled
    #[cfg(feature = "visualization")]
    {
        // For visualization, we need to process each image individually to create separate output files
        for (i, image_path) in existing_images.iter().enumerate() {
            if i < text_det_result.input_img.len() {
                let original_image = &text_det_result.input_img[i];
                let input_filename = Path::new(image_path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");
                let output_filename = format!("{input_filename}_detection.jpg");
                let output_path = Path::new(&args.output_dir).join(&output_filename);

                // Create a single-image result for visualization
                let single_result = oar_ocr::predictor::db_detector::TextDetResult {
                    input_path: vec![text_det_result.input_path[i].clone()],
                    index: vec![text_det_result.index[i]],
                    input_img: vec![text_det_result.input_img[i].clone()],
                    dt_polys: vec![text_det_result.dt_polys[i].clone()],
                    dt_scores: vec![text_det_result.dt_scores[i].clone()],
                };

                if let Err(e) = visualize_detection_results(
                    original_image,
                    &single_result,
                    output_path.to_str().unwrap(),
                ) {
                    error!("Visualization failed for {}: {}", image_path, e);
                }
            }
        }
    }

    #[cfg(not(feature = "visualization"))]
    {
        if !args.output_dir.is_empty() {
            warn!(
                "Visualization feature is disabled. To enable visualization, compile with --features visualization"
            );
        }
    }

    info!("Example completed!");
    Ok(())
}
