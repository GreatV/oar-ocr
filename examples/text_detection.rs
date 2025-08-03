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
use oar_ocr::core::{
    BatchData, init_tracing,
    traits::{BasePredictor, Sampler},
};
use oar_ocr::predictor::TextDetPredictorBuilder;
use std::path::Path;
use std::sync::Arc;
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

    /// Enable batch processing mode
    #[arg(short, long)]
    batch: bool,
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
        // Create batches using the predictor's batch sampler
        let string_paths: Vec<String> = existing_images.iter().map(|s| s.to_string()).collect();
        let batches = predictor.batch_sampler().sample(string_paths);

        // Process each batch directly using BasePredictor::process
        for batch_data in batches.into_iter() {
            // Use BasePredictor::process to get TextDetResult directly
            let text_det_result = predictor.process(batch_data)?;

            // Display results
            info!("{}", text_det_result);
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

            // Create BatchData for single image processing
            let path_str = image_path.to_string();
            let batch_data =
                BatchData::from_shared_arc_paths(vec![Arc::from(path_str.as_str())], vec![0]);

            // Use BasePredictor::process to get TextDetResult directly
            let text_det_result = predictor.process(batch_data)?;

            // Display results
            info!("{}", text_det_result);

            // Save visualization if feature is enabled
            #[cfg(feature = "visualization")]
            {
                let original_image = image::open(image_path)?.to_rgb8();
                let input_filename = Path::new(image_path)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");
                let output_filename = format!("{}_detection.jpg", input_filename);
                let output_path = Path::new(&args.output_dir).join(&output_filename);

                if let Err(e) = visualize_detection_results(
                    &original_image,
                    &text_det_result,
                    output_path.to_str().unwrap(),
                ) {
                    error!("Visualization failed for {}: {}", image_path, e);
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
        }
    }

    info!("Example completed!");
    Ok(())
}
