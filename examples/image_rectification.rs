//! Image Rectification Example
//!
//! This example demonstrates how to use the DocTr image rectification functionality
//! to correct perspective distortions in document images.
//!
//! # Usage
//! ```bash
//! cargo run --example image_rectification -- \
//!   --model-path /path/to/model.onnx \
//!   --output-dir /path/to/output \
//!   [--batch] \
//!   /path/to/image1.jpg [/path/to/image2.jpg ...]
//! ```
//!
//! # Arguments
//! * `--model-path` - Path to the DocTr rectification model file
//! * `--output-dir` - Directory to save rectified images
//! * `--batch` - Process multiple images in batch mode
//! * `images` - Paths to input images to rectify

use clap::Parser;
use oar_ocr::core::{Predictor, init_tracing};
use oar_ocr::predictor::DoctrRectifierPredictorBuilder;
use oar_ocr::predictor::doctr_rectifier::DoctrRectifierResult;
use std::path::Path;
use std::sync::Arc;
use tracing::{error, info};

/// Command-line arguments for the image rectification example
#[derive(Parser)]
#[command(name = "image_rectification")]
#[command(about = "Image Rectification Example - rectifies document images")]
struct Args {
    /// Path to the DocTr rectification model file
    #[arg(short, long)]
    model_path: String,

    /// Paths to input images to rectify
    #[arg(required = true)]
    images: Vec<String>,

    /// Directory to save rectified images
    #[arg(short, long)]
    output_dir: String,

    /// Process multiple images in batch mode
    #[arg(short, long)]
    batch: bool,
}

use image::{Rgb, RgbImage};

/// Creates a comparison image showing the original and rectified images side by side
///
/// # Arguments
/// * `original` - The original image
/// * `rectified` - The rectified image
/// * `output_path` - Path where to save the comparison image
///
/// # Returns
/// * `Ok(())` if the image was successfully created and saved
/// * `Err` if there was an error during image processing or saving
fn create_comparison_image(
    original: &RgbImage,
    rectified: &RgbImage,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Calculate dimensions for the comparison image
    let (orig_width, orig_height) = original.dimensions();
    let (rect_width, rect_height) = rectified.dimensions();

    // Determine the size of the comparison image
    let max_height = orig_height.max(rect_height);
    let total_width = orig_width + rect_width + 20; // 20 pixels spacing between images

    // Create a white background for the comparison image
    let mut comparison = RgbImage::new(total_width, max_height);
    for pixel in comparison.pixels_mut() {
        *pixel = Rgb([255, 255, 255]); // White background
    }

    // Copy the original image to the left side of the comparison
    for y in 0..orig_height {
        for x in 0..orig_width {
            let pixel = original.get_pixel(x, y);
            comparison.put_pixel(x, y, *pixel);
        }
    }

    // Copy the rectified image to the right side of the comparison
    let rect_start_x = orig_width + 20; // Add spacing between images
    for y in 0..rect_height {
        for x in 0..rect_width {
            let pixel = rectified.get_pixel(x, y);
            comparison.put_pixel(rect_start_x + x, y, *pixel);
        }
    }

    // Save the comparison image
    comparison.save(output_path)?;
    info!("Comparison image saved to: {}", output_path);

    Ok(())
}

/// Visualizes the rectification results by saving the rectified image and creating a comparison
///
/// # Arguments
/// * `result` - The rectification result containing input and rectified images
/// * `output_path` - Base path for saving the output images
///
/// # Returns
/// * `Ok(())` if visualization was successful
/// * `Err` if there was an error during visualization
fn visualize_rectification_results(
    result: &DoctrRectifierResult,
    output_path: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    // Check if we have images to visualize
    if result.input_img.is_empty() || result.rectified_img.is_empty() {
        return Err("No images to visualize".into());
    }

    // Save the rectified image
    let rectified_img = &result.rectified_img[0];
    rectified_img.save(output_path)?;
    info!("Rectified image saved to: {}", output_path);

    // Create a comparison image with original and rectified images
    let original_img = &result.input_img[0];
    let comparison_path = output_path.replace(".jpg", "_comparison.jpg");
    create_comparison_image(original_img, rectified_img, &comparison_path)?;

    // Log image dimensions for information
    let (orig_w, orig_h) = original_img.dimensions();
    let (rect_w, rect_h) = rectified_img.dimensions();
    info!("Original dimensions: {}x{}", orig_w, orig_h);
    info!("Rectified dimensions: {}x{}", rect_w, rect_h);

    Ok(())
}

/// Main function for the image rectification example
///
/// This function:
/// 1. Parses command-line arguments
/// 2. Initializes the DocTr rectifier predictor
/// 3. Processes input images (either in batch or individual mode)
/// 4. Saves rectified images and comparison images
///
/// # Returns
/// * `Ok(())` if the example completed successfully
/// * `Err` if there was an error during execution
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    init_tracing();

    // Parse command-line arguments
    let args = Args::parse();

    info!("Image Rectification Example");

    // Validate model path
    let model_path = &args.model_path;
    if !Path::new(model_path).exists() {
        error!("Model file not found: {}", model_path);
        return Err("Model file not found".into());
    }

    // Filter out non-existent image files
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

    // Check if we have any valid images to process
    if existing_images.is_empty() {
        error!("No valid image files found");
        return Err("No valid image files found".into());
    }

    // Initialize the DocTr rectifier predictor
    let mut predictor = DoctrRectifierPredictorBuilder::new()
        .model_name("DocTr_Image_Rectification".to_string())
        .build(Path::new(model_path))?;

    // Process images in batch mode if requested and we have multiple images
    if args.batch && existing_images.len() > 1 {
        info!(
            "Batch rectification for {} images...",
            existing_images.len()
        );

        // Convert image paths to Path objects
        let batch_paths: Vec<_> = existing_images.iter().map(Path::new).collect();

        // Perform batch prediction
        let batch_results = predictor.predict_batch(&batch_paths)?;

        // Log information about batch results
        for (i, batch_result) in batch_results.iter().enumerate() {
            let (_, batch_rectified_img) = match batch_result {
                oar_ocr::core::PredictionResult::Rectification {
                    input_img,
                    rectified_img,
                    ..
                } => (input_img, rectified_img),
                _ => continue,
            };

            info!(
                "Batch {}: {} rectified images",
                i + 1,
                batch_rectified_img.len()
            );
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

            // Perform single image prediction
            let result = predictor.predict_single(Path::new(image_path))?;

            // Extract input and rectified images from the result
            let (input_img, rectified_img) = match &result {
                oar_ocr::core::PredictionResult::Rectification {
                    input_img,
                    rectified_img,
                    ..
                } => (input_img, rectified_img),
                _ => return Err("Unexpected result type".into()),
            };

            info!("Processed {} images", input_img.len());

            // Create a DoctrRectifierResult for visualization
            let doctr_result = DoctrRectifierResult {
                input_path: result
                    .input_paths()
                    .iter()
                    .map(|p| Arc::from(p.as_ref()))
                    .collect(),
                index: result.indices().to_vec(),
                input_img: input_img.clone(),
                rectified_img: rectified_img.clone(),
            };

            // Generate output filename and path
            let output_filename = format!("rectified_result_{}.jpg", i + 1);
            let output_path = Path::new(&args.output_dir).join(&output_filename);

            // Visualize the rectification results
            if let Err(e) =
                visualize_rectification_results(&doctr_result, output_path.to_str().unwrap())
            {
                error!("Visualization failed for {}: {}", image_path, e);
            }
        }
    }

    info!("Example completed!");
    Ok(())
}
