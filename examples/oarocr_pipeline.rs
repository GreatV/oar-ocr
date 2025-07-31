//! Complete OCR pipeline example using the OAROCR library.
//!
//! This example demonstrates how to use the full OCR pipeline to process
//! images and extract text. It includes document orientation classification,
//! text detection, text recognition, and text line classification.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example oarocr_pipeline -- \
//!     --text-detection-model path/to/detection_model.onnx \
//!     --text-recognition-model path/to/recognition_model.onnx \
//!     --char-dict path/to/char_dict.txt \
//!     image1.jpg image2.png
//! ```
//!
//! To use text line orientation classification:
//!
//! ```bash
//! cargo run --example oarocr_pipeline -- \
//!     --text-detection-model path/to/detection_model.onnx \
//!     --text-recognition-model path/to/recognition_model.onnx \
//!     --textline-orientation-model path/to/orientation_model.onnx \
//!     --char-dict path/to/char_dict.txt \
//!     --use-textline-orientation \
//!     image1.jpg
//! ```

use clap::Parser;
use oar_ocr::core::init_tracing;
use oar_ocr::pipeline::OAROCRBuilder;
use std::path::Path;
use tracing::{error, info};

/// Command-line arguments for the OCR pipeline example.
///
/// This struct defines all the command-line arguments that can be passed
/// to the OCR pipeline example. It uses clap for parsing.
#[derive(Parser)]
#[command(name = "oarocr_pipeline")]
#[command(about = "OAROCR Pipeline Example - complete OCR pipeline")]
struct Args {
    /// List of image files to process.
    ///
    /// At least one image file must be provided. The pipeline will process
    /// each image in sequence.
    #[arg(required = true)]
    images: Vec<String>,

    /// Path to the text detection model file.
    ///
    /// This model is used to detect text regions in the images.
    #[arg(long)]
    text_detection_model: String,

    /// Path to the text recognition model file.
    ///
    /// This model is used to recognize text within the detected regions.
    #[arg(long)]
    text_recognition_model: String,

    /// Path to the text line orientation classification model file.
    ///
    /// This model is used to classify the orientation of text lines.
    /// Only required if `use_textline_orientation` is true.
    #[arg(long)]
    textline_orientation_model: String,

    /// Path to the character dictionary file.
    ///
    /// This file contains the characters that the recognition model can identify,
    /// one character per line.
    #[arg(long)]
    char_dict: String,

    /// Whether to use text line orientation classification.
    ///
    /// If true, the pipeline will classify the orientation of text lines
    /// using the specified model.
    #[arg(long)]
    use_textline_orientation: bool,
}

/// Main function for the OCR pipeline example.
///
/// This function initializes the OCR pipeline with the provided models,
/// processes each input image, and prints the results.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    init_tracing();

    // Parse command-line arguments
    let args = Args::parse();

    info!("OAROCR Pipeline Example");

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

    // Create the OCR pipeline builder with the required models and character dictionary
    let mut builder = OAROCRBuilder::new(
        args.text_detection_model,
        args.text_recognition_model,
        args.char_dict,
    )
    // Set batch sizes for detection and recognition to 1
    .text_detection_batch_size(1)
    .text_recognition_batch_size(1)
    // Set minimum score threshold for text recognition results
    .text_rec_score_thresh(0.0)
    // Set input shape for text recognition model (channels, height, width)
    .text_rec_input_shape((3, 48, 320));

    // Configure text line orientation classification if requested
    if args.use_textline_orientation {
        builder = builder
            .textline_orientation_classify_model_path(args.textline_orientation_model)
            .textline_orientation_classify_batch_size(1)
            .use_textline_orientation(true);
    }

    // Build the OCR pipeline and process images
    match builder.build() {
        Ok(mut oarocr) => {
            info!("Pipeline built successfully!");

            // Process each image in sequence
            for (i, image_path) in existing_images.iter().enumerate() {
                info!(
                    "Processing image {} of {}: {}",
                    i + 1,
                    existing_images.len(),
                    image_path
                );

                // Run OCR on the current image
                match oarocr.predict(Path::new(image_path)) {
                    Ok(result) => {
                        info!("OCR completed for {}!", image_path);
                        info!("Detected {} text boxes", result.text_boxes.len());
                        info!("Recognized {} texts", result.rec_texts.len());

                        // Print recognized texts with their confidence scores
                        for (j, (text, score)) in result
                            .rec_texts
                            .iter()
                            .zip(result.rec_scores.iter())
                            .enumerate()
                        {
                            info!("  {}. '{}' (confidence: {:.3})", j + 1, text, score);
                        }

                        // Print document orientation angle if available
                        if let Some(angle) = result.orientation_angle {
                            info!("Document orientation: {:.1}°", angle);
                        }
                    }
                    Err(e) => {
                        error!("OCR failed for {}: {}", image_path, e);
                        continue;
                    }
                }
            }
        }
        Err(e) => {
            error!("Failed to build pipeline: {}", e);
            info!("Ensure model files are available in the specified directory");
            return Err(e.into());
        }
    }

    info!("Example completed!");
    Ok(())
}
