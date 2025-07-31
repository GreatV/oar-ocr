//! Text Recognition Example
//!
//! This example demonstrates how to use the OCR pipeline to recognize text in images.
//! It loads a text recognition model, processes input images, and displays the recognized text
//! along with confidence scores.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example text_recognition -- [OPTIONS] <IMAGES>...
//! ```
//!
//! # Arguments
//!
//! * `-m, --model-path` - Path to the text recognition model file
//! * `-d, --char-dict-path` - Path to the character dictionary file
//! * `-b, --batch` - Enable batch processing mode
//! * `<IMAGES>...` - Paths to input images to process
//!
//! # Example
//!
//! ```bash
//! cargo run --example text_recognition -- -m model.onnx -d dict.txt image1.jpg image2.jpg
//! ```

use clap::Parser;
use oar_ocr::core::{Predictor, init_tracing};
use oar_ocr::predictor::TextRecPredictorBuilder;
use std::path::Path;
use tracing::{error, info};

/// Command-line arguments for the text recognition example
#[derive(Parser)]
#[command(name = "text_recognition")]
#[command(about = "Text Recognition Example - recognizes text from images")]
struct Args {
    /// Path to the text recognition model file
    #[arg(short, long)]
    model_path: String,

    /// Path to the character dictionary file
    #[arg(short = 'd', long)]
    char_dict_path: String,

    /// Paths to input images to process
    #[arg(required = true)]
    images: Vec<String>,

    /// Enable batch processing mode
    #[arg(short, long)]
    batch: bool,
}

/// Display the recognition results for text in images
///
/// This function prints the recognition results for each image, including
/// the image path, recognized text, and confidence score.
///
/// # Parameters
/// * `image_paths` - Paths to the processed images
/// * `texts` - Recognized texts for each image
/// * `scores` - Confidence scores for each recognition
fn display_recognition_results(
    image_paths: &[std::borrow::Cow<'_, str>],
    texts: &[std::borrow::Cow<'_, str>],
    scores: &[f32],
) {
    for (i, ((path, text), &score)) in image_paths
        .iter()
        .zip(texts.iter())
        .zip(scores.iter())
        .enumerate()
    {
        info!("{}. {}: '{}' (confidence: {:.3})", i + 1, path, text, score);
    }
}

/// Main function for the text recognition example
///
/// This function initializes the OCR pipeline, loads the text recognition model,
/// processes input images, and displays the recognized text. It supports both
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

    info!("Text Recognition Example");

    // Get the model path and character dictionary path from arguments
    let model_path = &args.model_path;
    let char_dict_path = &args.char_dict_path;

    // Verify that the model file exists
    if !Path::new(model_path).exists() {
        error!("Model file not found: {}", model_path);
        return Err("Model file not found".into());
    }

    // Verify that the character dictionary file exists
    if !Path::new(char_dict_path).exists() {
        error!("Character dictionary file not found: {}", char_dict_path);
        return Err("Character dictionary file not found".into());
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

    // Read the character dictionary file
    let char_dict_lines = std::fs::read_to_string(char_dict_path)?
        .lines()
        .map(|l| l.to_string())
        .collect();

    // Create a text recognition predictor with specified parameters
    let mut predictor = TextRecPredictorBuilder::new()
        .rec_image_shape([3, 48, 320]) // Input image shape for the model
        .batch_size(8) // Process 8 images at a time
        .character_dict(char_dict_lines) // Character dictionary for recognition
        .model_name("PP-OCRv5_mobile_rec".to_string()) // Model name
        .build(Path::new(model_path))?;

    // Process images in batch mode if requested and multiple images are provided
    if args.batch && existing_images.len() > 1 {
        info!("Batch recognition for {} images...", existing_images.len());
        // Convert image paths to Path references for batch processing
        let batch_paths: Vec<&Path> = existing_images.iter().map(Path::new).collect();
        // Perform batch recognition
        match predictor.predict_batch(&batch_paths) {
            Ok(results) => {
                // Process and display results for each batch
                for (batch_idx, result) in results.iter().enumerate() {
                    // Extract recognition text and scores from the result
                    let (batch_input_path, batch_rec_text, batch_rec_score) = match result {
                        oar_ocr::core::PredictionResult::Recognition {
                            input_path,
                            rec_text,
                            rec_score,
                            ..
                        } => (input_path, rec_text, rec_score),
                        _ => continue,
                    };

                    info!("Batch {}:", batch_idx + 1);
                    // Display the recognition results for this batch
                    display_recognition_results(batch_input_path, batch_rec_text, batch_rec_score);
                }
            }
            Err(e) => {
                error!("Batch recognition failed: {}", e);
                return Err("Batch recognition failed".into());
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
            // Perform text recognition on a single image
            match predictor.predict_single(Path::new(image_path)) {
                Ok(result) => {
                    // Extract recognition text and scores from the result
                    let (input_path, rec_text, rec_score) = match &result {
                        oar_ocr::core::PredictionResult::Recognition {
                            input_path,
                            rec_text,
                            rec_score,
                            ..
                        } => (input_path, rec_text, rec_score),
                        _ => return Err("Unexpected result type".into()),
                    };

                    // Display the recognition results
                    display_recognition_results(input_path, rec_text, rec_score);
                }
                Err(e) => {
                    error!("Recognition failed for {}: {}", image_path, e);
                    continue;
                }
            }
        }
    }

    // Demonstrate changing the image shape after processing
    predictor.set_rec_image_shape([3, 32, 256]);
    info!("Updated image shape to [3, 32, 256]");

    info!("Example completed!");
    Ok(())
}
