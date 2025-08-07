//! Text Recognition Example
//!
//! This example demonstrates how to use the OCR pipeline to recognize text in images.
//! It loads a text recognition model, processes input images, and displays the recognized text
//! along with confidence scores. The example automatically handles both single and multiple
//! images efficiently.
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
//! * `<IMAGES>...` - Paths to input images to process
//!
//! # Example
//!
//! ```bash
//! cargo run --example text_recognition -- -m model.onnx -d dict.txt image1.jpg image2.jpg
//! ```

use clap::Parser;
use oar_ocr::core::traits::StandardPredictor;
use oar_ocr::predictor::TextRecPredictorBuilder;
use oar_ocr::utils::init_tracing;
use oar_ocr::utils::load_image;
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
}

/// Display the recognition results for text in images
///
/// This function prints the recognition results for each image, including
/// the image path, recognized text, and confidence score.
///
/// # Parameters
/// * `image_paths` - Paths to the processed images (as strings)
/// * `texts` - Recognized texts for each image
/// * `scores` - Confidence scores for each recognition
fn display_recognition_results(
    image_paths: &[String],
    texts: &[std::sync::Arc<str>],
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
        .model_input_shape([3, 48, 320]) // Model input shape for image resizing
        .batch_size(8) // Process 8 images at a time
        .character_dict(char_dict_lines) // Character dictionary for recognition
        .model_name("PP-OCRv5_mobile_rec".to_string()) // Model name
        .build(Path::new(model_path))?;

    // Load all images into memory
    info!("Processing {} images...", existing_images.len());
    let mut images = Vec::new();
    let mut image_paths = Vec::new();

    for image_path in &existing_images {
        match load_image(Path::new(image_path)) {
            Ok(img) => {
                images.push(img);
                image_paths.push(image_path.clone());
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

    // Perform recognition using the predict API (handles both single and batch automatically)
    match predictor.predict(images, None) {
        Ok(result) => {
            info!("Processing completed for {} images", result.rec_text.len());

            // Display the recognition results
            display_recognition_results(&image_paths, &result.rec_text, &result.rec_score);
        }
        Err(e) => {
            error!("Recognition failed: {}", e);
            return Err("Recognition failed".into());
        }
    }

    // Demonstrate changing the model input shape after processing
    predictor.set_model_input_shape([3, 32, 256]);
    info!("Updated model input shape to [3, 32, 256]");

    info!("Example completed!");
    Ok(())
}
