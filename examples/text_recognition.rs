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
//! * `--device` - Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
//! * `<IMAGES>...` - Paths to input images to process
//!
//! # Example
//!
//! ```bash
//! cargo run --example text_recognition -- -m model.onnx -d dict.txt --device cuda image1.jpg image2.jpg
//! ```

use clap::Parser;
use oar_ocr::core::config::onnx::{OrtExecutionProvider, OrtSessionConfig};
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

    /// Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
    #[arg(long, default_value = "cpu")]
    device: String,
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

/// Parse device string and create appropriate ONNX execution provider
///
/// # Arguments
///
/// * `device` - Device string (e.g., "cpu", "cuda", "cuda:0")
///
/// # Returns
///
/// Vector of execution providers in order of preference
fn parse_device(device: &str) -> Result<Vec<OrtExecutionProvider>, Box<dyn std::error::Error>> {
    let device = device.to_lowercase();

    if device == "cpu" {
        Ok(vec![OrtExecutionProvider::CPU])
    } else if device == "cuda" {
        #[cfg(feature = "cuda")]
        {
            Ok(vec![
                OrtExecutionProvider::CUDA {
                    device_id: Some(0),
                    gpu_mem_limit: None,
                    arena_extend_strategy: None,
                    cudnn_conv_algo_search: None,
                    do_copy_in_default_stream: None,
                    cudnn_conv_use_max_workspace: None,
                },
                OrtExecutionProvider::CPU,
            ])
        }
        #[cfg(not(feature = "cuda"))]
        {
            error!("CUDA support not compiled in. Falling back to CPU.");
            Ok(vec![OrtExecutionProvider::CPU])
        }
    } else if device.starts_with("cuda:") {
        #[cfg(feature = "cuda")]
        {
            let device_id_str = device.strip_prefix("cuda:").unwrap();
            let device_id: i32 = device_id_str
                .parse()
                .map_err(|_| format!("Invalid CUDA device ID: {}", device_id_str))?;

            Ok(vec![
                OrtExecutionProvider::CUDA {
                    device_id: Some(device_id),
                    gpu_mem_limit: None,
                    arena_extend_strategy: None,
                    cudnn_conv_algo_search: None,
                    do_copy_in_default_stream: None,
                    cudnn_conv_use_max_workspace: None,
                },
                OrtExecutionProvider::CPU,
            ])
        }
        #[cfg(not(feature = "cuda"))]
        {
            error!("CUDA support not compiled in. Falling back to CPU.");
            Ok(vec![OrtExecutionProvider::CPU])
        }
    } else {
        Err(format!(
            "Unsupported device: {}. Supported devices: cpu, cuda, cuda:N",
            device
        )
        .into())
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

    // Parse device configuration
    let execution_providers = parse_device(&args.device)?;
    info!(
        "Using device: {} with providers: {:?}",
        args.device, execution_providers
    );

    // Create ONNX session configuration with device settings
    let ort_config = OrtSessionConfig::new().with_execution_providers(execution_providers);

    // Create a text recognition predictor with specified parameters
    let predictor = TextRecPredictorBuilder::new()
        .model_input_shape([3, 48, 320]) // Model input shape for image resizing
        .batch_size(8) // Process 8 images at a time
        .character_dict(char_dict_lines) // Character dictionary for recognition
        .model_name("PP-OCRv5_mobile_rec".to_string()) // Model name
        .ort_session(ort_config) // Set device configuration
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

    info!("Example completed!");
    Ok(())
}
