//! Text Line Classification Example
//!
//! This example demonstrates how to use the TextLineClasPredictor to classify
//! the orientation of text lines in images. It automatically handles both single
//! and multiple images efficiently.
//!
//! The example uses the PP-LCNet model for text line classification, which can
//! identify text orientations such as 0° and 180°.

use clap::Parser;
use oar_ocr::core::config::onnx::{OrtExecutionProvider, OrtSessionConfig};
use oar_ocr::core::traits::StandardPredictor;
use oar_ocr::predictor::TextLineClasPredictorBuilder;
use oar_ocr::utils::init_tracing;
use oar_ocr::utils::load_image;
use std::path::Path;
use tracing::{error, info};

/// Command-line arguments for the text line classification example
#[derive(Parser)]
#[command(name = "text_line_classification")]
#[command(about = "Text Line Classification Example - classifies text line orientation")]
struct Args {
    /// Path to the model file
    #[arg(short, long)]
    model_path: String,

    /// Paths to input image files
    #[arg(required = true)]
    images: Vec<String>,

    /// Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
    #[arg(short, long, default_value = "cpu")]
    device: String,
}

/// Display the classification results for text line orientation
///
/// This function prints the classification results for each image, including
/// the image path, detected orientation, and confidence score.
///
/// # Parameters
/// * `image_paths` - Paths to the processed images (as strings)
/// * `class_ids` - Classification IDs for each image
/// * `scores` - Confidence scores for each classification
/// * `label_names` - Label names for each classification
fn display_classification_results(
    image_paths: &[String],
    class_ids: &[Vec<usize>],
    scores: &[Vec<f32>],
    label_names: &[Vec<std::sync::Arc<str>>],
) {
    for (i, (((path, ids), scores_list), labels)) in image_paths
        .iter()
        .zip(class_ids.iter())
        .zip(scores.iter())
        .zip(label_names.iter())
        .enumerate()
    {
        info!("{}. {}", i + 1, path);
        if let (Some(&_class_id), Some(&score), Some(label)) =
            (ids.first(), scores_list.first(), labels.first())
        {
            let orientation = match label.as_ref() {
                "0" => "0°",
                "180" => "180°",
                _ => label,
            };
            info!("   Orientation: {} (confidence: {:.3})", orientation, score);
        }
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

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    init_tracing();

    // Parse command-line arguments
    let args = Args::parse();

    info!("Text Line Classification Example");

    // Get the model path from arguments
    let model_path = &args.model_path;

    // Check if the model file exists
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

    // Check if any valid images remain after filtering
    if existing_images.is_empty() {
        error!("No valid image files found");
        return Err("No valid image files found".into());
    }

    // Parse device configuration
    let execution_providers = parse_device(&args.device)?;
    info!(
        "Using device: {} with providers: {:?}",
        args.device, execution_providers
    );

    // Create ONNX session configuration with device settings
    let ort_config = OrtSessionConfig::new().with_execution_providers(execution_providers);

    // Create a text line classifier predictor with specified parameters
    let predictor = TextLineClasPredictorBuilder::new()
        .topk(2) // Return top 2 predictions
        .batch_size(4) // Process 4 images at a time
        .model_name("PP-LCNet_x0_25_text_line_clas") // Model name
        .input_shape((160, 80)) // Input image dimensions
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

    // Perform classification using the predict API (handles both single and batch automatically)
    match predictor.predict(images, None) {
        Ok(result) => {
            info!("Processing completed for {} images", result.class_ids.len());

            // Display the classification results
            display_classification_results(
                &image_paths,
                &result.class_ids,
                &result.scores,
                &result.label_names,
            );
        }
        Err(e) => {
            error!("Classification failed: {}", e);
            return Err("Classification failed".into());
        }
    }

    // Demonstrate using different predictor parameters
    if !existing_images.is_empty() {
        info!("Testing with topk=3 on first image...");
        // Create another predictor with different parameters
        let adjusted_predictor = TextLineClasPredictorBuilder::new()
            .topk(3) // Return top 3 predictions instead of 2
            .batch_size(2) // Different batch size
            .model_name("PP-LCNet_x0_25_text_line_clas_adjusted") // Different model name
            .input_shape((160, 80)) // Same input dimensions
            .build(Path::new(model_path))?;

        // Use the first image for testing
        let first_image = &existing_images[0];

        // Load the image into memory
        let image = match load_image(Path::new(first_image)) {
            Ok(img) => img,
            Err(e) => {
                error!("Failed to load image for adjusted parameter test: {}", e);
                return Err("Failed to load image".into());
            }
        };

        match adjusted_predictor.predict(vec![image], None) {
            Ok(result) => {
                // Create display data
                let display_paths = vec![first_image.clone()];

                // Display the classification results with adjusted parameters
                display_classification_results(
                    &display_paths,
                    &result.class_ids,
                    &result.scores,
                    &result.label_names,
                );
            }
            Err(e) => error!("Adjusted parameter test failed: {}", e),
        }
    }

    info!("Example completed!");
    Ok(())
}
