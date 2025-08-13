//! Document Orientation Classification Example
//!
//! This example demonstrates how to use the OAR-OCR library to classify the orientation
//! of document images. It supports both single image processing and batch processing.
//!
//! The classifier can identify four orientations: 0°, 90°, 180°, and 270°.
//!
//! Usage:
//! ```
//! cargo run --example doc_orientation_classification -- --model-path <path_to_model> <image_paths>...
//! ```
//!
//! The example automatically handles both single and multiple images efficiently.

use clap::Parser;
use oar_ocr::core::config::onnx::{OrtExecutionProvider, OrtSessionConfig};
use oar_ocr::core::format_orientation_label;
use oar_ocr::core::traits::StandardPredictor;
use oar_ocr::predictor::DocOrientationClassifierBuilder;
use oar_ocr::utils::init_tracing;
use oar_ocr::utils::load_image;
use std::path::Path;
use std::sync::Arc;
use tracing::{error, info};

/// Command-line arguments for the document orientation classification example
#[derive(Parser)]
#[command(name = "doc_orientation_classification")]
#[command(about = "Document Orientation Classification Example - classifies document orientation")]
struct Args {
    /// Path to the model file
    #[arg(short, long)]
    model_path: String,

    /// Image file paths to process
    #[arg(required = true)]
    images: Vec<String>,

    /// Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
    #[arg(short, long, default_value = "cpu")]
    device: String,
}

/// Display the classification results for document orientation
///
/// # Arguments
///
/// * `image_paths` - A slice of strings containing the paths to the processed images
/// * `class_ids` - A slice of vectors containing the class IDs for each image
/// * `scores` - A slice of vectors containing the confidence scores for each prediction
/// * `label_names` - A slice of vectors containing the label names for each prediction
fn display_classification_results(
    image_paths: &[String],
    class_ids: &[Vec<usize>],
    scores: &[Vec<f32>],
    label_names: &[Vec<Arc<str>>],
) {
    // Iterate through each image and its corresponding results
    for (i, (((path, ids), scores_list), labels)) in image_paths
        .iter()
        .zip(class_ids.iter())
        .zip(scores.iter())
        .zip(label_names.iter())
        .enumerate()
    {
        info!("{}. {}", i + 1, path);
        // Get the top prediction for each image (first element in the vectors)
        if let (Some(&_class_id), Some(&score), Some(label)) =
            (ids.first(), scores_list.first(), labels.first())
        {
            // Convert numeric labels to degree representations
            let orientation = format_orientation_label(label.as_ref());
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

/// Main function for the document orientation classification example
///
/// This function demonstrates how to use the DocOrientationClassifier to classify
/// the orientation of document images. It supports both single image processing
/// and batch processing.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    init_tracing();

    // Parse command-line arguments
    let args = Args::parse();

    info!("Document Orientation Classification Example");

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

    // Exit if no valid image files were found
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

    // Create a document orientation classifier with specified parameters
    // topk(4) means we want to get the top 4 predictions (all possible orientations)
    // input_shape((224, 224)) specifies the input size expected by the model
    let classifier = DocOrientationClassifierBuilder::new()
        .topk(4)
        .input_shape((224, 224))
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
    match classifier.predict(images, None) {
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

    // Demonstrate how changing parameters affects the results
    // This time we use topk(2) to get only the top 2 predictions
    if !existing_images.is_empty() {
        info!("Testing with topk=2 on first image...");
        let adjusted_classifier = DocOrientationClassifierBuilder::new()
            .topk(2)
            .input_shape((224, 224))
            .build(Path::new(model_path))?;

        let first_image = &existing_images[0];

        // Load the image into memory
        let image = match load_image(Path::new(first_image)) {
            Ok(img) => img,
            Err(e) => {
                error!("Failed to load image for adjusted parameter test: {}", e);
                return Err("Failed to load image".into());
            }
        };

        match adjusted_classifier.predict(vec![image], None) {
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
