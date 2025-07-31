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
//! For batch processing, add the `--batch` flag:
//! ```
//! cargo run --example doc_orientation_classification -- --model-path <path_to_model> --batch <image_paths>...
//! ```

use clap::Parser;
use oar_ocr::core::{Predictor, init_tracing};
use oar_ocr::predictor::DocOrientationClassifierBuilder;
use std::path::Path;
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

    /// Enable batch processing
    #[arg(short, long)]
    batch: bool,
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
    label_names: &[Vec<String>],
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
            let orientation = match label.as_str() {
                "0" => "0°",
                "90" => "90°",
                "180" => "180°",
                "270" => "270°",
                _ => label,
            };
            info!("   Orientation: {} (confidence: {:.3})", orientation, score);
        }
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

    // Create a document orientation classifier with specified parameters
    // topk(4) means we want to get the top 4 predictions (all possible orientations)
    // input_shape((224, 224)) specifies the input size expected by the model
    let mut classifier = DocOrientationClassifierBuilder::new()
        .topk(4)
        .input_shape((224, 224))
        .build(Path::new(model_path))?;

    // Process images in batch mode if requested and there are multiple images
    if args.batch && existing_images.len() > 1 {
        info!(
            "Batch classification for {} images...",
            existing_images.len()
        );
        // Convert image paths to a vector of Path references
        let batch_paths: Vec<&Path> = existing_images.iter().map(Path::new).collect();
        // Perform batch classification
        match classifier.predict_batch(&batch_paths) {
            Ok(results) => {
                // Extract the results from the first prediction (batch results are structured differently)
                let (batch_input_path, batch_class_ids, batch_scores, batch_label_names) =
                    match &results[0] {
                        oar_ocr::core::PredictionResult::Classification {
                            input_path,
                            class_ids,
                            scores,
                            label_names,
                            ..
                        } => (input_path, class_ids, scores, label_names),
                        _ => return Err("Unexpected result type".into()),
                    };

                // Convert Cow<str> to String for display
                let batch_input_path_strings: Vec<String> =
                    batch_input_path.iter().map(|cow| cow.to_string()).collect();
                let batch_label_names_strings: Vec<Vec<String>> = batch_label_names
                    .iter()
                    .map(|vec| vec.iter().map(|cow| cow.to_string()).collect())
                    .collect();

                // Display the classification results
                display_classification_results(
                    &batch_input_path_strings,
                    batch_class_ids,
                    batch_scores,
                    &batch_label_names_strings,
                );
            }
            Err(e) => {
                error!("Batch classification failed: {}", e);
                return Err("Batch classification failed".into());
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
            // Classify the orientation of a single image
            match classifier.predict_single(Path::new(image_path)) {
                Ok(result) => {
                    // Extract the results from the prediction
                    let (input_path, class_ids, scores, label_names) = match &result {
                        oar_ocr::core::PredictionResult::Classification {
                            input_path,
                            class_ids,
                            scores,
                            label_names,
                            ..
                        } => (input_path, class_ids, scores, label_names),
                        _ => return Err("Unexpected result type".into()),
                    };

                    // Convert Cow<str> to String for display
                    let input_path_strings: Vec<String> =
                        input_path.iter().map(|cow| cow.to_string()).collect();
                    let label_names_strings: Vec<Vec<String>> = label_names
                        .iter()
                        .map(|vec| vec.iter().map(|cow| cow.to_string()).collect())
                        .collect();

                    // Display the classification results
                    display_classification_results(
                        &input_path_strings,
                        class_ids,
                        scores,
                        &label_names_strings,
                    );
                }
                Err(e) => {
                    error!("Classification failed for {}: {}", image_path, e);
                    continue;
                }
            }
        }
    }

    // Demonstrate how changing parameters affects the results
    // This time we use topk(2) to get only the top 2 predictions
    if !existing_images.is_empty() {
        info!("Testing with topk=2 on first image...");
        let mut adjusted_classifier = DocOrientationClassifierBuilder::new()
            .topk(2)
            .input_shape((224, 224))
            .build(Path::new(model_path))?;

        let first_image = &existing_images[0];
        match adjusted_classifier.predict_single(Path::new(first_image)) {
            Ok(result) => {
                // Extract the results from the prediction
                let (
                    adjusted_input_path,
                    adjusted_class_ids,
                    adjusted_scores,
                    adjusted_label_names,
                ) = match &result {
                    oar_ocr::core::PredictionResult::Classification {
                        input_path,
                        class_ids,
                        scores,
                        label_names,
                        ..
                    } => (input_path, class_ids, scores, label_names),
                    _ => return Err("Unexpected result type".into()),
                };

                // Convert Cow<str> to String for display
                let adjusted_input_path_strings: Vec<String> = adjusted_input_path
                    .iter()
                    .map(|cow| cow.to_string())
                    .collect();
                let adjusted_label_names_strings: Vec<Vec<String>> = adjusted_label_names
                    .iter()
                    .map(|vec| vec.iter().map(|cow| cow.to_string()).collect())
                    .collect();

                // Display the classification results with adjusted parameters
                display_classification_results(
                    &adjusted_input_path_strings,
                    adjusted_class_ids,
                    adjusted_scores,
                    &adjusted_label_names_strings,
                );
            }
            Err(e) => error!("Adjusted parameter test failed: {}", e),
        }
    }

    info!("Example completed!");
    Ok(())
}
