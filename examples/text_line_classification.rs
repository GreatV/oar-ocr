//! Text Line Classification Example
//!
//! This example demonstrates how to use the TextLineClasPredictor to classify
//! the orientation of text lines in images. It supports both single image processing
//! and batch processing modes.
//!
//! The example uses the PP-LCNet model for text line classification, which can
//! identify text orientations such as 0° and 180°.

use clap::Parser;
use oar_ocr::core::{Predictor, init_tracing};
use oar_ocr::predictor::TextLineClasPredictorBuilder;
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

    /// Enable batch processing mode
    #[arg(short, long)]
    batch: bool,
}

/// Display the classification results for text line orientation
///
/// This function prints the classification results for each image, including
/// the image path, detected orientation, and confidence score.
///
/// # Parameters
/// * `image_paths` - Paths to the processed images
/// * `class_ids` - Classification IDs for each image
/// * `scores` - Confidence scores for each classification
/// * `label_names` - Label names for each classification
fn display_classification_results(
    image_paths: &[std::borrow::Cow<'_, str>],
    class_ids: &[Vec<usize>],
    scores: &[Vec<f32>],
    label_names: &[Vec<std::borrow::Cow<'_, str>>],
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

    // Create a text line classifier predictor with specified parameters
    let mut predictor = TextLineClasPredictorBuilder::new()
        .topk(2) // Return top 2 predictions
        .batch_size(4) // Process 4 images at a time
        .model_name("PP-LCNet_x0_25_text_line_clas") // Model name
        .input_shape((160, 80)) // Input image dimensions
        .build(Path::new(model_path))?;

    // Process images in batch mode if requested and multiple images are provided
    if args.batch && existing_images.len() > 1 {
        info!(
            "Batch classification for {} images...",
            existing_images.len()
        );
        // Convert image paths to Path objects for batch processing
        let batch_paths: Vec<_> = existing_images.iter().map(Path::new).collect();
        match predictor.predict_batch(&batch_paths) {
            Ok(results) => {
                // Process each batch result
                for (batch_idx, result) in results.iter().enumerate() {
                    // Extract classification data from the prediction result
                    let (batch_input_path, batch_class_ids, batch_scores, batch_label_names) =
                        match result {
                            oar_ocr::core::PredictionResult::Classification {
                                input_path,
                                class_ids,
                                scores,
                                label_names,
                                ..
                            } => (input_path, class_ids, scores, label_names),
                            _ => continue,
                        };

                    info!("Batch {}:", batch_idx + 1);
                    // Display the classification results for this batch
                    display_classification_results(
                        batch_input_path,
                        batch_class_ids,
                        batch_scores,
                        batch_label_names,
                    );
                }
            }
            Err(e) => {
                error!("Batch classification failed: {}", e);
                return Err("Batch classification failed".into());
            }
        }
    } else {
        // Process images one by one
        for (i, image_path) in existing_images.iter().enumerate() {
            info!(
                "Processing image {} of {}: {}",
                i + 1,
                existing_images.len(),
                image_path
            );
            // Classify a single image
            match predictor.predict_single(Path::new(image_path)) {
                Ok(result) => {
                    // Extract classification data from the prediction result
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

                    // Display the classification results
                    display_classification_results(input_path, class_ids, scores, label_names);
                }
                Err(e) => {
                    error!("Classification failed for {}: {}", image_path, e);
                    continue;
                }
            }
        }
    }

    // Demonstrate using different predictor parameters
    if !existing_images.is_empty() {
        info!("Testing with topk=3 on first image...");
        // Create another predictor with different parameters
        let mut adjusted_predictor = TextLineClasPredictorBuilder::new()
            .topk(3) // Return top 3 predictions instead of 2
            .batch_size(2) // Different batch size
            .model_name("PP-LCNet_x0_25_text_line_clas_adjusted") // Different model name
            .input_shape((160, 80)) // Same input dimensions
            .build(Path::new(model_path))?;

        // Use the first image for testing
        let first_image = &existing_images[0];
        match adjusted_predictor.predict_single(Path::new(first_image)) {
            Ok(result) => {
                // Extract classification data from the prediction result
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

                // Display the classification results with adjusted parameters
                display_classification_results(
                    adjusted_input_path,
                    adjusted_class_ids,
                    adjusted_scores,
                    adjusted_label_names,
                );
            }
            Err(e) => error!("Adjusted parameter test failed: {}", e),
        }
    }

    info!("Example completed!");
    Ok(())
}
