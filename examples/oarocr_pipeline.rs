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
//!     --output-dir ./visualizations \
//!     --font-path path/to/font.ttf \
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
//!     --output-dir ./visualizations \
//!     --font-path path/to/font.ttf \
//!     image1.jpg
//! ```

use clap::Parser;
use oar_ocr::pipeline::OAROCRBuilder;
use oar_ocr::utils::init_tracing;
#[cfg(feature = "visualization")]
use oar_ocr::utils::visualization::visualize_ocr_results;
use std::path::Path;
use tracing::{error, info, warn};

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

    /// Output directory for visualization images.
    ///
    /// If specified, the pipeline will generate visualization images showing
    /// the original image on the left and detected text with bounding boxes on the right.
    /// If not specified, no visualization will be performed.
    #[arg(long)]
    output_dir: Option<String>,

    /// Path to the font file for text rendering in visualizations.
    ///
    /// If not specified, a default font will be used for rendering text
    /// in the visualization images.
    #[arg(long)]
    font_path: Option<String>,
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
    // Set model input shape for text recognition (channels, height, width)
    .text_rec_model_input_shape((3, 48, 320));

    // Configure text line orientation classification if requested
    if args.use_textline_orientation {
        builder = builder
            .textline_orientation_classify_model_path(args.textline_orientation_model)
            .textline_orientation_classify_batch_size(1)
            .textline_orientation_classify_input_shape((160, 80))
            .use_textline_orientation(true);
    }

    // Enable parallel processing for better performance with multiple images
    builder = builder
        .enable_parallel_processing(true)
        .max_parallel_threads(None); // Use all available CPU cores

    // Build the OCR pipeline and process images
    match builder.build() {
        Ok(oarocr) => {
            info!("Pipeline built successfully!");

            // Process all images at once using the new parallel processing capabilities
            info!(
                "Processing {} images using parallel processing...",
                existing_images.len()
            );
            let image_paths: Vec<&Path> = existing_images.iter().map(Path::new).collect();

            let start_time = std::time::Instant::now();
            match oarocr.predict(&image_paths) {
                Ok(results) => {
                    let processing_time = start_time.elapsed();
                    info!(
                        "Successfully processed {} images in {:?} ({:.2} images/sec)",
                        results.len(),
                        processing_time,
                        results.len() as f64 / processing_time.as_secs_f64()
                    );

                    // Process each result
                    for (i, result) in results.iter().enumerate() {
                        info!(
                            "Results for image {} of {}: {}",
                            i + 1,
                            results.len(),
                            existing_images[i]
                        );

                        // Display OCR results using the Display trait
                        info!("{}", result);

                        // Generate visualization if output directory is specified
                        #[cfg(feature = "visualization")]
                        if let Some(ref output_dir) = args.output_dir {
                            let output_dir_path = Path::new(output_dir);
                            let image_path = Path::new(&existing_images[i]);

                            // Create output directory if it doesn't exist
                            if !output_dir_path.exists() {
                                if let Err(e) = std::fs::create_dir_all(output_dir_path) {
                                    warn!(
                                        "Failed to create output directory {}: {}",
                                        output_dir, e
                                    );
                                } else {
                                    // Generate output filename based on input image
                                    let input_filename = image_path
                                        .file_stem()
                                        .and_then(|s| s.to_str())
                                        .unwrap_or("unknown");
                                    let output_filename =
                                        format!("{input_filename}_visualization.jpg");
                                    let output_path = output_dir_path.join(output_filename);

                                    // Create visualization
                                    let font_path = args.font_path.as_ref().map(Path::new);
                                    match visualize_ocr_results(result, &output_path, font_path) {
                                        Ok(()) => {
                                            info!(
                                                "Visualization saved to: {}",
                                                output_path.display()
                                            );
                                        }
                                        Err(e) => {
                                            warn!(
                                                "Failed to create visualization for {}: {}",
                                                image_path.display(),
                                                e
                                            );
                                        }
                                    }
                                }
                            } else {
                                // Generate output filename based on input image
                                let input_filename = image_path
                                    .file_stem()
                                    .and_then(|s| s.to_str())
                                    .unwrap_or("unknown");
                                let output_filename = format!("{input_filename}_visualization.jpg");
                                let output_path = output_dir_path.join(output_filename);

                                // Create visualization
                                let font_path = args.font_path.as_ref().map(Path::new);
                                match visualize_ocr_results(result, &output_path, font_path) {
                                    Ok(()) => {
                                        info!(
                                            "📊 Visualization saved to: {}",
                                            output_path.display()
                                        );
                                    }
                                    Err(e) => {
                                        warn!(
                                            "Failed to create visualization for {}: {}",
                                            image_path.display(),
                                            e
                                        );
                                    }
                                }
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Failed to process images: {}", e);
                    return Err(e.into());
                }
            }

            // Also demonstrate single image processing for comparison
            if !existing_images.is_empty() {
                info!("\nDemonstrating single image processing for comparison...");
                let single_image = &existing_images[0];
                let start_time = std::time::Instant::now();

                match oarocr.predict(&[Path::new(single_image)]) {
                    Ok(results) => {
                        if let Some(result) = results.first() {
                            // Display OCR results using the Display trait
                            info!("{}", result);

                            // Generate visualization if output directory is specified
                            #[cfg(feature = "visualization")]
                            if let Some(ref output_dir) = args.output_dir {
                                let output_dir_path = Path::new(output_dir);
                                let image_path = Path::new(single_image);

                                // Create output directory if it doesn't exist
                                if !output_dir_path.exists() {
                                    if let Err(e) = std::fs::create_dir_all(output_dir_path) {
                                        warn!(
                                            "Failed to create output directory {}: {}",
                                            output_dir, e
                                        );
                                    } else {
                                        // Generate output filename based on input image
                                        let input_filename = image_path
                                            .file_stem()
                                            .and_then(|s| s.to_str())
                                            .unwrap_or("unknown");
                                        let output_filename =
                                            format!("{input_filename}_single_visualization.jpg");
                                        let output_path = output_dir_path.join(output_filename);

                                        // Create visualization
                                        let font_path = args.font_path.as_ref().map(Path::new);
                                        match visualize_ocr_results(result, &output_path, font_path)
                                        {
                                            Ok(()) => {
                                                info!(
                                                    "Single image visualization saved to: {}",
                                                    output_path.display()
                                                );
                                            }
                                            Err(e) => {
                                                warn!(
                                                    "Failed to create visualization for {}: {}",
                                                    image_path.display(),
                                                    e
                                                );
                                            }
                                        }
                                    }
                                } else {
                                    // Generate output filename based on input image
                                    let input_filename = image_path
                                        .file_stem()
                                        .and_then(|s| s.to_str())
                                        .unwrap_or("unknown");
                                    let output_filename =
                                        format!("{input_filename}_single_visualization.jpg");
                                    let output_path = output_dir_path.join(output_filename);

                                    // Create visualization
                                    let font_path = args.font_path.as_ref().map(Path::new);
                                    match visualize_ocr_results(result, &output_path, font_path) {
                                        Ok(()) => {
                                            info!(
                                                "Single image visualization saved to: {}",
                                                output_path.display()
                                            );
                                        }
                                        Err(e) => {
                                            warn!(
                                                "Failed to create visualization for {}: {}",
                                                image_path.display(),
                                                e
                                            );
                                        }
                                    }
                                }
                            }
                        }

                        // Handle case when visualization feature is disabled but user wants to save
                        let processing_time = start_time.elapsed();
                        info!("Single image processed in {:?}", processing_time);

                        #[cfg(not(feature = "visualization"))]
                        if args.output_dir.is_some() {
                            warn!(
                                "Visualization feature is disabled. To enable visualization, compile with --features visualization"
                            );
                        }
                    }
                    Err(e) => {
                        error!("Single image OCR failed for {}: {}", single_image, e);
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
