//! Document Orientation Classification Example
//!
//! This example demonstrates how to use the OCR pipeline to classify document orientation.
//! It loads a document orientation model, processes input images, and predicts the rotation
//! angle (0°, 90°, 180°, or 270°) with confidence scores.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example document_orientation -- [OPTIONS] <IMAGES>...
//! ```
//!
//! # Arguments
//!
//! * `-m, --model-path` - Path to the document orientation model file
//! * `-o, --output-dir` - Directory to save visualization results (optional)
//! * `--device` - Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
//! * `<IMAGES>...` - Paths to input document images to process
//!
//! # Example
//!
//! ```bash
//! cargo run --example document_orientation -- \
//!     -m models/pplcnet_x1_0_doc_ori.onnx \
//!     document1.jpg document2.jpg
//! ```

use clap::Parser;
use oar_ocr::core::traits::adapter::{AdapterBuilder, ModelAdapter};
use oar_ocr::core::traits::task::{ImageTaskInput, Task};
use oar_ocr::domain::tasks::document_orientation::{
    DocumentOrientationConfig, DocumentOrientationTask,
};
use oar_ocr::models::classification::pp_lcnet_adapter::PPLCNetAdapterBuilder;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info, warn};

/// Command-line arguments for the document orientation example
#[derive(Parser)]
#[command(name = "document_orientation")]
#[command(about = "Document Orientation Classification Example - detects document rotation")]
struct Args {
    /// Path to the document orientation model file
    #[arg(short, long)]
    model_path: PathBuf,

    /// Paths to input document images to process
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Directory to save visualization results
    #[arg(short, long)]
    output_dir: Option<PathBuf>,

    /// Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Score threshold for classification (default: 0.5)
    #[arg(long, default_value = "0.5")]
    score_thresh: f32,

    /// Number of top predictions to return (default: 4)
    #[arg(long, default_value = "4")]
    topk: usize,

    /// Session pool size for concurrent inference (default: 1)
    #[arg(long, default_value = "1")]
    session_pool_size: usize,

    /// Model input height (default: 224)
    #[arg(long, default_value = "224")]
    input_height: u32,

    /// Model input width (default: 224)
    #[arg(long, default_value = "224")]
    input_width: u32,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    oar_ocr::utils::init_tracing();

    // Parse command-line arguments
    let args = Args::parse();

    info!("Document Orientation Classification Example");

    // Verify that the model file exists
    if !args.model_path.exists() {
        error!("Model file not found: {}", args.model_path.display());
        return Err("Model file not found".into());
    }

    // Filter out non-existent image files and log errors for missing files
    let existing_images: Vec<PathBuf> = args
        .images
        .iter()
        .filter(|path| {
            let exists = path.exists();
            if !exists {
                error!("Image file not found: {}", path.display());
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

    // Log device configuration
    info!("Using device: {}", args.device);
    if args.device != "cpu" {
        warn!("GPU support not yet implemented in new architecture. Using CPU.");
    }

    // Create orientation classification configuration
    let config = DocumentOrientationConfig {
        score_threshold: args.score_thresh,
        topk: args.topk,
    };

    if args.verbose {
        info!("Classification Configuration:");
        info!("  Score threshold: {}", config.score_threshold);
        info!("  Top-k: {}", config.topk);
        info!(
            "  Input shape: ({}, {})",
            args.input_height, args.input_width
        );
    }

    // Build the orientation classifier adapter
    if args.verbose {
        info!("Building orientation classifier adapter...");
        info!("  Model: {}", args.model_path.display());
        info!("  Session pool size: {}", args.session_pool_size);
    }

    let adapter = PPLCNetAdapterBuilder::<DocumentOrientationTask>::new()
        .with_config(config.clone())
        .input_shape((args.input_height, args.input_width))
        .session_pool_size(args.session_pool_size)
        .build(&args.model_path)?;

    info!("Orientation classifier adapter built successfully");
    if args.verbose {
        info!("  Task type: {:?}", adapter.info().task_type);
        info!("  Model name: {}", adapter.info().model_name);
        info!("  Version: {}", adapter.info().version);
    }

    // Load all images into memory
    info!("Processing {} images...", existing_images.len());
    let mut images = Vec::new();

    for image_path in &existing_images {
        match image::open(image_path) {
            Ok(img) => {
                let rgb_img = img.to_rgb8();
                if args.verbose {
                    info!(
                        "Loaded image: {} ({}x{})",
                        image_path.display(),
                        rgb_img.width(),
                        rgb_img.height()
                    );
                }
                images.push(rgb_img);
            }
            Err(e) => {
                error!("Failed to load image {}: {}", image_path.display(), e);
                continue;
            }
        }
    }

    if images.is_empty() {
        error!("No images could be loaded for processing");
        return Err("No images could be loaded".into());
    }

    // Create task input
    let input = ImageTaskInput::new(images.clone());

    // Create task for validation
    let task = DocumentOrientationTask::new(config.clone());
    task.validate_input(&input)?;

    // Run orientation classification
    info!("Running document orientation classification...");
    let start = Instant::now();
    let output = adapter.execute(input, Some(&config))?;
    let duration = start.elapsed();

    info!(
        "Classification completed in {:.2}ms",
        duration.as_secs_f64() * 1000.0
    );

    // Display results for each image
    info!("\n=== Classification Results ===");
    for (idx, (image_path, class_ids, scores, labels)) in existing_images
        .iter()
        .zip(output.class_ids.iter())
        .zip(output.scores.iter())
        .zip(output.label_names.iter())
        .map(|(((path, ids), scores), labels)| (path, ids, scores, labels))
        .enumerate()
    {
        info!("\nImage {}: {}", idx + 1, image_path.display());

        if class_ids.is_empty() {
            warn!("  No predictions available");
        } else {
            // Show top prediction prominently
            let top_label = &labels[0];
            let top_score = scores[0];
            info!("  Detected orientation: {}°", top_label);
            info!("  Confidence: {:.2}%", top_score * 100.0);

            // Show all predictions if verbose
            if args.verbose && class_ids.len() > 1 {
                info!("  All predictions:");
                for (rank, (label, score)) in labels.iter().zip(scores.iter()).enumerate() {
                    info!("    [{}] {}° - {:.2}%", rank + 1, label, score * 100.0);
                }
            }
        }
    }

    // Save visualization if output directory is provided
    #[cfg(feature = "visualization")]
    if let Some(output_dir) = args.output_dir {
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&output_dir)?;

        info!("\nSaving visualizations to: {}", output_dir.display());

        for (image_path, rgb_img, labels, scores) in existing_images
            .iter()
            .zip(images.iter())
            .zip(output.label_names.iter())
            .zip(output.scores.iter())
            .map(|(((path, img), labels), scores)| (path, img, labels, scores))
        {
            if !labels.is_empty() {
                let input_filename = image_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown");
                let output_filename = format!("{}_orientation.jpg", input_filename);
                let output_path = output_dir.join(&output_filename);

                // Get top prediction
                let orientation = &labels[0];
                let confidence = scores[0];

                let visualized = visualize_orientation(rgb_img, orientation, confidence);
                visualized.save(&output_path)?;
                info!("  Saved: {}", output_path.display());
            } else {
                warn!(
                    "  Skipping visualization for {} (no predictions)",
                    image_path.display()
                );
            }
        }
    }

    Ok(())
}

/// Visualizes document orientation by drawing the predicted angle and confidence on the image
#[cfg(feature = "visualization")]
fn visualize_orientation(
    img: &image::RgbImage,
    orientation: &str,
    confidence: f32,
) -> image::RgbImage {
    use image::Rgb;
    use imageproc::drawing::{draw_filled_rect_mut, draw_text_mut};
    use imageproc::rect::Rect;

    let mut output = img.clone();
    let text_color = Rgb([255u8, 255u8, 255u8]); // White text
    let bg_color = Rgb([0u8, 0u8, 0u8]); // Black background

    // Try to load a font for text rendering
    let font = load_font();

    if let Some(ref font) = font {
        // Draw the orientation label with background
        let label = format!("Orientation: {}° ({:.1}%)", orientation, confidence * 100.0);

        // Draw background rectangle for text
        let text_x = 10;
        let text_y = 10;
        let text_width = label.len() as u32 * 10; // Approximate
        let text_height = 30;

        if text_x + text_width < output.width() && text_y + text_height < output.height() {
            let bg_rect = Rect::at(text_x as i32, text_y as i32).of_size(text_width, text_height);
            draw_filled_rect_mut(&mut output, bg_rect, bg_color);

            // Draw text on top
            draw_text_mut(
                &mut output,
                text_color,
                text_x as i32,
                (text_y + 5) as i32,
                20.0,
                font,
                &label,
            );
        }

        // Optionally rotate the image to show corrected orientation
        // This is commented out as it would require additional dependencies
        /*
        if let Ok(angle) = orientation.parse::<i32>() {
            if angle != 0 {
                info!("Note: Image would need to be rotated {}° for correct orientation", (360 - angle) % 360);
            }
        }
        */
    }

    output
}

#[cfg(feature = "visualization")]
fn load_font() -> Option<ab_glyph::FontVec> {
    use ab_glyph::FontVec;

    // Try common font paths
    let font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/System/Library/Fonts/Arial.ttf",
        "C:\\Windows\\Fonts\\arial.ttf",
    ];

    for path in &font_paths {
        if let Ok(font_data) = std::fs::read(path)
            && let Ok(font) = FontVec::try_from_vec(font_data)
        {
            return Some(font);
        }
    }

    None
}
