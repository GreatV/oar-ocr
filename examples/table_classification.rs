//! Table Classification Example
//!
//! This example demonstrates how to use the OCR pipeline to classify table images.
//! It loads a table classification model, processes input images, and predicts whether
//! the table is "wired" (with borders) or "wireless" (without borders) with confidence scores.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example table_classification -- [OPTIONS] <IMAGES>...
//! ```
//!
//! # Arguments
//!
//! * `-m, --model-path` - Path to the table classification model file
//! * `-o, --output-dir` - Directory to save visualization results (optional)
//! * `--device` - Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
//! * `<IMAGES>...` - Paths to input table images to process
//!
//! # Example
//!
//! ```bash
//! cargo run --example table_classification -- \
//!     -m models/pp-lcnet_x1_0_table_cls.onnx \
//!     images/table_recognition.jpg
//! ```

mod utils;

use clap::Parser;
use oar_ocr::predictors::TableClassificationPredictor;
use oar_ocr::utils::load_image;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info, warn};
use utils::parse_device_config;

/// Command-line arguments for the table classification example
#[derive(Parser)]
#[command(name = "table_classification")]
#[command(about = "Table Classification Example - classifies tables as wired or wireless")]
struct Args {
    /// Path to the table classification model file
    #[arg(short, long)]
    model_path: PathBuf,

    /// Paths to input table images to process
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

    /// Number of top predictions to return (default: 2)
    #[arg(long, default_value = "2")]
    topk: usize,

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
    utils::init_tracing();

    // Parse command-line arguments
    let args = Args::parse();

    info!("Table Classification Example");

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
    let ort_config = parse_device_config(&args.device)?.unwrap_or_default();

    if ort_config.execution_providers.is_some() {
        info!("CUDA execution provider configured successfully");
    }

    if args.verbose {
        info!("Classification Configuration:");
        info!("  Score threshold: {}", args.score_thresh);
        info!("  Top-k: {}", args.topk);
        info!(
            "  Input shape: ({}, {})",
            args.input_height, args.input_width
        );
    }

    // Build the table classifier predictor
    if args.verbose {
        info!("Building table classifier predictor...");
        info!("  Model: {}", args.model_path.display());
    }

    let predictor = TableClassificationPredictor::builder()
        .score_threshold(args.score_thresh)
        .topk(args.topk)
        .input_shape((args.input_height, args.input_width))
        .with_ort_config(ort_config)
        .build(&args.model_path)?;

    info!("Table classifier predictor built successfully");

    // Load all images into memory
    info!("Processing {} images...", existing_images.len());
    let mut images = Vec::new();

    for image_path in &existing_images {
        match load_image(image_path) {
            Ok(rgb_img) => {
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

    // Run table classification
    info!("Running table classification...");
    let start = Instant::now();
    let output = predictor.predict(images.clone())?;
    let duration = start.elapsed();

    info!(
        "Classification completed in {:.2}ms",
        duration.as_secs_f64() * 1000.0
    );

    // Display results for each image
    info!("\n=== Classification Results ===");
    for (idx, (image_path, classifications)) in existing_images
        .iter()
        .zip(output.classifications.iter())
        .enumerate()
    {
        info!("\nImage {}: {}", idx + 1, image_path.display());

        if classifications.is_empty() {
            warn!("  No predictions available");
        } else {
            // Show top prediction prominently
            let top = &classifications[0];
            info!("  Table type: {}", top.label);
            info!("  Confidence: {:.2}%", top.score * 100.0);

            // Show all predictions if verbose
            if args.verbose && classifications.len() > 1 {
                info!("  All predictions:");
                for (rank, c) in classifications.iter().enumerate() {
                    info!("    [{}] {} - {:.2}%", rank + 1, c.label, c.score * 100.0);
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

        for (image_path, rgb_img, classifications) in existing_images
            .iter()
            .zip(images.iter())
            .zip(output.classifications.iter())
            .map(|((path, img), classifications)| (path, img, classifications))
        {
            if !classifications.is_empty() {
                // Extract labels and scores from classifications
                let labels: Vec<_> = classifications.iter().map(|c| c.label.clone()).collect();
                let scores: Vec<_> = classifications.iter().map(|c| c.score).collect();
                // Use the original filename for output
                let output_filename = image_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown.jpg");
                let output_path = output_dir.join(output_filename);

                // Get top prediction
                let table_type = &labels[0];
                let confidence = scores[0];

                let visualized = visualize_table_classification(rgb_img, table_type, confidence);
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

/// Visualizes table classification by drawing the predicted type and confidence on the image
/// The classification result is displayed in an additional space above the original image
#[cfg(feature = "visualization")]
fn visualize_table_classification(
    img: &image::RgbImage,
    table_type: &str,
    confidence: f32,
) -> image::RgbImage {
    use image::{Rgb, RgbImage};
    use imageproc::drawing::{draw_filled_rect_mut, draw_text_mut};
    use imageproc::rect::Rect;

    let original_width = img.width();
    let original_height = img.height();

    // Create additional space above the image for classification result
    let header_height = 60;
    let total_height = header_height + original_height;

    // Create new image with extra space on top
    let mut output = RgbImage::new(original_width, total_height);

    // Fill header area with background color
    let bg_color = Rgb([240u8, 240u8, 240u8]); // Light gray background

    for y in 0..header_height {
        for x in 0..original_width {
            output.put_pixel(x, y, bg_color);
        }
    }

    // Copy original image to the bottom part
    for y in 0..original_height {
        for x in 0..original_width {
            output.put_pixel(x, y + header_height, *img.get_pixel(x, y));
        }
    }

    let text_color = table_classification_color(table_type);

    // Try to load a font for text rendering
    let font = load_font();

    if let Some(ref font) = font {
        // Draw the table type label
        let label = format!(
            "Table Type: {} (Confidence: {:.1}%)",
            table_type,
            confidence * 100.0
        );

        // Center the text horizontally
        let text_x = 10;
        let text_y = 20;

        // Draw text
        draw_text_mut(&mut output, text_color, text_x, text_y, 24.0, font, &label);
    } else {
        // Fallback: draw a simple colored rectangle to indicate classification
        let indicator_color = table_classification_color(table_type);

        let indicator_rect = Rect::at(10, 15).of_size(30, 30);
        draw_filled_rect_mut(&mut output, indicator_rect, indicator_color);
    }

    output
}

#[cfg(feature = "visualization")]
fn table_classification_color(table_type: &str) -> image::Rgb<u8> {
    use image::Rgb;

    match table_type {
        "Wired" => Rgb([0u8, 128u8, 0u8]), // Green
        _ => Rgb([220u8, 20u8, 60u8]),     // Red for non-wired classifications
    }
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
