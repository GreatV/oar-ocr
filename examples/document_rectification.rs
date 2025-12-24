//! Document Rectification Example
//!
//! This example demonstrates how to use the OCR pipeline to rectify distorted document images.
//! It loads a document rectification model, processes input images with geometric distortions
//! (such as perspective skew, wrinkles, or curves), and outputs corrected/straightened images.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example document_rectification -- [OPTIONS] <IMAGES>...
//! ```
//!
//! # Arguments
//!
//! * `-m, --model-path` - Path to the document rectification model file
//! * `-o, --output-dir` - Directory to save rectified images (required)
//! * `--device` - Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
//! * `<IMAGES>...` - Paths to input document images to rectify
//!
//! # Example
//!
//! ```bash
//! cargo run --example document_rectification -- \
//!     -m models/uvdoc_rectifier.onnx \
//!     -o output/ \
//!     distorted_doc1.jpg distorted_doc2.jpg
//! ```

mod utils;

use clap::Parser;
use oar_ocr::predictors::DocumentRectificationPredictor;
use oar_ocr::utils::load_image;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info};
use utils::parse_device_config;

/// Command-line arguments for the document rectification example
#[derive(Parser)]
#[command(name = "document_rectification")]
#[command(about = "Document Rectification Example - corrects distortions in document images")]
struct Args {
    /// Path to the document rectification model file
    #[arg(short, long)]
    model_path: PathBuf,

    /// Paths to input document images to rectify
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Directory to save rectified images
    #[arg(short, long)]
    output_dir: PathBuf,

    /// Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Model input height (default: dynamic input size)
    #[arg(long, default_value = "0")]
    input_height: usize,

    /// Model input width (default: dynamic input size)
    #[arg(long, default_value = "0")]
    input_width: usize,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,

    /// Save side-by-side comparison images
    #[arg(long)]
    save_comparison: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    utils::init_tracing();

    // Parse command-line arguments
    let args = Args::parse();

    info!("Document Rectification Example");

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

    // Build the rectification predictor
    if args.verbose {
        info!("Building rectification predictor...");
        info!("  Model: {}", args.model_path.display());
        if args.input_height > 0 && args.input_width > 0 {
            info!(
                "  Input shape override: [3, {}, {}]",
                args.input_height, args.input_width
            );
        } else {
            info!("  Input shape override: dynamic (use original image size)");
        }
    }

    let predictor = DocumentRectificationPredictor::builder()
        .with_ort_config(ort_config)
        .build(&args.model_path)?;

    info!("Rectification predictor built successfully");

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

    // Run rectification
    info!("Running document rectification...");
    let start = Instant::now();
    let output = predictor.predict(images.clone())?;
    let duration = start.elapsed();

    info!(
        "Rectification completed in {:.2}ms ({:.2}ms per image)",
        duration.as_secs_f64() * 1000.0,
        duration.as_secs_f64() * 1000.0 / existing_images.len() as f64
    );

    // Display results
    info!("\n=== Rectification Results ===");
    for (idx, (image_path, original_img, rectified_img)) in existing_images
        .iter()
        .zip(images.iter())
        .zip(output.images.iter())
        .map(|((path, orig), rect)| (path, orig, rect))
        .enumerate()
    {
        info!("\nImage {}: {}", idx + 1, image_path.display());
        info!(
            "  Original size: {}x{}",
            original_img.width(),
            original_img.height()
        );
        info!(
            "  Rectified size: {}x{}",
            rectified_img.width(),
            rectified_img.height()
        );
    }

    // Save rectified images
    std::fs::create_dir_all(&args.output_dir)?;
    info!(
        "\nSaving rectified images to: {}",
        args.output_dir.display()
    );

    for (image_path, original_img, rectified_img) in existing_images
        .iter()
        .zip(images.iter())
        .zip(output.images.iter())
        .map(|((path, orig), rect)| (path, orig, rect))
    {
        // Use the original filename for output
        let output_filename = image_path
            .file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown.jpg");

        // Save rectified image
        let rectified_path = args.output_dir.join(output_filename);
        rectified_img.save(&rectified_path)?;
        info!("  Saved rectified: {}", rectified_path.display());

        // Save comparison image if requested
        if args.save_comparison {
            let input_filename = image_path
                .file_stem()
                .and_then(|s| s.to_str())
                .unwrap_or("unknown");
            let comparison_filename = format!("{}_comparison.jpg", input_filename);
            let comparison_path = args.output_dir.join(&comparison_filename);
            let comparison_img = create_comparison_image(original_img, rectified_img);
            comparison_img.save(&comparison_path)?;
            info!("  Saved comparison: {}", comparison_path.display());
        }
    }

    info!("\nRectification complete!");

    Ok(())
}

/// Creates a side-by-side comparison image showing original and rectified versions.
fn create_comparison_image(
    original: &image::RgbImage,
    rectified: &image::RgbImage,
) -> image::RgbImage {
    use image::{Rgb, RgbImage};

    // Calculate dimensions for side-by-side layout
    let max_height = original.height().max(rectified.height());
    let total_width = original.width() + rectified.width() + 20; // 20px padding between images
    let padding = 10;

    // Create output image with white background
    let mut output = RgbImage::from_pixel(total_width, max_height, Rgb([255, 255, 255]));

    // Copy original image to left side
    for y in 0..original.height() {
        for x in 0..original.width() {
            output.put_pixel(x, y, *original.get_pixel(x, y));
        }
    }

    // Copy rectified image to right side (with padding)
    let x_offset = original.width() + padding * 2;
    for y in 0..rectified.height() {
        for x in 0..rectified.width() {
            output.put_pixel(x + x_offset, y, *rectified.get_pixel(x, y));
        }
    }

    // Add labels if visualization feature is enabled
    #[cfg(feature = "visualization")]
    {
        use imageproc::drawing::draw_text_mut;
        let text_color = Rgb([0u8, 0u8, 0u8]); // Black text

        if let Some(font) = load_font() {
            // Label for original image
            draw_text_mut(&mut output, text_color, 10, 10, 24.0, &font, "Original");

            // Label for rectified image
            draw_text_mut(
                &mut output,
                text_color,
                (x_offset + 10) as i32,
                10,
                24.0,
                &font,
                "Rectified",
            );
        }
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
