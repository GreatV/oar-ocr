//! Text Recognition Example
//!
//! This example demonstrates how to use the OCR pipeline to recognize text from cropped text images.
//! It loads a text recognition model and character dictionary, processes input images, and outputs
//! the recognized text with confidence scores.
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
//! * `-d, --dict-path` - Path to the character dictionary file
//! * `-o, --output-dir` - Directory to save visualization results (optional)
//! * `--device` - Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
//! * `<IMAGES>...` - Paths to input text images to process
//!
//! # Example
//!
//! ```bash
//! cargo run --example text_recognition -- \
//!     -m models/ppocrv4_mobile_rec.onnx \
//!     -d models/ppocr_keys_v1.txt \
//!     text1.jpg text2.jpg
//! ```

mod utils;

use clap::Parser;
use oar_ocr::predictors::TextRecognitionPredictor;
use oar_ocr::utils::load_image;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info, warn};
use utils::parse_device_config;

#[cfg(feature = "visualization")]
use image::RgbImage;

/// Command-line arguments for the text recognition example
#[derive(Parser)]
#[command(name = "text_recognition")]
#[command(about = "Text Recognition Example - recognizes text from cropped text images")]
struct Args {
    /// Path to the text recognition model file
    #[arg(short, long)]
    model_path: PathBuf,

    /// Path to the character dictionary file
    #[arg(short, long)]
    dict_path: PathBuf,

    /// Paths to input text images to process
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Directory to save visualization results
    #[arg(short, long)]
    output_dir: Option<PathBuf>,

    /// Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Score threshold for recognition (default: 0.0)
    #[arg(long, default_value = "0.0")]
    score_thresh: f32,

    /// Maximum image width for resizing (optional, e.g., 320)
    #[arg(long)]
    max_img_w: Option<usize>,

    /// Model input height (default: 48)
    #[arg(long, default_value = "48")]
    input_height: usize,

    /// Model input width (default: 320)
    #[arg(long, default_value = "320")]
    input_width: usize,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    utils::init_tracing();

    // Parse command-line arguments
    let args = Args::parse();

    info!("Text Recognition Example");

    // Verify that the model file exists
    if !args.model_path.exists() {
        error!("Model file not found: {}", args.model_path.display());
        return Err("Model file not found".into());
    }

    // Verify that the dictionary file exists
    if !args.dict_path.exists() {
        error!("Dictionary file not found: {}", args.dict_path.display());
        return Err("Dictionary file not found".into());
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
        info!("Recognition Configuration:");
        info!("  Score threshold: {}", args.score_thresh);
        info!(
            "  Input shape: [3, {}, {}]",
            args.input_height, args.input_width
        );
    }

    // Build the recognition predictor
    if args.verbose {
        info!("Building recognition predictor...");
        info!("  Model: {}", args.model_path.display());
    }

    let predictor = TextRecognitionPredictor::builder()
        .score_threshold(args.score_thresh)
        .dict_path(&args.dict_path)
        .with_ort_config(ort_config)
        .build(&args.model_path)?;

    info!("Recognition predictor built successfully");

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

    // Run recognition
    info!("Running text recognition...");
    let start = Instant::now();
    let output = predictor.predict(images.clone())?;
    let duration = start.elapsed();

    info!(
        "Recognition completed in {:.2}ms",
        duration.as_secs_f64() * 1000.0
    );

    // Display results for each image
    info!("\n=== Recognition Results ===");
    for (idx, (image_path, text, score)) in existing_images
        .iter()
        .zip(output.texts.iter())
        .zip(output.scores.iter())
        .map(|((path, text), score)| (path, text, score))
        .enumerate()
    {
        info!("\nImage {}: {}", idx + 1, image_path.display());
        if text.is_empty() {
            warn!("  No text recognized (below threshold)");
        } else {
            info!("  Text: \"{}\"", text);
            info!("  Confidence: {:.2}%", score * 100.0);
        }
    }

    // Save visualization if output directory is provided
    #[cfg(feature = "visualization")]
    if let Some(output_dir) = args.output_dir {
        // Create output directory if it doesn't exist
        std::fs::create_dir_all(&output_dir)?;

        info!("\nSaving visualizations to: {}", output_dir.display());

        for (image_path, rgb_img, text, score) in existing_images
            .iter()
            .zip(images.iter())
            .zip(output.texts.iter())
            .zip(output.scores.iter())
            .map(|(((path, img), text), score)| (path, img, text, score))
        {
            if !text.is_empty() {
                // Use the original filename for output
                let output_filename = image_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown.jpg");
                let output_path = output_dir.join(output_filename);

                let visualized = visualize_recognition(rgb_img, text, *score);
                visualized.save(&output_path)?;
                info!("  Saved: {}", output_path.display());
            } else {
                warn!(
                    "  Skipping visualization for {} (no text recognized)",
                    image_path.display()
                );
            }
        }
    }

    Ok(())
}

/// Visualizes recognized text by drawing it on the image
#[cfg(feature = "visualization")]
fn visualize_recognition(img: &RgbImage, text: &str, score: f32) -> RgbImage {
    use image::Rgb;
    use imageproc::drawing::draw_text_mut;

    let mut output = img.clone();
    let text_color = Rgb([255u8, 0u8, 0u8]); // Red for text

    // Try to load a font for text rendering
    let font = load_font();

    if let Some(ref font) = font {
        // Draw the recognized text at the top
        let label = format!("{} ({:.1}%)", text, score * 100.0);
        let text_x = 10;
        let text_y = 10;

        draw_text_mut(&mut output, text_color, text_x, text_y, 20.0, font, &label);
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
