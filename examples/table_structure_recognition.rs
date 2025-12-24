//! Table Structure Recognition Example
//!
//! This example demonstrates how to use the OCR pipeline to recognize table structure.
//! It loads a table structure recognition model, processes input images, and predicts
//! the HTML structure with bounding boxes for table cells.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example table_structure_recognition -- [OPTIONS] <IMAGES>...
//! ```
//!
//! # Arguments
//!
//! * `-m, --model-path` - Path to the table structure recognition model file
//! * `--dict-path` - Path to table structure dictionary file (required)
//!   - `table_structure_dict_ch.txt` - Chinese dictionary (48 entries)
//!   - `table_structure_dict.txt` - English dictionary (28 entries)
//!   - `table_master_structure_dict.txt` - Master dictionary with extended tags
//! * `<IMAGES>...` - Paths to input table images to process
//!
//! # Output
//!
//! that match standard output format for easy comparison and verification.
//!
//! # Usage
//!
//! Simple run with default settings:
//!
//! ```bash
//! cargo run --example table_structure_recognition -- \
//!     --model-path path/to/model.onnx \
//!     --dict-path path/to/dict.txt \
//!     --image-path path/to/image.jpg
//! ```
//!
//! With custom dictionary:
//!
//! ```bash
//! cargo run --example table_structure_recognition -- \
//!     --model-path path/to/model.onnx \
//!     --dict-path /path/to/table_structure_dict_ch.txt \
//!     --image-path path/to/image.jpg
//! ```
//!
//! With wireless table model (requires different dictionary):
//!
//! ```bash
//! cargo run --example table_structure_recognition -- \
//!     --model-path path/to/model.onnx \
//!     --dict-path /path/to/table_structure_dict.txt \
//!     --table-type wireless \
//!     --image-path path/to/image.jpg
//! ```

mod utils;

use clap::Parser;
use oar_ocr::predictors::TableStructureRecognitionPredictor;
use oar_ocr::utils::load_image;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info};
use utils::parse_device_config;

/// Command-line arguments for the table structure recognition example
#[derive(Parser)]
#[command(name = "table_structure_recognition")]
#[command(about = "Table Structure Recognition Example - recognizes table structure as HTML")]
struct Args {
    /// Path to the table structure recognition model file
    #[arg(short, long)]
    model_path: PathBuf,

    /// Paths to input table images to process
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Path to table structure dictionary (required)
    #[arg(long)]
    dict_path: PathBuf,

    /// Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Score threshold for recognition (default: 0.5)
    #[arg(long, default_value = "0.5")]
    score_thresh: f32,

    /// Maximum structure sequence length (default: 500)
    #[arg(long, default_value = "500")]
    max_length: usize,

    /// Model input height (default: 512 for wired tables)
    #[arg(long, default_value = "512")]
    input_height: u32,

    /// Model input width (default: 512 for wired tables)
    #[arg(long, default_value = "512")]
    input_width: u32,

    /// Directory to save visualization results (requires `visualization` feature)
    #[cfg(feature = "visualization")]
    #[arg(short = 'o', long = "output-dir")]
    output_dir: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Parse command-line arguments
    let args = Args::parse();

    // Initialize tracing for logging
    utils::init_tracing();

    info!("Table Structure Recognition Example");

    // Verify that the model file exists
    if !args.model_path.exists() {
        error!("Model file not found: {}", args.model_path.display());
        return Err("Model file not found".into());
    }

    // Verify dictionary exists
    if !args.dict_path.exists() {
        error!("Dictionary file not found: {}", args.dict_path.display());
        return Err("Dictionary file not found".into());
    }

    // Filter out non-existent image files
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

    info!("Recognition Configuration:");
    info!("  Score threshold: {}", args.score_thresh);
    info!("  Max structure length: {}", args.max_length);
    info!(
        "  Input shape: ({}, {})",
        args.input_height, args.input_width
    );
    info!("  Dictionary: {}", args.dict_path.display());

    // Build the predictor
    info!("Building table structure recognition predictor...");
    info!("  Model: {}", args.model_path.display());

    let start_build = Instant::now();
    let predictor = TableStructureRecognitionPredictor::builder()
        .score_threshold(args.score_thresh)
        .dict_path(&args.dict_path)
        .input_shape(args.input_height, args.input_width)
        .with_ort_config(ort_config)
        .build(&args.model_path)?;

    info!(
        "Predictor built in {:.2}ms",
        start_build.elapsed().as_secs_f64() * 1000.0
    );

    // Load all images
    info!("Processing {} images...", existing_images.len());
    let mut images = Vec::new();

    for image_path in &existing_images {
        match load_image(image_path) {
            Ok(rgb_img) => {
                info!(
                    "Loaded image: {} ({}x{})",
                    image_path.display(),
                    rgb_img.width(),
                    rgb_img.height()
                );
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
    info!("Running table structure recognition...");
    let start = Instant::now();
    let output = predictor.predict(images)?;
    let duration = start.elapsed();

    info!(
        "Recognition completed in {:.2}ms",
        duration.as_secs_f64() * 1000.0
    );

    // Display results
    info!("\n=== Structure Recognition Results ===");
    for (idx, (structure, bboxes)) in output
        .structures
        .iter()
        .zip(output.bboxes.iter())
        .enumerate()
    {
        info!(
            "\nImage {}: {}",
            idx,
            existing_images
                .get(idx)
                .map(|p| p.display().to_string())
                .unwrap_or_else(|| "N/A".to_string())
        );
        info!("  Structure tokens ({}): {:?}", structure.len(), structure);
        info!("  Cell bboxes ({}): {:?}", bboxes.len(), bboxes);
    }

    #[cfg(feature = "visualization")]
    {
        if let Some(ref output_dir) = args.output_dir {
            std::fs::create_dir_all(output_dir)?;

            for (idx, structure) in output.structures.iter().enumerate() {
                let structure_html = structure.join("");
                let html_stem = existing_images
                    .get(idx)
                    .and_then(|path| path.file_stem())
                    .and_then(|name| name.to_str())
                    .unwrap_or("table_structure");
                let html_path = output_dir.join(format!("{}_{}_structure.html", html_stem, idx));

                if let Err(e) = std::fs::write(&html_path, structure_html) {
                    error!(
                        "Failed to write structure HTML {}: {}",
                        html_path.display(),
                        e
                    );
                } else {
                    info!("Structure HTML saved to: {}", html_path.display());
                }
            }
        }
    }

    Ok(())
}
