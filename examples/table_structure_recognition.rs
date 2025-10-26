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
//! The example outputs complete structure tokens, bounding boxes, and confidence scores
//! that match PaddleOCR format for easy comparison and verification.
//!
//! # Examples
//!
//! Basic usage:
//! ```bash
//! cargo run --example table_structure_recognition -- \
//!     -m models/slanext_wired.onnx \
//!     --dict-path models/table_structure_dict_ch.txt \
//!     images/table_recognition.jpg
//! ```
//!
//! With custom dictionary from PaddleOCR:
//! ```bash
//! cargo run --example table_structure_recognition -- \
//!     -m models/slanext_wired.onnx \
//!     --dict-path ~/repos/PaddleOCR/ppocr/utils/dict/table_structure_dict_ch.txt \
//!     images/table_recognition.jpg
//! ```
//!
//! Using English dictionary:
//! ```bash
//! cargo run --example table_structure_recognition -- \
//!     -m models/slanext_wired.onnx \
//!     --dict-path ~/repos/PaddleOCR/ppocr/utils/dict/table_structure_dict.txt \
//!     images/table_recognition.jpg
//! ```
//!
//! Output will show complete structure tokens and bounding boxes for verification.

mod common;

use clap::Parser;
use common::{load_rgb_image, parse_device_config};
use oar_ocr::core::traits::adapter::{AdapterBuilder, ModelAdapter};
use oar_ocr::core::traits::task::{ImageTaskInput, Task};
use oar_ocr::domain::adapters::SLANetWiredAdapterBuilder;
use oar_ocr::domain::tasks::table_structure_recognition::{
    TableStructureRecognitionConfig, TableStructureRecognitionTask,
};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info};

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

    /// Session pool size for concurrent inference (default: 1)
    #[arg(long, default_value = "1")]
    session_pool_size: usize,

    /// Model input height (default: 512)
    #[arg(long, default_value = "512")]
    input_height: u32,

    /// Model input width (default: 512)
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

    // Set log level to info for token and bbox output
    if std::env::var("RUST_LOG").is_err() {
        unsafe {
            std::env::set_var("RUST_LOG", "info");
        }
    }

    // Initialize tracing for logging
    oar_ocr::utils::init_tracing();

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
    let ort_config = parse_device_config(&args.device)?;

    if ort_config.is_some() {
        info!("CUDA execution provider configured successfully");
    }

    // Create configuration
    let config = TableStructureRecognitionConfig {
        score_threshold: args.score_thresh,
        max_structure_length: args.max_length,
    };

    info!("Recognition Configuration:");
    info!("  Score threshold: {}", config.score_threshold);
    info!("  Max structure length: {}", config.max_structure_length);
    info!(
        "  Input shape: ({}, {})",
        args.input_height, args.input_width
    );
    info!("  Dictionary: {}", args.dict_path.display());

    // Build the adapter
    info!("Building table structure recognition adapter...");
    info!("  Model: {}", args.model_path.display());
    info!("  Session pool size: {}", args.session_pool_size);

    let start_build = Instant::now();
    let mut adapter_builder = SLANetWiredAdapterBuilder::new()
        .with_config(config.clone())
        .input_shape((args.input_height, args.input_width))
        .session_pool_size(args.session_pool_size)
        .dict_path(&args.dict_path);

    if let Some(ort_cfg) = ort_config {
        adapter_builder = adapter_builder.with_ort_config(ort_cfg);
    }

    let adapter = adapter_builder.build(&args.model_path)?;
    let info_adapter = adapter.info();

    info!(
        "Adapter built in {:.2}ms",
        start_build.elapsed().as_secs_f64() * 1000.0
    );
    info!("  Task type: {:?}", info_adapter.task_type);
    info!("  Model name: {}", info_adapter.model_name);
    info!("  Version: {}", info_adapter.version);

    // Load all images
    info!("Processing {} images...", existing_images.len());
    let mut images = Vec::new();

    for image_path in &existing_images {
        match load_rgb_image(image_path) {
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

    // Create task input
    let input = ImageTaskInput::new(images.clone());

    // Create task for validation
    let task = TableStructureRecognitionTask::new(config.clone());
    task.validate_input(&input)?;

    // Run recognition (rebuild adapter for execution)
    info!("Running table structure recognition...");
    let start = Instant::now();

    let adapter = SLANetWiredAdapterBuilder::new()
        .with_config(config.clone())
        .input_shape((args.input_height, args.input_width))
        .session_pool_size(args.session_pool_size)
        .dict_path(&args.dict_path)
        .build(&args.model_path)?;
    let output = adapter.execute(input, Some(&config))?;

    let duration = start.elapsed();

    info!(
        "Recognition completed in {:.2}ms",
        duration.as_secs_f64() * 1000.0
    );

    // Display results
    info!("\n=== Structure Recognition Results ===");
    info!("\nImage: {}", existing_images[0].display());
    info!(
        "  Structure tokens ({}): {:?}",
        output.structure.len(),
        output.structure
    );
    info!("  Cell bboxes ({}): {:?}", output.bbox.len(), output.bbox);
    info!("  Confidence: {:.6}", output.structure_score);

    #[cfg(feature = "visualization")]
    {
        if let Some(ref output_dir) = args.output_dir {
            std::fs::create_dir_all(output_dir)?;

            let structure_html = output.structure.join("");
            let html_stem = existing_images
                .get(0)
                .and_then(|path| path.file_stem())
                .and_then(|name| name.to_str())
                .unwrap_or("table_structure");
            let html_path = output_dir.join(format!("{}_structure.html", html_stem));

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

    Ok(())
}
