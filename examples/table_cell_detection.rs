//! Table Cell Detection Example
//!
//! This example runs the table cell detection models (wired / wireless) exported
//! from table detection models and prints the detected cell bounding boxes.
//! It will also produce an annotated image when visualization is enabled.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example table_cell_detection -- [OPTIONS] <IMAGES>...
//! ```
//!
//! # Arguments
//!
//! * `-m, --model-path` - Path to the table cell detection model (.onnx)
//! * `-o, --output-dir` - Output directory for results
//! * `--vis` - Enable visualization output
//! * `--model-type` - Explicit model type override (e.g., 'rt-detr-l_wired_table_cell_det')
//! * `--score-threshold` - Score threshold for detections (default: 0.3)
//! * `--max-cells` - Maximum number of cells per image (default: 300)
//! * `--device` - Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
//! * `<IMAGES>...` - Input document images containing tables

mod utils;

use clap::Parser;
use image::Rgb;
use oar_ocr::predictors::{TableCellDetectionPredictor, TableCellModelVariant};
use oar_ocr::utils::load_image;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info, warn};
use utils::device_config::parse_device_config;
use utils::visualization::{Detection, DetectionVisConfig, save_rgb_image, visualize_detections};

/// Command line arguments.
#[derive(Parser)]
#[command(name = "table_cell_detection")]
#[command(about = "Detect table cells using RT-DETR models")]
struct Args {
    /// Path to the table cell detection model (.onnx)
    #[arg(short, long)]
    model_path: PathBuf,

    /// Input document images containing tables
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Output directory for results
    #[arg(short, long)]
    output_dir: Option<PathBuf>,

    /// Enable visualization output
    #[arg(long)]
    vis: bool,

    /// Explicit model type override (e.g., `rt-detr-l_wired_table_cell_det`)
    #[arg(long)]
    model_type: Option<String>,

    /// Score threshold for detections
    #[arg(long, default_value_t = 0.3)]
    score_threshold: f32,

    /// Maximum number of cells per image
    #[arg(long, default_value_t = 300)]
    max_cells: usize,

    /// Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
    #[arg(long, default_value = "cpu")]
    device: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    utils::init_tracing();

    let args = Args::parse();
    info!("Loading table cell detection model: {:?}", args.model_path);

    let variant = if let Some(ref model_type) = args.model_type {
        parse_model_variant(model_type).map_err(|supported| {
            format!(
                "Unknown model type '{}'. Supported types: {}",
                model_type,
                supported.join(", ")
            )
        })?
    } else {
        TableCellModelVariant::detect_from_path(&args.model_path).ok_or_else(|| {
            format!(
                "Could not infer model type from filename '{}'. Specify --model-type explicitly.",
                args.model_path.display()
            )
        })?
    };

    info!("Detected model type: {}", variant.as_str());

    // Log device configuration
    info!("Using device: {}", args.device);
    let ort_config = parse_device_config(&args.device)?;

    if ort_config.is_some() {
        info!("CUDA execution provider configured successfully");
    }

    // Build the table cell detection predictor
    let mut predictor_builder = TableCellDetectionPredictor::builder()
        .score_threshold(args.score_threshold)
        .model_variant(variant);

    if let Some(ort_cfg) = ort_config {
        predictor_builder = predictor_builder.with_ort_config(ort_cfg);
    }

    let predictor = predictor_builder.build(&args.model_path)?;

    // Load all images into memory
    let mut images = Vec::new();
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

    for image_path in &existing_images {
        match load_image(image_path) {
            Ok(img) => {
                info!(
                    "Loaded image: {} ({}x{})",
                    image_path.display(),
                    img.width(),
                    img.height()
                );
                images.push(img);
            }
            Err(e) => {
                error!("Failed to load image {:?}: {}", image_path, e);
                continue;
            }
        }
    }

    if images.is_empty() {
        error!("No images could be loaded for processing");
        return Err("No images could be loaded".into());
    }

    // Run detection
    info!("Running table cell detection...");
    let start = Instant::now();
    let output = predictor.predict(images.clone())?;
    let elapsed = start.elapsed();

    info!("Detection completed in {:.2?}", elapsed);

    // Display results for each image
    for (idx, (image_path, cells)) in existing_images.iter().zip(output.cells.iter()).enumerate() {
        info!("\n=== Results for image {} ===", idx + 1);
        info!("Image: {}", image_path.display());
        info!("Detected {} table cells", cells.len());

        if cells.is_empty() {
            warn!("No cells detected in this image");
        } else {
            for (cell_idx, cell) in cells.iter().enumerate() {
                if let Some((min_x, min_y, max_x, max_y)) = bbox_bounds(&cell.bbox) {
                    info!(
                        "  [{}] {}: ({:.0},{:.0})-({:.0},{:.0}), score: {:.3}",
                        cell_idx, cell.label, min_x, min_y, max_x, max_y, cell.score
                    );
                } else {
                    info!(
                        "  [{}] {}: <empty bbox>, score: {:.3}",
                        cell_idx, cell.label, cell.score
                    );
                }
            }
        }
    }

    // Save visualization if --vis is enabled
    if args.vis {
        let output_dir = args
            .output_dir
            .as_ref()
            .ok_or("--output-dir is required when --vis is enabled")?;

        // Create output directory if it doesn't exist
        std::fs::create_dir_all(output_dir)?;

        info!("\nSaving visualizations to: {}", output_dir.display());

        // Use cyan color for table cells
        let vis_config = DetectionVisConfig::default()
            .with_box_color(Rgb([0, 200, 255]))
            .with_label_color(Rgb([0, 200, 255]));

        for (image_path, rgb_img, cells) in existing_images
            .iter()
            .zip(images.iter())
            .zip(output.cells.iter())
            .map(|((path, img), cells)| (path, img, cells))
        {
            if !cells.is_empty() {
                // Build detection list for visualization with labels
                let vis_detections: Vec<Detection> = cells
                    .iter()
                    .map(|c| Detection::new(&c.bbox, c.score).with_label(&c.label))
                    .collect();

                // Use the original filename for output
                let output_filename = image_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("result.png");
                let output_path = output_dir.join(output_filename);

                let visualized = visualize_detections(rgb_img, &vis_detections, &vis_config);
                save_rgb_image(&visualized, &output_path)
                    .map_err(|e| format!("Failed to save visualization: {}", e))?;
                info!("  Saved: {}", output_path.display());
            } else {
                warn!(
                    "  Skipping visualization for {} (no cells detected)",
                    image_path.display()
                );
            }
        }
    }

    Ok(())
}

fn parse_model_variant(model_type: &str) -> Result<TableCellModelVariant, Vec<&'static str>> {
    TableCellModelVariant::from_model_type(model_type).ok_or_else(|| {
        vec![
            TableCellModelVariant::RTDetrLWired.as_str(),
            TableCellModelVariant::RTDetrLWireless.as_str(),
        ]
    })
}

fn bbox_bounds(bbox: &oar_ocr::processors::BoundingBox) -> Option<(f32, f32, f32, f32)> {
    if bbox.points.is_empty() {
        return None;
    }
    let min_x = bbox
        .points
        .iter()
        .map(|p| p.x)
        .fold(f32::INFINITY, f32::min);
    let min_y = bbox
        .points
        .iter()
        .map(|p| p.y)
        .fold(f32::INFINITY, f32::min);
    let max_x = bbox
        .points
        .iter()
        .map(|p| p.x)
        .fold(f32::NEG_INFINITY, f32::max);
    let max_y = bbox
        .points
        .iter()
        .map(|p| p.y)
        .fold(f32::NEG_INFINITY, f32::max);

    Some((min_x, min_y, max_x, max_y))
}
