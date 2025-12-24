//! Table Cell Detection Example
//!
//! This example runs the table cell detection models (wired / wireless) exported
//! from table detection models and prints the detected cell bounding boxes. When the
//! `visualization` feature is enabled it will also produce an annotated image.
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
//! * `-o, --output-dir` - Output directory for visualizations
//! * `--model-type` - Explicit model type override (e.g., 'rt-detr-l_wired_table_cell_det')
//! * `--score-threshold` - Score threshold for detections (default: 0.3)
//! * `--max-cells` - Maximum number of cells per image (default: 300)
//! * `--device` - Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
//! * `<IMAGES>...` - Input document images containing tables

mod utils;

use clap::Parser;
use oar_ocr::predictors::{TableCellDetectionPredictor, TableCellModelVariant};
use oar_ocr::utils::load_image;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info, warn};
use utils::parse_device_config;

#[cfg(feature = "visualization")]
use image::RgbImage;
#[cfg(feature = "visualization")]
use std::fs;
#[cfg(feature = "visualization")]
use std::path::Path;

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

    /// Output directory for visualizations (enabled with the `visualization` feature)
    #[cfg_attr(
        not(feature = "visualization"),
        doc = " (requires `visualization` feature)"
    )]
    #[arg(short, long)]
    output_dir: Option<PathBuf>,

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

    #[cfg(feature = "visualization")]
    if let Some(ref output_dir) = args.output_dir {
        fs::create_dir_all(output_dir)?;
    }

    for (idx, image_path) in args.images.iter().enumerate() {
        info!(
            "Processing image {}/{}: {:?}",
            idx + 1,
            args.images.len(),
            image_path
        );

        let img = match load_image(image_path) {
            Ok(img) => img,
            Err(e) => {
                error!("Failed to load image {:?}: {}", image_path, e);
                continue;
            }
        };

        info!("Image size: {}x{}", img.width(), img.height());

        #[cfg(feature = "visualization")]
        let img_for_vis = img.clone();

        let start = Instant::now();
        let output = match predictor.predict(vec![img]) {
            Ok(output) => output,
            Err(e) => {
                error!("Table cell detection failed for {:?}: {}", image_path, e);
                continue;
            }
        };
        let elapsed = start.elapsed();

        info!("Detection completed in {:.2?}", elapsed);

        if let Some(cells) = output.cells.first() {
            info!("Detected {} table cells", cells.len());
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

            #[cfg(feature = "visualization")]
            if let Some(ref output_dir) = args.output_dir
                && let Err(e) =
                    visualize_cells(&img_for_vis, cells, output_dir.as_path(), image_path)
            {
                error!("Failed to save visualization: {}", e);
            }
        } else {
            warn!("No cells detected for {:?}", image_path);
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

#[cfg(feature = "visualization")]
fn visualize_cells(
    img: &RgbImage,
    cells: &[oar_ocr::domain::tasks::TableCell],
    output_dir: &Path,
    image_path: &Path,
) -> Result<(), Box<dyn std::error::Error>> {
    use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
    use imageproc::rect::Rect;

    let mut canvas = image::DynamicImage::ImageRgb8(img.clone()).to_rgba8();
    let color = image::Rgba([0, 200, 255, 255]);

    let font = load_font();

    for (idx, cell) in cells.iter().enumerate() {
        if let Some((min_x, min_y, max_x, max_y)) = bbox_bounds(&cell.bbox) {
            let rect = Rect::at(min_x.max(0.0) as i32, min_y.max(0.0) as i32).of_size(
                (max_x - min_x).max(1.0) as u32,
                (max_y - min_y).max(1.0) as u32,
            );
            draw_hollow_rect_mut(&mut canvas, rect, color);

            if let Some(ref font) = font {
                let label = format!("{} #{}, {:.1}%", cell.label, idx, cell.score * 100.0);
                let text_x = min_x.max(0.0) as i32;
                let text_y = (min_y - 18.0).max(0.0) as i32;
                draw_text_mut(&mut canvas, color, text_x, text_y, 18.0, font, &label);
            }
        }
    }

    // Use the original filename for output
    let output_filename = image_path
        .file_name()
        .and_then(|s| s.to_str())
        .unwrap_or("result.png");
    let output_path = output_dir.join(output_filename);

    // Convert RGBA to RGB if saving as JPEG (JPEG doesn't support alpha channel)
    let extension = output_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    if extension == "jpg" || extension == "jpeg" {
        let rgb_canvas = image::DynamicImage::ImageRgba8(canvas).to_rgb8();
        rgb_canvas
            .save(&output_path)
            .map_err(|e| format!("Failed to save visualization to {:?}: {}", output_path, e).into())
    } else {
        canvas
            .save(&output_path)
            .map_err(|e| format!("Failed to save visualization to {:?}: {}", output_path, e).into())
    }
}

#[cfg(feature = "visualization")]
fn load_font() -> Option<ab_glyph::FontVec> {
    use ab_glyph::FontVec;

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
