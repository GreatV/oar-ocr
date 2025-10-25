//! Table Cell Detection Example
//!
//! This example runs the table cell detection models (wired / wireless) exported
//! from PaddleOCR/PaddleX and prints the detected cell bounding boxes. When the
//! `visualization` feature is enabled it will also produce an annotated image.

use clap::Parser;
use oar_ocr::core::traits::{
    adapter::{AdapterBuilder, ModelAdapter},
    task::{ImageTaskInput, Task},
};
use oar_ocr::domain::adapters::RTDetrTableCellAdapterBuilder;
use oar_ocr::domain::tasks::{TableCellDetectionConfig, TableCellDetectionTask};
use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::time::Instant;
use tracing::{error, info, warn};

#[cfg(feature = "visualization")]
use image::RgbImage;
#[cfg(feature = "visualization")]
use std::fs;

/// Supported table cell model variants.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum TableCellModelVariant {
    /// RT-DETR-L wired table cell detector.
    RTDetrLWired,
    /// RT-DETR-L wireless table cell detector.
    RTDetrLWireless,
}

impl TableCellModelVariant {
    fn detect_from_filename(path: &Path) -> Option<Self> {
        let name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_ascii_lowercase();

        if name.contains("wired_table_cell") {
            Some(TableCellModelVariant::RTDetrLWired)
        } else if name.contains("wireless_table_cell") {
            Some(TableCellModelVariant::RTDetrLWireless)
        } else {
            None
        }
    }

    fn as_str(&self) -> &'static str {
        match self {
            TableCellModelVariant::RTDetrLWired => "rt-detr-l_wired_table_cell_det",
            TableCellModelVariant::RTDetrLWireless => "rt-detr-l_wireless_table_cell_det",
        }
    }
}

/// Command line arguments.
#[derive(Parser)]
#[command(name = "table_cell_detection")]
#[command(about = "Detect table cells using RT-DETR models exported from PaddleOCR/PaddleX")]
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
    #[arg(long, default_value_t = 0.5)]
    score_threshold: f32,

    /// Maximum number of cells per image
    #[arg(long, default_value_t = 300)]
    max_cells: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

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
        TableCellModelVariant::detect_from_filename(&args.model_path).ok_or_else(|| {
            format!(
                "Could not infer model type from filename '{}'. Specify --model-type explicitly.",
                args.model_path.display()
            )
        })?
    };

    info!("Detected model type: {}", variant.as_str());

    let task_config = TableCellDetectionConfig {
        score_threshold: args.score_threshold,
        max_cells: args.max_cells,
    };

    let adapter: Box<dyn ModelAdapter<Task = TableCellDetectionTask>> = match variant {
        TableCellModelVariant::RTDetrLWired => Box::new(
            RTDetrTableCellAdapterBuilder::new()
                .task_config(task_config.clone())
                .build(&args.model_path)
                .map_err(|e| format!("Failed to build RT-DETR wired table cell adapter: {}", e))?,
        ),
        TableCellModelVariant::RTDetrLWireless => Box::new(
            RTDetrTableCellAdapterBuilder::wireless()
                .task_config(task_config.clone())
                .build(&args.model_path)
                .map_err(|e| {
                    format!("Failed to build RT-DETR wireless table cell adapter: {}", e)
                })?,
        ),
    };

    let task = TableCellDetectionTask::new(task_config.clone());

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

        let img = match image::open(image_path) {
            Ok(img) => img.to_rgb8(),
            Err(e) => {
                error!("Failed to load image {:?}: {}", image_path, e);
                continue;
            }
        };

        info!("Image size: {}x{}", img.width(), img.height());

        let input = ImageTaskInput::new(vec![img.clone()]);
        if let Err(e) = task.validate_input(&input) {
            error!("Input validation failed for {:?}: {}", image_path, e);
            continue;
        }

        let start = Instant::now();
        let output = match adapter.execute(input, Some(&task_config)) {
            Ok(output) => output,
            Err(e) => {
                error!("Table cell detection failed for {:?}: {}", image_path, e);
                continue;
            }
        };
        let elapsed = start.elapsed();

        if let Err(e) = task.validate_output(&output) {
            error!("Output validation failed for {:?}: {}", image_path, e);
            continue;
        }

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
                && let Err(e) = visualize_cells(&img, cells, output_dir.as_path(), image_path)
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
    let normalized = model_type.to_ascii_lowercase();
    match normalized.as_str() {
        "rt-detr-l_wired_table_cell_det" | "rtdetr_l_wired_table_cell_det" => {
            Ok(TableCellModelVariant::RTDetrLWired)
        }
        "rt-detr-l_wireless_table_cell_det" | "rtdetr_l_wireless_table_cell_det" => {
            Ok(TableCellModelVariant::RTDetrLWireless)
        }
        _ => Err(HashSet::from([
            TableCellModelVariant::RTDetrLWired.as_str(),
            TableCellModelVariant::RTDetrLWireless.as_str(),
        ])
        .into_iter()
        .collect()),
    }
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

    let output_filename = format!(
        "table_cells_{}.png",
        image_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("result")
    );
    let output_path = output_dir.join(output_filename);
    canvas
        .save(&output_path)
        .map_err(|e| format!("Failed to save visualization to {:?}: {}", output_path, e).into())
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
