//! Layout Detection Example
//!
//! This example demonstrates how to use the OCR pipeline to detect layout elements in document images.
//! It loads a layout detection model, processes input images, and identifies document structure elements
//! such as text blocks, titles, lists, tables, and figures.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example layout_detection -- [OPTIONS] <IMAGES>...
//! ```
//!
//! # Arguments
//!
//! * `-m, --model-path` - Path to the layout detection model file
//! * `-o, --output-dir` - Directory to save visualization results (optional)
//! * `--model-type` - Model type to use (auto-detected from filename if not specified)
//! * `--score-threshold` - Score threshold for layout elements (default: 0.5)
//! * `--max-elements` - Maximum number of layout elements to detect (default: 100)
//! * `--device` - Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
//! * `<IMAGES>...` - Paths to input document images to process
//!
//! # Example
//!
//! ```bash
//! cargo run --example layout_detection -- \
//!     -m .oar/pp-doclayout_plus-l.onnx \
//!     -o output/ \
//!     document1.jpg document2.jpg
//! ```

use clap::Parser;
use oar_ocr::core::traits::{
    adapter::{AdapterBuilder, ModelAdapter},
    task::{ImageTaskInput, Task},
};
use oar_ocr::domain::adapters::{
    LayoutDetectionAdapterBuilder, LayoutModelConfig, PPDocLayoutAdapterBuilder,
    PicoDetLayoutAdapterBuilder, RTDetrLayoutAdapterBuilder,
};
use oar_ocr::domain::tasks::{LayoutDetectionConfig, LayoutDetectionTask};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info, warn};

#[cfg(feature = "visualization")]
use std::fs;

#[cfg(feature = "visualization")]
use image::RgbImage;

/// Command-line arguments for the layout detection example
#[derive(Parser)]
#[command(name = "layout_detection")]
#[command(about = "Layout Detection Example - detects document structure elements")]
struct Args {
    /// Path to the layout detection model file
    #[arg(short, long)]
    model_path: PathBuf,

    /// Paths to input document images to process
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Directory to save visualization results (if visualization feature is enabled)
    #[arg(short, long)]
    output_dir: Option<PathBuf>,

    /// Model type to use (auto-detected from filename if not specified)
    #[arg(long)]
    model_type: Option<String>,

    /// Score threshold for layout elements (0.0 to 1.0)
    #[arg(long, default_value_t = 0.5)]
    score_threshold: f32,

    /// Maximum number of layout elements to detect
    #[arg(long, default_value_t = 100)]
    max_elements: usize,

    /// Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
    #[arg(long, default_value = "cpu")]
    device: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize logging
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive(tracing::Level::INFO.into()),
        )
        .init();

    // Parse command line arguments
    let args = Args::parse();

    info!("Loading layout detection model: {:?}", args.model_path);

    // Auto-detect model type from filename if not specified
    let model_type = if let Some(ref mt) = args.model_type {
        mt.clone()
    } else {
        let filename = args
            .model_path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("")
            .to_lowercase();

        // Detect model type from filename
        if filename.contains("pp-docblocklayout") || filename.contains("pp_docblocklayout") {
            "pp_docblocklayout".to_string()
        } else if filename.contains("pp-doclayout_plus-l")
            || filename.contains("pp_doclayout_plus_l")
        {
            "pp_doclayout_plus_l".to_string()
        } else if filename.contains("pp-doclayout-l") || filename.contains("pp_doclayout_l") {
            "pp_doclayout_l".to_string()
        } else if filename.contains("pp-doclayout-m") || filename.contains("pp_doclayout_m") {
            "pp_doclayout_m".to_string()
        } else if filename.contains("pp-doclayout-s") || filename.contains("pp_doclayout_s") {
            "pp_doclayout_s".to_string()
        } else if filename.contains("picodet-s_layout_3cls")
            || filename.contains("picodet_s_layout_3cls")
        {
            "picodet_s_layout_3cls".to_string()
        } else if filename.contains("picodet-l_layout_3cls")
            || filename.contains("picodet_l_layout_3cls")
        {
            "picodet_l_layout_3cls".to_string()
        } else if filename.contains("picodet-s_layout_17cls")
            || filename.contains("picodet_s_layout_17cls")
        {
            "picodet_s_layout_17cls".to_string()
        } else if filename.contains("picodet-l_layout_17cls")
            || filename.contains("picodet_l_layout_17cls")
        {
            "picodet_l_layout_17cls".to_string()
        } else if filename.contains("picodet_layout_1x_table") {
            "picodet_layout_1x_table".to_string()
        } else if filename.contains("rt-detr-h_layout_3cls")
            || filename.contains("rtdetr_h_layout_3cls")
        {
            "rtdetr_h_layout_3cls".to_string()
        } else if filename.contains("rt-detr-h_layout_17cls")
            || filename.contains("rtdetr_h_layout_17cls")
        {
            "rtdetr_h_layout_17cls".to_string()
        } else if filename.contains("picodet_layout_1x") {
            "picodet_layout_1x".to_string()
        } else {
            warn!(
                "Could not auto-detect model type from filename '{}', using default 'picodet_layout_1x'",
                filename
            );
            "picodet_layout_1x".to_string()
        }
    };

    info!("Model type: {}", model_type);

    // Create model configuration based on model type
    let _model_config = match model_type.as_str() {
        "pp_docblocklayout" => LayoutModelConfig::pp_docblocklayout(),
        "picodet_layout_1x_table"
        | "picodet-layout_1x_table"
        | "picodet_layout_1x-table"
        | "picodet-layout-1x-table" => LayoutModelConfig::picodet_layout_1x_table(),
        "picodet_layout_1x" => LayoutModelConfig::picodet_layout_1x(),
        "picodet_s_layout_3cls" => LayoutModelConfig::picodet_s_layout_3cls(),
        "picodet_l_layout_3cls" => LayoutModelConfig::picodet_l_layout_3cls(),
        "picodet_s_layout_17cls" => LayoutModelConfig::picodet_s_layout_17cls(),
        "picodet_l_layout_17cls" => LayoutModelConfig::picodet_l_layout_17cls(),
        "pp_doclayout_s" | "pp-doclayout-s" => LayoutModelConfig::pp_doclayout_s(),
        "pp_doclayout_m" | "pp-doclayout-m" => LayoutModelConfig::pp_doclayout_m(),
        "pp_doclayout_l" | "pp-doclayout-l" => LayoutModelConfig::pp_doclayout_l(),
        "pp_doclayout_plus_l" | "pp-doclayout_plus-l" => LayoutModelConfig::pp_doclayout_plus_l(),
        "rtdetr_h_layout_3cls" | "rt-detr-h_layout_3cls" => {
            LayoutModelConfig::rtdetr_h_layout_3cls()
        }
        "rtdetr_h_layout_17cls" | "rt-detr-h_layout_17cls" => {
            LayoutModelConfig::rtdetr_h_layout_17cls()
        }
        _ => {
            error!("Unknown model type: {}", model_type);
            error!(
                "Available types: pp_docblocklayout, picodet_layout_1x, picodet_layout_1x_table, picodet_s_layout_3cls, picodet_l_layout_3cls, picodet_s_layout_17cls, picodet_l_layout_17cls, pp_doclayout_s, pp_doclayout_m, pp_doclayout_l, pp_doclayout_plus_l, rtdetr_h_layout_3cls, rtdetr_h_layout_17cls"
            );
            std::process::exit(1);
        }
    };

    // Create task configuration
    let config = LayoutDetectionConfig {
        score_threshold: args.score_threshold,
        max_elements: args.max_elements,
    };

    // Build the adapter based on model type
    let adapter: Box<dyn ModelAdapter<Task = LayoutDetectionTask>> = match model_type.as_str() {
        "pp_docblocklayout"
        | "pp-docblocklayout"
        | "pp_doclayout_s"
        | "pp-doclayout-s"
        | "pp_doclayout_m"
        | "pp-doclayout-m"
        | "pp_doclayout_l"
        | "pp-doclayout-l"
        | "pp_doclayout_plus_l"
        | "pp-doclayout_plus-l" => Box::new(
            PPDocLayoutAdapterBuilder::new(&model_type)
                .task_config(config.clone())
                .build(&args.model_path)
                .map_err(|e| {
                    format!(
                        "Failed to build PP-DocLayout adapter from model {:?}: {}",
                        args.model_path, e
                    )
                })?,
        ),
        "rtdetr_h_layout_3cls" | "rt-detr-h_layout_3cls" => Box::new(
            RTDetrLayoutAdapterBuilder::new()
                .task_config(config.clone())
                .build(&args.model_path)
                .map_err(|e| {
                    format!(
                        "Failed to build RT-DETR adapter from model {:?}: {}",
                        args.model_path, e
                    )
                })?,
        ),
        "rtdetr_h_layout_17cls" | "rt-detr-h_layout_17cls" => Box::new(
            RTDetrLayoutAdapterBuilder::new_17cls()
                .task_config(config.clone())
                .build(&args.model_path)
                .map_err(|e| {
                    format!(
                        "Failed to build RT-DETR adapter (17cls) from model {:?}: {}",
                        args.model_path, e
                    )
                })?,
        ),
        "picodet_layout_1x_table"
        | "picodet-layout_1x_table"
        | "picodet_layout_1x-table"
        | "picodet-layout-1x-table" => Box::new(
            LayoutDetectionAdapterBuilder::new()
                .model_config(LayoutModelConfig::picodet_layout_1x_table())
                .task_config(config.clone())
                .build(&args.model_path)
                .map_err(|e| {
                    format!(
                        "Failed to build PicoDet adapter (table) from model {:?}: {}",
                        args.model_path, e
                    )
                })?,
        ),
        "picodet_s_layout_3cls" => Box::new(
            PicoDetLayoutAdapterBuilder::new_3cls()
                .task_config(config.clone())
                .build(&args.model_path)
                .map_err(|e| {
                    format!(
                        "Failed to build PicoDet-S adapter from model {:?}: {}",
                        args.model_path, e
                    )
                })?,
        ),
        "picodet_l_layout_3cls" => Box::new(
            LayoutDetectionAdapterBuilder::new()
                .model_config(LayoutModelConfig::picodet_l_layout_3cls())
                .task_config(config.clone())
                .build(&args.model_path)
                .map_err(|e| {
                    format!(
                        "Failed to build PicoDet-L adapter (3cls) from model {:?}: {}",
                        args.model_path, e
                    )
                })?,
        ),
        "picodet_s_layout_17cls" => Box::new(
            LayoutDetectionAdapterBuilder::new()
                .model_config(LayoutModelConfig::picodet_s_layout_17cls())
                .task_config(config.clone())
                .build(&args.model_path)
                .map_err(|e| {
                    format!(
                        "Failed to build PicoDet-S adapter (17cls) from model {:?}: {}",
                        args.model_path, e
                    )
                })?,
        ),
        "picodet_l_layout_17cls" => Box::new(
            LayoutDetectionAdapterBuilder::new()
                .model_config(LayoutModelConfig::picodet_l_layout_17cls())
                .task_config(config.clone())
                .build(&args.model_path)
                .map_err(|e| {
                    format!(
                        "Failed to build PicoDet-L adapter (17cls) from model {:?}: {}",
                        args.model_path, e
                    )
                })?,
        ),
        _ => {
            // Default to PicoDet for other models
            Box::new(
                PicoDetLayoutAdapterBuilder::new()
                    .task_config(config.clone())
                    .build(&args.model_path)
                    .map_err(|e| {
                        format!(
                            "Failed to build PicoDet adapter from model {:?}: {}",
                            args.model_path, e
                        )
                    })?,
            )
        }
    };

    // Create the task
    let task = LayoutDetectionTask::new(config);

    // Create output directory if needed
    #[cfg(feature = "visualization")]
    if let Some(ref output_dir) = args.output_dir {
        fs::create_dir_all(output_dir)?;
    }

    // Process each image
    for (img_idx, image_path) in args.images.iter().enumerate() {
        info!(
            "Processing image {}/{}: {:?}",
            img_idx + 1,
            args.images.len(),
            image_path
        );

        // Load input image
        let img = match image::open(image_path) {
            Ok(img) => img.to_rgb8(),
            Err(e) => {
                error!("Failed to load image {:?}: {}", image_path, e);
                continue;
            }
        };

        let (width, height) = (img.width(), img.height());
        info!("Image size: {}x{}", width, height);

        // Create input
        let input = ImageTaskInput::new(vec![img.clone()]);

        // Validate input
        if let Err(e) = task.validate_input(&input) {
            error!("Input validation failed for {:?}: {}", image_path, e);
            continue;
        }

        // Run layout detection
        let start = Instant::now();
        let output = match adapter.execute(input, None) {
            Ok(output) => output,
            Err(e) => {
                error!("Layout detection failed for {:?}: {}", image_path, e);
                continue;
            }
        };
        let duration = start.elapsed();

        // Validate output
        if let Err(e) = task.validate_output(&output) {
            error!("Output validation failed for {:?}: {}", image_path, e);
            continue;
        }

        info!("Detection completed in {:.2?}", duration);

        // Process results
        if !output.elements.is_empty() {
            let elements = &output.elements[0];
            info!("Detected {} layout elements", elements.len());

            // Group elements by type
            let mut type_counts = std::collections::HashMap::new();
            for element in elements {
                *type_counts.entry(element.element_type.clone()).or_insert(0) += 1;
            }

            info!("Layout element summary:");
            for (element_type, count) in type_counts {
                let type_name = format_element_type(&element_type);
                info!("  {}: {}", type_name, count);
            }

            // Show detailed results
            for (idx, element) in elements.iter().enumerate() {
                let type_name = format_element_type(&element.element_type);

                // Get bounding box corners
                let bbox = &element.bbox;
                if !bbox.points.is_empty() {
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

                    info!(
                        "  [{}] {}: ({:.0},{:.0})-({:.0},{:.0}), score: {:.3}",
                        idx, type_name, min_x, min_y, max_x, max_y, element.score
                    );
                }
            }

            // Visualization
            #[cfg(feature = "visualization")]
            if let Some(ref output_dir) = args.output_dir {
                let output_filename = format!(
                    "layout_detection_{}.png",
                    image_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("result")
                );
                let output_path = output_dir.join(output_filename);

                if let Err(e) = visualize_layout(&img, elements, &output_path) {
                    error!("Failed to save visualization: {}", e);
                } else {
                    info!("Visualization saved to: {:?}", output_path);
                }
            }
        } else {
            warn!("No layout elements detected in {:?}", image_path);
        }
    }

    Ok(())
}

/// Format element type for display
fn format_element_type(element_type: &str) -> String {
    // Capitalize first letter for display
    if element_type.is_empty() {
        "Unknown".to_string()
    } else {
        let mut chars = element_type.chars();
        match chars.next() {
            None => "Unknown".to_string(),
            Some(first) => first.to_uppercase().chain(chars).collect(),
        }
    }
}

/// Visualize layout detection results
#[cfg(feature = "visualization")]
fn visualize_layout(
    img: &RgbImage,
    elements: &[oar_ocr::domain::tasks::LayoutElement],
    output_path: &PathBuf,
) -> Result<(), Box<dyn std::error::Error>> {
    use imageproc::drawing::{draw_hollow_rect_mut, draw_text_mut};
    use imageproc::rect::Rect;

    // Convert to DynamicImage for drawing
    let mut img = image::DynamicImage::ImageRgb8(img.clone()).to_rgba8();

    // Define colors for different element types
    let colors = [
        image::Rgba([255, 0, 0, 255]),   // Red for Text
        image::Rgba([0, 255, 0, 255]),   // Green for Title
        image::Rgba([0, 0, 255, 255]),   // Blue for List
        image::Rgba([255, 255, 0, 255]), // Yellow for Table
        image::Rgba([255, 0, 255, 255]), // Magenta for Figure
    ];

    // Try to load a font for text rendering
    let font = load_font();

    // Draw bounding boxes and labels
    for element in elements {
        // Determine color based on element type string
        let color = match element.element_type.to_lowercase().as_str() {
            "text" => colors[0],
            "title" | "paragraph_title" | "doc_title" => colors[1],
            "list" => colors[2],
            "table" => colors[3],
            _ => colors[4], // Default to magenta for figure, image, and others
        };

        // Get bounding rectangle
        let bbox = &element.bbox;
        if !bbox.points.is_empty() {
            let min_x = bbox
                .points
                .iter()
                .map(|p| p.x as i32)
                .min()
                .unwrap_or(0)
                .max(0);
            let min_y = bbox
                .points
                .iter()
                .map(|p| p.y as i32)
                .min()
                .unwrap_or(0)
                .max(0);
            let max_x = bbox
                .points
                .iter()
                .map(|p| p.x as i32)
                .max()
                .unwrap_or(0)
                .min(img.width() as i32);
            let max_y = bbox
                .points
                .iter()
                .map(|p| p.y as i32)
                .max()
                .unwrap_or(0)
                .min(img.height() as i32);

            if max_x > min_x && max_y > min_y {
                let rect =
                    Rect::at(min_x, min_y).of_size((max_x - min_x) as u32, (max_y - min_y) as u32);
                draw_hollow_rect_mut(&mut img, rect, color);

                // Draw label with element type and confidence score
                if let Some(ref font) = font {
                    let type_name = format_element_type(&element.element_type);
                    let label = format!("{} {:.1}%", type_name, element.score * 100.0);
                    let label_x = min_x.max(0);
                    let label_y = (min_y - 20).max(0);

                    // Draw text label
                    draw_text_mut(&mut img, color, label_x, label_y, 20.0, font, &label);
                }
            }
        }
    }

    // Save visualization
    img.save(output_path)
        .map_err(|e| format!("Failed to save visualization to {:?}: {}", output_path, e))?;

    Ok(())
}

/// Load a font for text rendering
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
