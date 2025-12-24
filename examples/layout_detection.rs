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
//! * `--model-name` - Model name to explicitly specify the model type. Supported names:
//!   - `pp_docblocklayout` - PP-DocBlockLayout model
//!   - `pp_doclayout_s` - PP-DocLayout Small variant
//!   - `pp_doclayout_m` - PP-DocLayout Medium variant
//!   - `pp_doclayout_l` - PP-DocLayout Large variant
//!   - `pp_doclayout_plus_l` - PP-DocLayout Plus Large variant
//!   - `picodet_layout_1x` - PicoDet Layout 1x model
//!   - `picodet_layout_1x_table` - PicoDet Layout 1x Table model
//!   - `picodet_s_layout_3cls` - PicoDet Small Layout 3-class model
//!   - `picodet_l_layout_3cls` - PicoDet Large Layout 3-class model
//!   - `picodet_s_layout_17cls` - PicoDet Small Layout 17-class model
//!   - `picodet_l_layout_17cls` - PicoDet Large Layout 17-class model
//!   - `rtdetr_h_layout_3cls` - RT-DETR High Layout 3-class model
//!   - `rtdetr_h_layout_17cls` - RT-DETR High Layout 17-class model
//! * `--score-threshold` - Score threshold for layout elements (default: 0.5)
//! * `--max-elements` - Maximum number of layout elements to detect (default: 100)
//! * `--device` - Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
//! * `<IMAGES>...` - Paths to input document images to process
//!
//! # Examples
//!
//! Basic usage with auto-detection:
//! ```bash
//! cargo run --example layout_detection -- \
//!     -m models/pp-doclayout_plus-l.onnx \
//!     document1.jpg document2.jpg
//! ```
//!
//! With explicit model name:
//! ```bash
//! cargo run --example layout_detection -- \
//!     -m models/pp-doclayout_plus-l.onnx \
//!     --model-name pp_doclayout_plus_l \
//!     -o output/ \
//!     document1.jpg document2.jpg
//! ```

mod utils;

use clap::Parser;
use oar_ocr::predictors::LayoutDetectionPredictor;
use oar_ocr::utils::load_image;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info, warn};
use utils::parse_device_config;

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

    /// Model name to explicitly specify the model type (auto-detected from filename if not specified).
    /// Supported: pp_docblocklayout, pp_doclayout_s, pp_doclayout_m, pp_doclayout_l, pp_doclayout_plus_l,
    /// picodet_layout_1x, picodet_layout_1x_table, picodet_s_layout_3cls, picodet_l_layout_3cls,
    /// picodet_s_layout_17cls, picodet_l_layout_17cls, rtdetr_h_layout_3cls, rtdetr_h_layout_17cls
    #[arg(long)]
    model_name: Option<String>,

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
    utils::init_tracing();

    // Parse command line arguments
    let args = Args::parse();

    info!("Loading layout detection model: {:?}", args.model_path);

    // Auto-detect model type from filename if not specified
    // Priority: --model-name > auto-detect from filename
    let model_type = if let Some(ref mn) = args.model_name {
        mn.clone()
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

    info!("Detected model type: {}", model_type);

    // Parse device configuration
    info!("Using device: {}", args.device);
    let ort_config = parse_device_config(&args.device)?;

    if ort_config.is_some() {
        info!("CUDA execution provider configured successfully");
    }

    // Build the layout detection predictor
    let mut predictor_builder = LayoutDetectionPredictor::builder()
        .model_name(model_type)
        .score_threshold(args.score_threshold);

    if let Some(ort_cfg) = ort_config {
        predictor_builder = predictor_builder.with_ort_config(ort_cfg);
    }

    let predictor = predictor_builder.build(&args.model_path)?;

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
        let img = match load_image(image_path) {
            Ok(img) => img,
            Err(e) => {
                error!("Failed to load image {:?}: {}", image_path, e);
                continue;
            }
        };

        let (width, height) = (img.width(), img.height());
        info!("Image size: {}x{}", width, height);

        #[cfg(feature = "visualization")]
        let img_for_vis = img.clone();

        // Run layout detection
        let start = Instant::now();
        let output = match predictor.predict(vec![img]) {
            Ok(output) => output,
            Err(e) => {
                error!("Layout detection failed for {:?}: {}", image_path, e);
                continue;
            }
        };
        let duration = start.elapsed();

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
                // Use the original filename for output
                let output_filename = image_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("result.png");
                let output_path = output_dir.join(output_filename);

                if let Err(e) = visualize_layout(&img_for_vis, elements, &output_path) {
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
    use imageproc::drawing::draw_hollow_rect_mut;
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

    // Draw bounding boxes
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
            }
        }
    }

    // Save visualization
    // Convert RGBA to RGB if saving as JPEG (JPEG doesn't support alpha channel)
    let extension = output_path
        .extension()
        .and_then(|s| s.to_str())
        .unwrap_or("")
        .to_lowercase();

    if extension == "jpg" || extension == "jpeg" {
        let rgb_img = image::DynamicImage::ImageRgba8(img).to_rgb8();
        rgb_img
            .save(output_path)
            .map_err(|e| format!("Failed to save visualization to {:?}: {}", output_path, e))?;
    } else {
        img.save(output_path)
            .map_err(|e| format!("Failed to save visualization to {:?}: {}", output_path, e))?;
    }

    Ok(())
}
