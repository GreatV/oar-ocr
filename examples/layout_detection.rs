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
//! * `-o, --output-dir` - Directory to save output results
//! * `--vis` - Enable visualization output
//! * `--model-name` - Model name to explicitly specify the model type
//! * `--score-threshold` - Score threshold for layout elements (default: 0.5)
//! * `--device` - Device to use for inference (e.g., 'cpu', 'cuda', 'cuda:0')
//! * `<IMAGES>...` - Paths to input document images to process
//!
//! # Example
//!
//! ```bash
//! cargo run --example layout_detection -- \
//!     -m models/pp-doclayout_plus-l.onnx \
//!     -o output/ --vis \
//!     document1.jpg document2.jpg
//! ```

mod utils;

use clap::Parser;
use oar_ocr::domain::tasks::layout_detection::LayoutDetectionConfig;
use oar_ocr::predictors::LayoutDetectionPredictor;
use oar_ocr::utils::load_image;
use serde_json::json;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info, warn};
use utils::device_config::parse_device_config;
use utils::visualization::{LayoutItem, save_image, visualize_layout};

use std::fs;

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

    /// Directory to save output results
    #[arg(short, long)]
    output_dir: Option<PathBuf>,

    /// Enable visualization output
    #[arg(long)]
    vis: bool,

    /// Model name to explicitly specify the model type (auto-detected from filename if not specified).
    /// Supported: pp_docblocklayout, pp_doclayout_s, pp_doclayout_m, pp_doclayout_l, pp_doclayout_plus_l,
    /// picodet_layout_1x, picodet_layout_1x_table, picodet_s_layout_3cls, picodet_l_layout_3cls,
    /// picodet_s_layout_17cls, picodet_l_layout_17cls, rtdetr_h_layout_3cls, rtdetr_h_layout_17cls
    #[arg(long)]
    model_name: Option<String>,

    /// Score threshold for layout elements (overrides model defaults)
    #[arg(long)]
    score_threshold: Option<f32>,

    /// Maximum number of layout elements to detect
    #[arg(long, default_value_t = 100)]
    max_elements: usize,

    /// Dump layout elements as JSON to stdout
    #[arg(long)]
    dump_json: bool,

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
    let mut base_config = match model_type.as_str() {
        "pp_doclayoutv2" => LayoutDetectionConfig::with_pp_doclayoutv2_defaults(),
        "pp_doclayoutv3" => LayoutDetectionConfig::with_pp_doclayoutv3_defaults(),
        _ => LayoutDetectionConfig::default(),
    };
    base_config.max_elements = args.max_elements;
    if let Some(threshold) = args.score_threshold {
        base_config.score_threshold = threshold;
    }

    let mut predictor_builder = LayoutDetectionPredictor::builder()
        .with_config(base_config)
        .model_name(model_type);

    if let Some(ort_cfg) = ort_config {
        predictor_builder = predictor_builder.with_ort_config(ort_cfg);
    }

    let predictor = predictor_builder.build(&args.model_path)?;

    // Create output directory if needed
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

            if args.dump_json {
                let elements_json: Vec<_> = elements
                    .iter()
                    .enumerate()
                    .map(|(idx, element)| {
                        let bbox = &element.bbox;
                        json!({
                            "order": idx + 1,
                            "label": element.element_type.as_str(),
                            "score": element.score,
                            "bbox": [
                                bbox.x_min().round() as i32,
                                bbox.y_min().round() as i32,
                                bbox.x_max().round() as i32,
                                bbox.y_max().round() as i32
                            ]
                        })
                    })
                    .collect();
                let payload = json!({
                    "image": image_path,
                    "width": width,
                    "height": height,
                    "elements": elements_json
                });
                println!("{}", serde_json::to_string_pretty(&payload)?);
            }

            // Visualization
            if args.vis {
                let output_dir = args
                    .output_dir
                    .as_ref()
                    .ok_or("--output-dir is required when --vis is enabled")?;

                // Use the original filename for output
                let output_filename = image_path
                    .file_name()
                    .and_then(|s| s.to_str())
                    .unwrap_or("result.png");
                let output_path = output_dir.join(output_filename);

                // Build layout items for visualization
                let layout_items: Vec<LayoutItem> = elements
                    .iter()
                    .map(|e| LayoutItem {
                        bbox: &e.bbox,
                        element_type: &e.element_type,
                        score: e.score,
                    })
                    .collect();

                let vis_img = visualize_layout(&img_for_vis, &layout_items, 2, true);
                save_image(&vis_img, &output_path)
                    .map_err(|e| format!("Failed to save visualization: {}", e))?;
                info!("Visualization saved to: {:?}", output_path);
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
