//! High-level OCR Pipeline Example
//!
//! This example shows how to run the end-to-end OCR pipeline using `OAROCRBuilder`.
//! It wires together text detection and recognition with optional components such as
//! document orientation correction and rectification, then prints recognized text and
//! bounding boxes for each input image.
//!
//! # Usage
//!
//! ```bash
//! cargo run --example ocr -- [OPTIONS] <IMAGES>...
//! ```
//!
//! # Arguments
//!
//! * `--det-model` - Path to the text detection model file
//! * `--rec-model` - Path to the text recognition model file
//! * `--dict-path` - Path to the character dictionary file
//! * `--document-image-orientation-model` - Optional document orientation classification model
//! * `--text-line-orientation-model` - Optional text line orientation classification model
//! * `--rectification-model` - Optional document rectification model
//! * `--text-type` - Text type hint (`seal` for curved seal text)
//! * `--return-word-box` - Enable word-level boxes from recognition output
//! * `--device` - Device to use (`cpu`, `cuda`, `cuda:0`, etc.)
//! * Detection config: `--det-score-thresh`, `--det-box-thresh`, `--det-unclip`, `--det-max-candidates`
//! * Recognition config: `--rec-score-thresh`, `--rec-max-text-length`
//! * Batch sizes: `--image-batch-size` (detection sessions), `--region-batch-size` (recognition)
//! * `-o, --output-dir` - Directory to save visualizations (requires `visualization` feature)
//! * `--vis-font-path` - Optional font for visualization (set for non-Latin languages like Chinese)
//! * `<IMAGES>...` - One or more document images to process
//!
//! # Example
//!
//! ```bash
//! cargo run --example ocr -- \
//!   --det-model models/ppocrv4_mobile_det.onnx \
//!   --rec-model models/ppocrv4_mobile_rec.onnx \
//!   --dict-path models/ppocr_keys_v1.txt \
//!   document1.jpg document2.jpg
//! ```

mod utils;

use clap::Parser;
use oar_ocr::domain::tasks::{TextDetectionConfig, TextRecognitionConfig};
use oar_ocr::oarocr::OAROCRBuilder;
use oar_ocr::processors::LimitType;
use oar_ocr::utils::load_image;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info, warn};
use utils::parse_device_config;

#[cfg(feature = "visualization")]
use oar_ocr::utils::visualization::{VisualizationConfig, create_ocr_visualization};

/// Command-line arguments for the OCR pipeline example
#[derive(Parser)]
#[command(name = "ocr")]
#[command(about = "Run the high-level OCR pipeline (detection + recognition)")]
struct Args {
    /// Path to the text detection model file
    #[arg(long = "det-model")]
    det_model: PathBuf,

    /// Path to the text recognition model file
    #[arg(long = "rec-model")]
    rec_model: PathBuf,

    /// Path to the character dictionary file
    #[arg(long = "dict-path")]
    dict_path: PathBuf,

    /// Input images to process
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Optional document image orientation classification model
    #[arg(long = "document-image-orientation-model")]
    document_image_orientation_model: Option<PathBuf>,

    /// Optional text line orientation classification model
    #[arg(long = "text-line-orientation-model")]
    text_line_orientation_model: Option<PathBuf>,

    /// Optional document rectification model
    #[arg(long)]
    rectification_model: Option<PathBuf>,

    /// Enable word-level bounding boxes derived from recognition output
    #[arg(long, default_value_t = false)]
    return_word_box: bool,

    /// Device to use for inference (cpu, cuda, cuda:0, etc.)
    #[arg(long, default_value = "cpu")]
    device: String,

    /// Text detection score threshold (default: 0.3)
    #[arg(long, default_value_t = 0.3)]
    det_score_thresh: f32,

    /// Text detection box threshold (default: 0.6)
    #[arg(long, default_value_t = 0.6)]
    det_box_thresh: f32,

    /// Text detection unclip ratio (default: 1.5)
    #[arg(long, default_value_t = 1.5)]
    det_unclip: f32,

    /// Maximum text detection candidates (default: 1000)
    #[arg(long, default_value_t = 1000)]
    det_max_candidates: usize,

    /// Text detection limit side length (default: 960 for general, 736 for seal)
    #[arg(long)]
    det_limit_side_len: Option<u32>,

    /// Text detection limit type (min/max/resize_long)
    #[arg(long)]
    det_limit_type: Option<String>,

    /// Text detection max side length (default: 4000)
    #[arg(long)]
    det_max_side_len: Option<u32>,

    /// Text recognition score threshold (default: 0.0)
    #[arg(long, default_value_t = 0.0)]
    rec_score_thresh: f32,

    /// Maximum text length for recognition (default: 100)
    #[arg(long, default_value_t = 100)]
    rec_max_text_length: usize,

    /// Detection session pool size (sets detection batch concurrency)
    #[arg(long)]
    image_batch_size: Option<usize>,

    /// Recognition session pool size (batching cropped regions)
    #[arg(long)]
    region_batch_size: Option<usize>,

    /// Directory to save visualization images (requires `visualization` feature)
    #[cfg(feature = "visualization")]
    #[arg(short = 'o', long = "output-dir")]
    output_dir: Option<PathBuf>,

    /// Custom font path for visualization (useful for Chinese or other non-Latin text)
    #[cfg(feature = "visualization")]
    #[arg(long = "vis-font-path")]
    vis_font_path: Option<PathBuf>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Initialize tracing for logging
    utils::init_tracing();

    // Parse CLI arguments
    let args = Args::parse();

    info!("Running OCR pipeline example");

    // Validate required files
    if !args.det_model.exists() {
        error!("Detection model not found: {}", args.det_model.display());
        return Err("Detection model not found".into());
    }
    if !args.rec_model.exists() {
        error!("Recognition model not found: {}", args.rec_model.display());
        return Err("Recognition model not found".into());
    }
    if !args.dict_path.exists() {
        error!("Dictionary file not found: {}", args.dict_path.display());
        return Err("Dictionary file not found".into());
    }

    // Filter out missing images
    let existing_images: Vec<PathBuf> = args
        .images
        .iter()
        .filter(|path| {
            let exists = path.exists();
            if !exists {
                error!("Image not found: {}", path.display());
            }
            exists
        })
        .cloned()
        .collect();

    if existing_images.is_empty() {
        return Err("No valid input images provided".into());
    }

    // Build device/ORT configuration
    let ort_config = parse_device_config(&args.device)?;

    // Prepare model configs
    let det_config = TextDetectionConfig {
        score_threshold: args.det_score_thresh,
        box_threshold: args.det_box_thresh,
        unclip_ratio: args.det_unclip,
        max_candidates: args.det_max_candidates,
        limit_side_len: args.det_limit_side_len,
        limit_type: args
            .det_limit_type
            .as_deref()
            .map(|s| match s.to_lowercase().as_str() {
                "min" => Ok(LimitType::Min),
                "max" => Ok(LimitType::Max),
                "resize_long" | "resizelong" | "resize-long" => Ok(LimitType::ResizeLong),
                other => Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!("Invalid --det-limit-type value: '{}'", other),
                )),
            })
            .transpose()?,
        max_side_len: args.det_max_side_len,
    };

    let rec_config = TextRecognitionConfig {
        score_threshold: args.rec_score_thresh,
        max_text_length: args.rec_max_text_length,
    };

    // Construct OCR pipeline
    let mut builder = OAROCRBuilder::new(&args.det_model, &args.rec_model, &args.dict_path)
        .text_detection_config(det_config)
        .text_recognition_config(rec_config)
        .return_word_box(args.return_word_box);

    if let Some(config) = ort_config.clone() {
        builder = builder.ort_session(config);
    }

    if let Some(size) = args.image_batch_size {
        builder = builder.image_batch_size(size);
    }

    if let Some(size) = args.region_batch_size {
        builder = builder.region_batch_size(size);
    }

    let build_start = Instant::now();
    let ocr = builder.build()?;
    info!(
        "OCR pipeline built in {:.2}ms",
        build_start.elapsed().as_secs_f64() * 1000.0
    );

    // Load images
    let mut images = Vec::new();
    for path in &existing_images {
        match load_image(path) {
            Ok(img) => {
                info!(
                    "Loaded image {} ({}x{})",
                    path.display(),
                    img.width(),
                    img.height()
                );
                images.push(img);
            }
            Err(err) => warn!("Failed to load {}: {}", path.display(), err),
        }
    }

    if images.is_empty() {
        return Err("No images could be loaded".into());
    }

    // Run inference
    let start = Instant::now();
    let results = ocr.predict(images)?;
    info!(
        "OCR completed in {:.2}ms",
        start.elapsed().as_secs_f64() * 1000.0
    );

    // Report results
    info!("\n=== OCR Results ===");
    for (idx, (path, result)) in existing_images.iter().zip(results.iter()).enumerate() {
        info!("\nImage {}: {}", idx + 1, path.display());
        if let Some(angle) = result.orientation_angle {
            info!("  Overall image orientation: {} degrees", angle);
        }
        info!("  {} text regions", result.text_regions.len());

        for (region_idx, region) in result.text_regions.iter().enumerate() {
            let bbox = &region.bounding_box;
            let text = region
                .text
                .as_ref()
                .map(|t| t.to_string())
                .unwrap_or_else(|| "<no text>".to_string());
            let score = region.confidence.unwrap_or(0.0) * 100.0;
            let line_orientation = region
                .orientation_angle
                .map_or("N/A".to_string(), |a| format!("{:.1}°", a));

            info!(
                "  [{}] \"{}\" ({:.1}%) at [{:.1},{:.1}] - [{:.1},{:.1}] (Line Orientation: {})",
                region_idx + 1,
                text,
                score,
                bbox.x_min(),
                bbox.y_min(),
                bbox.x_max(),
                bbox.y_max(),
                line_orientation
            );

            if let Some(word_boxes) = &region.word_boxes {
                info!("    {} word boxes", word_boxes.len());
                if let Some(full_text) = region.text.as_ref() {
                    let chars: Vec<char> = full_text.chars().collect();
                    for (i, word_bbox) in word_boxes.iter().enumerate() {
                        if let Some(char_content) = chars.get(i) {
                            info!(
                                "      Word Box {}: '{}' at [{:.1},{:.1}] - [{:.1},{:.1}]",
                                i + 1,
                                char_content,
                                word_bbox.x_min(),
                                word_bbox.y_min(),
                                word_bbox.x_max(),
                                word_bbox.y_max()
                            );
                        }
                    }
                }
            }
        }
    }

    // Save visualizations when enabled
    #[cfg(feature = "visualization")]
    if let Some(output_dir) = args.output_dir {
        std::fs::create_dir_all(&output_dir)?;
        let vis_config = if let Some(font_path) = args.vis_font_path {
            VisualizationConfig::with_font_path(&font_path).unwrap_or_else(|err| {
                warn!(
                    "Failed to load font {}: {}. Falling back to system font.",
                    font_path.display(),
                    err
                );
                VisualizationConfig::with_system_font()
            })
        } else {
            VisualizationConfig::with_system_font()
        };

        for (path, result) in existing_images.iter().zip(results.iter()) {
            let vis_img = create_ocr_visualization(result, &vis_config)?;
            let filename = path
                .file_name()
                .and_then(|s| s.to_str())
                .map(|s| s.to_string())
                .unwrap_or_else(|| "visualization.jpg".to_string());
            let output_path = output_dir.join(filename);
            vis_img.save(&output_path)?;
            info!("Saved visualization to {}", output_path.display());
        }
    }

    Ok(())
}
