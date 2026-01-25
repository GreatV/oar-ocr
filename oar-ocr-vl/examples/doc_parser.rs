//! Unified Document Parser Example
//!
//! This example demonstrates the unified DocParser API that supports multiple
//! recognition models for two-stage document parsing (layout detection + recognition).
//!
//! # Usage
//!
//! ```bash
//! # Using UniRec model (default, lighter)
//! cargo run -p oar-ocr-vl --example doc_parser -- \
//!     --model-name unirec \
//!     --model-dir models/unirec-0.1b \
//!     --layout-model models/pp-doclayoutv2.onnx \
//!     document.jpg
//!
//! # Using PaddleOCR-VL model (heavier, more accurate)
//! cargo run -p oar-ocr-vl --example doc_parser -- \
//!     --model-name paddleocr-vl \
//!     --model-dir PaddleOCR-VL \
//!     --layout-model models/pp-doclayoutv2.onnx \
//!     document.jpg
//!
//! # Using LightOnOCR model (end-to-end OCR)
//! cargo run -p oar-ocr-vl --example doc_parser -- \
//!     --model-name lightonocr \
//!     --model-dir LightOnOCR-2-1B \
//!     document.jpg
//! ```

mod utils;

use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use std::time::Instant;

use tracing::{error, info};

use oar_ocr_core::domain::LayoutDetectionConfig;
use oar_ocr_core::predictors::LayoutDetectionPredictor;
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::{DocParser, DocParserConfig};

/// Recognition model type
#[derive(Debug, Clone, Copy, ValueEnum)]
enum ModelName {
    /// UniRec: Lightweight unified recognition
    Unirec,
    /// PaddleOCR-VL: Large VLM with task prompts
    #[value(name = "paddleocr-vl")]
    PaddleOcrVl,
    /// HunyuanOCR: OCR expert VLM (HunYuanVL)
    #[value(name = "hunyuanocr")]
    HunyuanOcr,
    /// LightOnOCR: End-to-end OCR VLM
    #[value(name = "lightonocr")]
    LightOnOcr,
}

/// Command-line arguments
#[derive(Parser)]
#[command(name = "doc_parser")]
#[command(
    about = "Unified Document Parser - supports UniRec, PaddleOCR-VL, HunyuanOCR, and LightOnOCR models"
)]
struct Args {
    /// Recognition model to use
    #[arg(short = 'n', long, value_enum, default_value = "unirec")]
    model_name: ModelName,

    /// Path to the model directory
    #[arg(short, long)]
    model_dir: PathBuf,

    /// Path to the PP-DocLayoutV2 ONNX model file (required unless using lightonocr)
    #[arg(short, long)]
    layout_model: Option<PathBuf>,

    /// Paths to input document images
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Layout model name for label mapping
    #[arg(long, default_value = "pp-doclayoutv2")]
    layout_model_name: String,

    /// Device to run on: cpu, cuda, or cuda:N
    #[arg(short, long, default_value = "cpu")]
    device: String,

    /// Directory to save markdown output
    #[arg(short, long)]
    output_dir: Option<PathBuf>,

    /// Maximum tokens to generate per region
    #[arg(long, default_value = "4096")]
    max_tokens: usize,

    /// Enable verbose output
    #[arg(short, long)]
    verbose: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    use oar_ocr_vl::{HunyuanOcr, LightOnOcr, PaddleOcrVl, UniRec};

    utils::init_tracing();
    let args = Args::parse();

    info!("Unified Document Parser Example");
    info!("Model: {:?}", args.model_name);

    // Verify files exist
    if !args.model_dir.exists() {
        error!("Model directory not found: {}", args.model_dir.display());
        return Err("Model directory not found".into());
    }
    let needs_layout = !matches!(args.model_name, ModelName::LightOnOcr);
    let layout_model_path = if needs_layout {
        let path = args.layout_model.as_ref().ok_or_else(|| {
            error!(
                "Layout model is required for {:?} (not needed for LightOnOcr)",
                args.model_name
            );
            "Layout model not provided"
        })?;
        if !path.exists() {
            error!("Layout model not found: {}", path.display());
            return Err("Layout model not found".into());
        }
        Some(path)
    } else {
        None
    };

    // Filter valid images
    let existing_images: Vec<PathBuf> =
        args.images.iter().filter(|p| p.exists()).cloned().collect();

    if existing_images.is_empty() {
        return Err("No valid image files found".into());
    }

    // Create output directory if needed
    if let Some(ref dir) = args.output_dir {
        std::fs::create_dir_all(dir)?;
    }

    let device = parse_device(&args.device)?;
    info!("Device: {:?}", device);

    let layout_predictor = if let Some(layout_path) = layout_model_path {
        info!("Loading layout model...");
        let normalized_layout_name = args.layout_model_name.to_lowercase().replace('-', "_");
        let layout_config = match normalized_layout_name.as_str() {
            "pp_doclayoutv2" | "pp_doclayout_v2" => {
                Some(LayoutDetectionConfig::with_pp_doclayoutv2_defaults())
            }
            _ => None,
        };

        let mut layout_builder =
            LayoutDetectionPredictor::builder().model_name(&args.layout_model_name);
        if let Some(config) = layout_config {
            layout_builder = layout_builder.with_config(config);
        }
        Some(layout_builder.build(layout_path)?)
    } else {
        None
    };

    // Create config
    let config = DocParserConfig {
        max_tokens: args.max_tokens,
        ..DocParserConfig::default()
    };

    // Process images with the selected model
    match args.model_name {
        ModelName::Unirec => {
            info!("Loading UniRec model...");
            let load_start = Instant::now();
            let unirec = UniRec::from_dir(&args.model_dir, device)?;
            info!(
                "UniRec loaded in {:.2}ms",
                load_start.elapsed().as_secs_f64() * 1000.0
            );

            let parser = DocParser::with_config(&unirec, config);
            process_images(&parser, layout_predictor.as_ref(), &existing_images, &args)?;
        }
        ModelName::PaddleOcrVl => {
            info!("Loading PaddleOCR-VL model...");
            let load_start = Instant::now();
            let vl = PaddleOcrVl::from_dir(&args.model_dir, device)?;
            info!(
                "PaddleOCR-VL loaded in {:.2}ms",
                load_start.elapsed().as_secs_f64() * 1000.0
            );

            let parser = DocParser::with_config(&vl, config);
            process_images(&parser, layout_predictor.as_ref(), &existing_images, &args)?;
        }
        ModelName::HunyuanOcr => {
            info!("Loading HunyuanOCR model...");
            let load_start = Instant::now();
            let model = HunyuanOcr::from_dir(&args.model_dir, device)?;
            info!(
                "HunyuanOCR loaded in {:.2}ms",
                load_start.elapsed().as_secs_f64() * 1000.0
            );

            let parser = DocParser::with_config(&model, config);
            process_images(&parser, layout_predictor.as_ref(), &existing_images, &args)?;
        }
        ModelName::LightOnOcr => {
            info!("Loading LightOnOCR model...");
            let load_start = Instant::now();
            let model = LightOnOcr::from_dir(&args.model_dir, device)?;
            info!(
                "LightOnOCR loaded in {:.2}ms",
                load_start.elapsed().as_secs_f64() * 1000.0
            );

            let parser = DocParser::with_config(&model, config);
            process_images(&parser, layout_predictor.as_ref(), &existing_images, &args)?;
        }
    }
    Ok(())
}

fn process_images<B: oar_ocr_vl::RecognitionBackend>(
    parser: &DocParser<B>,
    layout_predictor: Option<&LayoutDetectionPredictor>,
    images: &[PathBuf],
    args: &Args,
) -> Result<(), Box<dyn std::error::Error>> {
    info!("\n=== Processing {} images ===", images.len());
    let ignore_labels = &parser.config().markdown_ignore_labels;

    for image_path in images {
        info!("\nProcessing: {}", image_path.display());

        let rgb_img = match load_image(image_path) {
            Ok(img) => {
                if args.verbose {
                    info!("  Image size: {}x{}", img.width(), img.height());
                }
                img
            }
            Err(e) => {
                error!("  Failed to load: {}", e);
                continue;
            }
        };

        let start = Instant::now();
        let result = match layout_predictor {
            Some(predictor) => parser.parse(predictor, rgb_img),
            None => parser.parse_without_layout(rgb_img),
        };
        match result {
            Ok(result) => {
                info!("  Parsed in {:.2}s", start.elapsed().as_secs_f64());
                info!("  Elements: {}", result.layout_elements.len());

                // Get markdown (OpenOCR-compatible) from the parsed result.
                let markdown = oar_ocr_vl::utils::to_markdown_openocr(
                    &result.layout_elements,
                    ignore_labels,
                    true,
                );

                // Save or print
                if let Some(ref dir) = args.output_dir {
                    let name = image_path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("out");
                    let path = dir.join(format!("{}.md", name));
                    std::fs::write(&path, &markdown)?;
                    info!("  Saved: {}", path.display());
                } else {
                    println!("\n--- Markdown ---\n{}\n--- End ---", markdown);
                }

                if args.verbose {
                    for (i, el) in result.layout_elements.iter().enumerate() {
                        let preview = el
                            .text
                            .as_ref()
                            .map(|t| {
                                if t.chars().count() > 40 {
                                    format!("{}...", t.chars().take(40).collect::<String>())
                                } else {
                                    t.clone()
                                }
                            })
                            .unwrap_or_default();
                        info!("  [{}] {:?}: {}", i, el.element_type, preview);
                    }
                }
            }
            Err(e) => error!("  Failed: {}", e),
        }
    }

    Ok(())
}
