//! Unified Document Parser Example
//!
//! This example demonstrates the unified DocParser API that supports multiple
//! recognition models for two-stage document parsing (layout detection + recognition).
//!
//! # Usage
//!
//! ```bash
//! # Using UniRec model (default, lighter)
//! cargo run --features vl --example doc_parser -- \
//!     --model-name unirec \
//!     --model-dir models/unirec-0.1b \
//!     --layout-model models/pp-doclayoutv2.onnx \
//!     document.jpg
//!
//! # Using PaddleOCR-VL model (heavier, more accurate)
//! cargo run --features vl --example doc_parser -- \
//!     --model-name paddleocr-vl \
//!     --model-dir PaddleOCR-VL \
//!     --layout-model models/pp-doclayoutv2.onnx \
//!     document.jpg
//! ```

#[cfg(feature = "vl")]
mod utils;

use clap::{Parser, ValueEnum};
use std::path::PathBuf;

#[cfg(feature = "vl")]
use std::time::Instant;
#[cfg(feature = "vl")]
use tracing::{error, info};

#[cfg(feature = "vl")]
use oar_ocr::domain::LayoutDetectionConfig;
#[cfg(feature = "vl")]
use oar_ocr::predictors::LayoutDetectionPredictor;
#[cfg(feature = "vl")]
use oar_ocr::utils::load_image;
#[cfg(feature = "vl")]
use oar_ocr::vl::{DocParser, DocParserConfig};
#[cfg(feature = "vl")]
use utils::candle_device::parse_candle_device;

/// Recognition model type
#[derive(Debug, Clone, Copy, ValueEnum)]
enum ModelName {
    /// UniRec: Lightweight unified recognition
    Unirec,
    /// PaddleOCR-VL: Large VLM with task prompts
    #[value(name = "paddleocr-vl")]
    PaddleOcrVl,
}

/// Command-line arguments
#[derive(Parser)]
#[command(name = "doc_parser")]
#[command(about = "Unified Document Parser - supports UniRec and PaddleOCR-VL models")]
struct Args {
    /// Recognition model to use
    #[arg(short = 'n', long, value_enum, default_value = "unirec")]
    model_name: ModelName,

    /// Path to the model directory
    #[arg(short, long)]
    model_dir: PathBuf,

    /// Path to the PP-DocLayoutV2 ONNX model file
    #[arg(short, long)]
    layout_model: PathBuf,

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

#[cfg(not(feature = "vl"))]
fn main() {
    eprintln!("This example requires the 'vl' feature.");
    eprintln!("Run with: cargo run --features vl --example doc_parser -- ...");
    std::process::exit(1);
}

#[cfg(feature = "vl")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    use oar_ocr::vl::{PaddleOcrVl, UniRec};

    utils::init_tracing();
    let args = Args::parse();

    info!("Unified Document Parser Example");
    info!("Model: {:?}", args.model_name);

    // Verify files exist
    if !args.model_dir.exists() {
        error!("Model directory not found: {}", args.model_dir.display());
        return Err("Model directory not found".into());
    }
    if !args.layout_model.exists() {
        error!("Layout model not found: {}", args.layout_model.display());
        return Err("Layout model not found".into());
    }

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

    let device = parse_candle_device(&args.device)?;
    info!("Device: {:?}", device);

    // Load layout model
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
    let layout_predictor = layout_builder.build(&args.layout_model)?;

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
            process_images(&parser, &layout_predictor, &existing_images, &args)?;
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
            process_images(&parser, &layout_predictor, &existing_images, &args)?;
        }
    }
    Ok(())
}

#[cfg(feature = "vl")]
fn process_images<B: oar_ocr::vl::RecognitionBackend>(
    parser: &DocParser<B>,
    layout_predictor: &LayoutDetectionPredictor,
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
        match parser.parse(layout_predictor, rgb_img) {
            Ok(result) => {
                info!("  Parsed in {:.2}s", start.elapsed().as_secs_f64());
                info!("  Elements: {}", result.layout_elements.len());

                // Get markdown (OpenOCR-compatible) from the parsed result.
                let markdown = oar_ocr::vl::utils::to_markdown_openocr(
                    &result.layout_elements,
                    &ignore_labels,
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
                                if t.len() > 40 {
                                    format!("{}...", &t[..40])
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
