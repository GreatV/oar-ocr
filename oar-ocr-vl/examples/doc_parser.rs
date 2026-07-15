//! Unified Document Parser Example
//!
//! This example demonstrates the unified DocParser API for external
//! layout-first document parsing (layout detection + region recognition).
//!
//! HunyuanOCR and the MinerU models are intentionally not exposed here: their
//! reference-quality usage is full-page prompt-driven parsing or a model-native
//! two-step pipeline, not forced external layout crops.
//!
//! # Usage
//!
//! ```bash
//! # Using PaddleOCR-VL model
//! cargo run -p oar-ocr-vl --features download-binaries --example doc_parser -- \
//!     --model-name paddleocr-vl \
//!     --model-dir PaddleOCR-VL \
//!     --layout-model models/pp-doclayoutv3.onnx \
//!     document.jpg
//!
//! # Using PaddleOCR-VL-1.5 model
//! cargo run -p oar-ocr-vl --features download-binaries --example doc_parser -- \
//!     --model-name paddleocr-vl-1.5 \
//!     --model-dir PaddleOCR-VL-1.5 \
//!     --layout-model models/pp-doclayoutv3.onnx \
//!     document.jpg
//!
//! # Using PaddleOCR-VL-1.6 model
//! cargo run -p oar-ocr-vl --features download-binaries --example doc_parser -- \
//!     --model-name paddleocr-vl-1.6 \
//!     --model-dir PaddleOCR-VL-1.6 \
//!     --layout-model models/pp-doclayoutv3.onnx \
//!     document.jpg
//!
//! # Using GLM-OCR model
//! cargo run -p oar-ocr-vl --features download-binaries --example doc_parser -- \
//!     --model-name glmocr \
//!     --model-dir models/GLM-OCR \
//!     --layout-model models/pp-doclayoutv3.onnx \
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
    /// PaddleOCR-VL 0.9B: VLM with task prompts
    #[value(name = "paddleocr-vl")]
    PaddleOcrVl,
    /// PaddleOCR-VL 1.5 (0.9B): spotting and seal recognition
    #[value(name = "paddleocr-vl-1.5")]
    PaddleOcrVl15,
    /// PaddleOCR-VL 1.6 (0.9B): region-aware enhanced recognition
    #[value(name = "paddleocr-vl-1.6")]
    PaddleOcrVl16,
    /// GLM-OCR: OCR expert VLM (GLM-V)
    #[value(name = "glmocr")]
    GlmOcr,
}

/// Command-line arguments
#[derive(Parser)]
#[command(name = "doc_parser")]
#[command(
    about = "Unified external-layout DocParser - supports PaddleOCR-VL, PaddleOCR-VL-1.5/1.6, and GLM-OCR"
)]
struct Args {
    /// Recognition model to use
    #[arg(short = 'n', long, value_enum, default_value = "paddleocr-vl-1.5")]
    model_name: ModelName,

    /// Path to the model directory
    #[arg(short, long)]
    model_dir: PathBuf,

    /// Path to the PP-DocLayout ONNX model file (v2/v3, required)
    #[arg(short, long)]
    layout_model: Option<PathBuf>,

    /// Paths to input document images
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Layout model name for label mapping
    #[arg(long, default_value = "pp-doclayoutv3")]
    layout_model_name: String,

    /// Device to run on: cpu, cuda, cuda:N, or metal
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
    use oar_ocr_vl::{GlmOcr, PaddleOcrVl};

    utils::init_tracing();
    let args = Args::parse();

    info!("Unified Document Parser Example");
    info!("Model: {:?}", args.model_name);

    // Verify files exist
    if !args.model_dir.exists() {
        error!("Model directory not found: {}", args.model_dir.display());
        return Err("Model directory not found".into());
    }
    let layout_model_path = args.layout_model.as_ref().ok_or_else(|| {
        error!(
            "Layout model is required for {:?}. Use the hunyuanocr/mineru examples for model-native full-page parsing.",
            args.model_name
        );
        "Layout model not provided"
    })?;
    if !layout_model_path.exists() {
        error!("Layout model not found: {}", layout_model_path.display());
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

    let device = parse_device(&args.device)?;
    info!("Device: {:?}", device);

    info!("Loading layout model...");
    let normalized_layout_name = args.layout_model_name.to_lowercase().replace('-', "_");
    let layout_config = match normalized_layout_name.as_str() {
        "pp_doclayoutv2" | "pp_doclayout_v2" => {
            Some(LayoutDetectionConfig::with_pp_doclayoutv2_defaults())
        }
        "pp_doclayoutv3" | "pp_doclayout_v3" => {
            Some(LayoutDetectionConfig::with_pp_doclayoutv3_defaults())
        }
        "pp_structurev3" | "pp_structure_v3" => {
            Some(LayoutDetectionConfig::with_pp_structurev3_defaults())
        }
        _ => None,
    };

    let mut layout_builder =
        LayoutDetectionPredictor::builder().model_name(&args.layout_model_name);
    if let Some(config) = layout_config {
        layout_builder = layout_builder.with_config(config);
    }
    let layout_predictor = layout_builder.build(layout_model_path)?;

    // Create config
    let config = DocParserConfig {
        max_tokens: args.max_tokens,
        ..DocParserConfig::default()
    };

    // Process images with the selected model
    match args.model_name {
        ModelName::PaddleOcrVl | ModelName::PaddleOcrVl15 | ModelName::PaddleOcrVl16 => {
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
        ModelName::GlmOcr => {
            info!("Loading GLM-OCR model...");
            let load_start = Instant::now();
            let model = GlmOcr::from_dir(&args.model_dir, device)?;
            info!(
                "GLM-OCR loaded in {:.2}ms",
                load_start.elapsed().as_secs_f64() * 1000.0
            );

            let parser = DocParser::with_config(&model, config);
            process_images(&parser, &layout_predictor, &existing_images, &args)?;
        }
    }
    Ok(())
}

fn process_images<B: oar_ocr_vl::RecognitionBackend>(
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
        let result = parser.parse(layout_predictor, rgb_img);
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
