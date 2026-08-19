//! NaviDC-OCR Document Recognition Example (Candle-based)
//!
//! Runs the ~1.2B StarDoc-AI NaviDC-OCR checkpoint on images with the
//! official per-task prompts: text, table (OTSL), formula (LaTeX), code,
//! layout, distorted layout, and scientific-figure table extraction. The
//! bundled `assets/` images in the model directory make handy inputs.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p oar-ocr-vl --example navidc_ocr -- [OPTIONS] <IMAGES>...
//! ```
//!
//! # Example
//!
//! ```bash
//! cargo run -p oar-ocr-vl --features cuda --example navidc_ocr -- \
//!     --model-dir StarDoc-AI/NaviDC-OCR \
//!     --device cuda:0 \
//!     --task table \
//!     StarDoc-AI/NaviDC-OCR/assets/table.png
//! ```

mod utils;

use clap::{Parser, ValueEnum};
use image::imageops;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info};

use oar_ocr_vl::utils::convert_otsl_to_html;
use oar_ocr_vl::utils::image::load_image;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::{NaviDcOcr, NaviDcTask};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Task {
    /// Plain text recognition
    Text,
    /// Table recognition; OTSL is converted to HTML unless --raw
    Table,
    /// Formula recognition (LaTeX, wrapped as $$..$$ unless --raw)
    Formula,
    /// Code snippet parsing
    Code,
    /// Full-page layout analysis (input resized to 1036x1036)
    Layout,
    /// Multi-point layout segmentation for distorted pages (1036x1036)
    LayoutDistorted,
    /// Table extraction from scientific figures (OTSL unless --raw)
    ScientificFigure,
}

impl Task {
    fn to_model(self) -> NaviDcTask {
        match self {
            Self::Text => NaviDcTask::Text,
            Self::Table => NaviDcTask::Table,
            Self::Formula => NaviDcTask::Formula,
            Self::Code => NaviDcTask::Code,
            Self::Layout => NaviDcTask::Layout,
            Self::LayoutDistorted => NaviDcTask::LayoutDistorted,
            Self::ScientificFigure => NaviDcTask::ScientificFigure,
        }
    }
}

#[derive(Parser)]
#[command(name = "navidc_ocr")]
#[command(about = "NaviDC-OCR document recognition - text, table, formula, code, and layout tasks")]
struct Args {
    /// Path to a NaviDC-OCR model directory
    #[arg(short, long, default_value = "StarDoc-AI/NaviDC-OCR")]
    model_dir: PathBuf,

    /// Paths to input images to process
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Device to run on: cpu, cuda, cuda:N, or metal (default: cpu)
    #[arg(short, long, default_value = "cpu")]
    device: String,

    /// Recognition task (default: text)
    #[arg(short, long, value_enum, default_value = "text")]
    task: Task,

    /// Free-form prompt override; skips the task prompt
    #[arg(short, long)]
    prompt: Option<String>,

    /// Maximum number of tokens to generate (default: 4096)
    #[arg(long, default_value = "4096")]
    max_tokens: usize,

    /// Print raw model output without OTSL/formula post-processing
    #[arg(long, default_value_t = false)]
    raw: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    utils::init_tracing();
    let args = Args::parse();

    if !args.model_dir.exists() {
        error!("Model directory not found: {}", args.model_dir.display());
        return Err("Model directory not found".into());
    }

    let existing_images: Vec<PathBuf> = args
        .images
        .into_iter()
        .filter(|path| {
            if path.exists() {
                true
            } else {
                error!("Image file not found: {}", path.display());
                false
            }
        })
        .collect();
    if existing_images.is_empty() {
        return Err("No valid image files found".into());
    }

    let task = args.task.to_model();
    let prompt: String = args
        .prompt
        .clone()
        .unwrap_or_else(|| task.prompt().to_string());

    let device = parse_device(&args.device)?;
    info!("Using device: {:?}", device);

    info!(
        "Loading NaviDC-OCR model from: {}",
        args.model_dir.display()
    );
    let load_start = Instant::now();
    let model = NaviDcOcr::from_dir(&args.model_dir, device)?;
    info!(
        "Model loaded in {:.2}ms",
        load_start.elapsed().as_secs_f64() * 1000.0
    );

    info!(
        "\n=== Processing {} images (task prompt: {:?}) ===",
        existing_images.len(),
        prompt
    );
    for image_path in &existing_images {
        info!("\nProcessing: {}", image_path.display());
        let rgb_img = match load_image(image_path) {
            Ok(img) => img,
            Err(e) => {
                error!("  Failed to load image: {}", e);
                continue;
            }
        };

        // The model card's quickstart resizes layout inputs to 1036x1036
        // (bicubic) before the layout prompts.
        let model_input = match task.resize_square() {
            Some(side) => imageops::resize(&rgb_img, side, side, imageops::FilterType::CatmullRom),
            None => rgb_img,
        };

        let infer_start = Instant::now();
        let tokens = match model
            .generate_tokens(&[model_input], &[prompt.as_str()], args.max_tokens)?
            .into_iter()
            .next()
        {
            Some(tokens) => tokens,
            None => {
                error!("  Inference returned no result");
                continue;
            }
        };
        info!(
            "  Inference time: {:.2}ms, tokens: {}, fingerprint: {:016x}",
            infer_start.elapsed().as_secs_f64() * 1000.0,
            tokens.len(),
            utils::token_fingerprint(&tokens)
        );

        let raw = model.decode_tokens(&tokens)?;
        let output = if args.raw {
            raw.trim().to_string()
        } else if task.outputs_otsl() {
            convert_otsl_to_html(raw.trim())
        } else {
            task.postprocess(&raw)
        };
        println!(
            "\n--- Output ({}) ---\n{}\n--- End ---",
            image_path.display(),
            output
        );
    }

    Ok(())
}
