//! Xiaomi-OCR-0 OCR and document parsing example (Candle-based).
//!
//! Runs the 0.8B SeerRay-Lab/Xiaomi-OCR-0 checkpoint with the official task
//! prompts: whole-page document parsing (Markdown; OTSL tables are converted
//! to HTML unless `--raw`), plus text, table, formula, and
//! key-information-extraction region tasks.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p oar-ocr-vl --example xiaomi_ocr -- [OPTIONS] <IMAGES>...
//! ```
//!
//! # Example
//!
//! ```bash
//! cargo run -p oar-ocr-vl --features cuda --example xiaomi_ocr -- \
//!     --model-dir SeerRay-Lab/Xiaomi-OCR-0 \
//!     --device cuda:0 \
//!     --task table \
//!     .oar/images/table3_crop.jpg
//! ```

mod utils;

use clap::{Parser, ValueEnum};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info};

use oar_ocr_vl::utils::convert_otsl_to_html;
use oar_ocr_vl::utils::image::load_image;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::xiaomi_ocr::{
    DEFAULT_MAX_NEW_TOKENS, DEFAULT_PROMPT, FORMULA_REGION_PROMPT, TABLE_REGION_PROMPT,
    TEXT_REGION_PROMPT, key_information_extraction_prompt,
};
use oar_ocr_vl::xiaomi_ocr::{XiaomiOcr, finalize_markdown};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Task {
    /// Whole-page parsing to Markdown (official post-processing unless --raw)
    Document,
    /// Text-region recognition
    Text,
    /// Table-region recognition; OTSL is converted to HTML unless --raw
    Table,
    /// Formula recognition (LaTeX)
    Formula,
    /// Key information extraction (JSON; supply --schema)
    Kie,
}

impl Task {
    fn instruction(self, schema: Option<&str>) -> Result<String, String> {
        match self {
            Self::Document => Ok(DEFAULT_PROMPT.to_string()),
            Self::Text => Ok(TEXT_REGION_PROMPT.to_string()),
            Self::Table => Ok(TABLE_REGION_PROMPT.to_string()),
            Self::Formula => Ok(FORMULA_REGION_PROMPT.to_string()),
            Self::Kie => {
                let schema = schema.ok_or("--task kie requires --schema (inline JSON or @file)")?;
                Ok(key_information_extraction_prompt(schema))
            }
        }
    }
}

#[derive(Parser)]
#[command(name = "xiaomi_ocr")]
#[command(about = "Xiaomi-OCR-0 full-page parsing and text/table/formula/KIE region tasks")]
struct Args {
    /// Path to the Xiaomi-OCR-0 model directory
    #[arg(short, long)]
    model_dir: PathBuf,

    /// Paths to one or more page or region images
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Device to run on: auto, cpu, cuda, cuda:N, or metal
    #[arg(short, long, default_value = "auto")]
    device: String,

    /// Task prompt to run (default: document)
    #[arg(short, long, value_enum, default_value = "document")]
    task: Task,

    /// Key-information-extraction JSON schema: inline JSON or @path/to/file
    #[arg(long)]
    schema: Option<String>,

    /// Maximum number of tokens to generate (default: 4096)
    #[arg(long, default_value_t = DEFAULT_MAX_NEW_TOKENS)]
    max_tokens: usize,

    /// Print the cleaned model output without OTSL→HTML / finalization
    #[arg(long, default_value_t = false)]
    raw: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    utils::init_tracing();
    let args = Args::parse();
    let mut had_errors = false;

    if !args.model_dir.exists() {
        error!("Model directory not found: {}", args.model_dir.display());
        return Err("Model directory not found".into());
    }

    let schema = match &args.schema {
        Some(schema) if schema.starts_with('@') => std::fs::read_to_string(&schema[1..])
            .map_err(|err| format!("failed to read --schema file {}: {err}", &schema[1..]))?,
        Some(schema) => schema.clone(),
        None => String::new(),
    };
    let instruction = args
        .task
        .instruction(Some(schema.as_str()).filter(|schema| !schema.is_empty()))
        .map_err(|message| -> Box<dyn std::error::Error> { message.into() })?;

    let mut image_paths = Vec::new();
    let mut images = Vec::new();
    for image_path in args.images {
        if !image_path.exists() {
            error!("Image file not found: {}", image_path.display());
            had_errors = true;
            continue;
        }
        match load_image(&image_path) {
            Ok(image) => {
                image_paths.push(image_path);
                images.push(image);
            }
            Err(err) => {
                error!("Failed to load {}: {err}", image_path.display());
                had_errors = true;
            }
        }
    }
    if images.is_empty() {
        return Err("No valid image files found".into());
    }

    let device = parse_device(&args.device)?;
    info!("Using device: {:?}", device);

    info!(
        "Loading Xiaomi-OCR-0 model from: {}",
        args.model_dir.display()
    );
    let load_start = Instant::now();
    let model = XiaomiOcr::from_dir(&args.model_dir, device)?;
    info!(
        "Model loaded in {:.2}ms",
        load_start.elapsed().as_secs_f64() * 1000.0
    );

    let page_count = images.len();
    info!("Processing {page_count} image(s) (task: {:?})", args.task);
    let infer_start = Instant::now();
    let mut results = Vec::with_capacity(page_count);
    for image in &images {
        results.push(model.generate_tokens_with_prompt(image, &instruction, args.max_tokens));
    }
    let infer_ms = infer_start.elapsed().as_secs_f64() * 1000.0;
    info!(
        "Inference completed for {page_count} image(s) in {infer_ms:.2}ms ({:.2}ms/image)",
        infer_ms / page_count as f64
    );

    for (index, (image_path, result)) in image_paths.iter().zip(results).enumerate() {
        let tokens = match result {
            Ok(tokens) => tokens,
            Err(err) => {
                error!("Inference failed for {}: {err}", image_path.display());
                had_errors = true;
                continue;
            }
        };
        info!(
            "  tokens: {}, fingerprint: {:016x}",
            tokens.len(),
            utils::token_fingerprint(&tokens)
        );
        // decode_tokens keeps the official truncated-tail cleanup; --raw
        // stops there, otherwise the document/table tasks apply their
        // OTSL→HTML conversions.
        let raw = model.decode_tokens(&tokens)?;
        let output = if args.raw {
            raw
        } else {
            match args.task {
                Task::Document => finalize_markdown(&raw),
                Task::Table => convert_otsl_to_html(&raw),
                Task::Text | Task::Formula | Task::Kie => raw,
            }
        };
        if index > 0 {
            println!();
        }
        println!("--- Output ({}) ---", image_path.display());
        println!("{output}");
        println!("--- End ---");
    }

    if had_errors {
        Err("One or more Xiaomi-OCR-0 images failed".into())
    } else {
        Ok(())
    }
}
