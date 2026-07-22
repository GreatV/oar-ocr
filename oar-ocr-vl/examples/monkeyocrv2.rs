//! MonkeyOCRv2-S/B-Parsing inference example (Candle-based).
//!
//! ```bash
//! cargo run --release -p oar-ocr-vl --features cuda,download-binaries \
//!   --example monkeyocrv2 -- \
//!   --model-dir "$MONKEYOCR_MODEL_DIR" \
//!   --device cuda:0 --task end-to-end document.jpg
//! ```

mod utils;

use clap::{Parser, ValueEnum};
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::monkeyocrv2::DEFAULT_MAX_NEW_TOKENS;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::{MonkeyOcrV2, MonkeyOcrV2Task};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info};

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Task {
    Layout,
    EndToEnd,
    Text,
    Formula,
    Table,
}

impl From<Task> for MonkeyOcrV2Task {
    fn from(task: Task) -> Self {
        match task {
            Task::Layout => Self::Layout,
            Task::EndToEnd => Self::EndToEnd,
            Task::Text => Self::Text,
            Task::Formula => Self::Formula,
            Task::Table => Self::Table,
        }
    }
}

#[derive(Parser)]
#[command(name = "monkeyocrv2")]
#[command(about = "MonkeyOCRv2-S/B-Parsing native Candle inference")]
struct Args {
    /// Path to a MonkeyOCRv2-S-Parsing or MonkeyOCRv2-B-Parsing model directory
    #[arg(short, long)]
    model_dir: PathBuf,

    /// Paths to one or more input images
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Official inference task
    #[arg(short, long, value_enum, default_value = "end-to-end")]
    task: Task,

    /// Device: cpu, cuda, cuda:N, or metal
    #[arg(short, long, default_value = "cpu")]
    device: String,

    /// Maximum generated tokens per image
    #[arg(long, default_value_t = DEFAULT_MAX_NEW_TOKENS)]
    max_tokens: usize,

    /// Override the official task prompt
    #[arg(long)]
    prompt: Option<String>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    utils::init_tracing();
    let args = Args::parse();
    if !args.model_dir.exists() {
        return Err(format!("Model directory not found: {}", args.model_dir.display()).into());
    }

    let mut paths = Vec::new();
    let mut images = Vec::new();
    for path in args.images {
        if !path.exists() {
            error!("Image not found: {}", path.display());
            continue;
        }
        match load_image(&path) {
            Ok(image) => {
                paths.push(path);
                images.push(image);
            }
            Err(err) => error!("Failed to load {}: {err}", path.display()),
        }
    }
    if images.is_empty() {
        return Err("No valid input images".into());
    }

    let device = parse_device(&args.device)?;
    info!("Loading {}", args.model_dir.display());
    let started = Instant::now();
    let model = MonkeyOcrV2::from_dir(&args.model_dir, device)?;
    info!("Model loaded in {:.2}s", started.elapsed().as_secs_f64());

    let started = Instant::now();
    let results = if let Some(prompt) = args.prompt {
        let prompts = vec![prompt; images.len()];
        model.generate_with_prompts(&images, &prompts, args.max_tokens)
    } else {
        let tasks = vec![MonkeyOcrV2Task::from(args.task); images.len()];
        model.generate(&images, &tasks, args.max_tokens)
    };
    info!(
        "Processed {} image(s) in {:.2}s",
        images.len(),
        started.elapsed().as_secs_f64()
    );

    let mut failed = false;
    for (path, result) in paths.iter().zip(results) {
        match result {
            Ok(text) => {
                println!("<!-- MonkeyOCRv2 source: {} -->", path.display());
                println!("{text}");
            }
            Err(err) => {
                error!("Inference failed for {}: {err}", path.display());
                failed = true;
            }
        }
    }
    if failed {
        Err("One or more MonkeyOCRv2 inferences failed".into())
    } else {
        Ok(())
    }
}
