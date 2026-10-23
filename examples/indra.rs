//! Indra OCR Example
//!
//! Reads page images with Datalab's Indra OCR models: lines with boxes,
//! layout blocks in reading order, and Markdown with tables as HTML.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --example indra -- --model-dir <DIR> [OPTIONS] <IMAGES>...
//! ```
//!
//! # Arguments
//!
//! * `--model-dir` - Local copy of the `datalab-to/indra-ocr` Hugging Face repository
//! * `--format` - `markdown` (default), `json` (the shared page JSON), or `text` (one line per detected line)
//! * `--math` - Write lines with math as inline LaTeX (needs the math models)
//! * `--keep-furniture` - Keep page headers and footers in the Markdown
//! * `--table-heads` - Build tables from the order model's row, column and cell heads
//! * `--device` - Device to use (`cpu`, `auto`, `cuda:0`, ...)
//! * `--workers` - CPU page workers, each with its own model sessions (default: 1)
//! * `<IMAGES>...` - One or more page images
//!
//! # Example
//!
//! ```bash
//! hf download datalab-to/indra-ocr --local-dir models/datalab-to/indra-ocr
//! cargo run --release --example indra -- --model-dir models/datalab-to/indra-ocr page.png
//! ```

mod utils;

use clap::{Parser, ValueEnum};
use oar_ocr::indra::IndraOcrBuilder;
use oar_ocr::utils::load_image;
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;
use utils::device_config::parse_device_config;

#[derive(Clone, Copy, ValueEnum)]
enum Format {
    Markdown,
    Json,
    Text,
}

/// Command-line arguments for the Indra OCR example
#[derive(Parser)]
#[command(name = "indra")]
#[command(about = "Read pages with Indra OCR (lines, layout blocks, Markdown)")]
struct Args {
    /// Folder with det.onnx, rec.onnx, order.onnx and meta.json
    #[arg(long)]
    model_dir: PathBuf,

    /// Input images to process
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Output format
    #[arg(long, value_enum, default_value = "markdown")]
    format: Format,

    /// Write lines with math as inline LaTeX
    #[arg(long)]
    math: bool,

    /// Keep page headers and footers in the Markdown
    #[arg(long)]
    keep_furniture: bool,

    /// Build tables from the order model's row, column and cell heads
    #[arg(long)]
    table_heads: bool,

    /// Device (cpu, auto, cuda:N, ...); accelerators require one worker
    #[arg(long, default_value = "cpu")]
    device: String,

    /// CPU page workers; unset ONNX thread counts default to one with multiple workers
    #[arg(long, default_value_t = 1)]
    workers: usize,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    utils::init_tracing();
    let args = Args::parse();

    let mut builder = IndraOcrBuilder::new(&args.model_dir)
        .math(args.math)
        .keep_furniture(args.keep_furniture)
        .table_heads(args.table_heads)
        .workers(args.workers);
    if let Some(config) = parse_device_config(&args.device)? {
        builder = builder.ort_session(config);
    }
    let start = Instant::now();
    let ocr = builder.build()?;
    info!("Loaded the models in {:.0?}", start.elapsed());

    let images = args
        .images
        .iter()
        .map(|path| {
            load_image(path).map_err(|error| format!("Failed to read {}: {error}", path.display()))
        })
        .collect::<Result<_, _>>()?;
    let start = Instant::now();
    let pages = ocr.predict(images)?;
    info!("Read {} pages in {:.0?}", pages.len(), start.elapsed());
    for (path, page) in args.images.iter().zip(pages) {
        info!(
            "{}: {} lines, {} blocks, rotation {}, skew {}, vertical {}",
            path.display(),
            page.lines.len(),
            page.blocks.len(),
            page.rotation,
            page.skew,
            page.vertical
        );
        match args.format {
            Format::Markdown => println!("{}", page.markdown),
            Format::Json => println!("{}", serde_json::to_string_pretty(&page.to_json())?),
            Format::Text => {
                for line in &page.lines {
                    println!("{}", line.text);
                }
            }
        }
    }
    Ok(())
}
