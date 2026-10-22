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

    /// Device (cpu, auto, cuda:N, ...); the models are built for CPU
    #[arg(long, default_value = "cpu")]
    device: String,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    utils::init_tracing();
    let args = Args::parse();

    let mut builder = IndraOcrBuilder::new(&args.model_dir)
        .math(args.math)
        .keep_furniture(args.keep_furniture)
        .table_heads(args.table_heads);
    if let Some(config) = parse_device_config(&args.device)? {
        builder = builder.ort_session(config);
    }
    let start = Instant::now();
    let ocr = builder.build()?;
    info!("Loaded the models in {:.0?}", start.elapsed());

    for path in &args.images {
        let image = load_image(path)?;
        let start = Instant::now();
        let page = ocr.predict(vec![image])?.remove(0);
        info!(
            "{}: {} lines, {} blocks, rotation {}, skew {}, vertical {} in {:.0?}",
            path.display(),
            page.lines.len(),
            page.blocks.len(),
            page.rotation,
            page.skew,
            page.vertical,
            start.elapsed()
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
