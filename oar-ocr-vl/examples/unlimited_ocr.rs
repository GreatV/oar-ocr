//! Unlimited-OCR-2 multi-page document parsing example (Candle-based).
//!
//! All pages go into one request, in order, so the model can use
//! cross-page context; the official converter then joins them into one
//! Markdown document.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p oar-ocr-vl --example unlimited_ocr -- \
//!     --model-dir baidu/Unlimited-OCR-2 \
//!     --device cpu \
//!     page-1.png page-2.png
//! ```

mod utils;

use clap::Parser;
use std::path::PathBuf;
use std::time::Instant;
use tracing::info;

use oar_ocr_vl::unlimited_ocr::to_markdown;
use oar_ocr_vl::utils::image::load_image;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::{UnlimitedOcr, UnlimitedOcrGenerationOptions};

#[derive(Parser)]
#[command(name = "unlimited_ocr")]
#[command(about = "Unlimited-OCR-2 single- and multi-page document-to-Markdown parsing")]
struct Args {
    /// Path to the Unlimited-OCR-2 model directory
    #[arg(short, long)]
    model_dir: PathBuf,

    /// Page images, in document order
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Device to run on: auto, cpu, cuda, cuda:N, or metal
    #[arg(short, long, default_value = "auto")]
    device: String,

    /// Output budget for the whole request (default: the rest of the context)
    #[arg(long)]
    max_tokens: Option<usize>,

    /// No-repeat n-gram size; 0 disables the guard
    #[arg(long, default_value_t = 35)]
    ngram_size: usize,

    /// Print the raw layout-block output instead of Markdown
    #[arg(long)]
    raw: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    utils::init_tracing();
    let args = Args::parse();
    let pages = args
        .images
        .iter()
        .map(load_image)
        .collect::<Result<Vec<_>, _>>()?;
    let device = parse_device(&args.device)?;
    info!("Device: {device:?}");

    let load_start = Instant::now();
    let model = UnlimitedOcr::from_dir(&args.model_dir, device)?;
    info!(
        "Unlimited-OCR-2 loaded in {:.2}s",
        load_start.elapsed().as_secs_f64()
    );

    let mut options =
        UnlimitedOcrGenerationOptions::default().with_no_repeat_ngram_size(args.ngram_size);
    if let Some(max_tokens) = args.max_tokens {
        options = options.with_max_new_tokens(max_tokens);
    }
    let generate_start = Instant::now();
    let raw = model.generate(&pages, &options)?;
    info!(
        "Parsed {} page(s) in {:.2}s",
        pages.len(),
        generate_start.elapsed().as_secs_f64()
    );
    if args.raw {
        println!("{raw}");
    } else {
        println!("{}", to_markdown(&raw, false));
    }
    Ok(())
}
