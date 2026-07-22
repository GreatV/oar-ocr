//! MinerU-Diffusion-V1 document OCR example (Candle-based).
//!
//! Runs `opendatalab/MinerU-Diffusion-V1-0320-2.5B` — a block-diffusion OCR
//! model — over one or more images. The text is produced by parallel diffusion
//! decoding (each block of output tokens is denoised from `<|MASK|>`) rather
//! than autoregressively.
//!
//! By default this runs MinerU's **two-step extraction**: a `Layout Detection`
//! pass locates every region, then each crop is routed to the matching
//! recognizer (`Text`/`Table`/`Formula Recognition`) and the structured blocks
//! are emitted as JSON — the same shape the `mineru` example produces. The
//! diffusion model shares the MinerU2.5 token vocabulary, so its layout pass
//! frames regions with `<|box_start|>…<|ref_start|>type<|ref_end|>` and its
//! table pass emits OTSL (`<fcel>`/`<nl>`), which we convert to HTML.
//!
//! Pass `--single-pass` for the legacy behaviour: one full-page
//! `Text Recognition` pass yielding flat text (no layout, no structure).
//!
//! # Usage
//!
//! ```bash
//! cargo run -p oar-ocr-vl --features cuda,download-binaries --example mineru_diffusion -- \
//!     --model-dir opendatalab/MinerU-Diffusion-V1-0320-2.5B \
//!     --device cuda:0 \
//!     document.jpg
//! ```

mod utils;

use std::path::PathBuf;
use std::time::Instant;

use clap::Parser;
use image::{RgbImage, imageops};
use tracing::{error, info};

use oar_ocr_core::utils::load_image;
use oar_ocr_vl::mineru_diffusion::DEFAULT_PROMPT;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::utils::{convert_otsl_to_html, truncate_repetitive_content};
use oar_ocr_vl::{DiffusionGenerationConfig, MinerUDiffusion};

use utils::mineru_layout::{
    ContentBlock, LAYOUT_IMAGE_SIZE, LAYOUT_PROMPT, parse_layout_output, prepare_for_extract,
};

#[derive(Parser)]
#[command(name = "mineru_diffusion")]
#[command(about = "MinerU-Diffusion-V1 block-diffusion document OCR (two-step layout routing)")]
struct Args {
    /// Path to the MinerU-Diffusion model directory
    #[arg(short, long)]
    model_dir: PathBuf,

    /// Paths to input images to process
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Device: cpu, cuda, cuda:N, or metal
    #[arg(short, long, default_value = "cpu")]
    device: String,

    /// Run a single full-page `Text Recognition` pass (flat text, no layout
    /// routing) instead of the default two-step extraction.
    #[arg(long, default_value_t = false)]
    single_pass: bool,

    /// Instruction prompt for `--single-pass` mode (default: "\nText Recognition:")
    #[arg(long)]
    prompt: Option<String>,

    /// Total generation length per pass (must be a multiple of --block-length)
    #[arg(long, default_value_t = 1024)]
    gen_length: usize,

    /// Diffusion block length
    #[arg(long, default_value_t = 32)]
    block_length: usize,

    /// Denoising steps per block
    #[arg(long, default_value_t = 32)]
    denoising_steps: usize,

    /// Confidence threshold for committing a position within a denoising step
    #[arg(long, default_value_t = 0.95)]
    dynamic_threshold: f32,

    /// Sampling temperature (reference default 1.0). Use <= 0 for deterministic
    /// greedy (argmax) decoding.
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,

    /// Top-k logit filter before sampling (0 disables, reference default)
    #[arg(long, default_value_t = 0)]
    top_k: usize,

    /// Top-p (nucleus) logit filter before sampling (1.0 disables, reference default)
    #[arg(long, default_value_t = 1.0)]
    top_p: f32,

    /// Seed for the multinomial sampler (reproducible decode)
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Minimum edge length for cropped blocks (two-step mode)
    #[arg(long, default_value_t = 28)]
    min_image_edge: u32,

    /// Max edge ratio before padding (two-step mode)
    #[arg(long, default_value_t = 50.0)]
    max_image_edge_ratio: f32,

    /// Print the raw layout-detection output (two-step mode)
    #[arg(long, default_value_t = false)]
    dump_layout: bool,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let args = Args::parse();
    let device = parse_device(&args.device)?;
    info!("Using device: {device:?}");

    info!(
        "Loading MinerU-Diffusion model from: {}",
        args.model_dir.display()
    );
    let load_start = Instant::now();
    let model = MinerUDiffusion::from_dir(&args.model_dir, device)?;
    info!("Model loaded in {:.2?}", load_start.elapsed());

    let gen_cfg = DiffusionGenerationConfig {
        gen_length: args.gen_length,
        block_length: args.block_length,
        denoising_steps: args.denoising_steps,
        dynamic_threshold: args.dynamic_threshold,
        temperature: args.temperature,
        top_k: args.top_k,
        top_p: args.top_p,
        seed: args.seed,
    };

    for image_path in &args.images {
        let image = match load_image(image_path) {
            Ok(img) => img,
            Err(e) => {
                error!("Failed to load {}: {e}", image_path.display());
                continue;
            }
        };
        info!(
            "Processing {} ({}x{})",
            image_path.display(),
            image.width(),
            image.height()
        );
        let start = Instant::now();

        if args.single_pass {
            let prompt = args.prompt.as_deref().unwrap_or(DEFAULT_PROMPT);
            match model.generate(&image, prompt, &gen_cfg) {
                Ok(text) => {
                    info!("Done in {:.2?}", start.elapsed());
                    println!("===== {} =====", image_path.display());
                    println!("{}", text.trim());
                }
                Err(e) => error!("Generation failed for {}: {e}", image_path.display()),
            }
            continue;
        }

        match two_step_extract(&model, &image, &gen_cfg, &args) {
            Ok(blocks) => {
                info!("Done in {:.2?}", start.elapsed());
                match serde_json::to_string_pretty(&blocks) {
                    Ok(json) => println!("{json}"),
                    Err(e) => error!("Failed to serialize output: {e}"),
                }
            }
            Err(e) => error!(
                "Two-step extraction failed for {}: {e}",
                image_path.display()
            ),
        }
    }

    Ok(())
}

/// MinerU two-step extraction adapted for the block-diffusion model: a
/// `Layout Detection` pass (decoded with special tokens preserved so the
/// `<|box_start|>…<|ref_end|>` framing survives), then a per-region recognition
/// pass routed by block type. Mirrors `examples/mineru.rs`.
fn two_step_extract(
    model: &MinerUDiffusion,
    image: &RgbImage,
    gen_cfg: &DiffusionGenerationConfig,
    args: &Args,
) -> Result<Vec<ContentBlock>, Box<dyn std::error::Error>> {
    // Step 1: layout detection on a square-resized page. `generate_raw` keeps
    // the box/ref special tokens that `parse_layout_output` keys on.
    let layout_image = imageops::resize(
        image,
        LAYOUT_IMAGE_SIZE,
        LAYOUT_IMAGE_SIZE,
        imageops::FilterType::CatmullRom,
    );
    let layout = model.generate_raw(&layout_image, LAYOUT_PROMPT, gen_cfg)?;
    if args.dump_layout {
        info!("Layout raw output:\n{layout}");
    }

    let mut blocks = parse_layout_output(&layout);
    if blocks.is_empty() {
        return Ok(blocks);
    }

    // Step 2: content extraction on cropped blocks, one region per call.
    let (block_images, prompts, indices) = prepare_for_extract(
        image,
        &blocks,
        args.min_image_edge,
        args.max_image_edge_ratio,
    );

    for (i, (block_image, prompt)) in block_images.into_iter().zip(prompts.iter()).enumerate() {
        let idx = indices[i];
        match model.generate(&block_image, prompt, gen_cfg) {
            Ok(content) => {
                let cleaned = truncate_repetitive_content(&content, 10, 10, 10);
                let content = if blocks[idx].block_type == "table" {
                    convert_otsl_to_html(&cleaned)
                } else {
                    cleaned.trim().to_string()
                };
                blocks[idx].content = Some(content);
            }
            Err(e) => error!("  Block inference failed (idx={idx}): {e}"),
        }
    }

    Ok(blocks)
}
