//! MinerU2.5 / MinerU2.5-Pro Two-Step Document Extraction Example (Candle-based)
//!
//! This example runs `opendatalab/MinerU2.5-2509-1.2B` or
//! `opendatalab/MinerU2.5-Pro-2605-1.2B` in Rust using the same two-step
//! extraction pipeline: layout detection followed by content extraction.
//!
//! # Usage
//!
//! ```bash
//! cargo run -p oar-ocr-vl --example mineru -- [OPTIONS] <IMAGES>...
//! ```
//!
//! # Example
//!
//! ```bash
//! cargo run -p oar-ocr-vl --features cuda --example mineru -- \
//!     --model-dir opendatalab/MinerU2.5-2509-1.2B \
//!     --device cuda:0 \
//!     document.jpg
//!
//! # MinerU2.5-Pro uses the same loader and pipeline
//! cargo run -p oar-ocr-vl --features cuda --example mineru -- \
//!     --model-dir opendatalab/MinerU2.5-Pro-2605-1.2B \
//!     --device cuda:0 \
//!     document.jpg
//! ```

mod utils;

use clap::Parser;
use image::{RgbImage, imageops};
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info};

use oar_ocr_vl::utils::image::load_image;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::utils::{convert_otsl_to_html, truncate_repetitive_content};
use oar_ocr_vl::{Error as OarError, MinerU};

use utils::mineru_layout::{
    ContentBlock, LAYOUT_IMAGE_SIZE, LAYOUT_PROMPT, parse_layout_output, prepare_for_extract,
};
use utils::token_fingerprint;

#[derive(Parser)]
#[command(name = "mineru")]
#[command(
    about = "MinerU2.5 / MinerU2.5-Pro two-step extraction - layout detection + content extraction"
)]
struct Args {
    /// Path to a MinerU2.5 or MinerU2.5-Pro model directory
    #[arg(short, long)]
    model_dir: PathBuf,

    /// Paths to input images to process
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Device to run on: cpu, cuda, cuda:N, or metal (default: cpu)
    #[arg(short, long, default_value = "cpu")]
    device: String,

    /// Maximum number of tokens to generate (default: 4096)
    #[arg(long, default_value = "4096")]
    max_tokens: usize,

    /// Number of cropped regions per content-extraction batch
    #[arg(long, default_value = "2")]
    region_batch_size: usize,

    /// Minimum edge length for cropped blocks
    #[arg(long, default_value = "28")]
    min_image_edge: u32,

    /// Max edge ratio before padding
    #[arg(long, default_value = "50")]
    max_image_edge_ratio: f32,

    /// Print raw layout output
    #[arg(long, default_value_t = false)]
    dump_layout: bool,
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

    let device = parse_device(&args.device)?;
    info!("Using device: {:?}", device);

    info!("Loading MinerU model from: {}", args.model_dir.display());
    let load_start = Instant::now();
    let model = MinerU::from_dir(&args.model_dir, device)?;
    info!(
        "Model loaded in {:.2}ms",
        load_start.elapsed().as_secs_f64() * 1000.0
    );

    info!("\n=== Processing {} images ===", existing_images.len());
    for image_path in &existing_images {
        info!("\nProcessing: {}", image_path.display());
        let rgb_img = match load_image(image_path) {
            Ok(img) => img,
            Err(e) => {
                error!("  Failed to load image: {}", e);
                continue;
            }
        };

        let infer_start = Instant::now();
        match two_step_extract(
            &model,
            &rgb_img,
            args.max_tokens,
            args.region_batch_size,
            args.min_image_edge,
            args.max_image_edge_ratio,
            args.dump_layout,
        ) {
            Ok(blocks) => {
                info!(
                    "  Inference time: {:.2}ms",
                    infer_start.elapsed().as_secs_f64() * 1000.0
                );
                match serde_json::to_string_pretty(&blocks) {
                    Ok(json) => println!("{}", json),
                    Err(e) => error!("  Failed to serialize output: {}", e),
                }
            }
            Err(e) => error!("  Inference failed: {}", e),
        }
    }

    Ok(())
}

fn two_step_extract(
    model: &MinerU,
    image: &RgbImage,
    max_tokens: usize,
    region_batch_size: usize,
    min_image_edge: u32,
    max_image_edge_ratio: f32,
    dump_layout: bool,
) -> Result<Vec<ContentBlock>, Box<dyn std::error::Error>> {
    // Step 1: Layout detection on resized image
    let layout_image = imageops::resize(
        image,
        LAYOUT_IMAGE_SIZE,
        LAYOUT_IMAGE_SIZE,
        imageops::FilterType::CatmullRom,
    );
    let layout_tokens = model
        .generate_tokens(&[layout_image], &[LAYOUT_PROMPT], max_tokens)?
        .into_iter()
        .next()
        .ok_or("Layout detection returned no result")?;
    info!(
        "  Layout tokens: {}, fingerprint: {:016x}",
        layout_tokens.len(),
        token_fingerprint(&layout_tokens)
    );
    let layout = model.decode_tokens(&layout_tokens)?;

    if dump_layout {
        info!("Layout raw output:\n{}", layout);
    }
    let mut blocks = parse_layout_output(&layout);
    if blocks.is_empty() {
        return Ok(blocks);
    }

    // Step 2: Content extraction on cropped blocks. MinerU masks the leading
    // padding KV for unequal prompt lengths, so differently sized crops can be
    // decoded in one batch.
    let (block_images, prompts, indices) =
        prepare_for_extract(image, &blocks, min_image_edge, max_image_edge_ratio);
    if block_images.is_empty() {
        return Ok(blocks);
    }

    let region_batch_size = region_batch_size.max(1);
    for start in (0..block_images.len()).step_by(region_batch_size) {
        let end = (start + region_batch_size).min(block_images.len());
        let expected = end - start;
        let (fallback_reason, generated) = recover_batch_results(
            expected,
            model.generate_tokens(&block_images[start..end], &prompts[start..end], max_tokens),
            |offset| {
                let position = start + offset;
                let mut tokens = model.generate_tokens(
                    std::slice::from_ref(&block_images[position]),
                    std::slice::from_ref(&prompts[position]),
                    max_tokens,
                )?;
                if tokens.len() != 1 {
                    return Err(OarError::InvalidInput {
                        message: format!(
                            "Content extraction returned {} results for block {}",
                            tokens.len(),
                            indices[position]
                        ),
                    });
                }
                Ok(tokens.pop().expect("single result length was checked"))
            },
        );
        if let Some(reason) = fallback_reason {
            error!(
                "  Batch inference failed for blocks {:?}: {}; retrying individually",
                &indices[start..end],
                reason
            );
        }

        for (&idx, result) in indices[start..end].iter().zip(generated) {
            match result {
                Ok(tokens) => {
                    info!(
                        "  Block {} tokens: {}, fingerprint: {:016x}",
                        idx,
                        tokens.len(),
                        token_fingerprint(&tokens)
                    );
                    let content = model.decode_tokens(&tokens)?;
                    let cleaned = truncate_repetitive_content(&content, 10, 10, 10);
                    let content = if blocks[idx].block_type == "table" {
                        convert_otsl_to_html(&cleaned)
                    } else {
                        cleaned.trim().to_string()
                    };
                    blocks[idx].content = Some(content);
                }
                Err(error) => error!("  Block inference failed (idx={}): {}", idx, error),
            }
        }
    }

    Ok(blocks)
}

fn recover_batch_results<T, E>(
    expected: usize,
    batch: Result<Vec<T>, E>,
    mut retry_one: impl FnMut(usize) -> Result<T, E>,
) -> (Option<String>, Vec<Result<T, E>>)
where
    E: std::fmt::Display,
{
    match batch {
        Ok(results) if results.len() == expected => (None, results.into_iter().map(Ok).collect()),
        Ok(results) => {
            let reason = format!(
                "content extraction returned {} results for {expected} blocks",
                results.len()
            );
            (Some(reason), (0..expected).map(&mut retry_one).collect())
        }
        Err(error) if expected == 1 => (None, vec![Err(error)]),
        Err(error) => {
            let reason = error.to_string();
            (Some(reason), (0..expected).map(&mut retry_one).collect())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::recover_batch_results;
    use std::cell::Cell;

    #[test]
    fn successful_batch_does_not_retry_items() {
        let retries = Cell::new(0);
        let (reason, results) = recover_batch_results::<_, String>(2, Ok(vec![10, 20]), |_| {
            retries.set(retries.get() + 1);
            Ok(0)
        });

        assert!(reason.is_none());
        assert_eq!(retries.get(), 0);
        assert_eq!(
            results.into_iter().collect::<Result<Vec<_>, _>>(),
            Ok(vec![10, 20])
        );
    }

    #[test]
    fn failed_batch_retries_each_item_and_preserves_item_errors() {
        let (reason, results) =
            recover_batch_results(3, Err("batch failed"), |index| match index {
                1 => Err("bad crop"),
                _ => Ok(index),
            });

        assert_eq!(reason.as_deref(), Some("batch failed"));
        assert_eq!(results, vec![Ok(0), Err("bad crop"), Ok(2)]);
    }

    #[test]
    fn single_item_failure_is_not_retried() {
        let retries = Cell::new(0);
        let (reason, results) = recover_batch_results(1, Err("bad crop"), |_| {
            retries.set(retries.get() + 1);
            Ok(0)
        });

        assert!(reason.is_none());
        assert_eq!(retries.get(), 0);
        assert_eq!(results, vec![Err("bad crop")]);
    }

    #[test]
    fn wrong_batch_result_count_retries_each_item() {
        let (reason, results) =
            recover_batch_results::<_, String>(2, Ok(vec![10]), |index| Ok(index + 20));

        assert_eq!(
            reason.as_deref(),
            Some("content extraction returned 1 results for 2 blocks")
        );
        assert_eq!(results, vec![Ok(20), Ok(21)]);
    }
}
