//! MinerU2.5 Two-Step Document Extraction Example (Candle-based)
//!
//! This example demonstrates how to run `opendatalab/MinerU2.5-2509-1.2B` in Rust
//! using the two-step extraction pipeline: layout detection followed by content extraction.
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
//! cargo run -p oar-ocr-vl --example mineru -- \
//!     --model-dir /path/to/MinerU2.5-2509-1.2B \
//!     --device cuda:0 \
//!     document.jpg
//! ```

mod utils;

use clap::Parser;
use image::{RgbImage, imageops};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::Serialize;
use std::path::PathBuf;
use std::time::Instant;
use tracing::{error, info};

use oar_ocr_core::utils::load_image;
use oar_ocr_vl::MinerU;
use oar_ocr_vl::utils::image::resize_for_mineru;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::utils::{convert_otsl_to_html, truncate_repetitive_content};

const LAYOUT_PROMPT: &str = "\nLayout Detection:";
const TABLE_PROMPT: &str = "\nTable Recognition:";
const EQUATION_PROMPT: &str = "\nFormula Recognition:";
const DEFAULT_PROMPT: &str = "\nText Recognition:";
const LAYOUT_IMAGE_SIZE: u32 = 1036;

const LAYOUT_RE: &str = r"^<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>(.*)$";

static LAYOUT_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(LAYOUT_RE).expect("layout regex"));

#[derive(Debug, Clone, Serialize)]
struct ContentBlock {
    #[serde(rename = "type")]
    block_type: String,
    bbox: [f32; 4],
    angle: Option<u16>,
    content: Option<String>,
}

#[derive(Parser)]
#[command(name = "mineru")]
#[command(about = "MinerU2.5 Two-Step Document Extraction - layout detection + content extraction")]
struct Args {
    /// Path to the MinerU2.5 model directory
    #[arg(short, long)]
    model_dir: PathBuf,

    /// Paths to input images to process
    #[arg(required = true)]
    images: Vec<PathBuf>,

    /// Device to run on: cpu, cuda, or cuda:N (default: cpu)
    #[arg(short, long, default_value = "cpu")]
    device: String,

    /// Maximum number of tokens to generate (default: 4096)
    #[arg(long, default_value = "4096")]
    max_tokens: usize,

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

    info!("Loading MinerU2.5 model from: {}", args.model_dir.display());
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
    let layout = model
        .generate(&[layout_image], &[LAYOUT_PROMPT], max_tokens)
        .into_iter()
        .next()
        .ok_or("Layout detection returned no result")??;

    if dump_layout {
        info!("Layout raw output:\n{}", layout);
    }
    let mut blocks = parse_layout_output(&layout);
    if blocks.is_empty() {
        return Ok(blocks);
    }

    // Step 2: Content extraction on cropped blocks
    // Note: Processing one at a time due to batched inference issues with different padding
    let (block_images, prompts, indices) =
        prepare_for_extract(image, &blocks, min_image_edge, max_image_edge_ratio);
    if block_images.is_empty() {
        return Ok(blocks);
    }

    for (i, (block_image, prompt)) in block_images.into_iter().zip(prompts.iter()).enumerate() {
        let idx = indices[i];
        let output = model
            .generate(&[block_image], &[prompt], max_tokens)
            .into_iter()
            .next();
        match output {
            Some(Ok(content)) => {
                let cleaned = truncate_repetitive_content(&content, 10, 10, 10);
                let content = if blocks[idx].block_type == "table" {
                    convert_otsl_to_html(&cleaned)
                } else {
                    cleaned.trim().to_string()
                };
                blocks[idx].content = Some(content);
            }
            Some(Err(e)) => {
                error!("  Block inference failed (idx={}): {}", idx, e);
            }
            None => {
                error!("  Block inference returned no result (idx={})", idx);
            }
        }
    }

    Ok(blocks)
}

fn parse_layout_output(output: &str) -> Vec<ContentBlock> {
    let mut blocks = Vec::new();
    for line in output.lines() {
        let Some(caps) = LAYOUT_REGEX.captures(line) else {
            continue;
        };
        let Some((x1, y1, x2, y2)) = (|| {
            Some((
                caps.get(1)?.as_str().parse().ok()?,
                caps.get(2)?.as_str().parse().ok()?,
                caps.get(3)?.as_str().parse().ok()?,
                caps.get(4)?.as_str().parse().ok()?,
            ))
        })() else {
            continue;
        };
        let ref_type = caps
            .get(5)
            .map(|m| m.as_str().to_lowercase())
            .unwrap_or_default();
        let tail = caps.get(6).map(|m| m.as_str()).unwrap_or("");

        let Some((x1, y1, x2, y2)) = normalize_bbox(x1, y1, x2, y2) else {
            continue;
        };
        if !is_block_type(&ref_type) {
            continue;
        }

        let angle = parse_angle(tail);
        blocks.push(ContentBlock {
            block_type: ref_type,
            bbox: [x1, y1, x2, y2],
            angle,
            content: None,
        });
    }
    blocks
}

fn normalize_bbox(x1: i32, y1: i32, x2: i32, y2: i32) -> Option<(f32, f32, f32, f32)> {
    if [x1, y1, x2, y2].iter().any(|&v| !(0..=1000).contains(&v)) {
        return None;
    }
    let (x1, x2) = if x2 < x1 { (x2, x1) } else { (x1, x2) };
    let (y1, y2) = if y2 < y1 { (y2, y1) } else { (y1, y2) };
    if x1 == x2 || y1 == y2 {
        return None;
    }
    Some((
        x1 as f32 / 1000.0,
        y1 as f32 / 1000.0,
        x2 as f32 / 1000.0,
        y2 as f32 / 1000.0,
    ))
}

fn parse_angle(tail: &str) -> Option<u16> {
    if tail.contains("<|rotate_up|>") {
        Some(0)
    } else if tail.contains("<|rotate_right|>") {
        Some(90)
    } else if tail.contains("<|rotate_down|>") {
        Some(180)
    } else if tail.contains("<|rotate_left|>") {
        Some(270)
    } else {
        None
    }
}

fn is_block_type(value: &str) -> bool {
    matches!(
        value,
        "text"
            | "title"
            | "table"
            | "image"
            | "code"
            | "algorithm"
            | "header"
            | "footer"
            | "page_number"
            | "page_footnote"
            | "aside_text"
            | "equation"
            | "equation_block"
            | "ref_text"
            | "list"
            | "phonetic"
            | "table_caption"
            | "image_caption"
            | "code_caption"
            | "table_footnote"
            | "image_footnote"
            | "unknown"
    )
}

fn prepare_for_extract(
    image: &RgbImage,
    blocks: &[ContentBlock],
    min_image_edge: u32,
    max_image_edge_ratio: f32,
) -> (Vec<RgbImage>, Vec<String>, Vec<usize>) {
    let mut block_images = Vec::new();
    let mut prompts = Vec::new();
    let mut indices = Vec::new();

    let width = image.width() as f32;
    let height = image.height() as f32;

    for (idx, block) in blocks.iter().enumerate() {
        if matches!(
            block.block_type.as_str(),
            "image" | "list" | "equation_block"
        ) {
            continue;
        }
        let (x1, y1, x2, y2) = (
            (block.bbox[0] * width).round(),
            (block.bbox[1] * height).round(),
            (block.bbox[2] * width).round(),
            (block.bbox[3] * height).round(),
        );
        let x1 = x1.clamp(0.0, width - 1.0) as u32;
        let y1 = y1.clamp(0.0, height - 1.0) as u32;
        let x2 = x2.clamp(0.0, width) as u32;
        let y2 = y2.clamp(0.0, height) as u32;
        if x2 <= x1 || y2 <= y1 {
            continue;
        }
        let mut crop = imageops::crop_imm(image, x1, y1, x2 - x1, y2 - y1).to_image();
        if let Some(angle) = block.angle {
            crop = match angle {
                90 => imageops::rotate90(&crop),
                180 => imageops::rotate180(&crop),
                270 => imageops::rotate270(&crop),
                _ => crop,
            };
        }
        crop = resize_for_mineru(&crop, min_image_edge, max_image_edge_ratio);
        block_images.push(crop);
        prompts.push(prompt_for_block(&block.block_type).to_string());
        indices.push(idx);
    }

    (block_images, prompts, indices)
}

fn prompt_for_block(block_type: &str) -> &'static str {
    match block_type {
        "table" => TABLE_PROMPT,
        "equation" => EQUATION_PROMPT,
        _ => DEFAULT_PROMPT,
    }
}
