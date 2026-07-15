//! Shared MinerU two-step layout parsing helpers for the `mineru` and
//! `mineru_diffusion` examples. Both run the same `Layout Detection`, followed by
//! per-region recognition flow and only differ in the model's generate API,
//! so the layout-output parsing and crop preparation live here.

use image::{RgbImage, imageops};
use once_cell::sync::Lazy;
use regex::Regex;
use serde::Serialize;

use oar_ocr_vl::utils::image::resize_for_mineru;

/// Prompt that triggers the layout-detection pass.
pub const LAYOUT_PROMPT: &str = "\nLayout Detection:";
const TABLE_PROMPT: &str = "\nTable Recognition:";
const EQUATION_PROMPT: &str = "\nFormula Recognition:";
const DEFAULT_PROMPT: &str = "\nText Recognition:";
/// Square edge length the page is resized to before layout detection.
pub const LAYOUT_IMAGE_SIZE: u32 = 1036;

const LAYOUT_RE: &str = r"^<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>(.*)$";

static LAYOUT_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(LAYOUT_RE).expect("layout regex"));

/// A single detected layout block plus its (optionally) recognized content.
#[derive(Debug, Clone, Serialize)]
pub struct ContentBlock {
    #[serde(rename = "type")]
    pub block_type: String,
    pub bbox: [f32; 4],
    pub angle: Option<u16>,
    pub content: Option<String>,
}

/// Parse the raw layout-detection output into a list of [`ContentBlock`]s.
pub fn parse_layout_output(output: &str) -> Vec<ContentBlock> {
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

/// Crop each recognizable block out of the original page, applying any
/// detected rotation, and pair it with the recognition prompt for its type.
/// Returns `(crops, prompts, original_block_indices)`.
pub fn prepare_for_extract(
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
