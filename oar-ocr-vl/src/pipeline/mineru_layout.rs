//! Shared MinerU two-step layout parsing helpers for the `mineru` and
//! `mineru_diffusion` examples. Both run the same `Layout Detection`, followed by
//! per-region recognition flow and only differ in the model's generate API,
//! so the layout-output parsing and crop preparation live here.

use image::{RgbImage, imageops};
use once_cell::sync::Lazy;
use regex::Regex;

use crate::document::page::DocumentBlock;
use crate::runtime::image::resize_for_mineru;

/// Prompt that triggers the layout-detection pass.
pub const LAYOUT_PROMPT: &str = "\nLayout Detection:";
const TABLE_PROMPT: &str = "\nTable Recognition:";
const EQUATION_PROMPT: &str = "\nFormula Recognition:";
const DEFAULT_PROMPT: &str = "\nText Recognition:";
/// Square edge length the page is resized to before layout detection.
pub const LAYOUT_IMAGE_SIZE: u32 = 1036;

const LAYOUT_RE: &str =
    r"<\|box_start\|>(\d+)\s+(\d+)\s+(\d+)\s+(\d+)<\|box_end\|><\|ref_start\|>(\w+?)<\|ref_end\|>";

static LAYOUT_REGEX: Lazy<Regex> = Lazy::new(|| Regex::new(LAYOUT_RE).expect("layout regex"));

/// Backwards-compatible name used by the original examples.
pub type ContentBlock = DocumentBlock;

/// Parse the raw layout-detection output into a list of [`ContentBlock`]s.
pub fn parse_layout_output(output: &str) -> Vec<ContentBlock> {
    let mut blocks = Vec::new();
    let records: Vec<_> = LAYOUT_REGEX.captures_iter(output).collect();
    for (index, caps) in records.iter().enumerate() {
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
        let end = records
            .get(index + 1)
            .map_or(output.len(), |record| record.get(0).unwrap().start());
        let tail = &output[caps.get(0).unwrap().end()..end];

        let Some((x1, y1, x2, y2)) = normalize_bbox(x1, y1, x2, y2) else {
            continue;
        };
        if !is_block_type(&ref_type) {
            continue;
        }

        if ref_type == "inline_formula" {
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
    // Table-internal text/formulas are recognized by the table pass itself.
    let tables: Vec<_> = blocks
        .iter()
        .filter(|block| block.block_type == "table")
        .map(|block| block.bbox)
        .collect();
    blocks.retain(|block| {
        !matches!(
            block.block_type.as_str(),
            "text" | "equation" | "equation_block"
        ) || !tables
            .iter()
            .any(|table| cover_ratio(*table, block.bbox) > 0.9)
    });
    blocks.dedup_by(|next, prev| {
        next.block_type == prev.block_type && next.bbox == prev.bbox && next.angle == prev.angle
    });
    blocks
}

pub(crate) fn cover_ratio(container: [f32; 4], bbox: [f32; 4]) -> f32 {
    let area = (bbox[2] - bbox[0]).max(0.0) * (bbox[3] - bbox[1]).max(0.0);
    if area == 0.0 {
        return 0.0;
    }
    (container[2].min(bbox[2]) - container[0].max(bbox[0])).max(0.0)
        * (container[3].min(bbox[3]) - container[1].max(bbox[1])).max(0.0)
        / area
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
            | "chart"
            | "image_block"
            | "inline_formula"
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
            "image" | "image_block" | "list" | "equation_block"
        ) {
            continue;
        }
        let (x1, y1, x2, y2) = (
            (block.bbox[0] * width).round_ties_even(),
            (block.bbox[1] * height).round_ties_even(),
            (block.bbox[2] * width).round_ties_even(),
            (block.bbox[3] * height).round_ties_even(),
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
                // Pillow positive angles rotate counterclockwise.
                90 => imageops::rotate270(&crop),
                180 => imageops::rotate180(&crop),
                270 => imageops::rotate90(&crop),
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
        "chart" => "\nImage Analysis:",
        _ => DEFAULT_PROMPT,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn adjacent_layout_records_keep_angles_and_filter_table_internal_text() {
        let raw = concat!(
            "<|box_start|>0 0 500 500<|box_end|><|ref_start|>table<|ref_end|><|rotate_right|>",
            "<|box_start|>10 10 100 100<|box_end|><|ref_start|>text<|ref_end|><|rotate_up|>",
            "<|box_start|>500 0 1000 500<|box_end|><|ref_start|>chart<|ref_end|><|rotate_left|>"
        );
        let blocks = parse_layout_output(raw);
        assert_eq!(blocks.len(), 2);
        assert_eq!(blocks[0].angle, Some(90));
        assert_eq!(blocks[1].angle, Some(270));
        assert_eq!(prompt_for_block(&blocks[1].block_type), "\nImage Analysis:");
    }

    #[test]
    fn crop_rotation_matches_pillow_positive_angles() {
        let image = RgbImage::from_fn(3, 2, |x, y| image::Rgb([(y * 3 + x + 1) as u8, 0, 0]));
        for (angle, expected) in [(90, vec![3, 6, 2, 5, 1, 4]), (270, vec![4, 1, 5, 2, 6, 3])] {
            let block = ContentBlock {
                block_type: "text".into(),
                bbox: [0.0, 0.0, 1.0, 1.0],
                angle: Some(angle),
                content: None,
            };
            let (images, _, _) = prepare_for_extract(&image, &[block], 1, 50.0);
            assert_eq!(
                images[0].pixels().map(|p| p[0]).collect::<Vec<_>>(),
                expected
            );
        }
    }

    #[test]
    fn parses_normalized_layout_and_rotation() {
        let output =
            "<|box_start|>100 200 500 600<|box_end|><|ref_start|>Table<|ref_end|><|rotate_right|>";
        let blocks = parse_layout_output(output);
        assert_eq!(blocks.len(), 1);
        assert_eq!(blocks[0].block_type, "table");
        assert_eq!(blocks[0].bbox, [0.1, 0.2, 0.5, 0.6]);
        assert_eq!(blocks[0].angle, Some(90));
    }

    #[test]
    fn rejects_invalid_or_unknown_layout_entries() {
        let output = concat!(
            "<|box_start|>0 0 0 10<|box_end|><|ref_start|>text<|ref_end|>\n",
            "<|box_start|>0 0 10 10<|box_end|><|ref_start|>unsupported<|ref_end|>\n",
            "not a layout record"
        );
        assert!(parse_layout_output(output).is_empty());
    }

    #[test]
    fn prepares_only_recognizable_blocks() {
        let image = RgbImage::new(100, 100);
        let blocks = vec![
            ContentBlock {
                block_type: "table".to_string(),
                bbox: [0.1, 0.1, 0.9, 0.9],
                angle: None,
                content: None,
            },
            ContentBlock {
                block_type: "image".to_string(),
                bbox: [0.0, 0.0, 1.0, 1.0],
                angle: None,
                content: None,
            },
        ];
        let (images, prompts, indices) = prepare_for_extract(&image, &blocks, 28, 50.0);
        assert_eq!(images.len(), 1);
        assert_eq!(prompts, [TABLE_PROMPT]);
        assert_eq!(indices, [0]);
    }
}
