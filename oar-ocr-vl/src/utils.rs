//! Utility functions for VL models and document parsing.
//!
//! This module provides:
//! - Device configuration for Candle-based models
//! - Candle tensor utilities for model inference
//! - Markdown conversion for parsed documents
//! - OTSL to HTML table conversion
//! - Image processing helpers

pub mod image;
pub mod table;
pub mod text;

use ::image::{GrayImage, RgbImage};
use candle_core::{Device, IndexOp, Tensor};
use oar_ocr_core::core::OCRError;
use oar_ocr_core::domain::structure::{LayoutElement, LayoutElementType};
use oar_ocr_core::processors::BoundingBox;
use once_cell::sync::Lazy;
use regex::Regex;
use std::collections::HashSet;

/// Parses a device string and creates a Candle [`Device`].
///
/// # Supported formats
///
/// - `"cpu"` → CPU device
/// - `"cuda"` or `"gpu"` → CUDA device 0
/// - `"cuda:N"` → CUDA device N (e.g., `"cuda:1"`)
///
/// # Errors
///
/// Returns an error if:
/// - The device string is invalid
/// - CUDA is requested but the `cuda` feature is not enabled
/// - CUDA device creation fails
///
/// # Examples
///
/// ```no_run
/// use oar_ocr_vl::utils::parse_device;
///
/// # fn main() -> Result<(), oar_ocr_core::core::OCRError> {
/// let cpu = parse_device("cpu")?;
/// let cuda = parse_device("cuda")?;
/// let cuda1 = parse_device("cuda:1")?;
/// # let _ = (cpu, cuda, cuda1);
/// # Ok(())
/// # }
/// ```
#[cfg(not(feature = "cuda"))]
fn cuda_not_enabled() -> OCRError {
    OCRError::ConfigError {
        message: "CUDA support not enabled. Compile with --features cuda".to_string(),
    }
}

pub fn parse_device(device_str: &str) -> Result<Device, OCRError> {
    let device_str = device_str.to_lowercase();
    match device_str.as_str() {
        "cpu" => Ok(Device::Cpu),
        "cuda" | "gpu" => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).map_err(|e| OCRError::ConfigError {
                    message: format!("Failed to create CUDA device: {}", e),
                })
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(cuda_not_enabled())
            }
        }
        s if s.starts_with("cuda:") => {
            #[cfg(feature = "cuda")]
            {
                let ordinal_str = s
                    .strip_prefix("cuda:")
                    .ok_or_else(|| OCRError::ConfigError {
                        message: format!("Invalid CUDA device string '{}'", s),
                    })?;
                let ordinal: usize = ordinal_str.parse().map_err(|_| OCRError::ConfigError {
                    message: format!("Invalid CUDA device ordinal in '{}'", s),
                })?;
                Device::new_cuda(ordinal).map_err(|e| OCRError::ConfigError {
                    message: format!("Failed to create CUDA device {}: {}", ordinal, e),
                })
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(cuda_not_enabled())
            }
        }
        _ => Err(OCRError::ConfigError {
            message: format!(
                "Unknown device: '{}'. Use 'cpu', 'cuda', or 'cuda:N'",
                device_str
            ),
        }),
    }
}

/// Convert Candle error to OCRError for inference operations.
pub fn candle_to_ocr_inference(
    model_name: &str,
    context: impl Into<String>,
    err: candle_core::Error,
) -> OCRError {
    OCRError::Inference {
        model_name: model_name.to_string(),
        context: context.into(),
        source: Box::new(err),
    }
}

/// Convert Candle error to OCRError for processing operations.
pub fn candle_to_ocr_processing(
    kind: oar_ocr_core::core::errors::ProcessingStage,
    context: impl Into<String>,
    err: candle_core::Error,
) -> OCRError {
    OCRError::Processing {
        kind,
        context: context.into(),
        source: Box::new(err),
    }
}

/// Rotate half of the tensor dimensions for RoPE.
pub fn rotate_half(x: &Tensor) -> Result<Tensor, OCRError> {
    let d = x.dim(candle_core::D::Minus1).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "rotate_half dim failed",
            e,
        )
    })?;
    let half = d / 2;
    let x1 = x.i((.., .., .., 0..half)).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "rotate_half slice x1 failed",
            e,
        )
    })?;
    let x2 = x.i((.., .., .., half..d)).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "rotate_half slice x2 failed",
            e,
        )
    })?;
    let nx2 = x2.neg().map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "rotate_half neg failed",
            e,
        )
    })?;
    Tensor::cat(&[&nx2, &x1], candle_core::D::Minus1).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "rotate_half cat failed",
            e,
        )
    })
}

/// Convert layout elements to Markdown format.
///
/// # Arguments
/// * `elements` - Layout elements with recognized text
/// * `ignore_labels` - Labels to skip in markdown output
pub fn to_markdown(elements: &[LayoutElement], ignore_labels: &[String]) -> String {
    let mut markdown = String::new();

    for (i, element) in elements.iter().enumerate() {
        let text = match &element.text {
            Some(t) if !t.trim().is_empty() => t.trim(),
            _ => continue,
        };

        if let Some(label) = &element.label
            && ignore_labels.iter().any(|l| l == label)
        {
            continue;
        }

        let content = match element.element_type {
            LayoutElementType::DocTitle => format_heading(text, 1),
            LayoutElementType::ParagraphTitle => format_heading(text, 2),
            LayoutElementType::Table => text::format_table(text),
            LayoutElementType::Formula => text::format_formula(text),
            LayoutElementType::Image | LayoutElementType::Chart | LayoutElementType::Seal => {
                format_figure(text, i)
            }
            LayoutElementType::List => format_list(text),
            LayoutElementType::Algorithm => format_code(text),
            _ => text::format_text(text),
        };

        if !content.is_empty() {
            markdown.push_str(&content);
            markdown.push_str("\n\n");
        }
    }

    markdown.trim().to_string()
}

// --- OpenOCR / PaddleX markdown compatibility ---

// Matches PaddleX `compile_title_pattern()` in:
// paddlex/inference/pipelines/layout_parsing/result_v2.py
static OPENOCR_TITLE_RE_PATTERN: Lazy<Result<Regex, regex::Error>> = Lazy::new(|| {
    // Note: Rust's `regex` is stricter about escapes inside character classes than Python `re`.
    // Keep the pattern semantically identical while avoiding unnecessary escapes.
    Regex::new(
        r"^\s*((?:[1-9][0-9]*(?:\.[1-9][0-9]*)*[.、]?|[(（](?:[1-9][0-9]*|[一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+)[)）]|[一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+[、.]?|(?:I|II|III|IV|V|VI|VII|VIII|IX|X)(?:\.|\s)))(\s*)(.*)$",
    )
});

fn openocr_format_title(text: &str) -> String {
    let mut title = text.to_string();
    if let Ok(re) = OPENOCR_TITLE_RE_PATTERN.as_ref()
        && let Some(caps) = re.captures(&title)
    {
        let numbering_raw = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        let numbering = numbering_raw.trim();
        let title_content = caps.get(3).map(|m| m.as_str()).unwrap_or("").trim_start();
        if !numbering.is_empty() {
            title = format!("{numbering} {title_content}");
        }
    }

    title = title.trim_end_matches('.').to_string();
    let level = if title.contains('.') {
        title.chars().filter(|&c| c == '.').count() + 1
    } else {
        1
    };

    format!("{} {}", "#".repeat(level + 1), title)
        .replace("-\n", "")
        .replace('\n', " ")
}

fn openocr_format_centered_by_html(text: &str) -> String {
    let content = text.replace("-\n", "").replace('\n', " ");
    format!("<div style=\"text-align: center;\">{}</div>\n", content)
}

fn openocr_format_table_center_func(html: &str) -> String {
    let mut table_content = html.to_string();
    table_content = table_content.replace(
        "<table>",
        "<table border=1 style='margin: auto; word-wrap: break-word;'>",
    );
    table_content = table_content.replace(
        "<th>",
        "<th style='text-align: center; word-wrap: break-word;'>",
    );
    table_content = table_content.replace(
        "<td>",
        "<td style='text-align: center; word-wrap: break-word;'>",
    );
    table_content
}

fn openocr_format_text_block(text: &str) -> String {
    text.replace("\n\n", "\n").replace('\n', "\n\n")
}

fn openocr_format_content_block(text: &str) -> String {
    text.replace("-\n", "  \n").replace('\n', "  \n")
}

fn openocr_format_first_line(
    text: &str,
    templates_lower: &[&str],
    format_first_line: impl Fn(&str) -> String,
    splitter: &str,
) -> String {
    let mut parts: Vec<String> = text.split(splitter).map(|s| s.to_string()).collect();
    for part in parts.iter_mut() {
        if part.trim().is_empty() {
            continue;
        }
        let lower = part.to_lowercase();
        if templates_lower.iter().any(|t| *t == lower) {
            *part = format_first_line(part);
        }
        break;
    }
    parts.join(splitter)
}

/// Convert layout elements to OpenOCR (PaddleX) markdown format.
///
/// This matches `PaddleOCRVLResult._to_markdown(pretty=...)` when labels come from PP-DocLayoutV2/V3.
pub fn to_markdown_openocr(
    elements: &[LayoutElement],
    ignore_labels: &[String],
    pretty: bool,
) -> String {
    let mut markdown = String::new();

    for element in elements {
        let label = element.label.as_deref().unwrap_or("");
        if ignore_labels.iter().any(|l| l == label) {
            continue;
        }

        let content = element.text.as_deref().unwrap_or("");

        let formatted = match label {
            // Titles
            "paragraph_title" | "abstract_title" | "reference_title" | "content_title" => {
                openocr_format_title(content)
            }
            "doc_title" => format!("# {}", content)
                .replace("-\n", "")
                .replace('\n', " "),

            // Captions (centered in pretty mode)
            "table_title" | "figure_title" | "chart_title" => {
                if pretty {
                    openocr_format_centered_by_html(content)
                } else {
                    content.to_string()
                }
            }

            // Text blocks
            "text" | "ocr" | "vertical_text" | "reference_content" => {
                openocr_format_text_block(content)
            }

            // Special sections
            "abstract" => openocr_format_first_line(
                content,
                &["摘要", "abstract"],
                |l| format!("## {l}\n"),
                " ",
            ),
            "reference" => openocr_format_first_line(
                content,
                &["参考文献", "references"],
                |l| format!("## {l}"),
                "\n",
            ),
            "content" => openocr_format_content_block(content),

            // Tables
            "table" => {
                if pretty {
                    format!("\n{}", openocr_format_table_center_func(content))
                } else {
                    // `simplify_table_func("\n" + block.content)`
                    format!("\n{}", content)
                        .replace("<html>", "")
                        .replace("</html>", "")
                        .replace("<body>", "")
                        .replace("</body>", "")
                }
            }

            // Formulas are already formatted with $$ in the pipeline.
            "formula" | "display_formula" | "inline_formula" => content.to_string(),

            // Algorithm blocks.
            "algorithm" => content.trim_matches('\n').to_string(),

            // Fallback: use existing heuristic mapping by element_type.
            _ => match element.element_type {
                LayoutElementType::ParagraphTitle => openocr_format_title(content),
                LayoutElementType::DocTitle => format!("# {}", content)
                    .replace("-\n", "")
                    .replace('\n', " "),
                LayoutElementType::Table => {
                    if pretty {
                        format!("\n{}", openocr_format_table_center_func(content))
                    } else {
                        content.to_string()
                    }
                }
                _ => content.to_string(),
            },
        };

        if markdown.is_empty() {
            markdown.push_str(&formatted);
        } else {
            markdown.push_str("\n\n");
            markdown.push_str(&formatted);
        }
    }

    markdown
}

fn format_heading(text: &str, level: usize) -> String {
    let prefix = "#".repeat(level.min(6));
    let cleaned = remove_newlines_in_heading(text);
    let processed = text::process_text(cleaned.trim());
    format!("{} {}", prefix, processed)
}

fn format_figure(text: &str, index: usize) -> String {
    if text.starts_with("![") {
        return text.to_string();
    }
    if text.starts_with("figures/") || text.starts_with("imgs/") {
        return format!("![Figure {}]({})", index + 1, text);
    }
    if text.starts_with("data:image/") {
        return format!("![Figure {}]({})", index + 1, text);
    }
    format!("*Figure {}: {}*", index + 1, text)
}

fn format_list(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut result = String::new();
    for line in lines {
        let trimmed = line.trim();
        if !trimmed.is_empty() {
            if trimmed.starts_with('-')
                || trimmed.starts_with('*')
                || trimmed
                    .chars()
                    .next()
                    .map(|c| c.is_ascii_digit())
                    .unwrap_or(false)
            {
                result.push_str(trimmed);
            } else {
                result.push_str("- ");
                result.push_str(trimmed);
            }
            result.push('\n');
        }
    }
    result.trim_end().to_string()
}

fn format_code(text: &str) -> String {
    format!("```\n{}\n```", text.trim())
}

fn remove_newlines_in_heading(text: &str) -> String {
    fn is_chinese(c: char) -> bool {
        ('\u{4e00}'..='\u{9fff}').contains(&c)
    }
    if text.chars().any(is_chinese) {
        text.replace('\n', "")
    } else {
        text.replace('\n', " ")
    }
}

pub use self::table::convert_otsl_to_html;
pub use self::text::truncate_repetitive_content;

/// Calculate the area of a bounding box.
#[inline]
pub fn calculate_bbox_area(bbox: &BoundingBox) -> f32 {
    (bbox.x_max() - bbox.x_min()).abs() * (bbox.y_max() - bbox.y_min()).abs()
}

/// Calculate overlap ratio between two bounding boxes.
pub fn calculate_overlap_ratio(bbox1: &BoundingBox, bbox2: &BoundingBox, mode: &str) -> f32 {
    let x_min_inter = bbox1.x_min().max(bbox2.x_min());
    let y_min_inter = bbox1.y_min().max(bbox2.y_min());
    let x_max_inter = bbox1.x_max().min(bbox2.x_max());
    let y_max_inter = bbox1.y_max().min(bbox2.y_max());

    let inter_width = (x_max_inter - x_min_inter).max(0.0);
    let inter_height = (y_max_inter - y_min_inter).max(0.0);
    let inter_area = inter_width * inter_height;

    let bbox1_area = calculate_bbox_area(bbox1);
    let bbox2_area = calculate_bbox_area(bbox2);

    let ref_area = match mode {
        "union" => bbox1_area + bbox2_area - inter_area,
        "small" => bbox1_area.min(bbox2_area),
        "large" => bbox1_area.max(bbox2_area),
        _ => bbox1_area + bbox2_area - inter_area,
    };

    if ref_area == 0.0 {
        0.0
    } else {
        inter_area / ref_area
    }
}

/// Calculate projection overlap ratio between two bounding boxes.
pub fn calculate_projection_overlap_ratio(
    bbox1: &BoundingBox,
    bbox2: &BoundingBox,
    direction: &str,
    mode: &str,
) -> f32 {
    let (start1, end1, start2, end2) = if direction == "horizontal" {
        (bbox1.x_min(), bbox1.x_max(), bbox2.x_min(), bbox2.x_max())
    } else {
        (bbox1.y_min(), bbox1.y_max(), bbox2.y_min(), bbox2.y_max())
    };

    let intersection_start = start1.max(start2);
    let intersection_end = end1.min(end2);
    let overlap = intersection_end - intersection_start;

    if overlap <= 0.0 {
        return 0.0;
    }

    let ref_width = match mode {
        "union" => end1.max(end2) - start1.min(start2),
        "small" => (end1 - start1).min(end2 - start2),
        "large" => (end1 - start1).max(end2 - start2),
        _ => end1.max(end2) - start1.min(start2),
    };

    if ref_width > 0.0 {
        overlap / ref_width
    } else {
        0.0
    }
}

/// Detected layout box with label and score.
#[derive(Debug, Clone)]
pub struct DetectedBox {
    pub bbox: BoundingBox,
    pub label: String,
    pub score: f32,
}

/// Filter overlapping boxes from layout detection results.
pub fn filter_overlap_boxes(boxes: Vec<DetectedBox>, overlap_threshold: f32) -> Vec<DetectedBox> {
    let boxes: Vec<DetectedBox> = boxes
        .into_iter()
        .filter(|b| b.label != "reference")
        .collect();

    let mut dropped_indexes: HashSet<usize> = HashSet::new();

    for i in 0..boxes.len() {
        for j in (i + 1)..boxes.len() {
            if dropped_indexes.contains(&i) || dropped_indexes.contains(&j) {
                continue;
            }

            let overlap_ratio = calculate_overlap_ratio(&boxes[i].bbox, &boxes[j].bbox, "small");

            if overlap_ratio > overlap_threshold {
                let area_i = calculate_bbox_area(&boxes[i].bbox);
                let area_j = calculate_bbox_area(&boxes[j].bbox);

                if (boxes[i].label == "image" || boxes[j].label == "image")
                    && boxes[i].label != boxes[j].label
                {
                    continue;
                }

                if area_i >= area_j {
                    dropped_indexes.insert(j);
                } else {
                    dropped_indexes.insert(i);
                }
            }
        }
    }

    boxes
        .into_iter()
        .enumerate()
        .filter(|(idx, _)| !dropped_indexes.contains(idx))
        .map(|(_, b)| b)
        .collect()
}

/// Crop white margins from a formula image.
pub fn crop_margin(img: &RgbImage) -> RgbImage {
    let gray: GrayImage = ::image::imageops::grayscale(img);
    let (min_val, max_val) = gray.pixels().fold((255u8, 0u8), |(min, max), p| {
        (min.min(p.0[0]), max.max(p.0[0]))
    });

    if max_val == min_val {
        return img.clone();
    }

    let threshold = 200u8;
    let mut x_min = img.width();
    let mut y_min = img.height();
    let mut x_max = 0;
    let mut y_max = 0;
    let mut found = false;

    for (x, y, pixel) in gray.enumerate_pixels() {
        let normalized = ((pixel.0[0] as f32 - min_val as f32) / (max_val as f32 - min_val as f32)
            * 255.0) as u8;
        if normalized < threshold {
            if x < x_min {
                x_min = x;
            }
            if x > x_max {
                x_max = x;
            }
            if y < y_min {
                y_min = y;
            }
            if y > y_max {
                y_max = y;
            }
            found = true;
        }
    }

    if !found {
        return img.clone();
    }

    let width = (x_max - x_min + 1).min(img.width() - x_min);
    let height = (y_max - y_min + 1).min(img.height() - y_min);

    if width == 0 || height == 0 {
        return img.clone();
    }

    ::image::imageops::crop_imm(img, x_min, y_min, width, height).to_image()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_heading() {
        // Avoid false-positive roman numerals (e.g., "Impact" should not become "I mpact").
        assert_eq!(
            openocr_format_title("Impact of Class Labels"),
            "## Impact of Class Labels"
        );
        assert_eq!(
            openocr_format_title("Vision Transformers"),
            "## Vision Transformers"
        );
        assert_eq!(openocr_format_title("Learning Curve"), "## Learning Curve");
        assert_eq!(openocr_format_title("1.2.3 Section"), "#### 1.2.3 Section");
        assert_eq!(openocr_format_title("1.2Title"), "### 1.2 Title");
        assert_eq!(
            openocr_format_title("I. Introduction"),
            "### I. Introduction"
        );
    }
}
