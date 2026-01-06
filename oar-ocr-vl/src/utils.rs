//! Utility functions for VL models and document parsing.
//!
//! This module provides:
//! - Device configuration for Candle-based models
//! - Candle tensor utilities for model inference
//! - Markdown conversion for parsed documents
//! - OTSL to HTML table conversion
//! - Image processing helpers

use candle_core::{Device, IndexOp, Tensor};
use image::{GrayImage, RgbImage};
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
/// let cpu = parse_device("cpu").unwrap();
/// let cuda = parse_device("cuda").unwrap();
/// let cuda1 = parse_device("cuda:1").unwrap();
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
                let ordinal: usize = s.strip_prefix("cuda:").unwrap().parse().map_err(|_| {
                    OCRError::ConfigError {
                        message: format!("Invalid CUDA device ordinal in '{}'", s),
                    }
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
            LayoutElementType::Table => format_table(text),
            LayoutElementType::Formula => format_formula(text),
            LayoutElementType::Image | LayoutElementType::Chart | LayoutElementType::Seal => {
                format_figure(text, i)
            }
            LayoutElementType::List => format_list(text),
            LayoutElementType::Algorithm => format_code(text),
            _ => format_text(text),
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
static OPENOCR_TITLE_RE_PATTERN: Lazy<Regex> = Lazy::new(|| {
    // Note: Rust's `regex` is stricter about escapes inside character classes than Python `re`.
    // Keep the pattern semantically identical while avoiding unnecessary escapes.
    Regex::new(
        r"^\s*((?:[1-9][0-9]*(?:\.[1-9][0-9]*)*[.、]?|[(（](?:[1-9][0-9]*|[一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+)[)）]|[一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+[、.]?|(?:I|II|III|IV|V|VI|VII|VIII|IX|X)\.?))(\s*)(.*)$",
    )
    .expect("OPENOCR_TITLE_RE_PATTERN must compile")
});

fn openocr_format_title(text: &str) -> String {
    fn should_treat_prefix_as_numbering(prefix: &str, suffix_ws: &str) -> bool {
        if !suffix_ws.is_empty() {
            return true;
        }

        if prefix.starts_with('(') || prefix.starts_with('（') {
            return true;
        }

        if prefix.ends_with('.')
            || prefix.ends_with('、')
            || prefix.ends_with(')')
            || prefix.ends_with('）')
        {
            return true;
        }

        // Numeric headings like `1.2Title` should still be treated as numbering even without space.
        if prefix.contains('.') || prefix.contains('、') {
            return true;
        }

        false
    }

    let mut title = text.to_string();
    if let Some(caps) = OPENOCR_TITLE_RE_PATTERN.captures(&title) {
        let numbering_raw = caps.get(1).map(|m| m.as_str()).unwrap_or("");
        let suffix_ws = caps.get(2).map(|m| m.as_str()).unwrap_or("");
        let numbering = numbering_raw.trim();
        let title_content = caps.get(3).map(|m| m.as_str()).unwrap_or("").trim_start();
        if !numbering.is_empty() && should_treat_prefix_as_numbering(numbering, suffix_ws) {
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
        "<table border=1 style='margin: auto; width: max-content;'>",
    );
    table_content = table_content.replace("<th>", "<th style='text-align: center;'>");
    table_content = table_content.replace("<td>", "<td style='text-align: center;'>");
    // PaddleX outputs compact single-line table HTML in markdown (no tag-newlines).
    let re = Regex::new(r">\s*\n+\s*").unwrap_or_else(|_| Regex::new(r">").unwrap());
    table_content = re.replace_all(&table_content, ">").to_string();
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
/// This matches `PaddleOCRVLResult._to_markdown(pretty=...)` when labels come from PP-DocLayoutV2.
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
                    // PaddleX table blocks typically include extra trailing blank lines (`\n\n\n`)
                    // in the markdown output (see OpenOCR `infer_doc.py` + PaddleX markdown export).
                    let compact = openocr_format_table_center_func(content)
                        .trim_end_matches('\n')
                        .to_string();
                    format!("\n{}\n\n\n", compact)
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
    let processed = process_text(cleaned.trim());
    format!("{} {}", prefix, processed)
}

fn format_table(text: &str) -> String {
    let mut result = text.to_string();
    result = result.replace("<tdcolspan=", "<td colspan=");
    result = result.replace("<tdrowspan=", "<td rowspan=");
    result = result.replace("\"colspan=", "\" colspan=");
    result = clean_special_tokens(&result);
    result = result.replace("\\(", "$").replace("\\)", "$");
    result = result.replace("\\[", "$$").replace("\\]", "$$");
    let re = Regex::new(r">\s*\n+\s*").unwrap_or_else(|_| Regex::new(r">").unwrap());
    result = re.replace_all(&result, ">").to_string();
    result
}

fn format_formula(text: &str) -> String {
    let mut result = text.to_string();
    result = clean_special_tokens(&result);
    result = result.replace(r"\upmu", r"\mu");
    result = result.replace("\\]", "");
    result = result.replace("\\[", "");
    result = result.replace("\\)", "");
    result = result.replace("\\(", "");
    result = result.trim().trim_matches('$').trim().to_string();
    result = result.replace('\n', "\\\\\n");
    result = fix_latex_brackets(&result);
    format!("$${}$$", result)
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

fn format_text(text: &str) -> String {
    let mut result = clean_special_tokens(text);
    if result.contains("\\(") && result.contains("\\)") {
        result = result.replace("\\(", " $ ").replace("\\)", " $ ");
    }
    if result.contains("\\[") && result.contains("\\]") {
        result = result.replace("\\[", " $$ ").replace("\\]", " $$ ");
    }
    result = result.replace("$\\bullet$", "•");
    if result.contains("<table>") {
        let re = Regex::new(r"</?(table|tr|th|td|thead|tbody|tfoot)[^>]*>")
            .unwrap_or_else(|_| Regex::new(r"<table>").unwrap());
        result = re.replace_all(&result, "").to_string();
    }
    process_text(&result)
}

fn clean_special_tokens(text: &str) -> String {
    text.replace("-<|sn|>", "")
        .replace("<|sn|>", " ")
        .replace("<|unk|>", "")
        .replace('\u{FFFF}', "")
}

fn process_text(text: &str) -> String {
    let mut result = text.to_string();
    let underscore_re = Regex::new(r"_{4,}").unwrap_or_else(|_| Regex::new(r"_").unwrap());
    result = underscore_re.replace_all(&result, "___").to_string();
    let dots_re = Regex::new(r"\.{4,}").unwrap_or_else(|_| Regex::new(r"\.").unwrap());
    result = dots_re.replace_all(&result, "...").to_string();
    result.trim().to_string()
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

fn fix_latex_brackets(text: &str) -> String {
    let pattern = Regex::new(
        r"\\(big|Big|bigg|Bigg|bigl|bigr|Bigl|Bigr|biggr|biggl|Biggl|Biggr)\{(\\?[{}\[\]\(\)\|])\}",
    )
    .unwrap_or_else(|_| Regex::new(r"\\big").unwrap());
    pattern.replace_all(text, r"\$1$2").to_string()
}

/// Truncate repeated tail patterns.
pub fn truncate_repeated_tail(text: &str, threshold: usize, keep: usize) -> String {
    if text.is_empty() {
        return text.to_string();
    }
    let max_pattern_len = (text.len() / threshold).min(100);
    for pattern_len in 1..=max_pattern_len {
        if text.len() < pattern_len {
            break;
        }
        let pattern = &text[text.len() - pattern_len..];
        let mut count = 0;
        let mut pos = text.len();
        while pos >= pattern_len {
            if &text[pos - pattern_len..pos] == pattern {
                count += 1;
                pos -= pattern_len;
            } else {
                break;
            }
        }
        if count > threshold {
            let non_repeat_part = &text[..pos];
            let kept_repeats = pattern.repeat(keep);
            return format!("{}{}", non_repeat_part, kept_repeats);
        }
    }
    text.to_string()
}

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
    let gray: GrayImage = image::imageops::grayscale(img);
    let (min_val, max_val) = gray.pixels().fold((255u8, 0u8), |(min, max), p| {
        (min.min(p.0[0]), max.max(p.0[0]))
    });

    if max_val == min_val {
        return img.clone();
    }

    let threshold = 200u8;
    let mut non_white_coords: Vec<(u32, u32)> = Vec::new();

    for (x, y, pixel) in gray.enumerate_pixels() {
        let normalized = ((pixel.0[0] as f32 - min_val as f32) / (max_val as f32 - min_val as f32)
            * 255.0) as u8;
        if normalized < threshold {
            non_white_coords.push((x, y));
        }
    }

    if non_white_coords.is_empty() {
        return img.clone();
    }

    let x_min = non_white_coords.iter().map(|(x, _)| *x).min().unwrap_or(0);
    let y_min = non_white_coords.iter().map(|(_, y)| *y).min().unwrap_or(0);
    let x_max = non_white_coords
        .iter()
        .map(|(x, _)| *x)
        .max()
        .unwrap_or(img.width());
    let y_max = non_white_coords
        .iter()
        .map(|(_, y)| *y)
        .max()
        .unwrap_or(img.height());

    let width = (x_max - x_min + 1).min(img.width() - x_min);
    let height = (y_max - y_min + 1).min(img.height() - y_min);

    if width == 0 || height == 0 {
        return img.clone();
    }

    image::imageops::crop_imm(img, x_min, y_min, width, height).to_image()
}

fn find_shortest_repeating_substring(s: &str) -> Option<String> {
    let n = s.len();
    for i in 1..=n / 2 {
        if n.is_multiple_of(i) {
            let substring = &s[..i];
            if substring.repeat(n / i) == s {
                return Some(substring.to_string());
            }
        }
    }
    None
}

fn find_repeating_suffix(
    s: &str,
    min_len: usize,
    min_repeats: usize,
) -> Option<(String, String, usize)> {
    for i in (min_len..=s.len() / min_repeats).rev() {
        if s.len() < i * min_repeats {
            continue;
        }
        let unit = &s[s.len() - i..];
        let pattern = unit.repeat(min_repeats);
        if s.ends_with(&pattern) {
            let mut count = 0;
            let mut temp_s = s;
            while temp_s.ends_with(unit) {
                temp_s = &temp_s[..temp_s.len() - i];
                count += 1;
            }
            let start_index = s.len() - (count * i);
            return Some((s[..start_index].to_string(), unit.to_string(), count));
        }
    }
    None
}

/// Detect and truncate repetitive content.
pub fn truncate_repetitive_content(
    content: &str,
    line_threshold: usize,
    char_threshold: usize,
    min_len: usize,
) -> String {
    let stripped = content.trim();
    if stripped.is_empty() {
        return content.to_string();
    }

    if !stripped.contains('\n')
        && stripped.len() > 100
        && let Some((prefix, unit, count)) = find_repeating_suffix(stripped, 8, 5)
        && unit.len() * count > stripped.len() / 2
    {
        return prefix;
    }

    if !stripped.contains('\n')
        && stripped.len() > min_len
        && let Some(unit) = find_shortest_repeating_substring(stripped)
    {
        let count = stripped.len() / unit.len();
        if count >= char_threshold {
            return unit;
        }
    }

    let lines: Vec<&str> = content
        .lines()
        .map(|l| l.trim())
        .filter(|l| !l.is_empty())
        .collect();

    if lines.is_empty() {
        return content.to_string();
    }

    let total_lines = lines.len();
    if total_lines < line_threshold {
        return content.to_string();
    }

    let mut counts = std::collections::HashMap::new();
    for line in &lines {
        *counts.entry(*line).or_insert(0usize) += 1;
    }

    if let Some((most_common, count)) = counts.into_iter().max_by_key(|(_, c)| *c)
        && count >= line_threshold
        && (count as f32 / total_lines as f32) >= 0.8
    {
        return most_common.to_string();
    }

    content.to_string()
}

const OTSL_NL: &str = "<nl>";
const OTSL_FCEL: &str = "<fcel>";
const OTSL_ECEL: &str = "<ecel>";
const OTSL_LCEL: &str = "<lcel>";
const OTSL_UCEL: &str = "<ucel>";
const OTSL_XCEL: &str = "<xcel>";

fn is_otsl_tag(s: &str) -> bool {
    matches!(
        s,
        "<nl>" | "<fcel>" | "<ecel>" | "<lcel>" | "<ucel>" | "<xcel>"
    )
}

fn otsl_extract_tokens_and_text(s: &str) -> (Vec<String>, Vec<String>) {
    let pattern = Regex::new(r"(<nl>|<fcel>|<ecel>|<lcel>|<ucel>|<xcel>)").unwrap();
    let tokens: Vec<String> = pattern
        .find_iter(s)
        .map(|m| m.as_str().to_string())
        .collect();
    let text_parts: Vec<String> = pattern
        .split(s)
        .map(|p| p.to_string())
        .filter(|p| !p.trim().is_empty())
        .collect();
    (tokens, text_parts)
}

#[derive(Debug, Clone)]
struct TableCell {
    row_span: usize,
    col_span: usize,
    start_row: usize,
    end_row: usize,
    start_col: usize,
    end_col: usize,
    text: String,
}

impl Default for TableCell {
    fn default() -> Self {
        Self {
            row_span: 1,
            col_span: 1,
            start_row: 0,
            end_row: 1,
            start_col: 0,
            end_col: 1,
            text: String::new(),
        }
    }
}

fn otsl_parse_texts(texts: &[String], tokens: &[String]) -> (Vec<TableCell>, Vec<Vec<String>>) {
    let mut split_row_tokens: Vec<Vec<String>> = Vec::new();
    let mut current_row: Vec<String> = Vec::new();

    for token in tokens {
        if token == OTSL_NL {
            if !current_row.is_empty() {
                split_row_tokens.push(current_row);
                current_row = Vec::new();
            }
        } else {
            current_row.push(token.clone());
        }
    }
    if !current_row.is_empty() {
        split_row_tokens.push(current_row);
    }

    if split_row_tokens.is_empty() {
        return (Vec::new(), split_row_tokens);
    }

    let max_cols = split_row_tokens.iter().map(|r| r.len()).max().unwrap_or(0);
    for row in &mut split_row_tokens {
        while row.len() < max_cols {
            row.push(OTSL_ECEL.to_string());
        }
    }

    let mut table_cells: Vec<TableCell> = Vec::new();
    let mut text_idx = 0usize;

    let mut combined_texts: Vec<String> = Vec::new();
    for row in &split_row_tokens {
        for token in row {
            combined_texts.push(token.clone());
            if text_idx < texts.len() && &texts[text_idx] == token {
                text_idx += 1;
                if text_idx < texts.len() && !is_otsl_tag(&texts[text_idx]) {
                    combined_texts.push(texts[text_idx].clone());
                    text_idx += 1;
                }
            }
        }
        combined_texts.push(OTSL_NL.to_string());
        if text_idx < texts.len() && texts[text_idx] == OTSL_NL {
            text_idx += 1;
        }
    }

    let count_right = |tokens: &[Vec<String>], c: usize, r: usize, which: &[&str]| -> usize {
        let mut span = 0;
        let mut c_iter = c;
        while c_iter < tokens[r].len() && which.contains(&tokens[r][c_iter].as_str()) {
            span += 1;
            c_iter += 1;
        }
        span
    };

    let count_down = |tokens: &[Vec<String>], c: usize, r: usize, which: &[&str]| -> usize {
        let mut span = 0;
        let mut r_iter = r;
        while r_iter < tokens.len()
            && c < tokens[r_iter].len()
            && which.contains(&tokens[r_iter][c].as_str())
        {
            span += 1;
            r_iter += 1;
        }
        span
    };

    let mut r_idx = 0usize;
    let mut c_idx = 0usize;
    let mut i = 0;
    while i < combined_texts.len() {
        let text = &combined_texts[i];

        if text == OTSL_FCEL || text == OTSL_ECEL {
            let mut row_span = 1usize;
            let mut col_span = 1usize;
            let mut cell_text = String::new();
            let mut right_offset = 1;

            if text != OTSL_ECEL
                && i + 1 < combined_texts.len()
                && !is_otsl_tag(&combined_texts[i + 1])
            {
                cell_text = combined_texts[i + 1].clone();
                right_offset = 2;
            }

            let next_right_cell = if i + right_offset < combined_texts.len() {
                &combined_texts[i + right_offset]
            } else {
                ""
            };

            let next_bottom_cell = if r_idx + 1 < split_row_tokens.len()
                && c_idx < split_row_tokens[r_idx + 1].len()
            {
                &split_row_tokens[r_idx + 1][c_idx]
            } else {
                ""
            };

            if next_right_cell == OTSL_LCEL || next_right_cell == OTSL_XCEL {
                col_span +=
                    count_right(&split_row_tokens, c_idx + 1, r_idx, &[OTSL_LCEL, OTSL_XCEL]);
            }

            if next_bottom_cell == OTSL_UCEL || next_bottom_cell == OTSL_XCEL {
                row_span +=
                    count_down(&split_row_tokens, c_idx, r_idx + 1, &[OTSL_UCEL, OTSL_XCEL]);
            }

            table_cells.push(TableCell {
                row_span,
                col_span,
                start_row: r_idx,
                end_row: r_idx + row_span,
                start_col: c_idx,
                end_col: c_idx + col_span,
                text: cell_text.trim().to_string(),
            });
        }

        if text == OTSL_FCEL
            || text == OTSL_ECEL
            || text == OTSL_LCEL
            || text == OTSL_UCEL
            || text == OTSL_XCEL
        {
            c_idx += 1;
        }

        if text == OTSL_NL {
            r_idx += 1;
            c_idx = 0;
        }

        i += 1;
    }

    (table_cells, split_row_tokens)
}

fn export_to_html(cells: &[TableCell], num_rows: usize, num_cols: usize) -> String {
    if cells.is_empty() {
        return String::new();
    }

    let mut grid: Vec<Vec<Option<&TableCell>>> = vec![vec![None; num_cols]; num_rows];
    for cell in cells {
        for row in grid
            .iter_mut()
            .take(cell.end_row.min(num_rows))
            .skip(cell.start_row)
        {
            for col in row
                .iter_mut()
                .take(cell.end_col.min(num_cols))
                .skip(cell.start_col)
            {
                *col = Some(cell);
            }
        }
    }

    let mut body = String::new();
    for (i, row) in grid.iter().enumerate().take(num_rows) {
        body.push_str("<tr>");
        for (j, col) in row.iter().enumerate().take(num_cols) {
            if let Some(cell) = col {
                if cell.start_row != i || cell.start_col != j {
                    continue;
                }

                let content = html_escape::encode_text(&cell.text);
                let mut opening_tag = String::from("td");

                if cell.row_span > 1 {
                    opening_tag.push_str(&format!(" rowspan=\"{}\"", cell.row_span));
                }
                if cell.col_span > 1 {
                    opening_tag.push_str(&format!(" colspan=\"{}\"", cell.col_span));
                }

                body.push_str(&format!("<{}>{}</td>", opening_tag, content));
            }
        }
        body.push_str("</tr>");
    }

    format!("<table>{}</table>", body)
}

fn otsl_pad_to_square(otsl_str: &str) -> String {
    let otsl_str = otsl_str.trim();
    if !otsl_str.contains(OTSL_NL) {
        return format!("{}{}", otsl_str, OTSL_NL);
    }

    let lines: Vec<&str> = otsl_str.split(OTSL_NL).filter(|l| !l.is_empty()).collect();
    let mut row_data: Vec<(Vec<&str>, usize, usize)> = Vec::new();

    let cell_pattern = Regex::new(r"<fcel>|<ecel>|<lcel>|<ucel>|<xcel>").unwrap();

    for line in &lines {
        let raw_cells: Vec<&str> = cell_pattern.find_iter(line).map(|m| m.as_str()).collect();
        if raw_cells.is_empty() {
            continue;
        }

        let total_len = raw_cells.len();
        let mut min_len = 0;
        for (i, cell) in raw_cells.iter().enumerate() {
            if *cell == OTSL_FCEL {
                min_len = i + 1;
            }
        }
        row_data.push((raw_cells, total_len, min_len));
    }

    if row_data.is_empty() {
        return OTSL_NL.to_string();
    }

    let global_min_width = row_data.iter().map(|(_, _, m)| *m).max().unwrap_or(0);
    let max_total_len = row_data.iter().map(|(_, t, _)| *t).max().unwrap_or(0);

    let search_start = global_min_width;
    let search_end = global_min_width.max(max_total_len);

    let mut min_total_cost = usize::MAX;
    let mut optimal_width = search_end;

    for width in search_start..=search_end {
        let current_cost: usize = row_data
            .iter()
            .map(|(_, total, _)| (*total as isize - width as isize).unsigned_abs())
            .sum();
        if current_cost < min_total_cost {
            min_total_cost = current_cost;
            optimal_width = width;
        }
    }

    let mut repaired_lines: Vec<String> = Vec::new();
    for (cells, _, _) in &row_data {
        let current_len = cells.len();
        let new_cells: Vec<&str> = if current_len > optimal_width {
            cells[..optimal_width].to_vec()
        } else {
            let mut padded = cells.clone();
            while padded.len() < optimal_width {
                padded.push(OTSL_ECEL);
            }
            padded
        };
        repaired_lines.push(new_cells.join(""));
    }

    format!("{}{}", repaired_lines.join(OTSL_NL), OTSL_NL)
}

/// Convert OTSL table format to HTML.
pub fn convert_otsl_to_html(otsl_content: &str) -> String {
    if otsl_content.contains("<table") {
        return clean_html_table(otsl_content);
    }

    if !otsl_content.contains("<fcel>") && !otsl_content.contains("<ecel>") {
        return simple_otsl_conversion(otsl_content);
    }

    let padded = otsl_pad_to_square(otsl_content);
    let (tokens, mixed_texts) = otsl_extract_tokens_and_text(&padded);
    let (table_cells, split_row_tokens) = otsl_parse_texts(&mixed_texts, &tokens);

    let num_rows = split_row_tokens.len();
    let num_cols = split_row_tokens.iter().map(|r| r.len()).max().unwrap_or(0);

    export_to_html(&table_cells, num_rows, num_cols)
}

fn simple_otsl_conversion(text: &str) -> String {
    let mut html = String::from("<table>");
    for line in text.lines() {
        if line.trim().is_empty() {
            continue;
        }
        html.push_str("<tr>");
        for cell in line.split('\t') {
            html.push_str("<td>");
            html.push_str(cell.trim());
            html.push_str("</td>");
        }
        html.push_str("</tr>");
    }
    html.push_str("</table>");
    html
}

fn clean_html_table(text: &str) -> String {
    let mut result = text.to_string();
    result = result.replace("<tdcolspan=", "<td colspan=");
    result = result.replace("<tdrowspan=", "<td rowspan=");
    result = result.replace("\"colspan=", "\" colspan=");
    result = result.replace("<|sn|>", "");
    result = result.replace("<|unk|>", "");
    result = result.replace('\u{FFFF}', "");
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_heading() {
        assert_eq!(format_heading("Title", 1), "# Title");
        assert_eq!(format_heading("Section", 2), "## Section");
    }

    #[test]
    fn test_format_formula() {
        assert_eq!(format_formula("x + y = z"), "$$x + y = z$$");
        assert_eq!(format_formula("\\[x^2\\]"), "$$x^2$$");
    }

    #[test]
    fn test_clean_special_tokens() {
        assert_eq!(clean_special_tokens("hello<|sn|>world"), "hello world");
        assert_eq!(clean_special_tokens("test<|unk|>"), "test");
    }

    #[test]
    fn test_truncate_repeated_tail() {
        let text = "hello".to_string() + &"!".repeat(50);
        let result = truncate_repeated_tail(&text, 20, 1);
        assert_eq!(result, "hello!");
    }

    #[test]
    fn test_calculate_overlap_ratio() {
        let bbox1 = BoundingBox::from_coords(0.0, 0.0, 10.0, 10.0);
        let bbox2 = BoundingBox::from_coords(5.0, 5.0, 15.0, 15.0);
        let ratio = calculate_overlap_ratio(&bbox1, &bbox2, "union");
        assert!(ratio > 0.0 && ratio < 1.0);
    }

    #[test]
    fn test_truncate_repetitive_content() {
        let text = "hello\nhello\nhello\nhello\nhello\nhello\nhello\nhello\nhello\nhello\nhello";
        let result = truncate_repetitive_content(text, 10, 10, 10);
        assert_eq!(result, "hello");
    }

    #[test]
    fn test_find_shortest_repeating() {
        assert_eq!(
            find_shortest_repeating_substring("abcabcabc"),
            Some("abc".to_string())
        );
        assert_eq!(find_shortest_repeating_substring("hello"), None);
    }

    #[test]
    fn test_simple_otsl_conversion() {
        let input = "a\tb\tc\nd\te\tf";
        let html = simple_otsl_conversion(input);
        assert!(html.contains("<table>"));
        assert!(html.contains("<td>a</td>"));
    }

    #[test]
    fn test_openocr_format_title_numbering_heuristics() {
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
