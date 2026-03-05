//! Document structure analysis result types.
//!
//! This module defines the result types for document structure analysis,
//! including layout detection, table recognition, and formula recognition.

use super::text_region::TextRegion;
use crate::processors::BoundingBox;
use image::RgbImage;
use once_cell::sync::Lazy;
use regex::Regex;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::Path;
use std::sync::Arc;

/// Title numbering pattern for detecting section numbers like 1, 1.2, 1.2.3, (1), 一、etc.
/// This follows standard title numbering pattern.
static TITLE_NUMBERING_REGEX: Lazy<Regex> = Lazy::new(|| {
    Regex::new(
        r"(?x)
        ^\s*
        (
            # Arabic numerals: 1, 1.2, 1.2.3, etc.
            [1-9][0-9]*(?:\.[1-9][0-9]*)*[\.、]?
            |
            # Parenthesized Arabic numerals: (1), (1.2), etc.
            [(（][1-9][0-9]*(?:\.[1-9][0-9]*)*[)）]
            |
            # Chinese numerals with punctuation: 一、 二、
            [一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾][、.]?
            |
            # Parenthesized Chinese numerals: （一）
            [(（][一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+[)）]
            |
            # Roman numerals with delimiter (period or followed by space)
            (?:I|II|III|IV|V|VI|VII|VIII|IX|X)(?:\.|\b)
        )
        (\s+)
        (.*)
        $
    ",
    )
    .unwrap_or_else(|e| panic!("Invalid title numbering regex: {e}"))
});

/// Format a paragraph title with automatic level detection based on numbering.
///
/// Following PaddleX's title formatting logic:
/// - Extracts numbering prefix (1.2.3, etc.)
/// - Determines heading level from number of dots
/// - Returns (level, formatted_title) where level starts from 2 (## for paragraph titles)
///
/// PaddleX logic: `level = dots + 1`, then uses `#{'#' * level}` which means:
/// - "1 Introduction" (0 dots) -> level=1 -> `## 1 Introduction`
/// - "2.1 Method" (1 dot) -> level=2 -> `### 2.1 Method`
/// - "2.1.1 Details" (2 dots) -> level=3 -> `#### 2.1.1 Details`
///
/// To align with PaddleX, we return level+1 to account for the extra `#`:
/// - "1 Introduction" -> (2, "1 Introduction") -> `## 1 Introduction`
/// - "2.1 Method" -> (3, "2.1 Method") -> `### 2.1 Method`
/// - "2.1.1 Details" -> (4, "2.1.1 Details") -> `#### 2.1.1 Details`
fn semantic_title_level_and_format(cleaned: &str) -> Option<(usize, String)> {
    let trimmed = cleaned.trim();

    // Common unnumbered top-level section headers.
    let keyword = trimmed.trim_end_matches(':').to_ascii_uppercase();
    if matches!(
        keyword.as_str(),
        "ABSTRACT" | "INTRODUCTION" | "REFERENCES" | "REFERENCE"
    ) {
        return Some((1, trimmed.to_string()));
    }

    if let Some(captures) = TITLE_NUMBERING_REGEX.captures(cleaned) {
        let numbering = captures.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        let title_content = captures.get(3).map(|m| m.as_str()).unwrap_or("");

        let dot_count = numbering.matches('.').count();
        let level = (dot_count + 2).clamp(2, 6);

        let formatted = if title_content.is_empty() {
            numbering.trim_end_matches('.').to_string()
        } else {
            format!(
                "{} {}",
                numbering.trim_end_matches('.'),
                title_content.trim_start()
            )
        };
        return Some((level, formatted));
    }

    None
}

fn semantic_title_level(text: &str) -> Option<usize> {
    let cleaned = text.replace("-\n", "").replace('\n', " ");
    semantic_title_level_and_format(&cleaned).map(|(level, _)| level)
}

fn format_title_with_level(title: &str, clustered_level: Option<usize>) -> (usize, String) {
    // Clean up line breaks
    let cleaned = title.replace("-\n", "").replace('\n', " ");
    if let Some((level, formatted)) = semantic_title_level_and_format(&cleaned) {
        return (level, formatted);
    }

    // No semantic signal: use voting hint from relative/font-size signals.
    let level = clustered_level.unwrap_or(2).clamp(2, 6);
    (level, cleaned)
}

/// Estimate per-title heading levels using three-signal voting:
/// 1) semantic numbering/keyword level
/// 2) relative indentation order
/// 3) font-size k-means (k<=4)
///
fn infer_paragraph_title_levels(elements: &[LayoutElement]) -> HashMap<usize, usize> {
    let title_indices: Vec<usize> = elements
        .iter()
        .enumerate()
        .filter(|(_, e)| e.element_type == LayoutElementType::ParagraphTitle)
        .map(|(idx, _)| idx)
        .collect();
    if title_indices.is_empty() {
        return HashMap::new();
    }

    let height_samples: Vec<(usize, f32)> = title_indices
        .iter()
        .filter_map(|&idx| {
            let e = &elements[idx];
            let height = (e.bbox.y_max() - e.bbox.y_min()).max(1.0);
            let line_h = height / e.num_lines.unwrap_or(1).max(1) as f32;
            let v = line_h.max(1.0);
            if v.is_finite() { Some((idx, v)) } else { None }
        })
        .collect();

    let indent_samples: Vec<(usize, f32)> = title_indices
        .iter()
        .filter_map(|&idx| {
            let x = elements[idx].bbox.x_min();
            if x.is_finite() { Some((idx, x)) } else { None }
        })
        .collect();
    let semantic_levels: HashMap<usize, usize> = title_indices
        .iter()
        .filter_map(|&idx| {
            elements[idx]
                .text
                .as_deref()
                .and_then(semantic_title_level)
                .map(|level| (idx, level))
        })
        .collect();

    let font_levels = infer_levels_by_kmeans_feature(&height_samples, true);
    // Smaller x_min (less indent) -> higher-level heading.
    let relative_levels = infer_levels_by_kmeans_feature(&indent_samples, false);

    let mut voted = HashMap::new();
    for idx in title_indices {
        let semantic_level = semantic_levels.get(&idx).copied();
        let font_level = font_levels.get(&idx).copied();
        let relative_level = relative_levels.get(&idx).copied();

        let mut score = [0u8; 7];
        if let Some(level) = semantic_level {
            score[level.clamp(1, 6)] += 2;
        }
        if let Some(level) = font_level {
            score[level.clamp(1, 6)] += 1;
        }
        if let Some(level) = relative_level {
            score[level.clamp(1, 6)] += 1;
        }

        let mut best_level = semantic_level.unwrap_or(2);
        let mut best_score = 0u8;
        for (level, &s) in score.iter().enumerate().skip(1) {
            if s > best_score {
                best_score = s;
                best_level = level;
            } else if s == best_score && s > 0 {
                let is_semantic = semantic_level == Some(level);
                let best_is_semantic = semantic_level == Some(best_level);
                if (is_semantic && !best_is_semantic)
                    || (is_semantic == best_is_semantic && level < best_level)
                {
                    best_level = level;
                }
            }
        }

        if best_score == 0 {
            best_level = semantic_level
                .or(font_level)
                .or(relative_level)
                .unwrap_or(2);
        }

        voted.insert(idx, best_level.clamp(1, 6));
    }

    voted
}

/// Cluster one scalar feature into heading levels with 1D k-means.
///
/// `descending=true` means larger feature -> higher-level heading (smaller markdown depth).
/// `descending=false` means smaller feature -> higher-level heading.
fn infer_levels_by_kmeans_feature(
    samples: &[(usize, f32)],
    descending: bool,
) -> HashMap<usize, usize> {
    let clean_samples: Vec<(usize, f32)> = samples
        .iter()
        .copied()
        .filter(|(_, v)| v.is_finite())
        .collect();
    if clean_samples.len() < 2 {
        return HashMap::new();
    }

    let mut values: Vec<f32> = clean_samples.iter().map(|(_, v)| *v).collect();
    values.sort_by(|a, b| a.total_cmp(b));
    let unique_count = values
        .windows(2)
        .filter(|w| (w[1] - w[0]).abs() > 1e-3)
        .count()
        + 1;
    let k = unique_count.clamp(1, 4).min(clean_samples.len());
    if k <= 1 {
        return HashMap::new();
    }

    let mut centroids = (0..k)
        .map(|i| {
            let pos = ((i as f32 + 0.5) / k as f32 * values.len() as f32).floor() as usize;
            values[pos.min(values.len() - 1)]
        })
        .collect::<Vec<_>>();

    for _ in 0..16 {
        let mut sums = vec![0.0f32; k];
        let mut counts = vec![0usize; k];
        for (_, value) in &clean_samples {
            let mut best_idx = 0usize;
            let mut best_dist = f32::INFINITY;
            for (idx, c) in centroids.iter().enumerate() {
                let dist = (value - c).abs();
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = idx;
                }
            }
            sums[best_idx] += *value;
            counts[best_idx] += 1;
        }
        for idx in 0..k {
            if counts[idx] > 0 {
                centroids[idx] = sums[idx] / counts[idx] as f32;
            }
        }
    }

    let mut centroid_order: Vec<(usize, f32)> = centroids.iter().copied().enumerate().collect();
    if descending {
        centroid_order.sort_by(|a, b| b.1.total_cmp(&a.1));
    } else {
        centroid_order.sort_by(|a, b| a.1.total_cmp(&b.1));
    }
    let rank_by_cluster: HashMap<usize, usize> = centroid_order
        .into_iter()
        .enumerate()
        .map(|(rank, (cluster_idx, _))| (cluster_idx, rank))
        .collect();

    let mut result = HashMap::new();
    for (element_idx, value) in &clean_samples {
        let mut best_idx = 0usize;
        let mut best_dist = f32::INFINITY;
        for (idx, c) in centroids.iter().enumerate() {
            let dist = (value - c).abs();
            if dist < best_dist {
                best_dist = dist;
                best_idx = idx;
            }
        }
        let rank = rank_by_cluster.get(&best_idx).copied().unwrap_or(0);
        let level = (rank + 2).clamp(2, 6);
        result.insert(*element_idx, level);
    }

    result
}

/// A detected document region block (from PP-DocBlockLayout).
///
/// Region blocks represent hierarchical groupings of layout elements,
/// typically columns or logical sections of a document. They are used
/// for hierarchical reading order determination.
///
/// # PP-StructureV3 Alignment
///
/// PP-DocBlockLayout detects "region" type blocks that group related
/// layout elements together. Elements within the same region should
/// be read together before moving to the next region.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RegionBlock {
    /// Bounding box of the region
    pub bbox: BoundingBox,
    /// Confidence score of the detection
    pub confidence: f32,
    /// Index of this region in the reading order
    pub order_index: Option<u32>,
    /// Indices of layout elements that belong to this region
    pub element_indices: Vec<usize>,
}

/// Page continuation flags for multi-page document processing.
///
/// These flags indicate whether the page starts or ends in the middle of
/// a semantic paragraph, which is crucial for properly concatenating
/// markdown output from multiple pages.
///
/// - `paragraph_start`: `false` means this page continues a paragraph from previous page
/// - `paragraph_end`: `false` means this page's content continues to next page
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PageContinuationFlags {
    /// Whether the first element on this page is a paragraph continuation
    pub paragraph_start: bool,
    /// Whether the last element on this page continues to the next page
    pub paragraph_end: bool,
}

impl PageContinuationFlags {
    pub fn new(paragraph_start: bool, paragraph_end: bool) -> Self {
        Self {
            paragraph_start,
            paragraph_end,
        }
    }

    /// Returns the tuple format (is_start, is_end) for compatibility
    pub fn as_tuple(&self) -> (bool, bool) {
        (self.paragraph_start, self.paragraph_end)
    }
}

/// Result of document structure analysis.
///
/// This struct contains all the results from analyzing a document's structure,
/// including layout elements, tables, formulas, and OCR results.
///
/// # Coordinate System
///
/// The coordinate system of bounding boxes depends on which preprocessing was applied:
///
/// - **No preprocessing**: Boxes are in the original input image's coordinate system.
///
/// - **Orientation correction only** (`orientation_angle` set, `rectified_img` is None):
///   Boxes are transformed back to the original input image's coordinate system.
///
/// - **Rectification applied** (`rectified_img` is Some):
///   Boxes remain in the **rectified image's coordinate system**. Neural network-based
///   rectification (UVDoc) warps cannot be precisely inverted, so use `rectified_img`
///   for visualization instead of the original image.
///
/// - **Both orientation and rectification**: Boxes are in the rectified coordinate system
///   (rectification takes precedence since it's applied after orientation correction).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructureResult {
    /// Path to the input image file
    pub input_path: Arc<str>,
    /// Index of the image in a batch (0 for single image processing)
    pub index: usize,
    /// Detected layout elements (text regions, tables, figures, etc.)
    pub layout_elements: Vec<LayoutElement>,
    /// Recognized tables with their structure and content
    pub tables: Vec<TableResult>,
    /// Recognized mathematical formulas
    pub formulas: Vec<FormulaResult>,
    /// OCR text regions (if OCR was integrated)
    pub text_regions: Option<Vec<TextRegion>>,
    /// Document orientation angle (if orientation correction was used)
    pub orientation_angle: Option<f32>,
    /// Detected region blocks for hierarchical ordering (PP-DocBlockLayout)
    /// When present, layout_elements are already sorted by region hierarchy
    pub region_blocks: Option<Vec<RegionBlock>>,
    /// Rectified image (if document rectification was used)
    /// Note: Bounding boxes are already transformed back to original coordinates for rotation,
    /// but for rectification (UVDoc), boxes are in the rectified image's coordinate system.
    /// Use this image for visualization when rectification was applied.
    #[serde(skip)]
    pub rectified_img: Option<Arc<RgbImage>>,
    /// Page continuation flags for multi-page document processing.
    /// This indicates whether this page continues a paragraph from the previous page
    /// or continues to the next page, which is crucial for proper markdown concatenation.
    pub page_continuation_flags: Option<PageContinuationFlags>,
}

impl StructureResult {
    /// Creates a new structure result.
    pub fn new(input_path: impl Into<Arc<str>>, index: usize) -> Self {
        Self {
            input_path: input_path.into(),
            index,
            layout_elements: Vec::new(),
            tables: Vec::new(),
            formulas: Vec::new(),
            text_regions: None,
            orientation_angle: None,
            region_blocks: None,
            rectified_img: None,
            page_continuation_flags: None,
        }
    }

    /// Adds layout elements to the result.
    pub fn with_layout_elements(mut self, elements: Vec<LayoutElement>) -> Self {
        self.layout_elements = elements;
        self
    }

    /// Adds tables to the result.
    pub fn with_tables(mut self, tables: Vec<TableResult>) -> Self {
        self.tables = tables;
        self
    }

    /// Adds formulas to the result.
    pub fn with_formulas(mut self, formulas: Vec<FormulaResult>) -> Self {
        self.formulas = formulas;
        self
    }

    /// Adds OCR text regions to the result.
    pub fn with_text_regions(mut self, regions: Vec<TextRegion>) -> Self {
        self.text_regions = Some(regions);
        self
    }

    /// Adds region blocks to the result (PP-DocBlockLayout).
    ///
    /// Region blocks represent hierarchical groupings of layout elements.
    /// When set, layout_elements should already be sorted by region hierarchy.
    pub fn with_region_blocks(mut self, blocks: Vec<RegionBlock>) -> Self {
        self.region_blocks = Some(blocks);
        self
    }

    /// Sets page continuation flags for multi-page document processing.
    pub fn with_page_continuation_flags(mut self, flags: PageContinuationFlags) -> Self {
        self.page_continuation_flags = Some(flags);
        self
    }

    /// Converts the result to a Markdown string.
    ///
    /// Follows PP-StructureV3's formatting rules:
    /// - DocTitle: `# title`
    /// - ParagraphTitle: Auto-detect numbering (1.2.3 -> ###)
    /// - Formula: `$$latex$$`
    /// - Table: HTML with border
    /// - Images: `![Figure](caption)`
    ///
    /// Note: Low-confidence text elements that overlap with table regions are filtered out
    /// to avoid duplicate content from table OCR.
    pub fn to_markdown(&self) -> String {
        // Collect table bboxes for overlap filtering
        let table_bboxes: Vec<&BoundingBox> = self
            .layout_elements
            .iter()
            .filter(|e| e.element_type == LayoutElementType::Table)
            .map(|e| &e.bbox)
            .collect();

        // Compute original image width for image scaling (PaddleX: original_image_width)
        let original_image_width = self
            .rectified_img
            .as_ref()
            .map(|img| img.width() as f32)
            .or_else(|| {
                // Estimate from max element x-coordinate
                self.layout_elements
                    .iter()
                    .map(|e| e.bbox.x_max())
                    .fold(None, |acc, x| Some(acc.map_or(x, |max: f32| max.max(x))))
            })
            .unwrap_or(1.0);

        let mut md = String::new();
        let elements = &self.layout_elements;
        let paragraph_title_levels = infer_paragraph_title_levels(elements);
        let mut last_label: Option<LayoutElementType> = None;
        let mut prev_element: Option<&LayoutElement> = None;

        for (idx, element) in elements.iter().enumerate() {
            // PP-StructureV3 markdown ignores auxiliary labels.
            if matches!(
                element.element_type,
                LayoutElementType::Number
                    | LayoutElementType::Footnote
                    | LayoutElementType::Header
                    | LayoutElementType::HeaderImage
                    | LayoutElementType::Footer
                    | LayoutElementType::FooterImage
                    | LayoutElementType::AsideText
            ) {
                continue;
            }

            // Filter out low-confidence text elements that overlap with tables
            // These are typically OCR artifacts from table cell text that shouldn't be
            // output separately in markdown
            if element.element_type == LayoutElementType::Text {
                let overlaps_table = table_bboxes.iter().any(|table_bbox| {
                    element.bbox.ioa(table_bbox) > 0.3 // >30% of text is inside table
                });

                // Skip low-confidence text that overlaps with table regions
                // Standard logic filters these in the stitching phase
                if overlaps_table && element.confidence < 0.7 {
                    continue;
                }
            }

            // Determine seg_start_flag for paragraph continuity (PaddleX get_seg_flag).
            // When both current and previous are "text" and seg_start_flag is false,
            // they belong to the same paragraph — join without \n\n separator.
            let seg_start_flag = get_seg_flag(element, prev_element);

            let is_continuation = element.element_type == LayoutElementType::Text
                && last_label == Some(LayoutElementType::Text)
                && !seg_start_flag;

            // Add separator between elements
            if !is_continuation {
                // Normal case: separate elements with blank line
            }

            match element.element_type {
                // Document title
                LayoutElementType::DocTitle => {
                    if !md.is_empty() {
                        md.push_str("\n\n");
                    }
                    md.push_str("# ");
                    if let Some(text) = &element.text {
                        let cleaned = clean_ocr_text(text);
                        md.push_str(&cleaned);
                    }
                }
                // Paragraph/section title - auto-detect numbering for level
                LayoutElementType::ParagraphTitle => {
                    if !md.is_empty() {
                        md.push_str("\n\n");
                    }
                    if let Some(text) = &element.text {
                        let cleaned = clean_ocr_text(text);
                        let clustered = paragraph_title_levels.get(&idx).copied();
                        let (level, formatted_title) = format_title_with_level(&cleaned, clustered);
                        for _ in 0..level {
                            md.push('#');
                        }
                        md.push(' ');
                        md.push_str(&formatted_title);
                    } else {
                        md.push_str("## ");
                    }
                }
                // Table - preserve HTML structure with border and center alignment
                // Following PaddleX's format with <div style="text-align: center;"> wrapper
                LayoutElementType::Table => {
                    if !md.is_empty() {
                        md.push_str("\n\n");
                    }
                    if let Some(table) =
                        self.tables.iter().find(|t| t.bbox.iou(&element.bbox) > 0.5)
                    {
                        if let Some(html) = &table.html_structure {
                            // Simplify table HTML (remove html/body wrappers) and add border
                            let simplified = simplify_table_html(html);
                            let table_with_border =
                                simplified.replacen("<table>", "<table border=\"1\">", 1);
                            // PaddleX format_centered_by_html: clean newlines then wrap
                            let cleaned = clean_ocr_text(&table_with_border);
                            md.push_str("<div style=\"text-align: center;\">");
                            md.push_str(&cleaned);
                            md.push_str("</div>");
                        } else {
                            md.push_str("[Table]");
                        }
                    } else {
                        md.push_str("[Table]");
                    }
                }
                // FormulaNumber - equation labels like "(1)", "Eq. 1" etc.
                // PaddleX does NOT include formula_number in handle_funcs_dict,
                // so these are silently skipped in markdown output.
                LayoutElementType::FormulaNumber => {
                    continue;
                }
                // Formula - detect inline vs display formula based on context
                LayoutElementType::Formula => {
                    // Extract and clean LaTeX content; skip if empty
                    let raw_content = element.text.as_deref().map(|s| s.trim()).unwrap_or("");
                    if raw_content.is_empty() {
                        continue;
                    }
                    // Remove only outer $$ or $ wrappers if present (from table cell injection)
                    let latex_content =
                        if raw_content.starts_with("$$") && raw_content.ends_with("$$") {
                            &raw_content[2..raw_content.len() - 2]
                        } else if raw_content.starts_with('$') && raw_content.ends_with('$') {
                            &raw_content[1..raw_content.len() - 1]
                        } else {
                            raw_content
                        };

                    // Check if this formula is on the same line as adjacent text elements
                    // to determine if it's an inline formula or display formula
                    let is_inline = {
                        // Look for previous non-formula text element on the same line
                        let has_prev_text = (0..idx).rev().any(|i| {
                            let prev = &elements[i];
                            !prev.element_type.is_formula()
                                && (prev.element_type == LayoutElementType::Text
                                    || prev.element_type == LayoutElementType::ReferenceContent)
                                && is_same_line(&element.bbox, &prev.bbox)
                        });

                        // Look for next non-formula text element on the same line
                        let has_next_text = ((idx + 1)..elements.len()).any(|i| {
                            let next = &elements[i];
                            !next.element_type.is_formula()
                                && (next.element_type == LayoutElementType::Text
                                    || next.element_type == LayoutElementType::ReferenceContent)
                                && is_same_line(&element.bbox, &next.bbox)
                        });

                        has_prev_text || has_next_text
                    };

                    if is_inline {
                        // Inline formula: use $...$
                        md.push('$');
                        md.push_str(latex_content);
                        md.push_str("$ ");
                    } else {
                        // Display formula: use $$...$$
                        if !md.is_empty() {
                            md.push_str("\n\n");
                        }
                        md.push_str("$$");
                        md.push_str(latex_content);
                        md.push_str("$$");
                    }
                }
                // Image/Chart - figure format with center alignment
                LayoutElementType::Image | LayoutElementType::Chart => {
                    if !md.is_empty() {
                        md.push_str("\n\n");
                    }
                    // Use HTML img tag with center alignment for better rendering
                    md.push_str("<div style=\"text-align: center;\"><img src=\"");
                    // Generate a placeholder image name based on element bbox
                    let img_name = format!(
                        "imgs/img_in_{}_box_{:.0}_{:.0}_{:.0}_{:.0}.jpg",
                        if element.element_type == LayoutElementType::Chart {
                            "chart"
                        } else {
                            "image"
                        },
                        element.bbox.x_min(),
                        element.bbox.y_min(),
                        element.bbox.x_max(),
                        element.bbox.y_max()
                    );
                    md.push_str(&img_name);
                    md.push_str("\" alt=\"Image\" width=\"");
                    // Calculate width percentage relative to original image width (PaddleX logic)
                    let image_width = element.bbox.x_max() - element.bbox.x_min();
                    let width_pct = (image_width / original_image_width * 100.0) as u32;
                    let width_pct = width_pct.clamp(1, 100);
                    md.push_str(&format!("{}%", width_pct));
                    md.push_str("\" /></div>");
                }
                // Seal - show as image with text
                LayoutElementType::Seal => {
                    if !md.is_empty() {
                        md.push_str("\n\n");
                    }
                    md.push_str("![Seal]");
                    if let Some(text) = &element.text {
                        md.push_str("\n> ");
                        md.push_str(text);
                    }
                }
                // Captions - with center alignment following PaddleX
                _ if element.element_type.is_caption() => {
                    if let Some(text) = &element.text {
                        if !md.is_empty() {
                            md.push_str("\n\n");
                        }
                        let cleaned = clean_ocr_text(text);
                        md.push_str("<div style=\"text-align: center;\">");
                        md.push_str(&cleaned);
                        md.push_str(" </div>");
                    }
                }
                // Abstract - following PaddleX's format_first_line_func with spliter=" "
                LayoutElementType::Abstract => {
                    if let Some(text) = &element.text {
                        if !md.is_empty() {
                            md.push_str("\n\n");
                        }
                        let formatted = format_first_line(text, " ", &["abstract", "摘要"], "## ");
                        md.push_str(&formatted);
                    }
                }
                // Reference - following PaddleX's format_first_line_func with spliter="\n"
                LayoutElementType::Reference => {
                    if let Some(text) = &element.text {
                        if !md.is_empty() {
                            md.push_str("\n\n");
                        }
                        let formatted =
                            format_first_line(text, "\n", &["references", "参考文献"], "## ");
                        md.push_str(&formatted);
                    }
                }
                // Content (table of contents) - following PaddleX's soft breaks
                LayoutElementType::Content => {
                    if let Some(text) = &element.text {
                        if !md.is_empty() {
                            md.push_str("\n\n");
                        }
                        let formatted = format_content_block(text);
                        md.push_str(&formatted);
                    }
                }
                // Footnote - following PaddleX's vision_footnote handling
                LayoutElementType::Footnote => {
                    if let Some(text) = &element.text {
                        if !md.is_empty() {
                            md.push_str("\n\n");
                        }
                        let formatted = format_vision_footnote_block(text);
                        md.push_str(&formatted);
                    }
                }
                // List
                LayoutElementType::List => {
                    if let Some(text) = &element.text {
                        if !md.is_empty() {
                            md.push_str("\n\n");
                        }
                        let cleaned = format_text_block(text);
                        // Split by newlines and format as list items
                        for line in cleaned.lines() {
                            let line = line.trim();
                            if !line.is_empty() {
                                md.push_str("- ");
                                md.push_str(line);
                                md.push('\n');
                            }
                        }
                    }
                }
                // Algorithm block - PaddleX: block.content.strip("\n")
                LayoutElementType::Algorithm => {
                    if let Some(text) = &element.text {
                        if !md.is_empty() {
                            md.push_str("\n\n");
                        }
                        md.push_str(text.trim_matches('\n'));
                    }
                }
                // Header/Footer - smaller text (typically excluded from markdown)
                _ if element.element_type.is_header() || element.element_type.is_footer() => {
                    // Skip headers and footers in markdown output
                    // They typically contain page numbers and repeating info
                    continue;
                }
                // Default text elements - following PaddleX's text handling
                _ => {
                    if let Some(text) = &element.text {
                        // For text continuation (same paragraph), join directly
                        if is_continuation {
                            let formatted = format_text_block(text);
                            md.push_str(&formatted);
                        } else {
                            if !md.is_empty() {
                                md.push_str("\n\n");
                            }
                            let formatted = format_text_block(text);
                            md.push_str(&formatted);
                        }
                    }
                }
            }

            last_label = Some(element.element_type);
            prev_element = Some(element);
        }
        md.trim().to_string()
    }

    /// Calculates the page continuation flags for this result.
    ///
    /// This follows PaddleX's `get_seg_flag` logic to determine whether
    /// the page starts/ends in the middle of a semantic paragraph.
    ///
    /// Returns (paragraph_start, paragraph_end) where:
    /// - `paragraph_start`: false means page continues from previous
    /// - `paragraph_end`: false means content continues to next page
    pub fn calculate_continuation_flags(&self) -> PageContinuationFlags {
        let elements = &self.layout_elements;

        if elements.is_empty() {
            return PageContinuationFlags::new(true, true);
        }

        // Estimate page width from rectified image or element bboxes
        let page_width = self
            .rectified_img
            .as_ref()
            .map(|img| img.width() as f32)
            .or_else(|| {
                elements
                    .iter()
                    .map(|e| e.bbox.x_max())
                    .fold(None, |acc, x| Some(acc.map_or(x, |max: f32| max.max(x))))
            });

        // Filter to only text elements for continuation analysis
        let text_elements: Vec<_> = elements
            .iter()
            .filter(|e| {
                matches!(
                    e.element_type,
                    LayoutElementType::Text
                        | LayoutElementType::DocTitle
                        | LayoutElementType::ParagraphTitle
                        | LayoutElementType::Abstract
                        | LayoutElementType::Reference
                )
            })
            .collect();

        if text_elements.is_empty() {
            return PageContinuationFlags::new(true, true);
        }

        // Calculate paragraph start flag
        let first = &text_elements[0];
        let paragraph_start = is_new_paragraph_start(first, page_width);

        // Calculate paragraph end flag
        let last = &text_elements[text_elements.len() - 1];
        let paragraph_end = is_paragraph_complete(last, page_width);

        PageContinuationFlags::new(paragraph_start, paragraph_end)
    }

    /// Converts the result to an HTML string.
    ///
    /// Follows PP-StructureV3's formatting rules with semantic HTML tags.
    pub fn to_html(&self) -> String {
        let mut html = String::from(
            "<!DOCTYPE html>\n<html>\n<head>\n<meta charset=\"UTF-8\">\n</head>\n<body>\n",
        );

        for element in &self.layout_elements {
            match element.element_type {
                // Document title
                LayoutElementType::DocTitle => {
                    html.push_str("<h1>");
                    if let Some(text) = &element.text {
                        html.push_str(&Self::escape_html(text));
                    }
                    html.push_str("</h1>\n");
                }
                // Paragraph/section title
                LayoutElementType::ParagraphTitle => {
                    html.push_str("<h2>");
                    if let Some(text) = &element.text {
                        html.push_str(&Self::escape_html(text));
                    }
                    html.push_str("</h2>\n");
                }
                // Table - embed HTML structure with simplified markup
                LayoutElementType::Table => {
                    if let Some(table) =
                        self.tables.iter().find(|t| t.bbox.iou(&element.bbox) > 0.5)
                    {
                        if let Some(table_html) = &table.html_structure {
                            // Simplify table HTML (remove html/body wrappers) and add border styling
                            let simplified = simplify_table_html(table_html);
                            let styled = simplified.replacen(
                                "<table>",
                                "<table border=\"1\" style=\"border-collapse: collapse;\">",
                                1,
                            );
                            html.push_str(&styled);
                            html.push('\n');
                        } else {
                            html.push_str("<p>[Table]</p>\n");
                        }
                    } else {
                        html.push_str("<p>[Table]</p>\n");
                    }
                }
                // Formula - use math tags
                LayoutElementType::Formula | LayoutElementType::FormulaNumber => {
                    html.push_str("<p class=\"formula\">$$");
                    if let Some(latex) = &element.text {
                        html.push_str(&Self::escape_html(latex));
                    }
                    html.push_str("$$</p>\n");
                }
                // Image/Chart
                LayoutElementType::Image | LayoutElementType::Chart => {
                    html.push_str("<figure>\n<img alt=\"Figure\" />\n");
                    if let Some(caption) = &element.text {
                        html.push_str("<figcaption>");
                        html.push_str(&Self::escape_html(caption));
                        html.push_str("</figcaption>\n");
                    }
                    html.push_str("</figure>\n");
                }
                // Seal
                LayoutElementType::Seal => {
                    html.push_str("<figure class=\"seal\">\n<img alt=\"Seal\" />\n");
                    if let Some(text) = &element.text {
                        html.push_str("<figcaption>");
                        html.push_str(&Self::escape_html(text));
                        html.push_str("</figcaption>\n");
                    }
                    html.push_str("</figure>\n");
                }
                // Captions
                _ if element.element_type.is_caption() => {
                    if let Some(text) = &element.text {
                        html.push_str("<figcaption>");
                        html.push_str(&Self::escape_html(text));
                        html.push_str("</figcaption>\n");
                    }
                }
                // Abstract
                LayoutElementType::Abstract => {
                    html.push_str("<section class=\"abstract\">\n<h3>Abstract</h3>\n<p>");
                    if let Some(text) = &element.text {
                        html.push_str(&Self::escape_html(text));
                    }
                    html.push_str("</p>\n</section>\n");
                }
                // Reference
                LayoutElementType::Reference | LayoutElementType::ReferenceContent => {
                    html.push_str("<section class=\"references\">\n<p>");
                    if let Some(text) = &element.text {
                        html.push_str(&Self::escape_html(text));
                    }
                    html.push_str("</p>\n</section>\n");
                }
                // List
                LayoutElementType::List => {
                    html.push_str("<ul>\n");
                    if let Some(text) = &element.text {
                        for line in text.lines() {
                            html.push_str("<li>");
                            html.push_str(&Self::escape_html(line));
                            html.push_str("</li>\n");
                        }
                    }
                    html.push_str("</ul>\n");
                }
                // Header
                _ if element.element_type.is_header() => {
                    html.push_str("<header>");
                    if let Some(text) = &element.text {
                        html.push_str(&Self::escape_html(text));
                    }
                    html.push_str("</header>\n");
                }
                // Footer
                _ if element.element_type.is_footer() => {
                    html.push_str("<footer>");
                    if let Some(text) = &element.text {
                        html.push_str(&Self::escape_html(text));
                    }
                    html.push_str("</footer>\n");
                }
                // Default text
                _ => {
                    if let Some(text) = &element.text {
                        html.push_str("<p>");
                        html.push_str(&Self::escape_html(text));
                        html.push_str("</p>\n");
                    }
                }
            }
        }
        html.push_str("</body>\n</html>");
        html
    }

    /// Escapes HTML special characters.
    fn escape_html(text: &str) -> String {
        text.replace('&', "&amp;")
            .replace('<', "&lt;")
            .replace('>', "&gt;")
            .replace('"', "&quot;")
            .replace('\'', "&#39;")
    }

    /// Converts the result to a JSON Value.
    pub fn to_json_value(&self) -> serde_json::Result<serde_json::Value> {
        serde_json::to_value(self)
    }

    /// Saves the analysis results to the specified directory.
    ///
    /// This generates:
    /// - `*_res.json`: The full structured result
    /// - `*_res.html`: An HTML representation
    ///
    /// Note: Markdown export with image extraction should use the example utilities
    /// (`examples/utils/markdown.rs`) instead, as that requires I/O operations
    /// that belong in the application layer. Use `StructureResult::to_markdown()`
    /// for pure markdown generation without side effects.
    ///
    /// # Arguments
    ///
    /// * `output_dir` - Directory to save the output files
    /// * `to_json` - If true, save a JSON representation
    /// * `to_html` - If true, save an HTML representation
    pub fn save_results(
        &self,
        output_dir: impl AsRef<Path>,
        to_json: bool,
        to_html: bool,
    ) -> std::io::Result<()> {
        let output_dir = output_dir.as_ref();
        if !output_dir.exists() {
            std::fs::create_dir_all(output_dir)?;
        }

        let input_path = Path::new(self.input_path.as_ref());
        // Extract file stem, handling PDF page suffix (e.g., "file.pdf#3" -> "file_003")
        let stem = if let Some(path_str) = input_path.to_str() {
            if let Some(hash_idx) = path_str.rfind('#') {
                // This is a PDF page reference like "file.pdf#3"
                let base = &path_str[..hash_idx];
                let page_num = &path_str[hash_idx + 1..];
                let base_stem = Path::new(base)
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("result");
                format!("{}_{}", base_stem, page_num)
            } else {
                input_path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("result")
                    .to_string()
            }
        } else {
            "result".to_string()
        };

        // Save JSON
        if to_json {
            let json_path = output_dir.join(format!("{}.json", stem));
            let json_file = std::fs::File::create(json_path)?;
            serde_json::to_writer_pretty(json_file, self)?;
        }

        // Save HTML
        if to_html {
            let html_path = output_dir.join(format!("{}.html", stem));
            std::fs::write(html_path, self.to_html())?;
        }

        Ok(())
    }
}

/// Determines paragraph continuity flags for the current element relative to the previous.
///
/// This implements PaddleX's `get_seg_flag` logic from `layout_parsing/utils.py`:
/// - `seg_start_flag = true` means this element starts a NEW paragraph
/// - `seg_start_flag = false` means this element CONTINUES the previous paragraph
///
/// The logic checks whether:
/// 1. Previous block's last line ends near the right edge (text fills to right)
/// 2. Current block's first line starts near the left edge (no indentation)
/// 3. Previous block has more than one line
/// 4. The two blocks are horizontally close enough
///
/// Returns `seg_start_flag` (true = new paragraph, false = continuation).
fn get_seg_flag(current: &LayoutElement, prev: Option<&LayoutElement>) -> bool {
    const COORD_THRESHOLD: f32 = 10.0;

    let seg_start = current.seg_start_x.unwrap_or(current.bbox.x_min());
    let mut context_left = current.bbox.x_min();
    let mut context_right = current.bbox.x_max();

    if let Some(prev) = prev {
        let prev_seg_end = prev.seg_end_x.unwrap_or(prev.bbox.x_max());
        let prev_num_lines = prev.num_lines.unwrap_or(1);

        // Check if blocks overlap horizontally
        let overlap_blocks = context_left < prev.bbox.x_max() && context_right > prev.bbox.x_min();

        let edge_distance;
        if overlap_blocks {
            context_left = context_left.min(prev.bbox.x_min());
            context_right = context_right.max(prev.bbox.x_max());
            edge_distance = 0.0;
        } else {
            edge_distance = (current.bbox.x_min() - prev.bbox.x_max()).abs();
        }

        let prev_end_space_small = (context_right - prev_seg_end).abs() < COORD_THRESHOLD;
        let current_start_space_small = seg_start - context_left < COORD_THRESHOLD;
        let prev_lines_more_than_one = prev_num_lines > 1;
        let blocks_close = edge_distance
            < (prev.bbox.x_max() - prev.bbox.x_min())
                .max(current.bbox.x_max() - current.bbox.x_min());

        if prev_end_space_small
            && current_start_space_small
            && prev_lines_more_than_one
            && blocks_close
        {
            return false; // continuation
        }

        true // new paragraph
    } else {
        // First element: check if text starts near the left edge
        if seg_start - context_left < COORD_THRESHOLD {
            return false; // continuation from previous page (no indentation)
        }
        true
    }
}

/// Checks if a text element appears to start a new paragraph.
///
/// Following PaddleX's logic: if the text starts near the left edge of the page
/// (within 5% of page width), it's likely the start of a new paragraph.
fn is_new_paragraph_start(element: &LayoutElement, page_width: Option<f32>) -> bool {
    let left = element.bbox.x_min();
    let threshold = page_width.map_or(50.0, |w| w * 0.05); // 5% of page width
    left <= threshold
}

/// Checks if a text element appears to complete its paragraph on this page.
///
/// Following PaddleX's logic: if the text ends before the right edge of the page
/// (not within 10% of right margin), the paragraph likely ends here.
fn is_paragraph_complete(element: &LayoutElement, page_width: Option<f32>) -> bool {
    let right = element.bbox.x_max();

    // If we have page width info, check if element ends before the right edge
    if let Some(width) = page_width {
        let right_margin = width * 0.1;
        return right <= (width - right_margin);
    }

    // Conservative default: assume paragraphs end
    true
}

/// Concatenates markdown content from multiple pages into a single document.
///
/// This follows PaddleX's `concatenate_markdown_pages` logic to intelligently
/// merge pages while preserving paragraph continuity.
///
/// # Arguments
///
/// * `results` - Slice of structure results from multiple pages (in order)
///
/// # Returns
///
/// A single markdown string with all pages properly concatenated
pub fn concatenate_markdown_pages(results: &[StructureResult]) -> String {
    if results.is_empty() {
        return String::new();
    }

    if results.len() == 1 {
        return results[0].to_markdown();
    }

    let mut markdown = String::new();
    let mut prev_page_end_flag = true; // First page is treated as starting fresh

    for result in results.iter() {
        let flags = result
            .page_continuation_flags
            .as_ref()
            .cloned()
            .unwrap_or_else(|| result.calculate_continuation_flags());

        let page_markdown = result.to_markdown();

        // Skip empty pages
        if page_markdown.trim().is_empty() {
            prev_page_end_flag = flags.paragraph_end;
            continue;
        }

        let page_first_continues = !flags.paragraph_start;
        let _page_last_continues = !flags.paragraph_end;

        // Determine how to join this page
        if page_first_continues && !prev_page_end_flag {
            // Both pages are in the middle of the same paragraph
            // Check for Chinese characters to decide spacing
            let last_char = markdown.chars().last();
            let first_char = page_markdown.chars().next();

            let last_is_chinese = last_char.is_some_and(is_chinese_char);
            let first_is_chinese = first_char.is_some_and(is_chinese_char);

            if !last_is_chinese && !first_is_chinese {
                // Non-Chinese text: add space
                markdown.push(' ');
                markdown.push_str(page_markdown.trim_start());
            } else {
                // Chinese or mixed: direct concatenation
                markdown.push_str(page_markdown.trim_start());
            }
        } else {
            // New paragraph or section
            if !markdown.is_empty() {
                markdown.push_str("\n\n");
            }
            markdown.push_str(&page_markdown);
        }

        prev_page_end_flag = flags.paragraph_end;
    }

    markdown.trim().to_string()
}

/// Cleans OCR text content by removing common artifacts.
///
/// This function removes PDF line-break hyphens and fixes spacing issues
/// in OCR text content. It should only be applied to raw OCR text, not to
/// formatted markdown or HTML.
///
/// Following PaddleX's approach:
/// 1. Remove hyphenation artifacts: `-\n` -> `` (join words)
/// 2. Convert newlines to spaces: `\n` -> ` `
fn clean_ocr_text(text: &str) -> String {
    // First remove hyphenation (word breaks), then convert newlines to spaces
    text.replace("-\n", "").replace('\n', " ")
}

/// Formats the first non-empty line of a block if it matches a template keyword.
///
/// This is the Rust equivalent of PaddleX's `format_first_line_func`:
/// 1. Split text by `spliter`
/// 2. Find the first non-empty token
/// 3. If it matches any template (case-insensitive exact match), replace it with `format_func(token)`
/// 4. Rejoin with `spliter`
///
/// For abstract: `spliter=" "`, templates=["abstract","摘要"], format_func= `## {}\n`
/// For reference: `spliter="\n"`, templates=["references","参考文献"], format_func= `## {}`
fn format_first_line(
    text: &str,
    spliter: &str,
    templates: &[&str],
    heading_prefix: &str,
) -> String {
    let parts: Vec<&str> = text.split(spliter).collect();
    let mut result_parts: Vec<String> = Vec::with_capacity(parts.len());
    let mut found_first = false;

    for part in &parts {
        if !found_first {
            let trimmed = part.trim();
            if trimmed.is_empty() {
                result_parts.push(part.to_string());
                continue;
            }
            found_first = true;
            // Check if the first non-empty token matches a template (case-insensitive)
            if templates.iter().any(|t| trimmed.eq_ignore_ascii_case(t)) {
                // Replace with formatted heading: "## <original_text>\n"
                result_parts.push(format!("{}{}\n", heading_prefix, trimmed));
            } else {
                result_parts.push(part.to_string());
            }
        } else {
            result_parts.push(part.to_string());
        }
    }

    result_parts.join(spliter)
}

/// Formats text blocks following PaddleX's text handling:
/// 1. First remove hyphenation: `-\n` -> `` (join broken words)
/// 2. Then: `.replace("\n\n", "\n").replace("\n", "\n\n")`
///
/// This converts OCR line breaks into proper paragraph breaks.
fn format_text_block(text: &str) -> String {
    // First, remove hyphenation artifacts (word breaks at line ends)
    let dehyphenated = text.replace("-\n", "");
    // Collapse double newlines to single (undo paragraph breaks)
    let step1 = dehyphenated.replace("\n\n", "\n");
    // Then, convert single newlines to paragraph breaks
    step1.replace('\n', "\n\n")
}

/// Formats content blocks (table of contents) following PaddleX:
/// `.replace("-\n", "  \n").replace("\n", "  \n")`
///
/// This uses markdown's soft line break (two spaces at end of line).
fn format_content_block(text: &str) -> String {
    // Handle PDF hyphen line breaks first
    let step1 = text.replace("-\n", "  \n");
    // Convert newlines to soft breaks
    step1.replace('\n', "  \n")
}

/// Formats vision footnote blocks following PaddleX:
/// 1. First remove hyphenation: `-\n` -> ``
/// 2. Then: `.replace("\n\n", "\n").replace("\n", "\n\n")`
fn format_vision_footnote_block(text: &str) -> String {
    let dehyphenated = text.replace("-\n", "");
    let step1 = dehyphenated.replace("\n\n", "\n");
    step1.replace('\n', "\n\n")
}

/// Checks if a character is a Chinese character.
///
/// Used to determine spacing rules when concatenating pages.
fn is_chinese_char(c: char) -> bool {
    match c {
        '\u{4E00}'..='\u{9FFF}' | // CJK Unified Ideographs
        '\u{3400}'..='\u{4DBF}' | // CJK Unified Ideographs Extension A
        '\u{20000}'..='\u{2A6DF}' | // CJK Unified Ideographs Extension B
        '\u{2A700}'..='\u{2B73F}' | // CJK Unified Ideographs Extension C
        '\u{2B740}'..='\u{2B81F}' | // CJK Unified Ideographs Extension D
        '\u{2B820}'..='\u{2CEAF}' | // CJK Unified Ideographs Extension E
        '\u{2CEB0}'..='\u{2EBEF}' => // CJK Unified Ideographs Extension F
            true,
        _ => false,
    }
}

/// Checks if a character is a lowercase letter.
fn is_lowercase(c: char) -> bool {
    c.is_ascii_lowercase()
}

/// Checks if a character is an uppercase letter.
fn is_uppercase(c: char) -> bool {
    c.is_ascii_uppercase()
}

/// Checks if a character is a digit.
fn is_digit(c: char) -> bool {
    c.is_ascii_digit()
}

/// Removes PDF hyphenation artifacts from text.
/// Dehyphenation: only handles hyphen-newline patterns (word breaks across lines).
///
/// Matches PaddleX's behavior where hyphens are only stripped at line boundaries
/// (hyphen immediately followed by newline). Mid-word hyphens in compound words
/// like "real-time", "end-to-end", "one-to-many" are preserved.
fn dehyphenate(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let len = chars.len();
    let mut i = 0;

    // Helper to check if we're in a URL-like pattern
    let is_url_context = |pos: usize| -> bool {
        let start = pos.saturating_sub(10);
        let end = (pos + 5).min(len);
        let window: String = chars[start..end].iter().collect();
        window.contains("http") || window.contains("www") || window.contains("://")
    };

    while i < len {
        if chars[i] == '-' {
            if is_url_context(i) {
                result.push('-');
                i += 1;
                continue;
            }

            // Only dehyphenate when hyphen is followed by newline (line-break hyphenation).
            // Pattern: "word-\nletter" → "wordletter"
            let is_artifact = if i + 1 < len && chars[i + 1] == '\n' {
                // Hyphen followed by newline — check if next line starts with lowercase
                if i + 2 < len {
                    is_lowercase(chars[i + 2])
                } else {
                    false
                }
            } else {
                false
            };

            if is_artifact {
                // Skip the hyphen and the following newline
                i += 1; // skip newline (will be incremented again at end of loop)
            } else {
                result.push('-');
            }
        } else {
            result.push(chars[i]);
        }
        i += 1;
    }

    result
}

/// Fixes missing spaces between merged words.
///
/// OCR and PDF extraction can result in merged words like
/// "enhancetheencoder'sfeaturerepresentation" or "48.1%AP".
/// This function detects and fixes common patterns.
fn fix_merged_words(text: &str) -> String {
    let mut result = String::with_capacity(text.len());
    let chars: Vec<char> = text.chars().collect();
    let mut i = 0;

    while i < chars.len() {
        let current = chars[i];

        if i > 0 {
            let prev = chars[i - 1];

            // Detect missing space between lowercase and lowercase (after apostrophe or consonant)
            // e.g., "encoder'sfeature" -> "encoder's feature"
            if is_lowercase(prev) && is_lowercase(current) {
                // Only add space if previous was apostrophe or word boundary context
                // This is a heuristic - in practice you'd want more sophisticated NLP
                if i > 1 && chars[i - 2] == '\'' {
                    result.push(' ');
                }
                // Also detect lowercase followed by uppercase
                // e.g., "RT-DETRis" -> "RT-DETR is"
            } else if is_lowercase(prev) && is_uppercase(current) {
                // Check if the uppercase starts a new word (not an acronym)
                // If next char is lowercase, it's likely a new word
                if i + 1 < chars.len() && is_lowercase(chars[i + 1]) {
                    result.push(' ');
                }
            }
            // Detect digit/percent followed by letter, or letter-digit-letter pattern
            // e.g., "48.1%AP" -> "48.1% AP"
            // e.g., "RT-DETRv3" shouldn't be split, but "model 100instances" -> "model 100 instances"
            else if ((is_digit(prev) || prev == '%') && is_uppercase(current))
                || (is_letter(prev)
                    && is_digit(current)
                    && i + 1 < chars.len()
                    && is_letter(chars[i + 1]))
            {
                result.push(' ');
            }
        }

        result.push(current);
        i += 1;
    }

    result
}

/// Checks if a character is a letter.
fn is_letter(c: char) -> bool {
    is_lowercase(c) || is_uppercase(c)
}

/// Simplifies table HTML by removing wrapper tags, following PaddleX's `simplify_table_func`.
///
/// This removes `<html>`, `</html>`, `<body>`, and `</body>` tags from table HTML
/// to produce cleaner markdown output.
fn simplify_table_html(html: &str) -> String {
    html.replace("<html>", "")
        .replace("</html>", "")
        .replace("<body>", "")
        .replace("</body>", "")
}

/// Post-processes text content to fix common OCR/PDF artifacts.
///
/// This applies multiple cleanup steps:
/// 1. Dehyphenation - removes line-break hyphens
/// 2. Word merging fixes - adds missing spaces
/// 3. Spacing normalization - fixes multiple spaces
pub fn postprocess_text(text: &str) -> String {
    let text = dehyphenate(text);
    let text = fix_merged_words(&text);

    // Normalize whitespace (collapse multiple spaces, fix spacing after punctuation)
    let mut result = String::new();
    let mut in_space = false;

    for c in text.chars() {
        if c.is_whitespace() {
            if !in_space && !result.is_empty() {
                result.push(' ');
                in_space = true;
            }
        } else {
            // Fix missing space after period (when followed by letter)
            if c == '.' && !result.is_empty() {
                let last = result.chars().last().unwrap();
                if is_letter(last) || is_digit(last) {
                    result.push('.');
                    in_space = true;
                    continue;
                }
            }
            // Fix spacing after punctuation
            if in_space && matches!(c, '.' | ',' | '!' | '?' | ';' | ':' | ')' | ']' | '}') {
                result.pop(); // Remove the space before punctuation
                result.push(c);
                continue;
            }
            result.push(c);
            in_space = false;
        }
    }

    result
}

/// Removes duplicate section headers from concatenated markdown.
///
/// When concatenating pages, section headers like "**Abstract**" or
/// "**References**" may appear multiple times. This function deduplicates
/// them while preserving the first occurrence.
fn deduplicate_sections(markdown: &str) -> String {
    let mut result = String::new();
    let mut seen_sections: std::collections::HashSet<String> = std::collections::HashSet::new();

    for line in markdown.lines() {
        let trimmed = line.trim();

        // Check for common section header patterns
        let is_section_header =
            trimmed.starts_with("**") && trimmed.ends_with("**") && trimmed.len() > 4;

        let section_name = if is_section_header {
            trimmed[2..trimmed.len() - 2].to_string()
        } else {
            String::new()
        };

        if is_section_header {
            if seen_sections.contains(&section_name) {
                // Skip duplicate section header
                continue;
            }
            seen_sections.insert(section_name);
        }

        if !result.is_empty() {
            result.push('\n');
        }
        result.push_str(line);
    }

    result
}

/// Checks if two bounding boxes are on the same line (have significant vertical overlap).
///
/// Two boxes are considered on the same line if their vertical overlap is greater than
/// 50% of the smaller box's height.
fn is_same_line(bbox1: &BoundingBox, bbox2: &BoundingBox) -> bool {
    let y1_min = bbox1.y_min();
    let y1_max = bbox1.y_max();
    let y2_min = bbox2.y_min();
    let y2_max = bbox2.y_max();

    // Calculate vertical overlap
    let overlap_start = y1_min.max(y2_min);
    let overlap_end = y1_max.min(y2_max);
    let overlap = (overlap_end - overlap_start).max(0.0);

    // Calculate minimum height
    let height1 = y1_max - y1_min;
    let height2 = y2_max - y2_min;
    let min_height = height1.min(height2);

    // Consider same line if overlap > 50% of min height
    min_height > 0.0 && overlap / min_height > 0.5
}

/// Filters empty formula blocks from markdown.
///
/// Formula blocks with no LaTeX content like `$$\n$$` are removed.
fn filter_empty_formulas(markdown: &str) -> String {
    let mut result = String::new();
    let lines: Vec<&str> = markdown.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];

        // Check for empty formula block pattern
        if line.trim() == "$$" {
            // Check if next line is also $$ (empty formula)
            if i + 1 < lines.len() && lines[i + 1].trim() == "$$" {
                // Skip both lines
                i += 2;
                // Also skip the blank line after
                if i < lines.len() && lines[i].trim().is_empty() {
                    i += 1;
                }
                continue;
            }
            // Check if the next non-empty line contains actual content
            let mut j = i + 1;
            let has_content = if j < lines.len() {
                let mut found = false;
                while j < lines.len() {
                    if lines[j].trim() == "$$" {
                        break;
                    }
                    if !lines[j].trim().is_empty() {
                        found = true;
                        break;
                    }
                    j += 1;
                }
                found
            } else {
                false
            };

            if !has_content {
                // Skip to closing $$
                while i < lines.len() && lines[i].trim() != "$$" {
                    i += 1;
                }
                if i < lines.len() {
                    i += 1; // Skip closing $$
                }
                continue;
            }
        }

        if !result.is_empty() {
            result.push('\n');
        }
        result.push_str(line);
        i += 1;
    }

    result
}

/// Applies all post-processing steps to concatenated markdown.
///
/// This is the main entry point for cleaning up concatenated markdown output.
pub fn postprocess_markdown(markdown: &str) -> String {
    let markdown = filter_empty_formulas(markdown);
    let markdown = deduplicate_sections(&markdown);

    // Apply text post-processing line by line for text content
    let mut result = String::new();
    let mut in_code_block = false;
    let mut in_formula = false;

    for line in markdown.lines() {
        let trimmed = line.trim();

        // Detect code blocks
        if trimmed.starts_with("```") {
            in_code_block = !in_code_block;
            result.push_str(line);
            result.push('\n');
            continue;
        }

        // Detect formula blocks
        if trimmed == "$$" {
            in_formula = !in_formula;
            result.push_str(line);
            result.push('\n');
            continue;
        }

        // Skip processing inside code blocks
        if in_code_block {
            result.push_str(line);
            result.push('\n');
            continue;
        }

        // If inside a formula block, ensure it doesn't contain unescaped dollar signs
        // which cause KaTeX "Can't use function '$' in math mode" errors.
        if in_formula {
            // If the formula content looks like regular text (many spaces, few backslashes)
            // and contains a $, KaTeX will fail. We escape the $ inside the math block.
            let contains_dollar = line.contains('$');
            let is_plain_text = line.split_whitespace().count() > 3 && !line.contains('\\');

            if contains_dollar && is_plain_text {
                result.push_str(&line.replace('$', "\\$"));
            } else if contains_dollar {
                // Remove redundant dollar signs inside the block
                result.push_str(&line.replace('$', ""));
            } else {
                result.push_str(line);
            }
            result.push('\n');
            continue;
        }

        // Process text content (skip headers, lists, etc.)
        if trimmed.starts_with('#')
            || trimmed.starts_with('*')
            || trimmed.starts_with('>')
            || trimmed.starts_with('|')
            || trimmed.starts_with('-')
            || trimmed.starts_with('+')
        {
            result.push_str(line);
        } else {
            result.push_str(&postprocess_text(line));
        }
        result.push('\n');
    }

    result
}

/// Extension trait for convenient multi-page processing.
pub trait StructureResultExt {
    /// Converts multiple results to a single concatenated markdown.
    fn to_concatenated_markdown(results: &[Self]) -> String
    where
        Self: Sized;

    /// Saves multiple results with concatenated markdown.
    fn save_multi_page_results(
        results: &[Self],
        output_dir: impl AsRef<std::path::Path>,
        base_name: &str,
        to_json: bool,
        to_markdown: bool,
        to_html: bool,
    ) -> std::io::Result<()>
    where
        Self: Sized;
}

impl StructureResultExt for StructureResult {
    fn to_concatenated_markdown(results: &[Self]) -> String {
        concatenate_markdown_pages(results)
    }

    fn save_multi_page_results(
        results: &[Self],
        output_dir: impl AsRef<std::path::Path>,
        base_name: &str,
        to_json: bool,
        to_markdown: bool,
        to_html: bool,
    ) -> std::io::Result<()>
    where
        Self: Sized,
    {
        let output_dir = output_dir.as_ref();
        if !output_dir.exists() {
            std::fs::create_dir_all(output_dir)?;
        }

        // Save individual page results
        for (idx, result) in results.iter().enumerate() {
            let page_dir = output_dir.join(format!("page_{:03}", idx));
            std::fs::create_dir_all(&page_dir)?;
            result.save_results(&page_dir, to_json, to_html)?;
        }

        // Save concatenated markdown
        if to_markdown {
            let concat_md_path = output_dir.join(format!("{}.md", base_name));
            std::fs::write(concat_md_path, Self::to_concatenated_markdown(results))?;
        }

        // Save concatenated JSON (array of results)
        if to_json {
            let concat_json_path = output_dir.join(format!("{}.json", base_name));
            let json_file = std::fs::File::create(concat_json_path)?;
            serde_json::to_writer_pretty(json_file, &results)?;
        }

        Ok(())
    }
}

/// A layout element detected in the document.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutElement {
    /// Bounding box of the element
    pub bbox: BoundingBox,
    /// Type of the layout element
    pub element_type: LayoutElementType,
    /// Confidence score for the detection
    pub confidence: f32,
    /// Optional label for the element (original model label)
    pub label: Option<String>,
    /// Optional text content for the element
    pub text: Option<String>,
    /// Reading order index (1-based, assigned during stitching)
    ///
    /// This index represents the element's position in the reading order.
    /// Only elements that should be included in reading flow (text, tables,
    /// formulas, images, etc.) will have an order index assigned.
    /// Headers, footers, and other auxiliary elements may have `None`.
    pub order_index: Option<u32>,
    /// X-coordinate of the first text span's left edge within this element.
    /// Used by `get_seg_flag` to detect paragraph continuity across blocks.
    /// Computed during stitching from the first OCR region (after spatial sort).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seg_start_x: Option<f32>,
    /// X-coordinate of the last text span's right edge within this element.
    /// Used by `get_seg_flag` to detect paragraph continuity across blocks.
    /// Computed during stitching from the last OCR region (after spatial sort).
    #[serde(skip_serializing_if = "Option::is_none")]
    pub seg_end_x: Option<f32>,
    /// Number of text lines within this element.
    /// Used by `get_seg_flag` to detect paragraph continuity across blocks.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub num_lines: Option<u32>,
}

impl LayoutElement {
    /// Creates a new layout element.
    pub fn new(bbox: BoundingBox, element_type: LayoutElementType, confidence: f32) -> Self {
        Self {
            bbox,
            element_type,
            confidence,
            label: None,
            text: None,
            order_index: None,
            seg_start_x: None,
            seg_end_x: None,
            num_lines: None,
        }
    }

    /// Sets the label for the element.
    pub fn with_label(mut self, label: impl Into<String>) -> Self {
        self.label = Some(label.into());
        self
    }

    /// Sets the text content for the element.
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }
}

/// Layout element type supporting PP-StructureV3's full label set.
///
/// This enum represents both **semantic categories** and **fine-grained labels** for layout elements.
/// PP-StructureV3 models output 20 or 23 class labels depending on the model variant.
///
/// The original model-specific label is preserved in `LayoutElement.label` field.
///
/// # PP-StructureV3 Label Categories
///
/// **Document structure:**
/// - `DocTitle` - Document title (doc_title)
/// - `ParagraphTitle` - Section/paragraph title (paragraph_title)
/// - `Text` - General text content
/// - `Content` - Table of contents (content)
/// - `Abstract` - Abstract section
///
/// **Visual elements:**
/// - `Image` - Images/figures (image, figure)
/// - `Table` - Tables
/// - `Chart` - Charts/graphs
/// - `Formula` - Mathematical formulas
///
/// **Captions and titles:**
/// - `FigureTitle` - Figure caption (figure_title)
/// - `TableTitle` - Table caption (table_title)
/// - `ChartTitle` - Chart caption (chart_title)
/// - `FigureTableChartTitle` - Combined caption type
///
/// **Page structure:**
/// - `Header` - Page header
/// - `HeaderImage` - Header image
/// - `Footer` - Page footer
/// - `FooterImage` - Footer image
/// - `Footnote` - Footnotes
///
/// **Special elements:**
/// - `Seal` - Stamps/official seals
/// - `Number` - Page numbers
/// - `Reference` - References section
/// - `ReferenceContent` - Reference content
/// - `Algorithm` - Algorithm blocks
/// - `FormulaNumber` - Formula numbers
/// - `AsideText` - Marginal/aside text
/// - `List` - List items
///
/// - `Other` - Unknown/unmapped labels
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum LayoutElementType {
    /// Document title
    DocTitle,
    /// Paragraph/section title
    ParagraphTitle,
    /// General text content
    Text,
    /// Table of contents
    Content,
    /// Abstract section
    Abstract,

    /// Image or figure
    Image,
    /// Table
    Table,
    /// Chart or graph
    Chart,
    /// Mathematical formula
    Formula,

    /// Figure caption/title
    FigureTitle,
    /// Table caption/title
    TableTitle,
    /// Chart caption/title
    ChartTitle,
    /// Combined figure/table/chart title (PP-DocLayout)
    FigureTableChartTitle,

    /// Page header
    Header,
    /// Header image
    HeaderImage,
    /// Page footer
    Footer,
    /// Footer image
    FooterImage,
    /// Footnote
    Footnote,

    /// Stamp or official seal
    Seal,
    /// Page number
    Number,
    /// Reference section
    Reference,
    /// Reference content (PP-DocLayout_plus-L)
    ReferenceContent,
    /// Algorithm block
    Algorithm,
    /// Formula number
    FormulaNumber,
    /// Marginal/aside text
    AsideText,
    /// List items
    List,

    /// Generic document region block (PP-DocBlockLayout)
    /// Used for hierarchical layout ordering and block grouping
    Region,

    /// Other/unknown (original label preserved in LayoutElement.label)
    Other,
}

impl LayoutElementType {
    /// Returns the string representation of the element type.
    ///
    /// This returns the PP-StructureV3 compatible label string.
    pub fn as_str(&self) -> &'static str {
        match self {
            // Document Structure
            LayoutElementType::DocTitle => "doc_title",
            LayoutElementType::ParagraphTitle => "paragraph_title",
            LayoutElementType::Text => "text",
            LayoutElementType::Content => "content",
            LayoutElementType::Abstract => "abstract",

            // Visual Elements
            LayoutElementType::Image => "image",
            LayoutElementType::Table => "table",
            LayoutElementType::Chart => "chart",
            LayoutElementType::Formula => "formula",

            // Captions
            LayoutElementType::FigureTitle => "figure_title",
            LayoutElementType::TableTitle => "table_title",
            LayoutElementType::ChartTitle => "chart_title",
            LayoutElementType::FigureTableChartTitle => "figure_table_chart_title",

            // Page Structure
            LayoutElementType::Header => "header",
            LayoutElementType::HeaderImage => "header_image",
            LayoutElementType::Footer => "footer",
            LayoutElementType::FooterImage => "footer_image",
            LayoutElementType::Footnote => "footnote",

            // Special Elements
            LayoutElementType::Seal => "seal",
            LayoutElementType::Number => "number",
            LayoutElementType::Reference => "reference",
            LayoutElementType::ReferenceContent => "reference_content",
            LayoutElementType::Algorithm => "algorithm",
            LayoutElementType::FormulaNumber => "formula_number",
            LayoutElementType::AsideText => "aside_text",
            LayoutElementType::List => "list",

            // Region (PP-DocBlockLayout)
            LayoutElementType::Region => "region",

            // Fallback
            LayoutElementType::Other => "other",
        }
    }

    /// Creates a LayoutElementType from a string label with fine-grained mapping.
    ///
    /// This method maps model output labels to their corresponding fine-grained types,
    /// preserving the full PP-StructureV3 label set (20/23 classes).
    pub fn from_label(label: &str) -> Self {
        match label.to_lowercase().as_str() {
            // Document Structure
            "doc_title" => LayoutElementType::DocTitle,
            "paragraph_title" | "title" => LayoutElementType::ParagraphTitle,
            "text" | "paragraph" => LayoutElementType::Text,
            "content" => LayoutElementType::Content,
            "abstract" => LayoutElementType::Abstract,

            // Visual Elements
            "image" | "figure" => LayoutElementType::Image,
            "table" => LayoutElementType::Table,
            "chart" | "flowchart" => LayoutElementType::Chart,
            "formula" | "equation" | "display_formula" | "inline_formula" => {
                LayoutElementType::Formula
            }

            // Captions
            "figure_title" => LayoutElementType::FigureTitle,
            "table_title" => LayoutElementType::TableTitle,
            "chart_title" => LayoutElementType::ChartTitle,
            "figure_table_chart_title" | "caption" => LayoutElementType::FigureTableChartTitle,

            // Page Structure
            "header" => LayoutElementType::Header,
            "header_image" => LayoutElementType::HeaderImage,
            "footer" => LayoutElementType::Footer,
            "footer_image" => LayoutElementType::FooterImage,
            "footnote" | "vision_footnote" => LayoutElementType::Footnote,

            // Special Elements
            "seal" => LayoutElementType::Seal,
            "number" => LayoutElementType::Number,
            "reference" => LayoutElementType::Reference,
            "reference_content" => LayoutElementType::ReferenceContent,
            "algorithm" => LayoutElementType::Algorithm,
            "formula_number" => LayoutElementType::FormulaNumber,
            "aside_text" => LayoutElementType::AsideText,
            "list" => LayoutElementType::List,
            "vertical_text" => LayoutElementType::Text,

            // Region (PP-DocBlockLayout)
            "region" => LayoutElementType::Region,

            // Everything else maps to Other
            // The original label is preserved in LayoutElement.label
            _ => LayoutElementType::Other,
        }
    }

    /// Returns the semantic category for this element type.
    ///
    /// This method groups fine-grained types into broader semantic categories,
    /// useful for processing logic that doesn't need fine-grained distinctions.
    ///
    /// # Categories
    ///
    /// - **Title**: DocTitle, ParagraphTitle
    /// - **Text**: Text, Content, Abstract
    /// - **Visual**: Image, Chart
    /// - **Table**: Table
    /// - **Caption**: FigureTitle, TableTitle, ChartTitle, FigureTableChartTitle
    /// - **Header**: Header, HeaderImage
    /// - **Footer**: Footer, FooterImage, Footnote
    /// - **Formula**: Formula, FormulaNumber
    /// - **Special**: Seal, Number, Reference, ReferenceContent, Algorithm, AsideText
    /// - **List**: List
    /// - **Other**: Other
    pub fn semantic_category(&self) -> &'static str {
        match self {
            // Title category
            LayoutElementType::DocTitle | LayoutElementType::ParagraphTitle => "title",

            // Text category
            LayoutElementType::Text | LayoutElementType::Content | LayoutElementType::Abstract => {
                "text"
            }

            // Visual category
            LayoutElementType::Image | LayoutElementType::Chart => "visual",

            // Table category
            LayoutElementType::Table => "table",

            // Caption category
            LayoutElementType::FigureTitle
            | LayoutElementType::TableTitle
            | LayoutElementType::ChartTitle
            | LayoutElementType::FigureTableChartTitle => "caption",

            // Header category
            LayoutElementType::Header | LayoutElementType::HeaderImage => "header",

            // Footer category
            LayoutElementType::Footer
            | LayoutElementType::FooterImage
            | LayoutElementType::Footnote => "footer",

            // Formula category
            LayoutElementType::Formula | LayoutElementType::FormulaNumber => "formula",

            // Special category
            LayoutElementType::Seal
            | LayoutElementType::Number
            | LayoutElementType::Reference
            | LayoutElementType::ReferenceContent
            | LayoutElementType::Algorithm
            | LayoutElementType::AsideText => "special",

            // List category
            LayoutElementType::List => "list",

            // Region category (PP-DocBlockLayout)
            LayoutElementType::Region => "region",

            // Other
            LayoutElementType::Other => "other",
        }
    }

    /// Returns whether this element type is a title variant.
    pub fn is_title(&self) -> bool {
        matches!(
            self,
            LayoutElementType::DocTitle | LayoutElementType::ParagraphTitle
        )
    }

    /// Returns whether this element type is a visual element (image, chart, figure).
    pub fn is_visual(&self) -> bool {
        matches!(self, LayoutElementType::Image | LayoutElementType::Chart)
    }

    /// Returns whether this element type is a caption variant.
    pub fn is_caption(&self) -> bool {
        matches!(
            self,
            LayoutElementType::FigureTitle
                | LayoutElementType::TableTitle
                | LayoutElementType::ChartTitle
                | LayoutElementType::FigureTableChartTitle
        )
    }

    /// Returns whether this element type is a header variant.
    pub fn is_header(&self) -> bool {
        matches!(
            self,
            LayoutElementType::Header | LayoutElementType::HeaderImage
        )
    }

    /// Returns whether this element type is a footer variant.
    pub fn is_footer(&self) -> bool {
        matches!(
            self,
            LayoutElementType::Footer
                | LayoutElementType::FooterImage
                | LayoutElementType::Footnote
        )
    }

    /// Returns whether this element type is a formula variant.
    pub fn is_formula(&self) -> bool {
        matches!(
            self,
            LayoutElementType::Formula | LayoutElementType::FormulaNumber
        )
    }

    /// Returns whether this element type contains text content that should be OCR'd.
    pub fn should_ocr(&self) -> bool {
        matches!(
            self,
            LayoutElementType::Text
                | LayoutElementType::Content
                | LayoutElementType::Abstract
                | LayoutElementType::DocTitle
                | LayoutElementType::ParagraphTitle
                | LayoutElementType::FigureTitle
                | LayoutElementType::TableTitle
                | LayoutElementType::ChartTitle
                | LayoutElementType::FigureTableChartTitle
                | LayoutElementType::Header
                | LayoutElementType::HeaderImage
                | LayoutElementType::Footer
                | LayoutElementType::FooterImage
                | LayoutElementType::Footnote
                | LayoutElementType::Reference
                | LayoutElementType::ReferenceContent
                | LayoutElementType::Algorithm
                | LayoutElementType::AsideText
                | LayoutElementType::List
                | LayoutElementType::Number
        )
    }
}

/// Removes heavily-overlapping layout elements in-place.
///
/// This mirrors PP-Structure-style overlap suppression where text takes priority over images.
/// Returns the number of elements removed.
pub fn remove_overlapping_layout_elements(
    layout_elements: &mut Vec<LayoutElement>,
    overlap_threshold: f32,
) -> usize {
    use std::collections::HashSet;

    if layout_elements.len() <= 1 {
        return 0;
    }

    let bboxes: Vec<_> = layout_elements.iter().map(|e| e.bbox.clone()).collect();
    let labels: Vec<&str> = layout_elements
        .iter()
        .map(|e| e.element_type.as_str())
        .collect();

    let remove_indices =
        crate::processors::get_overlap_removal_indices(&bboxes, &labels, overlap_threshold);
    if remove_indices.is_empty() {
        return 0;
    }

    let remove_set: HashSet<usize> = remove_indices.into_iter().collect();
    let before = layout_elements.len();

    let mut idx = 0;
    layout_elements.retain(|_| {
        let keep = !remove_set.contains(&idx);
        idx += 1;
        keep
    });

    before.saturating_sub(layout_elements.len())
}

/// Applies small, PP-Structure-style label fixes to layout elements.
///
/// This is intended to capture lightweight "glue" heuristics that shouldn't live in `predict`.
pub fn apply_standardized_layout_label_fixes(layout_elements: &mut [LayoutElement]) {
    if layout_elements.is_empty() {
        return;
    }

    let mut footnote_indices: Vec<usize> = Vec::new();
    let mut paragraph_title_indices: Vec<usize> = Vec::new();
    let mut bottom_text_y_max: f32 = 0.0;
    let mut max_block_area: f32 = 0.0;
    let mut doc_title_num: usize = 0;

    for (idx, elem) in layout_elements.iter().enumerate() {
        let area =
            (elem.bbox.x_max() - elem.bbox.x_min()) * (elem.bbox.y_max() - elem.bbox.y_min());
        max_block_area = max_block_area.max(area);

        match elem.element_type {
            LayoutElementType::Footnote => footnote_indices.push(idx),
            LayoutElementType::ParagraphTitle => paragraph_title_indices.push(idx),
            LayoutElementType::Text => {
                bottom_text_y_max = bottom_text_y_max.max(elem.bbox.y_max());
            }
            LayoutElementType::DocTitle => doc_title_num += 1,
            _ => {}
        }
    }

    for idx in footnote_indices {
        if layout_elements[idx].bbox.y_max() < bottom_text_y_max {
            layout_elements[idx].element_type = LayoutElementType::Text;
            layout_elements[idx].label = Some("text".to_string());
        }
    }

    let only_one_paragraph_title = paragraph_title_indices.len() == 1 && doc_title_num == 0;
    if only_one_paragraph_title {
        let idx = paragraph_title_indices[0];
        let area = (layout_elements[idx].bbox.x_max() - layout_elements[idx].bbox.x_min())
            * (layout_elements[idx].bbox.y_max() - layout_elements[idx].bbox.y_min());

        let title_area_ratio_threshold = 0.3f32;
        if area > max_block_area * title_area_ratio_threshold {
            layout_elements[idx].element_type = LayoutElementType::DocTitle;
            layout_elements[idx].label = Some("doc_title".to_string());
        }
    }
}

/// Result of table recognition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableResult {
    /// Bounding box of the table in the original image
    pub bbox: BoundingBox,
    /// Table type (wired or wireless)
    pub table_type: TableType,
    /// Confidence score for table type classification (None if classifier wasn't configured/run)
    pub classification_confidence: Option<f32>,
    /// Confidence score for table structure recognition (None if structure recognition failed)
    pub structure_confidence: Option<f32>,
    /// Detected table cells
    pub cells: Vec<TableCell>,
    /// HTML structure of the table (if available)
    pub html_structure: Option<String>,
    /// OCR text content for each cell (if OCR was integrated)
    pub cell_texts: Option<Vec<Option<String>>>,
    /// Structure tokens from table structure recognition (used for HTML generation after stitching)
    #[serde(skip)]
    pub structure_tokens: Option<Vec<String>>,
    /// Detected cell bounding boxes from the cell detection model (in page coordinates).
    /// Stored separately from `cells` (which carry structure/grid metadata from the structure model)
    /// and used by the stitcher for row-aware IoA-based OCR matching.
    #[serde(skip)]
    pub detected_cell_bboxes: Option<Vec<BoundingBox>>,
    /// Whether the table was processed in end-to-end (E2E) mode.
    /// When true, cells come from the structure model only (no separate cell detection).
    /// Used by the stitcher to select the appropriate OCR matching strategy.
    #[serde(skip)]
    pub is_e2e: bool,
}

impl TableResult {
    /// Creates a new table result.
    pub fn new(bbox: BoundingBox, table_type: TableType) -> Self {
        Self {
            bbox,
            table_type,
            classification_confidence: None,
            structure_confidence: None,
            cells: Vec::new(),
            html_structure: None,
            cell_texts: None,
            structure_tokens: None,
            detected_cell_bboxes: None,
            is_e2e: false,
        }
    }

    /// Sets the classification confidence.
    pub fn with_classification_confidence(mut self, confidence: f32) -> Self {
        self.classification_confidence = Some(confidence);
        self
    }

    /// Sets the structure recognition confidence.
    pub fn with_structure_confidence(mut self, confidence: f32) -> Self {
        self.structure_confidence = Some(confidence);
        self
    }

    /// Sets the table cells.
    pub fn with_cells(mut self, cells: Vec<TableCell>) -> Self {
        self.cells = cells;
        self
    }

    /// Sets the HTML structure.
    pub fn with_html_structure(mut self, html: impl Into<String>) -> Self {
        self.html_structure = Some(html.into());
        self
    }

    /// Sets the cell texts from OCR.
    pub fn with_cell_texts(mut self, texts: Vec<Option<String>>) -> Self {
        self.cell_texts = Some(texts);
        self
    }

    /// Sets the structure tokens for later HTML generation.
    pub fn with_structure_tokens(mut self, tokens: Vec<String>) -> Self {
        self.structure_tokens = Some(tokens);
        self
    }

    /// Stores detected cell bounding boxes for the stitcher's row-aware IoA matcher.
    pub fn with_detected_cell_bboxes(mut self, bboxes: Vec<BoundingBox>) -> Self {
        self.detected_cell_bboxes = Some(bboxes);
        self
    }

    /// Marks this table as processed in end-to-end (E2E) mode.
    pub fn with_e2e(mut self, is_e2e: bool) -> Self {
        self.is_e2e = is_e2e;
        self
    }

    /// Returns the best available confidence score for this table.
    ///
    /// This method provides a unified confidence API for callers who want to filter
    /// tables by confidence without caring whether classification or structure
    /// recognition was used. Priority:
    /// 1. If both classification and structure confidence are available, returns
    ///    the minimum (most conservative estimate)
    /// 2. If only structure confidence is available (common when classifier isn't
    ///    configured), returns that
    /// 3. If only classification confidence is available, returns that
    /// 4. Returns `None` only if neither confidence is available (stub result)
    pub fn confidence(&self) -> Option<f32> {
        match (self.classification_confidence, self.structure_confidence) {
            (Some(cls), Some(str)) => Some(cls.min(str)),
            (None, Some(str)) => Some(str),
            (Some(cls), None) => Some(cls),
            (None, None) => None,
        }
    }

    /// Returns true if this table has valid structure data.
    ///
    /// A table is considered valid if it has either cells or an HTML structure.
    /// Stub results (created when structure recognition fails) will return false.
    pub fn has_structure(&self) -> bool {
        !self.cells.is_empty() || self.html_structure.is_some()
    }
}

/// Type of table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TableType {
    /// Table with visible borders
    Wired,
    /// Table without visible borders
    Wireless,
    /// Unknown table type
    Unknown,
}

/// A cell in a table.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableCell {
    /// Bounding box of the cell
    pub bbox: BoundingBox,
    /// Row index (0-based)
    pub row: Option<usize>,
    /// Column index (0-based)
    pub col: Option<usize>,
    /// Row span
    pub row_span: Option<usize>,
    /// Column span
    pub col_span: Option<usize>,
    /// Confidence score for the cell detection
    pub confidence: f32,
    /// Text content of the cell (if available)
    pub text: Option<String>,
}

impl TableCell {
    /// Creates a new table cell.
    pub fn new(bbox: BoundingBox, confidence: f32) -> Self {
        Self {
            bbox,
            row: None,
            col: None,
            row_span: None,
            col_span: None,
            confidence,
            text: None,
        }
    }

    /// Sets the row and column indices.
    pub fn with_position(mut self, row: usize, col: usize) -> Self {
        self.row = Some(row);
        self.col = Some(col);
        self
    }

    /// Sets the row and column spans.
    pub fn with_span(mut self, row_span: usize, col_span: usize) -> Self {
        self.row_span = Some(row_span);
        self.col_span = Some(col_span);
        self
    }

    /// Sets the text content.
    pub fn with_text(mut self, text: impl Into<String>) -> Self {
        self.text = Some(text.into());
        self
    }
}

/// Result of formula recognition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormulaResult {
    /// Bounding box of the formula in the original image
    pub bbox: BoundingBox,
    /// LaTeX representation of the formula
    pub latex: String,
    /// Confidence score for the recognition
    pub confidence: f32,
}

impl FormulaResult {
    /// Creates a new formula result.
    pub fn new(bbox: BoundingBox, latex: impl Into<String>, confidence: f32) -> Self {
        Self {
            bbox,
            latex: latex.into(),
            confidence,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_structure_result_creation() {
        let result = StructureResult::new("test.jpg", 0);
        assert_eq!(result.input_path.as_ref(), "test.jpg");
        assert_eq!(result.index, 0);
        assert!(result.layout_elements.is_empty());
        assert!(result.tables.is_empty());
        assert!(result.formulas.is_empty());
        assert!(result.text_regions.is_none());
    }

    #[test]
    fn test_layout_element_type_as_str() {
        assert_eq!(LayoutElementType::Text.as_str(), "text");
        assert_eq!(LayoutElementType::Table.as_str(), "table");
        assert_eq!(LayoutElementType::Formula.as_str(), "formula");
    }

    #[test]
    fn test_table_result_creation() {
        let bbox = BoundingBox::from_coords(0.0, 0.0, 100.0, 100.0);
        let table = TableResult::new(bbox, TableType::Wired);
        assert_eq!(table.table_type, TableType::Wired);
        assert!(table.cells.is_empty());
        assert!(table.html_structure.is_none());
    }

    #[test]
    fn test_structure_result_export() {
        let bbox = BoundingBox::from_coords(0.0, 0.0, 100.0, 100.0);
        let mut result = StructureResult::new("test.jpg", 0);

        let title = LayoutElement::new(bbox.clone(), LayoutElementType::DocTitle, 1.0)
            .with_text("Test Document");

        let text =
            LayoutElement::new(bbox.clone(), LayoutElementType::Text, 1.0).with_text("Hello world");

        result = result.with_layout_elements(vec![title, text]);

        let md = result.to_markdown();
        assert!(md.contains("# Test Document"));
        assert!(md.contains("Hello world"));

        let html = result.to_html();
        assert!(html.contains("<h1>Test Document</h1>"));
        assert!(html.contains("<p>Hello world</p>"));
    }

    #[test]
    fn test_format_title_with_level_keywords() {
        let (level, text) = format_title_with_level("Abstract", None);
        assert_eq!(level, 1);
        assert_eq!(text, "Abstract");

        let (level, text) = format_title_with_level("References:", None);
        assert_eq!(level, 1);
        assert_eq!(text, "References:");
    }

    #[test]
    fn test_format_title_with_level_cluster_fallback() {
        let (level, text) = format_title_with_level("Unnumbered Heading", Some(4));
        assert_eq!(level, 4);
        assert_eq!(text, "Unnumbered Heading");
    }

    #[test]
    fn test_to_markdown_skips_footnote() {
        let mut result = StructureResult::new("test.jpg", 0);
        let body = LayoutElement::new(
            BoundingBox::from_coords(0.0, 0.0, 100.0, 30.0),
            LayoutElementType::Text,
            1.0,
        )
        .with_text("Body");
        let footnote = LayoutElement::new(
            BoundingBox::from_coords(0.0, 40.0, 100.0, 60.0),
            LayoutElementType::Footnote,
            1.0,
        )
        .with_text("Footnote text");
        result = result.with_layout_elements(vec![body, footnote]);

        let md = result.to_markdown();
        assert!(md.contains("Body"));
        assert!(!md.contains("Footnote text"));
    }

    #[test]
    fn test_to_markdown_doc_title_joins_lines_with_space() {
        let mut result = StructureResult::new("test.jpg", 0);
        let title = LayoutElement::new(
            BoundingBox::from_coords(0.0, 0.0, 100.0, 20.0),
            LayoutElementType::DocTitle,
            1.0,
        )
        .with_text("Main\nTitle");
        result = result.with_layout_elements(vec![title]);
        let md = result.to_markdown();
        assert!(md.contains("# Main Title"));
    }

    #[test]
    fn test_to_markdown_content_uses_soft_breaks() {
        let mut result = StructureResult::new("test.jpg", 0);
        let toc = LayoutElement::new(
            BoundingBox::from_coords(0.0, 0.0, 100.0, 40.0),
            LayoutElementType::Content,
            1.0,
        )
        .with_text("1 Intro\n2 Method");
        result = result.with_layout_elements(vec![toc]);
        let md = result.to_markdown();
        assert!(md.contains("1 Intro  \n2 Method"));
    }

    #[test]
    fn test_infer_paragraph_title_levels_by_height() {
        let titles = vec![
            LayoutElement::new(
                BoundingBox::from_coords(0.0, 0.0, 100.0, 40.0),
                LayoutElementType::ParagraphTitle,
                1.0,
            )
            .with_text("Large"),
            LayoutElement::new(
                BoundingBox::from_coords(0.0, 50.0, 100.0, 74.0),
                LayoutElementType::ParagraphTitle,
                1.0,
            )
            .with_text("Medium"),
            LayoutElement::new(
                BoundingBox::from_coords(0.0, 80.0, 100.0, 98.0),
                LayoutElementType::ParagraphTitle,
                1.0,
            )
            .with_text("Small"),
        ];

        let levels = infer_paragraph_title_levels(&titles);
        let l0 = levels.get(&0).copied().unwrap_or(2);
        let l1 = levels.get(&1).copied().unwrap_or(2);
        let l2 = levels.get(&2).copied().unwrap_or(2);
        assert!(l0 <= l1 && l1 <= l2);
    }

    #[test]
    fn test_infer_paragraph_title_levels_semantic_vote_wins_tie() {
        let titles = vec![
            LayoutElement::new(
                BoundingBox::from_coords(0.0, 0.0, 100.0, 40.0),
                LayoutElementType::ParagraphTitle,
                1.0,
            )
            .with_text("1.1 Detail"),
            LayoutElement::new(
                BoundingBox::from_coords(0.0, 50.0, 100.0, 70.0),
                LayoutElementType::ParagraphTitle,
                1.0,
            )
            .with_text("2 Intro"),
        ];

        let levels = infer_paragraph_title_levels(&titles);
        assert_eq!(levels.get(&0).copied(), Some(3));
        assert_eq!(levels.get(&1).copied(), Some(2));
    }

    #[test]
    fn test_infer_paragraph_title_levels_uses_relative_indent_signal() {
        let titles = vec![
            LayoutElement::new(
                BoundingBox::from_coords(0.0, 0.0, 100.0, 24.0),
                LayoutElementType::ParagraphTitle,
                1.0,
            )
            .with_text("Heading A"),
            LayoutElement::new(
                BoundingBox::from_coords(40.0, 40.0, 140.0, 64.0),
                LayoutElementType::ParagraphTitle,
                1.0,
            )
            .with_text("Heading B"),
        ];

        let levels = infer_paragraph_title_levels(&titles);
        let left_level = levels.get(&0).copied().unwrap_or(2);
        let indented_level = levels.get(&1).copied().unwrap_or(2);
        assert!(left_level < indented_level);
    }
}
