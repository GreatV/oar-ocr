//! Markdown export utilities with image extraction for examples
//!
//! This module provides I/O functions for exporting markdown with extracted images.
//! The library crate provides pure transformations (e.g., `StructureResult::to_markdown()`),
//! while these examples utilities handle the file system operations.

use oar_ocr::domain::structure::{LayoutElementType, StructureResult};
use oar_ocr::processors::BoundingBox;
use std::path::Path;

/// Exports markdown with extracted images saved to disk.
///
/// This function generates markdown content and extracts image/chart regions
/// from the source image, saving them to the output directory.
///
/// # Arguments
///
/// * `result` - Structure result containing layout elements and rectified image
/// * `output_dir` - Directory to save extracted images (an `imgs` subdirectory will be created)
///
/// # Returns
///
/// A markdown string with relative paths to the saved images
pub fn export_markdown_with_images(
    result: &StructureResult,
    output_dir: impl AsRef<Path>,
) -> std::io::Result<String> {
    let output_dir = output_dir.as_ref();
    let imgs_dir = output_dir.join("imgs");

    // Create imgs directory if it doesn't exist
    if !imgs_dir.exists() {
        std::fs::create_dir_all(&imgs_dir)?;
    }

    // Collect table bboxes for overlap filtering
    let table_bboxes: Vec<&BoundingBox> = result
        .layout_elements
        .iter()
        .filter(|e| e.element_type == LayoutElementType::Table)
        .map(|e| &e.bbox)
        .collect();

    let mut md = String::new();
    let mut img_counter = 0usize;
    let elements = &result.layout_elements;

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
        if element.element_type == LayoutElementType::Text {
            let overlaps_table = table_bboxes
                .iter()
                .any(|table_bbox| element.bbox.ioa(table_bbox) > 0.3);
            if overlaps_table && element.confidence < 0.7 {
                continue;
            }
        }

        match element.element_type {
            // Document title
            LayoutElementType::DocTitle => {
                md.push_str("\n# ");
                if let Some(text) = &element.text {
                    let cleaned = clean_ocr_text(text);
                    md.push_str(&cleaned);
                }
                md.push_str("\n\n");
            }
            // Paragraph/section title
            LayoutElementType::ParagraphTitle => {
                if let Some(text) = &element.text {
                    let cleaned = clean_ocr_text(text);
                    let (level, formatted_title) = format_title_with_level(&cleaned);
                    md.push('\n');
                    for _ in 0..level {
                        md.push('#');
                    }
                    md.push(' ');
                    md.push_str(&formatted_title);
                    md.push_str("\n\n");
                } else {
                    md.push_str("\n## \n\n");
                }
            }
            // Table
            LayoutElementType::Table => {
                if let Some(table) = result
                    .tables
                    .iter()
                    .find(|t| t.bbox.iou(&element.bbox) > 0.5)
                {
                    if let Some(html) = &table.html_structure {
                        let simplified = simplify_table_html(html);
                        let table_with_border =
                            simplified.replacen("<table>", "<table border=\"1\">", 1);
                        md.push_str("\n<div style=\"text-align: center;\">");
                        md.push_str(&table_with_border);
                        md.push_str("</div>\n\n");
                    } else {
                        md.push_str("\n[Table]\n\n");
                    }
                } else {
                    md.push_str("\n[Table]\n\n");
                }
            }
            // Formula - detect inline vs display formula based on context
            LayoutElementType::Formula | LayoutElementType::FormulaNumber => {
                let is_inline = {
                    let has_prev_text = (0..idx).rev().any(|i| {
                        let prev = &elements[i];
                        !prev.element_type.is_formula()
                            && (prev.element_type == LayoutElementType::Text
                                || prev.element_type == LayoutElementType::ReferenceContent)
                            && is_same_line(&element.bbox, &prev.bbox)
                    });

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
                    md.push('$');
                    if let Some(latex) = &element.text {
                        md.push_str(latex);
                    }
                    md.push_str("$ ");
                } else {
                    md.push_str("\n$$");
                    if let Some(latex) = &element.text {
                        md.push_str(latex);
                    }
                    md.push_str("$$\n\n");
                }
            }
            // Image/Chart - extract and save image region
            LayoutElementType::Image | LayoutElementType::Chart => {
                let type_name = if element.element_type == LayoutElementType::Chart {
                    "chart"
                } else {
                    "image"
                };

                // Generate image filename
                let img_name = format!(
                    "img_in_{}_box_{:.0}_{:.0}_{:.0}_{:.0}.jpg",
                    type_name,
                    element.bbox.x_min(),
                    element.bbox.y_min(),
                    element.bbox.x_max(),
                    element.bbox.y_max()
                );
                let img_path = imgs_dir.join(&img_name);
                let relative_path = format!("imgs/{}", img_name);

                // Extract and save image region if we have the source image
                if let Some(ref img) = result.rectified_img {
                    let x = element.bbox.x_min().max(0.0) as u32;
                    let y = element.bbox.y_min().max(0.0) as u32;
                    let width = ((element.bbox.x_max() - element.bbox.x_min()) as u32)
                        .min(img.width().saturating_sub(x));
                    let height = ((element.bbox.y_max() - element.bbox.y_min()) as u32)
                        .min(img.height().saturating_sub(y));

                    if width > 0 && height > 0 {
                        let cropped =
                            image::imageops::crop_imm(img.as_ref(), x, y, width, height).to_image();
                        // Save as JPEG
                        if let Err(e) = cropped.save(&img_path) {
                            tracing::warn!("Failed to save image {}: {}", img_path.display(), e);
                        }
                    }
                }

                // Calculate width percentage
                let width_pct =
                    ((element.bbox.x_max() - element.bbox.x_min()) / 12.0).clamp(20.0, 100.0);

                md.push_str("\n<div style=\"text-align: center;\"><img src=\"");
                md.push_str(&relative_path);
                md.push_str("\" alt=\"Image\" width=\"");
                md.push_str(&format!("{:.0}%", width_pct));
                md.push_str("\" /></div>\n\n");

                img_counter += 1;
            }
            // Seal
            LayoutElementType::Seal => {
                md.push_str("\n![Seal]");
                if let Some(text) = &element.text {
                    md.push_str("\n> ");
                    md.push_str(text);
                }
                md.push_str("\n\n");
            }
            // Captions
            _ if element.element_type.is_caption() => {
                if let Some(text) = &element.text {
                    md.push_str("\n<div style=\"text-align: center;\">");
                    md.push_str(text);
                    md.push_str(" </div>\n\n");
                }
            }
            // Abstract
            LayoutElementType::Abstract => {
                if let Some(text) = &element.text {
                    let lower = text.to_lowercase();
                    if lower.contains("abstract") || lower.contains("摘要") {
                        md.push_str("\n## **Abstract**\n\n");
                    }
                    let formatted = format_text_block(text);
                    md.push_str(&formatted);
                    md.push_str("\n\n");
                }
            }
            // Reference
            LayoutElementType::Reference => {
                if let Some(text) = &element.text {
                    let formatted = format_reference_block(text);
                    md.push('\n');
                    md.push_str(&formatted);
                    md.push_str("\n\n");
                }
            }
            // Content
            LayoutElementType::Content => {
                if let Some(text) = &element.text {
                    let formatted = format_content_block(text);
                    md.push('\n');
                    md.push_str(&formatted);
                    md.push_str("\n\n");
                }
            }
            // Footnote
            LayoutElementType::Footnote => {
                if let Some(text) = &element.text {
                    let formatted = format_vision_footnote_block(text);
                    md.push('\n');
                    md.push_str(&formatted);
                    md.push_str("\n\n");
                }
            }
            // List
            LayoutElementType::List => {
                if let Some(text) = &element.text {
                    let cleaned = format_text_block(text);
                    for line in cleaned.lines() {
                        let line = line.trim();
                        if !line.is_empty() {
                            md.push_str("- ");
                            md.push_str(line);
                            md.push('\n');
                        }
                    }
                    md.push('\n');
                }
            }
            // Header/Footer - skip
            _ if element.element_type.is_header() || element.element_type.is_footer() => {
                continue;
            }
            // Default text
            _ => {
                if let Some(text) = &element.text {
                    let formatted = format_text_block(text);
                    md.push_str(&formatted);
                    md.push_str("\n\n");
                }
            }
        }
    }

    tracing::debug!("Extracted {} images to {:?}", img_counter, imgs_dir);
    Ok(md.trim().to_string())
}

/// Exports concatenated markdown from multiple pages with images.
///
/// This follows the same concatenation logic as `concatenate_markdown_pages`
/// but also handles image extraction for all pages.
///
/// # Arguments
///
/// * `results` - Slice of structure results from multiple pages (in order)
/// * `output_dir` - Directory to save extracted images
///
/// # Returns
///
/// A single markdown string with all pages properly concatenated and images extracted
pub fn export_concatenated_markdown_with_images(
    results: &[StructureResult],
    output_dir: impl AsRef<Path>,
) -> std::io::Result<String> {
    let output_dir = output_dir.as_ref();

    if results.is_empty() {
        return Ok(String::new());
    }

    if results.len() == 1 {
        return export_markdown_with_images(&results[0], output_dir);
    }

    let mut markdown = String::new();
    let mut prev_page_end_flag = true;

    for result in results.iter() {
        let flags = result
            .page_continuation_flags
            .as_ref()
            .cloned()
            .unwrap_or_else(|| result.calculate_continuation_flags());

        let page_markdown = export_markdown_with_images(result, output_dir)?;

        if page_markdown.trim().is_empty() {
            prev_page_end_flag = flags.paragraph_end;
            continue;
        }

        let page_first_continues = !flags.paragraph_start;

        if page_first_continues && !prev_page_end_flag {
            let last_char = markdown.chars().last();
            let first_char = page_markdown.chars().next();

            let last_is_chinese = last_char.is_some_and(is_chinese_char);
            let first_is_chinese = first_char.is_some_and(is_chinese_char);

            if !last_is_chinese && !first_is_chinese {
                markdown.push(' ');
                markdown.push_str(page_markdown.trim_start());
            } else {
                markdown.push_str(page_markdown.trim_start());
            }
        } else {
            if !markdown.is_empty() {
                markdown.push_str("\n\n");
            }
            markdown.push_str(&page_markdown);
        }

        prev_page_end_flag = flags.paragraph_end;
    }

    Ok(markdown.trim().to_string())
}

// ============================================================================
// Helper functions (replicated from lib crate)
// ============================================================================

/// Cleans OCR text content by removing common artifacts.
fn clean_ocr_text(text: &str) -> String {
    text.replace("-\n", "").replace('\n', " ")
}

/// Formats text blocks following PaddleX's text handling.
fn format_text_block(text: &str) -> String {
    let dehyphenated = text.replace("-\n", "");
    let step1 = dehyphenated.replace("\n\n", "\n");
    step1.replace('\n', "\n\n")
}

/// Formats content blocks (table of contents).
fn format_content_block(text: &str) -> String {
    let step1 = text.replace("-\n", "  \n");
    step1.replace('\n', "  \n")
}

/// Formats reference blocks.
fn format_reference_block(text: &str) -> String {
    let dehyphenated = text.replace("-\n", "");
    let lines: Vec<&str> = dehyphenated.lines().collect();

    let mut result = String::new();
    let mut added_heading = false;

    for (i, line) in lines.iter().enumerate() {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            continue;
        }

        if !added_heading && (trimmed.contains("References") || trimmed.contains("参考文献")) {
            result.push_str("## **References**\n\n");
            added_heading = true;
            continue;
        }

        if i > 0 || result.is_empty() {
            if !result.is_empty() {
                result.push('\n');
            }
            result.push_str(trimmed);
        }
    }

    if result.is_empty() {
        dehyphenated
    } else {
        result
    }
}

/// Formats vision footnote blocks.
fn format_vision_footnote_block(text: &str) -> String {
    let dehyphenated = text.replace("-\n", "");
    let step1 = dehyphenated.replace("\n\n", "\n");
    step1.replace('\n', "\n\n")
}

/// Simplifies table HTML by removing wrapper tags.
fn simplify_table_html(html: &str) -> String {
    html.replace("<html>", "")
        .replace("</html>", "")
        .replace("<body>", "")
        .replace("</body>", "")
}

/// Checks if two bounding boxes are on the same line.
fn is_same_line(bbox1: &BoundingBox, bbox2: &BoundingBox) -> bool {
    let y1_min = bbox1.y_min();
    let y1_max = bbox1.y_max();
    let y2_min = bbox2.y_min();
    let y2_max = bbox2.y_max();

    let overlap_start = y1_min.max(y2_min);
    let overlap_end = y1_max.min(y2_max);
    let overlap = (overlap_end - overlap_start).max(0.0);

    let height1 = y1_max - y1_min;
    let height2 = y2_max - y2_min;
    let min_height = height1.min(height2);

    min_height > 0.0 && overlap / min_height > 0.5
}

/// Checks if a character is a Chinese character.
fn is_chinese_char(c: char) -> bool {
    matches!(c,
        '\u{4E00}'..='\u{9FFF}' | // CJK Unified Ideographs
        '\u{3400}'..='\u{4DBF}' | // CJK Unified Ideographs Extension A
        '\u{20000}'..='\u{2A6DF}' | // CJK Unified Ideographs Extension B
        '\u{2A700}'..='\u{2B73F}' | // CJK Unified Ideographs Extension C
        '\u{2B740}'..='\u{2B81F}' | // CJK Unified Ideographs Extension D
        '\u{2B820}'..='\u{2CEAF}' | // CJK Unified Ideographs Extension E
        '\u{2CEB0}'..='\u{2EBEF}'   // CJK Unified Ideographs Extension F
    )
}

/// Title numbering pattern for detecting section numbers.
fn is_numbered_title(title: &str) -> (bool, usize, String) {
    use regex::Regex;

    let numbering_regex = Regex::new(
        r"(?x)
        ^\s*
        (
            [1-9][0-9]*(?:\.[1-9][0-9]*)*[\.、]?
            |
            [(（][1-9][0-9]*(?:\.[1-9][0-9]*)*[)）]
            |
            [一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾][、.]?
            |
            [(（][一二三四五六七八九十百千万亿零壹贰叁肆伍陆柒捌玖拾]+[)）]
            |
            (?:I|II|III|IV|V|VI|VII|VIII|IX|X)(?:\.|\b)
        )
        (\s+)
        (.*)
        $
    ",
    )
    .unwrap();

    let cleaned = title.replace("-\n", "").replace('\n', " ");

    if let Some(captures) = numbering_regex.captures(&cleaned) {
        let numbering = captures.get(1).map(|m| m.as_str().trim()).unwrap_or("");
        let title_content = captures.get(3).map(|m| m.as_str()).unwrap_or("");

        let dot_count = numbering.matches('.').count();
        let level = dot_count + 2;

        let formatted = if title_content.is_empty() {
            numbering.trim_end_matches('.').to_string()
        } else {
            format!(
                "{} {}",
                numbering.trim_end_matches('.'),
                title_content.trim_start()
            )
        };

        (true, level.clamp(2, 6), formatted)
    } else {
        (false, 2, cleaned)
    }
}

/// Formats paragraph title with automatic level detection.
fn format_title_with_level(title: &str) -> (usize, String) {
    let (is_numbered, level, formatted) = is_numbered_title(title);
    if is_numbered {
        (level, formatted)
    } else {
        (2, title.replace("-\n", "").replace('\n', " "))
    }
}
