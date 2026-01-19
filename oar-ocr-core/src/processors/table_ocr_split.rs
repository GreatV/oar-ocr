//! Table OCR box splitting utilities.
//!
//! This module provides functionality to split OCR boxes that span multiple table cells,
//! which improves table recognition accuracy for complex tables with rowspan/colspan.
//!
//! ## Problem
//!
//! When processing complex tables, OCR text boxes sometimes span multiple cells. Without
//! splitting, the entire text is assigned to a single cell, causing incorrect table structure.
//!
//! ## Solution
//!
//! This module detects cross-cell OCR boxes and splits them at cell boundaries:
//! 1. Detect OCR boxes that overlap with multiple cells (using IoA)
//! 2. Determine split boundaries based on cell edges
//! 3. Split the bounding box and distribute text proportionally
//!
//! ## Reference
//!
//! This implementation is inspired by PaddleX's `split_ocr_bboxes_by_table_cells()`.

use crate::domain::structure::TableCell;
use crate::domain::text_region::TextRegion;
use crate::processors::BoundingBox;
use std::sync::Arc;

/// Configuration for OCR box splitting.
#[derive(Debug, Clone)]
pub struct SplitConfig {
    /// Minimum overlap ratio (IoA) for considering a cell as affected.
    /// An OCR box is considered to overlap a cell if IoA > this threshold.
    pub min_overlap_ratio: f32,

    /// Minimum number of cells an OCR box must span to be considered for splitting.
    /// Boxes spanning fewer cells are not split.
    pub min_cells_to_split: usize,

    /// Whether to split horizontally (along x-axis) when OCR box spans columns.
    pub split_horizontal: bool,

    /// Whether to split vertically (along y-axis) when OCR box spans rows.
    pub split_vertical: bool,
}

impl Default for SplitConfig {
    fn default() -> Self {
        Self {
            min_overlap_ratio: 0.05, // Low threshold to catch partial overlaps
            min_cells_to_split: 2,
            split_horizontal: true,
            split_vertical: true,
        }
    }
}

/// Detection result for a cross-cell OCR box.
#[derive(Debug, Clone)]
pub struct CrossCellDetection {
    /// Index of the OCR box in the original array.
    pub ocr_index: usize,

    /// Indices of cells that this OCR box overlaps with.
    pub affected_cell_indices: Vec<usize>,

    /// Split boundaries - x-coordinates for horizontal splits.
    pub x_boundaries: Vec<f32>,

    /// Split boundaries - y-coordinates for vertical splits.
    pub y_boundaries: Vec<f32>,

    /// Split direction: true for horizontal (column-wise), false for vertical (row-wise).
    pub is_horizontal_split: bool,
}

/// A segment resulting from splitting an OCR box.
#[derive(Debug, Clone)]
pub struct SplitSegment {
    /// The bounding box of this segment.
    pub bbox: BoundingBox,

    /// The text content assigned to this segment.
    pub text: String,

    /// Index of the cell this segment belongs to.
    pub cell_index: usize,
}

/// Result of splitting a single OCR box.
#[derive(Debug, Clone)]
pub struct SplitOcrResult {
    /// Original OCR box bounding box.
    pub original_bbox: BoundingBox,

    /// Original text content.
    pub original_text: String,

    /// Confidence score from original OCR.
    pub confidence: Option<f32>,

    /// Resulting segments after splitting.
    pub segments: Vec<SplitSegment>,
}

/// Detects OCR boxes that span multiple table cells.
///
/// This function analyzes each OCR box to determine if it overlaps with multiple cells
/// based on the IoA (Intersection over Area of OCR box) metric.
///
/// # Arguments
///
/// * `text_regions` - Slice of text regions from OCR.
/// * `cells` - Slice of table cells.
/// * `config` - Configuration for the detection.
///
/// # Returns
///
/// Vector of `CrossCellDetection` results for OCR boxes that span multiple cells.
pub fn detect_cross_cell_ocr_boxes(
    text_regions: &[TextRegion],
    cells: &[TableCell],
    config: &SplitConfig,
) -> Vec<CrossCellDetection> {
    let mut detections = Vec::new();

    if cells.is_empty() || text_regions.is_empty() {
        return detections;
    }

    for (ocr_idx, region) in text_regions.iter().enumerate() {
        // Skip regions without text
        if region.text.is_none() {
            continue;
        }

        let ocr_bbox = &region.bounding_box;
        let ocr_area = calculate_bbox_area(ocr_bbox);

        if ocr_area <= 0.0 {
            continue;
        }

        // Find all cells that this OCR box overlaps with
        let mut overlapping_cells: Vec<(usize, f32)> = Vec::new();

        for (cell_idx, cell) in cells.iter().enumerate() {
            let inter_area = ocr_bbox.intersection_area(&cell.bbox);
            let ioa = inter_area / ocr_area; // IoA = intersection / OCR area

            if ioa > config.min_overlap_ratio {
                overlapping_cells.push((cell_idx, ioa));
            }
        }

        // Only consider splitting if OCR box spans multiple cells
        if overlapping_cells.len() >= config.min_cells_to_split {
            // Sort by cell index to maintain consistent ordering
            overlapping_cells.sort_by_key(|(idx, _)| *idx);

            let affected_cell_indices: Vec<usize> =
                overlapping_cells.iter().map(|(idx, _)| *idx).collect();

            // Determine split direction and boundaries
            let (x_boundaries, y_boundaries, is_horizontal) =
                compute_split_boundaries(ocr_bbox, &affected_cell_indices, cells, config);

            // Only add detection if we have valid split boundaries
            if !x_boundaries.is_empty() || !y_boundaries.is_empty() {
                detections.push(CrossCellDetection {
                    ocr_index: ocr_idx,
                    affected_cell_indices,
                    x_boundaries,
                    y_boundaries,
                    is_horizontal_split: is_horizontal,
                });
            }
        }
    }

    detections
}

/// Computes split boundaries based on cell edges.
///
/// Returns (x_boundaries, y_boundaries, is_horizontal_split).
fn compute_split_boundaries(
    ocr_bbox: &BoundingBox,
    cell_indices: &[usize],
    cells: &[TableCell],
    config: &SplitConfig,
) -> (Vec<f32>, Vec<f32>, bool) {
    if cell_indices.is_empty() {
        return (Vec::new(), Vec::new(), true);
    }

    // Collect all cell boundaries within the OCR box range
    let mut x_edges: Vec<f32> = Vec::new();
    let mut y_edges: Vec<f32> = Vec::new();

    let ocr_x_min = ocr_bbox.x_min();
    let ocr_x_max = ocr_bbox.x_max();
    let ocr_y_min = ocr_bbox.y_min();
    let ocr_y_max = ocr_bbox.y_max();

    for &cell_idx in cell_indices {
        let cell = &cells[cell_idx];

        // Collect vertical boundaries (x edges) for horizontal splitting
        if config.split_horizontal {
            let cell_x_min = cell.bbox.x_min();
            let cell_x_max = cell.bbox.x_max();

            // Add left edge if it's inside the OCR box
            if cell_x_min > ocr_x_min && cell_x_min < ocr_x_max {
                x_edges.push(cell_x_min);
            }
            // Add right edge if it's inside the OCR box
            if cell_x_max > ocr_x_min && cell_x_max < ocr_x_max {
                x_edges.push(cell_x_max);
            }
        }

        // Collect horizontal boundaries (y edges) for vertical splitting
        if config.split_vertical {
            let cell_y_min = cell.bbox.y_min();
            let cell_y_max = cell.bbox.y_max();

            // Add top edge if it's inside the OCR box
            if cell_y_min > ocr_y_min && cell_y_min < ocr_y_max {
                y_edges.push(cell_y_min);
            }
            // Add bottom edge if it's inside the OCR box
            if cell_y_max > ocr_y_min && cell_y_max < ocr_y_max {
                y_edges.push(cell_y_max);
            }
        }
    }

    // Remove duplicates and sort
    x_edges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    x_edges.dedup_by(|a, b| (*a - *b).abs() < 1.0); // Remove edges within 1 pixel

    y_edges.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    y_edges.dedup_by(|a, b| (*a - *b).abs() < 1.0);

    // Determine primary split direction based on which has more boundaries
    // or based on the OCR box aspect ratio
    let ocr_width = ocr_x_max - ocr_x_min;
    let ocr_height = ocr_y_max - ocr_y_min;

    let is_horizontal = if !x_edges.is_empty() && !y_edges.is_empty() {
        // Both directions have boundaries - prefer horizontal for wide boxes
        ocr_width >= ocr_height
    } else {
        !x_edges.is_empty()
    };

    // Return only the relevant boundaries based on split direction
    if is_horizontal {
        (x_edges, Vec::new(), true)
    } else {
        (Vec::new(), y_edges, false)
    }
}

/// Splits an OCR box at cell boundaries and distributes text proportionally.
///
/// # Arguments
///
/// * `region` - The text region to split.
/// * `detection` - The cross-cell detection result.
/// * `cells` - Slice of table cells.
///
/// # Returns
///
/// `SplitOcrResult` containing the original box info and resulting segments.
pub fn split_ocr_box_at_cell_boundaries(
    region: &TextRegion,
    detection: &CrossCellDetection,
    cells: &[TableCell],
) -> SplitOcrResult {
    let original_bbox = region.bounding_box.clone();
    let original_text = region
        .text
        .as_ref()
        .map(|s| s.to_string())
        .unwrap_or_default();
    let confidence = region.confidence;

    if original_text.is_empty() || detection.affected_cell_indices.is_empty() {
        return SplitOcrResult {
            original_bbox,
            original_text,
            confidence,
            segments: Vec::new(),
        };
    }

    let segments = if detection.is_horizontal_split && !detection.x_boundaries.is_empty() {
        split_horizontally(
            &original_bbox,
            &original_text,
            &detection.x_boundaries,
            &detection.affected_cell_indices,
            cells,
        )
    } else if !detection.y_boundaries.is_empty() {
        split_vertically(
            &original_bbox,
            &original_text,
            &detection.y_boundaries,
            &detection.affected_cell_indices,
            cells,
        )
    } else {
        // Fallback: assign to first affected cell
        vec![SplitSegment {
            bbox: original_bbox.clone(),
            text: original_text.clone(),
            cell_index: detection.affected_cell_indices[0],
        }]
    };

    SplitOcrResult {
        original_bbox,
        original_text,
        confidence,
        segments,
    }
}

/// Splits a bounding box horizontally (along x-axis) and distributes text.
fn split_horizontally(
    ocr_bbox: &BoundingBox,
    text: &str,
    x_boundaries: &[f32],
    cell_indices: &[usize],
    cells: &[TableCell],
) -> Vec<SplitSegment> {
    let mut segments = Vec::new();

    let ocr_x_min = ocr_bbox.x_min();
    let ocr_x_max = ocr_bbox.x_max();
    let ocr_y_min = ocr_bbox.y_min();
    let ocr_y_max = ocr_bbox.y_max();
    let ocr_width = ocr_x_max - ocr_x_min;

    if ocr_width <= 0.0 {
        return segments;
    }

    // Build segment x-ranges from boundaries
    let mut x_ranges: Vec<(f32, f32)> = Vec::new();
    let mut prev_x = ocr_x_min;

    for &boundary_x in x_boundaries {
        if boundary_x > prev_x && boundary_x < ocr_x_max {
            x_ranges.push((prev_x, boundary_x));
            prev_x = boundary_x;
        }
    }
    // Add final segment
    if prev_x < ocr_x_max {
        x_ranges.push((prev_x, ocr_x_max));
    }

    if x_ranges.is_empty() {
        return segments;
    }

    // Calculate width ratios for text distribution
    let total_width: f32 = x_ranges.iter().map(|(x1, x2)| x2 - x1).sum();
    let ratios: Vec<f32> = x_ranges
        .iter()
        .map(|(x1, x2)| (x2 - x1) / total_width)
        .collect();

    // Split text by ratios
    let text_parts = split_text_by_ratio(text, &ratios);

    // Create segments and assign to cells
    for ((x1, x2), text_part) in x_ranges.iter().zip(text_parts.iter()) {
        let segment_bbox = BoundingBox::from_coords(*x1, ocr_y_min, *x2, ocr_y_max);

        // Find the best matching cell for this segment
        let cell_index = find_best_matching_cell(&segment_bbox, cell_indices, cells);

        segments.push(SplitSegment {
            bbox: segment_bbox,
            text: text_part.clone(),
            cell_index,
        });
    }

    segments
}

/// Splits a bounding box vertically (along y-axis) and distributes text.
fn split_vertically(
    ocr_bbox: &BoundingBox,
    text: &str,
    y_boundaries: &[f32],
    cell_indices: &[usize],
    cells: &[TableCell],
) -> Vec<SplitSegment> {
    let mut segments = Vec::new();

    let ocr_x_min = ocr_bbox.x_min();
    let ocr_x_max = ocr_bbox.x_max();
    let ocr_y_min = ocr_bbox.y_min();
    let ocr_y_max = ocr_bbox.y_max();
    let ocr_height = ocr_y_max - ocr_y_min;

    if ocr_height <= 0.0 {
        return segments;
    }

    // Build segment y-ranges from boundaries
    let mut y_ranges: Vec<(f32, f32)> = Vec::new();
    let mut prev_y = ocr_y_min;

    for &boundary_y in y_boundaries {
        if boundary_y > prev_y && boundary_y < ocr_y_max {
            y_ranges.push((prev_y, boundary_y));
            prev_y = boundary_y;
        }
    }
    // Add final segment
    if prev_y < ocr_y_max {
        y_ranges.push((prev_y, ocr_y_max));
    }

    if y_ranges.is_empty() {
        return segments;
    }

    // For vertical splits, try to split by lines first
    let lines: Vec<&str> = text.lines().collect();

    if lines.len() >= y_ranges.len() {
        // Distribute lines across segments
        let lines_per_segment = lines.len() / y_ranges.len();
        let mut line_idx = 0;

        for (i, (y1, y2)) in y_ranges.iter().enumerate() {
            let segment_bbox = BoundingBox::from_coords(ocr_x_min, *y1, ocr_x_max, *y2);

            // Calculate how many lines this segment gets
            let num_lines = if i == y_ranges.len() - 1 {
                lines.len() - line_idx // Last segment gets remaining lines
            } else {
                lines_per_segment
            };

            let segment_text: String = lines[line_idx..line_idx + num_lines].join("\n");
            line_idx += num_lines;

            let cell_index = find_best_matching_cell(&segment_bbox, cell_indices, cells);

            segments.push(SplitSegment {
                bbox: segment_bbox,
                text: segment_text,
                cell_index,
            });
        }
    } else {
        // Fall back to ratio-based splitting
        let total_height: f32 = y_ranges.iter().map(|(y1, y2)| y2 - y1).sum();
        let ratios: Vec<f32> = y_ranges
            .iter()
            .map(|(y1, y2)| (y2 - y1) / total_height)
            .collect();

        let text_parts = split_text_by_ratio(text, &ratios);

        for ((y1, y2), text_part) in y_ranges.iter().zip(text_parts.iter()) {
            let segment_bbox = BoundingBox::from_coords(ocr_x_min, *y1, ocr_x_max, *y2);
            let cell_index = find_best_matching_cell(&segment_bbox, cell_indices, cells);

            segments.push(SplitSegment {
                bbox: segment_bbox,
                text: text_part.clone(),
                cell_index,
            });
        }
    }

    segments
}

/// Finds the best matching cell for a segment bbox.
fn find_best_matching_cell(
    segment_bbox: &BoundingBox,
    candidate_indices: &[usize],
    cells: &[TableCell],
) -> usize {
    let mut best_cell_idx = candidate_indices.first().copied().unwrap_or(0);
    let mut best_iou = 0.0f32;

    for &cell_idx in candidate_indices {
        if cell_idx >= cells.len() {
            continue;
        }

        let iou = segment_bbox.iou(&cells[cell_idx].bbox);
        if iou > best_iou {
            best_iou = iou;
            best_cell_idx = cell_idx;
        }
    }

    best_cell_idx
}

/// Splits text into parts based on given ratios.
///
/// This function attempts to split text at word boundaries when possible,
/// distributing characters roughly according to the specified ratios.
///
/// # Arguments
///
/// * `text` - The text to split.
/// * `ratios` - Slice of ratios (should sum to ~1.0).
///
/// # Returns
///
/// Vector of text parts, one for each ratio.
pub fn split_text_by_ratio(text: &str, ratios: &[f32]) -> Vec<String> {
    if ratios.is_empty() {
        return vec![text.to_string()];
    }

    if ratios.len() == 1 {
        return vec![text.to_string()];
    }

    let chars: Vec<char> = text.chars().collect();
    let total_chars = chars.len();

    if total_chars == 0 {
        return ratios.iter().map(|_| String::new()).collect();
    }

    // Normalize ratios
    let total_ratio: f32 = ratios.iter().sum();
    let normalized_ratios: Vec<f32> = if total_ratio > 0.0 {
        ratios.iter().map(|r| r / total_ratio).collect()
    } else {
        let equal = 1.0 / ratios.len() as f32;
        vec![equal; ratios.len()]
    };

    let mut result = Vec::with_capacity(ratios.len());
    let mut start_idx = 0;

    for (i, ratio) in normalized_ratios.iter().enumerate() {
        let chars_for_segment = if i == normalized_ratios.len() - 1 {
            // Last segment gets remaining characters
            total_chars - start_idx
        } else {
            (total_chars as f32 * ratio).round() as usize
        };

        let end_idx = (start_idx + chars_for_segment).min(total_chars);

        // Try to find a word boundary near the split point
        let adjusted_end_idx = if end_idx < total_chars && end_idx > start_idx {
            find_word_boundary(&chars, start_idx, end_idx)
        } else {
            end_idx
        };

        let segment: String = chars[start_idx..adjusted_end_idx].iter().collect();
        result.push(segment.trim().to_string());

        start_idx = adjusted_end_idx;
    }

    // Handle any remaining characters
    if start_idx < total_chars && !result.is_empty() {
        let remaining: String = chars[start_idx..].iter().collect();
        if let Some(last) = result.last_mut()
            && !remaining.trim().is_empty()
        {
            last.push_str(remaining.trim());
        }
    }

    result
}

/// Finds a suitable word boundary near the target split point.
fn find_word_boundary(chars: &[char], start: usize, target_end: usize) -> usize {
    // Search within a small window around the target
    let window = 5.min(target_end - start);

    // Look for space or punctuation near the target
    for offset in 0..window {
        let check_idx = target_end.saturating_sub(offset);
        if check_idx > start
            && check_idx < chars.len()
            && (chars[check_idx].is_whitespace()
                || chars[check_idx] == ','
                || chars[check_idx] == '.')
        {
            return check_idx + 1;
        }
    }

    // No word boundary found, use original target
    target_end
}

/// Calculates the area of a bounding box.
fn calculate_bbox_area(bbox: &BoundingBox) -> f32 {
    let width = bbox.x_max() - bbox.x_min();
    let height = bbox.y_max() - bbox.y_min();
    (width * height).max(0.0)
}

/// Creates expanded OCR results after splitting cross-cell boxes.
///
/// This is a convenience function that takes the original text regions,
/// detects cross-cell boxes, splits them, and returns an expanded list
/// of text regions suitable for table stitching.
///
/// # Arguments
///
/// * `text_regions` - Original text regions from OCR.
/// * `cells` - Table cells.
/// * `config` - Optional split configuration.
///
/// # Returns
///
/// A tuple of (expanded_regions, processed_indices) where:
/// - expanded_regions: New text regions after splitting
/// - processed_indices: Indices of original regions that were split
pub fn create_expanded_ocr_for_table(
    text_regions: &[TextRegion],
    cells: &[TableCell],
    config: Option<&SplitConfig>,
) -> (Vec<TextRegion>, std::collections::HashSet<usize>) {
    let default_config = SplitConfig::default();
    let config = config.unwrap_or(&default_config);

    let detections = detect_cross_cell_ocr_boxes(text_regions, cells, config);

    let mut expanded_regions = Vec::new();
    let mut processed_indices = std::collections::HashSet::new();

    for detection in &detections {
        processed_indices.insert(detection.ocr_index);

        let region = &text_regions[detection.ocr_index];
        let split_result = split_ocr_box_at_cell_boundaries(region, detection, cells);

        for segment in split_result.segments {
            if !segment.text.is_empty() {
                let new_region = TextRegion::with_recognition(
                    segment.bbox,
                    Some(Arc::from(segment.text.as_str())),
                    split_result.confidence,
                );
                expanded_regions.push(new_region);
            }
        }
    }

    (expanded_regions, processed_indices)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_region(x1: f32, y1: f32, x2: f32, y2: f32, text: &str) -> TextRegion {
        TextRegion::with_recognition(
            BoundingBox::from_coords(x1, y1, x2, y2),
            Some(Arc::from(text)),
            Some(0.9),
        )
    }

    fn make_cell(x1: f32, y1: f32, x2: f32, y2: f32) -> TableCell {
        TableCell::new(BoundingBox::from_coords(x1, y1, x2, y2), 0.9)
    }

    #[test]
    fn test_detect_no_cross_cell_ocr() {
        // OCR box fully inside one cell
        let regions = vec![make_region(10.0, 10.0, 90.0, 40.0, "Hello World")];

        let cells = vec![
            make_cell(0.0, 0.0, 100.0, 50.0),
            make_cell(100.0, 0.0, 200.0, 50.0),
        ];

        let config = SplitConfig::default();
        let detections = detect_cross_cell_ocr_boxes(&regions, &cells, &config);

        assert!(
            detections.is_empty(),
            "Should not detect cross-cell for box fully inside one cell"
        );
    }

    #[test]
    fn test_detect_cross_cell_horizontal() {
        // OCR box spans two cells horizontally
        let regions = vec![make_region(50.0, 10.0, 150.0, 40.0, "Header Text")];

        let cells = vec![
            make_cell(0.0, 0.0, 100.0, 50.0),
            make_cell(100.0, 0.0, 200.0, 50.0),
        ];

        let config = SplitConfig::default();
        let detections = detect_cross_cell_ocr_boxes(&regions, &cells, &config);

        assert_eq!(detections.len(), 1, "Should detect one cross-cell OCR box");
        assert_eq!(detections[0].affected_cell_indices.len(), 2);
        assert!(detections[0].is_horizontal_split);
    }

    #[test]
    fn test_split_text_by_ratio_equal() {
        let text = "ABCDEFGHIJ";
        let ratios = vec![0.5, 0.5];

        let parts = split_text_by_ratio(text, &ratios);

        assert_eq!(parts.len(), 2);
        // Total should be original length
        let total_len: usize = parts.iter().map(|s| s.len()).sum();
        assert_eq!(total_len, text.len());
    }

    #[test]
    fn test_split_text_by_ratio_unequal() {
        let text = "Hello World";
        let ratios = vec![0.3, 0.7];

        let parts = split_text_by_ratio(text, &ratios);

        assert_eq!(parts.len(), 2);
        // Parts should be non-empty
        assert!(!parts[0].is_empty() || !parts[1].is_empty());
    }

    #[test]
    fn test_split_text_empty() {
        let text = "";
        let ratios = vec![0.5, 0.5];

        let parts = split_text_by_ratio(text, &ratios);

        assert_eq!(parts.len(), 2);
        assert!(parts[0].is_empty());
        assert!(parts[1].is_empty());
    }

    #[test]
    fn test_split_ocr_box_horizontal() {
        let region = make_region(50.0, 10.0, 150.0, 40.0, "Col1 Col2");

        let cells = vec![
            make_cell(0.0, 0.0, 100.0, 50.0),
            make_cell(100.0, 0.0, 200.0, 50.0),
        ];

        let detection = CrossCellDetection {
            ocr_index: 0,
            affected_cell_indices: vec![0, 1],
            x_boundaries: vec![100.0],
            y_boundaries: Vec::new(),
            is_horizontal_split: true,
        };

        let result = split_ocr_box_at_cell_boundaries(&region, &detection, &cells);

        assert_eq!(result.segments.len(), 2, "Should produce 2 segments");

        // Verify segment bboxes don't overlap
        let seg1_x_max = result.segments[0].bbox.x_max();
        let seg2_x_min = result.segments[1].bbox.x_min();
        assert!(
            seg1_x_max <= seg2_x_min + 1.0,
            "Segments should not overlap"
        );
    }

    #[test]
    fn test_create_expanded_ocr_for_table() {
        let regions = vec![
            make_region(10.0, 10.0, 90.0, 40.0, "Cell1 Only"), // Inside cell 0
            make_region(50.0, 10.0, 150.0, 40.0, "Across Cells"), // Spans cells 0 and 1
        ];

        let cells = vec![
            make_cell(0.0, 0.0, 100.0, 50.0),
            make_cell(100.0, 0.0, 200.0, 50.0),
        ];

        let config = SplitConfig::default();
        let (expanded, processed) = create_expanded_ocr_for_table(&regions, &cells, Some(&config));

        // Second region should be split
        assert!(processed.contains(&1));
        assert!(!processed.contains(&0));

        // Should have created new regions from the split
        assert!(!expanded.is_empty());
    }
}
