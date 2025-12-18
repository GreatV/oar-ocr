//! Layout parsing utilities.
//!
//! This module provides utilities for layout analysis, including sorting layout boxes
//! and associating OCR results with layout regions. The implementation follows
//! established approaches.

use crate::processors::BoundingBox;
use std::collections::HashSet;

/// Result of associating OCR boxes with layout regions.
#[derive(Debug, Clone)]
pub struct LayoutOCRAssociation {
    /// Indices of OCR boxes that are within the layout regions
    pub matched_indices: Vec<usize>,
    /// Indices of OCR boxes that are outside all layout regions
    pub unmatched_indices: Vec<usize>,
}

/// Get indices of OCR boxes that overlap with layout regions.
///
/// This function checks which OCR boxes have significant overlap with any of the
/// layout regions. An overlap is considered significant if the intersection has
/// both width and height greater than the threshold (default: 3 pixels).
///
/// This follows standard overlap detection implementation.
///
/// # Arguments
///
/// * `ocr_boxes` - Slice of OCR bounding boxes
/// * `layout_regions` - Slice of layout region bounding boxes
/// * `threshold` - Minimum intersection dimension (default: 3.0 pixels)
///
/// # Returns
///
/// Vector of indices of OCR boxes that overlap with any layout region
pub fn get_overlap_boxes_idx(
    ocr_boxes: &[BoundingBox],
    layout_regions: &[BoundingBox],
    threshold: f32,
) -> Vec<usize> {
    let mut matched_indices = Vec::new();

    if ocr_boxes.is_empty() || layout_regions.is_empty() {
        return matched_indices;
    }

    // For each layout region, find overlapping OCR boxes
    for layout_region in layout_regions {
        for (idx, ocr_box) in ocr_boxes.iter().enumerate() {
            if ocr_box.overlaps_with(layout_region, threshold) {
                matched_indices.push(idx);
            }
        }
    }

    matched_indices
}

/// Associate OCR results with layout regions.
///
/// This function filters OCR boxes based on whether they are within or outside
/// the specified layout regions.
///
/// This follows standard region association implementation.
///
/// # Arguments
///
/// * `ocr_boxes` - Slice of OCR bounding boxes
/// * `layout_regions` - Slice of layout region bounding boxes
/// * `flag_within` - If true, return boxes within regions; if false, return boxes outside regions
/// * `threshold` - Minimum intersection dimension for overlap detection
///
/// # Returns
///
/// `LayoutOCRAssociation` containing matched and unmatched indices
pub fn associate_ocr_with_layout(
    ocr_boxes: &[BoundingBox],
    layout_regions: &[BoundingBox],
    flag_within: bool,
    threshold: f32,
) -> LayoutOCRAssociation {
    let overlap_indices = get_overlap_boxes_idx(ocr_boxes, layout_regions, threshold);
    let overlap_set: HashSet<usize> = overlap_indices.into_iter().collect();

    let mut matched_indices = Vec::new();
    let mut unmatched_indices = Vec::new();

    for (idx, _) in ocr_boxes.iter().enumerate() {
        let is_overlapping = overlap_set.contains(&idx);

        if flag_within {
            // Return boxes within regions
            if is_overlapping {
                matched_indices.push(idx);
            } else {
                unmatched_indices.push(idx);
            }
        } else {
            // Return boxes outside regions
            if !is_overlapping {
                matched_indices.push(idx);
            } else {
                unmatched_indices.push(idx);
            }
        }
    }

    LayoutOCRAssociation {
        matched_indices,
        unmatched_indices,
    }
}

/// Layout element with bounding box and label.
#[derive(Debug, Clone)]
pub struct LayoutElement {
    /// Bounding box of the layout element
    pub bbox: BoundingBox,
    /// Label/type of the layout element (e.g., "text", "title", "table", "figure")
    pub label: String,
    /// Optional content text
    pub content: Option<String>,
}

impl LayoutElement {
    /// Create a new layout element.
    pub fn new(bbox: BoundingBox, label: String) -> Self {
        Self {
            bbox,
            label,
            content: None,
        }
    }

    /// Create a layout element with content.
    pub fn with_content(bbox: BoundingBox, label: String, content: String) -> Self {
        Self {
            bbox,
            label,
            content: Some(content),
        }
    }
}

/// Sort layout boxes in reading order with column detection.
///
/// This function sorts layout boxes from top to bottom, left to right, with special
/// handling for two-column layouts. Boxes are first sorted by y-coordinate, then
/// separated into left and right columns based on their x-coordinate.
///
/// The algorithm:
/// 1. Sort boxes by (y, x) coordinates
/// 2. Identify left column boxes (x1 < w/4 and x2 < 3w/5)
/// 3. Identify right column boxes (x1 > 2w/5)
/// 4. Other boxes are considered full-width
/// 5. Within each column, sort by y-coordinate
///
/// This follows standard layout sorting implementation.
///
/// # Arguments
///
/// * `elements` - Slice of layout elements to sort
/// * `image_width` - Width of the image for column detection
///
/// # Returns
///
/// A vector of sorted layout elements
pub fn sort_layout_boxes(elements: &[LayoutElement], image_width: f32) -> Vec<LayoutElement> {
    let num_boxes = elements.len();

    if num_boxes <= 1 {
        return elements.to_vec();
    }

    // Sort by y-coordinate first, then x-coordinate
    let mut sorted: Vec<LayoutElement> = elements.to_vec();
    sorted.sort_by(|a, b| {
        let a_y = a.bbox.y_min();
        let a_x = a.bbox.x_min();
        let b_y = b.bbox.y_min();
        let b_x = b.bbox.x_min();

        match a_y.partial_cmp(&b_y) {
            Some(std::cmp::Ordering::Equal) => {
                a_x.partial_cmp(&b_x).unwrap_or(std::cmp::Ordering::Equal)
            }
            other => other.unwrap_or(std::cmp::Ordering::Equal),
        }
    });

    let mut result = Vec::new();
    let mut left_column = Vec::new();
    let mut right_column = Vec::new();

    let w = image_width;
    let mut i = 0;

    while i < num_boxes {
        let elem = &sorted[i];
        let x1 = elem.bbox.x_min();
        let x2 = elem.bbox.x_max();

        // Check if box is in left column
        if x1 < w / 4.0 && x2 < 3.0 * w / 5.0 {
            left_column.push(elem.clone());
        }
        // Check if box is in right column
        else if x1 > 2.0 * w / 5.0 {
            right_column.push(elem.clone());
        }
        // Full-width box - flush columns and add this box
        else {
            // Add accumulated column boxes
            result.append(&mut left_column);
            result.append(&mut right_column);
            result.push(elem.clone());
        }

        i += 1;
    }

    // Sort left and right columns by y-coordinate
    left_column.sort_by(|a, b| {
        a.bbox
            .y_min()
            .partial_cmp(&b.bbox.y_min())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    right_column.sort_by(|a, b| {
        a.bbox
            .y_min()
            .partial_cmp(&b.bbox.y_min())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Add remaining column boxes
    result.append(&mut left_column);
    result.append(&mut right_column);

    result
}

/// Reconciles structure recognition cells with detected cells.
///
/// This function aligns the number of output cells with the structure cells (N).
/// - If multiple detected cells map to one structure cell, they are merged (compressed).
/// - If a structure cell has no matching detected cell, the original structure box is kept (filled).
///
/// # Arguments
/// * `structure_cells` - Cells derived from structure recognition (provides logical N)
/// * `detected_cells` - Cells from detection model (provides precise geometry)
///
/// # Returns
/// * `Vec<BoundingBox>` - Reconciled bounding boxes of length N.
pub fn reconcile_table_cells(
    structure_cells: &[BoundingBox],
    detected_cells: &[BoundingBox],
) -> Vec<BoundingBox> {
    let n = structure_cells.len();
    if n == 0 {
        return Vec::new();
    }
    if detected_cells.is_empty() {
        return structure_cells.to_vec();
    }

    // If detection produces significantly more cells than the table structure,
    // reduce them using KMeans-style clustering on box centers.
    let mut det_boxes: Vec<BoundingBox> = detected_cells.to_vec();
    if det_boxes.len() > n {
        det_boxes = combine_rectangles_kmeans(&det_boxes, n);
    }

    // Assignments: structure_idx -> list of detected_indices
    let mut assignments: Vec<Vec<usize>> = vec![Vec::new(); n];

    // Assign each detected cell to the best matching structure cell
    for (det_idx, det_box) in det_boxes.iter().enumerate() {
        let mut best_iou = 0.001f32; // Minimal threshold
        let mut best_struct_idx: Option<usize> = None;

        for (struct_idx, struct_box) in structure_cells.iter().enumerate() {
            // Use IoU for assignment
            // Note: We could also use intersection over detection area to handle
            // cases where detection is much smaller than structure cell
            let iou = calculate_iou(det_box, struct_box);
            if iou > best_iou {
                best_iou = iou;
                best_struct_idx = Some(struct_idx);
            }
        }

        if let Some(idx) = best_struct_idx {
            assignments[idx].push(det_idx);
        }
    }

    // Build result
    let mut reconciled = Vec::with_capacity(n);
    for i in 0..n {
        let assigned = &assignments[i];
        if assigned.is_empty() {
            // Fill: No matching detection, keep original structure box
            reconciled.push(structure_cells[i].clone());
        } else if assigned.len() == 1 {
            // Exact match: Use detected box
            reconciled.push(det_boxes[assigned[0]].clone());
        } else {
            // Compress: Multiple detections map to one structure cell
            // Merge them by taking the bounding box of all detections
            let mut merged = det_boxes[assigned[0]].clone();
            for &idx in &assigned[1..] {
                merged = merged.union(&det_boxes[idx]);
            }
            reconciled.push(merged);
        }
    }

    reconciled
}

/// Reprocesses detected table cell boxes using OCR boxes to better match the
/// structure model's expected cell count.
///
/// This mirrors cell detection results reprocessing in
/// `table_recognition/pipeline_v2.py`:
/// - If detected cells > target_n, keep top-N by score.
/// - Find OCR boxes not sufficiently covered by any cell (IoA >= 0.6).
/// - If missing OCR boxes exist, supplement/merge boxes with KMeans-style clustering.
/// - If final count is too small, fall back to clustering OCR boxes.
///
/// All boxes must be in the same coordinate system (typically table-crop coords).
pub fn reprocess_table_cells_with_ocr(
    detected_cells: &[BoundingBox],
    detected_scores: &[f32],
    ocr_boxes: &[BoundingBox],
    target_n: usize,
) -> Vec<BoundingBox> {
    if target_n == 0 {
        return Vec::new();
    }

    // If no detected cells, fall back to OCR clustering.
    if detected_cells.is_empty() {
        return combine_rectangles_kmeans(ocr_boxes, target_n);
    }

    // Defensive: scores length mismatch -> assume uniform.
    let scores: Vec<f32> = if detected_scores.len() == detected_cells.len() {
        detected_scores.to_vec()
    } else {
        vec![1.0; detected_cells.len()]
    };

    let mut cells: Vec<BoundingBox> = detected_cells.to_vec();

    let mut more_cells_flag = false;
    if cells.len() == target_n {
        return cells;
    } else if cells.len() > target_n {
        more_cells_flag = true;
        // Keep top target_n by score (descending).
        let mut idxs: Vec<usize> = (0..cells.len()).collect();
        idxs.sort_by(|&a, &b| {
            scores[b]
                .partial_cmp(&scores[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        idxs.truncate(target_n);
        cells = idxs.iter().map(|&i| cells[i].clone()).collect();
    }

    // Compute IoA (intersection / ocr_area) between OCR and cell boxes.
    fn ioa_ocr_in_cell(ocr: &BoundingBox, cell: &BoundingBox) -> f32 {
        let inter = ocr.intersection_area(cell);
        if inter <= 0.0 {
            return 0.0;
        }
        let area = (ocr.x_max() - ocr.x_min()) * (ocr.y_max() - ocr.y_min());
        if area <= 0.0 { 0.0 } else { inter / area }
    }

    let iou_threshold = 0.6f32;
    let mut ocr_miss_boxes: Vec<BoundingBox> = Vec::new();

    for ocr_box in ocr_boxes {
        let mut has_large_ioa = false;
        let mut merge_ioa_sum = 0.0f32;
        for cell_box in &cells {
            let ioa = ioa_ocr_in_cell(ocr_box, cell_box);
            if ioa > 0.0 {
                merge_ioa_sum += ioa;
            }
            if ioa >= iou_threshold || merge_ioa_sum >= iou_threshold {
                has_large_ioa = true;
                break;
            }
        }
        if !has_large_ioa {
            ocr_miss_boxes.push(ocr_box.clone());
        }
    }

    let mut final_results: Vec<BoundingBox>;

    if ocr_miss_boxes.is_empty() {
        final_results = cells;
    } else if more_cells_flag {
        // More cells than expected: merge cells + missing OCR boxes to target_n.
        let mut merged = cells.clone();
        merged.extend(ocr_miss_boxes);
        final_results = combine_rectangles_kmeans(&merged, target_n);
    } else {
        // Fewer cells than expected: supplement with clustered missing OCR boxes.
        let need_n = target_n.saturating_sub(cells.len());
        let supp = combine_rectangles_kmeans(&ocr_miss_boxes, need_n);
        final_results = cells;
        final_results.extend(supp);
    }

    // If still too few, fall back to clustering OCR boxes.
    if final_results.len() as f32 <= 0.6 * target_n as f32 {
        final_results = combine_rectangles_kmeans(ocr_boxes, target_n);
    }

    final_results
}

/// Combines rectangles into at most `target_n` rectangles using KMeans-style clustering
/// on box centers.
pub fn combine_rectangles_kmeans(rectangles: &[BoundingBox], target_n: usize) -> Vec<BoundingBox> {
    let num_rects = rectangles.len();
    if num_rects == 0 || target_n == 0 {
        return Vec::new();
    }
    if target_n >= num_rects {
        return rectangles.to_vec();
    }

    // Represent each rectangle by its center point (x, y)
    let points: Vec<(f32, f32)> = rectangles
        .iter()
        .map(|r| {
            let cx = (r.x_min() + r.x_max()) * 0.5;
            let cy = (r.y_min() + r.y_max()) * 0.5;
            (cx, cy)
        })
        .collect();

    // Initialize cluster centers using the first target_n points
    let mut centers: Vec<(f32, f32)> = points.iter().take(target_n).cloned().collect();
    let mut labels: Vec<usize> = vec![0; num_rects];

    let max_iters = 10;
    for _ in 0..max_iters {
        let mut changed = false;

        // Assignment step: assign each point to nearest center
        for (i, &(px, py)) in points.iter().enumerate() {
            let mut best_idx = 0usize;
            let mut best_dist = f32::MAX;
            for (c_idx, &(cx, cy)) in centers.iter().enumerate() {
                let dx = px - cx;
                let dy = py - cy;
                let dist = dx * dx + dy * dy;
                if dist < best_dist {
                    best_dist = dist;
                    best_idx = c_idx;
                }
            }
            if labels[i] != best_idx {
                labels[i] = best_idx;
                changed = true;
            }
        }

        // Recompute centers
        let mut sums: Vec<(f32, f32, usize)> = vec![(0.0, 0.0, 0); target_n];
        for (i, &(px, py)) in points.iter().enumerate() {
            let l = labels[i];
            sums[l].0 += px;
            sums[l].1 += py;
            sums[l].2 += 1;
        }
        for (c_idx, center) in centers.iter_mut().enumerate() {
            let (sx, sy, count) = sums[c_idx];
            if count > 0 {
                center.0 = sx / count as f32;
                center.1 = sy / count as f32;
            }
        }

        if !changed {
            break;
        }
    }

    // Build combined rectangles per cluster
    let mut combined: Vec<BoundingBox> = Vec::new();
    for cluster_idx in 0..target_n {
        let mut first = true;
        let mut min_x = 0.0f32;
        let mut min_y = 0.0f32;
        let mut max_x = 0.0f32;
        let mut max_y = 0.0f32;

        for (i, rect) in rectangles.iter().enumerate() {
            if labels[i] == cluster_idx {
                if first {
                    min_x = rect.x_min();
                    min_y = rect.y_min();
                    max_x = rect.x_max();
                    max_y = rect.y_max();
                    first = false;
                } else {
                    min_x = min_x.min(rect.x_min());
                    min_y = min_y.min(rect.y_min());
                    max_x = max_x.max(rect.x_max());
                    max_y = max_y.max(rect.y_max());
                }
            }
        }

        if !first {
            combined.push(BoundingBox::from_coords(min_x, min_y, max_x, max_y));
        }
    }

    if combined.is_empty() {
        rectangles.to_vec()
    } else {
        combined
    }
}

/// Calculates Intersection over Union (IoU) between two bounding boxes.
fn calculate_iou(a: &BoundingBox, b: &BoundingBox) -> f32 {
    let inter_x1 = a.x_min().max(b.x_min());
    let inter_y1 = a.y_min().max(b.y_min());
    let inter_x2 = a.x_max().min(b.x_max());
    let inter_y2 = a.y_max().min(b.y_max());

    let inter_area = (inter_x2 - inter_x1).max(0.0) * (inter_y2 - inter_y1).max(0.0);

    let area_a = (a.x_max() - a.x_min()) * (a.y_max() - a.y_min());
    let area_b = (b.x_max() - b.x_min()) * (b.y_max() - b.y_min());

    let union_area = area_a + area_b - inter_area;

    if union_area <= 0.0 {
        0.0
    } else {
        inter_area / union_area
    }
}

/// Calculates Intersection over Area (IoA) - intersection / smaller box area.
fn calculate_ioa_smaller(a: &BoundingBox, b: &BoundingBox) -> f32 {
    let inter_x1 = a.x_min().max(b.x_min());
    let inter_y1 = a.y_min().max(b.y_min());
    let inter_x2 = a.x_max().min(b.x_max());
    let inter_y2 = a.y_max().min(b.y_max());

    let inter_area = (inter_x2 - inter_x1).max(0.0) * (inter_y2 - inter_y1).max(0.0);

    let area_a = (a.x_max() - a.x_min()) * (a.y_max() - a.y_min());
    let area_b = (b.x_max() - b.x_min()) * (b.y_max() - b.y_min());

    let smaller_area = area_a.min(area_b);

    if smaller_area <= 0.0 {
        0.0
    } else {
        inter_area / smaller_area
    }
}

/// Result of overlap removal.
#[derive(Debug, Clone)]
pub struct OverlapRemovalResult<T> {
    /// Elements that were kept after overlap removal
    pub kept: Vec<T>,
    /// Indices of elements that were removed
    pub removed_indices: Vec<usize>,
}

/// Removes overlapping layout blocks based on overlap ratio threshold.
///
/// This follows standard overlap removal implementation in
/// `layout_parsing/utils.py`. When two blocks overlap significantly:
/// - If one is an image and one is not, the image is removed (text takes priority)
/// - Otherwise, the smaller block is removed
///
/// # Arguments
///
/// * `elements` - Slice of layout elements to process
/// * `threshold` - Overlap ratio threshold (default: 0.65)
///   If intersection/smaller_area > threshold, blocks are considered overlapping
///
/// # Returns
///
/// `OverlapRemovalResult` containing kept elements and indices of removed elements
///
/// # Example
///
/// ```ignore
/// use oar_ocr::processors::layout_utils::{remove_overlap_blocks, LayoutElement};
/// use oar_ocr::processors::BoundingBox;
///
/// let elements = vec![
///     LayoutElement::new(BoundingBox::from_coords(0.0, 0.0, 100.0, 100.0), "text".to_string()),
///     LayoutElement::new(BoundingBox::from_coords(10.0, 10.0, 90.0, 90.0), "text".to_string()),
/// ];
///
/// let result = remove_overlap_blocks(&elements, 0.65);
/// assert_eq!(result.kept.len(), 1); // Smaller overlapping box was removed
/// ```
pub fn remove_overlap_blocks(
    elements: &[LayoutElement],
    threshold: f32,
) -> OverlapRemovalResult<LayoutElement> {
    let n = elements.len();
    if n <= 1 {
        return OverlapRemovalResult {
            kept: elements.to_vec(),
            removed_indices: Vec::new(),
        };
    }

    let mut dropped_indices: HashSet<usize> = HashSet::new();

    // Compare all pairs of elements
    for i in 0..n {
        if dropped_indices.contains(&i) {
            continue;
        }

        for j in (i + 1)..n {
            if dropped_indices.contains(&j) {
                continue;
            }

            let elem_i = &elements[i];
            let elem_j = &elements[j];

            // Calculate overlap ratio (intersection / smaller area)
            let overlap_ratio = calculate_ioa_smaller(&elem_i.bbox, &elem_j.bbox);

            if overlap_ratio > threshold {
                // Determine which element to remove
                let is_i_image = elem_i.label == "image";
                let is_j_image = elem_j.label == "image";

                let drop_index = if is_i_image != is_j_image {
                    // One is image, one is not: remove the image (text takes priority)
                    if is_i_image { i } else { j }
                } else {
                    // Same type: remove the smaller one
                    let area_i = (elem_i.bbox.x_max() - elem_i.bbox.x_min())
                        * (elem_i.bbox.y_max() - elem_i.bbox.y_min());
                    let area_j = (elem_j.bbox.x_max() - elem_j.bbox.x_min())
                        * (elem_j.bbox.y_max() - elem_j.bbox.y_min());

                    if area_i < area_j { i } else { j }
                };

                dropped_indices.insert(drop_index);
                tracing::debug!(
                    "Removing overlapping element {} (label={}, overlap={:.2})",
                    drop_index,
                    elements[drop_index].label,
                    overlap_ratio
                );
            }
        }
    }

    // Build result
    let mut kept = Vec::new();
    let mut removed_indices: Vec<usize> = dropped_indices.into_iter().collect();
    removed_indices.sort();

    for (idx, elem) in elements.iter().enumerate() {
        if !removed_indices.contains(&idx) {
            kept.push(elem.clone());
        }
    }

    tracing::info!(
        "Overlap removal: {} elements -> {} kept, {} removed",
        n,
        kept.len(),
        removed_indices.len()
    );

    OverlapRemovalResult {
        kept,
        removed_indices,
    }
}

/// Removes overlapping layout blocks, returning only indices to remove.
///
/// A lighter-weight version that works with any bbox type implementing the required traits.
/// This is useful when you want to apply overlap removal to `LayoutElement` from `domain::structure`.
///
/// # Arguments
///
/// * `bboxes` - Slice of bounding boxes
/// * `labels` - Slice of labels corresponding to each bbox
/// * `threshold` - Overlap ratio threshold
///
/// # Returns
///
/// Set of indices that should be removed
pub fn get_overlap_removal_indices(
    bboxes: &[BoundingBox],
    labels: &[&str],
    threshold: f32,
) -> HashSet<usize> {
    let n = bboxes.len();
    if n <= 1 || n != labels.len() {
        return HashSet::new();
    }

    let mut dropped_indices: HashSet<usize> = HashSet::new();

    for i in 0..n {
        if dropped_indices.contains(&i) {
            continue;
        }

        for j in (i + 1)..n {
            if dropped_indices.contains(&j) {
                continue;
            }

            let overlap_ratio = calculate_ioa_smaller(&bboxes[i], &bboxes[j]);

            if overlap_ratio > threshold {
                let is_i_image = labels[i] == "image";
                let is_j_image = labels[j] == "image";

                let drop_index = if is_i_image != is_j_image {
                    if is_i_image { i } else { j }
                } else {
                    let area_i = (bboxes[i].x_max() - bboxes[i].x_min())
                        * (bboxes[i].y_max() - bboxes[i].y_min());
                    let area_j = (bboxes[j].x_max() - bboxes[j].x_min())
                        * (bboxes[j].y_max() - bboxes[j].y_min());

                    if area_i < area_j { i } else { j }
                };

                dropped_indices.insert(drop_index);
            }
        }
    }

    dropped_indices
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_overlap_boxes_idx() {
        // Create OCR boxes
        let ocr_boxes = vec![
            BoundingBox::from_coords(10.0, 10.0, 50.0, 30.0), // inside region
            BoundingBox::from_coords(60.0, 60.0, 100.0, 80.0), // inside region
            BoundingBox::from_coords(200.0, 200.0, 250.0, 220.0), // outside region
        ];

        // Create layout region
        let layout_regions = vec![BoundingBox::from_coords(0.0, 0.0, 150.0, 150.0)];

        let matched = get_overlap_boxes_idx(&ocr_boxes, &layout_regions, 3.0);

        // First two boxes should match
        assert_eq!(matched.len(), 2);
        assert!(matched.contains(&0));
        assert!(matched.contains(&1));
        assert!(!matched.contains(&2));
    }

    #[test]
    fn test_associate_ocr_with_layout_within() {
        let ocr_boxes = vec![
            BoundingBox::from_coords(10.0, 10.0, 50.0, 30.0),
            BoundingBox::from_coords(200.0, 200.0, 250.0, 220.0),
        ];

        let layout_regions = vec![BoundingBox::from_coords(0.0, 0.0, 100.0, 100.0)];

        let association = associate_ocr_with_layout(&ocr_boxes, &layout_regions, true, 3.0);

        assert_eq!(association.matched_indices.len(), 1);
        assert_eq!(association.matched_indices[0], 0);
        assert_eq!(association.unmatched_indices.len(), 1);
        assert_eq!(association.unmatched_indices[0], 1);
    }

    #[test]
    fn test_associate_ocr_with_layout_outside() {
        let ocr_boxes = vec![
            BoundingBox::from_coords(10.0, 10.0, 50.0, 30.0),
            BoundingBox::from_coords(200.0, 200.0, 250.0, 220.0),
        ];

        let layout_regions = vec![BoundingBox::from_coords(0.0, 0.0, 100.0, 100.0)];

        let association = associate_ocr_with_layout(&ocr_boxes, &layout_regions, false, 3.0);

        // flag_within=false returns boxes outside regions
        assert_eq!(association.matched_indices.len(), 1);
        assert_eq!(association.matched_indices[0], 1);
    }

    #[test]
    fn test_sort_layout_boxes_single_column() {
        let elements = vec![
            LayoutElement::new(
                BoundingBox::from_coords(10.0, 50.0, 200.0, 70.0),
                "text".to_string(),
            ), // bottom
            LayoutElement::new(
                BoundingBox::from_coords(10.0, 10.0, 200.0, 30.0),
                "title".to_string(),
            ), // top
        ];

        let sorted = sort_layout_boxes(&elements, 300.0);

        assert_eq!(sorted[0].label, "title"); // top first
        assert_eq!(sorted[1].label, "text"); // bottom second
    }

    #[test]
    fn test_sort_layout_boxes_two_columns() {
        let image_width = 400.0;

        let elements = vec![
            // Left column boxes (x < w/4 and x2 < 3w/5)
            LayoutElement::new(
                BoundingBox::from_coords(10.0, 100.0, 90.0, 120.0),
                "left_bottom".to_string(),
            ),
            LayoutElement::new(
                BoundingBox::from_coords(10.0, 50.0, 90.0, 70.0),
                "left_top".to_string(),
            ),
            // Right column boxes (x > 2w/5)
            LayoutElement::new(
                BoundingBox::from_coords(250.0, 100.0, 390.0, 120.0),
                "right_bottom".to_string(),
            ),
            LayoutElement::new(
                BoundingBox::from_coords(250.0, 50.0, 390.0, 70.0),
                "right_top".to_string(),
            ),
            // Full-width box (neither left nor right)
            LayoutElement::new(
                BoundingBox::from_coords(10.0, 10.0, 390.0, 30.0),
                "title".to_string(),
            ),
        ];

        let sorted = sort_layout_boxes(&elements, image_width);

        // Expected order:
        // 1. title (full-width, top)
        // 2. left_top (left column, higher)
        // 3. right_top (right column, higher)
        // 4. left_bottom (left column, lower)
        // 5. right_bottom (right column, lower)

        assert_eq!(sorted[0].label, "title");
        // Left column should come before right column
        let left_top_idx = sorted.iter().position(|e| e.label == "left_top").unwrap();
        let left_bottom_idx = sorted
            .iter()
            .position(|e| e.label == "left_bottom")
            .unwrap();
        let right_top_idx = sorted.iter().position(|e| e.label == "right_top").unwrap();
        let right_bottom_idx = sorted
            .iter()
            .position(|e| e.label == "right_bottom")
            .unwrap();

        // Within left column, top should come before bottom
        assert!(left_top_idx < left_bottom_idx);
        // Within right column, top should come before bottom
        assert!(right_top_idx < right_bottom_idx);
    }

    #[test]
    fn test_sort_layout_boxes_empty() {
        let elements: Vec<LayoutElement> = Vec::new();
        let sorted = sort_layout_boxes(&elements, 300.0);
        assert!(sorted.is_empty());
    }

    #[test]
    fn test_sort_layout_boxes_single_element() {
        let elements = vec![LayoutElement::new(
            BoundingBox::from_coords(10.0, 10.0, 100.0, 30.0),
            "text".to_string(),
        )];

        let sorted = sort_layout_boxes(&elements, 300.0);
        assert_eq!(sorted.len(), 1);
        assert_eq!(sorted[0].label, "text");
    }
}
