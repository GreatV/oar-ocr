//! Text box sorting utilities for OCR.
//!
//! This module provides sorting algorithms for detected text boxes to establish
//! reading order, following the standard approach of sorting from top to bottom,
//! left to right.
//!
//! ## Algorithms
//!
//! - **Simple sorting**: Basic top-to-bottom, left-to-right sorting
//! - **XY-cut sorting**: Recursive projection-based sorting (PP-StructureV3 compatible)
//!
//! ## XY-cut Algorithm
//!
//! The XY-cut algorithm recursively divides the page by projecting bounding boxes
//! onto the X and Y axes, finding gaps in the projection, and recursively sorting
//! each partition. This produces a more accurate reading order for complex layouts
//! like multi-column documents.

use crate::processors::BoundingBox;

/// Sort quad boxes (4-point bounding boxes) in reading order.
///
/// The boxes are sorted from top to bottom, left to right. Boxes on the same
/// horizontal line (within a threshold of 10 pixels) are sorted by their x-coordinate.
///
/// This follows standard quad box sorting implementation.
///
/// # Arguments
///
/// * `boxes` - Slice of bounding boxes to sort
///
/// # Returns
///
/// A vector of sorted bounding boxes
pub fn sort_quad_boxes(boxes: &[BoundingBox]) -> Vec<BoundingBox> {
    if boxes.is_empty() {
        return Vec::new();
    }

    let mut sorted: Vec<BoundingBox> = boxes.to_vec();

    // Sort by y-coordinate first (top-left point), then by x-coordinate
    sorted.sort_by(|a, b| {
        let a_y = a.y_min();
        let a_x = a.x_min();
        let b_y = b.y_min();
        let b_x = b.x_min();

        // Primary sort: y-coordinate
        match a_y.partial_cmp(&b_y) {
            Some(std::cmp::Ordering::Equal) => {
                // Secondary sort: x-coordinate
                a_x.partial_cmp(&b_x).unwrap_or(std::cmp::Ordering::Equal)
            }
            other => other.unwrap_or(std::cmp::Ordering::Equal),
        }
    });

    // Bubble sort to handle boxes on the same line
    // If two adjacent boxes have y-coordinates within 10 pixels and the lower one
    // has a smaller x-coordinate, swap them
    let num_boxes = sorted.len();
    for i in 0..num_boxes.saturating_sub(1) {
        for j in (0..=i).rev() {
            if j + 1 >= sorted.len() {
                break;
            }

            let curr_y = sorted[j].y_min();
            let next_y = sorted[j + 1].y_min();
            let curr_x = sorted[j].x_min();
            let next_x = sorted[j + 1].x_min();

            // Check if boxes are on the same horizontal line (within 10 pixels)
            if (next_y - curr_y).abs() < 10.0 && next_x < curr_x {
                sorted.swap(j, j + 1);
            } else {
                break;
            }
        }
    }

    sorted
}

/// Sort polygon boxes (N-point bounding boxes) in reading order.
///
/// The boxes are sorted by their minimum y-coordinate (top edge), from top to bottom.
/// This is a simpler sorting strategy suitable for arbitrary polygon shapes.
///
/// This follows standard poly box sorting implementation.
///
/// # Arguments
///
/// * `boxes` - Slice of bounding boxes to sort
///
/// # Returns
///
/// A vector of sorted bounding boxes
pub fn sort_poly_boxes(boxes: &[BoundingBox]) -> Vec<BoundingBox> {
    if boxes.is_empty() {
        return Vec::new();
    }

    let mut indexed_boxes: Vec<(usize, f32, &BoundingBox)> = boxes
        .iter()
        .enumerate()
        .map(|(idx, bbox)| (idx, bbox.y_min(), bbox))
        .collect();

    // Sort by minimum y-coordinate
    indexed_boxes.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    indexed_boxes
        .into_iter()
        .map(|(_, _, bbox)| bbox.clone())
        .collect()
}

// XY-cut Algorithm Implementation (PP-StructureV3 compatible)

/// Direction for XY-cut sorting.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortDirection {
    /// Horizontal direction (left-to-right reading order)
    Horizontal,
    /// Vertical direction (top-to-bottom reading order)
    Vertical,
}

/// Sort bounding boxes using the XY-cut algorithm.
///
/// The XY-cut algorithm recursively divides the page by projecting bounding boxes
/// onto axes, finding gaps in the projection, and sorting each partition.
/// This produces accurate reading order for complex multi-column layouts.
///
/// Follows the PP-StructureV3 `sort_by_xycut` implementation.
///
/// # Arguments
///
/// * `boxes` - Slice of bounding boxes to sort
/// * `direction` - Initial cut direction (Vertical for Y-first, Horizontal for X-first)
/// * `min_gap` - Minimum gap width to consider a separation (default: 1)
///
/// # Returns
///
/// Indices representing the sorted order of input boxes
///
/// # Example
///
/// ```
/// use oar_ocr_core::processors::{BoundingBox, SortDirection, sort_by_xycut};
///
/// let boxes = vec![
///     BoundingBox::from_coords(10.0, 10.0, 50.0, 30.0),
///     BoundingBox::from_coords(60.0, 10.0, 100.0, 30.0),
///     BoundingBox::from_coords(10.0, 40.0, 100.0, 60.0),
/// ];
/// let sorted_indices = sort_by_xycut(&boxes, SortDirection::Vertical, 1);
/// ```
pub fn sort_by_xycut(boxes: &[BoundingBox], direction: SortDirection, min_gap: i32) -> Vec<usize> {
    if boxes.is_empty() {
        return Vec::new();
    }

    // Convert boxes to integer coordinates for projection
    let int_boxes: Vec<[i32; 4]> = boxes
        .iter()
        .map(|b| {
            [
                b.x_min() as i32,
                b.y_min() as i32,
                b.x_max() as i32,
                b.y_max() as i32,
            ]
        })
        .collect();

    let indices: Vec<usize> = (0..boxes.len()).collect();
    let mut result = Vec::new();

    match direction {
        SortDirection::Vertical => {
            recursive_yx_cut(&int_boxes, &indices, &mut result, min_gap);
        }
        SortDirection::Horizontal => {
            recursive_xy_cut(&int_boxes, &indices, &mut result, min_gap);
        }
    }

    result
}

/// Sort bounding boxes using XY-cut and return sorted boxes (convenience wrapper).
///
/// # Arguments
///
/// * `boxes` - Slice of bounding boxes to sort
/// * `direction` - Initial cut direction
///
/// # Returns
///
/// A vector of sorted bounding boxes
pub fn sort_boxes_xycut(boxes: &[BoundingBox], direction: SortDirection) -> Vec<BoundingBox> {
    let indices = sort_by_xycut(boxes, direction, 1);
    indices.into_iter().map(|i| boxes[i].clone()).collect()
}

/// Generate a 1D projection histogram from bounding boxes along a specified axis.
///
/// # Arguments
///
/// * `boxes` - Array of bounding boxes as `[x_min, y_min, x_max, y_max]`
/// * `axis` - 0 for X-axis projection, 1 for Y-axis projection
///
/// # Returns
///
/// A 1D vector representing the projection histogram
fn projection_by_bboxes(boxes: &[[i32; 4]], axis: usize) -> Vec<i32> {
    assert!(axis <= 1, "axis must be 0 or 1");

    if boxes.is_empty() {
        return Vec::new();
    }

    // Find the maximum coordinate
    let max_length = boxes
        .iter()
        .map(|b| b[axis + 2].unsigned_abs() as usize)
        .max()
        .unwrap_or(0);

    if max_length == 0 {
        return Vec::new();
    }

    let mut projection = vec![0i32; max_length + 1];

    // Increment projection histogram over the interval defined by each bounding box
    for b in boxes {
        let start = b[axis].unsigned_abs() as usize;
        let end = b[axis + 2].unsigned_abs() as usize;
        let (start, end) = if start <= end {
            (start, end)
        } else {
            (end, start)
        };
        for i in start..end.min(projection.len()) {
            projection[i] += 1;
        }
    }

    projection
}

/// Split the projection profile into segments based on gaps.
///
/// # Arguments
///
/// * `arr_values` - 1D array representing the projection profile
/// * `min_value` - Minimum value threshold to consider significant
/// * `min_gap` - Minimum gap width to consider a separation
///
/// # Returns
///
/// Optional tuple of (segment_starts, segment_ends)
fn split_projection_profile(
    arr_values: &[i32],
    min_value: i32,
    min_gap: i32,
) -> Option<(Vec<usize>, Vec<usize>)> {
    // Find indices where the projection exceeds the minimum value
    let significant_indices: Vec<usize> = arr_values
        .iter()
        .enumerate()
        .filter(|(_, val)| **val > min_value)
        .map(|(idx, _)| idx)
        .collect();

    if significant_indices.is_empty() {
        return None;
    }

    // Calculate gaps between significant indices
    let mut segment_starts = Vec::new();
    let mut segment_ends = Vec::new();

    segment_starts.push(significant_indices[0]);

    for i in 1..significant_indices.len() {
        let gap = (significant_indices[i] - significant_indices[i - 1]) as i32;
        if gap > min_gap {
            segment_ends.push(significant_indices[i - 1] + 1);
            segment_starts.push(significant_indices[i]);
        }
    }

    segment_ends.push(significant_indices[significant_indices.len() - 1] + 1);

    Some((segment_starts, segment_ends))
}

/// Recursively sort boxes using Y-axis first, then X-axis (YX-cut).
///
/// This is the preferred order for vertical reading direction (top-to-bottom).
fn recursive_yx_cut(boxes: &[[i32; 4]], indices: &[usize], result: &mut Vec<usize>, min_gap: i32) {
    if boxes.is_empty() || indices.is_empty() {
        return;
    }

    // Sort by y_min for Y-axis projection
    let mut y_sorted: Vec<(usize, &[i32; 4], usize)> = boxes
        .iter()
        .enumerate()
        .map(|(i, b)| (i, b, indices[i]))
        .collect();
    y_sorted.sort_by_key(|(_, b, _)| b[1]);

    let y_sorted_boxes: Vec<[i32; 4]> = y_sorted.iter().map(|(_, b, _)| **b).collect();
    let y_sorted_indices: Vec<usize> = y_sorted.iter().map(|(_, _, idx)| *idx).collect();

    // Perform Y-axis projection
    let y_projection = projection_by_bboxes(&y_sorted_boxes, 1);
    let y_intervals = split_projection_profile(&y_projection, 0, 1);

    let Some((y_starts, y_ends)) = y_intervals else {
        return;
    };

    // Process each segment defined by Y-axis projection
    for (y_start, y_end) in y_starts.iter().zip(y_ends.iter()) {
        // Select boxes within the current y interval
        let y_chunk: Vec<(usize, [i32; 4])> = y_sorted_boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| {
                let y_min = b[1] as usize;
                y_min >= *y_start && y_min < *y_end
            })
            .map(|(i, b)| (y_sorted_indices[i], *b))
            .collect();

        if y_chunk.is_empty() {
            continue;
        }

        // Sort by x_min for X-axis projection
        let mut x_sorted = y_chunk.clone();
        x_sorted.sort_by_key(|(_, b)| b[0]);

        let x_sorted_boxes: Vec<[i32; 4]> = x_sorted.iter().map(|(_, b)| *b).collect();
        let x_sorted_indices: Vec<usize> = x_sorted.iter().map(|(idx, _)| *idx).collect();

        // Perform X-axis projection
        let x_projection = projection_by_bboxes(&x_sorted_boxes, 0);
        let x_intervals = split_projection_profile(&x_projection, 0, min_gap);

        let Some((x_starts, x_ends)) = x_intervals else {
            continue;
        };

        // If X-axis cannot be further segmented, add current indices to results
        if x_starts.len() == 1 {
            result.extend(x_sorted_indices);
            continue;
        }

        // Recursively process each segment defined by X-axis projection
        for (x_start, x_end) in x_starts.iter().zip(x_ends.iter()) {
            let x_chunk_boxes: Vec<[i32; 4]> = x_sorted_boxes
                .iter()
                .enumerate()
                .filter(|(_, b)| {
                    let x_min = b[0].unsigned_abs() as usize;
                    x_min >= *x_start && x_min < *x_end
                })
                .map(|(_, b)| *b)
                .collect();

            let x_chunk_indices: Vec<usize> = x_sorted_boxes
                .iter()
                .enumerate()
                .filter(|(_, b)| {
                    let x_min = b[0].unsigned_abs() as usize;
                    x_min >= *x_start && x_min < *x_end
                })
                .map(|(i, _)| x_sorted_indices[i])
                .collect();

            recursive_yx_cut(&x_chunk_boxes, &x_chunk_indices, result, min_gap);
        }
    }
}

/// Recursively sort boxes using X-axis first, then Y-axis (XY-cut).
///
/// This is the preferred order for horizontal reading direction (left-to-right).
fn recursive_xy_cut(boxes: &[[i32; 4]], indices: &[usize], result: &mut Vec<usize>, min_gap: i32) {
    if boxes.is_empty() || indices.is_empty() {
        return;
    }

    // Sort by x_min for X-axis projection
    let mut x_sorted: Vec<(usize, &[i32; 4], usize)> = boxes
        .iter()
        .enumerate()
        .map(|(i, b)| (i, b, indices[i]))
        .collect();
    x_sorted.sort_by_key(|(_, b, _)| b[0]);

    let x_sorted_boxes: Vec<[i32; 4]> = x_sorted.iter().map(|(_, b, _)| **b).collect();
    let x_sorted_indices: Vec<usize> = x_sorted.iter().map(|(_, _, idx)| *idx).collect();

    // Perform X-axis projection
    let x_projection = projection_by_bboxes(&x_sorted_boxes, 0);
    let x_intervals = split_projection_profile(&x_projection, 0, 1);

    let Some((x_starts, x_ends)) = x_intervals else {
        return;
    };

    // Process each segment defined by X-axis projection
    for (x_start, x_end) in x_starts.iter().zip(x_ends.iter()) {
        // Select boxes within the current x interval
        let x_chunk: Vec<(usize, [i32; 4])> = x_sorted_boxes
            .iter()
            .enumerate()
            .filter(|(_, b)| {
                let x_min = b[0].unsigned_abs() as usize;
                x_min >= *x_start && x_min < *x_end
            })
            .map(|(i, b)| (x_sorted_indices[i], *b))
            .collect();

        if x_chunk.is_empty() {
            continue;
        }

        // Sort by y_min for Y-axis projection
        let mut y_sorted = x_chunk.clone();
        y_sorted.sort_by_key(|(_, b)| b[1]);

        let y_sorted_boxes: Vec<[i32; 4]> = y_sorted.iter().map(|(_, b)| *b).collect();
        let y_sorted_indices: Vec<usize> = y_sorted.iter().map(|(idx, _)| *idx).collect();

        // Perform Y-axis projection
        let y_projection = projection_by_bboxes(&y_sorted_boxes, 1);
        let y_intervals = split_projection_profile(&y_projection, 0, min_gap);

        let Some((y_starts, y_ends)) = y_intervals else {
            continue;
        };

        // If Y-axis cannot be further segmented, add current indices to results
        if y_starts.len() == 1 {
            result.extend(y_sorted_indices);
            continue;
        }

        // Recursively process each segment defined by Y-axis projection
        for (y_start, y_end) in y_starts.iter().zip(y_ends.iter()) {
            let y_chunk_boxes: Vec<[i32; 4]> = y_sorted_boxes
                .iter()
                .enumerate()
                .filter(|(_, b)| {
                    let y_min = b[1] as usize;
                    y_min >= *y_start && y_min < *y_end
                })
                .map(|(_, b)| *b)
                .collect();

            let y_chunk_indices: Vec<usize> = y_sorted_boxes
                .iter()
                .enumerate()
                .filter(|(_, b)| {
                    let y_min = b[1] as usize;
                    y_min >= *y_start && y_min < *y_end
                })
                .map(|(i, _)| y_sorted_indices[i])
                .collect();

            recursive_xy_cut(&y_chunk_boxes, &y_chunk_indices, result, min_gap);
        }
    }
}

/// Represents a sortable region for hierarchical sorting.
///
/// This is a lightweight internal structure used for sorting regions.
/// For document structure regions, use `RegionBlock` from `domain::structure`.
#[derive(Debug, Clone)]
pub struct SortableRegion {
    /// Region bounding box
    pub bbox: BoundingBox,
    /// Original index of this region in the input
    pub original_index: usize,
}

impl SortableRegion {
    /// Creates a new sortable region.
    pub fn new(bbox: BoundingBox, original_index: usize) -> Self {
        Self {
            bbox,
            original_index,
        }
    }

    /// Calculates the area of this region.
    pub fn area(&self) -> f32 {
        let width = self.bbox.x_max() - self.bbox.x_min();
        let height = self.bbox.y_max() - self.bbox.y_min();
        width * height
    }

    /// Gets the center point of this region.
    pub fn center(&self) -> (f32, f32) {
        (
            (self.bbox.x_min() + self.bbox.x_max()) / 2.0,
            (self.bbox.y_min() + self.bbox.y_max()) / 2.0,
        )
    }
}

/// Calculates the IoU (Intersection over Union) between two bounding boxes.
pub fn calculate_iou(a: &BoundingBox, b: &BoundingBox) -> f32 {
    let x1 = a.x_min().max(b.x_min());
    let y1 = a.y_min().max(b.y_min());
    let x2 = a.x_max().min(b.x_max());
    let y2 = a.y_max().min(b.y_max());

    let intersection_width = (x2 - x1).max(0.0);
    let intersection_height = (y2 - y1).max(0.0);
    let intersection_area = intersection_width * intersection_height;

    let area_a = (a.x_max() - a.x_min()) * (a.y_max() - a.y_min());
    let area_b = (b.x_max() - b.x_min()) * (b.y_max() - b.y_min());
    let union_area = area_a + area_b - intersection_area;

    if union_area > 0.0 {
        intersection_area / union_area
    } else {
        0.0
    }
}

/// Calculates the overlap ratio of box a covered by box b.
/// Returns intersection_area / area_a
pub fn calculate_overlap_ratio(a: &BoundingBox, b: &BoundingBox) -> f32 {
    let x1 = a.x_min().max(b.x_min());
    let y1 = a.y_min().max(b.y_min());
    let x2 = a.x_max().min(b.x_max());
    let y2 = a.y_max().min(b.y_max());

    let intersection_width = (x2 - x1).max(0.0);
    let intersection_height = (y2 - y1).max(0.0);
    let intersection_area = intersection_width * intersection_height;

    let area_a = (a.x_max() - a.x_min()) * (a.y_max() - a.y_min());

    if area_a > 0.0 {
        intersection_area / area_a
    } else {
        0.0
    }
}

/// Assigns elements to regions based on overlap threshold.
///
/// # Arguments
///
/// * `elements` - Bounding boxes of layout elements
/// * `regions` - Sortable regions for hierarchical ordering
/// * `threshold` - Minimum overlap ratio to assign element to region (0.0-1.0)
///
/// # Returns
///
/// A vector where index i contains the region index that element i belongs to,
/// or None if the element doesn't belong to any region.
pub fn assign_elements_to_regions(
    elements: &[BoundingBox],
    regions: &[SortableRegion],
    threshold: f32,
) -> Vec<Option<usize>> {
    elements
        .iter()
        .map(|element| {
            // Find the region with the highest overlap ratio
            let mut best_region: Option<usize> = None;
            let mut best_overlap = threshold;

            for (region_idx, region) in regions.iter().enumerate() {
                let overlap = calculate_overlap_ratio(element, &region.bbox);
                if overlap > best_overlap {
                    best_overlap = overlap;
                    best_region = Some(region_idx);
                }
            }

            best_region
        })
        .collect()
}

/// Sorts regions in reading order (top-to-bottom, left-to-right).
///
/// Uses XY-cut algorithm for consistent ordering.
pub fn sort_regions(regions: &[SortableRegion]) -> Vec<usize> {
    if regions.is_empty() {
        return Vec::new();
    }

    let bboxes: Vec<BoundingBox> = regions.iter().map(|r| r.bbox.clone()).collect();
    sort_by_xycut(&bboxes, SortDirection::Vertical, 1)
}

/// Sorts elements within each region using XY-cut, then orders regions.
///
/// This implements the PP-StructureV3 hierarchical reading order:
/// 1. Group elements by their assigned regions
/// 2. Apply XY-cut sorting within each region
/// 3. Sort regions themselves
/// 4. Concatenate results following region order
///
/// # Arguments
///
/// * `elements` - Bounding boxes of layout elements
/// * `regions` - Sortable regions for hierarchical grouping
/// * `assignments` - Region assignment for each element (from `assign_elements_to_regions`)
///
/// # Returns
///
/// Indices representing the sorted order of input elements
pub fn sort_elements_with_regions(
    elements: &[BoundingBox],
    regions: &[SortableRegion],
    assignments: &[Option<usize>],
) -> Vec<usize> {
    if elements.is_empty() {
        return Vec::new();
    }

    // If no regions, fall back to simple XY-cut
    if regions.is_empty() {
        return sort_by_xycut(elements, SortDirection::Vertical, 1);
    }

    // Sort regions first
    let sorted_region_indices = sort_regions(regions);

    // Group elements by region
    let mut region_elements: Vec<Vec<usize>> = vec![Vec::new(); regions.len()];
    let mut unassigned_elements: Vec<usize> = Vec::new();

    for (elem_idx, assignment) in assignments.iter().enumerate() {
        match assignment {
            Some(region_idx) => region_elements[*region_idx].push(elem_idx),
            None => unassigned_elements.push(elem_idx),
        }
    }

    // Sort elements within each region
    let mut result: Vec<usize> = Vec::new();

    for region_idx in sorted_region_indices {
        let region_elem_indices = &region_elements[region_idx];
        if region_elem_indices.is_empty() {
            continue;
        }

        // Extract boxes for this region
        let region_boxes: Vec<BoundingBox> = region_elem_indices
            .iter()
            .map(|&idx| elements[idx].clone())
            .collect();

        // Sort within region
        let sorted_within = sort_by_xycut(&region_boxes, SortDirection::Vertical, 1);

        // Map back to original indices
        for sorted_idx in sorted_within {
            result.push(region_elem_indices[sorted_idx]);
        }
    }

    // Handle unassigned elements - sort and append
    if !unassigned_elements.is_empty() {
        let unassigned_boxes: Vec<BoundingBox> = unassigned_elements
            .iter()
            .map(|&idx| elements[idx].clone())
            .collect();
        let sorted_unassigned = sort_by_xycut(&unassigned_boxes, SortDirection::Vertical, 1);

        for sorted_idx in sorted_unassigned {
            result.push(unassigned_elements[sorted_idx]);
        }
    }

    result
}

/// Convenience function that combines region detection output with layout elements
/// to produce a hierarchically sorted reading order.
///
/// # Arguments
///
/// * `elements` - Bounding boxes of layout elements (from layout detection)
/// * `region_bboxes` - Bounding boxes of regions (from PP-DocBlockLayout)
/// * `overlap_threshold` - Minimum overlap to assign element to region
///
/// # Returns
///
/// Indices representing the sorted order of input elements
pub fn sort_with_region_hierarchy(
    elements: &[BoundingBox],
    region_bboxes: &[BoundingBox],
    overlap_threshold: f32,
) -> Vec<usize> {
    if elements.is_empty() {
        return Vec::new();
    }

    // Convert to SortableRegions
    let regions: Vec<SortableRegion> = region_bboxes
        .iter()
        .enumerate()
        .map(|(idx, bbox)| SortableRegion::new(bbox.clone(), idx))
        .collect();

    // Assign elements to regions
    let assignments = assign_elements_to_regions(elements, &regions, overlap_threshold);

    // Sort with hierarchy
    sort_elements_with_regions(elements, &regions, &assignments)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sort_quad_boxes_vertical() {
        // Create boxes at different vertical positions
        let box1 = BoundingBox::from_coords(10.0, 50.0, 50.0, 70.0); // bottom
        let box2 = BoundingBox::from_coords(10.0, 10.0, 50.0, 30.0); // top
        let box3 = BoundingBox::from_coords(10.0, 30.0, 50.0, 50.0); // middle

        let boxes = vec![box1, box2, box3];
        let sorted = sort_quad_boxes(&boxes);

        assert_eq!(sorted[0].y_min(), 10.0); // top box first
        assert_eq!(sorted[1].y_min(), 30.0); // middle box second
        assert_eq!(sorted[2].y_min(), 50.0); // bottom box third
    }

    #[test]
    fn test_sort_quad_boxes_same_line() {
        // Create boxes on the same horizontal line
        let box1 = BoundingBox::from_coords(60.0, 10.0, 100.0, 30.0); // right
        let box2 = BoundingBox::from_coords(10.0, 12.0, 50.0, 32.0); // left (y within 10px)

        let boxes = vec![box1, box2];
        let sorted = sort_quad_boxes(&boxes);

        // Left box should come first
        assert!(sorted[0].x_min() < sorted[1].x_min());
    }

    #[test]
    fn test_sort_quad_boxes_mixed() {
        // Create boxes with mixed positions
        let box1 = BoundingBox::from_coords(60.0, 10.0, 100.0, 30.0); // top-right
        let box2 = BoundingBox::from_coords(10.0, 11.0, 50.0, 31.0); // top-left (same line)
        let box3 = BoundingBox::from_coords(10.0, 50.0, 50.0, 70.0); // bottom-left
        let box4 = BoundingBox::from_coords(60.0, 52.0, 100.0, 72.0); // bottom-right

        let boxes = vec![box1.clone(), box2.clone(), box3.clone(), box4.clone()];
        let sorted = sort_quad_boxes(&boxes);

        // Expected order: box2 (top-left), box1 (top-right), box3 (bottom-left), box4 (bottom-right)
        assert!(sorted[0].x_min() < sorted[1].x_min()); // top line: left before right
        assert!(sorted[0].y_min() < sorted[2].y_min()); // top before bottom
        assert!(sorted[2].x_min() < sorted[3].x_min()); // bottom line: left before right
    }

    #[test]
    fn test_sort_poly_boxes() {
        // Create boxes at different vertical positions
        let box1 = BoundingBox::from_coords(10.0, 50.0, 50.0, 70.0); // bottom
        let box2 = BoundingBox::from_coords(10.0, 10.0, 50.0, 30.0); // top
        let box3 = BoundingBox::from_coords(10.0, 30.0, 50.0, 50.0); // middle

        let boxes = vec![box1, box2, box3];
        let sorted = sort_poly_boxes(&boxes);

        assert_eq!(sorted[0].y_min(), 10.0); // top box first
        assert_eq!(sorted[1].y_min(), 30.0); // middle box second
        assert_eq!(sorted[2].y_min(), 50.0); // bottom box third
    }

    #[test]
    fn test_sort_empty_boxes() {
        let boxes: Vec<BoundingBox> = Vec::new();
        let sorted = sort_quad_boxes(&boxes);
        assert!(sorted.is_empty());

        let sorted = sort_poly_boxes(&boxes);
        assert!(sorted.is_empty());
    }

    // XY-cut algorithm tests

    #[test]
    fn test_xycut_single_column() {
        // Single column layout - boxes stacked vertically
        let boxes = vec![
            BoundingBox::from_coords(10.0, 10.0, 100.0, 30.0), // top
            BoundingBox::from_coords(10.0, 40.0, 100.0, 60.0), // middle
            BoundingBox::from_coords(10.0, 70.0, 100.0, 90.0), // bottom
        ];

        let sorted_indices = sort_by_xycut(&boxes, SortDirection::Vertical, 1);

        // Should maintain top-to-bottom order
        assert_eq!(sorted_indices.len(), 3);
        // First box should be the top one (index 0)
        assert_eq!(sorted_indices[0], 0);
        // Second box should be middle one (index 1)
        assert_eq!(sorted_indices[1], 1);
        // Third box should be bottom one (index 2)
        assert_eq!(sorted_indices[2], 2);
    }

    #[test]
    fn test_xycut_two_columns() {
        // Two-column layout with clear separation
        let boxes = vec![
            BoundingBox::from_coords(10.0, 10.0, 45.0, 30.0), // left-col, row 1
            BoundingBox::from_coords(55.0, 10.0, 90.0, 30.0), // right-col, row 1
            BoundingBox::from_coords(10.0, 40.0, 45.0, 60.0), // left-col, row 2
            BoundingBox::from_coords(55.0, 40.0, 90.0, 60.0), // right-col, row 2
        ];

        let sorted_indices = sort_by_xycut(&boxes, SortDirection::Vertical, 1);

        // With YX-cut (Vertical direction), should read left column first, then right
        // Expected order: left-col-row1, left-col-row2, right-col-row1, right-col-row2
        assert_eq!(sorted_indices.len(), 4);
    }

    #[test]
    fn test_xycut_empty_boxes() {
        let boxes: Vec<BoundingBox> = Vec::new();
        let sorted_indices = sort_by_xycut(&boxes, SortDirection::Vertical, 1);
        assert!(sorted_indices.is_empty());
    }

    #[test]
    fn test_sort_boxes_xycut_wrapper() {
        let boxes = vec![
            BoundingBox::from_coords(10.0, 50.0, 50.0, 70.0), // bottom
            BoundingBox::from_coords(10.0, 10.0, 50.0, 30.0), // top
        ];

        let sorted = sort_boxes_xycut(&boxes, SortDirection::Vertical);

        // Top box should come first
        assert_eq!(sorted.len(), 2);
        assert!(sorted[0].y_min() < sorted[1].y_min());
    }

    #[test]
    fn test_projection_by_bboxes() {
        let boxes = vec![[10i32, 0, 20, 10], [15, 0, 25, 10]];

        // X-axis projection
        let x_proj = projection_by_bboxes(&boxes, 0);
        assert!(!x_proj.is_empty());
        // Should have overlap in 15-20 range (count = 2)
        assert_eq!(x_proj[15], 2);
        assert_eq!(x_proj[10], 1);
    }

    #[test]
    fn test_split_projection_profile() {
        // Profile with a gap
        let profile = vec![1, 1, 0, 0, 0, 1, 1];
        let result = split_projection_profile(&profile, 0, 1);

        assert!(result.is_some());
        let (starts, ends) = result.unwrap();
        assert_eq!(starts.len(), 2);
        assert_eq!(ends.len(), 2);
    }
}
