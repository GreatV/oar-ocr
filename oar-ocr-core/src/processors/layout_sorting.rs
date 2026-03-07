//! Enhanced layout sorting logic — `xycut_enhanced` algorithm.
//!
//! Faithful port of PaddleX's `xycut_enhanced` strategy:
//! 1. Header/Footer separation
//! 2. Cross-layout detection (blocks spanning multiple columns)
//! 3. Direction-aware XY-cut sorting
//! 4. Overlapping box shrinking before projection
//! 5. Weighted distance insertion for special blocks
//! 6. Child block association (vision titles → vision parents)

use crate::domain::structure::LayoutElementType;
use crate::processors::sorting::calculate_overlap_ratio;
use crate::processors::{BoundingBox, SortDirection, sort_by_xycut};

/// XYCUT_SETTINGS constants (matching PaddleX setting.py)
const EDGE_DISTANCE_COMPARE_TOLERANCE_LEN: f32 = 2.0;
const EDGE_WEIGHT: f32 = 10000.0; // 10^4
const UP_EDGE_WEIGHT: f32 = 1.0;
const LEFT_EDGE_WEIGHT: f32 = 2.0;
const CROSS_LAYOUT_REF_TEXT_BLOCK_WORDS_NUM_THRESHOLD: f32 = 10.0;

/// Label used for sorting logic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OrderLabel {
    Header,
    Footer,
    DocTitle,
    ParagraphTitle,
    Vision,
    VisionTitle,
    Unordered,
    NormalText,
    CrossLayout,
    CrossReference,
    Reference,
}

impl OrderLabel {
    pub fn from_element_type(et: LayoutElementType) -> Self {
        match et {
            LayoutElementType::Header | LayoutElementType::HeaderImage => OrderLabel::Header,

            LayoutElementType::Footer
            | LayoutElementType::FooterImage
            | LayoutElementType::Footnote => OrderLabel::Footer,

            LayoutElementType::DocTitle => OrderLabel::DocTitle,

            LayoutElementType::ParagraphTitle | LayoutElementType::Content => {
                OrderLabel::ParagraphTitle
            }

            LayoutElementType::Reference => OrderLabel::Reference,

            LayoutElementType::Image
            | LayoutElementType::Table
            | LayoutElementType::Chart
            | LayoutElementType::Algorithm => OrderLabel::Vision,

            LayoutElementType::FigureTitle
            | LayoutElementType::TableTitle
            | LayoutElementType::ChartTitle
            | LayoutElementType::FigureTableChartTitle => OrderLabel::VisionTitle,

            LayoutElementType::AsideText
            | LayoutElementType::Seal
            | LayoutElementType::Number
            | LayoutElementType::FormulaNumber => OrderLabel::Unordered,

            LayoutElementType::Text
            | LayoutElementType::List
            | LayoutElementType::Abstract
            | LayoutElementType::ReferenceContent
            | LayoutElementType::Formula => OrderLabel::NormalText,

            _ => OrderLabel::NormalText,
        }
    }
}

/// A wrapper around layout elements with properties needed for sorting.
#[derive(Debug, Clone)]
pub struct SortableBlock {
    pub bbox: BoundingBox,
    pub original_index: usize,
    pub order_label: OrderLabel,
    pub element_type: LayoutElementType,
    pub direction: SortDirection,
    pub num_lines: u32,
    pub text_line_height: f32,
}

impl SortableBlock {
    pub fn new(
        bbox: BoundingBox,
        original_index: usize,
        element_type: LayoutElementType,
        num_lines: Option<u32>,
    ) -> Self {
        let order_label = OrderLabel::from_element_type(element_type);
        let width = bbox.x_max() - bbox.x_min();
        let height = bbox.y_max() - bbox.y_min();
        let direction = if width >= height {
            SortDirection::Horizontal
        } else {
            SortDirection::Vertical
        };
        let num_lines = num_lines.unwrap_or(1).max(1);
        let text_line_height = if num_lines > 0 {
            height / num_lines as f32
        } else {
            height
        };

        Self {
            bbox,
            original_index,
            order_label,
            element_type,
            direction,
            num_lines,
            text_line_height,
        }
    }

    pub fn width(&self) -> f32 {
        self.bbox.x_max() - self.bbox.x_min()
    }

    pub fn height(&self) -> f32 {
        self.bbox.y_max() - self.bbox.y_min()
    }

    pub fn area(&self) -> f32 {
        self.width() * self.height()
    }

    pub fn center(&self) -> (f32, f32) {
        (
            (self.bbox.x_min() + self.bbox.x_max()) / 2.0,
            (self.bbox.y_min() + self.bbox.y_max()) / 2.0,
        )
    }

    pub fn long_side_length(&self) -> f32 {
        self.width().max(self.height())
    }
}

/// Input element for enhanced sorting.
pub struct SortableElement {
    pub bbox: BoundingBox,
    pub element_type: LayoutElementType,
    pub num_lines: Option<u32>,
}

/// Main entry point for enhanced sorting.
///
/// Returns a list of original indices in the correct reading order.
pub fn sort_layout_enhanced(
    elements: &[SortableElement],
    page_width: f32,
    _page_height: f32,
) -> Vec<usize> {
    if elements.is_empty() {
        return Vec::new();
    }

    let blocks: Vec<SortableBlock> = elements
        .iter()
        .enumerate()
        .map(|(i, e)| SortableBlock::new(e.bbox.clone(), i, e.element_type, e.num_lines))
        .collect();

    // Separate headers/footers
    let mut header_blocks = Vec::new();
    let mut footer_blocks = Vec::new();
    let mut main_blocks = Vec::new();

    for block in blocks {
        match block.order_label {
            OrderLabel::Header => header_blocks.push(block),
            OrderLabel::Footer => footer_blocks.push(block),
            _ => main_blocks.push(block),
        }
    }

    sort_blocks_by_y(&mut header_blocks);
    sort_blocks_by_y(&mut footer_blocks);

    let sorted_main = sort_main_blocks(main_blocks, page_width);

    let mut result = Vec::with_capacity(elements.len());
    result.extend(header_blocks.into_iter().map(|b| b.original_index));
    result.extend(sorted_main.into_iter().map(|b| b.original_index));
    result.extend(footer_blocks.into_iter().map(|b| b.original_index));

    result
}

fn sort_blocks_by_y(blocks: &mut [SortableBlock]) {
    blocks.sort_by(|a, b| {
        a.bbox
            .y_min()
            .partial_cmp(&b.bbox.y_min())
            .unwrap_or(std::cmp::Ordering::Equal)
    });
}

fn sort_main_blocks(mut blocks: Vec<SortableBlock>, page_width: f32) -> Vec<SortableBlock> {
    if blocks.is_empty() {
        return blocks;
    }

    // 1. Cross-layout detection (PaddleX get_layout_structure)
    detect_cross_layout(&mut blocks, page_width);

    // 2. Separate blocks for XY-cut vs special insertion
    // PaddleX SKIP_ORDER_LABELS are inserted by weighted distance after main XY-cut.
    let mut xy_cut_blocks = Vec::new();
    let mut doc_title_blocks = Vec::new();
    let mut weighted_insert_blocks = Vec::new();
    let mut unordered_blocks = Vec::new();

    for block in blocks {
        match block.order_label {
            OrderLabel::CrossLayout
            | OrderLabel::CrossReference
            | OrderLabel::Vision
            | OrderLabel::VisionTitle => weighted_insert_blocks.push(block),
            OrderLabel::DocTitle => doc_title_blocks.push(block),
            OrderLabel::Unordered => unordered_blocks.push(block),
            _ => xy_cut_blocks.push(block),
        }
    }

    // 3. Direction-aware XY-cut on xy_cut_blocks
    let mut sorted_blocks = if !xy_cut_blocks.is_empty() {
        direction_aware_xycut_sort(&mut xy_cut_blocks)
    } else {
        Vec::new()
    };

    // 4. Match unsorted blocks using weighted distance insertion
    // Order: doc_title first (PaddleX inserts first doc_title at position 0)
    sort_blocks_by_y(&mut doc_title_blocks);
    for (i, block) in doc_title_blocks.into_iter().enumerate() {
        if i == 0 && sorted_blocks.is_empty() {
            sorted_blocks.push(block);
        } else if i == 0 {
            sorted_blocks.insert(0, block);
        } else {
            weighted_distance_insert(block, &mut sorted_blocks, SortDirection::Horizontal);
        }
    }

    // Vision/cross-layout/title blocks are inserted after XY-cut.
    sort_blocks_by_y(&mut weighted_insert_blocks);
    for block in weighted_insert_blocks {
        weighted_distance_insert(block, &mut sorted_blocks, SortDirection::Horizontal);
    }

    // Unordered blocks using manhattan distance
    sort_blocks_by_y(&mut unordered_blocks);
    for block in unordered_blocks {
        manhattan_insert(block, &mut sorted_blocks);
    }

    // 5. Associate child blocks (vision titles next to vision parents)
    associate_child_blocks(&mut sorted_blocks);

    sorted_blocks
}

/// Direction-aware XY-cut sorting (PaddleX xycut_enhanced lines 539-584).
///
/// If single column or all blocks have 1 line → use secondary direction (xy_cut).
/// If multi-column → use primary direction (yx_cut).
fn direction_aware_xycut_sort(blocks: &mut [SortableBlock]) -> Vec<SortableBlock> {
    let bboxes: Vec<BoundingBox> = blocks.iter().map(|b| b.bbox.clone()).collect();
    let max_text_lines = blocks.iter().map(|b| b.num_lines).max().unwrap_or(1);

    // Check column structure using horizontal projection
    let discontinuous = calculate_discontinuous_projection(&bboxes, SortDirection::Horizontal);

    // Shrink overlapping boxes before XY-cut
    shrink_overlapping_boxes(blocks, SortDirection::Vertical);

    let shrunk_bboxes: Vec<BoundingBox> = blocks.iter().map(|b| b.bbox.clone()).collect();

    let sorted_indices = if discontinuous.len() == 1 || max_text_lines == 1 {
        // Single column: use secondary direction (XY-cut = X first, then Y)
        sort_by_xycut(&shrunk_bboxes, SortDirection::Horizontal, 1)
    } else {
        // Multi-column: use primary direction (YX-cut = Y first, then X)
        sort_by_xycut(&shrunk_bboxes, SortDirection::Vertical, 1)
    };

    sorted_indices
        .into_iter()
        .map(|i| blocks[i].clone())
        .collect()
}

/// Cross-layout detection (port of PaddleX `get_layout_structure`).
///
/// Marks blocks that span multiple columns as `CrossLayout`.
///
/// The naive algorithm is O(n³). We reduce it to O(n² + k²) per outer block
/// (where k = |h_neighbors| ≪ n for typical sparse layouts) by precomputing
/// horizontal-projection overlaps once and building per-block neighbor lists.
/// Both the 2D bbox overlap and the projection-overlap conditions require
/// horizontal overlap with `block_idx`, so any block outside its h_neighbors
/// can never trigger a cross-layout classification and is safely skipped.
fn detect_cross_layout(blocks: &mut [SortableBlock], _page_width: f32) {
    if blocks.len() < 2 {
        return;
    }

    // Sort by x_min, then width (matching PaddleX)
    blocks.sort_by(|a, b| {
        a.bbox
            .x_min()
            .partial_cmp(&b.bbox.x_min())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| {
                a.width()
                    .partial_cmp(&b.width())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
    });

    let mask_labels = [
        OrderLabel::DocTitle,
        OrderLabel::CrossLayout,
        OrderLabel::CrossReference,
    ];

    let n = blocks.len();

    let block_data: Vec<(BoundingBox, OrderLabel, f32, f32)> = blocks
        .iter()
        .map(|b| {
            (
                b.bbox.clone(),
                b.order_label,
                b.area(),
                b.long_side_length(),
            )
        })
        .collect();

    let text_line_heights: Vec<f32> = blocks.iter().map(|b| b.text_line_height).collect();

    // Precompute the full horizontal-projection overlap matrix (O(n²)) so that
    // inner loops can do a single table lookup instead of recomputing the ratio.
    let h_proj: Vec<Vec<f32>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    calculate_projection_overlap_ratio(
                        &block_data[i].0,
                        &block_data[j].0,
                        SortDirection::Horizontal,
                    )
                })
                .collect()
        })
        .collect();

    // For each block, the set of other blocks that horizontally overlap with it.
    // Both inner loops only act on blocks in this set, so we iterate only over
    // neighbors rather than 0..n.
    let h_neighbors: Vec<Vec<usize>> = (0..n)
        .map(|i| (0..n).filter(|&j| j != i && h_proj[i][j] > 0.0).collect())
        .collect();

    for block_idx in 0..n {
        if mask_labels.contains(&block_data[block_idx].1) {
            continue;
        }

        let mut mark_block_cross = false;

        // Iterate only over blocks that horizontally overlap with block_idx.
        // Any block without horizontal overlap has bbox_overlap == 0 and
        // match_proj == 0, so it cannot affect the cross-layout decision.
        for &ref_idx in &h_neighbors[block_idx] {
            if mask_labels.contains(&block_data[ref_idx].1) {
                continue;
            }
            if blocks[ref_idx].order_label == OrderLabel::CrossLayout {
                continue;
            }
            if blocks[block_idx].order_label == OrderLabel::CrossLayout {
                break;
            }

            let bbox_overlap =
                calculate_overlap_ratio(&block_data[block_idx].0, &block_data[ref_idx].0);

            if bbox_overlap > 0.0 {
                if block_data[ref_idx].1 == OrderLabel::Vision {
                    blocks[ref_idx].order_label = OrderLabel::CrossLayout;
                    continue;
                }
                if bbox_overlap > 0.1 && block_data[block_idx].2 < block_data[ref_idx].2 {
                    mark_block_cross = true;
                    break;
                }
            }

            // h_proj[block_idx][ref_idx] > 0 is guaranteed by h_neighbors, so
            // the match_proj > 0 guard from the original is always satisfied here.

            // Iterate over the same neighbor set for second_ref: every triggering
            // condition (bbox_overlap2 > 0.1 or second_match_proj > 0) requires
            // horizontal overlap with block_idx, which is exactly h_neighbors.
            for &second_ref_idx in &h_neighbors[block_idx] {
                if second_ref_idx == ref_idx || mask_labels.contains(&block_data[second_ref_idx].1)
                {
                    continue;
                }
                if blocks[second_ref_idx].order_label == OrderLabel::CrossLayout {
                    continue;
                }

                let bbox_overlap2 = calculate_overlap_ratio(
                    &block_data[block_idx].0,
                    &block_data[second_ref_idx].0,
                );

                if bbox_overlap2 > 0.1 {
                    if block_data[second_ref_idx].1 == OrderLabel::Vision {
                        blocks[second_ref_idx].order_label = OrderLabel::CrossLayout;
                        continue;
                    }
                    if block_data[block_idx].1 == OrderLabel::Vision
                        || block_data[block_idx].2 < block_data[second_ref_idx].2
                    {
                        mark_block_cross = true;
                        break;
                    }
                }

                // second_match_proj > 0 is guaranteed (second_ref_idx ∈ h_neighbors[block_idx]).
                // Use precomputed table for ref_match_proj to avoid re-computing.
                let ref_match_proj = h_proj[ref_idx][second_ref_idx];
                let secondary_ref_match = calculate_projection_overlap_ratio(
                    &block_data[ref_idx].0,
                    &block_data[second_ref_idx].0,
                    SortDirection::Vertical,
                );

                if ref_match_proj == 0.0 && secondary_ref_match > 0.0 {
                    if block_data[block_idx].1 == OrderLabel::Vision {
                        mark_block_cross = true;
                        break;
                    }
                    // Both ref blocks are normal text with sufficient width
                    if block_data[ref_idx].1 == OrderLabel::NormalText
                        && block_data[second_ref_idx].1 == OrderLabel::NormalText
                        && block_data[ref_idx].3
                            > text_line_heights[ref_idx]
                                * CROSS_LAYOUT_REF_TEXT_BLOCK_WORDS_NUM_THRESHOLD
                        && block_data[second_ref_idx].3
                            > text_line_heights[second_ref_idx]
                                * CROSS_LAYOUT_REF_TEXT_BLOCK_WORDS_NUM_THRESHOLD
                    {
                        mark_block_cross = true;
                        break;
                    }
                }
            }

            if mark_block_cross {
                break;
            }
        }

        if mark_block_cross {
            if block_data[block_idx].1 == OrderLabel::Reference {
                blocks[block_idx].order_label = OrderLabel::CrossReference;
            } else {
                blocks[block_idx].order_label = OrderLabel::CrossLayout;
            }
        }
    }
}

/// Calculate discontinuous projection intervals along a direction.
///
/// Returns merged intervals where boxes project onto the axis.
/// Single interval = single column; multiple = multi-column.
fn calculate_discontinuous_projection(
    bboxes: &[BoundingBox],
    direction: SortDirection,
) -> Vec<(i32, i32)> {
    if bboxes.is_empty() {
        return Vec::new();
    }

    let mut intervals: Vec<(i32, i32)> = bboxes
        .iter()
        .map(|b| match direction {
            SortDirection::Horizontal => (b.x_min() as i32, b.x_max() as i32),
            SortDirection::Vertical => (b.y_min() as i32, b.y_max() as i32),
        })
        .collect();

    intervals.sort_by_key(|&(start, _)| start);

    let mut merged = Vec::new();
    let (mut current_start, mut current_end) = intervals[0];

    for &(start, end) in &intervals[1..] {
        if start <= current_end {
            current_end = current_end.max(end);
        } else {
            merged.push((current_start, current_end));
            current_start = start;
            current_end = end;
        }
    }
    merged.push((current_start, current_end));

    merged
}

/// Shrink slightly overlapping boxes at their midpoint (PaddleX `shrink_overlapping_boxes`).
///
/// For consecutive blocks sorted by position, if they have small overlap in the
/// cut direction (0 < overlap < 10%), split at the midpoint of overlap.
fn shrink_overlapping_boxes(blocks: &mut [SortableBlock], direction: SortDirection) {
    if blocks.len() < 2 {
        return;
    }

    // Sort by the end coordinate of the cut direction
    match direction {
        SortDirection::Vertical => {
            blocks.sort_by(|a, b| {
                a.bbox
                    .y_max()
                    .partial_cmp(&b.bbox.y_max())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
        SortDirection::Horizontal => {
            blocks.sort_by(|a, b| {
                a.bbox
                    .x_max()
                    .partial_cmp(&b.bbox.x_max())
                    .unwrap_or(std::cmp::Ordering::Equal)
            });
        }
    }

    for i in 0..blocks.len() - 1 {
        let perp_direction = match direction {
            SortDirection::Vertical => SortDirection::Horizontal,
            SortDirection::Horizontal => SortDirection::Vertical,
        };

        let cut_iou =
            calculate_projection_overlap_ratio(&blocks[i].bbox, &blocks[i + 1].bbox, direction);
        let match_iou = calculate_projection_overlap_ratio(
            &blocks[i].bbox,
            &blocks[i + 1].bbox,
            perp_direction,
        );

        match direction {
            SortDirection::Vertical => {
                let y2 = blocks[i].bbox.y_max();
                let y1_prime = blocks[i + 1].bbox.y_min();
                if (match_iou > 0.0 && cut_iou > 0.0 && cut_iou < 0.1)
                    || y2 == y1_prime
                    || (y2 - y1_prime).abs() <= 3.0
                {
                    let overlap_y_min = blocks[i].bbox.y_min().max(blocks[i + 1].bbox.y_min());
                    let overlap_y_max = blocks[i].bbox.y_max().min(blocks[i + 1].bbox.y_max());
                    let split_y = ((overlap_y_min + overlap_y_max) / 2.0).floor();

                    if blocks[i].bbox.y_min() < blocks[i + 1].bbox.y_min() {
                        let new_bbox = BoundingBox::from_coords(
                            blocks[i].bbox.x_min(),
                            blocks[i].bbox.y_min(),
                            blocks[i].bbox.x_max(),
                            split_y - 1.0,
                        );
                        blocks[i].bbox = new_bbox;
                        let new_bbox2 = BoundingBox::from_coords(
                            blocks[i + 1].bbox.x_min(),
                            split_y + 1.0,
                            blocks[i + 1].bbox.x_max(),
                            blocks[i + 1].bbox.y_max(),
                        );
                        blocks[i + 1].bbox = new_bbox2;
                    } else {
                        let new_bbox = BoundingBox::from_coords(
                            blocks[i].bbox.x_min(),
                            split_y - 1.0,
                            blocks[i].bbox.x_max(),
                            blocks[i].bbox.y_max(),
                        );
                        blocks[i].bbox = new_bbox;
                        let new_bbox2 = BoundingBox::from_coords(
                            blocks[i + 1].bbox.x_min(),
                            blocks[i + 1].bbox.y_min(),
                            blocks[i + 1].bbox.x_max(),
                            split_y + 1.0,
                        );
                        blocks[i + 1].bbox = new_bbox2;
                    }
                }
            }
            SortDirection::Horizontal => {
                let x2 = blocks[i].bbox.x_max();
                let x1_prime = blocks[i + 1].bbox.x_min();
                if (match_iou > 0.0 && cut_iou > 0.0 && cut_iou < 0.1)
                    || x2 == x1_prime
                    || (x2 - x1_prime).abs() <= 3.0
                {
                    let overlap_x_min = blocks[i].bbox.x_min().max(blocks[i + 1].bbox.x_min());
                    let overlap_x_max = blocks[i].bbox.x_max().min(blocks[i + 1].bbox.x_max());
                    let split_x = ((overlap_x_min + overlap_x_max) / 2.0).floor();

                    if blocks[i].bbox.x_min() < blocks[i + 1].bbox.x_min() {
                        let new_bbox = BoundingBox::from_coords(
                            blocks[i].bbox.x_min(),
                            blocks[i].bbox.y_min(),
                            split_x - 1.0,
                            blocks[i].bbox.y_max(),
                        );
                        blocks[i].bbox = new_bbox;
                        let new_bbox2 = BoundingBox::from_coords(
                            split_x + 1.0,
                            blocks[i + 1].bbox.y_min(),
                            blocks[i + 1].bbox.x_max(),
                            blocks[i + 1].bbox.y_max(),
                        );
                        blocks[i + 1].bbox = new_bbox2;
                    } else {
                        let new_bbox = BoundingBox::from_coords(
                            split_x - 1.0,
                            blocks[i].bbox.y_min(),
                            blocks[i].bbox.x_max(),
                            blocks[i].bbox.y_max(),
                        );
                        blocks[i].bbox = new_bbox;
                        let new_bbox2 = BoundingBox::from_coords(
                            blocks[i + 1].bbox.x_min(),
                            blocks[i + 1].bbox.y_min(),
                            split_x + 1.0,
                            blocks[i + 1].bbox.y_max(),
                        );
                        blocks[i + 1].bbox = new_bbox2;
                    }
                }
            }
        }
    }
}

/// Associate vision title blocks with their nearest vision parent (PaddleX `insert_child_blocks`).
///
/// Moves VisionTitle blocks adjacent to their nearest Vision block.
fn associate_child_blocks(sorted_blocks: &mut Vec<SortableBlock>) {
    if sorted_blocks.len() < 2 {
        return;
    }

    // Find vision title indices that need to be moved
    let mut moves: Vec<(usize, usize)> = Vec::new(); // (from_idx, target_vision_idx)

    for (i, block) in sorted_blocks.iter().enumerate() {
        if block.order_label != OrderLabel::VisionTitle {
            continue;
        }

        // Find nearest Vision block by edge distance
        let mut best_vision_idx = None;
        let mut best_distance = f32::INFINITY;

        for (j, other) in sorted_blocks.iter().enumerate() {
            if other.order_label != OrderLabel::Vision {
                continue;
            }
            let dist = get_nearest_edge_distance(&block.bbox, &other.bbox, &[1.0, 1.0, 1.0, 1.0]);
            if dist < best_distance {
                best_distance = dist;
                best_vision_idx = Some(j);
            }
        }

        // Only move if close enough (< 2 * text_line_height of the vision block)
        if let Some(vision_idx) = best_vision_idx {
            let threshold = sorted_blocks[vision_idx].text_line_height * 3.0;
            if best_distance < threshold {
                // Should be placed right before or after the vision block
                if block.bbox.y_min() < sorted_blocks[vision_idx].bbox.y_min() {
                    moves.push((i, vision_idx)); // place before
                } else {
                    moves.push((i, vision_idx + 1)); // place after
                }
            }
        }
    }

    // Apply moves (process in reverse order to maintain indices)
    for (from_idx, target_idx) in moves.into_iter().rev() {
        // Only move if the title is not already adjacent
        if from_idx == target_idx || from_idx + 1 == target_idx {
            continue;
        }
        let block = sorted_blocks.remove(from_idx);
        let adjusted_target = if from_idx < target_idx {
            target_idx - 1
        } else {
            target_idx
        };
        let insert_pos = adjusted_target.min(sorted_blocks.len());
        sorted_blocks.insert(insert_pos, block);
    }
}

/// Insert a block using Manhattan distance (for unordered blocks).
fn manhattan_insert(block: SortableBlock, sorted_blocks: &mut Vec<SortableBlock>) {
    if sorted_blocks.is_empty() {
        sorted_blocks.push(block);
        return;
    }

    let mut min_distance = f32::INFINITY;
    let mut nearest_index = 0;

    for (idx, sorted_block) in sorted_blocks.iter().enumerate() {
        let distance = (block.bbox.x_min() - sorted_block.bbox.x_min()).abs()
            + (block.bbox.y_min() - sorted_block.bbox.y_min()).abs();
        if distance < min_distance {
            min_distance = distance;
            nearest_index = idx;
        }
    }

    sorted_blocks.insert(nearest_index + 1, block);
}

/// Insert a block using weighted distance logic (PaddleX `weighted_distance_insert`).
fn weighted_distance_insert(
    block: SortableBlock,
    sorted_blocks: &mut Vec<SortableBlock>,
    region_direction: SortDirection,
) {
    if sorted_blocks.is_empty() {
        sorted_blocks.push(block);
        return;
    }

    let tolerance_len = EDGE_DISTANCE_COMPARE_TOLERANCE_LEN;
    let (x1, y1, x2, _y2) = (
        block.bbox.x_min(),
        block.bbox.y_min(),
        block.bbox.x_max(),
        block.bbox.y_max(),
    );

    let mut min_weighted_distance = f32::INFINITY;
    let mut _min_edge_distance = f32::INFINITY;
    let mut min_up_edge_distance = f32::INFINITY;
    let mut nearest_index = 0;

    for (idx, sorted_block) in sorted_blocks.iter().enumerate() {
        let (x1_prime, y1_prime, x2_prime, y2_prime) = (
            sorted_block.bbox.x_min(),
            sorted_block.bbox.y_min(),
            sorted_block.bbox.x_max(),
            sorted_block.bbox.y_max(),
        );

        let weight = get_weights(&block.order_label, block.direction);
        let raw_edge_distance = get_nearest_edge_distance(&block.bbox, &sorted_block.bbox, &weight);

        // Quantize edge distance to 50px buckets to ignore minor vertical misalignments
        // between columns, allowing left_dist to correctly resolve reading order.
        let edge_distance = (raw_edge_distance / 50.0).floor() * 50.0;

        let (mut up_dist, mut left_dist) = match region_direction {
            SortDirection::Horizontal => (y1_prime, x1_prime),
            SortDirection::Vertical => (-x2_prime, y1_prime),
        };

        let is_below = match region_direction {
            SortDirection::Horizontal => y2_prime < y1,
            SortDirection::Vertical => x1_prime > x2,
        };

        // Flip signs for special blocks that are below
        let is_special = !matches!(block.order_label, OrderLabel::Unordered)
            || matches!(
                block.order_label,
                OrderLabel::DocTitle
                    | OrderLabel::ParagraphTitle
                    | OrderLabel::Vision
                    | OrderLabel::VisionTitle
                    | OrderLabel::CrossLayout
            );

        if is_special && is_below {
            up_dist = -up_dist;
            left_dist = -left_dist;
        }

        if (min_up_edge_distance - up_dist).abs() <= tolerance_len {
            up_dist = min_up_edge_distance;
        }

        let weighted_dist =
            edge_distance * EDGE_WEIGHT + up_dist * UP_EDGE_WEIGHT + left_dist * LEFT_EDGE_WEIGHT;

        _min_edge_distance = _min_edge_distance.min(edge_distance);
        min_up_edge_distance = min_up_edge_distance.min(up_dist);

        if weighted_dist < min_weighted_distance {
            min_weighted_distance = weighted_dist;

            let y1_i = (y1.floor() as i32) / 2;
            let y1_p_i = (y1_prime.floor() as i32) / 2;

            let (sorted_dist_val, block_dist_val) = if (y1_i - y1_p_i).abs() > 0 {
                (y1_prime, y1)
            } else if matches!(region_direction, SortDirection::Horizontal) {
                let x1_i = (x1.floor() as i32) / 2;
                let x2_i = (x2.floor() as i32) / 2;
                if (x1_i - x2_i).abs() > 0 {
                    (x1_prime, x1)
                } else {
                    let (cx, cy) = block.center();
                    let (scx, scy) = sorted_block.center();
                    (scx * scx + scy * scy, cx * cx + cy * cy)
                }
            } else {
                (x1_prime, x1)
            };

            if block_dist_val > sorted_dist_val {
                nearest_index = idx + 1;
            } else {
                nearest_index = idx;
            }
        }
    }

    if nearest_index > sorted_blocks.len() {
        nearest_index = sorted_blocks.len();
    }

    sorted_blocks.insert(nearest_index, block);
}

fn get_weights(label: &OrderLabel, direction: SortDirection) -> [f32; 4] {
    match label {
        OrderLabel::DocTitle => {
            if matches!(direction, SortDirection::Horizontal) {
                [1.0, 0.1, 0.1, 1.0]
            } else {
                [0.2, 0.1, 1.0, 1.0]
            }
        }
        OrderLabel::ParagraphTitle
        | OrderLabel::Vision
        | OrderLabel::VisionTitle
        | OrderLabel::CrossLayout => [1.0, 1.0, 0.1, 1.0],
        _ => [1.0, 1.0, 1.0, 0.1],
    }
}

/// Calculate nearest edge distance between two boxes.
fn get_nearest_edge_distance(b1: &BoundingBox, b2: &BoundingBox, weights: &[f32; 4]) -> f32 {
    let h_overlap = calculate_projection_overlap_ratio(b1, b2, SortDirection::Horizontal);
    let v_overlap = calculate_projection_overlap_ratio(b1, b2, SortDirection::Vertical);

    if h_overlap > 0.0 && v_overlap > 0.0 {
        return 0.0;
    }

    let mut min_x = 0.0;
    let mut min_y = 0.0;

    if h_overlap == 0.0 {
        let d1 = (b1.x_min() - b2.x_max()).abs();
        let d2 = (b1.x_max() - b2.x_min()).abs();
        let w = if b1.x_max() < b2.x_min() {
            weights[0]
        } else {
            weights[1]
        };
        min_x = d1.min(d2) * w;
    }

    if v_overlap == 0.0 {
        let d1 = (b1.y_min() - b2.y_max()).abs();
        let d2 = (b1.y_max() - b2.y_min()).abs();
        let w = if b1.y_max() < b2.y_min() {
            weights[2]
        } else {
            weights[3]
        };
        min_y = d1.min(d2) * w;
    }

    min_x + min_y
}

/// Calculate projection overlap ratio (IoU) along a single axis.
fn calculate_projection_overlap_ratio(
    b1: &BoundingBox,
    b2: &BoundingBox,
    direction: SortDirection,
) -> f32 {
    let (min1, max1, min2, max2) = match direction {
        SortDirection::Horizontal => (b1.x_min(), b1.x_max(), b2.x_min(), b2.x_max()),
        SortDirection::Vertical => (b1.y_min(), b1.y_max(), b2.y_min(), b2.y_max()),
    };

    let intersection = (max1.min(max2) - min1.max(min2)).max(0.0);
    let union = max1.max(max2) - min1.min(min2);

    if union > 0.0 {
        intersection / union
    } else {
        0.0
    }
}
