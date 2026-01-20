//! Stitching module for combining OCR results.
//!
//! This module provides functionality to associate recognized text regions with
//! layout elements (such as tables and paragraphs) to create a unified structured result.
//!
//! ## PP-StructureV3 Alignment
//!
//! The stitching logic follows PP-StructureV3's fusion strategy:
//! 1. **Label-based filtering**: Special regions (formula, table, seal) are excluded from OCR matching
//! 2. **Content preservation**: Formulas retain LaTeX, tables retain HTML structure
//! 3. **Reading order**: Elements are assigned `order_index` based on spatial sorting
//! 4. **Orphan handling**: Unmatched OCR regions create new text elements

use crate::oarocr::TextRegion;
use oar_ocr_core::domain::structure::{
    FormulaResult, LayoutElement, LayoutElementType, StructureResult, TableResult,
};
use oar_ocr_core::processors::{
    BoundingBox, SplitConfig as OcrSplitConfig, create_expanded_ocr_for_table,
};
use std::cmp::Ordering;

/// Source of an OCR region reference, distinguishing between regions that were
/// split across cell boundaries and original regions.
#[derive(Clone, Copy, Debug)]
enum OcrSource {
    /// Index into the split_regions vector (created by cross-cell splitting)
    Split(usize),
    /// Index into the original text_regions slice
    Original(usize),
}

/// Labels that should be excluded from OCR text matching.
/// These regions have their own specialized content (LaTeX, HTML, etc.)
const EXCLUDED_FROM_OCR_LABELS: [LayoutElementType; 4] = [
    LayoutElementType::Formula,
    LayoutElementType::FormulaNumber,
    LayoutElementType::Table,
    LayoutElementType::Seal,
];

#[derive(Clone)]
pub struct StitchConfig {
    pub overlap_min_pixels: f32,
    pub cell_text_min_ioa: f32,
    pub require_text_center_inside_cell: bool,
    pub cell_merge_min_iou: f32,
    pub formula_to_cell_min_iou: f32,
    pub same_line_y_tolerance: f32,
    /// Whether to enable cross-cell OCR box splitting.
    /// When enabled, OCR boxes that span multiple table cells will be split
    /// at cell boundaries and their text distributed proportionally.
    pub enable_cross_cell_split: bool,
}

impl Default for StitchConfig {
    fn default() -> Self {
        Self {
            overlap_min_pixels: 3.0,
            cell_text_min_ioa: 0.6,
            require_text_center_inside_cell: true,
            cell_merge_min_iou: 0.3,
            formula_to_cell_min_iou: 0.01,
            same_line_y_tolerance: 10.0,
            enable_cross_cell_split: true,
        }
    }
}

/// Stitcher for combining results from different OCR tasks.
pub struct ResultStitcher;

impl ResultStitcher {
    /// Stitches text regions into layout elements and tables within the structure result.
    ///
    /// This method follows PP-StructureV3's fusion strategy:
    /// 1. Stitch OCR text into tables (cell-level matching)
    /// 2. Stitch OCR text into layout elements (excluding formula/table/seal)
    /// 3. Fill formula elements with LaTeX content from formula results
    /// 4. Create new text elements for orphan OCR regions
    /// 5. Sort elements and assign reading order indices
    pub fn stitch(result: &mut StructureResult) {
        let cfg = StitchConfig::default();
        Self::stitch_with_config(result, &cfg);
    }

    pub fn stitch_with_config(result: &mut StructureResult, cfg: &StitchConfig) {
        // Track which regions have been used
        let mut used_region_indices = std::collections::HashSet::new();

        // Get text regions (clone to avoid borrow issues)
        let regions = result.text_regions.clone().unwrap_or_default();

        tracing::debug!("Stitching: {} text regions", regions.len());

        // 1. Stitch text into tables
        // For tables, we also want recognized formulas to participate in cell content
        // matching, similar to how formulas are injected into the OCR results used
        // for table recognition.
        Self::stitch_tables(
            &mut result.tables,
            &regions,
            &result.formulas,
            &mut used_region_indices,
            cfg,
        );

        tracing::debug!(
            "After stitch_tables: {} regions used",
            used_region_indices.len()
        );

        // 2. Stitch text into layout elements (excluding special types)
        Self::stitch_layout_elements(
            &mut result.layout_elements,
            &regions,
            &mut used_region_indices,
            cfg,
        );

        tracing::debug!(
            "After stitch_layout_elements: {} regions used",
            used_region_indices.len()
        );

        // 3. Fill formula elements with LaTeX content
        Self::fill_formula_content(&mut result.layout_elements, &result.formulas);

        // 4. Mark text regions that overlap with excluded element types (Formula, Seal)
        // as used to prevent them from becoming orphans.
        // - Formulas: content comes from LaTeX recognition, OCR is redundant/noise.
        // - Seals: content comes from specialized seal OCR.
        // - Tables: content comes from OCR stitching. We do NOT suppress tables here because
        //   text inside a table that wasn't assigned to a cell (in step 1) should be preserved
        //   as an orphan (e.g. caption, header, or matching failure).
        for element in &result.layout_elements {
            if matches!(
                element.element_type,
                LayoutElementType::Formula | LayoutElementType::Seal
            ) {
                for (idx, region) in regions.iter().enumerate() {
                    if Self::is_overlapping(&element.bbox, &region.bounding_box, cfg) {
                        used_region_indices.insert(idx);
                    }
                }
            }
        }

        // 5. Handle unmatched text regions (create new layout elements)
        // PP-StructureV3 alignment: Filter out orphan text regions that significantly overlap
        // with table regions, as these are likely table cell text that failed to match cells.
        // These shouldn't become separate layout elements.
        let table_bboxes: Vec<&BoundingBox> = result
            .layout_elements
            .iter()
            .filter(|e| e.element_type == LayoutElementType::Table)
            .map(|e| &e.bbox)
            .collect();

        let original_element_count = result.layout_elements.len();
        let mut new_elements = Vec::new();
        for (idx, region) in regions.iter().enumerate() {
            if !used_region_indices.contains(&idx)
                && let Some(text) = &region.text
            {
                // Filter out text that overlaps significantly with tables
                // These are likely table cell text that didn't match any cell
                let overlaps_table = table_bboxes
                    .iter()
                    .any(|table_bbox| region.bounding_box.ioa(table_bbox) > 0.3);

                if overlaps_table {
                    // Skip - this text is inside a table and should not be a separate element
                    continue;
                }

                // Create a new layout element for this orphan text
                // We treat it as a generic "text" element
                let element = LayoutElement::new(
                    region.bounding_box.clone(),
                    LayoutElementType::Text,
                    region.confidence.unwrap_or(0.0),
                )
                .with_text(text.as_ref().to_string());

                new_elements.push(element);
            }
        }

        // If region_blocks exist, assign orphan elements to their containing regions
        // and update element_indices to maintain proper grouping
        if let Some(ref mut region_blocks) = result.region_blocks {
            for (new_idx, new_element) in new_elements.iter().enumerate() {
                let element_index = original_element_count + new_idx;

                // Find the region that best contains this orphan element
                let mut best_region_idx: Option<usize> = None;
                let mut best_overlap = 0.0f32;

                for (region_idx, region) in region_blocks.iter().enumerate() {
                    // Check if this element overlaps with the region bbox
                    let overlap = new_element.bbox.intersection_area(&region.bbox);
                    if overlap > best_overlap {
                        best_overlap = overlap;
                        best_region_idx = Some(region_idx);
                    }
                }

                // Add to the best matching region, or leave unassigned if no overlap
                if let Some(region_idx) = best_region_idx {
                    region_blocks[region_idx]
                        .element_indices
                        .push(element_index);
                }
            }
        }

        result.layout_elements.extend(new_elements);

        // 6. Sort all layout elements spatially and assign order indices
        // PP-StructureV3: When region_blocks is present, elements are already sorted
        // by hierarchical region order - skip re-sorting to preserve the structure
        let width = if let Some(img) = &result.rectified_img {
            img.width() as f32
        } else {
            // Estimate width from max x coordinate
            result
                .layout_elements
                .iter()
                .map(|e| e.bbox.x_max())
                .fold(0.0f32, f32::max)
                .max(1000.0) // default fallback
        };

        // When region_blocks exist, layout_elements are already sorted correctly
        // by XY-cut with region hierarchy in structure.rs - do NOT re-sort here.
        // Only sort when region_blocks is NOT present.
        if result.region_blocks.is_none() {
            Self::sort_layout_elements(&mut result.layout_elements, width, cfg);
        }

        // Assign order indices regardless of sorting
        Self::assign_order_indices(&mut result.layout_elements);
    }

    /// Fills formula layout elements with their corresponding LaTeX content.
    ///
    /// Matches formula results to layout elements by bounding box overlap (IOU > 0.5).
    fn fill_formula_content(elements: &mut [LayoutElement], formulas: &[FormulaResult]) {
        for element in elements.iter_mut() {
            if element.element_type.is_formula() {
                // Find the best matching formula result by IOU
                if let Some(formula) = formulas
                    .iter()
                    .filter(|f| element.bbox.iou(&f.bbox) > 0.5)
                    .max_by(|a, b| {
                        element
                            .bbox
                            .iou(&a.bbox)
                            .partial_cmp(&element.bbox.iou(&b.bbox))
                            .unwrap_or(Ordering::Equal)
                    })
                {
                    element.text = Some(formula.latex.clone());
                }
            }
        }
    }

    /// Assigns reading order indices to layout elements.
    ///
    /// Only elements that should be included in reading order get an index.
    /// PP-StructureV3 includes: text, titles, tables, formulas, images, seals, etc.
    fn assign_order_indices(elements: &mut [LayoutElement]) {
        let mut order_index = 1u32;
        for element in elements.iter_mut() {
            // Assign order index to elements that should be in reading order
            // (matching PP-StructureV3's visualize_index_labels)
            if Self::should_have_order_index(element.element_type) {
                element.order_index = Some(order_index);
                order_index += 1;
            }
        }
    }

    /// Determines if an element type should have a reading order index.
    ///
    /// Based on PP-StructureV3's `visualize_index_labels`.
    fn should_have_order_index(element_type: LayoutElementType) -> bool {
        matches!(
            element_type,
            LayoutElementType::Text
                | LayoutElementType::Content
                | LayoutElementType::Abstract
                | LayoutElementType::DocTitle
                | LayoutElementType::ParagraphTitle
                | LayoutElementType::Table
                | LayoutElementType::Image
                | LayoutElementType::Chart
                | LayoutElementType::Formula
                | LayoutElementType::Seal
                | LayoutElementType::Reference
                | LayoutElementType::ReferenceContent
                | LayoutElementType::List
                | LayoutElementType::FigureTitle
                | LayoutElementType::TableTitle
                | LayoutElementType::ChartTitle
                | LayoutElementType::FigureTableChartTitle
        )
    }

    fn stitch_tables(
        tables: &mut [TableResult],
        text_regions: &[TextRegion],
        formulas: &[FormulaResult],
        used_indices: &mut std::collections::HashSet<usize>,
        cfg: &StitchConfig,
    ) {
        for (table_idx, table) in tables.iter_mut().enumerate() {
            if table.cells.is_empty() {
                continue;
            }

            // 1. Filter relevant text regions (those overlapping the table area)
            let table_bbox = table.bbox.clone(); // Use table bbox
            let relevant_indices: Vec<usize> = text_regions
                .iter()
                .enumerate()
                .filter(|(idx, region)| {
                    !used_indices.contains(idx)
                        && Self::is_overlapping(&table_bbox, &region.bounding_box, cfg)
                })
                .map(|(idx, _)| idx)
                .collect();

            // 1.5. Cross-cell OCR splitting (new step)
            // Detect OCR boxes that span multiple cells and split them at cell boundaries.
            // This improves accuracy for complex tables with rowspan/colspan.
            let (split_regions, split_ocr_indices, split_cell_assignments) =
                if cfg.enable_cross_cell_split {
                    Self::split_cross_cell_ocr_boxes(text_regions, &relevant_indices, &table.cells)
                } else {
                    (
                        Vec::new(),
                        std::collections::HashSet::new(),
                        std::collections::HashMap::new(),
                    )
                };

            // 2. Match OCR boxes to Cells (global OCR fallback for cells without per-cell OCR)
            // Map: cell_index -> List of OCR sources (split or original regions)
            let mut cell_to_ocr: std::collections::HashMap<usize, Vec<OcrSource>> =
                std::collections::HashMap::new();

            // First, add pre-assigned split regions to cell_to_ocr
            for (cell_idx, ocr_indices) in &split_cell_assignments {
                for &ocr_idx in ocr_indices {
                    cell_to_ocr
                        .entry(*cell_idx)
                        .or_default()
                        .push(OcrSource::Split(ocr_idx));
                }
            }

            // Mark split original indices as used and process non-split regions
            for &ocr_idx in &relevant_indices {
                // Skip indices that were split
                if split_ocr_indices.contains(&ocr_idx) {
                    used_indices.insert(ocr_idx);
                    continue;
                }

                let region = &text_regions[ocr_idx];

                // Compute cost for each cell: (1 - IoU, L1_Distance)
                // We want to minimize this cost.
                let mut best_cell_idx = None;
                let mut min_cost = (1.0f32, f32::MAX);

                // Find candidate cells (optimization: filter by coarse intersection first if needed)
                for (cell_idx, cell) in table.cells.iter().enumerate() {
                    let iou = Self::calculate_iou(&region.bounding_box, &cell.bbox);

                    // Consider candidates with IoU > 0
                    if iou > 0.0 {
                        let dist = Self::l1_distance(&region.bounding_box, &cell.bbox);
                        // Primary sort key: 1 - IoU (maximizing IoU)
                        // Secondary sort key: L1 distance (minimizing distance)
                        let cost = (1.0 - iou, dist);

                        if cost < min_cost {
                            min_cost = cost;
                            best_cell_idx = Some(cell_idx);
                        }
                    }
                }

                // Assign to best cell if found
                if let Some(cell_idx) = best_cell_idx {
                    cell_to_ocr
                        .entry(cell_idx)
                        .or_default()
                        .push(OcrSource::Original(ocr_idx));
                    used_indices.insert(ocr_idx);
                }
            }

            // 3. Assign text to cells:
            // - If per-cell OCR has already filled cell.text, keep it
            // - Otherwise, fall back to global OCR mapped via IoU/distance
            for (cell_idx, cell) in table.cells.iter_mut().enumerate() {
                let has_text = cell
                    .text
                    .as_ref()
                    .map(|t| !t.trim().is_empty())
                    .unwrap_or(false);

                if has_text {
                    continue;
                }

                if let Some(ocr_sources) = cell_to_ocr.get(&cell_idx) {
                    let mut cell_text_regions: Vec<(&TextRegion, &str)> = ocr_sources
                        .iter()
                        .filter_map(|&source| match source {
                            OcrSource::Split(idx) => split_regions
                                .get(idx)
                                .and_then(|r| r.text.as_deref().map(|t| (r, t))),
                            OcrSource::Original(idx) => text_regions
                                .get(idx)
                                .and_then(|r| r.text.as_deref().map(|t| (r, t))),
                        })
                        .collect();

                    // Sort text within cell (top-to-bottom, left-to-right)
                    Self::sort_and_join_texts(
                        &mut cell_text_regions,
                        Some(&cell.bbox),
                        cfg,
                        |joined| {
                            if !joined.is_empty() {
                                cell.text = Some(joined);
                            }
                        },
                    );
                }
            }

            // 4. Attach formulas
            Self::attach_formulas_to_cells(table, formulas, cfg);

            // 5. Regenerate HTML
            if let Some(ref structure_tokens) = table.structure_tokens {
                let cell_texts: Vec<Option<String>> =
                    table.cells.iter().map(|c| c.text.clone()).collect();
                let html_structure =
                    crate::processors::wrap_table_html_with_content(structure_tokens, &cell_texts);
                table.html_structure = Some(html_structure);
                table.cell_texts = Some(cell_texts);
            }

            tracing::debug!("Table {}: matching complete.", table_idx);
        }
    }

    /// Detects and splits OCR boxes that span multiple table cells.
    ///
    /// Returns:
    /// - Vec<TextRegion>: New text regions created from split OCR boxes
    /// - HashSet<usize>: Indices of original regions that were split
    /// - HashMap<usize, Vec<usize>>: Mapping from cell_idx -> indices in the new split_regions vec
    fn split_cross_cell_ocr_boxes(
        text_regions: &[TextRegion],
        relevant_indices: &[usize],
        cells: &[oar_ocr_core::domain::structure::TableCell],
    ) -> (
        Vec<TextRegion>,
        std::collections::HashSet<usize>,
        std::collections::HashMap<usize, Vec<usize>>,
    ) {
        let mut split_regions: Vec<TextRegion> = Vec::new();
        let mut split_ocr_indices: std::collections::HashSet<usize> =
            std::collections::HashSet::new();
        let mut cell_assignments: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();

        // Build a subset of text regions for the table
        let table_regions: Vec<TextRegion> = relevant_indices
            .iter()
            .map(|&idx| text_regions[idx].clone())
            .collect();

        if table_regions.is_empty() || cells.is_empty() {
            return (split_regions, split_ocr_indices, cell_assignments);
        }

        // Use the cross-cell splitting utility
        let split_config = OcrSplitConfig::default();
        let (expanded, processed_local_indices) =
            create_expanded_ocr_for_table(&table_regions, cells, Some(&split_config));

        // Map local indices back to original indices
        for local_idx in processed_local_indices {
            if local_idx < relevant_indices.len() {
                split_ocr_indices.insert(relevant_indices[local_idx]);
            }
        }

        // Add expanded regions and track cell assignments
        for region in expanded {
            let region_idx = split_regions.len();

            // Find the best matching cell for this expanded region
            let mut best_cell_idx = 0usize;
            let mut best_iou = 0.0f32;

            for (cell_idx, cell) in cells.iter().enumerate() {
                let iou = region.bounding_box.iou(&cell.bbox);
                if iou > best_iou {
                    best_iou = iou;
                    best_cell_idx = cell_idx;
                }
            }

            cell_assignments
                .entry(best_cell_idx)
                .or_default()
                .push(region_idx);

            split_regions.push(region);
        }

        tracing::debug!(
            "Cross-cell OCR splitting: {} original regions processed, {} new regions created",
            split_ocr_indices.len(),
            split_regions.len()
        );

        (split_regions, split_ocr_indices, cell_assignments)
    }

    /// Attaches recognized formulas to the best-matching table cells.
    ///
    /// This mirrors behavior where formula recognition results are merged into the
    /// OCR content used for table structure recognition. Here we approximate that behavior by:
    /// - For each formula, finding the cell with maximum IoU
    /// - If IoU exceeds a small threshold, appending `$latex$` to that cell's text
    fn attach_formulas_to_cells(
        table: &mut TableResult,
        formulas: &[FormulaResult],
        cfg: &StitchConfig,
    ) {
        if formulas.is_empty() || table.cells.is_empty() {
            return;
        }

        for formula in formulas {
            let bbox = &formula.bbox;

            // Skip degenerate boxes
            let w = bbox.x_max() - bbox.x_min();
            let h = bbox.y_max() - bbox.y_min();
            if w <= 1.0 || h <= 1.0 {
                continue;
            }

            // Only consider formulas that overlap the table bbox at all
            if !Self::is_overlapping(&table.bbox, bbox, cfg) {
                continue;
            }

            // Find best-matching cell by IoU
            let mut best_cell_idx: Option<usize> = None;
            let mut best_iou = 0.0f32;

            for (cell_idx, cell) in table.cells.iter().enumerate() {
                let iou = Self::calculate_iou(&cell.bbox, bbox);
                if iou > best_iou {
                    best_iou = iou;
                    best_cell_idx = Some(cell_idx);
                }
            }

            if let Some(cell_idx) = best_cell_idx
                && best_iou > cfg.formula_to_cell_min_iou
            {
                let cell = &mut table.cells[cell_idx];

                // Append formula as LaTeX wrapped in $...$
                let formatted = if formula.latex.starts_with('$') && formula.latex.ends_with('$') {
                    formula.latex.clone()
                } else {
                    format!("${}$", formula.latex)
                };

                match &mut cell.text {
                    Some(existing) => {
                        if !existing.is_empty() {
                            existing.push(' ');
                        }
                        existing.push_str(&formatted);
                    }
                    None => {
                        cell.text = Some(formatted);
                    }
                }
            }
        }
    }

    /// Calculates the Intersection over Union (IoU) between two bounding boxes.
    fn calculate_iou(bbox1: &BoundingBox, bbox2: &BoundingBox) -> f32 {
        let x1_min = bbox1.x_min();
        let y1_min = bbox1.y_min();
        let x1_max = bbox1.x_max();
        let y1_max = bbox1.y_max();

        let x2_min = bbox2.x_min();
        let y2_min = bbox2.y_min();
        let x2_max = bbox2.x_max();
        let y2_max = bbox2.y_max();

        let inter_x_min = x1_min.max(x2_min);
        let inter_y_min = y1_min.max(y2_min);
        let inter_x_max = x1_max.min(x2_max);
        let inter_y_max = y1_max.min(y2_max);

        let inter_w = (inter_x_max - inter_x_min).max(0.0);
        let inter_h = (inter_y_max - inter_y_min).max(0.0);
        let inter_area = inter_w * inter_h;

        let area1 = (x1_max - x1_min) * (y1_max - y1_min);
        let area2 = (x2_max - x2_min) * (y2_max - y2_min);
        let union_area = area1 + area2 - inter_area;

        if union_area > 0.0 {
            inter_area / union_area
        } else {
            0.0
        }
    }

    /// Calculates the L1 distance between two axis-aligned boxes.
    fn l1_distance(bbox1: &BoundingBox, bbox2: &BoundingBox) -> f32 {
        let b1 = [bbox1.x_min(), bbox1.y_min(), bbox1.x_max(), bbox1.y_max()];
        let b2 = [bbox2.x_min(), bbox2.y_min(), bbox2.x_max(), bbox2.y_max()];

        (b2[0] - b1[0]).abs()
            + (b2[1] - b1[1]).abs()
            + (b2[2] - b1[2]).abs()
            + (b2[3] - b1[3]).abs()
    }

    fn stitch_layout_elements(
        elements: &mut [LayoutElement],
        text_regions: &[TextRegion],
        used_indices: &mut std::collections::HashSet<usize>,
        cfg: &StitchConfig,
    ) {
        tracing::debug!(
            "stitch_layout_elements: {} elements, {} regions, {} already used",
            elements.len(),
            text_regions.len(),
            used_indices.len()
        );

        for (elem_idx, element) in elements.iter_mut().enumerate() {
            // Skip special types that have their own content handling:
            // - Table: handled separately with cell-level matching
            // - Formula: filled with LaTeX content
            // - Seal: may have specialized seal OCR results
            // This matches PP-StructureV3's behavior in standardized_data()
            if EXCLUDED_FROM_OCR_LABELS.contains(&element.element_type) {
                continue;
            }

            let mut element_texts: Vec<(&TextRegion, &str)> = Vec::new();

            for (idx, region) in text_regions.iter().enumerate() {
                if let Some(text) = &region.text
                    && Self::is_overlapping(&element.bbox, &region.bounding_box, cfg)
                {
                    element_texts.push((region, text));
                    // Only mark as used if not already used (to allow sharing if needed,
                    // though typically strict assignment is better. Some systems allow one-to-many
                    // matching, but here we track usage to find orphans)
                    used_indices.insert(idx);
                }
            }

            if !element_texts.is_empty() {
                tracing::debug!(
                    "Element {} ({:?}): matched {} regions",
                    elem_idx,
                    element.element_type,
                    element_texts.len()
                );
            }

            Self::sort_and_join_texts(&mut element_texts, Some(&element.bbox), cfg, |joined| {
                element.text = Some(joined);
            });
        }
    }

    /// Checks if two bounding boxes overlap significantly (intersection dimensions > 3px).
    /// Matches `get_overlap_boxes_idx` logic.
    fn is_overlapping(bbox1: &BoundingBox, bbox2: &BoundingBox, cfg: &StitchConfig) -> bool {
        let x1_min = bbox1.x_min();
        let y1_min = bbox1.y_min();
        let x1_max = bbox1.x_max();
        let y1_max = bbox1.y_max();

        let x2_min = bbox2.x_min();
        let y2_min = bbox2.y_min();
        let x2_max = bbox2.x_max();
        let y2_max = bbox2.y_max();

        let inter_x_min = x1_min.max(x2_min);
        let inter_y_min = y1_min.max(y2_min);
        let inter_x_max = x1_max.min(x2_max);
        let inter_y_max = y1_max.min(y2_max);

        let inter_w = inter_x_max - inter_x_min;
        let inter_h = inter_y_max - inter_y_min;

        inter_w > cfg.overlap_min_pixels && inter_h > cfg.overlap_min_pixels
    }

    fn sort_and_join_texts<F>(
        texts: &mut Vec<(&TextRegion, &str)>,
        container_bbox: Option<&BoundingBox>,
        cfg: &StitchConfig,
        update_fn: F,
    ) where
        F: FnOnce(String),
    {
        if texts.is_empty() {
            return;
        }

        // Sort spatially: top-to-bottom, then left-to-right
        texts.sort_by(|(r1, _), (r2, _)| {
            let c1 = r1.bounding_box.center();
            let c2 = r2.bounding_box.center();

            // Y-difference tolerance for same line (10 pixels)
            if (c1.y - c2.y).abs() < cfg.same_line_y_tolerance {
                c1.x.partial_cmp(&c2.x).unwrap_or(Ordering::Equal)
            } else {
                c1.y.partial_cmp(&c2.y).unwrap_or(Ordering::Equal)
            }
        });

        // Smart text joining following format_line logic:
        // - Texts on the same line are joined directly (no separator)
        // - A space is added only if the previous text ends with an English letter
        // - Newlines are added conditionally based on geometric gap (paragraph break detection)
        let mut result = String::new();
        let mut prev_y: Option<f32> = None;
        let mut prev_region: Option<&TextRegion> = None;

        for (region, text) in texts.iter() {
            if text.is_empty() {
                continue;
            }

            let current_y = region.bounding_box.center().y;

            if let Some(py) = prev_y {
                // Check if this is a new line (Y-difference > tolerance)
                if (current_y - py).abs() > cfg.same_line_y_tolerance {
                    // New visual line detected.
                    // Check for hyphenation: if previous text ends with '-' and current starts with lowercase,
                    // this is likely a word break that should be joined without the hyphen.
                    let prev_ends_hyphen = result.ends_with('-');
                    let current_starts_lower =
                        text.chars().next().is_some_and(|c| c.is_lowercase());

                    if prev_ends_hyphen && current_starts_lower {
                        // Remove the trailing hyphen and join directly (dehyphenation)
                        result.pop();
                        // Don't add any separator - words should be joined
                    } else {
                        // Decide whether to insert '\n' (hard break) or ' ' (soft break/wrap).
                        let mut add_newline = false;

                        if let Some(container) = container_bbox
                            && let Some(last_region) = prev_region
                        {
                            let container_width = container.x_max() - container.x_min();
                            // If the previous line ended far from the right edge, it's likely a paragraph break.
                            // Heuristic: gap > 30% of container width
                            // Note: We use container.x_max because we assume LTR text.
                            let right_gap = container.x_max() - last_region.bounding_box.x_max();
                            if right_gap > container_width * 0.3 {
                                add_newline = true;
                            }
                        }
                        // If no container info, we default to NO newline (soft wrap) to avoid discontinuity,
                        // unless specific patterns dictate otherwise (future work).

                        if add_newline {
                            if !result.ends_with('\n') {
                                result.push('\n');
                            }
                        } else {
                            // Soft wrap - treat as space if needed (English) or join (CJK)
                            if let Some(last_char) = result.chars().last()
                                && last_char != '\n'
                                && needs_space_after(last_char)
                            {
                                result.push(' ');
                            }
                        }
                    }
                } else {
                    // Same visual line - join with smart spacing
                    if let Some(last_char) = result.chars().last()
                        && last_char != '\n'
                        && needs_space_after(last_char)
                    {
                        result.push(' ');
                    }
                }
            }

            result.push_str(text);
            prev_y = Some(current_y);
            prev_region = Some(region);
        }

        // Trim trailing whitespace
        let joined = result.trim_end().to_string();
        update_fn(joined);
    }

    /// Sorts layout elements using the XY-cut algorithm.
    ///
    /// When region blocks are not available, this provides a robust column-aware reading
    /// order that matches PP-StructureV3's `sort_by_xycut` behavior.
    fn sort_layout_elements(elements: &mut Vec<LayoutElement>, _width: f32, _cfg: &StitchConfig) {
        if elements.len() <= 1 {
            return;
        }

        // Use shared XY-cut implementation from processors module.
        let bboxes: Vec<BoundingBox> = elements.iter().map(|e| e.bbox.clone()).collect();
        let order = crate::processors::sort_by_xycut(
            &bboxes,
            crate::processors::SortDirection::Vertical,
            1,
        );

        if order.len() != elements.len() {
            return;
        }

        let mut reordered = Vec::with_capacity(elements.len());
        for idx in order {
            reordered.push(elements[idx].clone());
        }

        *elements = reordered;
    }
}

/// Checks if a space should be added after the given character.
/// Based on format_line logic: add space only after English letters.
fn needs_space_after(c: char) -> bool {
    c.is_ascii_alphabetic()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::oarocr::TextRegion;
    use oar_ocr_core::processors::BoundingBox;

    #[test]
    fn test_is_overlapping_threshold() {
        let b1 = BoundingBox::from_coords(0.0, 0.0, 10.0, 10.0);
        let b2 = BoundingBox::from_coords(5.0, 5.0, 20.0, 20.0);
        let cfg = StitchConfig::default();
        assert!(ResultStitcher::is_overlapping(&b1, &b2, &cfg));
        let cfg2 = StitchConfig {
            overlap_min_pixels: 5.0,
            ..cfg.clone()
        };
        assert!(!ResultStitcher::is_overlapping(&b1, &b2, &cfg2));
    }

    #[test]
    fn test_sort_and_join_texts_tolerance() {
        let b1 = BoundingBox::from_coords(0.0, 0.0, 10.0, 10.0);
        let b2 = BoundingBox::from_coords(12.0, 1.0, 20.0, 11.0);
        let r1 = TextRegion {
            bounding_box: b1.clone(),
            dt_poly: Some(b1.clone()),
            rec_poly: Some(b1),
            text: Some("A".into()),
            confidence: Some(0.9),
            orientation_angle: None,
            word_boxes: None,
        };
        let r2 = TextRegion {
            bounding_box: b2.clone(),
            dt_poly: Some(b2.clone()),
            rec_poly: Some(b2),
            text: Some("B".into()),
            confidence: Some(0.9),
            orientation_angle: None,
            word_boxes: None,
        };
        let mut texts = vec![(&r1, "A"), (&r2, "B")];
        let cfg = StitchConfig::default();
        let mut joined = String::new();
        ResultStitcher::sort_and_join_texts(&mut texts, None, &cfg, |j| {
            joined = j;
        });
        assert_eq!(joined, "A B");
    }
}
