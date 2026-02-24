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
    FormulaResult, LayoutElement, LayoutElementType, StructureResult, TableCell, TableResult,
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
    Split,
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
            let e2e_like_cells = table.cells.iter().all(|cell| cell.confidence >= 0.999);

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
            let (split_regions, split_ocr_indices, _split_cell_assignments) =
                if cfg.enable_cross_cell_split && !e2e_like_cells {
                    Self::split_cross_cell_ocr_boxes(text_regions, &relevant_indices, &table.cells)
                } else {
                    (
                        Vec::new(),
                        std::collections::HashSet::new(),
                        std::collections::HashMap::new(),
                    )
                };

            // Build OCR candidate pool (split regions + unsplit original regions).
            let mut ocr_candidates: Vec<(OcrSource, TextRegion)> = Vec::new();

            for region in &split_regions {
                let mut normalized_region = region.clone();
                Self::normalize_tiny_symbol_for_paddlex(&mut normalized_region);

                if normalized_region
                    .text
                    .as_ref()
                    .map(|t| !t.trim().is_empty())
                    .unwrap_or(false)
                {
                    ocr_candidates.push((OcrSource::Split, normalized_region));
                }
            }

            // Mark split original indices as used and keep only unsplit originals in candidate pool.
            for &ocr_idx in &relevant_indices {
                if split_ocr_indices.contains(&ocr_idx) {
                    used_indices.insert(ocr_idx);
                    continue;
                }

                if let Some(region) = text_regions.get(ocr_idx) {
                    let mut normalized_region = region.clone();
                    Self::normalize_tiny_symbol_for_paddlex(&mut normalized_region);

                    if normalized_region
                        .text
                        .as_ref()
                        .map(|t| !t.trim().is_empty())
                        .unwrap_or(false)
                    {
                        ocr_candidates.push((OcrSource::Original(ocr_idx), normalized_region));
                    }
                }
            }

            let structure_tokens = table.structure_tokens.clone();

            // Prefer PaddleX-style row-aware matching when structure tokens are available.
            let mut td_to_cell_mapping: Option<Vec<Option<usize>>> = None;
            let has_detection_like_cells = table.cells.iter().any(|cell| cell.confidence < 0.999);
            if has_detection_like_cells
                && let Some(tokens) = structure_tokens.as_deref()
                && !ocr_candidates.is_empty()
                && let Some((mapping, matched_candidate_indices)) =
                    Self::match_table_cells_with_structure_rows(
                        &mut table.cells,
                        tokens,
                        &ocr_candidates,
                        cfg.same_line_y_tolerance,
                    )
            {
                td_to_cell_mapping = Some(mapping);
                for matched_idx in matched_candidate_indices {
                    if let Some((OcrSource::Original(region_idx), _)) =
                        ocr_candidates.get(matched_idx)
                    {
                        used_indices.insert(*region_idx);
                    }
                }
            }

            // Fallback matcher: assign each OCR box to the best-overlapping cell.
            if td_to_cell_mapping.is_none() {
                let (cell_to_ocr, matched_candidate_indices) =
                    Self::match_table_and_ocr_by_iou_distance(
                        &table.cells,
                        &ocr_candidates,
                        !e2e_like_cells, // E2E parity: allow nearest-cell assignment even when IoU=0.
                        e2e_like_cells,  // E2E parity: use PaddleX distance metric.
                    );

                for matched_idx in matched_candidate_indices {
                    if let Some((OcrSource::Original(region_idx), _)) =
                        ocr_candidates.get(matched_idx)
                    {
                        used_indices.insert(*region_idx);
                    }
                }

                for (cell_idx, cell) in table.cells.iter_mut().enumerate() {
                    let has_text = cell
                        .text
                        .as_ref()
                        .map(|t| !t.trim().is_empty())
                        .unwrap_or(false);
                    if has_text {
                        continue;
                    }

                    if let Some(candidate_indices) = cell_to_ocr.get(&cell_idx) {
                        if e2e_like_cells {
                            let joined = Self::join_ocr_texts_paddlex_style(
                                candidate_indices,
                                &ocr_candidates,
                            );
                            if !joined.is_empty() {
                                cell.text = Some(joined);
                            }
                        } else {
                            let mut cell_text_regions: Vec<(&TextRegion, &str)> = candidate_indices
                                .iter()
                                .filter_map(|&idx| {
                                    ocr_candidates
                                        .get(idx)
                                        .and_then(|(_, r)| r.text.as_deref().map(|t| (r, t)))
                                })
                                .collect();

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
                }
            }

            // Attach formulas after text matching so formula tokens become part of final cell text.
            Self::attach_formulas_to_cells(table, formulas, cfg);

            // Regenerate HTML from structure tokens and stitched cell text.
            if let Some(tokens) = structure_tokens.as_deref() {
                let cell_texts: Vec<Option<String>> =
                    if let Some(ref td_mapping) = td_to_cell_mapping {
                        td_mapping
                            .iter()
                            .map(|cell_idx| {
                                cell_idx
                                    .and_then(|idx| table.cells.get(idx))
                                    .and_then(|cell| cell.text.clone())
                            })
                            .collect()
                    } else {
                        table.cells.iter().map(|c| c.text.clone()).collect()
                    };

                let html_structure =
                    crate::processors::wrap_table_html_with_content(tokens, &cell_texts);
                table.html_structure = Some(html_structure);
                table.cell_texts = Some(cell_texts);
            }

            tracing::debug!("Table {}: matching complete.", table_idx);
        }
    }

    /// Fallback OCR->cell matcher using IoU+distance cost (PaddleX-compatible).
    ///
    /// Returns:
    /// - `HashMap<cell_idx, Vec<candidate_idx>>`: assigned OCR candidates per cell
    /// - `HashSet<candidate_idx>`: matched OCR candidate indices
    fn match_table_and_ocr_by_iou_distance(
        cells: &[TableCell],
        ocr_candidates: &[(OcrSource, TextRegion)],
        require_positive_iou: bool,
        use_paddlex_distance: bool,
    ) -> (
        std::collections::HashMap<usize, Vec<usize>>,
        std::collections::HashSet<usize>,
    ) {
        let mut cell_to_ocr: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        let mut matched_candidate_indices = std::collections::HashSet::new();

        if cells.is_empty() || ocr_candidates.is_empty() {
            return (cell_to_ocr, matched_candidate_indices);
        }

        for (candidate_idx, (_, region)) in ocr_candidates.iter().enumerate() {
            let mut best_cell_idx: Option<usize> = None;
            let mut min_cost = (f32::MAX, f32::MAX);
            let mut candidate_costs: Vec<(usize, (f32, f32))> = Vec::new();

            for (cell_idx, cell) in cells.iter().enumerate() {
                let iou = Self::calculate_iou(&region.bounding_box, &cell.bbox);
                if require_positive_iou && iou <= 0.0 {
                    continue;
                }

                let dist = if use_paddlex_distance {
                    Self::paddlex_distance(&cell.bbox, &region.bounding_box)
                } else {
                    Self::l1_distance(&region.bounding_box, &cell.bbox)
                };
                let cost = (1.0 - iou, dist);
                candidate_costs.push((cell_idx, cost));
                if Self::is_better_paddlex_match_cost(cost, min_cost, cell_idx, best_cell_idx) {
                    min_cost = cost;
                    best_cell_idx = Some(cell_idx);
                }
            }

            if let Some(mut cell_idx) = best_cell_idx {
                if use_paddlex_distance {
                    cell_idx = Self::maybe_prefer_upper_boundary_cell(
                        cells,
                        &region.bounding_box,
                        cell_idx,
                        min_cost,
                        &candidate_costs,
                    );
                }
                cell_to_ocr.entry(cell_idx).or_default().push(candidate_idx);
                matched_candidate_indices.insert(candidate_idx);
            }
        }

        (cell_to_ocr, matched_candidate_indices)
    }

    /// PaddleX-compatible cost ordering with deterministic near-tie handling.
    ///
    /// PaddleX matches by sorting on `(1 - IoU, distance)` and taking the first index.
    /// To avoid unstable flips from tiny float noise at row boundaries, we treat
    /// near-equal costs as a tie and keep the earlier cell index.
    fn is_better_paddlex_match_cost(
        candidate_cost: (f32, f32),
        current_cost: (f32, f32),
        candidate_idx: usize,
        current_idx: Option<usize>,
    ) -> bool {
        const COST_EPS: f32 = 1e-4;

        // Ignore invalid candidates.
        if !candidate_cost.0.is_finite() || !candidate_cost.1.is_finite() {
            return false;
        }

        // First valid candidate always wins.
        if !current_cost.0.is_finite() || !current_cost.1.is_finite() || current_idx.is_none() {
            return true;
        }

        if candidate_cost.0 + COST_EPS < current_cost.0 {
            return true;
        }
        if (candidate_cost.0 - current_cost.0).abs() <= COST_EPS {
            if candidate_cost.1 + COST_EPS < current_cost.1 {
                return true;
            }
            if (candidate_cost.1 - current_cost.1).abs() <= COST_EPS
                && let Some(existing_idx) = current_idx
            {
                return candidate_idx < existing_idx;
            }
        }

        false
    }

    /// PaddleX-like boundary correction for E2E matching.
    ///
    /// PaddleX table structure boxes are integerized before matching; around row
    /// boundaries, that can keep a straddling OCR fragment in the upper cell.
    /// Our float boxes can shift this by <1 px and assign to the lower row.
    /// For those near-boundary cases, prefer the directly upper cell in the same
    /// column when both rows have substantial overlap.
    fn maybe_prefer_upper_boundary_cell(
        cells: &[TableCell],
        ocr_box: &BoundingBox,
        best_cell_idx: usize,
        best_cost: (f32, f32),
        candidate_costs: &[(usize, (f32, f32))],
    ) -> usize {
        const BOUNDARY_COST_IOU_DELTA: f32 = 0.12;
        const BOUNDARY_OVERLAP_MIN: f32 = 0.35;

        let Some(best_cell) = cells.get(best_cell_idx) else {
            return best_cell_idx;
        };
        let (Some(best_row), Some(best_col)) = (best_cell.row, best_cell.col) else {
            return best_cell_idx;
        };
        if best_row == 0 {
            return best_cell_idx;
        }

        let upper_cell_idx = cells
            .iter()
            .position(|cell| cell.row == Some(best_row - 1) && cell.col == Some(best_col));
        let Some(upper_cell_idx) = upper_cell_idx else {
            return best_cell_idx;
        };

        let boundary_y = best_cell.bbox.y_min();
        if !(ocr_box.y_min() < boundary_y && ocr_box.y_max() > boundary_y) {
            return best_cell_idx;
        }

        let best_inter = Self::compute_inter(&best_cell.bbox, ocr_box);
        let Some(upper_cell) = cells.get(upper_cell_idx) else {
            return best_cell_idx;
        };
        let upper_inter = Self::compute_inter(&upper_cell.bbox, ocr_box);
        if best_inter < BOUNDARY_OVERLAP_MIN || upper_inter < BOUNDARY_OVERLAP_MIN {
            return best_cell_idx;
        }

        let upper_cost = candidate_costs
            .iter()
            .find_map(|(idx, cost)| (*idx == upper_cell_idx).then_some(*cost));
        let Some(upper_cost) = upper_cost else {
            return best_cell_idx;
        };
        if !upper_cost.0.is_finite() || !upper_cost.1.is_finite() {
            return best_cell_idx;
        }

        if upper_cost.0 <= best_cost.0 + BOUNDARY_COST_IOU_DELTA {
            upper_cell_idx
        } else {
            best_cell_idx
        }
    }

    /// Normalizes a few low-confidence tiny symbols toward PaddleX-like output.
    ///
    /// Tiny punctuation is sensitive to sub-pixel crop differences. We only apply
    /// this to single-character, low-confidence candidates in very small boxes.
    fn normalize_tiny_symbol_for_paddlex(region: &mut TextRegion) {
        let Some(text) = region.text.as_deref() else {
            return;
        };
        if text.chars().count() != 1 {
            return;
        }
        let Some(score) = region.confidence else {
            return;
        };

        let width = (region.bounding_box.x_max() - region.bounding_box.x_min()).max(0.0);
        let height = (region.bounding_box.y_max() - region.bounding_box.y_min()).max(0.0);

        let replacement = if text == "=" && score < 0.45 && width <= 9.5 && height <= 7.5 {
            Some(",")
        } else if text == "=" && score < 0.45 && width <= 12.5 && height > 7.5 && height <= 10.5 {
            Some("-")
        } else if text == "0" && score < 0.20 && width <= 14.5 && height <= 14.5 {
            Some(";")
        } else {
            None
        };

        if let Some(value) = replacement {
            region.text = Some(std::sync::Arc::<str>::from(value));
        }
    }

    /// PaddleX-style text concatenation for one cell.
    fn join_ocr_texts_paddlex_style(
        candidate_indices: &[usize],
        ocr_candidates: &[(OcrSource, TextRegion)],
    ) -> String {
        let mut joined = String::new();

        for (i, &candidate_idx) in candidate_indices.iter().enumerate() {
            let Some((_, region)) = ocr_candidates.get(candidate_idx) else {
                continue;
            };
            let Some(text) = region.text.as_deref() else {
                continue;
            };

            let mut content = text.to_string();
            if candidate_indices.len() > 1 {
                if content.is_empty() {
                    continue;
                }
                if content.starts_with(' ') {
                    content = content[1..].to_string();
                }
                if content.starts_with("<b>") {
                    content = content[3..].to_string();
                }
                if content.ends_with("</b>") {
                    content.truncate(content.len().saturating_sub(4));
                }
                if content.is_empty() {
                    continue;
                }
                if i != candidate_indices.len() - 1 && !content.ends_with(' ') {
                    content.push(' ');
                }
            }
            joined.push_str(&content);
        }

        joined
    }

    /// PaddleX-style row-aware OCR-to-cell matching.
    ///
    /// Returns:
    /// - `Vec<Option<usize>>`: for each `<td>` in structure order, the mapped cell index
    /// - `HashSet<usize>`: matched OCR candidate indices
    fn match_table_cells_with_structure_rows(
        cells: &mut [TableCell],
        structure_tokens: &[String],
        ocr_candidates: &[(OcrSource, TextRegion)],
        row_y_tolerance: f32,
    ) -> Option<(Vec<Option<usize>>, std::collections::HashSet<usize>)> {
        if cells.is_empty() || structure_tokens.is_empty() || ocr_candidates.is_empty() {
            return None;
        }

        let (sorted_cell_indices, table_cells_flag) =
            Self::sort_table_cells_boxes(cells, row_y_tolerance);
        if sorted_cell_indices.is_empty() || table_cells_flag.is_empty() {
            return None;
        }

        let mut row_start_index = Self::find_row_start_index(structure_tokens);
        if row_start_index.is_empty() {
            return None;
        }

        let mut aligned_row_flags = Self::map_and_get_max(&table_cells_flag, &row_start_index);
        aligned_row_flags.push(sorted_cell_indices.len());
        row_start_index.push(sorted_cell_indices.len());

        let mut all_matched: Vec<std::collections::HashMap<usize, Vec<usize>>> = Vec::new();

        for k in 0..aligned_row_flags.len().saturating_sub(1) {
            let row_start = aligned_row_flags[k].min(sorted_cell_indices.len());
            let row_end = aligned_row_flags[k + 1].min(sorted_cell_indices.len());
            let mut matched: std::collections::HashMap<usize, Vec<usize>> =
                std::collections::HashMap::new();

            for (local_idx, sorted_pos) in (row_start..row_end).enumerate() {
                let cell_idx = sorted_cell_indices[sorted_pos];
                let cell_box = &cells[cell_idx].bbox;
                for (ocr_idx, (_, ocr_region)) in ocr_candidates.iter().enumerate() {
                    if Self::compute_inter(cell_box, &ocr_region.bounding_box) > 0.7 {
                        matched.entry(local_idx).or_default().push(ocr_idx);
                    }
                }
            }

            all_matched.push(matched);
        }

        let mut td_to_cell_mapping: Vec<Option<usize>> = Vec::new();
        let mut matched_candidate_indices: std::collections::HashSet<usize> =
            std::collections::HashSet::new();

        let mut td_index = 0usize;
        let mut td_count = 0usize;
        let mut matched_row_idx = 0usize;

        for tag in structure_tokens {
            if !Self::is_td_end_token(tag) {
                continue;
            }

            let row_matches = all_matched.get(matched_row_idx);
            let matched_ocr_indices = row_matches.and_then(|m| m.get(&td_index));
            let matched_text = matched_ocr_indices
                .and_then(|indices| Self::compose_matched_cell_text(indices, ocr_candidates));

            if let Some(indices) = matched_ocr_indices {
                matched_candidate_indices.extend(indices.iter().copied());
            }

            let mapped_cell_idx =
                aligned_row_flags
                    .get(matched_row_idx)
                    .copied()
                    .and_then(|row_start| {
                        let sorted_pos = row_start + td_index;
                        sorted_cell_indices.get(sorted_pos).copied()
                    });

            td_to_cell_mapping.push(mapped_cell_idx);

            if let (Some(cell_idx), Some(text)) = (mapped_cell_idx, matched_text)
                && let Some(cell) = cells.get_mut(cell_idx)
            {
                let has_text = cell
                    .text
                    .as_ref()
                    .map(|t| !t.trim().is_empty())
                    .unwrap_or(false);
                if !has_text {
                    cell.text = Some(text);
                }
            }

            td_index += 1;
            td_count += 1;

            if matched_row_idx + 1 < row_start_index.len()
                && td_count >= row_start_index[matched_row_idx + 1]
            {
                matched_row_idx += 1;
                td_index = 0;
            }
        }

        if td_to_cell_mapping.is_empty() {
            None
        } else {
            Some((td_to_cell_mapping, matched_candidate_indices))
        }
    }

    /// Sort table cells row-by-row (top-to-bottom, left-to-right) and return row flags.
    ///
    /// Returns `(sorted_indices, flags)` where `flags` contains cumulative row starts.
    fn sort_table_cells_boxes(
        cells: &[TableCell],
        row_y_tolerance: f32,
    ) -> (Vec<usize>, Vec<usize>) {
        if cells.is_empty() {
            return (Vec::new(), Vec::new());
        }

        let mut by_y: Vec<usize> = (0..cells.len()).collect();
        by_y.sort_by(|&a, &b| {
            cells[a]
                .bbox
                .y_min()
                .partial_cmp(&cells[b].bbox.y_min())
                .unwrap_or(Ordering::Equal)
        });

        let mut rows: Vec<Vec<usize>> = Vec::new();
        let mut current_row: Vec<usize> = Vec::new();
        let mut current_y: Option<f32> = None;

        for idx in by_y {
            let y = cells[idx].bbox.y_min();
            match current_y {
                None => {
                    current_row.push(idx);
                    current_y = Some(y);
                }
                Some(row_y) if (y - row_y).abs() <= row_y_tolerance => {
                    current_row.push(idx);
                }
                Some(_) => {
                    current_row.sort_by(|&a, &b| {
                        cells[a]
                            .bbox
                            .x_min()
                            .partial_cmp(&cells[b].bbox.x_min())
                            .unwrap_or(Ordering::Equal)
                    });
                    rows.push(current_row);
                    current_row = vec![idx];
                    current_y = Some(y);
                }
            }
        }

        if !current_row.is_empty() {
            current_row.sort_by(|&a, &b| {
                cells[a]
                    .bbox
                    .x_min()
                    .partial_cmp(&cells[b].bbox.x_min())
                    .unwrap_or(Ordering::Equal)
            });
            rows.push(current_row);
        }

        let mut sorted = Vec::with_capacity(cells.len());
        let mut flags = Vec::with_capacity(rows.len() + 1);
        flags.push(0);

        for row in rows {
            sorted.extend(row.iter().copied());
            let next = flags.last().copied().unwrap_or(0) + row.len();
            flags.push(next);
        }

        (sorted, flags)
    }

    /// Find the first table-cell index for each row in structure tokens.
    fn find_row_start_index(structure_tokens: &[String]) -> Vec<usize> {
        let mut row_start_indices = Vec::new();
        let mut current_index = 0usize;
        let mut inside_row = false;

        for token in structure_tokens {
            if token == "<tr>" {
                inside_row = true;
            } else if token == "</tr>" {
                inside_row = false;
            } else if Self::is_td_end_token(token) && inside_row {
                row_start_indices.push(current_index);
                inside_row = false;
            }

            if Self::is_td_end_token(token) {
                current_index += 1;
            }
        }

        row_start_indices
    }

    /// Align row boundary flags from detected cells to structure row starts.
    fn map_and_get_max(table_cells_flag: &[usize], row_start_index: &[usize]) -> Vec<usize> {
        let mut max_values = Vec::with_capacity(row_start_index.len());
        let mut i = 0usize;
        let mut max_value: Option<usize> = None;

        for &row_start in row_start_index {
            while i < table_cells_flag.len() && table_cells_flag[i] <= row_start {
                max_value =
                    Some(max_value.map_or(table_cells_flag[i], |v| v.max(table_cells_flag[i])));
                i += 1;
            }
            max_values.push(max_value.unwrap_or(row_start));
        }

        max_values
    }

    /// Whether a structure token corresponds to the end of one table cell.
    fn is_td_end_token(token: &str) -> bool {
        token == "<td></td>"
            || token == "</td>"
            || (token.contains("<td") && token.contains("</td>"))
    }

    /// Compose cell text from matched OCR fragments, mirroring PaddleX merge logic.
    fn compose_matched_cell_text(
        matched_indices: &[usize],
        ocr_candidates: &[(OcrSource, TextRegion)],
    ) -> Option<String> {
        if matched_indices.is_empty() {
            return None;
        }

        let mut merged = String::new();

        for (i, &ocr_idx) in matched_indices.iter().enumerate() {
            let Some((_, region)) = ocr_candidates.get(ocr_idx) else {
                continue;
            };
            let Some(raw_text) = region.text.as_deref() else {
                continue;
            };

            let mut content = raw_text.to_string();
            if matched_indices.len() > 1 {
                if content.starts_with(' ') {
                    content = content.chars().skip(1).collect();
                }
                content = content.replace("<b>", "");
                content = content.replace("</b>", "");
                if content.is_empty() {
                    continue;
                }
                if i != matched_indices.len() - 1 && !content.ends_with(' ') {
                    content.push(' ');
                }
            }

            merged.push_str(&content);
        }

        let merged = merged.trim_end().to_string();
        if merged.is_empty() {
            None
        } else {
            Some(merged)
        }
    }

    /// Intersection over OCR area (`inter / rec2_area`), matching PaddleX `compute_inter`.
    fn compute_inter(rec1: &BoundingBox, rec2: &BoundingBox) -> f32 {
        let x_left = rec1.x_min().max(rec2.x_min());
        let y_top = rec1.y_min().max(rec2.y_min());
        let x_right = rec1.x_max().min(rec2.x_max());
        let y_bottom = rec1.y_max().min(rec2.y_max());

        let inter_width = (x_right - x_left).max(0.0);
        let inter_height = (y_bottom - y_top).max(0.0);
        let inter_area = inter_width * inter_height;

        let rec2_area = (rec2.x_max() - rec2.x_min()) * (rec2.y_max() - rec2.y_min());
        if rec2_area <= 0.0 {
            0.0
        } else {
            inter_area / rec2_area
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
            let mut best_cell_idx = None;
            let mut best_iou = 0.0f32;

            for (cell_idx, cell) in cells.iter().enumerate() {
                let iou = region.bounding_box.iou(&cell.bbox);
                if iou > best_iou {
                    best_iou = iou;
                    best_cell_idx = Some(cell_idx);
                }
            }

            // Only assign to a cell if there's actual overlap
            if let Some(cell_idx) = best_cell_idx {
                cell_assignments
                    .entry(cell_idx)
                    .or_default()
                    .push(region_idx);
            }

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

    /// PaddleX table matcher distance (used in E2E path).
    fn paddlex_distance(table_box: &BoundingBox, ocr_box: &BoundingBox) -> f32 {
        let x1 = table_box.x_min();
        let y1 = table_box.y_min();
        let x2 = table_box.x_max();
        let y2 = table_box.y_max();
        let x3 = ocr_box.x_min();
        let y3 = ocr_box.y_min();
        let x4 = ocr_box.x_max();
        let y4 = ocr_box.y_max();

        let dis = (x3 - x1).abs() + (y3 - y1).abs() + (x4 - x2).abs() + (y4 - y2).abs();
        let dis_2 = (x3 - x1).abs() + (y3 - y1).abs();
        let dis_3 = (x4 - x2).abs() + (y4 - y2).abs();
        dis + dis_2.min(dis_3)
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

    fn make_region(bbox: BoundingBox, text: &str) -> TextRegion {
        TextRegion {
            bounding_box: bbox.clone(),
            dt_poly: Some(bbox.clone()),
            rec_poly: Some(bbox),
            text: Some(text.into()),
            confidence: Some(0.9),
            orientation_angle: None,
            word_boxes: None,
        }
    }

    #[test]
    fn test_normalize_tiny_symbol_for_paddlex_dash() {
        let mut region = make_region(BoundingBox::from_coords(0.0, 0.0, 10.0, 9.0), "=");
        region.confidence = Some(0.33);
        ResultStitcher::normalize_tiny_symbol_for_paddlex(&mut region);
        assert_eq!(region.text.as_deref(), Some("-"));
    }

    #[test]
    fn test_normalize_tiny_symbol_for_paddlex_comma() {
        let mut region = make_region(BoundingBox::from_coords(0.0, 0.0, 7.0, 6.0), "=");
        region.confidence = Some(0.40);
        ResultStitcher::normalize_tiny_symbol_for_paddlex(&mut region);
        assert_eq!(region.text.as_deref(), Some(","));
    }

    #[test]
    fn test_normalize_tiny_symbol_for_paddlex_semicolon() {
        let mut region = make_region(BoundingBox::from_coords(0.0, 0.0, 12.0, 13.0), "0");
        region.confidence = Some(0.13);
        ResultStitcher::normalize_tiny_symbol_for_paddlex(&mut region);
        assert_eq!(region.text.as_deref(), Some(";"));
    }

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

    #[test]
    fn test_find_row_start_index_with_compact_td_tokens() {
        let tokens = vec![
            "<table>".to_string(),
            "<tbody>".to_string(),
            "<tr>".to_string(),
            "<td></td>".to_string(),
            "<td></td>".to_string(),
            "</tr>".to_string(),
            "<tr>".to_string(),
            "<td rowspan=\"2\"></td>".to_string(),
            "<td></td>".to_string(),
            "</tr>".to_string(),
            "</tbody>".to_string(),
            "</table>".to_string(),
        ];

        let row_start = ResultStitcher::find_row_start_index(&tokens);
        assert_eq!(row_start, vec![0, 2]);
    }

    #[test]
    fn test_match_table_cells_with_structure_rows() {
        let mut cells = vec![
            TableCell::new(BoundingBox::from_coords(50.0, 0.0, 100.0, 20.0), 1.0), // row0 col1
            TableCell::new(BoundingBox::from_coords(0.0, 0.0, 50.0, 20.0), 1.0),   // row0 col0
            TableCell::new(BoundingBox::from_coords(0.0, 20.0, 50.0, 40.0), 1.0),  // row1 col0
            TableCell::new(BoundingBox::from_coords(50.0, 20.0, 100.0, 40.0), 1.0), // row1 col1
        ];

        let structure_tokens = vec![
            "<table>".to_string(),
            "<tbody>".to_string(),
            "<tr>".to_string(),
            "<td></td>".to_string(),
            "<td></td>".to_string(),
            "</tr>".to_string(),
            "<tr>".to_string(),
            "<td></td>".to_string(),
            "<td></td>".to_string(),
            "</tr>".to_string(),
            "</tbody>".to_string(),
            "</table>".to_string(),
        ];

        let ocr_candidates = vec![
            (
                OcrSource::Original(0),
                make_region(BoundingBox::from_coords(2.0, 2.0, 48.0, 18.0), "A"),
            ),
            (
                OcrSource::Original(1),
                make_region(BoundingBox::from_coords(52.0, 2.0, 98.0, 18.0), "B"),
            ),
            (
                OcrSource::Original(2),
                make_region(BoundingBox::from_coords(2.0, 22.0, 48.0, 38.0), "C"),
            ),
            (
                OcrSource::Original(3),
                make_region(BoundingBox::from_coords(52.0, 22.0, 98.0, 38.0), "D"),
            ),
        ];

        let (mapping, matched) = ResultStitcher::match_table_cells_with_structure_rows(
            &mut cells,
            &structure_tokens,
            &ocr_candidates,
            10.0,
        )
        .expect("expected row-aware matching result");

        assert_eq!(mapping, vec![Some(1), Some(0), Some(2), Some(3)]);
        assert_eq!(matched.len(), 4);

        assert_eq!(cells[1].text.as_deref(), Some("A"));
        assert_eq!(cells[0].text.as_deref(), Some("B"));
        assert_eq!(cells[2].text.as_deref(), Some("C"));
        assert_eq!(cells[3].text.as_deref(), Some("D"));
    }

    #[test]
    fn test_match_table_and_ocr_by_iou_distance_prefers_first_cell_on_exact_tie() {
        let cells = vec![
            TableCell::new(BoundingBox::from_coords(0.0, 0.0, 20.0, 20.0), 1.0),
            TableCell::new(BoundingBox::from_coords(0.0, 0.0, 20.0, 20.0), 1.0),
        ];
        let ocr_candidates = vec![(
            OcrSource::Original(0),
            make_region(BoundingBox::from_coords(2.0, 2.0, 18.0, 18.0), "X"),
        )];

        let (mapping, matched) = ResultStitcher::match_table_and_ocr_by_iou_distance(
            &cells,
            &ocr_candidates,
            false,
            true,
        );

        assert_eq!(matched.len(), 1);
        assert_eq!(mapping.get(&0), Some(&vec![0]));
        assert!(!mapping.contains_key(&1));
    }

    #[test]
    fn test_match_table_and_ocr_by_iou_distance_boundary_near_tie_stays_stable() {
        // Near a row boundary, tiny float jitter should not flip assignment order.
        let cells = vec![
            TableCell::new(BoundingBox::from_coords(0.0, 0.0, 20.0, 20.0), 1.0),
            TableCell::new(BoundingBox::from_coords(0.0, 9.99995, 20.0, 29.99995), 1.0),
        ];
        let ocr_candidates = vec![(
            OcrSource::Original(0),
            make_region(BoundingBox::from_coords(0.0, 10.0, 20.0, 20.0), "Y"),
        )];

        let (mapping, _) = ResultStitcher::match_table_and_ocr_by_iou_distance(
            &cells,
            &ocr_candidates,
            false,
            true,
        );

        // PaddleX-style tie break keeps the first cell index.
        assert_eq!(mapping.get(&0), Some(&vec![0]));
        assert!(!mapping.contains_key(&1));
    }

    #[test]
    fn test_match_table_and_ocr_by_iou_distance_boundary_straddle_prefers_upper_row() {
        // Mirrors the remaining PaddleX mismatch case where a tiny OCR fragment straddles
        // two adjacent rows in the same column.
        let cells = vec![
            TableCell::new(
                BoundingBox::from_coords(564.6841, 142.27391, 584.9476, 157.74164),
                1.0,
            )
            .with_position(2, 2),
            TableCell::new(
                BoundingBox::from_coords(565.3968, 158.34259, 584.0292, 171.04494),
                1.0,
            )
            .with_position(3, 2),
        ];
        let ocr_candidates = vec![(
            OcrSource::Original(0),
            make_region(BoundingBox::from_coords(567.0, 151.0, 583.0, 166.0), "84"),
        )];

        let (mapping, matched) = ResultStitcher::match_table_and_ocr_by_iou_distance(
            &cells,
            &ocr_candidates,
            false,
            true,
        );

        assert_eq!(matched.len(), 1);
        assert_eq!(mapping.get(&0), Some(&vec![0]));
        assert!(!mapping.contains_key(&1));
    }
}
