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
    BoundingBox, SplitConfig as OcrSplitConfig, create_expanded_ocr_for_table, parse_cell_grid_info,
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
/// Labels excluded from OCR text matching in `stitch_layout_elements`.
/// PaddleX: formula results are injected into the OCR pool (via
/// `convert_formula_res_to_ocr_format`), so formula blocks participate
/// in normal OCR matching — only Table and Seal are excluded.
///
/// NOTE: After inline formula injection, formula elements have been absorbed
/// into text regions and should be excluded from stitching to prevent duplication.
const EXCLUDED_FROM_OCR_LABELS: [LayoutElementType; 3] = [
    LayoutElementType::Table,
    LayoutElementType::Seal,
    LayoutElementType::Formula, // Exclude formulas to prevent duplicate rendering after injection
];

#[derive(Clone)]
pub struct StitchConfig {
    pub overlap_min_pixels: f32,
    pub cell_text_min_ioa: f32,
    pub require_text_center_inside_cell: bool,
    pub cell_merge_min_iou: f32,
    pub formula_to_cell_min_iou: f32,
    /// Fallback pixel tolerance for line grouping.
    pub same_line_y_tolerance: f32,
    /// Minimum vertical overlap ratio (intersection / min(line_height)) to treat two spans as one line.
    pub line_height_iou_threshold: f32,
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
            line_height_iou_threshold: 0.6,
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

        // Get text regions (clone to avoid borrow issues, make mutable for injection)
        let mut regions = result.text_regions.clone().unwrap_or_default();

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

        // 1.5. Fill formula elements with LaTeX content FIRST
        // This must happen before inject_inline_formulas so formulas have text content
        Self::fill_formula_elements(&mut result.layout_elements, &result.formulas, cfg);

        // 1.6. Inject inline formulas into text regions
        // PaddleX: Small formula elements that overlap with text elements should be
        // absorbed into the text flow, not kept as separate layout elements.
        // This creates TextRegion entries with label="formula" that will be wrapped
        // with $...$ delimiters during text joining.
        Self::inject_inline_formulas(&mut result.layout_elements, &mut regions, cfg);

        // 2. Stitch text into layout elements (excluding special types)
        // Note: after inject_inline_formulas, some formula elements have had their text cleared
        // These won't be rendered separately in to_markdown
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

        // Note: fill_formula_elements was already called before inject_inline_formulas
        // Do NOT call it again here, as it would re-fill formulas that were injected and cleared

        // 3. Mark text regions that overlap with Seal elements as used
        // to prevent them from becoming orphans.
        // - Seals: content comes from specialized seal OCR.
        // - Tables: content comes from OCR stitching. We do NOT suppress tables here because
        //   text inside a table that wasn't assigned to a cell (in step 1) should be preserved
        //   as an orphan (e.g. caption, header, or matching failure).
        // - Formulas: now handled through normal OCR matching (step 2), already marked used.
        for element in &result.layout_elements {
            if element.element_type == LayoutElementType::Seal {
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

        let image_chart_bboxes: Vec<&BoundingBox> = result
            .layout_elements
            .iter()
            .filter(|e| {
                matches!(
                    e.element_type,
                    LayoutElementType::Image | LayoutElementType::Chart
                )
            })
            .map(|e| &e.bbox)
            .collect();

        // Collect figure/chart caption bboxes to infer undetected figure regions.
        // When the layout model detects a caption (e.g. "Figure 3...") but misses
        // the figure image itself, OCR text from the figure diagram becomes orphans.
        // We infer the figure area as the region above each caption within its x-range.
        let figure_caption_bboxes: Vec<&BoundingBox> = result
            .layout_elements
            .iter()
            .filter(|e| {
                matches!(
                    e.element_type,
                    LayoutElementType::FigureTitle
                        | LayoutElementType::ChartTitle
                        | LayoutElementType::FigureTableChartTitle
                )
            })
            .map(|e| &e.bbox)
            .collect();

        // Collect text/title element bboxes to check if an orphan is already
        // covered by a known content element (avoid filtering legitimate text)
        let content_element_bboxes: Vec<&BoundingBox> = result
            .layout_elements
            .iter()
            .filter(|e| {
                matches!(
                    e.element_type,
                    LayoutElementType::Text
                        | LayoutElementType::DocTitle
                        | LayoutElementType::ParagraphTitle
                        | LayoutElementType::Abstract
                )
            })
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

                // Filter out text inside Image/Chart regions
                let overlaps_image_chart = image_chart_bboxes
                    .iter()
                    .any(|bbox| region.bounding_box.ioa(bbox) > 0.5);

                if overlaps_image_chart {
                    continue;
                }

                // Filter out text in inferred figure regions (above figure/chart captions).
                // When the layout model detects a caption but not the figure itself,
                // OCR'd annotations from the figure diagram leak as orphan text.
                // Check: orphan is above a caption, within its x-range, and not inside
                // any existing text/title element.
                let in_inferred_figure_region = figure_caption_bboxes.iter().any(|cap| {
                    let orphan_bb = &region.bounding_box;
                    // Orphan must be above or overlapping with the caption's top
                    let above_caption = orphan_bb.y_max() < cap.y_max();
                    // Orphan must be within the caption's horizontal range (with margin)
                    let x_margin = (cap.x_max() - cap.x_min()) * 0.1;
                    let in_x_range = orphan_bb.x_min() >= (cap.x_min() - x_margin)
                        && orphan_bb.x_max() <= (cap.x_max() + x_margin);
                    above_caption && in_x_range
                });

                if in_inferred_figure_region {
                    // Verify the orphan is NOT inside any existing text/title element
                    let inside_content_element = content_element_bboxes
                        .iter()
                        .any(|bbox| region.bounding_box.ioa(bbox) > 0.5);
                    if !inside_content_element {
                        continue;
                    }
                }

                // Check if this orphan region is a formula
                // Create a new layout element for this orphan text
                // If it's a formula (label="formula"), create a Formula element, otherwise Text
                let element_type = if region.is_formula() {
                    LayoutElementType::Formula
                } else {
                    LayoutElementType::Text
                };

                let element = LayoutElement::new(
                    region.bounding_box.clone(),
                    element_type,
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
            let height = if let Some(img) = &result.rectified_img {
                img.height() as f32
            } else {
                result
                    .layout_elements
                    .iter()
                    .map(|e| e.bbox.y_max())
                    .fold(0.0f32, f32::max)
                    .max(1000.0)
            };
            Self::sort_layout_elements_enhanced(&mut result.layout_elements, width, height);
        }

        // Assign order indices regardless of sorting
        Self::assign_order_indices(&mut result.layout_elements);
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
            // Use the explicit is_e2e flag from the table analyzer to determine
            // the matching strategy, instead of inferring from confidence values.
            let has_detected_cells = table.detected_cell_bboxes.is_some();
            let e2e_like_cells = table.is_e2e && !has_detected_cells;

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

            // PaddleX: inject formula results into table OCR candidate pool with $...$
            // wrapping (table_contents_for_img). This lets formulas participate in normal
            // cell matching, so formula content appears in the correct table cells.
            for formula in formulas {
                let w = formula.bbox.x_max() - formula.bbox.x_min();
                let h = formula.bbox.y_max() - formula.bbox.y_min();
                if w <= 1.0 || h <= 1.0 {
                    continue;
                }
                if !Self::is_overlapping(&table_bbox, &formula.bbox, cfg) {
                    continue;
                }
                let latex = &formula.latex;
                let formatted = if latex.starts_with('$') && latex.ends_with('$') {
                    latex.clone()
                } else {
                    format!("${}$", latex)
                };
                let mut formula_region = TextRegion::new(formula.bbox.clone());
                formula_region.text = Some(formatted.into());
                formula_region.confidence = Some(1.0);
                ocr_candidates.push((OcrSource::Split, formula_region));
            }

            let structure_tokens = table.structure_tokens.clone();

            // Prefer PaddleX-style row-aware matching when structure tokens are available.
            // Use row-aware matching when cell detection was used (non-E2E mode).
            let mut td_to_cell_mapping: Option<Vec<Option<usize>>> = None;
            if !e2e_like_cells
                && let Some(tokens) = structure_tokens.as_deref()
                && !ocr_candidates.is_empty()
                && let Some((mapping, matched_candidate_indices)) =
                    Self::match_table_cells_with_structure_rows(
                        &mut table.cells,
                        tokens,
                        &ocr_candidates,
                        cfg.same_line_y_tolerance,
                        table.detected_cell_bboxes.as_deref(),
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

            // Formulas are now injected into the OCR candidate pool above,
            // so they participate in normal cell matching — no separate attach step needed.

            // Optional postprocess for checkbox-style tables:
            // normalize common OCR confusions like ü/L/X into ✓/✗ when the table
            // clearly exhibits both positive and negative marker patterns.
            Self::normalize_checkbox_symbols_in_table(&mut table.cells);

            // Regenerate HTML from structure tokens and stitched cell text.
            if let Some(tokens) = structure_tokens.as_deref() {
                let cell_texts: Vec<Option<String>> =
                    if let Some(ref td_mapping) = td_to_cell_mapping {
                        // Use the mapping from row-aware matching
                        td_mapping
                            .iter()
                            .map(|cell_idx| {
                                cell_idx
                                    .and_then(|idx| table.cells.get(idx))
                                    .and_then(|cell| cell.text.clone())
                            })
                            .collect()
                    } else {
                        // Fallback: cells may not be in the same order as structure_tokens.
                        // We need to create a mapping from cell bbox to its index, then
                        // iterate through tokens to collect texts in the correct order.
                        Self::collect_cell_texts_for_tokens(&table.cells, tokens)
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
            let ocr_bbox = &region.bounding_box;

            // Strategy 1: Center-point-in-cell with high IoA (strongest signal).
            // If the OCR box center falls inside a cell AND the box has high overlap
            // with that cell (IoA > 0.7), assign directly. The IoA check avoids
            // misassignment for boxes that straddle cell boundaries.
            let ocr_cx = (ocr_bbox.x_min() + ocr_bbox.x_max()) / 2.0;
            let ocr_cy = (ocr_bbox.y_min() + ocr_bbox.y_max()) / 2.0;
            let center_cell = cells.iter().enumerate().find(|(_, cell)| {
                ocr_cx >= cell.bbox.x_min()
                    && ocr_cx <= cell.bbox.x_max()
                    && ocr_cy >= cell.bbox.y_min()
                    && ocr_cy <= cell.bbox.y_max()
                    && ocr_bbox.ioa(&cell.bbox) > 0.7
            });

            if let Some((cell_idx, _)) = center_cell {
                cell_to_ocr.entry(cell_idx).or_default().push(candidate_idx);
                matched_candidate_indices.insert(candidate_idx);
                continue;
            }

            // Strategy 2+3: IoU + distance fallback
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

    fn normalize_checkbox_symbols_in_table(cells: &mut [TableCell]) {
        let mut has_positive_candidate = false;
        let mut has_negative_candidate = false;

        for cell in cells.iter() {
            let Some(text) = cell.text.as_deref() else {
                continue;
            };
            let trimmed = text.trim();
            if trimmed.chars().count() != 1 {
                continue;
            }
            match trimmed.chars().next().unwrap_or_default() {
                '✓' | 'ü' | 'Ü' | 'L' | '√' | '☑' => has_positive_candidate = true,
                '✗' | 'X' | 'x' | '✕' | '✖' | '☒' => has_negative_candidate = true,
                _ => {}
            }
        }

        for cell in cells.iter_mut() {
            let Some(text) = cell.text.clone() else {
                continue;
            };
            let trimmed = text.trim();
            if trimmed.chars().count() != 1 {
                continue;
            }
            let mapped = match trimmed.chars().next().unwrap_or_default() {
                // Safe positive normalization.
                'ü' | 'Ü' | '√' | '☑' => Some("✓"),
                // Ambiguous L is normalized only when the table appears checkbox-like.
                'L' if has_positive_candidate && has_negative_candidate => Some("✓"),
                // Safe negative normalization.
                '✕' | '✖' | '☒' => Some("✗"),
                // Ambiguous X/x are normalized only when the table appears checkbox-like.
                'X' | 'x' if has_positive_candidate && has_negative_candidate => Some("✗"),
                _ => None,
            };

            if let Some(symbol) = mapped {
                cell.text = Some(symbol.to_string());
            }
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
        cell_bboxes_override: Option<&[BoundingBox]>,
    ) -> Option<(Vec<Option<usize>>, std::collections::HashSet<usize>)> {
        if cells.is_empty() || structure_tokens.is_empty() || ocr_candidates.is_empty() {
            return None;
        }

        // --- Sort cells into rows ---
        // When detected bboxes are available we sort them (better spatial accuracy)
        // to pick the IoA bbox for OCR matching.  We also independently sort the
        // structure cells so that the td→cell text-assignment step uses a valid
        // index into `cells[]`.  Without this separation the det-bbox sort indices
        // are silently reused as structure-cell indices, misassigning OCR to wrong
        // cells whenever the two orderings differ.
        let (match_sorted_indices, cell_sorted_indices, match_row_flags) =
            if let Some(det_bboxes) = cell_bboxes_override {
                let temp_cells: Vec<TableCell> = det_bboxes
                    .iter()
                    .map(|b| TableCell::new(b.clone(), 0.5))
                    .collect();
                let (det_sorted, row_flags) =
                    Self::sort_table_cells_boxes(&temp_cells, row_y_tolerance);
                // Sort structure cells independently so their indices stay valid.
                let (cell_sorted, _) = Self::sort_table_cells_boxes(cells, row_y_tolerance);
                (det_sorted, cell_sorted, row_flags)
            } else {
                let (sorted, row_flags) = Self::sort_table_cells_boxes(cells, row_y_tolerance);
                // When there is no override the two index lists are identical.
                (sorted.clone(), sorted, row_flags)
            };

        if match_sorted_indices.is_empty() || match_row_flags.is_empty() {
            return None;
        }

        let mut row_start_index = Self::find_row_start_index(structure_tokens);
        if row_start_index.is_empty() {
            return None;
        }

        // Align match row flags with structure token row boundaries
        let mut match_aligned = Self::map_and_get_max(&match_row_flags, &row_start_index);
        match_aligned.push(match_sorted_indices.len());
        row_start_index.push(
            structure_tokens
                .iter()
                .filter(|t| Self::is_td_end_token(t))
                .count(),
        );

        // --- Per-row matching: cell → OCR (PaddleX style) ---
        // For each cell in the row, collect ALL OCR boxes with IoA > 0.7.
        // No cross-row deduplication — each row independently checks all OCR boxes,
        // matching PaddleX v2 behavior. The 0.7 IoA threshold naturally prevents
        // false cross-row matches.
        let mut all_matched: Vec<std::collections::HashMap<usize, Vec<usize>>> = Vec::new();

        for k in 0..match_aligned.len().saturating_sub(1) {
            let row_start = match_aligned[k].min(match_sorted_indices.len());
            let row_end = match_aligned[k + 1].min(match_sorted_indices.len());

            let mut matched: std::collections::HashMap<usize, Vec<usize>> =
                std::collections::HashMap::new();

            for (local_idx, &bbox_idx) in
                match_sorted_indices[row_start..row_end].iter().enumerate()
            {
                // Use detected bbox directly when available, else structure cell bbox
                let cell_box = cell_bboxes_override
                    .and_then(|bbs| bbs.get(bbox_idx))
                    .unwrap_or_else(|| &cells[bbox_idx.min(cells.len() - 1)].bbox);

                for (ocr_idx, (_, ocr_region)) in ocr_candidates.iter().enumerate() {
                    // IoA = intersection / OCR_area (PaddleX compute_inter > 0.7)
                    let ioa = ocr_region.bounding_box.ioa(cell_box);
                    if ioa > 0.7 {
                        matched.entry(local_idx).or_default().push(ocr_idx);
                    }
                }
            }

            all_matched.push(matched);
        }

        // --- Build td_to_cell_mapping by iterating structure tokens ---
        // table.cells maps exactly 1:1 with td tokens in structure order.
        let mut td_to_cell_mapping: Vec<Option<usize>> = Vec::new();
        let mut matched_candidate_indices: std::collections::HashSet<usize> =
            std::collections::HashSet::new();

        let mut td_index = 0usize;
        let mut td_count = 0usize;
        let mut matched_row_idx = 0usize;

        for tag in structure_tokens {
            if tag == "<tr>" {
                td_index = 0; // Reset cell index at row start
                continue;
            }
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

            // Map td position to the original cell index via sorted ordering.
            // match_aligned[matched_row_idx] + td_index gives the position in the
            // sorted cell list.  Use cell_sorted_indices (indices into cells[])
            // rather than match_sorted_indices (which may be indices into det_bboxes
            // when cell_bboxes_override is active).
            let mapped_cell_idx = match_aligned
                .get(matched_row_idx)
                .copied()
                .and_then(|row_start| {
                    let sorted_pos = row_start + td_index;
                    cell_sorted_indices.get(sorted_pos).copied()
                })
                .filter(|&idx| idx < cells.len());

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
            }
        }

        if td_to_cell_mapping.is_empty() {
            None
        } else {
            Some((td_to_cell_mapping, matched_candidate_indices))
        }
    }

    /// Collects cell texts in the order they appear in structure tokens.
    ///
    /// Uses grid-based `(row, col)` matching when cells have grid info, which
    /// correctly handles rowspan/colspan cases where cells.len() != td_count.
    /// Falls back to index-based matching when grid info is unavailable.
    fn collect_cell_texts_for_tokens(
        cells: &[TableCell],
        tokens: &[String],
    ) -> Vec<Option<String>> {
        if cells.is_empty() {
            return Vec::new();
        }

        // Parse grid positions for each <td> token
        let token_grid = parse_cell_grid_info(tokens);
        let td_count = token_grid.len();

        // Build a lookup from (row, col) -> cell index for cells that have grid info
        let mut grid_to_cell: std::collections::HashMap<(usize, usize), usize> =
            std::collections::HashMap::new();
        let mut has_grid_info = false;

        for (cell_idx, cell) in cells.iter().enumerate() {
            if let (Some(row), Some(col)) = (cell.row, cell.col) {
                grid_to_cell.insert((row, col), cell_idx);
                has_grid_info = true;
            }
        }

        if has_grid_info {
            // Grid-based matching: match tokens to cells by (row, col) position
            token_grid
                .iter()
                .map(|gi| {
                    grid_to_cell
                        .get(&(gi.row, gi.col))
                        .and_then(|&idx| cells.get(idx))
                        .and_then(|cell| cell.text.clone())
                })
                .collect()
        } else {
            // Fallback: cells don't have grid info, use index-based matching
            (0..td_count)
                .map(|i| cells.get(i).and_then(|cell| cell.text.clone()))
                .collect()
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

    /// Marks small inline formulas to be absorbed into the text flow.
    ///
    /// PaddleX: Small formula elements should be absorbed into the text flow,
    /// not kept as separate layout elements.
    ///
    /// This function:
    /// 1. Finds small formula elements that should be inline (not display formulas)
    /// 2. Clears their text and order_index so the formula element won't be rendered
    /// 3. The corresponding TextRegion with label="formula" (already created in structure.rs)
    ///    will become an orphan and be handled with proper $...$ wrapping
    fn inject_inline_formulas(
        elements: &mut [LayoutElement],
        _text_regions: &mut Vec<TextRegion>,
        _cfg: &StitchConfig,
    ) {
        use oar_ocr_core::domain::structure::LayoutElementType;

        let mut inline_formula_indices: Vec<usize> = Vec::new();

        // Size threshold: formulas smaller than 80k pixels² are likely inline
        const INLINE_FORMULA_MAX_AREA: f32 = 80000.0;

        for (idx, element) in elements.iter().enumerate() {
            if element.element_type != LayoutElementType::Formula {
                continue;
            }

            // Only process formulas that have text
            let formula_text = if let Some(text) = &element.text {
                if !text.is_empty() {
                    text
                } else {
                    continue;
                }
            } else {
                continue;
            };

            let formula_area = element.bbox.area();
            tracing::debug!(
                "Formula idx {}: area={:.1}, text={}",
                idx,
                formula_area,
                formula_text
            );

            // Small formulas are treated as inline
            if formula_area < INLINE_FORMULA_MAX_AREA {
                inline_formula_indices.push(idx);
                tracing::debug!(
                    "Marking formula idx {} as inline (area {:.1} < {})",
                    idx,
                    formula_area,
                    INLINE_FORMULA_MAX_AREA
                );
            }
        }

        // Clear inline formula elements so they won't be rendered separately
        for idx in &inline_formula_indices {
            if let Some(element) = elements.get_mut(*idx) {
                tracing::debug!(
                    "Clearing inline formula idx {} to use TextRegion with label=formula",
                    idx
                );
                element.text = None;
                element.order_index = None;
            }
        }

        if !inline_formula_indices.is_empty() {
            tracing::debug!("Marked {} formulas as inline", inline_formula_indices.len());
        }
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

                // Debug: log all text regions being joined
                for (region, text) in &element_texts {
                    tracing::debug!("  - region with label={:?}, text={:?}", region.label, text);
                }

                // Compute seg metadata (seg_start_x, seg_end_x, num_lines) for get_seg_flag.
                // Sort a copy to find first/last spans and count lines.
                let mut sorted_for_meta = element_texts.clone();
                sorted_for_meta.sort_by(|(r1, _), (r2, _)| {
                    r1.bounding_box
                        .center()
                        .y
                        .partial_cmp(&r2.bounding_box.center().y)
                        .unwrap_or(Ordering::Equal)
                });
                let mut lines = Vec::new();
                let mut current_line = Vec::new();
                for item in std::mem::take(&mut sorted_for_meta) {
                    if current_line.is_empty() {
                        current_line.push(item);
                    } else {
                        let first_in_line = &current_line[0].0.bounding_box;
                        if Self::is_same_text_line_bbox(first_in_line, &item.0.bounding_box, cfg) {
                            current_line.push(item);
                        } else {
                            current_line.sort_by(|(r1, _), (r2, _)| {
                                r1.bounding_box
                                    .center()
                                    .x
                                    .partial_cmp(&r2.bounding_box.center().x)
                                    .unwrap_or(Ordering::Equal)
                            });
                            lines.push(current_line);
                            current_line = vec![item];
                        }
                    }
                }
                if !current_line.is_empty() {
                    current_line.sort_by(|(r1, _), (r2, _)| {
                        r1.bounding_box
                            .center()
                            .x
                            .partial_cmp(&r2.bounding_box.center().x)
                            .unwrap_or(Ordering::Equal)
                    });
                    lines.push(current_line);
                }
                for mut line in lines {
                    sorted_for_meta.append(&mut line);
                }

                // seg_start_x: first span's left edge (PaddleX: line[0].spans[0].box[0])
                element.seg_start_x = Some(sorted_for_meta[0].0.bounding_box.x_min());
                // seg_end_x: last span's right edge (PaddleX: line[-1].spans[-1].box[2])
                element.seg_end_x = Some(sorted_for_meta.last().unwrap().0.bounding_box.x_max());

                // Count distinct lines (Y-groups)
                let mut num_lines = 1u32;
                let mut prev_bbox = &sorted_for_meta[0].0.bounding_box;
                for (region, _) in &sorted_for_meta[1..] {
                    if !Self::is_same_text_line_bbox(prev_bbox, &region.bounding_box, cfg) {
                        num_lines += 1;
                        prev_bbox = &region.bounding_box;
                    }
                }
                element.num_lines = Some(num_lines);
            }

            Self::sort_and_join_texts(&mut element_texts, Some(&element.bbox), cfg, |joined| {
                element.text = Some(joined);
            });
        }
    }

    /// Fills formula layout elements with LaTeX content from formula recognition results.
    ///
    /// This ensures formula elements have correct content even if OCR matching
    /// thresholds prevented proper association.
    fn fill_formula_elements(
        elements: &mut [LayoutElement],
        formulas: &[FormulaResult],
        _cfg: &StitchConfig,
    ) {
        for element in elements.iter_mut() {
            if element.element_type != LayoutElementType::Formula {
                continue;
            }

            // Skip if element already has content from OCR matching
            if element.text.is_some() {
                continue;
            }

            // Find the best matching formula result by bidirectional IoA.
            // IoA (intersection / self_area) is much more permissive than IoU for
            // size-mismatched bboxes. PaddleX uses simple intersection overlap (>3px).
            let mut best_formula: Option<&FormulaResult> = None;
            let mut best_score = 0.0f32;

            for formula in formulas {
                let ioa_element = element.bbox.ioa(&formula.bbox);
                let ioa_formula = formula.bbox.ioa(&element.bbox);
                let score = ioa_element.max(ioa_formula);
                if score > best_score {
                    best_score = score;
                    best_formula = Some(formula);
                }
            }

            // Fallback: if no IoA match, try center-containment matching.
            // Find formula whose center is within the element bbox (or vice versa).
            if best_score < 0.05 {
                let elem_center = element.bbox.center();
                let mut best_dist = f32::MAX;

                for formula in formulas {
                    let fc = formula.bbox.center();
                    let fc_inside = fc.x >= element.bbox.x_min()
                        && fc.x <= element.bbox.x_max()
                        && fc.y >= element.bbox.y_min()
                        && fc.y <= element.bbox.y_max();
                    let ec_inside = elem_center.x >= formula.bbox.x_min()
                        && elem_center.x <= formula.bbox.x_max()
                        && elem_center.y >= formula.bbox.y_min()
                        && elem_center.y <= formula.bbox.y_max();

                    if fc_inside || ec_inside {
                        let dx = fc.x - elem_center.x;
                        let dy = fc.y - elem_center.y;
                        let dist = dx * dx + dy * dy;
                        if dist < best_dist {
                            best_dist = dist;
                            best_formula = Some(formula);
                            best_score = 0.05;
                        }
                    }
                }
            }

            if best_score >= 0.05
                && let Some(formula) = best_formula
            {
                element.text = Some(formula.latex.clone());
            }
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

    /// Checks whether two OCR spans should be grouped into the same visual line.
    ///
    /// Primary signal follows PaddleX-style line-height overlap:
    /// vertical_overlap / min(height1, height2) >= threshold.
    /// A small adaptive center-Y fallback is kept for robustness on noisy boxes.
    fn is_same_text_line_bbox(
        bbox1: &BoundingBox,
        bbox2: &BoundingBox,
        cfg: &StitchConfig,
    ) -> bool {
        let h1 = (bbox1.y_max() - bbox1.y_min()).max(1.0);
        let h2 = (bbox2.y_max() - bbox2.y_min()).max(1.0);
        let inter_h =
            (bbox1.y_max().min(bbox2.y_max()) - bbox1.y_min().max(bbox2.y_min())).max(0.0);
        let overlap_ratio = inter_h / h1.min(h2);
        if overlap_ratio >= cfg.line_height_iou_threshold {
            return true;
        }

        let adaptive_tol = (h1.min(h2) * 0.5).max(1.0);
        let center_delta = (bbox1.center().y - bbox2.center().y).abs();
        center_delta <= adaptive_tol.max(cfg.same_line_y_tolerance * 0.25)
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
            r1.bounding_box
                .center()
                .y
                .partial_cmp(&r2.bounding_box.center().y)
                .unwrap_or(Ordering::Equal)
        });
        let mut lines = Vec::new();
        let mut current_line = Vec::new();
        for item in std::mem::take(texts) {
            if current_line.is_empty() {
                current_line.push(item);
            } else {
                let first_in_line = &current_line[0].0.bounding_box;
                if Self::is_same_text_line_bbox(first_in_line, &item.0.bounding_box, cfg) {
                    current_line.push(item);
                } else {
                    current_line.sort_by(|(r1, _), (r2, _)| {
                        r1.bounding_box
                            .center()
                            .x
                            .partial_cmp(&r2.bounding_box.center().x)
                            .unwrap_or(Ordering::Equal)
                    });
                    lines.push(current_line);
                    current_line = vec![item];
                }
            }
        }
        if !current_line.is_empty() {
            current_line.sort_by(|(r1, _), (r2, _)| {
                r1.bounding_box
                    .center()
                    .x
                    .partial_cmp(&r2.bounding_box.center().x)
                    .unwrap_or(Ordering::Equal)
            });
            lines.push(current_line);
        }
        for mut line in lines {
            texts.append(&mut line);
        }

        // Smart text joining following format_line logic:
        // - Texts on the same line are joined directly (no separator)
        // - A space is added only if the previous text ends with an English letter
        // - Newlines are added conditionally based on geometric gap (paragraph break detection)
        let mut result = String::new();
        let mut prev_region: Option<&TextRegion> = None;

        tracing::debug!(
            "sort_and_join_texts: processing {} text regions",
            texts.len()
        );

        for (region, text) in texts.iter() {
            if text.is_empty() {
                continue;
            }

            if let Some(last_region) = prev_region {
                if !Self::is_same_text_line_bbox(
                    &last_region.bounding_box,
                    &region.bounding_box,
                    cfg,
                ) {
                    // New visual line detected.
                    // Decide whether to insert '\n' (hard break) or ' ' (soft break/wrap).
                    let mut add_newline = false;
                    let mut is_line_wrap = false;

                    if let Some(container) = container_bbox {
                        let container_width = container.x_max() - container.x_min();
                        let right_gap = container.x_max() - last_region.bounding_box.x_max();
                        let tail_char = last_non_whitespace_char(&result);
                        let ends_with_non_break_punct =
                            tail_char.is_some_and(is_non_break_line_end_punctuation);
                        // PaddleX: English lines use a larger right-gap threshold.
                        let paragraph_gap_ratio =
                            if tail_char.is_some_and(|c| c.is_ascii_alphabetic()) {
                                0.5
                            } else {
                                0.3
                            };

                        if !ends_with_non_break_punct
                            && right_gap > container_width * paragraph_gap_ratio
                        {
                            // Previous line ended far from the right edge → paragraph break.
                            add_newline = true;
                        } else {
                            // Previous line extends close to the right edge → line wrap.
                            is_line_wrap = true;
                        }
                    }

                    // Dehyphenation: only strip trailing hyphen when the previous line
                    // is a wrapped line (extends close to container right edge).
                    // This preserves hyphens in compound words like "real-time",
                    // "end-to-end", "one-to-many" that end short lines.
                    // Matches PaddleX format_line behavior where hyphens are stripped
                    // at line-wrap boundaries.
                    let prev_ends_hyphen = result.ends_with('-');
                    if prev_ends_hyphen && is_line_wrap {
                        // Line wraps at hyphen → word-break hyphen, remove it
                        result.pop();
                        // Don't add any separator - words should be joined
                    } else if add_newline {
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
                } else {
                    // Same visual line - join with smart spacing
                    // PaddleX format_line: add space after English letters OR after formulas
                    let needs_spacing = if let Some(last_char) = result.chars().last()
                        && last_char != '\n'
                        && needs_space_after(last_char)
                    {
                        true
                    } else {
                        // PaddleX: add space after formula when next content is on same line
                        last_region.is_formula()
                    };

                    if needs_spacing {
                        result.push(' ');
                    }
                }
            }

            // PaddleX: formula spans are wrapped with $...$ delimiters
            // Inline formulas (mixed with text on same line): $formula$
            // Display formulas (standalone line): $$formula$$ (display math)
            let is_formula = region.is_formula();
            let text_to_add = if is_formula {
                // Don't double-wrap if formula model already added delimiters
                let already_wrapped =
                    text.starts_with('$') || text.starts_with("\\(") || text.starts_with("\\[");
                if already_wrapped {
                    text.to_string()
                } else {
                    // Check if this is a display formula (starts a new line with no other content yet on this line)
                    // Display formulas typically appear at the start of a line after a newline
                    let is_display = result.is_empty() || result.ends_with('\n');

                    if is_display {
                        // Display formula: $$...$$
                        format!("$${}$$", text)
                    } else {
                        // Inline formula: $...$
                        format!("${}$", text)
                    }
                }
            } else {
                text.to_string()
            };

            result.push_str(&text_to_add);
            prev_region = Some(region);
        }

        // Trim trailing whitespace
        let joined = result.trim_end().to_string();
        update_fn(joined);
    }

    /// Sorts layout elements using the enhanced xycut_enhanced algorithm.
    ///
    /// Uses cross-layout detection, direction-aware XY-cut, overlapping box shrinking,
    /// weighted distance insertion, and child block association for accurate reading order.
    fn sort_layout_elements_enhanced(
        elements: &mut Vec<LayoutElement>,
        page_width: f32,
        page_height: f32,
    ) {
        use oar_ocr_core::processors::layout_sorting::{SortableElement, sort_layout_enhanced};

        if elements.is_empty() {
            return;
        }

        let sortable_elements: Vec<_> = elements
            .iter()
            .map(|e| SortableElement {
                bbox: e.bbox.clone(),
                element_type: e.element_type,
                num_lines: e.num_lines,
            })
            .collect();

        let sorted_indices = sort_layout_enhanced(&sortable_elements, page_width, page_height);
        if sorted_indices.len() != elements.len() {
            return;
        }

        let sorted_elements: Vec<_> = sorted_indices
            .into_iter()
            .map(|idx| elements[idx].clone())
            .collect();
        *elements = sorted_elements;
    }

    /// Sorts layout elements using the XY-cut algorithm (legacy fallback).
    #[allow(dead_code)]
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

fn last_non_whitespace_char(text: &str) -> Option<char> {
    text.chars().rev().find(|c| !c.is_whitespace())
}

/// Punctuation that should not trigger hard paragraph breaks across line wraps.
fn is_non_break_line_end_punctuation(c: char) -> bool {
    matches!(c, ',' | '，' | '、' | ';' | '；' | ':' | '：')
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
            label: None,
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
            label: None,
        };
        let r2 = TextRegion {
            bounding_box: b2.clone(),
            dt_poly: Some(b2.clone()),
            rec_poly: Some(b2),
            text: Some("B".into()),
            confidence: Some(0.9),
            orientation_angle: None,
            word_boxes: None,
            label: None,
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
    fn test_sort_and_join_texts_english_line_uses_larger_paragraph_gap_threshold() {
        let r1 = make_region(BoundingBox::from_coords(0.0, 0.0, 60.0, 10.0), "Line");
        let r2 = make_region(BoundingBox::from_coords(0.0, 20.0, 40.0, 30.0), "next");
        let mut texts = vec![(&r1, "Line"), (&r2, "next")];
        let cfg = StitchConfig::default();
        let container = BoundingBox::from_coords(0.0, 0.0, 100.0, 40.0);
        let mut joined = String::new();
        ResultStitcher::sort_and_join_texts(&mut texts, Some(&container), &cfg, |j| joined = j);
        assert_eq!(joined, "Line next");
    }

    #[test]
    fn test_sort_and_join_texts_non_english_tail_keeps_original_paragraph_gap_threshold() {
        let r1 = make_region(BoundingBox::from_coords(0.0, 0.0, 60.0, 10.0), "2024");
        let r2 = make_region(BoundingBox::from_coords(0.0, 20.0, 40.0, 30.0), "next");
        let mut texts = vec![(&r1, "2024"), (&r2, "next")];
        let cfg = StitchConfig::default();
        let container = BoundingBox::from_coords(0.0, 0.0, 100.0, 40.0);
        let mut joined = String::new();
        ResultStitcher::sort_and_join_texts(&mut texts, Some(&container), &cfg, |j| joined = j);
        assert_eq!(joined, "2024\nnext");
    }

    #[test]
    fn test_sort_and_join_texts_non_break_punctuation_suppresses_newline() {
        let r1 = make_region(BoundingBox::from_coords(0.0, 0.0, 20.0, 10.0), "Note:");
        let r2 = make_region(BoundingBox::from_coords(0.0, 20.0, 40.0, 30.0), "next");
        let mut texts = vec![(&r1, "Note:"), (&r2, "next")];
        let cfg = StitchConfig::default();
        let container = BoundingBox::from_coords(0.0, 0.0, 100.0, 40.0);
        let mut joined = String::new();
        ResultStitcher::sort_and_join_texts(&mut texts, Some(&container), &cfg, |j| joined = j);
        assert_eq!(joined, "Note:next");
    }

    #[test]
    fn test_normalize_checkbox_symbols_in_table_checkbox_like() {
        let mut cells = vec![
            TableCell::new(BoundingBox::from_coords(0.0, 0.0, 10.0, 10.0), 1.0).with_text("ü"),
            TableCell::new(BoundingBox::from_coords(10.0, 0.0, 20.0, 10.0), 1.0).with_text("X"),
            TableCell::new(BoundingBox::from_coords(20.0, 0.0, 30.0, 10.0), 1.0).with_text("L"),
        ];

        ResultStitcher::normalize_checkbox_symbols_in_table(&mut cells);

        assert_eq!(cells[0].text.as_deref(), Some("✓"));
        assert_eq!(cells[1].text.as_deref(), Some("✗"));
        assert_eq!(cells[2].text.as_deref(), Some("✓"));
    }

    #[test]
    fn test_normalize_checkbox_symbols_in_table_keeps_ambiguous_when_not_checkbox_like() {
        let mut cells = vec![
            TableCell::new(BoundingBox::from_coords(0.0, 0.0, 10.0, 10.0), 1.0).with_text("L"),
            TableCell::new(BoundingBox::from_coords(10.0, 0.0, 20.0, 10.0), 1.0).with_text("A"),
        ];

        ResultStitcher::normalize_checkbox_symbols_in_table(&mut cells);

        assert_eq!(cells[0].text.as_deref(), Some("L"));
        assert_eq!(cells[1].text.as_deref(), Some("A"));
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
            None,
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
