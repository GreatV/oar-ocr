//! # Stage Definition: Table Analysis
//!
//! This service is considered "Done" when it fulfills the following contract:
//!
//! - **Inputs**: Full page `image::RgbImage`, detected `LayoutElement`s, `FormulaResult`s, and `TextRegion`s.
//! - **Outputs**: `Vec<TableResult>` containing parsed grid structure, cells mapped to page coordinates, and HTML.
//! - **Logging**: Traces table classification, rotation, and whether E2E or cell-detection mode was selected.
//! - **Error Behavior**: Returns `OCRError` for fatal adapter failures, but emits stub `TableResult` for soft recognition failures to maintain alignment with layout boxes.
//! - **Invariants**:
//!     - Output cell coordinates are always transformed back to the original page space.
//!     - Table results include both logical grid info (row/col) and visual bounding boxes.
//!     - OCR results overlapping table regions are used to refine cell bounding boxes.

use crate::core::OCRError;
use crate::core::registry::{DynModelAdapter, DynTaskInput};
use crate::core::traits::task::ImageTaskInput;
use crate::domain::structure::{
    FormulaResult, LayoutElement, LayoutElementType, TableCell, TableResult, TableType,
};
use crate::oarocr::TextRegion;
use crate::processors::BoundingBox;
use crate::utils::BBoxCrop;
use std::sync::Arc;

/// Configuration for creating a TableAnalyzer.
#[derive(Debug, Clone, Default)]
pub(crate) struct TableAnalyzerConfig {
    pub table_classification_adapter: Option<Arc<dyn DynModelAdapter>>,
    pub table_orientation_adapter: Option<Arc<dyn DynModelAdapter>>,
    pub table_structure_recognition_adapter: Option<Arc<dyn DynModelAdapter>>,
    pub wired_table_structure_adapter: Option<Arc<dyn DynModelAdapter>>,
    pub wireless_table_structure_adapter: Option<Arc<dyn DynModelAdapter>>,
    pub table_cell_detection_adapter: Option<Arc<dyn DynModelAdapter>>,
    pub wired_table_cell_adapter: Option<Arc<dyn DynModelAdapter>>,
    pub wireless_table_cell_adapter: Option<Arc<dyn DynModelAdapter>>,
    pub use_e2e_wired_table_rec: bool,
    pub use_e2e_wireless_table_rec: bool,
}

#[derive(Debug, Clone)]
pub(crate) struct TableAnalyzer {
    table_classification_adapter: Option<Arc<dyn DynModelAdapter>>,
    table_orientation_adapter: Option<Arc<dyn DynModelAdapter>>,

    table_structure_recognition_adapter: Option<Arc<dyn DynModelAdapter>>,
    wired_table_structure_adapter: Option<Arc<dyn DynModelAdapter>>,
    wireless_table_structure_adapter: Option<Arc<dyn DynModelAdapter>>,

    table_cell_detection_adapter: Option<Arc<dyn DynModelAdapter>>,
    wired_table_cell_adapter: Option<Arc<dyn DynModelAdapter>>,
    wireless_table_cell_adapter: Option<Arc<dyn DynModelAdapter>>,

    use_e2e_wired_table_rec: bool,
    use_e2e_wireless_table_rec: bool,
}

impl TableAnalyzer {
    pub(crate) fn new(config: TableAnalyzerConfig) -> Self {
        Self {
            table_classification_adapter: config.table_classification_adapter,
            table_orientation_adapter: config.table_orientation_adapter,
            table_structure_recognition_adapter: config.table_structure_recognition_adapter,
            wired_table_structure_adapter: config.wired_table_structure_adapter,
            wireless_table_structure_adapter: config.wireless_table_structure_adapter,
            table_cell_detection_adapter: config.table_cell_detection_adapter,
            wired_table_cell_adapter: config.wired_table_cell_adapter,
            wireless_table_cell_adapter: config.wireless_table_cell_adapter,
            use_e2e_wired_table_rec: config.use_e2e_wired_table_rec,
            use_e2e_wireless_table_rec: config.use_e2e_wireless_table_rec,
        }
    }

    pub(crate) fn analyze_tables(
        &self,
        page_image: &image::RgbImage,
        layout_elements: &[LayoutElement],
        formulas: &[FormulaResult],
        text_regions: &[TextRegion],
    ) -> Result<Vec<TableResult>, OCRError> {
        let table_regions: Vec<_> = layout_elements
            .iter()
            .filter(|e| e.element_type == LayoutElementType::Table)
            .collect();

        let mut tables = Vec::new();
        for (idx, table_element) in table_regions.iter().enumerate() {
            if let Some(table_result) = self.analyze_single_table(
                idx,
                table_element,
                page_image,
                layout_elements,
                formulas,
                text_regions,
            )? {
                tables.push(table_result);
            }
        }

        Ok(tables)
    }

    fn analyze_single_table(
        &self,
        idx: usize,
        table_element: &LayoutElement,
        page_image: &image::RgbImage,
        layout_elements: &[LayoutElement],
        formulas: &[FormulaResult],
        text_regions: &[TextRegion],
    ) -> Result<Option<TableResult>, OCRError> {
        let table_bbox = &table_element.bbox;

        let cropped_table = match BBoxCrop::crop_bounding_box(page_image, table_bbox) {
            Ok(img) => {
                tracing::debug!(
                    target: "structure",
                    table_index = idx,
                    bbox = ?[
                        table_bbox.x_min(),
                        table_bbox.y_min(),
                        table_bbox.x_max(),
                        table_bbox.y_max()
                    ],
                    crop_size = ?(img.width(), img.height()),
                    "Cropped table region"
                );
                img
            }
            Err(e) => {
                tracing::warn!(
                    target: "structure",
                    table_index = idx,
                    error = %e,
                    "Failed to crop table region; skipping"
                );
                return Ok(None);
            }
        };

        // Use floor() to align with BBoxCrop which truncates coordinates to u32.
        let table_x_offset = table_bbox.x_min().max(0.0).floor();
        let table_y_offset = table_bbox.y_min().max(0.0).floor();

        let (table_for_recognition, table_rotation) =
            if let Some(ref orientation_adapter) = self.table_orientation_adapter {
                match crate::oarocr::preprocess::correct_image_orientation(
                    cropped_table.clone(),
                    orientation_adapter,
                ) {
                    Ok((rotated, correction)) => {
                        if let Some(c) = correction
                            && c.angle.abs() > 1.0
                        {
                            tracing::debug!(
                                target: "structure",
                                table_index = idx,
                                rotation_angle = c.angle,
                                "Rotating table to correct orientation"
                            );
                        }
                        (rotated, correction)
                    }
                    Err(e) => {
                        tracing::warn!(
                            target: "structure",
                            table_index = idx,
                            error = %e,
                            "Table orientation detection failed; proceeding without rotation"
                        );
                        (cropped_table.clone(), None)
                    }
                }
            } else {
                (cropped_table.clone(), None)
            };

        let (table_type, classification_confidence) = if let Some(ref cls_adapter) =
            self.table_classification_adapter
        {
            let input =
                DynTaskInput::from_images(ImageTaskInput::new(vec![table_for_recognition.clone()]));
            if let Ok(cls_output) = cls_adapter.execute_dyn(input)
                && let Ok(cls_result) = cls_output.into_table_classification()
                && let Some(classifications) = cls_result.classifications.first()
                && let Some(top_cls) = classifications.first()
            {
                let table_type = match top_cls.label.to_lowercase().as_str() {
                    "wired" | "wired_table" => TableType::Wired,
                    "wireless" | "wireless_table" => TableType::Wireless,
                    _ => TableType::Unknown,
                };
                (table_type, Some(top_cls.score))
            } else {
                (TableType::Unknown, None)
            }
        } else {
            (TableType::Unknown, None)
        };

        let use_e2e_mode = match table_type {
            TableType::Wired => self.use_e2e_wired_table_rec,
            TableType::Wireless => self.use_e2e_wireless_table_rec,
            TableType::Unknown => self.use_e2e_wireless_table_rec,
        };

        let structure_adapter: Option<&Arc<dyn DynModelAdapter>> = match table_type {
            TableType::Wired => self
                .wired_table_structure_adapter
                .as_ref()
                .or(self.table_structure_recognition_adapter.as_ref()),
            TableType::Wireless => self
                .wireless_table_structure_adapter
                .as_ref()
                .or(self.table_structure_recognition_adapter.as_ref()),
            TableType::Unknown => self
                .table_structure_recognition_adapter
                .as_ref()
                .or(self.wireless_table_structure_adapter.as_ref())
                .or(self.wired_table_structure_adapter.as_ref()),
        };

        let cell_adapter: Option<&Arc<dyn DynModelAdapter>> = if use_e2e_mode {
            tracing::info!(
                target: "structure",
                table_index = idx,
                table_type = ?table_type,
                "Using E2E mode: skipping cell detection"
            );
            None
        } else {
            tracing::info!(
                target: "structure",
                table_index = idx,
                table_type = ?table_type,
                "Using cell detection mode (E2E disabled)"
            );
            match table_type {
                TableType::Wired => self
                    .wired_table_cell_adapter
                    .as_ref()
                    .or(self.table_cell_detection_adapter.as_ref())
                    .or(self.wireless_table_cell_adapter.as_ref()),
                TableType::Wireless => self
                    .wireless_table_cell_adapter
                    .as_ref()
                    .or(self.table_cell_detection_adapter.as_ref())
                    .or(self.wired_table_cell_adapter.as_ref()),
                TableType::Unknown => self
                    .table_cell_detection_adapter
                    .as_ref()
                    .or(self.wired_table_cell_adapter.as_ref())
                    .or(self.wireless_table_cell_adapter.as_ref()),
            }
        };

        let table_output = match structure_adapter {
            Some(adapter) => {
                let input = DynTaskInput::from_images(ImageTaskInput::new(vec![
                    table_for_recognition.clone(),
                ]));
                match adapter.execute_dyn(input) {
                    Ok(output) => output,
                    Err(e) => {
                        tracing::warn!(
                            target: "structure",
                            table_index = idx,
                            table_type = ?table_type,
                            error = %e,
                            "Structure adapter failed; adding stub result"
                        );
                        let mut table_result = TableResult::new(table_bbox.clone(), table_type);
                        if let Some(conf) = classification_confidence {
                            table_result = table_result.with_classification_confidence(conf);
                        }
                        return Ok(Some(table_result));
                    }
                }
            }
            None => {
                tracing::warn!(
                    target: "structure",
                    table_index = idx,
                    table_type = ?table_type,
                    "No structure adapter available; adding stub result"
                );
                let mut table_result = TableResult::new(table_bbox.clone(), table_type);
                if let Some(conf) = classification_confidence {
                    table_result = table_result.with_classification_confidence(conf);
                }
                return Ok(Some(table_result));
            }
        };

        let structure_parsed = table_output.into_table_structure_recognition().ok();
        let has_valid_structure = structure_parsed
            .as_ref()
            .map(|r| !r.structures.is_empty())
            .unwrap_or(false);

        if !has_valid_structure {
            let mut table_result = TableResult::new(table_bbox.clone(), table_type);
            if let Some(conf) = classification_confidence {
                table_result = table_result.with_classification_confidence(conf);
            }
            return Ok(Some(table_result));
        }

        let table_result = structure_parsed.unwrap();
        let Some((structure, bboxes, structure_score)) = table_result
            .structures
            .first()
            .zip(table_result.bboxes.first())
            .zip(table_result.structure_scores.first())
            .map(|((s, b), sc)| (s, b, sc))
        else {
            tracing::warn!(
                "Table {}: Structure recognition returned mismatched data (structures: {}, bboxes: {}, scores: {}). Adding stub result.",
                idx,
                table_result.structures.len(),
                table_result.bboxes.len(),
                table_result.structure_scores.len()
            );
            let mut stub_result = TableResult::new(table_bbox.clone(), table_type);
            if let Some(conf) = classification_confidence {
                stub_result = stub_result.with_classification_confidence(conf);
            }
            return Ok(Some(stub_result));
        };

        let grid_info = crate::processors::parse_cell_grid_info(structure);

        let mut cells: Vec<TableCell> = bboxes
            .iter()
            .enumerate()
            .map(|(cell_idx, bbox_coords)| {
                let mut bbox_crop = if bbox_coords.len() >= 8 {
                    let xs = [
                        bbox_coords[0],
                        bbox_coords[2],
                        bbox_coords[4],
                        bbox_coords[6],
                    ];
                    let ys = [
                        bbox_coords[1],
                        bbox_coords[3],
                        bbox_coords[5],
                        bbox_coords[7],
                    ];
                    let x_min = xs.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
                    let y_min = ys.iter().fold(f32::INFINITY, |acc, &y| acc.min(y));
                    let x_max = xs.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
                    let y_max = ys.iter().fold(f32::NEG_INFINITY, |acc, &y| acc.max(y));
                    BoundingBox::from_coords(x_min, y_min, x_max, y_max)
                } else if bbox_coords.len() >= 4 {
                    BoundingBox::from_coords(
                        bbox_coords[0],
                        bbox_coords[1],
                        bbox_coords[2],
                        bbox_coords[3],
                    )
                } else {
                    BoundingBox::from_coords(0.0, 0.0, 0.0, 0.0)
                };

                if let Some(rot) = table_rotation
                    && rot.angle.abs() > 1.0
                {
                    bbox_crop = bbox_crop.rotate_back_to_original(
                        rot.angle,
                        rot.rotated_width,
                        rot.rotated_height,
                    );
                }

                let bbox = bbox_crop.translate(table_x_offset, table_y_offset);

                let mut cell = TableCell::new(bbox, 1.0);
                if let Some(info) = grid_info.get(cell_idx) {
                    cell = cell
                        .with_position(info.row, info.col)
                        .with_span(info.row_span, info.col_span);
                }
                cell
            })
            .collect();

        if let Some(cell_detection_adapter) = cell_adapter {
            let cell_input =
                DynTaskInput::from_images(ImageTaskInput::new(vec![table_for_recognition.clone()]));
            if let Ok(cell_output) = cell_detection_adapter.execute_dyn(cell_input)
                && let Ok(cell_result) = cell_output.into_table_cell_detection()
                && let Some(detected_cells) = cell_result.cells.first()
                && !detected_cells.is_empty()
            {
                let structure_bboxes_crop: Vec<_> = cells
                    .iter()
                    .map(|c| {
                        BoundingBox::from_coords(
                            c.bbox.x_min() - table_x_offset,
                            c.bbox.y_min() - table_y_offset,
                            c.bbox.x_max() - table_x_offset,
                            c.bbox.y_max() - table_y_offset,
                        )
                    })
                    .collect();

                let detected_bboxes: Vec<_> = detected_cells
                    .iter()
                    .map(|c| {
                        let mut bbox = c.bbox.clone();
                        if let Some(rot) = table_rotation
                            && rot.angle.abs() > 1.0
                        {
                            bbox = bbox.rotate_back_to_original(
                                rot.angle,
                                rot.rotated_width,
                                rot.rotated_height,
                            );
                        }
                        bbox
                    })
                    .collect();
                let detected_scores: Vec<f32> = detected_cells.iter().map(|c| c.score).collect();

                let mut ocr_boxes_crop: Vec<BoundingBox> = Vec::new();
                for region in text_regions {
                    let b = &region.bounding_box;
                    if b.x_min() >= table_bbox.x_min()
                        && b.y_min() >= table_bbox.y_min()
                        && b.x_max() <= table_bbox.x_max()
                        && b.y_max() <= table_bbox.y_max()
                    {
                        ocr_boxes_crop.push(BoundingBox::from_coords(
                            b.x_min() - table_x_offset,
                            b.y_min() - table_y_offset,
                            b.x_max() - table_x_offset,
                            b.y_max() - table_y_offset,
                        ));
                    }
                }
                for formula in formulas {
                    let b = &formula.bbox;
                    if b.x_min() >= table_bbox.x_min()
                        && b.y_min() >= table_bbox.y_min()
                        && b.x_max() <= table_bbox.x_max()
                        && b.y_max() <= table_bbox.y_max()
                    {
                        ocr_boxes_crop.push(BoundingBox::from_coords(
                            b.x_min() - table_x_offset,
                            b.y_min() - table_y_offset,
                            b.x_max() - table_x_offset,
                            b.y_max() - table_y_offset,
                        ));
                    }
                }
                for elem in layout_elements {
                    if matches!(
                        elem.element_type,
                        LayoutElementType::Image | LayoutElementType::Chart
                    ) {
                        let b = &elem.bbox;
                        if b.x_min() >= table_bbox.x_min()
                            && b.y_min() >= table_bbox.y_min()
                            && b.x_max() <= table_bbox.x_max()
                            && b.y_max() <= table_bbox.y_max()
                        {
                            ocr_boxes_crop.push(BoundingBox::from_coords(
                                b.x_min() - table_x_offset,
                                b.y_min() - table_y_offset,
                                b.x_max() - table_x_offset,
                                b.y_max() - table_y_offset,
                            ));
                        }
                    }
                }

                let expected_n = structure_bboxes_crop.len();
                let processed_detected_crop = crate::processors::reprocess_table_cells_with_ocr(
                    &detected_bboxes,
                    &detected_scores,
                    &ocr_boxes_crop,
                    expected_n,
                );

                let reconciled_bboxes = crate::processors::reconcile_table_cells(
                    &structure_bboxes_crop,
                    &processed_detected_crop,
                );

                for (cell, new_bbox_crop) in cells.iter_mut().zip(reconciled_bboxes.into_iter()) {
                    cell.bbox = BoundingBox::from_coords(
                        new_bbox_crop.x_min() + table_x_offset,
                        new_bbox_crop.y_min() + table_y_offset,
                        new_bbox_crop.x_max() + table_x_offset,
                        new_bbox_crop.y_max() + table_y_offset,
                    );
                }
            }
        }

        let html_structure = crate::processors::wrap_table_html(structure);
        let mut final_result = TableResult::new(table_bbox.clone(), table_type)
            .with_cells(cells)
            .with_html_structure(html_structure)
            .with_structure_tokens(structure.clone())
            .with_structure_confidence(*structure_score);

        if let Some(conf) = classification_confidence {
            final_result = final_result.with_classification_confidence(conf);
        }

        Ok(Some(final_result))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a minimal TableAnalyzer with no adapters for testing stub behavior.
    fn create_stub_analyzer() -> TableAnalyzer {
        TableAnalyzer::new(TableAnalyzerConfig {
            use_e2e_wired_table_rec: true,
            use_e2e_wireless_table_rec: true,
            ..Default::default()
        })
    }

    /// Creates a test image with specified dimensions.
    fn create_test_image(width: u32, height: u32) -> image::RgbImage {
        image::RgbImage::new(width, height)
    }

    #[test]
    fn test_table_offset_floor_alignment() {
        // Verify that offset calculation uses floor() to align with BBoxCrop
        // which truncates f32 coordinates to u32.

        // Case 1: Positive fractional coordinates
        let bbox = BoundingBox::from_coords(10.7, 20.3, 100.0, 200.0);
        let x_offset = bbox.x_min().max(0.0).floor();
        let y_offset = bbox.y_min().max(0.0).floor();
        assert_eq!(x_offset, 10.0);
        assert_eq!(y_offset, 20.0);

        // Case 2: Integer coordinates (no change)
        let bbox = BoundingBox::from_coords(15.0, 25.0, 100.0, 200.0);
        let x_offset = bbox.x_min().max(0.0).floor();
        let y_offset = bbox.y_min().max(0.0).floor();
        assert_eq!(x_offset, 15.0);
        assert_eq!(y_offset, 25.0);

        // Case 3: Negative coordinates clamped to 0
        let bbox = BoundingBox::from_coords(-5.5, -10.2, 100.0, 200.0);
        let x_offset = bbox.x_min().max(0.0).floor();
        let y_offset = bbox.y_min().max(0.0).floor();
        assert_eq!(x_offset, 0.0);
        assert_eq!(y_offset, 0.0);

        // Case 4: High precision fractional coordinates
        let bbox = BoundingBox::from_coords(99.999, 199.001, 300.0, 400.0);
        let x_offset = bbox.x_min().max(0.0).floor();
        let y_offset = bbox.y_min().max(0.0).floor();
        assert_eq!(x_offset, 99.0);
        assert_eq!(y_offset, 199.0);
    }

    #[test]
    fn test_cell_bbox_rotation_90_degrees() {
        // When table is rotated 90° to correct orientation, cell boxes need
        // to be transformed back to original page coordinates.
        let rotated_width = 100;
        let rotated_height = 200;

        // Cell bbox in the rotated image
        let cell_bbox = BoundingBox::from_coords(10.0, 20.0, 30.0, 40.0);

        // Transform back to original
        let original = cell_bbox.rotate_back_to_original(90.0, rotated_width, rotated_height);

        // For 90° rotation: (x, y) -> (rotated_height - 1 - y, x)
        // Original points: (10, 20), (30, 20), (30, 40), (10, 40)
        // Expected: (179, 10), (179, 30), (159, 30), (159, 10)
        assert!((original.x_min() - 159.0).abs() < 0.01);
        assert!((original.y_min() - 10.0).abs() < 0.01);
        assert!((original.x_max() - 179.0).abs() < 0.01);
        assert!((original.y_max() - 30.0).abs() < 0.01);
    }

    #[test]
    fn test_cell_bbox_rotation_180_degrees() {
        let rotated_width = 100;
        let rotated_height = 200;

        let cell_bbox = BoundingBox::from_coords(10.0, 20.0, 30.0, 40.0);
        let original = cell_bbox.rotate_back_to_original(180.0, rotated_width, rotated_height);

        // For 180° rotation: (x, y) -> (rotated_width - 1 - x, rotated_height - 1 - y)
        // Expected corners: (69, 159), (89, 159), (89, 179), (69, 179)
        assert!((original.x_min() - 69.0).abs() < 0.01);
        assert!((original.y_min() - 159.0).abs() < 0.01);
        assert!((original.x_max() - 89.0).abs() < 0.01);
        assert!((original.y_max() - 179.0).abs() < 0.01);
    }

    #[test]
    fn test_cell_bbox_rotation_270_degrees() {
        let rotated_width = 100;
        let rotated_height = 200;

        let cell_bbox = BoundingBox::from_coords(10.0, 20.0, 30.0, 40.0);
        let original = cell_bbox.rotate_back_to_original(270.0, rotated_width, rotated_height);

        // For 270° rotation: (x, y) -> (y, rotated_width - 1 - x)
        // Expected corners: (20, 69), (40, 69), (40, 89), (20, 89)
        assert!((original.x_min() - 20.0).abs() < 0.01);
        assert!((original.y_min() - 69.0).abs() < 0.01);
        assert!((original.x_max() - 40.0).abs() < 0.01);
        assert!((original.y_max() - 89.0).abs() < 0.01);
    }

    #[test]
    fn test_cell_bbox_rotation_skipped_for_small_angles() {
        // Rotation should be skipped when angle < 1.0 degrees
        let rotated_width = 100;
        let rotated_height = 200;

        let cell_bbox = BoundingBox::from_coords(10.0, 20.0, 30.0, 40.0);

        // Small angle - should not transform
        let result = cell_bbox.rotate_back_to_original(0.5, rotated_width, rotated_height);
        assert_eq!(result.x_min(), cell_bbox.x_min());
        assert_eq!(result.y_min(), cell_bbox.y_min());
        assert_eq!(result.x_max(), cell_bbox.x_max());
        assert_eq!(result.y_max(), cell_bbox.y_max());

        // Zero angle - no transform
        let result = cell_bbox.rotate_back_to_original(0.0, rotated_width, rotated_height);
        assert_eq!(result.x_min(), cell_bbox.x_min());
        assert_eq!(result.y_min(), cell_bbox.y_min());
    }

    #[test]
    fn test_cell_bbox_translate_to_page_coordinates() {
        // Cells detected in cropped table image need to be translated
        // back to page coordinates by adding the table offset.
        let table_x_offset = 50.0;
        let table_y_offset = 100.0;

        // Cell bbox in crop coordinates
        let cell_crop = BoundingBox::from_coords(10.0, 20.0, 30.0, 40.0);

        // Translate to page coordinates
        let cell_page = cell_crop.translate(table_x_offset, table_y_offset);

        assert_eq!(cell_page.x_min(), 60.0); // 10 + 50
        assert_eq!(cell_page.y_min(), 120.0); // 20 + 100
        assert_eq!(cell_page.x_max(), 80.0); // 30 + 50
        assert_eq!(cell_page.y_max(), 140.0); // 40 + 100
    }

    #[test]
    fn test_cell_bbox_translate_to_crop_coordinates() {
        // For cell reconciliation, page coordinates need to be converted
        // back to crop coordinates by subtracting the table offset.
        let table_x_offset = 50.0;
        let table_y_offset = 100.0;

        // Cell bbox in page coordinates
        let cell_page = BoundingBox::from_coords(60.0, 120.0, 80.0, 140.0);

        // Convert to crop coordinates
        let cell_crop = BoundingBox::from_coords(
            cell_page.x_min() - table_x_offset,
            cell_page.y_min() - table_y_offset,
            cell_page.x_max() - table_x_offset,
            cell_page.y_max() - table_y_offset,
        );

        assert_eq!(cell_crop.x_min(), 10.0);
        assert_eq!(cell_crop.y_min(), 20.0);
        assert_eq!(cell_crop.x_max(), 30.0);
        assert_eq!(cell_crop.y_max(), 40.0);
    }

    #[test]
    fn test_combined_rotation_and_translation() {
        // Test the combined transformation: rotation back + translation
        // This simulates the full cell bbox transformation pipeline.
        let rotated_width = 100;
        let rotated_height = 200;
        let table_x_offset = 50.0;
        let table_y_offset = 100.0;

        // Cell in rotated crop space
        let cell_rotated_crop = BoundingBox::from_coords(0.0, 0.0, 10.0, 20.0);

        // Step 1: Rotate back to original crop orientation (90° case)
        let cell_original_crop =
            cell_rotated_crop.rotate_back_to_original(90.0, rotated_width, rotated_height);

        // Step 2: Translate to page coordinates
        let cell_page = cell_original_crop.translate(table_x_offset, table_y_offset);

        // The cell should now be in original page coordinates
        // Verify it's offset from the table origin
        assert!(cell_page.x_min() >= table_x_offset);
        assert!(cell_page.y_min() >= table_y_offset);
    }

    #[test]
    fn test_stub_analyzer_returns_none_when_no_tables() {
        let analyzer = create_stub_analyzer();
        let page_image = create_test_image(800, 600);

        // No table elements in layout
        let layout_elements: Vec<LayoutElement> = vec![];
        let formulas: Vec<FormulaResult> = vec![];
        let text_regions: Vec<TextRegion> = vec![];

        let result = analyzer
            .analyze_tables(&page_image, &layout_elements, &formulas, &text_regions)
            .expect("analyze_tables should not error");

        assert!(result.is_empty());
    }

    #[test]
    fn test_stub_result_has_correct_bbox_and_type() {
        // When structure adapter is not available, a stub result should be returned
        // with the original bbox and table type.
        let table_bbox = BoundingBox::from_coords(100.0, 100.0, 400.0, 300.0);
        let table_type = TableType::Wired;

        let stub = TableResult::new(table_bbox.clone(), table_type);

        assert_eq!(stub.bbox.x_min(), 100.0);
        assert_eq!(stub.bbox.y_min(), 100.0);
        assert_eq!(stub.bbox.x_max(), 400.0);
        assert_eq!(stub.bbox.y_max(), 300.0);
        assert!(matches!(stub.table_type, TableType::Wired));
        assert!(stub.cells.is_empty());
        assert!(stub.html_structure.is_none());
    }

    #[test]
    fn test_stub_result_with_classification_confidence() {
        let table_bbox = BoundingBox::from_coords(100.0, 100.0, 400.0, 300.0);
        let stub =
            TableResult::new(table_bbox, TableType::Wireless).with_classification_confidence(0.95);

        assert_eq!(stub.classification_confidence, Some(0.95));
        assert!(stub.structure_confidence.is_none()); // Not set for stubs
    }

    #[test]
    fn test_ocr_box_fully_inside_table() {
        let table_bbox = BoundingBox::from_coords(100.0, 100.0, 500.0, 400.0);

        // OCR box fully inside table
        let ocr_bbox = BoundingBox::from_coords(150.0, 150.0, 200.0, 180.0);

        let is_inside = ocr_bbox.x_min() >= table_bbox.x_min()
            && ocr_bbox.y_min() >= table_bbox.y_min()
            && ocr_bbox.x_max() <= table_bbox.x_max()
            && ocr_bbox.y_max() <= table_bbox.y_max();

        assert!(is_inside);
    }

    #[test]
    fn test_ocr_box_partially_outside_table() {
        let table_bbox = BoundingBox::from_coords(100.0, 100.0, 500.0, 400.0);

        // OCR box extends beyond table boundary
        let ocr_bbox = BoundingBox::from_coords(450.0, 350.0, 550.0, 420.0);

        let is_inside = ocr_bbox.x_min() >= table_bbox.x_min()
            && ocr_bbox.y_min() >= table_bbox.y_min()
            && ocr_bbox.x_max() <= table_bbox.x_max()
            && ocr_bbox.y_max() <= table_bbox.y_max();

        assert!(!is_inside);
    }

    #[test]
    fn test_ocr_box_conversion_to_crop_coordinates() {
        let table_x_offset = 100.0;
        let table_y_offset = 100.0;

        // OCR box in page coordinates
        let ocr_page = BoundingBox::from_coords(150.0, 150.0, 200.0, 180.0);

        // Convert to crop coordinates
        let ocr_crop = BoundingBox::from_coords(
            ocr_page.x_min() - table_x_offset,
            ocr_page.y_min() - table_y_offset,
            ocr_page.x_max() - table_x_offset,
            ocr_page.y_max() - table_y_offset,
        );

        assert_eq!(ocr_crop.x_min(), 50.0);
        assert_eq!(ocr_crop.y_min(), 50.0);
        assert_eq!(ocr_crop.x_max(), 100.0);
        assert_eq!(ocr_crop.y_max(), 80.0);
    }

    #[test]
    fn test_cell_bbox_from_8_point_polygon() {
        // Table structure recognition can return 8-point polygons
        let bbox_coords: Vec<f32> = vec![
            10.0, 20.0, // top-left
            90.0, 20.0, // top-right
            90.0, 80.0, // bottom-right
            10.0, 80.0, // bottom-left
        ];

        let xs = [
            bbox_coords[0],
            bbox_coords[2],
            bbox_coords[4],
            bbox_coords[6],
        ];
        let ys = [
            bbox_coords[1],
            bbox_coords[3],
            bbox_coords[5],
            bbox_coords[7],
        ];
        let x_min = xs.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
        let y_min = ys.iter().fold(f32::INFINITY, |acc, &y| acc.min(y));
        let x_max = xs.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
        let y_max = ys.iter().fold(f32::NEG_INFINITY, |acc, &y| acc.max(y));

        assert_eq!(x_min, 10.0);
        assert_eq!(y_min, 20.0);
        assert_eq!(x_max, 90.0);
        assert_eq!(y_max, 80.0);
    }

    #[test]
    fn test_cell_bbox_from_4_point_rect() {
        // 4-value format: [x_min, y_min, x_max, y_max]
        let bbox_coords: Vec<f32> = vec![10.0, 20.0, 90.0, 80.0];

        let bbox = if bbox_coords.len() >= 4 {
            BoundingBox::from_coords(
                bbox_coords[0],
                bbox_coords[1],
                bbox_coords[2],
                bbox_coords[3],
            )
        } else {
            BoundingBox::from_coords(0.0, 0.0, 0.0, 0.0)
        };

        assert_eq!(bbox.x_min(), 10.0);
        assert_eq!(bbox.y_min(), 20.0);
        assert_eq!(bbox.x_max(), 90.0);
        assert_eq!(bbox.y_max(), 80.0);
    }

    #[test]
    fn test_cell_bbox_fallback_for_empty_coords() {
        let bbox_coords: Vec<f32> = vec![];

        let bbox = if bbox_coords.len() >= 8 {
            let xs = [
                bbox_coords[0],
                bbox_coords[2],
                bbox_coords[4],
                bbox_coords[6],
            ];
            let ys = [
                bbox_coords[1],
                bbox_coords[3],
                bbox_coords[5],
                bbox_coords[7],
            ];
            let x_min = xs.iter().fold(f32::INFINITY, |acc, &x| acc.min(x));
            let y_min = ys.iter().fold(f32::INFINITY, |acc, &y| acc.min(y));
            let x_max = xs.iter().fold(f32::NEG_INFINITY, |acc, &x| acc.max(x));
            let y_max = ys.iter().fold(f32::NEG_INFINITY, |acc, &y| acc.max(y));
            BoundingBox::from_coords(x_min, y_min, x_max, y_max)
        } else if bbox_coords.len() >= 4 {
            BoundingBox::from_coords(
                bbox_coords[0],
                bbox_coords[1],
                bbox_coords[2],
                bbox_coords[3],
            )
        } else {
            BoundingBox::from_coords(0.0, 0.0, 0.0, 0.0)
        };

        // Fallback to zero-sized bbox
        assert_eq!(bbox.x_min(), 0.0);
        assert_eq!(bbox.y_min(), 0.0);
        assert_eq!(bbox.x_max(), 0.0);
        assert_eq!(bbox.y_max(), 0.0);
    }

    #[test]
    fn test_table_cell_with_position_and_span() {
        let bbox = BoundingBox::from_coords(10.0, 20.0, 100.0, 80.0);
        let cell = TableCell::new(bbox, 0.95)
            .with_position(1, 2)
            .with_span(2, 3);

        assert_eq!(cell.row, Some(1));
        assert_eq!(cell.col, Some(2));
        assert_eq!(cell.row_span, Some(2));
        assert_eq!(cell.col_span, Some(3));
        assert!((cell.confidence - 0.95).abs() < 0.001);
    }

    #[test]
    fn test_table_cell_default_values() {
        let bbox = BoundingBox::from_coords(10.0, 20.0, 100.0, 80.0);
        let cell = TableCell::new(bbox, 1.0);

        assert!(cell.row.is_none());
        assert!(cell.col.is_none());
        assert!(cell.row_span.is_none());
        assert!(cell.col_span.is_none());
        assert!(cell.text.is_none());
    }

    #[test]
    fn test_e2e_mode_selection_wired() {
        let table_type = TableType::Wired;
        let use_e2e_wired = true;
        let use_e2e_wireless = false;

        let use_e2e = match table_type {
            TableType::Wired => use_e2e_wired,
            TableType::Wireless => use_e2e_wireless,
            TableType::Unknown => use_e2e_wireless,
        };

        assert!(use_e2e);
    }

    #[test]
    fn test_e2e_mode_selection_wireless() {
        let table_type = TableType::Wireless;
        let use_e2e_wired = true;
        let use_e2e_wireless = false;

        let use_e2e = match table_type {
            TableType::Wired => use_e2e_wired,
            TableType::Wireless => use_e2e_wireless,
            TableType::Unknown => use_e2e_wireless,
        };

        assert!(!use_e2e);
    }

    #[test]
    fn test_e2e_mode_selection_unknown_defaults_to_wireless() {
        let table_type = TableType::Unknown;
        let use_e2e_wired = true;
        let use_e2e_wireless = false;

        let use_e2e = match table_type {
            TableType::Wired => use_e2e_wired,
            TableType::Wireless => use_e2e_wireless,
            TableType::Unknown => use_e2e_wireless,
        };

        assert!(!use_e2e); // Unknown defaults to wireless behavior
    }

    #[test]
    fn test_analyze_tables_with_table_element_no_adapters() {
        // When a table element exists but no structure adapter is available,
        // the analyzer should return a stub TableResult.
        let analyzer = create_stub_analyzer();
        let page_image = create_test_image(800, 600);

        // Create a table layout element
        let table_element = LayoutElement::new(
            BoundingBox::from_coords(100.0, 100.0, 400.0, 300.0),
            LayoutElementType::Table,
            0.9,
        );
        let layout_elements = vec![table_element];
        let formulas: Vec<FormulaResult> = vec![];
        let text_regions: Vec<TextRegion> = vec![];

        let result = analyzer
            .analyze_tables(&page_image, &layout_elements, &formulas, &text_regions)
            .expect("analyze_tables should not error");

        // Should produce one stub result
        assert_eq!(result.len(), 1);
        let table = &result[0];

        // Stub should have correct bbox
        assert!((table.bbox.x_min() - 100.0).abs() < 0.01);
        assert!((table.bbox.y_min() - 100.0).abs() < 0.01);
        assert!((table.bbox.x_max() - 400.0).abs() < 0.01);
        assert!((table.bbox.y_max() - 300.0).abs() < 0.01);

        // Stub should have no cells or HTML (no structure recognition)
        assert!(table.cells.is_empty());
        assert!(table.html_structure.is_none());

        // Classification confidence should be None (no classifier)
        assert!(table.classification_confidence.is_none());
    }

    #[test]
    fn test_analyze_tables_skips_non_table_elements() {
        let analyzer = create_stub_analyzer();
        let page_image = create_test_image(800, 600);

        // Create non-table layout elements
        let text_element = LayoutElement::new(
            BoundingBox::from_coords(50.0, 50.0, 200.0, 100.0),
            LayoutElementType::Text,
            0.95,
        );
        let image_element = LayoutElement::new(
            BoundingBox::from_coords(300.0, 300.0, 500.0, 500.0),
            LayoutElementType::Image,
            0.85,
        );
        let layout_elements = vec![text_element, image_element];
        let formulas: Vec<FormulaResult> = vec![];
        let text_regions: Vec<TextRegion> = vec![];

        let result = analyzer
            .analyze_tables(&page_image, &layout_elements, &formulas, &text_regions)
            .expect("analyze_tables should not error");

        // No tables should be returned since there are no Table elements
        assert!(result.is_empty());
    }

    #[test]
    fn test_analyze_tables_multiple_tables() {
        let analyzer = create_stub_analyzer();
        let page_image = create_test_image(1000, 800);

        // Create multiple table layout elements
        let table1 = LayoutElement::new(
            BoundingBox::from_coords(50.0, 50.0, 300.0, 200.0),
            LayoutElementType::Table,
            0.9,
        );
        let table2 = LayoutElement::new(
            BoundingBox::from_coords(50.0, 250.0, 300.0, 400.0),
            LayoutElementType::Table,
            0.85,
        );
        let table3 = LayoutElement::new(
            BoundingBox::from_coords(400.0, 50.0, 700.0, 300.0),
            LayoutElementType::Table,
            0.95,
        );
        let layout_elements = vec![table1, table2, table3];
        let formulas: Vec<FormulaResult> = vec![];
        let text_regions: Vec<TextRegion> = vec![];

        let result = analyzer
            .analyze_tables(&page_image, &layout_elements, &formulas, &text_regions)
            .expect("analyze_tables should not error");

        // Should produce three stub results
        assert_eq!(result.len(), 3);

        // Verify each has distinct bbox
        let bboxes: Vec<_> = result
            .iter()
            .map(|t| (t.bbox.x_min(), t.bbox.y_min()))
            .collect();
        assert!(bboxes.contains(&(50.0, 50.0)));
        assert!(bboxes.contains(&(50.0, 250.0)));
        assert!(bboxes.contains(&(400.0, 50.0)));
    }

    #[test]
    fn test_analyze_tables_handles_edge_crop_region() {
        let analyzer = create_stub_analyzer();
        // Small image where table bbox extends beyond bounds
        let page_image = create_test_image(100, 100);

        // Table element with bbox starting outside image bounds
        // BBoxCrop clamps coordinates but can still produce a valid crop region
        let table_element = LayoutElement::new(
            BoundingBox::from_coords(200.0, 200.0, 500.0, 400.0),
            LayoutElementType::Table,
            0.9,
        );
        let layout_elements = vec![table_element];
        let formulas: Vec<FormulaResult> = vec![];
        let text_regions: Vec<TextRegion> = vec![];

        let result = analyzer
            .analyze_tables(&page_image, &layout_elements, &formulas, &text_regions)
            .expect("analyze_tables should not error");

        // BBoxCrop clamps to image bounds and produces a 1x1 crop
        // which is still valid, resulting in a stub table result
        assert_eq!(result.len(), 1);
    }
}
