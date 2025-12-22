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
    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new(
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
    ) -> Self {
        Self {
            table_classification_adapter,
            table_orientation_adapter,
            table_structure_recognition_adapter,
            wired_table_structure_adapter,
            wireless_table_structure_adapter,
            table_cell_detection_adapter,
            wired_table_cell_adapter,
            wireless_table_cell_adapter,
            use_e2e_wired_table_rec,
            use_e2e_wireless_table_rec,
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
                    "wired" | "wired_table" | "有线表格" => TableType::Wired,
                    "wireless" | "wireless_table" | "无线表格" => TableType::Wireless,
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
