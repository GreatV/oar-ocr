//! PaddleOCR-VL document parsing helpers (layout-first).
//!
//! PaddleOCR-VL's recommended document parsing flow uses a layout model (PP-DocLayoutV2)
//! to segment a page and determine reading order, then runs the VLM per block.

use super::{PaddleOcrVl, PaddleOcrVlTask};
use crate::core::OCRError;
use crate::domain::structure::{
    LayoutElement, LayoutElementType, StructureResult, TableResult, TableType,
};
use crate::predictors::LayoutDetectionPredictor;
use crate::processors::BoundingBox;
use crate::processors::layout_sorting::sort_layout_enhanced;
use crate::utils::BBoxCrop;
use image::RgbImage;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct PaddleOcrVlDocParserConfig {
    /// Adds extra padding around each detected region before cropping.
    pub crop_pad_ratio: f32,
    /// Drops header/footer/aside/number regions early to reduce VLM calls.
    pub skip_auxiliary_regions: bool,
    /// Drops PP-DocBlockLayout regions (if present).
    pub skip_region_blocks: bool,
}

impl Default for PaddleOcrVlDocParserConfig {
    fn default() -> Self {
        Self {
            crop_pad_ratio: 0.01,
            skip_auxiliary_regions: true,
            skip_region_blocks: true,
        }
    }
}

impl PaddleOcrVl {
    /// Parse a document image using a layout-first pipeline (PP-DocLayoutV2 -> PaddleOCR-VL).
    pub fn parse_document(
        &self,
        layout_predictor: &LayoutDetectionPredictor,
        image: RgbImage,
        max_new_tokens: usize,
    ) -> Result<StructureResult, OCRError> {
        parse_document_with_layout_config(
            self,
            layout_predictor,
            "<memory>",
            0,
            image,
            &PaddleOcrVlDocParserConfig::default(),
            max_new_tokens,
        )
    }
}

pub fn parse_document_with_layout_config(
    vl: &PaddleOcrVl,
    layout_predictor: &LayoutDetectionPredictor,
    input_path: impl Into<Arc<str>>,
    index: usize,
    image: RgbImage,
    cfg: &PaddleOcrVlDocParserConfig,
    max_new_tokens: usize,
) -> Result<StructureResult, OCRError> {
    if max_new_tokens == 0 {
        return Err(OCRError::InvalidInput {
            message: "PaddleOCR-VL doc parser: max_new_tokens must be > 0".to_string(),
        });
    }

    let input_path: Arc<str> = input_path.into();

    let (page_w, page_h) = (image.width() as f32, image.height() as f32);
    let layout_result =
        layout_predictor
            .predict(vec![image.clone()])
            .map_err(|e| OCRError::Inference {
                model_name: "layout_detection".to_string(),
                context: "PaddleOCR-VL doc parser: layout prediction failed".to_string(),
                source: Box::new(std::io::Error::other(e.to_string())),
            })?;

    let detected = layout_result
        .elements
        .into_iter()
        .next()
        .unwrap_or_default();

    if detected.is_empty() {
        let text = vl.generate(image.clone(), PaddleOcrVlTask::Ocr, max_new_tokens)?;
        let element = LayoutElement::new(
            BoundingBox::from_coords(0.0, 0.0, page_w, page_h),
            LayoutElementType::Text,
            1.0,
        )
        .with_label("text")
        .with_text(text.trim());
        return Ok(StructureResult::new(input_path, index).with_layout_elements(vec![element]));
    }

    let mut elements: Vec<LayoutElement> = Vec::with_capacity(detected.len());
    for element in detected {
        let element_type = LayoutElementType::from_label(&element.element_type);
        if cfg.skip_region_blocks && element_type == LayoutElementType::Region {
            continue;
        }
        if cfg.skip_auxiliary_regions
            && matches!(
                element_type,
                LayoutElementType::Number
                    | LayoutElementType::Footnote
                    | LayoutElementType::Header
                    | LayoutElementType::HeaderImage
                    | LayoutElementType::Footer
                    | LayoutElementType::FooterImage
                    | LayoutElementType::AsideText
            )
        {
            continue;
        }

        elements.push(
            LayoutElement::new(element.bbox, element_type, element.score)
                .with_label(element.element_type),
        );
    }

    if elements.is_empty() {
        let text = vl.generate(image.clone(), PaddleOcrVlTask::Ocr, max_new_tokens)?;
        let element = LayoutElement::new(
            BoundingBox::from_coords(0.0, 0.0, page_w, page_h),
            LayoutElementType::Text,
            1.0,
        )
        .with_label("text")
        .with_text(text.trim());
        return Ok(StructureResult::new(input_path, index).with_layout_elements(vec![element]));
    }

    // Use reading order from model if available (PP-DocLayoutV2 outputs 8-dim with col/row indices)
    // Otherwise, apply the enhanced sorting algorithm
    let mut sorted_elements: Vec<LayoutElement> = if layout_result.is_reading_order_sorted {
        // Elements are already in reading order from the layout model
        elements
    } else {
        // Apply enhanced sorting algorithm for models without reading order info
        let sortable: Vec<(BoundingBox, LayoutElementType)> = elements
            .iter()
            .map(|e| (e.bbox.clone(), e.element_type))
            .collect();
        let sorted_indices = sort_layout_enhanced(&sortable, page_w, page_h);
        sorted_indices
            .into_iter()
            .filter_map(|idx| elements.get(idx).cloned())
            .collect()
    };

    assign_order_indices(&mut sorted_elements);

    let mut tables: Vec<TableResult> = Vec::new();
    for element in sorted_elements.iter_mut() {
        let Some(task) = task_for_element_type(element.element_type) else {
            continue;
        };

        let crop_bbox = if cfg.crop_pad_ratio > 0.0 {
            pad_bbox(&element.bbox, page_w, page_h, cfg.crop_pad_ratio)
        } else {
            element.bbox.clone()
        };

        let cropped = match BBoxCrop::crop_bounding_box(&image, &crop_bbox) {
            Ok(cropped) => cropped,
            Err(_) => continue,
        };
        let generated = vl.generate(cropped, task, max_new_tokens)?;
        if generated.trim().is_empty() {
            continue;
        }

        if element.element_type == LayoutElementType::Table {
            tables.push(
                TableResult::new(element.bbox.clone(), TableType::Unknown)
                    .with_html_structure(generated.clone()),
            );
        }

        element.text = Some(generated);
    }

    Ok(StructureResult::new(input_path, index)
        .with_layout_elements(sorted_elements)
        .with_tables(tables))
}

fn task_for_element_type(element_type: LayoutElementType) -> Option<PaddleOcrVlTask> {
    match element_type {
        LayoutElementType::Table => Some(PaddleOcrVlTask::Table),
        LayoutElementType::Chart => Some(PaddleOcrVlTask::Chart),
        LayoutElementType::Formula => Some(PaddleOcrVlTask::Formula),
        // Formula numbers are usually short; OCR is sufficient and avoids wrapping issues.
        LayoutElementType::FormulaNumber => Some(PaddleOcrVlTask::Ocr),
        // Skip pure visual regions by default; keeping them in the structure output is enough.
        LayoutElementType::Image
        | LayoutElementType::HeaderImage
        | LayoutElementType::FooterImage => None,
        _ => Some(PaddleOcrVlTask::Ocr),
    }
}

fn pad_bbox(bbox: &BoundingBox, page_w: f32, page_h: f32, pad_ratio: f32) -> BoundingBox {
    let x1 = bbox.x_min();
    let y1 = bbox.y_min();
    let x2 = bbox.x_max();
    let y2 = bbox.y_max();
    let w = (x2 - x1).max(1.0);
    let h = (y2 - y1).max(1.0);
    let pad_x = w * pad_ratio;
    let pad_y = h * pad_ratio;

    BoundingBox::from_coords(
        (x1 - pad_x).max(0.0),
        (y1 - pad_y).max(0.0),
        (x2 + pad_x).min(page_w),
        (y2 + pad_y).min(page_h),
    )
}

fn assign_order_indices(elements: &mut [LayoutElement]) {
    let mut order_index = 1u32;
    for element in elements.iter_mut() {
        if should_have_order_index(element.element_type) {
            element.order_index = Some(order_index);
            order_index += 1;
        }
    }
}

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
