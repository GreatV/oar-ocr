//! Unified document parser using layout-first approach.
//!
//! This module provides a generic document parsing pipeline that:
//! 1. Detects layout elements with PP-DocLayout (v2/v3)
//! 2. Filters and sorts elements by reading order
//! 3. Crops regions and recognizes each with a pluggable backend
//! 4. Returns structured document results
//!
//! Supported backends:
//! - `UniRec` - Lightweight unified recognition
//! - `PaddleOcrVl` - Larger VLM with task-specific prompts
//! - `HunyuanOcr` - OCR expert VLM (HunYuanVL)
//! - `GlmOcr` - GLM-OCR OCR expert VLM
//! - `LightOnOcr` - End-to-end OCR VLM

use super::utils::{
    DetectedBox, calculate_overlap_ratio, calculate_projection_overlap_ratio, convert_otsl_to_html,
    crop_margin, filter_overlap_boxes, text, truncate_repetitive_content,
};
use image::RgbImage;
use image::{Rgb, imageops};
use oar_ocr_core::core::{OCRError, OpaqueError};
use oar_ocr_core::domain::structure::{
    LayoutElement, LayoutElementType, StructureResult, TableResult, TableType,
};
use oar_ocr_core::predictors::LayoutDetectionPredictor;
use oar_ocr_core::processors::BoundingBox;
use oar_ocr_core::processors::layout_sorting::sort_layout_enhanced;
use oar_ocr_core::utils::BBoxCrop;
use std::sync::Arc;

/// Recognition task for a layout element.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RecognitionTask {
    /// General OCR/text recognition
    Ocr,
    /// Table structure recognition (outputs HTML)
    Table,
    /// Formula recognition (outputs LaTeX)
    Formula,
    /// Chart recognition
    Chart,
}

/// Trait for recognition backends that can process cropped regions.
pub trait RecognitionBackend {
    /// Generate text/content for a cropped image region.
    ///
    /// # Arguments
    /// * `image` - Cropped region image
    /// * `task` - Recognition task type
    /// * `max_tokens` - Maximum tokens to generate
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, OCRError>;

    /// Whether this backend requires post-processing for table output.
    /// UniRec outputs OTSL format that needs conversion; PaddleOCR-VL outputs HTML directly.
    fn needs_table_postprocess(&self) -> bool {
        false
    }

    /// Whether this backend benefits from margin cropping for formulas.
    fn needs_formula_preprocess(&self) -> bool {
        false
    }

    /// Whether this backend output may contain repetitive content.
    fn needs_repetition_truncation(&self) -> bool {
        false
    }
}

/// Configuration for the unified document parser.
#[derive(Debug, Clone)]
pub struct DocParserConfig {
    /// Adds extra padding around each detected region before cropping.
    pub crop_pad_ratio: f32,
    /// Maximum tokens for generation per region.
    pub max_tokens: usize,
    /// Drops header/footer/aside/number regions early to reduce recognition calls.
    pub skip_auxiliary_regions: bool,
    /// Drops PP-DocBlockLayout regions (if present).
    pub skip_region_blocks: bool,
    /// Labels to ignore when converting to markdown.
    pub markdown_ignore_labels: Vec<String>,
}

impl Default for DocParserConfig {
    fn default() -> Self {
        Self {
            // OpenOCR's CropByBoxes does not add padding around regions.
            crop_pad_ratio: 0.0,
            max_tokens: 4096,
            skip_auxiliary_regions: true,
            skip_region_blocks: true,
            markdown_ignore_labels: vec![
                "number".to_string(),
                "footnote".to_string(),
                "header".to_string(),
                "header_image".to_string(),
                "footer".to_string(),
                "footer_image".to_string(),
                "aside_text".to_string(),
                // PaddleOCR-VL markdown output skips `formula_number` blocks by default (unless explicitly
                // merged into the preceding formula), so ignore it for OpenOCR/PaddleX parity.
                "formula_number".to_string(),
            ],
        }
    }
}

/// Unified document parser combining layout detection with recognition backend.
pub struct DocParser<'a, B: RecognitionBackend> {
    backend: &'a B,
    config: DocParserConfig,
}

impl<'a, B: RecognitionBackend> DocParser<'a, B> {
    /// Create a new document parser with the given backend.
    pub fn new(backend: &'a B) -> Self {
        Self {
            backend,
            config: DocParserConfig::default(),
        }
    }

    /// Create a new document parser with custom configuration.
    pub fn with_config(backend: &'a B, config: DocParserConfig) -> Self {
        Self { backend, config }
    }

    /// Returns a reference to the parser's configuration.
    pub fn config(&self) -> &DocParserConfig {
        &self.config
    }

    /// Parse a document image using layout-first pipeline.
    pub fn parse(
        &self,
        layout_predictor: &LayoutDetectionPredictor,
        image: RgbImage,
    ) -> Result<StructureResult, OCRError> {
        self.parse_with_path(layout_predictor, "<memory>", 0, image)
    }

    /// Parse a document image without layout detection (single full-image OCR).
    ///
    /// Use this for end-to-end models (e.g. LightOnOCR) that handle layout internally.
    /// For models requiring separate layout detection, use [`parse`](Self::parse) instead.
    pub fn parse_without_layout(&self, image: RgbImage) -> Result<StructureResult, OCRError> {
        self.recognize_full_image("<memory>".into(), 0, image)
    }

    /// Parse a document image with source path information.
    pub fn parse_with_path(
        &self,
        layout_predictor: &LayoutDetectionPredictor,
        input_path: impl Into<Arc<str>>,
        index: usize,
        image: RgbImage,
    ) -> Result<StructureResult, OCRError> {
        let input_path: Arc<str> = input_path.into();
        let (page_w, page_h) = (image.width() as f32, image.height() as f32);

        // Step 1: Layout detection
        let layout_result =
            layout_predictor
                .predict(vec![image.clone()])
                .map_err(|e| OCRError::Inference {
                    model_name: "layout_detection".to_string(),
                    context: "DocParser: layout prediction failed".to_string(),
                    source: Box::new(OpaqueError::from_display(e)),
                })?;

        let detected = layout_result
            .elements
            .into_iter()
            .next()
            .unwrap_or_default();

        // OpenOCR applies an additional overlap filter (overlap_ratio > 0.7, mode=small)
        // after PP-DocLayout (v2/v3) to remove redundant boxes.
        let detected: Vec<DetectedBox> = detected
            .into_iter()
            .map(|e| DetectedBox {
                bbox: e.bbox,
                label: e.element_type,
                score: e.score,
            })
            .collect();
        let detected = filter_overlap_boxes(detected, 0.7);

        // If no layout elements detected, run OCR on the whole image
        if detected.is_empty() {
            return self.recognize_full_image(input_path, index, image);
        }

        // Step 2: Filter and prepare elements
        let mut elements: Vec<LayoutElement> = Vec::with_capacity(detected.len());
        for element in detected {
            let element_type = LayoutElementType::from_label(&element.label);
            if self.config.skip_region_blocks && element_type == LayoutElementType::Region {
                continue;
            }
            if self.config.skip_auxiliary_regions && is_auxiliary_element(element_type) {
                continue;
            }

            elements.push(
                LayoutElement::new(element.bbox, element_type, element.score)
                    .with_label(element.label),
            );
        }

        if elements.is_empty() {
            return self.recognize_full_image(input_path, index, image);
        }

        // Step 3: Sort by reading order
        let mut sorted_elements: Vec<LayoutElement> = if layout_result.is_reading_order_sorted {
            elements
        } else {
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

        // Step 4: Recognize each element (with OpenOCR-style block merging)
        //
        // OpenOCR merges adjacent "text" blocks before recognition by vertically stacking crops.
        // This can improve recognition quality for fragmented detections.
        let merge_groups = compute_openocr_merge_groups(&sorted_elements);
        let mut group_first_for_index: Vec<Option<usize>> = vec![None; sorted_elements.len()];
        let mut group_by_first: std::collections::HashMap<usize, MergeGroup> =
            std::collections::HashMap::new();
        for group in merge_groups {
            let first = group.indices[0];
            for &idx in &group.indices {
                if idx < group_first_for_index.len() {
                    group_first_for_index[idx] = Some(first);
                }
            }
            group_by_first.insert(first, group);
        }

        let element_bboxes: Vec<BoundingBox> =
            sorted_elements.iter().map(|el| el.bbox.clone()).collect();

        let mut tables: Vec<TableResult> = Vec::new();
        let mut merged_by_first: std::collections::HashMap<usize, bool> =
            std::collections::HashMap::new();
        for (idx, element) in sorted_elements.iter_mut().enumerate() {
            if let Some(first) = group_first_for_index.get(idx).and_then(|v| *v)
                && first != idx
            {
                // OpenOCR only drops non-first blocks *when a merge actually happens*.
                // When the group is not merged (aspect_ratio >= 3), each block is recognized separately.
                if merged_by_first.get(&first).copied().unwrap_or(false) {
                    continue;
                }
            }

            // Determine task for element
            let Some(task) = task_for_element_type(element.element_type) else {
                continue;
            };

            let group = group_by_first.get(&idx);
            let mut cropped = if let Some(group) = group {
                // Merge all crops in the group into a single stacked image.
                let mut crops: Vec<RgbImage> = Vec::with_capacity(group.indices.len());
                for &g_idx in &group.indices {
                    let Some(bbox) = element_bboxes.get(g_idx) else {
                        continue;
                    };
                    let crop_bbox = if self.config.crop_pad_ratio > 0.0 {
                        pad_bbox(bbox, page_w, page_h, self.config.crop_pad_ratio)
                    } else {
                        bbox.clone()
                    };
                    let crop = match BBoxCrop::crop_bounding_box(&image, &crop_bbox) {
                        Ok(crop) => crop,
                        Err(_) => continue,
                    };
                    crops.push(crop);
                }
                if crops.is_empty() {
                    continue;
                }
                // Decide whether to merge based on aspect ratio (OpenOCR: skip merge if h/w >= 3).
                let max_w = crops.iter().map(|c| c.width()).max().unwrap_or(0);
                let sum_h: u32 = crops.iter().map(|c| c.height()).sum();
                let aspect_ratio = if max_w == 0 {
                    f32::INFINITY
                } else {
                    sum_h as f32 / max_w as f32
                };
                if aspect_ratio >= 3.0 || crops.len() == 1 {
                    merged_by_first.insert(idx, false);
                    // Fallback: use this element's own crop only.
                    let crop_bbox = if self.config.crop_pad_ratio > 0.0 {
                        pad_bbox(&element.bbox, page_w, page_h, self.config.crop_pad_ratio)
                    } else {
                        element.bbox.clone()
                    };
                    match BBoxCrop::crop_bounding_box(&image, &crop_bbox) {
                        Ok(crop) => crop,
                        Err(_) => continue,
                    }
                } else {
                    merged_by_first.insert(idx, true);
                    merge_images_vertically(&crops, &group.aligns)
                }
            } else {
                let crop_bbox = if self.config.crop_pad_ratio > 0.0 {
                    pad_bbox(&element.bbox, page_w, page_h, self.config.crop_pad_ratio)
                } else {
                    element.bbox.clone()
                };
                match BBoxCrop::crop_bounding_box(&image, &crop_bbox) {
                    Ok(cropped) => cropped,
                    Err(_) => continue,
                }
            };

            // Apply preprocessing if needed
            if task == RecognitionTask::Formula && self.backend.needs_formula_preprocess() {
                cropped = crop_margin(&cropped);
            }

            // Generate text
            let mut generated = self
                .backend
                .recognize(cropped, task, self.config.max_tokens)?;
            if generated.trim().is_empty() {
                continue;
            }

            // Apply repetition truncation if needed
            if self.backend.needs_repetition_truncation() {
                generated = truncate_repetitive_content(&generated, 10, 10, 10);
            }

            // Apply post-processing based on task.
            //
            // Note: Table blocks should remain HTML/OTSL-derived HTML; do not run the text normalizer
            // on table markup.
            let processed = if task == RecognitionTask::Table {
                if self.backend.needs_table_postprocess() {
                    convert_otsl_to_html(&generated)
                } else {
                    generated.trim().to_string()
                }
            } else if task == RecognitionTask::Formula {
                text::format_formula(&generated)
            } else {
                text::format_text(&generated)
            };

            if element.element_type == LayoutElementType::Table {
                tables.push(
                    TableResult::new(element.bbox.clone(), TableType::Unknown)
                        .with_html_structure(processed.clone()),
                );
            }

            element.text = Some(processed);
        }

        Ok(StructureResult::new(input_path, index)
            .with_layout_elements(sorted_elements)
            .with_tables(tables))
    }

    /// Parse a document and convert to markdown.
    pub fn parse_to_markdown(
        &self,
        layout_predictor: &LayoutDetectionPredictor,
        image: RgbImage,
    ) -> Result<String, OCRError> {
        let result = self.parse(layout_predictor, image)?;
        Ok(super::utils::to_markdown(
            &result.layout_elements,
            &self.config.markdown_ignore_labels,
        ))
    }

    /// Parse a document and convert to OpenOCR/PaddleX-compatible markdown (pretty HTML mode).
    pub fn parse_to_markdown_openocr(
        &self,
        layout_predictor: &LayoutDetectionPredictor,
        image: RgbImage,
    ) -> Result<String, OCRError> {
        let result = self.parse(layout_predictor, image)?;
        Ok(super::utils::to_markdown_openocr(
            &result.layout_elements,
            &self.config.markdown_ignore_labels,
            true,
        ))
    }

    fn recognize_full_image(
        &self,
        input_path: Arc<str>,
        index: usize,
        image: RgbImage,
    ) -> Result<StructureResult, OCRError> {
        let (page_w, page_h) = (image.width() as f32, image.height() as f32);
        let text = self
            .backend
            .recognize(image, RecognitionTask::Ocr, self.config.max_tokens)?;
        let element = LayoutElement::new(
            BoundingBox::from_coords(0.0, 0.0, page_w, page_h),
            LayoutElementType::Text,
            1.0,
        )
        .with_label("text")
        .with_text(text.trim());
        Ok(StructureResult::new(input_path, index).with_layout_elements(vec![element]))
    }
}

use super::unirec::UniRec;

impl RecognitionBackend for UniRec {
    fn recognize(
        &self,
        image: RgbImage,
        _task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, OCRError> {
        // UniRec doesn't use task-specific prompts; it's a unified model
        self.generate(&[image], max_tokens)
            .into_iter()
            .next()
            .unwrap_or_else(|| {
                Err(OCRError::InvalidInput {
                    message: "UniRec: no result returned".to_string(),
                })
            })
    }

    fn needs_table_postprocess(&self) -> bool {
        true // UniRec outputs OTSL format
    }

    fn needs_formula_preprocess(&self) -> bool {
        true // Benefit from margin cropping
    }

    fn needs_repetition_truncation(&self) -> bool {
        true // May produce repetitive output
    }
}

use super::glmocr::GlmOcr;
use super::hunyuanocr::HunyuanOcr;
use super::lightonocr::LightOnOcr;
use super::paddleocr_vl::{PaddleOcrVl, PaddleOcrVlTask};

impl RecognitionBackend for PaddleOcrVl {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, OCRError> {
        let vl_task = match task {
            RecognitionTask::Ocr => PaddleOcrVlTask::Ocr,
            RecognitionTask::Table => PaddleOcrVlTask::Table,
            RecognitionTask::Formula => PaddleOcrVlTask::Formula,
            RecognitionTask::Chart => PaddleOcrVlTask::Chart,
        };

        // PaddleOCR-VL's reference pipeline truncates repetitive tails on the *raw* model output,
        // before per-task postprocessing (e.g., table-token conversion).
        let results = self.generate_with_raw(&[image], &[vl_task], max_tokens);
        let (raw, _) = results.into_iter().next().ok_or(OCRError::InvalidInput {
            message: "PaddleOCR-VL: no result returned".to_string(),
        })??;
        let raw = truncate_repetitive_content(&raw, 10, 10, 10);
        Ok(vl_task.postprocess(raw))
    }

    fn needs_table_postprocess(&self) -> bool {
        false // PaddleOCR-VL outputs HTML directly
    }

    fn needs_formula_preprocess(&self) -> bool {
        true // Match PaddleOCR-VL pipeline: crop formula margins before recognition
    }

    fn needs_repetition_truncation(&self) -> bool {
        false // Handled inside `recognize()` (must run before per-task postprocess)
    }
}

impl RecognitionBackend for HunyuanOcr {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, OCRError> {
        let prompt = match task {
            RecognitionTask::Ocr => {
                "Detect and recognize text in the image, and output the text coordinates in a formatted manner."
            }
            RecognitionTask::Table => "Parse the table in the image into HTML.",
            RecognitionTask::Formula => {
                "Identify the formula in the image and represent it using LaTeX format."
            }
            RecognitionTask::Chart => {
                "Parse the chart in the image; use Mermaid format for flowcharts and Markdown for other charts."
            }
        };
        let out = self
            .generate(&[image], &[prompt], max_tokens)
            .into_iter()
            .next()
            .unwrap_or_else(|| {
                Err(OCRError::InvalidInput {
                    message: "HunyuanOCR: no result returned".to_string(),
                })
            })?;
        Ok(truncate_repetitive_content(&out, 10, 10, 10)
            .trim()
            .to_string())
    }

    fn needs_table_postprocess(&self) -> bool {
        false
    }

    fn needs_formula_preprocess(&self) -> bool {
        false
    }

    fn needs_repetition_truncation(&self) -> bool {
        false // handled inside `recognize()`
    }
}

impl RecognitionBackend for GlmOcr {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, OCRError> {
        let prompt = match task {
            RecognitionTask::Ocr => "Text Recognition:",
            RecognitionTask::Table => "Table Recognition:",
            RecognitionTask::Formula => "Formula Recognition:",
            RecognitionTask::Chart => "Text Recognition:",
        };
        let out = self
            .generate(&[image], &[prompt], max_tokens)
            .into_iter()
            .next()
            .unwrap_or_else(|| {
                Err(OCRError::InvalidInput {
                    message: "GLM-OCR: no result returned".to_string(),
                })
            })?;
        Ok(truncate_repetitive_content(&out, 10, 10, 10)
            .trim()
            .to_string())
    }

    fn needs_table_postprocess(&self) -> bool {
        false
    }

    fn needs_formula_preprocess(&self) -> bool {
        false
    }

    fn needs_repetition_truncation(&self) -> bool {
        false // handled inside `recognize()`
    }
}

impl RecognitionBackend for LightOnOcr {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, OCRError> {
        let prompt = match task {
            RecognitionTask::Ocr => "",
            RecognitionTask::Table => "Parse the table in the image into HTML.",
            RecognitionTask::Formula => {
                "Identify the formula in the image and represent it using LaTeX format."
            }
            RecognitionTask::Chart => {
                "Parse the chart in the image; use Mermaid format for flowcharts and Markdown for other charts."
            }
        };
        let out = self
            .generate(&[image], &[prompt], max_tokens)
            .into_iter()
            .next()
            .unwrap_or_else(|| {
                Err(OCRError::InvalidInput {
                    message: "LightOnOCR: no result returned".to_string(),
                })
            })?;
        Ok(truncate_repetitive_content(&out, 10, 10, 10)
            .trim()
            .to_string())
    }

    fn needs_table_postprocess(&self) -> bool {
        false
    }

    fn needs_formula_preprocess(&self) -> bool {
        false
    }

    fn needs_repetition_truncation(&self) -> bool {
        false // handled inside `recognize()`
    }
}

fn is_auxiliary_element(element_type: LayoutElementType) -> bool {
    matches!(
        element_type,
        LayoutElementType::Number
            | LayoutElementType::Footnote
            | LayoutElementType::Header
            | LayoutElementType::HeaderImage
            | LayoutElementType::Footer
            | LayoutElementType::FooterImage
            | LayoutElementType::AsideText
    )
}

fn task_for_element_type(element_type: LayoutElementType) -> Option<RecognitionTask> {
    match element_type {
        LayoutElementType::Table => Some(RecognitionTask::Table),
        LayoutElementType::Chart => Some(RecognitionTask::Chart),
        LayoutElementType::Formula => Some(RecognitionTask::Formula),
        LayoutElementType::FormulaNumber => Some(RecognitionTask::Ocr),
        // Skip pure visual regions
        LayoutElementType::Image
        | LayoutElementType::HeaderImage
        | LayoutElementType::FooterImage
        | LayoutElementType::Seal => None,
        _ => Some(RecognitionTask::Ocr),
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

/// Document parser using UniRec backend.
pub type UniRecDocParser<'a> = DocParser<'a, UniRec>;

/// Document parser using PaddleOCR-VL backend.
pub type PaddleOcrVlDocParser<'a> = DocParser<'a, PaddleOcrVl>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum MergeAlign {
    Left,
    Right,
    Center,
}

#[derive(Debug, Clone)]
struct MergeGroup {
    indices: Vec<usize>,
    aligns: Vec<MergeAlign>,
}

fn merge_images_vertically(images: &[RgbImage], aligns: &[MergeAlign]) -> RgbImage {
    if images.is_empty() {
        return RgbImage::new(0, 0);
    }
    if images.len() == 1 {
        return images[0].clone();
    }

    let mut merged = images[0].clone();
    for (i, img2) in images.iter().enumerate().skip(1) {
        let align = aligns.get(i - 1).copied().unwrap_or(MergeAlign::Center);
        let w = merged.width().max(img2.width());
        let h = merged.height() + img2.height();
        let mut new_img = RgbImage::from_pixel(w, h, Rgb([255, 255, 255]));

        let (x1, x2) = match align {
            MergeAlign::Center => (
                ((w - merged.width()) / 2) as i64,
                ((w - img2.width()) / 2) as i64,
            ),
            MergeAlign::Right => ((w - merged.width()) as i64, (w - img2.width()) as i64),
            MergeAlign::Left => (0, 0),
        };

        imageops::overlay(&mut new_img, &merged, x1, 0);
        imageops::overlay(&mut new_img, img2, x2, merged.height() as i64);
        merged = new_img;
    }
    merged
}

fn compute_openocr_merge_groups(elements: &[LayoutElement]) -> Vec<MergeGroup> {
    // Match OpenOCR: merge only non-image/table blocks; actual merge conditions are label == "text".
    const NON_MERGE_LABELS: [&str; 6] = [
        "image",
        "header_image",
        "footer_image",
        "seal",
        "table",
        "chart",
    ];

    let mut blocks_to_merge: Vec<usize> = Vec::new();
    for (idx, element) in elements.iter().enumerate() {
        let label = element.label.as_deref().unwrap_or("");
        if NON_MERGE_LABELS.contains(&label) {
            continue;
        }
        blocks_to_merge.push(idx);
    }

    if blocks_to_merge.len() < 2 {
        return Vec::new();
    }

    let mut merged_groups: Vec<MergeGroup> = Vec::new();
    let mut current_indices: Vec<usize> = Vec::new();
    let mut current_aligns: Vec<MergeAlign> = Vec::new();

    fn is_aligned(a1: f32, a2: f32) -> bool {
        (a1 - a2).abs() <= 5.0
    }

    fn rect_xyxy(bbox: &BoundingBox) -> (f32, f32, f32, f32) {
        (bbox.x_min(), bbox.y_min(), bbox.x_max(), bbox.y_max())
    }

    fn get_alignment(curr: (f32, f32, f32, f32), prev: (f32, f32, f32, f32)) -> MergeAlign {
        if is_aligned(curr.0, prev.0) {
            MergeAlign::Left
        } else if is_aligned(curr.2, prev.2) {
            MergeAlign::Right
        } else {
            MergeAlign::Center
        }
    }

    fn overlapwith_other_box(
        block_idx: usize,
        prev_idx: usize,
        elements: &[LayoutElement],
    ) -> bool {
        let prev_bbox = &elements[prev_idx].bbox;
        let block_bbox = &elements[block_idx].bbox;

        let (px1, py1, px2, py2) = rect_xyxy(prev_bbox);
        let (bx1, by1, bx2, by2) = rect_xyxy(block_bbox);

        let min_box =
            BoundingBox::from_coords(px1.min(bx1), py1.min(by1), px2.max(bx2), py2.max(by2));

        for (idx, other) in elements.iter().enumerate() {
            if idx == block_idx || idx == prev_idx {
                continue;
            }
            if calculate_overlap_ratio(&min_box, &other.bbox, "union") > 0.0 {
                return true;
            }
        }
        false
    }

    for (i, &idx) in blocks_to_merge.iter().enumerate() {
        if current_indices.is_empty() {
            current_indices.push(idx);
            continue;
        }

        let prev_idx = blocks_to_merge[i - 1];
        let prev_label = elements[prev_idx].label.as_deref().unwrap_or("");
        let curr_label = elements[idx].label.as_deref().unwrap_or("");

        let prev_rect = rect_xyxy(&elements[prev_idx].bbox);
        let curr_rect = rect_xyxy(&elements[idx].bbox);

        let iou_h = calculate_projection_overlap_ratio(
            &elements[idx].bbox,
            &elements[prev_idx].bbox,
            "horizontal",
            "union",
        );

        let prev_w = (prev_rect.2 - prev_rect.0).max(0.0);
        let curr_w = (curr_rect.2 - curr_rect.0).max(0.0);
        let prev_h = (prev_rect.3 - prev_rect.1).max(0.0);
        let curr_h = (curr_rect.3 - curr_rect.1).max(0.0);

        let is_cross = iou_h == 0.0
            && curr_label == "text"
            && curr_label == prev_label
            && curr_rect.0 > prev_rect.2
            && curr_rect.1 < prev_rect.3
            && (curr_rect.0 - prev_rect.2) < prev_w.max(curr_w) * 0.3;

        let left_aligned = is_aligned(curr_rect.0, prev_rect.0);
        let right_aligned = is_aligned(curr_rect.2, prev_rect.2);

        let is_updown_align = iou_h > 0.0
            && curr_label == "text"
            && curr_label == prev_label
            && curr_rect.3 >= prev_rect.1
            && (curr_rect.1 - prev_rect.3).abs() < prev_h.max(curr_h) * 0.5
            && (left_aligned ^ right_aligned)
            && overlapwith_other_box(idx, prev_idx, elements);

        let align_mode = if is_cross {
            Some(MergeAlign::Center)
        } else if is_updown_align {
            Some(get_alignment(curr_rect, prev_rect))
        } else {
            None
        };

        if is_cross || is_updown_align {
            current_indices.push(idx);
            if let Some(a) = align_mode {
                current_aligns.push(a);
            }
        } else {
            merged_groups.push(MergeGroup {
                indices: std::mem::take(&mut current_indices),
                aligns: std::mem::take(&mut current_aligns),
            });
            current_indices.push(idx);
        }
    }

    if !current_indices.is_empty() {
        merged_groups.push(MergeGroup {
            indices: current_indices,
            aligns: current_aligns,
        });
    }

    merged_groups
        .into_iter()
        .filter(|g| g.indices.len() > 1 && g.aligns.len() + 1 == g.indices.len())
        .collect()
}
