//! Unified layout-first document parser: detect layout (PP-DocLayout v2/v3),
//! then crop and recognize each region with a pluggable backend into structured
//! results. The layout source supplies the regions in reading order.
//!
//! Supported backends include PaddleOCR-VL, HunyuanOCR, GLM-OCR, MinerU2.5,
//! MonkeyOCRv2, OvisOCR2, and MinerU-Diffusion.
//!
//! The `doc_parser` example exposes the layout-first PaddleOCR-VL and GLM-OCR
//! paths. For reference-quality full-page parsing, prefer HunyuanOCR's native
//! page prompt and the MinerU models' native two-step extraction pipeline.

use crate::api::error::Error;
use crate::api::generation::GenerationOptions;
use crate::document::geometry::BoundingBox;
use crate::document::page::ParseDiagnostic;
use crate::document::structure::{
    LayoutElement, LayoutElementType, StructureResult, TableResult, TableType,
};
use crate::pipeline::geometry::{
    calculate_overlap_ratio, calculate_projection_overlap_ratio, crop_margin,
};
use crate::pipeline::layout::LayoutSource;
use crate::render::table::convert_otsl_to_html_with_options;
use crate::render::text::{self, truncate_repetitive_content};
use image::RgbImage;
use image::{Rgb, imageops};
use std::borrow::Cow;
use std::sync::Arc;

pub use crate::api::recognition::{
    AuxiliaryLabelAction, AuxiliaryLabelRule, DocParserOptions, DocParserProfile,
    EmptyLayoutBehavior, FormulaNumberComposition, RecognitionBackend, RecognitionTask,
    RegionErrorBehavior,
};

#[derive(Debug)]
pub(crate) struct ParsedPage {
    pub structure: StructureResult,
    pub diagnostics: Vec<ParseDiagnostic>,
    pub raw_output: Option<String>,
    pub width: u32,
    pub height: u32,
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
    /// Centre captions and tables with inline HTML in markdown output, as the
    /// PaddleOCR-VL reference does.
    pub markdown_pretty: bool,
}

impl Default for DocParserConfig {
    fn default() -> Self {
        Self {
            // Keep detected region bounds unchanged by default.
            crop_pad_ratio: 0.0,
            max_tokens: 4096,
            skip_auxiliary_regions: true,
            skip_region_blocks: true,
            markdown_ignore_labels: crate::render::markdown::default_markdown_ignore_labels(),
            markdown_pretty: true,
        }
    }
}

/// Unified document parser combining layout detection with recognition backend.
pub struct DocParser<'a, B: RecognitionBackend + ?Sized> {
    backend: &'a B,
    config: DocParserConfig,
    config_override: Option<DocParserConfig>,
    options: DocParserOptions,
    profile: DocParserProfile,
    region_batch_size: usize,
    page_preprocessor: Option<std::panic::AssertUnwindSafe<&'a dyn crate::PagePreprocessor>>,
}

impl<'a, B: RecognitionBackend + ?Sized> DocParser<'a, B> {
    /// Create a new document parser with the given backend.
    pub fn new(backend: &'a B) -> Self {
        Self::build(backend, None)
    }

    /// Create a document parser with an explicit six-field configuration.
    pub fn with_config(backend: &'a B, config: DocParserConfig) -> Self {
        Self::build(backend, Some(config))
    }

    fn build(backend: &'a B, config_override: Option<DocParserConfig>) -> Self {
        let mut parser = Self {
            backend,
            config: DocParserConfig::default(),
            config_override,
            options: DocParserOptions::default(),
            profile: DocParserProfile::default(),
            region_batch_size: 1,
            page_preprocessor: None,
        };
        parser.resolve_options();
        parser
    }

    /// Apply document overrides above the backend profile and explicit config.
    pub fn with_options(mut self, options: DocParserOptions) -> Self {
        self.options = self.options.merge(&options);
        self.resolve_options();
        self
    }

    /// Set the page transform applied before layout detection.
    pub fn with_page_preprocessor(mut self, preprocessor: &'a dyn crate::PagePreprocessor) -> Self {
        self.page_preprocessor = Some(std::panic::AssertUnwindSafe(preprocessor));
        self
    }

    fn resolve_options(&mut self) {
        let mut profile = self
            .options
            .profile
            .clone()
            .unwrap_or_else(|| self.backend.doc_parser_profile());
        if self.options.profile.is_none()
            && self.options.merge_labels.is_none()
            && !self.backend.merge_adjacent_text_regions()
        {
            profile.merge_labels.clear();
        }
        if let Some(config) = &self.config_override {
            profile.crop_pad_ratio = config.crop_pad_ratio;
            profile.max_tokens = config.max_tokens;
            profile.skip_auxiliary_regions = config.skip_auxiliary_regions;
            profile.skip_region_blocks = config.skip_region_blocks;
            profile.markdown_ignore_labels = config.markdown_ignore_labels.clone();
            profile.markdown_pretty = config.markdown_pretty;
        }
        self.options.apply(&mut profile);
        self.config = DocParserConfig {
            crop_pad_ratio: profile.crop_pad_ratio,
            max_tokens: profile.max_tokens,
            skip_auxiliary_regions: profile.skip_auxiliary_regions,
            skip_region_blocks: profile.skip_region_blocks,
            markdown_ignore_labels: profile.markdown_ignore_labels.clone(),
            markdown_pretty: profile.markdown_pretty,
        };
        self.profile = profile;
    }

    fn skips_auxiliary(&self, label: &str, kind: LayoutElementType) -> bool {
        if let Some(skip) = self.options.skip_auxiliary_regions {
            return skip && is_auxiliary_element(kind);
        }
        if let Some(rules) = &self.options.auxiliary_labels
            && let Some(rule) = rules.iter().find(|rule| rule.label == label)
        {
            return rule.action == AuxiliaryLabelAction::Drop;
        }
        if let Some(config) = &self.config_override {
            return config.skip_auxiliary_regions && is_auxiliary_element(kind);
        }
        match self
            .profile
            .auxiliary_labels
            .iter()
            .find(|rule| rule.label == label)
        {
            Some(rule) => rule.action == AuxiliaryLabelAction::Drop,
            None => self.profile.skip_auxiliary_regions && is_auxiliary_element(kind),
        }
    }

    fn crop_region(
        &self,
        image: &RgbImage,
        bbox: &BoundingBox,
        polygon: Option<&BoundingBox>,
        figures: Option<(&super::table_figures::TableFigures, usize)>,
    ) -> Result<RgbImage, Error> {
        super::crop::crop_with_polygon_and_paint(
            image,
            [bbox.x_min(), bbox.y_min(), bbox.x_max(), bbox.y_max()].map(f64::from),
            polygon,
            &self.profile.crop,
            |crop, origin| {
                if let Some((figures, index)) = figures {
                    figures.paint(index, crop, origin);
                }
            },
        )
    }

    fn empty_layout_result(
        &self,
        input_path: Arc<str>,
        index: usize,
        image: Cow<'_, RgbImage>,
        raw_output: Option<String>,
        diagnostics: Vec<ParseDiagnostic>,
    ) -> Result<ParsedPage, Error> {
        let (width, height) = image.dimensions();
        let result = match self.profile.empty_layout {
            EmptyLayoutBehavior::Empty => StructureResult::new(input_path, index),
            EmptyLayoutBehavior::FullPageOcr => {
                self.recognize_full_image(input_path, index, image.into_owned())?
            }
        };
        Ok(ParsedPage {
            structure: result,
            diagnostics,
            raw_output,
            width,
            height,
        })
    }

    /// Render recognized regions with this parser's document settings.
    pub fn render_markdown(&self, result: &StructureResult) -> String {
        let mut ignored = self.config.markdown_ignore_labels.clone();
        if self.options.markdown_ignore_labels.is_none()
            && (self.options.auxiliary_labels.is_some() || self.config_override.is_none())
        {
            for rule in &self.profile.auxiliary_labels {
                match rule.action {
                    AuxiliaryLabelAction::HideAtRender if !ignored.contains(&rule.label) => {
                        ignored.push(rule.label.clone())
                    }
                    AuxiliaryLabelAction::Recognize => ignored.retain(|label| label != &rule.label),
                    _ => {}
                }
            }
        }
        let mut projected = std::borrow::Cow::Borrowed(result.layout_elements.as_slice());
        if self.profile.table_figure_tokens {
            let figures = super::table_figures::TableFigures::new(&projected);
            ignored.push(crate::render::table::TABLE_FIGURE_LABEL.into());
            for (index, element) in projected.to_mut().iter_mut().enumerate() {
                if figures.absorbed(index) {
                    element.label = Some(crate::render::table::TABLE_FIGURE_LABEL.into());
                }
            }
        }
        let elements = crate::render::markdown::assembly::compose_formula_numbers(
            &projected,
            self.profile.formula_numbers,
        );
        crate::render::markdown::to_markdown_with_style(
            &elements,
            &ignored,
            self.config.markdown_pretty,
            &self.profile.render_style,
        )
    }

    /// Set the maximum number of same-task regions sent to the recognition
    /// backend at once. A value of zero is treated as one.
    ///
    /// The default is one so existing callers retain their memory use and
    /// output behavior. PaddleOCR-VL, HunyuanOCR, and MinerU perform native
    /// batched inference; the other backends use a correct sequential path.
    /// PaddleOCR-VL and HunyuanOCR additionally group equal image-token counts,
    /// which keeps the common Metal decode path padding-free. Any padded native
    /// batch remains correct but uses the slower masked attention path.
    pub fn with_region_batch_size(mut self, size: usize) -> Self {
        self.region_batch_size = size.max(1);
        self
    }

    /// Return the configured cropped-region batch size.
    pub fn region_batch_size(&self) -> usize {
        self.region_batch_size
    }

    /// Returns a reference to the parser's configuration.
    pub fn config(&self) -> &DocParserConfig {
        &self.config
    }

    /// Parse a document image using layout-first pipeline.
    ///
    /// `layout` is any [`LayoutSource`]: [`PpDocLayout`](crate::PpDocLayout),
    /// which runs on Candle like the rest of this crate, or a detector of your
    /// own. Its regions are used in the order returned.
    pub fn parse<L: LayoutSource + ?Sized>(
        &self,
        layout: &L,
        image: RgbImage,
    ) -> Result<StructureResult, Error> {
        self.parse_with_path(layout, "<memory>", 0, image)
    }

    /// Parse a document image without layout detection (single full-image OCR).
    ///
    /// Use this for end-to-end models that handle layout internally.
    /// For models requiring separate layout detection, use [`parse`](Self::parse) instead.
    pub fn parse_without_layout(&self, image: RgbImage) -> Result<StructureResult, Error> {
        let image = super::page_preprocessing::prepare_page(
            Cow::Owned(image),
            self.page_preprocessor
                .as_ref()
                .map(|hook| hook.0)
                .or_else(|| self.backend.page_preprocessor()),
        )?;
        self.recognize_full_image("<memory>".into(), 0, image.into_owned())
    }

    /// Parse a document image with source path information.
    pub fn parse_with_path<L: LayoutSource + ?Sized>(
        &self,
        layout: &L,
        input_path: impl Into<Arc<str>>,
        index: usize,
        image: RgbImage,
    ) -> Result<StructureResult, Error> {
        self.parse_with_path_image(layout, input_path.into(), index, Cow::Owned(image))
            .map(|parsed| parsed.structure)
    }

    pub(crate) fn parse_image<L: LayoutSource + ?Sized>(
        &self,
        layout: &L,
        image: &RgbImage,
    ) -> Result<ParsedPage, Error> {
        self.parse_with_path_image(layout, "<memory>".into(), 0, Cow::Borrowed(image))
    }

    fn parse_with_path_image<L: LayoutSource + ?Sized>(
        &self,
        layout: &L,
        input_path: Arc<str>,
        index: usize,
        image: Cow<'_, RgbImage>,
    ) -> Result<ParsedPage, Error> {
        let image = super::page_preprocessing::prepare_page(
            image,
            self.page_preprocessor
                .as_ref()
                .map(|hook| hook.0)
                .or_else(|| self.backend.page_preprocessor()),
        )?;
        let (page_w, page_h) = (image.width() as f32, image.height() as f32);

        let mut prediction = match &self.profile.layout {
            Some(options) => layout.detect_with_options(&image, options)?,
            None => layout.detect(&image)?.into(),
        };
        let table_figure_pool = self.profile.table_figure_tokens.then(|| {
            prediction
                .detections
                .elements
                .iter()
                .map(|element| {
                    LayoutElement::new(
                        element.bbox.clone(),
                        LayoutElementType::from_label(&element.element_type),
                        element.score,
                    )
                    .with_label(element.element_type.clone())
                })
                .collect::<Vec<_>>()
        });
        super::geometry::layout_filter::filter_document_layout(
            &mut prediction,
            self.profile.layout.as_ref().unwrap_or(&Default::default()),
        );
        let detected = prediction.detections.elements;
        let mut polygons = Vec::with_capacity(detected.len());
        let mut crop_polygons = prediction.polygons.into_iter();

        // If no layout elements detected, run OCR on the whole image
        if detected.is_empty() {
            return self.empty_layout_result(
                input_path,
                index,
                image,
                prediction.raw_output,
                prediction.diagnostics,
            );
        }

        // Step 2: Filter and prepare elements
        let mut elements: Vec<LayoutElement> = Vec::with_capacity(detected.len());
        for element in detected {
            let polygon = crop_polygons.next().flatten();
            let element_type = LayoutElementType::from_label(&element.element_type);
            if self.config.skip_region_blocks && element_type == LayoutElementType::Region {
                continue;
            }
            if self.skips_auxiliary(&element.element_type, element_type) {
                continue;
            }

            elements.push(
                LayoutElement::new(element.bbox, element_type, element.score)
                    .with_label(element.element_type),
            );
            polygons.push(polygon);
        }

        if elements.is_empty() {
            return self.empty_layout_result(
                input_path,
                index,
                image,
                prediction.raw_output,
                prediction.diagnostics,
            );
        }

        // A backend may refine detector order using its trained page protocol.
        // Invalid predictions preserve every region and report a diagnostic.
        let mut diagnostics = prediction.diagnostics;
        if elements.len() > 1
            && let Some(order) =
                self.backend
                    .predict_reading_order(&image, &elements, self.config.max_tokens)?
        {
            let mut sorted_ids = order.clone();
            sorted_ids.sort_unstable();
            if sorted_ids.iter().copied().eq(0..elements.len()) {
                let mut slots: Vec<_> = elements.into_iter().map(Some).collect();
                polygons = order.iter().map(|&i| polygons[i].clone()).collect();
                elements = order
                    .into_iter()
                    .map(|index| slots[index].take().expect("validated permutation"))
                    .collect();
            } else {
                diagnostics.push(ParseDiagnostic {
                    block_index: None,
                    stage: "reading_order".into(),
                    message: "Invalid reading-order permutation; retained detector order".into(),
                });
            }
        }
        let mut sorted_elements = elements;
        assign_order_indices(&mut sorted_elements);

        // Step 4: Recognize each element with adjacent text-block merging.
        //
        // Vertically stack compatible adjacent text crops before recognition.
        // This improves recognition quality for fragmented detections.
        let merge_groups = if !self.profile.merge_labels.is_empty() {
            compute_adjacent_text_merge_groups_with_profile(&sorted_elements, &self.profile)
        } else {
            Vec::new()
        };
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

        struct RecognitionJob {
            element_index: usize,
            image: RgbImage,
            task: RecognitionTask,
            label: String,
            max_tokens: usize,
        }

        let mut jobs_by_batch: std::collections::BTreeMap<
            (usize, usize, u64),
            Vec<RecognitionJob>,
        > = std::collections::BTreeMap::new();
        let max_pending_jobs = self.region_batch_size.saturating_mul(8).max(8);
        let mut pending_jobs = 0usize;
        let mut tasks_by_element = vec![None; sorted_elements.len()];
        let mut generated_by_element: Vec<Option<Result<String, Error>>> =
            std::iter::repeat_with(|| None)
                .take(sorted_elements.len())
                .collect();
        let flush_batch = |jobs: &mut Vec<RecognitionJob>,
                           generated: &mut [Option<Result<String, Error>>]|
         -> Result<(), Error> {
            let max_tokens = jobs[0].max_tokens;
            let batch = std::mem::take(jobs);
            let element_indices: Vec<usize> = batch.iter().map(|job| job.element_index).collect();
            let tasks: Vec<RecognitionTask> = batch.iter().map(|job| job.task).collect();
            let (images, labels): (Vec<_>, Vec<_>) =
                batch.into_iter().map(|job| (job.image, job.label)).unzip();
            let labels: Vec<_> = labels.iter().map(String::as_str).collect();
            let results = self.backend.recognize_labeled_batch_with_options(
                images,
                &tasks,
                &labels,
                &GenerationOptions::new(max_tokens),
            )?;
            if results.len() != element_indices.len() {
                return Err(Error::InvalidInput {
                    message: format!(
                        "recognition backend returned {} results for {} inputs",
                        results.len(),
                        element_indices.len()
                    ),
                });
            }

            for (element_index, result) in element_indices.into_iter().zip(results) {
                generated[element_index] = Some(result);
            }
            Ok(())
        };
        let table_figures = table_figure_pool
            .as_ref()
            .map(|pool| super::table_figures::TableFigures::from_pool(&sorted_elements, pool));
        let mut merged_by_first: std::collections::HashMap<usize, bool> =
            std::collections::HashMap::new();
        for (idx, element) in sorted_elements.iter_mut().enumerate() {
            if let Some(first) = group_first_for_index.get(idx).and_then(|v| *v)
                && first != idx
            {
                // Drop non-first blocks only when a merge actually happens.
                // When the group is not merged (aspect_ratio >= 3), each block is recognized separately.
                if merged_by_first.get(&first).copied().unwrap_or(false) {
                    continue;
                }
            }

            // Determine task for element
            let label = element
                .label
                .as_deref()
                .unwrap_or(element.element_type.as_str());
            let Some(task) = self
                .backend
                .recognition_task_for_label(label, task_for_element_type(element.element_type))
            else {
                continue;
            };
            // Backends that cannot read charts leave those regions
            // unrecognized; the element stays in the document without text
            // rather than collecting whatever the model emits for a figure
            // its prompt tells it to ignore.
            if task == RecognitionTask::Chart && !self.backend.capabilities().supports_chart {
                continue;
            }

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
                    let crop = match self.crop_region(
                        &image,
                        &crop_bbox,
                        polygons.get(g_idx).and_then(Option::as_ref),
                        table_figures.as_ref().map(|figures| (figures, g_idx)),
                    ) {
                        Ok(crop) => crop,
                        Err(error) => {
                            diagnostics.push(ParseDiagnostic {
                                block_index: Some(g_idx),
                                stage: "crop".to_string(),
                                message: error.to_string(),
                            });
                            continue;
                        }
                    };
                    crops.push(crop);
                }
                if crops.is_empty() {
                    continue;
                }
                // Skip merging when the stacked crop would be too tall (h/w >= 3).
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
                    match self.crop_region(
                        &image,
                        &crop_bbox,
                        polygons.get(idx).and_then(Option::as_ref),
                        table_figures.as_ref().map(|figures| (figures, idx)),
                    ) {
                        Ok(crop) => crop,
                        Err(error) => {
                            diagnostics.push(ParseDiagnostic {
                                block_index: Some(idx),
                                stage: "crop".to_string(),
                                message: error.to_string(),
                            });
                            continue;
                        }
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
                match self.crop_region(
                    &image,
                    &crop_bbox,
                    polygons.get(idx).and_then(Option::as_ref),
                    table_figures.as_ref().map(|figures| (figures, idx)),
                ) {
                    Ok(cropped) => cropped,
                    Err(error) => {
                        diagnostics.push(ParseDiagnostic {
                            block_index: Some(idx),
                            stage: "crop".to_string(),
                            message: error.to_string(),
                        });
                        continue;
                    }
                }
            };

            // Apply preprocessing if needed
            if task == RecognitionTask::Formula
                && self.backend.capabilities().preprocess_formula_margin
            {
                cropped = crop_margin(&cropped);
            }

            tasks_by_element[idx] = Some(task);
            let max_tokens = self
                .backend
                .region_token_budget(label, task)
                .map_or(self.config.max_tokens, |budget| {
                    budget.min(self.config.max_tokens)
                });
            let batch_key = (
                recognition_task_index(task),
                max_tokens,
                self.backend.recognition_batch_key(&cropped, task),
            );
            {
                let jobs = jobs_by_batch.entry(batch_key).or_default();
                jobs.push(RecognitionJob {
                    element_index: idx,
                    image: cropped,
                    task,
                    label: label.to_string(),
                    max_tokens,
                });
                pending_jobs += 1;
                if jobs.len() == self.region_batch_size {
                    flush_batch(jobs, &mut generated_by_element)?;
                    pending_jobs -= self.region_batch_size;
                }
            }

            // Exact token-count bucketing can produce many sparse queues on a
            // page with varied region sizes. Bound retained crop memory and
            // flush the fullest queue when the cap is reached.
            if pending_jobs >= max_pending_jobs
                && let Some(key) = jobs_by_batch
                    .iter()
                    .filter(|(_, jobs)| !jobs.is_empty())
                    .max_by_key(|(_, jobs)| jobs.len())
                    .map(|(key, _)| *key)
            {
                let jobs = jobs_by_batch
                    .get_mut(&key)
                    .expect("selected recognition queue must exist");
                pending_jobs -= jobs.len();
                flush_batch(jobs, &mut generated_by_element)?;
            }
        }

        // Native VLM batches are most efficient when prompts/tasks are
        // homogeneous. Full queues and the global pending-job cap keep retained
        // crop memory bounded rather than proportional to all page regions.
        for jobs in jobs_by_batch.values_mut() {
            if !jobs.is_empty() {
                flush_batch(jobs, &mut generated_by_element)?;
            }
        }

        let mut tables: Vec<TableResult> = Vec::new();
        let mut table_indices = Vec::new();
        for (idx, generated) in generated_by_element.into_iter().enumerate() {
            let Some(generated) = generated else {
                continue;
            };
            let Some(task) = tasks_by_element[idx] else {
                continue;
            };
            let element = &mut sorted_elements[idx];
            let mut generated = match generated {
                Ok(value) => value,
                Err(error) if self.profile.region_errors == RegionErrorBehavior::KeepRegion => {
                    diagnostics.push(ParseDiagnostic {
                        block_index: Some(idx),
                        stage: "recognition".into(),
                        message: error.to_string(),
                    });
                    continue;
                }
                Err(error) => return Err(error),
            };
            if generated.trim().is_empty() {
                continue;
            }

            // Apply repetition truncation if needed. Table markup is
            // exempt: legitimate tables repeat identical rows well past
            // the threshold, and collapsing them yields invalid HTML.
            if let Some(options) = &self.profile.repetition {
                generated =
                    text::clean_repetition(&generated, task == RecognitionTask::Table, options);
            } else if self.backend.capabilities().truncate_repetitive_output
                && task != RecognitionTask::Table
            {
                generated = truncate_repetitive_content(&generated, 10, 10, 10);
            }

            // Apply post-processing based on task.
            //
            // Note: Table blocks should remain HTML/OTSL-derived HTML; do not run the text normalizer
            // on table markup.
            let label = element
                .label
                .as_deref()
                .unwrap_or(element.element_type.as_str());
            let processed = if let Some(native) = self
                .backend
                .postprocess_region_output(label, task, &generated)
            {
                native
            } else if task == RecognitionTask::Table {
                if self.backend.capabilities().table_output_is_otsl {
                    convert_otsl_to_html_with_options(&generated, &self.profile.table)
                } else {
                    generated.trim().to_string()
                }
            } else if task == RecognitionTask::Formula {
                text::format_formula_with_options(&generated, &self.profile.formula)
            } else if let Some(options) = &self.profile.delimiters {
                text::normalize_math_delimiters(&generated, options)
            } else {
                text::format_text(&generated)
            };

            if element.element_type == LayoutElementType::Table {
                table_indices.push(idx);
                tables.push(
                    TableResult::new(element.bbox.clone(), TableType::Unknown)
                        .with_html_structure(processed.clone()),
                );
            }

            element.text = Some(processed);
        }

        if let Some(figures) = &table_figures {
            for (&index, table) in table_indices.iter().zip(&mut tables) {
                if let Some(content) = sorted_elements[index].text.as_deref() {
                    let restored = figures.restore(index, content, &sorted_elements);
                    sorted_elements[index].text = Some(restored.clone());
                    table.html_structure = Some(restored);
                }
            }
        }

        if self.profile.render_style.normalize_bullets {
            let formulas = crate::render::markdown::assembly::compose_formula_numbers(
                &sorted_elements,
                self.profile.formula_numbers,
            );
            let projected = crate::render::markdown::assembly::prepare_text(
                &formulas,
                self.profile.render_style.merge_hyphenated_text,
                Some(crate::render::markdown::assembly::BulletAlignment {
                    page_width: page_w,
                    grid: self.profile.crop.coordinate_grid.unwrap_or(1000),
                    tolerance: 10.,
                }),
            );
            let numbered: Vec<_> = projected
                .iter()
                .filter(|element| {
                    element.label.as_deref() == Some("text")
                        && element.text.as_deref().is_some_and(|s| s.starts_with("- "))
                })
                .filter_map(|element| element.order_index)
                .collect();
            for element in &mut sorted_elements {
                if element
                    .order_index
                    .is_some_and(|index| numbered.contains(&index))
                    && element.label.as_deref() == Some("text")
                    && !element.text.as_deref().unwrap_or("").starts_with("- ")
                {
                    element.text = Some(format!("- {}", element.text.as_deref().unwrap_or("")));
                }
            }
        }

        let result = StructureResult::new(input_path, index)
            .with_layout_elements(sorted_elements)
            .with_tables(tables);
        Ok(ParsedPage {
            structure: result,
            diagnostics,
            raw_output: prediction.raw_output,
            width: page_w as u32,
            height: page_h as u32,
        })
    }

    /// Parse a document and convert to markdown.
    ///
    /// Uses [`DocParserConfig::markdown_ignore_labels`] and
    /// [`DocParserConfig::markdown_pretty`].
    pub fn parse_to_markdown<L: LayoutSource + ?Sized>(
        &self,
        layout: &L,
        image: RgbImage,
    ) -> Result<String, Error> {
        let result = self.parse(layout, image)?;
        Ok(self.render_markdown(&result))
    }

    fn recognize_full_image(
        &self,
        input_path: Arc<str>,
        index: usize,
        image: RgbImage,
    ) -> Result<StructureResult, Error> {
        let (page_w, page_h) = (image.width() as f32, image.height() as f32);
        let text = self.backend.recognize_with_options(
            image,
            RecognitionTask::Ocr,
            &GenerationOptions::new(self.config.max_tokens),
        )?;
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

const fn recognition_task_index(task: RecognitionTask) -> usize {
    match task {
        RecognitionTask::Ocr => 0,
        RecognitionTask::Table => 1,
        RecognitionTask::Formula => 2,
        RecognitionTask::Chart => 3,
    }
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

/// Document parser using PaddleOCR-VL backend.
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

fn compute_adjacent_text_merge_groups_with_profile(
    elements: &[LayoutElement],
    profile: &DocParserProfile,
) -> Vec<MergeGroup> {
    // Keep semantic regions in the sequence and apply the configured merge labels.
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
        let blocked = profile.merge_blocker_labels.as_ref().map_or_else(
            || NON_MERGE_LABELS.contains(&label),
            |labels| labels.iter().any(|v| v == label),
        );
        if blocked {
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
        blockers: Option<&[String]>,
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
            let label = other
                .label
                .as_deref()
                .unwrap_or(other.element_type.as_str());
            if blockers.is_some_and(|labels| !labels.iter().any(|v| v == label)) {
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
            && profile.merge_labels.iter().any(|label| label == curr_label)
            && curr_label == prev_label
            && curr_rect.0 > prev_rect.2
            && curr_rect.1 < prev_rect.3
            && (curr_rect.0 - prev_rect.2) < prev_w.max(curr_w) * 0.3;

        let left_aligned = is_aligned(curr_rect.0, prev_rect.0);
        let right_aligned = is_aligned(curr_rect.2, prev_rect.2);

        let is_updown_align = iou_h > 0.0
            && profile.merge_labels.iter().any(|label| label == curr_label)
            && curr_label == prev_label
            && curr_rect.3 >= prev_rect.1
            && (curr_rect.1 - prev_rect.3).abs() < prev_h.max(curr_h) * 0.5
            && (left_aligned ^ right_aligned)
            && overlapwith_other_box(
                idx,
                prev_idx,
                elements,
                profile.merge_blocker_labels.as_deref(),
            );

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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layout::LayoutDetectionElement;
    use crate::layout::StaticLayout;
    use std::cell::{Cell, RefCell};

    /// Records the crops it is handed and echoes back a per-call marker.
    struct RecordingBackend {
        crops: RefCell<Vec<(u32, u32)>>,
    }

    impl RecognitionBackend for RecordingBackend {
        fn recognize(
            &self,
            image: RgbImage,
            _task: RecognitionTask,
            _max_tokens: usize,
        ) -> Result<String, Error> {
            let mut crops = self.crops.borrow_mut();
            crops.push((image.width(), image.height()));
            Ok(format!("region {}", crops.len()))
        }
    }

    struct BatchRecordingBackend {
        batch_sizes: RefCell<Vec<usize>>,
    }

    impl RecognitionBackend for BatchRecordingBackend {
        fn recognize(
            &self,
            _image: RgbImage,
            _task: RecognitionTask,
            _max_tokens: usize,
        ) -> Result<String, Error> {
            panic!("native batch hook should be used")
        }

        fn recognize_batch(
            &self,
            images: Vec<RgbImage>,
            tasks: &[RecognitionTask],
            _max_tokens: usize,
        ) -> crate::error::BatchResult<String> {
            assert_eq!(images.len(), tasks.len());
            self.batch_sizes.borrow_mut().push(images.len());
            Ok(images
                .into_iter()
                .map(|_| Ok("batched region".to_string()))
                .collect())
        }

        fn recognition_batch_key(&self, image: &RgbImage, _task: RecognitionTask) -> u64 {
            image.width() as u64
        }
    }

    struct SparseBatchKeyBackend {
        keys_seen: Cell<usize>,
        first_flush_at: Cell<Option<usize>>,
    }

    struct FailingBatchBackend;

    impl RecognitionBackend for FailingBatchBackend {
        fn recognize(
            &self,
            _image: RgbImage,
            _task: RecognitionTask,
            _max_tokens: usize,
        ) -> Result<String, Error> {
            panic!("native batch hook should be used")
        }

        fn recognize_batch(
            &self,
            _images: Vec<RgbImage>,
            _tasks: &[RecognitionTask],
            _max_tokens: usize,
        ) -> crate::error::BatchResult<String> {
            Err(Error::Config {
                message: "native batch failed".to_string(),
            })
        }
    }

    impl RecognitionBackend for SparseBatchKeyBackend {
        fn recognize(
            &self,
            _image: RgbImage,
            _task: RecognitionTask,
            _max_tokens: usize,
        ) -> Result<String, Error> {
            panic!("native batch hook should be used")
        }

        fn recognize_batch(
            &self,
            images: Vec<RgbImage>,
            tasks: &[RecognitionTask],
            _max_tokens: usize,
        ) -> crate::error::BatchResult<String> {
            assert_eq!(images.len(), tasks.len());
            if self.first_flush_at.get().is_none() {
                self.first_flush_at.set(Some(self.keys_seen.get()));
            }
            Ok(images
                .into_iter()
                .map(|_| Ok("bounded region".to_string()))
                .collect())
        }

        fn recognition_batch_key(&self, image: &RgbImage, _task: RecognitionTask) -> u64 {
            self.keys_seen.set(self.keys_seen.get() + 1);
            image.width() as u64
        }
    }

    fn element(label: &str, x1: f32, y1: f32, x2: f32, y2: f32) -> LayoutDetectionElement {
        LayoutDetectionElement {
            bbox: BoundingBox::from_coords(x1, y1, x2, y2),
            element_type: label.to_string(),
            score: 0.9,
        }
    }

    /// The parser must run end to end on a caller-supplied `LayoutSource`,
    /// which is the path that needs no ONNX Runtime.
    /// Truncation-enabled backend echoing a table whose rows repeat
    /// identically — the shape text-level repetition truncation must not
    /// touch.
    struct TableEchoBackend {
        crops: RefCell<Vec<RgbImage>>,
    }

    impl RecognitionBackend for TableEchoBackend {
        fn recognition_task_for_label(
            &self,
            label: &str,
            suggested: Option<RecognitionTask>,
        ) -> Option<RecognitionTask> {
            if label == "seal" {
                Some(RecognitionTask::Ocr)
            } else {
                suggested
            }
        }

        fn recognize(
            &self,
            image: RgbImage,
            task: RecognitionTask,
            _max_tokens: usize,
        ) -> Result<String, Error> {
            self.crops.borrow_mut().push(image);
            if task == RecognitionTask::Table {
                let mut rows = String::new();
                for _ in 0..12 {
                    rows.push_str("<tr><td>[F2] [F3]</td><td>[F404]</td></tr>\n");
                }
                Ok(format!("<table>\n{rows}</table>"))
            } else {
                Ok("plain text\n".repeat(12))
            }
        }

        fn capabilities(&self) -> crate::api::recognition::BackendCapabilities {
            crate::api::recognition::BackendCapabilities {
                truncate_repetitive_output: true,
                ..crate::api::recognition::BackendCapabilities::default()
            }
        }
    }

    /// A backend that cannot read charts: chart regions must be left as
    /// placeholders instead of reaching the model.
    struct NoChartBackend;

    impl RecognitionBackend for NoChartBackend {
        fn recognize(
            &self,
            _image: RgbImage,
            task: RecognitionTask,
            _max_tokens: usize,
        ) -> Result<String, Error> {
            match task {
                RecognitionTask::Chart => {
                    panic!("chart regions must not reach a backend without chart support")
                }
                other => Ok(format!("recognized {other:?}")),
            }
        }

        fn capabilities(&self) -> crate::api::recognition::BackendCapabilities {
            crate::api::recognition::BackendCapabilities {
                supports_chart: false,
                ..crate::api::recognition::BackendCapabilities::default()
            }
        }
    }

    #[test]
    fn backend_order_and_native_formatting_preserve_regions_and_code() {
        struct NativeBackend {
            invalid: bool,
            inference_error: bool,
        }
        impl RecognitionBackend for NativeBackend {
            fn recognize(
                &self,
                image: RgbImage,
                _: RecognitionTask,
                _: usize,
            ) -> Result<String, Error> {
                assert!(image.pixels().any(|p| p.0 == [0; 3]));
                assert!(image.pixels().any(|p| p.0 == [255; 3]));
                Ok("```\n    a_____b....c\n```".into())
            }
            fn predict_reading_order(
                &self,
                _: &RgbImage,
                _: &[LayoutElement],
                _: usize,
            ) -> Result<Option<Vec<usize>>, Error> {
                if self.inference_error {
                    return Err(Error::Inference {
                        model_name: "test".into(),
                        context: "reading-order inference failed".into(),
                        source: Box::new(std::io::Error::other("device failure")),
                    });
                }
                Ok(Some(if self.invalid { vec![0, 0] } else { vec![1, 0] }))
            }
            fn doc_parser_profile(&self) -> DocParserProfile {
                DocParserProfile::default()
                    .with_layout(Some(crate::LayoutPostprocessOptions::default()))
                    .with_crop(
                        crate::pipeline::crop::CropOptions::default()
                            .with_shape_mode(crate::pipeline::layout::LayoutShapeMode::Poly),
                    )
            }
            fn merge_adjacent_text_regions(&self) -> bool {
                false
            }
            fn postprocess_region_output(
                &self,
                _: &str,
                _: RecognitionTask,
                raw: &str,
            ) -> Option<String> {
                Some(raw.to_string())
            }
        }
        struct PolygonLayout(StaticLayout);
        impl LayoutSource for PolygonLayout {
            fn detect(&self, image: &RgbImage) -> Result<crate::LayoutDetections, Error> {
                self.0.detect(image)
            }
            fn detect_with_options(
                &self,
                image: &RgbImage,
                _: &crate::LayoutPostprocessOptions,
            ) -> Result<crate::LayoutPrediction, Error> {
                Ok(crate::LayoutPrediction {
                    detections: self.detect(image)?,
                    polygons: vec![
                        Some(BoundingBox::from_coords(20., 10., 180., 90.)),
                        Some(BoundingBox::from_coords(20., 100., 180., 180.)),
                    ],
                    raw_output: Some("layout trace".into()),
                    ..Default::default()
                })
            }
        }
        let layout = PolygonLayout(StaticLayout::new(vec![
            element("text", 10., 10., 190., 90.),
            element("code", 10., 100., 190., 180.),
        ]));
        for invalid in [false, true] {
            let backend = NativeBackend {
                invalid,
                inference_error: false,
            };
            let parser = DocParser::new(&backend);
            let ParsedPage {
                structure: output,
                diagnostics,
                raw_output,
                ..
            } = parser
                .parse_image(&layout, &RgbImage::new(200, 200))
                .unwrap();
            assert_eq!(raw_output.as_deref(), Some("layout trace"));
            assert_eq!(output.layout_elements.len(), 2);
            assert_eq!(
                output.layout_elements[0].label.as_deref(),
                Some(if invalid { "text" } else { "code" })
            );
            assert_eq!(
                output.layout_elements[0].text.as_deref(),
                Some("```\n    a_____b....c\n```")
            );
            assert_eq!(diagnostics.len(), usize::from(invalid));
            if invalid {
                assert_eq!(diagnostics[0].stage, "reading_order");
            }
        }
        let backend = NativeBackend {
            invalid: false,
            inference_error: true,
        };
        let error = DocParser::new(&backend)
            .parse_image(&layout, &RgbImage::new(200, 200))
            .unwrap_err();
        assert!(matches!(error, Error::Inference { .. }));
        let formula = LayoutElement::new(
            BoundingBox::from_coords(0., 0., 100., 30.),
            LayoutElementType::Formula,
            1.,
        )
        .with_label("display_formula")
        .with_text("$$\nx=1\n$$");
        let number = LayoutElement::new(
            BoundingBox::from_coords(100., 0., 120., 30.),
            LayoutElementType::FormulaNumber,
            1.,
        )
        .with_label("formula_number")
        .with_text("(3)");
        let result = StructureResult::new("<memory>", 0)
            .with_layout_elements(vec![number.clone(), formula.clone()]);
        let parser = DocParser::new(&backend).with_options(
            DocParserOptions::default().with_formula_numbers(FormulaNumberComposition::AdjacentTag),
        );
        assert_eq!(parser.render_markdown(&result), "$$\nx=1 \\tag{3}\n$$");
        assert_eq!(result.layout_elements.len(), 2);
        assert_eq!(
            result.layout_elements[1].text.as_deref(),
            Some("$$\nx=1\n$$")
        );
        let result =
            StructureResult::new("<memory>", 0).with_layout_elements(vec![formula, number]);
        let parser = parser.with_options(
            DocParserOptions::default()
                .with_formula_numbers(FormulaNumberComposition::FollowingStarTag),
        );
        assert_eq!(parser.render_markdown(&result), "$$\nx=1\n \\tag*{(3)}$$");
        let parser = parser.with_options(
            DocParserOptions::default()
                .with_formula_numbers(FormulaNumberComposition::SeparateParagraph)
                .with_markdown_ignore_labels(Vec::new()),
        );
        assert_eq!(parser.render_markdown(&result), "$$\nx=1\n$$\n\n(3)");
    }

    #[test]
    fn borrowed_backends_receive_native_region_labels_and_can_enable_seals() {
        struct LabeledBackend;
        impl RecognitionBackend for LabeledBackend {
            fn recognize(
                &self,
                _: RgbImage,
                _: RecognitionTask,
                _: usize,
            ) -> Result<String, Error> {
                panic!("labeled dispatch expected");
            }
            fn recognition_task_for_label(
                &self,
                label: &str,
                suggested: Option<RecognitionTask>,
            ) -> Option<RecognitionTask> {
                if matches!(label, "seal" | "inline_formula") {
                    Some(RecognitionTask::Ocr)
                } else {
                    suggested
                }
            }
            fn recognize_labeled_batch_with_options(
                &self,
                images: Vec<RgbImage>,
                tasks: &[RecognitionTask],
                labels: &[&str],
                _: &GenerationOptions,
            ) -> crate::BatchResult<String> {
                assert_eq!(images.len(), labels.len());
                assert!(tasks.iter().all(|task| *task == RecognitionTask::Ocr));
                Ok(labels
                    .iter()
                    .map(|label| Ok(format!("recognized {label}")))
                    .collect())
            }
        }
        let backend = LabeledBackend;
        let parser = DocParser::new(&backend);
        let layout = StaticLayout::new(vec![
            element("seal", 10., 10., 190., 90.),
            element("inline_formula", 10., 100., 190., 180.),
        ]);
        let output = parser.parse(&layout, RgbImage::new(200, 200)).unwrap();
        assert_eq!(
            output.layout_elements[0].text.as_deref(),
            Some("recognized seal")
        );
        assert_eq!(
            output.layout_elements[1].text.as_deref(),
            Some("recognized inline_formula")
        );
    }

    #[test]
    fn unsupported_chart_regions_are_skipped_not_recognized() {
        assert!(crate::api::recognition::BackendCapabilities::default().supports_chart);
        let backend = NoChartBackend;
        let parser = DocParser::new(&backend);
        let layout = StaticLayout::new(vec![
            element("chart", 10.0, 10.0, 190.0, 100.0),
            element("text", 10.0, 110.0, 190.0, 210.0),
        ]);

        let result = parser
            .parse(&layout, RgbImage::new(200, 210))
            .expect("parse succeeds");

        // The chart element stays in the document, unrecognized; the text
        // region still runs.
        assert_eq!(result.layout_elements.len(), 2);
        assert_eq!(result.layout_elements[0].text, None);
        assert_eq!(
            result.layout_elements[1].text.as_deref(),
            Some("recognized Ocr")
        );
    }

    #[test]
    fn table_output_survives_repetition_truncation() {
        let backend = TableEchoBackend {
            crops: RefCell::new(Vec::new()),
        };
        let parser = DocParser::new(&backend);
        let layout = StaticLayout::new(vec![
            element("table", 10.0, 10.0, 190.0, 100.0),
            element("text", 10.0, 110.0, 190.0, 210.0),
        ]);

        let result = parser
            .parse(&layout, RgbImage::new(200, 210))
            .expect("parse succeeds");

        // The table keeps every row: collapsing identical rows would
        // yield invalid HTML.
        let table_text = result.layout_elements[0].text.as_deref().unwrap();
        assert_eq!(
            table_text.matches("<tr>").count(),
            12,
            "table rows must not be collapsed by repetition truncation"
        );
        // Non-table text with the same repetition still gets truncated.
        let text = result.layout_elements[1].text.as_deref().unwrap();
        assert!(
            !text.contains("plain text\nplain text"),
            "repeated text should still be truncated"
        );
    }

    #[test]
    fn table_figure_tokens_follow_paddlex() {
        let backend = TableEchoBackend {
            crops: RefCell::new(Vec::new()),
        };
        let profile = crate::DocParserProfile::default()
            .with_table_figure_tokens(true)
            .with_layout(Some(
                crate::pipeline::layout::LayoutPostprocessOptions::default()
                    .with_visual_labels(vec!["image".into(), "seal".into()])
                    .with_min_size(6.),
            ))
            .with_render_style(
                crate::render::markdown::RenderStyle::default()
                    .with_image_template(Some("FIGURE-{index}".into())),
            );
        let parser = DocParser::new(&backend)
            .with_options(crate::DocParserOptions::default().with_profile(profile));
        let layout = StaticLayout::new(vec![
            element("table", 10., 10., 190., 100.),
            element("image", 30., 30., 70., 70.),
        ]);
        let result = parser.parse(&layout, RgbImage::new(200, 210)).unwrap();
        let html = result.layout_elements[0].text.as_deref().unwrap();
        assert!(html.contains("<img src=\"image_2.png\" alt=\"Image\" />"));
        assert!(html.contains("[F404]"));
        assert!(!html.contains("[F2]"));
        assert_eq!(result.layout_elements.len(), 2);
        assert_eq!(result.tables[0].html_structure.as_deref(), Some(html));
        assert!(!parser.render_markdown(&result).contains("FIGURE-"));
        let crops = backend.crops.borrow();
        let crop = crops.last().unwrap();
        assert_eq!(crop.get_pixel(20, 20).0, [255; 3]);
        assert_eq!(crop.get_pixel(10, 10).0, [0; 3]);
        drop(crops);

        let tiny = StaticLayout::new(vec![
            element("table", 10., 10., 190., 100.),
            element("image", 30., 30., 40., 40.),
        ]);
        let result = parser.parse(&tiny, RgbImage::new(200, 210)).unwrap();
        assert!(
            result.layout_elements[0]
                .text
                .as_deref()
                .unwrap()
                .contains("[F2]")
        );
        assert!(
            backend
                .crops
                .borrow()
                .last()
                .unwrap()
                .pixels()
                .all(|p| p.0 == [0; 3])
        );
        assert!(!parser.render_markdown(&result).contains("FIGURE-"));
        let fractional = StaticLayout::new(vec![
            element("table", 10.8, 10.8, 190.8, 100.8),
            element("image", 30.8, 30.8, 70.8, 70.8),
        ]);
        parser.parse(&fractional, RgbImage::new(200, 210)).unwrap();
        let crops = backend.crops.borrow();
        let crop = crops.last().unwrap();
        assert_eq!(crop.get_pixel(19, 19).0, [255; 3]);
        assert_eq!(crop.get_pixel(18, 19).0, [0; 3]);
        drop(crops);
        let seal = StaticLayout::new(vec![
            element("table", 10., 10., 190., 100.),
            element("seal", 30., 30., 70., 70.),
        ]);
        let result = parser.parse(&seal, RgbImage::new(200, 210)).unwrap();
        let html = result.layout_elements[0].text.as_deref().unwrap();
        assert!(html.contains("image_2.png"));
        assert!(html.contains("plain text"));
        let discarded = StaticLayout::new(vec![
            element("table", 10., 10., 190., 100.),
            element("image", 30., 30., 70., 70.),
            element("image", 195., 30., 199., 34.),
        ]);
        let result = parser.parse(&discarded, RgbImage::new(200, 210)).unwrap();
        let html = result.layout_elements[0].text.as_deref().unwrap();
        assert!(html.contains("image_2.png"));
        assert!(!html.contains("[F3]"));
        assert!(html.contains("[F2]"));
        let header = StaticLayout::new(vec![
            element("table", 10., 10., 190., 100.),
            element("header_image", 30., 30., 70., 70.),
        ]);
        let result = parser.parse(&header, RgbImage::new(200, 210)).unwrap();
        assert!(
            result.layout_elements[0]
                .text
                .as_deref()
                .unwrap()
                .contains("[F2]")
        );
    }

    #[test]
    fn parses_with_a_static_layout_source() {
        let backend = RecordingBackend {
            crops: RefCell::new(Vec::new()),
        };
        let parser = DocParser::new(&backend);
        let layout = StaticLayout::new(vec![
            element("doc_title", 10.0, 10.0, 190.0, 40.0),
            element("text", 10.0, 60.0, 190.0, 140.0),
        ]);

        let result = parser
            .parse(&layout, RgbImage::new(200, 200))
            .expect("parse succeeds");

        assert_eq!(result.layout_elements.len(), 2);
        assert_eq!(backend.crops.borrow().len(), 2);
        assert_eq!(
            result.layout_elements[0].element_type,
            LayoutElementType::DocTitle
        );
    }

    #[test]
    fn batches_same_task_regions_and_preserves_element_results() {
        let backend = BatchRecordingBackend {
            batch_sizes: RefCell::new(Vec::new()),
        };
        let parser = DocParser::new(&backend).with_region_batch_size(2);
        let layout = StaticLayout::new(vec![
            element("doc_title", 10.0, 0.0, 190.0, 20.0),
            element("paragraph_title", 10.0, 30.0, 190.0, 50.0),
            element("content", 10.0, 60.0, 190.0, 80.0),
            element("abstract", 10.0, 90.0, 190.0, 110.0),
            element("list", 10.0, 120.0, 190.0, 140.0),
        ]);

        let result = parser
            .parse(&layout, RgbImage::new(200, 160))
            .expect("parse succeeds");

        assert_eq!(*backend.batch_sizes.borrow(), vec![2, 2, 1]);
        assert_eq!(result.layout_elements.len(), 5);
        assert!(
            result
                .layout_elements
                .iter()
                .all(|element| element.text.as_deref() == Some("batched region"))
        );
    }

    #[test]
    fn separates_same_task_regions_with_different_batch_keys() {
        let backend = BatchRecordingBackend {
            batch_sizes: RefCell::new(Vec::new()),
        };
        let parser = DocParser::new(&backend).with_region_batch_size(2);
        let layout = StaticLayout::new(vec![
            element("doc_title", 0.0, 0.0, 100.0, 20.0),
            element("paragraph_title", 0.0, 30.0, 120.0, 50.0),
            element("content", 0.0, 60.0, 100.0, 80.0),
            element("abstract", 0.0, 90.0, 120.0, 110.0),
        ]);

        let result = parser
            .parse(&layout, RgbImage::new(140, 120))
            .expect("parse succeeds");

        assert_eq!(*backend.batch_sizes.borrow(), vec![2, 2]);
        assert_eq!(result.layout_elements.len(), 4);
    }

    #[test]
    fn sparse_batch_keys_flush_before_retaining_the_whole_page() {
        let backend = SparseBatchKeyBackend {
            keys_seen: Cell::new(0),
            first_flush_at: Cell::new(None),
        };
        let parser = DocParser::new(&backend).with_region_batch_size(2);
        let layout = StaticLayout::new(
            (0..20)
                .map(|index| {
                    let width = 20.0 + index as f32;
                    let top = index as f32 * 3.0;
                    element("content", 0.0, top, width, top + 2.0)
                })
                .collect(),
        );

        let result = parser
            .parse(&layout, RgbImage::new(64, 64))
            .expect("parse succeeds");

        assert_eq!(result.layout_elements.len(), 20);
        assert_eq!(backend.first_flush_at.get(), Some(16));
    }

    #[test]
    fn propagates_one_batch_error_without_rewriting_it_per_item() {
        let parser = DocParser::new(&FailingBatchBackend).with_region_batch_size(2);
        let layout = StaticLayout::new(vec![
            element("text", 0.0, 0.0, 50.0, 20.0),
            element("text", 0.0, 30.0, 50.0, 50.0),
        ]);

        let error = parser
            .parse(&layout, RgbImage::new(64, 64))
            .expect_err("batch error must fail parsing");
        assert!(matches!(
            error,
            Error::Config { message } if message == "native batch failed"
        ));
    }

    /// An empty detection list falls back to whole-page recognition.
    #[test]
    fn falls_back_to_full_image_when_layout_is_empty() {
        let backend = RecordingBackend {
            crops: RefCell::new(Vec::new()),
        };
        let parser = DocParser::new(&backend);

        let result = parser
            .parse(&StaticLayout::new(Vec::new()), RgbImage::new(120, 80))
            .expect("parse succeeds");

        assert_eq!(result.layout_elements.len(), 1);
        assert_eq!(*backend.crops.borrow(), vec![(120, 80)]);
    }

    /// `parse_to_markdown` and `StructureResult::to_markdown` must drop the
    /// same labels; they used to disagree, so a page header reached the prose
    /// through one path but not the other.
    #[test]
    fn both_markdown_entry_points_skip_the_same_labels() {
        let backend = RecordingBackend {
            crops: RefCell::new(Vec::new()),
        };
        let config = DocParserConfig {
            // Keep the auxiliary regions so they reach the renderer at all.
            skip_auxiliary_regions: false,
            ..DocParserConfig::default()
        };
        let parser = DocParser::with_config(&backend, config);
        let layout = StaticLayout::new(vec![
            element("header", 10.0, 5.0, 190.0, 20.0),
            element("text", 10.0, 60.0, 190.0, 140.0),
        ]);

        let result = parser
            .parse(&layout, RgbImage::new(200, 200))
            .expect("parse succeeds");
        assert!(
            result
                .layout_elements
                .iter()
                .any(|e| e.element_type == LayoutElementType::Header),
            "the header must survive parsing for this test to mean anything"
        );

        let from_config = crate::render::markdown::to_markdown(
            &result.layout_elements,
            &parser.config().markdown_ignore_labels,
            parser.config().markdown_pretty,
        );
        assert_eq!(result.to_markdown(), from_config);
    }
}
