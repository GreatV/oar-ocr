//! Layout Detection Adapter
//!
//! This module provides adapters for layout detection models.

use crate::core::inference::OrtInfer;
use crate::core::traits::{
    adapter::{AdapterBuilder, AdapterInfo, ModelAdapter},
    task::Task,
};
use crate::core::{OCRError, TaskType, Tensor4D};
use crate::domain::tasks::{
    LayoutDetectionConfig, LayoutDetectionOutput, LayoutDetectionTask, LayoutElement,
};
use crate::models::detection::{
    PPDocLayoutModel, PPDocLayoutModelBuilder, PPDocLayoutPostprocessConfig, PicoDetModel,
    PicoDetModelBuilder, PicoDetPostprocessConfig, RTDetrModel, RTDetrModelBuilder,
    RTDetrPostprocessConfig,
};
use crate::processors::LayoutPostProcess;
use std::collections::HashMap;
use std::path::Path;

/// Configuration for layout detection models.
#[derive(Debug, Clone)]
pub struct LayoutModelConfig {
    /// Model name
    pub model_name: String,
    /// Number of classes
    pub num_classes: usize,
    /// Class label mapping (class_id -> label string)
    pub class_labels: HashMap<usize, String>,
    /// Model type (e.g., "picodet", "rtdetr", "pp-doclayout")
    pub model_type: String,
    /// Optional fixed input image size (height, width)
    pub input_size: Option<(u32, u32)>,
}

impl LayoutModelConfig {
    /// Create configuration for PicoDet layout 1x model (5 classes).
    pub fn picodet_layout_1x() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "text".to_string());
        class_labels.insert(1, "title".to_string());
        class_labels.insert(2, "list".to_string());
        class_labels.insert(3, "table".to_string());
        class_labels.insert(4, "figure".to_string());

        Self {
            model_name: "picodet_layout_1x".to_string(),
            num_classes: 5,
            class_labels,
            model_type: "picodet".to_string(),
            input_size: Some((800, 608)),
        }
    }

    /// Create configuration for PicoDet layout 1x table-only model.
    pub fn picodet_layout_1x_table() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "table".to_string());

        Self {
            model_name: "picodet_layout_1x_table".to_string(),
            num_classes: 1,
            class_labels,
            model_type: "picodet".to_string(),
            input_size: Some((800, 608)),
        }
    }

    /// Create configuration for PicoDet-S layout 3 class model.
    pub fn picodet_s_layout_3cls() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "image".to_string());
        class_labels.insert(1, "table".to_string());
        class_labels.insert(2, "seal".to_string()); // seal treated as separate class

        Self {
            model_name: "picodet-s_layout_3cls".to_string(),
            num_classes: 3,
            class_labels,
            model_type: "picodet".to_string(),
            input_size: Some((480, 480)),
        }
    }

    /// Create configuration for PicoDet-L layout 3 class model.
    pub fn picodet_l_layout_3cls() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "image".to_string());
        class_labels.insert(1, "table".to_string());
        class_labels.insert(2, "seal".to_string());

        Self {
            model_name: "picodet-l_layout_3cls".to_string(),
            num_classes: 3,
            class_labels,
            model_type: "picodet".to_string(),
            input_size: Some((640, 640)),
        }
    }

    /// Create configuration for PicoDet-S layout 17 class model.
    pub fn picodet_s_layout_17cls() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "paragraph_title".to_string());
        class_labels.insert(1, "image".to_string());
        class_labels.insert(2, "text".to_string());
        class_labels.insert(3, "number".to_string());
        class_labels.insert(4, "abstract".to_string());
        class_labels.insert(5, "content".to_string());
        class_labels.insert(6, "figure_title".to_string());
        class_labels.insert(7, "formula".to_string());
        class_labels.insert(8, "table".to_string());
        class_labels.insert(9, "table_title".to_string());
        class_labels.insert(10, "reference".to_string());
        class_labels.insert(11, "doc_title".to_string());
        class_labels.insert(12, "footnote".to_string());
        class_labels.insert(13, "header".to_string());
        class_labels.insert(14, "algorithm".to_string());
        class_labels.insert(15, "footer".to_string());
        class_labels.insert(16, "seal".to_string());

        Self {
            model_name: "picodet-s_layout_17cls".to_string(),
            num_classes: 17,
            class_labels,
            model_type: "picodet".to_string(),
            input_size: Some((480, 480)),
        }
    }

    /// Create configuration for PicoDet-L layout 17 class model.
    pub fn picodet_l_layout_17cls() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "paragraph_title".to_string());
        class_labels.insert(1, "image".to_string());
        class_labels.insert(2, "text".to_string());
        class_labels.insert(3, "number".to_string());
        class_labels.insert(4, "abstract".to_string());
        class_labels.insert(5, "content".to_string());
        class_labels.insert(6, "figure_title".to_string());
        class_labels.insert(7, "formula".to_string());
        class_labels.insert(8, "table".to_string());
        class_labels.insert(9, "table_title".to_string());
        class_labels.insert(10, "reference".to_string());
        class_labels.insert(11, "doc_title".to_string());
        class_labels.insert(12, "footnote".to_string());
        class_labels.insert(13, "header".to_string());
        class_labels.insert(14, "algorithm".to_string());
        class_labels.insert(15, "footer".to_string());
        class_labels.insert(16, "seal".to_string());

        Self {
            model_name: "picodet-l_layout_17cls".to_string(),
            num_classes: 17,
            class_labels,
            model_type: "picodet".to_string(),
            input_size: Some((640, 640)),
        }
    }

    /// Create configuration for RT-DETR-H layout 3 class model.
    pub fn rtdetr_h_layout_3cls() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "figure".to_string()); // image
        class_labels.insert(1, "table".to_string());
        class_labels.insert(2, "seal".to_string()); // seal

        Self {
            model_name: "rt-detr-h_layout_3cls".to_string(),
            num_classes: 3,
            class_labels,
            model_type: "rtdetr".to_string(),
            input_size: Some((640, 640)),
        }
    }

    /// Create configuration for RT-DETR-H layout 17 class model.
    pub fn rtdetr_h_layout_17cls() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "paragraph_title".to_string());
        class_labels.insert(1, "image".to_string());
        class_labels.insert(2, "text".to_string());
        class_labels.insert(3, "number".to_string());
        class_labels.insert(4, "abstract".to_string());
        class_labels.insert(5, "content".to_string());
        class_labels.insert(6, "figure_title".to_string());
        class_labels.insert(7, "formula".to_string());
        class_labels.insert(8, "table".to_string());
        class_labels.insert(9, "table_title".to_string());
        class_labels.insert(10, "reference".to_string());
        class_labels.insert(11, "doc_title".to_string());
        class_labels.insert(12, "footnote".to_string());
        class_labels.insert(13, "header".to_string());
        class_labels.insert(14, "algorithm".to_string());
        class_labels.insert(15, "footer".to_string());
        class_labels.insert(16, "seal".to_string());

        Self {
            model_name: "rt-detr-h_layout_17cls".to_string(),
            num_classes: 17,
            class_labels,
            model_type: "rtdetr".to_string(),
            input_size: Some((640, 640)),
        }
    }

    /// Create configuration for PP-DocBlockLayout model (1 class: Region).
    /// This model uses 640x640 input size and only detects generic regions.
    pub fn pp_docblocklayout() -> Self {
        let mut class_labels = HashMap::new();
        // PP-DocBlockLayout has only 1 class: Region (generic layout block)
        class_labels.insert(0, "region".to_string());

        Self {
            model_name: "pp-docblocklayout".to_string(),
            num_classes: 1,
            class_labels,
            model_type: "pp-doclayout".to_string(),
            input_size: Some((640, 640)),
        }
    }

    /// Create configuration for PP-DocLayout-S model (23 classes).
    pub fn pp_doclayout_s() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "paragraph_title".to_string());
        class_labels.insert(1, "image".to_string());
        class_labels.insert(2, "text".to_string());
        class_labels.insert(3, "number".to_string());
        class_labels.insert(4, "abstract".to_string());
        class_labels.insert(5, "content".to_string());
        class_labels.insert(6, "figure_title".to_string());
        class_labels.insert(7, "formula".to_string());
        class_labels.insert(8, "table".to_string());
        class_labels.insert(9, "table_title".to_string());
        class_labels.insert(10, "reference".to_string());
        class_labels.insert(11, "doc_title".to_string());
        class_labels.insert(12, "footnote".to_string());
        class_labels.insert(13, "header".to_string());
        class_labels.insert(14, "algorithm".to_string());
        class_labels.insert(15, "footer".to_string());
        class_labels.insert(16, "seal".to_string());
        class_labels.insert(17, "chart_title".to_string());
        class_labels.insert(18, "chart".to_string());
        class_labels.insert(19, "formula_number".to_string());
        class_labels.insert(20, "header_image".to_string());
        class_labels.insert(21, "footer_image".to_string());
        class_labels.insert(22, "aside_text".to_string());

        Self {
            model_name: "pp-doclayout-s".to_string(),
            num_classes: 23,
            class_labels,
            model_type: "pp-doclayout".to_string(),
            input_size: Some((480, 480)),
        }
    }

    /// Create configuration for PP-DocLayout-M model (23 classes).
    pub fn pp_doclayout_m() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "paragraph_title".to_string());
        class_labels.insert(1, "image".to_string());
        class_labels.insert(2, "text".to_string());
        class_labels.insert(3, "number".to_string());
        class_labels.insert(4, "abstract".to_string());
        class_labels.insert(5, "content".to_string());
        class_labels.insert(6, "figure_title".to_string());
        class_labels.insert(7, "formula".to_string());
        class_labels.insert(8, "table".to_string());
        class_labels.insert(9, "table_title".to_string());
        class_labels.insert(10, "reference".to_string());
        class_labels.insert(11, "doc_title".to_string());
        class_labels.insert(12, "footnote".to_string());
        class_labels.insert(13, "header".to_string());
        class_labels.insert(14, "algorithm".to_string());
        class_labels.insert(15, "footer".to_string());
        class_labels.insert(16, "seal".to_string());
        class_labels.insert(17, "chart_title".to_string());
        class_labels.insert(18, "chart".to_string());
        class_labels.insert(19, "formula_number".to_string());
        class_labels.insert(20, "header_image".to_string());
        class_labels.insert(21, "footer_image".to_string());
        class_labels.insert(22, "aside_text".to_string());

        Self {
            model_name: "pp-doclayout-m".to_string(),
            num_classes: 23,
            class_labels,
            model_type: "pp-doclayout".to_string(),
            input_size: Some((640, 640)),
        }
    }

    /// Create configuration for PP-DocLayout-L model (23 classes).
    pub fn pp_doclayout_l() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "paragraph_title".to_string());
        class_labels.insert(1, "image".to_string());
        class_labels.insert(2, "text".to_string());
        class_labels.insert(3, "number".to_string());
        class_labels.insert(4, "abstract".to_string());
        class_labels.insert(5, "content".to_string());
        class_labels.insert(6, "figure_title".to_string());
        class_labels.insert(7, "formula".to_string());
        class_labels.insert(8, "table".to_string());
        class_labels.insert(9, "table_title".to_string());
        class_labels.insert(10, "reference".to_string());
        class_labels.insert(11, "doc_title".to_string());
        class_labels.insert(12, "footnote".to_string());
        class_labels.insert(13, "header".to_string());
        class_labels.insert(14, "algorithm".to_string());
        class_labels.insert(15, "footer".to_string());
        class_labels.insert(16, "seal".to_string());
        class_labels.insert(17, "chart_title".to_string());
        class_labels.insert(18, "chart".to_string());
        class_labels.insert(19, "formula_number".to_string());
        class_labels.insert(20, "header_image".to_string());
        class_labels.insert(21, "footer_image".to_string());
        class_labels.insert(22, "aside_text".to_string());

        Self {
            model_name: "pp-doclayout-l".to_string(),
            num_classes: 23,
            class_labels,
            model_type: "pp-doclayout".to_string(),
            input_size: Some((640, 640)),
        }
    }

    /// Create configuration for PP-DocLayout-plus-L model (20 classes).
    pub fn pp_doclayout_plus_l() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "paragraph_title".to_string());
        class_labels.insert(1, "image".to_string());
        class_labels.insert(2, "text".to_string());
        class_labels.insert(3, "number".to_string());
        class_labels.insert(4, "abstract".to_string());
        class_labels.insert(5, "content".to_string());
        class_labels.insert(6, "figure_title".to_string());
        class_labels.insert(7, "formula".to_string());
        class_labels.insert(8, "table".to_string());
        class_labels.insert(9, "reference".to_string());
        class_labels.insert(10, "doc_title".to_string());
        class_labels.insert(11, "footnote".to_string());
        class_labels.insert(12, "header".to_string());
        class_labels.insert(13, "algorithm".to_string());
        class_labels.insert(14, "footer".to_string());
        class_labels.insert(15, "seal".to_string());
        class_labels.insert(16, "chart".to_string());
        class_labels.insert(17, "formula_number".to_string());
        class_labels.insert(18, "aside_text".to_string());
        class_labels.insert(19, "reference_content".to_string());

        Self {
            model_name: "pp-doclayout_plus-l".to_string(),
            num_classes: 20,
            class_labels,
            model_type: "pp-doclayout".to_string(),
            input_size: Some((800, 800)),
        }
    }
}

/// Enum for different layout detection model types.
#[derive(Debug)]
enum LayoutModel {
    PicoDet(PicoDetModel),
    RTDetr(RTDetrModel),
    PPDocLayout(PPDocLayoutModel),
}

/// Generic layout detection adapter.
///
/// This adapter uses one of the layout detection models (PicoDet, RT-DETR, or PP-DocLayout)
/// and adapts the model output to the LayoutDetectionTask output format.
#[derive(Debug)]
pub struct LayoutDetectionAdapter {
    model: LayoutModel,
    postprocessor: LayoutPostProcess,
    model_config: LayoutModelConfig,
    info: AdapterInfo,
    config: LayoutDetectionConfig,
}

impl LayoutDetectionAdapter {
    /// Creates a new layout detection adapter with PicoDet model.
    pub fn new_picodet(
        model: PicoDetModel,
        postprocessor: LayoutPostProcess,
        model_config: LayoutModelConfig,
        info: AdapterInfo,
        config: LayoutDetectionConfig,
    ) -> Self {
        Self {
            model: LayoutModel::PicoDet(model),
            postprocessor,
            model_config,
            info,
            config,
        }
    }

    /// Creates a new layout detection adapter with RT-DETR model.
    pub fn new_rtdetr(
        model: RTDetrModel,
        postprocessor: LayoutPostProcess,
        model_config: LayoutModelConfig,
        info: AdapterInfo,
        config: LayoutDetectionConfig,
    ) -> Self {
        Self {
            model: LayoutModel::RTDetr(model),
            postprocessor,
            model_config,
            info,
            config,
        }
    }

    /// Creates a new layout detection adapter with PP-DocLayout model.
    pub fn new_pp_doclayout(
        model: PPDocLayoutModel,
        postprocessor: LayoutPostProcess,
        model_config: LayoutModelConfig,
        info: AdapterInfo,
        config: LayoutDetectionConfig,
    ) -> Self {
        Self {
            model: LayoutModel::PPDocLayout(model),
            postprocessor,
            model_config,
            info,
            config,
        }
    }

    /// Postprocesses model predictions to layout elements.
    fn postprocess(
        &self,
        predictions: &Tensor4D,
        img_shapes: Vec<[f32; 4]>,
        config: &LayoutDetectionConfig,
    ) -> LayoutDetectionOutput {
        let (boxes, class_ids, scores) = self.postprocessor.apply(predictions, img_shapes);

        let mut elements = Vec::with_capacity(boxes.len());

        // Convert to layout elements
        for img_idx in 0..boxes.len() {
            let img_boxes = &boxes[img_idx];
            let img_classes = &class_ids[img_idx];
            let img_scores = &scores[img_idx];

            let mut img_elements = Vec::new();

            for ((bbox, &class_id), &score) in img_boxes
                .iter()
                .zip(img_classes.iter())
                .zip(img_scores.iter())
            {
                if score >= config.score_threshold {
                    // Map class ID to element type
                    let element_type = self
                        .model_config
                        .class_labels
                        .get(&class_id)
                        .cloned()
                        .unwrap_or_else(|| "unknown".to_string());

                    let element = LayoutElement {
                        bbox: bbox.clone(),
                        element_type,
                        score,
                    };

                    img_elements.push(element);

                    if img_elements.len() >= config.max_elements {
                        break;
                    }
                }
            }

            elements.push(img_elements);
        }

        LayoutDetectionOutput { elements }
    }
}

impl ModelAdapter for LayoutDetectionAdapter {
    type Task = LayoutDetectionTask;

    fn info(&self) -> AdapterInfo {
        self.info.clone()
    }

    fn execute(
        &self,
        input: <Self::Task as Task>::Input,
        config: Option<&<Self::Task as Task>::Config>,
    ) -> Result<<Self::Task as Task>::Output, OCRError> {
        // Use provided config or fall back to stored config
        let effective_config = config.unwrap_or(&self.config);

        // Run model-specific forward pass
        let (predictions, img_shapes) = match &self.model {
            LayoutModel::PicoDet(model) => {
                let postprocess_config = PicoDetPostprocessConfig {
                    num_classes: self.model_config.num_classes,
                };
                let (output, img_shapes) = model.forward(input.images, &postprocess_config)?;
                (output.predictions, img_shapes)
            }
            LayoutModel::RTDetr(model) => {
                let postprocess_config = RTDetrPostprocessConfig {
                    num_classes: self.model_config.num_classes,
                };
                let (output, img_shapes) = model.forward(input.images, &postprocess_config)?;
                (output.predictions, img_shapes)
            }
            LayoutModel::PPDocLayout(model) => {
                let postprocess_config = PPDocLayoutPostprocessConfig {
                    num_classes: self.model_config.num_classes,
                };
                let (output, img_shapes) = model.forward(input.images, &postprocess_config)?;
                (output.predictions, img_shapes)
            }
        };

        // Postprocess predictions
        let output = self.postprocess(&predictions, img_shapes, effective_config);

        Ok(output)
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn recommended_batch_size(&self) -> usize {
        4 // Default batch size for layout detection
    }
}

/// Builder for layout detection adapters.
#[derive(Debug)]
pub struct LayoutDetectionAdapterBuilder {
    model_config: Option<LayoutModelConfig>,
    task_config: LayoutDetectionConfig,
}

impl Default for LayoutDetectionAdapterBuilder {
    fn default() -> Self {
        Self {
            model_config: None,
            task_config: LayoutDetectionConfig::default(),
        }
    }
}

impl LayoutDetectionAdapterBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the model configuration.
    pub fn model_config(mut self, config: LayoutModelConfig) -> Self {
        self.model_config = Some(config);
        self
    }

    /// Sets the task configuration.
    pub fn task_config(mut self, config: LayoutDetectionConfig) -> Self {
        self.task_config = config;
        self
    }

    /// Sets the score threshold.
    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.task_config.score_threshold = threshold;
        self
    }

    /// Sets the maximum number of elements.
    pub fn max_elements(mut self, max: usize) -> Self {
        self.task_config.max_elements = max;
        self
    }

    /// Builds the adapter with the specified model configuration.
    fn build_with_config(
        self,
        model_path: &Path,
        model_config: LayoutModelConfig,
    ) -> Result<LayoutDetectionAdapter, OCRError> {
        // Create ONNX inference engine with proper input name based on model type
        let inference = match model_config.model_type.as_str() {
            "pp-doclayout" => {
                // PP-DocLayout models use "image" as the input name
                OrtInfer::new(model_path, Some("image"))?
            }
            _ => {
                // Other models use default or auto-detect
                OrtInfer::new(model_path, None)?
            }
        };

        // Create postprocessor
        let postprocessor = LayoutPostProcess::new(
            model_config.num_classes,
            self.task_config.score_threshold,
            0.5, // NMS threshold
            self.task_config.max_elements,
            model_config.model_type.clone(),
        );

        // Create adapter info
        let info = AdapterInfo::new(
            format!("LayoutDetection_{}", model_config.model_name),
            "1.0.0",
            TaskType::LayoutDetection,
            format!(
                "Layout detection adapter for {} with {} classes",
                model_config.model_name, model_config.num_classes
            ),
        );

        // Build model based on model type
        let adapter = match model_config.model_type.as_str() {
            "picodet" => {
                let mut builder = PicoDetModelBuilder::new();
                if let Some((height, width)) = model_config.input_size {
                    builder = builder.image_shape(height, width);
                }
                let model = builder.build(inference)?;
                LayoutDetectionAdapter::new_picodet(
                    model,
                    postprocessor,
                    model_config,
                    info,
                    self.task_config,
                )
            }
            "rtdetr" => {
                let model = RTDetrModelBuilder::new().build(inference)?;
                LayoutDetectionAdapter::new_rtdetr(
                    model,
                    postprocessor,
                    model_config,
                    info,
                    self.task_config,
                )
            }
            "pp-doclayout" => {
                // PP-DocLayout models - different variants use different input sizes
                let model = if model_config.model_name == "pp-doclayout-s" {
                    // pp-doclayout-s uses 480x480
                    PPDocLayoutModelBuilder::new()
                        .image_shape(480, 480)
                        .build(inference)?
                } else if model_config.model_name == "pp-docblocklayout"
                    || model_config.model_name == "pp-doclayout-m"
                    || model_config.model_name == "pp-doclayout-l"
                {
                    // pp-docblocklayout, pp-doclayout-m, and pp-doclayout-l use 640x640
                    PPDocLayoutModelBuilder::new()
                        .image_shape(640, 640)
                        .build(inference)?
                } else {
                    // pp-doclayout_plus-l uses 800x800 (default)
                    PPDocLayoutModelBuilder::new().build(inference)?
                };
                LayoutDetectionAdapter::new_pp_doclayout(
                    model,
                    postprocessor,
                    model_config,
                    info,
                    self.task_config,
                )
            }
            _ => {
                return Err(OCRError::InvalidInput {
                    message: format!("Unknown model type: {}", model_config.model_type),
                });
            }
        };

        Ok(adapter)
    }
}

impl AdapterBuilder for LayoutDetectionAdapterBuilder {
    type Config = LayoutDetectionConfig;
    type Adapter = LayoutDetectionAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        let model_config = self
            .model_config
            .clone()
            .ok_or_else(|| OCRError::InvalidInput {
                message: "Model configuration is required".to_string(),
            })?;

        self.build_with_config(model_path, model_config)
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.task_config = config;
        self
    }

    fn adapter_type(&self) -> &str {
        "LayoutDetection"
    }
}

// Type aliases and builders for specific models

/// PicoDet layout detection adapter.
pub type PicoDetLayoutAdapter = LayoutDetectionAdapter;

/// Builder for PicoDet layout detection adapter.
pub struct PicoDetLayoutAdapterBuilder {
    inner: LayoutDetectionAdapterBuilder,
}

impl Default for PicoDetLayoutAdapterBuilder {
    fn default() -> Self {
        Self {
            inner: LayoutDetectionAdapterBuilder::new()
                .model_config(LayoutModelConfig::picodet_layout_1x()),
        }
    }
}

impl PicoDetLayoutAdapterBuilder {
    /// Creates a new builder with default PicoDet layout 1x configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new builder with PicoDet-S layout 3 class configuration.
    pub fn new_3cls() -> Self {
        Self {
            inner: LayoutDetectionAdapterBuilder::new()
                .model_config(LayoutModelConfig::picodet_s_layout_3cls()),
        }
    }

    /// Sets the task configuration.
    pub fn task_config(mut self, config: LayoutDetectionConfig) -> Self {
        self.inner = self.inner.task_config(config);
        self
    }

    /// Sets the score threshold.
    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.inner = self.inner.score_threshold(threshold);
        self
    }

    /// Sets the maximum number of elements.
    pub fn max_elements(mut self, max: usize) -> Self {
        self.inner = self.inner.max_elements(max);
        self
    }
}

impl AdapterBuilder for PicoDetLayoutAdapterBuilder {
    type Config = LayoutDetectionConfig;
    type Adapter = PicoDetLayoutAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        self.inner.build(model_path)
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.inner = self.inner.with_config(config);
        self
    }

    fn adapter_type(&self) -> &str {
        "PicoDetLayout"
    }
}

/// RT-DETR layout detection adapter.
pub type RTDetrLayoutAdapter = LayoutDetectionAdapter;

/// Builder for RT-DETR layout detection adapter.
pub struct RTDetrLayoutAdapterBuilder {
    inner: LayoutDetectionAdapterBuilder,
}

impl Default for RTDetrLayoutAdapterBuilder {
    fn default() -> Self {
        Self {
            inner: LayoutDetectionAdapterBuilder::new()
                .model_config(LayoutModelConfig::rtdetr_h_layout_3cls()),
        }
    }
}

impl RTDetrLayoutAdapterBuilder {
    /// Creates a new builder with default RT-DETR-H layout 3 class configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new builder with RT-DETR-H layout 17 class configuration.
    pub fn new_17cls() -> Self {
        Self {
            inner: LayoutDetectionAdapterBuilder::new()
                .model_config(LayoutModelConfig::rtdetr_h_layout_17cls()),
        }
    }

    /// Sets the task configuration.
    pub fn task_config(mut self, config: LayoutDetectionConfig) -> Self {
        self.inner = self.inner.task_config(config);
        self
    }

    /// Sets the score threshold.
    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.inner = self.inner.score_threshold(threshold);
        self
    }

    /// Sets the maximum number of elements.
    pub fn max_elements(mut self, max: usize) -> Self {
        self.inner = self.inner.max_elements(max);
        self
    }
}

impl AdapterBuilder for RTDetrLayoutAdapterBuilder {
    type Config = LayoutDetectionConfig;
    type Adapter = RTDetrLayoutAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        self.inner.build(model_path)
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.inner = self.inner.with_config(config);
        self
    }

    fn adapter_type(&self) -> &str {
        "RTDetrLayout"
    }
}

/// PP-DocLayout detection adapter.
pub type PPDocLayoutAdapter = LayoutDetectionAdapter;

/// Builder for PP-DocLayout detection adapter.
pub struct PPDocLayoutAdapterBuilder {
    inner: LayoutDetectionAdapterBuilder,
}

impl Default for PPDocLayoutAdapterBuilder {
    fn default() -> Self {
        Self {
            inner: LayoutDetectionAdapterBuilder::new()
                .model_config(LayoutModelConfig::pp_doclayout_l()),
        }
    }
}

impl PPDocLayoutAdapterBuilder {
    /// Creates a new builder with the specified PP-DocLayout model variant.
    ///
    /// # Arguments
    ///
    /// * `model_name` - Model variant name. Supported values:
    ///   - `"pp-doclayout-s"` or `"pp_doclayout_s"` - Small model (480x480)
    ///   - `"pp-doclayout-m"` or `"pp_doclayout_m"` - Medium model (640x640)
    ///   - `"pp-doclayout-l"` or `"pp_doclayout_l"` - Large model (640x640, default)
    ///   - `"pp-doclayout_plus-l"` or `"pp_doclayout_plus_l"` - Plus-Large model (800x800)
    ///   - `"pp-docblocklayout"` or `"pp_docblocklayout"` - Block layout model (640x640)
    ///
    /// # Example
    ///
    /// ```no_run
    /// use oar_ocr::domain::adapters::PPDocLayoutAdapterBuilder;
    ///
    /// let adapter = PPDocLayoutAdapterBuilder::new("pp-doclayout-s")
    ///     .build("model.onnx")?;
    /// ```
    pub fn new(model_name: impl AsRef<str>) -> Self {
        let name = model_name.as_ref();
        let config = match name {
            "pp-doclayout-s" | "pp_doclayout_s" => LayoutModelConfig::pp_doclayout_s(),
            "pp-doclayout-m" | "pp_doclayout_m" => LayoutModelConfig::pp_doclayout_m(),
            "pp-doclayout-l" | "pp_doclayout_l" => LayoutModelConfig::pp_doclayout_l(),
            "pp-doclayout_plus-l" | "pp_doclayout_plus_l" => {
                LayoutModelConfig::pp_doclayout_plus_l()
            }
            "pp-docblocklayout" | "pp_docblocklayout" => LayoutModelConfig::pp_docblocklayout(),
            _ => {
                // Default to pp-doclayout-l for unknown variants
                LayoutModelConfig::pp_doclayout_l()
            }
        };

        Self {
            inner: LayoutDetectionAdapterBuilder::new().model_config(config),
        }
    }

    /// Sets the task configuration.
    pub fn task_config(mut self, config: LayoutDetectionConfig) -> Self {
        self.inner = self.inner.task_config(config);
        self
    }

    /// Sets the score threshold.
    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.inner = self.inner.score_threshold(threshold);
        self
    }

    /// Sets the maximum number of elements.
    pub fn max_elements(mut self, max: usize) -> Self {
        self.inner = self.inner.max_elements(max);
        self
    }
}

impl AdapterBuilder for PPDocLayoutAdapterBuilder {
    type Config = LayoutDetectionConfig;
    type Adapter = PPDocLayoutAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        self.inner.build(model_path)
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.inner = self.inner.with_config(config);
        self
    }

    fn adapter_type(&self) -> &str {
        "PPDocLayout"
    }
}
