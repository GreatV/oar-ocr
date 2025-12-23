//! Table cell detection adapter.
//!
//! This adapter wraps RT-DETR models that detect table cells (wired / wireless
//! cell structures) and adapts their outputs to the [`TableCellDetectionTask`].

use crate::core::inference::OrtInfer;
use crate::core::traits::adapter::{AdapterBuilder, AdapterInfo, ModelAdapter};
use crate::core::traits::task::{Task, TaskType};
use crate::core::{OCRError, Tensor4D};
use crate::domain::tasks::{
    TableCell, TableCellDetectionConfig, TableCellDetectionOutput, TableCellDetectionTask,
};
use crate::models::detection::{RTDetrModel, RTDetrModelBuilder, RTDetrPostprocessConfig};
use crate::processors::{ImageScaleInfo, LayoutPostProcess};
use std::collections::HashMap;
use std::path::Path;

/// Configuration describing a table cell detection model.
#[derive(Debug, Clone)]
pub struct TableCellModelConfig {
    /// Model name (e.g., `rt-detr-l_wired_table_cell_det`)
    pub model_name: String,
    /// Number of classes (currently 1 for table cells)
    pub num_classes: usize,
    /// Mapping from class id to label string.
    pub class_labels: HashMap<usize, String>,
    /// Model family identifier (e.g., `rtdetr`)
    pub model_type: String,
    /// Optional fixed input size (height, width).
    pub input_size: Option<(u32, u32)>,
}

impl TableCellModelConfig {
    /// Creates configuration for the wired table cell detector.
    pub fn rtdetr_l_wired_table_cell_det() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "cell".to_string());
        Self {
            model_name: "rt-detr-l_wired_table_cell_det".to_string(),
            num_classes: 1,
            class_labels,
            model_type: "rtdetr".to_string(),
            input_size: Some((640, 640)),
        }
    }

    /// Creates configuration for the wireless table cell detector.
    pub fn rtdetr_l_wireless_table_cell_det() -> Self {
        let mut class_labels = HashMap::new();
        class_labels.insert(0, "cell".to_string());
        Self {
            model_name: "rt-detr-l_wireless_table_cell_det".to_string(),
            num_classes: 1,
            class_labels,
            model_type: "rtdetr".to_string(),
            input_size: Some((640, 640)),
        }
    }
}

/// Underlying model enum.
#[derive(Debug)]
enum TableCellModel {
    RTDetr(RTDetrModel),
}

/// Adapter for table cell detection.
#[derive(Debug)]
pub struct TableCellDetectionAdapter {
    model: TableCellModel,
    postprocessor: LayoutPostProcess,
    model_config: TableCellModelConfig,
    info: AdapterInfo,
    config: TableCellDetectionConfig,
}

impl TableCellDetectionAdapter {
    fn new_rtdetr(
        model: RTDetrModel,
        postprocessor: LayoutPostProcess,
        model_config: TableCellModelConfig,
        info: AdapterInfo,
        config: TableCellDetectionConfig,
    ) -> Self {
        Self {
            model: TableCellModel::RTDetr(model),
            postprocessor,
            model_config,
            info,
            config,
        }
    }

    fn postprocess(
        &self,
        predictions: &Tensor4D,
        img_shapes: Vec<ImageScaleInfo>,
        config: &TableCellDetectionConfig,
    ) -> TableCellDetectionOutput {
        let (boxes, class_ids, scores) = self.postprocessor.apply(predictions, img_shapes);
        let mut all_cells = Vec::with_capacity(boxes.len());

        for (img_boxes, (img_classes, img_scores)) in boxes
            .into_iter()
            .zip(class_ids.into_iter().zip(scores.into_iter()))
        {
            let mut cells = Vec::new();
            for (bbox, (class_id, score)) in img_boxes
                .into_iter()
                .zip(img_classes.into_iter().zip(img_scores.into_iter()))
            {
                if score < config.score_threshold {
                    continue;
                }

                let label = self
                    .model_config
                    .class_labels
                    .get(&class_id)
                    .cloned()
                    .unwrap_or_else(|| "cell".to_string());

                cells.push(TableCell { bbox, score, label });

                if cells.len() >= config.max_cells {
                    break;
                }
            }
            all_cells.push(cells);
        }

        TableCellDetectionOutput { cells: all_cells }
    }
}

impl ModelAdapter for TableCellDetectionAdapter {
    type Task = TableCellDetectionTask;

    fn info(&self) -> AdapterInfo {
        self.info.clone()
    }

    fn execute(
        &self,
        input: <Self::Task as Task>::Input,
        config: Option<&<Self::Task as Task>::Config>,
    ) -> Result<<Self::Task as Task>::Output, OCRError> {
        let effective_config = config.unwrap_or(&self.config);
        let batch_len = input.images.len();

        let (predictions, img_shapes) = match &self.model {
            TableCellModel::RTDetr(model) => {
                let postprocess_config = RTDetrPostprocessConfig {
                    num_classes: self.model_config.num_classes,
                };
                let (output, img_shapes) = model
                    .forward(input.images, &postprocess_config)
                    .map_err(|e| {
                        OCRError::adapter_execution_error(
                            "TableCellDetectionAdapter",
                            format!("RTDetr forward (batch_size={})", batch_len),
                            e,
                        )
                    })?;
                (output.predictions, img_shapes)
            }
        };

        Ok(self.postprocess(&predictions, img_shapes, effective_config))
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn recommended_batch_size(&self) -> usize {
        4
    }
}

/// Builder for table cell detection adapters.
#[derive(Debug, Default)]
pub struct TableCellDetectionAdapterBuilder {
    config: super::builder_config::AdapterBuilderConfig<TableCellDetectionConfig>,
    model_config: Option<TableCellModelConfig>,
}

impl TableCellDetectionAdapterBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the model configuration.
    pub fn model_config(mut self, config: TableCellModelConfig) -> Self {
        self.model_config = Some(config);
        self
    }

    /// Sets the task configuration.
    pub fn task_config(mut self, config: TableCellDetectionConfig) -> Self {
        self.config = self.config.with_task_config(config);
        self
    }

    /// Sets the score threshold.
    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.config.task_config.score_threshold = threshold;
        self
    }

    /// Sets the maximum number of cells per image.
    pub fn max_cells(mut self, max: usize) -> Self {
        self.config.task_config.max_cells = max;
        self
    }

    /// Sets the ONNX Runtime session configuration.
    pub fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.config = self.config.with_ort_config(config);
        self
    }

    fn build_with_config(
        model_path: &Path,
        model_config: TableCellModelConfig,
        task_config: TableCellDetectionConfig,
        ort_config: Option<crate::core::config::OrtSessionConfig>,
    ) -> Result<TableCellDetectionAdapter, OCRError> {
        let inference = if ort_config.is_some() {
            use crate::core::config::ModelInferenceConfig;
            let common_config = ModelInferenceConfig {
                ort_session: ort_config,
                ..Default::default()
            };
            OrtInfer::from_config(&common_config, model_path, None)?
        } else {
            OrtInfer::new(model_path, None)?
        };

        let postprocessor = LayoutPostProcess::new(
            model_config.num_classes,
            task_config.score_threshold,
            0.5,
            task_config.max_cells,
            model_config.model_type.clone(),
        );

        let info = AdapterInfo::new(
            format!("TableCellDetection_{}", model_config.model_name),
            "1.0.0",
            TaskType::TableCellDetection,
            format!(
                "Table cell detection adapter for {} with {} classes",
                model_config.model_name, model_config.num_classes
            ),
        );

        let model = match model_config.model_type.as_str() {
            "rtdetr" => {
                let mut builder = RTDetrModelBuilder::new();
                if let Some((height, width)) = model_config.input_size {
                    builder = builder.image_shape(height, width);
                }
                builder.build(inference)?
            }
            other => {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Unsupported model type '{}' for table cell detection",
                        other
                    ),
                });
            }
        };

        Ok(TableCellDetectionAdapter::new_rtdetr(
            model,
            postprocessor,
            model_config,
            info,
            task_config,
        ))
    }
}

impl AdapterBuilder for TableCellDetectionAdapterBuilder {
    type Config = TableCellDetectionConfig;
    type Adapter = TableCellDetectionAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        let model_config = self.model_config.ok_or_else(|| OCRError::InvalidInput {
            message: "Table cell model configuration is required".to_string(),
        })?;

        let (task_config, _session_pool_size, ort_config) = self
            .config
            .into_validated_parts()
            .map_err(|err| OCRError::ConfigError {
                message: err.to_string(),
            })?;

        TableCellDetectionAdapterBuilder::build_with_config(
            model_path,
            model_config,
            task_config,
            ort_config,
        )
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.config = self.config.with_task_config(config);
        self
    }

    fn adapter_type(&self) -> &str {
        "TableCellDetection"
    }
}

/// Builder for RT-DETR table cell detection adapters.
#[derive(Debug)]
pub struct RTDetrTableCellAdapterBuilder {
    inner: TableCellDetectionAdapterBuilder,
}

impl Default for RTDetrTableCellAdapterBuilder {
    fn default() -> Self {
        Self {
            inner: TableCellDetectionAdapterBuilder::new()
                .model_config(TableCellModelConfig::rtdetr_l_wired_table_cell_det()),
        }
    }
}

impl RTDetrTableCellAdapterBuilder {
    /// Creates a new builder with wired variant as default.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a builder configured for the wireless model.
    pub fn wireless() -> Self {
        Self {
            inner: TableCellDetectionAdapterBuilder::new()
                .model_config(TableCellModelConfig::rtdetr_l_wireless_table_cell_det()),
        }
    }

    /// Sets the task configuration.
    pub fn task_config(mut self, config: TableCellDetectionConfig) -> Self {
        self.inner = self.inner.task_config(config);
        self
    }

    /// Sets the score threshold.
    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.inner = self.inner.score_threshold(threshold);
        self
    }

    /// Sets the maximum number of cells.
    pub fn max_cells(mut self, max: usize) -> Self {
        self.inner = self.inner.max_cells(max);
        self
    }

    /// Sets the ONNX Runtime session configuration.
    pub fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.inner = self.inner.with_ort_config(config);
        self
    }
}

impl AdapterBuilder for RTDetrTableCellAdapterBuilder {
    type Config = TableCellDetectionConfig;
    type Adapter = TableCellDetectionAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        self.inner.build(model_path)
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.inner = self.inner.with_config(config);
        self
    }

    fn adapter_type(&self) -> &str {
        "RTDetrTableCell"
    }
}
