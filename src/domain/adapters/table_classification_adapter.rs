//! Table Classification Adapter
//!
//! This adapter uses the PP-LCNet model to classify table images as wired or wireless.

use crate::core::OCRError;
use crate::core::traits::{
    adapter::{AdapterBuilder, AdapterInfo, ModelAdapter},
    task::{Task, TaskType},
};
use crate::domain::tasks::{
    Classification, TableClassificationConfig, TableClassificationOutput, TableClassificationTask,
};
use crate::models::classification::{PPLCNetModel, PPLCNetModelBuilder, PPLCNetPostprocessConfig};
use std::path::Path;

/// Table classification adapter that uses the PP-LCNet model.
#[derive(Debug)]
pub struct TableClassificationAdapter {
    /// The underlying PP-LCNet model
    model: PPLCNetModel,
    /// Adapter information
    info: AdapterInfo,
    /// Task configuration
    config: TableClassificationConfig,
    /// Postprocessing configuration
    postprocess_config: PPLCNetPostprocessConfig,
}

impl TableClassificationAdapter {
    /// Creates a new table classification adapter.
    pub fn new(
        model: PPLCNetModel,
        info: AdapterInfo,
        config: TableClassificationConfig,
        postprocess_config: PPLCNetPostprocessConfig,
    ) -> Self {
        Self {
            model,
            info,
            config,
            postprocess_config,
        }
    }

    /// Default input shape for table classification (224x224 as per model spec).
    pub const DEFAULT_INPUT_SHAPE: (u32, u32) = (224, 224);

    /// Class labels for table classification.
    pub fn labels() -> Vec<String> {
        vec!["wired_table".to_string(), "wireless_table".to_string()]
    }
}

impl ModelAdapter for TableClassificationAdapter {
    type Task = TableClassificationTask;

    fn info(&self) -> AdapterInfo {
        self.info.clone()
    }

    fn execute(
        &self,
        input: <Self::Task as Task>::Input,
        config: Option<&<Self::Task as Task>::Config>,
    ) -> Result<<Self::Task as Task>::Output, OCRError> {
        let effective_config = config.unwrap_or(&self.config);

        // Update postprocess config with task-specific topk
        let mut postprocess_config = self.postprocess_config.clone();
        postprocess_config.topk = effective_config.topk;

        // Use model to get predictions with error context
        let model_output = self
            .model
            .forward(input.images, &postprocess_config)
            .map_err(|e| {
                OCRError::adapter_execution_error(
                    "TableClassificationAdapter",
                    format!(
                        "failed to classify table type (topk={})",
                        effective_config.topk
                    ),
                    e,
                )
            })?;

        // Convert model output to task-specific output with structured classifications
        let label_names = model_output.label_names.unwrap_or_else(|| {
            model_output
                .class_ids
                .iter()
                .map(|ids| {
                    ids.iter()
                        .map(|&id| {
                            if id == 0 {
                                "wired_table".to_string()
                            } else {
                                "wireless_table".to_string()
                            }
                        })
                        .collect()
                })
                .collect()
        });

        // Create structured classifications
        let classifications = model_output
            .class_ids
            .into_iter()
            .zip(model_output.scores)
            .zip(label_names)
            .map(|((class_ids, scores), labels)| {
                class_ids
                    .into_iter()
                    .zip(scores)
                    .zip(labels)
                    .map(|((class_id, score), label)| Classification::new(class_id, label, score))
                    .collect()
            })
            .collect();

        Ok(TableClassificationOutput { classifications })
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn recommended_batch_size(&self) -> usize {
        32
    }
}

/// Builder for table classification adapter.
pub struct TableClassificationAdapterBuilder {
    config: super::builder_config::AdapterBuilderConfig<TableClassificationConfig>,
    /// Input shape (height, width)
    input_shape: (u32, u32),
    /// Optional override for the registered model name
    model_name_override: Option<String>,
}

impl TableClassificationAdapterBuilder {
    /// Creates a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: super::builder_config::AdapterBuilderConfig::default(),
            input_shape: TableClassificationAdapter::DEFAULT_INPUT_SHAPE,
            model_name_override: None,
        }
    }

    /// Sets the input shape.
    pub fn input_shape(mut self, input_shape: (u32, u32)) -> Self {
        self.input_shape = input_shape;
        self
    }

    /// Sets the session pool size.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.config = self.config.with_session_pool_size(size);
        self
    }

    /// Sets a custom model name for registry registration.
    pub fn model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name_override = Some(model_name.into());
        self
    }

    /// Sets the ONNX Runtime session configuration.
    pub fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.config = self.config.with_ort_config(config);
        self
    }
}

impl Default for TableClassificationAdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AdapterBuilder for TableClassificationAdapterBuilder {
    type Config = TableClassificationConfig;
    type Adapter = TableClassificationAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        let (task_config, session_pool_size, ort_config) = self
            .config
            .into_validated_parts()
            .map_err(|err| OCRError::ConfigError {
                message: err.to_string(),
            })?;

        // Build the PP-LCNet model
        let preprocess_config = super::preprocessing::pp_lcnet_preprocess(self.input_shape);

        let mut model_builder = PPLCNetModelBuilder::new()
            .session_pool_size(session_pool_size)
            .preprocess_config(preprocess_config);

        if let Some(ort_config) = ort_config {
            model_builder = model_builder.with_ort_config(ort_config);
        }

        let model = model_builder.build(model_path)?;

        // Create postprocessing configuration
        let postprocess_config = PPLCNetPostprocessConfig {
            labels: TableClassificationAdapter::labels(),
            topk: 1, // Will be overridden by task config
        };

        // Create adapter info
        let mut info = AdapterInfo::new(
            "table_classification",
            "1.0.0",
            TaskType::TableClassification,
            "Table classification (wired/wireless) using PP-LCNet model",
        );
        if let Some(model_name) = self.model_name_override {
            info.model_name = model_name;
        }

        Ok(TableClassificationAdapter::new(
            model,
            info,
            task_config,
            postprocess_config,
        ))
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.config = self.config.with_task_config(config);
        self
    }

    fn adapter_type(&self) -> &str {
        "TableClassification"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = TableClassificationAdapterBuilder::new();
        assert_eq!(builder.adapter_type(), "TableClassification");
    }

    #[test]
    fn test_builder_with_config() {
        let config = TableClassificationConfig {
            score_threshold: 0.7,
            topk: 2,
        };

        let builder = TableClassificationAdapterBuilder::new().with_config(config.clone());
        assert_eq!(builder.config.task_config().topk, 2);
        assert_eq!(builder.config.task_config().score_threshold, 0.7);
    }

    #[test]
    fn test_builder_fluent_api() {
        let builder = TableClassificationAdapterBuilder::new()
            .input_shape((256, 256))
            .session_pool_size(4);

        assert_eq!(builder.input_shape, (256, 256));
        assert_eq!(builder.config.session_pool_size(), 4);
    }

    #[test]
    fn test_default_builder() {
        let builder = TableClassificationAdapterBuilder::default();
        assert_eq!(builder.adapter_type(), "TableClassification");
        assert_eq!(
            builder.input_shape,
            TableClassificationAdapter::DEFAULT_INPUT_SHAPE
        );
    }

    #[test]
    fn test_labels() {
        let labels = TableClassificationAdapter::labels();
        assert_eq!(labels, vec!["wired_table", "wireless_table"]);
    }
}
