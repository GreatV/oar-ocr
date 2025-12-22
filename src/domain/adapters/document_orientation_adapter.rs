//! Document Orientation Classification Adapter
//!
//! This adapter uses the PP-LCNet model to classify document image orientation.

use crate::core::OCRError;
use crate::core::traits::{
    adapter::{AdapterBuilder, AdapterInfo, ModelAdapter},
    task::{Task, TaskType},
};
use crate::domain::tasks::{
    Classification, DocumentOrientationConfig, DocumentOrientationOutput, DocumentOrientationTask,
};
use crate::models::classification::{PPLCNetModel, PPLCNetModelBuilder, PPLCNetPostprocessConfig};
use std::path::Path;

/// Document orientation classification adapter that uses the PP-LCNet model.
#[derive(Debug)]
pub struct DocumentOrientationAdapter {
    /// The underlying PP-LCNet model
    model: PPLCNetModel,
    /// Adapter information
    info: AdapterInfo,
    /// Task configuration
    config: DocumentOrientationConfig,
    /// Postprocessing configuration
    postprocess_config: PPLCNetPostprocessConfig,
}

impl DocumentOrientationAdapter {
    /// Creates a new document orientation adapter.
    pub fn new(
        model: PPLCNetModel,
        info: AdapterInfo,
        config: DocumentOrientationConfig,
        postprocess_config: PPLCNetPostprocessConfig,
    ) -> Self {
        Self {
            model,
            info,
            config,
            postprocess_config,
        }
    }

    /// Default input shape for document orientation classification.
    /// PP-LCNet document orientation models are trained with 224x224 inputs.
    pub const DEFAULT_INPUT_SHAPE: (u32, u32) = (224, 224);

    /// Class labels for document orientation.
    pub fn labels() -> Vec<String> {
        vec![
            "0".to_string(),
            "90".to_string(),
            "180".to_string(),
            "270".to_string(),
        ]
    }
}

impl ModelAdapter for DocumentOrientationAdapter {
    type Task = DocumentOrientationTask;

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
                    "DocumentOrientationAdapter",
                    format!(
                        "failed to classify document orientation (topk={})",
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
                .map(|ids| ids.iter().map(|&id| format!("{}", id * 90)).collect())
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

        Ok(DocumentOrientationOutput { classifications })
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn recommended_batch_size(&self) -> usize {
        32
    }
}

/// Builder for document orientation adapter.
pub struct DocumentOrientationAdapterBuilder {
    config: super::builder_config::AdapterBuilderConfig<DocumentOrientationConfig>,
    /// Input shape (height, width)
    input_shape: (u32, u32),
    /// Optional override for the registered model name
    model_name_override: Option<String>,
}

impl DocumentOrientationAdapterBuilder {
    /// Creates a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: super::builder_config::AdapterBuilderConfig::default(),
            input_shape: DocumentOrientationAdapter::DEFAULT_INPUT_SHAPE,
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

impl Default for DocumentOrientationAdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AdapterBuilder for DocumentOrientationAdapterBuilder {
    type Config = DocumentOrientationConfig;
    type Adapter = DocumentOrientationAdapter;

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
            labels: DocumentOrientationAdapter::labels(),
            topk: 1, // Will be overridden by task config
        };

        // Create adapter info
        let mut info = AdapterInfo::new(
            "document_orientation",
            "1.0.0",
            TaskType::DocumentOrientation,
            "Document orientation classification using PP-LCNet model",
        );
        if let Some(model_name) = self.model_name_override {
            info.model_name = model_name;
        }

        Ok(DocumentOrientationAdapter::new(
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
        "DocumentOrientation"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = DocumentOrientationAdapterBuilder::new();
        assert_eq!(builder.adapter_type(), "DocumentOrientation");
    }

    #[test]
    fn test_builder_with_config() {
        let config = DocumentOrientationConfig {
            score_threshold: 0.7,
            topk: 2,
        };

        let builder = DocumentOrientationAdapterBuilder::new().with_config(config.clone());
        assert_eq!(builder.config.task_config().topk, 2);
        assert_eq!(builder.config.task_config().score_threshold, 0.7);
    }

    #[test]
    fn test_builder_fluent_api() {
        let builder = DocumentOrientationAdapterBuilder::new()
            .input_shape((224, 224))
            .session_pool_size(4);

        assert_eq!(builder.input_shape, (224, 224));
        assert_eq!(builder.config.session_pool_size(), 4);
    }

    #[test]
    fn test_default_builder() {
        let builder = DocumentOrientationAdapterBuilder::default();
        assert_eq!(builder.adapter_type(), "DocumentOrientation");
        assert_eq!(
            builder.input_shape,
            DocumentOrientationAdapter::DEFAULT_INPUT_SHAPE
        );
    }

    #[test]
    fn test_labels() {
        let labels = DocumentOrientationAdapter::labels();
        assert_eq!(labels, vec!["0", "90", "180", "270"]);
    }
}
