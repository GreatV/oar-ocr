//! Document Orientation Classification Adapter
//!
//! This adapter uses the PP-LCNet model to classify document image orientation.

use crate::core::OCRError;
use crate::core::traits::{
    adapter::{AdapterBuilder, AdapterInfo, ModelAdapter},
    task::{Task, TaskType},
};
use crate::domain::tasks::{DocumentOrientationConfig, DocumentOrientationTask};
use crate::models::classification::{
    PPLCNetModel, PPLCNetModelBuilder, PPLCNetPostprocessConfig, PPLCNetPreprocessConfig,
};
use image::imageops::FilterType;
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
    pub const DEFAULT_INPUT_SHAPE: (u32, u32) = (192, 48);

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

        // Use model to get predictions
        let model_output = self.model.forward(input.images, &postprocess_config)?;

        // Convert model output to task-specific output
        let label_names = model_output.label_names.unwrap_or_else(|| {
            model_output
                .class_ids
                .iter()
                .map(|ids| ids.iter().map(|&id| format!("{}", id * 90)).collect())
                .collect()
        });

        Ok(crate::domain::tasks::DocumentOrientationOutput {
            class_ids: model_output.class_ids,
            scores: model_output.scores,
            label_names,
        })
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
    /// Task configuration
    task_config: DocumentOrientationConfig,
    /// Input shape (height, width)
    input_shape: (u32, u32),
    /// Session pool size for ONNX Runtime
    session_pool_size: usize,
    /// Optional override for the registered model name
    model_name_override: Option<String>,
}

impl DocumentOrientationAdapterBuilder {
    /// Creates a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            task_config: DocumentOrientationConfig::default(),
            input_shape: DocumentOrientationAdapter::DEFAULT_INPUT_SHAPE,
            session_pool_size: 1,
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
        self.session_pool_size = size;
        self
    }

    /// Sets a custom model name for registry registration.
    pub fn model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name_override = Some(model_name.into());
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
        // Build the PP-LCNet model
        let mut preprocess_config = PPLCNetPreprocessConfig::default();
        preprocess_config.input_shape = self.input_shape;
        preprocess_config.resize_filter = FilterType::Lanczos3;

        let model = PPLCNetModelBuilder::new()
            .session_pool_size(self.session_pool_size)
            .preprocess_config(preprocess_config)
            .build(model_path)?;

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
            self.task_config,
            postprocess_config,
        ))
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.task_config = config;
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
        assert_eq!(builder.task_config.topk, 2);
        assert_eq!(builder.task_config.score_threshold, 0.7);
    }

    #[test]
    fn test_builder_fluent_api() {
        let builder = DocumentOrientationAdapterBuilder::new()
            .input_shape((224, 224))
            .session_pool_size(4);

        assert_eq!(builder.input_shape, (224, 224));
        assert_eq!(builder.session_pool_size, 4);
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
        assert_eq!(labels, vec!["0", "180"]);
    }
}
