//! UVDoc rectifier adapter implementation.
//!
//! This adapter uses the UVDoc model and adapts its output to the DocumentRectification task format.

use crate::core::OCRError;
use crate::core::traits::adapter::{AdapterBuilder, AdapterInfo, ModelAdapter};
use crate::core::traits::task::{Task, TaskType};
use crate::domain::tasks::document_rectification::{
    DocumentRectificationConfig, DocumentRectificationOutput, DocumentRectificationTask,
};
use crate::models::rectification::uvdoc::{UVDocModel, UVDocModelBuilder, UVDocPreprocessConfig};
use std::path::Path;

/// UVDoc rectifier adapter that uses the UVDoc model.
#[derive(Debug)]
pub struct UVDocRectifierAdapter {
    /// The underlying UVDoc model
    model: UVDocModel,
    /// Adapter information
    info: AdapterInfo,
    /// Task configuration (stored for potential future use)
    _config: DocumentRectificationConfig,
}

impl ModelAdapter for UVDocRectifierAdapter {
    type Task = DocumentRectificationTask;

    fn info(&self) -> AdapterInfo {
        self.info.clone()
    }

    fn execute(
        &self,
        input: <Self::Task as Task>::Input,
        _config: Option<&<Self::Task as Task>::Config>,
    ) -> Result<<Self::Task as Task>::Output, OCRError> {
        let batch_len = input.images.len();
        // Use the UVDoc model to rectify images
        let model_output = self.model.forward(input.images).map_err(|e| {
            OCRError::adapter_execution_error(
                "UVDocRectifierAdapter",
                format!("model forward (batch_size={})", batch_len),
                e,
            )
        })?;

        // Adapt model output to task output
        Ok(DocumentRectificationOutput {
            rectified_images: model_output.images,
        })
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn recommended_batch_size(&self) -> usize {
        // Document rectification is computationally intensive
        // Use smaller batch size for better memory management
        8
    }
}

/// Builder for UVDoc rectifier adapter.
///
/// This builder provides a fluent API for configuring and creating
/// a UVDoc rectifier adapter instance.
pub struct UVDocRectifierAdapterBuilder {
    config: super::builder_config::AdapterBuilderConfig<DocumentRectificationConfig>,
    /// Preprocessing configuration
    preprocess_config: UVDocPreprocessConfig,
    /// Optional override for the registered model name
    model_name_override: Option<String>,
}

impl UVDocRectifierAdapterBuilder {
    /// Creates a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            config: super::builder_config::AdapterBuilderConfig::default(),
            preprocess_config: UVDocPreprocessConfig::default(),
            model_name_override: None,
        }
    }

    /// Sets the input shape for the rectification model.
    ///
    /// # Arguments
    ///
    /// * `shape` - Input shape as [channels, height, width]
    pub fn input_shape(mut self, shape: [usize; 3]) -> Self {
        self.config.task_config.rec_image_shape = shape;
        self.preprocess_config.rec_image_shape = shape;
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

impl Default for UVDocRectifierAdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AdapterBuilder for UVDocRectifierAdapterBuilder {
    type Config = DocumentRectificationConfig;
    type Adapter = UVDocRectifierAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        let (task_config, session_pool_size, ort_config) = self
            .config
            .into_validated_parts()
            .map_err(|err| OCRError::ConfigError {
                message: err.to_string(),
            })?;

        // Build the UVDoc model
        let mut model_builder = UVDocModelBuilder::new()
            .preprocess_config(self.preprocess_config)
            .session_pool_size(session_pool_size);

        if let Some(ort_config) = ort_config {
            model_builder = model_builder.with_ort_config(ort_config);
        }

        let model = model_builder.build(model_path)?;

        // Create adapter info
        let mut info = AdapterInfo::new(
            "uvdoc_rectifier",
            "1.0.0",
            TaskType::DocumentRectification,
            "UVDoc document rectifier for correcting image distortions",
        );
        if let Some(model_name) = self.model_name_override {
            info.model_name = model_name;
        }

        Ok(UVDocRectifierAdapter {
            model,
            info,
            _config: task_config,
        })
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.preprocess_config.rec_image_shape = config.rec_image_shape;
        self.config = self.config.with_task_config(config);
        self
    }

    fn adapter_type(&self) -> &str {
        "uvdoc_rectifier"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = UVDocRectifierAdapterBuilder::new();
        assert_eq!(builder.adapter_type(), "uvdoc_rectifier");
    }

    #[test]
    fn test_builder_with_config() {
        let config = DocumentRectificationConfig {
            rec_image_shape: [3, 1024, 1024],
        };

        let builder = UVDocRectifierAdapterBuilder::new().with_config(config.clone());
        assert_eq!(
            builder.config.task_config().rec_image_shape,
            [3, 1024, 1024]
        );
        assert_eq!(builder.preprocess_config.rec_image_shape, [3, 1024, 1024]);
    }

    #[test]
    fn test_builder_fluent_api() {
        let builder = UVDocRectifierAdapterBuilder::new().input_shape([3, 768, 768]);

        assert_eq!(builder.config.task_config().rec_image_shape, [3, 768, 768]);
        assert_eq!(builder.preprocess_config.rec_image_shape, [3, 768, 768]);
    }

    #[test]
    fn test_default_builder() {
        let builder = UVDocRectifierAdapterBuilder::default();
        assert_eq!(builder.adapter_type(), "uvdoc_rectifier");
        assert_eq!(builder.config.task_config().rec_image_shape, [3, 0, 0]);
    }

    #[test]
    fn test_builder_with_session_pool() {
        let builder = UVDocRectifierAdapterBuilder::new().session_pool_size(4);

        assert_eq!(builder.adapter_type(), "uvdoc_rectifier");
    }
}
