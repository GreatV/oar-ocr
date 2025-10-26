//! Text Line Orientation Classification Adapter
//!
//! This adapter uses the PP-LCNet model to classify text line orientation.

use crate::core::OCRError;
use crate::core::traits::{
    adapter::{AdapterBuilder, AdapterInfo, ModelAdapter},
    task::{Task, TaskType},
};
use crate::domain::tasks::{TextLineOrientationConfig, TextLineOrientationTask};
use crate::models::classification::{
    PPLCNetModel, PPLCNetModelBuilder, PPLCNetPostprocessConfig, PPLCNetPreprocessConfig,
};
use image::imageops::FilterType;
use std::path::Path;

/// Text line orientation classification adapter that uses the PP-LCNet model.
#[derive(Debug)]
pub struct TextLineOrientationAdapter {
    /// The underlying PP-LCNet model
    model: PPLCNetModel,
    /// Adapter information
    info: AdapterInfo,
    /// Task configuration
    config: TextLineOrientationConfig,
    /// Postprocessing configuration
    postprocess_config: PPLCNetPostprocessConfig,
}

impl TextLineOrientationAdapter {
    /// Creates a new text line orientation adapter.
    pub fn new(
        model: PPLCNetModel,
        info: AdapterInfo,
        config: TextLineOrientationConfig,
        postprocess_config: PPLCNetPostprocessConfig,
    ) -> Self {
        Self {
            model,
            info,
            config,
            postprocess_config,
        }
    }

    /// Default input shape for text line orientation classification.
    pub const DEFAULT_INPUT_SHAPE: (u32, u32) = (192, 48);

    /// Class labels for text line orientation.
    pub fn labels() -> Vec<String> {
        vec!["0".to_string(), "180".to_string()]
    }
}

impl ModelAdapter for TextLineOrientationAdapter {
    type Task = TextLineOrientationTask;

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
                .map(|ids| ids.iter().map(|&id| format!("{}", id * 180)).collect())
                .collect()
        });

        Ok(crate::domain::tasks::TextLineOrientationOutput {
            class_ids: model_output.class_ids,
            scores: model_output.scores,
            label_names,
        })
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn recommended_batch_size(&self) -> usize {
        64
    }
}

/// Builder for text line orientation adapter.
pub struct TextLineOrientationAdapterBuilder {
    /// Task configuration
    task_config: TextLineOrientationConfig,
    /// Input shape (height, width)
    input_shape: (u32, u32),
    /// Session pool size for ONNX Runtime
    session_pool_size: usize,
    /// Optional override for the registered model name
    model_name_override: Option<String>,
    /// ONNX Runtime session configuration
    ort_config: Option<crate::core::config::OrtSessionConfig>,
}

impl TextLineOrientationAdapterBuilder {
    /// Creates a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            task_config: TextLineOrientationConfig::default(),
            input_shape: TextLineOrientationAdapter::DEFAULT_INPUT_SHAPE,
            session_pool_size: 1,
            model_name_override: None,
            ort_config: None,
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

    /// Sets the ONNX Runtime session configuration.
    pub fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.ort_config = Some(config);
        self
    }
}

impl Default for TextLineOrientationAdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AdapterBuilder for TextLineOrientationAdapterBuilder {
    type Config = TextLineOrientationConfig;
    type Adapter = TextLineOrientationAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        // Build the PP-LCNet model
        let preprocess_config = PPLCNetPreprocessConfig {
            input_shape: self.input_shape,
            resize_filter: FilterType::Lanczos3,
            normalize_mean: vec![0.5, 0.5, 0.5],
            normalize_std: vec![0.5, 0.5, 0.5],
            ..Default::default()
        };

        let mut model_builder = PPLCNetModelBuilder::new()
            .session_pool_size(self.session_pool_size)
            .preprocess_config(preprocess_config);

        if let Some(ort_config) = self.ort_config {
            model_builder = model_builder.with_ort_config(ort_config);
        }

        let model = model_builder.build(model_path)?;

        // Create postprocessing configuration
        let postprocess_config = PPLCNetPostprocessConfig {
            labels: TextLineOrientationAdapter::labels(),
            topk: 1, // Will be overridden by task config
        };

        // Create adapter info
        let mut info = AdapterInfo::new(
            "text_line_orientation",
            "1.0.0",
            TaskType::TextLineOrientation,
            "Text line orientation classification using PP-LCNet model",
        );
        if let Some(model_name) = self.model_name_override {
            info.model_name = model_name;
        }

        Ok(TextLineOrientationAdapter::new(
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
        "TextLineOrientation"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = TextLineOrientationAdapterBuilder::new();
        assert_eq!(builder.adapter_type(), "TextLineOrientation");
    }

    #[test]
    fn test_builder_with_config() {
        let config = TextLineOrientationConfig {
            score_threshold: 0.7,
            topk: 2,
        };

        let builder = TextLineOrientationAdapterBuilder::new().with_config(config.clone());
        assert_eq!(builder.task_config.topk, 2);
        assert_eq!(builder.task_config.score_threshold, 0.7);
    }

    #[test]
    fn test_builder_fluent_api() {
        let builder = TextLineOrientationAdapterBuilder::new()
            .input_shape((224, 224))
            .session_pool_size(4);

        assert_eq!(builder.input_shape, (224, 224));
        assert_eq!(builder.session_pool_size, 4);
    }

    #[test]
    fn test_default_builder() {
        let builder = TextLineOrientationAdapterBuilder::default();
        assert_eq!(builder.adapter_type(), "TextLineOrientation");
        assert_eq!(
            builder.input_shape,
            TextLineOrientationAdapter::DEFAULT_INPUT_SHAPE
        );
    }

    #[test]
    fn test_labels() {
        let labels = TextLineOrientationAdapter::labels();
        assert_eq!(labels, vec!["0", "180"]);
    }
}
