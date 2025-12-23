//! Text Detection Task Adapter
//!
//! This adapter uses the DB model and adapts its output to the TextDetection task format.

use crate::core::OCRError;
use crate::core::traits::{
    adapter::{AdapterBuilder, AdapterInfo, ModelAdapter},
    task::{Task, TaskType},
};
use crate::domain::tasks::{
    Detection, TextDetectionConfig, TextDetectionOutput, TextDetectionTask,
};
use crate::models::detection::db::{DBModel, DBModelBuilder, DBPostprocessConfig};
use crate::processors::{BoxType, ScoreMode};
use std::path::Path;

/// Text detection adapter that uses the DB model.
#[derive(Debug)]
pub struct TextDetectionAdapter {
    /// The underlying DB model
    model: DBModel,
    /// Adapter information
    info: AdapterInfo,
    /// Task configuration
    config: TextDetectionConfig,
}

impl ModelAdapter for TextDetectionAdapter {
    type Task = TextDetectionTask;

    fn info(&self) -> AdapterInfo {
        self.info.clone()
    }

    fn execute(
        &self,
        input: <Self::Task as Task>::Input,
        config: Option<&<Self::Task as Task>::Config>,
    ) -> Result<<Self::Task as Task>::Output, OCRError> {
        let effective_config = config.unwrap_or(&self.config);

        // Use the DB model to detect text with error context
        let model_output = self.model
            .forward(
                input.images,
                effective_config.score_threshold,
                effective_config.box_threshold,
                effective_config.unclip_ratio,
            )
            .map_err(|e| {
                OCRError::adapter_execution_error(
                    "TextDetectionAdapter",
                    format!(
                        "failed to detect text (score_threshold={}, box_threshold={}, unclip_ratio={})",
                        effective_config.score_threshold,
                        effective_config.box_threshold,
                        effective_config.unclip_ratio
                    ),
                    e,
                )
            })?;

        // Convert model output to structured detections
        let detections = model_output
            .boxes
            .into_iter()
            .zip(model_output.scores)
            .map(|(boxes, scores)| {
                boxes
                    .into_iter()
                    .zip(scores)
                    .map(|(bbox, score)| Detection::new(bbox, score))
                    .collect()
            })
            .collect();

        Ok(TextDetectionOutput { detections })
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn recommended_batch_size(&self) -> usize {
        8
    }
}

/// Builder for text detection adapter.
pub struct TextDetectionAdapterBuilder {
    config: super::builder_config::AdapterBuilderConfig<TextDetectionConfig>,
    text_type: Option<String>,
}

impl TextDetectionAdapterBuilder {
    /// Creates a new text detection adapter builder.
    pub fn new() -> Self {
        Self {
            config: super::builder_config::AdapterBuilderConfig::default(),
            text_type: None,
        }
    }

    /// Sets the task configuration.
    pub fn with_config(mut self, config: TextDetectionConfig) -> Self {
        self.config = self.config.with_task_config(config);
        self
    }

    /// Sets the session pool size.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.config = self.config.with_session_pool_size(size);
        self
    }

    /// Sets the ONNX Runtime session configuration.
    pub fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.config = self.config.with_ort_config(config);
        self
    }

    /// Sets the text type for preprocessing and postprocessing configuration.
    ///
    /// This matches the text_type parameter:
    /// - "seal": Uses seal-specific preprocessing (limit_side_len=736, limit_type=Min) and polygon boxes
    /// - Other values or None: Uses general text configuration (limit_side_len=960, limit_type=Max) and quad boxes
    pub fn text_type(mut self, text_type: impl Into<String>) -> Self {
        self.text_type = Some(text_type.into());
        self
    }
}

impl AdapterBuilder for TextDetectionAdapterBuilder {
    type Config = TextDetectionConfig;
    type Adapter = TextDetectionAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        let (task_config, session_pool_size, ort_config) = self
            .config
            .into_validated_parts()
            .map_err(|err| OCRError::ConfigError {
                message: err.to_string(),
            })?;

        // Determine if this is seal text (uses different preprocessing and box type)
        let is_seal_text = self
            .text_type
            .as_ref()
            .map(|t| t.to_lowercase() == "seal")
            .unwrap_or(false);

        // Configure DB model preprocessing based on text type
        // Matches standard behavior:
        // - General text: limit_side_len=960, limit_type=Max
        // - Seal text: limit_side_len=736, limit_type=Min
        let mut preprocess_config =
            super::preprocessing::db_preprocess_for_text_type(self.text_type.as_deref());

        // Override with config values if present
        if let Some(limit) = task_config.limit_side_len {
            preprocess_config.limit_side_len = Some(limit);
        }
        if let Some(limit_type) = task_config.limit_type.clone() {
            preprocess_config.limit_type = Some(limit_type);
        }
        if let Some(max_limit) = task_config.max_side_len {
            preprocess_config.max_side_limit = Some(max_limit);
        }

        // Configure postprocessing based on text type
        // Seal text uses polygon boxes for curved text, general text uses quad boxes
        let box_type = if is_seal_text {
            BoxType::Poly
        } else {
            BoxType::Quad
        };

        let postprocess_config = DBPostprocessConfig {
            score_threshold: task_config.score_threshold,
            box_threshold: task_config.box_threshold,
            unclip_ratio: task_config.unclip_ratio,
            max_candidates: task_config.max_candidates,
            use_dilation: false,
            score_mode: ScoreMode::Fast,
            box_type,
        };

        // Build the DB model
        let mut model_builder = DBModelBuilder::new()
            .preprocess_config(preprocess_config)
            .postprocess_config(postprocess_config)
            .session_pool_size(session_pool_size);

        if let Some(ort_config) = ort_config {
            model_builder = model_builder.with_ort_config(ort_config);
        }

        let model = model_builder.build(model_path)?;

        // Create adapter info
        let info = AdapterInfo::new(
            "TextDetection",
            "1.0.0",
            TaskType::TextDetection,
            "Text detection using DB model",
        );

        Ok(TextDetectionAdapter {
            model,
            info,
            config: task_config,
        })
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.config = self.config.with_task_config(config);
        self
    }

    fn adapter_type(&self) -> &str {
        "TextDetection"
    }
}

impl Default for TextDetectionAdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}
