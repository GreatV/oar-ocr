//! Seal Text Detection Task Adapter
//!
//! This adapter uses the DB model configured for seal text detection (curved text).

use crate::core::OCRError;
use crate::core::traits::{
    adapter::{AdapterBuilder, AdapterInfo, ModelAdapter},
    task::{Task, TaskType},
};
use crate::domain::tasks::{
    Detection, SealTextDetectionConfig, SealTextDetectionOutput, SealTextDetectionTask,
};
use crate::models::detection::db::{DBModel, DBModelBuilder, DBPostprocessConfig};
use crate::processors::{BoxType, ScoreMode};
use std::path::Path;

/// Seal text detection adapter that uses the DB model.
#[derive(Debug)]
pub struct SealTextDetectionAdapter {
    /// The underlying DB model
    model: DBModel,
    /// Adapter information
    info: AdapterInfo,
    /// Task configuration
    config: SealTextDetectionConfig,
}

impl ModelAdapter for SealTextDetectionAdapter {
    type Task = SealTextDetectionTask;

    fn info(&self) -> AdapterInfo {
        self.info.clone()
    }

    fn execute(
        &self,
        input: <Self::Task as Task>::Input,
        config: Option<&<Self::Task as Task>::Config>,
    ) -> Result<<Self::Task as Task>::Output, OCRError> {
        let effective_config = config.unwrap_or(&self.config);

        // Use the DB model to detect seal text with error context
        let model_output = self.model
            .forward(
                input.images,
                effective_config.score_threshold,
                effective_config.box_threshold,
                effective_config.unclip_ratio,
            )
            .map_err(|e| {
                OCRError::adapter_execution_error(
                    "SealTextDetectionAdapter",
                    format!(
                        "failed to detect seal text (score_threshold={}, box_threshold={}, unclip_ratio={})",
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

        Ok(SealTextDetectionOutput { detections })
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn recommended_batch_size(&self) -> usize {
        8
    }
}

/// Builder for seal text detection adapter.
pub struct SealTextDetectionAdapterBuilder {
    config: super::builder_config::AdapterBuilderConfig<SealTextDetectionConfig>,
}

impl SealTextDetectionAdapterBuilder {
    /// Creates a new seal text detection adapter builder.
    pub fn new() -> Self {
        Self {
            config: super::builder_config::AdapterBuilderConfig::default(),
        }
    }

    /// Sets the task configuration.
    pub fn with_config(mut self, config: SealTextDetectionConfig) -> Self {
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
}

impl AdapterBuilder for SealTextDetectionAdapterBuilder {
    type Config = SealTextDetectionConfig;
    type Adapter = SealTextDetectionAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        let (task_config, session_pool_size, ort_config) = self
            .config
            .into_validated_parts()
            .map_err(|err| OCRError::ConfigError {
                message: err.to_string(),
            })?;

        // Configure DB model for seal text detection
        // Use seal text preprocessing configuration (limit_side_len=736, limit_type=Min)
        let preprocess_config = super::preprocessing::db_preprocess_for_text_type(Some("seal"));

        let postprocess_config = DBPostprocessConfig {
            score_threshold: task_config.score_threshold,
            box_threshold: task_config.box_threshold,
            unclip_ratio: task_config.unclip_ratio,
            max_candidates: task_config.max_candidates,
            use_dilation: false,
            score_mode: ScoreMode::Fast,
            box_type: BoxType::Poly, // Seal detection uses polygon boxes for curved text
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
            "SealTextDetection",
            "1.0.0",
            TaskType::SealTextDetection,
            "Seal text detection using DB model with polygon output",
        );

        Ok(SealTextDetectionAdapter {
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
        "SealTextDetection"
    }
}

impl Default for SealTextDetectionAdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}
