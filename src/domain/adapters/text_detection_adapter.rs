//! Text Detection Task Adapter
//!
//! This adapter uses the DB model and adapts its output to the TextDetection task format.

use crate::core::traits::{
    adapter::{AdapterBuilder, AdapterInfo, ModelAdapter},
    task::{Task, TaskType},
};
use crate::core::{OCRError, ProcessingStage};
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
            .map_err(|e| OCRError::Processing {
                kind: ProcessingStage::AdapterExecution,
                context: format!(
                    "TextDetectionAdapter failed to detect text (score_threshold={}, box_threshold={}, unclip_ratio={})",
                    effective_config.score_threshold,
                    effective_config.box_threshold,
                    effective_config.unclip_ratio
                ),
                source: Box::new(e),
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
    /// Task configuration
    task_config: TextDetectionConfig,
    /// Session pool size
    session_pool_size: usize,
    /// ONNX Runtime session configuration
    ort_config: Option<crate::core::config::OrtSessionConfig>,
}

impl TextDetectionAdapterBuilder {
    /// Creates a new text detection adapter builder.
    pub fn new() -> Self {
        Self {
            task_config: TextDetectionConfig::default(),
            session_pool_size: 1,
            ort_config: None,
        }
    }

    /// Sets the task configuration.
    pub fn with_config(mut self, config: TextDetectionConfig) -> Self {
        self.task_config = config;
        self
    }

    /// Sets the session pool size.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.session_pool_size = size;
        self
    }

    /// Sets the ONNX Runtime session configuration.
    pub fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.ort_config = Some(config);
        self
    }
}

impl AdapterBuilder for TextDetectionAdapterBuilder {
    type Config = TextDetectionConfig;
    type Adapter = TextDetectionAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        // Configure DB model for text detection
        // Use default preprocessing (limit_side_len=960)
        let preprocess_config = super::preprocessing::db_preprocess_with_limit_side_len(960);

        let postprocess_config = DBPostprocessConfig {
            score_threshold: self.task_config.score_threshold,
            box_threshold: self.task_config.box_threshold,
            unclip_ratio: self.task_config.unclip_ratio,
            max_candidates: self.task_config.max_candidates,
            use_dilation: false,
            score_mode: ScoreMode::Fast,
            box_type: BoxType::Quad, // Text detection uses quadrilateral boxes
        };

        // Build the DB model
        let mut model_builder = DBModelBuilder::new()
            .preprocess_config(preprocess_config)
            .postprocess_config(postprocess_config)
            .session_pool_size(self.session_pool_size);

        if let Some(ort_config) = self.ort_config {
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
            config: self.task_config,
        })
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.task_config = config;
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
