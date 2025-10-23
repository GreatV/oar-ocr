//! Seal Text Detection Task Adapter
//!
//! This adapter uses the DB model configured for seal text detection (curved text).

use crate::core::OCRError;
use crate::core::traits::{
    adapter::{AdapterBuilder, AdapterInfo, ModelAdapter},
    task::{Task, TaskType},
};
use crate::domain::tasks::{
    SealTextDetectionConfig, SealTextDetectionOutput, SealTextDetectionTask,
};
use crate::models::detection::db::{
    DBModel, DBModelBuilder, DBPostprocessConfig, DBPreprocessConfig,
};
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

        // Use the DB model to detect seal text
        let model_output = self.model.forward(
            input.images,
            effective_config.score_threshold,
            effective_config.box_threshold,
            effective_config.unclip_ratio,
        )?;

        // Adapt model output to task output
        Ok(SealTextDetectionOutput {
            boxes: model_output.boxes,
            scores: model_output.scores,
        })
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
    /// Task configuration
    task_config: SealTextDetectionConfig,
    /// Session pool size
    session_pool_size: usize,
}

impl SealTextDetectionAdapterBuilder {
    /// Creates a new seal text detection adapter builder.
    pub fn new() -> Self {
        Self {
            task_config: SealTextDetectionConfig::default(),
            session_pool_size: 1,
        }
    }

    /// Sets the task configuration.
    pub fn with_config(mut self, config: SealTextDetectionConfig) -> Self {
        self.task_config = config;
        self
    }

    /// Sets the session pool size.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.session_pool_size = size;
        self
    }
}

impl AdapterBuilder for SealTextDetectionAdapterBuilder {
    type Config = SealTextDetectionConfig;
    type Adapter = SealTextDetectionAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        // Configure DB model for seal text detection
        let preprocess_config = DBPreprocessConfig {
            limit_side_len: None,
            limit_type: None,
            max_side_limit: None,
            resize_long: Some(736), // Specific for seal detection
        };

        let postprocess_config = DBPostprocessConfig {
            score_threshold: self.task_config.score_threshold,
            box_threshold: self.task_config.box_threshold,
            unclip_ratio: self.task_config.unclip_ratio,
            max_candidates: self.task_config.max_candidates,
            use_dilation: false,
            score_mode: ScoreMode::Fast,
            box_type: BoxType::Poly, // Seal detection uses polygon boxes for curved text
        };

        // Build the DB model
        let model = DBModelBuilder::new()
            .preprocess_config(preprocess_config)
            .postprocess_config(postprocess_config)
            .session_pool_size(self.session_pool_size)
            .build(model_path)?;

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
            config: self.task_config,
        })
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.task_config = config;
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
