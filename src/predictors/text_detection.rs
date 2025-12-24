//! Text Detection Predictor
//!
//! This module provides a high-level API for text detection in images.

use super::builder::PredictorBuilderState;
use crate::core::traits::adapter::AdapterBuilder;
use crate::core::traits::task::ImageTaskInput;
use crate::domain::adapters::TextDetectionAdapterBuilder;
use crate::domain::tasks::text_detection::{TextDetectionConfig, TextDetectionTask};
use crate::predictors::TaskPredictorCore;
use image::RgbImage;
use std::path::Path;

/// Text detection prediction result
#[derive(Debug, Clone)]
pub struct TextDetectionResult {
    /// Detected text regions for each input image
    pub detections: Vec<Vec<crate::domain::tasks::text_detection::Detection>>,
}

/// Text detection predictor
pub struct TextDetectionPredictor {
    core: TaskPredictorCore<TextDetectionTask>,
}

impl TextDetectionPredictor {
    /// Create a new builder for the text detection predictor
    pub fn builder() -> TextDetectionPredictorBuilder {
        TextDetectionPredictorBuilder::new()
    }

    /// Predict text regions in the given images.
    pub fn predict(
        &self,
        images: Vec<RgbImage>,
    ) -> Result<TextDetectionResult, Box<dyn std::error::Error>> {
        // Create task input
        let input = ImageTaskInput::new(images);

        // Use core predictor for validation and execution
        let output = self.core.predict(input)?;

        Ok(TextDetectionResult {
            detections: output.detections,
        })
    }
}

/// Builder for text detection predictor
pub struct TextDetectionPredictorBuilder {
    state: PredictorBuilderState<TextDetectionConfig>,
}

impl TextDetectionPredictorBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            state: PredictorBuilderState::new(TextDetectionConfig {
                score_threshold: 0.3,
                box_threshold: 0.6,
                unclip_ratio: 1.5,
                max_candidates: 1000,
                limit_side_len: None,
                limit_type: None,
                max_side_len: None,
            }),
        }
    }

    /// Set the score threshold
    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.state.config_mut().score_threshold = threshold;
        self
    }

    /// Set the box threshold
    pub fn box_threshold(mut self, threshold: f32) -> Self {
        self.state.config_mut().box_threshold = threshold;
        self
    }

    /// Set the unclip ratio
    pub fn unclip_ratio(mut self, ratio: f32) -> Self {
        self.state.config_mut().unclip_ratio = ratio;
        self
    }

    /// Set the maximum candidates
    pub fn max_candidates(mut self, max: usize) -> Self {
        self.state.config_mut().max_candidates = max;
        self
    }

    /// Build the text detection predictor
    pub fn build<P: AsRef<Path>>(
        self,
        model_path: P,
    ) -> Result<TextDetectionPredictor, Box<dyn std::error::Error>> {
        let (config, ort_config) = self.state.into_parts();
        let mut adapter_builder = TextDetectionAdapterBuilder::new().with_config(config.clone());

        if let Some(ort_cfg) = ort_config {
            adapter_builder = adapter_builder.with_ort_config(ort_cfg);
        }

        let adapter = Box::new(adapter_builder.build(model_path.as_ref())?);
        let task = TextDetectionTask::new(config.clone());

        Ok(TextDetectionPredictor {
            core: TaskPredictorCore::new(adapter, task, config),
        })
    }
}

impl_task_predictor_builder!(TextDetectionPredictorBuilder, TextDetectionConfig);

impl Default for TextDetectionPredictorBuilder {
    fn default() -> Self {
        Self::new()
    }
}
