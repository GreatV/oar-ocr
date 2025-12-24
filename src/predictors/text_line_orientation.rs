//! Text Line Orientation Predictor
//!
//! This module provides a high-level API for text line orientation classification.

use super::builder::PredictorBuilderState;
use crate::core::traits::adapter::AdapterBuilder;
use crate::core::traits::task::ImageTaskInput;
use crate::domain::adapters::TextLineOrientationAdapterBuilder;
use crate::domain::tasks::document_orientation::Classification;
use crate::domain::tasks::text_line_orientation::{
    TextLineOrientationConfig, TextLineOrientationTask,
};
use crate::predictors::TaskPredictorCore;
use image::RgbImage;
use std::path::Path;

/// Text line orientation prediction result
#[derive(Debug, Clone)]
pub struct TextLineOrientationResult {
    /// Orientation classifications for each input image
    pub orientations: Vec<Vec<Classification>>,
}

/// Text line orientation predictor
pub struct TextLineOrientationPredictor {
    core: TaskPredictorCore<TextLineOrientationTask>,
}

impl TextLineOrientationPredictor {
    pub fn builder() -> TextLineOrientationPredictorBuilder {
        TextLineOrientationPredictorBuilder::new()
    }

    /// Predict text line orientations in the given images.
    pub fn predict(
        &self,
        images: Vec<RgbImage>,
    ) -> Result<TextLineOrientationResult, Box<dyn std::error::Error>> {
        let input = ImageTaskInput::new(images);
        let output = self.core.predict(input)?;
        Ok(TextLineOrientationResult {
            orientations: output.classifications,
        })
    }
}

pub struct TextLineOrientationPredictorBuilder {
    state: PredictorBuilderState<TextLineOrientationConfig>,
    input_shape: (u32, u32),
}

impl TextLineOrientationPredictorBuilder {
    pub fn new() -> Self {
        Self {
            state: PredictorBuilderState::new(TextLineOrientationConfig {
                score_threshold: 0.5,
                topk: 2,
            }),
            input_shape: (192, 48),
        }
    }

    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.state.config_mut().score_threshold = threshold;
        self
    }

    pub fn topk(mut self, k: usize) -> Self {
        self.state.config_mut().topk = k;
        self
    }

    pub fn input_shape(mut self, shape: (u32, u32)) -> Self {
        self.input_shape = shape;
        self
    }

    pub fn build<P: AsRef<Path>>(
        self,
        model_path: P,
    ) -> Result<TextLineOrientationPredictor, Box<dyn std::error::Error>> {
        let Self { state, input_shape } = self;
        let (config, ort_config) = state.into_parts();
        let mut adapter_builder = TextLineOrientationAdapterBuilder::new()
            .with_config(config.clone())
            .input_shape(input_shape);

        if let Some(ort_cfg) = ort_config {
            adapter_builder = adapter_builder.with_ort_config(ort_cfg);
        }

        let adapter = Box::new(adapter_builder.build(model_path.as_ref())?);
        let task = TextLineOrientationTask::new(config.clone());
        Ok(TextLineOrientationPredictor {
            core: TaskPredictorCore::new(adapter, task, config),
        })
    }
}

impl_task_predictor_builder!(
    TextLineOrientationPredictorBuilder,
    TextLineOrientationConfig
);

impl Default for TextLineOrientationPredictorBuilder {
    fn default() -> Self {
        Self::new()
    }
}
