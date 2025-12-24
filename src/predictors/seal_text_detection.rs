//! Seal Text Detection Predictor
//!
//! This module provides a high-level API for seal text detection in images.

use super::builder::PredictorBuilderState;
use crate::core::traits::adapter::AdapterBuilder;
use crate::core::traits::task::ImageTaskInput;
use crate::domain::adapters::SealTextDetectionAdapterBuilder;
use crate::domain::tasks::seal_text_detection::{SealTextDetectionConfig, SealTextDetectionTask};
use crate::predictors::TaskPredictorCore;
use image::RgbImage;
use std::path::Path;

/// Seal text detection prediction result
#[derive(Debug, Clone)]
pub struct SealTextDetectionResult {
    /// Detected seal text regions for each input image
    pub detections: Vec<Vec<crate::domain::tasks::text_detection::Detection>>,
}

/// Seal text detection predictor
pub struct SealTextDetectionPredictor {
    core: TaskPredictorCore<SealTextDetectionTask>,
}

impl SealTextDetectionPredictor {
    pub fn builder() -> SealTextDetectionPredictorBuilder {
        SealTextDetectionPredictorBuilder::new()
    }

    /// Predict seal text regions in the given images.
    pub fn predict(
        &self,
        images: Vec<RgbImage>,
    ) -> Result<SealTextDetectionResult, Box<dyn std::error::Error>> {
        let input = ImageTaskInput::new(images);
        let output = self.core.predict(input)?;
        Ok(SealTextDetectionResult {
            detections: output.detections,
        })
    }
}

pub struct SealTextDetectionPredictorBuilder {
    state: PredictorBuilderState<SealTextDetectionConfig>,
}

impl SealTextDetectionPredictorBuilder {
    pub fn new() -> Self {
        Self {
            state: PredictorBuilderState::new(SealTextDetectionConfig {
                score_threshold: 0.2,
                box_threshold: 0.6,
                unclip_ratio: 0.5,
                max_candidates: 1000,
            }),
        }
    }

    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.state.config_mut().score_threshold = threshold;
        self
    }

    pub fn build<P: AsRef<Path>>(
        self,
        model_path: P,
    ) -> Result<SealTextDetectionPredictor, Box<dyn std::error::Error>> {
        let (config, ort_config) = self.state.into_parts();
        let mut adapter_builder =
            SealTextDetectionAdapterBuilder::new().with_config(config.clone());

        if let Some(ort_cfg) = ort_config {
            adapter_builder = adapter_builder.with_ort_config(ort_cfg);
        }

        let adapter = Box::new(adapter_builder.build(model_path.as_ref())?);
        let task = SealTextDetectionTask::with_config(config.clone());
        Ok(SealTextDetectionPredictor {
            core: TaskPredictorCore::new(adapter, task, config),
        })
    }
}

impl_task_predictor_builder!(SealTextDetectionPredictorBuilder, SealTextDetectionConfig);

impl Default for SealTextDetectionPredictorBuilder {
    fn default() -> Self {
        Self::new()
    }
}
