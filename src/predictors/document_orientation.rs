//! Document Orientation Predictor
//!
//! This module provides a high-level API for document orientation classification.

use super::builder::PredictorBuilderState;
use crate::core::traits::adapter::AdapterBuilder;
use crate::core::traits::task::ImageTaskInput;
use crate::domain::adapters::DocumentOrientationAdapterBuilder;
use crate::domain::tasks::document_orientation::{
    Classification, DocumentOrientationConfig, DocumentOrientationTask,
};
use crate::predictors::TaskPredictorCore;
use image::RgbImage;
use std::path::Path;

/// Document orientation prediction result
#[derive(Debug, Clone)]
pub struct DocumentOrientationResult {
    /// Orientation classifications for each input image
    pub orientations: Vec<Vec<Classification>>,
}

/// Document orientation predictor
pub struct DocumentOrientationPredictor {
    core: TaskPredictorCore<DocumentOrientationTask>,
}

impl DocumentOrientationPredictor {
    /// Create a new builder for the document orientation predictor
    pub fn builder() -> DocumentOrientationPredictorBuilder {
        DocumentOrientationPredictorBuilder::new()
    }

    /// Predict document orientations in the given images.
    pub fn predict(
        &self,
        images: Vec<RgbImage>,
    ) -> Result<DocumentOrientationResult, Box<dyn std::error::Error>> {
        let input = ImageTaskInput::new(images);
        let output = self.core.predict(input)?;
        Ok(DocumentOrientationResult {
            orientations: output.classifications,
        })
    }
}

/// Builder for document orientation predictor
pub struct DocumentOrientationPredictorBuilder {
    state: PredictorBuilderState<DocumentOrientationConfig>,
    input_shape: (u32, u32),
}

impl DocumentOrientationPredictorBuilder {
    pub fn new() -> Self {
        Self {
            state: PredictorBuilderState::new(DocumentOrientationConfig {
                score_threshold: 0.5,
                topk: 4,
            }),
            input_shape: (224, 224),
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
    ) -> Result<DocumentOrientationPredictor, Box<dyn std::error::Error>> {
        let Self { state, input_shape } = self;
        let (config, ort_config) = state.into_parts();
        let mut adapter_builder = DocumentOrientationAdapterBuilder::new()
            .with_config(config.clone())
            .input_shape(input_shape);

        if let Some(ort_cfg) = ort_config {
            adapter_builder = adapter_builder.with_ort_config(ort_cfg);
        }

        let adapter = Box::new(adapter_builder.build(model_path.as_ref())?);
        let task = DocumentOrientationTask::new(config.clone());
        Ok(DocumentOrientationPredictor {
            core: TaskPredictorCore::new(adapter, task, config),
        })
    }
}

impl_task_predictor_builder!(
    DocumentOrientationPredictorBuilder,
    DocumentOrientationConfig
);

impl Default for DocumentOrientationPredictorBuilder {
    fn default() -> Self {
        Self::new()
    }
}
