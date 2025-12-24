//! Table Classification Predictor
//!
//! This module provides a high-level API for table classification (wired vs wireless tables).

use super::builder::PredictorBuilderState;
use crate::core::traits::adapter::AdapterBuilder;
use crate::core::traits::task::ImageTaskInput;
use crate::domain::adapters::TableClassificationAdapterBuilder;
use crate::domain::tasks::document_orientation::Classification;
use crate::domain::tasks::table_classification::{
    TableClassificationConfig, TableClassificationTask,
};
use crate::predictors::TaskPredictorCore;
use image::RgbImage;
use std::path::Path;

/// Table classification prediction result
#[derive(Debug, Clone)]
pub struct TableClassificationResult {
    /// Classification results for each input image
    pub classifications: Vec<Vec<Classification>>,
}

/// Table classification predictor
pub struct TableClassificationPredictor {
    core: TaskPredictorCore<TableClassificationTask>,
}

impl TableClassificationPredictor {
    /// Create a new builder for the table classification predictor
    pub fn builder() -> TableClassificationPredictorBuilder {
        TableClassificationPredictorBuilder::new()
    }

    /// Predict table classifications in the given images.
    pub fn predict(
        &self,
        images: Vec<RgbImage>,
    ) -> Result<TableClassificationResult, Box<dyn std::error::Error>> {
        let input = ImageTaskInput::new(images);
        let output = self.core.predict(input)?;
        Ok(TableClassificationResult {
            classifications: output.classifications,
        })
    }
}

/// Builder for table classification predictor
pub struct TableClassificationPredictorBuilder {
    state: PredictorBuilderState<TableClassificationConfig>,
    input_shape: (u32, u32),
}

impl TableClassificationPredictorBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            state: PredictorBuilderState::new(TableClassificationConfig {
                score_threshold: 0.5,
                topk: 2,
            }),
            input_shape: (224, 224),
        }
    }

    /// Set the score threshold
    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.state.config_mut().score_threshold = threshold;
        self
    }

    /// Set the top-k predictions to return
    pub fn topk(mut self, k: usize) -> Self {
        self.state.config_mut().topk = k;
        self
    }

    /// Set the model input shape (height, width)
    pub fn input_shape(mut self, shape: (u32, u32)) -> Self {
        self.input_shape = shape;
        self
    }

    /// Build the table classification predictor
    pub fn build<P: AsRef<Path>>(
        self,
        model_path: P,
    ) -> Result<TableClassificationPredictor, Box<dyn std::error::Error>> {
        let Self { state, input_shape } = self;
        let (config, ort_config) = state.into_parts();
        let mut adapter_builder = TableClassificationAdapterBuilder::new()
            .with_config(config.clone())
            .input_shape(input_shape);

        if let Some(ort_cfg) = ort_config {
            adapter_builder = adapter_builder.with_ort_config(ort_cfg);
        }

        let adapter = Box::new(adapter_builder.build(model_path.as_ref())?);
        let task = TableClassificationTask::new(config.clone());

        Ok(TableClassificationPredictor {
            core: TaskPredictorCore::new(adapter, task, config),
        })
    }
}

impl_task_predictor_builder!(
    TableClassificationPredictorBuilder,
    TableClassificationConfig
);

impl Default for TableClassificationPredictorBuilder {
    fn default() -> Self {
        Self::new()
    }
}
