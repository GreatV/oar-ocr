//! Document Rectification Predictor
//!
//! This module provides a high-level API for document rectification (dewarp).

use super::builder::PredictorBuilderState;
use crate::core::traits::adapter::AdapterBuilder;
use crate::core::traits::task::ImageTaskInput;
use crate::domain::adapters::UVDocRectifierAdapterBuilder;
use crate::domain::tasks::document_rectification::{
    DocumentRectificationConfig, DocumentRectificationTask,
};
use crate::predictors::TaskPredictorCore;
use image::RgbImage;
use std::path::Path;

/// Document rectification prediction result
#[derive(Debug, Clone)]
pub struct DocumentRectificationResult {
    /// Rectified images
    pub images: Vec<RgbImage>,
}

/// Document rectification predictor
pub struct DocumentRectificationPredictor {
    core: TaskPredictorCore<DocumentRectificationTask>,
}

impl DocumentRectificationPredictor {
    pub fn builder() -> DocumentRectificationPredictorBuilder {
        DocumentRectificationPredictorBuilder::new()
    }

    /// Predict document rectification for the given images.
    pub fn predict(
        &self,
        images: Vec<RgbImage>,
    ) -> Result<DocumentRectificationResult, Box<dyn std::error::Error>> {
        let input = ImageTaskInput::new(images);
        let output = self.core.predict(input)?;
        Ok(DocumentRectificationResult {
            images: output.rectified_images,
        })
    }
}

pub struct DocumentRectificationPredictorBuilder {
    state: PredictorBuilderState<DocumentRectificationConfig>,
}

impl DocumentRectificationPredictorBuilder {
    pub fn new() -> Self {
        Self {
            state: PredictorBuilderState::new(DocumentRectificationConfig {
                rec_image_shape: [3, 0, 0],
            }),
        }
    }

    pub fn build<P: AsRef<Path>>(
        self,
        model_path: P,
    ) -> Result<DocumentRectificationPredictor, Box<dyn std::error::Error>> {
        let (config, ort_config) = self.state.into_parts();
        let mut adapter_builder = UVDocRectifierAdapterBuilder::new().with_config(config.clone());

        if let Some(ort_cfg) = ort_config {
            adapter_builder = adapter_builder.with_ort_config(ort_cfg);
        }

        let adapter = Box::new(adapter_builder.build(model_path.as_ref())?);
        let task = DocumentRectificationTask::new(config.clone());
        Ok(DocumentRectificationPredictor {
            core: TaskPredictorCore::new(adapter, task, config),
        })
    }
}

impl_task_predictor_builder!(
    DocumentRectificationPredictorBuilder,
    DocumentRectificationConfig
);

impl Default for DocumentRectificationPredictorBuilder {
    fn default() -> Self {
        Self::new()
    }
}
