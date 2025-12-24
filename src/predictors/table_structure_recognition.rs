//! Table Structure Recognition Predictor
//!
//! This module provides a high-level API for table structure recognition.

use super::builder::PredictorBuilderState;
use crate::core::traits::adapter::AdapterBuilder;
use crate::core::traits::task::ImageTaskInput;
use crate::domain::adapters::SLANetWiredAdapterBuilder;
use crate::domain::tasks::table_structure_recognition::{
    TableStructureRecognitionConfig, TableStructureRecognitionTask,
};
use crate::predictors::TaskPredictorCore;
use image::RgbImage;
use std::path::{Path, PathBuf};

/// Table structure recognition prediction result
#[derive(Debug, Clone)]
pub struct TableStructureRecognitionResult {
    /// Recognized table structures in HTML format (one per image)
    pub structures: Vec<Vec<String>>,
    /// Bounding boxes for table cells as 8-point coordinates (one per image)
    pub bboxes: Vec<Vec<Vec<f32>>>,
}

/// Table structure recognition predictor
pub struct TableStructureRecognitionPredictor {
    core: TaskPredictorCore<TableStructureRecognitionTask>,
}

impl TableStructureRecognitionPredictor {
    pub fn builder() -> TableStructureRecognitionPredictorBuilder {
        TableStructureRecognitionPredictorBuilder::new()
    }

    /// Predict table structures in the given images.
    pub fn predict(
        &self,
        images: Vec<RgbImage>,
    ) -> Result<TableStructureRecognitionResult, Box<dyn std::error::Error>> {
        let input = ImageTaskInput::new(images);
        let output = self.core.predict(input)?;
        Ok(TableStructureRecognitionResult {
            structures: output.structures,
            bboxes: output.bboxes,
        })
    }
}

pub struct TableStructureRecognitionPredictorBuilder {
    state: PredictorBuilderState<TableStructureRecognitionConfig>,
    dict_path: Option<PathBuf>,
    /// Custom input shape (height, width). If None, uses adapter default.
    input_shape: Option<(u32, u32)>,
}

impl TableStructureRecognitionPredictorBuilder {
    pub fn new() -> Self {
        Self {
            state: PredictorBuilderState::new(TableStructureRecognitionConfig {
                score_threshold: 0.5,
                max_structure_length: 500,
            }),
            dict_path: None,
            input_shape: None,
        }
    }

    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.state.config_mut().score_threshold = threshold;
        self
    }

    pub fn dict_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.dict_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Sets the input shape for the model.
    ///
    /// If not set, the input shape will be auto-detected from the ONNX model.
    /// Example ONNX shapes: SLANeXt_wired=512×512, SLANet_plus=488×488.
    pub fn input_shape(mut self, height: u32, width: u32) -> Self {
        self.input_shape = Some((height, width));
        self
    }

    pub fn build<P: AsRef<Path>>(
        self,
        model_path: P,
    ) -> Result<TableStructureRecognitionPredictor, Box<dyn std::error::Error>> {
        let Self {
            state,
            dict_path,
            input_shape,
        } = self;
        let (config, ort_config) = state.into_parts();
        let dict_path =
            dict_path.ok_or("Dictionary path is required for table structure recognition")?;

        let mut adapter_builder = SLANetWiredAdapterBuilder::new()
            .with_config(config.clone())
            .dict_path(dict_path);

        if let Some((h, w)) = input_shape {
            adapter_builder = adapter_builder.input_shape((h, w));
        }

        if let Some(ort_cfg) = ort_config {
            adapter_builder = adapter_builder.with_ort_config(ort_cfg);
        }

        let adapter = Box::new(adapter_builder.build(model_path.as_ref())?);
        let task = TableStructureRecognitionTask::new(config.clone());
        Ok(TableStructureRecognitionPredictor {
            core: TaskPredictorCore::new(adapter, task, config),
        })
    }
}

impl_task_predictor_builder!(
    TableStructureRecognitionPredictorBuilder,
    TableStructureRecognitionConfig
);

impl Default for TableStructureRecognitionPredictorBuilder {
    fn default() -> Self {
        Self::new()
    }
}
