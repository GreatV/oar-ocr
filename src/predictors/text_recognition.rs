//! Text Recognition Predictor
//!
//! This module provides a high-level API for text recognition from cropped text images.

use super::builder::PredictorBuilderState;
use crate::core::traits::adapter::AdapterBuilder;
use crate::domain::adapters::TextRecognitionAdapterBuilder;
use crate::domain::tasks::text_recognition::{
    TextRecognitionConfig, TextRecognitionInput, TextRecognitionTask,
};
use crate::predictors::TaskPredictorCore;
use image::RgbImage;
use std::path::{Path, PathBuf};

/// Text recognition prediction result
#[derive(Debug, Clone)]
pub struct TextRecognitionResult {
    /// Recognized text for each input image
    pub texts: Vec<String>,
    /// Confidence scores for each recognition
    pub scores: Vec<f32>,
}

/// Text recognition predictor
pub struct TextRecognitionPredictor {
    core: TaskPredictorCore<TextRecognitionTask>,
}

impl TextRecognitionPredictor {
    /// Create a new builder for the text recognition predictor
    pub fn builder() -> TextRecognitionPredictorBuilder {
        TextRecognitionPredictorBuilder::new()
    }

    /// Predict text in the given images.
    pub fn predict(
        &self,
        images: Vec<RgbImage>,
    ) -> Result<TextRecognitionResult, Box<dyn std::error::Error>> {
        let input = TextRecognitionInput::new(images.to_vec());
        let output = self.core.predict(input)?;
        Ok(TextRecognitionResult {
            texts: output.texts,
            scores: output.scores,
        })
    }
}

/// Builder for text recognition predictor
pub struct TextRecognitionPredictorBuilder {
    state: PredictorBuilderState<TextRecognitionConfig>,
    dict_path: Option<PathBuf>,
}

impl TextRecognitionPredictorBuilder {
    pub fn new() -> Self {
        Self {
            state: PredictorBuilderState::new(TextRecognitionConfig {
                score_threshold: 0.0,
                max_text_length: 100,
            }),
            dict_path: None,
        }
    }

    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.state.config_mut().score_threshold = threshold;
        self
    }

    pub fn max_text_length(mut self, max_length: usize) -> Self {
        self.state.config_mut().max_text_length = max_length;
        self
    }

    pub fn dict_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.dict_path = Some(path.as_ref().to_path_buf());
        self
    }

    pub fn build<P: AsRef<Path>>(
        self,
        model_path: P,
    ) -> Result<TextRecognitionPredictor, Box<dyn std::error::Error>> {
        let Self { state, dict_path } = self;
        let (config, ort_config) = state.into_parts();

        let dict_path = dict_path.ok_or("Dictionary path is required for text recognition")?;

        // Load character dictionary from file
        let character_dict = std::fs::read_to_string(&dict_path)?
            .lines()
            .map(|s| s.to_string())
            .collect::<Vec<String>>();

        let mut adapter_builder = TextRecognitionAdapterBuilder::new()
            .with_config(config.clone())
            .character_dict(character_dict);

        if let Some(ort_cfg) = ort_config {
            adapter_builder = adapter_builder.with_ort_config(ort_cfg);
        }

        let adapter = Box::new(adapter_builder.build(model_path.as_ref())?);
        let task = TextRecognitionTask::new(config.clone());
        Ok(TextRecognitionPredictor {
            core: TaskPredictorCore::new(adapter, task, config),
        })
    }
}

impl_task_predictor_builder!(TextRecognitionPredictorBuilder, TextRecognitionConfig);

impl Default for TextRecognitionPredictorBuilder {
    fn default() -> Self {
        Self::new()
    }
}
