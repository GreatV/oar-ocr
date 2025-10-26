//! Text Recognition Task Adapter
//!
//! This adapter uses the CRNN model and adapts its output to the TextRecognition task format.

use crate::core::OCRError;
use crate::core::traits::{
    adapter::{AdapterBuilder, AdapterInfo, ModelAdapter},
    task::{Task, TaskType},
};
use crate::domain::tasks::{TextRecognitionConfig, TextRecognitionOutput, TextRecognitionTask};
use crate::models::recognition::crnn::{CRNNModel, CRNNModelBuilder, CRNNPreprocessConfig};
use std::path::Path;

/// Text recognition adapter that uses the CRNN model.
#[derive(Debug)]
pub struct TextRecognitionAdapter {
    /// The underlying CRNN model
    model: CRNNModel,
    /// Adapter information
    info: AdapterInfo,
    /// Task configuration
    config: TextRecognitionConfig,
}

impl ModelAdapter for TextRecognitionAdapter {
    type Task = TextRecognitionTask;

    fn info(&self) -> AdapterInfo {
        self.info.clone()
    }

    fn execute(
        &self,
        input: <Self::Task as Task>::Input,
        config: Option<&<Self::Task as Task>::Config>,
    ) -> Result<<Self::Task as Task>::Output, OCRError> {
        let effective_config = config.unwrap_or(&self.config);

        // Use the CRNN model to recognize text
        let model_output = self.model.forward(input.images)?;

        // Filter by score threshold and adapt to task output
        let filtered: Vec<(String, f32)> = model_output
            .texts
            .into_iter()
            .zip(model_output.scores)
            .filter(|(_, score)| *score >= effective_config.score_threshold)
            .collect();

        let (texts, scores): (Vec<String>, Vec<f32>) = filtered.into_iter().unzip();

        Ok(TextRecognitionOutput { texts, scores })
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn recommended_batch_size(&self) -> usize {
        32
    }
}

/// Builder for text recognition adapter.
pub struct TextRecognitionAdapterBuilder {
    /// Task configuration
    task_config: TextRecognitionConfig,
    /// Model preprocessing configuration
    preprocess_config: CRNNPreprocessConfig,
    /// Character dictionary
    character_dict: Option<Vec<String>>,
    /// Session pool size
    session_pool_size: usize,
    /// ONNX Runtime session configuration
    ort_config: Option<crate::core::config::OrtSessionConfig>,
}

impl TextRecognitionAdapterBuilder {
    /// Creates a new text recognition adapter builder.
    pub fn new() -> Self {
        Self {
            task_config: TextRecognitionConfig::default(),
            preprocess_config: CRNNPreprocessConfig::default(),
            character_dict: None,
            session_pool_size: 1,
            ort_config: None,
        }
    }

    /// Sets the task configuration.
    pub fn with_config(mut self, config: TextRecognitionConfig) -> Self {
        self.task_config = config;
        self
    }

    /// Sets the model input shape.
    pub fn model_input_shape(mut self, shape: [usize; 3]) -> Self {
        self.preprocess_config.model_input_shape = shape;
        self
    }

    /// Sets the character dictionary.
    pub fn character_dict(mut self, character_dict: Vec<String>) -> Self {
        self.character_dict = Some(character_dict);
        self
    }

    /// Sets the score threshold.
    pub fn score_thresh(mut self, score_thresh: f32) -> Self {
        self.task_config.score_threshold = score_thresh;
        self
    }

    /// Sets the session pool size.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.session_pool_size = size;
        self
    }

    /// Sets the maximum image width.
    pub fn max_img_w(mut self, max_img_w: usize) -> Self {
        self.preprocess_config.max_img_w = Some(max_img_w);
        self
    }

    /// Sets the ONNX Runtime session configuration.
    pub fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.ort_config = Some(config);
        self
    }
}

impl AdapterBuilder for TextRecognitionAdapterBuilder {
    type Config = TextRecognitionConfig;
    type Adapter = TextRecognitionAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        // Build the CRNN model
        let mut model_builder = CRNNModelBuilder::new()
            .preprocess_config(self.preprocess_config)
            .session_pool_size(self.session_pool_size);

        if let Some(character_dict) = self.character_dict {
            model_builder = model_builder.character_dict(character_dict);
        }

        if let Some(ort_config) = self.ort_config {
            model_builder = model_builder.with_ort_config(ort_config);
        }

        let model = model_builder.build(model_path)?;

        // Create adapter info
        let info = AdapterInfo::new(
            "TextRecognition",
            "1.0.0",
            TaskType::TextRecognition,
            "Text recognition using CRNN model",
        );

        Ok(TextRecognitionAdapter {
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
        "TextRecognition"
    }
}

impl Default for TextRecognitionAdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}
