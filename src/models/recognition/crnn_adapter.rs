//! CRNN Text Recognition Adapter
//!
//! This module provides an adapter for CRNN text recognition that directly implements
//! preprocessing, ONNX inference, and postprocessing without using predictor wrappers.

use crate::core::inference::OrtInfer;
use crate::core::traits::{
    adapter::{AdapterBuilder, AdapterInfo, ModelAdapter},
    task::Task,
};
use crate::core::{OCRError, Tensor3D, Tensor4D};
use crate::domain::tasks::{TextRecognitionConfig, TextRecognitionOutput, TextRecognitionTask};
use crate::processors::{CTCLabelDecode, NormalizeImage, OCRResize};
use image::RgbImage;
use std::path::Path;

/// Adapter for CRNN text recognition model.
///
/// This adapter directly implements preprocessing, ONNX inference, and postprocessing
/// for CRNN text recognition without using predictor wrappers.
#[derive(Debug)]
pub struct CRNNTextRecognitionAdapter {
    /// ONNX Runtime inference engine
    inference: OrtInfer,
    /// Image resizer for preprocessing
    resizer: OCRResize,
    /// Image normalizer for preprocessing
    normalizer: NormalizeImage,
    /// CTC decoder for postprocessing
    decoder: CTCLabelDecode,
    /// Adapter information
    info: AdapterInfo,
    /// Configuration for recognition
    config: TextRecognitionConfig,
}

impl CRNNTextRecognitionAdapter {
    /// Creates a new CRNN text recognition adapter.
    pub fn new(
        inference: OrtInfer,
        resizer: OCRResize,
        normalizer: NormalizeImage,
        decoder: CTCLabelDecode,
        info: AdapterInfo,
        config: TextRecognitionConfig,
    ) -> Self {
        Self {
            inference,
            resizer,
            normalizer,
            decoder,
            info,
            config,
        }
    }

    /// Preprocesses images for recognition.
    fn preprocess(&self, images: Vec<RgbImage>) -> Result<Tensor4D, OCRError> {
        // Resize images
        let resized_images = self.resizer.apply(&images)?;

        // Convert to DynamicImage for normalization
        let dynamic_images: Vec<image::DynamicImage> = resized_images
            .into_iter()
            .map(image::DynamicImage::ImageRgb8)
            .collect();

        // Normalize and convert to tensor
        let batch_tensor = self.normalizer.normalize_batch_to(dynamic_images)?;

        // Return 4D tensor - OrtInfer will handle the conversion
        Ok(batch_tensor)
    }

    /// Postprocesses model predictions to text strings.
    fn postprocess(
        &self,
        predictions: &Tensor3D,
        config: &TextRecognitionConfig,
    ) -> TextRecognitionOutput {
        // Decode CTC predictions
        let (texts, scores) = self.decoder.apply(predictions);

        // Filter by score threshold
        let filtered: Vec<(String, f32)> = texts
            .into_iter()
            .zip(scores)
            .filter(|(_, score)| *score >= config.score_threshold)
            .collect();

        let (texts, scores): (Vec<String>, Vec<f32>) = filtered.into_iter().unzip();

        TextRecognitionOutput { texts, scores }
    }
}

impl ModelAdapter for CRNNTextRecognitionAdapter {
    type Task = TextRecognitionTask;

    fn info(&self) -> AdapterInfo {
        self.info.clone()
    }

    fn execute(
        &self,
        input: <Self::Task as Task>::Input,
        config: Option<&<Self::Task as Task>::Config>,
    ) -> Result<<Self::Task as Task>::Output, OCRError> {
        // Use provided config or fall back to stored config
        let effective_config = config.unwrap_or(&self.config);

        // Preprocess images
        let batch_tensor = self.preprocess(input.images)?;

        // Run inference
        let predictions = self.inference.infer_3d(&batch_tensor)?;

        // Postprocess predictions
        let output = self.postprocess(&predictions, effective_config);

        Ok(output)
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn recommended_batch_size(&self) -> usize {
        32 // Default batch size for CRNN
    }
}

/// Builder for CRNN text recognition adapter.
pub struct CRNNTextRecognitionAdapterBuilder {
    /// Task configuration
    task_config: TextRecognitionConfig,
    /// Model input shape [channels, height, width]
    model_input_shape: [usize; 3],
    /// Character dictionary
    character_dict: Option<Vec<String>>,
    /// Session pool size for ONNX Runtime
    session_pool_size: usize,
    /// Maximum image width
    max_img_w: Option<usize>,
    /// Optional override for the registered model name
    model_name_override: Option<String>,
}

impl CRNNTextRecognitionAdapterBuilder {
    /// Creates a new CRNN text recognition adapter builder.
    pub fn new() -> Self {
        Self {
            task_config: TextRecognitionConfig::default(),
            model_input_shape: [3, 48, 320], // Default CRNN input shape
            character_dict: None,
            session_pool_size: 1,
            max_img_w: None,
            model_name_override: None,
        }
    }

    /// Sets the model input shape.
    pub fn model_input_shape(mut self, shape: [usize; 3]) -> Self {
        self.model_input_shape = shape;
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
        self.max_img_w = Some(max_img_w);
        self
    }

    /// Sets a custom model name for registry registration.
    pub fn model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name_override = Some(model_name.into());
        self
    }
}

impl Default for CRNNTextRecognitionAdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AdapterBuilder for CRNNTextRecognitionAdapterBuilder {
    type Config = TextRecognitionConfig;
    type Adapter = CRNNTextRecognitionAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        // Create ONNX inference engine
        let inference = if self.session_pool_size > 1 {
            // Use session pool for concurrent inference
            use crate::core::config::CommonBuilderConfig;
            let common_config = CommonBuilderConfig {
                session_pool_size: Some(self.session_pool_size),
                ..Default::default()
            };
            OrtInfer::from_common(&common_config, model_path, Some("x"))?
        } else {
            OrtInfer::new(model_path, Some("x"))?
        };

        // Create resizer
        let resizer = if let Some(max_w) = self.max_img_w {
            OCRResize::with_max_width(
                Some(self.model_input_shape),
                None, // input_shape
                Some(max_w),
            )
        } else {
            OCRResize::new(Some(self.model_input_shape), None)
        };

        // Create normalizer (OCR recognition normalization)
        let normalizer = NormalizeImage::for_ocr_recognition()?;

        // Create CTC decoder
        let decoder = if let Some(char_dict) = self.character_dict {
            CTCLabelDecode::from_string_list(Some(&char_dict), true, false)
        } else {
            // Use default character set
            CTCLabelDecode::new(None, true)
        };

        let mut info = AdapterInfo::new(
            "CRNN",
            "1.0.0",
            crate::core::traits::task::TaskType::TextRecognition,
            "Convolutional Recurrent Neural Network text recognizer",
        );
        if let Some(model_name) = self.model_name_override {
            info.model_name = model_name;
        }

        Ok(CRNNTextRecognitionAdapter::new(
            inference,
            resizer,
            normalizer,
            decoder,
            info,
            self.task_config,
        ))
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.task_config = config;
        self
    }

    fn adapter_type(&self) -> &str {
        "CRNN-TextRecognition"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = CRNNTextRecognitionAdapterBuilder::new();
        assert_eq!(builder.adapter_type(), "CRNN-TextRecognition");
    }

    #[test]
    fn test_builder_with_config() {
        let config = TextRecognitionConfig {
            score_threshold: 0.8,
            max_text_length: 200,
        };
        let builder = CRNNTextRecognitionAdapterBuilder::new().with_config(config);
        assert_eq!(builder.task_config.score_threshold, 0.8);
        assert_eq!(builder.task_config.max_text_length, 200);
    }

    #[test]
    fn test_builder_with_character_dict() {
        let char_dict = vec!["a".to_string(), "b".to_string(), "c".to_string()];
        let builder = CRNNTextRecognitionAdapterBuilder::new().character_dict(char_dict.clone());

        assert_eq!(builder.adapter_type(), "CRNN-TextRecognition");
    }

    #[test]
    fn test_builder_with_session_pool() {
        let builder = CRNNTextRecognitionAdapterBuilder::new().session_pool_size(4);

        assert_eq!(builder.adapter_type(), "CRNN-TextRecognition");
    }

    #[test]
    fn test_builder_with_input_shape() {
        let builder = CRNNTextRecognitionAdapterBuilder::new().model_input_shape([3, 48, 320]);

        assert_eq!(builder.adapter_type(), "CRNN-TextRecognition");
    }
}
