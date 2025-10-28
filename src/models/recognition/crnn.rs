//! CRNN (Convolutional Recurrent Neural Network) Model
//!
//! This module provides a pure implementation of the CRNN text recognition model.
//! The model handles preprocessing, inference, and postprocessing independently of tasks.

use crate::core::inference::OrtInfer;
use crate::core::{OCRError, Tensor3D, Tensor4D};
use crate::processors::{CTCLabelDecode, NormalizeImage, OCRResize};
use image::RgbImage;
use std::path::Path;

/// CRNN model output containing recognized text and confidence scores.
#[derive(Debug, Clone)]
pub struct CRNNModelOutput {
    /// Recognized text strings for each image in the batch
    pub texts: Vec<String>,
    /// Confidence scores for each recognized text
    pub scores: Vec<f32>,
}

/// Pure CRNN model implementation.
///
/// This model implements the CRNN architecture for text recognition.
#[derive(Debug)]
pub struct CRNNModel {
    /// ONNX Runtime inference engine
    inference: OrtInfer,
    /// Image resizer for preprocessing
    resizer: OCRResize,
    /// Image normalizer for preprocessing
    normalizer: NormalizeImage,
    /// CTC decoder for postprocessing
    decoder: CTCLabelDecode,
}

impl CRNNModel {
    /// Creates a new CRNN model.
    pub fn new(
        inference: OrtInfer,
        resizer: OCRResize,
        normalizer: NormalizeImage,
        decoder: CTCLabelDecode,
    ) -> Self {
        Self {
            inference,
            resizer,
            normalizer,
            decoder,
        }
    }

    /// Preprocesses images for recognition.
    ///
    /// # Arguments
    ///
    /// * `images` - Input RGB images
    ///
    /// # Returns
    ///
    /// A 4D tensor ready for inference
    pub fn preprocess(&self, images: Vec<RgbImage>) -> Result<Tensor4D, OCRError> {
        // Resize images
        let resized_images = self.resizer.apply(&images)?;

        // Convert to DynamicImage for normalization
        let dynamic_images: Vec<image::DynamicImage> = resized_images
            .into_iter()
            .map(image::DynamicImage::ImageRgb8)
            .collect();

        // Normalize and convert to tensor
        let batch_tensor = self.normalizer.normalize_batch_to(dynamic_images)?;

        Ok(batch_tensor)
    }

    /// Runs inference on the preprocessed tensor.
    ///
    /// # Arguments
    ///
    /// * `batch_tensor` - Preprocessed 4D tensor
    ///
    /// # Returns
    ///
    /// A 3D tensor containing CTC predictions
    pub fn infer(&self, batch_tensor: &Tensor4D) -> Result<Tensor3D, OCRError> {
        self.inference
            .infer_3d(batch_tensor)
            .map_err(|e| OCRError::Inference {
                model_name: "CRNN".to_string(),
                context: format!(
                    "failed to run inference on batch with shape {:?}",
                    batch_tensor.shape()
                ),
                source: Box::new(e),
            })
    }

    /// Postprocesses model predictions to text strings.
    ///
    /// # Arguments
    ///
    /// * `predictions` - 3D tensor from model inference
    ///
    /// # Returns
    ///
    /// Model output containing recognized texts and scores
    pub fn postprocess(&self, predictions: &Tensor3D) -> CRNNModelOutput {
        // Decode CTC predictions
        let (texts, scores) = self.decoder.apply(predictions);

        CRNNModelOutput { texts, scores }
    }

    /// Runs the complete forward pass: preprocess -> infer -> postprocess.
    ///
    /// # Arguments
    ///
    /// * `images` - Input RGB images
    ///
    /// # Returns
    ///
    /// Model output containing recognized texts and scores
    pub fn forward(&self, images: Vec<RgbImage>) -> Result<CRNNModelOutput, OCRError> {
        let batch_tensor = self.preprocess(images)?;
        let predictions = self.infer(&batch_tensor)?;
        Ok(self.postprocess(&predictions))
    }
}

/// Configuration for CRNN model preprocessing.
#[derive(Debug, Clone)]
pub struct CRNNPreprocessConfig {
    /// Model input shape [channels, height, width]
    pub model_input_shape: [usize; 3],
    /// Maximum image width (None for dynamic width)
    pub max_img_w: Option<usize>,
}

impl Default for CRNNPreprocessConfig {
    fn default() -> Self {
        Self {
            model_input_shape: [3, 48, 320],
            max_img_w: None,
        }
    }
}

/// Builder for CRNN model.
pub struct CRNNModelBuilder {
    /// Preprocessing configuration
    preprocess_config: CRNNPreprocessConfig,
    /// Character dictionary
    character_dict: Option<Vec<String>>,
    /// Session pool size for ONNX Runtime
    session_pool_size: usize,
    /// ONNX Runtime session configuration
    ort_config: Option<crate::core::config::OrtSessionConfig>,
}

impl CRNNModelBuilder {
    /// Creates a new CRNN model builder with default settings.
    pub fn new() -> Self {
        Self {
            preprocess_config: CRNNPreprocessConfig::default(),
            character_dict: None,
            session_pool_size: 1,
            ort_config: None,
        }
    }

    /// Sets the preprocessing configuration.
    pub fn preprocess_config(mut self, config: CRNNPreprocessConfig) -> Self {
        self.preprocess_config = config;
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

    /// Sets the maximum image width.
    pub fn max_img_w(mut self, max_img_w: usize) -> Self {
        self.preprocess_config.max_img_w = Some(max_img_w);
        self
    }

    /// Sets the session pool size.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.session_pool_size = size;
        self
    }

    /// Sets the ONNX Runtime session configuration.
    pub fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.ort_config = Some(config);
        self
    }

    /// Builds the CRNN model.
    pub fn build(self, model_path: &Path) -> Result<CRNNModel, OCRError> {
        // Create ONNX inference engine
        let inference = if self.session_pool_size > 1 || self.ort_config.is_some() {
            // Use session pool for concurrent inference
            let common = crate::core::config::ModelInferenceConfig {
                session_pool_size: Some(self.session_pool_size),
                ort_session: self.ort_config,
                ..Default::default()
            };
            OrtInfer::from_config(&common, model_path, None)?
        } else {
            // Single session
            OrtInfer::new(model_path, None)?
        };

        // Create resizer
        let resizer = OCRResize::new(Some(self.preprocess_config.model_input_shape), None);

        // Create normalizer with PaddleOCR parameters
        // PaddleOCR uses: (x / 255 - 0.5) / 0.5 which normalizes to [-1, 1]
        // This is equivalent to: scale=1.0/255.0, mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]
        let normalizer = NormalizeImage::new(
            Some(1.0 / 255.0),
            Some(vec![0.5, 0.5, 0.5]),
            Some(vec![0.5, 0.5, 0.5]),
            None,
        )?;

        // Create CTC decoder
        let decoder = if let Some(character_dict) = self.character_dict {
            CTCLabelDecode::from_string_list(Some(&character_dict), true, false)
        } else {
            // Use default character dictionary
            CTCLabelDecode::new(None, true)
        };

        Ok(CRNNModel::new(inference, resizer, normalizer, decoder))
    }
}

impl Default for CRNNModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}
