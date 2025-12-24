//! UniMERNet Model
//!
//! This module provides a pure implementation of the UniMERNet formula recognition model.
//! The model is independent of any specific task and can be reused in different contexts.

use crate::core::inference::OrtInfer;
use crate::core::{OCRError, Tensor4D};
use crate::processors::{UniMERNetPreprocessParams, UniMERNetPreprocessor};
use image::RgbImage;
use ndarray::{ArrayBase, Axis, Data, Ix2};

/// Preprocessing configuration for UniMERNet model.
#[derive(Debug, Clone)]
pub struct UniMERNetPreprocessConfig {
    /// Target size (width, height)
    pub target_size: (u32, u32),
    /// Threshold for binarizing margins during cropping
    pub crop_threshold: u8,
    /// Padding alignment for tensor export (UniMERNet uses 32 instead of 16)
    pub padding_multiple: usize,
    /// Channel-wise normalization mean
    pub normalize_mean: [f32; 3],
    /// Channel-wise normalization std
    pub normalize_std: [f32; 3],
}

impl Default for UniMERNetPreprocessConfig {
    fn default() -> Self {
        Self {
            target_size: (672, 192), // UniMERNet uses (672, 192) by default
            crop_threshold: 200,
            padding_multiple: 32, // UniMERNet uses 32 instead of 16
            normalize_mean: [0.7931, 0.7931, 0.7931],
            normalize_std: [0.1738, 0.1738, 0.1738],
        }
    }
}

/// Postprocessing configuration for UniMERNet model.
#[derive(Debug, Clone)]
pub struct UniMERNetPostprocessConfig {
    /// Start-of-sequence token id
    pub sos_token_id: i64,
    /// End-of-sequence token id
    pub eos_token_id: i64,
}

impl Default for UniMERNetPostprocessConfig {
    fn default() -> Self {
        Self {
            sos_token_id: 0,
            eos_token_id: 2,
        }
    }
}

/// Output from UniMERNet model.
#[derive(Debug, Clone)]
pub struct UniMERNetModelOutput {
    /// Token IDs for each image in the batch [batch_size, max_length]
    pub token_ids: ndarray::Array2<i64>,
}

/// UniMERNet formula recognition model.
///
/// This is a pure model implementation that handles:
/// - Preprocessing: Image cropping, resizing, and normalization using UniMERNet-specific logic
/// - Inference: Running the ONNX model
/// - Postprocessing: Returning raw token IDs
///
/// The model is independent of any specific task or adapter.
#[derive(Debug)]
pub struct UniMERNetModel {
    inference: OrtInfer,
    preprocessor: UniMERNetPreprocessor,
    _preprocess_config: UniMERNetPreprocessConfig,
}

impl UniMERNetModel {
    /// Creates a new UniMERNet model.
    pub fn new(
        inference: OrtInfer,
        preprocess_config: UniMERNetPreprocessConfig,
    ) -> Result<Self, OCRError> {
        // Create UniMERNet-specific preprocessor
        let params = UniMERNetPreprocessParams {
            target_size: preprocess_config.target_size,
            crop_threshold: preprocess_config.crop_threshold,
            padding_multiple: preprocess_config.padding_multiple,
            normalize_mean: preprocess_config.normalize_mean,
            normalize_std: preprocess_config.normalize_std,
        };

        let preprocessor = UniMERNetPreprocessor::new(params);

        Ok(Self {
            inference,
            preprocessor,
            _preprocess_config: preprocess_config,
        })
    }

    /// Preprocesses images for formula recognition.
    ///
    /// Returns a batch tensor ready for inference.
    pub fn preprocess(&self, images: Vec<RgbImage>) -> Result<Tensor4D, OCRError> {
        self.preprocessor.preprocess_batch(&images)
    }

    /// Runs inference on the preprocessed batch tensor.
    ///
    /// Returns raw token IDs [batch_size, max_length].
    pub fn infer(&self, batch_tensor: &Tensor4D) -> Result<ndarray::Array2<i64>, OCRError> {
        self.inference
            .infer_2d_i64(batch_tensor)
            .map_err(|e| OCRError::Inference {
                model_name: "UniMERNet".to_string(),
                context: format!(
                    "failed to run inference on batch with shape {:?}",
                    batch_tensor.shape()
                ),
                source: Box::new(e),
            })
    }

    /// Postprocesses model predictions.
    ///
    /// For UniMERNet, we just return the raw token IDs.
    /// The adapter layer will handle tokenization and LaTeX decoding.
    pub fn postprocess(
        &self,
        token_ids: ndarray::Array2<i64>,
        _config: &UniMERNetPostprocessConfig,
    ) -> Result<UniMERNetModelOutput, OCRError> {
        Ok(UniMERNetModelOutput { token_ids })
    }

    /// Runs the complete forward pass: preprocess -> infer -> postprocess.
    pub fn forward(
        &self,
        images: Vec<RgbImage>,
        config: &UniMERNetPostprocessConfig,
    ) -> Result<UniMERNetModelOutput, OCRError> {
        let batch_tensor = self.preprocess(images)?;
        let token_ids = self.infer(&batch_tensor)?;
        let output = self.postprocess(token_ids, config)?;
        Ok(output)
    }

    /// Helper method to filter tokens based on configuration.
    ///
    /// This is used by adapters to filter out special tokens before decoding.
    pub fn filter_tokens<D>(
        token_ids: &ArrayBase<D, Ix2>,
        config: &UniMERNetPostprocessConfig,
    ) -> Vec<Vec<u32>>
    where
        D: Data<Elem = i64>,
    {
        let batch_size = token_ids.shape()[0];
        let mut filtered_tokens = Vec::with_capacity(batch_size);

        for batch_idx in 0..batch_size {
            let row = token_ids.index_axis(Axis(0), batch_idx);

            let tokens: Vec<u32> = row
                .iter()
                .copied()
                .take_while(|&id| id != config.eos_token_id)
                .filter(|&id| id >= 0 && id != config.sos_token_id)
                .map(|id| id as u32)
                .collect();

            filtered_tokens.push(tokens);
        }

        filtered_tokens
    }
}

/// Builder for UniMERNet model.
#[derive(Debug, Default)]
pub struct UniMERNetModelBuilder {
    preprocess_config: Option<UniMERNetPreprocessConfig>,
    session_pool_size: usize,
    ort_config: Option<crate::core::config::OrtSessionConfig>,
}

impl UniMERNetModelBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            preprocess_config: None,
            session_pool_size: 1,
            ort_config: None,
        }
    }

    /// Sets the preprocessing configuration.
    pub fn preprocess_config(mut self, config: UniMERNetPreprocessConfig) -> Self {
        self.preprocess_config = Some(config);
        self
    }

    /// Sets the target size.
    pub fn target_size(mut self, width: u32, height: u32) -> Self {
        let mut config = self.preprocess_config.unwrap_or_default();
        config.target_size = (width, height);
        self.preprocess_config = Some(config);
        self
    }

    /// Sets the padding multiple.
    pub fn padding_multiple(mut self, multiple: usize) -> Self {
        let mut config = self.preprocess_config.unwrap_or_default();
        config.padding_multiple = multiple;
        self.preprocess_config = Some(config);
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

    /// Builds the UniMERNet model.
    pub fn build(self, model_path: &std::path::Path) -> Result<UniMERNetModel, OCRError> {
        // Create ONNX inference engine
        let inference = if self.session_pool_size > 1 || self.ort_config.is_some() {
            use crate::core::config::ModelInferenceConfig;
            let common_config = ModelInferenceConfig {
                session_pool_size: Some(self.session_pool_size),
                ort_session: self.ort_config,
                ..Default::default()
            };
            OrtInfer::from_config(&common_config, model_path, None)?
        } else {
            OrtInfer::new(model_path, None)?
        };

        // Determine target size
        let mut preprocess_config = self.preprocess_config.unwrap_or_default();

        // Try to detect target size from model input shape if not explicitly set
        if preprocess_config.target_size == (672, 192)
            && let Some(shape) = inference.primary_input_shape()
            && shape.len() >= 4
        {
            let height = shape[shape.len() - 2];
            let width = shape[shape.len() - 1];
            if height > 0 && width > 0 {
                preprocess_config.target_size = (width as u32, height as u32);
            }
        }

        UniMERNetModel::new(inference, preprocess_config)
    }
}
