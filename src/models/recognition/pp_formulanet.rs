//! PP-FormulaNet Model
//!
//! This module provides a pure implementation of the PP-FormulaNet formula recognition model.
//! The model is independent of any specific task and can be reused in different contexts.

use crate::core::inference::OrtInfer;
use crate::core::{OCRError, Tensor4D};
use crate::processors::{FormulaPreprocessParams, FormulaPreprocessor};
use image::RgbImage;
use ndarray::{ArrayBase, Axis, Data, Ix2};

/// Preprocessing configuration for PP-FormulaNet model.
#[derive(Debug, Clone)]
pub struct PPFormulaNetPreprocessConfig {
    /// Target size (width, height)
    pub target_size: (u32, u32),
    /// Threshold for binarizing margins during cropping
    pub crop_threshold: u8,
    /// Padding alignment for tensor export
    pub padding_multiple: usize,
    /// Channel-wise normalization mean
    pub normalize_mean: [f32; 3],
    /// Channel-wise normalization std
    pub normalize_std: [f32; 3],
}

impl Default for PPFormulaNetPreprocessConfig {
    fn default() -> Self {
        Self {
            target_size: (384, 384),
            crop_threshold: 200,
            padding_multiple: 16,
            normalize_mean: [0.7931, 0.7931, 0.7931],
            normalize_std: [0.1738, 0.1738, 0.1738],
        }
    }
}

/// Postprocessing configuration for PP-FormulaNet model.
#[derive(Debug, Clone)]
pub struct PPFormulaNetPostprocessConfig {
    /// Start-of-sequence token id
    pub sos_token_id: i64,
    /// End-of-sequence token id
    pub eos_token_id: i64,
}

impl Default for PPFormulaNetPostprocessConfig {
    fn default() -> Self {
        Self {
            sos_token_id: 0,
            eos_token_id: 2,
        }
    }
}

/// Output from PP-FormulaNet model.
#[derive(Debug, Clone)]
pub struct PPFormulaNetModelOutput {
    /// Token IDs for each image in the batch [batch_size, max_length]
    pub token_ids: ndarray::Array2<i64>,
}

/// PP-FormulaNet formula recognition model.
///
/// This is a pure model implementation that handles:
/// - Preprocessing: Image cropping, resizing, and normalization
/// - Inference: Running the ONNX model
/// - Postprocessing: Returning raw token IDs
///
/// The model is independent of any specific task or adapter.
#[derive(Debug)]
pub struct PPFormulaNetModel {
    inference: OrtInfer,
    preprocessor: FormulaPreprocessor,
    #[allow(dead_code)]
    preprocess_config: PPFormulaNetPreprocessConfig,
}

impl PPFormulaNetModel {
    /// Creates a new PP-FormulaNet model.
    pub fn new(
        inference: OrtInfer,
        preprocess_config: PPFormulaNetPreprocessConfig,
    ) -> Result<Self, OCRError> {
        // Create preprocessor
        let params = FormulaPreprocessParams {
            target_size: preprocess_config.target_size,
            crop_threshold: preprocess_config.crop_threshold,
            padding_multiple: preprocess_config.padding_multiple,
            normalize_mean: preprocess_config.normalize_mean,
            normalize_std: preprocess_config.normalize_std,
        };

        let preprocessor = FormulaPreprocessor::new(params);

        Ok(Self {
            inference,
            preprocessor,
            preprocess_config,
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
        // Debug: log input tensor stats
        tracing::info!("Input tensor shape: {:?}", batch_tensor.shape());
        let min_val = batch_tensor.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = batch_tensor
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        tracing::info!("Input tensor min/max: {:.4} / {:.4}", min_val, max_val);

        self.inference.infer_2d_i64(batch_tensor)
    }

    /// Postprocesses model predictions.
    ///
    /// For PP-FormulaNet, we just return the raw token IDs.
    /// The adapter layer will handle tokenization and LaTeX decoding.
    pub fn postprocess(
        &self,
        token_ids: ndarray::Array2<i64>,
        _config: &PPFormulaNetPostprocessConfig,
    ) -> Result<PPFormulaNetModelOutput, OCRError> {
        // Debug: print raw model output shape
        tracing::info!("Model output shape: {:?}", token_ids.shape());

        Ok(PPFormulaNetModelOutput { token_ids })
    }

    /// Runs the complete forward pass: preprocess -> infer -> postprocess.
    pub fn forward(
        &self,
        images: Vec<RgbImage>,
        config: &PPFormulaNetPostprocessConfig,
    ) -> Result<PPFormulaNetModelOutput, OCRError> {
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
        config: &PPFormulaNetPostprocessConfig,
    ) -> Vec<Vec<u32>>
    where
        D: Data<Elem = i64>,
    {
        let mut filtered_tokens = Vec::new();

        for batch_idx in 0..token_ids.shape()[0] {
            let row = token_ids.index_axis(Axis(0), batch_idx);

            // Debug: print first 20 raw token IDs
            let first_tokens: Vec<i64> = row.iter().copied().take(20).collect();
            tracing::info!(
                "First 20 raw tokens for batch {}: {:?}",
                batch_idx,
                first_tokens
            );

            let tokens: Vec<u32> = row
                .iter()
                .copied()
                .take_while(|&id| id != config.eos_token_id)
                .filter(|&id| id >= 0 && id != config.sos_token_id)
                .map(|id| id as u32)
                .collect();

            tracing::debug!(
                "Filtered tokens for batch {}: {:?}",
                batch_idx,
                &tokens[..tokens.len().min(50)]
            );

            filtered_tokens.push(tokens);
        }

        filtered_tokens
    }
}

/// Builder for PP-FormulaNet model.
#[derive(Debug, Default)]
pub struct PPFormulaNetModelBuilder {
    preprocess_config: Option<PPFormulaNetPreprocessConfig>,
    session_pool_size: usize,
    ort_config: Option<crate::core::config::OrtSessionConfig>,
}

impl PPFormulaNetModelBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            preprocess_config: None,
            session_pool_size: 1,
            ort_config: None,
        }
    }

    /// Sets the preprocessing configuration.
    pub fn preprocess_config(mut self, config: PPFormulaNetPreprocessConfig) -> Self {
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

    /// Builds the PP-FormulaNet model.
    pub fn build(self, model_path: &std::path::Path) -> Result<PPFormulaNetModel, OCRError> {
        // Create ONNX inference engine
        let inference = if self.session_pool_size > 1 || self.ort_config.is_some() {
            use crate::core::config::CommonBuilderConfig;
            let common_config = CommonBuilderConfig {
                session_pool_size: Some(self.session_pool_size),
                ort_session: self.ort_config,
                ..Default::default()
            };
            OrtInfer::from_common_with_auto_input(&common_config, model_path)?
        } else {
            OrtInfer::with_auto_input_name(model_path)?
        };

        // Determine target size
        let mut preprocess_config = self.preprocess_config.unwrap_or_default();

        // Try to detect target size from model input shape if not explicitly set
        if preprocess_config.target_size == (384, 384)
            && let Some(detected) = inference.primary_input_shape().and_then(|shape| {
                tracing::debug!("Model input shape: {:?}", shape);
                if shape.len() >= 4 {
                    let height = shape[shape.len() - 2];
                    let width = shape[shape.len() - 1];
                    tracing::debug!("Detected height={}, width={}", height, width);
                    if height > 0 && width > 0 {
                        Some((width as u32, height as u32))
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
        {
            tracing::info!("Using detected target size: {:?}", detected);
            preprocess_config.target_size = detected;
        }

        PPFormulaNetModel::new(inference, preprocess_config)
    }
}
