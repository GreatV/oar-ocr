//! PP-FormulaNet Model
//!
//! This module provides a pure implementation of the PP-FormulaNet formula recognition model.
//! The model is independent of any specific task and can be reused in different contexts.

use crate::core::OCRError;
use crate::core::inference::{OrtInfer, TensorInput, TensorOutput};
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
    /// Tokenizer vocabulary size. Non-negative IDs at or above this value are
    /// treated as padding/sentinel values emitted by exported ONNX models.
    pub vocab_size: i64,
}

impl Default for PPFormulaNetPostprocessConfig {
    fn default() -> Self {
        Self {
            sos_token_id: 0,
            eos_token_id: 2,
            vocab_size: i64::MAX,
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
    _preprocess_config: PPFormulaNetPreprocessConfig,
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
            _preprocess_config: preprocess_config,
        })
    }

    /// Preprocesses images for formula recognition.
    ///
    /// Returns a batch tensor ready for inference.
    pub fn preprocess(&self, images: Vec<RgbImage>) -> Result<ndarray::Array4<f32>, OCRError> {
        self.preprocessor.preprocess_batch(&images)
    }

    /// Runs inference on the preprocessed batch tensor.
    ///
    /// Returns raw token IDs [batch_size, max_length].
    pub fn infer(
        &self,
        batch_tensor: &ndarray::Array4<f32>,
    ) -> Result<ndarray::Array2<i64>, OCRError> {
        let input_name = self.inference.input_name();
        let inputs = vec![(input_name, TensorInput::Array4(batch_tensor))];

        let outputs = self
            .inference
            .infer(&inputs)
            .map_err(|e| OCRError::Inference {
                model_name: "PP-FormulaNet".to_string(),
                context: format!(
                    "failed to run inference on batch with shape {:?}",
                    batch_tensor.shape()
                ),
                source: Box::new(e),
            })?;

        tracing::debug!(
            "PP-FormulaNet declared output shapes: {:?}",
            self.inference.output_shapes()
        );

        // Some exported PP-FormulaNet ONNX models emit multiple tensors
        // (e.g. token IDs + per-step scores). The token-ID tensor is the
        // unique 2-D i64 output; pick it explicitly rather than trusting
        // graph output order, which has bitten us before when exporters
        // reordered metadata vs ids.
        let i64_2d_count = outputs
            .iter()
            .filter(|(_, t)| matches!(t, TensorOutput::I64 { shape, .. } if shape.len() == 2))
            .count();
        if i64_2d_count != 1 {
            // Defer the (name, dtype, shape) walk to the error path: on the
            // happy path we don't pay for a `Vec` we'd immediately drop.
            let candidates: Vec<(String, &'static str, Vec<i64>)> = outputs
                .iter()
                .map(|(name, t)| (name.clone(), t.dtype_name(), t.shape().to_vec()))
                .collect();
            return Err(OCRError::Inference {
                model_name: "PP-FormulaNet".to_string(),
                context: format!(
                    "expected exactly one 2-D i64 output (token ids); found {} candidate(s) among outputs {:?}",
                    i64_2d_count, candidates
                ),
                source: Box::new(OCRError::InvalidInput {
                    message: "PP-FormulaNet: ambiguous or missing token-id output".to_string(),
                }),
            });
        }
        let (name, tensor) = outputs
            .into_iter()
            .find(|(_, t)| matches!(t, TensorOutput::I64 { shape, .. } if shape.len() == 2))
            .expect("i64_2d_count == 1 checked above");
        tracing::debug!(
            "PP-FormulaNet selected output '{}' dtype={} runtime_shape={:?}",
            name,
            tensor.dtype_name(),
            tensor.shape()
        );

        tensor
            .try_into_array2_i64()
            .map_err(|e| OCRError::Inference {
                model_name: "PP-FormulaNet".to_string(),
                context: format!("failed to convert output '{name}' to 2-D i64 array"),
                source: Box::new(e),
            })
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
        let batch_size = token_ids.shape()[0];
        let mut filtered_tokens = Vec::with_capacity(batch_size);

        for batch_idx in 0..batch_size {
            let row = token_ids.index_axis(Axis(0), batch_idx);

            let tokens: Vec<u32> = row
                .iter()
                .copied()
                .take_while(|&id| id != config.eos_token_id)
                .take_while(|&id| id < 0 || id < config.vocab_size)
                .filter(|&id| id >= 0 && id != config.sos_token_id)
                .map(|id| id as u32)
                .collect();

            filtered_tokens.push(tokens);
        }

        filtered_tokens
    }
}

/// Builder for PP-FormulaNet model.
#[derive(Debug, Default)]
pub struct PPFormulaNetModelBuilder {
    preprocess_config: Option<PPFormulaNetPreprocessConfig>,
    ort_config: Option<crate::core::config::OrtSessionConfig>,
}

impl PPFormulaNetModelBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            preprocess_config: None,
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

    /// Sets the ONNX Runtime session configuration.
    pub fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.ort_config = Some(config);
        self
    }

    /// Builds the PP-FormulaNet model.
    pub fn build(self, model_path: &std::path::Path) -> Result<PPFormulaNetModel, OCRError> {
        // Create ONNX inference engine
        let inference = if self.ort_config.is_some() {
            use crate::core::config::ModelInferenceConfig;
            let common_config = ModelInferenceConfig {
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
        if preprocess_config.target_size == (384, 384)
            && let Some(shape) = inference.primary_input_shape()
            && shape.len() >= 4
        {
            let height = shape[shape.len() - 2];
            let width = shape[shape.len() - 1];
            if height > 0 && width > 0 {
                preprocess_config.target_size = (width as u32, height as u32);
            }
        }

        PPFormulaNetModel::new(inference, preprocess_config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn filter_tokens_stops_at_vocab_sentinel() {
        let token_ids = arr2(&[[0, 42, 49_999, 4_096_990_134i64, 77, 2]]);
        let config = PPFormulaNetPostprocessConfig {
            sos_token_id: 0,
            eos_token_id: 2,
            vocab_size: 50_000,
        };

        let filtered = PPFormulaNetModel::filter_tokens(&token_ids, &config);

        assert_eq!(filtered, vec![vec![42, 49_999]]);
    }

    #[test]
    fn filter_tokens_still_stops_at_eos() {
        let token_ids = arr2(&[[0, 42, 2, 43]]);
        let config = PPFormulaNetPostprocessConfig {
            sos_token_id: 0,
            eos_token_id: 2,
            vocab_size: 50_000,
        };

        let filtered = PPFormulaNetModel::filter_tokens(&token_ids, &config);

        assert_eq!(filtered, vec![vec![42]]);
    }
}
