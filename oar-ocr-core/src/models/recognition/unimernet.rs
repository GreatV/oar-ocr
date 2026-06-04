//! UniMERNet Model
//!
//! This module provides a pure implementation of the UniMERNet formula recognition model.
//! The model is independent of any specific task and can be reused in different contexts.

use crate::core::OCRError;
use crate::core::config::{OrtExecutionProvider, OrtGraphOptimizationLevel, OrtSessionConfig};
use crate::core::inference::{OrtInfer, TensorInput};
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
    /// Tokenizer vocabulary size. Non-negative IDs at or above this value are
    /// treated as padding/sentinel values emitted by exported ONNX models.
    pub vocab_size: i64,
}

impl Default for UniMERNetPostprocessConfig {
    fn default() -> Self {
        Self {
            sos_token_id: 0,
            eos_token_id: 2,
            vocab_size: i64::MAX,
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
                model_name: "UniMERNet".to_string(),
                context: format!(
                    "failed to run inference on batch with shape {:?}",
                    batch_tensor.shape()
                ),
                source: Box::new(e),
            })?;

        let output = outputs
            .into_iter()
            .next()
            .ok_or_else(|| OCRError::InvalidInput {
                message: "UniMERNet: no output returned from inference".to_string(),
            })?;

        output
            .1
            .try_into_array2_i64()
            .map_err(|e| OCRError::Inference {
                model_name: "UniMERNet".to_string(),
                context: "failed to convert output to 2D i64 array".to_string(),
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
                .take_while(|&id| id < 0 || id < config.vocab_size)
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
    ort_config: Option<crate::core::config::OrtSessionConfig>,
}

impl UniMERNetModelBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            preprocess_config: None,
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

    /// Sets the ONNX Runtime session configuration.
    pub fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.ort_config = Some(config);
        self
    }

    /// Builds the UniMERNet model.
    pub fn build(self, model_path: &std::path::Path) -> Result<UniMERNetModel, OCRError> {
        // Create ONNX inference engine
        let ort_config = self.ort_config.map(Self::configure_unimernet_ort_for_cuda);

        let inference = if ort_config.is_some() {
            use crate::core::config::ModelInferenceConfig;
            let common_config = ModelInferenceConfig {
                ort_session: ort_config,
                // Identify the model so name-based configuration switching works
                // and it is not reported as "unknown_model".
                model_name: Some("UniMERNet".to_string()),
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

    fn configure_unimernet_ort_for_cuda(mut config: OrtSessionConfig) -> OrtSessionConfig {
        if !Self::uses_cuda(&config) {
            return config;
        }

        config.optimization_level = Some(OrtGraphOptimizationLevel::Level1);
        if config.enable_mem_pattern.is_none() {
            config.enable_mem_pattern = Some(false);
        }

        let entries = config
            .session_config_entries
            .get_or_insert_with(Default::default);
        let disabled_optimizers = entries
            .entry("optimization.disable_specified_optimizers".to_string())
            .or_default();
        if !disabled_optimizers
            .split(',')
            .any(|name| name.trim() == "ConstantFolding")
        {
            if !disabled_optimizers.trim().is_empty() {
                disabled_optimizers.push(',');
            }
            disabled_optimizers.push_str("ConstantFolding");
        }

        config
    }

    fn uses_cuda(config: &OrtSessionConfig) -> bool {
        config
            .execution_providers
            .as_ref()
            .is_some_and(|providers| {
                providers
                    .iter()
                    .any(|provider| matches!(provider, OrtExecutionProvider::CUDA { .. }))
            })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn filter_tokens_stops_at_vocab_sentinel() {
        let token_ids = arr2(&[[0, 42, 49_999, 4_096_990_134i64, 77, 2]]);
        let config = UniMERNetPostprocessConfig {
            sos_token_id: 0,
            eos_token_id: 2,
            vocab_size: 50_000,
        };

        let filtered = UniMERNetModel::filter_tokens(&token_ids, &config);

        assert_eq!(filtered, vec![vec![42, 49_999]]);
    }

    #[test]
    fn filter_tokens_still_stops_at_eos() {
        let token_ids = arr2(&[[0, 42, 2, 43]]);
        let config = UniMERNetPostprocessConfig {
            sos_token_id: 0,
            eos_token_id: 2,
            vocab_size: 50_000,
        };

        let filtered = UniMERNetModel::filter_tokens(&token_ids, &config);

        assert_eq!(filtered, vec![vec![42]]);
    }

    #[test]
    fn cuda_config_disables_constant_folding_for_unimernet() {
        let config = OrtSessionConfig::new().with_execution_providers(vec![
            OrtExecutionProvider::CUDA {
                device_id: Some(0),
                gpu_mem_limit: None,
                arena_extend_strategy: None,
                cudnn_conv_algo_search: None,
                cudnn_conv_use_max_workspace: None,
            },
            OrtExecutionProvider::CPU,
        ]);

        let configured = UniMERNetModelBuilder::configure_unimernet_ort_for_cuda(config);

        assert!(matches!(
            configured.optimization_level,
            Some(OrtGraphOptimizationLevel::Level1)
        ));
        assert_eq!(configured.enable_mem_pattern, Some(false));
        assert_eq!(
            configured
                .session_config_entries
                .as_ref()
                .and_then(|entries| entries.get("optimization.disable_specified_optimizers"))
                .map(String::as_str),
            Some("ConstantFolding")
        );
    }

    #[test]
    fn cpu_config_keeps_unimernet_ort_config_unchanged() {
        let config =
            OrtSessionConfig::new().with_optimization_level(OrtGraphOptimizationLevel::All);

        let configured = UniMERNetModelBuilder::configure_unimernet_ort_for_cuda(config);

        assert!(matches!(
            configured.optimization_level,
            Some(OrtGraphOptimizationLevel::All)
        ));
        assert!(configured.session_config_entries.is_none());
    }
}
