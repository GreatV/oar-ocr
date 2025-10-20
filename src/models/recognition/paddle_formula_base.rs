//! Base implementations for PaddleOCR-style formula recognition adapters.

use crate::core::inference::OrtInfer;
use crate::core::traits::{
    adapter::{AdapterInfo, ModelAdapter},
    task::{Task, TaskType},
};
use crate::core::{OCRError, Tensor4D};
use crate::domain::tasks::{
    FormulaRecognitionConfig, FormulaRecognitionOutput, FormulaRecognitionTask,
};
use crate::processors::{FormulaPreprocessParams, FormulaPreprocessor, normalize_latex};
use image::RgbImage;
use ndarray::{ArrayBase, Axis, Data, Ix2};
use std::fmt::Debug;
use std::marker::PhantomData;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// Specification hook for Paddle-style formula models.
pub trait PaddleFormulaSpec: Send + Sync + Debug + 'static {
    /// Display name reported by adapter info.
    fn model_name() -> &'static str;
    /// Human readable model description.
    fn description() -> &'static str;
    /// Adapter semantic version.
    fn adapter_version() -> &'static str {
        "1.0.0"
    }
    /// Start-of-sequence token id.
    fn sos_token_id() -> i64 {
        0
    }
    /// End-of-sequence token id.
    fn eos_token_id() -> i64 {
        2
    }
    /// Threshold for binarizing margins during cropping.
    fn crop_threshold() -> u8 {
        200
    }
    /// Padding alignment for tensor export.
    fn padding_multiple() -> usize {
        16
    }
    /// Optional fallback target size when model input dims are dynamic.
    fn default_target_size() -> Option<(u32, u32)> {
        None
    }
    /// Channel-wise normalization mean applied during preprocessing.
    fn normalize_mean() -> [f32; 3] {
        [0.7931, 0.7931, 0.7931]
    }
    /// Channel-wise normalization std applied during preprocessing.
    fn normalize_std() -> [f32; 3] {
        [0.1738, 0.1738, 0.1738]
    }
    /// Optional tokenizer path fallback relative to the model dir.
    fn default_tokenizer_filename() -> &'static str {
        "tokenizer.json"
    }
    /// Normalization hook applied to decoded LaTeX.
    fn normalize_latex(text: &str) -> String {
        normalize_latex(text)
    }
    /// Filters tokens that should be removed before decoding.
    fn should_keep_token(id: i64) -> bool {
        id >= 0 && id != Self::sos_token_id()
    }
}

/// Internal adapter shared by concrete model implementations.
#[derive(Debug)]
pub struct PaddleFormulaAdapterBase<S: PaddleFormulaSpec> {
    pub(crate) inference: OrtInfer,
    pub(crate) tokenizer: Tokenizer,
    pub(crate) preprocessor: FormulaPreprocessor,
    pub(crate) info: AdapterInfo,
    pub(crate) config: FormulaRecognitionConfig,
    spec: PhantomData<S>,
}

impl<S: PaddleFormulaSpec> PaddleFormulaAdapterBase<S> {
    /// Creates an adapter base from its components.
    pub fn new(
        inference: OrtInfer,
        tokenizer: Tokenizer,
        preprocessor: FormulaPreprocessor,
        info: AdapterInfo,
        config: FormulaRecognitionConfig,
    ) -> Self {
        Self {
            inference,
            tokenizer,
            preprocessor,
            info,
            config,
            spec: PhantomData,
        }
    }

    /// Returns adapter metadata.
    pub fn info(&self) -> AdapterInfo {
        self.info.clone()
    }

    /// Runs preprocessing on a batch of images using the configured preprocessor.
    pub fn preprocess(&self, images: Vec<RgbImage>) -> Result<Tensor4D, OCRError> {
        self.preprocessor.preprocess_batch(&images)
    }

    /// Executes ONNX inference and returns the raw token ids.
    pub fn infer_tokens(&self, batch_tensor: &Tensor4D) -> Result<ndarray::Array2<i64>, OCRError> {
        // Debug: log input tensor stats
        tracing::info!("Input tensor shape: {:?}", batch_tensor.shape());
        let min_val = batch_tensor.iter().fold(f32::INFINITY, |a, &b| a.min(b));
        let max_val = batch_tensor
            .iter()
            .fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        tracing::info!("Input tensor min/max: {:.4} / {:.4}", min_val, max_val);

        self.inference.infer_2d_i64(batch_tensor)
    }

    /// Converts model outputs to LaTeX strings using the tokenizer and config.
    pub fn postprocess<D>(
        &self,
        token_ids: &ArrayBase<D, Ix2>,
        config: &FormulaRecognitionConfig,
    ) -> FormulaRecognitionOutput
    where
        D: Data<Elem = i64>,
    {
        let mut formulas = Vec::new();
        let mut scores = Vec::new();

        // Debug: print raw model output shape and first few tokens
        tracing::info!("Model output shape: {:?}", token_ids.shape());

        for batch_idx in 0..token_ids.shape()[0] {
            let row = token_ids.index_axis(Axis(0), batch_idx);

            // Debug: print first 20 raw token IDs
            let first_tokens: Vec<i64> = row.iter().copied().take(20).collect();
            tracing::info!(
                "First 20 raw tokens for batch {}: {:?}",
                batch_idx,
                first_tokens
            );

            // Guardrail: warn if any token id exceeds tokenizer vocab size
            let vocab_size = self.tokenizer.get_vocab_size(true) as i64;
            if let Some(&max_id) = row.iter().max()
                && max_id >= vocab_size
            {
                tracing::warn!(
                    "Token id(s) exceed tokenizer vocab (max_id={} >= vocab_size={}). \\nThis usually means model/tokenizer mismatch. If you're using PaddleOCR models, please supply the matching tokenizer from PaddleOCR via --tokenizer-path.",
                    max_id,
                    vocab_size
                );
            }

            let tokens: Vec<u32> = row
                .iter()
                .copied()
                .take_while(|&id| id != S::eos_token_id())
                .filter(|&id| S::should_keep_token(id))
                .map(|id| id as u32)
                .collect();

            tracing::debug!(
                "Filtered tokens for batch {}: {:?}",
                batch_idx,
                &tokens[..tokens.len().min(50)]
            );

            let latex = match self.tokenizer.decode(&tokens, true) {
                Ok(text) => {
                    tracing::debug!("Decoded LaTeX before normalization: {}", text);
                    S::normalize_latex(&text)
                }
                Err(err) => {
                    tracing::warn!("Failed to decode tokens for batch {}: {}", batch_idx, err);
                    String::new()
                }
            };

            if let Some(score) = None::<f32> {
                if score >= config.score_threshold {
                    formulas.push(latex);
                    scores.push(Some(score));
                } else {
                    formulas.push(String::new());
                    scores.push(Some(score));
                }
            } else {
                formulas.push(latex);
                scores.push(None);
            }
        }

        FormulaRecognitionOutput { formulas, scores }
    }
}

/// Shared builder for Paddle-style adapters.
#[derive(Debug)]
pub struct PaddleFormulaAdapterBuilderBase<S: PaddleFormulaSpec> {
    pub(crate) task_config: FormulaRecognitionConfig,
    pub(crate) target_size: Option<(u32, u32)>,
    pub(crate) tokenizer_path: Option<PathBuf>,
    pub(crate) session_pool_size: usize,
    pub(crate) model_name_override: Option<String>,
    spec: PhantomData<S>,
}

impl<S: PaddleFormulaSpec> PaddleFormulaAdapterBuilderBase<S> {
    /// Creates a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            task_config: FormulaRecognitionConfig::default(),
            target_size: None,
            tokenizer_path: None,
            session_pool_size: 1,
            model_name_override: None,
            spec: PhantomData,
        }
    }

    /// Overrides the preprocessing target size.
    pub fn target_size(mut self, width: u32, height: u32) -> Self {
        self.target_size = Some((width, height));
        self
    }

    /// Overrides the tokenizer file path.
    pub fn tokenizer_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.tokenizer_path = Some(path.into());
        self
    }

    /// Overrides the session pool size for ONNX runtime.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.session_pool_size = size;
        self
    }

    /// Overrides the display name reported via `AdapterInfo`.
    pub fn model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name_override = Some(name.into());
        self
    }

    /// Sets the score threshold filter.
    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.task_config.score_threshold = threshold;
        self
    }

    /// Sets the maximum sequence length.
    pub fn max_length(mut self, length: usize) -> Self {
        self.task_config.max_length = length;
        self
    }

    /// Replaces the entire task configuration.
    pub fn with_config(mut self, config: FormulaRecognitionConfig) -> Self {
        self.task_config = config;
        self
    }

    /// Consumes the builder and constructs the adapter base.
    pub fn build(self, model_path: &Path) -> Result<PaddleFormulaAdapterBase<S>, OCRError> {
        let inference = if self.session_pool_size > 1 {
            use crate::core::config::CommonBuilderConfig;
            let common_config = CommonBuilderConfig {
                session_pool_size: Some(self.session_pool_size),
                ..Default::default()
            };
            OrtInfer::from_common_with_auto_input(&common_config, model_path)?
        } else {
            OrtInfer::with_auto_input_name(model_path)?
        };

        let target_size = if let Some(size) = self.target_size {
            tracing::debug!("Using user-provided target size: {:?}", size);
            size
        } else {
            let detected = inference
                .primary_input_shape()
                .and_then(|shape| {
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
                .or_else(S::default_target_size);

            let detected = detected.unwrap_or((384, 384));
            tracing::info!("Using detected/default target size: {:?}", detected);
            detected
        };

        let tokenizer_path = if let Some(path) = self.tokenizer_path {
            path
        } else {
            let model_dir = model_path.parent().ok_or_else(|| OCRError::InvalidInput {
                message: "Cannot determine model directory".to_string(),
            })?;
            model_dir.join(S::default_tokenizer_filename())
        };

        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|err| OCRError::InvalidInput {
                message: format!(
                    "Failed to load tokenizer from {:?}: {}",
                    tokenizer_path, err
                ),
            })?;

        let params = FormulaPreprocessParams {
            target_size,
            crop_threshold: S::crop_threshold(),
            padding_multiple: S::padding_multiple(),
            normalize_mean: S::normalize_mean(),
            normalize_std: S::normalize_std(),
        };

        let preprocessor = FormulaPreprocessor::new(params);

        let info = AdapterInfo::new(
            self.model_name_override
                .unwrap_or_else(|| S::model_name().to_string()),
            S::adapter_version(),
            TaskType::FormulaRecognition,
            S::description(),
        );

        Ok(PaddleFormulaAdapterBase::new(
            inference,
            tokenizer,
            preprocessor,
            info,
            self.task_config,
        ))
    }
}

impl<S: PaddleFormulaSpec> Default for PaddleFormulaAdapterBuilderBase<S> {
    fn default() -> Self {
        Self::new()
    }
}

/// Adapter wrapper implementing the `ModelAdapter` trait.
#[derive(Debug)]
pub struct PaddleFormulaAdapter<S: PaddleFormulaSpec> {
    inner: PaddleFormulaAdapterBase<S>,
}

impl<S: PaddleFormulaSpec> PaddleFormulaAdapter<S> {
    pub fn new(inner: PaddleFormulaAdapterBase<S>) -> Self {
        Self { inner }
    }
}

impl<S: PaddleFormulaSpec> ModelAdapter for PaddleFormulaAdapter<S> {
    type Task = FormulaRecognitionTask;

    fn info(&self) -> AdapterInfo {
        self.inner.info()
    }

    fn execute(
        &self,
        input: <Self::Task as Task>::Input,
        config: Option<&<Self::Task as Task>::Config>,
    ) -> Result<<Self::Task as Task>::Output, OCRError> {
        let effective_config = config.unwrap_or(&self.inner.config);
        let batch_tensor = self.inner.preprocess(input.images)?;
        let token_ids = self.inner.infer_tokens(&batch_tensor)?;
        Ok(self.inner.postprocess(&token_ids, effective_config))
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn recommended_batch_size(&self) -> usize {
        8
    }
}
