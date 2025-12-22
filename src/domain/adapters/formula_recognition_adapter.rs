//! Formula recognition adapter using formula recognition models.

use crate::core::OCRError;
use crate::core::traits::adapter::{AdapterBuilder, AdapterInfo, ModelAdapter};
use crate::core::traits::task::{Task, TaskType};
use crate::domain::tasks::{
    FormulaRecognitionConfig, FormulaRecognitionOutput, FormulaRecognitionTask,
};
use crate::models::recognition::{
    PPFormulaNetModel, PPFormulaNetModelBuilder, PPFormulaNetPostprocessConfig, UniMERNetModel,
    UniMERNetModelBuilder, UniMERNetPostprocessConfig,
};
use crate::processors::normalize_latex;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// Formula model enum to support different model types.
#[derive(Debug)]
enum FormulaModel {
    PPFormulaNet(PPFormulaNetModel),
    UniMERNet(UniMERNetModel),
}

impl FormulaModel {
    fn preprocess(&self, images: Vec<image::RgbImage>) -> Result<crate::core::Tensor4D, OCRError> {
        match self {
            FormulaModel::PPFormulaNet(model) => model.preprocess(images),
            FormulaModel::UniMERNet(model) => model.preprocess(images),
        }
    }

    fn infer(
        &self,
        batch_tensor: &crate::core::Tensor4D,
    ) -> Result<ndarray::Array2<i64>, OCRError> {
        match self {
            FormulaModel::PPFormulaNet(model) => model.infer(batch_tensor),
            FormulaModel::UniMERNet(model) => model.infer(batch_tensor),
        }
    }

    fn filter_tokens(
        &self,
        token_ids: &ndarray::Array2<i64>,
        sos_token_id: i64,
        eos_token_id: i64,
    ) -> Vec<Vec<u32>> {
        match self {
            FormulaModel::PPFormulaNet(_) => {
                let config = PPFormulaNetPostprocessConfig {
                    sos_token_id,
                    eos_token_id,
                };
                PPFormulaNetModel::filter_tokens(token_ids, &config)
            }
            FormulaModel::UniMERNet(_) => {
                let config = UniMERNetPostprocessConfig {
                    sos_token_id,
                    eos_token_id,
                };
                UniMERNetModel::filter_tokens(token_ids, &config)
            }
        }
    }
}

/// Formula model configuration.
#[derive(Debug, Clone)]
pub struct FormulaModelConfig {
    pub model_name: String,
    pub description: String,
    pub sos_token_id: i64,
    pub eos_token_id: i64,
}

impl FormulaModelConfig {
    /// PP-FormulaNet configuration.
    pub fn pp_formulanet() -> Self {
        Self {
            model_name: "PP-FormulaNet".to_string(),
            description: "PP-FormulaNet formula recognition model".to_string(),
            sos_token_id: 0,
            eos_token_id: 2,
        }
    }

    /// UniMERNet configuration.
    pub fn unimernet() -> Self {
        Self {
            model_name: "UniMERNet".to_string(),
            description: "UniMERNet formula recognition model".to_string(),
            sos_token_id: 0,
            eos_token_id: 2,
        }
    }
}

/// Formula recognition adapter.
#[derive(Debug)]
pub struct FormulaRecognitionAdapter {
    model: FormulaModel,
    tokenizer: Tokenizer,
    model_config: FormulaModelConfig,
    info: AdapterInfo,
    config: FormulaRecognitionConfig,
}

impl ModelAdapter for FormulaRecognitionAdapter {
    type Task = FormulaRecognitionTask;

    fn info(&self) -> AdapterInfo {
        self.info.clone()
    }

    fn execute(
        &self,
        input: <Self::Task as Task>::Input,
        config: Option<&<Self::Task as Task>::Config>,
    ) -> Result<<Self::Task as Task>::Output, OCRError> {
        let effective_config = config.unwrap_or(&self.config);
        let batch_len = input.images.len();

        // Preprocess and infer
        let batch_tensor = self.model.preprocess(input.images).map_err(|e| {
            OCRError::adapter_execution_error(
                "FormulaRecognitionAdapter",
                format!("preprocess (batch_size={})", batch_len),
                e,
            )
        })?;
        let token_ids = self.model.infer(&batch_tensor).map_err(|e| {
            OCRError::adapter_execution_error(
                "FormulaRecognitionAdapter",
                format!("infer (batch_size={})", batch_len),
                e,
            )
        })?;

        // Filter tokens and decode
        let filtered_tokens = self.model.filter_tokens(
            &token_ids,
            self.model_config.sos_token_id,
            self.model_config.eos_token_id,
        );

        let mut formulas = Vec::new();
        let mut scores = Vec::new();

        // Decode tokens to LaTeX
        for (batch_idx, tokens) in filtered_tokens.iter().enumerate() {
            // Apply max_length constraint by truncating token sequences
            let tokens_to_decode = if tokens.len() > effective_config.max_length {
                tracing::debug!(
                    "Truncating formula tokens from {} to {} (max_length)",
                    tokens.len(),
                    effective_config.max_length
                );
                &tokens[..effective_config.max_length]
            } else {
                tokens.as_slice()
            };

            // Warn if any token id exceeds tokenizer vocab size
            let vocab_size = self.tokenizer.get_vocab_size(true) as u32;
            if let Some(&max_id) = tokens_to_decode.iter().max()
                && max_id >= vocab_size
            {
                tracing::warn!(
                    "Token id(s) exceed tokenizer vocab (max_id={} >= vocab_size={}). \
                     This usually means model/tokenizer mismatch. If you're using external models, \
                     please supply the matching tokenizer via --tokenizer-path.",
                    max_id,
                    vocab_size
                );
            }

            let latex = match self.tokenizer.decode(tokens_to_decode, true) {
                Ok(text) => {
                    tracing::debug!("Decoded LaTeX before normalization: {}", text);
                    normalize_latex(&text)
                }
                Err(err) => {
                    tracing::warn!("Failed to decode tokens for batch {}: {}", batch_idx, err);
                    String::new()
                }
            };

            // Note: Confidence score computation is not currently implemented.
            // The current model interface only returns token IDs via infer_2d_i64(),
            // not the underlying logits or probabilities from which confidence could be computed.
            //
            // To implement score_threshold filtering, we would need to:
            // 1. Modify the model inference to also return logits/probabilities
            // 2. Compute confidence scores (e.g., mean/min token probability, or sequence probability)
            // 3. Filter formulas based on: score >= effective_config.score_threshold
            // 4. Only push formulas that pass the threshold
            //
            // Example implementation once probabilities are available:
            // ```
            // let confidence = compute_sequence_confidence(&token_probs);
            // if confidence >= effective_config.score_threshold {
            //     formulas.push(latex);
            //     scores.push(Some(confidence));
            // } else {
            //     tracing::debug!("Filtered formula with confidence {} < threshold {}",
            //                    confidence, effective_config.score_threshold);
            // }
            // ```
            //
            // For now, we accept all formulas without filtering:
            formulas.push(latex);
            scores.push(None);
        }

        Ok(FormulaRecognitionOutput { formulas, scores })
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn recommended_batch_size(&self) -> usize {
        8
    }
}

/// Model type for formula recognition.
#[derive(Debug, Clone, Copy)]
enum FormulaModelType {
    PPFormulaNet,
    UniMERNet,
}

/// Builder for formula recognition adapter.
#[derive(Debug)]
pub struct FormulaRecognitionAdapterBuilder {
    config: super::builder_config::AdapterBuilderConfig<FormulaRecognitionConfig>,
    model_config: Option<FormulaModelConfig>,
    model_type: FormulaModelType,
    tokenizer_path: Option<PathBuf>,
    model_name_override: Option<String>,
    target_size: Option<(u32, u32)>,
}

impl FormulaRecognitionAdapterBuilder {
    /// Creates a new builder with the specified model configuration and type.
    fn new_with_config(model_config: FormulaModelConfig, model_type: FormulaModelType) -> Self {
        Self {
            config: super::builder_config::AdapterBuilderConfig::default(),
            model_config: Some(model_config),
            model_type,
            tokenizer_path: None,
            model_name_override: None,
            target_size: None,
        }
    }

    /// Sets the task configuration.
    pub fn task_config(mut self, config: FormulaRecognitionConfig) -> Self {
        self.config = self.config.with_task_config(config);
        self
    }

    /// Sets the target size.
    pub fn target_size(mut self, width: u32, height: u32) -> Self {
        self.target_size = Some((width, height));
        self
    }

    /// Sets the tokenizer path.
    pub fn tokenizer_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.tokenizer_path = Some(path.into());
        self
    }

    /// Sets the session pool size.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.config = self.config.with_session_pool_size(size);
        self
    }

    /// Sets the model name override.
    pub fn model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name_override = Some(name.into());
        self
    }

    /// Sets the score threshold.
    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.config.task_config.score_threshold = threshold;
        self
    }

    /// Sets the maximum sequence length.
    pub fn max_length(mut self, length: usize) -> Self {
        self.config.task_config.max_length = length;
        self
    }

    /// Sets the ONNX Runtime session configuration.
    pub fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.config = self.config.with_ort_config(config);
        self
    }
}

impl AdapterBuilder for FormulaRecognitionAdapterBuilder {
    type Config = FormulaRecognitionConfig;
    type Adapter = FormulaRecognitionAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        let (task_config, session_pool_size, ort_config) = self
            .config
            .into_validated_parts()
            .map_err(|err| OCRError::ConfigError {
                message: err.to_string(),
            })?;

        let model_config = self.model_config.ok_or_else(|| OCRError::InvalidInput {
            message: "Model configuration not set".to_string(),
        })?;

        // Build the model based on type
        let model = match self.model_type {
            FormulaModelType::PPFormulaNet => {
                let mut builder =
                    PPFormulaNetModelBuilder::new().session_pool_size(session_pool_size);
                if let Some((width, height)) = self.target_size {
                    builder = builder.target_size(width, height);
                }
                if let Some(ort_config) = ort_config.clone() {
                    builder = builder.with_ort_config(ort_config);
                }
                FormulaModel::PPFormulaNet(builder.build(model_path)?)
            }
            FormulaModelType::UniMERNet => {
                let mut builder = UniMERNetModelBuilder::new().session_pool_size(session_pool_size);
                if let Some((width, height)) = self.target_size {
                    builder = builder.target_size(width, height);
                }
                if let Some(ort_config) = ort_config {
                    builder = builder.with_ort_config(ort_config);
                }
                FormulaModel::UniMERNet(builder.build(model_path)?)
            }
        };

        // Load tokenizer - tokenizer path is required
        let tokenizer_path = self.tokenizer_path.ok_or_else(|| OCRError::InvalidInput {
            message: "Tokenizer path is required. Please provide it via --tokenizer-path or tokenizer_path() builder method.".to_string(),
        })?;

        let tokenizer =
            Tokenizer::from_file(&tokenizer_path).map_err(|err| OCRError::InvalidInput {
                message: format!(
                    "Failed to load tokenizer from {:?}: {}",
                    tokenizer_path, err
                ),
            })?;

        // Create adapter info
        let info = AdapterInfo::new(
            self.model_name_override
                .unwrap_or_else(|| model_config.model_name.clone()),
            "1.0.0",
            TaskType::FormulaRecognition,
            &model_config.description,
        );

        Ok(FormulaRecognitionAdapter {
            model,
            tokenizer,
            model_config,
            info,
            config: task_config,
        })
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.config = self.config.with_task_config(config);
        self
    }

    fn adapter_type(&self) -> &str {
        "FormulaRecognition"
    }
}

/// Type alias for PP-FormulaNet adapter.
pub type PPFormulaNetAdapter = FormulaRecognitionAdapter;

/// Builder for PP-FormulaNet adapter.
#[derive(Debug)]
pub struct PPFormulaNetAdapterBuilder {
    inner: FormulaRecognitionAdapterBuilder,
}

impl PPFormulaNetAdapterBuilder {
    /// Creates a new PP-FormulaNet adapter builder.
    pub fn new() -> Self {
        Self {
            inner: FormulaRecognitionAdapterBuilder::new_with_config(
                FormulaModelConfig::pp_formulanet(),
                FormulaModelType::PPFormulaNet,
            ),
        }
    }

    /// Sets the target size.
    pub fn target_size(mut self, width: u32, height: u32) -> Self {
        self.inner = self.inner.target_size(width, height);
        self
    }

    /// Sets the tokenizer path.
    pub fn tokenizer_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.inner = self.inner.tokenizer_path(path);
        self
    }

    /// Sets the session pool size.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.inner = self.inner.session_pool_size(size);
        self
    }

    /// Sets the model name override.
    pub fn model_name(mut self, name: impl Into<String>) -> Self {
        self.inner = self.inner.model_name(name);
        self
    }

    /// Sets the score threshold.
    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.inner = self.inner.score_threshold(threshold);
        self
    }

    /// Sets the maximum sequence length.
    pub fn max_length(mut self, length: usize) -> Self {
        self.inner = self.inner.max_length(length);
        self
    }

    /// Sets the task configuration.
    pub fn task_config(mut self, config: FormulaRecognitionConfig) -> Self {
        self.inner = self.inner.task_config(config);
        self
    }

    /// Sets the ONNX Runtime session configuration.
    pub fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.inner = self.inner.with_ort_config(config);
        self
    }
}

impl Default for PPFormulaNetAdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AdapterBuilder for PPFormulaNetAdapterBuilder {
    type Config = FormulaRecognitionConfig;
    type Adapter = PPFormulaNetAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        self.inner.build(model_path)
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.inner = self.inner.with_config(config);
        self
    }

    fn adapter_type(&self) -> &str {
        "PPFormulaNet"
    }
}

/// Type alias for UniMERNet adapter.
pub type UniMERNetFormulaAdapter = FormulaRecognitionAdapter;

/// Builder for UniMERNet adapter.
#[derive(Debug)]
pub struct UniMERNetFormulaAdapterBuilder {
    inner: FormulaRecognitionAdapterBuilder,
}

impl UniMERNetFormulaAdapterBuilder {
    /// Creates a new UniMERNet adapter builder.
    pub fn new() -> Self {
        Self {
            inner: FormulaRecognitionAdapterBuilder::new_with_config(
                FormulaModelConfig::unimernet(),
                FormulaModelType::UniMERNet,
            ),
        }
    }

    /// Sets the target size.
    pub fn target_size(mut self, width: u32, height: u32) -> Self {
        self.inner = self.inner.target_size(width, height);
        self
    }

    /// Sets the tokenizer path.
    pub fn tokenizer_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
        self.inner = self.inner.tokenizer_path(path);
        self
    }

    /// Sets the session pool size.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.inner = self.inner.session_pool_size(size);
        self
    }

    /// Sets the model name override.
    pub fn model_name(mut self, name: impl Into<String>) -> Self {
        self.inner = self.inner.model_name(name);
        self
    }

    /// Sets the score threshold.
    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.inner = self.inner.score_threshold(threshold);
        self
    }

    /// Sets the maximum sequence length.
    pub fn max_length(mut self, length: usize) -> Self {
        self.inner = self.inner.max_length(length);
        self
    }

    /// Sets the task configuration.
    pub fn task_config(mut self, config: FormulaRecognitionConfig) -> Self {
        self.inner = self.inner.task_config(config);
        self
    }

    /// Sets the ONNX Runtime session configuration.
    pub fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.inner = self.inner.with_ort_config(config);
        self
    }
}

impl Default for UniMERNetFormulaAdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AdapterBuilder for UniMERNetFormulaAdapterBuilder {
    type Config = FormulaRecognitionConfig;
    type Adapter = UniMERNetFormulaAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        self.inner.build(model_path)
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.inner = self.inner.with_config(config);
        self
    }

    fn adapter_type(&self) -> &str {
        "UniMERNet"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pp_formulanet_builder_creation() {
        let builder = PPFormulaNetAdapterBuilder::new();
        assert_eq!(builder.adapter_type(), "PPFormulaNet");
    }

    #[test]
    fn test_pp_formulanet_builder_with_config() {
        let config = FormulaRecognitionConfig {
            score_threshold: 0.8,
            max_length: 512,
        };

        let builder = PPFormulaNetAdapterBuilder::new().with_config(config.clone());
        assert_eq!(builder.inner.config.task_config().score_threshold, 0.8);
        assert_eq!(builder.inner.config.task_config().max_length, 512);
    }

    #[test]
    fn test_pp_formulanet_builder_fluent_api() {
        let builder = PPFormulaNetAdapterBuilder::new()
            .score_threshold(0.9)
            .max_length(1024)
            .session_pool_size(4)
            .target_size(640, 640);

        assert_eq!(builder.inner.config.task_config().score_threshold, 0.9);
        assert_eq!(builder.inner.config.task_config().max_length, 1024);
        assert_eq!(builder.inner.config.session_pool_size(), 4);
        assert_eq!(builder.inner.target_size, Some((640, 640)));
    }

    #[test]
    fn test_pp_formulanet_default_builder() {
        let builder = PPFormulaNetAdapterBuilder::default();
        assert_eq!(builder.adapter_type(), "PPFormulaNet");
        // Default config values
        assert_eq!(builder.inner.config.task_config().score_threshold, 0.0);
        assert_eq!(builder.inner.config.task_config().max_length, 1536);
    }

    #[test]
    fn test_unimernet_builder_creation() {
        let builder = UniMERNetFormulaAdapterBuilder::new();
        assert_eq!(builder.adapter_type(), "UniMERNet");
    }

    #[test]
    fn test_unimernet_builder_with_config() {
        let config = FormulaRecognitionConfig {
            score_threshold: 0.7,
            max_length: 2048,
        };

        let builder = UniMERNetFormulaAdapterBuilder::new().with_config(config.clone());
        assert_eq!(builder.inner.config.task_config().score_threshold, 0.7);
        assert_eq!(builder.inner.config.task_config().max_length, 2048);
    }

    #[test]
    fn test_unimernet_builder_fluent_api() {
        let builder = UniMERNetFormulaAdapterBuilder::new()
            .score_threshold(0.85)
            .max_length(768)
            .session_pool_size(2)
            .target_size(512, 512);

        assert_eq!(builder.inner.config.task_config().score_threshold, 0.85);
        assert_eq!(builder.inner.config.task_config().max_length, 768);
        assert_eq!(builder.inner.config.session_pool_size(), 2);
        assert_eq!(builder.inner.target_size, Some((512, 512)));
    }

    #[test]
    fn test_unimernet_default_builder() {
        let builder = UniMERNetFormulaAdapterBuilder::default();
        assert_eq!(builder.adapter_type(), "UniMERNet");
        // Default config values
        assert_eq!(builder.inner.config.task_config().score_threshold, 0.0);
        assert_eq!(builder.inner.config.task_config().max_length, 1536);
    }

    #[test]
    fn test_formula_model_config_pp_formulanet() {
        let config = FormulaModelConfig::pp_formulanet();
        assert_eq!(config.model_name, "PP-FormulaNet");
        assert_eq!(config.sos_token_id, 0);
        assert_eq!(config.eos_token_id, 2);
    }

    #[test]
    fn test_formula_model_config_unimernet() {
        let config = FormulaModelConfig::unimernet();
        assert_eq!(config.model_name, "UniMERNet");
        assert_eq!(config.sos_token_id, 0);
        assert_eq!(config.eos_token_id, 2);
    }
}
