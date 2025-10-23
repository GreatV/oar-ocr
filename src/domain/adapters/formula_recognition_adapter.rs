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
    pub default_tokenizer_filename: String,
    pub sos_token_id: i64,
    pub eos_token_id: i64,
}

impl FormulaModelConfig {
    /// PP-FormulaNet configuration.
    pub fn pp_formulanet() -> Self {
        Self {
            model_name: "PP-FormulaNet".to_string(),
            description: "PaddleOCR PP-FormulaNet formula recognition model".to_string(),
            default_tokenizer_filename: "tokenizer.json".to_string(),
            sos_token_id: 0,
            eos_token_id: 2,
        }
    }

    /// UniMERNet configuration.
    pub fn unimernet() -> Self {
        Self {
            model_name: "UniMERNet".to_string(),
            description: "PaddleOCR UniMERNet formula recognition model".to_string(),
            default_tokenizer_filename: "tokenizer.json".to_string(),
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

        // Preprocess and infer
        let batch_tensor = self.model.preprocess(input.images)?;
        let token_ids = self.model.infer(&batch_tensor)?;

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
            // Warn if any token id exceeds tokenizer vocab size
            let vocab_size = self.tokenizer.get_vocab_size(true) as u32;
            if let Some(&max_id) = tokens.iter().max() {
                if max_id >= vocab_size {
                    tracing::warn!(
                        "Token id(s) exceed tokenizer vocab (max_id={} >= vocab_size={}). \
                         This usually means model/tokenizer mismatch. If you're using PaddleOCR models, \
                         please supply the matching tokenizer from PaddleOCR via --tokenizer-path.",
                        max_id,
                        vocab_size
                    );
                }
            }

            let latex = match self.tokenizer.decode(tokens, true) {
                Ok(text) => {
                    tracing::debug!("Decoded LaTeX before normalization: {}", text);
                    normalize_latex(&text)
                }
                Err(err) => {
                    tracing::warn!("Failed to decode tokens for batch {}: {}", batch_idx, err);
                    String::new()
                }
            };

            // For now, we don't have confidence scores from the model
            // In the future, we could compute them from the token probabilities
            if latex.is_empty() || effective_config.score_threshold > 0.0 {
                formulas.push(latex);
                scores.push(None);
            } else {
                formulas.push(latex);
                scores.push(None);
            }
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
    model_config: Option<FormulaModelConfig>,
    model_type: FormulaModelType,
    task_config: FormulaRecognitionConfig,
    tokenizer_path: Option<PathBuf>,
    session_pool_size: usize,
    model_name_override: Option<String>,
    target_size: Option<(u32, u32)>,
}

impl FormulaRecognitionAdapterBuilder {
    /// Creates a new builder with the specified model configuration and type.
    fn new_with_config(model_config: FormulaModelConfig, model_type: FormulaModelType) -> Self {
        Self {
            model_config: Some(model_config),
            model_type,
            task_config: FormulaRecognitionConfig::default(),
            tokenizer_path: None,
            session_pool_size: 1,
            model_name_override: None,
            target_size: None,
        }
    }

    /// Sets the task configuration.
    pub fn task_config(mut self, config: FormulaRecognitionConfig) -> Self {
        self.task_config = config;
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
        self.session_pool_size = size;
        self
    }

    /// Sets the model name override.
    pub fn model_name(mut self, name: impl Into<String>) -> Self {
        self.model_name_override = Some(name.into());
        self
    }

    /// Sets the score threshold.
    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.task_config.score_threshold = threshold;
        self
    }

    /// Sets the maximum sequence length.
    pub fn max_length(mut self, length: usize) -> Self {
        self.task_config.max_length = length;
        self
    }
}

impl AdapterBuilder for FormulaRecognitionAdapterBuilder {
    type Config = FormulaRecognitionConfig;
    type Adapter = FormulaRecognitionAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        let model_config = self.model_config.ok_or_else(|| OCRError::InvalidInput {
            message: "Model configuration not set".to_string(),
        })?;

        // Build the model based on type
        let model = match self.model_type {
            FormulaModelType::PPFormulaNet => {
                let mut builder =
                    PPFormulaNetModelBuilder::new().session_pool_size(self.session_pool_size);
                if let Some((width, height)) = self.target_size {
                    builder = builder.target_size(width, height);
                }
                FormulaModel::PPFormulaNet(builder.build(model_path)?)
            }
            FormulaModelType::UniMERNet => {
                let mut builder =
                    UniMERNetModelBuilder::new().session_pool_size(self.session_pool_size);
                if let Some((width, height)) = self.target_size {
                    builder = builder.target_size(width, height);
                }
                FormulaModel::UniMERNet(builder.build(model_path)?)
            }
        };

        // Load tokenizer
        let tokenizer_path = if let Some(path) = self.tokenizer_path {
            path
        } else {
            let model_dir = model_path.parent().ok_or_else(|| OCRError::InvalidInput {
                message: "Cannot determine model directory".to_string(),
            })?;
            model_dir.join(&model_config.default_tokenizer_filename)
        };

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
            config: self.task_config,
        })
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.task_config = config;
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
