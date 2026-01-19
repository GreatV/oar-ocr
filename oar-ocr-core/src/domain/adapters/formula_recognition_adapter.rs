//! Formula recognition adapter using formula recognition models.

use crate::apply_ort_config;
use crate::core::OCRError;
use crate::core::traits::adapter::{AdapterInfo, ModelAdapter};
use crate::core::traits::task::Task;
use crate::domain::tasks::{
    FormulaRecognitionConfig, FormulaRecognitionOutput, FormulaRecognitionTask,
};
use crate::impl_adapter_builder;
use crate::models::recognition::{
    PPFormulaNetModel, PPFormulaNetModelBuilder, UniMERNetModel, UniMERNetModelBuilder,
    pp_formulanet::PPFormulaNetPostprocessConfig, unimernet::UniMERNetPostprocessConfig,
};
use crate::processors::normalize_latex;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

/// Special token IDs extracted from a tokenizer.
#[derive(Debug, Clone)]
pub struct SpecialTokenIds {
    /// Start-of-sequence (BOS) token ID
    pub sos_token_id: i64,
    /// End-of-sequence (EOS) token ID
    pub eos_token_id: i64,
}

impl Default for SpecialTokenIds {
    fn default() -> Self {
        Self {
            sos_token_id: 0,
            eos_token_id: 2,
        }
    }
}

/// Extracts special token IDs from a tokenizer.
///
/// This function attempts to find the BOS and EOS token IDs by checking common
/// token representations used by different tokenizer implementations.
///
/// # Token search order
/// - BOS tokens: `<s>`, `[BOS]`, `<bos>`, `[CLS]`
/// - EOS tokens: `</s>`, `[EOS]`, `<eos>`, `[SEP]`
///
/// If tokens are not found, falls back to default values (BOS=0, EOS=2).
pub fn extract_special_token_ids(tokenizer: &Tokenizer) -> SpecialTokenIds {
    // Common BOS token representations
    let bos_candidates = ["<s>", "[BOS]", "<bos>", "[CLS]"];
    // Common EOS token representations
    let eos_candidates = ["</s>", "[EOS]", "<eos>", "[SEP]"];

    let sos_token_id = bos_candidates
        .iter()
        .find_map(|&token| tokenizer.token_to_id(token))
        .map(|id| id as i64)
        .unwrap_or_else(|| {
            tracing::debug!("BOS token not found in tokenizer vocabulary, using default value 0");
            0
        });

    let eos_token_id = eos_candidates
        .iter()
        .find_map(|&token| tokenizer.token_to_id(token))
        .map(|id| id as i64)
        .unwrap_or_else(|| {
            tracing::debug!("EOS token not found in tokenizer vocabulary, using default value 2");
            2
        });

    tracing::debug!(
        "Extracted special token IDs: sos_token_id={}, eos_token_id={}",
        sos_token_id,
        eos_token_id
    );

    SpecialTokenIds {
        sos_token_id,
        eos_token_id,
    }
}

/// Formula model enum to support different model types.
#[derive(Debug)]
pub(crate) enum FormulaModel {
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

/// Formula recognition adapter.
#[derive(Debug)]
pub struct FormulaRecognitionAdapter {
    model: FormulaModel,
    tokenizer: Tokenizer,
    model_config: FormulaModelConfig,
    info: AdapterInfo,
    config: FormulaRecognitionConfig,
}

impl FormulaRecognitionAdapter {
    /// Creates a new formula recognition adapter.
    pub(crate) fn new(
        model: FormulaModel,
        tokenizer: Tokenizer,
        model_config: FormulaModelConfig,
        info: AdapterInfo,
        config: FormulaRecognitionConfig,
    ) -> Self {
        Self {
            model,
            tokenizer,
            model_config,
            info,
            config,
        }
    }
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

/// Formula model configuration.
#[derive(Debug, Clone)]
pub struct FormulaModelConfig {
    pub model_name: String,
    pub description: String,
    pub sos_token_id: i64,
    pub eos_token_id: i64,
}

impl FormulaModelConfig {
    /// PP-FormulaNet configuration with dynamically extracted token IDs.
    pub fn pp_formulanet_with_tokens(token_ids: SpecialTokenIds) -> Self {
        Self {
            model_name: "PP-FormulaNet".to_string(),
            description: "PP-FormulaNet formula recognition model".to_string(),
            sos_token_id: token_ids.sos_token_id,
            eos_token_id: token_ids.eos_token_id,
        }
    }

    /// UniMERNet configuration with dynamically extracted token IDs.
    pub fn unimernet_with_tokens(token_ids: SpecialTokenIds) -> Self {
        Self {
            model_name: "UniMERNet".to_string(),
            description: "UniMERNet formula recognition model".to_string(),
            sos_token_id: token_ids.sos_token_id,
            eos_token_id: token_ids.eos_token_id,
        }
    }
}

impl_adapter_builder! {
    builder_name: PPFormulaNetAdapterBuilder,
    adapter_name: FormulaRecognitionAdapter,
    config_type: FormulaRecognitionConfig,
    adapter_type: "formula_recognition_pp_formulanet",
    adapter_desc: "Recognizes mathematical formulas from images and converts to LaTeX",
    task_type: FormulaRecognition,

    fields: {
        tokenizer_path: Option<PathBuf> = None,
        target_size: Option<(u32, u32)> = None,
        model_name_override: Option<String> = None,
    },

    methods: {
        /// Sets the target size for image preprocessing.
        pub fn target_size(mut self, width: u32, height: u32) -> Self {
            self.target_size = Some((width, height));
            self
        }

        /// Sets the tokenizer path.
        pub fn tokenizer_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
            self.tokenizer_path = Some(path.into());
            self
        }

        /// Sets a custom model name for registry registration.
        pub fn model_name(mut self, name: impl Into<String>) -> Self {
            self.model_name_override = Some(name.into());
            self
        }

        /// Sets the maximum sequence length.
        pub fn max_length(mut self, length: usize) -> Self {
            self.config.task_config.max_length = length;
            self
        }

        /// Sets the task configuration (alias for with_config).
        pub fn task_config(mut self, config: FormulaRecognitionConfig) -> Self {
            self.config = self.config.with_task_config(config);
            self
        }

        /// Sets the score threshold for filtering low-confidence results.
        pub fn score_threshold(mut self, threshold: f32) -> Self {
            self.config.task_config.score_threshold = threshold;
            self
        }
    }

    build: |builder: PPFormulaNetAdapterBuilder, model_path: &Path| -> Result<FormulaRecognitionAdapter, OCRError> {
        let (task_config, ort_config) = builder.config
            .into_validated_parts()
            .map_err(|err| OCRError::ConfigError {
                message: err.to_string(),
            })?;

        // Build PP-FormulaNet model
        let mut model_builder = PPFormulaNetModelBuilder::new();
        if let Some((width, height)) = builder.target_size {
            model_builder = model_builder.target_size(width, height);
        }
        let model = FormulaModel::PPFormulaNet(
            apply_ort_config!(model_builder, ort_config).build(model_path)?
        );

        // Tokenizer path is required
        let tokenizer_path = builder.tokenizer_path.ok_or_else(|| OCRError::InvalidInput {
            message: "Tokenizer path is required. Use .tokenizer_path() to specify the path.".to_string(),
        })?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|err| OCRError::InvalidInput {
            message: format!("Failed to load tokenizer from {:?}: {}", tokenizer_path, err),
        })?;

        // Extract special token IDs dynamically from tokenizer
        let special_tokens = extract_special_token_ids(&tokenizer);
        let model_config = FormulaModelConfig::pp_formulanet_with_tokens(special_tokens);

        // Create adapter info using the helper
        let mut info = PPFormulaNetAdapterBuilder::base_adapter_info();
        if let Some(model_name) = builder.model_name_override {
            info.model_name = model_name;
        }

        Ok(FormulaRecognitionAdapter::new(
            model,
            tokenizer,
            model_config,
            info,
            task_config,
        ))
    },
}

impl_adapter_builder! {
    builder_name: UniMERNetAdapterBuilder,
    adapter_name: FormulaRecognitionAdapter,
    config_type: FormulaRecognitionConfig,
    adapter_type: "formula_recognition_unimernet",
    adapter_desc: "Recognizes mathematical formulas from images and converts to LaTeX",
    task_type: FormulaRecognition,

    fields: {
        tokenizer_path: Option<PathBuf> = None,
        target_size: Option<(u32, u32)> = None,
        model_name_override: Option<String> = None,
    },

    methods: {
        /// Sets the target size for image preprocessing.
        pub fn target_size(mut self, width: u32, height: u32) -> Self {
            self.target_size = Some((width, height));
            self
        }

        /// Sets the tokenizer path.
        pub fn tokenizer_path<P: Into<PathBuf>>(mut self, path: P) -> Self {
            self.tokenizer_path = Some(path.into());
            self
        }

        /// Sets a custom model name for registry registration.
        pub fn model_name(mut self, name: impl Into<String>) -> Self {
            self.model_name_override = Some(name.into());
            self
        }

        /// Sets the maximum sequence length.
        pub fn max_length(mut self, length: usize) -> Self {
            self.config.task_config.max_length = length;
            self
        }

        /// Sets the task configuration (alias for with_config).
        pub fn task_config(mut self, config: FormulaRecognitionConfig) -> Self {
            self.config = self.config.with_task_config(config);
            self
        }

        /// Sets the score threshold for filtering low-confidence results.
        pub fn score_threshold(mut self, threshold: f32) -> Self {
            self.config.task_config.score_threshold = threshold;
            self
        }
    }

    build: |builder: UniMERNetAdapterBuilder, model_path: &Path| -> Result<FormulaRecognitionAdapter, OCRError> {
        let (task_config, ort_config) = builder.config
            .into_validated_parts()
            .map_err(|err| OCRError::ConfigError {
                message: err.to_string(),
            })?;

        // Build UniMERNet model
        let mut model_builder = UniMERNetModelBuilder::new();
        if let Some((width, height)) = builder.target_size {
            model_builder = model_builder.target_size(width, height);
        }
        let model = FormulaModel::UniMERNet(
            apply_ort_config!(model_builder, ort_config).build(model_path)?
        );

        // Tokenizer path is required
        let tokenizer_path = builder.tokenizer_path.ok_or_else(|| OCRError::InvalidInput {
            message: "Tokenizer path is required. Use .tokenizer_path() to specify the path.".to_string(),
        })?;

        let tokenizer = Tokenizer::from_file(&tokenizer_path).map_err(|err| OCRError::InvalidInput {
            message: format!("Failed to load tokenizer from {:?}: {}", tokenizer_path, err),
        })?;

        // Extract special token IDs dynamically from tokenizer
        let special_tokens = extract_special_token_ids(&tokenizer);
        let model_config = FormulaModelConfig::unimernet_with_tokens(special_tokens);

        // Create adapter info using the helper
        let mut info = UniMERNetAdapterBuilder::base_adapter_info();
        if let Some(model_name) = builder.model_name_override {
            info.model_name = model_name;
        }

        Ok(FormulaRecognitionAdapter::new(
            model,
            tokenizer,
            model_config,
            info,
            task_config,
        ))
    },
}

/// Type alias for PP-FormulaNet adapter.
pub type PPFormulaNetAdapter = FormulaRecognitionAdapter;

/// Type alias for UniMERNet adapter.
pub type UniMERNetFormulaAdapter = FormulaRecognitionAdapter;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::traits::adapter::AdapterBuilder;

    #[test]
    fn test_pp_formulanet_builder_creation() {
        let builder = PPFormulaNetAdapterBuilder::new();
        assert_eq!(builder.adapter_type(), "formula_recognition_pp_formulanet");
    }

    #[test]
    fn test_pp_formulanet_builder_with_config() {
        let config = FormulaRecognitionConfig {
            score_threshold: 0.8,
            max_length: 512,
        };

        let builder = PPFormulaNetAdapterBuilder::new().with_config(config);
        assert_eq!(builder.config.task_config().score_threshold, 0.8);
        assert_eq!(builder.config.task_config().max_length, 512);
    }

    #[test]
    fn test_pp_formulanet_builder_fluent_api() {
        let builder = PPFormulaNetAdapterBuilder::new()
            .score_threshold(0.9)
            .max_length(1024)
            .target_size(640, 640);

        assert_eq!(builder.config.task_config().score_threshold, 0.9);
        assert_eq!(builder.config.task_config().max_length, 1024);
        assert_eq!(builder.target_size, Some((640, 640)));
    }

    #[test]
    fn test_pp_formulanet_default_builder() {
        let builder = PPFormulaNetAdapterBuilder::default();
        assert_eq!(builder.adapter_type(), "formula_recognition_pp_formulanet");
        // Default config values
        assert_eq!(builder.config.task_config().score_threshold, 0.0);
        assert_eq!(builder.config.task_config().max_length, 1536);
    }

    #[test]
    fn test_unimernet_builder_creation() {
        let builder = UniMERNetAdapterBuilder::new();
        assert_eq!(builder.adapter_type(), "formula_recognition_unimernet");
    }

    #[test]
    fn test_unimernet_builder_with_config() {
        let config = FormulaRecognitionConfig {
            score_threshold: 0.7,
            max_length: 2048,
        };

        let builder = UniMERNetAdapterBuilder::new().with_config(config);
        assert_eq!(builder.config.task_config().score_threshold, 0.7);
        assert_eq!(builder.config.task_config().max_length, 2048);
    }

    #[test]
    fn test_unimernet_builder_fluent_api() {
        let builder = UniMERNetAdapterBuilder::new()
            .score_threshold(0.85)
            .max_length(768)
            .target_size(512, 512);

        assert_eq!(builder.config.task_config().score_threshold, 0.85);
        assert_eq!(builder.config.task_config().max_length, 768);
        assert_eq!(builder.target_size, Some((512, 512)));
    }

    #[test]
    fn test_unimernet_default_builder() {
        let builder = UniMERNetAdapterBuilder::default();
        assert_eq!(builder.adapter_type(), "formula_recognition_unimernet");
        // Default config values
        assert_eq!(builder.config.task_config().score_threshold, 0.0);
        assert_eq!(builder.config.task_config().max_length, 1536);
    }

    #[test]
    fn test_formula_model_config_pp_formulanet() {
        let token_ids = SpecialTokenIds {
            sos_token_id: 1,
            eos_token_id: 2,
        };
        let config = FormulaModelConfig::pp_formulanet_with_tokens(token_ids);
        assert_eq!(config.model_name, "PP-FormulaNet");
        assert_eq!(config.sos_token_id, 1);
        assert_eq!(config.eos_token_id, 2);
    }

    #[test]
    fn test_formula_model_config_unimernet() {
        let token_ids = SpecialTokenIds {
            sos_token_id: 0,
            eos_token_id: 3,
        };
        let config = FormulaModelConfig::unimernet_with_tokens(token_ids);
        assert_eq!(config.model_name, "UniMERNet");
        assert_eq!(config.sos_token_id, 0);
        assert_eq!(config.eos_token_id, 3);
    }

    #[test]
    fn test_special_token_ids_default() {
        let default_ids = SpecialTokenIds::default();
        assert_eq!(default_ids.sos_token_id, 0);
        assert_eq!(default_ids.eos_token_id, 2);
    }

    #[test]
    fn test_unimernet_formula_adapter_builder_alias() {
        // Test that the type alias works for backward compatibility
        let builder = UniMERNetAdapterBuilder::new();
        assert_eq!(builder.adapter_type(), "formula_recognition_unimernet");
    }
}
