//! Formula Recognition Predictor
//!
//! This module provides a high-level API for mathematical formula recognition in images.

use super::builder::PredictorBuilderState;
use crate::TaskPredictorBuilder;
use crate::core::OcrResult;
use crate::core::errors::OCRError;
use crate::core::traits::OrtConfigurable;
use crate::core::traits::task::ImageTaskInput;
use crate::domain::adapters::{PPFormulaNetAdapterBuilder, UniMERNetAdapterBuilder};
use crate::domain::tasks::formula_recognition::{FormulaRecognitionConfig, FormulaRecognitionTask};
use crate::predictors::TaskPredictorCore;
use image::RgbImage;
use std::path::{Path, PathBuf};

/// Formula recognition model type
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FormulaModelKind {
    /// UniMERNet formula recognition model
    UniMERNet,
    /// PP-FormulaNet formula recognition model
    PPFormulaNet,
}

impl FormulaModelKind {
    /// Stable configuration spelling for this model family.
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::UniMERNet => "unimernet",
            Self::PPFormulaNet => "pp_formulanet",
        }
    }

    /// Infer model kind from a known model name or naming pattern.
    ///
    /// Returns `None` rather than silently selecting an incompatible decoder
    /// when the name does not identify either supported model family.
    pub fn from_model_name(name: &str) -> Option<Self> {
        match name {
            "UniMERNet" => Some(FormulaModelKind::UniMERNet),
            "PP-FormulaNet-S"
            | "PP-FormulaNet-L"
            | "PP-FormulaNet_plus-S"
            | "PP-FormulaNet_plus-M"
            | "PP-FormulaNet_plus-L" => Some(FormulaModelKind::PPFormulaNet),
            _ => {
                let name_lower = name.to_lowercase().replace('_', "-");
                if name_lower.contains("unimernet") {
                    Some(FormulaModelKind::UniMERNet)
                } else if name_lower.contains("pp-formulanet")
                    || name_lower.contains("ppformulanet")
                {
                    Some(FormulaModelKind::PPFormulaNet)
                } else {
                    None
                }
            }
        }
    }
}

impl AsRef<str> for FormulaModelKind {
    fn as_ref(&self) -> &str {
        self.as_str()
    }
}

impl std::str::FromStr for FormulaModelKind {
    type Err = String;

    fn from_str(value: &str) -> Result<Self, Self::Err> {
        match value.trim().to_ascii_lowercase().replace('-', "_").as_str() {
            "pp_formulanet" => Ok(Self::PPFormulaNet),
            "unimernet" => Ok(Self::UniMERNet),
            _ => Err(format!(
                "unknown formula model kind {value:?}; expected 'pp_formulanet' or 'unimernet'"
            )),
        }
    }
}

/// Formula recognition prediction result
#[derive(Debug, Clone)]
pub struct FormulaRecognitionResult {
    /// Recognized LaTeX formulas for each input image
    pub formulas: Vec<String>,
    /// Confidence scores for each formula (if available)
    pub scores: Vec<Option<f32>>,
}

/// Formula recognition predictor
pub struct FormulaRecognitionPredictor {
    core: TaskPredictorCore<FormulaRecognitionTask>,
}

impl FormulaRecognitionPredictor {
    /// Create a new builder for the formula recognition predictor
    pub fn builder() -> FormulaRecognitionPredictorBuilder {
        FormulaRecognitionPredictorBuilder::new()
    }

    /// Predict formulas in the given images.
    pub fn predict(&self, images: Vec<RgbImage>) -> OcrResult<FormulaRecognitionResult> {
        // Create task input
        let input = ImageTaskInput::new(images);

        // Execute prediction through core
        let output = self.core.predict(input)?;

        Ok(FormulaRecognitionResult {
            formulas: output.formulas,
            scores: output.scores,
        })
    }
}

/// Builder for formula recognition predictor
#[derive(TaskPredictorBuilder)]
#[builder(config = FormulaRecognitionConfig)]
pub struct FormulaRecognitionPredictorBuilder {
    state: PredictorBuilderState<FormulaRecognitionConfig>,
    model_name: String,
    tokenizer_path: Option<PathBuf>,
    target_size: Option<(u32, u32)>,
    model_kind: Option<FormulaModelKind>,
}

impl FormulaRecognitionPredictorBuilder {
    /// Create a new builder with default configuration
    pub fn new() -> Self {
        Self {
            state: PredictorBuilderState::new(FormulaRecognitionConfig {
                score_threshold: 0.0,
                max_length: 1536,
                batch_size: 8,
            }),
            model_name: "UniMERNet".to_string(),
            tokenizer_path: None,
            target_size: None,
            model_kind: None,
        }
    }

    /// Set the score threshold
    pub fn score_threshold(mut self, threshold: f32) -> Self {
        self.state.config_mut().score_threshold = threshold;
        self
    }

    /// Set the maximum formula length in tokens
    pub fn max_length(mut self, max: usize) -> Self {
        self.state.config_mut().max_length = max;
        self
    }

    /// Set the model name
    pub fn model_name(mut self, name: &str) -> Self {
        self.model_name = name.to_string();
        self
    }

    /// Set the tokenizer path (required)
    pub fn tokenizer_path<P: AsRef<Path>>(mut self, path: P) -> Self {
        self.tokenizer_path = Some(path.as_ref().to_path_buf());
        self
    }

    /// Set the target image size (width, height)
    pub fn target_size(mut self, width: u32, height: u32) -> Self {
        self.target_size = Some((width, height));
        self
    }

    /// Explicitly set the model kind
    pub fn model_kind(mut self, kind: FormulaModelKind) -> Self {
        self.model_kind = Some(kind);
        self
    }

    /// Build the formula recognition predictor
    pub fn build(
        self,
        model_source: impl Into<crate::core::ModelSource>,
    ) -> OcrResult<FormulaRecognitionPredictor> {
        let Self {
            state,
            model_name,
            tokenizer_path,
            target_size,
            model_kind,
        } = self;

        let (config, ort_config) = state.into_parts();

        let tokenizer_path = tokenizer_path.ok_or_else(|| {
            OCRError::missing_field("tokenizer_path", "FormulaRecognitionPredictor")
        })?;
        let tokenizer_path = super::resolve_asset_path(&tokenizer_path)?;

        // Determine model kind
        let model_kind = match model_kind {
            Some(kind) => kind,
            None => FormulaModelKind::from_model_name(&model_name).ok_or_else(|| {
                OCRError::config_error_detailed(
                    "formula_recognition",
                    format!(
                        "Cannot infer formula model kind from '{model_name}'; call model_kind() with FormulaModelKind::PPFormulaNet or FormulaModelKind::UniMERNet"
                    ),
                )
            })?,
        };

        let adapter = match model_kind {
            FormulaModelKind::UniMERNet => {
                let mut builder = UniMERNetAdapterBuilder::new()
                    .with_config(config.clone())
                    .model_name(&model_name)
                    .tokenizer_path(tokenizer_path);

                if let Some((width, height)) = target_size {
                    builder = builder.target_size(width, height);
                }

                if let Some(ort_cfg) = ort_config.clone() {
                    builder = builder.with_ort_config(ort_cfg);
                }

                super::build_adapter(builder, model_source)?
            }
            FormulaModelKind::PPFormulaNet => {
                let mut builder = PPFormulaNetAdapterBuilder::new()
                    .with_config(config.clone())
                    .model_name(&model_name)
                    .tokenizer_path(tokenizer_path);

                if let Some((width, height)) = target_size {
                    builder = builder.target_size(width, height);
                }

                if let Some(ort_cfg) = ort_config.clone() {
                    builder = builder.with_ort_config(ort_cfg);
                }

                super::build_adapter(builder, model_source)?
            }
        };

        Ok(FormulaRecognitionPredictor {
            core: TaskPredictorCore::new(
                adapter,
                FormulaRecognitionTask::new(config.clone()),
                config,
            ),
        })
    }
}

impl Default for FormulaRecognitionPredictorBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::{FormulaModelKind, FormulaRecognitionPredictorBuilder};

    #[test]
    fn formula_builder_defaults_to_unimernet() {
        let builder = FormulaRecognitionPredictorBuilder::new();
        assert_eq!(builder.model_name, "UniMERNet");
        assert_eq!(builder.model_kind, None);
    }

    #[test]
    fn formula_model_kind_inference_rejects_unknown_names() {
        assert_eq!(
            FormulaModelKind::from_model_name("UniMERNet"),
            Some(FormulaModelKind::UniMERNet)
        );
        assert_eq!(
            FormulaModelKind::from_model_name("PP-FormulaNet_plus-L"),
            Some(FormulaModelKind::PPFormulaNet)
        );
        assert_eq!(
            FormulaModelKind::from_model_name("pp_formulanet"),
            Some(FormulaModelKind::PPFormulaNet)
        );
        assert_eq!(
            FormulaModelKind::from_model_name("custom_unimernet_export"),
            Some(FormulaModelKind::UniMERNet)
        );
        assert_eq!(FormulaModelKind::from_model_name("formula.onnx"), None);
        assert_eq!("pp_formulanet".parse(), Ok(FormulaModelKind::PPFormulaNet));
        assert_eq!("PP-FormulaNet".parse(), Ok(FormulaModelKind::PPFormulaNet));
        assert!(
            "custom_unimernet_export"
                .parse::<FormulaModelKind>()
                .is_err()
        );
        assert!("mystery".parse::<FormulaModelKind>().is_err());
    }
}
