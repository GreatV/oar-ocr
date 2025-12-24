//! Formula Recognition Predictor
//!
//! This module provides a high-level API for mathematical formula recognition in images.

use super::builder::PredictorBuilderState;
use crate::core::traits::adapter::AdapterBuilder;
use crate::core::traits::task::ImageTaskInput;
use crate::domain::adapters::{PPFormulaNetAdapterBuilder, UniMERNetFormulaAdapterBuilder};
use crate::domain::tasks::formula_recognition::{FormulaRecognitionConfig, FormulaRecognitionTask};
use crate::predictors::TaskPredictorCore;
use image::RgbImage;
use std::path::{Path, PathBuf};

/// Formula recognition model type
#[derive(Clone, Debug)]
pub enum FormulaModelKind {
    /// UniMERNet formula recognition model
    UniMERNet,
    /// PP-FormulaNet formula recognition model
    PPFormulaNet,
}

impl FormulaModelKind {
    /// Infer model kind from model name
    pub fn from_model_name(name: &str) -> Self {
        match name {
            "UniMERNet" => FormulaModelKind::UniMERNet,
            "PP-FormulaNet-S"
            | "PP-FormulaNet-L"
            | "PP-FormulaNet_plus-S"
            | "PP-FormulaNet_plus-M"
            | "PP-FormulaNet_plus-L" => FormulaModelKind::PPFormulaNet,
            _ => {
                // Fallback: try to infer from name pattern
                let name_lower = name.to_lowercase();
                if name_lower.contains("unimernet") {
                    FormulaModelKind::UniMERNet
                } else if name_lower.contains("pp-formulanet")
                    || name_lower.contains("ppformulanet")
                {
                    FormulaModelKind::PPFormulaNet
                } else {
                    // Default to UniMERNet
                    FormulaModelKind::UniMERNet
                }
            }
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
    pub fn predict(
        &self,
        images: Vec<RgbImage>,
    ) -> Result<FormulaRecognitionResult, Box<dyn std::error::Error>> {
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
            }),
            model_name: "FormulaRecognition".to_string(),
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
    pub fn build<P: AsRef<Path>>(
        self,
        model_path: P,
    ) -> Result<FormulaRecognitionPredictor, Box<dyn std::error::Error>> {
        let Self {
            state,
            model_name,
            tokenizer_path,
            target_size,
            model_kind,
        } = self;

        let (config, ort_config) = state.into_parts();

        let tokenizer_path =
            tokenizer_path.ok_or("Tokenizer path is required for formula recognition")?;

        // Determine model kind
        let model_kind =
            model_kind.unwrap_or_else(|| FormulaModelKind::from_model_name(&model_name));

        let adapter = match model_kind {
            FormulaModelKind::UniMERNet => {
                let mut builder = UniMERNetFormulaAdapterBuilder::new()
                    .with_config(config.clone())
                    .model_name(&model_name)
                    .tokenizer_path(tokenizer_path);

                if let Some((width, height)) = target_size {
                    builder = builder.target_size(width, height);
                }

                if let Some(ort_cfg) = ort_config.clone() {
                    builder = builder.with_ort_config(ort_cfg);
                }

                Box::new(builder.build(model_path.as_ref())?)
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

                Box::new(builder.build(model_path.as_ref())?)
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

impl_task_predictor_builder!(FormulaRecognitionPredictorBuilder, FormulaRecognitionConfig);

impl Default for FormulaRecognitionPredictorBuilder {
    fn default() -> Self {
        Self::new()
    }
}
