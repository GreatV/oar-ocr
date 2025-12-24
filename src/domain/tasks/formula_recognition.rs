//! Concrete task implementations for formula recognition.
//!
//! This module provides the formula recognition task that converts mathematical
//! formulas in images to LaTeX strings.

use super::validation::ensure_non_empty_images;
use crate::core::OCRError;
use crate::core::traits::task::{ImageTaskInput, Task, TaskSchema, TaskType};
use crate::impl_config_validator;
use crate::utils::{ScoreValidator, validate_length_match};
use serde::{Deserialize, Serialize};

/// Configuration for formula recognition task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FormulaRecognitionConfig {
    /// Score threshold for filtering low-confidence results (default: 0.0)
    pub score_threshold: f32,
    /// Maximum formula length in tokens (default: 1536)
    pub max_length: usize,
}

impl Default for FormulaRecognitionConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.0,
            max_length: 1536,
        }
    }
}

impl_config_validator!(FormulaRecognitionConfig {
    score_threshold: range(0.0, 1.0),
    max_length: min(1),
});

/// Output from formula recognition task.
#[derive(Debug, Clone)]
pub struct FormulaRecognitionOutput {
    /// Recognized LaTeX formulas for each image
    pub formulas: Vec<String>,
    /// Confidence scores for each formula (if available)
    pub scores: Vec<Option<f32>>,
}

impl FormulaRecognitionOutput {
    /// Creates an empty formula recognition output.
    pub fn empty() -> Self {
        Self {
            formulas: Vec::new(),
            scores: Vec::new(),
        }
    }

    /// Creates a formula recognition output with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            formulas: Vec::with_capacity(capacity),
            scores: Vec::with_capacity(capacity),
        }
    }
}

/// Formula recognition task implementation.
#[derive(Debug, Default)]
pub struct FormulaRecognitionTask {
    _config: FormulaRecognitionConfig,
}

impl FormulaRecognitionTask {
    /// Creates a new formula recognition task.
    pub fn new(config: FormulaRecognitionConfig) -> Self {
        Self { _config: config }
    }
}

impl Task for FormulaRecognitionTask {
    type Config = FormulaRecognitionConfig;
    type Input = ImageTaskInput;
    type Output = FormulaRecognitionOutput;

    fn task_type(&self) -> TaskType {
        TaskType::FormulaRecognition
    }

    fn schema(&self) -> TaskSchema {
        TaskSchema::new(
            TaskType::FormulaRecognition,
            vec!["image".to_string()],
            vec!["latex_formula".to_string(), "confidence".to_string()],
        )
    }

    fn validate_input(&self, input: &Self::Input) -> Result<(), OCRError> {
        ensure_non_empty_images(&input.images, "No images provided for formula recognition")?;

        Ok(())
    }

    fn validate_output(&self, output: &Self::Output) -> Result<(), OCRError> {
        // Validate that formulas and scores have matching lengths
        validate_length_match(
            output.formulas.len(),
            output.scores.len(),
            "formulas",
            "scores",
        )?;

        // Validate scores are in valid range when present
        let validator = ScoreValidator::new_unit_range("score");
        for (idx, score_opt) in output.scores.iter().enumerate() {
            if let Some(score) = score_opt {
                validator.validate_score(*score, &format!("Formula {}", idx))?;
            }
        }

        Ok(())
    }

    fn empty_output(&self) -> Self::Output {
        FormulaRecognitionOutput::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_formula_recognition_task_creation() {
        let task = FormulaRecognitionTask::default();
        assert_eq!(task.task_type(), TaskType::FormulaRecognition);
    }

    #[test]
    fn test_input_validation() {
        let task = FormulaRecognitionTask::default();

        // Empty images should fail
        let empty_input = ImageTaskInput::new(vec![]);
        assert!(task.validate_input(&empty_input).is_err());

        // Valid images should pass
        let valid_input = ImageTaskInput::new(vec![RgbImage::new(100, 100)]);
        assert!(task.validate_input(&valid_input).is_ok());
    }

    #[test]
    fn test_output_validation() {
        let task = FormulaRecognitionTask::default();

        // Valid output should pass
        let output = FormulaRecognitionOutput {
            formulas: vec!["\\frac{1}{2}".to_string()],
            scores: vec![Some(0.95)],
        };
        assert!(task.validate_output(&output).is_ok());

        // Mismatched lengths should fail
        let bad_output = FormulaRecognitionOutput {
            formulas: vec!["\\frac{1}{2}".to_string()],
            scores: vec![Some(0.95), Some(0.90)],
        };
        assert!(task.validate_output(&bad_output).is_err());
    }

    #[test]
    fn test_schema() {
        let task = FormulaRecognitionTask::default();
        let schema = task.schema();
        assert_eq!(schema.task_type, TaskType::FormulaRecognition);
        assert!(schema.input_types.contains(&"image".to_string()));
        assert!(schema.output_types.contains(&"latex_formula".to_string()));
    }
}
