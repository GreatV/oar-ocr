//! Concrete task implementations for text recognition.
//!
//! This module provides the text recognition task that converts text regions to strings.

use super::validation::ensure_non_empty_images;
use crate::core::OCRError;
use crate::core::traits::TaskDefinition;
use crate::core::traits::task::{ImageTaskInput, Task, TaskSchema, TaskType};
use crate::impl_config_validator;
use crate::utils::{ScoreValidator, validate_length_match, validate_max_value};
use serde::{Deserialize, Serialize};

/// Configuration for text recognition task.
///
/// Default values are aligned with PP-StructureV3.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextRecognitionConfig {
    /// Score threshold for recognition (default: 0.0, no filtering)
    pub score_threshold: f32,
    /// Maximum text length (default: 25)
    pub max_text_length: usize,
}

impl Default for TextRecognitionConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.0,
            max_text_length: 25,
        }
    }
}

impl_config_validator!(TextRecognitionConfig {
    score_threshold: range(0.0, 1.0),
    max_text_length: min(1),
});

/// Output from text recognition task.
#[derive(Debug, Clone)]
pub struct TextRecognitionOutput {
    /// Recognized text strings
    pub texts: Vec<String>,
    /// Confidence scores for each text
    pub scores: Vec<f32>,
    /// Character/word positions within each text line (optional)
    /// Each inner vector contains normalized x-positions (0.0-1.0) for characters
    /// Only populated when word box detection is enabled
    pub char_positions: Vec<Vec<f32>>,
    /// Column indices for each character in the CTC output
    /// Used for accurate word box generation with compatible approach
    pub char_col_indices: Vec<Vec<usize>>,
    /// Total number of columns (sequence length) in the CTC output for each text line
    pub sequence_lengths: Vec<usize>,
}

impl TextRecognitionOutput {
    /// Creates an empty text recognition output.
    pub fn empty() -> Self {
        Self {
            texts: Vec::new(),
            scores: Vec::new(),
            char_positions: Vec::new(),
            char_col_indices: Vec::new(),
            sequence_lengths: Vec::new(),
        }
    }

    /// Creates a text recognition output with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            texts: Vec::with_capacity(capacity),
            scores: Vec::with_capacity(capacity),
            char_positions: Vec::with_capacity(capacity),
            char_col_indices: Vec::with_capacity(capacity),
            sequence_lengths: Vec::with_capacity(capacity),
        }
    }
}

impl Default for TextRecognitionOutput {
    fn default() -> Self {
        Self::empty()
    }
}

impl TaskDefinition for TextRecognitionOutput {
    const TASK_NAME: &'static str = "text_recognition";
    const TASK_DOC: &'static str = "Text recognition - converting text regions to strings";

    fn empty() -> Self {
        Self::empty()
    }
}

/// Text recognition task implementation.
#[derive(Debug, Default)]
pub struct TextRecognitionTask {
    config: TextRecognitionConfig,
}

impl TextRecognitionTask {
    /// Creates a new text recognition task.
    pub fn new(config: TextRecognitionConfig) -> Self {
        Self { config }
    }
}

impl Task for TextRecognitionTask {
    type Config = TextRecognitionConfig;
    type Input = ImageTaskInput;
    type Output = TextRecognitionOutput;

    fn task_type(&self) -> TaskType {
        TaskType::TextRecognition
    }

    fn schema(&self) -> TaskSchema {
        TaskSchema::new(
            TaskType::TextRecognition,
            vec!["text_boxes".to_string()],
            vec!["text_strings".to_string(), "scores".to_string()],
        )
    }

    fn validate_input(&self, input: &Self::Input) -> Result<(), OCRError> {
        ensure_non_empty_images(&input.images, "No images provided for text recognition")?;

        Ok(())
    }

    fn validate_output(&self, output: &Self::Output) -> Result<(), OCRError> {
        // Validate that texts and scores have matching lengths
        validate_length_match(output.texts.len(), output.scores.len(), "texts", "scores")?;

        // Validate score ranges
        let validator = ScoreValidator::new_unit_range("score");
        validator.validate_scores_with(&output.scores, |idx| format!("Text {}", idx))?;

        // Validate text lengths
        for (idx, text) in output.texts.iter().enumerate() {
            validate_max_value(
                text.len(),
                self.config.max_text_length,
                "length",
                &format!("Text {}", idx),
            )?;
        }

        Ok(())
    }

    fn empty_output(&self) -> Self::Output {
        TextRecognitionOutput::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_text_recognition_task_creation() {
        let task = TextRecognitionTask::default();
        assert_eq!(task.task_type(), TaskType::TextRecognition);
    }

    #[test]
    fn test_input_validation() {
        let task = TextRecognitionTask::default();

        // Empty images should fail
        let empty_input = ImageTaskInput::new(vec![]);
        assert!(task.validate_input(&empty_input).is_err());

        // Valid images should pass
        let valid_input = ImageTaskInput::new(vec![RgbImage::new(100, 32)]);
        assert!(task.validate_input(&valid_input).is_ok());
    }

    #[test]
    fn test_output_validation() {
        let task = TextRecognitionTask::default();

        // Matching texts and scores should pass
        let output = TextRecognitionOutput {
            texts: vec!["Hello".to_string()],
            scores: vec![0.95],
            ..Default::default()
        };
        assert!(task.validate_output(&output).is_ok());

        // Mismatched lengths should fail
        let bad_output = TextRecognitionOutput {
            texts: vec![],
            scores: vec![0.95],
            ..Default::default()
        };
        assert!(task.validate_output(&bad_output).is_err());

        // Invalid score should fail
        let bad_score = TextRecognitionOutput {
            texts: vec!["Hello".to_string()],
            scores: vec![1.5],
            ..Default::default()
        };
        assert!(task.validate_output(&bad_score).is_err());
    }

    #[test]
    fn test_schema() {
        let task = TextRecognitionTask::default();
        let schema = task.schema();
        assert_eq!(schema.task_type, TaskType::TextRecognition);
        assert!(schema.input_types.contains(&"text_boxes".to_string()));
        assert!(schema.output_types.contains(&"text_strings".to_string()));
    }
}
