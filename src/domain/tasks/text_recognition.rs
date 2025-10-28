//! Concrete task implementations for text recognition.
//!
//! This module provides the text recognition task that converts text regions to strings.

use super::validation::ensure_non_empty_images;
use crate::core::OCRError;
use crate::core::traits::task::{Task, TaskSchema, TaskType};
use image::RgbImage;
use serde::{Deserialize, Serialize};

/// Configuration for text recognition task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextRecognitionConfig {
    /// Score threshold for recognition (default: 0.5)
    pub score_threshold: f32,
    /// Maximum text length (default: 100)
    pub max_text_length: usize,
}

impl Default for TextRecognitionConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.5,
            max_text_length: 100,
        }
    }
}

/// Input for text recognition task (cropped text images).
#[derive(Debug, Clone)]
pub struct TextRecognitionInput {
    /// Cropped text images to recognize
    pub images: Vec<RgbImage>,
}

impl TextRecognitionInput {
    /// Creates a new text recognition input.
    pub fn new(images: Vec<RgbImage>) -> Self {
        Self { images }
    }
}

/// Output from text recognition task.
#[derive(Debug, Clone)]
pub struct TextRecognitionOutput {
    /// Recognized text strings
    pub texts: Vec<String>,
    /// Confidence scores for each text
    pub scores: Vec<f32>,
}

impl TextRecognitionOutput {
    /// Creates an empty text recognition output.
    pub fn empty() -> Self {
        Self {
            texts: Vec::new(),
            scores: Vec::new(),
        }
    }

    /// Creates a text recognition output with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            texts: Vec::with_capacity(capacity),
            scores: Vec::with_capacity(capacity),
        }
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
    type Input = TextRecognitionInput;
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
        if output.texts.len() != output.scores.len() {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "Mismatch between texts count ({}) and scores count ({})",
                    output.texts.len(),
                    output.scores.len()
                ),
            });
        }

        // Validate score ranges
        for (idx, &score) in output.scores.iter().enumerate() {
            if !(0.0..=1.0).contains(&score) {
                return Err(OCRError::InvalidInput {
                    message: format!("Text {}: score {} is out of valid range [0, 1]", idx, score),
                });
            }
        }

        // Validate text lengths
        for (idx, text) in output.texts.iter().enumerate() {
            if text.len() > self.config.max_text_length {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Text {}: length {} exceeds maximum {}",
                        idx,
                        text.len(),
                        self.config.max_text_length
                    ),
                });
            }
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

    #[test]
    fn test_text_recognition_task_creation() {
        let task = TextRecognitionTask::default();
        assert_eq!(task.task_type(), TaskType::TextRecognition);
    }

    #[test]
    fn test_input_validation() {
        let task = TextRecognitionTask::default();

        // Empty images should fail
        let empty_input = TextRecognitionInput::new(vec![]);
        assert!(task.validate_input(&empty_input).is_err());

        // Valid images should pass
        let valid_input = TextRecognitionInput::new(vec![RgbImage::new(100, 32)]);
        assert!(task.validate_input(&valid_input).is_ok());
    }

    #[test]
    fn test_output_validation() {
        let task = TextRecognitionTask::default();

        // Matching texts and scores should pass
        let output = TextRecognitionOutput {
            texts: vec!["Hello".to_string()],
            scores: vec![0.95],
        };
        assert!(task.validate_output(&output).is_ok());

        // Mismatched lengths should fail
        let bad_output = TextRecognitionOutput {
            texts: vec![],
            scores: vec![0.95],
        };
        assert!(task.validate_output(&bad_output).is_err());

        // Invalid score should fail
        let bad_score = TextRecognitionOutput {
            texts: vec!["Hello".to_string()],
            scores: vec![1.5],
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
