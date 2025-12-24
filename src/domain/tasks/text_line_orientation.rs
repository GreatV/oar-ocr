//! Concrete task implementations for text line orientation classification.
//!
//! This module provides the text line orientation task that detects text line rotation.

use super::document_orientation::Classification;
use super::validation::ensure_non_empty_images;
use crate::core::OCRError;
use crate::core::traits::task::{ImageTaskInput, Task, TaskSchema, TaskType};
use crate::impl_config_validator;
use crate::utils::ScoreValidator;
use serde::{Deserialize, Serialize};

/// Configuration for text line orientation classification task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextLineOrientationConfig {
    /// Score threshold for classification (default: 0.5)
    pub score_threshold: f32,
    /// Number of top predictions to return (default: 2)
    pub topk: usize,
}

impl Default for TextLineOrientationConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.5,
            topk: 2,
        }
    }
}

impl_config_validator!(TextLineOrientationConfig {
    score_threshold: range(0.0, 1.0),
    topk: min(1),
});

/// Output from text line orientation classification task.
#[derive(Debug, Clone)]
pub struct TextLineOrientationOutput {
    /// Classification results per image
    pub classifications: Vec<Vec<Classification>>,
}

impl TextLineOrientationOutput {
    /// Creates an empty text line orientation output.
    pub fn empty() -> Self {
        Self {
            classifications: Vec::new(),
        }
    }

    /// Creates a text line orientation output with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            classifications: Vec::with_capacity(capacity),
        }
    }
}

/// Text line orientation classification task implementation.
#[derive(Debug, Default)]
pub struct TextLineOrientationTask {
    _config: TextLineOrientationConfig,
}

impl TextLineOrientationTask {
    /// Creates a new text line orientation task.
    pub fn new(config: TextLineOrientationConfig) -> Self {
        Self { _config: config }
    }
}

impl Task for TextLineOrientationTask {
    type Config = TextLineOrientationConfig;
    type Input = ImageTaskInput;
    type Output = TextLineOrientationOutput;

    fn task_type(&self) -> TaskType {
        TaskType::TextLineOrientation
    }

    fn schema(&self) -> TaskSchema {
        TaskSchema::new(
            TaskType::TextLineOrientation,
            vec!["image".to_string()],
            vec!["orientation_labels".to_string(), "scores".to_string()],
        )
    }

    fn validate_input(&self, input: &Self::Input) -> Result<(), OCRError> {
        ensure_non_empty_images(
            &input.images,
            "No images provided for text line orientation classification",
        )?;

        Ok(())
    }

    fn validate_output(&self, output: &Self::Output) -> Result<(), OCRError> {
        let validator = ScoreValidator::new_unit_range("score");

        for (idx, classifications) in output.classifications.iter().enumerate() {
            for classification in classifications.iter() {
                // Validate class IDs (should be 0-1 for 2 orientations: 0° and 180°)
                if classification.class_id > 1 {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Image {}: invalid class_id {}. Expected 0-1 (0°, 180°)",
                            idx, classification.class_id
                        ),
                    });
                }
            }

            // Validate score ranges
            let scores: Vec<f32> = classifications.iter().map(|c| c.score).collect();
            validator.validate_scores_with(&scores, |class_idx| {
                format!("Image {}, classification {}", idx, class_idx)
            })?;
        }

        Ok(())
    }

    fn empty_output(&self) -> Self::Output {
        TextLineOrientationOutput::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_text_line_orientation_task_creation() {
        let task = TextLineOrientationTask::default();
        assert_eq!(task.task_type(), TaskType::TextLineOrientation);
    }

    #[test]
    fn test_input_validation() {
        let task = TextLineOrientationTask::default();

        // Empty images should fail
        let empty_input = ImageTaskInput::new(vec![]);
        assert!(task.validate_input(&empty_input).is_err());

        // Valid images should pass
        let valid_input = ImageTaskInput::new(vec![RgbImage::new(100, 100)]);
        assert!(task.validate_input(&valid_input).is_ok());
    }

    #[test]
    fn test_output_validation() {
        let task = TextLineOrientationTask::default();

        // Valid output should pass
        let classification1 = Classification::new(0, "0".to_string(), 0.95);
        let classification2 = Classification::new(1, "180".to_string(), 0.05);
        let output = TextLineOrientationOutput {
            classifications: vec![vec![classification1, classification2]],
        };
        assert!(task.validate_output(&output).is_ok());

        // Invalid class ID should fail
        let bad_classification = Classification::new(2, "invalid".to_string(), 0.95); // Invalid: should be 0-1
        let bad_output = TextLineOrientationOutput {
            classifications: vec![vec![bad_classification]],
        };
        assert!(task.validate_output(&bad_output).is_err());

        // Invalid score should fail
        let bad_score_classification = Classification::new(0, "0".to_string(), 1.5); // Invalid score
        let bad_score_output = TextLineOrientationOutput {
            classifications: vec![vec![bad_score_classification]],
        };
        assert!(task.validate_output(&bad_score_output).is_err());
    }

    #[test]
    fn test_schema() {
        let task = TextLineOrientationTask::default();
        let schema = task.schema();
        assert_eq!(schema.task_type, TaskType::TextLineOrientation);
        assert!(schema.input_types.contains(&"image".to_string()));
        assert!(
            schema
                .output_types
                .contains(&"orientation_labels".to_string())
        );
    }
}
