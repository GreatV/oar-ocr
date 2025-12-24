//! Concrete task implementations for document orientation classification.
//!
//! This module provides the document orientation task that detects document rotation.

use super::validation::ensure_non_empty_images;
use crate::core::OCRError;
use crate::core::traits::task::{ImageTaskInput, Task, TaskSchema, TaskType};
use crate::impl_config_validator;
use crate::utils::ScoreValidator;
use serde::{Deserialize, Serialize};

/// A single classification result with class ID, label, and confidence score.
#[derive(Debug, Clone)]
pub struct Classification {
    /// The predicted class ID
    pub class_id: usize,
    /// The human-readable label for this class
    pub label: String,
    /// Confidence score for this classification (0.0 to 1.0)
    pub score: f32,
}

impl Classification {
    /// Creates a new classification.
    pub fn new(class_id: usize, label: String, score: f32) -> Self {
        Self {
            class_id,
            label,
            score,
        }
    }
}

/// Configuration for document orientation classification task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentOrientationConfig {
    /// Score threshold for classification (default: 0.5)
    pub score_threshold: f32,
    /// Number of top predictions to return (default: 4)
    pub topk: usize,
}

impl Default for DocumentOrientationConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.5,
            topk: 4,
        }
    }
}

impl_config_validator!(DocumentOrientationConfig {
    score_threshold: range(0.0, 1.0),
    topk: min(1),
});

/// Output from document orientation classification task.
#[derive(Debug, Clone)]
pub struct DocumentOrientationOutput {
    /// Classification results per image
    pub classifications: Vec<Vec<Classification>>,
}

impl DocumentOrientationOutput {
    /// Creates an empty document orientation output.
    pub fn empty() -> Self {
        Self {
            classifications: Vec::new(),
        }
    }

    /// Creates a document orientation output with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            classifications: Vec::with_capacity(capacity),
        }
    }
}

/// Document orientation classification task implementation.
#[derive(Debug, Default)]
pub struct DocumentOrientationTask {
    _config: DocumentOrientationConfig,
}

impl DocumentOrientationTask {
    /// Creates a new document orientation task.
    pub fn new(config: DocumentOrientationConfig) -> Self {
        Self { _config: config }
    }
}

impl Task for DocumentOrientationTask {
    type Config = DocumentOrientationConfig;
    type Input = ImageTaskInput;
    type Output = DocumentOrientationOutput;

    fn task_type(&self) -> TaskType {
        TaskType::DocumentOrientation
    }

    fn schema(&self) -> TaskSchema {
        TaskSchema::new(
            TaskType::DocumentOrientation,
            vec!["image".to_string()],
            vec!["orientation_labels".to_string(), "scores".to_string()],
        )
    }

    fn validate_input(&self, input: &Self::Input) -> Result<(), OCRError> {
        ensure_non_empty_images(
            &input.images,
            "No images provided for document orientation classification",
        )?;

        Ok(())
    }

    fn validate_output(&self, output: &Self::Output) -> Result<(), OCRError> {
        let validator = ScoreValidator::new_unit_range("score");

        for (idx, classifications) in output.classifications.iter().enumerate() {
            for classification in classifications.iter() {
                // Validate class IDs (should be 0-3 for 4 orientations)
                if classification.class_id > 3 {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Image {}: invalid class_id {}. Expected 0-3 (0°, 90°, 180°, 270°)",
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
        DocumentOrientationOutput::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_document_orientation_task_creation() {
        let task = DocumentOrientationTask::default();
        assert_eq!(task.task_type(), TaskType::DocumentOrientation);
    }

    #[test]
    fn test_input_validation() {
        let task = DocumentOrientationTask::default();

        // Empty images should fail
        let empty_input = ImageTaskInput::new(vec![]);
        assert!(task.validate_input(&empty_input).is_err());

        // Valid images should pass
        let valid_input = ImageTaskInput::new(vec![RgbImage::new(100, 100)]);
        assert!(task.validate_input(&valid_input).is_ok());
    }

    #[test]
    fn test_output_validation() {
        let task = DocumentOrientationTask::default();

        // Valid output should pass
        let classification1 = Classification::new(0, "0".to_string(), 0.95);
        let classification2 = Classification::new(1, "90".to_string(), 0.03);
        let output = DocumentOrientationOutput {
            classifications: vec![vec![classification1, classification2]],
        };
        assert!(task.validate_output(&output).is_ok());

        // Invalid class ID should fail (should be 0-3)
        let bad_classification = Classification::new(5, "invalid".to_string(), 0.95);
        let bad_output = DocumentOrientationOutput {
            classifications: vec![vec![bad_classification]],
        };
        assert!(task.validate_output(&bad_output).is_err());

        // Invalid score should fail
        let bad_score_classification = Classification::new(0, "0".to_string(), 1.5);
        let bad_score_output = DocumentOrientationOutput {
            classifications: vec![vec![bad_score_classification]],
        };
        assert!(task.validate_output(&bad_score_output).is_err());
    }

    #[test]
    fn test_schema() {
        let task = DocumentOrientationTask::default();
        let schema = task.schema();
        assert_eq!(schema.task_type, TaskType::DocumentOrientation);
        assert!(schema.input_types.contains(&"image".to_string()));
        assert!(
            schema
                .output_types
                .contains(&"orientation_labels".to_string())
        );
    }
}
