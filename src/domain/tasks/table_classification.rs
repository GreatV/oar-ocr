//! Concrete task implementations for table classification.
//!
//! This module provides the table classification task that classifies table images
//! as either "wired_table" (tables with borders) or "wireless_table" (tables without borders).

use super::document_orientation::Classification;
use super::validation::ensure_non_empty_images;
use crate::core::OCRError;
use crate::core::traits::task::{ImageTaskInput, Task, TaskSchema, TaskType};
use crate::impl_config_validator;
use crate::utils::ScoreValidator;
use serde::{Deserialize, Serialize};

/// Configuration for table classification task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableClassificationConfig {
    /// Score threshold for classification (default: 0.5)
    pub score_threshold: f32,
    /// Number of top predictions to return (default: 2)
    pub topk: usize,
}

impl Default for TableClassificationConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.5,
            topk: 2,
        }
    }
}

impl_config_validator!(TableClassificationConfig {
    score_threshold: range(0.0, 1.0),
    topk: min(1),
});

/// Output from table classification task.
#[derive(Debug, Clone)]
pub struct TableClassificationOutput {
    /// Classification results per image
    pub classifications: Vec<Vec<Classification>>,
}

impl TableClassificationOutput {
    /// Creates an empty table classification output.
    pub fn empty() -> Self {
        Self {
            classifications: Vec::new(),
        }
    }

    /// Creates a table classification output with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            classifications: Vec::with_capacity(capacity),
        }
    }
}

/// Table classification task implementation.
#[derive(Debug, Default)]
pub struct TableClassificationTask {
    _config: TableClassificationConfig,
}

impl TableClassificationTask {
    /// Creates a new table classification task.
    pub fn new(config: TableClassificationConfig) -> Self {
        Self { _config: config }
    }
}

impl Task for TableClassificationTask {
    type Config = TableClassificationConfig;
    type Input = ImageTaskInput;
    type Output = TableClassificationOutput;

    fn task_type(&self) -> TaskType {
        TaskType::TableClassification
    }

    fn schema(&self) -> TaskSchema {
        TaskSchema::new(
            TaskType::TableClassification,
            vec!["image".to_string()],
            vec!["table_type_labels".to_string(), "scores".to_string()],
        )
    }

    fn validate_input(&self, input: &Self::Input) -> Result<(), OCRError> {
        ensure_non_empty_images(&input.images, "No images provided for table classification")?;

        Ok(())
    }

    fn validate_output(&self, output: &Self::Output) -> Result<(), OCRError> {
        let validator = ScoreValidator::new_unit_range("score");

        for (idx, classifications) in output.classifications.iter().enumerate() {
            for classification in classifications.iter() {
                // Validate class IDs (should be 0-1 for 2 table types)
                if classification.class_id > 1 {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Image {}: invalid class_id {}. Expected 0-1 (wired_table, wireless_table)",
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
        TableClassificationOutput::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_table_classification_task_creation() {
        let task = TableClassificationTask::default();
        assert_eq!(task.task_type(), TaskType::TableClassification);
    }

    #[test]
    fn test_input_validation() {
        let task = TableClassificationTask::default();

        // Empty images should fail
        let empty_input = ImageTaskInput::new(vec![]);
        assert!(task.validate_input(&empty_input).is_err());

        // Valid images should pass
        let valid_input = ImageTaskInput::new(vec![RgbImage::new(100, 100)]);
        assert!(task.validate_input(&valid_input).is_ok());
    }

    #[test]
    fn test_output_validation() {
        let task = TableClassificationTask::default();

        // Valid output should pass
        let classification1 = Classification::new(0, "wired_table".to_string(), 0.85);
        let classification2 = Classification::new(1, "wireless_table".to_string(), 0.15);
        let output = TableClassificationOutput {
            classifications: vec![vec![classification1, classification2]],
        };
        assert!(task.validate_output(&output).is_ok());

        // Invalid class ID should fail (should be 0-1)
        let bad_classification = Classification::new(2, "invalid".to_string(), 0.95);
        let bad_output = TableClassificationOutput {
            classifications: vec![vec![bad_classification]],
        };
        assert!(task.validate_output(&bad_output).is_err());

        // Invalid score should fail
        let bad_score_classification = Classification::new(0, "wired_table".to_string(), 1.5);
        let bad_score_output = TableClassificationOutput {
            classifications: vec![vec![bad_score_classification]],
        };
        assert!(task.validate_output(&bad_score_output).is_err());
    }

    #[test]
    fn test_schema() {
        let task = TableClassificationTask::default();
        let schema = task.schema();
        assert_eq!(schema.task_type, TaskType::TableClassification);
        assert!(schema.input_types.contains(&"image".to_string()));
        assert!(
            schema
                .output_types
                .contains(&"table_type_labels".to_string())
        );
    }
}
