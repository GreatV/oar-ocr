//! Concrete task implementations for text line orientation classification.
//!
//! This module provides the text line orientation task that detects text line rotation.

use crate::core::OCRError;
use crate::core::traits::task::{ImageTaskInput, Task, TaskSchema, TaskType};
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

/// Output from text line orientation classification task.
#[derive(Debug, Clone)]
pub struct TextLineOrientationOutput {
    /// Predicted class IDs per image (0=0°, 1=180°)
    pub class_ids: Vec<Vec<usize>>,
    /// Confidence scores for each prediction
    pub scores: Vec<Vec<f32>>,
    /// Label names for each prediction (e.g., "0", "180")
    pub label_names: Vec<Vec<String>>,
}

impl TextLineOrientationOutput {
    /// Creates an empty text line orientation output.
    pub fn empty() -> Self {
        Self {
            class_ids: Vec::new(),
            scores: Vec::new(),
            label_names: Vec::new(),
        }
    }

    /// Creates a text line orientation output with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            class_ids: Vec::with_capacity(capacity),
            scores: Vec::with_capacity(capacity),
            label_names: Vec::with_capacity(capacity),
        }
    }
}

/// Text line orientation classification task implementation.
#[derive(Debug, Default)]
pub struct TextLineOrientationTask {
    #[allow(dead_code)]
    config: TextLineOrientationConfig,
}

impl TextLineOrientationTask {
    /// Creates a new text line orientation task.
    pub fn new(config: TextLineOrientationConfig) -> Self {
        Self { config }
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
        if input.images.is_empty() {
            return Err(OCRError::InvalidInput {
                message: "No images provided for text line orientation classification".to_string(),
            });
        }

        // Validate image dimensions
        for (idx, img) in input.images.iter().enumerate() {
            if img.width() == 0 || img.height() == 0 {
                return Err(OCRError::InvalidInput {
                    message: format!("Image at index {} has zero dimensions", idx),
                });
            }
        }

        Ok(())
    }

    fn validate_output(&self, output: &Self::Output) -> Result<(), OCRError> {
        // Validate that all arrays have matching lengths
        if output.class_ids.len() != output.scores.len()
            || output.class_ids.len() != output.label_names.len()
        {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "Mismatched output lengths: class_ids={}, scores={}, labels={}",
                    output.class_ids.len(),
                    output.scores.len(),
                    output.label_names.len()
                ),
            });
        }

        // Validate each image's predictions
        for (idx, (class_ids, scores)) in output
            .class_ids
            .iter()
            .zip(output.scores.iter())
            .enumerate()
        {
            if class_ids.len() != scores.len() {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Image {}: class_ids count ({}) doesn't match scores count ({})",
                        idx,
                        class_ids.len(),
                        scores.len()
                    ),
                });
            }

            // Validate class IDs (should be 0-1 for 2 orientations: 0° and 180°)
            for &class_id in class_ids {
                if class_id > 1 {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Image {}: invalid class_id {}. Expected 0-1 (0°, 180°)",
                            idx, class_id
                        ),
                    });
                }
            }

            // Validate score ranges
            for (pred_idx, &score) in scores.iter().enumerate() {
                if !(0.0..=1.0).contains(&score) {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Image {}, prediction {}: score {} is out of valid range [0, 1]",
                            idx, pred_idx, score
                        ),
                    });
                }
            }
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
        let output = TextLineOrientationOutput {
            class_ids: vec![vec![0, 1]],
            scores: vec![vec![0.95, 0.05]],
            label_names: vec![vec!["0".to_string(), "180".to_string()]],
        };
        assert!(task.validate_output(&output).is_ok());

        // Invalid class ID should fail
        let bad_output = TextLineOrientationOutput {
            class_ids: vec![vec![2]], // Invalid: should be 0-1
            scores: vec![vec![0.95]],
            label_names: vec![vec!["invalid".to_string()]],
        };
        assert!(task.validate_output(&bad_output).is_err());

        // Mismatched lengths should fail
        let bad_output2 = TextLineOrientationOutput {
            class_ids: vec![vec![0]],
            scores: vec![vec![0.95, 0.05]], // Mismatch
            label_names: vec![vec!["0".to_string()]],
        };
        assert!(task.validate_output(&bad_output2).is_err());
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
