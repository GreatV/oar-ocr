//! Concrete task implementations for table structure recognition.
//!
//! This module provides the table structure recognition task that converts table images
//! into HTML structure with bounding boxes for cells.

use super::validation::ensure_non_empty_images;
use crate::core::OCRError;
use crate::core::traits::task::{ImageTaskInput, Task, TaskSchema, TaskType};
use crate::impl_config_validator;
use crate::utils::{ScoreValidator, validate_max_value};
use serde::{Deserialize, Serialize};

/// Configuration for table structure recognition task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableStructureRecognitionConfig {
    /// Score threshold for recognition (default: 0.5)
    pub score_threshold: f32,
    /// Maximum structure sequence length (default: 500)
    pub max_structure_length: usize,
}

impl Default for TableStructureRecognitionConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.5,
            max_structure_length: 500,
        }
    }
}

impl_config_validator!(TableStructureRecognitionConfig {
    score_threshold: range(0.0, 1.0),
    max_structure_length: min(1),
});

/// Output from table structure recognition task.
#[derive(Debug, Clone)]
pub struct TableStructureRecognitionOutput {
    /// HTML structure tokens with full HTML wrapping (one per image)
    pub structures: Vec<Vec<String>>,
    /// Bounding boxes for table cells as 8-point coordinates (floating point) (one per image)
    pub bboxes: Vec<Vec<Vec<f32>>>,
    /// Confidence scores for structure prediction (one per image)
    pub structure_scores: Vec<f32>,
}

impl TableStructureRecognitionOutput {
    /// Creates an empty table structure recognition output.
    pub fn empty() -> Self {
        Self {
            structures: Vec::new(),
            bboxes: Vec::new(),
            structure_scores: Vec::new(),
        }
    }

    /// Creates a table structure recognition output with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            structures: Vec::with_capacity(capacity),
            bboxes: Vec::with_capacity(capacity),
            structure_scores: Vec::with_capacity(capacity),
        }
    }
}

/// Table structure recognition task implementation.
#[derive(Debug, Default)]
pub struct TableStructureRecognitionTask {
    config: TableStructureRecognitionConfig,
}

impl TableStructureRecognitionTask {
    /// Creates a new table structure recognition task.
    pub fn new(config: TableStructureRecognitionConfig) -> Self {
        Self { config }
    }
}

impl Task for TableStructureRecognitionTask {
    type Config = TableStructureRecognitionConfig;
    type Input = ImageTaskInput;
    type Output = TableStructureRecognitionOutput;

    fn task_type(&self) -> TaskType {
        TaskType::TableStructureRecognition
    }

    fn schema(&self) -> TaskSchema {
        TaskSchema::new(
            TaskType::TableStructureRecognition,
            vec!["image".to_string()],
            vec![
                "structure".to_string(),
                "bbox".to_string(),
                "structure_score".to_string(),
            ],
        )
    }

    fn validate_input(&self, input: &Self::Input) -> Result<(), OCRError> {
        ensure_non_empty_images(
            &input.images,
            "No images provided for table structure recognition",
        )?;

        Ok(())
    }

    fn validate_output(&self, output: &Self::Output) -> Result<(), OCRError> {
        // Validate that all output vectors have the same length
        let num_images = output.structures.len();
        if output.bboxes.len() != num_images || output.structure_scores.len() != num_images {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "Output length mismatch: structures={}, bboxes={}, scores={}",
                    num_images,
                    output.bboxes.len(),
                    output.structure_scores.len()
                ),
            });
        }

        // Validate each image's output
        let validator = ScoreValidator::new_unit_range("score");

        for (img_idx, (structure, bboxes, score)) in output
            .structures
            .iter()
            .zip(output.bboxes.iter())
            .zip(output.structure_scores.iter())
            .map(|((s, b), sc)| (s, b, sc))
            .enumerate()
        {
            // Validate structure length
            validate_max_value(
                structure.len(),
                self.config.max_structure_length,
                "Structure length",
                &format!("Image {}", img_idx),
            )?;

            // Validate score range
            validator.validate_score(*score, &format!("Image {}", img_idx))?;

            // Validate bboxes (each should have 8 floating point coordinates)
            for (bbox_idx, bbox) in bboxes.iter().enumerate() {
                if bbox.len() != 8 {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Image {}, Bbox {}: expected 8 coordinates, got {}",
                            img_idx,
                            bbox_idx,
                            bbox.len()
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    fn empty_output(&self) -> Self::Output {
        TableStructureRecognitionOutput::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_table_structure_recognition_task_creation() {
        let task = TableStructureRecognitionTask::default();
        assert_eq!(task.task_type(), TaskType::TableStructureRecognition);
    }

    #[test]
    fn test_input_validation() {
        let task = TableStructureRecognitionTask::default();

        // Empty images should fail
        let empty_input = ImageTaskInput::new(vec![]);
        assert!(task.validate_input(&empty_input).is_err());

        // Valid images should pass
        let valid_input = ImageTaskInput::new(vec![RgbImage::new(100, 100)]);
        assert!(task.validate_input(&valid_input).is_ok());
    }

    #[test]
    fn test_output_validation() {
        let task = TableStructureRecognitionTask::default();

        // Valid output should pass
        let output = TableStructureRecognitionOutput {
            structures: vec![vec!["<html>".to_string(), "<table>".to_string()]],
            bboxes: vec![vec![vec![10.0, 10.0, 50.0, 10.0, 50.0, 30.0, 10.0, 30.0]]],
            structure_scores: vec![0.95],
        };
        assert!(task.validate_output(&output).is_ok());

        // Invalid bbox coordinates should fail
        let bad_bbox_output = TableStructureRecognitionOutput {
            structures: vec![vec!["<html>".to_string()]],
            bboxes: vec![vec![vec![10.0, 10.0, 50.0]]], // Invalid - only 3 coords instead of 8
            structure_scores: vec![0.95],
        };
        assert!(task.validate_output(&bad_bbox_output).is_err());

        // Mismatched lengths should fail
        let mismatched_output = TableStructureRecognitionOutput {
            structures: vec![vec!["<html>".to_string()]],
            bboxes: vec![vec![vec![10.0, 10.0, 50.0, 10.0, 50.0, 30.0, 10.0, 30.0]]],
            structure_scores: vec![0.95, 0.90], // Extra score
        };
        assert!(task.validate_output(&mismatched_output).is_err());
    }

    #[test]
    fn test_schema() {
        let task = TableStructureRecognitionTask::default();
        let schema = task.schema();
        assert_eq!(schema.task_type, TaskType::TableStructureRecognition);
        assert!(schema.input_types.contains(&"image".to_string()));
        assert!(schema.output_types.contains(&"structure".to_string()));
        assert!(schema.output_types.contains(&"bbox".to_string()));
    }
}
