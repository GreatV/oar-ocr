//! Concrete task implementations for table structure recognition.
//!
//! This module provides the table structure recognition task that converts table images
//! into HTML structure with bounding boxes for cells.

use crate::core::OCRError;
use crate::core::traits::task::{ImageTaskInput, Task, TaskSchema, TaskType};
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

/// Output from table structure recognition task.
#[derive(Debug, Clone)]
pub struct TableStructureRecognitionOutput {
    /// HTML structure tokens with full HTML wrapping
    pub structure: Vec<String>,
    /// Bounding boxes for table cells as 8-point coordinates (integer)
    pub bbox: Vec<Vec<i32>>,
    /// Confidence score for structure prediction
    pub structure_score: f32,
}

impl TableStructureRecognitionOutput {
    /// Creates an empty table structure recognition output.
    pub fn empty() -> Self {
        Self {
            structure: Vec::new(),
            bbox: Vec::new(),
            structure_score: 0.0,
        }
    }

    /// Creates a table structure recognition output with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            structure: Vec::with_capacity(capacity),
            bbox: Vec::with_capacity(capacity),
            structure_score: 0.0,
        }
    }
}

/// Table structure recognition task implementation.
#[derive(Debug, Default)]
pub struct TableStructureRecognitionTask {
    #[allow(dead_code)]
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
        if input.images.is_empty() {
            return Err(OCRError::InvalidInput {
                message: "No images provided for table structure recognition".to_string(),
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
        // Validate structure length
        if output.structure.len() > self.config.max_structure_length {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "Structure length {} exceeds maximum {}",
                    output.structure.len(),
                    self.config.max_structure_length
                ),
            });
        }

        // Validate score range
        if !(0.0..=1.0).contains(&output.structure_score) {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "Score {} is out of valid range [0, 1]",
                    output.structure_score
                ),
            });
        }

        // Validate bboxes (each should have 8 integer coordinates)
        for (bbox_idx, bbox) in output.bbox.iter().enumerate() {
            if bbox.len() != 8 {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Bbox {}: expected 8 coordinates, got {}",
                        bbox_idx,
                        bbox.len()
                    ),
                });
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
            structure: vec!["<html>".to_string(), "<table>".to_string()],
            bbox: vec![vec![10, 10, 50, 10, 50, 30, 10, 30]],
            structure_score: 0.95,
        };
        assert!(task.validate_output(&output).is_ok());

        // Invalid bbox coordinates should fail
        let bad_bbox_output = TableStructureRecognitionOutput {
            structure: vec!["<html>".to_string()],
            bbox: vec![vec![10, 10, 50]], // Invalid - only 3 coords instead of 8
            structure_score: 0.95,
        };
        assert!(task.validate_output(&bad_bbox_output).is_err());
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
