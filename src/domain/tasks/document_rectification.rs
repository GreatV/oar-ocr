//! Concrete task implementations for document rectification.
//!
//! This module provides the document rectification task that corrects distortions in document images.

use crate::core::OCRError;
use crate::core::traits::task::{ImageTaskInput, Task, TaskSchema, TaskType};
use image::RgbImage;
use serde::{Deserialize, Serialize};

/// Configuration for document rectification task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentRectificationConfig {
    /// Input shape for the rectification model [channels, height, width]
    pub rec_image_shape: [usize; 3],
}

impl Default for DocumentRectificationConfig {
    fn default() -> Self {
        Self {
            rec_image_shape: [3, 0, 0],
        }
    }
}

/// Output from document rectification task.
#[derive(Debug, Clone)]
pub struct DocumentRectificationOutput {
    /// Rectified images
    pub rectified_images: Vec<RgbImage>,
}

impl DocumentRectificationOutput {
    /// Creates an empty document rectification output.
    pub fn empty() -> Self {
        Self {
            rectified_images: Vec::new(),
        }
    }

    /// Creates a document rectification output with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            rectified_images: Vec::with_capacity(capacity),
        }
    }
}

/// Document rectification task implementation.
#[derive(Debug, Default)]
pub struct DocumentRectificationTask {
    #[allow(dead_code)]
    config: DocumentRectificationConfig,
}

impl DocumentRectificationTask {
    /// Creates a new document rectification task.
    pub fn new(config: DocumentRectificationConfig) -> Self {
        Self { config }
    }
}

impl Task for DocumentRectificationTask {
    type Config = DocumentRectificationConfig;
    type Input = ImageTaskInput;
    type Output = DocumentRectificationOutput;

    fn task_type(&self) -> TaskType {
        TaskType::DocumentRectification
    }

    fn schema(&self) -> TaskSchema {
        TaskSchema::new(
            TaskType::DocumentRectification,
            vec!["image".to_string()],
            vec!["rectified_image".to_string()],
        )
    }

    fn validate_input(&self, input: &Self::Input) -> Result<(), OCRError> {
        if input.images.is_empty() {
            return Err(OCRError::InvalidInput {
                message: "No images provided for document rectification".to_string(),
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
        // Validate that we have rectified images
        if output.rectified_images.is_empty() {
            return Err(OCRError::InvalidInput {
                message: "No rectified images in output".to_string(),
            });
        }

        // Validate each rectified image
        for (idx, img) in output.rectified_images.iter().enumerate() {
            if img.width() == 0 || img.height() == 0 {
                return Err(OCRError::InvalidInput {
                    message: format!("Rectified image at index {} has zero dimensions", idx),
                });
            }
        }

        Ok(())
    }

    fn empty_output(&self) -> Self::Output {
        DocumentRectificationOutput::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_document_rectification_task_creation() {
        let task = DocumentRectificationTask::default();
        assert_eq!(task.task_type(), TaskType::DocumentRectification);
    }

    #[test]
    fn test_input_validation() {
        let task = DocumentRectificationTask::default();

        // Empty images should fail
        let empty_input = ImageTaskInput::new(vec![]);
        assert!(task.validate_input(&empty_input).is_err());

        // Valid images should pass
        let valid_input = ImageTaskInput::new(vec![RgbImage::new(100, 100)]);
        assert!(task.validate_input(&valid_input).is_ok());
    }

    #[test]
    fn test_output_validation() {
        let task = DocumentRectificationTask::default();

        // Valid output should pass
        let output = DocumentRectificationOutput {
            rectified_images: vec![RgbImage::new(100, 100)],
        };
        assert!(task.validate_output(&output).is_ok());

        // Empty output should fail
        let empty_output = DocumentRectificationOutput::empty();
        assert!(task.validate_output(&empty_output).is_err());
    }

    #[test]
    fn test_schema() {
        let task = DocumentRectificationTask::default();
        let schema = task.schema();
        assert_eq!(schema.task_type, TaskType::DocumentRectification);
        assert!(schema.input_types.contains(&"image".to_string()));
        assert!(schema.output_types.contains(&"rectified_image".to_string()));
    }
}
