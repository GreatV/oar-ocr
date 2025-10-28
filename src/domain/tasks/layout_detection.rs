//! Concrete task implementations for layout detection.
//!
//! This module provides the layout detection task that identifies document layout elements.

use super::validation::ensure_non_empty_images;
use crate::core::OCRError;
use crate::core::traits::task::{ImageTaskInput, Task, TaskSchema, TaskType};
use crate::processors::BoundingBox;
use serde::{Deserialize, Serialize};

/// Configuration for layout detection task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayoutDetectionConfig {
    /// Score threshold for detection (default: 0.5)
    pub score_threshold: f32,
    /// Maximum number of layout elements (default: 100)
    pub max_elements: usize,
}

impl Default for LayoutDetectionConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.5,
            max_elements: 100,
        }
    }
}

/// A detected layout element.
#[derive(Debug, Clone)]
pub struct LayoutElement {
    /// Bounding box of the element
    pub bbox: BoundingBox,
    /// Type of layout element (string label)
    pub element_type: String,
    /// Confidence score
    pub score: f32,
}

/// Output from layout detection task.
#[derive(Debug, Clone)]
pub struct LayoutDetectionOutput {
    /// Detected layout elements per image
    pub elements: Vec<Vec<LayoutElement>>,
}

impl LayoutDetectionOutput {
    /// Creates an empty layout detection output.
    pub fn empty() -> Self {
        Self {
            elements: Vec::new(),
        }
    }

    /// Creates a layout detection output with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            elements: Vec::with_capacity(capacity),
        }
    }
}

/// Layout detection task implementation.
#[derive(Debug, Default)]
pub struct LayoutDetectionTask {
    config: LayoutDetectionConfig,
}

impl LayoutDetectionTask {
    /// Creates a new layout detection task.
    pub fn new(config: LayoutDetectionConfig) -> Self {
        Self { config }
    }
}

impl Task for LayoutDetectionTask {
    type Config = LayoutDetectionConfig;
    type Input = ImageTaskInput;
    type Output = LayoutDetectionOutput;

    fn task_type(&self) -> TaskType {
        TaskType::LayoutDetection
    }

    fn schema(&self) -> TaskSchema {
        TaskSchema::new(
            TaskType::LayoutDetection,
            vec!["image".to_string()],
            vec!["layout_elements".to_string()],
        )
    }

    fn validate_input(&self, input: &Self::Input) -> Result<(), OCRError> {
        ensure_non_empty_images(&input.images, "No images provided for layout detection")?;

        Ok(())
    }

    fn validate_output(&self, output: &Self::Output) -> Result<(), OCRError> {
        // Validate that each image doesn't exceed max_elements
        for (idx, elements) in output.elements.iter().enumerate() {
            if elements.len() > self.config.max_elements {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Image {}: element count ({}) exceeds maximum ({})",
                        idx,
                        elements.len(),
                        self.config.max_elements
                    ),
                });
            }

            // Validate scores
            for (elem_idx, element) in elements.iter().enumerate() {
                if !(0.0..=1.0).contains(&element.score) {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Image {}, element {}: score {} is out of valid range [0, 1]",
                            idx, elem_idx, element.score
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    fn empty_output(&self) -> Self::Output {
        LayoutDetectionOutput::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processors::Point;
    use image::RgbImage;

    #[test]
    fn test_layout_detection_task_creation() {
        let task = LayoutDetectionTask::default();
        assert_eq!(task.task_type(), TaskType::LayoutDetection);
    }

    #[test]
    fn test_input_validation() {
        let task = LayoutDetectionTask::default();

        // Empty images should fail
        let empty_input = ImageTaskInput::new(vec![]);
        assert!(task.validate_input(&empty_input).is_err());

        // Valid images should pass
        let valid_input = ImageTaskInput::new(vec![RgbImage::new(100, 100)]);
        assert!(task.validate_input(&valid_input).is_ok());
    }

    #[test]
    fn test_output_validation() {
        let task = LayoutDetectionTask::default();

        // Valid output should pass
        let box1 = BoundingBox::new(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ]);
        let element = LayoutElement {
            bbox: box1,
            element_type: "text".to_string(),
            score: 0.95,
        };
        let output = LayoutDetectionOutput {
            elements: vec![vec![element]],
        };
        assert!(task.validate_output(&output).is_ok());
    }

    #[test]
    fn test_schema() {
        let task = LayoutDetectionTask::default();
        let schema = task.schema();
        assert_eq!(schema.task_type, TaskType::LayoutDetection);
        assert!(schema.input_types.contains(&"image".to_string()));
        assert!(schema.output_types.contains(&"layout_elements".to_string()));
    }
}
