//! Concrete task implementations for text detection.
//!
//! This module provides the text detection task that locates text regions in images.

use crate::core::OCRError;
use crate::core::traits::task::{ImageTaskInput, Task, TaskSchema, TaskType};
use crate::processors::BoundingBox;
use serde::{Deserialize, Serialize};

/// Configuration for text detection task.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDetectionConfig {
    /// Score threshold for detection (default: 0.3)
    pub score_threshold: f32,
    /// Box threshold for filtering (default: 0.7)
    pub box_threshold: f32,
    /// Unclip ratio for expanding detected regions (default: 1.5)
    pub unclip_ratio: f32,
    /// Maximum candidates to consider (default: 1000)
    pub max_candidates: usize,
}

impl Default for TextDetectionConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.3,
            box_threshold: 0.7,
            unclip_ratio: 1.5,
            max_candidates: 1000,
        }
    }
}

/// Output from text detection task.
#[derive(Debug, Clone)]
pub struct TextDetectionOutput {
    /// Detected text bounding boxes per image
    pub boxes: Vec<Vec<BoundingBox>>,
    /// Confidence scores for each box
    pub scores: Vec<Vec<f32>>,
}

impl TextDetectionOutput {
    /// Creates an empty text detection output.
    pub fn empty() -> Self {
        Self {
            boxes: Vec::new(),
            scores: Vec::new(),
        }
    }

    /// Creates a text detection output with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            boxes: Vec::with_capacity(capacity),
            scores: Vec::with_capacity(capacity),
        }
    }
}

/// Text detection task implementation.
#[derive(Debug, Default)]
pub struct TextDetectionTask {
    #[allow(dead_code)]
    config: TextDetectionConfig,
}

impl TextDetectionTask {
    /// Creates a new text detection task.
    pub fn new(config: TextDetectionConfig) -> Self {
        Self { config }
    }
}

impl Task for TextDetectionTask {
    type Config = TextDetectionConfig;
    type Input = ImageTaskInput;
    type Output = TextDetectionOutput;

    fn task_type(&self) -> TaskType {
        TaskType::TextDetection
    }

    fn schema(&self) -> TaskSchema {
        TaskSchema::new(
            TaskType::TextDetection,
            vec!["image".to_string()],
            vec!["text_boxes".to_string(), "scores".to_string()],
        )
    }

    fn validate_input(&self, input: &Self::Input) -> Result<(), OCRError> {
        if input.images.is_empty() {
            return Err(OCRError::InvalidInput {
                message: "No images provided for text detection".to_string(),
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
        // Validate that boxes and scores have matching lengths
        if output.boxes.len() != output.scores.len() {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "Mismatch between boxes count ({}) and scores count ({})",
                    output.boxes.len(),
                    output.scores.len()
                ),
            });
        }

        // Validate that each image's boxes and scores match
        for (idx, (boxes, scores)) in output.boxes.iter().zip(output.scores.iter()).enumerate() {
            if boxes.len() != scores.len() {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Image {}: boxes count ({}) doesn't match scores count ({})",
                        idx,
                        boxes.len(),
                        scores.len()
                    ),
                });
            }

            // Validate score ranges
            for (box_idx, &score) in scores.iter().enumerate() {
                if !(0.0..=1.0).contains(&score) {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Image {}, box {}: score {} is out of valid range [0, 1]",
                            idx, box_idx, score
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    fn empty_output(&self) -> Self::Output {
        TextDetectionOutput::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processors::Point;
    use image::RgbImage;

    #[test]
    fn test_text_detection_task_creation() {
        let task = TextDetectionTask::default();
        assert_eq!(task.task_type(), TaskType::TextDetection);
    }

    #[test]
    fn test_input_validation() {
        let task = TextDetectionTask::default();

        // Empty images should fail
        let empty_input = ImageTaskInput::new(vec![]);
        assert!(task.validate_input(&empty_input).is_err());

        // Valid images should pass
        let valid_input = ImageTaskInput::new(vec![RgbImage::new(100, 100)]);
        assert!(task.validate_input(&valid_input).is_ok());
    }

    #[test]
    fn test_output_validation() {
        let task = TextDetectionTask::default();

        // Matching boxes and scores should pass
        let box1 = BoundingBox::new(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ]);
        let output = TextDetectionOutput {
            boxes: vec![vec![box1]],
            scores: vec![vec![0.95]],
        };
        assert!(task.validate_output(&output).is_ok());

        // Mismatched lengths should fail
        let bad_output = TextDetectionOutput {
            boxes: vec![vec![]],
            scores: vec![vec![0.95]],
        };
        assert!(task.validate_output(&bad_output).is_err());
    }

    #[test]
    fn test_schema() {
        let task = TextDetectionTask::default();
        let schema = task.schema();
        assert_eq!(schema.task_type, TaskType::TextDetection);
        assert!(schema.input_types.contains(&"image".to_string()));
        assert!(schema.output_types.contains(&"text_boxes".to_string()));
    }
}
