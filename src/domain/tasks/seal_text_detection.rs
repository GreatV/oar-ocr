//! Seal text detection task definitions.
//!
//! This module provides task definitions for detecting text in seal/stamp images,
//! which often contain curved text arranged in circular patterns.

use crate::core::OCRError;
use crate::core::traits::task::{ImageTaskInput, Task, TaskSchema, TaskType};
use crate::processors::BoundingBox;
use serde::{Deserialize, Serialize};

/// Configuration for seal text detection models.
///
/// These parameters control how text regions are extracted from seal images.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SealTextDetectionConfig {
    /// Pixel-level threshold for text detection (0.2 default)
    pub score_threshold: f32,
    /// Box-level threshold for filtering detections (0.6 default)
    pub box_threshold: f32,
    /// Expansion ratio for detected regions using Vatti clipping (0.5 default)
    pub unclip_ratio: f32,
    /// Maximum number of candidate detections (1000 default)
    pub max_candidates: usize,
}

impl Default for SealTextDetectionConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.2, // Lower than regular detection
            box_threshold: 0.6,   // From official config
            unclip_ratio: 0.5,    // Much lower than regular detection
            max_candidates: 1000,
        }
    }
}

/// Output from seal text detection models.
///
/// Contains polygon bounding boxes that can handle curved text regions.
#[derive(Debug, Clone)]
pub struct SealTextDetectionOutput {
    /// Detected text region polygons per image
    pub boxes: Vec<Vec<BoundingBox>>,
    /// Confidence scores for each detection
    pub scores: Vec<Vec<f32>>,
}

/// Seal text detection task.
///
/// This task is specialized for detecting text in seal and stamp images,
/// where text often follows curved paths along circular borders.
#[derive(Debug, Default)]
pub struct SealTextDetectionTask {
    pub config: SealTextDetectionConfig,
}

impl SealTextDetectionTask {
    /// Creates a new seal text detection task with default config.
    pub fn new() -> Self {
        Self::default()
    }

    /// Creates a new seal text detection task with custom config.
    pub fn with_config(config: SealTextDetectionConfig) -> Self {
        Self { config }
    }
}

impl Task for SealTextDetectionTask {
    type Config = SealTextDetectionConfig;
    type Input = ImageTaskInput;
    type Output = SealTextDetectionOutput;

    fn task_type(&self) -> TaskType {
        TaskType::SealTextDetection
    }

    fn schema(&self) -> TaskSchema {
        TaskSchema::new(
            self.task_type(),
            vec!["image".to_string()],
            vec!["seal_text_boxes".to_string(), "scores".to_string()],
        )
    }

    fn validate_input(&self, input: &Self::Input) -> Result<(), OCRError> {
        if input.images.is_empty() {
            return Err(OCRError::InvalidInput {
                message: "Input images cannot be empty for seal text detection".to_string(),
            });
        }

        for (idx, image) in input.images.iter().enumerate() {
            if image.width() == 0 || image.height() == 0 {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Image at index {} has invalid dimensions: {}x{}",
                        idx,
                        image.width(),
                        image.height()
                    ),
                });
            }
        }

        Ok(())
    }

    fn validate_output(&self, output: &Self::Output) -> Result<(), OCRError> {
        if output.boxes.len() != output.scores.len() {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "Mismatch between number of box sets ({}) and score sets ({})",
                    output.boxes.len(),
                    output.scores.len()
                ),
            });
        }

        for (batch_idx, (boxes, scores)) in output.boxes.iter().zip(&output.scores).enumerate() {
            if boxes.len() != scores.len() {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Batch {}: mismatch between number of boxes ({}) and scores ({})",
                        batch_idx,
                        boxes.len(),
                        scores.len()
                    ),
                });
            }

            // Validate score ranges
            for (box_idx, score) in scores.iter().enumerate() {
                if !(*score >= 0.0 && *score <= 1.0) {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Batch {}, box {}: invalid score value {}. Must be in range [0, 1]",
                            batch_idx, box_idx, score
                        ),
                    });
                }
            }

            // Validate bounding boxes
            for (box_idx, bbox) in boxes.iter().enumerate() {
                if bbox.points.is_empty() {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Batch {}, box {}: empty bounding box points",
                            batch_idx, box_idx
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    fn empty_output(&self) -> Self::Output {
        SealTextDetectionOutput {
            boxes: Vec::new(),
            scores: Vec::new(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processors::Point;
    use image::RgbImage;

    #[test]
    fn test_seal_text_detection_task_creation() {
        let task = SealTextDetectionTask::new();
        assert_eq!(task.task_type(), TaskType::SealTextDetection);
    }

    #[test]
    fn test_input_validation() {
        let task = SealTextDetectionTask::new();

        // Test empty input
        let empty_input = ImageTaskInput::new(vec![]);
        assert!(task.validate_input(&empty_input).is_err());

        // Test valid input
        let valid_input = ImageTaskInput::new(vec![RgbImage::new(100, 100)]);
        assert!(task.validate_input(&valid_input).is_ok());

        // Test zero-dimension image
        let invalid_input = ImageTaskInput::new(vec![RgbImage::new(0, 100)]);
        assert!(task.validate_input(&invalid_input).is_err());
    }

    #[test]
    fn test_output_validation() {
        let task = SealTextDetectionTask::new();

        // Valid output
        let valid_output = SealTextDetectionOutput {
            boxes: vec![vec![BoundingBox {
                points: vec![
                    Point { x: 10.0, y: 10.0 },
                    Point { x: 50.0, y: 10.0 },
                    Point { x: 50.0, y: 30.0 },
                    Point { x: 10.0, y: 30.0 },
                ],
            }]],
            scores: vec![vec![0.95]],
        };
        assert!(task.validate_output(&valid_output).is_ok());

        // Mismatched lengths
        let invalid_output = SealTextDetectionOutput {
            boxes: vec![vec![]],
            scores: vec![vec![], vec![]],
        };
        assert!(task.validate_output(&invalid_output).is_err());

        // Invalid score
        let invalid_score_output = SealTextDetectionOutput {
            boxes: vec![vec![BoundingBox {
                points: vec![Point { x: 0.0, y: 0.0 }],
            }]],
            scores: vec![vec![1.5]], // Score > 1.0
        };
        assert!(task.validate_output(&invalid_score_output).is_err());
    }

    #[test]
    fn test_schema() {
        let task = SealTextDetectionTask::new();
        let schema = task.schema();
        assert_eq!(schema.task_type, TaskType::SealTextDetection);
        assert_eq!(schema.input_types, vec!["image"]);
        assert_eq!(schema.output_types, vec!["seal_text_boxes", "scores"]);
    }
}
