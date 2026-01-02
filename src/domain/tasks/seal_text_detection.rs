//! Seal text detection task definitions.
//!
//! This module provides task definitions for detecting text in seal/stamp images,
//! which often contain curved text arranged in circular patterns.

use super::text_detection::Detection;
use super::validation::ensure_non_empty_images;
use crate::core::OCRError;
use crate::core::traits::TaskDefinition;
use crate::core::traits::task::{ImageTaskInput, Task, TaskSchema, TaskType};
use crate::impl_config_validator;
use crate::utils::ScoreValidator;
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

impl_config_validator!(SealTextDetectionConfig {
    score_threshold: range(0.0, 1.0),
    box_threshold: range(0.0, 1.0),
    unclip_ratio: min(0.0),
    max_candidates: min(1),
});

/// Output from seal text detection models.
///
/// Contains polygon bounding boxes that can handle curved text regions.
#[derive(Debug, Clone)]
pub struct SealTextDetectionOutput {
    /// Detected text regions per image with confidence scores
    pub detections: Vec<Vec<Detection>>,
}

impl SealTextDetectionOutput {
    /// Creates an empty seal text detection output.
    pub fn empty() -> Self {
        Self {
            detections: Vec::new(),
        }
    }

    /// Creates a seal text detection output with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            detections: Vec::with_capacity(capacity),
        }
    }
}

impl TaskDefinition for SealTextDetectionOutput {
    const TASK_NAME: &'static str = "seal_text_detection";
    const TASK_DOC: &'static str =
        "Seal text detection - locating text regions in seal/stamp images";

    fn empty() -> Self {
        SealTextDetectionOutput::empty()
    }
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
        ensure_non_empty_images(
            &input.images,
            "Input images cannot be empty for seal text detection",
        )?;

        Ok(())
    }

    fn validate_output(&self, output: &Self::Output) -> Result<(), OCRError> {
        let validator = ScoreValidator::new_unit_range("score");

        for (batch_idx, detections) in output.detections.iter().enumerate() {
            // Validate score ranges
            let scores: Vec<f32> = detections.iter().map(|d| d.score).collect();
            validator.validate_scores_with(&scores, |det_idx| {
                format!("Batch {}, detection {}", batch_idx, det_idx)
            })?;

            // Validate bounding boxes
            for (det_idx, detection) in detections.iter().enumerate() {
                if detection.bbox.points.is_empty() {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "Batch {}, detection {}: empty bounding box points",
                            batch_idx, det_idx
                        ),
                    });
                }
            }
        }

        Ok(())
    }

    fn empty_output(&self) -> Self::Output {
        SealTextDetectionOutput::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processors::{BoundingBox, Point};
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
        let valid_bbox = BoundingBox {
            points: vec![
                Point { x: 10.0, y: 10.0 },
                Point { x: 50.0, y: 10.0 },
                Point { x: 50.0, y: 30.0 },
                Point { x: 10.0, y: 30.0 },
            ],
        };
        let valid_detection = Detection::new(valid_bbox, 0.95);
        let valid_output = SealTextDetectionOutput {
            detections: vec![vec![valid_detection]],
        };
        assert!(task.validate_output(&valid_output).is_ok());

        // Invalid score
        let invalid_bbox = BoundingBox {
            points: vec![Point { x: 0.0, y: 0.0 }],
        };
        let invalid_detection = Detection::new(invalid_bbox, 1.5); // Score > 1.0
        let invalid_score_output = SealTextDetectionOutput {
            detections: vec![vec![invalid_detection]],
        };
        assert!(task.validate_output(&invalid_score_output).is_err());

        // Empty bounding box points
        let empty_bbox = BoundingBox { points: vec![] };
        let empty_bbox_detection = Detection::new(empty_bbox, 0.95);
        let empty_bbox_output = SealTextDetectionOutput {
            detections: vec![vec![empty_bbox_detection]],
        };
        assert!(task.validate_output(&empty_bbox_output).is_err());
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
