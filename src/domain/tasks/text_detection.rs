//! Concrete task implementations for text detection.
//!
//! This module provides the text detection task that locates text regions in images.

use super::validation::ensure_non_empty_images;
use crate::core::OCRError;
use crate::core::traits::task::{ImageTaskInput, Task, TaskSchema, TaskType};
use crate::impl_config_validator;
use crate::processors::{BoundingBox, LimitType};
use crate::utils::ScoreValidator;
use serde::{Deserialize, Serialize};

/// A single text detection result with bounding box and confidence score.
#[derive(Debug, Clone)]
pub struct Detection {
    /// The bounding box polygon coordinates
    pub bbox: BoundingBox,
    /// Confidence score for this detection (0.0 to 1.0)
    pub score: f32,
}

impl Detection {
    /// Creates a new detection.
    pub fn new(bbox: BoundingBox, score: f32) -> Self {
        Self { bbox, score }
    }
}

/// Configuration for text detection task.
///
/// Default values are aligned with PP-StructureV3.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDetectionConfig {
    /// Score threshold for detection (default: 0.3)
    pub score_threshold: f32,
    /// Box threshold for filtering (default: 0.6)
    pub box_threshold: f32,
    /// Unclip ratio for expanding detected regions (default: 1.5)
    pub unclip_ratio: f32,
    /// Maximum candidates to consider (default: 1000)
    pub max_candidates: usize,
    /// Target side length for image resizing (optional)
    pub limit_side_len: Option<u32>,
    /// Limit type for resizing (optional)
    pub limit_type: Option<LimitType>,
    /// Maximum side length to prevent OOM (optional)
    pub max_side_len: Option<u32>,
}

impl Default for TextDetectionConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.3,
            box_threshold: 0.6,
            unclip_ratio: 1.5,
            max_candidates: 1000,
            limit_side_len: None,
            limit_type: None,
            max_side_len: None,
        }
    }
}

impl_config_validator!(TextDetectionConfig {
    score_threshold: range(0.0, 1.0),
    box_threshold: range(0.0, 1.0),
    unclip_ratio: min(0.0),
    max_candidates: min(1),
});

/// Output from text detection task.
#[derive(Debug, Clone)]
pub struct TextDetectionOutput {
    /// Detected text regions per image
    pub detections: Vec<Vec<Detection>>,
}

impl TextDetectionOutput {
    /// Creates an empty text detection output.
    pub fn empty() -> Self {
        Self {
            detections: Vec::new(),
        }
    }

    /// Creates a text detection output with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            detections: Vec::with_capacity(capacity),
        }
    }
}

/// Text detection task implementation.
#[derive(Debug, Default)]
pub struct TextDetectionTask {
    _config: TextDetectionConfig,
}

impl TextDetectionTask {
    /// Creates a new text detection task.
    pub fn new(config: TextDetectionConfig) -> Self {
        Self { _config: config }
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
        ensure_non_empty_images(&input.images, "No images provided for text detection")?;

        Ok(())
    }

    fn validate_output(&self, output: &Self::Output) -> Result<(), OCRError> {
        let validator = ScoreValidator::new_unit_range("score");

        // Validate each image's detections
        for (idx, detections) in output.detections.iter().enumerate() {
            let scores: Vec<f32> = detections.iter().map(|d| d.score).collect();
            validator.validate_scores_with(&scores, |det_idx| {
                format!("Image {}, detection {}", idx, det_idx)
            })?;
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

        // Valid detection should pass
        let box1 = BoundingBox::new(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ]);
        let detection1 = Detection::new(box1, 0.95);
        let output = TextDetectionOutput {
            detections: vec![vec![detection1]],
        };
        assert!(task.validate_output(&output).is_ok());

        // Invalid score should fail
        let box2 = BoundingBox::new(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ]);
        let detection2 = Detection::new(box2, 1.5); // Invalid score > 1.0
        let bad_output = TextDetectionOutput {
            detections: vec![vec![detection2]],
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
