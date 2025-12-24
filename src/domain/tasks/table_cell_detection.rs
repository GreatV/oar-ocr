//! Table cell detection task definitions.
//!
//! This module provides the task configuration and output structures for
//! detecting table cells using object detection models (e.g., RT-DETR).

use super::validation::ensure_non_empty_images;
use crate::core::OCRError;
use crate::core::traits::task::{ImageTaskInput, Task, TaskSchema, TaskType};
use crate::impl_config_validator;
use crate::processors::BoundingBox;
use crate::utils::{ScoreValidator, validate_max_value};
use serde::{Deserialize, Serialize};

/// Configuration for table cell detection.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TableCellDetectionConfig {
    /// Score threshold for detections (default: 0.3).
    pub score_threshold: f32,
    /// Maximum number of cells to keep per image (default: 300).
    pub max_cells: usize,
}

impl Default for TableCellDetectionConfig {
    fn default() -> Self {
        Self {
            // Table cell detection threshold defaults to 0.3.
            score_threshold: 0.3,
            max_cells: 300,
        }
    }
}

impl_config_validator!(TableCellDetectionConfig {
    score_threshold: range(0.0, 1.0),
    max_cells: min(1),
});

/// A detected table cell.
#[derive(Debug, Clone)]
pub struct TableCell {
    /// Bounding box of the detected cell.
    pub bbox: BoundingBox,
    /// Confidence score associated with the detection.
    pub score: f32,
    /// Optional label for the cell (e.g., "cell").
    pub label: String,
}

/// Output from the table cell detection task.
#[derive(Debug, Clone)]
pub struct TableCellDetectionOutput {
    /// Detected cells per image, preserving input order.
    pub cells: Vec<Vec<TableCell>>,
}

impl TableCellDetectionOutput {
    /// Creates an empty table cell detection output.
    pub fn empty() -> Self {
        Self { cells: Vec::new() }
    }

    /// Creates a table cell detection output with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            cells: Vec::with_capacity(capacity),
        }
    }
}

/// Table cell detection task implementation.
#[derive(Debug, Default)]
pub struct TableCellDetectionTask {
    config: TableCellDetectionConfig,
}

impl TableCellDetectionTask {
    /// Creates a new table cell detection task.
    pub fn new(config: TableCellDetectionConfig) -> Self {
        Self { config }
    }
}

impl Task for TableCellDetectionTask {
    type Config = TableCellDetectionConfig;
    type Input = ImageTaskInput;
    type Output = TableCellDetectionOutput;

    fn task_type(&self) -> TaskType {
        TaskType::TableCellDetection
    }

    fn schema(&self) -> TaskSchema {
        TaskSchema::new(
            TaskType::TableCellDetection,
            vec!["image".to_string()],
            vec!["table_cells".to_string()],
        )
    }

    fn validate_input(&self, input: &Self::Input) -> Result<(), OCRError> {
        ensure_non_empty_images(&input.images, "No images provided for table cell detection")?;

        Ok(())
    }

    fn validate_output(&self, output: &Self::Output) -> Result<(), OCRError> {
        let validator = ScoreValidator::new_unit_range("score");

        for (idx, cells) in output.cells.iter().enumerate() {
            // Validate cell count
            validate_max_value(
                cells.len(),
                self.config.max_cells,
                "cell count",
                &format!("Image {}", idx),
            )?;

            // Validate scores
            let scores: Vec<f32> = cells.iter().map(|c| c.score).collect();
            validator.validate_scores_with(&scores, |cell_idx| {
                format!("Image {}, cell {}", idx, cell_idx)
            })?;
        }

        Ok(())
    }

    fn empty_output(&self) -> Self::Output {
        TableCellDetectionOutput::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processors::Point;
    use image::RgbImage;

    #[test]
    fn test_task_creation() {
        let task = TableCellDetectionTask::default();
        assert_eq!(task.task_type(), TaskType::TableCellDetection);
    }

    #[test]
    fn test_input_validation() {
        let task = TableCellDetectionTask::default();
        let empty_input = ImageTaskInput::new(vec![]);
        assert!(task.validate_input(&empty_input).is_err());

        let valid_input = ImageTaskInput::new(vec![RgbImage::new(100, 100)]);
        assert!(task.validate_input(&valid_input).is_ok());
    }

    #[test]
    fn test_output_validation() {
        let task = TableCellDetectionTask::default();
        let bbox = BoundingBox::new(vec![
            Point::new(0.0, 0.0),
            Point::new(10.0, 0.0),
            Point::new(10.0, 10.0),
            Point::new(0.0, 10.0),
        ]);
        let cell = TableCell {
            bbox,
            score: 0.95,
            label: "cell".to_string(),
        };
        let output = TableCellDetectionOutput {
            cells: vec![vec![cell]],
        };
        assert!(task.validate_output(&output).is_ok());
    }

    #[test]
    fn test_schema() {
        let task = TableCellDetectionTask::default();
        let schema = task.schema();
        assert_eq!(schema.task_type, TaskType::TableCellDetection);
        assert!(schema.input_types.contains(&"image".to_string()));
        assert!(schema.output_types.contains(&"table_cells".to_string()));
    }
}
