//! Task trait definitions for the OCR pipeline.
//!
//! This module defines the `Task` trait and related types that represent
//! different OCR tasks (text detection, recognition, layout analysis, etc.).
//! Tasks define input/output schemas and execution contracts.

use crate::core::OCRError;
use image::RgbImage;
use serde::{Deserialize, Serialize};
use std::fmt::Debug;

/// Represents the type of OCR task being performed.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum TaskType {
    /// Text detection - locating text regions in images
    TextDetection,
    /// Text recognition - converting text regions to strings
    TextRecognition,
    /// Document orientation classification
    DocumentOrientation,
    /// Text line orientation classification
    TextLineOrientation,
    /// Document rectification/unwarp
    DocumentRectification,
    /// Layout detection/analysis
    LayoutDetection,
    /// Table cell detection - locating cells within table regions
    TableCellDetection,
    /// Formula recognition - converting mathematical formulas to LaTeX
    FormulaRecognition,
    /// Seal text detection - locating text regions in seal/stamp images
    SealTextDetection,
    /// Table classification - classifying table images as wired or wireless
    TableClassification,
    /// Table structure recognition - recognizing table structure as HTML with bboxes
    TableStructureRecognition,
}

impl TaskType {
    /// Returns a human-readable name for the task type.
    pub fn name(&self) -> &'static str {
        match self {
            TaskType::TextDetection => "text_detection",
            TaskType::TextRecognition => "text_recognition",
            TaskType::DocumentOrientation => "document_orientation",
            TaskType::TextLineOrientation => "text_line_orientation",
            TaskType::DocumentRectification => "document_rectification",
            TaskType::LayoutDetection => "layout_detection",
            TaskType::TableCellDetection => "table_cell_detection",
            TaskType::FormulaRecognition => "formula_recognition",
            TaskType::SealTextDetection => "seal_text_detection",
            TaskType::TableClassification => "table_classification",
            TaskType::TableStructureRecognition => "table_structure_recognition",
        }
    }
}

/// Schema definition for task inputs and outputs.
///
/// This allows for validation that models produce outputs compatible
/// with what downstream tasks expect.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TaskSchema {
    /// Task type
    pub task_type: TaskType,
    /// Expected input types (e.g., "image", "text_boxes")
    pub input_types: Vec<String>,
    /// Expected output types (e.g., "text_boxes", "text_strings")
    pub output_types: Vec<String>,
    /// Optional metadata schema
    pub metadata_schema: Option<String>,
}

impl TaskSchema {
    /// Creates a new task schema.
    pub fn new(task_type: TaskType, input_types: Vec<String>, output_types: Vec<String>) -> Self {
        Self {
            task_type,
            input_types,
            output_types,
            metadata_schema: None,
        }
    }

    /// Validates that this schema is compatible with another schema.
    ///
    /// Returns true if the output types of this schema match the input types
    /// of the target schema.
    pub fn is_compatible_with(&self, target: &TaskSchema) -> bool {
        // Check if any of our output types match any of target's input types
        self.output_types
            .iter()
            .any(|output| target.input_types.contains(output))
    }
}

/// Core trait for OCR tasks.
///
/// Tasks represent distinct operations in the OCR pipeline (detection, recognition, etc.).
/// Each task defines its input/output schema and can be executed with various model adapters.
pub trait Task: Send + Sync + Debug {
    /// Configuration type for this task
    type Config: Send + Sync + Debug + Clone;

    /// Input type for this task
    type Input: Send + Sync + Debug;

    /// Output type from this task
    type Output: Send + Sync + Debug;

    /// Returns the task type identifier.
    fn task_type(&self) -> TaskType;

    /// Returns the schema defining inputs and outputs for this task.
    fn schema(&self) -> TaskSchema;

    /// Validates that the given input is suitable for this task.
    ///
    /// # Arguments
    ///
    /// * `input` - The input to validate
    ///
    /// # Returns
    ///
    /// Result indicating success or validation error
    fn validate_input(&self, input: &Self::Input) -> Result<(), OCRError>;

    /// Validates that the given output matches the expected schema.
    ///
    /// # Arguments
    ///
    /// * `output` - The output to validate
    ///
    /// # Returns
    ///
    /// Result indicating success or validation error
    fn validate_output(&self, output: &Self::Output) -> Result<(), OCRError>;

    /// Returns an empty output instance for when no valid results are produced.
    fn empty_output(&self) -> Self::Output;

    /// Returns a human-readable description of this task.
    fn description(&self) -> String {
        format!("Task: {}", self.task_type().name())
    }
}

/// A task runner that executes tasks using a model adapter.
///
/// This struct coordinates the execution of a task with a specific model,
/// handling validation and error propagation.
#[derive(Debug)]
pub struct TaskRunner<T: Task> {
    /// The task to execute
    task: T,
    /// Configuration for the task
    config: T::Config,
}

impl<T: Task> TaskRunner<T> {
    /// Creates a new task runner.
    ///
    /// # Arguments
    ///
    /// * `task` - The task to execute
    /// * `config` - Configuration for the task
    pub fn new(task: T, config: T::Config) -> Self {
        Self { task, config }
    }

    /// Returns a reference to the task.
    pub fn task(&self) -> &T {
        &self.task
    }

    /// Returns a reference to the configuration.
    pub fn config(&self) -> &T::Config {
        &self.config
    }

    /// Returns the task type.
    pub fn task_type(&self) -> TaskType {
        self.task.task_type()
    }

    /// Validates input before execution.
    pub fn validate_input(&self, input: &T::Input) -> Result<(), OCRError> {
        self.task.validate_input(input)
    }

    /// Validates output after execution.
    pub fn validate_output(&self, output: &T::Output) -> Result<(), OCRError> {
        self.task.validate_output(output)
    }
}

/// Common input type for image-based tasks.
#[derive(Debug, Clone)]
pub struct ImageTaskInput {
    /// Input images
    pub images: Vec<RgbImage>,
    /// Optional metadata per image
    pub metadata: Vec<Option<String>>,
}

impl ImageTaskInput {
    /// Creates a new image task input.
    pub fn new(images: Vec<RgbImage>) -> Self {
        let count = images.len();
        Self {
            images,
            metadata: vec![None; count],
        }
    }

    /// Creates a new image task input with metadata.
    pub fn with_metadata(images: Vec<RgbImage>, metadata: Vec<Option<String>>) -> Self {
        Self { images, metadata }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_type_name() {
        assert_eq!(TaskType::TextDetection.name(), "text_detection");
        assert_eq!(TaskType::TextRecognition.name(), "text_recognition");
    }

    #[test]
    fn test_schema_compatibility() {
        let detection_schema = TaskSchema::new(
            TaskType::TextDetection,
            vec!["image".to_string()],
            vec!["text_boxes".to_string()],
        );

        let recognition_schema = TaskSchema::new(
            TaskType::TextRecognition,
            vec!["text_boxes".to_string()],
            vec!["text_strings".to_string()],
        );

        // Detection output (text_boxes) should be compatible with recognition input (text_boxes)
        assert!(detection_schema.is_compatible_with(&recognition_schema));

        // Recognition output (text_strings) is not compatible with detection input (image)
        assert!(!recognition_schema.is_compatible_with(&detection_schema));
    }

    #[test]
    fn test_image_task_input_creation() {
        let images = vec![RgbImage::new(100, 100), RgbImage::new(200, 200)];
        let input = ImageTaskInput::new(images.clone());

        assert_eq!(input.images.len(), 2);
        assert_eq!(input.metadata.len(), 2);
        assert!(input.metadata.iter().all(|m| m.is_none()));
    }
}
