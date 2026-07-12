//! Concrete task implementations for text recognition.
//!
//! This module provides the text recognition task that converts text regions to strings.

use super::validation::ensure_non_empty_images;
use crate::ConfigValidator;
use crate::core::OCRError;
use crate::core::traits::TaskDefinition;
use crate::core::traits::task::{ImageTaskInput, Task, TaskSchema, TaskType};
use crate::utils::ScoreValidator;
use serde::{Deserialize, Serialize};
use unicode_bidi::BidiInfo;

/// Configuration for text recognition task.
///
/// Default values are aligned with PP-StructureV3.
#[derive(Debug, Clone, Serialize, Deserialize, ConfigValidator)]
pub struct TextRecognitionConfig {
    /// Score threshold for recognition (default: 0.0, no filtering)
    #[validate(range(min = 0.0, max = 1.0))]
    pub score_threshold: f32,
}

impl Default for TextRecognitionConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.0,
        }
    }
}

fn reorder_bidi_line(line: &str) -> String {
    let bidi_info = BidiInfo::new(line, None);
    let Some(para) = bidi_info.paragraphs.first() else {
        return line.to_string();
    };

    bidi_info
        .reorder_line(para, para.range.clone())
        .into_owned()
}

fn reorder_bidi_text(text: &str) -> (String, bool) {
    let mut out = String::with_capacity(text.len());
    let mut changed_order = false;

    for segment in text.split_inclusive('\n') {
        if let Some(line) = segment.strip_suffix('\n') {
            let converted = reorder_bidi_line(line);
            changed_order |= converted != line;
            out.push_str(&converted);
            out.push('\n');
        } else {
            let converted = reorder_bidi_line(segment);
            changed_order |= converted != segment;
            out.push_str(&converted);
        }
    }

    (out, changed_order)
}

/// Applies Unicode bidi post-processing to decoded OCR text.
pub fn postprocess_text_direction(text: String) -> String {
    postprocess_text_direction_with_order_change(text).0
}

/// Applies Unicode bidi post-processing and reports whether character order changed.
pub fn postprocess_text_direction_with_order_change(text: String) -> (String, bool) {
    reorder_bidi_text(&text)
}

/// Output from text recognition task.
#[derive(Debug, Clone)]
pub struct TextRecognitionOutput {
    /// Recognized text strings
    pub texts: Vec<String>,
    /// Confidence scores for each text
    pub scores: Vec<f32>,
    /// Character/word positions within each text line (optional)
    /// Each inner vector contains normalized x-positions (0.0-1.0) for characters
    /// Only populated when word box detection is enabled
    pub char_positions: Vec<Vec<f32>>,
    /// Column indices for each character in the CTC output
    /// Used for accurate word box generation with compatible approach
    pub char_col_indices: Vec<Vec<usize>>,
    /// Total number of columns (sequence length) in the CTC output for each text line
    pub sequence_lengths: Vec<usize>,
}

impl TextRecognitionOutput {
    /// Creates an empty text recognition output.
    pub fn empty() -> Self {
        Self {
            texts: Vec::new(),
            scores: Vec::new(),
            char_positions: Vec::new(),
            char_col_indices: Vec::new(),
            sequence_lengths: Vec::new(),
        }
    }

    /// Creates a text recognition output with the given capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            texts: Vec::with_capacity(capacity),
            scores: Vec::with_capacity(capacity),
            char_positions: Vec::with_capacity(capacity),
            char_col_indices: Vec::with_capacity(capacity),
            sequence_lengths: Vec::with_capacity(capacity),
        }
    }
}

impl Default for TextRecognitionOutput {
    fn default() -> Self {
        Self::empty()
    }
}

impl TaskDefinition for TextRecognitionOutput {
    const TASK_NAME: &'static str = "text_recognition";
    const TASK_DOC: &'static str = "Text recognition - converting text regions to strings";

    fn empty() -> Self {
        TextRecognitionOutput::empty()
    }
}

/// Text recognition task implementation.
#[derive(Debug, Default)]
pub struct TextRecognitionTask;

impl TextRecognitionTask {
    /// Creates a new text recognition task.
    pub fn new(_config: TextRecognitionConfig) -> Self {
        Self
    }
}

impl Task for TextRecognitionTask {
    type Config = TextRecognitionConfig;
    type Input = ImageTaskInput;
    type Output = TextRecognitionOutput;

    fn task_type(&self) -> TaskType {
        TaskType::TextRecognition
    }

    fn schema(&self) -> TaskSchema {
        TaskSchema::new(
            TaskType::TextRecognition,
            vec!["text_boxes".to_string()],
            vec!["text_strings".to_string(), "scores".to_string()],
        )
    }

    fn validate_input(&self, input: &Self::Input) -> Result<(), OCRError> {
        ensure_non_empty_images(&input.images, "No images provided for text recognition")?;

        Ok(())
    }

    fn validate_output(&self, output: &Self::Output) -> Result<(), OCRError> {
        // Validate score ranges
        let validator = ScoreValidator::new_unit_range("score");
        validator.validate_scores_with(&output.scores, |idx| format!("Text {}", idx))?;

        Ok(())
    }

    fn empty_output(&self) -> Self::Output {
        TextRecognitionOutput::empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::RgbImage;

    #[test]
    fn test_text_recognition_task_creation() {
        let task = TextRecognitionTask;
        assert_eq!(task.task_type(), TaskType::TextRecognition);
    }

    #[test]
    fn test_input_validation() {
        let task = TextRecognitionTask;

        // Empty images should fail
        let empty_input = ImageTaskInput::new(vec![]);
        assert!(task.validate_input(&empty_input).is_err());

        // Valid images should pass
        let valid_input = ImageTaskInput::new(vec![RgbImage::new(100, 32)]);
        assert!(task.validate_input(&valid_input).is_ok());
    }

    #[test]
    fn test_output_validation() {
        let task = TextRecognitionTask;

        // Matching texts and scores should pass
        let output = TextRecognitionOutput {
            texts: vec!["Hello".to_string()],
            scores: vec![0.95],
            ..Default::default()
        };
        assert!(task.validate_output(&output).is_ok());

        // Invalid score should fail
        let bad_score = TextRecognitionOutput {
            texts: vec!["Hello".to_string()],
            scores: vec![1.5],
            ..Default::default()
        };
        assert!(task.validate_output(&bad_score).is_err());
    }

    #[test]
    fn test_schema() {
        let task = TextRecognitionTask;
        let schema = task.schema();
        assert_eq!(schema.task_type, TaskType::TextRecognition);
        assert!(schema.input_types.contains(&"text_boxes".to_string()));
        assert!(schema.output_types.contains(&"text_strings".to_string()));
    }

    #[test]
    fn bidi_postprocess_leaves_ltr_text_unchanged() {
        assert_eq!(
            postprocess_text_direction("hello 123".to_string()),
            "hello 123"
        );
    }

    #[test]
    fn bidi_postprocess_reports_order_changes() {
        assert_eq!(
            postprocess_text_direction_with_order_change("hello 123".to_string()),
            ("hello 123".to_string(), false)
        );
        assert_eq!(
            postprocess_text_direction_with_order_change("ابحرم".to_string()),
            ("مرحبا".to_string(), true)
        );
    }
}
