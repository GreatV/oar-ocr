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
use unicode_bidi::{BidiClass, bidi_class};

/// Reading direction for recognized text post-processing.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TextDirection {
    /// Keep decoded text in the model's left-to-right sequence order.
    #[default]
    Ltr,
    /// Convert visual right-to-left OCR output into logical string order.
    Rtl,
    /// Convert only when decoded text contains right-to-left characters.
    Auto,
}

/// Configuration for text recognition task.
///
/// Default values are aligned with PP-StructureV3.
#[derive(Debug, Clone, Serialize, Deserialize, ConfigValidator)]
pub struct TextRecognitionConfig {
    /// Score threshold for recognition (default: 0.0, no filtering)
    #[validate(range(min = 0.0, max = 1.0))]
    pub score_threshold: f32,
    /// Reading direction used for text post-processing.
    #[serde(default)]
    pub text_direction: TextDirection,
}

impl Default for TextRecognitionConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.0,
            text_direction: TextDirection::Ltr,
        }
    }
}

fn is_rtl_char(c: char) -> bool {
    matches!(
        bidi_class(c),
        BidiClass::AL | BidiClass::R | BidiClass::RLE | BidiClass::RLI | BidiClass::RLO
    )
}

fn is_combining_mark(c: char) -> bool {
    bidi_class(c) == BidiClass::NSM
}

fn is_ltr_token_char(c: char) -> bool {
    matches!(
        bidi_class(c),
        BidiClass::L
            | BidiClass::EN
            | BidiClass::AN
            | BidiClass::ES
            | BidiClass::ET
            | BidiClass::CS
    ) || "._:/%+-#@&".contains(c)
}

fn preserve_ltr_phrase_order(chars: &mut [char]) {
    let mut i = 0;
    while i < chars.len() {
        if !is_ltr_token_char(chars[i]) {
            i += 1;
            continue;
        }

        let start = i;
        i += 1;
        while i < chars.len() && (is_ltr_token_char(chars[i]) || chars[i].is_whitespace()) {
            i += 1;
        }
        let mut end = i;
        while end > start && chars[end - 1].is_whitespace() {
            end -= 1;
        }
        chars[start..end].reverse();
    }
}

fn normalize_leading_combining_marks(chars: Vec<char>) -> Vec<char> {
    let mut normalized = Vec::with_capacity(chars.len());
    let mut pending_marks = Vec::new();

    for c in chars {
        if is_combining_mark(c) {
            pending_marks.push(c);
            continue;
        }

        normalized.push(c);
        if is_rtl_char(c) && !pending_marks.is_empty() {
            normalized.append(&mut pending_marks);
        } else if !pending_marks.is_empty() {
            let insertion = normalized.len().saturating_sub(1);
            normalized.splice(insertion..insertion, pending_marks.drain(..));
        }
    }

    normalized.extend(pending_marks);
    normalized
}

fn visual_rtl_line_to_logical(line: &str) -> String {
    if !line.chars().any(is_rtl_char) {
        return line.to_string();
    }

    // PaddleOCR/PaddleX Arabic CTC recognizers can expose visual RTL order.
    // Convert to logical order while using Unicode bidi classes for RTL,
    // non-spacing marks, and embedded LTR/number phrases.
    let mut chars: Vec<char> = line.chars().rev().collect();
    preserve_ltr_phrase_order(&mut chars);
    normalize_leading_combining_marks(chars)
        .into_iter()
        .collect()
}

fn visual_rtl_to_logical(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    for segment in text.split_inclusive('\n') {
        if let Some(line) = segment.strip_suffix('\n') {
            out.push_str(&visual_rtl_line_to_logical(line));
            out.push('\n');
        } else {
            out.push_str(&visual_rtl_line_to_logical(segment));
        }
    }
    out
}

/// Applies configured reading-direction post-processing to decoded OCR text.
pub fn postprocess_text_direction(text: String, direction: TextDirection) -> String {
    match direction {
        TextDirection::Ltr => text,
        TextDirection::Rtl => visual_rtl_to_logical(&text),
        TextDirection::Auto if text.chars().any(is_rtl_char) => visual_rtl_to_logical(&text),
        TextDirection::Auto => text,
    }
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
    fn rtl_postprocess_reverses_visual_arabic_order() {
        assert_eq!(
            postprocess_text_direction(
                "دحاو نآ يف لوؤسمو يعاديإ لمع مجرتملا لمع".to_string(),
                TextDirection::Rtl,
            ),
            "عمل المترجم عمل إيداعي ومسؤول في آن واحد"
        );
    }

    #[test]
    fn rtl_postprocess_preserves_ltr_token_order() {
        assert_eq!(
            postprocess_text_direction("abc 123 ابحرم".to_string(), TextDirection::Rtl),
            "مرحبا abc 123"
        );
    }

    #[test]
    fn ltr_postprocess_leaves_text_unchanged() {
        let text = "دحاو نآ يف".to_string();
        assert_eq!(
            postprocess_text_direction(text.clone(), TextDirection::Ltr),
            text
        );
    }

    #[test]
    fn auto_postprocess_only_reverses_rtl_text() {
        assert_eq!(
            postprocess_text_direction("دحاو نآ يف".to_string(), TextDirection::Auto),
            "في آن واحد"
        );
        assert_eq!(
            postprocess_text_direction("hello 123".to_string(), TextDirection::Auto),
            "hello 123"
        );
    }

    #[test]
    fn rtl_postprocess_keeps_combining_marks_on_rtl_base() {
        assert_eq!(
            postprocess_text_direction("لكشي ّراطإ".to_string(), TextDirection::Rtl),
            "إطارّ يشكل"
        );
    }

    #[test]
    fn rtl_postprocess_preserves_paired_punctuation() {
        assert_eq!(
            postprocess_text_direction(")OCR 2026( ابحرم".to_string(), TextDirection::Rtl),
            "مرحبا (OCR 2026)"
        );
    }

    #[test]
    fn rtl_postprocess_preserves_arabic_indic_number_order() {
        assert_eq!(
            postprocess_text_direction("١٢٣ ابحرم".to_string(), TextDirection::Rtl),
            "مرحبا ١٢٣"
        );
    }
}
