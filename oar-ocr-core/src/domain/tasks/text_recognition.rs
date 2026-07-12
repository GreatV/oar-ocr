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
    /// Convert only when a decoded line's first strong directional character is right-to-left.
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
}

impl Default for TextRecognitionConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.0,
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

fn is_strong_ltr_char(c: char) -> bool {
    bidi_class(c) == BidiClass::L
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
            | BidiClass::NSM
    ) || "._:/%+-#@&'’`".contains(c)
}

fn matching_open_bracket(close: char) -> Option<char> {
    match close {
        ')' => Some('('),
        ']' => Some('['),
        '}' => Some('{'),
        '>' => Some('<'),
        _ => None,
    }
}

fn is_open_bracket(c: char) -> bool {
    matches!(c, '(' | '[' | '{' | '<')
}

fn preserve_ltr_phrase_order(chars: &mut [char]) {
    let mut i = 0;
    while i < chars.len() {
        if !is_ltr_token_char(chars[i]) {
            i += 1;
            continue;
        }

        let mut start = i;
        i += 1;
        while i < chars.len()
            && (is_ltr_token_char(chars[i])
                || chars[i].is_whitespace()
                || (is_open_bracket(chars[i])
                    && chars
                        .get(i + 1)
                        .is_some_and(|next| is_ltr_token_char(*next))))
        {
            i += 1;
        }
        let mut end = i;
        while end > start && chars[end - 1].is_whitespace() {
            end -= 1;
        }

        if start > 0
            && let Some(open) = matching_open_bracket(chars[start - 1])
            && chars[start..end].contains(&open)
        {
            start -= 1;
        }

        chars[start..end].reverse();
    }
}

fn normalize_leading_combining_marks(chars: Vec<char>) -> Vec<char> {
    let mut normalized: Vec<char> = Vec::with_capacity(chars.len());
    let mut pending_marks = Vec::new();

    for (idx, c) in chars.iter().copied().enumerate() {
        if is_combining_mark(c) {
            let last_base = normalized
                .iter()
                .rev()
                .copied()
                .find(|last| !is_combining_mark(*last));
            let next_base = chars
                .iter()
                .skip(idx + 1)
                .copied()
                .find(|next| !is_combining_mark(*next));
            if last_base.is_some_and(is_ltr_token_char)
                || next_base.is_none_or(|next| next.is_whitespace())
            {
                normalized.push(c);
                continue;
            }
            pending_marks.push(c);
            continue;
        }

        normalized.push(c);
        if !pending_marks.is_empty() && !c.is_whitespace() {
            normalized.append(&mut pending_marks);
        }
    }

    normalized.extend(pending_marks);
    normalized
}

fn visual_rtl_line_to_logical(line: &str) -> String {
    // PaddleOCR/PaddleX Arabic CTC recognizers can expose visual RTL order.
    // Convert to logical order while using Unicode bidi classes for RTL,
    // non-spacing marks, and embedded LTR/number phrases.
    let mut chars: Vec<char> = line.chars().rev().collect();
    preserve_ltr_phrase_order(&mut chars);
    normalize_leading_combining_marks(chars)
        .into_iter()
        .collect()
}

fn has_rtl_char(text: &str) -> bool {
    text.chars().any(is_rtl_char)
}

fn has_rtl_visual_line_end(line: &str) -> bool {
    for c in line.chars().rev() {
        if c.is_whitespace() || is_combining_mark(c) {
            continue;
        }
        if is_rtl_char(c) {
            return true;
        }
        if is_strong_ltr_char(c) || is_ltr_token_char(c) {
            return false;
        }
    }
    false
}

fn should_convert_visual_rtl_line(line: &str, direction: TextDirection) -> bool {
    match direction {
        TextDirection::Ltr => false,
        TextDirection::Rtl => has_rtl_char(line),
        TextDirection::Auto => has_rtl_visual_line_end(line),
    }
}

fn visual_rtl_to_logical(text: &str, direction: TextDirection) -> (String, bool) {
    let mut out = String::with_capacity(text.len());
    let mut changed_order = false;

    for segment in text.split_inclusive('\n') {
        if let Some(line) = segment.strip_suffix('\n') {
            if should_convert_visual_rtl_line(line, direction) {
                let converted = visual_rtl_line_to_logical(line);
                changed_order |= converted != line;
                out.push_str(&converted);
            } else {
                out.push_str(line);
            }
            out.push('\n');
        } else if should_convert_visual_rtl_line(segment, direction) {
            let converted = visual_rtl_line_to_logical(segment);
            changed_order |= converted != segment;
            out.push_str(&converted);
        } else {
            out.push_str(segment);
        }
    }

    (out, changed_order)
}

/// Applies configured reading-direction post-processing to decoded OCR text.
pub fn postprocess_text_direction(text: String, direction: TextDirection) -> String {
    postprocess_text_direction_with_order_change(text, direction).0
}

/// Applies reading-direction post-processing and reports whether character order changed.
pub fn postprocess_text_direction_with_order_change(
    text: String,
    direction: TextDirection,
) -> (String, bool) {
    match direction {
        TextDirection::Ltr => (text, false),
        TextDirection::Rtl | TextDirection::Auto => visual_rtl_to_logical(&text, direction),
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
    fn auto_postprocess_leaves_ltr_base_mixed_text_unchanged() {
        assert_eq!(
            postprocess_text_direction("Issue رقم 123".to_string(), TextDirection::Auto),
            "Issue رقم 123"
        );
    }

    #[test]
    fn auto_postprocess_detects_rtl_from_visual_line_end() {
        assert_eq!(
            postprocess_text_direction("abc 123 ابحرم".to_string(), TextDirection::Auto),
            "مرحبا abc 123"
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
    fn rtl_postprocess_keeps_multiple_combining_marks_on_each_rtl_base() {
        assert_eq!(
            postprocess_text_direction("بَتِكُ".to_string(), TextDirection::Rtl),
            "كُتِبَ"
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
    fn rtl_postprocess_preserves_internal_paired_punctuation_in_ltr_token() {
        assert_eq!(
            postprocess_text_direction("abc(def) ابحرم".to_string(), TextDirection::Rtl),
            "مرحبا abc(def)"
        );
    }

    #[test]
    fn rtl_postprocess_preserves_ltr_combining_marks() {
        assert_eq!(
            postprocess_text_direction("cafe\u{301} ابحرم".to_string(), TextDirection::Rtl),
            "مرحبا cafe\u{301}"
        );
    }

    #[test]
    fn rtl_postprocess_preserves_ltr_combining_marks_before_another_ltr_word() {
        assert_eq!(
            postprocess_text_direction("cafe\u{301} xyz ابحرم".to_string(), TextDirection::Rtl),
            "مرحبا cafe\u{301} xyz"
        );
    }

    #[test]
    fn rtl_postprocess_preserves_apostrophes_inside_ltr_tokens() {
        assert_eq!(
            postprocess_text_direction("O'Reilly ابحرم".to_string(), TextDirection::Rtl),
            "مرحبا O'Reilly"
        );
    }

    #[test]
    fn postprocess_reports_order_changes() {
        assert_eq!(
            postprocess_text_direction_with_order_change(
                "Issue رقم 123".to_string(),
                TextDirection::Auto,
            ),
            ("Issue رقم 123".to_string(), false)
        );
        assert_eq!(
            postprocess_text_direction_with_order_change("ابحرم".to_string(), TextDirection::Rtl),
            ("مرحبا".to_string(), true)
        );
        assert_eq!(
            postprocess_text_direction_with_order_change("م".to_string(), TextDirection::Rtl),
            ("م".to_string(), false)
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
