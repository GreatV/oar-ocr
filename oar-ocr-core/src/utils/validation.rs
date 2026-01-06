//! Reusable validation components for OCR tasks.
//!
//! This module provides composable validators that can be used across different tasks
//! to validate common patterns like score ranges, dimensions, and other constraints.

use crate::core::OCRError;

/// A reusable validator for score ranges.
///
/// This validator can be configured with custom min/max bounds and field names,
/// making it suitable for validating confidence scores, probabilities, and other
/// numerical ranges across different tasks.
///
/// # Examples
///
/// ```rust,no_run
/// # use oar_ocr::utils::validation::ScoreValidator;
/// # use oar_ocr::core::OCRError;
/// # fn example() -> Result<(), OCRError> {
/// let validator = ScoreValidator::new_unit_range("confidence");
/// validator.validate_scores(&[0.5, 0.8, 0.95], "Detection")?;
/// #     Ok(())
/// # }
/// # example().unwrap();
/// ```
#[derive(Debug, Clone)]
pub struct ScoreValidator {
    min: f32,
    max: f32,
    field_name: String,
}

impl ScoreValidator {
    /// Creates a new score validator with custom bounds.
    ///
    /// # Arguments
    ///
    /// * `min` - Minimum valid score (inclusive)
    /// * `max` - Maximum valid score (inclusive)
    /// * `field_name` - Name of the field being validated (for error messages)
    pub fn new(min: f32, max: f32, field_name: impl Into<String>) -> Self {
        Self {
            min,
            max,
            field_name: field_name.into(),
        }
    }

    /// Creates a validator for unit range scores [0.0, 1.0].
    ///
    /// This is the most common case for confidence scores and probabilities.
    pub fn new_unit_range(field_name: impl Into<String>) -> Self {
        Self::new(0.0, 1.0, field_name)
    }

    /// Validates a single score value.
    ///
    /// # Errors
    ///
    /// Returns `OCRError::InvalidInput` if the score is outside the valid range.
    pub fn validate_score(&self, score: f32, context: &str) -> Result<(), OCRError> {
        if !(self.min..=self.max).contains(&score) {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "{}: {} {} is out of valid range [{}, {}]",
                    context, self.field_name, score, self.min, self.max
                ),
            });
        }
        Ok(())
    }

    /// Validates a collection of scores.
    ///
    /// # Errors
    ///
    /// Returns `OCRError::InvalidInput` if any score is outside the valid range.
    /// The error message includes the index of the invalid score.
    pub fn validate_scores(&self, scores: &[f32], context_prefix: &str) -> Result<(), OCRError> {
        for (idx, &score) in scores.iter().enumerate() {
            self.validate_score(score, &format!("{} {}", context_prefix, idx))?;
        }
        Ok(())
    }

    /// Validates a collection of scores with a custom index formatter.
    ///
    /// This is useful when you want to provide more context in error messages,
    /// such as "Image 3, detection 2" instead of just an index.
    pub fn validate_scores_with<F>(&self, scores: &[f32], format_context: F) -> Result<(), OCRError>
    where
        F: Fn(usize) -> String,
    {
        for (idx, &score) in scores.iter().enumerate() {
            self.validate_score(score, &format_context(idx))?;
        }
        Ok(())
    }
}

/// Validates that a vector's length matches an expected size.
///
/// # Errors
///
/// Returns `OCRError::InvalidInput` if lengths don't match.
pub fn validate_length_match(
    actual: usize,
    expected: usize,
    actual_name: &str,
    expected_name: &str,
) -> Result<(), OCRError> {
    if actual != expected {
        return Err(OCRError::InvalidInput {
            message: format!(
                "Mismatch between {} count ({}) and {} count ({})",
                actual_name, actual, expected_name, expected
            ),
        });
    }
    Ok(())
}

/// Validates that a value doesn't exceed a maximum.
///
/// # Errors
///
/// Returns `OCRError::InvalidInput` if value exceeds maximum.
pub fn validate_max_value<T: PartialOrd + std::fmt::Display>(
    value: T,
    max: T,
    field_name: &str,
    context: &str,
) -> Result<(), OCRError> {
    if value > max {
        return Err(OCRError::InvalidInput {
            message: format!(
                "{}: {} {} exceeds maximum {}",
                context, field_name, value, max
            ),
        });
    }
    Ok(())
}

/// Validates that dimensions are positive (non-zero).
///
/// # Errors
///
/// Returns `OCRError::InvalidInput` if either dimension is zero.
pub fn validate_positive_dimensions(
    width: u32,
    height: u32,
    context: &str,
) -> Result<(), OCRError> {
    if width == 0 || height == 0 {
        return Err(OCRError::InvalidInput {
            message: format!(
                "{}: invalid dimensions width={}, height={} (must be positive)",
                context, width, height
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_validator_unit_range() {
        let validator = ScoreValidator::new_unit_range("score");

        // Valid scores should pass
        assert!(validator.validate_score(0.0, "test").is_ok());
        assert!(validator.validate_score(0.5, "test").is_ok());
        assert!(validator.validate_score(1.0, "test").is_ok());

        // Invalid scores should fail
        assert!(validator.validate_score(-0.1, "test").is_err());
        assert!(validator.validate_score(1.1, "test").is_err());
    }

    #[test]
    fn test_score_validator_custom_range() {
        let validator = ScoreValidator::new(0.5, 2.0, "custom");

        // Valid scores should pass
        assert!(validator.validate_score(0.5, "test").is_ok());
        assert!(validator.validate_score(1.0, "test").is_ok());
        assert!(validator.validate_score(2.0, "test").is_ok());

        // Invalid scores should fail
        assert!(validator.validate_score(0.4, "test").is_err());
        assert!(validator.validate_score(2.1, "test").is_err());
    }

    #[test]
    fn test_validate_scores() {
        let validator = ScoreValidator::new_unit_range("score");

        // All valid scores
        assert!(validator.validate_scores(&[0.1, 0.5, 0.9], "test").is_ok());

        // One invalid score
        assert!(validator.validate_scores(&[0.1, 1.5, 0.9], "test").is_err());
    }

    #[test]
    fn test_validate_scores_with_formatter() {
        let validator = ScoreValidator::new_unit_range("score");

        let result = validator
            .validate_scores_with(&[0.5, 1.5], |idx| format!("Image 0, detection {}", idx));

        assert!(result.is_err());
        let err_msg = format!("{:?}", result.unwrap_err());
        assert!(err_msg.contains("detection 1"));
    }

    #[test]
    fn test_validate_length_match() {
        assert!(validate_length_match(3, 3, "texts", "scores").is_ok());
        assert!(validate_length_match(3, 5, "texts", "scores").is_err());
    }

    #[test]
    fn test_validate_max_value() {
        assert!(validate_max_value(50, 100, "length", "text 0").is_ok());
        assert!(validate_max_value(150, 100, "length", "text 0").is_err());
    }

    #[test]
    fn test_validate_positive_dimensions() {
        assert!(validate_positive_dimensions(100, 200, "image 0").is_ok());
        assert!(validate_positive_dimensions(0, 200, "image 0").is_err());
        assert!(validate_positive_dimensions(100, 0, "image 0").is_err());
    }
}
