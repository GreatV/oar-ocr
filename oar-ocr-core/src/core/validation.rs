//! Input Validation Utilities
//!
//! This module provides comprehensive validation utilities to prevent runtime panics
//! and ensure data integrity across the OCR pipeline.

use crate::core::OCRError;

/// Validates that a float value is finite (not NaN or infinite).
#[inline]
pub fn validate_finite(value: f32, param_name: &str) -> Result<(), OCRError> {
    if !value.is_finite() {
        return Err(OCRError::InvalidInput {
            message: format!("Parameter '{}' must be finite, got: {}", param_name, value),
        });
    }
    Ok(())
}

/// Validates that a value is within a specified range (inclusive).
#[inline]
pub fn validate_range<T: PartialOrd + std::fmt::Display>(
    value: T,
    min: T,
    max: T,
    param_name: &str,
) -> Result<(), OCRError> {
    if value < min || value > max {
        return Err(OCRError::InvalidInput {
            message: format!(
                "Parameter '{}' must be in range [{}, {}], got: {}",
                param_name, min, max, value
            ),
        });
    }
    Ok(())
}

/// Validates that a value is positive (> 0).
#[inline]
pub fn validate_positive<T: PartialOrd + std::fmt::Display + Default>(
    value: T,
    param_name: &str,
) -> Result<(), OCRError> {
    if value <= T::default() {
        return Err(OCRError::InvalidInput {
            message: format!(
                "Parameter '{}' must be positive, got: {}",
                param_name, value
            ),
        });
    }
    Ok(())
}

/// Validates that a value is non-negative (>= 0).
#[inline]
pub fn validate_non_negative<T: PartialOrd + std::fmt::Display + Default>(
    value: T,
    param_name: &str,
) -> Result<(), OCRError> {
    if value < T::default() {
        return Err(OCRError::InvalidInput {
            message: format!(
                "Parameter '{}' must be non-negative, got: {}",
                param_name, value
            ),
        });
    }
    Ok(())
}

/// Validates that a collection is not empty.
#[inline]
pub fn validate_non_empty<T>(items: &[T], param_name: &str) -> Result<(), OCRError> {
    if items.is_empty() {
        return Err(OCRError::InvalidInput {
            message: format!("Parameter '{}' cannot be empty", param_name),
        });
    }
    Ok(())
}

/// Validates that two collections have the same length.
#[inline]
pub fn validate_same_length<T, U>(
    items1: &[T],
    items2: &[U],
    name1: &str,
    name2: &str,
) -> Result<(), OCRError> {
    if items1.len() != items2.len() {
        return Err(OCRError::InvalidInput {
            message: format!(
                "Length mismatch: {} has {} elements, but {} has {} elements",
                name1,
                items1.len(),
                name2,
                items2.len()
            ),
        });
    }
    Ok(())
}

/// Validates tensor shape dimensions.
pub fn validate_tensor_shape(
    shape: &[usize],
    expected_dims: usize,
    tensor_name: &str,
) -> Result<(), OCRError> {
    if shape.len() != expected_dims {
        return Err(OCRError::InvalidInput {
            message: format!(
                "Tensor '{}' expected {}D shape, got {}D: {:?}",
                tensor_name,
                expected_dims,
                shape.len(),
                shape
            ),
        });
    }
    Ok(())
}

/// Validates that tensor batch size is positive.
pub fn validate_batch_size(shape: &[usize], tensor_name: &str) -> Result<usize, OCRError> {
    validate_tensor_shape(shape, 4, tensor_name)?;

    let batch_size = shape[0];
    if batch_size == 0 {
        return Err(OCRError::InvalidInput {
            message: format!(
                "Tensor '{}' has zero batch size. Shape: {:?}",
                tensor_name, shape
            ),
        });
    }

    Ok(batch_size)
}

/// Validates image dimensions.
pub fn validate_image_dimensions(height: u32, width: u32, context: &str) -> Result<(), OCRError> {
    if height == 0 || width == 0 {
        return Err(OCRError::InvalidInput {
            message: format!(
                "{}: image dimensions must be positive, got {}x{}",
                context, height, width
            ),
        });
    }

    // Reasonable upper bounds to prevent memory issues
    const MAX_DIMENSION: u32 = 32768;
    if height > MAX_DIMENSION || width > MAX_DIMENSION {
        return Err(OCRError::InvalidInput {
            message: format!(
                "{}: image dimensions exceed maximum of {}x{}, got {}x{}",
                context, MAX_DIMENSION, MAX_DIMENSION, height, width
            ),
        });
    }

    Ok(())
}

/// Validates that array index is within bounds.
#[inline]
pub fn validate_index_bounds<T>(
    slice: &[T],
    index: usize,
    slice_name: &str,
) -> Result<(), OCRError> {
    if index >= slice.len() {
        return Err(OCRError::InvalidInput {
            message: format!(
                "Index out of bounds for '{}': index {} >= length {}",
                slice_name,
                index,
                slice.len()
            ),
        });
    }
    Ok(())
}

/// Validates division operands to prevent division by zero.
#[inline]
pub fn validate_division(numerator: f32, denominator: f32, context: &str) -> Result<(), OCRError> {
    validate_finite(numerator, &format!("{} numerator", context))?;
    validate_finite(denominator, &format!("{} denominator", context))?;

    if denominator.abs() < f32::EPSILON {
        return Err(OCRError::InvalidInput {
            message: format!(
                "{}: division by zero (denominator: {})",
                context, denominator
            ),
        });
    }

    Ok(())
}

/// Validates normalization parameters (mean and std).
pub fn validate_normalization_params(
    mean: &[f32],
    std: &[f32],
    num_channels: usize,
) -> Result<(), OCRError> {
    // Check lengths match expected channels
    if mean.len() != num_channels {
        return Err(OCRError::InvalidInput {
            message: format!(
                "Mean length {} does not match number of channels {}",
                mean.len(),
                num_channels
            ),
        });
    }

    if std.len() != num_channels {
        return Err(OCRError::InvalidInput {
            message: format!(
                "Std length {} does not match number of channels {}",
                std.len(),
                num_channels
            ),
        });
    }

    // Validate all values are finite
    for (i, &m) in mean.iter().enumerate() {
        validate_finite(m, &format!("mean[{}]", i))?;
    }

    // Validate std values are positive and finite
    for (i, &s) in std.iter().enumerate() {
        validate_finite(s, &format!("std[{}]", i))?;
        validate_positive(s, &format!("std[{}]", i))?;
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_finite() {
        assert!(validate_finite(1.0, "test").is_ok());
        assert!(validate_finite(0.0, "test").is_ok());
        assert!(validate_finite(-1.0, "test").is_ok());
        assert!(validate_finite(f32::NAN, "test").is_err());
        assert!(validate_finite(f32::INFINITY, "test").is_err());
        assert!(validate_finite(f32::NEG_INFINITY, "test").is_err());
    }

    #[test]
    fn test_validate_range() {
        assert!(validate_range(5.0, 0.0, 10.0, "test").is_ok());
        assert!(validate_range(0.0, 0.0, 10.0, "test").is_ok());
        assert!(validate_range(10.0, 0.0, 10.0, "test").is_ok());
        assert!(validate_range(-1.0, 0.0, 10.0, "test").is_err());
        assert!(validate_range(11.0, 0.0, 10.0, "test").is_err());
    }

    #[test]
    fn test_validate_positive() {
        assert!(validate_positive(1.0, "test").is_ok());
        assert!(validate_positive(0.1, "test").is_ok());
        assert!(validate_positive(0.0, "test").is_err());
        assert!(validate_positive(-1.0, "test").is_err());
    }

    #[test]
    fn test_validate_non_negative() {
        assert!(validate_non_negative(1.0, "test").is_ok());
        assert!(validate_non_negative(0.0, "test").is_ok());
        assert!(validate_non_negative(-1.0, "test").is_err());
    }

    #[test]
    fn test_validate_non_empty() {
        assert!(validate_non_empty(&[1, 2, 3], "test").is_ok());
        assert!(validate_non_empty(&[1], "test").is_ok());
        assert!(validate_non_empty::<i32>(&[], "test").is_err());
    }

    #[test]
    fn test_validate_same_length() {
        assert!(validate_same_length(&[1, 2], &[3, 4], "a", "b").is_ok());
        assert!(validate_same_length(&[1], &[2], "a", "b").is_ok());
        assert!(validate_same_length(&[1, 2], &[3], "a", "b").is_err());
    }

    #[test]
    fn test_validate_tensor_shape() {
        assert!(validate_tensor_shape(&[1, 3, 224, 224], 4, "test").is_ok());
        assert!(validate_tensor_shape(&[1, 3, 224], 3, "test").is_ok());
        assert!(validate_tensor_shape(&[1, 3, 224], 4, "test").is_err());
    }

    #[test]
    fn test_validate_batch_size() {
        match validate_batch_size(&[2, 3, 224, 224], "test") {
            Ok(batch_size) => assert_eq!(batch_size, 2),
            Err(err) => panic!("expected validate_batch_size to succeed: {err}"),
        }
        match validate_batch_size(&[1, 3, 224, 224], "test") {
            Ok(batch_size) => assert_eq!(batch_size, 1),
            Err(err) => panic!("expected validate_batch_size to succeed: {err}"),
        }
        assert!(validate_batch_size(&[0, 3, 224, 224], "test").is_err());
        assert!(validate_batch_size(&[1, 3, 224], "test").is_err());
    }

    #[test]
    fn test_validate_image_dimensions() {
        assert!(validate_image_dimensions(224, 224, "test").is_ok());
        assert!(validate_image_dimensions(1, 1, "test").is_ok());
        assert!(validate_image_dimensions(0, 224, "test").is_err());
        assert!(validate_image_dimensions(224, 0, "test").is_err());
        assert!(validate_image_dimensions(99999, 99999, "test").is_err());
    }

    #[test]
    fn test_validate_division() {
        assert!(validate_division(10.0, 2.0, "test").is_ok());
        assert!(validate_division(0.0, 2.0, "test").is_ok());
        assert!(validate_division(10.0, 0.0, "test").is_err());
        assert!(validate_division(10.0, 1e-10, "test").is_err());
        assert!(validate_division(f32::NAN, 2.0, "test").is_err());
        assert!(validate_division(10.0, f32::INFINITY, "test").is_err());
    }

    #[test]
    fn test_validate_normalization_params() {
        assert!(
            validate_normalization_params(&[0.485, 0.456, 0.406], &[0.229, 0.224, 0.225], 3)
                .is_ok()
        );

        // Wrong length
        assert!(validate_normalization_params(&[0.485, 0.456], &[0.229, 0.224, 0.225], 3).is_err());

        // NaN in mean
        assert!(
            validate_normalization_params(&[f32::NAN, 0.456, 0.406], &[0.229, 0.224, 0.225], 3)
                .is_err()
        );

        // Zero in std
        assert!(
            validate_normalization_params(&[0.485, 0.456, 0.406], &[0.0, 0.224, 0.225], 3).is_err()
        );

        // Negative in std
        assert!(
            validate_normalization_params(&[0.485, 0.456, 0.406], &[-0.229, 0.224, 0.225], 3)
                .is_err()
        );
    }
}
