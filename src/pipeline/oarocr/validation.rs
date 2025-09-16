//! Validation utilities for the OAROCR builder.
//!
//! This module contains validation functions used by the OAROCRBuilder
//! to ensure configuration parameters are within valid ranges.

use tracing::warn;

/// Validates and clamps a threshold value to the range [0.0, 1.0].
///
/// # Arguments
///
/// * `threshold` - The threshold value to validate
/// * `param_name` - The name of the parameter for logging purposes
///
/// # Returns
///
/// The validated and potentially clamped threshold value
pub fn validate_threshold(threshold: f32, param_name: &str) -> f32 {
    if (0.0..=1.0).contains(&threshold) {
        threshold
    } else {
        warn!("{param_name} out of range [{threshold}], clamping to [0.0, 1.0]");
        threshold.clamp(0.0, 1.0)
    }
}

/// Validates and ensures a size value is at least 1.
///
/// # Arguments
///
/// * `size` - The size value to validate
/// * `param_name` - The name of the parameter for logging purposes
///
/// # Returns
///
/// The validated size value (minimum 1)
pub fn validate_min_size_usize(size: usize, param_name: &str) -> usize {
    if size >= 1 {
        size
    } else {
        warn!("{param_name} must be >= 1, got {size}; using 1");
        1
    }
}

/// Validates and ensures a size value is at least 1.
///
/// # Arguments
///
/// * `size` - The size value to validate
/// * `param_name` - The name of the parameter for logging purposes
///
/// # Returns
///
/// The validated size value (minimum 1)
pub fn validate_min_size_u32(size: u32, param_name: &str) -> u32 {
    if size >= 1 {
        size
    } else {
        warn!("{param_name} must be >= 1, got {size}; using 1");
        1
    }
}

/// Validates and ensures a dimension value is greater than 0.
///
/// # Arguments
///
/// * `dimension` - The dimension value to validate
/// * `param_name` - The name of the parameter for logging purposes
///
/// # Returns
///
/// The validated dimension value (minimum 1)
pub fn validate_dimension(dimension: u32, param_name: &str) -> u32 {
    if dimension > 0 {
        dimension
    } else {
        warn!("{param_name} {} <= 0; using 1", dimension);
        1u32
    }
}

/// Validates and ensures a positive float value.
///
/// # Arguments
///
/// * `value` - The value to validate
/// * `param_name` - The name of the parameter for logging purposes
/// * `default` - The default value to use if validation fails
///
/// # Returns
///
/// The validated value or the default
pub fn validate_positive_f32(value: f32, param_name: &str, default: f32) -> f32 {
    if value > 0.0 {
        value
    } else {
        warn!("{param_name} must be > 0.0, got {value}; using {default}");
        default
    }
}

/// Validates and ensures a non-negative float value.
///
/// # Arguments
///
/// * `value` - The value to validate
/// * `param_name` - The name of the parameter for logging purposes
///
/// # Returns
///
/// The validated value (minimum 0.0)
pub fn validate_non_negative_f32(value: f32, param_name: &str) -> f32 {
    if value >= 0.0 {
        value
    } else {
        warn!("{param_name} must be >= 0.0, got {value}; using 0.0");
        0.0
    }
}