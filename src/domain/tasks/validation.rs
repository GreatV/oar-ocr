//! Shared validation helpers for task inputs.
//!
//! This module re-exports validation functions from `core::validation` and provides
//! backwards-compatible wrappers for existing code.

use crate::core::OCRError;
use crate::core::validation::validate_image_batch_with_message;
use image::RgbImage;

/// Ensures an image batch is non-empty and each image has positive dimensions.
pub(crate) fn ensure_non_empty_images(
    images: &[RgbImage],
    empty_batch_message: &str,
) -> Result<(), OCRError> {
    ensure_images_with(images, empty_batch_message, |idx, width, height| {
        format!(
            "Invalid image dimensions for item {idx}: width={width}, height={height} must be positive. Please check your input image."
        )
    })
}

/// Generic helper for validating non-empty RgbImage collections with custom error messaging.
///
/// Delegates to `crate::core::validation::validate_image_batch_with_message`.
pub(crate) fn ensure_images_with(
    images: &[RgbImage],
    empty_batch_message: &str,
    zero_dim_message: impl Fn(usize, u32, u32) -> String,
) -> Result<(), OCRError> {
    validate_image_batch_with_message(images, empty_batch_message, zero_dim_message)
}
