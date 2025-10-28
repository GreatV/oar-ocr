//! Shared validation helpers for task inputs.
//!
//! Centralizes common input validation logic to avoid duplicating checks
//! like ensuring batches are non-empty and images have positive dimensions.

use crate::core::OCRError;
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

/// Generic helper for validating non-empty image collections with custom error messaging.
pub(crate) fn ensure_images_with(
    images: &[RgbImage],
    empty_batch_message: &str,
    zero_dim_message: impl Fn(usize, u32, u32) -> String,
) -> Result<(), OCRError> {
    if images.is_empty() {
        return Err(OCRError::InvalidInput {
            message: empty_batch_message.to_string(),
        });
    }

    for (idx, img) in images.iter().enumerate() {
        let (width, height) = (img.width(), img.height());
        if width == 0 || height == 0 {
            return Err(OCRError::InvalidInput {
                message: zero_dim_message(idx, width, height),
            });
        }
    }

    Ok(())
}
