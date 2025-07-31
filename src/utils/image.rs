//! Utility functions for image processing.
//!
//! This module provides functions for loading, converting, and manipulating images
//! in the OCR pipeline. It includes functions for converting between different
//! image formats, loading single or batch images from files, and creating images
//! from raw data.

use crate::core::OCRError;
use image::{DynamicImage, GrayImage, ImageBuffer, RgbImage};

/// Converts a DynamicImage to an RgbImage.
///
/// This function takes a DynamicImage (which can be in any format) and converts
/// it to an RgbImage (8-bit RGB format).
///
/// # Arguments
///
/// * `img` - The DynamicImage to convert
///
/// # Returns
///
/// * `RgbImage` - The converted RGB image
pub fn dynamic_to_rgb(img: DynamicImage) -> RgbImage {
    img.to_rgb8()
}

/// Converts a DynamicImage to a GrayImage.
///
/// This function takes a DynamicImage (which can be in any format) and converts
/// it to a GrayImage (8-bit grayscale format).
///
/// # Arguments
///
/// * `img` - The DynamicImage to convert
///
/// # Returns
///
/// * `GrayImage` - The converted grayscale image
pub fn dynamic_to_gray(img: DynamicImage) -> GrayImage {
    img.to_luma8()
}

/// Loads an image from a file path and converts it to RgbImage.
///
/// This function opens an image from the specified file path and converts it
/// to an RgbImage. It handles any image format supported by the image crate.
///
/// # Arguments
///
/// * `path` - A reference to the path of the image file to load
///
/// # Returns
///
/// * `Ok(RgbImage)` - The loaded and converted RGB image
/// * `Err(OCRError)` - An error if the image could not be loaded or converted
///
/// # Errors
///
/// This function will return an `OCRError::ImageLoad` error if the image cannot
/// be loaded from the specified path, or if there is an error during conversion.
pub fn load_image(path: &std::path::Path) -> Result<RgbImage, OCRError> {
    let img = image::open(path).map_err(OCRError::ImageLoad)?;
    Ok(dynamic_to_rgb(img))
}

/// Creates an RgbImage from raw pixel data.
///
/// This function creates an RgbImage from raw pixel data. The data must be
/// in RGB format (3 bytes per pixel) and the length must match the specified
/// width and height.
///
/// # Arguments
///
/// * `width` - The width of the image in pixels
/// * `height` - The height of the image in pixels
/// * `data` - A vector containing the raw pixel data (RGB format)
///
/// # Returns
///
/// * `Some(RgbImage)` - The created RGB image if the data is valid
/// * `None` - If the data length doesn't match the specified dimensions
pub fn create_rgb_image(width: u32, height: u32, data: Vec<u8>) -> Option<RgbImage> {
    if data.len() != (width * height * 3) as usize {
        return None;
    }

    ImageBuffer::from_raw(width, height, data)
}

/// Loads a batch of images from file paths.
///
/// This function loads multiple images from the specified file paths and
/// converts them to RgbImages. It uses parallel processing when the number
/// of images exceeds the default parallel threshold.
///
/// # Arguments
///
/// * `paths` - A slice of paths to the image files to load
///
/// # Returns
///
/// * `Ok(Vec<RgbImage>)` - A vector of loaded RGB images
/// * `Err(OCRError)` - An error if any image could not be loaded
///
/// # Errors
///
/// This function will return an `OCRError` if any image cannot be loaded
/// from its specified path.
pub fn load_images_batch<P: AsRef<std::path::Path> + Send + Sync>(
    paths: &[P],
) -> Result<Vec<RgbImage>, OCRError> {
    load_images_batch_with_threshold(paths, None)
}

/// Loads a batch of images from file paths with a custom parallel threshold.
///
/// This function loads multiple images from the specified file paths and
/// converts them to RgbImages. It uses parallel processing when the number
/// of images exceeds the specified threshold, or the default threshold if
/// none is provided.
///
/// # Arguments
///
/// * `paths` - A slice of paths to the image files to load
/// * `parallel_threshold` - An optional threshold for parallel processing.
///   If `None`, the default threshold from `DEFAULT_PARALLEL_THRESHOLD` is used.
///
/// # Returns
///
/// * `Ok(Vec<RgbImage>)` - A vector of loaded RGB images
/// * `Err(OCRError)` - An error if any image could not be loaded
///
/// # Errors
///
/// This function will return an `OCRError` if any image cannot be loaded
/// from its specified path.
pub fn load_images_batch_with_threshold<P: AsRef<std::path::Path> + Send + Sync>(
    paths: &[P],
    parallel_threshold: Option<usize>,
) -> Result<Vec<RgbImage>, OCRError> {
    use crate::core::constants::DEFAULT_PARALLEL_THRESHOLD;

    let threshold = parallel_threshold.unwrap_or(DEFAULT_PARALLEL_THRESHOLD);

    if paths.len() > threshold {
        use rayon::prelude::*;
        paths.par_iter().map(|p| load_image(p.as_ref())).collect()
    } else {
        paths.iter().map(|p| load_image(p.as_ref())).collect()
    }
}
