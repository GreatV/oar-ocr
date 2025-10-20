//! Formula preprocessing utilities for PaddleOCR-style models.
//!
//! This module provides reusable preprocessing pipelines for formula recognition,
//! including image normalization, margin cropping, and tensor formatting.

use crate::core::{OCRError, Tensor4D};
use image::imageops::{FilterType, overlay, resize};
use image::{DynamicImage, RgbImage};
use ndarray::{Array2, Array3, Array4};

/// Configuration parameters for formula preprocessing pipeline.
#[derive(Debug, Clone, Copy)]
pub struct FormulaPreprocessParams {
    /// Target image size (width, height) after preprocessing
    pub target_size: (u32, u32),
    /// Threshold for binarizing image during margin cropping (0-255)
    pub crop_threshold: u8,
    /// Padding alignment requirement for output tensor dimensions
    pub padding_multiple: usize,
    /// Mean values for channel-wise normalization in HWC order
    pub normalize_mean: [f32; 3],
    /// Standard deviation for channel-wise normalization in HWC order
    pub normalize_std: [f32; 3],
}

/// Preprocessor implementing the standard PaddleOCR formula recognition pipeline.
///
/// This preprocessor applies the following transformations:
/// 1. Margin cropping - removes background margins by binarization
/// 2. Resize and padding - scales image to target size with aspect ratio preservation
/// 3. Normalization and grayscale conversion - normalizes pixel values and converts to grayscale
/// 4. Tensor formatting - converts to 4D tensor with proper padding alignment
#[derive(Debug, Clone)]
pub struct FormulaPreprocessor {
    params: FormulaPreprocessParams,
}

impl FormulaPreprocessor {
    /// Creates a new preprocessor with the specified parameters.
    pub fn new(params: FormulaPreprocessParams) -> Self {
        Self { params }
    }

    /// Preprocesses a batch of images through the complete pipeline.
    ///
    /// # Arguments
    /// * `images` - Input images as RGB8 format
    ///
    /// # Returns
    /// A 4D tensor of shape [batch, channels, height, width] ready for model inference
    pub fn preprocess_batch(&self, images: &[RgbImage]) -> Result<Tensor4D, OCRError> {
        let mut normalized = Vec::with_capacity(images.len());

        for image in images {
            let cropped = self.crop_margin(image);
            let resized = self.resize_and_pad(&cropped);
            let normalized_image = self.normalize_and_to_grayscale(&resized);
            normalized.push(normalized_image);
        }

        self.format_to_tensor(normalized)
    }

    /// Removes background margins by binarization and cropping.
    ///
    /// The algorithm:
    /// 1. Converts to grayscale
    /// 2. Normalizes pixel values to 0-255 range
    /// 3. Binarizes using the configured threshold
    /// 4. Finds bounding box of foreground pixels
    /// 5. Crops to the bounding box
    fn crop_margin(&self, img: &RgbImage) -> RgbImage {
        let gray = DynamicImage::ImageRgb8(img.clone()).to_luma8();
        let (width, height) = gray.dimensions();

        // Find min and max pixel values for normalization
        let mut min_val = u8::MAX;
        let mut max_val = u8::MIN;
        for pixel in gray.pixels() {
            let val = pixel[0];
            min_val = min_val.min(val);
            max_val = max_val.max(val);
        }

        // If image is uniform, return as-is
        if max_val == min_val {
            return img.clone();
        }

        // Create binary image using threshold
        let mut binary = image::GrayImage::new(width, height);
        for (x, y, pixel) in gray.enumerate_pixels() {
            let normalized = ((pixel[0] as f32 - min_val as f32)
                / (max_val as f32 - min_val as f32)
                * 255.0) as u8;
            binary.put_pixel(
                x,
                y,
                image::Luma([if normalized < self.params.crop_threshold {
                    255
                } else {
                    0
                }]),
            );
        }

        // Find bounding box of foreground pixels
        let mut min_x = width;
        let mut min_y = height;
        let mut max_x = 0;
        let mut max_y = 0;
        for (x, y, pixel) in binary.enumerate_pixels() {
            if pixel[0] > 0 {
                min_x = min_x.min(x);
                min_y = min_y.min(y);
                max_x = max_x.max(x);
                max_y = max_y.max(y);
            }
        }

        // Return original if no foreground found or invalid bounds
        if min_x >= max_x || min_y >= max_y {
            return img.clone();
        }

        // Crop to bounding box
        image::imageops::crop_imm(img, min_x, min_y, max_x - min_x + 1, max_y - min_y + 1)
            .to_image()
    }

    /// Resizes image to target size while preserving aspect ratio, then pads with black.
    ///
    /// The resize strategy:
    /// 1. Calculates scale factor based on the smaller dimension of target size
    /// 2. Resizes maintaining aspect ratio
    /// 3. Centers the resized image on black background of target size
    fn resize_and_pad(&self, img: &RgbImage) -> RgbImage {
        let (target_width, target_height) = self.params.target_size;
        let (img_width, img_height) = img.dimensions();

        if img_width == 0 || img_height == 0 {
            return RgbImage::new(target_width, target_height);
        }

        // Calculate scale to fit within target size
        let min_size = target_width.min(target_height);
        let scale = (min_size as f32) / (img_width.max(img_height) as f32);
        let new_width = (img_width as f32 * scale) as u32;
        let new_height = (img_height as f32 * scale) as u32;

        let final_width = new_width.min(target_width);
        let final_height = new_height.min(target_height);

        let resized = resize(img, final_width, final_height, FilterType::Lanczos3);

        // Calculate padding to center the image
        let delta_width = target_width - final_width;
        let delta_height = target_height - final_height;
        let pad_left = delta_width / 2;
        let pad_top = delta_height / 2;

        // Create black background and overlay resized image
        let mut padded = RgbImage::from_pixel(target_width, target_height, image::Rgb([0, 0, 0]));
        overlay(&mut padded, &resized, pad_left as i64, pad_top as i64);

        padded
    }

    /// Normalizes pixel values and converts to grayscale representation.
    ///
    /// The normalization follows UniMERNet's preprocessing:
    /// 1. Normalizes RGB channels using mean and std (in BGR order for OpenCV compatibility)
    /// 2. Converts to grayscale using standard weights (0.299*R + 0.587*G + 0.114*B)
    /// 3. Replicates grayscale to 3 channels for model input
    fn normalize_and_to_grayscale(&self, img: &RgbImage) -> Array3<f32> {
        let (width, height) = img.dimensions();

        const SCALE: f32 = 1.0 / 255.0;
        let mean = self.params.normalize_mean;
        let std = self.params.normalize_std;

        // Normalize RGB channels (BGR order for OpenCV compatibility)
        let mut normalized = Array3::<f32>::zeros((height as usize, width as usize, 3));
        for (x, y, pixel) in img.enumerate_pixels() {
            let r = pixel[0] as f32;
            let g = pixel[1] as f32;
            let b = pixel[2] as f32;

            // BGR order to match OpenCV convention
            normalized[[y as usize, x as usize, 0]] =
                (b * SCALE - mean[0]) / std[0].max(f32::EPSILON);
            normalized[[y as usize, x as usize, 1]] =
                (g * SCALE - mean[1]) / std[1].max(f32::EPSILON);
            normalized[[y as usize, x as usize, 2]] =
                (r * SCALE - mean[2]) / std[2].max(f32::EPSILON);
        }

        // Convert to grayscale using standard luminance formula
        // Y = 0.299*R + 0.587*G + 0.114*B (in RGB order)
        // In BGR order: Y = 0.114*B + 0.587*G + 0.299*R
        let mut grayscale = Array2::<f32>::zeros((height as usize, width as usize));
        for y in 0..height as usize {
            for x in 0..width as usize {
                let b = normalized[[y, x, 0]];
                let g = normalized[[y, x, 1]];
                let r = normalized[[y, x, 2]];
                grayscale[[y, x]] = 0.114 * b + 0.587 * g + 0.299 * r;
            }
        }

        // Replicate grayscale channel to 3 channels
        let mut result = Array3::<f32>::zeros((height as usize, width as usize, 3));
        for y in 0..height as usize {
            for x in 0..width as usize {
                let gray_val = grayscale[[y, x]];
                result[[y, x, 0]] = gray_val;
                result[[y, x, 1]] = gray_val;
                result[[y, x, 2]] = gray_val;
            }
        }

        result
    }

    /// Formats preprocessed images into a properly padded 4D tensor.
    ///
    /// Creates a tensor with dimensions padded to multiples of the configured value,
    /// which is required by some models for efficient computation.
    fn format_to_tensor(&self, images: Vec<Array3<f32>>) -> Result<Tensor4D, OCRError> {
        let (target_width, target_height) = self.params.target_size;
        let batch_size = images.len();

        // Pad dimensions to multiples of padding_multiple
        let padded_height = ((target_height as f32 / self.params.padding_multiple as f32).ceil()
            * self.params.padding_multiple as f32) as usize;
        let padded_width = ((target_width as f32 / self.params.padding_multiple as f32).ceil()
            * self.params.padding_multiple as f32) as usize;

        // Use 1.0 as padding value after normalization
        let padding_value = 1.0_f32;

        // Create output tensor [batch, channels=1, height, width]
        let mut tensor =
            Array4::<f32>::from_elem((batch_size, 1, padded_height, padded_width), padding_value);

        // Copy first channel of each image to tensor
        for (batch_idx, img) in images.iter().enumerate() {
            for y in 0..target_height as usize {
                for x in 0..target_width as usize {
                    tensor[[batch_idx, 0, y, x]] = img[[y, x, 0]];
                }
            }
        }

        Ok(tensor)
    }
}

/// Normalizes decoded LaTeX text using PaddleOCR UniMERNet normalization rules.
///
/// This function applies the following transformations:
/// - Removes Chinese text wrapping (e.g., `\text{中文}` → `中文`)
/// - Removes quotation marks
/// - Removes unnecessary whitespace between symbols while preserving LaTeX space commands
///
/// # Arguments
/// * `latex` - Raw LaTeX string from model output
///
/// # Returns
/// Normalized LaTeX string suitable for rendering
pub fn normalize_latex(latex: &str) -> String {
    use regex::Regex;

    let mut result = latex.to_string();

    // Remove \text{} wrapping around Chinese characters
    let chinese_text_pattern =
        Regex::new(r"\\text\s*\{([^{}]*[\u{4e00}-\u{9fff}]+[^{}]*)\}").unwrap();
    result = chinese_text_pattern.replace_all(&result, "$1").to_string();

    // Remove quotation marks
    result = result.replace('"', "");

    // Define patterns for removing unnecessary spaces
    // Pattern 1: non-letter (not backslash) followed by space(s) followed by non-letter
    let re1 = Regex::new(r"([\W&&[^\\]])\s+?([\W_^\d])").unwrap();
    // Pattern 2: non-letter (not backslash) followed by space(s) followed by letter
    let re2 = Regex::new(r"([\W&&[^\\]])\s+?([a-zA-Z])").unwrap();
    // Pattern 3: letter followed by space(s) followed by non-letter
    let re3 = Regex::new(r"([a-zA-Z])\s+?([\W_^\d])").unwrap();

    // Iteratively remove spaces until convergence
    let mut prev_result = String::new();
    let max_iterations = 10; // Safety limit to prevent infinite loops
    let mut iterations = 0;

    while prev_result != result && iterations < max_iterations {
        prev_result = result.clone();
        result = re1.replace_all(&result, "$1$2").to_string();
        result = re2.replace_all(&result, "$1$2").to_string();
        result = re3.replace_all(&result, "$1$2").to_string();
        iterations += 1;
    }

    result.trim().to_string()
}
