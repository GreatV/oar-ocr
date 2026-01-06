//! UniMERNet-specific preprocessing functions
//!
//! This module implements the specific preprocessing logic for UniMERNet model,
//! which differs significantly from PP-FormulaNet's preprocessing.

use crate::core::{OCRError, Tensor4D};
use image::{ImageBuffer, Luma, Rgb, RgbImage, imageops};
use ndarray::{Array4, s};

/// Parameters for UniMERNet preprocessing
#[derive(Debug, Clone)]
pub struct UniMERNetPreprocessParams {
    /// Target size (width, height) - typically (672, 192) for UniMERNet
    pub target_size: (u32, u32),
    /// Threshold for cropping margins
    pub crop_threshold: u8,
    /// Padding multiple (32 for UniMERNet)
    pub padding_multiple: usize,
    /// Mean for normalization
    pub normalize_mean: [f32; 3],
    /// Std for normalization
    pub normalize_std: [f32; 3],
}

impl Default for UniMERNetPreprocessParams {
    fn default() -> Self {
        Self {
            target_size: (672, 192),
            crop_threshold: 200,
            padding_multiple: 32,
            normalize_mean: [0.7931, 0.7931, 0.7931],
            normalize_std: [0.1738, 0.1738, 0.1738],
        }
    }
}

/// UniMERNet preprocessor that matches Python's UniMERNetImgDecode logic
#[derive(Debug)]
pub struct UniMERNetPreprocessor {
    params: UniMERNetPreprocessParams,
}

impl UniMERNetPreprocessor {
    pub fn new(params: UniMERNetPreprocessParams) -> Self {
        Self { params }
    }

    /// Crop margin of the image based on grayscale thresholding
    /// This matches Python's UniMERNetImgDecode.crop_margin
    fn crop_margin(&self, img: &RgbImage) -> Result<RgbImage, OCRError> {
        // Convert to grayscale
        let gray = imageops::grayscale(img);
        let (width, height) = gray.dimensions();

        // Find min and max values
        let mut min_val = 255u8;
        let mut max_val = 0u8;
        for pixel in gray.pixels() {
            let val = pixel[0];
            if val < min_val {
                min_val = val;
            }
            if val > max_val {
                max_val = val;
            }
        }

        // If all pixels are the same, return original
        if min_val == max_val {
            return Ok(img.clone());
        }

        // Normalize to 0-255 range
        let mut binary = ImageBuffer::<Luma<u8>, Vec<u8>>::new(width, height);
        for (x, y, pixel) in binary.enumerate_pixels_mut() {
            let orig_val = gray.get_pixel(x, y)[0];
            // Normalize: (val - min) / (max - min) * 255
            let normalized = ((orig_val as f32 - min_val as f32)
                / (max_val as f32 - min_val as f32)
                * 255.0) as u8;
            // Binary threshold at 200 (matching Python)
            pixel[0] = if normalized < self.params.crop_threshold {
                255
            } else {
                0
            };
        }

        // Find bounding box of non-zero pixels
        let mut min_x = width;
        let mut max_x = 0;
        let mut min_y = height;
        let mut max_y = 0;
        let mut found = false;

        for (x, y, pixel) in binary.enumerate_pixels() {
            if pixel[0] == 255 {
                found = true;
                if x < min_x {
                    min_x = x;
                }
                if x > max_x {
                    max_x = x;
                }
                if y < min_y {
                    min_y = y;
                }
                if y > max_y {
                    max_y = y;
                }
            }
        }

        if !found {
            return Ok(img.clone());
        }

        // Crop the original image
        let cropped = imageops::crop_imm(img, min_x, min_y, max_x - min_x + 1, max_y - min_y + 1);
        Ok(cropped.to_image())
    }

    /// Resize image following UniMERNet's specific logic
    /// This matches Python's resize and thumbnail logic
    fn resize_unimernet(&self, img: &RgbImage) -> RgbImage {
        let (width, height) = img.dimensions();
        let (target_w, target_h) = self.params.target_size;

        // First, resize to have minimum dimension match the smaller of target dimensions
        let min_target = target_h.min(target_w);
        let scale = if width <= height {
            min_target as f32 / width as f32
        } else {
            min_target as f32 / height as f32
        };

        let new_width = (width as f32 * scale) as u32;
        let new_height = (height as f32 * scale) as u32;

        let mut resized = imageops::resize(
            img,
            new_width,
            new_height,
            imageops::FilterType::Triangle, // Bilinear
        );

        // Then apply thumbnail logic (don't exceed target dimensions)
        let (curr_w, curr_h) = resized.dimensions();
        if curr_w > target_w || curr_h > target_h {
            let scale_w = target_w as f32 / curr_w as f32;
            let scale_h = target_h as f32 / curr_h as f32;
            let scale = scale_w.min(scale_h);

            let final_w = (curr_w as f32 * scale) as u32;
            let final_h = (curr_h as f32 * scale) as u32;

            resized = imageops::resize(&resized, final_w, final_h, imageops::FilterType::Triangle);
        }

        resized
    }

    /// Add padding to reach target size
    fn add_padding(&self, img: &RgbImage) -> RgbImage {
        let (width, height) = img.dimensions();
        let (target_w, target_h) = self.params.target_size;

        let delta_width = target_w.saturating_sub(width);
        let delta_height = target_h.saturating_sub(height);

        // Center padding (matching Python's delta_width // 2)
        let pad_left = delta_width / 2;
        let pad_top = delta_height / 2;
        let _pad_right = delta_width - pad_left;
        let _pad_bottom = delta_height - pad_top;

        // Create a new image with target size, filled with white
        let mut padded = ImageBuffer::from_pixel(target_w, target_h, Rgb([255u8, 255u8, 255u8]));

        // Copy original image to the padded image
        for (x, y, pixel) in img.enumerate_pixels() {
            if x + pad_left < target_w && y + pad_top < target_h {
                padded.put_pixel(x + pad_left, y + pad_top, *pixel);
            }
        }

        padded
    }

    /// Process a single image following UniMERNet's preprocessing pipeline
    pub fn preprocess_single(&self, img: &RgbImage) -> Result<RgbImage, OCRError> {
        // Step 1: Crop margins
        let cropped = self.crop_margin(img)?;

        // Step 2: Resize with UniMERNet logic
        let resized = self.resize_unimernet(&cropped);

        // Step 3: Add padding to reach target size
        let padded = self.add_padding(&resized);

        Ok(padded)
    }

    /// Convert image to tensor and normalize
    fn image_to_tensor(&self, img: &RgbImage) -> Tensor4D {
        let (width, height) = img.dimensions();

        // Ensure dimensions are multiples of padding_multiple
        let padding_multiple = self.params.padding_multiple as u32;
        let padded_h = height.div_ceil(padding_multiple) * padding_multiple;
        let padded_w = width.div_ceil(padding_multiple) * padding_multiple;

        let mut tensor = Array4::<f32>::zeros((1, 1, padded_h as usize, padded_w as usize));

        // Convert to grayscale and normalize
        for y in 0..height {
            for x in 0..width {
                let pixel = img.get_pixel(x, y);
                // Convert to grayscale
                let gray =
                    (0.299 * pixel[0] as f32 + 0.587 * pixel[1] as f32 + 0.114 * pixel[2] as f32)
                        / 255.0;

                // Apply normalization
                let normalized =
                    (gray - self.params.normalize_mean[0]) / self.params.normalize_std[0];

                tensor[[0, 0, y as usize, x as usize]] = normalized;
            }
        }

        // Padding area remains as zeros (which corresponds to white after normalization)
        for y in height..padded_h {
            for x in 0..padded_w {
                let normalized =
                    (1.0 - self.params.normalize_mean[0]) / self.params.normalize_std[0];
                tensor[[0, 0, y as usize, x as usize]] = normalized;
            }
        }
        for x in width..padded_w {
            for y in 0..height {
                let normalized =
                    (1.0 - self.params.normalize_mean[0]) / self.params.normalize_std[0];
                tensor[[0, 0, y as usize, x as usize]] = normalized;
            }
        }

        tensor
    }

    /// Process a batch of images
    pub fn preprocess_batch(&self, images: &[RgbImage]) -> Result<Tensor4D, OCRError> {
        if images.is_empty() {
            return Err(OCRError::InvalidInput {
                message: "Empty image batch".to_string(),
            });
        }

        let mut batch_tensors = Vec::new();

        for img in images {
            // Preprocess the image
            let processed = self.preprocess_single(img)?;

            // Convert to tensor
            let tensor = self.image_to_tensor(&processed);
            batch_tensors.push(tensor);
        }

        // Concatenate along batch dimension
        let batch_size = batch_tensors.len();
        let shape = batch_tensors[0].shape();
        let mut result = Array4::<f32>::zeros((batch_size, shape[1], shape[2], shape[3]));

        for (i, tensor) in batch_tensors.iter().enumerate() {
            result.slice_mut(s![i..i + 1, .., .., ..]).assign(tensor);
        }

        Ok(result)
    }
}
