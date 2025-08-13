//! Image normalization utilities for OCR processing.
//!
//! This module provides functionality to normalize images for OCR processing,
//! including standard normalization with mean and standard deviation, as well as
//! specialized normalization for OCR recognition tasks.

use crate::core::OCRError;
use crate::processors::types::ChannelOrder;
use image::DynamicImage;
use rayon::prelude::*;

/// Normalizes images for OCR processing.
///
/// This struct encapsulates the parameters needed to normalize images,
/// including scaling factors, mean values, standard deviations, and channel ordering.
/// It provides methods to apply normalization to single images or batches of images.
#[derive(Debug)]
pub struct NormalizeImage {
    /// Scaling factors for each channel (alpha = scale / std)
    pub alpha: Vec<f32>,
    /// Offset values for each channel (beta = -mean / std)
    pub beta: Vec<f32>,
    /// Channel ordering (CHW or HWC)
    pub order: ChannelOrder,
}

impl NormalizeImage {
    /// Creates a new NormalizeImage instance with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `scale` - Optional scaling factor (defaults to 1.0/255.0)
    /// * `mean` - Optional mean values for each channel (defaults to [0.485, 0.456, 0.406])
    /// * `std` - Optional standard deviation values for each channel (defaults to [0.229, 0.224, 0.225])
    /// * `order` - Optional channel ordering (defaults to CHW)
    ///
    /// # Returns
    ///
    /// A Result containing the new NormalizeImage instance or an OCRError if validation fails.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * Scale is less than or equal to 0
    /// * Mean or std vectors don't have exactly 3 elements
    /// * Any standard deviation value is less than or equal to 0
    pub fn new(
        scale: Option<f32>,
        mean: Option<Vec<f32>>,
        std: Option<Vec<f32>>,
        order: Option<ChannelOrder>,
    ) -> Result<Self, OCRError> {
        let scale = scale.unwrap_or(1.0 / 255.0);
        let mean = mean.unwrap_or_else(|| vec![0.485, 0.456, 0.406]);
        let std = std.unwrap_or_else(|| vec![0.229, 0.224, 0.225]);
        let order = order.unwrap_or(ChannelOrder::CHW);

        if scale <= 0.0 {
            return Err(OCRError::ConfigError {
                message: "Scale must be greater than 0".to_string(),
            });
        }

        if mean.len() != 3 {
            return Err(OCRError::ConfigError {
                message: "Mean must have exactly 3 elements for RGB".to_string(),
            });
        }

        if std.len() != 3 {
            return Err(OCRError::ConfigError {
                message: "Std must have exactly 3 elements for RGB".to_string(),
            });
        }

        for (i, &s) in std.iter().enumerate() {
            if s <= 0.0 {
                return Err(OCRError::ConfigError {
                    message: format!(
                        "Standard deviation at index {i} must be greater than 0, got {s}"
                    ),
                });
            }
        }

        let alpha: Vec<f32> = std.iter().map(|s| scale / s).collect();
        let beta: Vec<f32> = mean.iter().zip(&std).map(|(m, s)| -m / s).collect();

        Ok(Self { alpha, beta, order })
    }

    /// Validates the configuration of the NormalizeImage instance.
    ///
    /// # Returns
    ///
    /// A Result indicating success or an OCRError if validation fails.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * Alpha or beta vectors don't have exactly 3 elements
    /// * Any alpha or beta value is not finite
    pub fn validate_config(&self) -> Result<(), OCRError> {
        if self.alpha.len() != 3 || self.beta.len() != 3 {
            return Err(OCRError::ConfigError {
                message: "Alpha and beta must have exactly 3 elements for RGB".to_string(),
            });
        }

        for (i, &alpha) in self.alpha.iter().enumerate() {
            if !alpha.is_finite() {
                return Err(OCRError::ConfigError {
                    message: format!("Alpha value at index {i} is not finite: {alpha}"),
                });
            }
        }

        for (i, &beta) in self.beta.iter().enumerate() {
            if !beta.is_finite() {
                return Err(OCRError::ConfigError {
                    message: format!("Beta value at index {i} is not finite: {beta}"),
                });
            }
        }

        Ok(())
    }

    /// Creates a NormalizeImage instance with parameters suitable for OCR recognition.
    ///
    /// This creates a normalization configuration with:
    /// * Scale: 2.0/255.0
    /// * Mean: [1.0, 1.0, 1.0]
    /// * Std: [1.0, 1.0, 1.0]
    /// * Order: CHW
    ///
    /// # Returns
    ///
    /// A Result containing the new NormalizeImage instance or an OCRError.
    pub fn for_ocr_recognition() -> Result<Self, OCRError> {
        Self::new(
            Some(2.0 / 255.0),
            Some(vec![1.0, 1.0, 1.0]),
            Some(vec![1.0, 1.0, 1.0]),
            Some(ChannelOrder::CHW),
        )
    }

    /// Applies normalization to a vector of images.
    ///
    /// # Arguments
    ///
    /// * `imgs` - A vector of DynamicImage instances to normalize
    ///
    /// # Returns
    ///
    /// A vector of normalized images represented as vectors of f32 values
    pub fn apply(&self, imgs: Vec<DynamicImage>) -> Vec<Vec<f32>> {
        imgs.into_iter().map(|img| self.normalize(img)).collect()
    }

    /// Validates inputs for batch processing operations.
    ///
    /// # Arguments
    ///
    /// * `imgs_len` - Number of images in the batch
    /// * `shapes` - Shapes of the images as (channels, height, width) tuples
    /// * `batch_tensor` - The batch tensor to validate against
    ///
    /// # Returns
    ///
    /// A Result containing a tuple of (batch_size, channels, height, max_width) or an OCRError.
    fn validate_batch_inputs(
        &self,
        imgs_len: usize,
        shapes: &[(usize, usize, usize)],
        batch_tensor: &[f32],
    ) -> Result<(usize, usize, usize, usize), OCRError> {
        if imgs_len != shapes.len() {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "Images and shapes length mismatch: {} images vs {} shapes",
                    imgs_len,
                    shapes.len()
                ),
            });
        }

        let batch_size = imgs_len;
        if batch_size == 0 {
            return Ok((0, 0, 0, 0));
        }

        let max_width = shapes.iter().map(|(_, _, w)| *w).max().unwrap_or(0);
        let channels = shapes.first().map(|(c, _, _)| *c).unwrap_or(0);
        let height = shapes.first().map(|(_, h, _)| *h).unwrap_or(0);
        let img_size = channels * height * max_width;

        if batch_tensor.len() < batch_size * img_size {
            return Err(OCRError::BufferTooSmall {
                expected: batch_size * img_size,
                actual: batch_tensor.len(),
            });
        }

        Ok((batch_size, channels, height, max_width))
    }

    /// Applies normalization to a batch of images and stores the result in a pre-allocated tensor.
    ///
    /// # Arguments
    ///
    /// * `imgs` - A vector of DynamicImage instances to normalize
    /// * `batch_tensor` - A mutable slice where the normalized batch will be stored
    /// * `shapes` - Shapes of the images as (channels, height, width) tuples
    ///
    /// # Returns
    ///
    /// A Result indicating success or an OCRError if validation fails.
    pub fn apply_to_batch(
        &self,
        imgs: Vec<DynamicImage>,
        batch_tensor: &mut [f32],
        shapes: &[(usize, usize, usize)],
    ) -> Result<(), OCRError> {
        let (batch_size, channels, height, max_width) =
            self.validate_batch_inputs(imgs.len(), shapes, batch_tensor)?;

        if batch_size == 0 {
            return Ok(());
        }

        let img_size = channels * height * max_width;

        for (batch_idx, (img, &(_c, h, w))) in imgs.into_iter().zip(shapes.iter()).enumerate() {
            let normalized_img = self.normalize(img);

            let batch_offset = batch_idx * img_size;

            for ch in 0.._c {
                for y in 0..h {
                    for x in 0..w {
                        let src_idx = ch * h * w + y * w + x;
                        let dst_idx = batch_offset + ch * height * max_width + y * max_width + x;
                        if src_idx < normalized_img.len() && dst_idx < batch_tensor.len() {
                            batch_tensor[dst_idx] = normalized_img[src_idx];
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Applies normalization to a batch of images and stores the result in a pre-allocated tensor,
    /// processing images in a streaming fashion.
    ///
    /// # Arguments
    ///
    /// * `imgs` - A vector of DynamicImage instances to normalize
    /// * `batch_tensor` - A mutable slice where the normalized batch will be stored
    /// * `shapes` - Shapes of the images as (channels, height, width) tuples
    ///
    /// # Returns
    ///
    /// A Result indicating success or an OCRError if validation fails.
    pub fn normalize_streaming_to_batch(
        &self,
        imgs: Vec<DynamicImage>,
        batch_tensor: &mut [f32],
        shapes: &[(usize, usize, usize)],
    ) -> Result<(), OCRError> {
        let (batch_size, channels, height, max_width) =
            self.validate_batch_inputs(imgs.len(), shapes, batch_tensor)?;

        if batch_size == 0 {
            return Ok(());
        }

        let img_size = channels * height * max_width;
        batch_tensor.fill(0.0);

        for (batch_idx, (img, &(_c, h, w))) in imgs.into_iter().zip(shapes.iter()).enumerate() {
            let rgb_img = img.to_rgb8();
            let (width, height_img) = rgb_img.dimensions();
            let batch_offset = batch_idx * img_size;

            match self.order {
                ChannelOrder::CHW => {
                    for c in 0..channels.min(3) {
                        for y in 0..h.min(height_img as usize) {
                            for x in 0..w.min(width as usize) {
                                let pixel = rgb_img.get_pixel(x as u32, y as u32);
                                let channel_value = pixel[c] as f32;
                                let dst_idx =
                                    batch_offset + c * height * max_width + y * max_width + x;
                                if dst_idx < batch_tensor.len() {
                                    batch_tensor[dst_idx] =
                                        channel_value * self.alpha[c] + self.beta[c];
                                }
                            }
                        }
                    }
                }
                ChannelOrder::HWC => {
                    for y in 0..h.min(height_img as usize) {
                        for x in 0..w.min(width as usize) {
                            let pixel = rgb_img.get_pixel(x as u32, y as u32);
                            for c in 0..channels.min(3) {
                                let channel_value = pixel[c] as f32;
                                let dst_idx =
                                    batch_offset + y * max_width * channels + x * channels + c;
                                if dst_idx < batch_tensor.len() {
                                    batch_tensor[dst_idx] =
                                        channel_value * self.alpha[c] + self.beta[c];
                                }
                            }
                        }
                    }
                }
            }
        }

        Ok(())
    }

    /// Normalizes a single image.
    ///
    /// # Arguments
    ///
    /// * `img` - The DynamicImage to normalize
    ///
    /// # Returns
    ///
    /// A vector of normalized pixel values as f32
    fn normalize(&self, img: DynamicImage) -> Vec<f32> {
        let rgb_img = img.to_rgb8();
        let (width, height) = rgb_img.dimensions();
        let channels = 3;

        match self.order {
            ChannelOrder::CHW => {
                let mut result = vec![0.0f32; (channels * height * width) as usize];

                for c in 0..channels {
                    for y in 0..height {
                        for x in 0..width {
                            let pixel = rgb_img.get_pixel(x, y);
                            let channel_value = pixel[c as usize] as f32;
                            let dst_idx = (c * height * width + y * width + x) as usize;

                            result[dst_idx] =
                                channel_value * self.alpha[c as usize] + self.beta[c as usize];
                        }
                    }
                }
                result
            }
            ChannelOrder::HWC => {
                let mut result = vec![0.0f32; (height * width * channels) as usize];

                for y in 0..height {
                    for x in 0..width {
                        let pixel = rgb_img.get_pixel(x, y);
                        for c in 0..channels {
                            let channel_value = pixel[c as usize] as f32;
                            let dst_idx = (y * width * channels + x * channels + c) as usize;

                            result[dst_idx] =
                                channel_value * self.alpha[c as usize] + self.beta[c as usize];
                        }
                    }
                }
                result
            }
        }
    }

    /// Normalizes a single image and returns it as a 4D tensor.
    ///
    /// # Arguments
    ///
    /// * `img` - The DynamicImage to normalize
    ///
    /// # Returns
    ///
    /// A Result containing the normalized image as a 4D tensor or an OCRError.
    pub fn normalize_to(&self, img: DynamicImage) -> Result<crate::core::Tensor4D, OCRError> {
        let rgb_img = img.to_rgb8();
        let (width, height) = rgb_img.dimensions();
        let channels = 3;

        match self.order {
            ChannelOrder::CHW => {
                let mut result = vec![0.0f32; (channels * height * width) as usize];

                for c in 0..channels {
                    for y in 0..height {
                        for x in 0..width {
                            let pixel = rgb_img.get_pixel(x, y);
                            let channel_value = pixel[c as usize] as f32;
                            let dst_idx = (c * height * width + y * width + x) as usize;

                            result[dst_idx] =
                                channel_value * self.alpha[c as usize] + self.beta[c as usize];
                        }
                    }
                }

                ndarray::Array4::from_shape_vec(
                    (1, channels as usize, height as usize, width as usize),
                    result.clone(),
                )
                .map_err(|e| {
                    OCRError::tensor_operation_error(
                        "normalization_tensor_creation_chw",
                        &[1, channels as usize, height as usize, width as usize],
                        &[result.len()],
                        &format!("Failed to create CHW normalization tensor for {}x{} image with {} channels",
                            width, height, channels),
                        e,
                    )
                })
            }
            ChannelOrder::HWC => {
                let mut result = vec![0.0f32; (height * width * channels) as usize];

                for y in 0..height {
                    for x in 0..width {
                        let pixel = rgb_img.get_pixel(x, y);
                        for c in 0..channels {
                            let channel_value = pixel[c as usize] as f32;
                            let dst_idx = (y * width * channels + x * channels + c) as usize;

                            result[dst_idx] =
                                channel_value * self.alpha[c as usize] + self.beta[c as usize];
                        }
                    }
                }

                ndarray::Array4::from_shape_vec(
                    (1, height as usize, width as usize, channels as usize),
                    result.clone(),
                )
                .map_err(|e| {
                    OCRError::tensor_operation_error(
                        "normalization_tensor_creation_hwc",
                        &[1, height as usize, width as usize, channels as usize],
                        &[result.len()],
                        &format!("Failed to create HWC normalization tensor for {}x{} image with {} channels",
                            width, height, channels),
                        e,
                    )
                })
            }
        }
    }

    /// Normalizes a batch of images and returns them as a 4D tensor.
    ///
    /// # Arguments
    ///
    /// * `imgs` - A vector of DynamicImage instances to normalize
    ///
    /// # Returns
    ///
    /// A Result containing the normalized batch as a 4D tensor or an OCRError.
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// * Images in the batch don't all have the same dimensions
    pub fn normalize_batch_to(
        &self,
        imgs: Vec<DynamicImage>,
    ) -> Result<crate::core::Tensor4D, OCRError> {
        if imgs.is_empty() {
            return Ok(ndarray::Array4::zeros((0, 0, 0, 0)));
        }

        let batch_size = imgs.len();

        let rgb_imgs: Vec<_> = imgs.into_iter().map(|img| img.to_rgb8()).collect();
        let dimensions: Vec<_> = rgb_imgs.iter().map(|img| img.dimensions()).collect();

        let (first_width, first_height) = dimensions.first().copied().unwrap_or((0, 0));
        for (i, &(width, height)) in dimensions.iter().enumerate() {
            if width != first_width || height != first_height {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "All images in batch must have the same dimensions. Image 0: {first_width}x{first_height}, Image {i}: {width}x{height}"
                    ),
                });
            }
        }

        let (width, height) = (first_width, first_height);
        let channels = 3;

        match self.order {
            ChannelOrder::CHW => {
                let mut result = vec![0.0f32; batch_size * (channels * height * width) as usize];

                let img_size = (channels * height * width) as usize;
                if batch_size <= 1 {
                    // Avoid rayon overhead for single-image batches
                    let rgb_img = &rgb_imgs[0];
                    let batch_slice = &mut result[0..img_size];
                    for c in 0..channels {
                        for y in 0..height {
                            for x in 0..width {
                                let pixel = rgb_img.get_pixel(x, y);
                                let channel_value = pixel[c as usize] as f32;
                                let dst_idx = (c * height * width + y * width + x) as usize;
                                batch_slice[dst_idx] =
                                    channel_value * self.alpha[c as usize] + self.beta[c as usize];
                            }
                        }
                    }
                } else {
                    result.par_chunks_mut(img_size).enumerate().for_each(
                        |(batch_idx, batch_slice)| {
                            let rgb_img = &rgb_imgs[batch_idx];
                            for c in 0..channels {
                                for y in 0..height {
                                    for x in 0..width {
                                        let pixel = rgb_img.get_pixel(x, y);
                                        let channel_value = pixel[c as usize] as f32;
                                        let dst_idx = (c * height * width + y * width + x) as usize;
                                        batch_slice[dst_idx] = channel_value
                                            * self.alpha[c as usize]
                                            + self.beta[c as usize];
                                    }
                                }
                            }
                        },
                    );
                }

                ndarray::Array4::from_shape_vec(
                    (
                        batch_size,
                        channels as usize,
                        height as usize,
                        width as usize,
                    ),
                    result,
                )
                .map_err(|e| {
                    OCRError::tensor_operation(
                        "Failed to create batch normalization tensor in CHW format",
                        e,
                    )
                })
            }
            ChannelOrder::HWC => {
                let mut result = vec![0.0f32; batch_size * (height * width * channels) as usize];

                let img_size = (height * width * channels) as usize;
                if batch_size <= 1 {
                    // Avoid rayon overhead for single-image batches
                    let rgb_img = &rgb_imgs[0];
                    let batch_slice = &mut result[0..img_size];
                    for y in 0..height {
                        for x in 0..width {
                            let pixel = rgb_img.get_pixel(x, y);
                            for c in 0..channels {
                                let channel_value = pixel[c as usize] as f32;
                                let dst_idx = (y * width * channels + x * channels + c) as usize;
                                batch_slice[dst_idx] =
                                    channel_value * self.alpha[c as usize] + self.beta[c as usize];
                            }
                        }
                    }
                } else {
                    result.par_chunks_mut(img_size).enumerate().for_each(
                        |(batch_idx, batch_slice)| {
                            let rgb_img = &rgb_imgs[batch_idx];
                            for y in 0..height {
                                for x in 0..width {
                                    let pixel = rgb_img.get_pixel(x, y);
                                    for c in 0..channels {
                                        let channel_value = pixel[c as usize] as f32;
                                        let dst_idx =
                                            (y * width * channels + x * channels + c) as usize;
                                        batch_slice[dst_idx] = channel_value
                                            * self.alpha[c as usize]
                                            + self.beta[c as usize];
                                    }
                                }
                            }
                        },
                    );
                }

                ndarray::Array4::from_shape_vec(
                    (
                        batch_size,
                        height as usize,
                        width as usize,
                        channels as usize,
                    ),
                    result,
                )
                .map_err(|e| {
                    OCRError::tensor_operation(
                        "Failed to create batch normalization tensor in HWC format",
                        e,
                    )
                })
            }
        }
    }
}
