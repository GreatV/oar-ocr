//! Image normalization utilities for OCR processing.
//!
//! This module provides functionality to normalize images for OCR processing,
//! including standard normalization with mean and standard deviation, as well as
//! specialized normalization for OCR recognition tasks.

use crate::core::OCRError;
use crate::processors::types::{ColorOrder, TensorLayout};
use image::{DynamicImage, RgbImage};
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
    /// Tensor data layout (CHW or HWC)
    pub order: TensorLayout,
    /// Color channel order (RGB or BGR)
    pub color_order: ColorOrder,
}

impl NormalizeImage {
    const PARALLEL_NORMALIZE_MIN_BYTES: usize = 1_048_576;

    fn should_parallelize(batch_size: usize, total_output_bytes: usize) -> bool {
        batch_size > 1 && total_output_bytes > Self::PARALLEL_NORMALIZE_MIN_BYTES
    }

    fn src_channels(&self) -> [usize; 3] {
        match self.color_order {
            ColorOrder::RGB => [0, 1, 2],
            ColorOrder::BGR => [2, 1, 0],
        }
    }

    fn image_len(width: u32, height: u32, channels: usize) -> usize {
        width as usize * height as usize * channels
    }

    /// Creates a new NormalizeImage instance with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `scale` - Optional scaling factor (defaults to 1.0/255.0)
    /// * `mean` - Optional mean values for each channel (defaults to [0.485, 0.456, 0.406])
    /// * `std` - Optional standard deviation values for each channel (defaults to [0.229, 0.224, 0.225])
    /// * `order` - Optional tensor data layout (defaults to CHW)
    /// * `color_order` - Optional color channel order (defaults to BGR)
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
        order: Option<TensorLayout>,
        color_order: Option<ColorOrder>,
    ) -> Result<Self, OCRError> {
        Self::with_color_order(scale, mean, std, order, color_order)
    }

    /// Creates a new NormalizeImage instance with the specified parameters including color order.
    ///
    /// # Arguments
    ///
    /// * `scale` - Optional scaling factor (defaults to 1.0/255.0)
    /// * `mean` - Optional mean values for each channel (defaults to [0.485, 0.456, 0.406])
    /// * `std` - Optional standard deviation values for each channel (defaults to [0.229, 0.224, 0.225])
    /// * `order` - Optional tensor data layout (defaults to CHW)
    /// * `color_order` - Optional color channel order (defaults to RGB)
    ///
    /// # Mean/Std Semantics
    ///
    /// `mean` and `std` must be provided in the **output channel order** specified by `color_order`.
    /// For example, if `color_order` is BGR, pass mean/std as `[B_mean, G_mean, R_mean]`.
    ///
    /// **Note:** This function does not validate that mean/std values match the specified
    /// `color_order`. Ensuring consistency is the caller's responsibility. If you have stats
    /// expressed in RGB order but need BGR output, prefer using
    /// [`NormalizeImage::with_color_order_from_rgb_stats`] or
    /// [`NormalizeImage::imagenet_bgr_from_rgb_stats`] which handle the reordering automatically.
    ///
    /// # Returns
    ///
    /// A Result containing the new NormalizeImage instance or an OCRError if validation fails.
    pub fn with_color_order(
        scale: Option<f32>,
        mean: Option<Vec<f32>>,
        std: Option<Vec<f32>>,
        order: Option<TensorLayout>,
        color_order: Option<ColorOrder>,
    ) -> Result<Self, OCRError> {
        let scale = scale.unwrap_or(1.0 / 255.0);
        let mean = mean.unwrap_or_else(|| vec![0.485, 0.456, 0.406]);
        let std = std.unwrap_or_else(|| vec![0.229, 0.224, 0.225]);
        let order = order.unwrap_or(TensorLayout::CHW);
        let color_order = color_order.unwrap_or_default();

        if scale <= 0.0 {
            return Err(OCRError::ConfigError {
                message: "Scale must be greater than 0".to_string(),
            });
        }

        if mean.len() != 3 {
            return Err(OCRError::ConfigError {
                message: "Mean must have exactly 3 elements (3-channel normalization)".to_string(),
            });
        }

        if std.len() != 3 {
            return Err(OCRError::ConfigError {
                message: "Std must have exactly 3 elements (3-channel normalization)".to_string(),
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

        Ok(Self {
            alpha,
            beta,
            order,
            color_order,
        })
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
                message: "Alpha and beta must have exactly 3 elements (3-channel normalization)"
                    .to_string(),
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
            Some(TensorLayout::CHW),
            Some(ColorOrder::BGR),
        )
    }

    /// Creates an ImageNet-style RGB normalizer (mean/std in RGB order).
    pub fn imagenet_rgb() -> Result<Self, OCRError> {
        Self::with_color_order(
            None,
            Some(vec![0.485, 0.456, 0.406]),
            Some(vec![0.229, 0.224, 0.225]),
            Some(TensorLayout::CHW),
            Some(ColorOrder::RGB),
        )
    }

    /// Creates an ImageNet-style BGR normalizer from RGB stats.
    ///
    /// This is useful for PaddlePaddle-exported models that expect BGR input,
    /// while configuration commonly provides ImageNet mean/std in RGB order.
    pub fn imagenet_bgr_from_rgb_stats() -> Result<Self, OCRError> {
        Self::with_color_order(
            None,
            Some(vec![0.406, 0.456, 0.485]),
            Some(vec![0.225, 0.224, 0.229]),
            Some(TensorLayout::CHW),
            Some(ColorOrder::BGR),
        )
    }

    /// Builds a normalizer for a given output `color_order` using RGB mean/std stats.
    ///
    /// Invariant: `mean`/`std` passed to `with_color_order` are interpreted in the output channel
    /// order (`ColorOrder`). This helper makes the conversion explicit at call sites.
    pub fn with_color_order_from_rgb_stats(
        scale: Option<f32>,
        mean_rgb: Vec<f32>,
        std_rgb: Vec<f32>,
        order: Option<TensorLayout>,
        output_color_order: ColorOrder,
    ) -> Result<Self, OCRError> {
        if mean_rgb.len() != 3 || std_rgb.len() != 3 {
            return Err(OCRError::ConfigError {
                message: format!(
                    "mean/std must have exactly 3 elements (got mean={}, std={})",
                    mean_rgb.len(),
                    std_rgb.len()
                ),
            });
        }

        let (mean, std) = match output_color_order {
            ColorOrder::RGB => (mean_rgb, std_rgb),
            ColorOrder::BGR => (
                vec![mean_rgb[2], mean_rgb[1], mean_rgb[0]],
                vec![std_rgb[2], std_rgb[1], std_rgb[0]],
            ),
        };

        Self::with_color_order(
            scale,
            Some(mean),
            Some(std),
            order,
            Some(output_color_order),
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

        // Pre-compute channel mapping for BGR support
        let src_channels: [usize; 3] = match self.color_order {
            ColorOrder::RGB => [0, 1, 2],
            ColorOrder::BGR => [2, 1, 0],
        };

        for (batch_idx, (img, &(_c, h, w))) in imgs.into_iter().zip(shapes.iter()).enumerate() {
            let rgb_img = img.to_rgb8();
            let (width, height_img) = rgb_img.dimensions();
            let batch_offset = batch_idx * img_size;

            match self.order {
                TensorLayout::CHW => {
                    for (c, &src_c) in src_channels.iter().enumerate().take(channels.min(3)) {
                        for y in 0..h.min(height_img as usize) {
                            for x in 0..w.min(width as usize) {
                                let pixel = rgb_img.get_pixel(x as u32, y as u32);
                                let channel_value = pixel[src_c] as f32;
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
                TensorLayout::HWC => {
                    for y in 0..h.min(height_img as usize) {
                        for x in 0..w.min(width as usize) {
                            let pixel = rgb_img.get_pixel(x as u32, y as u32);
                            for (c, &src_c) in src_channels.iter().enumerate().take(channels.min(3))
                            {
                                let channel_value = pixel[src_c] as f32;
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
        let rgb_img = into_rgb8_no_copy(img);
        self.normalize_rgb(&rgb_img)
    }

    fn normalize_rgb(&self, rgb_img: &RgbImage) -> Vec<f32> {
        let (width, height) = rgb_img.dimensions();
        let channels = 3usize;
        let src_channels = self.src_channels();
        let mut result = vec![0.0f32; Self::image_len(width, height, channels)];

        match self.order {
            TensorLayout::CHW => {
                let plane = width as usize * height as usize;
                for (pixel_idx, pixel) in rgb_img.pixels().enumerate() {
                    for (c, &src_c) in src_channels.iter().enumerate() {
                        result[c * plane + pixel_idx] =
                            pixel[src_c] as f32 * self.alpha[c] + self.beta[c];
                    }
                }
                result
            }
            TensorLayout::HWC => {
                for (pixel_idx, pixel) in rgb_img.pixels().enumerate() {
                    let dst_base = pixel_idx * channels;
                    for (c, &src_c) in src_channels.iter().enumerate() {
                        result[dst_base + c] = pixel[src_c] as f32 * self.alpha[c] + self.beta[c];
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
    pub fn normalize_to(&self, img: DynamicImage) -> Result<ndarray::Array4<f32>, OCRError> {
        let rgb_img = into_rgb8_no_copy(img);
        let (width, height) = rgb_img.dimensions();
        let channels = 3usize;
        let image_len = Self::image_len(width, height, channels);

        match self.order {
            TensorLayout::CHW => {
                let result = self.normalize_rgb(&rgb_img);

                ndarray::Array4::from_shape_vec(
                    (1, channels, height as usize, width as usize),
                    result,
                )
                .map_err(|e| {
                    OCRError::tensor_operation_error(
                        "normalization_tensor_creation_chw",
                        &[1, channels, height as usize, width as usize],
                        &[image_len],
                        &format!("Failed to create CHW normalization tensor for {}x{} image with {} channels",
                            width, height, channels),
                        e,
                    )
                })
            }
            TensorLayout::HWC => {
                let result = self.normalize_rgb(&rgb_img);

                ndarray::Array4::from_shape_vec(
                    (1, height as usize, width as usize, channels),
                    result,
                )
                .map_err(|e| {
                    OCRError::tensor_operation_error(
                        "normalization_tensor_creation_hwc",
                        &[1, height as usize, width as usize, channels],
                        &[image_len],
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
    ) -> Result<ndarray::Array4<f32>, OCRError> {
        if imgs.is_empty() {
            return Ok(ndarray::Array4::zeros((0, 0, 0, 0)));
        }

        let batch_size = imgs.len();

        let rgb_imgs: Vec<_> = imgs.into_iter().map(into_rgb8_no_copy).collect();
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
        let channels = 3usize;

        let src_channels = self.src_channels();

        // Clone alpha/beta for parallel closure
        let alpha = self.alpha.clone();
        let beta = self.beta.clone();

        match self.order {
            TensorLayout::CHW => {
                let img_size = Self::image_len(width, height, channels);
                let plane = width as usize * height as usize;
                let mut result = vec![0.0f32; batch_size * img_size];

                // The threshold is based on total output size for the whole batch rather than
                // per-image size. This keeps tiny batches serial even when the batch has multiple
                // images, and avoids rayon overhead for common OCR crops.
                let use_parallel =
                    Self::should_parallelize(batch_size, result.len() * std::mem::size_of::<f32>());
                if !use_parallel {
                    for (batch_idx, rgb_img) in rgb_imgs.iter().enumerate() {
                        let batch_offset = batch_idx * img_size;
                        let batch_slice = &mut result[batch_offset..batch_offset + img_size];
                        for (pixel_idx, pixel) in rgb_img.pixels().enumerate() {
                            for (c, &src_c) in src_channels.iter().enumerate() {
                                batch_slice[c * plane + pixel_idx] =
                                    pixel[src_c] as f32 * alpha[c] + beta[c];
                            }
                        }
                    }
                } else {
                    result.par_chunks_mut(img_size).enumerate().for_each(
                        |(batch_idx, batch_slice)| {
                            let rgb_img = &rgb_imgs[batch_idx];
                            for (pixel_idx, pixel) in rgb_img.pixels().enumerate() {
                                for (c, &src_c) in src_channels.iter().enumerate() {
                                    batch_slice[c * plane + pixel_idx] =
                                        pixel[src_c] as f32 * alpha[c] + beta[c];
                                }
                            }
                        },
                    );
                }

                ndarray::Array4::from_shape_vec(
                    (batch_size, channels, height as usize, width as usize),
                    result,
                )
                .map_err(|e| {
                    OCRError::tensor_operation(
                        "Failed to create batch normalization tensor in CHW format",
                        e,
                    )
                })
            }
            TensorLayout::HWC => {
                let img_size = Self::image_len(width, height, channels);
                let mut result = vec![0.0f32; batch_size * img_size];

                // Match the CHW path: parallelism is gated by total batch output size so that
                // small OCR crops stay serial unless the batch is large enough to amortize rayon.
                let use_parallel =
                    Self::should_parallelize(batch_size, result.len() * std::mem::size_of::<f32>());
                if !use_parallel {
                    for (batch_idx, rgb_img) in rgb_imgs.iter().enumerate() {
                        let batch_offset = batch_idx * img_size;
                        let batch_slice = &mut result[batch_offset..batch_offset + img_size];
                        for (pixel_idx, pixel) in rgb_img.pixels().enumerate() {
                            let dst_base = pixel_idx * channels;
                            for (c, &src_c) in src_channels.iter().enumerate() {
                                batch_slice[dst_base + c] =
                                    pixel[src_c] as f32 * alpha[c] + beta[c];
                            }
                        }
                    }
                } else {
                    result.par_chunks_mut(img_size).enumerate().for_each(
                        |(batch_idx, batch_slice)| {
                            let rgb_img = &rgb_imgs[batch_idx];
                            for (pixel_idx, pixel) in rgb_img.pixels().enumerate() {
                                let dst_base = pixel_idx * channels;
                                for (c, &src_c) in src_channels.iter().enumerate() {
                                    batch_slice[dst_base + c] =
                                        pixel[src_c] as f32 * alpha[c] + beta[c];
                                }
                            }
                        },
                    );
                }

                ndarray::Array4::from_shape_vec(
                    (batch_size, height as usize, width as usize, channels),
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

fn into_rgb8_no_copy(img: DynamicImage) -> RgbImage {
    match img {
        DynamicImage::ImageRgb8(img) => img,
        img => img.to_rgb8(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};
    use ndarray::Axis;

    #[test]
    fn test_normalize_image_color_order_rgb_vs_bgr_chw() -> Result<(), OCRError> {
        let mut img = RgbImage::new(1, 1);
        img.put_pixel(0, 0, Rgb([10, 20, 30])); // R, G, B

        let rgb = NormalizeImage::with_color_order(
            Some(1.0),
            Some(vec![0.0, 0.0, 0.0]),
            Some(vec![1.0, 1.0, 1.0]),
            Some(TensorLayout::CHW),
            Some(ColorOrder::RGB),
        )?;
        let bgr = NormalizeImage::with_color_order(
            Some(1.0),
            Some(vec![0.0, 0.0, 0.0]),
            Some(vec![1.0, 1.0, 1.0]),
            Some(TensorLayout::CHW),
            Some(ColorOrder::BGR),
        )?;

        let rgb_out = rgb.apply(vec![DynamicImage::ImageRgb8(img.clone())]);
        let bgr_out = bgr.apply(vec![DynamicImage::ImageRgb8(img)]);

        assert_eq!(rgb_out.len(), 1);
        assert_eq!(bgr_out.len(), 1);
        assert_eq!(rgb_out[0], vec![10.0, 20.0, 30.0]);
        assert_eq!(bgr_out[0], vec![30.0, 20.0, 10.0]);
        Ok(())
    }

    #[test]
    fn test_normalize_image_mean_std_applied_in_output_channel_order() -> Result<(), OCRError> {
        let mut img = RgbImage::new(1, 1);
        img.put_pixel(0, 0, Rgb([11, 22, 33])); // R, G, B

        let rgb = NormalizeImage::with_color_order(
            Some(1.0),
            Some(vec![1.0, 2.0, 3.0]), // RGB means
            Some(vec![2.0, 4.0, 5.0]), // RGB stds
            Some(TensorLayout::CHW),
            Some(ColorOrder::RGB),
        )?;
        let bgr = NormalizeImage::with_color_order(
            Some(1.0),
            Some(vec![3.0, 2.0, 1.0]), // BGR means
            Some(vec![5.0, 4.0, 2.0]), // BGR stds
            Some(TensorLayout::CHW),
            Some(ColorOrder::BGR),
        )?;

        let rgb_out = rgb.apply(vec![DynamicImage::ImageRgb8(img.clone())]);
        let bgr_out = bgr.apply(vec![DynamicImage::ImageRgb8(img)]);

        assert_eq!(rgb_out[0], vec![5.0, 5.0, 6.0]); // (R-1)/2, (G-2)/4, (B-3)/5
        assert_eq!(bgr_out[0], vec![6.0, 5.0, 5.0]); // (B-3)/5, (G-2)/4, (R-1)/2
        Ok(())
    }

    #[test]
    fn test_should_parallelize_threshold_behavior() {
        assert!(!NormalizeImage::should_parallelize(
            1,
            NormalizeImage::PARALLEL_NORMALIZE_MIN_BYTES * 4,
        ));
        assert!(!NormalizeImage::should_parallelize(
            4,
            NormalizeImage::PARALLEL_NORMALIZE_MIN_BYTES,
        ));
        assert!(NormalizeImage::should_parallelize(
            4,
            NormalizeImage::PARALLEL_NORMALIZE_MIN_BYTES + 1,
        ));
    }

    #[test]
    fn test_normalize_batch_to_matches_single_image_paths_for_serial_and_parallel()
    -> Result<(), OCRError> {
        let normalizer = NormalizeImage::with_color_order(
            Some(1.0),
            Some(vec![0.0, 0.0, 0.0]),
            Some(vec![1.0, 1.0, 1.0]),
            Some(TensorLayout::CHW),
            Some(ColorOrder::RGB),
        )?;

        let mut small_a = RgbImage::new(1, 1);
        small_a.put_pixel(0, 0, Rgb([1, 2, 3]));
        let mut small_b = RgbImage::new(1, 1);
        small_b.put_pixel(0, 0, Rgb([4, 5, 6]));
        let small_batch = vec![
            DynamicImage::ImageRgb8(small_a.clone()),
            DynamicImage::ImageRgb8(small_b.clone()),
        ];
        let serial = normalizer.normalize_batch_to(small_batch)?;
        let serial_expected = [
            normalizer.normalize_to(DynamicImage::ImageRgb8(small_a))?,
            normalizer.normalize_to(DynamicImage::ImageRgb8(small_b))?,
        ];

        assert_eq!(serial.len_of(Axis(0)), serial_expected.len());
        for (idx, expected) in serial_expected.iter().enumerate() {
            assert_eq!(
                serial.index_axis(Axis(0), idx).to_owned(),
                expected.index_axis(Axis(0), 0)
            );
        }

        let large_a =
            RgbImage::from_fn(512, 512, |x, y| Rgb([(x % 251) as u8, (y % 241) as u8, 7]));
        let large_b =
            RgbImage::from_fn(512, 512, |x, y| Rgb([11, (x % 239) as u8, (y % 233) as u8]));
        let parallel_batch = vec![
            DynamicImage::ImageRgb8(large_a.clone()),
            DynamicImage::ImageRgb8(large_b.clone()),
        ];
        let parallel = normalizer.normalize_batch_to(parallel_batch)?;
        let parallel_expected = [
            normalizer.normalize_to(DynamicImage::ImageRgb8(large_a))?,
            normalizer.normalize_to(DynamicImage::ImageRgb8(large_b))?,
        ];

        assert_eq!(parallel.len_of(Axis(0)), parallel_expected.len());
        for (idx, expected) in parallel_expected.iter().enumerate() {
            assert_eq!(
                parallel.index_axis(Axis(0), idx).to_owned(),
                expected.index_axis(Axis(0), 0)
            );
        }

        Ok(())
    }

    #[test]
    fn test_normalize_batch_to_preserves_batch_and_layout_semantics() -> Result<(), OCRError> {
        let chw = NormalizeImage::with_color_order(
            Some(1.0),
            Some(vec![0.0, 0.0, 0.0]),
            Some(vec![1.0, 1.0, 1.0]),
            Some(TensorLayout::CHW),
            Some(ColorOrder::RGB),
        )?;
        let hwc = NormalizeImage::with_color_order(
            Some(1.0),
            Some(vec![0.0, 0.0, 0.0]),
            Some(vec![1.0, 1.0, 1.0]),
            Some(TensorLayout::HWC),
            Some(ColorOrder::RGB),
        )?;

        let img_a = RgbImage::from_fn(2, 2, |x, y| {
            let base = (y * 2 + x) as u8 * 3 + 1;
            Rgb([base, base + 1, base + 2])
        });
        let img_b = RgbImage::from_fn(2, 2, |x, y| {
            let base = (y * 2 + x) as u8 * 3 + 21;
            Rgb([base, base + 1, base + 2])
        });

        let chw_batch = chw.normalize_batch_to(vec![
            DynamicImage::ImageRgb8(img_a.clone()),
            DynamicImage::ImageRgb8(img_b.clone()),
        ])?;
        assert_eq!(chw_batch.shape(), &[2, 3, 2, 2]);
        assert_eq!(
            chw_batch.iter().copied().collect::<Vec<_>>(),
            vec![
                1.0, 4.0, 7.0, 10.0, 2.0, 5.0, 8.0, 11.0, 3.0, 6.0, 9.0, 12.0, 21.0, 24.0, 27.0,
                30.0, 22.0, 25.0, 28.0, 31.0, 23.0, 26.0, 29.0, 32.0,
            ]
        );

        let hwc_batch = hwc.normalize_batch_to(vec![
            DynamicImage::ImageRgb8(img_a),
            DynamicImage::ImageRgb8(img_b),
        ])?;
        assert_eq!(hwc_batch.shape(), &[2, 2, 2, 3]);
        assert_eq!(
            hwc_batch.iter().copied().collect::<Vec<_>>(),
            vec![
                1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 21.0, 22.0, 23.0,
                24.0, 25.0, 26.0, 27.0, 28.0, 29.0, 30.0, 31.0, 32.0,
            ]
        );

        Ok(())
    }
}
