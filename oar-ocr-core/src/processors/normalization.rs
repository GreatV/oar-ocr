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
    /// * `color_order` - Optional color channel order (defaults to RGB)
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
        let mut result = vec![0.0f32; Self::image_len(width, height, channels)];
        self.normalize_rgb_into(rgb_img, &mut result);
        result
    }

    /// Normalizes `rgb_img` into a pre-allocated, tightly packed output buffer.
    ///
    /// `out` must have length `width * height * 3` and is laid out per
    /// [`self.order`](TensorLayout). Writes through the SIMD kernels (see
    /// [`crate::processors::simd`]) operating on the raw interleaved RGB bytes,
    /// avoiding per-pixel `get_pixel`/`pixels()` overhead.
    fn normalize_rgb_into(&self, rgb_img: &RgbImage, out: &mut [f32]) {
        let (width, height) = rgb_img.dimensions();
        let (width, height) = (width as usize, height as usize);
        let src_channels = self.src_channels();
        let alpha = [self.alpha[0], self.alpha[1], self.alpha[2]];
        let beta = [self.beta[0], self.beta[1], self.beta[2]];
        let rgb = rgb_img.as_raw();

        match self.order {
            TensorLayout::CHW => crate::processors::simd::normalize_chw_into(
                rgb,
                width,
                height,
                src_channels,
                &alpha,
                &beta,
                out,
            ),
            TensorLayout::HWC => crate::processors::simd::normalize_hwc_into(
                rgb,
                width,
                height,
                src_channels,
                &alpha,
                &beta,
                out,
            ),
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
        let img_size = Self::image_len(width, height, channels);
        let mut result = vec![0.0f32; batch_size * img_size];

        // Parallelism is gated by total batch output size (not per-image size)
        // so tiny OCR crops stay serial unless the batch is large enough to
        // amortize rayon overhead. Each image is normalized into its own
        // contiguous `img_size` slice via the shared SIMD-backed kernel.
        let use_parallel =
            Self::should_parallelize(batch_size, result.len() * std::mem::size_of::<f32>());
        if !use_parallel {
            for (rgb_img, batch_slice) in rgb_imgs.iter().zip(result.chunks_mut(img_size)) {
                self.normalize_rgb_into(rgb_img, batch_slice);
            }
        } else {
            result
                .par_chunks_mut(img_size)
                .zip(rgb_imgs.par_iter())
                .for_each(|(batch_slice, rgb_img)| {
                    self.normalize_rgb_into(rgb_img, batch_slice);
                });
        }

        let shape = match self.order {
            TensorLayout::CHW => (batch_size, channels, height as usize, width as usize),
            TensorLayout::HWC => (batch_size, height as usize, width as usize, channels),
        };
        ndarray::Array4::from_shape_vec(shape, result).map_err(|e| {
            OCRError::tensor_operation("Failed to create batch normalization tensor", e)
        })
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
