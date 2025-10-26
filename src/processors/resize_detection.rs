//! Image resizing utilities for OCR preprocessing
//!
//! This module provides functionality to resize images for OCR processing,
//! supporting different resizing strategies based on the requirements of
//! various OCR models. The main struct `DetResizeForTest` handles different
//! types of resizing operations.
//!
//! # Resize Types
//! - Type0: Resize based on limit side length with different behaviors depending on the limit type
//! - Type1: Resize to specific dimensions with optional aspect ratio preservation
//! - Type2: Resize long side to specific length
//! - Type3: Resize to specific input shape
//!
//! # Limit Types
//! - Max: Resize if the longest side exceeds the limit
//! - Min: Resize if the shortest side is below the limit
//! - ResizeLong: Resize the long side to match the limit

use crate::core::constants::{DEFAULT_LIMIT_SIDE_LEN, DEFAULT_MAX_SIDE_LIMIT};
use crate::processors::types::{ImageScaleInfo, LimitType, ResizeType};
use image::{DynamicImage, GenericImageView};
use tracing::{error, warn};

/// A struct for resizing images for OCR testing
///
/// This struct encapsulates different resizing strategies for preparing
/// images for OCR processing. It supports multiple resize types based
/// on the input parameters.
#[derive(Debug)]
pub struct DetResizeForTest {
    /// The type of resizing to perform
    pub resize_type: ResizeType,
    /// The length to limit the side of the image to (optional)
    pub limit_side_len: Option<u32>,
    /// The type of limit to apply (min, max, or resize long) (optional)
    pub limit_type: Option<LimitType>,
    /// The maximum allowed side length
    pub max_side_limit: u32,
}

impl DetResizeForTest {
    /// Creates a new `DetResizeForTest` instance
    ///
    /// This constructor determines the resize type based on the provided parameters.
    /// The resize type is determined in the following order:
    /// 1. If `input_shape` is provided, uses Type3
    /// 2. If `image_shape` is provided, uses Type1
    /// 3. If `resize_long` is provided, uses Type2
    /// 4. Otherwise, uses Type0 (default)
    ///
    /// # Parameters
    /// * `input_shape` - Optional input shape (channels, height, width)
    /// * `image_shape` - Optional target image shape (height, width)
    /// * `keep_ratio` - Whether to maintain aspect ratio when resizing (used with image_shape)
    /// * `limit_side_len` - Optional limit for side length
    /// * `limit_type` - Optional limit type (min, max, or resize long)
    /// * `resize_long` - Optional length to resize the long side to
    /// * `max_side_limit` - Optional maximum side length limit
    ///
    /// # Returns
    /// A new `DetResizeForTest` instance with the determined resize type and parameters
    pub fn new(
        input_shape: Option<(u32, u32, u32)>,
        image_shape: Option<(u32, u32)>,
        keep_ratio: Option<bool>,
        limit_side_len: Option<u32>,
        limit_type: Option<LimitType>,
        resize_long: Option<u32>,
        max_side_limit: Option<u32>,
    ) -> Self {
        // Determine resize type based on provided parameters
        // Priority order: Type3 -> Type1 -> Type2 -> Type0 (default)
        let resize_type = if let Some(shape) = input_shape {
            // Type3: Resize to specific input shape (channels, height, width)
            ResizeType::Type3 { input_shape: shape }
        } else if let Some(shape) = image_shape {
            // Type1: Resize to specific dimensions with optional aspect ratio preservation
            ResizeType::Type1 {
                image_shape: shape,
                keep_ratio: keep_ratio.unwrap_or(false),
            }
        } else if let Some(long) = resize_long {
            // Type2: Resize long side to specific length
            ResizeType::Type2 { resize_long: long }
        } else {
            // Type0: Resize based on limit side length (default)
            ResizeType::Type0
        };

        Self {
            resize_type,
            limit_side_len: limit_side_len.or(Some(DEFAULT_LIMIT_SIDE_LEN)),
            limit_type: limit_type.or(Some(LimitType::Min)),
            max_side_limit: max_side_limit.unwrap_or(DEFAULT_MAX_SIDE_LIMIT),
        }
    }

    /// Applies resizing to a batch of images
    ///
    /// This method processes a vector of images, applying the configured
    /// resize operation to each one.
    ///
    /// # Parameters
    /// * `imgs` - Vector of images to resize
    /// * `limit_side_len` - Optional override for limit side length
    /// * `limit_type` - Optional override for limit type
    /// * `max_side_limit` - Optional override for maximum side limit
    ///
    /// # Returns
    /// A tuple containing:
    /// 1. Vector of resized images
    /// 2. Vector of image scale information (original dimensions and resize ratios)
    pub fn apply(
        &self,
        imgs: Vec<DynamicImage>,
        limit_side_len: Option<u32>,
        limit_type: Option<LimitType>,
        max_side_limit: Option<u32>,
    ) -> (Vec<DynamicImage>, Vec<ImageScaleInfo>) {
        let mut resize_imgs = Vec::new();
        let mut img_shapes = Vec::new();

        // Process each image in the batch
        for img in imgs {
            let (resized_img, shape) =
                self.resize(img, limit_side_len, limit_type.as_ref(), max_side_limit);
            resize_imgs.push(resized_img);
            img_shapes.push(shape);
        }

        (resize_imgs, img_shapes)
    }

    /// Resizes a single image based on the configured resize type
    ///
    /// This method applies the appropriate resize operation based on the
    /// `resize_type` field. It also handles small images by padding them
    /// if their dimensions are less than 64 pixels in total.
    ///
    /// # Parameters
    /// * `img` - The image to resize
    /// * `limit_side_len` - Optional override for limit side length
    /// * `limit_type` - Optional override for limit type
    /// * `max_side_limit` - Optional override for maximum side limit
    ///
    /// # Returns
    /// A tuple containing:
    /// 1. The resized image
    /// 2. Image scale information (original dimensions and resize ratios)
    fn resize(
        &self,
        mut img: DynamicImage,
        limit_side_len: Option<u32>,
        limit_type: Option<&LimitType>,
        max_side_limit: Option<u32>,
    ) -> (DynamicImage, ImageScaleInfo) {
        let (src_w, src_h) = img.dimensions();

        // Pad small images to avoid issues with OCR processing
        // Images with total dimensions less than 64 pixels can cause problems in OCR models
        if (src_h + src_w) < 64 {
            img = self.image_padding(img);
        }

        let (resized_img, ratios) = match &self.resize_type {
            ResizeType::Type0 => {
                self.resize_image_type0(img, limit_side_len, limit_type, max_side_limit)
            }
            ResizeType::Type1 {
                image_shape,
                keep_ratio,
            } => self.resize_image_type1(img, *image_shape, *keep_ratio),
            ResizeType::Type2 { resize_long } => self.resize_image_type2(img, *resize_long),
            ResizeType::Type3 { input_shape } => self.resize_image_type3(img, *input_shape),
        };

        let scale_info = ImageScaleInfo::new(src_h as f32, src_w as f32, ratios[0], ratios[1]);
        (resized_img, scale_info)
    }

    /// Pads small images to a minimum size
    ///
    /// Ensures that images have a minimum dimension of 32x32 pixels
    /// by padding them with black pixels if needed.
    ///
    /// # Parameters
    /// * `img` - The image to pad
    ///
    /// # Returns
    /// The padded image (or original if no padding was needed)
    fn image_padding(&self, img: DynamicImage) -> DynamicImage {
        let (w, h) = img.dimensions();
        // Ensure minimum dimension of 32 pixels for both width and height
        let new_w = w.max(32);
        let new_h = h.max(32);

        // If image is already at least 32x32, return it unchanged
        if new_w == w && new_h == h {
            return img;
        }

        // Create a new image with the padded dimensions
        let mut padded = DynamicImage::new_rgb8(new_w, new_h);
        // Overlay the original image onto the padded image at position (0,0)
        image::imageops::overlay(&mut padded, &img, 0, 0);
        padded
    }

    /// Resize type 0: Resize based on limit side length
    ///
    /// This method resizes the image based on a limit for the side length,
    /// with different behaviors depending on the limit type:
    /// - Max: Resize if the longest side exceeds the limit
    /// - Min: Resize if the shortest side is below the limit
    /// - ResizeLong: Resize the long side to match the limit
    ///
    /// The resized dimensions are also adjusted to be multiples of 32,
    /// and constrained by the maximum side limit.
    ///
    /// # Parameters
    /// * `img` - The image to resize
    /// * `limit_side_len` - Optional override for limit side length
    /// * `limit_type` - Optional override for limit type
    /// * `max_side_limit` - Optional override for maximum side limit
    ///
    /// # Returns
    /// A tuple containing:
    /// 1. The resized image
    /// 2. Array with resize ratios [ratio_h, ratio_w]
    fn resize_image_type0(
        &self,
        img: DynamicImage,
        limit_side_len: Option<u32>,
        limit_type: Option<&LimitType>,
        max_side_limit: Option<u32>,
    ) -> (DynamicImage, [f32; 2]) {
        let (w, h) = img.dimensions();
        let limit_side_len = limit_side_len
            .or(self.limit_side_len)
            .unwrap_or(DEFAULT_LIMIT_SIDE_LEN);
        let limit_type = limit_type
            .or(self.limit_type.as_ref())
            .unwrap_or(&LimitType::Min);
        let max_side_limit = max_side_limit.unwrap_or(self.max_side_limit);

        // Calculate resize ratio based on limit type
        let ratio = match limit_type {
            LimitType::Max => {
                // Resize if the longest side exceeds the limit
                if h.max(w) > limit_side_len {
                    limit_side_len as f32 / h.max(w) as f32
                } else {
                    1.0
                }
            }
            LimitType::Min => {
                // Resize if the shortest side is below the limit
                if h.min(w) < limit_side_len {
                    limit_side_len as f32 / h.min(w) as f32
                } else {
                    1.0
                }
            }
            LimitType::ResizeLong => {
                // Resize the long side to match the limit
                limit_side_len as f32 / h.max(w) as f32
            }
        };

        let mut resize_h = (h as f32 * ratio) as u32;
        let mut resize_w = (w as f32 * ratio) as u32;

        // Apply maximum side limit if exceeded
        if resize_h.max(resize_w) > max_side_limit {
            warn!(
                "Resized image size ({}x{}) exceeds max_side_limit of {}. Resizing to fit within limit.",
                resize_h, resize_w, max_side_limit
            );
            // Calculate ratio to scale down to fit within max_side_limit
            let limit_ratio = max_side_limit as f32 / resize_h.max(resize_w) as f32;
            resize_h = (resize_h as f32 * limit_ratio) as u32;
            resize_w = (resize_w as f32 * limit_ratio) as u32;
        }

        // Ensure dimensions are multiples of 32 and at least 32 pixels
        // Adding 16 before division ensures proper rounding to nearest multiple of 32
        resize_h = ((resize_h + 16) / 32 * 32).max(32);
        resize_w = ((resize_w + 16) / 32 * 32).max(32);

        // Return original if no resize is needed
        if resize_h == h && resize_w == w {
            return (img, [1.0, 1.0]);
        }

        // Handle invalid resize dimensions
        if resize_w == 0 || resize_h == 0 {
            error!("Invalid resize dimensions: {}x{}", resize_w, resize_h);
            return (img, [1.0, 1.0]);
        }

        let resized_img =
            img.resize_exact(resize_w, resize_h, image::imageops::FilterType::Lanczos3);
        let ratio_h = resize_h as f32 / h as f32;
        let ratio_w = resize_w as f32 / w as f32;

        (resized_img, [ratio_h, ratio_w])
    }

    /// Resize type 1: Resize to specific dimensions
    ///
    /// This method resizes the image to specific dimensions, with an option
    /// to maintain the aspect ratio. When keeping the ratio, the width is
    /// adjusted to maintain the aspect ratio and then rounded up to the
    /// nearest multiple of 32.
    ///
    /// # Parameters
    /// * `img` - The image to resize
    /// * `image_shape` - Target dimensions (height, width)
    /// * `keep_ratio` - Whether to maintain aspect ratio
    ///
    /// # Returns
    /// A tuple containing:
    /// 1. The resized image
    /// 2. Array with resize ratios [ratio_h, ratio_w]
    fn resize_image_type1(
        &self,
        img: DynamicImage,
        image_shape: (u32, u32),
        keep_ratio: bool,
    ) -> (DynamicImage, [f32; 2]) {
        let (ori_w, ori_h) = img.dimensions();
        let (resize_h, mut resize_w) = image_shape;

        // Adjust width to maintain aspect ratio if requested
        if keep_ratio {
            // Calculate new width based on aspect ratio: new_width = original_width * (target_height / original_height)
            resize_w = (ori_w * resize_h) / ori_h;
            // Round up to nearest multiple of 32 to ensure proper alignment for OCR models
            let n = resize_w.div_ceil(32);
            resize_w = n * 32;
        }

        // Return original if no resize is needed
        if resize_h == ori_h && resize_w == ori_w {
            return (img, [1.0, 1.0]);
        }

        let ratio_h = resize_h as f32 / ori_h as f32;
        let ratio_w = resize_w as f32 / ori_w as f32;
        let resized_img =
            img.resize_exact(resize_w, resize_h, image::imageops::FilterType::Lanczos3);

        (resized_img, [ratio_h, ratio_w])
    }

    /// Resize type 2: Resize long side to specific length
    ///
    /// This method resizes the image so that its longest side matches
    /// the specified length. The dimensions are then adjusted to be
    /// multiples of 128.
    ///
    /// # Parameters
    /// * `img` - The image to resize
    /// * `resize_long` - Target length for the long side
    ///
    /// # Returns
    /// A tuple containing:
    /// 1. The resized image
    /// 2. Array with resize ratios [ratio_h, ratio_w]
    fn resize_image_type2(&self, img: DynamicImage, resize_long: u32) -> (DynamicImage, [f32; 2]) {
        let (w, h) = img.dimensions();

        // Calculate resize ratio based on which side is longer
        // If height > width, resize based on height; otherwise resize based on width
        let ratio = if h > w {
            resize_long as f32 / h as f32
        } else {
            resize_long as f32 / w as f32
        };

        let mut resize_h = (h as f32 * ratio) as u32;
        let mut resize_w = (w as f32 * ratio) as u32;

        // Ensure dimensions are multiples of 128
        let max_stride = 128;
        // Round up to nearest multiple of 128 to ensure proper alignment
        resize_h = resize_h.div_ceil(max_stride) * max_stride;
        resize_w = resize_w.div_ceil(max_stride) * max_stride;

        // Return original if no resize is needed
        if resize_h == h && resize_w == w {
            return (img, [1.0, 1.0]);
        }

        let resized_img =
            img.resize_exact(resize_w, resize_h, image::imageops::FilterType::Lanczos3);
        let ratio_h = resize_h as f32 / h as f32;
        let ratio_w = resize_w as f32 / w as f32;

        (resized_img, [ratio_h, ratio_w])
    }

    /// Resize type 3: Resize to specific input shape
    ///
    /// This method resizes the image to match the exact dimensions
    /// specified in the input shape parameter (channels, height, width).
    ///
    /// # Parameters
    /// * `img` - The image to resize
    /// * `input_shape` - Target shape (channels, height, width)
    ///
    /// # Returns
    /// A tuple containing:
    /// 1. The resized image
    /// 2. Array with resize ratios [ratio_h, ratio_w]
    fn resize_image_type3(
        &self,
        img: DynamicImage,
        input_shape: (u32, u32, u32),
    ) -> (DynamicImage, [f32; 2]) {
        let (ori_w, ori_h) = img.dimensions();
        let (_, resize_h, resize_w) = input_shape;

        // Return original if no resize is needed
        if resize_h == ori_h && resize_w == ori_w {
            return (img, [1.0, 1.0]);
        }

        let ratio_h = resize_h as f32 / ori_h as f32;
        let ratio_w = resize_w as f32 / ori_w as f32;
        let resized_img =
            img.resize_exact(resize_w, resize_h, image::imageops::FilterType::Lanczos3);

        (resized_img, [ratio_h, ratio_w])
    }
}
