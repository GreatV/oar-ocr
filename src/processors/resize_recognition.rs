//! OCR-specific image resizing functionality.
//!
//! This module provides functionality for resizing images specifically for OCR processing.
//! It includes dynamic resizing based on image ratios and static resizing to fixed dimensions.

use crate::core::{
    OCRError,
    constants::{DEFAULT_MAX_IMG_WIDTH, DEFAULT_REC_IMAGE_SHAPE},
};
use crate::utils::{OCRResizePadConfig, ocr_resize_and_pad};
use image::RgbImage;

/// OCR-specific image resizer.
///
/// This struct handles resizing of images for OCR processing. It supports both dynamic
/// resizing based on the image's width-to-height ratio and static resizing to fixed
/// dimensions.
#[derive(Debug)]
pub struct OCRResize {
    pub rec_image_shape: [usize; 3],
    pub input_shape: Option<[usize; 3]>,
    pub max_img_w: usize,
}

impl OCRResize {
    /// Creates a new OCRResize instance with default maximum width.
    ///
    /// # Arguments
    ///
    /// * `rec_image_shape` - Optional shape for recognition images [channels, height, width].
    ///   If None, uses DEFAULT_REC_IMAGE_SHAPE.
    /// * `input_shape` - Optional input shape [channels, height, width].
    ///
    /// # Returns
    ///
    /// A new OCRResize instance.
    pub fn new(rec_image_shape: Option<[usize; 3]>, input_shape: Option<[usize; 3]>) -> Self {
        Self::with_max_width(rec_image_shape, input_shape, None)
    }

    /// Creates a new OCRResize instance with custom maximum width.
    ///
    /// # Arguments
    ///
    /// * `rec_image_shape` - Optional shape for recognition images [channels, height, width].
    ///   If None, uses DEFAULT_REC_IMAGE_SHAPE.
    /// * `input_shape` - Optional input shape [channels, height, width].
    /// * `max_img_w` - Optional maximum image width. If None, uses DEFAULT_MAX_IMG_WIDTH.
    ///
    /// # Returns
    ///
    /// A new OCRResize instance.
    pub fn with_max_width(
        rec_image_shape: Option<[usize; 3]>,
        input_shape: Option<[usize; 3]>,
        max_img_w: Option<usize>,
    ) -> Self {
        let rec_image_shape = rec_image_shape.unwrap_or(DEFAULT_REC_IMAGE_SHAPE);
        Self {
            rec_image_shape,
            input_shape,
            max_img_w: max_img_w.unwrap_or(DEFAULT_MAX_IMG_WIDTH),
        }
    }

    /// Resizes an image based on a maximum width-to-height ratio.
    ///
    /// This method resizes an image to fit within the specified dimensions while maintaining
    /// the aspect ratio. If the calculated width exceeds the maximum allowed width, the image
    /// is resized to the maximum width.
    ///
    /// # Arguments
    ///
    /// * `img` - The input RGB image to resize.
    /// * `max_wh_ratio` - The maximum width-to-height ratio for the resized image.
    ///
    /// # Returns
    ///
    /// A resized and padded RGB image.
    ///
    /// # Errors
    ///
    /// Returns an error if resizing fails due to invalid dimensions.
    pub fn resize_img(&self, img: &RgbImage, max_wh_ratio: f32) -> Result<RgbImage, OCRError> {
        let [_img_c, img_h, _img_w] = self.rec_image_shape;

        let config = OCRResizePadConfig::new(img_h as u32, self.max_img_w as u32);
        let (padded_image, _actual_width) =
            ocr_resize_and_pad(img, &config, Some(max_wh_ratio)).map_err(OCRError::from)?;

        Ok(padded_image)
    }

    /// Resizes an image using the default width-to-height ratio from rec_image_shape.
    ///
    /// This method calculates the width-to-height ratio from the configured rec_image_shape
    /// and uses it to resize the image via resize_img.
    ///
    /// # Arguments
    ///
    /// * `img` - The input RGB image to resize.
    ///
    /// # Returns
    ///
    /// A resized and padded RGB image.
    ///
    /// # Errors
    ///
    /// Returns an error if resizing fails due to invalid dimensions.
    pub fn resize(&self, img: &RgbImage) -> Result<RgbImage, OCRError> {
        let [_, img_h, img_w] = self.rec_image_shape;
        let max_wh_ratio = img_w as f32 / img_h as f32;
        self.resize_img(img, max_wh_ratio)
    }

    /// Resizes an image to a static size defined by input_shape.
    ///
    /// This method resizes an image to exact dimensions specified in the input_shape.
    /// It requires input_shape to be configured, otherwise it returns a ConfigError.
    ///
    /// # Arguments
    ///
    /// * `img` - The input RGB image to resize.
    ///
    /// # Returns
    ///
    /// A resized RGB image or an OCRError if input_shape is not configured.
    pub fn static_resize(&self, img: &RgbImage) -> Result<RgbImage, OCRError> {
        let [_img_c, img_h, img_w] = self.input_shape.ok_or_else(|| OCRError::ConfigError {
            message: "Input shape not configured for static resize".to_string(),
        })?;

        // Use Triangle (bilinear) to match cv2.resize INTER_LINEAR
        let resized_image = image::imageops::resize(
            img,
            img_w as u32,
            img_h as u32,
            image::imageops::FilterType::Triangle,
        );

        Ok(resized_image)
    }

    /// Applies resizing to a batch of images.
    ///
    /// This method applies either dynamic resizing (using resize) or static resizing
    /// (using static_resize) to a batch of images, depending on whether input_shape is configured.
    /// If input_shape is None, dynamic resizing is used; otherwise, static resizing is used.
    ///
    /// # Arguments
    ///
    /// * `imgs` - A slice of RGB images to resize.
    ///
    /// # Returns
    ///
    /// A vector of resized RGB images or an OCRError if static resizing fails.
    pub fn apply(&self, imgs: &[RgbImage]) -> Result<Vec<RgbImage>, OCRError> {
        if self.input_shape.is_none() {
            imgs.iter().map(|img| self.resize(img)).collect()
        } else {
            imgs.iter().map(|img| self.static_resize(img)).collect()
        }
    }

    /// Resizes an image to fit tensor shape requirements.
    ///
    /// This method resizes an image to fit within the dimensions specified by rec_image_shape,
    /// while maintaining the aspect ratio. If the calculated width exceeds the maximum allowed
    /// width, the image is resized to the maximum width. The resulting image is padded to
    /// match the target dimensions.
    ///
    /// # Arguments
    ///
    /// * `img` - The input RGB image to resize.
    ///
    /// # Returns
    ///
    /// A resized and padded RGB image or an OCRError.
    pub fn resize_to_tensor_shape(&self, img: &RgbImage) -> Result<RgbImage, OCRError> {
        let [_img_c, img_h, _img_w] = self.rec_image_shape;

        let config = OCRResizePadConfig::new(img_h as u32, self.max_img_w as u32);
        let (padded_image, _actual_width) =
            ocr_resize_and_pad(img, &config, None).map_err(OCRError::from)?;

        Ok(padded_image)
    }

    /// Applies tensor shape resizing to a batch of images.
    ///
    /// This method applies resize_to_tensor_shape to a batch of images. It handles
    /// empty batches by returning an empty vector.
    ///
    /// # Arguments
    ///
    /// * `imgs` - A slice of RGB images to resize.
    ///
    /// # Returns
    ///
    /// A vector of resized RGB images or an OCRError if resizing fails.
    pub fn apply_to_images(&self, imgs: &[RgbImage]) -> Result<Vec<RgbImage>, OCRError> {
        if imgs.is_empty() {
            return Ok(Vec::new());
        }

        let mut resized_images = Vec::with_capacity(imgs.len());

        for img in imgs {
            let resized_img = self.resize_to_tensor_shape(img)?;
            resized_images.push(resized_img);
        }

        Ok(resized_images)
    }
}
