//! Shared preprocessing helpers for model adapters.
//!
//! This module centralizes common preprocessing operations to reduce code duplication
//! across model implementations. It provides:
//! - Configuration helpers for common model types
//! - Reusable preprocessing pipelines for common patterns
//! - Utility functions for image format conversions

use crate::core::{OCRError, Tensor4D};
use crate::models::classification::PPLCNetPreprocessConfig;
use crate::models::detection::db::DBPreprocessConfig;
use crate::processors::{ImageScaleInfo, LimitType, NormalizeImage};
use image::{DynamicImage, RgbImage};

/// Construct a PP-LCNet preprocessing config with a custom input shape.
///
/// Leaves other fields at their `Default` values so adapters can override only
/// what they need (e.g., normalization statistics).
pub fn pp_lcnet_preprocess(input_shape: (u32, u32)) -> PPLCNetPreprocessConfig {
    PPLCNetPreprocessConfig {
        input_shape,
        ..Default::default()
    }
}

/// Construct a PP-LCNet preprocessing config with custom normalization stats.
///
/// Useful for adapters that expect zero-centered inputs but otherwise rely on
/// the standard defaults.
pub fn pp_lcnet_preprocess_with_norm(
    input_shape: (u32, u32),
    mean: [f32; 3],
    std: [f32; 3],
) -> PPLCNetPreprocessConfig {
    let mut config = pp_lcnet_preprocess(input_shape);
    config.normalize_mean = mean.to_vec();
    config.normalize_std = std.to_vec();
    config
}

/// Construct a DB preprocessing config that limits images by side length.
pub fn db_preprocess_with_limit_side_len(limit_side_len: u32) -> DBPreprocessConfig {
    DBPreprocessConfig {
        limit_side_len: Some(limit_side_len),
        ..Default::default()
    }
}

/// Construct a DB preprocessing config that resizes by long edge.
pub fn db_preprocess_with_resize_long(resize_long: u32) -> DBPreprocessConfig {
    DBPreprocessConfig {
        resize_long: Some(resize_long),
        ..Default::default()
    }
}

/// Construct a DB preprocessing config based on text type.
///
/// This function provides default preprocessing configurations:
/// - "general" (default): limit_side_len=960, limit_type=Max, max_side_limit=4000
/// - "seal": limit_side_len=736, limit_type=Min, max_side_limit=4000
///
/// Note: PP-StructureV3's overall OCR uses different defaults (736/min for all text).
/// Those defaults are applied in `OARStructureBuilder`, not here.
///
/// # Arguments
///
/// * `text_type` - Optional text type string ("general", "seal", etc.)
///
/// # Returns
///
/// DBPreprocessConfig configured for the specified text type
pub fn db_preprocess_for_text_type(text_type: Option<&str>) -> DBPreprocessConfig {
    match text_type {
        Some("seal") => DBPreprocessConfig {
            limit_side_len: Some(736),
            limit_type: Some(LimitType::Min),
            max_side_limit: Some(4000),
            ..Default::default()
        },
        _ => {
            // Default to "general" text configuration
            DBPreprocessConfig {
                limit_side_len: Some(960),
                limit_type: Some(LimitType::Max),
                max_side_limit: Some(4000),
                ..Default::default()
            }
        }
    }
}

/// Converts a batch of RGB images to dynamic images.
///
/// This is a common operation needed before most preprocessing steps.
///
/// # Arguments
///
/// * `images` - Vector of RGB images to convert
///
/// # Returns
///
/// Vector of dynamic images
///
/// # Example
///
/// ```rust,no_run
/// // let rgb_images: Vec<RgbImage> = load_images();
/// // let dynamic_images = rgb_to_dynamic(rgb_images);
/// ```
#[inline]
pub fn rgb_to_dynamic(images: Vec<RgbImage>) -> Vec<DynamicImage> {
    images.into_iter().map(DynamicImage::ImageRgb8).collect()
}

/// Applies a resizer and then normalizes the result to a tensor.
///
/// This is a common pattern used in recognition models like CRNN.
///
/// # Arguments
///
/// * `images` - Input RGB images
/// * `resizer` - Resizer implementing the apply method
/// * `normalizer` - Normalizer to convert to tensor
///
/// # Returns
///
/// Preprocessed 4D tensor ready for inference
///
/// # Example
///
/// ```rust,no_run
/// // let tensor = resize_and_normalize(
/// //     images,
/// //     &self.resizer,
/// //     &self.normalizer
/// // )?;
/// ```
pub fn resize_and_normalize<R>(
    images: Vec<RgbImage>,
    resizer: &R,
    normalizer: &NormalizeImage,
) -> Result<Tensor4D, OCRError>
where
    R: ResizeOperation,
{
    let resized_images = resizer.resize(images)?;
    let dynamic_images = rgb_to_dynamic(resized_images);
    normalizer.normalize_batch_to(dynamic_images)
}

/// Applies a detection resizer (with scale info) and then normalizes the result.
///
/// This is a common pattern used in detection models like DB and RT-DETR.
///
/// # Arguments
///
/// * `images` - Input RGB images
/// * `resizer` - Detection resizer that returns scale info
/// * `normalizer` - Normalizer to convert to tensor
///
/// # Returns
///
/// Tuple of (preprocessed tensor, scale information for each image)
///
/// # Example
///
/// ```rust,no_run
/// // let (tensor, scales) = detection_resize_and_normalize(
/// //     images,
/// //     &self.resizer,
/// //     &self.normalizer,
/// //     None,
/// //     None,
/// //     None,
/// // )?;
/// ```
pub fn detection_resize_and_normalize<R>(
    images: Vec<RgbImage>,
    resizer: &R,
    normalizer: &NormalizeImage,
) -> Result<(Tensor4D, Vec<ImageScaleInfo>), OCRError>
where
    R: DetectionResizeOperation,
{
    let dynamic_images = rgb_to_dynamic(images);
    let (resized_images, scale_info) = resizer.resize_with_scale(dynamic_images)?;
    let tensor = normalizer.normalize_batch_to(resized_images)?;
    Ok((tensor, scale_info))
}

/// Trait for resize operations that return only resized images.
///
/// Used for simple resize operations in recognition models.
pub trait ResizeOperation {
    /// Resizes a batch of images.
    fn resize(&self, images: Vec<RgbImage>) -> Result<Vec<RgbImage>, OCRError>;
}

/// Trait for detection resize operations that return scale information.
///
/// Used for detection models that need to map predictions back to original coordinates.
pub trait DetectionResizeOperation {
    /// Resizes a batch of images and returns scale information.
    fn resize_with_scale(
        &self,
        images: Vec<DynamicImage>,
    ) -> Result<(Vec<DynamicImage>, Vec<ImageScaleInfo>), OCRError>;
}

/// Builder for common preprocessing pipelines.
///
/// Provides a fluent interface for constructing preprocessing operations
/// without duplicating code across adapters.
///
/// # Example
///
/// ```rust,no_run
/// // let tensor = PreprocessPipelineBuilder::new()
/// //     .rgb_images(images)
/// //     .resize(&resizer)
/// //     .normalize(&normalizer)
/// //     .build()?;
/// ```
pub struct PreprocessPipelineBuilder {
    images: Option<Vec<RgbImage>>,
    dynamic_images: Option<Vec<DynamicImage>>,
}

impl PreprocessPipelineBuilder {
    /// Creates a new preprocessing pipeline builder.
    pub fn new() -> Self {
        Self {
            images: None,
            dynamic_images: None,
        }
    }

    /// Sets the input RGB images.
    pub fn rgb_images(mut self, images: Vec<RgbImage>) -> Self {
        self.images = Some(images);
        self
    }

    /// Sets the input dynamic images.
    pub fn dynamic_images(mut self, images: Vec<DynamicImage>) -> Self {
        self.dynamic_images = Some(images);
        self
    }

    /// Converts RGB images to dynamic images.
    pub fn to_dynamic(mut self) -> Self {
        if let Some(images) = self.images.take() {
            self.dynamic_images = Some(rgb_to_dynamic(images));
        }
        self
    }

    /// Applies a resize operation.
    pub fn resize<R>(mut self, resizer: &R) -> Result<Self, OCRError>
    where
        R: ResizeOperation,
    {
        if let Some(images) = self.images.take() {
            let resized = resizer.resize(images)?;
            self.images = Some(resized);
        }
        Ok(self)
    }

    /// Normalizes images to a tensor.
    pub fn normalize(self, normalizer: &NormalizeImage) -> Result<Tensor4D, OCRError> {
        let dynamic_images = if let Some(images) = self.images {
            rgb_to_dynamic(images)
        } else if let Some(images) = self.dynamic_images {
            images
        } else {
            return Err(OCRError::InvalidInput {
                message: "No images provided to preprocessing pipeline".to_string(),
            });
        };

        normalizer.normalize_batch_to(dynamic_images)
    }

    /// Builds the final tensor (alias for normalize for consistency).
    pub fn build(self, normalizer: &NormalizeImage) -> Result<Tensor4D, OCRError> {
        self.normalize(normalizer)
    }
}

impl Default for PreprocessPipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

use crate::processors::{DetResizeForTest, OCRResize};

/// Implement ResizeOperation for OCRResize (recognition models).
impl ResizeOperation for OCRResize {
    fn resize(&self, images: Vec<RgbImage>) -> Result<Vec<RgbImage>, OCRError> {
        self.apply(&images)
    }
}

/// Wrapper for DetResizeForTest to implement DetectionResizeOperation.
///
/// This allows DetResizeForTest to be used in the common preprocessing pipeline.
pub struct DetectionResizer<'a> {
    resizer: &'a DetResizeForTest,
}

impl<'a> DetectionResizer<'a> {
    /// Creates a new detection resizer wrapper.
    pub fn new(resizer: &'a DetResizeForTest) -> Self {
        Self { resizer }
    }
}

impl<'a> DetectionResizeOperation for DetectionResizer<'a> {
    fn resize_with_scale(
        &self,
        images: Vec<DynamicImage>,
    ) -> Result<(Vec<DynamicImage>, Vec<ImageScaleInfo>), OCRError> {
        let (resized, scales) = self.resizer.apply(images, None, None, None);
        Ok((resized, scales))
    }
}

/// Convenience function to wrap a DetResizeForTest for use in preprocessing pipelines.
#[inline]
pub fn wrap_detection_resizer(resizer: &DetResizeForTest) -> DetectionResizer<'_> {
    DetectionResizer::new(resizer)
}
