//! Scale-Aware Object Detector
//!
//! This module provides a unified, generic implementation for object detection models
//! that require scale information as auxiliary inputs during inference.
//!
//! Both PicoDet (general object detection) and PP-DocLayout (document layout detection)
//! are thin wrappers around this core implementation, eliminating code duplication
//! while maintaining clean, model-specific APIs.

use crate::core::inference::OrtInfer;
use crate::core::validation::{
    validate_division, validate_image_dimensions, validate_non_empty, validate_positive,
    validate_same_length,
};
use crate::core::{OCRError, Tensor4D};
use crate::processors::{
    ChannelOrder, DetResizeForTest, ImageScaleInfo, LimitType, NormalizeImage,
};
use image::{DynamicImage, RgbImage};

/// Specifies which auxiliary inputs the model requires for inference.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ScaleAwareDetectorInferenceMode {
    /// Only requires scale_factor (e.g., PicoDet S/M models)
    ScaleFactorOnly,
    /// Requires both scale_factor and im_shape (e.g., PP-DocLayout L/plus-L models)
    ScaleFactorAndImageShape,
}

/// Preprocessing configuration for scale-aware detection models.
#[derive(Debug, Clone)]
pub struct ScaleAwareDetectorPreprocessConfig {
    /// Target image shape (height, width)
    pub image_shape: (u32, u32),
    /// Whether to keep aspect ratio when resizing
    pub keep_ratio: bool,
    /// Limit side length
    pub limit_side_len: u32,
    /// Normalization scale factor
    pub scale: f32,
    /// Normalization mean values (RGB)
    pub mean: Vec<f32>,
    /// Normalization std values (RGB)
    pub std: Vec<f32>,
}

impl ScaleAwareDetectorPreprocessConfig {
    /// Creates configuration for PicoDet models (608x800 default)
    pub fn picodet() -> Self {
        Self {
            image_shape: (800, 608),
            keep_ratio: false,
            limit_side_len: 800,
            scale: 1.0 / 255.0,
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
        }
    }

    /// Creates configuration for PP-DocLayout models (800x800 default)
    pub fn pp_doclayout() -> Self {
        Self {
            image_shape: (800, 800),
            keep_ratio: false,
            limit_side_len: 800,
            scale: 1.0 / 255.0,
            mean: vec![0.0, 0.0, 0.0],
            std: vec![1.0, 1.0, 1.0],
        }
    }

    /// Validates the configuration parameters.
    pub fn validate(&self) -> Result<(), OCRError> {
        // Validate image dimensions
        validate_image_dimensions(
            self.image_shape.0,
            self.image_shape.1,
            "ScaleAwareDetectorPreprocessConfig",
        )?;

        // Validate limit_side_len
        validate_positive(self.limit_side_len, "limit_side_len")?;

        // Validate scale
        validate_positive(self.scale, "scale")?;

        // Validate mean and std have same length
        validate_same_length(&self.mean, &self.std, "mean", "std")?;

        // Validate mean and std lengths (should be 3 for RGB)
        validate_non_empty(&self.mean, "mean")?;
        validate_non_empty(&self.std, "std")?;

        // Validate std values are positive
        for (i, &std_val) in self.std.iter().enumerate() {
            validate_positive(std_val, &format!("std[{}]", i))?;
        }

        Ok(())
    }
}

/// Postprocessing configuration for scale-aware detection models.
#[derive(Debug, Clone)]
pub struct ScaleAwareDetectorPostprocessConfig {
    /// Number of classes
    pub num_classes: usize,
}

/// Output from scale-aware detection models.
#[derive(Debug, Clone)]
pub struct ScaleAwareDetectorModelOutput {
    /// Detection predictions tensor [batch_size, num_detections, 6]
    /// Each detection: [x1, y1, x2, y2, score, class_id]
    pub predictions: Tensor4D,
}

type ScaleAwareDetectorPreprocessArtifacts =
    (Tensor4D, Vec<ImageScaleInfo>, Vec<[f32; 2]>, Vec<[f32; 2]>);
type ScaleAwareDetectorPreprocessResult = Result<ScaleAwareDetectorPreprocessArtifacts, OCRError>;

/// Generic scale-aware object detection model.
///
/// This unified implementation supports detection models that require scale information
/// as auxiliary inputs (PicoDet, PP-DocLayout, etc.), eliminating code duplication
/// by parameterizing the inference mode.
#[derive(Debug)]
pub struct ScaleAwareDetectorModel {
    inference: OrtInfer,
    resizer: DetResizeForTest,
    normalizer: NormalizeImage,
    inference_mode: ScaleAwareDetectorInferenceMode,
    _preprocess_config: ScaleAwareDetectorPreprocessConfig,
}

impl ScaleAwareDetectorModel {
    /// Creates a new scale-aware detector model.
    pub fn new(
        inference: OrtInfer,
        preprocess_config: ScaleAwareDetectorPreprocessConfig,
        inference_mode: ScaleAwareDetectorInferenceMode,
    ) -> Result<Self, OCRError> {
        // Validate configuration before proceeding
        preprocess_config.validate()?;

        // Create resizer
        let resizer = DetResizeForTest::new(
            None,
            Some((
                preprocess_config.image_shape.0,
                preprocess_config.image_shape.1,
            )),
            Some(preprocess_config.keep_ratio),
            Some(preprocess_config.limit_side_len),
            Some(LimitType::Max),
            None,
            None,
        );

        // Create normalizer.
        // Paddle models expect BGR input; reorder ImageNet stats and swap channels.
        let normalizer = NormalizeImage::with_color_order_from_rgb_stats(
            Some(preprocess_config.scale),
            preprocess_config.mean.clone(),
            preprocess_config.std.clone(),
            Some(ChannelOrder::CHW),
            crate::processors::ColorOrder::BGR,
        )?;

        Ok(Self {
            inference,
            resizer,
            normalizer,
            inference_mode,
            _preprocess_config: preprocess_config,
        })
    }

    /// Preprocesses images for object detection.
    ///
    /// Returns:
    /// - Batch tensor ready for inference
    /// - Image shapes after resizing [h, w, ratio_h, ratio_w]
    /// - Original shapes [h, w]
    /// - Resized shapes [h, w]
    pub fn preprocess(&self, images: Vec<RgbImage>) -> ScaleAwareDetectorPreprocessResult {
        // Validate input is not empty
        validate_non_empty(&images, "images")?;

        // Store original dimensions
        let orig_shapes: Vec<[f32; 2]> = images
            .iter()
            .map(|img| [img.height() as f32, img.width() as f32])
            .collect();

        // Convert to DynamicImage
        let dynamic_images: Vec<DynamicImage> =
            images.into_iter().map(DynamicImage::ImageRgb8).collect();

        // Resize images
        let (resized_images, img_shapes) = self.resizer.apply(
            dynamic_images,
            None, // Use configured limit_side_length
            None, // Use configured limit_type
            None,
        );

        // Get resized dimensions
        let resized_shapes: Vec<[f32; 2]> = resized_images
            .iter()
            .map(|img| [img.height() as f32, img.width() as f32])
            .collect();

        // Normalize and convert to tensor
        let batch_tensor = self.normalizer.normalize_batch_to(resized_images)?;

        Ok((batch_tensor, img_shapes, orig_shapes, resized_shapes))
    }

    /// Runs inference on the preprocessed batch tensor.
    ///
    /// Automatically handles different inference modes:
    /// - ScaleFactorOnly: Only passes scale_factor
    /// - ScaleFactorAndImageShape: Passes both scale_factor and im_shape
    pub fn infer(
        &self,
        batch_tensor: &Tensor4D,
        orig_shapes: &[[f32; 2]],
        resized_shapes: &[[f32; 2]],
    ) -> Result<Tensor4D, OCRError> {
        let batch_size = batch_tensor.shape()[0];

        // Validate batch size
        if batch_size == 0 {
            return Err(OCRError::InvalidInput {
                message: "Batch size cannot be zero".to_string(),
            });
        }

        // Validate shape arrays match batch size
        validate_same_length(orig_shapes, resized_shapes, "orig_shapes", "resized_shapes")?;

        if orig_shapes.len() != batch_size {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "Shape arrays length ({}) does not match batch size ({})",
                    orig_shapes.len(),
                    batch_size
                ),
            });
        }

        // Calculate scale factors (resized / original) with bounds checking and division validation
        let mut scale_factors = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            // Safe indexing - already validated lengths match
            let orig_h = orig_shapes[i][0];
            let orig_w = orig_shapes[i][1];
            let resized_h = resized_shapes[i][0];
            let resized_w = resized_shapes[i][1];

            // Validate divisions
            validate_division(
                resized_h,
                orig_h,
                &format!("scale_y calculation for image {}", i),
            )?;
            validate_division(
                resized_w,
                orig_w,
                &format!("scale_x calculation for image {}", i),
            )?;

            let scale_y = resized_h / orig_h;
            let scale_x = resized_w / orig_w;
            scale_factors.push([scale_y, scale_x]);
        }

        // Create scale_factor array
        let scale_factor = ndarray::Array2::from_shape_vec(
            (batch_size, 2),
            scale_factors.into_iter().flatten().collect(),
        )
        .map_err(|e| OCRError::InvalidInput {
            message: format!("Failed to create scale_factor array: {}", e),
        })?;

        // Handle different inference modes
        match self.inference_mode {
            ScaleAwareDetectorInferenceMode::ScaleFactorOnly => {
                // PicoDet-style: only scale_factor
                self.inference
                    .infer_4d_layout(batch_tensor, Some(scale_factor), None)
            }
            ScaleAwareDetectorInferenceMode::ScaleFactorAndImageShape => {
                // PP-DocLayout-style: both scale_factor and image_shape
                let image_shape = ndarray::Array2::from_shape_vec(
                    (batch_size, 2),
                    resized_shapes
                        .iter()
                        .flat_map(|s| s.iter().copied())
                        .collect(),
                )
                .map_err(|e| OCRError::InvalidInput {
                    message: format!("Failed to create image_shape array: {}", e),
                })?;

                self.inference
                    .infer_4d_layout(batch_tensor, Some(scale_factor), Some(image_shape))
            }
        }
    }

    /// Postprocesses model predictions.
    ///
    /// Returns raw predictions for adapter layer to process.
    pub fn postprocess(
        &self,
        predictions: Tensor4D,
        _config: &ScaleAwareDetectorPostprocessConfig,
    ) -> Result<ScaleAwareDetectorModelOutput, OCRError> {
        Ok(ScaleAwareDetectorModelOutput { predictions })
    }

    /// Runs the complete forward pass: preprocess -> infer -> postprocess.
    pub fn forward(
        &self,
        images: Vec<RgbImage>,
        config: &ScaleAwareDetectorPostprocessConfig,
    ) -> Result<(ScaleAwareDetectorModelOutput, Vec<ImageScaleInfo>), OCRError> {
        let (batch_tensor, img_shapes, orig_shapes, resized_shapes) = self.preprocess(images)?;
        let predictions = self.infer(&batch_tensor, &orig_shapes, &resized_shapes)?;
        let output = self.postprocess(predictions, config)?;
        Ok((output, img_shapes))
    }
}

/// Builder for scale-aware detection models.
#[derive(Debug, Default)]
pub struct ScaleAwareDetectorModelBuilder {
    preprocess_config: Option<ScaleAwareDetectorPreprocessConfig>,
    inference_mode: Option<ScaleAwareDetectorInferenceMode>,
}

impl ScaleAwareDetectorModelBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Configures as PicoDet model (ScaleFactorOnly mode).
    pub fn picodet() -> Self {
        Self {
            preprocess_config: Some(ScaleAwareDetectorPreprocessConfig::picodet()),
            inference_mode: Some(ScaleAwareDetectorInferenceMode::ScaleFactorOnly),
        }
    }

    /// Configures as PP-DocLayout model (ScaleFactorAndImageShape mode).
    pub fn pp_doclayout() -> Self {
        Self {
            preprocess_config: Some(ScaleAwareDetectorPreprocessConfig::pp_doclayout()),
            inference_mode: Some(ScaleAwareDetectorInferenceMode::ScaleFactorAndImageShape),
        }
    }

    /// Sets the preprocessing configuration.
    pub fn preprocess_config(mut self, config: ScaleAwareDetectorPreprocessConfig) -> Self {
        self.preprocess_config = Some(config);
        self
    }

    /// Sets the inference mode.
    pub fn inference_mode(mut self, mode: ScaleAwareDetectorInferenceMode) -> Self {
        self.inference_mode = Some(mode);
        self
    }

    /// Sets the image shape.
    pub fn image_shape(mut self, height: u32, width: u32) -> Self {
        let mut config = self
            .preprocess_config
            .unwrap_or_else(ScaleAwareDetectorPreprocessConfig::picodet);
        config.image_shape = (height, width);
        config.limit_side_len = height.max(width);
        self.preprocess_config = Some(config);
        self
    }

    /// Builds the scale-aware detector model.
    pub fn build(self, inference: OrtInfer) -> Result<ScaleAwareDetectorModel, OCRError> {
        let preprocess_config = self
            .preprocess_config
            .unwrap_or_else(ScaleAwareDetectorPreprocessConfig::picodet);
        let inference_mode = self
            .inference_mode
            .unwrap_or(ScaleAwareDetectorInferenceMode::ScaleFactorOnly);

        ScaleAwareDetectorModel::new(inference, preprocess_config, inference_mode)
    }
}
