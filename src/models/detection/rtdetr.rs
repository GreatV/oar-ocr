//! RT-DETR Layout Detection Model
//!
//! This module provides a pure implementation of the RT-DETR model for layout detection.
//! The model is independent of any specific task and can be reused in different contexts.

use crate::core::inference::OrtInfer;
use crate::core::{OCRError, Tensor4D};
use crate::processors::{
    ChannelOrder, DetResizeForTest, ImageScaleInfo, LimitType, NormalizeImage,
};
use image::{DynamicImage, RgbImage};
use ndarray::Array2;

type RTDetrPreprocessArtifacts = (Tensor4D, Vec<ImageScaleInfo>, Vec<[f32; 2]>, Vec<[f32; 2]>);
type RTDetrPreprocessResult = Result<RTDetrPreprocessArtifacts, OCRError>;

/// Preprocessing configuration for RT-DETR model.
#[derive(Debug, Clone)]
pub struct RTDetrPreprocessConfig {
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

impl Default for RTDetrPreprocessConfig {
    fn default() -> Self {
        Self {
            image_shape: (640, 640),
            keep_ratio: false,
            limit_side_len: 640,
            scale: 1.0 / 255.0,
            // Paddle's RT-DETR exports expect inputs scaled to [0,1] without mean/std shift
            mean: vec![0.0, 0.0, 0.0],
            std: vec![1.0, 1.0, 1.0],
        }
    }
}

/// Postprocessing configuration for RT-DETR model.
#[derive(Debug, Clone)]
pub struct RTDetrPostprocessConfig {
    /// Number of classes
    pub num_classes: usize,
}

/// Output from RT-DETR model.
#[derive(Debug, Clone)]
pub struct RTDetrModelOutput {
    /// Detection predictions tensor [batch_size, num_detections, 6]
    /// Each detection: [x1, y1, x2, y2, score, class_id]
    pub predictions: Tensor4D,
}

/// RT-DETR layout detection model.
///
/// This is a pure model implementation that handles:
/// - Preprocessing: Image resizing and normalization
/// - Inference: Running the ONNX model
/// - Postprocessing: Returning raw predictions
///
/// The model is independent of any specific task or adapter.
#[derive(Debug)]
pub struct RTDetrModel {
    inference: OrtInfer,
    resizer: DetResizeForTest,
    normalizer: NormalizeImage,
    _preprocess_config: RTDetrPreprocessConfig,
}

impl RTDetrModel {
    /// Creates a new RT-DETR model.
    pub fn new(
        inference: OrtInfer,
        preprocess_config: RTDetrPreprocessConfig,
    ) -> Result<Self, OCRError> {
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
        // Paddle models expect BGR input; treat config mean/std as RGB and reorder.
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
            _preprocess_config: preprocess_config,
        })
    }

    /// Preprocesses images for RT-DETR model.
    ///
    /// Returns:
    /// - Batch tensor ready for inference
    /// - Image shapes after resizing [h, w, ratio_h, ratio_w]
    /// - Original shapes [h, w]
    /// - Resized shapes [h, w]
    pub fn preprocess(&self, images: Vec<RgbImage>) -> RTDetrPreprocessResult {
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
    /// RT-DETR requires both `scale_factor` and `im_shape` inputs.
    pub fn infer(
        &self,
        batch_tensor: &Tensor4D,
        scale_factor: Array2<f32>,
        im_shape: Array2<f32>,
    ) -> Result<Tensor4D, OCRError> {
        self.inference
            .infer_4d_layout(batch_tensor, Some(scale_factor), Some(im_shape))
    }

    /// Postprocesses model predictions.
    ///
    /// For RT-DETR, we just return the raw predictions.
    /// The adapter layer will handle converting these to task-specific outputs.
    pub fn postprocess(
        &self,
        predictions: Tensor4D,
        _config: &RTDetrPostprocessConfig,
    ) -> Result<RTDetrModelOutput, OCRError> {
        Ok(RTDetrModelOutput { predictions })
    }

    /// Runs the complete forward pass: preprocess -> infer -> postprocess.
    pub fn forward(
        &self,
        images: Vec<RgbImage>,
        config: &RTDetrPostprocessConfig,
    ) -> Result<(RTDetrModelOutput, Vec<ImageScaleInfo>), OCRError> {
        let (batch_tensor, img_shapes, _orig_shapes, resized_shapes) = self.preprocess(images)?;

        let batch_size = batch_tensor.shape()[0];

        // Build scale_factor array [ratio_h, ratio_w]
        let scale_data: Vec<f32> = img_shapes
            .iter()
            .flat_map(|shape| [shape.ratio_h, shape.ratio_w])
            .collect();
        let scale_factor = Array2::from_shape_vec((batch_size, 2), scale_data).map_err(|e| {
            OCRError::InvalidInput {
                message: format!("Failed to create scale_factor array: {}", e),
            }
        })?;

        // Build im_shape array using resized dimensions
        let im_shape_data: Vec<f32> = resized_shapes
            .iter()
            .flat_map(|shape| [shape[0], shape[1]])
            .collect();
        let im_shape = Array2::from_shape_vec((batch_size, 2), im_shape_data).map_err(|e| {
            OCRError::InvalidInput {
                message: format!("Failed to create im_shape array: {}", e),
            }
        })?;

        let predictions = self.infer(&batch_tensor, scale_factor, im_shape)?;
        let output = self.postprocess(predictions, config)?;
        Ok((output, img_shapes))
    }
}

/// Builder for RT-DETR model.
#[derive(Debug, Default)]
pub struct RTDetrModelBuilder {
    preprocess_config: Option<RTDetrPreprocessConfig>,
}

impl RTDetrModelBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the preprocessing configuration.
    pub fn preprocess_config(mut self, config: RTDetrPreprocessConfig) -> Self {
        self.preprocess_config = Some(config);
        self
    }

    /// Sets the image shape.
    pub fn image_shape(mut self, height: u32, width: u32) -> Self {
        let mut config = self.preprocess_config.unwrap_or_default();
        config.image_shape = (height, width);
        config.limit_side_len = height.max(width);
        self.preprocess_config = Some(config);
        self
    }

    /// Builds the RT-DETR model.
    pub fn build(self, inference: OrtInfer) -> Result<RTDetrModel, OCRError> {
        let preprocess_config = self.preprocess_config.unwrap_or_default();
        RTDetrModel::new(inference, preprocess_config)
    }
}
