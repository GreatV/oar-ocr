//! PP-DocLayout Detection Model
//!
//! This module provides a pure implementation of the PP-DocLayout model for layout detection.
//! The model is independent of any specific task and can be reused in different contexts.

use crate::core::inference::OrtInfer;
use crate::core::{OCRError, Tensor4D};
use crate::processors::{
    ChannelOrder, DetResizeForTest, ImageScaleInfo, LimitType, NormalizeImage,
};
use image::{DynamicImage, RgbImage};

type PPDocLayoutPreprocessArtifacts = (Tensor4D, Vec<ImageScaleInfo>, Vec<[f32; 2]>, Vec<[f32; 2]>);
type PPDocLayoutPreprocessResult = Result<PPDocLayoutPreprocessArtifacts, OCRError>;

/// Preprocessing configuration for PP-DocLayout model.
#[derive(Debug, Clone)]
pub struct PPDocLayoutPreprocessConfig {
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

impl Default for PPDocLayoutPreprocessConfig {
    fn default() -> Self {
        Self {
            image_shape: (800, 800),
            keep_ratio: false,
            limit_side_len: 800,
            scale: 1.0 / 255.0,
            mean: vec![0.485, 0.456, 0.406],
            std: vec![0.229, 0.224, 0.225],
        }
    }
}

/// Postprocessing configuration for PP-DocLayout model.
#[derive(Debug, Clone)]
pub struct PPDocLayoutPostprocessConfig {
    /// Number of classes
    pub num_classes: usize,
}

/// Output from PP-DocLayout model.
#[derive(Debug, Clone)]
pub struct PPDocLayoutModelOutput {
    /// Detection predictions tensor [batch_size, num_detections, 6]
    /// Each detection: [x1, y1, x2, y2, score, class_id]
    pub predictions: Tensor4D,
}

/// PP-DocLayout layout detection model.
///
/// This is a pure model implementation that handles:
/// - Preprocessing: Image resizing and normalization
/// - Inference: Running the ONNX model
/// - Postprocessing: Returning raw predictions
///
/// The model is independent of any specific task or adapter.
#[derive(Debug)]
pub struct PPDocLayoutModel {
    inference: OrtInfer,
    resizer: DetResizeForTest,
    normalizer: NormalizeImage,
    #[allow(dead_code)]
    preprocess_config: PPDocLayoutPreprocessConfig,
}

impl PPDocLayoutModel {
    /// Creates a new PP-DocLayout model.
    pub fn new(
        inference: OrtInfer,
        preprocess_config: PPDocLayoutPreprocessConfig,
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

        // Create normalizer
        let normalizer = NormalizeImage::new(
            Some(preprocess_config.scale),
            Some(preprocess_config.mean.clone()),
            Some(preprocess_config.std.clone()),
            Some(ChannelOrder::CHW),
        )?;

        Ok(Self {
            inference,
            resizer,
            normalizer,
            preprocess_config,
        })
    }

    /// Preprocesses images for PP-DocLayout model.
    ///
    /// Returns:
    /// - Batch tensor ready for inference
    /// - Image shapes after resizing [h, w, ratio_h, ratio_w]
    /// - Original shapes [h, w]
    /// - Resized shapes [h, w]
    pub fn preprocess(&self, images: Vec<RgbImage>) -> PPDocLayoutPreprocessResult {
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
    /// PP-DocLayout requires scale_factor and image_shape as additional inputs.
    pub fn infer(
        &self,
        batch_tensor: &Tensor4D,
        orig_shapes: &[[f32; 2]],
        resized_shapes: &[[f32; 2]],
    ) -> Result<Tensor4D, OCRError> {
        let batch_size = batch_tensor.shape()[0];

        // Calculate scale factors (resized / original)
        let mut scale_factors = Vec::new();
        for i in 0..batch_size {
            let scale_y = resized_shapes[i][0] / orig_shapes[i][0];
            let scale_x = resized_shapes[i][1] / orig_shapes[i][1];
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

        // Create image_shape array (resized shapes)
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

        // Run inference with scale_factor and image_shape
        self.inference
            .infer_4d_layout(batch_tensor, Some(scale_factor), Some(image_shape))
    }

    /// Postprocesses model predictions.
    ///
    /// For PP-DocLayout, we just return the raw predictions.
    /// The adapter layer will handle converting these to task-specific outputs.
    pub fn postprocess(
        &self,
        predictions: Tensor4D,
        _config: &PPDocLayoutPostprocessConfig,
    ) -> Result<PPDocLayoutModelOutput, OCRError> {
        Ok(PPDocLayoutModelOutput { predictions })
    }

    /// Runs the complete forward pass: preprocess -> infer -> postprocess.
    pub fn forward(
        &self,
        images: Vec<RgbImage>,
        config: &PPDocLayoutPostprocessConfig,
    ) -> Result<(PPDocLayoutModelOutput, Vec<ImageScaleInfo>), OCRError> {
        let (batch_tensor, img_shapes, orig_shapes, resized_shapes) = self.preprocess(images)?;
        let predictions = self.infer(&batch_tensor, &orig_shapes, &resized_shapes)?;
        let output = self.postprocess(predictions, config)?;
        Ok((output, img_shapes))
    }
}

/// Builder for PP-DocLayout model.
#[derive(Debug, Default)]
pub struct PPDocLayoutModelBuilder {
    preprocess_config: Option<PPDocLayoutPreprocessConfig>,
}

impl PPDocLayoutModelBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the preprocessing configuration.
    pub fn preprocess_config(mut self, config: PPDocLayoutPreprocessConfig) -> Self {
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

    /// Builds the PP-DocLayout model.
    pub fn build(self, inference: OrtInfer) -> Result<PPDocLayoutModel, OCRError> {
        let preprocess_config = self.preprocess_config.unwrap_or_default();
        PPDocLayoutModel::new(inference, preprocess_config)
    }
}
