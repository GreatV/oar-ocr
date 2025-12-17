//! UVDoc Document Rectification Model
//!
//! This module provides a pure implementation of the UVDoc model for document rectification.
//! The model takes distorted document images and outputs rectified (flattened) versions.

use crate::core::inference::OrtInfer;
use crate::core::{OCRError, Tensor4D};
use crate::processors::{ChannelOrder, NormalizeImage, UVDocPostProcess};
use image::{DynamicImage, RgbImage, imageops::FilterType};

/// Configuration for UVDoc model preprocessing.
#[derive(Debug, Clone)]
pub struct UVDocPreprocessConfig {
    /// Input shape [channels, height, width]
    pub rec_image_shape: [usize; 3],
}

impl Default for UVDocPreprocessConfig {
    fn default() -> Self {
        Self {
            rec_image_shape: [3, 512, 512],
        }
    }
}

/// Output from UVDoc model.
#[derive(Debug, Clone)]
pub struct UVDocModelOutput {
    /// Rectified images
    pub images: Vec<RgbImage>,
}

/// Pure UVDoc model implementation.
///
/// This model performs document rectification (unwarping) on distorted document images.
#[derive(Debug)]
pub struct UVDocModel {
    /// ONNX Runtime inference engine
    inference: OrtInfer,
    /// Image normalizer for preprocessing
    normalizer: NormalizeImage,
    /// UVDoc postprocessor for converting tensor to images
    postprocessor: UVDocPostProcess,
    /// Input shape [channels, height, width]
    rec_image_shape: [usize; 3],
}

impl UVDocModel {
    /// Creates a new UVDoc model.
    pub fn new(
        inference: OrtInfer,
        normalizer: NormalizeImage,
        postprocessor: UVDocPostProcess,
        rec_image_shape: [usize; 3],
    ) -> Self {
        Self {
            inference,
            normalizer,
            postprocessor,
            rec_image_shape,
        }
    }

    /// Preprocesses images for rectification.
    ///
    /// # Arguments
    ///
    /// * `images` - Input images to preprocess
    ///
    /// # Returns
    ///
    /// A tuple of (batch_tensor, original_sizes)
    pub fn preprocess(
        &self,
        images: Vec<RgbImage>,
    ) -> Result<(Tensor4D, Vec<(u32, u32)>), OCRError> {
        let mut original_sizes = Vec::with_capacity(images.len());
        let mut processed_images = Vec::with_capacity(images.len());

        let target_height = self.rec_image_shape[1] as u32;
        let target_width = self.rec_image_shape[2] as u32;
        let should_resize = target_height > 0 && target_width > 0;

        for img in images {
            let original_size = (img.width(), img.height());
            original_sizes.push(original_size);

            if should_resize && (img.width() != target_width || img.height() != target_height) {
                // Use cv2.INTER_LINEAR for UVDoc resize.
                let resized = DynamicImage::ImageRgb8(img).resize_exact(
                    target_width,
                    target_height,
                    FilterType::Triangle,
                );
                processed_images.push(resized);
            } else {
                processed_images.push(DynamicImage::ImageRgb8(img));
            }
        }

        // Normalize and convert to tensor
        let batch_tensor = self.normalizer.normalize_batch_to(processed_images)?;

        Ok((batch_tensor, original_sizes))
    }

    /// Runs inference on the preprocessed batch.
    ///
    /// # Arguments
    ///
    /// * `batch_tensor` - Preprocessed batch tensor
    ///
    /// # Returns
    ///
    /// Model predictions as a 4D tensor
    pub fn infer(&self, batch_tensor: &Tensor4D) -> Result<Tensor4D, OCRError> {
        self.inference
            .infer_4d(batch_tensor)
            .map_err(|e| OCRError::Inference {
                model_name: "UVDoc".to_string(),
                context: format!(
                    "failed to run inference on batch with shape {:?}",
                    batch_tensor.shape()
                ),
                source: Box::new(e),
            })
    }

    /// Postprocesses model predictions to rectified images.
    ///
    /// # Arguments
    ///
    /// * `predictions` - Model predictions
    /// * `original_sizes` - Original image sizes (width, height)
    ///
    /// # Returns
    ///
    /// Rectified images resized to original dimensions
    pub fn postprocess(
        &self,
        predictions: &Tensor4D,
        original_sizes: &[(u32, u32)],
    ) -> Result<Vec<RgbImage>, OCRError> {
        // Use UVDocPostProcess to convert tensor to images
        let mut images =
            self.postprocessor
                .apply_batch(predictions)
                .map_err(|e| OCRError::ConfigError {
                    message: format!("Failed to postprocess rectification output: {}", e),
                })?;

        if images.len() != original_sizes.len() {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "Mismatched rectification batch sizes: predictions={}, originals={}",
                    images.len(),
                    original_sizes.len()
                ),
            });
        }

        // Resize back to original dimensions
        for (img, &(orig_w, orig_h)) in images.iter_mut().zip(original_sizes) {
            if orig_w == 0 || orig_h == 0 {
                continue;
            }

            if img.width() != orig_w || img.height() != orig_h {
                // Use cv2.INTER_LINEAR for resizing outputs back to original size.
                let resized = DynamicImage::ImageRgb8(std::mem::take(img)).resize_exact(
                    orig_w,
                    orig_h,
                    FilterType::Triangle,
                );
                *img = resized.into_rgb8();
            }
        }

        Ok(images)
    }

    /// Performs complete forward pass: preprocess -> infer -> postprocess.
    ///
    /// # Arguments
    ///
    /// * `images` - Input images to rectify
    ///
    /// # Returns
    ///
    /// UVDocModelOutput containing rectified images
    pub fn forward(&self, images: Vec<RgbImage>) -> Result<UVDocModelOutput, OCRError> {
        let (batch_tensor, original_sizes) = self.preprocess(images)?;
        let predictions = self.infer(&batch_tensor)?;
        let rectified_images = self.postprocess(&predictions, &original_sizes)?;

        Ok(UVDocModelOutput {
            images: rectified_images,
        })
    }
}

/// Builder for UVDoc model.
#[derive(Debug, Default)]
pub struct UVDocModelBuilder {
    /// Session pool size for ONNX Runtime
    session_pool_size: usize,
    /// Preprocessing configuration
    preprocess_config: UVDocPreprocessConfig,
    /// ONNX Runtime session configuration
    ort_config: Option<crate::core::config::OrtSessionConfig>,
}

impl UVDocModelBuilder {
    /// Creates a new UVDoc model builder.
    pub fn new() -> Self {
        Self {
            session_pool_size: 1,
            preprocess_config: UVDocPreprocessConfig::default(),
            ort_config: None,
        }
    }

    /// Sets the session pool size for ONNX Runtime.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.session_pool_size = size;
        self
    }

    /// Sets the preprocessing configuration.
    pub fn preprocess_config(mut self, config: UVDocPreprocessConfig) -> Self {
        self.preprocess_config = config;
        self
    }

    /// Sets the input image shape.
    pub fn rec_image_shape(mut self, shape: [usize; 3]) -> Self {
        self.preprocess_config.rec_image_shape = shape;
        self
    }

    /// Sets the ONNX Runtime session configuration.
    pub fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.ort_config = Some(config);
        self
    }

    /// Builds the UVDoc model.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file
    ///
    /// # Returns
    ///
    /// A configured UVDoc model instance
    pub fn build(self, model_path: &std::path::Path) -> Result<UVDocModel, OCRError> {
        // Create ONNX inference engine
        let inference = if self.session_pool_size > 1 || self.ort_config.is_some() {
            use crate::core::config::ModelInferenceConfig;
            let common_config = ModelInferenceConfig {
                model_path: None,
                model_name: None,
                batch_size: None,
                enable_logging: None,
                ort_session: self.ort_config,
                session_pool_size: Some(self.session_pool_size),
            };
            OrtInfer::from_config(&common_config, model_path, Some("image"))?
        } else {
            OrtInfer::new(model_path, Some("image"))?
        };

        // Create normalizer (scale to [0, 1] without mean shift).
        // Images are read in BGR and UVDoc models are trained with BGR order,
        // so keep color order consistent here.
        let normalizer = NormalizeImage::with_color_order(
            Some(1.0 / 255.0),
            Some(vec![0.0, 0.0, 0.0]),
            Some(vec![1.0, 1.0, 1.0]),
            Some(ChannelOrder::CHW),
            Some(crate::processors::ColorOrder::BGR),
        )?;

        // Create postprocessor
        let postprocessor = UVDocPostProcess::new(255.0);

        Ok(UVDocModel::new(
            inference,
            normalizer,
            postprocessor,
            self.preprocess_config.rec_image_shape,
        ))
    }
}
