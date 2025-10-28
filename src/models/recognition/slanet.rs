//! SLANet (Structure Location Alignment Network) Model
//!
//! This module provides a pure implementation of the SLANet table structure recognition model.
//! The model predicts both HTML structure tokens and bounding boxes for table cells.

use crate::core::inference::OrtInfer;
use crate::core::{OCRError, Tensor3D, Tensor4D};
use crate::processors::NormalizeImage;
use image::{DynamicImage, RgbImage};
use std::path::Path;

/// Output from SLANet model containing structure predictions and bounding boxes.
#[derive(Debug, Clone)]
pub struct SLANetModelOutput {
    /// Structure token logits [batch, seq_len, vocab_size]
    pub structure_logits: Tensor3D,
    /// Bounding box predictions [batch, seq_len, 8] (4 corner points × 2 coords)
    pub bbox_preds: Tensor3D,
    /// Image shape information for bbox denormalization (h, w, ratio_h, ratio_w, pad_h, pad_w)
    pub shape_info: Vec<[f32; 6]>,
}

/// Pure SLANet model implementation.
#[derive(Debug)]
pub struct SLANetModel {
    /// ONNX Runtime inference engine
    inference: OrtInfer,
    /// Image normalizer for preprocessing
    normalizer: NormalizeImage,
    /// Input image size (height, width)
    input_size: (u32, u32),
}

impl SLANetModel {
    /// Creates a new SLANet model.
    pub fn new(inference: OrtInfer, normalizer: NormalizeImage, input_size: (u32, u32)) -> Self {
        Self {
            inference,
            normalizer,
            input_size,
        }
    }

    /// Preprocesses images for table structure recognition.
    /// Follows PaddleOCR preprocessing pipeline:
    /// 1. ResizeByLong (keep aspect ratio, longest side = target_size)
    /// 2. Pad to target_size x target_size
    /// 3. Normalize
    pub fn preprocess(&self, images: Vec<RgbImage>) -> Result<(Tensor4D, Vec<[f32; 6]>), OCRError> {
        let mut shape_info_list = Vec::with_capacity(images.len());
        let mut processed_images = Vec::with_capacity(images.len());

        for img in images {
            let (orig_h, orig_w) = (img.height() as f32, img.width() as f32);
            let target_size = self.input_size.0 as f32; // Assuming square target

            // Step 1: ResizeByLong - keep aspect ratio, longest side = target_size
            let longest_side = orig_h.max(orig_w);
            let scale = target_size / longest_side;
            let resized_h = (orig_h * scale).round() as u32;
            let resized_w = (orig_w * scale).round() as u32;

            let resized = image::imageops::resize(
                &img,
                resized_w,
                resized_h,
                image::imageops::FilterType::Lanczos3,
            );

            // Step 2: Pad to target_size x target_size
            let target_size_u32 = target_size as u32;
            let padded = if resized_h < target_size_u32 || resized_w < target_size_u32 {
                // Create new image with target size
                let mut padded_img = image::RgbImage::new(target_size_u32, target_size_u32);

                // Copy resized image to top-left
                for y in 0..resized_h.min(target_size_u32) {
                    for x in 0..resized_w.min(target_size_u32) {
                        padded_img.put_pixel(x, y, *resized.get_pixel(x, y));
                    }
                }

                // The rest is already zero (black padding)
                padded_img
            } else {
                resized
            };

            // Store shape information for bbox denormalization
            // [orig_h, orig_w, scale, pad_h, pad_w, target_size]
            let pad_h = target_size - (resized_h as f32);
            let pad_w = target_size - (resized_w as f32);
            shape_info_list.push([orig_h, orig_w, scale, pad_h, pad_w, target_size]);

            processed_images.push(DynamicImage::ImageRgb8(padded));
        }

        // Normalize and convert to tensor
        let batch_tensor = self.normalizer.normalize_batch_to(processed_images)?;

        Ok((batch_tensor, shape_info_list))
    }

    /// Runs inference on the preprocessed tensor.
    ///
    /// Returns dual outputs: structure logits and bbox predictions.
    pub fn infer(&self, batch_tensor: &Tensor4D) -> Result<(Tensor3D, Tensor3D), OCRError> {
        self.inference
            .infer_dual_3d(batch_tensor)
            .map_err(|e| OCRError::Inference {
                model_name: "SLANet".to_string(),
                context: format!(
                    "failed to run inference on batch with shape {:?}",
                    batch_tensor.shape()
                ),
                source: Box::new(e),
            })
    }

    /// Runs the complete forward pass: preprocess -> infer.
    ///
    /// Postprocessing is handled separately by TableStructureDecode.
    pub fn forward(&self, images: Vec<RgbImage>) -> Result<SLANetModelOutput, OCRError> {
        let (batch_tensor, shape_info) = self.preprocess(images)?;
        let (structure_logits, bbox_preds) = self.infer(&batch_tensor)?;

        Ok(SLANetModelOutput {
            structure_logits,
            bbox_preds,
            shape_info,
        })
    }
}

/// Builder for SLANet model.
#[derive(Debug, Default)]
pub struct SLANetModelBuilder {
    /// Session pool size for ONNX Runtime
    session_pool_size: usize,
    /// Input image size (height, width)
    input_size: (u32, u32),
    /// ONNX Runtime session configuration
    ort_config: Option<crate::core::config::OrtSessionConfig>,
}

impl SLANetModelBuilder {
    /// Creates a new SLANet model builder.
    pub fn new() -> Self {
        Self {
            session_pool_size: 1,
            input_size: (512, 512),
            ort_config: None,
        }
    }

    /// Sets the session pool size.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.session_pool_size = size;
        self
    }

    /// Sets the input image size.
    pub fn input_size(mut self, size: (u32, u32)) -> Self {
        self.input_size = size;
        self
    }

    /// Sets the ONNX Runtime session configuration.
    pub fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.ort_config = Some(config);
        self
    }

    /// Builds the SLANet model.
    pub fn build(self, model_path: &Path) -> Result<SLANetModel, OCRError> {
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
            OrtInfer::from_config(&common_config, model_path, None)?
        } else {
            OrtInfer::new(model_path, None)?
        };

        // Create normalizer (ImageNet normalization)
        let normalizer = NormalizeImage::new(
            Some(1.0 / 255.0),
            Some(vec![0.485, 0.456, 0.406]),
            Some(vec![0.229, 0.224, 0.225]),
            None,
        )?;

        Ok(SLANetModel::new(inference, normalizer, self.input_size))
    }
}
