//! SLANet (Structure Location Alignment Network) Model
//!
//! This module provides a pure implementation of the SLANet table structure recognition model.
//! The model predicts both HTML structure tokens and bounding boxes for table cells.
//!
//! # Important: BGR Channel Order
//!
//! SLANet models expect images in **BGR** channel order.
//! Configs apply ImageNet normalization on that BGR input. In this repo, images are
//! loaded as RGB; we keep them in RGB and rely on `NormalizeImage` with
//! `ColorOrder::BGR` to map channels (RGB -> BGR) without a manual swap.
//!
//! # Input Shape
//!
//! The model automatically parses input shape from the ONNX file. For models with
//! dynamic spatial dimensions, a default size must be provided.

use crate::core::OCRError;
use crate::core::config::InputShape;
use crate::core::inference::{OrtInfer, TensorInput};
use crate::processors::NormalizeImage;
use image::{DynamicImage, RgbImage};
use ndarray::s;
use std::path::Path;

/// Output from SLANet model containing structure predictions and bounding boxes.
#[derive(Debug, Clone)]
pub struct SLANetModelOutput {
    /// Structure token logits [batch, seq_len, vocab_size]
    pub structure_logits: ndarray::Array3<f32>,
    /// Bounding box predictions [batch, seq_len, 8] (4 corner points × 2 coords)
    pub bbox_preds: ndarray::Array3<f32>,
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
    /// Input shape parsed from ONNX model
    input_shape: InputShape,
}

impl SLANetModel {
    /// Creates a new SLANet model.
    pub fn new(inference: OrtInfer, normalizer: NormalizeImage, input_shape: InputShape) -> Self {
        Self {
            inference,
            normalizer,
            input_shape,
        }
    }

    /// Returns the input shape of this model.
    pub fn input_shape(&self) -> &InputShape {
        &self.input_shape
    }

    /// Preprocesses images for table structure recognition.
    ///
    /// Preprocessing follows PaddleX TablePredictor order:
    /// - ResizeByLong
    /// - Normalize
    /// - PaddingTableImage (for fixed-shape models)
    ///
    /// Important: padding happens **after normalization** and uses zero fill in normalized
    /// tensor space. This matches PaddleX behavior and avoids introducing large negative
    /// values from normalizing black pixels in padded regions.
    pub fn preprocess(
        &self,
        images: Vec<RgbImage>,
    ) -> Result<(ndarray::Array4<f32>, Vec<[f32; 6]>), OCRError> {
        let mut shape_info_list = Vec::with_capacity(images.len());
        let mut processed_tensors = Vec::with_capacity(images.len());

        // Determine preprocessing strategy based on input shape
        let needs_padding = self.input_shape.has_fixed_spatial();
        let (target_h, target_w) = self.input_shape.spatial_size_or(488, 488);
        let target_size = target_h.max(target_w) as f32;

        tracing::debug!(
            "SLANet preprocess: needs_padding={}, target_size={}",
            needs_padding,
            target_size
        );

        for img in images {
            let (orig_h, orig_w) = (img.height() as f32, img.width() as f32);

            // ResizeByLong - keep aspect ratio, longest side = target_size
            let longest_side = orig_h.max(orig_w);
            let scale = target_size / longest_side;
            let resized_h = (orig_h * scale).round() as u32;
            let resized_w = (orig_w * scale).round() as u32;

            tracing::debug!(
                "SLANet resize: orig={}x{}, resized={}x{}, scale={:.4}",
                orig_w,
                orig_h,
                resized_w,
                resized_h,
                scale
            );

            let resized = image::imageops::resize(
                &img,
                resized_w,
                resized_h,
                image::imageops::FilterType::Triangle,
            );

            // Normalize resized image first, then optionally pad in normalized space.
            let normalized = self
                .normalizer
                .normalize_to(DynamicImage::ImageRgb8(resized.clone()))?;

            let (tensor, pad_h, pad_w) = if needs_padding {
                let target_size_u32 = target_size as usize;
                let mut padded =
                    ndarray::Array4::<f32>::zeros((1, 3, target_size_u32, target_size_u32));
                padded
                    .slice_mut(s![0, .., 0..(resized_h as usize), 0..(resized_w as usize)])
                    .assign(&normalized.slice(s![0, .., .., ..]));

                let pad_h = target_size - (resized_h as f32);
                let pad_w = target_size - (resized_w as f32);
                (padded, pad_h, pad_w)
            } else {
                (normalized, 0.0, 0.0)
            };

            // Store shape information for bbox denormalization
            // [orig_h, orig_w, scale, pad_h, pad_w, target_size]
            // For non-padded case, target_size represents max_len used for ResizeByLong
            shape_info_list.push([orig_h, orig_w, scale, pad_h, pad_w, target_size]);

            processed_tensors.push(tensor);
        }

        // Concatenate per-image tensors into a batch tensor.
        let batch_tensor = if processed_tensors.is_empty() {
            ndarray::Array4::<f32>::zeros((0, 0, 0, 0))
        } else {
            let first_shape = processed_tensors[0].shape().to_vec();
            let (channels, height, width) = (first_shape[1], first_shape[2], first_shape[3]);
            if !processed_tensors.iter().all(|t| {
                t.shape()[1] == channels && t.shape()[2] == height && t.shape()[3] == width
            }) {
                return Err(OCRError::InvalidInput {
                    message: "SLANet preprocess produced tensors with inconsistent shapes"
                        .to_string(),
                });
            }

            let mut batch =
                ndarray::Array4::<f32>::zeros((processed_tensors.len(), channels, height, width));
            for (i, tensor) in processed_tensors.iter().enumerate() {
                batch
                    .slice_mut(s![i, .., .., ..])
                    .assign(&tensor.slice(s![0, .., .., ..]));
            }
            batch
        };

        Ok((batch_tensor, shape_info_list))
    }

    /// Runs inference on the preprocessed tensor.
    ///
    /// Returns dual outputs matching ONNX model order: (bbox predictions, structure logits).
    pub fn infer(
        &self,
        batch_tensor: &ndarray::Array4<f32>,
    ) -> Result<(ndarray::Array3<f32>, ndarray::Array3<f32>), OCRError> {
        let input_name = self.inference.input_name();
        let inputs = vec![(input_name, TensorInput::Array4(batch_tensor))];

        let outputs = self
            .inference
            .infer(&inputs)
            .map_err(|e| OCRError::Inference {
                model_name: "SLANet".to_string(),
                context: format!(
                    "failed to run inference on batch with shape {:?}",
                    batch_tensor.shape()
                ),
                source: Box::new(e),
            })?;

        if outputs.len() < 2 {
            return Err(OCRError::InvalidInput {
                message: format!("SLANet: expected at least 2 outputs, got {}", outputs.len()),
            });
        }

        let bbox_preds =
            outputs[0]
                .1
                .clone()
                .try_into_array3_f32()
                .map_err(|e| OCRError::Inference {
                    model_name: "SLANet".to_string(),
                    context: "failed to convert first output (bbox_preds) to 3D array".to_string(),
                    source: Box::new(e),
                })?;

        let structure_logits =
            outputs[1]
                .1
                .clone()
                .try_into_array3_f32()
                .map_err(|e| OCRError::Inference {
                    model_name: "SLANet".to_string(),
                    context: "failed to convert second output (structure_logits) to 3D array"
                        .to_string(),
                    source: Box::new(e),
                })?;

        Ok((bbox_preds, structure_logits))
    }

    /// Runs the complete forward pass: preprocess -> infer.
    ///
    /// Postprocessing is handled separately by TableStructureDecode.
    ///
    /// # Model Output Order
    ///
    /// The ONNX model outputs are in this order:
    /// - `fetch_name_0`: bbox predictions [batch, seq, 8]
    /// - `fetch_name_1`: structure logits [batch, seq, vocab_size]
    pub fn forward(&self, images: Vec<RgbImage>) -> Result<SLANetModelOutput, OCRError> {
        let (batch_tensor, shape_info) = self.preprocess(images)?;
        // CRITICAL: Model outputs bbox first, then structure logits
        let (bbox_preds, structure_logits) = self.infer(&batch_tensor)?;

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
    /// Input shape override (if None, will be parsed from ONNX)
    input_shape: Option<InputShape>,
    /// ONNX Runtime session configuration
    ort_config: Option<crate::core::config::OrtSessionConfig>,
}

impl SLANetModelBuilder {
    /// Creates a new SLANet model builder.
    pub fn new() -> Self {
        Self {
            input_shape: None,
            ort_config: None,
        }
    }

    /// Sets the input image size (height, width).
    ///
    /// This is a convenience method that creates an InputShape with dynamic batch.
    /// For more control, use `input_shape()` directly.
    pub fn input_size(mut self, size: (u32, u32)) -> Self {
        self.input_shape = Some(InputShape::dynamic_batch(3, size.0 as i64, size.1 as i64));
        self
    }

    /// Sets the input shape directly.
    ///
    /// If not set, the input shape will be parsed from the ONNX model.
    pub fn input_shape(mut self, shape: InputShape) -> Self {
        self.input_shape = Some(shape);
        self
    }

    /// Sets the ONNX Runtime session configuration.
    pub fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.ort_config = Some(config);
        self
    }

    /// Builds the SLANet model.
    ///
    /// The input shape is determined in the following order:
    /// 1. If explicitly set via `input_shape()` or `input_size()`, use that
    /// 2. Otherwise, parse from ONNX model metadata
    /// 3. If ONNX has dynamic spatial dimensions, use default 512x512
    pub fn build(self, model_path: &Path) -> Result<SLANetModel, OCRError> {
        // Create ONNX inference engine
        let inference = if self.ort_config.is_some() {
            use crate::core::config::ModelInferenceConfig;
            let common_config = ModelInferenceConfig {
                ort_session: self.ort_config,
                ..Default::default()
            };
            OrtInfer::from_config(&common_config, model_path, None)?
        } else {
            OrtInfer::new(model_path, None)?
        };

        // Determine input shape: user override > ONNX metadata > default
        let input_shape = if let Some(shape) = self.input_shape {
            // User explicitly set input shape
            shape
        } else if let Some(onnx_dims) = inference.primary_input_shape() {
            // Try to parse from ONNX model
            InputShape::from_onnx_dims(&onnx_dims).unwrap_or_else(|| {
                // Fallback to default if ONNX parsing fails
                InputShape::dynamic_batch(3, 512, 512)
            })
        } else {
            // No ONNX shape available, use default
            InputShape::dynamic_batch(3, 512, 512)
        };

        tracing::debug!(
            "SLANet input shape: {} (fixed_spatial: {})",
            input_shape,
            input_shape.has_fixed_spatial()
        );

        // Create normalizer for SLANet.
        // Configs normalize BGR images with ImageNet stats in that order
        // (B, G, R). Our images are loaded as RGB; rely on `ColorOrder::BGR` to map channels.
        let normalizer = NormalizeImage::with_color_order(
            Some(1.0 / 255.0),
            Some(vec![0.485, 0.456, 0.406]), // mean
            Some(vec![0.229, 0.224, 0.225]), // std
            None,
            Some(crate::processors::types::ColorOrder::BGR),
        )?;

        Ok(SLANetModel::new(inference, normalizer, input_shape))
    }
}
