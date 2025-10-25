//! SLANet (Structure Location Alignment Network) Model
//!
//! This module provides a pure implementation of the SLANet table structure recognition model.
//! The model predicts both HTML structure tokens and bounding boxes for table cells.

use crate::core::inference::OrtInfer;
use crate::core::{OCRError, Tensor3D, Tensor4D};
use crate::processors::NormalizeImage;
use image::{DynamicImage, RgbImage};
use ndarray::Array3;
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
        use ort::value::TensorRef;

        let input_shape = batch_tensor.shape().to_vec();
        tracing::trace!(
            "Running SLANet inference with input shape {:?}",
            input_shape
        );
        let input_tensor =
            TensorRef::from_array_view(batch_tensor.view()).map_err(|e| OCRError::ConfigError {
                message: format!("Failed to convert input tensor: {}", e),
            })?;

        let inputs = ort::inputs![self.inference.input_name() => input_tensor];

        // Get session and run inference
        let idx = 0; // Use first session for simplicity
        let mut session_guard =
            self.inference
                .get_session(idx)
                .map_err(|e| OCRError::ConfigError {
                    message: format!("Failed to acquire session: {}", e),
                })?;

        // Get output names from session before running inference
        let output_names: Vec<String> = session_guard
            .outputs
            .iter()
            .map(|output| output.name.clone())
            .collect();

        tracing::debug!("Model outputs: {:?}", output_names);

        let outputs = session_guard
            .run(inputs)
            .map_err(|e| OCRError::ConfigError {
                message: format!("Inference failed: {}", e),
            })?;

        // Extract structure logits and bbox predictions
        // Try to find outputs by index since names may vary
        let structure_output = if output_names.len() >= 2 {
            // Assume first output is structure, second is bbox
            outputs[output_names[0].as_str()]
                .try_extract_tensor::<f32>()
                .map_err(|e| OCRError::ConfigError {
                    message: format!(
                        "Failed to extract structure tensor from '{}': {}",
                        output_names[0], e
                    ),
                })?
        } else {
            return Err(OCRError::ConfigError {
                message: format!("Expected at least 2 outputs, got {}", output_names.len()),
            });
        };

        let bbox_output = outputs[output_names[1].as_str()]
            .try_extract_tensor::<f32>()
            .map_err(|e| OCRError::ConfigError {
                message: format!(
                    "Failed to extract bbox tensor from '{}': {}",
                    output_names[1], e
                ),
            })?;

        let (structure_shape, structure_data) = structure_output;
        let (bbox_shape, bbox_data) = bbox_output;

        // Convert to 3D tensors
        let structure_logits = Self::reshape_to_3d(structure_shape, structure_data)?;
        let bbox_preds = Self::reshape_to_3d(bbox_shape, bbox_data)?;

        Ok((structure_logits, bbox_preds))
    }

    /// Helper to reshape flat data to 3D tensor.
    fn reshape_to_3d(shape: &[i64], data: &[f32]) -> Result<Tensor3D, OCRError> {
        if shape.len() != 3 {
            return Err(OCRError::InvalidInput {
                message: format!("Expected 3D shape, got {:?}", shape),
            });
        }

        let dim0 = shape[0] as usize;
        let dim1 = shape[1] as usize;
        let dim2 = shape[2] as usize;

        Array3::from_shape_vec((dim0, dim1, dim2), data.to_vec()).map_err(|e| {
            OCRError::InvalidInput {
                message: format!("Failed to reshape tensor: {}", e),
            }
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
}

impl SLANetModelBuilder {
    /// Creates a new SLANet model builder.
    pub fn new() -> Self {
        Self {
            session_pool_size: 1,
            input_size: (512, 512),
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

    /// Builds the SLANet model.
    pub fn build(self, model_path: &Path) -> Result<SLANetModel, OCRError> {
        // Create ONNX inference engine
        let inference = if self.session_pool_size > 1 {
            use crate::core::config::CommonBuilderConfig;
            let common_config = CommonBuilderConfig {
                model_path: None,
                model_name: None,
                batch_size: None,
                enable_logging: None,
                ort_session: None,
                session_pool_size: Some(self.session_pool_size),
            };
            OrtInfer::from_common(&common_config, model_path, None)?
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
