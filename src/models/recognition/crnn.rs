//! CRNN (Convolutional Recurrent Neural Network) Model
//!
//! This module provides a pure implementation of the CRNN text recognition model.
//! The model handles preprocessing, inference, and postprocessing independently of tasks.

use crate::core::inference::OrtInfer;
use crate::core::{OCRError, Tensor3D, Tensor4D};
use crate::processors::{CTCLabelDecode, OCRResize};
use image::RgbImage;
use std::path::Path;

/// CRNN model output containing recognized text and confidence scores.
#[derive(Debug, Clone)]
pub struct CRNNModelOutput {
    /// Recognized text strings for each image in the batch
    pub texts: Vec<String>,
    /// Confidence scores for each recognized text
    pub scores: Vec<f32>,
    /// Character positions (normalized 0.0-1.0) for each text line
    /// Only populated when return_word_box is enabled
    pub char_positions: Vec<Vec<f32>>,
    /// Column indices for each character in the CTC output
    /// Used for accurate word box generation. Each value is the timestep index.
    pub char_col_indices: Vec<Vec<usize>>,
    /// Total number of columns (sequence length) in the CTC output for each text line
    pub sequence_lengths: Vec<usize>,
}

/// Pure CRNN model implementation.
///
/// This model implements the CRNN architecture for text recognition.
#[derive(Debug)]
pub struct CRNNModel {
    /// ONNX Runtime inference engine
    inference: OrtInfer,
    /// Image resizer for preprocessing
    resizer: OCRResize,
    /// CTC decoder for postprocessing
    decoder: CTCLabelDecode,
}

impl CRNNModel {
    /// Creates a new CRNN model.
    pub fn new(inference: OrtInfer, resizer: OCRResize, decoder: CTCLabelDecode) -> Self {
        Self {
            inference,
            resizer,
            decoder,
        }
    }

    /// Preprocesses images for recognition.
    ///
    /// # Arguments
    ///
    /// * `images` - Input RGB images
    ///
    /// # Returns
    ///
    /// A 4D tensor ready for inference
    pub fn preprocess(&self, images: Vec<RgbImage>) -> Result<Tensor4D, OCRError> {
        if images.is_empty() {
            return Ok(ndarray::Array4::zeros((0, 0, 0, 0)));
        }

        // Match standard behavior:
        // 1. Calculate max_wh_ratio to determine final tensor width
        // 2. For each image: resize maintaining aspect ratio, normalize, pad with zeros
        let [_img_c, img_h, img_w] = self.resizer.rec_image_shape;
        let base_ratio = img_w as f32 / img_h.max(1) as f32;
        let max_wh_ratio = images
            .iter()
            .map(|img| img.width() as f32 / img.height().max(1) as f32)
            .fold(base_ratio, |acc, r| acc.max(r));

        // Calculate final tensor width
        let tensor_width = ((img_h as f32 * max_wh_ratio) as usize).min(self.resizer.max_img_w);

        // Process each image: resize → normalize → pad
        let batch_size = images.len();
        let mut batch_tensor = ndarray::Array4::<f32>::zeros((batch_size, 3, img_h, tensor_width));

        for (batch_idx, img) in images.iter().enumerate() {
            let (orig_w, orig_h) = (img.width() as f32, img.height() as f32);
            let ratio = orig_w / orig_h;

            // Calculate resize width
            let resized_w = ((img_h as f32 * ratio).ceil() as usize).min(tensor_width);

            // Resize image (without padding)
            let resized = image::imageops::resize(
                img,
                resized_w as u32,
                img_h as u32,
                image::imageops::FilterType::Triangle,
            );

            // Normalize and copy to tensor with zero padding
            // Channel order: BGR, so we need to swap channels
            // Normalization: (pixel / 255 - 0.5) / 0.5
            for y in 0..img_h {
                for x in 0..resized_w {
                    let pixel = resized.get_pixel(x as u32, y as u32);
                    // BGR order for PaddlePaddle models
                    let b = (pixel[2] as f32 / 255.0 - 0.5) / 0.5;
                    let g = (pixel[1] as f32 / 255.0 - 0.5) / 0.5;
                    let r = (pixel[0] as f32 / 255.0 - 0.5) / 0.5;

                    batch_tensor[[batch_idx, 0, y, x]] = b;
                    batch_tensor[[batch_idx, 1, y, x]] = g;
                    batch_tensor[[batch_idx, 2, y, x]] = r;
                }
            }
            // Rest of the tensor remains zero (zero-padding)
        }

        Ok(batch_tensor)
    }

    /// Runs inference on the preprocessed tensor.
    ///
    /// # Arguments
    ///
    /// * `batch_tensor` - Preprocessed 4D tensor
    ///
    /// # Returns
    ///
    /// A 3D tensor containing CTC predictions
    pub fn infer(&self, batch_tensor: &Tensor4D) -> Result<Tensor3D, OCRError> {
        self.inference
            .infer_3d(batch_tensor)
            .map_err(|e| OCRError::Inference {
                model_name: "CRNN".to_string(),
                context: format!(
                    "failed to run inference on batch with shape {:?}",
                    batch_tensor.shape()
                ),
                source: Box::new(e),
            })
    }

    /// Postprocesses model predictions to text strings.
    ///
    /// # Arguments
    ///
    /// * `predictions` - 3D tensor from model inference
    /// * `return_positions` - Whether to return character positions for word boxes
    ///
    /// # Returns
    ///
    /// Model output containing recognized texts, scores, and optionally character positions
    pub fn postprocess(&self, predictions: &Tensor3D, return_positions: bool) -> CRNNModelOutput {
        if return_positions {
            // Decode CTC predictions with character positions and column indices
            let (texts, scores, char_positions, char_col_indices, sequence_lengths) =
                self.decoder.apply_with_positions(predictions);
            CRNNModelOutput {
                texts,
                scores,
                char_positions,
                char_col_indices,
                sequence_lengths,
            }
        } else {
            // Decode CTC predictions without positions
            let (texts, scores) = self.decoder.apply(predictions);
            CRNNModelOutput {
                texts,
                scores,
                char_positions: Vec::new(),
                char_col_indices: Vec::new(),
                sequence_lengths: Vec::new(),
            }
        }
    }

    /// Runs the complete forward pass: preprocess -> infer -> postprocess.
    ///
    /// # Arguments
    ///
    /// * `images` - Input RGB images
    /// * `return_positions` - Whether to return character positions for word boxes
    ///
    /// # Returns
    ///
    /// Model output containing recognized texts, scores, and optionally character positions
    pub fn forward(
        &self,
        images: Vec<RgbImage>,
        return_positions: bool,
    ) -> Result<CRNNModelOutput, OCRError> {
        tracing::debug!("CRNN forward: {} images", images.len());
        if !images.is_empty() {
            tracing::debug!(
                "First image size: {}x{}",
                images[0].width(),
                images[0].height()
            );
        }
        let batch_tensor = self.preprocess(images)?;
        tracing::debug!("CRNN preprocess output shape: {:?}", batch_tensor.shape());
        let predictions = self.infer(&batch_tensor)?;
        tracing::debug!("CRNN infer output shape: {:?}", predictions.shape());
        let output = self.postprocess(&predictions, return_positions);
        tracing::debug!(
            "CRNN postprocess: {} texts, first 3: {:?}",
            output.texts.len(),
            &output.texts[..3.min(output.texts.len())]
        );
        Ok(output)
    }
}

/// Configuration for CRNN model preprocessing.
#[derive(Debug, Clone)]
pub struct CRNNPreprocessConfig {
    /// Model input shape [channels, height, width]
    pub model_input_shape: [usize; 3],
    /// Maximum image width (None for dynamic width)
    pub max_img_w: Option<usize>,
}

impl Default for CRNNPreprocessConfig {
    fn default() -> Self {
        Self {
            model_input_shape: [3, 48, 320],
            max_img_w: None,
        }
    }
}

/// Builder for CRNN model.
pub struct CRNNModelBuilder {
    /// Preprocessing configuration
    preprocess_config: CRNNPreprocessConfig,
    /// Character dictionary
    character_dict: Option<Vec<String>>,
    /// Session pool size for ONNX Runtime
    session_pool_size: usize,
    /// ONNX Runtime session configuration
    ort_config: Option<crate::core::config::OrtSessionConfig>,
}

impl CRNNModelBuilder {
    /// Creates a new CRNN model builder with default settings.
    pub fn new() -> Self {
        Self {
            preprocess_config: CRNNPreprocessConfig::default(),
            character_dict: None,
            session_pool_size: 1,
            ort_config: None,
        }
    }

    /// Sets the preprocessing configuration.
    pub fn preprocess_config(mut self, config: CRNNPreprocessConfig) -> Self {
        self.preprocess_config = config;
        self
    }

    /// Sets the model input shape.
    pub fn model_input_shape(mut self, shape: [usize; 3]) -> Self {
        self.preprocess_config.model_input_shape = shape;
        self
    }

    /// Sets the character dictionary.
    pub fn character_dict(mut self, character_dict: Vec<String>) -> Self {
        self.character_dict = Some(character_dict);
        self
    }

    /// Sets the maximum image width.
    pub fn max_img_w(mut self, max_img_w: usize) -> Self {
        self.preprocess_config.max_img_w = Some(max_img_w);
        self
    }

    /// Sets the session pool size.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.session_pool_size = size;
        self
    }

    /// Sets the ONNX Runtime session configuration.
    pub fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.ort_config = Some(config);
        self
    }

    /// Builds the CRNN model.
    pub fn build(self, model_path: &Path) -> Result<CRNNModel, OCRError> {
        // Create ONNX inference engine
        let inference = if self.session_pool_size > 1 || self.ort_config.is_some() {
            // Use session pool for concurrent inference
            let common = crate::core::config::ModelInferenceConfig {
                session_pool_size: Some(self.session_pool_size),
                ort_session: self.ort_config,
                ..Default::default()
            };
            OrtInfer::from_config(&common, model_path, None)?
        } else {
            // Single session
            OrtInfer::new(model_path, None)?
        };

        // Create resizer
        let resizer = OCRResize::new(Some(self.preprocess_config.model_input_shape), None);

        // Create CTC decoder
        let decoder = if let Some(character_dict) = self.character_dict {
            CTCLabelDecode::from_string_list(Some(&character_dict), true, false)
        } else {
            // Use default character dictionary
            CTCLabelDecode::new(None, true)
        };

        Ok(CRNNModel::new(inference, resizer, decoder))
    }
}

impl Default for CRNNModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}
