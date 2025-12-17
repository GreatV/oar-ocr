//! PP-LCNet Classification Model
//!
//! This module provides a pure implementation of the PP-LCNet model for image classification.
//! PP-LCNet is a lightweight classification network that can be used for various classification
//! tasks such as document orientation and text line orientation.

use crate::core::inference::OrtInfer;
use crate::core::{OCRError, Tensor2D, Tensor4D};
use crate::domain::adapters::preprocessing::rgb_to_dynamic;
use crate::processors::{ChannelOrder, NormalizeImage};
use crate::utils::topk::Topk;
use image::{RgbImage, imageops::FilterType};

/// Configuration for PP-LCNet model preprocessing.
#[derive(Debug, Clone)]
pub struct PPLCNetPreprocessConfig {
    /// Input shape (height, width)
    pub input_shape: (u32, u32),
    /// Resizing filter to use
    pub resize_filter: FilterType,
    /// When set, resize by short edge to this size (keep ratio) then center-crop to `input_shape`.
    ///
    /// This matches `ResizeImage(resize_short=256)` + `CropImage(size=224)` used by
    /// most PP-LCNet classifiers (e.g. `PP-LCNet_x1_0_doc_ori`, `PP-LCNet_x1_0_table_cls`).
    ///
    /// Some specialized PP-LCNet classifiers (e.g. `PP-LCNet_x1_0_textline_ori`) use a direct
    /// resize to a fixed `(height,width)` without short-edge resize/crop; set this to `None`
    /// to match `ResizeImage(size=[w,h])`.
    pub resize_short: Option<u32>,
    /// Scaling factor applied before normalization (defaults to 1.0 / 255.0)
    pub normalize_scale: f32,
    /// Mean values for normalization
    pub normalize_mean: Vec<f32>,
    /// Standard deviation values for normalization
    pub normalize_std: Vec<f32>,
    /// Channel ordering for the normalized tensor
    pub channel_order: ChannelOrder,
}

impl Default for PPLCNetPreprocessConfig {
    fn default() -> Self {
        Self {
            input_shape: (224, 224),
            // Use cv2.INTER_LINEAR for PP-LCNet resize.
            resize_filter: FilterType::Triangle,
            // PP-LCNet classifiers default to resize_short=256 then center-crop.
            resize_short: Some(256),
            normalize_scale: 1.0 / 255.0,
            normalize_mean: vec![0.485, 0.456, 0.406],
            normalize_std: vec![0.229, 0.224, 0.225],
            channel_order: ChannelOrder::CHW,
        }
    }
}

/// Configuration for PP-LCNet model postprocessing.
#[derive(Debug, Clone)]
pub struct PPLCNetPostprocessConfig {
    /// Class labels
    pub labels: Vec<String>,
    /// Number of top predictions to return
    pub topk: usize,
}

impl Default for PPLCNetPostprocessConfig {
    fn default() -> Self {
        Self {
            labels: vec![],
            topk: 1,
        }
    }
}

/// Output from PP-LCNet model.
#[derive(Debug, Clone)]
pub struct PPLCNetModelOutput {
    /// Predicted class IDs per image
    pub class_ids: Vec<Vec<usize>>,
    /// Confidence scores for each prediction
    pub scores: Vec<Vec<f32>>,
    /// Label names for each prediction (if labels provided)
    pub label_names: Option<Vec<Vec<String>>>,
}

/// Pure PP-LCNet model implementation.
///
/// This model performs image classification using the PP-LCNet architecture.
#[derive(Debug)]
pub struct PPLCNetModel {
    /// ONNX Runtime inference engine
    inference: OrtInfer,
    /// Image normalizer for preprocessing
    normalizer: NormalizeImage,
    /// Top-k processor for postprocessing
    topk_processor: Topk,
    /// Input shape (height, width)
    input_shape: (u32, u32),
    /// Resizing filter
    resize_filter: FilterType,
    /// Optional short-edge resize (see `PPLCNetPreprocessConfig::resize_short`)
    resize_short: Option<u32>,
}

impl PPLCNetModel {
    /// Creates a new PP-LCNet model.
    pub fn new(
        inference: OrtInfer,
        normalizer: NormalizeImage,
        topk_processor: Topk,
        input_shape: (u32, u32),
        resize_filter: FilterType,
        resize_short: Option<u32>,
    ) -> Self {
        Self {
            inference,
            normalizer,
            topk_processor,
            input_shape,
            resize_filter,
            resize_short,
        }
    }

    /// Preprocesses images for classification.
    ///
    /// # Arguments
    ///
    /// * `images` - Input images to preprocess
    ///
    /// # Returns
    ///
    /// Preprocessed batch tensor
    pub fn preprocess(&self, images: Vec<RgbImage>) -> Result<Tensor4D, OCRError> {
        let (crop_h, crop_w) = self.input_shape;

        let resized_rgb: Vec<RgbImage> = if let Some(resize_short) = self.resize_short {
            // PP-LCNet classifier preprocessing:
            // 1) Resize by short edge (keep ratio)
            // 2) Center crop to the model input size
            //
            // This matches `ResizeImage(resize_short=256)` + `CropImage(size=224)` used by
            // models like `PP-LCNet_x1_0_doc_ori` / `PP-LCNet_x1_0_table_cls`.
            images
                .into_iter()
                .filter_map(|img| {
                    let (w, h) = (img.width(), img.height());
                    if w == 0 || h == 0 {
                        return None;
                    }

                    let short = w.min(h) as f32;
                    let scale = (resize_short as f32) / short;
                    let new_w = ((w as f32) * scale).round().max(crop_w as f32) as u32;
                    let new_h = ((h as f32) * scale).round().max(crop_h as f32) as u32;

                    let resized = image::imageops::resize(&img, new_w, new_h, self.resize_filter);

                    // Center crop to (crop_w, crop_h)
                    let x1 = (new_w.saturating_sub(crop_w)) / 2;
                    let y1 = (new_h.saturating_sub(crop_h)) / 2;
                    let cropped =
                        image::imageops::crop_imm(&resized, x1, y1, crop_w, crop_h).to_image();
                    Some(cropped)
                })
                .collect()
        } else {
            // Direct resize to input shape (height,width) without crop.
            // This matches `ResizeImage(size=[w,h])` used by
            // `PP-LCNet_x1_0_textline_ori` (80x160).
            images
                .into_iter()
                .filter_map(|img| {
                    let (w, h) = (img.width(), img.height());
                    if w == 0 || h == 0 {
                        return None;
                    }
                    Some(image::imageops::resize(
                        &img,
                        crop_w,
                        crop_h,
                        self.resize_filter,
                    ))
                })
                .collect()
        };

        // Convert to dynamic images and normalize using common helper
        let dynamic_images = rgb_to_dynamic(resized_rgb);
        self.normalizer.normalize_batch_to(dynamic_images)
    }

    /// Runs inference on the preprocessed batch.
    ///
    /// # Arguments
    ///
    /// * `batch_tensor` - Preprocessed batch tensor
    ///
    /// # Returns
    ///
    /// Model predictions as a 2D tensor (batch_size x num_classes)
    pub fn infer(&self, batch_tensor: &Tensor4D) -> Result<Tensor2D, OCRError> {
        self.inference
            .infer_2d(batch_tensor)
            .map_err(|e| OCRError::Inference {
                model_name: "PP-LCNet".to_string(),
                context: format!(
                    "failed to run inference on batch with shape {:?}",
                    batch_tensor.shape()
                ),
                source: Box::new(e),
            })
    }

    /// Postprocesses model predictions to class IDs and scores.
    ///
    /// # Arguments
    ///
    /// * `predictions` - Model predictions (batch_size x num_classes)
    /// * `config` - Postprocessing configuration
    ///
    /// # Returns
    ///
    /// PPLCNetModelOutput containing class IDs, scores, and optional label names
    pub fn postprocess(
        &self,
        predictions: &Tensor2D,
        config: &PPLCNetPostprocessConfig,
    ) -> Result<PPLCNetModelOutput, OCRError> {
        let predictions_vec: Vec<Vec<f32>> =
            predictions.outer_iter().map(|row| row.to_vec()).collect();

        let topk_result = self
            .topk_processor
            .process(&predictions_vec, config.topk)
            .unwrap_or_else(|_| crate::utils::topk::TopkResult {
                indexes: vec![],
                scores: vec![],
                class_names: None,
            });

        let class_ids = topk_result.indexes;
        let scores = topk_result.scores;

        // Map class IDs to label names if labels are provided
        let label_names = if !config.labels.is_empty() {
            Some(
                class_ids
                    .iter()
                    .map(|ids| {
                        ids.iter()
                            .map(|&id| {
                                config
                                    .labels
                                    .get(id)
                                    .cloned()
                                    .unwrap_or_else(|| format!("class_{}", id))
                            })
                            .collect()
                    })
                    .collect(),
            )
        } else {
            topk_result.class_names
        };

        Ok(PPLCNetModelOutput {
            class_ids,
            scores,
            label_names,
        })
    }

    /// Performs complete forward pass: preprocess -> infer -> postprocess.
    ///
    /// # Arguments
    ///
    /// * `images` - Input images to classify
    /// * `config` - Postprocessing configuration
    ///
    /// # Returns
    ///
    /// PPLCNetModelOutput containing classification results
    pub fn forward(
        &self,
        images: Vec<RgbImage>,
        config: &PPLCNetPostprocessConfig,
    ) -> Result<PPLCNetModelOutput, OCRError> {
        let batch_tensor = self.preprocess(images)?;
        let predictions = self.infer(&batch_tensor)?;
        self.postprocess(&predictions, config)
    }
}

/// Builder for PP-LCNet model.
#[derive(Debug, Default)]
pub struct PPLCNetModelBuilder {
    /// Session pool size for ONNX Runtime
    session_pool_size: usize,
    /// Preprocessing configuration
    preprocess_config: PPLCNetPreprocessConfig,
    /// ONNX Runtime session configuration
    ort_config: Option<crate::core::config::OrtSessionConfig>,
}

impl PPLCNetModelBuilder {
    /// Creates a new PP-LCNet model builder.
    pub fn new() -> Self {
        Self {
            session_pool_size: 1,
            preprocess_config: PPLCNetPreprocessConfig::default(),
            ort_config: None,
        }
    }

    /// Sets the session pool size for ONNX Runtime.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.session_pool_size = size;
        self
    }

    /// Sets the preprocessing configuration.
    pub fn preprocess_config(mut self, config: PPLCNetPreprocessConfig) -> Self {
        self.preprocess_config = config;
        self
    }

    /// Sets the input image shape.
    pub fn input_shape(mut self, shape: (u32, u32)) -> Self {
        self.preprocess_config.input_shape = shape;
        self
    }

    /// Sets the resizing filter.
    pub fn resize_filter(mut self, filter: FilterType) -> Self {
        self.preprocess_config.resize_filter = filter;
        self
    }

    /// Sets the ONNX Runtime session configuration.
    pub fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.ort_config = Some(config);
        self
    }

    /// Builds the PP-LCNet model.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the ONNX model file
    ///
    /// # Returns
    ///
    /// A configured PP-LCNet model instance
    pub fn build(self, model_path: &std::path::Path) -> Result<PPLCNetModel, OCRError> {
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

        // Create normalizer (ImageNet normalization).
        //
        // PP-LCNet classifiers read images as **RGB** by default (no `DecodeImage`
        // op in the official inference.yml), so we keep RGB order here.
        let mean = self.preprocess_config.normalize_mean.clone();
        let std = self.preprocess_config.normalize_std.clone();
        let normalizer = NormalizeImage::with_color_order(
            Some(self.preprocess_config.normalize_scale),
            Some(mean),
            Some(std),
            Some(self.preprocess_config.channel_order.clone()),
            Some(crate::processors::ColorOrder::RGB),
        )?;

        // Create top-k processor
        let topk_processor = Topk::new(None);

        Ok(PPLCNetModel::new(
            inference,
            normalizer,
            topk_processor,
            self.preprocess_config.input_shape,
            self.preprocess_config.resize_filter,
            self.preprocess_config.resize_short,
        ))
    }
}
