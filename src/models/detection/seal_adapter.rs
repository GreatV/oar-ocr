//! Seal text detection adapter for PP-OCR seal detection models.
//!
//! This module provides the adapter for seal/stamp text detection models,
//! which are specialized for detecting curved text in circular stamps.

use crate::core::inference::OrtInfer;
use crate::core::traits::adapter::{AdapterBuilder, AdapterInfo, ModelAdapter};
use crate::core::traits::task::{Task, TaskType};
use crate::core::{OCRError, Tensor4D};
use crate::domain::tasks::{
    SealTextDetectionConfig, SealTextDetectionOutput, SealTextDetectionTask,
};
use crate::processors::{
    BoxType, ChannelOrder, DBPostProcess, DetResizeForTest, NormalizeImage, ScoreMode,
};
use image::{DynamicImage, RgbImage};
use std::path::Path;
use tracing::debug;

/// Adapter for seal text detection models.
///
/// This adapter is based on the DB (Differentiable Binarization) architecture
/// but optimized for detecting text in seal and stamp images, where text
/// often follows curved paths.
#[derive(Debug)]
pub struct SealTextDetectionAdapter {
    inference: OrtInfer,
    resizer: DetResizeForTest,
    normalizer: NormalizeImage,
    postprocessor: DBPostProcess,
    info: AdapterInfo,
    config: SealTextDetectionConfig,
}

impl SealTextDetectionAdapter {
    /// Preprocesses images for the model.
    fn preprocess(&self, images: Vec<RgbImage>) -> Result<(Tensor4D, Vec<[f32; 4]>), OCRError> {
        // Convert to DynamicImage
        let dynamic_images: Vec<DynamicImage> =
            images.into_iter().map(DynamicImage::ImageRgb8).collect();

        // Apply detection resizing (default limit: 960px)
        let (resized_images, img_shapes) = self.resizer.apply(
            dynamic_images,
            None, // Use default limit_side_len
            None, // Use default limit_type
            None, // Use default max_side_limit
        );

        debug!("After resize: {} images", resized_images.len());
        for (i, (img, shape)) in resized_images.iter().zip(&img_shapes).enumerate() {
            debug!(
                "  Image {}: {}x{}, shape=[src_h={:.0}, src_w={:.0}, ratio_h={:.3}, ratio_w={:.3}]",
                i,
                img.width(),
                img.height(),
                shape[0],
                shape[1],
                shape[2],
                shape[3]
            );
        }

        // Convert resized images from RGB (image crate default) to BGR to match
        // the original Paddle preprocessing implementation that uses OpenCV.
        // This ensures mean/std normalization happens over the expected channel
        // order and keeps detection outputs aligned with the Python reference.
        let bgr_images: Vec<DynamicImage> = resized_images
            .into_iter()
            .map(|img| {
                let mut rgb = img.to_rgb8();
                for pixel in rgb.pixels_mut() {
                    pixel.0.swap(0, 2);
                }
                DynamicImage::ImageRgb8(rgb)
            })
            .collect();

        // Apply ImageNet normalization and convert to tensor
        let batch_tensor = self.normalizer.normalize_batch_to(bgr_images)?;
        debug!("Batch tensor shape: {:?}", batch_tensor.shape());

        // Debug: Print tensor statistics per channel
        if let Some(tensor_slice) = batch_tensor.as_slice() {
            let shape = batch_tensor.shape();
            if shape.len() == 4 && shape[0] > 0 {
                let c = shape[1];
                let h = shape[2];
                let w = shape[3];
                let channel_size = h * w;

                for ch in 0..c {
                    let start = ch * channel_size;
                    let end = start + channel_size;
                    let channel_data = &tensor_slice[start..end];

                    let sum: f32 = channel_data.iter().sum();
                    let mean = sum / channel_size as f32;

                    let variance: f32 = channel_data
                        .iter()
                        .map(|&x| (x - mean).powi(2))
                        .sum::<f32>()
                        / channel_size as f32;
                    let std = variance.sqrt();

                    debug!("  Channel {}: mean={:.4}, std={:.4}", ch, mean, std);
                }
            }
        }

        Ok((batch_tensor, img_shapes))
    }

    /// Postprocesses model predictions to extract bounding boxes.
    fn postprocess(
        &self,
        predictions: &Tensor4D,
        img_shapes: Vec<[f32; 4]>,
        config: &SealTextDetectionConfig,
    ) -> SealTextDetectionOutput {
        debug!("Predictions tensor shape: {:?}", predictions.shape());
        debug!("Image shapes for postprocessing: {:?}", img_shapes);

        // Debug: Print prediction map statistics
        if let Some(pred_map) = predictions.slice(ndarray::s![0, 0, .., ..]).as_slice() {
            let min_val = pred_map.iter().fold(f32::INFINITY, |a, &b| a.min(b));
            let max_val = pred_map.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
            let pixels_above_02 = pred_map.iter().filter(|&&x| x > 0.2).count();
            let pixels_above_06 = pred_map.iter().filter(|&&x| x > 0.6).count();
            debug!("Prediction map range: [{:.4}, {:.4}]", min_val, max_val);
            debug!("Pixels above 0.2: {}", pixels_above_02);
            debug!("Pixels above 0.6: {}", pixels_above_06);
        }

        let (boxes, scores) = self.postprocessor.apply(
            predictions,
            img_shapes,
            Some(config.score_threshold),
            Some(config.box_threshold),
            Some(config.unclip_ratio),
        );

        SealTextDetectionOutput { boxes, scores }
    }
}

impl ModelAdapter for SealTextDetectionAdapter {
    type Task = SealTextDetectionTask;

    fn info(&self) -> AdapterInfo {
        self.info.clone()
    }

    fn execute(
        &self,
        input: <Self::Task as Task>::Input,
        config: Option<&<Self::Task as Task>::Config>,
    ) -> Result<<Self::Task as Task>::Output, OCRError> {
        debug!(
            "Executing seal text detection for {} images",
            input.images.len()
        );

        // Use provided config or fall back to stored config
        let effective_config = config.unwrap_or(&self.config);

        // Preprocess
        let (batch_tensor, img_shapes) = self.preprocess(input.images)?;

        // Run inference
        let predictions = self.inference.infer_4d(&batch_tensor)?;

        // Postprocess
        let output = self.postprocess(&predictions, img_shapes, effective_config);

        debug!(
            "Detected {} seal text regions total",
            output.boxes.iter().map(|b| b.len()).sum::<usize>()
        );

        Ok(output)
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn recommended_batch_size(&self) -> usize {
        8
    }
}

/// Builder for seal text detection adapters.
pub struct SealTextDetectionAdapterBuilder {
    model_name: String,
    session_pool_size: usize,
}

impl SealTextDetectionAdapterBuilder {
    /// Creates a new builder.
    pub fn new() -> Self {
        Self {
            model_name: "PP-OCRv4_mobile_seal_det".to_string(),
            session_pool_size: 1,
        }
    }

    /// Sets the model name.
    pub fn model_name<S: Into<String>>(mut self, name: S) -> Self {
        self.model_name = name.into();
        self
    }

    /// Sets the session pool size for concurrent inference.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.session_pool_size = size;
        self
    }
}

impl Default for SealTextDetectionAdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AdapterBuilder for SealTextDetectionAdapterBuilder {
    type Config = SealTextDetectionConfig;
    type Adapter = SealTextDetectionAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        if !model_path.exists() {
            return Err(OCRError::InvalidInput {
                message: format!("Model file not found: {:?}", model_path),
            });
        }

        debug!("Building seal text detection adapter from {:?}", model_path);

        // Create inference engine
        let inference = if self.session_pool_size > 1 {
            use crate::core::config::CommonBuilderConfig;
            let common_config = CommonBuilderConfig {
                session_pool_size: Some(self.session_pool_size),
                ..Default::default()
            };
            OrtInfer::from_common(&common_config, model_path, Some("x"))?
        } else {
            OrtInfer::new(model_path, Some("x"))?
        };

        // Create preprocessing components
        // Seal detection uses resize_long instead of limit_side_len
        let resizer = DetResizeForTest::new(
            None,      // input_shape
            None,      // image_shape
            None,      // keep_ratio
            None,      // limit_side_len
            None,      // limit_type
            Some(736), // resize_long - specific for seal detection
            None,      // max_side_limit
        );

        // Create normalizer with ImageNet parameters. PaddleX converts inputs to RGB
        // before normalization, so we keep the canonical channel ordering here.
        let normalizer = NormalizeImage::new(
            Some(1.0 / 255.0),               // scale
            Some(vec![0.485, 0.456, 0.406]), // mean (RGB order)
            Some(vec![0.229, 0.224, 0.225]), // std (RGB order)
            Some(ChannelOrder::CHW),         // order
        )?;

        // Create postprocessor with seal-specific parameters
        // Note: Using parameters from official PP-OCRv4 seal config
        let postprocessor = DBPostProcess::new(
            Some(0.2),   // thresh
            Some(0.6),   // box_thresh
            Some(1000),  // max_candidates
            Some(0.5),   // unclip_ratio
            Some(false), // use_dilation
            Some(ScoreMode::Fast),
            Some(BoxType::Poly), // Use polygon for curved boundaries
        );

        // Create adapter info
        let info = AdapterInfo::new(
            self.model_name.clone(),
            "1.0.0",
            TaskType::SealTextDetection,
            "PP-OCR seal text detection for curved text in stamps",
        );

        Ok(SealTextDetectionAdapter {
            inference,
            resizer,
            normalizer,
            postprocessor,
            info,
            config: SealTextDetectionConfig::default(),
        })
    }

    fn with_config(self, _config: Self::Config) -> Self {
        // Config is applied at runtime during execute()
        self
    }

    fn adapter_type(&self) -> &str {
        "SealTextDetection"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = SealTextDetectionAdapterBuilder::new();
        assert_eq!(builder.adapter_type(), "SealTextDetection");
    }

    #[test]
    fn test_builder_with_session_pool() {
        let builder = SealTextDetectionAdapterBuilder::new().session_pool_size(4);
        assert_eq!(builder.session_pool_size, 4);
    }

    #[test]
    fn test_builder_with_model_name() {
        let builder = SealTextDetectionAdapterBuilder::new().model_name("PP-OCRv4_server_seal_det");
        assert_eq!(builder.model_name, "PP-OCRv4_server_seal_det");
    }

    #[test]
    fn test_adapter_info() {
        // This test would need a valid model path in real usage
        // For now, we just test the builder pattern
        let builder = SealTextDetectionAdapterBuilder::new()
            .model_name("test_model")
            .session_pool_size(2);
        assert_eq!(builder.adapter_type(), "SealTextDetection");
    }
}
