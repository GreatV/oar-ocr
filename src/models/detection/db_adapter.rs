//! DB (Differentiable Binarization) Text Detection Adapter
//!
//! This module provides an adapter for DB text detection that directly implements
//! preprocessing, ONNX inference, and postprocessing without using predictor wrappers.

use crate::core::inference::OrtInfer;
use crate::core::traits::{
    adapter::{AdapterBuilder, AdapterInfo, ModelAdapter},
    task::Task,
};
use crate::core::{OCRError, Tensor4D};
use crate::domain::tasks::{TextDetectionConfig, TextDetectionOutput, TextDetectionTask};
use crate::processors::{ChannelOrder, DBPostProcess, DetResizeForTest, LimitType, NormalizeImage};
use image::{DynamicImage, RgbImage};
use std::path::Path;

/// Adapter for DB text detection model.
///
/// This adapter directly implements preprocessing, ONNX inference, and postprocessing
/// for DB text detection without using predictor wrappers.
#[derive(Debug)]
pub struct DBTextDetectionAdapter {
    /// ONNX Runtime inference engine
    inference: OrtInfer,
    /// Image resizer for preprocessing
    resizer: DetResizeForTest,
    /// Image normalizer for preprocessing
    normalizer: NormalizeImage,
    /// Postprocessor for converting predictions to bounding boxes
    postprocessor: DBPostProcess,
    /// Adapter information
    info: AdapterInfo,
    /// Configuration for detection
    config: TextDetectionConfig,
}

impl DBTextDetectionAdapter {
    /// Creates a new DB text detection adapter.
    pub fn new(
        inference: OrtInfer,
        resizer: DetResizeForTest,
        normalizer: NormalizeImage,
        postprocessor: DBPostProcess,
        config: TextDetectionConfig,
    ) -> Self {
        let info = AdapterInfo::new(
            "DB",
            "1.0.0",
            crate::core::traits::task::TaskType::TextDetection,
            "Differentiable Binarization text detector",
        );
        Self {
            inference,
            resizer,
            normalizer,
            postprocessor,
            info,
            config,
        }
    }

    /// Preprocesses images for detection.
    fn preprocess(&self, images: Vec<RgbImage>) -> Result<(Tensor4D, Vec<[f32; 4]>), OCRError> {
        // Convert to DynamicImage
        let dynamic_images: Vec<DynamicImage> =
            images.into_iter().map(DynamicImage::ImageRgb8).collect();

        // Resize images
        let (resized_images, img_shapes) = self.resizer.apply(
            dynamic_images,
            None, // Use default limit_side_len
            None, // Use default limit_type
            None, // Use default max_side_limit
        );

        // Normalize and convert to tensor
        let batch_tensor = self.normalizer.normalize_batch_to(resized_images)?;

        Ok((batch_tensor, img_shapes))
    }

    /// Postprocesses model predictions to bounding boxes.
    fn postprocess(
        &self,
        predictions: &Tensor4D,
        img_shapes: Vec<[f32; 4]>,
        config: &TextDetectionConfig,
    ) -> TextDetectionOutput {
        let (boxes, scores) = self.postprocessor.apply(
            predictions,
            img_shapes,
            Some(config.score_threshold),
            Some(config.box_threshold),
            Some(config.unclip_ratio),
        );

        TextDetectionOutput { boxes, scores }
    }
}

impl ModelAdapter for DBTextDetectionAdapter {
    type Task = TextDetectionTask;

    fn info(&self) -> AdapterInfo {
        self.info.clone()
    }

    fn execute(
        &self,
        input: <Self::Task as Task>::Input,
        config: Option<&<Self::Task as Task>::Config>,
    ) -> Result<<Self::Task as Task>::Output, OCRError> {
        // Use provided config or fall back to stored config
        let effective_config = config.unwrap_or(&self.config);

        // Preprocess images
        let (batch_tensor, img_shapes) = self.preprocess(input.images)?;

        // Run inference
        let predictions = self.inference.infer_4d(&batch_tensor)?;

        // Postprocess predictions
        let output = self.postprocess(&predictions, img_shapes, effective_config);

        Ok(output)
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn recommended_batch_size(&self) -> usize {
        6 // Default batch size
    }
}

/// Builder for DB text detection adapter.
pub struct DBTextDetectionAdapterBuilder {
    /// Task configuration
    task_config: TextDetectionConfig,
    /// Resize configuration
    limit_side_len: Option<u32>,
    limit_type: Option<LimitType>,
    max_side_limit: Option<u32>,
    /// Session pool size for ONNX Runtime
    session_pool_size: usize,
}

impl DBTextDetectionAdapterBuilder {
    /// Creates a new DB text detection adapter builder.
    pub fn new() -> Self {
        Self {
            task_config: TextDetectionConfig::default(),
            limit_side_len: None,
            limit_type: None,
            max_side_limit: None,
            session_pool_size: 1,
        }
    }

    /// Sets the limit for the side length of the image.
    pub fn limit_side_len(mut self, limit_side_len: u32) -> Self {
        self.limit_side_len = Some(limit_side_len);
        self
    }

    /// Sets the type of limit to apply.
    pub fn limit_type(mut self, limit_type: LimitType) -> Self {
        self.limit_type = Some(limit_type);
        self
    }

    /// Sets the maximum side limit for the image.
    pub fn max_side_limit(mut self, max_side_limit: u32) -> Self {
        self.max_side_limit = Some(max_side_limit);
        self
    }

    /// Sets the session pool size.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.session_pool_size = size;
        self
    }
}

impl Default for DBTextDetectionAdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AdapterBuilder for DBTextDetectionAdapterBuilder {
    type Config = TextDetectionConfig;
    type Adapter = DBTextDetectionAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        // Create ONNX inference engine
        let inference = if self.session_pool_size > 1 {
            // Use session pool for concurrent inference
            use crate::core::config::CommonBuilderConfig;
            let common_config = CommonBuilderConfig {
                session_pool_size: Some(self.session_pool_size),
                ..Default::default()
            };
            OrtInfer::from_common(&common_config, model_path, Some("x"))?
        } else {
            OrtInfer::new(model_path, Some("x"))?
        };

        // Create resizer
        let resizer = DetResizeForTest::new(
            None,                // input_shape
            None,                // image_shape
            None,                // keep_ratio
            self.limit_side_len, // limit_side_len
            self.limit_type,     // limit_type
            None,                // resize_long
            self.max_side_limit, // max_side_limit
        );

        // Create normalizer (standard ImageNet normalization)
        let normalizer = NormalizeImage::new(
            Some(1.0 / 255.0),               // scale
            Some(vec![0.485, 0.456, 0.406]), // mean
            Some(vec![0.229, 0.224, 0.225]), // std
            Some(ChannelOrder::CHW),         // order
        )?;

        // Create postprocessor
        let postprocessor = DBPostProcess::new(
            Some(self.task_config.score_threshold),
            Some(self.task_config.box_threshold),
            None, // max_candidates
            Some(self.task_config.unclip_ratio),
            None, // use_dilation
            None, // score_mode
            None, // box_type
        );

        Ok(DBTextDetectionAdapter::new(
            inference,
            resizer,
            normalizer,
            postprocessor,
            self.task_config,
        ))
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.task_config = config;
        self
    }

    fn adapter_type(&self) -> &str {
        "DB-TextDetection"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = DBTextDetectionAdapterBuilder::new();
        assert_eq!(builder.adapter_type(), "DB-TextDetection");
    }

    #[test]
    fn test_builder_with_config() {
        let config = TextDetectionConfig {
            score_threshold: 0.5,
            box_threshold: 0.7,
            unclip_ratio: 2.0,
            max_candidates: 1000,
        };

        let builder = DBTextDetectionAdapterBuilder::new().with_config(config.clone());

        assert_eq!(builder.adapter_type(), "DB-TextDetection");
    }

    #[test]
    fn test_builder_with_session_pool() {
        let builder = DBTextDetectionAdapterBuilder::new().session_pool_size(4);

        assert_eq!(builder.adapter_type(), "DB-TextDetection");
    }

    #[test]
    fn test_adapter_info() {
        // Note: This test would require a valid model file to actually build the adapter
        // For now, we just test the builder configuration
        let builder = DBTextDetectionAdapterBuilder::new();
        assert_eq!(builder.adapter_type(), "DB-TextDetection");
    }
}
