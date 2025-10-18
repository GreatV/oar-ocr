//! DocTR rectifier adapter implementation.
//!
//! This module provides an adapter for document rectification that directly
//! implements preprocessing, ONNX inference, and postprocessing without using predictor wrappers.

use crate::core::inference::OrtInfer;
use crate::core::traits::adapter::{AdapterBuilder, AdapterInfo, ModelAdapter};
use crate::core::{OCRError, Tensor4D};
use crate::domain::tasks::document_rectification::{
    DocumentRectificationConfig, DocumentRectificationOutput, DocumentRectificationTask,
};
use crate::processors::{ChannelOrder, DocTrPostProcess, NormalizeImage};
use image::{DynamicImage, RgbImage, imageops::FilterType};
use std::path::Path;

/// Adapter for DocTR document rectifier.
///
/// This adapter directly implements document rectification using ONNX inference.
#[derive(Debug)]
pub struct DoctrRectifierAdapter {
    /// ONNX Runtime inference engine
    inference: OrtInfer,
    /// Image normalizer for preprocessing
    normalizer: NormalizeImage,
    /// DocTR postprocessor for converting tensor to images
    postprocessor: DocTrPostProcess,
    /// Input shape [channels, height, width]
    rec_image_shape: [usize; 3],
    /// Adapter information
    info: AdapterInfo,
    /// Configuration
    #[allow(dead_code)]
    config: DocumentRectificationConfig,
}

impl DoctrRectifierAdapter {
    /// Creates a new DocTR rectifier adapter.
    pub fn new(
        inference: OrtInfer,
        normalizer: NormalizeImage,
        postprocessor: DocTrPostProcess,
        rec_image_shape: [usize; 3],
        config: DocumentRectificationConfig,
    ) -> Self {
        let info = AdapterInfo::new(
            "doctr_rectifier",
            "1.0.0",
            crate::core::traits::task::TaskType::DocumentRectification,
            "DocTR document rectifier for correcting image distortions",
        );
        Self {
            inference,
            normalizer,
            postprocessor,
            rec_image_shape,
            info,
            config,
        }
    }

    /// Preprocesses images for rectification.
    fn preprocess(&self, images: Vec<RgbImage>) -> Result<(Tensor4D, Vec<(u32, u32)>), OCRError> {
        let mut original_sizes = Vec::with_capacity(images.len());
        let mut processed_images = Vec::with_capacity(images.len());

        let target_height = self.rec_image_shape[1] as u32;
        let target_width = self.rec_image_shape[2] as u32;
        let should_resize = target_height > 0 && target_width > 0;

        for img in images {
            let original_size = (img.width(), img.height());
            original_sizes.push(original_size);

            if should_resize && (img.width() != target_width || img.height() != target_height) {
                let resized = DynamicImage::ImageRgb8(img).resize_exact(
                    target_width,
                    target_height,
                    FilterType::Lanczos3,
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

    /// Postprocesses model predictions to rectified images.
    fn postprocess(
        &self,
        predictions: &Tensor4D,
        original_sizes: &[(u32, u32)],
    ) -> Result<Vec<RgbImage>, OCRError> {
        // Use DocTrPostProcess to convert tensor to images
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

        for (img, &(orig_w, orig_h)) in images.iter_mut().zip(original_sizes) {
            if orig_w == 0 || orig_h == 0 {
                continue;
            }

            if img.width() != orig_w || img.height() != orig_h {
                let resized = DynamicImage::ImageRgb8(std::mem::take(img)).resize_exact(
                    orig_w,
                    orig_h,
                    FilterType::Lanczos3,
                );
                *img = resized.into_rgb8();
            }
        }

        Ok(images)
    }
}

impl ModelAdapter for DoctrRectifierAdapter {
    type Task = DocumentRectificationTask;

    fn info(&self) -> AdapterInfo {
        self.info.clone()
    }

    fn execute(
        &self,
        input: <Self::Task as crate::core::traits::task::Task>::Input,
        _config: Option<&<Self::Task as crate::core::traits::task::Task>::Config>,
    ) -> Result<<Self::Task as crate::core::traits::task::Task>::Output, OCRError> {
        // Preprocess images
        let (batch_tensor, original_sizes) = self.preprocess(input.images)?;

        // Run inference
        let predictions = self.inference.infer_4d(&batch_tensor)?;

        // Postprocess predictions
        let rectified_images = self.postprocess(&predictions, &original_sizes)?;

        Ok(DocumentRectificationOutput { rectified_images })
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn recommended_batch_size(&self) -> usize {
        // Document rectification is computationally intensive
        // Use smaller batch size for better memory management
        8
    }
}

/// Builder for DocTR rectifier adapter.
///
/// This builder provides a fluent API for configuring and creating
/// a DocTR rectifier adapter instance.
pub struct DoctrRectifierAdapterBuilder {
    /// Task configuration
    task_config: DocumentRectificationConfig,
    /// Session pool size for ONNX Runtime
    session_pool_size: usize,
}

impl DoctrRectifierAdapterBuilder {
    /// Creates a new builder with default configuration.
    pub fn new() -> Self {
        Self {
            task_config: DocumentRectificationConfig::default(),
            session_pool_size: 1,
        }
    }

    /// Sets the input shape for the rectification model.
    ///
    /// # Arguments
    ///
    /// * `shape` - Input shape as [channels, height, width]
    pub fn input_shape(mut self, shape: [usize; 3]) -> Self {
        self.task_config.rec_image_shape = shape;
        self
    }

    /// Sets the session pool size.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.session_pool_size = size;
        self
    }
}

impl Default for DoctrRectifierAdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AdapterBuilder for DoctrRectifierAdapterBuilder {
    type Config = DocumentRectificationConfig;
    type Adapter = DoctrRectifierAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        // Create ONNX inference engine
        let inference = if self.session_pool_size > 1 {
            use crate::core::config::CommonBuilderConfig;
            let common_config = CommonBuilderConfig {
                session_pool_size: Some(self.session_pool_size),
                ..Default::default()
            };
            OrtInfer::from_common_with_auto_input(&common_config, model_path)?
        } else {
            OrtInfer::with_auto_input_name(model_path)?
        };

        // Create normalizer (scale to [0, 1] without mean shift)
        let normalizer = NormalizeImage::new(
            Some(1.0 / 255.0),
            Some(vec![0.0, 0.0, 0.0]),
            Some(vec![1.0, 1.0, 1.0]),
            Some(ChannelOrder::CHW),
        )?;

        // Create DocTR postprocessor
        let postprocessor = DocTrPostProcess::new(255.0);

        Ok(DoctrRectifierAdapter::new(
            inference,
            normalizer,
            postprocessor,
            self.task_config.rec_image_shape,
            self.task_config,
        ))
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.task_config = config;
        self
    }

    fn adapter_type(&self) -> &str {
        "doctr_rectifier"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = DoctrRectifierAdapterBuilder::new();
        assert_eq!(builder.adapter_type(), "doctr_rectifier");
    }

    #[test]
    fn test_builder_with_config() {
        let config = DocumentRectificationConfig {
            rec_image_shape: [3, 1024, 1024],
        };

        let builder = DoctrRectifierAdapterBuilder::new().with_config(config.clone());
        assert_eq!(builder.task_config.rec_image_shape, [3, 1024, 1024]);
    }

    #[test]
    fn test_builder_fluent_api() {
        let builder = DoctrRectifierAdapterBuilder::new().input_shape([3, 768, 768]);

        assert_eq!(builder.task_config.rec_image_shape, [3, 768, 768]);
    }

    #[test]
    fn test_default_builder() {
        let builder = DoctrRectifierAdapterBuilder::default();
        assert_eq!(builder.adapter_type(), "doctr_rectifier");
        assert_eq!(builder.task_config.rec_image_shape, [3, 0, 0]);
    }

    #[test]
    fn test_builder_with_session_pool() {
        let builder = DoctrRectifierAdapterBuilder::new().session_pool_size(4);

        assert_eq!(builder.adapter_type(), "doctr_rectifier");
    }
}
