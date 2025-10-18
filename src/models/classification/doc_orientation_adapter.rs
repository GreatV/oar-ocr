//! Document Orientation Classifier Adapter
//!
//! This module provides an adapter for document orientation classification that directly
//! implements preprocessing, ONNX inference, and postprocessing without using predictor wrappers.

use crate::core::inference::OrtInfer;
use crate::core::traits::{
    adapter::{AdapterBuilder, AdapterInfo, ModelAdapter},
    task::Task,
};
use crate::core::{OCRError, Tensor2D, Tensor4D};
use crate::domain::tasks::{
    DocumentOrientationConfig, DocumentOrientationOutput, DocumentOrientationTask,
};
use crate::processors::{ChannelOrder, NormalizeImage};
use crate::utils::topk::Topk;
use image::{DynamicImage, RgbImage};
use std::path::Path;

/// Adapter for document orientation classification model.
#[derive(Debug)]
pub struct DocOrientationAdapter {
    /// ONNX Runtime inference engine
    inference: OrtInfer,
    /// Image normalizer for preprocessing
    normalizer: NormalizeImage,
    /// Top-k processor for postprocessing
    topk_processor: Topk,
    /// Input shape (height, width)
    input_shape: (u32, u32),
    /// Adapter information
    info: AdapterInfo,
    /// Configuration
    config: DocumentOrientationConfig,
}

impl DocOrientationAdapter {
    /// Creates a new document orientation adapter.
    pub fn new(
        inference: OrtInfer,
        normalizer: NormalizeImage,
        topk_processor: Topk,
        input_shape: (u32, u32),
        config: DocumentOrientationConfig,
    ) -> Self {
        let info = AdapterInfo::new(
            "DocOrientationClassifier",
            "1.0.0",
            crate::core::traits::task::TaskType::DocumentOrientation,
            "Document orientation classifier (0°, 90°, 180°, 270°)",
        );
        Self {
            inference,
            normalizer,
            topk_processor,
            input_shape,
            info,
            config,
        }
    }

    /// Preprocesses images for classification.
    fn preprocess(&self, images: Vec<RgbImage>) -> Result<Tensor4D, OCRError> {
        // Resize images to input shape
        let resized_images: Vec<DynamicImage> = images
            .into_iter()
            .map(|img| {
                DynamicImage::ImageRgb8(image::imageops::resize(
                    &img,
                    self.input_shape.1,
                    self.input_shape.0,
                    image::imageops::FilterType::Lanczos3,
                ))
            })
            .collect();

        // Normalize and convert to tensor
        let batch_tensor = self.normalizer.normalize_batch_to(resized_images)?;

        // Return 4D tensor - OrtInfer will handle the conversion
        Ok(batch_tensor)
    }

    /// Postprocesses model predictions to orientation labels.
    fn postprocess(
        &self,
        predictions: &Tensor2D,
        config: &DocumentOrientationConfig,
    ) -> DocumentOrientationOutput {
        // Convert tensor to Vec<Vec<f32>> for topk processing
        let predictions_vec: Vec<Vec<f32>> =
            predictions.outer_iter().map(|row| row.to_vec()).collect();

        // Apply top-k processing
        let topk_result = self
            .topk_processor
            .process(&predictions_vec, config.topk)
            .unwrap_or_else(|_| {
                // Fallback to empty result on error
                crate::utils::topk::TopkResult {
                    indexes: vec![],
                    scores: vec![],
                    class_names: None,
                }
            });

        // Convert to output format
        let class_ids = topk_result.indexes;
        let scores = topk_result.scores;
        let label_names = topk_result.class_names.unwrap_or_else(|| {
            // Generate default labels if not provided
            class_ids
                .iter()
                .map(|ids| ids.iter().map(|&id| format!("{}", id * 90)).collect())
                .collect()
        });

        DocumentOrientationOutput {
            class_ids,
            scores,
            label_names,
        }
    }
}

impl ModelAdapter for DocOrientationAdapter {
    type Task = DocumentOrientationTask;

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
        let batch_tensor = self.preprocess(input.images)?;

        // Run inference
        let predictions = self.inference.infer_2d(&batch_tensor)?;

        // Postprocess predictions
        let output = self.postprocess(&predictions, effective_config);

        Ok(output)
    }

    fn supports_batching(&self) -> bool {
        true
    }

    fn recommended_batch_size(&self) -> usize {
        1 // Document orientation typically processes one image at a time
    }
}

/// Builder for document orientation adapter.
pub struct DocOrientationAdapterBuilder {
    /// Task configuration
    task_config: DocumentOrientationConfig,
    /// Input shape (height, width)
    input_shape: (u32, u32),
    /// Session pool size for ONNX Runtime
    session_pool_size: usize,
}

impl DocOrientationAdapterBuilder {
    /// Creates a new document orientation adapter builder.
    pub fn new() -> Self {
        Self {
            task_config: DocumentOrientationConfig::default(),
            input_shape: (224, 224), // Default input shape for orientation classification
            session_pool_size: 1,
        }
    }

    /// Sets the input shape.
    pub fn input_shape(mut self, input_shape: (u32, u32)) -> Self {
        self.input_shape = input_shape;
        self
    }

    /// Sets the number of top predictions to return.
    pub fn topk(mut self, topk: usize) -> Self {
        self.task_config.topk = topk;
        self
    }

    /// Sets the session pool size.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.session_pool_size = size;
        self
    }
}

impl Default for DocOrientationAdapterBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl AdapterBuilder for DocOrientationAdapterBuilder {
    type Config = DocumentOrientationConfig;
    type Adapter = DocOrientationAdapter;

    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError> {
        // Create ONNX inference engine
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

        // Create normalizer (standard ImageNet normalization)
        let normalizer = NormalizeImage::new(
            Some(1.0 / 255.0),
            Some(vec![0.485, 0.456, 0.406]),
            Some(vec![0.229, 0.224, 0.225]),
            Some(ChannelOrder::CHW),
        )?;

        // Create top-k processor with orientation labels
        let orientation_labels = vec![
            "0".to_string(),
            "90".to_string(),
            "180".to_string(),
            "270".to_string(),
        ];
        let topk_processor = Topk::from_class_names(orientation_labels);

        Ok(DocOrientationAdapter::new(
            inference,
            normalizer,
            topk_processor,
            self.input_shape,
            self.task_config,
        ))
    }

    fn with_config(mut self, config: Self::Config) -> Self {
        self.task_config = config;
        self
    }

    fn adapter_type(&self) -> &str {
        "DocOrientation-Classification"
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_builder_creation() {
        let builder = DocOrientationAdapterBuilder::new();
        assert_eq!(builder.adapter_type(), "DocOrientation-Classification");
    }

    #[test]
    fn test_builder_with_config() {
        let config = DocumentOrientationConfig {
            score_threshold: 0.8,
            topk: 2,
        };
        let builder = DocOrientationAdapterBuilder::new().with_config(config);
        assert_eq!(builder.task_config.score_threshold, 0.8);
        assert_eq!(builder.task_config.topk, 2);
    }

    #[test]
    fn test_builder_with_input_shape() {
        let builder = DocOrientationAdapterBuilder::new().input_shape((48, 192));

        assert_eq!(builder.adapter_type(), "DocOrientation-Classification");
    }

    #[test]
    fn test_builder_with_session_pool() {
        let builder = DocOrientationAdapterBuilder::new().session_pool_size(4);

        assert_eq!(builder.adapter_type(), "DocOrientation-Classification");
    }
}
