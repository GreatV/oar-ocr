//! DB (Differentiable Binarization) Model
//!
//! This module provides a pure implementation of the DB text detection model.
//! The model handles preprocessing, inference, and postprocessing independently of tasks.

use crate::core::inference::OrtInfer;
use crate::core::{OCRError, Tensor4D, validate_positive, validate_range};
use crate::processors::{
    BoundingBox, BoxType, ChannelOrder, DBPostProcess, DBPostProcessConfig, DetResizeForTest,
    ImageScaleInfo, LimitType, NormalizeImage, ScoreMode,
};
use image::{DynamicImage, RgbImage};
use std::path::Path;
use tracing::debug;

/// Configuration for DB model preprocessing.
#[derive(Debug, Clone, Default)]
pub struct DBPreprocessConfig {
    /// Limit for the side length of the image
    pub limit_side_len: Option<u32>,
    /// Type of limit to apply
    pub limit_type: Option<LimitType>,
    /// Maximum side limit for the image
    pub max_side_limit: Option<u32>,
    /// Resize long dimension (alternative to limit_side_len)
    pub resize_long: Option<u32>,
}

/// Configuration for DB model postprocessing.
#[derive(Debug, Clone)]
pub struct DBPostprocessConfig {
    /// Pixel-level threshold for text detection
    pub score_threshold: f32,
    /// Box-level threshold for filtering detections
    pub box_threshold: f32,
    /// Expansion ratio for detected regions using Vatti clipping
    pub unclip_ratio: f32,
    /// Maximum number of candidate detections
    pub max_candidates: usize,
    /// Whether to use dilation
    pub use_dilation: bool,
    /// Score calculation mode
    pub score_mode: ScoreMode,
    /// Type of bounding box (Quad or Poly)
    pub box_type: BoxType,
}

impl Default for DBPostprocessConfig {
    fn default() -> Self {
        Self {
            score_threshold: 0.3,
            box_threshold: 0.7,
            unclip_ratio: 1.5,
            max_candidates: 1000,
            use_dilation: false,
            score_mode: ScoreMode::Fast,
            box_type: BoxType::Quad,
        }
    }
}

impl DBPostprocessConfig {
    /// Validates the configuration parameters.
    pub fn validate(&self) -> Result<(), OCRError> {
        // Validate score_threshold is in [0, 1]
        validate_range(self.score_threshold, 0.0, 1.0, "score_threshold")?;

        // Validate box_threshold is in [0, 1]
        validate_range(self.box_threshold, 0.0, 1.0, "box_threshold")?;

        // Validate unclip_ratio is positive
        validate_positive(self.unclip_ratio, "unclip_ratio")?;

        // Validate max_candidates is positive
        validate_positive(self.max_candidates, "max_candidates")?;

        Ok(())
    }
}

/// DB model output containing bounding boxes and confidence scores.
#[derive(Debug, Clone)]
pub struct DBModelOutput {
    /// Detected bounding boxes for each image in the batch
    pub boxes: Vec<Vec<BoundingBox>>,
    /// Confidence scores for each bounding box
    pub scores: Vec<Vec<f32>>,
}

/// Pure DB model implementation.
///
/// This model implements the core DB architecture and can be configured
/// for different detection tasks through preprocessing and postprocessing configs.
#[derive(Debug)]
pub struct DBModel {
    /// ONNX Runtime inference engine
    inference: OrtInfer,
    /// Image resizer for preprocessing
    resizer: DetResizeForTest,
    /// Image normalizer for preprocessing
    normalizer: NormalizeImage,
    /// Postprocessor for converting predictions to bounding boxes
    postprocessor: DBPostProcess,
}

impl DBModel {
    /// Creates a new DB model.
    pub fn new(
        inference: OrtInfer,
        resizer: DetResizeForTest,
        normalizer: NormalizeImage,
        postprocessor: DBPostProcess,
    ) -> Self {
        Self {
            inference,
            resizer,
            normalizer,
            postprocessor,
        }
    }

    /// Preprocesses images for detection.
    pub fn preprocess(
        &self,
        images: Vec<RgbImage>,
    ) -> Result<(Tensor4D, Vec<ImageScaleInfo>), OCRError> {
        // Convert to DynamicImage
        let dynamic_images: Vec<DynamicImage> =
            images.into_iter().map(DynamicImage::ImageRgb8).collect();

        // Apply detection resizing
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
                shape.src_h,
                shape.src_w,
                shape.ratio_h,
                shape.ratio_w
            );
        }

        // Apply ImageNet normalization and convert to tensor.
        //
        // Note: External models often decode images as BGR and then normalize with
        // mean/std as provided in their configs. In this repo, input images are
        // loaded as RGB; we keep them in RGB here and rely on `NormalizeImage`
        // with `ColorOrder::BGR` to map channels (RGB -> BGR) without a manual swap.
        let batch_tensor = self.normalizer.normalize_batch_to(resized_images)?;
        debug!("Batch tensor shape: {:?}", batch_tensor.shape());

        Ok((batch_tensor, img_shapes))
    }

    /// Runs inference on the preprocessed batch.
    pub fn infer(&self, batch_tensor: &Tensor4D) -> Result<Tensor4D, OCRError> {
        self.inference
            .infer_4d(batch_tensor)
            .map_err(|e| OCRError::Inference {
                model_name: "DB".to_string(),
                context: format!(
                    "failed to run inference on batch with shape {:?}",
                    batch_tensor.shape()
                ),
                source: Box::new(e),
            })
    }

    /// Postprocesses model predictions to bounding boxes.
    pub fn postprocess(
        &self,
        predictions: &Tensor4D,
        img_shapes: Vec<ImageScaleInfo>,
        score_threshold: f32,
        box_threshold: f32,
        unclip_ratio: f32,
    ) -> DBModelOutput {
        let config = DBPostProcessConfig::new(score_threshold, box_threshold, unclip_ratio);
        let (boxes, scores) = self
            .postprocessor
            .apply(predictions, img_shapes, Some(&config));
        DBModelOutput { boxes, scores }
    }

    /// Runs the complete forward pass: preprocess -> infer -> postprocess.
    pub fn forward(
        &self,
        images: Vec<RgbImage>,
        score_threshold: f32,
        box_threshold: f32,
        unclip_ratio: f32,
    ) -> Result<DBModelOutput, OCRError> {
        let (batch_tensor, img_shapes) = self.preprocess(images)?;
        let predictions = self.infer(&batch_tensor)?;
        Ok(self.postprocess(
            &predictions,
            img_shapes,
            score_threshold,
            box_threshold,
            unclip_ratio,
        ))
    }
}

/// Builder for DB model.
pub struct DBModelBuilder {
    /// Preprocessing configuration
    preprocess_config: DBPreprocessConfig,
    /// Postprocessing configuration
    postprocess_config: DBPostprocessConfig,
    /// ONNX Runtime session configuration
    ort_config: Option<crate::core::config::OrtSessionConfig>,
}

impl DBModelBuilder {
    /// Creates a new DB model builder with default settings.
    pub fn new() -> Self {
        Self {
            preprocess_config: DBPreprocessConfig::default(),
            postprocess_config: DBPostprocessConfig::default(),
            ort_config: None,
        }
    }

    /// Sets the preprocessing configuration.
    pub fn preprocess_config(mut self, config: DBPreprocessConfig) -> Self {
        self.preprocess_config = config;
        self
    }

    /// Sets the postprocessing configuration.
    pub fn postprocess_config(mut self, config: DBPostprocessConfig) -> Self {
        self.postprocess_config = config;
        self
    }

    /// Sets the ONNX Runtime session configuration.
    pub fn with_ort_config(mut self, config: crate::core::config::OrtSessionConfig) -> Self {
        self.ort_config = Some(config);
        self
    }

    /// Builds the DB model.
    pub fn build(self, model_path: &Path) -> Result<DBModel, OCRError> {
        // Create ONNX inference engine
        let inference = if self.ort_config.is_some() {
            use crate::core::config::ModelInferenceConfig;
            let common_config = ModelInferenceConfig {
                ort_session: self.ort_config,
                ..Default::default()
            };
            OrtInfer::from_config(&common_config, model_path, Some("x"))?
        } else {
            OrtInfer::new(model_path, Some("x"))?
        };

        // Create resizer
        let resizer = DetResizeForTest::new(
            None,                                  // input_shape
            None,                                  // image_shape
            None,                                  // keep_ratio
            self.preprocess_config.limit_side_len, // limit_side_len
            self.preprocess_config.limit_type,     // limit_type
            self.preprocess_config.resize_long,    // resize_long
            self.preprocess_config.max_side_limit, // max_side_limit
        );

        // Create normalizer.
        // External models read images in BGR. Their configs use ImageNet stats
        // in that *same* channel order (B, G, R). Our images are loaded as RGB,
        // so we keep them in RGB and use `ColorOrder::BGR` to map channels
        // into BGR order during normalization.
        let normalizer = NormalizeImage::with_color_order(
            Some(1.0 / 255.0),               // scale
            Some(vec![0.485, 0.456, 0.406]), // mean
            Some(vec![0.229, 0.224, 0.225]), // std
            Some(ChannelOrder::CHW),         // order
            Some(crate::processors::ColorOrder::BGR),
        )?;

        // Create postprocessor
        let postprocessor = DBPostProcess::new(
            Some(self.postprocess_config.score_threshold),
            Some(self.postprocess_config.box_threshold),
            Some(self.postprocess_config.max_candidates),
            Some(self.postprocess_config.unclip_ratio),
            Some(self.postprocess_config.use_dilation),
            Some(self.postprocess_config.score_mode),
            Some(self.postprocess_config.box_type),
        );

        Ok(DBModel::new(inference, resizer, normalizer, postprocessor))
    }
}

impl Default for DBModelBuilder {
    fn default() -> Self {
        Self::new()
    }
}
