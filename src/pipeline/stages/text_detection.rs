//! Text detection stage processor for the extensible pipeline.
//!
//! This module provides a wrapper around the text detection predictor
//! to integrate it into the extensible pipeline system.

use image::RgbImage;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, warn};

use super::extensible::{PipelineStage, StageContext, StageData, StageDependency, StageId};
use super::types::{StageMetrics, StageResult};
use crate::core::config::ConfigValidator;
use crate::core::{OCRError, traits::StandardPredictor};
use crate::predictor::TextDetPredictor;
use crate::processors::BoundingBox;

/// Result of text detection processing.
#[derive(Debug, Clone)]
pub struct TextDetectionResult {
    /// Detected text bounding boxes
    pub text_boxes: Vec<BoundingBox>,
}

/// Configuration for text detection processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextDetectionConfig {
    /// Minimum confidence threshold for text detection
    pub confidence_threshold: Option<f32>,
    /// Whether to apply non-maximum suppression
    pub apply_nms: bool,
    /// NMS threshold if applied
    pub nms_threshold: f32,
}

impl Default for TextDetectionConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: None,
            apply_nms: true,
            nms_threshold: 0.3,
        }
    }
}

impl ConfigValidator for TextDetectionConfig {
    fn validate(&self) -> Result<(), crate::core::config::ConfigError> {
        if let Some(threshold) = self.confidence_threshold
            && !(0.0..=1.0).contains(&threshold)
        {
            return Err(crate::core::config::ConfigError::InvalidConfig {
                message: "confidence_threshold must be between 0.0 and 1.0".to_string(),
            });
        }

        if !(0.0..=1.0).contains(&self.nms_threshold) {
            return Err(crate::core::config::ConfigError::InvalidConfig {
                message: "nms_threshold must be between 0.0 and 1.0".to_string(),
            });
        }

        Ok(())
    }

    fn get_defaults() -> Self {
        Self::default()
    }
}

/// Text detection stage processor.
pub struct TextDetectionStageProcessor;

impl TextDetectionStageProcessor {
    /// Process text detection for a single image.
    pub fn process_single(
        image: &RgbImage,
        detector: Option<&TextDetPredictor>,
        config: Option<&TextDetectionConfig>,
    ) -> Result<StageResult<TextDetectionResult>, OCRError> {
        let start_time = Instant::now();
        let _config = config.cloned().unwrap_or_default();

        let text_boxes = if let Some(detector) = detector {
            debug!("Running text detection");
            let result = detector.predict(vec![image.clone()], None)?;
            result.dt_polys.into_iter().flatten().collect()
        } else {
            warn!("No text detector available");
            Vec::new()
        };

        let detection_count = text_boxes.len();
        let processing_time = start_time.elapsed();

        let metrics = StageMetrics::new(1, 0)
            .with_processing_time(processing_time)
            .with_info("stage", "text_detection")
            .with_info("detections", detection_count.to_string());

        let result = TextDetectionResult { text_boxes };

        debug!(
            "Text detection completed: {} regions detected",
            detection_count
        );

        Ok(StageResult::new(result, metrics))
    }
}

/// Extensible text detection stage that implements PipelineStage trait.
#[derive(Debug)]
pub struct ExtensibleTextDetectionStage {
    detector: Option<Arc<TextDetPredictor>>,
}

impl ExtensibleTextDetectionStage {
    /// Create a new extensible text detection stage.
    pub fn new(detector: Option<Arc<TextDetPredictor>>) -> Self {
        Self { detector }
    }
}

impl PipelineStage for ExtensibleTextDetectionStage {
    type Config = TextDetectionConfig;
    type Result = TextDetectionResult;

    fn stage_id(&self) -> StageId {
        StageId::new("text_detection")
    }

    fn stage_name(&self) -> &str {
        "Text Detection"
    }

    fn dependencies(&self) -> Vec<StageDependency> {
        // Text detection should run after orientation
        vec![StageDependency::After(StageId::new("orientation"))]
    }

    fn is_enabled(&self, _context: &StageContext, _config: Option<&Self::Config>) -> bool {
        self.detector.is_some()
    }

    fn process(
        &self,
        context: &mut StageContext,
        data: StageData,
        config: Option<&Self::Config>,
    ) -> Result<StageResult<Self::Result>, OCRError> {
        let stage_result = TextDetectionStageProcessor::process_single(
            &data.image,
            self.detector.as_ref().map(|d| d.as_ref()),
            config,
        )?;

        // Store text boxes in context for other stages to use
        context.set_stage_result(
            StageId::new("text_boxes"),
            stage_result.data.text_boxes.clone(),
        );

        Ok(stage_result)
    }

    fn validate_config(&self, config: &Self::Config) -> Result<(), OCRError> {
        config.validate().map_err(|e| OCRError::ConfigError {
            message: format!("TextDetectionConfig validation failed: {}", e),
        })
    }

    fn default_config(&self) -> Self::Config {
        TextDetectionConfig::get_defaults()
    }
}

/// Text line orientation stage for the extensible pipeline.
#[derive(Debug)]
pub struct ExtensibleTextLineOrientationStage {
    classifier: Option<Arc<crate::predictor::TextLineClasPredictor>>,
}

impl ExtensibleTextLineOrientationStage {
    /// Create a new extensible text line orientation stage.
    pub fn new(classifier: Option<Arc<crate::predictor::TextLineClasPredictor>>) -> Self {
        Self { classifier }
    }
}

/// Configuration for text line orientation processing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextLineOrientationConfig {
    /// Confidence threshold for accepting orientation predictions
    pub confidence_threshold: Option<f32>,
}

impl Default for TextLineOrientationConfig {
    fn default() -> Self {
        Self {
            confidence_threshold: Some(0.5),
        }
    }
}

impl ConfigValidator for TextLineOrientationConfig {
    fn validate(&self) -> Result<(), crate::core::config::ConfigError> {
        if let Some(threshold) = self.confidence_threshold
            && !(0.0..=1.0).contains(&threshold)
        {
            return Err(crate::core::config::ConfigError::InvalidConfig {
                message: "confidence_threshold must be between 0.0 and 1.0".to_string(),
            });
        }
        Ok(())
    }

    fn get_defaults() -> Self {
        Self::default()
    }
}

impl PipelineStage for ExtensibleTextLineOrientationStage {
    type Config = TextLineOrientationConfig;
    type Result = Vec<Option<f32>>;

    fn stage_id(&self) -> StageId {
        StageId::new("text_line_orientation")
    }

    fn stage_name(&self) -> &str {
        "Text Line Orientation Classification"
    }

    fn dependencies(&self) -> Vec<StageDependency> {
        // Text line orientation depends on cropping results
        vec![StageDependency::Requires(StageId::new("cropping"))]
    }

    fn is_enabled(&self, _context: &StageContext, _config: Option<&Self::Config>) -> bool {
        self.classifier.is_some()
    }

    fn process(
        &self,
        context: &mut StageContext,
        _data: StageData,
        config: Option<&Self::Config>,
    ) -> Result<StageResult<Self::Result>, OCRError> {
        let start_time = Instant::now();

        // Get cropped images from the cropping stage
        let cropping_result = context
            .get_stage_result::<super::cropping::CroppingResult>(&StageId::new("cropping"))
            .ok_or_else(|| {
                OCRError::processing_error(
                    crate::core::ProcessingStage::Generic,
                    "Cropping results not found in context",
                    crate::core::errors::SimpleError::new("Missing cropping results"),
                )
            })?;

        let mut text_line_orientations = Vec::new();
        let mut failed_orientations = 0;

        if let Some(ref classifier) = self.classifier {
            let valid_images: Vec<RgbImage> = cropping_result
                .cropped_images
                .iter()
                .filter_map(|o| o.as_ref().cloned())
                .collect();

            if !valid_images.is_empty() {
                let valid_images_count = valid_images.len();
                match classifier.predict(valid_images, None) {
                    Ok(result) => {
                        let mut result_idx = 0;
                        for cropped_img_opt in &cropping_result.cropped_images {
                            if cropped_img_opt.is_some() {
                                if let (Some(labels), Some(score_list)) = (
                                    result.label_names.get(result_idx),
                                    result.scores.get(result_idx),
                                ) {
                                    if let (Some(label), Some(&score)) =
                                        (labels.first(), score_list.first())
                                    {
                                        let confidence_threshold =
                                            config.and_then(|c| c.confidence_threshold);

                                        let orientation_result =
                                            crate::core::parse_text_line_orientation(
                                                label.as_ref(),
                                                score,
                                                confidence_threshold,
                                            );

                                        if orientation_result.is_confident {
                                            text_line_orientations
                                                .push(Some(orientation_result.angle));
                                        } else {
                                            text_line_orientations.push(None);
                                        }
                                    } else {
                                        text_line_orientations.push(None);
                                    }
                                } else {
                                    text_line_orientations.push(None);
                                }
                                result_idx += 1;
                            } else {
                                text_line_orientations.push(None);
                            }
                        }
                    }
                    Err(e) => {
                        failed_orientations = valid_images_count;
                        warn!("Text line orientation classification failed: {}", e);
                        text_line_orientations.resize(cropping_result.cropped_images.len(), None);
                    }
                }
            } else {
                text_line_orientations.resize(cropping_result.cropped_images.len(), None);
            }
        } else {
            text_line_orientations.resize(cropping_result.cropped_images.len(), None);
        }

        let processing_time = start_time.elapsed();
        let success_count = text_line_orientations.len() - failed_orientations;

        let metrics = StageMetrics::new(success_count, failed_orientations)
            .with_processing_time(processing_time)
            .with_info("stage", "text_line_orientation")
            .with_info("total_regions", text_line_orientations.len().to_string());

        Ok(StageResult::new(text_line_orientations, metrics))
    }

    fn validate_config(&self, config: &Self::Config) -> Result<(), OCRError> {
        config.validate().map_err(|e| OCRError::ConfigError {
            message: format!("TextLineOrientationConfig validation failed: {}", e),
        })
    }

    fn default_config(&self) -> Self::Config {
        TextLineOrientationConfig::get_defaults()
    }
}
