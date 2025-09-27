//! Document orientation classification stage processor.

use image::RgbImage;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use tracing::{debug, warn};

use super::extensible::{PipelineStage, StageContext, StageData, StageDependency, StageId};
use super::processor_helper::{BatchConfig, BatchProcessor, SingleItemProcessor};
use super::types::StageResult;
use crate::core::config::ConfigValidator;
use crate::core::{
    OCRError, apply_document_orientation, parse_document_orientation, traits::StandardPredictor,
};
use crate::predictor::DocOrientationClassifier;

/// Result of orientation classification processing
#[derive(Debug, Clone)]
pub struct OrientationResult {
    /// The detected orientation angle (None if no classifier or low confidence)
    pub orientation_angle: Option<f32>,
    /// The corrected image after applying orientation
    pub corrected_image: RgbImage,
}

/// Configuration for orientation processing
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OrientationConfig {
    /// Confidence threshold for accepting orientation predictions
    pub confidence_threshold: Option<f32>,
}

impl ConfigValidator for OrientationConfig {
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
        Self {
            confidence_threshold: Some(0.5),
        }
    }
}

/// Processor for document orientation classification stage.
///
/// This processor encapsulates the logic for:
/// - Document orientation classification with confidence thresholding
/// - Image correction based on detected orientation
/// - Consistent error handling and metrics collection
pub struct OrientationStageProcessor;

impl OrientationStageProcessor {
    /// Process document orientation classification for a single image.
    ///
    /// # Arguments
    ///
    /// * `image` - The input image to classify and correct
    /// * `classifier` - Optional orientation classifier
    /// * `config` - Configuration for orientation processing
    ///
    /// # Returns
    ///
    /// A StageResult containing the orientation result and processing metrics
    pub fn process_single(
        image: Arc<RgbImage>,
        classifier: Option<&DocOrientationClassifier>,
        config: Option<&OrientationConfig>,
    ) -> Result<StageResult<OrientationResult>, OCRError> {
        let processor = SingleItemProcessor::new("orientation");

        let (orientation_angle, corrected_image) = if let Some(classifier) = classifier {
            debug!("Processing document orientation classification");

            // Clone the image for classification
            let input_img_clone = image.as_ref().clone();
            let result = classifier.predict(vec![input_img_clone.clone()], None)?;

            let angle = if let (Some(labels), Some(scores)) =
                (result.label_names.first(), result.scores.first())
            {
                if let (Some(label), Some(&score)) = (labels.first(), scores.first()) {
                    let confidence_threshold = config.and_then(|c| c.confidence_threshold);

                    let orientation_result =
                        parse_document_orientation(label.as_ref(), score, confidence_threshold);

                    if orientation_result.is_confident {
                        debug!(
                            "Detected orientation: {} degrees (confidence: {:.3})",
                            orientation_result.angle, score
                        );
                        orientation_result.angle
                    } else {
                        debug!(
                            "Low confidence orientation detection: {} degrees (confidence: {:.3}, threshold: {:?})",
                            orientation_result.angle, score, confidence_threshold
                        );
                        0.0
                    }
                } else {
                    warn!("Invalid orientation classification result format");
                    0.0
                }
            } else {
                warn!("Empty orientation classification result");
                0.0
            };

            let corrected_img = if angle != 0.0 {
                debug!(
                    "Applying document orientation correction: {} degrees",
                    angle
                );
                apply_document_orientation(input_img_clone, angle)
            } else {
                input_img_clone
            };

            (Some(angle), corrected_img)
        } else {
            debug!("No orientation classifier available, skipping orientation correction");
            (None, image.as_ref().clone())
        };

        let result = OrientationResult {
            orientation_angle,
            corrected_image,
        };

        let additional_info = vec![(
            "angle_detected",
            orientation_angle.map_or("none".to_string(), |a| a.to_string()),
        )];

        Ok(processor.complete_with_info(result, true, additional_info))
    }

    /// Process document orientation classification for multiple images.
    ///
    /// # Arguments
    ///
    /// * `images` - Vector of input images to classify and correct
    /// * `classifier` - Optional orientation classifier
    /// * `config` - Configuration for orientation processing
    ///
    /// # Returns
    ///
    /// A StageResult containing the orientation results and processing metrics
    pub fn process_batch(
        images: Vec<Arc<RgbImage>>,
        classifier: Option<&DocOrientationClassifier>,
        config: Option<&OrientationConfig>,
    ) -> Result<StageResult<Vec<OrientationResult>>, OCRError> {
        let batch_config = BatchConfig::new("orientation_batch").with_fallback_results(true);

        let processor = BatchProcessor::new(&batch_config);

        // Keep a cheap Arc-cloned copy of inputs for safe fallbacks
        let originals = images.clone();

        processor.process_items(
            images,
            |image| Self::process_single(image, classifier, config).map(|result| result.data),
            move |e, index| {
                // Create enhanced error using the common helper
                let enhanced_error = OCRError::batch_item_error(
                    "orientation",
                    Some("orientation_batch"),
                    index,
                    None, // Total images not available in this context
                    "process_single",
                    e,
                );

                warn!(
                    "Orientation processing failed for image {}: {}",
                    index + 1,
                    enhanced_error
                );

                // Fallback: return the original image unchanged to avoid misleading placeholders
                originals
                    .get(index)
                    .cloned()
                    .map(|img_arc| OrientationResult {
                        orientation_angle: None,
                        corrected_image: img_arc.as_ref().clone(),
                    })
            },
        )
    }
}

/// Extensible orientation stage that implements PipelineStage trait.
#[derive(Debug)]
pub struct ExtensibleOrientationStage {
    classifier: Option<Arc<DocOrientationClassifier>>,
}

impl ExtensibleOrientationStage {
    /// Create a new extensible orientation stage.
    pub fn new(classifier: Option<Arc<DocOrientationClassifier>>) -> Self {
        Self { classifier }
    }
}

impl PipelineStage for ExtensibleOrientationStage {
    type Config = OrientationConfig;
    type Result = OrientationResult;

    fn stage_id(&self) -> StageId {
        StageId::new("orientation")
    }

    fn stage_name(&self) -> &str {
        "Document Orientation Classification"
    }

    fn dependencies(&self) -> Vec<StageDependency> {
        // Orientation should run early in the pipeline
        Vec::new()
    }

    fn is_enabled(&self, _context: &StageContext, _config: Option<&Self::Config>) -> bool {
        self.classifier.is_some()
    }

    fn process(
        &self,
        context: &mut StageContext,
        data: StageData,
        config: Option<&Self::Config>,
    ) -> Result<StageResult<Self::Result>, OCRError> {
        let image_arc = Arc::new(data.image);

        let stage_result = OrientationStageProcessor::process_single(
            image_arc,
            self.classifier.as_ref().map(|c| c.as_ref()),
            config,
        )?;

        // Update the context with the corrected image
        context.current_image = Arc::new(stage_result.data.corrected_image.clone());

        Ok(stage_result)
    }

    fn validate_config(&self, config: &Self::Config) -> Result<(), OCRError> {
        config.validate().map_err(|e| OCRError::ConfigError {
            message: format!("OrientationConfig validation failed: {}", e),
        })
    }

    fn default_config(&self) -> Self::Config {
        OrientationConfig::get_defaults()
    }
}
