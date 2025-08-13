//! Text recognition processing stage processor.
//!
//! This module provides a refactored text recognition pipeline that separates concerns
//! into dedicated components:
//!
//! ## Architecture
//!
//! The recognition stage has been refactored to separate mixed responsibilities:
//!
//! ### Grouping Strategies
//! - **`GroupingStrategy`** trait: Defines how images are grouped for batch processing
//! - **`ExactDimensionStrategy`**: Groups images by exact pixel dimensions
//! - **`AspectRatioBucketingStrategy`**: Groups images by aspect ratio ranges for better batching efficiency
//!
//! ### Orientation Correction
//! - **`OrientationCorrector`**: Handles text line orientation corrections separately from grouping logic
//! - Configurable through `OrientationCorrectionConfig`
//!
//! ### Configuration
//! - **`RecognitionConfig`**: New unified configuration using the separated components
//! - **`from_legacy_config()`**: Helper method for backward compatibility
//!
//! ## Example Usage
//!
//! ```rust
//! use oar_ocr::pipeline::stages::{
//!     RecognitionConfig, GroupingStrategyConfig, OrientationCorrectionConfig
//! };
//! use oar_ocr::processors::AspectRatioBucketingConfig;
//!
//! // Create config with aspect ratio bucketing and orientation correction
//! let config = RecognitionConfig {
//!     orientation_correction: OrientationCorrectionConfig { enabled: true },
//!     grouping_strategy: GroupingStrategyConfig::AspectRatioBucketing(
//!         AspectRatioBucketingConfig::default()
//!     ),
//! };
//!
//! // Or create from legacy config format
//! let legacy_config = RecognitionConfig::from_legacy_config(
//!     true, // use_textline_orientation
//!     Some(AspectRatioBucketingConfig::default()), // aspect_ratio_bucketing
//! );
//! ```

use image::RgbImage;
use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::metrics;
use tracing::{debug, warn};

use super::extensible::{PipelineStage, StageContext, StageData, StageDependency, StageId};

use super::types::StageResult;
use crate::core::config::ConfigValidator;
use crate::core::{OCRError, traits::StandardPredictor};
use crate::predictor::TextRecPredictor;

mod grouping;
mod orientation_correction;

pub use grouping::{GroupingStrategy, GroupingStrategyConfig, GroupingStrategyFactory};
pub use orientation_correction::{OrientationCorrectionConfig, OrientationCorrector};

/// Result of recognition processing
#[derive(Debug, Clone)]
pub struct RecognitionResult {
    /// Recognition texts in original order
    pub rec_texts: Vec<Arc<str>>,
    /// Recognition scores in original order
    pub rec_scores: Vec<f32>,
    /// Number of failed recognitions
    pub failed_recognitions: usize,
}

/// Configuration for recognition processing
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct RecognitionConfig {
    /// Configuration for orientation correction
    #[serde(default)]
    pub orientation_correction: OrientationCorrectionConfig,
    /// Configuration for grouping strategy
    #[serde(default)]
    pub grouping_strategy: GroupingStrategyConfig,
}

impl RecognitionConfig {
    /// Create a RecognitionConfig from the old config format
    pub fn from_legacy_config(
        use_textline_orientation: bool,
        aspect_ratio_bucketing: Option<crate::processors::AspectRatioBucketingConfig>,
    ) -> Self {
        let orientation_correction = OrientationCorrectionConfig {
            enabled: use_textline_orientation,
        };

        let grouping_strategy = if let Some(bucketing_config) = aspect_ratio_bucketing {
            GroupingStrategyConfig::AspectRatioBucketing(bucketing_config)
        } else {
            GroupingStrategyConfig::ExactDimensions
        };

        Self {
            orientation_correction,
            grouping_strategy,
        }
    }
}

impl crate::core::config::ConfigValidator for RecognitionConfig {
    fn validate(&self) -> Result<(), crate::core::config::ConfigError> {
        // RecognitionConfig validation - basic validation for now
        // Could add more specific validation for grouping strategy if needed
        Ok(())
    }

    fn get_defaults() -> Self {
        Self::default()
    }
}

/// Processor for text recognition grouping and processing stage.
///
/// This processor encapsulates the logic for:
/// - Grouping text images by aspect ratio buckets or exact dimensions
/// - Applying text line orientation corrections
/// - Batch recognition processing with consistent error handling
/// - Collecting and ordering recognition results
pub struct RecognitionStageProcessor;

impl RecognitionStageProcessor {
    /// Process text recognition for cropped text images.
    ///
    /// # Arguments
    ///
    /// * `cropped_images` - Vector of optional cropped images (None for failed crops)
    /// * `text_line_orientations` - Optional orientation angles for each text region
    /// * `recognizer` - Optional text recognizer
    /// * `config` - Configuration for recognition processing
    ///
    /// # Returns
    ///
    /// A StageResult containing the recognition result and processing metrics
    pub fn process_single(
        cropped_images: Vec<Option<RgbImage>>,
        text_line_orientations: Option<&[Option<f32>]>,
        recognizer: Option<&TextRecPredictor>,
        config: Option<&RecognitionConfig>,
    ) -> Result<StageResult<RecognitionResult>, OCRError> {
        use std::time::Instant;
        let start_time = Instant::now();
        let default_config = RecognitionConfig::default();
        let config = config.unwrap_or(&default_config);

        let mut failed_recognitions = 0;
        let (rec_texts, rec_scores) = if cropped_images.is_empty() {
            (Vec::new(), Vec::new())
        } else if let Some(recognizer) = recognizer {
            Self::process_recognition_groups(
                &cropped_images,
                text_line_orientations,
                recognizer,
                config,
                &mut failed_recognitions,
            )?
        } else {
            debug!("No text recognizer available, skipping recognition");
            let empty_texts = vec![Arc::from(""); cropped_images.len()];
            let empty_scores = vec![0.0; cropped_images.len()];
            (empty_texts, empty_scores)
        };

        let success_count = cropped_images.len() - failed_recognitions;
        let grouping_strategy_name = match config.grouping_strategy {
            GroupingStrategyConfig::AspectRatioBucketing(_) => "aspect_ratio_bucketing",
            GroupingStrategyConfig::ExactDimensions => "exact_dimensions",
        };
        let metrics = metrics!(success_count, failed_recognitions, start_time;
            stage = "recognition",
            text_regions = cropped_images.len(),
            grouping_strategy = grouping_strategy_name
        );

        let result = RecognitionResult {
            rec_texts,
            rec_scores,
            failed_recognitions,
        };

        Ok(StageResult::new(result, metrics))
    }

    /// Process recognition by grouping images and running batch recognition.
    fn process_recognition_groups(
        cropped_images: &[Option<RgbImage>],
        text_line_orientations: Option<&[Option<f32>]>,
        recognizer: &TextRecPredictor,
        config: &RecognitionConfig,
        failed_recognitions: &mut usize,
    ) -> Result<(Vec<Arc<str>>, Vec<f32>), OCRError> {
        // Prepare images for grouping (filter out None values)
        let images_for_grouping: Vec<(usize, RgbImage)> = cropped_images
            .iter()
            .enumerate()
            .filter_map(|(i, cropped_img_opt)| cropped_img_opt.as_ref().map(|img| (i, img.clone())))
            .collect();

        // Create grouping strategy and group images
        let grouping_strategy =
            GroupingStrategyFactory::create_strategy(&config.grouping_strategy)?;
        let groups = grouping_strategy.group_images(images_for_grouping)?;

        // Create orientation corrector
        let orientation_corrector =
            OrientationCorrector::new(config.orientation_correction.clone());

        // Process each group
        let mut recognition_results: Vec<(usize, Arc<str>, f32)> = Vec::new();
        for (group_name, group) in groups {
            Self::process_recognition_group(
                group_name,
                group,
                text_line_orientations,
                recognizer,
                &orientation_corrector,
                &mut recognition_results,
                failed_recognitions,
            )?;
        }

        // Sort results by original index and extract texts and scores
        recognition_results.sort_by_key(|(idx, _, _)| *idx);
        let mut rec_texts = vec![Arc::from(""); cropped_images.len()];
        let mut rec_scores = vec![0.0; cropped_images.len()];

        for (original_idx, text, score) in recognition_results {
            if original_idx < rec_texts.len() {
                rec_texts[original_idx] = text;
                rec_scores[original_idx] = score;
            }
        }

        Ok((rec_texts, rec_scores))
    }

    /// Process a single recognition group.
    fn process_recognition_group(
        group_name: String,
        group: Vec<(usize, RgbImage)>,
        text_line_orientations: Option<&[Option<f32>]>,
        recognizer: &TextRecPredictor,
        orientation_corrector: &OrientationCorrector,
        recognition_results: &mut Vec<(usize, Arc<str>, f32)>,
        failed_recognitions: &mut usize,
    ) -> Result<(), OCRError> {
        let (indices, mut images): (Vec<usize>, Vec<RgbImage>) = group.into_iter().unzip();

        // Apply text line orientation corrections using the corrector
        let corrections_applied =
            orientation_corrector.apply_corrections(&mut images, &indices, text_line_orientations);

        if corrections_applied > 0 {
            debug!(
                "Applied {} orientation corrections in group '{}'",
                corrections_applied, group_name
            );
        }

        debug!(
            "Processing recognition group '{}' with {} images",
            group_name,
            images.len()
        );

        match recognizer.predict(images, None) {
            Ok(result) => {
                for (original_idx, (text, score)) in indices
                    .into_iter()
                    .zip(result.rec_text.iter().zip(result.rec_score.iter()))
                {
                    recognition_results.push((original_idx, text.clone(), *score));
                }
            }
            Err(e) => {
                *failed_recognitions += indices.len();

                // Create enhanced error message using the common helper
                let enhanced_error = crate::core::OCRError::format_batch_error_message(
                    "text recognition",
                    &group_name,
                    &indices,
                    &e,
                );

                warn!(
                    "Text recognition failed for batch '{}' of {} images (indices: {:?}): {}",
                    group_name,
                    indices.len(),
                    indices,
                    enhanced_error
                );

                for original_idx in indices {
                    recognition_results.push((original_idx, Arc::from(""), 0.0));
                }
            }
        }

        Ok(())
    }
}

/// Extensible recognition stage that implements PipelineStage trait.
#[derive(Debug)]
pub struct ExtensibleRecognitionStage {
    recognizer: Option<Arc<TextRecPredictor>>,
}

impl ExtensibleRecognitionStage {
    /// Create a new extensible recognition stage.
    pub fn new(recognizer: Option<Arc<TextRecPredictor>>) -> Self {
        Self { recognizer }
    }
}

impl PipelineStage for ExtensibleRecognitionStage {
    type Config = RecognitionConfig;
    type Result = RecognitionResult;

    fn stage_id(&self) -> StageId {
        StageId::new("recognition")
    }

    fn stage_name(&self) -> &str {
        "Text Recognition"
    }

    fn dependencies(&self) -> Vec<StageDependency> {
        // Recognition depends on cropping results
        vec![StageDependency::Requires(StageId::new("cropping"))]
    }

    fn is_enabled(&self, _context: &StageContext, _config: Option<&Self::Config>) -> bool {
        self.recognizer.is_some()
    }

    fn process(
        &self,
        context: &mut StageContext,
        _data: StageData,
        config: Option<&Self::Config>,
    ) -> Result<StageResult<Self::Result>, OCRError> {
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

        // Get text line orientations if available
        let text_line_orientations =
            context.get_stage_result::<Vec<Option<f32>>>(&StageId::new("text_line_orientation"));

        let recognition_config = config.cloned().unwrap_or_default();

        let stage_result = RecognitionStageProcessor::process_single(
            cropping_result.cropped_images.clone(),
            text_line_orientations.map(|v| &**v),
            self.recognizer.as_ref().map(|r| r.as_ref()),
            Some(&recognition_config),
        )?;

        Ok(stage_result)
    }

    fn validate_config(&self, config: &Self::Config) -> Result<(), OCRError> {
        config.validate().map_err(|e| OCRError::ConfigError {
            message: format!("RecognitionConfig validation failed: {}", e),
        })
    }

    fn default_config(&self) -> Self::Config {
        RecognitionConfig::get_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processors::AspectRatioBucketingConfig;

    #[test]
    fn test_recognition_config_default() {
        let config = RecognitionConfig::default();
        assert!(!config.orientation_correction.enabled);
        match config.grouping_strategy {
            GroupingStrategyConfig::ExactDimensions => {
                // This is expected
            }
            _ => panic!("Default should be ExactDimensions"),
        }
    }

    #[test]
    fn test_recognition_config_from_legacy_config() {
        // Test with orientation enabled and aspect ratio bucketing
        let config = RecognitionConfig::from_legacy_config(
            true,
            Some(AspectRatioBucketingConfig::default()),
        );

        assert!(config.orientation_correction.enabled);
        match config.grouping_strategy {
            GroupingStrategyConfig::AspectRatioBucketing(_) => {
                // This is expected
            }
            _ => panic!("Should be AspectRatioBucketing"),
        }

        // Test with orientation disabled and no bucketing
        let config = RecognitionConfig::from_legacy_config(false, None);

        assert!(!config.orientation_correction.enabled);
        match config.grouping_strategy {
            GroupingStrategyConfig::ExactDimensions => {
                // This is expected
            }
            _ => panic!("Should be ExactDimensions"),
        }
    }

    #[test]
    fn test_recognition_config_serialization() {
        // Create a custom bucketing config without infinity to avoid JSON serialization issues
        let mut bucketing_config = AspectRatioBucketingConfig::default();
        // Replace the last bucket that uses infinity with a large finite value
        if let Some(last_bucket) = bucketing_config.buckets.last_mut() {
            last_bucket.max_ratio = 1000.0; // Use a large finite value instead of infinity
        }

        let config = RecognitionConfig::from_legacy_config(true, Some(bucketing_config));

        // Test that the config can be serialized and deserialized
        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: RecognitionConfig = serde_json::from_str(&serialized).unwrap();

        assert_eq!(
            config.orientation_correction.enabled,
            deserialized.orientation_correction.enabled
        );

        match (&config.grouping_strategy, &deserialized.grouping_strategy) {
            (
                GroupingStrategyConfig::AspectRatioBucketing(_),
                GroupingStrategyConfig::AspectRatioBucketing(_),
            ) => {
                // This is expected
            }
            _ => panic!("Grouping strategy should match after serialization"),
        }
    }
}
