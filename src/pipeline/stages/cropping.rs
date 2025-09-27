//! Text box cropping stage processor.

use image::RgbImage;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};

use tracing::{debug, warn};

use super::extensible::{PipelineStage, StageContext, StageData, StageDependency, StageId};
use super::processor_helper::{BatchConfig, BatchProcessor};
use super::types::{StageMetrics, StageResult};
use crate::core::OCRError;
use crate::core::config::ConfigValidator;
use crate::pipeline::oarocr::ImageProcessor;
use crate::processors::BoundingBox;

/// Result of cropping processing
#[derive(Debug, Clone)]
pub struct CroppingResult {
    /// Successfully cropped images (None for failed crops)
    pub cropped_images: Vec<Option<RgbImage>>,
    /// Number of failed cropping operations
    pub failed_crops: usize,
}

/// Configuration for cropping processing
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct CroppingConfig {
    // Currently empty, but kept for future configuration options
}

impl ConfigValidator for CroppingConfig {
    fn validate(&self) -> Result<(), crate::core::config::ConfigError> {
        // No validation needed for empty config
        Ok(())
    }

    fn get_defaults() -> Self {
        Self::default()
    }
}

impl CroppingConfig {
    /// Create a new CroppingConfig
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the effective parallel threshold from the provided policy
    pub fn effective_threshold(&self, policy_threshold: Option<usize>) -> usize {
        policy_threshold.unwrap_or(10) // Default threshold
    }
}

/// Processor for text box cropping stage.
///
/// This processor encapsulates the logic for:
/// - Cropping text boxes from images (both rotated and regular bounding boxes)
/// - Parallel processing based on configurable thresholds
/// - Consistent error handling and metrics collection
pub struct CroppingStageProcessor;

impl CroppingStageProcessor {
    /// Process text box cropping for a single image.
    ///
    /// # Arguments
    ///
    /// * `image` - The source image to crop from
    /// * `text_boxes` - Vector of bounding boxes to crop
    /// * `config` - Configuration for cropping processing
    ///
    /// # Returns
    ///
    /// A StageResult containing the cropping result and processing metrics
    pub fn process_single(
        image: &RgbImage,
        text_boxes: &[BoundingBox],
        config: Option<&CroppingConfig>,
    ) -> Result<StageResult<CroppingResult>, OCRError> {
        Self::process_single_with_policy(image, text_boxes, config, None)
    }

    /// Process text box cropping for a single image with policy threshold.
    ///
    /// # Arguments
    ///
    /// * `image` - The source image to crop from
    /// * `text_boxes` - Vector of bounding boxes to crop
    /// * `config` - Optional configuration for cropping processing
    /// * `policy_threshold` - Optional threshold from parallel policy
    ///
    /// # Returns
    ///
    /// A StageResult containing the cropping results and processing metrics
    pub fn process_single_with_policy(
        image: &RgbImage,
        text_boxes: &[BoundingBox],
        config: Option<&CroppingConfig>,
        policy_threshold: Option<usize>,
    ) -> Result<StageResult<CroppingResult>, OCRError> {
        use std::time::Instant;
        let start_time = Instant::now();
        let default_config = CroppingConfig::default();
        let config = config.unwrap_or(&default_config);

        debug!("Processing {} text boxes for cropping", text_boxes.len());

        if text_boxes.is_empty() {
            let metrics = StageMetrics::new(0, 0)
                .with_processing_time(start_time.elapsed())
                .with_info("stage", "cropping")
                .with_info("text_boxes", "0");

            return Ok(StageResult::new(
                CroppingResult {
                    cropped_images: Vec::new(),
                    failed_crops: 0,
                },
                metrics,
            ));
        }

        // Choose sequential or parallel processing based on threshold
        let effective_threshold = config.effective_threshold(policy_threshold);
        let use_parallel = text_boxes.len() > effective_threshold;
        let cropping_results: Vec<(usize, Result<RgbImage, OCRError>)> = if use_parallel {
            debug!(
                "Using parallel cropping for {} text boxes",
                text_boxes.len()
            );
            text_boxes
                .par_iter()
                .enumerate()
                .map(|(idx, bbox)| {
                    let crop_result = Self::crop_bounding_box(image, bbox);
                    (idx, crop_result)
                })
                .collect()
        } else {
            debug!(
                "Using sequential cropping for {} text boxes",
                text_boxes.len()
            );
            text_boxes
                .iter()
                .enumerate()
                .map(|(idx, bbox)| {
                    let crop_result = Self::crop_bounding_box(image, bbox);
                    (idx, crop_result)
                })
                .collect()
        };

        // Process results and count failures
        let mut failed_crops = 0;
        let cropped_images: Vec<Option<RgbImage>> = cropping_results
            .into_iter()
            .map(|(idx, crop_result)| match crop_result {
                Ok(img) => Some(img),
                Err(e) => {
                    failed_crops += 1;
                    warn!(
                        "Failed to crop text box {} with {} points: {}",
                        idx,
                        text_boxes[idx].points.len(),
                        e
                    );
                    None
                }
            })
            .collect();

        let success_count = text_boxes.len() - failed_crops;
        let result = CroppingResult {
            cropped_images,
            failed_crops,
        };

        let metrics = StageMetrics::new(success_count, failed_crops)
            .with_processing_time(start_time.elapsed())
            .with_info("stage", "cropping")
            .with_info("text_boxes", text_boxes.len().to_string())
            .with_info("parallel_processing", use_parallel.to_string());

        Ok(StageResult::new(result, metrics))
    }

    /// Crop a single bounding box from an image.
    ///
    /// This method handles both rotated (4-point) and regular (axis-aligned) bounding boxes.
    ///
    /// # Arguments
    ///
    /// * `image` - The source image to crop from
    /// * `bbox` - The bounding box to crop
    ///
    /// # Returns
    ///
    /// Result containing the cropped image or an error
    fn crop_bounding_box(image: &RgbImage, bbox: &BoundingBox) -> Result<RgbImage, OCRError> {
        if bbox.points.len() == 4 {
            // Rotated bounding box (quadrilateral)
            ImageProcessor::crop_rotated_bounding_box(image, bbox)
        } else {
            // Regular axis-aligned bounding box
            ImageProcessor::crop_bounding_box(image, bbox)
        }
    }

    /// Process text box cropping for multiple images.
    ///
    /// # Arguments
    ///
    /// * `images_and_boxes` - Vector of (image, text_boxes) pairs to process
    /// * `config` - Configuration for cropping processing
    ///
    /// # Returns
    ///
    /// A StageResult containing the cropping results and processing metrics
    pub fn process_batch(
        images_and_boxes: Vec<(&RgbImage, &[BoundingBox])>,
        config: Option<&CroppingConfig>,
    ) -> Result<StageResult<Vec<CroppingResult>>, OCRError> {
        Self::process_batch_with_policy(images_and_boxes, config, None)
    }

    /// Process text box cropping for multiple images with policy threshold.
    ///
    /// # Arguments
    ///
    /// * `images_and_boxes` - Vector of (image, text_boxes) pairs to process
    /// * `config` - Configuration for cropping processing
    /// * `policy_threshold` - Optional threshold from parallel policy
    ///
    /// # Returns
    ///
    /// A StageResult containing the cropping results and processing metrics
    pub fn process_batch_with_policy(
        images_and_boxes: Vec<(&RgbImage, &[BoundingBox])>,
        config: Option<&CroppingConfig>,
        _policy_threshold: Option<usize>,
    ) -> Result<StageResult<Vec<CroppingResult>>, OCRError> {
        let batch_config = BatchConfig::new("cropping_batch").with_fallback_results(true);

        let processor = BatchProcessor::new(&batch_config);

        // Convert to owned data for processing
        let owned_data: Vec<(RgbImage, Vec<BoundingBox>)> = images_and_boxes
            .into_iter()
            .map(|(image, boxes)| (image.clone(), boxes.to_vec()))
            .collect();

        let result = processor.process_items(
            owned_data,
            |(image, text_boxes)| {
                Self::process_single(&image, &text_boxes, config).map(|stage_result| {
                    // Return both the data and the metrics for aggregation
                    (
                        stage_result.data,
                        stage_result.metrics.success_count,
                        stage_result.metrics.failure_count,
                    )
                })
            },
            |e, index| {
                warn!("Cropping processing failed for image {}: {}", index, e);
                // Create a fallback result - we don't know the exact number of text boxes here
                // so we'll create an empty result
                Some((
                    CroppingResult {
                        cropped_images: Vec::new(),
                        failed_crops: 0,
                    },
                    0,
                    1,
                ))
            },
        )?;

        // Aggregate the results and metrics
        let mut cropping_results = Vec::new();
        let mut total_success = 0;
        let mut total_failures = 0;

        for (cropping_result, success_count, failure_count) in result.data {
            cropping_results.push(cropping_result);
            total_success += success_count;
            total_failures += failure_count;
        }

        // Create updated metrics with aggregated counts
        let mut updated_metrics = result.metrics;
        updated_metrics.success_count = total_success;
        updated_metrics.failure_count = total_failures;
        updated_metrics
            .additional_info
            .insert("batch_size".to_string(), cropping_results.len().to_string());

        Ok(StageResult::new(cropping_results, updated_metrics))
    }
}

/// Extensible cropping stage that implements PipelineStage trait.
#[derive(Debug)]
pub struct ExtensibleCroppingStage;

impl ExtensibleCroppingStage {
    /// Create a new extensible cropping stage.
    pub fn new() -> Self {
        Self
    }
}

impl Default for ExtensibleCroppingStage {
    fn default() -> Self {
        Self::new()
    }
}

impl PipelineStage for ExtensibleCroppingStage {
    type Config = CroppingConfig;
    type Result = CroppingResult;

    fn stage_id(&self) -> StageId {
        StageId::new("cropping")
    }

    fn stage_name(&self) -> &str {
        "Text Box Cropping"
    }

    fn dependencies(&self) -> Vec<StageDependency> {
        // Cropping depends on text detection results
        vec![StageDependency::Requires(StageId::new("text_detection"))]
    }

    fn is_enabled(&self, context: &StageContext, _config: Option<&Self::Config>) -> bool {
        // Check if we have text boxes from detection stage
        context
            .get_stage_result::<Vec<BoundingBox>>(&StageId::new("text_detection"))
            .is_some()
    }

    fn process(
        &self,
        context: &mut StageContext,
        data: StageData,
        config: Option<&Self::Config>,
    ) -> Result<StageResult<Self::Result>, OCRError> {
        // Get text boxes from the detection stage
        let text_boxes = context
            .get_stage_result::<Vec<BoundingBox>>(&StageId::new("text_detection"))
            .ok_or_else(|| {
                OCRError::processing_error(
                    crate::core::ProcessingStage::Generic,
                    "Text boxes not found in context",
                    crate::core::errors::SimpleError::new("Missing text detection results"),
                )
            })?;

        let cropping_config = config.cloned().unwrap_or_default();

        let stage_result = CroppingStageProcessor::process_single(
            &data.image,
            text_boxes,
            Some(&cropping_config),
        )?;

        Ok(stage_result)
    }

    fn validate_config(&self, config: &Self::Config) -> Result<(), OCRError> {
        config.validate().map_err(|e| OCRError::ConfigError {
            message: format!("CroppingConfig validation failed: {}", e),
        })
    }

    fn default_config(&self) -> Self::Config {
        CroppingConfig::get_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cropping_config_effective_threshold() {
        let config = CroppingConfig::new();

        // Test with policy threshold
        assert_eq!(config.effective_threshold(Some(5)), 5);

        // Test without policy threshold (uses default)
        assert_eq!(config.effective_threshold(None), 10);
    }

    #[test]
    fn test_cropping_config_serialization() {
        let config = CroppingConfig::new();

        let serialized = serde_json::to_string(&config).unwrap();
        let deserialized: CroppingConfig = serde_json::from_str(&serialized).unwrap();

        // Test that serialization/deserialization works correctly
        assert_eq!(
            config.effective_threshold(None),
            deserialized.effective_threshold(None)
        );
    }
}
