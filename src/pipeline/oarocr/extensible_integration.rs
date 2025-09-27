//! Integration layer for the extensible pipeline system with OAROCR.
//!
//! This module provides utilities to integrate the extensible pipeline system
//! with the existing OAROCR pipeline while maintaining backward compatibility.

use image::RgbImage;
use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info};

use crate::core::OCRError;
use crate::pipeline::oarocr::{OAROCRConfig, OAROCRResult};
use crate::pipeline::stages::{
    CroppingConfig, CroppingResult, ExtensibleCroppingStage, ExtensibleOrientationStage,
    ExtensiblePipeline, ExtensiblePipelineConfig, ExtensibleRecognitionStage,
    ExtensibleTextDetectionStage, ExtensibleTextLineOrientationStage, OrientationConfig,
    OrientationResult, RecognitionConfig, RecognitionResult, StageContext, StageData, StageId,
    TextDetectionConfig, TextDetectionResult, TextLineOrientationConfig,
};

/// Integration wrapper that bridges the extensible pipeline with OAROCR.
pub struct ExtensibleOAROCR {
    /// The extensible pipeline
    pipeline: ExtensiblePipeline,
    /// Configuration for the extensible pipeline
    config: ExtensiblePipelineConfig,
    /// Original OAROCR configuration for fallback
    oarocr_config: OAROCRConfig,
}

impl ExtensibleOAROCR {
    /// Create a new extensible OAROCR instance.
    pub fn new(
        oarocr_config: OAROCRConfig,
        extensible_config: ExtensiblePipelineConfig,
    ) -> Result<Self, OCRError> {
        let mut pipeline = ExtensiblePipeline::new();

        // Register stages based on configuration
        Self::register_standard_stages(&mut pipeline, &oarocr_config, &extensible_config)?;

        Ok(Self {
            pipeline,
            config: extensible_config,
            oarocr_config,
        })
    }

    /// Register standard OCR stages with the pipeline.
    fn register_standard_stages(
        pipeline: &mut ExtensiblePipeline,
        oarocr_config: &OAROCRConfig,
        extensible_config: &ExtensiblePipelineConfig,
    ) -> Result<(), OCRError> {
        // 1. Orientation Stage
        if extensible_config.is_stage_enabled("orientation") {
            let orientation_stage = ExtensibleOrientationStage::new(None); // Would use actual classifier
            let orientation_config = extensible_config
                .get_stage_config::<OrientationConfig>("orientation")
                .or_else(|| oarocr_config.orientation_stage.as_ref().cloned());
            pipeline.register_stage(orientation_stage, orientation_config)?;
            debug!("Registered orientation stage");
        }

        // 2. Text Detection Stage
        if extensible_config.is_stage_enabled("text_detection") {
            let detection_stage = ExtensibleTextDetectionStage::new(None); // Would use actual detector
            let detection_config = extensible_config
                .get_stage_config::<TextDetectionConfig>("text_detection")
                .unwrap_or_default();
            pipeline.register_stage(detection_stage, Some(detection_config))?;
            debug!("Registered text detection stage");
        }

        // 3. Cropping Stage
        if extensible_config.is_stage_enabled("cropping") {
            let cropping_stage = ExtensibleCroppingStage::new();
            let cropping_config = extensible_config
                .get_stage_config::<CroppingConfig>("cropping")
                .unwrap_or_default();
            pipeline.register_stage(cropping_stage, Some(cropping_config))?;
            debug!("Registered cropping stage");
        }

        // 4. Text Line Orientation Stage
        if extensible_config.is_stage_enabled("text_line_orientation") {
            let text_line_stage = ExtensibleTextLineOrientationStage::new(None); // Would use actual classifier
            let text_line_config = extensible_config
                .get_stage_config::<TextLineOrientationConfig>("text_line_orientation")
                .or_else(|| oarocr_config.text_line_orientation_stage.as_ref().cloned());
            pipeline.register_stage(text_line_stage, text_line_config)?;
            debug!("Registered text line orientation stage");
        }

        // 5. Recognition Stage
        if extensible_config.is_stage_enabled("recognition") {
            let recognition_stage = ExtensibleRecognitionStage::new(None); // Would use actual recognizer
            let recognition_config = extensible_config
                .get_stage_config::<RecognitionConfig>("recognition")
                .unwrap_or_else(|| {
                    RecognitionConfig::from_legacy_config(
                        oarocr_config.use_textline_orientation,
                        oarocr_config.aspect_ratio_bucketing.clone(),
                    )
                });
            pipeline.register_stage(recognition_stage, Some(recognition_config))?;
            debug!("Registered recognition stage");
        }

        Ok(())
    }

    /// Process a single image using the extensible pipeline.
    pub fn process_image(&mut self, image_path: &Path) -> Result<OAROCRResult, OCRError> {
        info!(
            "Processing image with extensible pipeline: {:?}",
            image_path
        );

        // Load image
        let input_img = crate::utils::load_image(image_path)?;
        let input_img_arc = Arc::new(input_img.clone());

        // Create stage context
        let context = StageContext::new(input_img_arc.clone(), input_img_arc.clone(), 0);

        // Create initial data
        let initial_data = StageData::new(input_img);

        // Execute pipeline - we need to modify the execute method to return the context
        // For now, we'll use a workaround by executing the pipeline manually
        self.execute_pipeline_and_convert(image_path, input_img_arc, context, initial_data)
    }

    /// Execute the pipeline and convert results to OAROCRResult format.
    fn execute_pipeline_and_convert(
        &mut self,
        image_path: &Path,
        input_img_arc: Arc<RgbImage>,
        mut context: StageContext,
        initial_data: StageData,
    ) -> Result<OAROCRResult, OCRError> {
        // Execute pipeline stages manually to retain access to context
        let execution_order = self.pipeline.registry_mut().resolve_execution_order()?;
        let mut current_data = initial_data;

        info!("Executing pipeline with {} stages", execution_order.len());

        for stage_id in execution_order {
            let stage = self
                .pipeline
                .registry()
                .get_stage(&stage_id)
                .ok_or_else(|| OCRError::ConfigError {
                    message: format!("Stage not found: {}", stage_id.as_str()),
                })?;

            let config = self.pipeline.registry().get_config(&stage_id);

            // Check if stage is enabled
            if !stage.is_enabled(&context, config) {
                debug!("Skipping disabled stage: {}", stage.stage_name());
                continue;
            }

            debug!("Executing stage: {}", stage.stage_name());

            // Execute the stage
            let stage_result = stage.process(&mut context, current_data, config)?;

            // Store the result in context for other stages
            context.set_stage_result(stage_id.clone(), stage_result.data);

            // Update current data - stages that modify the image should update the context
            current_data = StageData::new(context.current_image.as_ref().clone());

            debug!(
                "Stage {} completed in {:?}",
                stage.stage_name(),
                stage_result.metrics.processing_time
            );
        }

        // Convert extensible pipeline results to OAROCRResult format
        self.convert_pipeline_results_to_oarocr(image_path, input_img_arc, &context)
    }

    /// Convert extensible pipeline results to OAROCRResult format.
    fn convert_pipeline_results_to_oarocr(
        &self,
        image_path: &Path,
        input_img_arc: Arc<RgbImage>,
        context: &StageContext,
    ) -> Result<OAROCRResult, OCRError> {
        // Extract results from each stage
        let orientation_result = self.extract_orientation_result(context);
        let text_detection_result = self.extract_text_detection_result(context);
        let cropping_result = self.extract_cropping_result(context);
        let recognition_result = self.extract_recognition_result(context);

        // Build text regions by combining results from all stages
        let text_regions = self.build_text_regions(
            &text_detection_result,
            &cropping_result,
            &recognition_result,
        )?;

        // Calculate error metrics
        let error_metrics = self.calculate_error_metrics(
            &text_detection_result,
            &cropping_result,
            &recognition_result,
        );

        // Get the final processed image (may have been modified by orientation stage)
        let rectified_img = if context.current_image.as_ptr() != context.original_image.as_ptr() {
            Some(context.current_image.clone())
        } else {
            None
        };

        Ok(OAROCRResult {
            input_path: Arc::from(image_path.to_string_lossy().as_ref()),
            index: 0,
            input_img: input_img_arc,
            text_regions,
            orientation_angle: orientation_result.and_then(|r| r.orientation_angle),
            rectified_img,
            error_metrics,
        })
    }

    /// Extract orientation result from the stage context.
    fn extract_orientation_result(&self, context: &StageContext) -> Option<OrientationResult> {
        context
            .get_stage_result::<OrientationResult>(&StageId("orientation".to_string()))
            .cloned()
    }

    /// Extract text detection result from the stage context.
    fn extract_text_detection_result(&self, context: &StageContext) -> Option<TextDetectionResult> {
        context
            .get_stage_result::<TextDetectionResult>(&StageId("text_detection".to_string()))
            .cloned()
    }

    /// Extract cropping result from the stage context.
    fn extract_cropping_result(&self, context: &StageContext) -> Option<CroppingResult> {
        context
            .get_stage_result::<CroppingResult>(&StageId("cropping".to_string()))
            .cloned()
    }

    /// Extract recognition result from the stage context.
    fn extract_recognition_result(&self, context: &StageContext) -> Option<RecognitionResult> {
        context
            .get_stage_result::<RecognitionResult>(&StageId("recognition".to_string()))
            .cloned()
    }

    /// Build text regions by combining results from all stages.
    fn build_text_regions(
        &self,
        text_detection_result: &Option<TextDetectionResult>,
        _cropping_result: &Option<CroppingResult>,
        recognition_result: &Option<RecognitionResult>,
    ) -> Result<Vec<crate::pipeline::oarocr::TextRegion>, OCRError> {
        use crate::pipeline::oarocr::TextRegion;

        // Get text boxes from detection result
        let empty_boxes = Vec::new();
        let text_boxes = text_detection_result
            .as_ref()
            .map(|r| &r.text_boxes)
            .unwrap_or(&empty_boxes);

        if text_boxes.is_empty() {
            return Ok(Vec::new());
        }

        // Build text regions
        let mut text_regions = Vec::new();
        for (i, bbox) in text_boxes.iter().enumerate() {
            // Get recognition results if available
            let (text, confidence) = if let Some(rec_result) = recognition_result {
                let text = if i < rec_result.rec_texts.len() && !rec_result.rec_texts[i].is_empty()
                {
                    Some(rec_result.rec_texts[i].clone())
                } else {
                    None
                };

                let confidence =
                    if i < rec_result.rec_scores.len() && rec_result.rec_scores[i] > 0.0 {
                        Some(rec_result.rec_scores[i])
                    } else {
                        None
                    };

                (text, confidence)
            } else {
                (None, None)
            };

            // For now, we don't extract text line orientation from the extensible pipeline
            // This could be added later if text line orientation stage is implemented
            let orientation_angle = None;

            let text_region =
                TextRegion::with_all(bbox.clone(), text, confidence, orientation_angle);

            text_regions.push(text_region);
        }

        Ok(text_regions)
    }

    /// Calculate error metrics from stage results.
    fn calculate_error_metrics(
        &self,
        text_detection_result: &Option<TextDetectionResult>,
        cropping_result: &Option<CroppingResult>,
        recognition_result: &Option<RecognitionResult>,
    ) -> crate::pipeline::oarocr::ErrorMetrics {
        use crate::pipeline::oarocr::ErrorMetrics;

        let total_text_boxes = text_detection_result
            .as_ref()
            .map(|r| r.text_boxes.len())
            .unwrap_or(0);

        let failed_crops = cropping_result
            .as_ref()
            .map(|r| r.failed_crops)
            .unwrap_or(0);

        let failed_recognitions = recognition_result
            .as_ref()
            .map(|r| r.failed_recognitions)
            .unwrap_or(0);

        // Text line orientation failures are not tracked in the current extensible pipeline
        let failed_orientations = 0;

        ErrorMetrics {
            failed_crops,
            failed_recognitions,
            failed_orientations,
            total_text_boxes,
        }
    }

    /// Get the extensible pipeline configuration.
    pub fn extensible_config(&self) -> &ExtensiblePipelineConfig {
        &self.config
    }

    /// Get the original OAROCR configuration.
    pub fn oarocr_config(&self) -> &OAROCRConfig {
        &self.oarocr_config
    }

    /// Add a custom stage to the pipeline.
    pub fn add_custom_stage<S, C>(&mut self, stage: S, config: Option<C>) -> Result<(), OCRError>
    where
        S: crate::pipeline::stages::PipelineStage<Config = C> + 'static,
        C: Send
            + Sync
            + std::fmt::Debug
            + Clone
            + crate::core::config::ConfigValidator
            + Default
            + 'static,
    {
        self.pipeline.register_stage(stage, config)
    }
}

/// Builder for creating ExtensibleOAROCR instances.
pub struct ExtensibleOAROCRBuilder {
    oarocr_config: OAROCRConfig,
    extensible_config: Option<ExtensiblePipelineConfig>,
}

impl ExtensibleOAROCRBuilder {
    /// Create a new builder with the given OAROCR configuration.
    pub fn new(oarocr_config: OAROCRConfig) -> Self {
        Self {
            oarocr_config,
            extensible_config: None,
        }
    }

    /// Set the extensible pipeline configuration.
    pub fn extensible_config(mut self, config: ExtensiblePipelineConfig) -> Self {
        self.extensible_config = Some(config);
        self
    }

    /// Use the default OCR pipeline configuration.
    pub fn default_ocr_pipeline(mut self) -> Self {
        self.extensible_config = Some(ExtensiblePipelineConfig::default());
        self
    }

    /// Use the minimal pipeline configuration.
    pub fn minimal_pipeline(mut self) -> Self {
        self.extensible_config = Some(ExtensiblePipelineConfig::default());
        self
    }

    /// Use the layout-aware pipeline configuration.
    pub fn layout_aware_pipeline(mut self) -> Self {
        self.extensible_config = Some(ExtensiblePipelineConfig::default());
        self
    }

    /// Build the ExtensibleOAROCR instance.
    pub fn build(self) -> Result<ExtensibleOAROCR, OCRError> {
        let extensible_config = self.extensible_config.unwrap_or_default();

        ExtensibleOAROCR::new(self.oarocr_config, extensible_config)
    }
}

/// Utility functions for converting between pipeline formats.
pub mod conversion {}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pipeline::stages::ExtensiblePipelineConfig;
    use std::path::PathBuf;

    #[test]
    fn test_extensible_oarocr_creation() {
        let oarocr_config = OAROCRConfig::default();
        let extensible_config = ExtensiblePipelineConfig::default();

        let result = ExtensibleOAROCR::new(oarocr_config, extensible_config);
        assert!(result.is_ok());
    }

    #[test]
    fn test_builder_pattern() {
        let oarocr_config = OAROCRConfig::default();

        let result = ExtensibleOAROCRBuilder::new(oarocr_config)
            .default_ocr_pipeline()
            .build();

        assert!(result.is_ok());
    }

    #[test]
    fn test_conversion_with_empty_results() {
        let oarocr_config = OAROCRConfig::default();
        let extensible_config = ExtensiblePipelineConfig::default();
        let extensible_oarocr = ExtensibleOAROCR::new(oarocr_config, extensible_config).unwrap();

        // Test conversion with empty stage results
        let input_img = RgbImage::new(100, 100);
        let input_img_arc = Arc::new(input_img);
        let context = StageContext::new(input_img_arc.clone(), input_img_arc.clone(), 0);
        let image_path = PathBuf::from("test.jpg");

        let result = extensible_oarocr.convert_pipeline_results_to_oarocr(
            &image_path,
            input_img_arc,
            &context,
        );

        assert!(result.is_ok());
        let oarocr_result = result.unwrap();
        assert_eq!(oarocr_result.text_regions.len(), 0);
        assert_eq!(oarocr_result.orientation_angle, None);
        assert_eq!(oarocr_result.error_metrics.total_text_boxes, 0);
    }
}
