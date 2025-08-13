//! Integration layer for the extensible pipeline system with OAROCR.
//!
//! This module provides utilities to integrate the extensible pipeline system
//! with the existing OAROCR pipeline while maintaining backward compatibility.

use std::path::Path;
use std::sync::Arc;
use tracing::{debug, info};

use crate::core::OCRError;
use crate::pipeline::oarocr::{OAROCRConfig, OAROCRResult};
use crate::pipeline::stages::{
    CroppingConfig, ExtensibleCroppingStage, ExtensibleOrientationStage, ExtensiblePipeline,
    ExtensiblePipelineConfig, ExtensibleRecognitionStage, ExtensibleTextDetectionStage,
    ExtensibleTextLineOrientationStage, OrientationConfig, PipelineExecutor, RecognitionConfig,
    StageContext, StageData, TextDetectionConfig, TextLineOrientationConfig,
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

        // Execute pipeline
        let _result = PipelineExecutor::execute(&mut self.pipeline, context, initial_data)?;

        // For now, return a placeholder result
        // In a full implementation, this would convert the extensible pipeline results
        // to the OAROCRResult format
        Ok(OAROCRResult {
            input_path: Arc::from(image_path.to_string_lossy().as_ref()),
            index: 0,
            input_img: input_img_arc,
            text_regions: Vec::new(),
            orientation_angle: None,
            rectified_img: None,
            error_metrics: crate::pipeline::oarocr::ErrorMetrics {
                failed_crops: 0,
                failed_recognitions: 0,
                failed_orientations: 0,
                total_text_boxes: 0,
            },
        })
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
