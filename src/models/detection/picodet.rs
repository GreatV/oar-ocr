//! PicoDet Object Detection Model
//!
//! This module provides PicoDet-specific types and configurations.
//! PicoDet is a general-purpose object detection model (not limited to layout detection).
//! The actual implementation is shared via the generic ScaleAwareDetectorModel.

use super::scale_aware_detector::{
    ScaleAwareDetectorInferenceMode, ScaleAwareDetectorModel, ScaleAwareDetectorModelBuilder,
    ScaleAwareDetectorModelOutput, ScaleAwareDetectorPostprocessConfig,
    ScaleAwareDetectorPreprocessConfig,
};
use crate::core::OCRError;
use crate::core::inference::OrtInfer;

/// Preprocessing configuration for PicoDet model.
///
/// This is a type alias for the generic configuration with PicoDet defaults.
pub type PicoDetPreprocessConfig = ScaleAwareDetectorPreprocessConfig;

/// Postprocessing configuration for PicoDet model.
pub type PicoDetPostprocessConfig = ScaleAwareDetectorPostprocessConfig;

/// Output from PicoDet model.
pub type PicoDetModelOutput = ScaleAwareDetectorModelOutput;

/// PicoDet object detection model.
///
/// This is a specialized configuration of the generic ScaleAwareDetectorModel
/// that uses ScaleFactorOnly inference mode (no im_shape input required).
pub type PicoDetModel = ScaleAwareDetectorModel;

/// Builder for PicoDet model.
#[derive(Debug, Default)]
pub struct PicoDetModelBuilder {
    inner: ScaleAwareDetectorModelBuilder,
}

impl PicoDetModelBuilder {
    /// Creates a new PicoDet builder with default settings.
    pub fn new() -> Self {
        Self {
            inner: ScaleAwareDetectorModelBuilder::picodet(),
        }
    }

    /// Sets the preprocessing configuration.
    pub fn preprocess_config(mut self, config: PicoDetPreprocessConfig) -> Self {
        self.inner = self.inner.preprocess_config(config);
        self
    }

    /// Sets the image shape.
    pub fn image_shape(mut self, height: u32, width: u32) -> Self {
        self.inner = self.inner.image_shape(height, width);
        self
    }

    /// Builds the PicoDet model.
    pub fn build(self, inference: OrtInfer) -> Result<PicoDetModel, OCRError> {
        self.inner
            .inference_mode(ScaleAwareDetectorInferenceMode::ScaleFactorOnly)
            .build(inference)
    }
}
