//! PP-DocLayout Detection Model
//!
//! This module provides PP-DocLayout-specific types and configurations.
//! PP-DocLayout is specialized for document layout detection.
//! The actual implementation is shared via the generic ScaleAwareDetectorModel.

use super::scale_aware_detector::{
    ScaleAwareDetectorInferenceMode, ScaleAwareDetectorModel, ScaleAwareDetectorModelBuilder,
    ScaleAwareDetectorModelOutput, ScaleAwareDetectorPostprocessConfig,
    ScaleAwareDetectorPreprocessConfig,
};
use crate::core::OCRError;
use crate::core::inference::OrtInfer;

/// Preprocessing configuration for PP-DocLayout model.
///
/// This is a type alias for the generic configuration with PP-DocLayout defaults.
pub type PPDocLayoutPreprocessConfig = ScaleAwareDetectorPreprocessConfig;

/// Postprocessing configuration for PP-DocLayout model.
pub type PPDocLayoutPostprocessConfig = ScaleAwareDetectorPostprocessConfig;

/// Output from PP-DocLayout model.
pub type PPDocLayoutModelOutput = ScaleAwareDetectorModelOutput;

/// PP-DocLayout document layout detection model.
///
/// This is a specialized configuration of the generic ScaleAwareDetectorModel
/// that uses ScaleFactorAndImageShape inference mode (requires both scale_factor and im_shape).
pub type PPDocLayoutModel = ScaleAwareDetectorModel;

/// Builder for PP-DocLayout model.
#[derive(Debug, Default)]
pub struct PPDocLayoutModelBuilder {
    inner: ScaleAwareDetectorModelBuilder,
}

impl PPDocLayoutModelBuilder {
    /// Creates a new PP-DocLayout builder with default settings.
    pub fn new() -> Self {
        Self {
            inner: ScaleAwareDetectorModelBuilder::pp_doclayout(),
        }
    }

    /// Sets the preprocessing configuration.
    pub fn preprocess_config(mut self, config: PPDocLayoutPreprocessConfig) -> Self {
        self.inner = self.inner.preprocess_config(config);
        self
    }

    /// Sets the image shape.
    pub fn image_shape(mut self, height: u32, width: u32) -> Self {
        self.inner = self.inner.image_shape(height, width);
        self
    }

    /// Builds the PP-DocLayout model.
    ///
    /// Auto-detects the inference mode based on the model's declared inputs:
    /// uses `ScaleFactorAndImageShape` if the model accepts `im_shape`, otherwise
    /// falls back to `ScaleFactorOnly`.
    pub fn build(self, inference: OrtInfer) -> Result<PPDocLayoutModel, OCRError> {
        let input_names = inference.input_names_from_model();
        let mode = if input_names.iter().any(|n| n == "im_shape") {
            ScaleAwareDetectorInferenceMode::ScaleFactorAndImageShape
        } else {
            ScaleAwareDetectorInferenceMode::ScaleFactorOnly
        };
        self.inner.inference_mode(mode).build(inference)
    }
}
