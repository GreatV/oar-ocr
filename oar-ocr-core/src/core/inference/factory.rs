//! Factory for selecting an inference backend.
//!
//! Today only the ONNX Runtime (`OrtInfer`) backend is wired up. New backends
//! (e.g. RKNN for Rockchip NPUs) will plug in here without forcing call sites
//! to change.

use std::path::Path;

use crate::core::config::ModelInferenceConfig;
use crate::core::errors::OCRError;
use crate::core::inference::{InferenceBackend, OrtInfer};

/// Build an inference backend for the given model file.
///
/// `common` carries cross-backend configuration (model name, session options,
/// execution providers); `input_name` overrides the default input tensor name
/// when the model exposes a non-standard name.
pub fn build(
    common: Option<&ModelInferenceConfig>,
    model_path: impl AsRef<Path>,
    input_name: Option<&str>,
) -> Result<Box<dyn InferenceBackend>, OCRError> {
    let infer = match common {
        Some(cfg) => OrtInfer::from_config(cfg, model_path, input_name)?,
        None => OrtInfer::new(model_path, input_name)?,
    };
    Ok(Box::new(infer))
}
