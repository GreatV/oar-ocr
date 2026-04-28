//! Factory for selecting an inference backend.
//!
//! The factory dispatches on the model file extension:
//! * `*.rknn` → RKNN runtime (RK3588 NPU). Requires `feature = "rknpu"` and
//!   `target_arch = "aarch64"`; otherwise returns a config error.
//! * everything else → ONNX Runtime (`OrtInfer`) when `feature = "ort"` is on.

use std::path::Path;

use crate::core::config::ModelInferenceConfig;
use crate::core::errors::OCRError;
use crate::core::inference::InferenceBackend;

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
    let path = model_path.as_ref();

    if is_rknn_model(path) {
        return build_rknn(common, path, input_name);
    }

    build_ort(common, path, input_name)
}

fn is_rknn_model(path: &Path) -> bool {
    path.extension()
        .and_then(|e| e.to_str())
        .map(|e| e.eq_ignore_ascii_case("rknn"))
        .unwrap_or(false)
}

#[cfg(all(target_arch = "aarch64", feature = "rknpu"))]
fn build_rknn(
    common: Option<&ModelInferenceConfig>,
    path: &Path,
    input_name: Option<&str>,
) -> Result<Box<dyn InferenceBackend>, OCRError> {
    use crate::core::inference::RknnInfer;
    let model_name = common.and_then(|c| c.model_name.clone());
    let rknn_config = common.and_then(|c| c.rknn_session.as_ref());
    let infer = RknnInfer::from_file(path, input_name, model_name, rknn_config)?;
    Ok(Box::new(infer))
}

#[cfg(not(all(target_arch = "aarch64", feature = "rknpu")))]
fn build_rknn(
    _common: Option<&ModelInferenceConfig>,
    path: &Path,
    _input_name: Option<&str>,
) -> Result<Box<dyn InferenceBackend>, OCRError> {
    Err(OCRError::ConfigError {
        message: format!(
            "Model '{}' is an RKNN file but this build does not support the RKNN backend. \
             Build with `--features rknpu` on an aarch64 target.",
            path.display()
        ),
    })
}

#[cfg(feature = "ort")]
fn build_ort(
    common: Option<&ModelInferenceConfig>,
    path: &Path,
    input_name: Option<&str>,
) -> Result<Box<dyn InferenceBackend>, OCRError> {
    use crate::core::inference::OrtInfer;

    let infer = match common {
        Some(cfg) => OrtInfer::from_config(cfg, path, input_name)?,
        None => OrtInfer::new(path, input_name)?,
    };
    Ok(Box::new(infer))
}

#[cfg(not(feature = "ort"))]
fn build_ort(
    _common: Option<&ModelInferenceConfig>,
    path: &Path,
    _input_name: Option<&str>,
) -> Result<Box<dyn InferenceBackend>, OCRError> {
    Err(OCRError::ConfigError {
        message: format!(
            "Model '{}' is not an RKNN file and this build does not include the ONNX Runtime backend. \
             Build with default features or enable the `ort` feature.",
            path.display()
        ),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn is_rknn_model_matches_extension_case_insensitively() {
        assert!(is_rknn_model(Path::new("model.rknn")));
        assert!(is_rknn_model(Path::new("model.RKNN")));
        assert!(is_rknn_model(Path::new("archive.det.rKnN")));
        assert!(!is_rknn_model(Path::new("model.onnx")));
        assert!(!is_rknn_model(Path::new("rknn")));
        assert!(!is_rknn_model(Path::new("model.rknn.backup")));
    }

    #[cfg(not(all(target_arch = "aarch64", feature = "rknpu")))]
    #[test]
    fn rknn_model_returns_config_error_when_backend_is_not_available() {
        let err = build(None, "detector.rknn", None).expect_err("unsupported RKNN build");

        match err {
            OCRError::ConfigError { message } => {
                assert!(message.contains("does not support the RKNN backend"));
                assert!(message.contains("--features rknpu"));
                assert!(message.contains("aarch64"));
            }
            other => panic!("expected ConfigError, got {other:?}"),
        }
    }
}
