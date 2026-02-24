use super::{session, *};
use crate::core::config::ModelInferenceConfig;
use ort::logging::LogLevel;
use std::path::Path;
use std::sync::Mutex;

impl OrtInfer {
    /// Creates a new OrtInfer instance with default ONNX Runtime settings and a single session.
    pub fn new(model_path: impl AsRef<Path>, input_name: Option<&str>) -> Result<Self, OCRError> {
        let path = model_path.as_ref();
        let session = session::load_session_with(
            path,
            |builder| builder.with_log_level(LogLevel::Error),
            Some("verify model path and compatibility with selected execution providers"),
        )?;
        let model_name = "unknown_model".to_string();

        Ok(OrtInfer {
            sessions: vec![Mutex::new(session)],
            next_idx: std::sync::atomic::AtomicUsize::new(0),
            input_name: input_name.unwrap_or("x").to_string(),
            model_path: path.to_path_buf(),
            model_name,
        })
    }

    /// Creates a new OrtInfer instance from ModelInferenceConfig, applying ORT session
    /// configuration.
    pub fn from_config(
        common: &ModelInferenceConfig,
        model_path: impl AsRef<Path>,
        input_name: Option<&str>,
    ) -> Result<Self, OCRError> {
        let path = model_path.as_ref();
        let session = session::load_session_with(
            path,
            |builder| {
                if let Some(cfg) = &common.ort_session {
                    Self::apply_ort_config(builder, cfg)
                } else {
                    builder.with_log_level(LogLevel::Error)
                }
            },
            Some("check device/EP configuration and model file"),
        )?;

        let model_name = common
            .model_name
            .clone()
            .unwrap_or_else(|| "unknown_model".to_string());

        Ok(OrtInfer {
            sessions: vec![Mutex::new(session)],
            next_idx: std::sync::atomic::AtomicUsize::new(0),
            input_name: input_name.unwrap_or("x").to_string(),
            model_path: path.to_path_buf(),
            model_name,
        })
    }
}
