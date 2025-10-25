use super::*;
use crate::core::config::CommonBuilderConfig;
use ort::logging::LogLevel;
use ort::session::Session;
use std::path::Path;
use std::sync::Mutex;

impl OrtInfer {
    /// Creates a new OrtInfer instance with default ONNX Runtime settings and a single session.
    pub fn new(model_path: impl AsRef<Path>, input_name: Option<&str>) -> Result<Self, OCRError> {
        let path = model_path.as_ref();
        let session = Session::builder()?
            .with_log_level(LogLevel::Error)?
            .commit_from_file(path)
            .map_err(|e| {
                OCRError::model_load_error(
                    path,
                    "failed to create ONNX session",
                    Some("verify model path and compatibility with selected execution providers"),
                    Some(e),
                )
            })?;
        let model_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown_model")
            .to_string();

        Ok(OrtInfer {
            sessions: vec![Mutex::new(session)],
            next_idx: std::sync::atomic::AtomicUsize::new(0),
            input_name: input_name.unwrap_or("x").to_string(),
            output_name: None,
            model_path: path.to_path_buf(),
            model_name,
        })
    }

    /// Creates a new OrtInfer instance from CommonBuilderConfig, applying ORT session
    /// configuration and constructing a session pool for concurrent predictions.
    pub fn from_common(
        common: &CommonBuilderConfig,
        model_path: impl AsRef<Path>,
        input_name: Option<&str>,
    ) -> Result<Self, OCRError> {
        let path = model_path.as_ref();
        let pool_size = common.session_pool_size.unwrap_or(1).max(1);
        let mut sessions = Vec::with_capacity(pool_size);
        for _ in 0..pool_size {
            let builder = Session::builder()?;
            let builder = if let Some(cfg) = &common.ort_session {
                Self::apply_ort_config(builder, cfg)?
            } else {
                // Set default log level to Error to suppress ORT logs
                builder.with_log_level(LogLevel::Error)?
            };
            let session = builder.commit_from_file(path).map_err(|e| {
                OCRError::model_load_error(
                    path,
                    "failed to create ONNX session",
                    Some("check device/EP configuration and model file"),
                    Some(e),
                )
            })?;
            sessions.push(Mutex::new(session));
        }

        let model_name = common
            .model_name
            .clone()
            .or_else(|| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| "unknown_model".to_string());

        Ok(OrtInfer {
            sessions,
            next_idx: std::sync::atomic::AtomicUsize::new(0),
            input_name: input_name.unwrap_or("x").to_string(),
            output_name: None,
            model_path: path.to_path_buf(),
            model_name,
        })
    }

    /// Creates a new OrtInfer instance from CommonBuilderConfig with automatic input name detection,
    /// applying ORT session configuration and constructing a session pool for concurrent predictions.
    ///
    /// This method combines the functionality of `from_common` and `with_auto_input_name` to respect
    /// ORT configuration while automatically detecting the input tensor name.
    pub fn from_common_with_auto_input(
        common: &CommonBuilderConfig,
        model_path: impl AsRef<Path>,
    ) -> Result<Self, OCRError> {
        let path = model_path.as_ref();
        let pool_size = common.session_pool_size.unwrap_or(1).max(1);
        let mut sessions = Vec::with_capacity(pool_size);

        // Create the first session to detect input name
        let builder = Session::builder()?;
        let builder = if let Some(cfg) = &common.ort_session {
            Self::apply_ort_config(builder, cfg)?
        } else {
            // Set default log level to Error to suppress ORT logs
            builder.with_log_level(LogLevel::Error)?
        };
        let first_session = builder.commit_from_file(path).map_err(|e| {
            OCRError::model_load_error(
                path,
                "failed to create ONNX session",
                Some("ensure model is compatible and file exists"),
                Some(e),
            )
        })?;

        // Auto-detect input name from the first session
        let common_names = ["x", "input", "images", "data", "image"];
        let available_inputs: Vec<String> = first_session
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect();

        let input_name = common_names
            .iter()
            .find_map(|&name| {
                available_inputs
                    .iter()
                    .any(|input| input == name)
                    .then(|| name.to_string())
            })
            .or_else(|| available_inputs.first().cloned())
            .unwrap_or_else(|| "x".to_string());

        sessions.push(Mutex::new(first_session));

        // Create remaining sessions with the same configuration
        for _ in 1..pool_size {
            let builder = Session::builder()?;
            let builder = if let Some(cfg) = &common.ort_session {
                Self::apply_ort_config(builder, cfg)?
            } else {
                // Set default log level to Error to suppress ORT logs
                builder.with_log_level(LogLevel::Error)?
            };
            let session = builder.commit_from_file(path).map_err(|e| {
                OCRError::model_load_error(
                    path,
                    "failed to create ONNX session",
                    Some("ensure model is compatible and file exists"),
                    Some(e),
                )
            })?;
            sessions.push(Mutex::new(session));
        }

        let model_name = common
            .model_name
            .clone()
            .or_else(|| {
                path.file_stem()
                    .and_then(|s| s.to_str())
                    .map(|s| s.to_string())
            })
            .unwrap_or_else(|| "unknown_model".to_string());

        Ok(OrtInfer {
            sessions,
            next_idx: std::sync::atomic::AtomicUsize::new(0),
            input_name,
            output_name: None,
            model_path: path.to_path_buf(),
            model_name,
        })
    }

    /// Creates a new OrtInfer instance with a specified output tensor name.
    pub fn with_output_name(
        model_path: impl AsRef<Path>,
        input_name: Option<&str>,
        output_name: Option<&str>,
    ) -> Result<Self, OCRError> {
        let path = model_path.as_ref();
        let session = Session::builder()?
            .with_log_level(LogLevel::Error)?
            .commit_from_file(path)
            .map_err(|e| {
                OCRError::model_load_error(
                    path,
                    "failed to create ONNX session",
                    Some("verify model path and compatibility"),
                    Some(e),
                )
            })?;
        let model_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown_model")
            .to_string();

        Ok(OrtInfer {
            sessions: vec![Mutex::new(session)],
            next_idx: std::sync::atomic::AtomicUsize::new(0),
            input_name: input_name.unwrap_or("x").to_string(),
            output_name: output_name.map(|s| s.to_string()),
            model_path: path.to_path_buf(),
            model_name,
        })
    }

    /// Creates a new OrtInfer instance with an automatically detected input tensor name.
    pub fn with_auto_input_name(model_path: impl AsRef<Path>) -> Result<Self, OCRError> {
        let path = model_path.as_ref();
        let session = Session::builder()?
            .with_log_level(LogLevel::Error)?
            .commit_from_file(path)
            .map_err(|e| {
                OCRError::model_load_error(
                    path,
                    "failed to create ONNX session",
                    Some("verify model path and compatibility"),
                    Some(e),
                )
            })?;
        let model_name = path
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown_model")
            .to_string();

        let common_names = ["x", "input", "images", "data", "image"];

        let available_inputs: Vec<String> = session
            .inputs
            .iter()
            .map(|input| input.name.clone())
            .collect();

        let input_name = common_names
            .iter()
            .find_map(|&name| {
                available_inputs
                    .iter()
                    .any(|input| input == name)
                    .then(|| name.to_string())
            })
            .or_else(|| available_inputs.first().cloned())
            .unwrap_or_else(|| "x".to_string());

        Ok(OrtInfer {
            sessions: vec![Mutex::new(session)],
            next_idx: std::sync::atomic::AtomicUsize::new(0),
            input_name,
            output_name: None,
            model_path: path.to_path_buf(),
            model_name,
        })
    }
}
