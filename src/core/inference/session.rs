//! Helpers for working directly with ONNX Runtime sessions.

use crate::core::errors::OCRError;
use ort::logging::LogLevel;
use ort::session::{Session, builder::SessionBuilder};
use std::path::Path;

const SESSION_CREATION_FAILURE: &str = "failed to create ONNX session";

/// Loads a session with default logging configuration.
pub fn load_session(model_path: impl AsRef<Path>) -> Result<Session, OCRError> {
    load_session_with(
        model_path,
        |builder| builder.with_log_level(LogLevel::Error),
        Some("verify model file exists and is readable"),
    )
}

/// Builds a session using a caller-provided builder configuration.
pub(crate) fn load_session_with<F>(
    model_path: impl AsRef<Path>,
    configure_builder: F,
    suggestion: Option<&str>,
) -> Result<Session, OCRError>
where
    F: FnOnce(SessionBuilder) -> Result<SessionBuilder, ort::Error>,
{
    let path = model_path.as_ref();
    let builder = Session::builder()?;
    let builder = configure_builder(builder)?;
    let session = builder.commit_from_file(path).map_err(|e| {
        OCRError::model_load_error(path, SESSION_CREATION_FAILURE, suggestion, Some(e))
    })?;
    Ok(session)
}
