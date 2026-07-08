//! Helpers for working directly with ONNX Runtime sessions.

use crate::core::errors::OCRError;
use crate::core::inference::ModelSource;
use ort::logging::LogLevel;
use ort::session::{Session, builder::SessionBuilder};
use std::path::Path;

const SESSION_CREATION_FAILURE: &str = "failed to create ONNX session";

/// Loads a session with default logging configuration.
pub fn load_session(model_source: impl Into<ModelSource>) -> Result<Session, OCRError> {
    load_session_with(
        model_source,
        |builder| Ok(builder.with_log_level(LogLevel::Error)?),
        Some("verify model file exists and is readable"),
    )
}

/// Builds a session using a caller-provided builder configuration.
pub(crate) fn load_session_with<F>(
    model_source: impl Into<ModelSource>,
    configure_builder: F,
    suggestion: Option<&str>,
) -> Result<Session, OCRError>
where
    F: FnOnce(SessionBuilder) -> Result<SessionBuilder, ort::Error>,
{
    let source = model_source.into();
    let builder = Session::builder()?;
    let mut builder = configure_builder(builder)?;
    let session = match &source {
        ModelSource::Path(path) => builder.commit_from_file(path).map_err(|e| {
            OCRError::model_load_error(path, SESSION_CREATION_FAILURE, suggestion, Some(e))
        })?,
        ModelSource::Memory(bytes) => builder.commit_from_memory(bytes).map_err(|e| {
            OCRError::model_load_error(
                Path::new("<in-memory model>"),
                SESSION_CREATION_FAILURE,
                suggestion,
                Some(e),
            )
        })?,
    };
    Ok(session)
}
