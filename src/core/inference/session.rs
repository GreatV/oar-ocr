//! Helpers for working directly with ONNX Runtime sessions.

use crate::core::errors::OCRError;
use ort::session::Session;
use std::path::Path;

pub fn load_session(model_path: impl AsRef<Path>) -> Result<Session, OCRError> {
    let path = model_path.as_ref();
    let session = Session::builder()
        .and_then(|b| b.commit_from_file(path))
        .map_err(|e| {
            OCRError::model_load_error(
                path,
                "failed to create ONNX session",
                Some("verify model file exists and is readable"),
                Some(e),
            )
        })?;
    Ok(session)
}
