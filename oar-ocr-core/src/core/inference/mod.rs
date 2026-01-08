//! Structures and helpers for ONNX Runtime inference.
//!
//! This module centralizes the low level inference engine along with thin wrappers
//! that adapt it to the `InferenceEngine` trait used across the pipeline.

use crate::core::{
    batch::{Tensor2D, Tensor3D, Tensor4D},
    errors::OCRError,
};
use ort::{session::Session, value::ValueType};
use std::sync::Mutex;

pub mod session;
pub mod wrappers;

// OrtInfer implementation modules
#[path = "ort_infer_builders.rs"]
mod ort_infer_builders;
#[path = "ort_infer_config.rs"]
mod ort_infer_config;
#[path = "ort_infer_execution.rs"]
mod ort_infer_execution;

pub use session::load_session;
pub use wrappers::{OrtInfer2D, OrtInfer3D, OrtInfer4D};

/// Core ONNX Runtime inference engine with support for pooling and configurable sessions.
pub struct OrtInfer {
    pub(self) sessions: Vec<Mutex<Session>>,
    pub(self) next_idx: std::sync::atomic::AtomicUsize,
    pub(self) input_name: String,
    pub(self) output_name: Option<String>,
    pub(self) model_path: std::path::PathBuf,
    pub(self) model_name: String,
}

impl std::fmt::Debug for OrtInfer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrtInfer")
            .field("sessions", &self.sessions.len())
            .field("input_name", &self.input_name)
            .field("output_name", &self.output_name)
            .field("model_path", &self.model_path)
            .field("model_name", &self.model_name)
            .finish()
    }
}

impl OrtInfer {
    /// Returns the input tensor name.
    pub fn input_name(&self) -> &str {
        &self.input_name
    }

    /// Gets a session from the pool.
    pub fn get_session(&self, idx: usize) -> Result<std::sync::MutexGuard<'_, Session>, OCRError> {
        self.sessions[idx % self.sessions.len()]
            .lock()
            .map_err(|_| OCRError::ConfigError {
                message: "Failed to acquire session lock".to_string(),
            })
    }

    /// Attempts to retrieve the primary input tensor shape from the first session.
    ///
    /// Returns a vector of dimensions if available. Dynamic dimensions (e.g., -1) are returned as-is.
    pub fn primary_input_shape(&self) -> Option<Vec<i64>> {
        let session_mutex = self.sessions.first()?;
        let session_guard = session_mutex.lock().ok()?;
        let input = session_guard.inputs().first()?;
        match input.dtype() {
            ValueType::Tensor { shape, .. } => Some(shape.iter().copied().collect()),
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::ModelInferenceConfig;

    #[test]
    fn test_from_config_with_ort_session() {
        let common = ModelInferenceConfig::new();
        let result = OrtInfer::from_config(&common, "dummy_path.onnx", None);
        assert!(result.is_err()); // File doesn't exist
    }
}
