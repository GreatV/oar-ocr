//! Structures and helpers for ONNX Runtime inference.
//!
//! This module provides a unified inference interface that is backend-agnostic
//! and does not make assumptions about input/output semantics.

#[cfg(feature = "ort")]
use crate::core::errors::OCRError;
#[cfg(feature = "ort")]
use ort::{session::Session, value::ValueType};
#[cfg(feature = "ort")]
use std::sync::Mutex;

#[cfg(feature = "ort")]
pub mod session;
mod tensor_input;
mod tensor_output;

// OrtInfer implementation modules
#[cfg(feature = "ort")]
#[path = "ort_infer_builders.rs"]
mod ort_infer_builders;
#[cfg(feature = "ort")]
#[path = "ort_infer_config.rs"]
mod ort_infer_config;
#[cfg(feature = "ort")]
#[path = "ort_infer_execution.rs"]
mod ort_infer_execution;

mod backend;
mod factory;
pub(crate) mod rknn;
mod rknn_infer;

pub use backend::InferenceBackend;
pub use factory::build;
#[cfg(feature = "ort")]
pub use session::load_session;
pub use tensor_input::TensorInput;
pub use tensor_output::TensorOutput;

#[cfg(all(target_arch = "aarch64", feature = "rknpu"))]
pub use rknn_infer::RknnInfer;

/// Core ONNX Runtime inference engine with support for pooling and configurable sessions.
#[cfg(feature = "ort")]
pub struct OrtInfer {
    pub(self) sessions: Vec<Mutex<Session>>,
    pub(self) next_idx: std::sync::atomic::AtomicUsize,
    pub(self) input_name: String,
    pub(self) model_path: std::path::PathBuf,
    pub(self) model_name: String,
}

#[cfg(feature = "ort")]
impl std::fmt::Debug for OrtInfer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OrtInfer")
            .field("sessions", &self.sessions.len())
            .field("input_name", &self.input_name)
            .field("model_path", &self.model_path)
            .field("model_name", &self.model_name)
            .finish()
    }
}

#[cfg(feature = "ort")]
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

    /// Returns the declared input names from the model.
    pub fn input_names_from_model(&self) -> Vec<String> {
        let Some(session_mutex) = self.sessions.first() else {
            return Vec::new();
        };
        let Ok(session_guard) = session_mutex.lock() else {
            return Vec::new();
        };
        session_guard
            .inputs()
            .iter()
            .map(|i| i.name().to_string())
            .collect()
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

#[cfg(feature = "ort")]
impl backend::InferenceBackend for OrtInfer {
    fn model_path(&self) -> &std::path::Path {
        OrtInfer::model_path(self)
    }

    fn model_name(&self) -> &str {
        OrtInfer::model_name(self)
    }

    fn input_name(&self) -> &str {
        OrtInfer::input_name(self)
    }

    fn input_names_from_model(&self) -> Vec<String> {
        OrtInfer::input_names_from_model(self)
    }

    fn primary_input_shape(&self) -> Option<Vec<i64>> {
        OrtInfer::primary_input_shape(self)
    }

    fn infer(
        &self,
        inputs: &[(&str, TensorInput<'_>)],
    ) -> Result<Vec<(String, TensorOutput)>, OCRError> {
        OrtInfer::infer(self, inputs)
    }
}

#[cfg(all(test, feature = "ort"))]
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
