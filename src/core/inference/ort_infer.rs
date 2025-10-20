//! Core ONNX Runtime inference engine with support for pooling and configurable sessions.

use crate::core::{
    batch::{Tensor2D, Tensor3D, Tensor4D},
    errors::OCRError,
};
use ort::{session::Session, value::ValueType};
use std::sync::Mutex;

#[path = "ort_infer_builders.rs"]
mod ort_infer_builders;
#[path = "ort_infer_config.rs"]
mod ort_infer_config;
#[path = "ort_infer_execution.rs"]
mod ort_infer_execution;
#[cfg(test)]
#[path = "ort_infer_tests.rs"]
mod ort_infer_tests;

pub struct OrtInfer {
    pub(super) sessions: Vec<Mutex<Session>>,
    pub(super) next_idx: std::sync::atomic::AtomicUsize,
    pub(super) input_name: String,
    pub(super) output_name: Option<String>,
    pub(super) model_path: std::path::PathBuf,
    pub(super) model_name: String,
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
    /// Attempts to retrieve the primary input tensor shape from the first session.
    ///
    /// Returns a vector of dimensions if available. Dynamic dimensions (e.g., -1) are returned as-is.
    pub fn primary_input_shape(&self) -> Option<Vec<i64>> {
        let session_mutex = self.sessions.first()?;
        let session_guard = session_mutex.lock().ok()?;
        let input = session_guard.inputs.first()?;
        match &input.input_type {
            ValueType::Tensor { shape, .. } => Some(shape.iter().copied().collect()),
            _ => None,
        }
    }
}
