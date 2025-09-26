//! Core ONNX Runtime inference engine with support for pooling and configurable sessions.

use crate::core::{
    batch::{Tensor2D, Tensor3D, Tensor4D},
    errors::OCRError,
};
use ort::session::Session;
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
