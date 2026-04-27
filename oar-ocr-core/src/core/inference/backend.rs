//! Backend-agnostic inference interface.
//!
//! This trait isolates the rest of the codebase from the concrete inference
//! engine (ONNX Runtime today; RKNN, etc. in the future). All public model
//! types hold a `Box<dyn InferenceBackend>` so they are independent of the
//! underlying runtime.

use std::path::Path;

use crate::core::errors::OCRError;
use crate::core::inference::{TensorInput, TensorOutput};

/// A runtime-selectable inference engine.
///
/// Implementations must be safe to call from multiple threads (`Send + Sync`)
/// because models are routinely shared across worker threads in the OCR
/// pipeline. They must also support `Debug` for logging.
pub trait InferenceBackend: Send + Sync + std::fmt::Debug {
    /// Path to the model file backing this engine.
    fn model_path(&self) -> &Path;

    /// Logical name of the model (for diagnostics / errors).
    fn model_name(&self) -> &str;

    /// Default input tensor name to use when the caller does not supply one.
    fn input_name(&self) -> &str;

    /// All declared input tensor names from the model metadata.
    fn input_names_from_model(&self) -> Vec<String>;

    /// Shape of the primary input tensor as declared by the model, if known.
    /// Dynamic dimensions are reported as `-1`.
    fn primary_input_shape(&self) -> Option<Vec<i64>>;

    /// Run inference on a set of named inputs and return all model outputs.
    fn infer(
        &self,
        inputs: &[(&str, TensorInput<'_>)],
    ) -> Result<Vec<(String, TensorOutput)>, OCRError>;
}
