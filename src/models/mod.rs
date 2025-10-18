//! Model adapters for the OCR pipeline.
//!
//! This module contains adapters that wrap model implementations
//! to conform to the task-based architecture.
//!
//! All adapters have been refactored to directly use ONNX Runtime
//! without dependencies on the old predictor module.

pub mod classification;
pub mod detection;
pub mod recognition;
pub mod rectification;

pub use classification::*;
pub use detection::*;
pub use recognition::*;
pub use rectification::*;
