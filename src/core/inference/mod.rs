//! Structures and helpers for ONNX Runtime inference.
//!
//! This module centralizes the low level inference engine along with thin wrappers
//! that adapt it to the `InferenceEngine` trait used across the pipeline.

pub mod image_reader;
pub mod ort_infer;
pub mod session;
pub mod wrappers;

pub use image_reader::DefaultImageReader;
pub use ort_infer::OrtInfer;
pub use session::load_session;
pub use wrappers::{OrtInfer2D, OrtInfer3D, OrtInfer4D};
