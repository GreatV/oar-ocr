//! Vision-Language modules (optional).
//!
//! This module contains integrations that are not part of the core ONNX-based
//! OCR pipeline (e.g. large VLMs). These are behind feature flags to keep
//! default builds lightweight.

#[cfg(feature = "paddleocr-vl")]
pub mod config;

#[cfg(feature = "paddleocr-vl")]
pub mod model;

#[cfg(feature = "paddleocr-vl")]
pub mod processing;

#[cfg(feature = "paddleocr-vl")]
pub mod paddleocr_vl;

#[cfg(feature = "paddleocr-vl")]
pub mod paddleocr_vl_doc_parser;

#[cfg(feature = "paddleocr-vl")]
pub use paddleocr_vl::*;

#[cfg(feature = "paddleocr-vl")]
pub use paddleocr_vl_doc_parser::*;
