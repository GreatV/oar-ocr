//! Utility functions for the OCR pipeline.
//!
//! This module re-exports utilities from `oar-ocr-core` and adds OCR-specific
//! visualization capabilities.

// Re-export everything from oar-ocr-core utils
pub use oar_ocr_core::utils::*;

// OCR visualization (requires visualization feature)
#[cfg(feature = "visualization")]
pub mod visualization;
