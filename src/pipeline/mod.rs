//! The OCR pipeline module.
//!
//! This module provides the main OCR pipeline implementation that combines
//! multiple components to perform document orientation classification, text
//! detection, text recognition, and text line classification.

mod oarocr;

// Re-export the main OCR pipeline components for easier access
pub use oarocr::{OAROCR, OAROCRBuilder, OAROCRConfig, OAROCRResult};
