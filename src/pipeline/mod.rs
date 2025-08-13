//! The OCR pipeline module.
//!
//! This module provides the main OCR pipeline implementation that combines
//! multiple components to perform document orientation classification, text
//! detection, text recognition, and text line classification.

mod config;
pub mod oarocr;
pub mod stages;

// Re-export the main OCR pipeline components for easier access
pub use config::{ConfigFormat, ConfigLoader};
pub use oarocr::{
    ErrorMetrics, ExtensibleOAROCR, ExtensibleOAROCRBuilder, ImageProcessor, OAROCR, OAROCRBuilder,
    OAROCRConfig, OAROCRResult, TextRegion, configure_thread_pool_once,
};
pub use stages::{
    CroppingStageProcessor, OrientationStageProcessor, RecognitionStageProcessor, StageMetrics,
    StageResult,
};
