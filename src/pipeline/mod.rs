//! The OCR pipeline module.
//!
//! This module provides the task graph-based OCR pipeline implementation that combines
//! multiple model adapters to perform document orientation classification, text
//! detection, text recognition, and text line classification.
//!
//! # Task Graph Architecture
//!
//! The pipeline uses a flexible task graph architecture that allows:
//! - Declarative pipeline configuration via JSON or code
//! - Dynamic model swapping without code changes
//! - Edge processors for data transformation between tasks
//! - Comprehensive validation of pipeline structure
//!
//! See [`TaskGraphBuilder`] and [`TaskGraphConfig`] for details.

pub mod oarocr;

// Re-export the main OCR pipeline components for easier access
pub use oarocr::{
    EdgeProcessor, EdgeProcessorConfig, EdgeProcessorFactory, ErrorMetrics, ImageProcessor,
    ModelBinding, OAROCRResult, TaskGraphBuilder, TaskGraphConfig, TaskNode, TextRegion,
};
