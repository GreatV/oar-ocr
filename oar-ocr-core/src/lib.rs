//! # OAR OCR Core
//!
//! Core types, models, and predictors for the OAR OCR library.
//!
//! This crate provides:
//! - Error handling types
//! - Domain types (layout elements, structure results, etc.)
//! - Model implementations
//! - Image processors
//! - Task predictors
//!
//! ## Modules
//!
//! * [`core`] - Core traits, error handling, and batch processing
//! * [`domain`] - Domain types like orientation helpers and prediction models
//! * [`models`] - Model adapters for different OCR tasks
//! * [`processors`] - Image processing utilities
//! * [`utils`] - Utility functions for images and tensors
//! * [`predictors`] - Task-specific predictor interfaces

// Core modules
pub mod core;
pub mod domain;
pub mod models;
pub mod predictors;
pub mod processors;
pub mod utils;

// Re-export derive macros for convenient use
pub use oar_ocr_derive::{ConfigValidator, TaskPredictorBuilder};

/// Prelude module for convenient imports.
pub mod prelude {
    // Error Handling
    pub use crate::core::{OCRError, OcrResult};

    // Domain types
    pub use crate::domain::TextRegion;
    pub use crate::domain::structure::{
        FormulaResult, LayoutElement, LayoutElementType, RegionBlock, StructureResult, TableCell,
        TableResult, TableType,
    };

    // Geometry types
    pub use crate::processors::{BoundingBox, MinAreaRect, Point};

    // Image Utilities
    pub use crate::utils::{BBoxCrop, load_image, load_images};

    // Predictors (high-level API)
    pub use crate::predictors::*;
}
