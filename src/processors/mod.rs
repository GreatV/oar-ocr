//! Image processing utilities for OCR systems.
//!
//! This module provides a collection of image processing functions and utilities
//! specifically designed for OCR (Optical Character Recognition) systems. It includes
//! functionality for image resizing, normalization, geometric operations, text decoding,
//! and post-processing of OCR results.
//!
//! # Modules
//!
//! * `decode` - Text decoding utilities for converting model predictions to readable text
//! * `geometry` - Geometric primitives and algorithms for OCR processing
//! * `normalization` - Image normalization utilities for preparing images for OCR models
//! * `ocr_resize` - OCR-specific image resizing functionality
//! * `postprocess` - Post-processing utilities for OCR pipeline outputs
//! * `resize` - General image resizing utilities for OCR preprocessing
//! * `types` - Type definitions used across the processors module
//! * `utils` - Additional utility functions for image processing

mod decode;
mod geometry;
mod normalization;
mod ocr_resize;
mod postprocess;
mod resize;
mod types;
mod utils;

pub use decode::*;
pub use geometry::*;
pub use normalization::*;
pub use ocr_resize::*;
pub use postprocess::*;
pub use resize::*;
pub use types::*;
pub use utils::*;
