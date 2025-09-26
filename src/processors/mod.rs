//! Image processing utilities for OCR systems.
//!
//! This module provides a collection of image processing functions and utilities
//! specifically designed for OCR (Optical Character Recognition) systems. It includes
//! functionality for image resizing, normalization, geometric operations, text decoding,
//! and post-processing of OCR results.
//!
//! # Modules
//!
//! * `aspect_ratio_bucketing` - Aspect ratio bucketing for efficient batch processing
//! * `decode` - Text decoding utilities for converting model predictions to readable text
//! * `geometry` - Geometric primitives and algorithms for OCR processing
//! * `normalization` - Image normalization utilities for preparing images for OCR models
//! * `postprocess` - Post-processing utilities for OCR pipeline outputs
//! * `resize` - Resizing helpers for detection and recognition stages
//! * `types` - Type definitions used across the processors module

mod aspect_ratio_bucketing;
mod decode;
mod geometry;
mod normalization;
mod postprocess;
mod resize;
pub mod types;

pub use crate::utils::{Crop, Topk, TopkResult};
pub use aspect_ratio_bucketing::*;
pub use decode::*;
pub use geometry::*;
pub use normalization::*;
pub use postprocess::*;
pub use resize::*;
pub use types::*;
