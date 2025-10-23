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
//! * `db_postprocess` - DB detection post-processing
//! * `uvdoc_postprocess` - UVDoc rectification post-processing
//! * `formula_preprocess` - Formula recognition preprocessing
//! * `layout_postprocess` - Layout detection post-processing
//! * `resize_detection` - Resizing for detection models
//! * `resize_recognition` - Resizing for recognition models
//! * `types` - Type definitions used across the processors module

mod aspect_ratio_bucketing;
pub mod db_postprocess;
mod decode;
pub mod formula_preprocess;
mod geometry;
pub mod layout_postprocess;
mod normalization;
pub mod resize_detection;
pub mod resize_recognition;
pub mod types;
pub mod unimernet_preprocess;
pub mod uvdoc_postprocess;

pub use crate::utils::{Crop, Topk, TopkResult};
pub use aspect_ratio_bucketing::*;
pub use db_postprocess::*;
pub use decode::*;
pub use formula_preprocess::{FormulaPreprocessParams, FormulaPreprocessor, normalize_latex};
pub use geometry::*;
pub use layout_postprocess::*;
pub use normalization::*;
pub use resize_detection::*;
pub use resize_recognition::*;
pub use types::*;
pub use unimernet_preprocess::{UniMERNetPreprocessParams, UniMERNetPreprocessor};
pub use uvdoc_postprocess::*;
