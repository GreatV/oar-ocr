//! Post-processing utilities for OCR predictors.
//!
//! This module groups the different post-processing implementations used across
//! the OCR pipeline, including detection post-processing and document
//! transformation helpers.

pub mod db;
pub mod doctr;

pub use db::*;
pub use doctr::*;
