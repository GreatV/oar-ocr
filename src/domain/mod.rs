//! Domain-level structures shared across the OCR pipeline.
//!
//! This module groups the higher-level prediction and orientation types that
//! represent OCR-specific concepts used throughout the crate.

pub mod orientation;
pub mod predictions;
pub mod tasks;

pub use orientation::*;
pub use predictions::*;
pub use tasks::*;
