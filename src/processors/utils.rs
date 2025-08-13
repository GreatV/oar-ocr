//! Utility functions for image processing operations.
//!
//! This module provides various utility functions for image processing, including:
//! - Image cropping operations with different modes
//! - Top-k classification result processing
//! - Document transformation post-processing
//!
//! The module is organized into several components:
//! - `image_utils`: Helper functions for basic image operations
//! - `crop`: Image cropping functionality with different modes
//! - `topk`: Top-k classification result processing
//! - `doctr_postprocess`: Document transformation post-processing

pub mod crop;
pub mod doctr_postprocess;
pub mod image_utils;
pub mod topk;

// Re-export public types for convenience
pub use crop::Crop;
pub use doctr_postprocess::DocTrPostProcess;
pub use topk::{Topk, TopkResult};
