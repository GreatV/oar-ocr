//! Domain-level structures shared across the OCR pipeline.
//!
//! This module groups the higher-level prediction and orientation types that
//! represent OCR-specific concepts used throughout the crate, as well as
//! task-level adapters that combine models with task configurations.

pub mod adapters;
pub mod orientation;
pub mod predictions;
pub mod structure;
pub mod tasks;

pub use orientation::*;
pub use predictions::*;
// Note: structure module is not re-exported with * to avoid naming conflicts
// with tasks module (LayoutElement, TableCell). Use domain::structure::* explicitly.
pub use tasks::*;
