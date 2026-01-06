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
pub mod text_region;

pub use orientation::*;
pub use predictions::*;
pub use text_region::TextRegion;
// Note: structure module is not re-exported with * to keep explicit separation.
// Tasks exports: LayoutDetectionElement, TableCellDetection (detection outputs)
// Structure exports: LayoutElement, TableCell (enriched results)
// Use domain::structure::* explicitly when needed.
pub use tasks::*;
