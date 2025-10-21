//! Detection model adapters.
//!
//! This module contains adapters for text detection models.

pub mod db_adapter;
pub mod seal_adapter;

pub use db_adapter::{DBTextDetectionAdapter, DBTextDetectionAdapterBuilder};
pub use seal_adapter::{SealTextDetectionAdapter, SealTextDetectionAdapterBuilder};
