//! Detection model adapters.
//!
//! This module contains adapters for text detection models.

pub mod db_adapter;

pub use db_adapter::{DBTextDetectionAdapter, DBTextDetectionAdapterBuilder};
