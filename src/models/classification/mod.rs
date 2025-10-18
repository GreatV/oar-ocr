//! Classification model adapters.
//!
//! This module contains adapters for classification models (orientation, etc.).

pub mod doc_orientation_adapter;
pub mod text_line_orientation_adapter;

pub use doc_orientation_adapter::{DocOrientationAdapter, DocOrientationAdapterBuilder};
pub use text_line_orientation_adapter::{
    TextLineOrientationAdapter, TextLineOrientationAdapterBuilder,
};
