//! Recognition model adapters.
//!
//! This module contains adapters for text recognition models.

pub mod crnn_adapter;

pub use crnn_adapter::{CRNNTextRecognitionAdapter, CRNNTextRecognitionAdapterBuilder};
