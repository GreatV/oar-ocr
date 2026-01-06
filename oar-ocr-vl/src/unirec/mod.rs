//! UniRec Vision-Language model implementation using Candle.
//!
//! UniRec is a unified text and formula recognition model that combines:
//! - FocalSVTR visual encoder (Focal Modulation Networks)
//! - M2M100 decoder (multilingual translation decoder architecture)
//!
//! This module provides a native Rust implementation for efficient inference.

mod config;
mod decoder;
mod encoder;
mod model;

pub use config::UniRecConfig;
pub use model::UniRec;
