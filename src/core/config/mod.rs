//! Configuration management for the OCR pipeline.
//!
//! This module provides configuration types, validation traits, and utilities
//! for managing OCR pipeline configurations.

pub mod builder;
pub mod derive;
pub mod errors;
pub mod onnx;
pub mod transform;

// Re-export commonly used types
pub use builder::CommonBuilderConfig;
pub use errors::{ConfigError, ConfigValidator, ConfigValidatorExt};
pub use onnx::*;
pub use transform::{TransformConfig, TransformRegistry, TransformType};
