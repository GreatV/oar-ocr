//! Configuration management for the OCR pipeline.
//!
//! This module provides configuration types, validation traits, and utilities
//! for managing OCR pipeline configurations.

pub mod builder;
pub mod derive;
pub mod errors;
pub mod model_input;
pub mod onnx;
pub mod parallel;
pub mod transform;

// Re-export commonly used types
pub use builder::ModelInferenceConfig;
pub use errors::{ConfigDefaults, ConfigError, ConfigValidator, ConfigValidatorExt};
pub use model_input::{ChannelOrder, Dim, InputShape, ModelInputConfig, NormalizationConfig};
pub use onnx::*;
pub use parallel::ParallelPolicy;
pub use transform::{TransformConfig, TransformRegistry, TransformType};
