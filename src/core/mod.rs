//! The core module of the OCR pipeline.
//!
//! This module contains the fundamental components of the OCR pipeline, including:
//! - Batch processing utilities
//! - Configuration management
//! - Constants used throughout the pipeline
//! - Error handling
//! - Inference engine integration
//! - Prediction result types
//! - Traits defining interfaces for various components
//!
//! It also provides re-exports of commonly used types and functions for convenience.

pub mod batch;
pub mod config;
pub mod constants;
pub mod errors;
pub mod inference;
#[macro_use]
pub mod macros;
pub mod predictions;
pub mod traits;

// Image utilities are now available directly from oar_ocr::utils
// pub use crate::utils::{
//     create_rgb_image, dynamic_to_gray, dynamic_to_rgb, load_image, load_images_batch,
// };
pub use batch::{BatchData, BatchSampler, Tensor1D, Tensor2D, Tensor3D, Tensor4D, ToBatch};
pub use config::{
    CommonBuilderConfig, ConfigError, TransformConfig, TransformRegistry, TransformType,
};
pub use constants::*;
pub use errors::{OCRError, ProcessingStage};
pub use inference::{DefaultImageReader, OrtInfer, load_session};
pub use predictions::{
    IntoOwnedPrediction, IntoPrediction, OwnedPredictionResult, PipelineStats, PredictionResult,
};
pub use traits::{BasePredictor, ImageReader, PredictorBuilder, PredictorConfig, Sampler};

// init_tracing function has been moved to oar_ocr::utils::init_tracing
