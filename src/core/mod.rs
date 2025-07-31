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

pub use crate::utils::{
    create_rgb_image, dynamic_to_gray, dynamic_to_rgb, load_image, load_images_batch,
};
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
pub use traits::{
    BasePredictor, ImageReader, PipelineExecutor, Predictor, PredictorBuilder, PredictorConfig,
    Sampler,
};

/// Initializes the tracing subscriber for logging.
///
/// This function sets up the tracing subscriber with environment filter and formatting layer.
/// It's typically called at the start of an application to enable logging.
pub fn init_tracing() {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer())
        .init();
}
