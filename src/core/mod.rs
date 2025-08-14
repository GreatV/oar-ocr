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
pub mod dynamic_batch;
pub mod errors;
pub mod granular_traits;
pub mod inference;
#[macro_use]
pub mod macros;
pub mod orientation;
pub mod predictions;
pub mod traits;

// Image utilities are now available directly from oar_ocr::utils
// pub use crate::utils::{
//     create_rgb_image, dynamic_to_gray, dynamic_to_rgb, load_image, load_images_batch,
// };
pub use batch::{BatchData, BatchSampler, Tensor1D, Tensor2D, Tensor3D, Tensor4D, ToBatch};
pub use config::{
    CommonBuilderConfig, ConfigError, ConfigValidator, ConfigValidatorExt, TransformConfig,
    TransformRegistry, TransformType,
};
pub use constants::*;
pub use dynamic_batch::{
    BatchPerformanceMetrics, CompatibleBatch, CrossImageBatch, CrossImageItem,
    DefaultDynamicBatcher, DynamicBatchConfig, DynamicBatchResult, DynamicBatcher, MemoryStrategy,
    PaddingStrategy, ShapeCompatibilityStrategy,
};
pub use errors::{OCRError, ProcessingStage};
pub use granular_traits::{
    ImageReader as GranularImageReader, InferenceEngine, ModularPredictor, Postprocessor,
    Preprocessor,
};
pub use inference::{DefaultImageReader, OrtInfer, load_session};
pub use orientation::{
    OrientationResult, apply_document_orientation, apply_text_line_orientation,
    format_orientation_label, get_document_orientation_labels, get_text_line_orientation_labels,
    parse_document_orientation, parse_orientation_angle, parse_text_line_orientation,
};
pub use predictions::{
    IntoOwnedPrediction, IntoPrediction, OwnedPredictionResult, PipelineStats, PredictionResult,
};
pub use traits::{BasePredictor, ImageReader, PredictorBuilder, PredictorConfig, Sampler};

// init_tracing function has been moved to oar_ocr::utils::init_tracing
