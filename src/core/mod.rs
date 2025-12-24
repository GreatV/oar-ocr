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
pub mod image_reader;
pub mod inference;
#[macro_use]
pub mod macros;
pub mod registry;
pub mod traits;
pub mod validation;

// Image utilities are now available directly from oar_ocr::utils
// pub use crate::utils::{
//     create_rgb_image, dynamic_to_gray, dynamic_to_rgb, load_image, load_images,
// };
pub use crate::domain::{
    IntoOwnedPrediction, IntoPrediction, OrientationResult, OwnedPredictionResult,
    PredictionResult, apply_document_orientation, apply_text_line_orientation,
    format_orientation_label, get_document_orientation_labels, get_text_line_orientation_labels,
    parse_document_orientation, parse_orientation_angle, parse_text_line_orientation,
};
pub use batch::dynamic::{
    BatchPerformanceMetrics, CompatibleBatch, CrossImageBatch, CrossImageItem,
    DefaultDynamicBatcher, DynamicBatchConfig, DynamicBatchResult, DynamicBatcher, PaddingStrategy,
    ShapeCompatibilityStrategy,
};
pub use batch::{BatchData, BatchSampler, Tensor1D, Tensor2D, Tensor3D, Tensor4D, ToBatch};
pub use config::{
    ConfigError, ConfigValidator, ConfigValidatorExt, ModelInferenceConfig, TransformConfig,
    TransformRegistry, TransformType,
};
pub use constants::*;
pub use errors::{OCRError, OcrResult, ProcessingStage};
pub use image_reader::DefaultImageReader;
pub use inference::{OrtInfer, OrtInfer2D, OrtInfer3D, OrtInfer4D, load_session};
pub use traits::{
    AdapterBuilder, AdapterInfo, AdapterTask, BasePredictor, GranularImageReader, ImageReader,
    ImageTaskInput, InferenceEngine, ModelAdapter, ModularPredictor, Postprocessor,
    PredictorBuilder, PredictorConfig, Preprocessor, Sampler, StandardPredictor, Task, TaskRunner,
    TaskSchema, TaskType,
};

pub use registry::{DynModelAdapter, DynTaskInput, DynTaskOutput, TaskAdapter};
pub use validation::{
    validate_batch_size, validate_division, validate_finite, validate_image_dimensions,
    validate_index_bounds, validate_non_empty, validate_non_negative,
    validate_normalization_params, validate_positive, validate_range, validate_same_length,
    validate_tensor_shape,
};

// init_tracing function has been moved to oar_ocr::utils::init_tracing
