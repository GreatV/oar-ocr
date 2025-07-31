//! OAR OCR - A comprehensive OCR (Optical Character Recognition) library for document processing.
//!
//! This library provides a complete OCR pipeline for processing document images, including:
//! - Document orientation classification
//! - Text detection (finding text regions in images)
//! - Text recognition (recognizing characters in text regions)
//! - Document rectification (correcting document perspective)
//! - Text line classification
//!
//! The library is built on top of ONNX Runtime for efficient inference and provides
//! a modular architecture that allows for easy customization and extension.
//!
//! # Modules
//!
//! * [`core`] - Fundamental components and traits for the OCR pipeline
//! * [`predictor`] - Implementations of various OCR predictors
//! * [`pipeline`] - Main OCR pipeline that combines all components
//! * [`processors`] - Image processing utilities for OCR
//! * [`utils`] - Utility functions for image and tensor operations
//!
//! # Examples
//!
//! Basic usage:
//! ```rust
//! // Import everything from the prelude for convenience
//! use oar_ocr::prelude::*;
//!
//! // Build and use OCR components
//! // let detector = TextDetPredictorBuilder::new().build_typed(model_path)?;
//! // let recognizer = TextRecPredictorBuilder::new().build_typed(model_path)?;
//! ```
//!
//! For more detailed examples, see the `examples/` directory in the repository.

#[macro_use]
pub mod core;
#[macro_use]
pub mod predictor;
pub mod pipeline;
pub mod processors;
pub mod utils;

/// A prelude module for convenient imports.
///
/// This module re-exports commonly used types and traits from across the crate,
/// allowing users to import them with a single `use` statement.
///
/// # Re-exports
///
/// The prelude includes:
/// - Core traits and types for building OCR pipelines
/// - Processor utilities for image manipulation
/// - Predictor types for different OCR tasks
/// - Pipeline components for combining OCR functionality
/// - Utility functions for image and tensor operations
pub mod prelude {

    // Core components
    pub use crate::core::{
        BasePredictor, BatchData, BatchSampler, CommonBuilderConfig, ConfigError,
        DefaultImageReader, ImageReader, OCRError, OrtInfer, PipelineExecutor, PipelineStats,
        PredictionResult, Predictor, PredictorBuilder, PredictorConfig, ProcessingStage, Sampler,
        Tensor1D, Tensor2D, Tensor3D, Tensor4D, ToBatch, TransformConfig, TransformRegistry,
        TransformType, create_rgb_image, dynamic_to_gray, dynamic_to_rgb, load_image,
        load_images_batch,
    };

    // Processor utilities
    pub use crate::processors::{
        BaseRecLabelDecode, BoundingBox, BoxType, CTCLabelDecode, ChannelOrder, Crop, CropMode,
        DBPostProcess, DetResizeForTest, DocTrPostProcess, ImageProcessError, LimitType,
        MinAreaRect, NormalizeImage, OCRResize, Point, ResizeType, ScoreMode, Topk, TopkResult,
    };

    // Macros for building predictors
    pub use crate::impl_builder;
    pub use crate::impl_builder_config;
    pub use crate::impl_config_validation;
    pub use crate::impl_enhanced_builder;

    // Predictor types
    pub use crate::predictor::{
        DocOrientationClassifier, DocOrientationClassifierBuilder, DoctrRectifierPredictor,
        DoctrRectifierPredictorBuilder, TextDetPredictor, TextDetPredictorBuilder,
        TextLineClasPredictor, TextLineClasPredictorBuilder, TextRecPredictor,
        TextRecPredictorBuilder,
    };

    // Pipeline components
    pub use crate::pipeline::{OAROCR, OAROCRBuilder, OAROCRConfig, OAROCRResult};

    // Utility modules
    pub use crate::utils;

    // Tensor utilities
    pub use crate::utils::tensor::{
        stack_tensor2d, stack_tensor3d, tensor1d_to_vec, tensor2d_to_vec, tensor3d_slice,
        tensor3d_to_vec, tensor4d_slice, tensor4d_to_vec, vec_to_tensor1d, vec_to_tensor2d,
        vec_to_tensor3d, vec_to_tensor4d,
    };
}
