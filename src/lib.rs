//! # OAR OCR
//!
//! A Rust OCR library that extracts text from document images using ONNX models.
//! Supports text detection, recognition, document orientation, and rectification.
//!
//! ## Features
//!
//! - Complete OCR pipeline from image to text
//! - Modular components (use only what you need)
//! - Batch processing support
//! - ONNX Runtime integration for fast inference
//!
//! ## Components
//!
//! - **Text Detection**: Find text regions in images
//! - **Text Recognition**: Convert text regions to readable text
//! - **Document Orientation**: Detect document rotation (0°, 90°, 180°, 270°)
//! - **Document Rectification**: Fix perspective distortion
//! - **Text Line Classification**: Detect text line orientation
//!
//! ## Modules
//!
//! * [`core`] - Core traits, error handling, and batch processing
//! * [`predictor`] - OCR predictor implementations
//! * [`pipeline`] - Complete OCR pipeline
//! * [`processors`] - Image processing utilities
//! * [`utils`] - Utility functions for images and tensors
//!
//! ## Quick Start
//!
//! ### Complete OCR Pipeline
//!
//! ```rust,no_run
//! use oar_ocr::prelude::*;
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Build OCR pipeline
//! let mut ocr = OAROCRBuilder::new(
//!     "detection_model.onnx".to_string(),
//!     "recognition_model.onnx".to_string(),
//!     "char_dict.txt".to_string(),
//! ).build()?;
//!
//! // Process image
//! let result = ocr.predict(Path::new("document.jpg"))?;
//!
//! // Print results
//! for (text, score) in result.rec_texts.iter().zip(result.rec_scores.iter()) {
//!     println!("Text: {} (confidence: {:.2})", text, score);
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Individual Components
//!
//! ```rust,no_run
//! use oar_ocr::prelude::*;
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Text detection only
//! let mut detector = TextDetPredictorBuilder::new()
//!     .build(Path::new("detection_model.onnx"))?;
//!
//! let result = detector.predict_single(Path::new("image.jpg"))?;
//! println!("Detection result: {:?}", result);
//!
//! // Text recognition only
//! let char_dict = vec!["a".to_string(), "b".to_string()]; // Load your dictionary
//! let mut recognizer = TextRecPredictorBuilder::new()
//!     .character_dict(char_dict)
//!     .build(Path::new("recognition_model.onnx"))?;
//!
//! let result = recognizer.predict_single(Path::new("text_crop.jpg"))?;
//! println!("Recognition result: {:?}", result);
//! # Ok(())
//! # }
//! ```

// Core modules with macro support
#[macro_use]
pub mod core;
#[macro_use]
pub mod predictor;

pub mod pipeline;
pub mod processors;
pub mod utils;

/// Prelude module for convenient imports.
///
/// Import everything you need for OCR with a single use statement:
///
/// ```rust
/// use oar_ocr::prelude::*;
/// ```
///
/// This includes:
/// - All predictor types and builders
/// - Core traits like `Predictor` and `PipelineExecutor`
/// - Image processing utilities
/// - Error types and result types
/// - Tensor utilities
/// - The main `OAROCR` pipeline
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

    // Image processing utilities
    pub use crate::processors::{
        BaseRecLabelDecode, BoundingBox, BoxType, CTCLabelDecode, ChannelOrder, Crop, CropMode,
        DBPostProcess, DetResizeForTest, DocTrPostProcess, ImageProcessError, LimitType,
        MinAreaRect, NormalizeImage, OCRResize, Point, ResizeType, ScoreMode, Topk, TopkResult,
    };

    // Builder macros
    pub use crate::impl_builder;
    pub use crate::impl_builder_config;
    pub use crate::impl_config_validation;
    pub use crate::impl_enhanced_builder;

    // Predictor implementations
    pub use crate::predictor::{
        DocOrientationClassifier, DocOrientationClassifierBuilder, DoctrRectifierPredictor,
        DoctrRectifierPredictorBuilder, TextDetPredictor, TextDetPredictorBuilder,
        TextLineClasPredictor, TextLineClasPredictorBuilder, TextRecPredictor,
        TextRecPredictorBuilder,
    };

    // Pipeline components
    pub use crate::pipeline::{OAROCR, OAROCRBuilder, OAROCRConfig, OAROCRResult};

    // Utility functions
    pub use crate::utils;

    // Tensor utilities
    pub use crate::utils::tensor::{
        stack_tensor2d, stack_tensor3d, tensor1d_to_vec, tensor2d_to_vec, tensor3d_slice,
        tensor3d_to_vec, tensor4d_slice, tensor4d_to_vec, vec_to_tensor1d, vec_to_tensor2d,
        vec_to_tensor3d, vec_to_tensor4d,
    };
}
