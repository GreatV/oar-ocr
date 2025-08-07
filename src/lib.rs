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
//! // Process single image
//! let results = ocr.predict(&[Path::new("document.jpg")])?;
//! let result = &results[0];
//!
//! // Print results
//! for (text, score) in result.rec_texts.iter().zip(result.rec_scores.iter()) {
//!     println!("Text: {} (confidence: {:.2})", text, score);
//! }
//!
//! // Process multiple images
//! let results = ocr.predict(&[Path::new("doc1.jpg"), Path::new("doc2.jpg")])?;
//! for result in results {
//!     println!("Image {}: {} text regions found", result.index, result.text_boxes.len());
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Individual Components
//!
//! ```rust,no_run
//! use oar_ocr::prelude::*;
//! use oar_ocr::core::traits::StandardPredictor;
//! use oar_ocr::utils::load_image;
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Text detection only
//! let mut detector = TextDetPredictorBuilder::new()
//!     .build(Path::new("detection_model.onnx"))?;
//!
//! let image = load_image(Path::new("image.jpg"))?;
//! let result = detector.predict(vec![image], None)?;
//! println!("Detection result: {:?}", result);
//!
//! // Text recognition only
//! let char_dict = vec!["a".to_string(), "b".to_string()]; // Load your dictionary
//! let mut recognizer = TextRecPredictorBuilder::new()
//!     .character_dict(char_dict)
//!     .build(Path::new("recognition_model.onnx"))?;
//!
//! let image = load_image(Path::new("text_crop.jpg"))?;
//! let result = recognizer.predict(vec![image], None)?;
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
/// - Core traits like `BasePredictor` and `StandardPredictor`
/// - Image processing utilities
/// - Error types and result types
/// - Tensor utilities
/// - The main `OAROCR` pipeline
pub mod prelude {

    // Core components
    pub use crate::core::{
        BasePredictor, BatchData, BatchSampler, CommonBuilderConfig, ConfigError,
        DefaultImageReader, ImageReader, OCRError, OrtInfer, PipelineStats, PredictionResult,
        PredictorBuilder, PredictorConfig, ProcessingStage, Sampler, Tensor1D, Tensor2D, Tensor3D,
        Tensor4D, ToBatch, TransformConfig, TransformRegistry, TransformType,
    };

    // Image utilities and logging
    pub use crate::utils::{
        create_rgb_image, dynamic_to_gray, dynamic_to_rgb, init_tracing, load_image,
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
    pub use crate::pipeline::{
        ConfigFormat, ConfigLoader, OAROCR, OAROCRBuilder, OAROCRConfig, OAROCRResult,
    };

    // Utility functions
    pub use crate::utils;

    // Tensor utilities
    pub use crate::utils::tensor::{
        stack_tensor2d, stack_tensor3d, tensor1d_to_vec, tensor2d_to_vec, tensor3d_slice,
        tensor3d_to_vec, tensor4d_slice, tensor4d_to_vec, vec_to_tensor1d, vec_to_tensor2d,
        vec_to_tensor3d, vec_to_tensor4d,
    };

    // Transform utilities
    pub use crate::utils::transform::{Point2f, get_rotate_crop_image};
}
