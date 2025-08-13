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
//! let image = load_image(Path::new("document.jpg"))?;
//! let results = ocr.predict(&[image])?;
//! let result = &results[0];
//!
//! // Print results
//! for region in &result.text_regions {
//!     if let (Some(text), Some(confidence)) = (&region.text, region.confidence) {
//!         println!("Text: {} (confidence: {:.2})", text, confidence);
//!     }
//! }
//!
//! // Process multiple images
//! let images = load_images_batch(&[Path::new("doc1.jpg"), Path::new("doc2.jpg")])?;
//! let results = ocr.predict(&images)?;
//! for result in results {
//!     println!("Image {}: {} text regions found", result.index, result.text_regions.len());
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Advanced Configuration with Confidence Thresholding
//!
//! ```rust,no_run
//! use oar_ocr::prelude::*;
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Build OCR pipeline with confidence thresholding for orientation detection
//! let mut ocr = OAROCRBuilder::new(
//!     "detection_model.onnx".to_string(),
//!     "recognition_model.onnx".to_string(),
//!     "char_dict.txt".to_string(),
//! )
//! // Configure document orientation with confidence threshold
//! .doc_orientation_classify_model_path("orientation_model.onnx")
//! .doc_orientation_confidence_threshold(0.8) // Only accept predictions with 80% confidence
//! .use_doc_orientation_classify(true)
//! // Configure text line orientation with confidence threshold
//! .textline_orientation_classify_model_path("textline_orientation_model.onnx")
//! .textline_orientation_confidence_threshold(0.7) // Only accept predictions with 70% confidence
//! .use_textline_orientation(true)
//! // Set recognition score threshold
//! .text_rec_score_thresh(0.5)
//! .build()?;
//!
//! // Process image - low confidence orientations will fall back to defaults
//! let image = load_image(Path::new("document.jpg"))?;
//! let results = ocr.predict(&[image])?;
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

// Core modules
pub mod core;
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
/// This includes the stable public API:
/// - Main OCR pipeline (`OAROCR`, `OAROCRBuilder`, `OAROCRConfig`, `OAROCRResult`)
/// - Individual predictor builders for custom workflows
/// - Core traits for extending functionality
/// - Essential error and result types
/// - Image loading utilities
/// - Logging initialization
pub mod prelude {
    // === Main OCR Pipeline ===
    // The primary interface for most users
    pub use crate::pipeline::{OAROCR, OAROCRBuilder, OAROCRConfig, OAROCRResult, TextRegion};

    // === Individual Predictor Builders ===
    // For users who want to build custom OCR workflows
    pub use crate::predictor::{
        DocOrientationClassifierBuilder, DoctrRectifierPredictorBuilder, TextDetPredictorBuilder,
        TextLineClasPredictorBuilder, TextRecPredictorBuilder,
    };

    // === Core Traits ===
    // Essential traits for extending functionality
    pub use crate::core::traits::{BasePredictor, PredictorBuilder, StandardPredictor};

    // === Error Handling ===
    // Essential error types that users need to handle
    pub use crate::core::{OCRError, PredictionResult};

    // === Image Utilities ===
    // Essential utilities for loading and working with images
    pub use crate::utils::{init_tracing, load_image, load_images_batch};

    // === Configuration ===
    // For loading/saving pipeline configurations
    pub use crate::pipeline::{ConfigFormat, ConfigLoader};
}
