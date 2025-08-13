//! Predictor implementations for various OCR tasks.
//!
//! This module contains implementations of different predictors used in the OCR pipeline:
//! - Text detection (finding text regions in images)
//! - Text recognition (recognizing characters in text regions)
//! - Document orientation classification (determining document orientation)
//! - Document rectification (correcting document perspective)
//! - Text line classification (classifying text line properties)
//!
//! Each predictor module contains both the predictor implementation and its builder.
//!
//! Previously this module contained macro definitions for implementing predictor traits,
//! but they have been removed as they were unused throughout the codebase.

/// Text recognition predictor using CRNN (Convolutional Recurrent Neural Network)
pub mod crnn_recognizer;

/// Text detection predictor using DB (Differentiable Binarization) algorithm
pub mod db_detector;

/// Document orientation classifier for determining document orientation
pub mod doc_orientation_classifier;

/// Document rectifier using DocTR (Document Text Recognition) models
pub mod doctr_rectifier;

/// Text line classifier for classifying properties of text lines
pub mod text_line_classifier;

// Re-exports for easier access to predictor types
pub use crnn_recognizer::{TextRecPredictor, TextRecPredictorBuilder, TextRecPredictorConfig};
pub use db_detector::{TextDetPredictor, TextDetPredictorBuilder, TextDetPredictorConfig};
pub use doc_orientation_classifier::{
    DocOrientationClassifier, DocOrientationClassifierBuilder, DocOrientationClassifierConfig,
};
pub use doctr_rectifier::{
    DoctrRectifierPredictor, DoctrRectifierPredictorBuilder, DoctrRectifierPredictorConfig,
};
pub use text_line_classifier::{
    TextLineClasPredictor, TextLineClasPredictorBuilder, TextLineClasPredictorConfig,
};
