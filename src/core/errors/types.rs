//! Core error types for the OCR pipeline.
//!
//! This module defines the fundamental error types used throughout the OCR system,
//! including the main OCRError enum, ProcessingStage enum, and SimpleError struct.
//! These types provide the foundation for error handling across all pipeline components.

use thiserror::Error;

/// A simple error type for basic error messages.
///
/// This is a lightweight error type used internally for creating error chains
/// and providing simple error messages when more complex error types are not needed.
#[derive(Debug)]
pub struct SimpleError {
    message: String,
}

impl SimpleError {
    /// Creates a new SimpleError with the given message.
    ///
    /// # Arguments
    ///
    /// * `message` - The error message, can be any type that converts to String
    ///
    /// # Returns
    ///
    /// A new SimpleError instance
    pub fn new(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
        }
    }
}

impl std::fmt::Display for SimpleError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.message)
    }
}

impl std::error::Error for SimpleError {}

/// Enum representing different stages of processing in the OCR pipeline.
///
/// This enum is used to identify which stage of the OCR pipeline an error occurred in,
/// providing context for debugging and error handling.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProcessingStage {
    /// Error occurred during tensor operations.
    TensorOperation,
    /// Error occurred during image normalization.
    Normalization,
    /// Error occurred during image resizing.
    Resize,
    /// Error occurred during image processing operations.
    ImageProcessing,
    /// Error occurred during batch processing.
    BatchProcessing,
    /// Error occurred during post-processing.
    PostProcessing,
    /// Error occurred during pipeline execution.
    PipelineExecution,
    /// Generic processing error.
    Generic,
}

impl std::fmt::Display for ProcessingStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessingStage::TensorOperation => write!(f, "tensor operation"),
            ProcessingStage::Normalization => write!(f, "normalization"),
            ProcessingStage::Resize => write!(f, "resize"),
            ProcessingStage::ImageProcessing => write!(f, "image processing"),
            ProcessingStage::BatchProcessing => write!(f, "batch processing"),
            ProcessingStage::PostProcessing => write!(f, "post-processing"),
            ProcessingStage::PipelineExecution => write!(f, "pipeline execution"),
            ProcessingStage::Generic => write!(f, "processing"),
        }
    }
}

/// Enum representing various errors that can occur in the OCR pipeline.
///
/// This enum defines all the possible error types that can occur during
/// the OCR process, including image loading errors, processing errors,
/// inference errors, and configuration errors.
#[derive(Error, Debug)]
pub enum OCRError {
    /// Error occurred while loading an image.
    #[error("image load")]
    ImageLoad(#[source] image::ImageError),

    /// Error occurred during processing.
    #[error("{kind} failed: {context}")]
    Processing {
        /// The stage of processing where the error occurred.
        kind: ProcessingStage,
        /// Additional context about the error.
        context: String,
        /// The underlying error that caused this error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Error occurred during inference.
    #[error("inference failed in model '{model_name}': {context}")]
    Inference {
        /// The name of the model where inference failed.
        model_name: String,
        /// Additional context about the inference error.
        context: String,
        /// The underlying error that caused this error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Error occurred during model inference with detailed context.
    #[error(
        "model '{model_name}' inference failed: {operation} on batch[{batch_index}] with input shape {input_shape:?}"
    )]
    ModelInference {
        /// The name of the model where inference failed.
        model_name: String,
        /// The operation that failed (e.g., "forward_pass", "preprocessing").
        operation: String,
        /// The batch index where the error occurred.
        batch_index: usize,
        /// The input tensor shape.
        input_shape: Vec<usize>,
        /// Additional context about the error.
        context: String,
        /// The underlying error that caused this error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Error indicating invalid input.
    #[error("invalid input: {message}")]
    InvalidInput {
        /// A message describing the invalid input.
        message: String,
    },

    /// Error indicating a configuration problem.
    #[error("configuration: {message}")]
    ConfigError {
        /// A message describing the configuration error.
        message: String,
    },

    /// Error indicating a buffer is too small.
    #[error("buffer too small: expected at least {expected} bytes, got {actual} bytes")]
    BufferTooSmall {
        /// The expected minimum buffer size.
        expected: usize,
        /// The actual buffer size.
        actual: usize,
    },

    /// Error from the ONNX Runtime session.
    #[error(transparent)]
    Session(#[from] ort::Error),

    /// Error from tensor operations with detailed context.
    #[error(
        "tensor operation '{operation}' failed: expected shape {expected_shape:?}, got {actual_shape:?} in {context}"
    )]
    TensorOperation {
        /// The tensor operation that failed.
        operation: String,
        /// The expected tensor shape.
        expected_shape: Vec<usize>,
        /// The actual tensor shape.
        actual_shape: Vec<usize>,
        /// Additional context about where the error occurred.
        context: String,
        /// The underlying error that caused this error.
        #[source]
        source: Box<dyn std::error::Error + Send + Sync>,
    },

    /// Error from basic tensor operations (fallback for ndarray errors).
    #[error("tensor operation")]
    Tensor(#[from] ndarray::ShapeError),

    /// IO error.
    #[error("io")]
    Io(#[from] std::io::Error),

    /// Error loading a model file, with context and suggestions.
    #[error("model load failed for '{model_path}': {reason}{suggestion}")]
    ModelLoad {
        /// Path to the model that failed to load
        model_path: String,
        /// Short reason string
        reason: String,
        /// Optional suggestion (prefixed with '; ' when present)
        suggestion: String,
        /// Underlying source error
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },
}

// From trait implementations for automatic error conversions

impl From<image::ImageError> for OCRError {
    /// Converts an image::ImageError to OCRError::ImageLoad.
    fn from(error: image::ImageError) -> Self {
        Self::ImageLoad(error)
    }
}

impl From<crate::core::config::ConfigError> for OCRError {
    /// Converts a ConfigError to OCRError::ConfigError.
    fn from(error: crate::core::config::ConfigError) -> Self {
        Self::ConfigError {
            message: error.to_string(),
        }
    }
}

impl From<crate::processors::ImageProcessError> for OCRError {
    /// Converts an ImageProcessError to OCRError::Processing.
    fn from(error: crate::processors::ImageProcessError) -> Self {
        Self::Processing {
            kind: ProcessingStage::Generic,
            context: "Image processing failed".to_string(),
            source: Box::new(error),
        }
    }
}
