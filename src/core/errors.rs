//! Error types for the OCR pipeline.
//!
//! This module defines various error types that can occur during the OCR process,
//! including image loading errors, processing errors, inference errors, and
//! configuration errors. It also provides utility functions for creating these
//! errors with appropriate context.

use thiserror::Error;

/// Enum representing different stages of processing in the OCR pipeline.
///
/// This enum is used to identify which stage of the OCR pipeline an error occurred in.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ProcessingStage {
    /// Error occurred during tensor operations.
    TensorOperation,
    /// Error occurred during image normalization.
    Normalization,
    /// Error occurred during image resizing.
    Resize,
    /// Error occurred during batch processing.
    BatchProcessing,
    /// Error occurred during post-processing.
    PostProcessing,
    /// Generic processing error.
    Generic,
}

/// Implementation of Display for ProcessingStage.
///
/// This allows ProcessingStage to be converted to a string representation.
impl std::fmt::Display for ProcessingStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ProcessingStage::TensorOperation => write!(f, "tensor operation"),
            ProcessingStage::Normalization => write!(f, "normalization"),
            ProcessingStage::Resize => write!(f, "resize"),
            ProcessingStage::BatchProcessing => write!(f, "batch processing"),
            ProcessingStage::PostProcessing => write!(f, "post-processing"),
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
    #[error("inference")]
    Inference(#[source] Box<dyn std::error::Error + Send + Sync>),

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

    /// Error from tensor operations.
    #[error("tensor operation")]
    Tensor(#[from] ndarray::ShapeError),

    /// IO error.
    #[error("io")]
    Io(#[from] std::io::Error),
}

/// Implementation of OCRError with utility functions for creating errors.
impl OCRError {
    /// Creates an OCRError for tensor operations.
    ///
    /// # Arguments
    ///
    /// * `context` - Additional context about the error.
    /// * `error` - The underlying error that caused this error.
    ///
    /// # Returns
    ///
    /// An OCRError instance.
    pub fn tensor_operation(
        context: &str,
        error: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Processing {
            kind: ProcessingStage::TensorOperation,
            context: context.to_string(),
            source: Box::new(error),
        }
    }

    /// Creates an OCRError for post-processing operations.
    ///
    /// # Arguments
    ///
    /// * `context` - Additional context about the error.
    /// * `error` - The underlying error that caused this error.
    ///
    /// # Returns
    ///
    /// An OCRError instance.
    pub fn post_processing(
        context: &str,
        error: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Processing {
            kind: ProcessingStage::PostProcessing,
            context: context.to_string(),
            source: Box::new(error),
        }
    }

    /// Creates an OCRError for normalization operations.
    ///
    /// # Arguments
    ///
    /// * `context` - Additional context about the error.
    /// * `error` - The underlying error that caused this error.
    ///
    /// # Returns
    ///
    /// An OCRError instance.
    pub fn normalization(
        context: &str,
        error: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Processing {
            kind: ProcessingStage::Normalization,
            context: context.to_string(),
            source: Box::new(error),
        }
    }

    /// Creates an OCRError for resize operations.
    ///
    /// # Arguments
    ///
    /// * `context` - Additional context about the error.
    /// * `error` - The underlying error that caused this error.
    ///
    /// # Returns
    ///
    /// An OCRError instance.
    pub fn resize_error(
        context: &str,
        error: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Processing {
            kind: ProcessingStage::Resize,
            context: context.to_string(),
            source: Box::new(error),
        }
    }

    /// Creates an OCRError for batch processing operations.
    ///
    /// # Arguments
    ///
    /// * `context` - Additional context about the error.
    /// * `error` - The underlying error that caused this error.
    ///
    /// # Returns
    ///
    /// An OCRError instance.
    pub fn batch_processing(
        context: &str,
        error: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Processing {
            kind: ProcessingStage::BatchProcessing,
            context: context.to_string(),
            source: Box::new(error),
        }
    }

    /// Creates an OCRError for processing operations.
    ///
    /// # Arguments
    ///
    /// * `kind` - The stage of processing where the error occurred.
    /// * `context` - Additional context about the error.
    /// * `error` - The underlying error that caused this error.
    ///
    /// # Returns
    ///
    /// An OCRError instance.
    pub fn processing_error(
        kind: ProcessingStage,
        context: &str,
        error: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Processing {
            kind,
            context: context.to_string(),
            source: Box::new(error),
        }
    }

    /// Creates an OCRError for inference operations.
    ///
    /// # Arguments
    ///
    /// * `error` - The underlying error that caused this error.
    ///
    /// # Returns
    ///
    /// An OCRError instance.
    pub fn inference_error(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Inference(Box::new(error))
    }

    /// Creates an OCRError for invalid input.
    ///
    /// # Arguments
    ///
    /// * `message` - A message describing the invalid input.
    ///
    /// # Returns
    ///
    /// An OCRError instance.
    pub fn invalid_input(message: impl Into<String>) -> Self {
        Self::InvalidInput {
            message: message.into(),
        }
    }

    /// Creates an OCRError for configuration errors.
    ///
    /// # Arguments
    ///
    /// * `message` - A message describing the configuration error.
    ///
    /// # Returns
    ///
    /// An OCRError instance.
    pub fn config_error(message: impl Into<String>) -> Self {
        Self::ConfigError {
            message: message.into(),
        }
    }

    /// Creates an OCRError for configuration errors with context.
    ///
    /// # Arguments
    ///
    /// * `field` - The field where the error occurred.
    /// * `value` - The value of the field.
    /// * `reason` - The reason for the error.
    ///
    /// # Returns
    ///
    /// An OCRError instance.
    pub fn config_error_with_context(field: &str, value: &str, reason: &str) -> Self {
        Self::ConfigError {
            message: format!(
                "Configuration error in field '{}' with value '{}': {}",
                field, value, reason
            ),
        }
    }

    /// Creates an OCRError for validation errors.
    ///
    /// # Arguments
    ///
    /// * `component` - The component where the error occurred.
    /// * `field` - The field where the error occurred.
    /// * `expected` - The expected value.
    /// * `actual` - The actual value.
    ///
    /// # Returns
    ///
    /// An OCRError instance.
    pub fn validation_error(component: &str, field: &str, expected: &str, actual: &str) -> Self {
        Self::InvalidInput {
            message: format!(
                "Validation failed in {}: field '{}' expected {}, but got '{}'",
                component, field, expected, actual
            ),
        }
    }

    /// Creates an OCRError for resource limit errors.
    ///
    /// # Arguments
    ///
    /// * `resource` - The resource that exceeded its limit.
    /// * `limit` - The maximum allowed limit.
    /// * `requested` - The requested amount.
    ///
    /// # Returns
    ///
    /// An OCRError instance.
    pub fn resource_limit_error(resource: &str, limit: usize, requested: usize) -> Self {
        Self::InvalidInput {
            message: format!(
                "Resource limit exceeded for {}: requested {} but limit is {}",
                resource, requested, limit
            ),
        }
    }

    /// Creates an OCRError for processing operations with detailed context.
    ///
    /// # Arguments
    ///
    /// * `stage` - The stage of processing where the error occurred.
    /// * `operation` - The operation that failed.
    /// * `input_info` - Information about the input.
    /// * `error` - The underlying error that caused this error.
    ///
    /// # Returns
    ///
    /// An OCRError instance.
    pub fn processing_error_with_details(
        stage: ProcessingStage,
        operation: &str,
        input_info: &str,
        error: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Processing {
            kind: stage,
            context: format!(
                "Operation '{}' failed on input '{}': {}",
                operation, input_info, error
            ),
            source: Box::new(error),
        }
    }
}

/// Implementation of From<image::ImageError> for OCRError.
///
/// This allows image::ImageError to be automatically converted to OCRError.
impl From<image::ImageError> for OCRError {
    fn from(error: image::ImageError) -> Self {
        Self::ImageLoad(error)
    }
}

/// Implementation of From<crate::core::config::ConfigError> for OCRError.
///
/// This allows crate::core::config::ConfigError to be automatically converted to OCRError.
impl From<crate::core::config::ConfigError> for OCRError {
    fn from(error: crate::core::config::ConfigError) -> Self {
        Self::ConfigError {
            message: error.to_string(),
        }
    }
}

/// Implementation of From<crate::processors::ImageProcessError> for OCRError.
///
/// This allows crate::processors::ImageProcessError to be automatically converted to OCRError.
impl From<crate::processors::ImageProcessError> for OCRError {
    fn from(error: crate::processors::ImageProcessError) -> Self {
        Self::Processing {
            kind: ProcessingStage::Generic,
            context: "Image processing failed".to_string(),
            source: Box::new(error),
        }
    }
}
