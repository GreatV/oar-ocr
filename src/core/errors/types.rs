//! Core error types for the OCR pipeline.
//!
//! This module defines the fundamental error types used throughout the OCR system,
//! including the main OCRError enum and ProcessingStage enum.
//! These types provide the foundation for error handling across all pipeline components.

use thiserror::Error;

/// Errors that can occur during image processing operations.
#[derive(Debug, Error)]
pub enum ImageProcessError {
    /// The crop size is invalid (e.g., zero dimensions).
    #[error("Invalid crop size")]
    InvalidCropSize,
    /// The input image is smaller than the requested crop size.
    #[error(
        "Input image ({image_width}, {image_height}) smaller than the target size ({crop_width}, {crop_height})",
        image_width = image_size.0,
        image_height = image_size.1,
        crop_width = crop_size.0,
        crop_height = crop_size.1
    )]
    ImageTooSmall {
        /// The actual size of the image.
        image_size: (u32, u32),
        /// The requested crop size.
        crop_size: (u32, u32),
    },
    /// The requested interpolation mode is not supported.
    #[error("Unsupported interpolation method")]
    UnsupportedMode,
    /// The input data is invalid.
    #[error("Invalid input")]
    InvalidInput,
    /// The crop size is too large for the image.
    #[error("Crop size is too large for the image")]
    CropSizeTooLarge,
    /// The crop coordinates are out of bounds.
    #[error("Crop coordinates are out of bounds")]
    CropOutOfBounds,
    /// The crop coordinates are invalid.
    #[error("Invalid crop coordinates")]
    InvalidCropCoordinates,
}

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
    /// Error occurred during adapter execution.
    AdapterExecution,
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
            ProcessingStage::AdapterExecution => write!(f, "adapter execution"),
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

impl From<ImageProcessError> for OCRError {
    /// Converts an ImageProcessError to OCRError::Processing.
    fn from(error: ImageProcessError) -> Self {
        Self::Processing {
            kind: ProcessingStage::Generic,
            context: "Image processing failed".to_string(),
            source: Box::new(error),
        }
    }
}

impl OCRError {
    /// Creates a configuration error with enhanced context and details.
    ///
    /// # Arguments
    ///
    /// * `context` - High-level description of what was being configured
    /// * `details` - Specific details about what went wrong
    ///
    /// # Returns
    ///
    /// A new ConfigError with enhanced context
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use oar_ocr::core::errors::OCRError;
    /// let err = OCRError::config_error_detailed(
    ///     "task graph validation",
    ///     "task 'recognition' depends on 'detection' which does not exist"
    /// );
    /// assert!(matches!(err, OCRError::ConfigError { .. }));
    /// ```
    pub fn config_error_detailed(context: impl Into<String>, details: impl Into<String>) -> Self {
        Self::ConfigError {
            message: format!("{}: {}", context.into(), details.into()),
        }
    }

    /// Creates a configuration error with a suggestion for recovery.
    ///
    /// # Arguments
    ///
    /// * `context` - High-level description of what was being configured
    /// * `details` - Specific details about what went wrong
    /// * `suggestion` - Suggestion for how to fix the issue
    ///
    /// # Returns
    ///
    /// A new ConfigError with enhanced context and suggestion
    ///
    /// # Example
    ///
    /// ```rust,no_run
    /// # use oar_ocr::core::errors::OCRError;
    /// let err = OCRError::config_error_with_suggestion(
    ///     "model loading",
    ///     "model file not found at 'models/detection.onnx'",
    ///     "ensure the model has been downloaded and the path is correct"
    /// );
    /// assert!(matches!(err, OCRError::ConfigError { .. }));
    /// ```
    pub fn config_error_with_suggestion(
        context: impl Into<String>,
        details: impl Into<String>,
        suggestion: impl Into<String>,
    ) -> Self {
        Self::ConfigError {
            message: format!(
                "{}: {}; suggestion: {}",
                context.into(),
                details.into(),
                suggestion.into()
            ),
        }
    }

    /// Creates a configuration error for missing required fields.
    ///
    /// # Arguments
    ///
    /// * `field` - The name of the missing field
    /// * `context` - Context about where the field is required
    ///
    /// # Returns
    ///
    /// A new ConfigError for missing fields
    pub fn missing_field(field: impl Into<String>, context: impl Into<String>) -> Self {
        Self::ConfigError {
            message: format!(
                "missing required field '{}' in {}",
                field.into(),
                context.into()
            ),
        }
    }

    /// Creates a configuration error for invalid field values.
    ///
    /// # Arguments
    ///
    /// * `field` - The name of the field with an invalid value
    /// * `expected` - Description of what was expected
    /// * `actual` - Description of what was actually provided
    ///
    /// # Returns
    ///
    /// A new ConfigError for invalid field values
    pub fn invalid_field(
        field: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> Self {
        Self::ConfigError {
            message: format!(
                "invalid value for field '{}': expected {}, got {}",
                field.into(),
                expected.into(),
                actual.into()
            ),
        }
    }

    /// Creates a configuration error for type mismatches in task graphs.
    ///
    /// # Arguments
    ///
    /// * `task_id` - The ID of the task with the mismatch
    /// * `expected` - The expected type
    /// * `actual` - The actual type
    ///
    /// # Returns
    ///
    /// A new ConfigError for type mismatches
    pub fn type_mismatch(
        task_id: impl Into<String>,
        expected: impl Into<String>,
        actual: impl Into<String>,
    ) -> Self {
        Self::ConfigError {
            message: format!(
                "type mismatch in task '{}': expected {}, got {}",
                task_id.into(),
                expected.into(),
                actual.into()
            ),
        }
    }

    /// Creates a configuration error for dependency issues.
    ///
    /// # Arguments
    ///
    /// * `dependent` - The task that has a dependency
    /// * `dependency` - The missing or invalid dependency
    /// * `issue` - Description of the issue
    ///
    /// # Returns
    ///
    /// A new ConfigError for dependency issues
    pub fn dependency_error(
        dependent: impl Into<String>,
        dependency: impl Into<String>,
        issue: impl Into<String>,
    ) -> Self {
        Self::ConfigError {
            message: format!(
                "dependency error: task '{}' depends on '{}' which {}",
                dependent.into(),
                dependency.into(),
                issue.into()
            ),
        }
    }

    /// Wraps an error that occurred while executing a model adapter.
    pub fn adapter_execution_error(
        adapter: impl Into<String>,
        context: impl Into<String>,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Processing {
            kind: ProcessingStage::AdapterExecution,
            context: format!("{}: {}", adapter.into(), context.into()),
            source: Box::new(source),
        }
    }
}
