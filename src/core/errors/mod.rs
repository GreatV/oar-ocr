//! Error types for the OCR pipeline.
//!
//! This module provides a comprehensive error handling system for the OCR pipeline,
//! including various error types, helper constructors, and utilities for creating
//! well-structured errors with appropriate context and error chaining.
//!
//! # Usage
//!
//! ```rust
//! use oar_ocr::core::errors::{OCRError, ProcessingStage};
//!
//! // Create a processing error with context
//! let error = OCRError::tensor_operation(
//!     "Failed to reshape tensor for batch processing",
//!     std::io::Error::new(std::io::ErrorKind::InvalidData, "Invalid tensor shape")
//! );
//!
//! // Create a configuration error
//! let config_error = OCRError::config_error("Missing required model path");
//!
//! // Create a validation error with detailed context
//! let validation_error = OCRError::validation_error(
//!     "TextDetector",
//!     "input_size",
//!     "[640, 640]",
//!     "[320, 320]"
//! );
//! ```

// Module declarations
pub mod constructors;
pub mod types;

// Re-export all public types and functions for backward compatibility
pub use types::{ImageProcessError, OCRError, OpaqueError, ProcessingStage};

/// Convenient result alias for OCR operations.
pub type OcrResult<T> = Result<T, OCRError>;

// Note: Constructor methods are implemented directly on OCRError in the constructors module,
// so they are automatically available when OCRError is imported.
