//! Error types for the OCR pipeline.
//!
//! This module provides a comprehensive error handling system for the OCR pipeline,
//! including various error types, helper constructors, and utilities for creating
//! well-structured errors with appropriate context and error chaining.
//!
//! # Architecture
//!
//! The error system is organized into several modules:
//!
//! - [`types`] - Core error types (OCRError, ProcessingStage, SimpleError)
//! - [`constructors`] - Helper methods for creating errors with context
//! - [`tests`] - Comprehensive test suite for error handling
//!
//! # Main Error Types
//!
//! - [`OCRError`] - The main error type used throughout the OCR pipeline
//! - [`ProcessingStage`] - Enum identifying which pipeline stage an error occurred in
//! - [`SimpleError`] - Lightweight error type for basic error messages
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
//!
//! # Error Categories
//!
//! The error system supports several categories of errors:
//!
//! - **Processing Errors** - Errors during various pipeline stages
//! - **Inference Errors** - Model inference and prediction errors  
//! - **Configuration Errors** - Invalid configuration or validation failures
//! - **Input Errors** - Invalid input data or parameters
//! - **System Errors** - IO, tensor operations, and other system-level errors

// Module declarations
pub mod constructors;
pub mod types;

// Re-export all public types and functions for backward compatibility
pub use types::{OCRError, ProcessingStage, SimpleError};

/// Convenient result alias for OCR operations.
pub type OcrResult<T> = Result<T, OCRError>;

// Note: Constructor methods are implemented directly on OCRError in the constructors module,
// so they are automatically available when OCRError is imported.
