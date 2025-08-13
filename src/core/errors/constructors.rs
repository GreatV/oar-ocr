//! Error constructor utilities for the OCR pipeline.
//!
//! This module provides ergonomic helper functions for creating OCRError instances
//! with appropriate context and error chaining. These constructors make it easier
//! to create well-structured errors throughout the OCR pipeline.
//!
//! ## Batch Processing Error Helpers
//!
//! The module includes specialized helpers for creating consistent batch processing errors:
//!
//! ### `batch_item_error`
//! Creates standardized errors for individual item failures within batch processing:
//!
//! ```rust
//! use oar_ocr::core::OCRError;
//! use oar_ocr::core::errors::SimpleError;
//!
//! // Error for a specific item in a named batch
//! let error = OCRError::batch_item_error(
//!     "recognition",           // stage name
//!     Some("batch_123"),       // batch context
//!     2,                       // item index (0-based)
//!     Some(5),                 // total items
//!     "predict",               // operation
//!     SimpleError::new("timeout"),
//! );
//! // Results in: "recognition processing failed in batch 'batch_123' (item 3/5): operation 'predict'"
//!
//! // Error for item without batch context
//! let error = OCRError::batch_item_error(
//!     "orientation",
//!     None,
//!     0,
//!     None,
//!     "process",
//!     SimpleError::new("invalid input"),
//! );
//! // Results in: "orientation processing failed (item 1): operation 'process'"
//! ```
//!
//! ### `format_batch_error_message`
//! Creates formatted error messages for logging without wrapping in OCRError:
//!
//! ```rust
//! use oar_ocr::core::OCRError;
//! use oar_ocr::core::errors::SimpleError;
//!
//! let underlying_error = SimpleError::new("network timeout");
//! let affected_indices = vec![1, 3, 5];
//!
//! let message = OCRError::format_batch_error_message(
//!     "text recognition",
//!     "group_aspect_1.2",
//!     &affected_indices,
//!     &underlying_error,
//! );
//! // Results in: "text recognition batch 'group_aspect_1.2' failed: network timeout (affected indices: [1, 3, 5])"
//! ```

use super::types::{OCRError, ProcessingStage, SimpleError};

/// Implementation of OCRError with utility functions for creating errors.
impl OCRError {
    /// Creates an OCRError for tensor operations (simple variant).
    ///
    /// This is an alias for `tensor_operation_error` with default shape information.
    /// For detailed tensor shape errors, use `tensor_operation_error` directly.
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
        Self::tensor_operation_error("unknown", &[], &[], context, error)
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

    /// Creates an OCRError for image processing operations.
    ///
    /// # Arguments
    ///
    /// * `context` - Additional context about the error.
    /// * `error` - The underlying error that caused this error.
    ///
    /// # Returns
    ///
    /// An OCRError instance.
    pub fn image_processing(
        context: &str,
        error: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Processing {
            kind: ProcessingStage::ImageProcessing,
            context: context.to_string(),
            source: Box::new(error),
        }
    }

    /// Creates an OCRError for image processing operations with a simple message.
    ///
    /// This is an alias for `image_processing` with a SimpleError wrapper.
    /// For errors with underlying causes, use `image_processing` directly.
    ///
    /// # Arguments
    ///
    /// * `message` - The error message describing what went wrong.
    ///
    /// # Returns
    ///
    /// An OCRError instance.
    pub fn image_processing_error(message: impl Into<String>) -> Self {
        Self::image_processing(&message.into(), SimpleError::new("Image processing failed"))
    }

    /// Creates an OCRError for batch processing operations (simple variant).
    ///
    /// This is an alias for `batch_processing_error` with default batch information.
    /// For detailed batch processing errors, use `batch_processing_error` directly.
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
        Self::batch_processing_error(ProcessingStage::BatchProcessing, 0, 1, context, error)
    }

    /// Creates an OCRError for processing operations (simple variant).
    ///
    /// This is an alias for `processing_error_with_details` with default operation and input info.
    /// For detailed processing errors, use `processing_error_with_details` directly.
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
        Self::processing_error_with_details(kind, "unknown", context, error)
    }

    /// Creates an OCRError for basic inference operations (legacy method).
    ///
    /// # Arguments
    ///
    /// * `error` - The underlying error that caused this error.
    ///
    /// # Returns
    ///
    /// An OCRError instance.
    pub fn basic_inference_error(error: impl std::error::Error + Send + Sync + 'static) -> Self {
        Self::Inference {
            model_name: "Unknown".to_string(),
            context: "Basic inference error".to_string(),
            source: Box::new(error),
        }
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
                "Configuration error in field '{field}' with value '{value}': {reason}"
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
                "Validation failed in {component}: field '{field}' expected {expected}, but got '{actual}'"
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
                "Resource limit exceeded for {resource}: requested {requested} but limit is {limit}"
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
            context: format!("Operation '{operation}' failed on input '{input_info}': {error}"),
            source: Box::new(error),
        }
    }

    /// Creates an OCRError for model inference operations with detailed context.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model where inference failed.
    /// * `operation` - The operation that failed.
    /// * `batch_index` - The batch index where the error occurred.
    /// * `input_shape` - The input tensor shape.
    /// * `context` - Additional context about the error.
    /// * `error` - The underlying error that caused this error.
    ///
    /// # Returns
    ///
    /// An OCRError instance.
    pub fn model_inference_error(
        model_name: &str,
        operation: &str,
        batch_index: usize,
        input_shape: &[usize],
        context: &str,
        error: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::ModelInference {
            model_name: model_name.to_string(),
            operation: operation.to_string(),
            batch_index,
            input_shape: input_shape.to_vec(),
            context: context.to_string(),
            source: Box::new(error),
        }
    }

    /// Creates an OCRError for inference operations with model context (simple variant).
    ///
    /// This is an alias for `model_inference_error` with default operation and shape information.
    /// For detailed inference errors, use `model_inference_error` directly.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model where inference failed.
    /// * `context` - Additional context about the error.
    /// * `error` - The underlying error that caused this error.
    ///
    /// # Returns
    ///
    /// An OCRError instance.
    pub fn inference_error(
        model_name: &str,
        context: &str,
        error: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::model_inference_error(model_name, "unknown", 0, &[], context, error)
    }

    /// Creates an OCRError for tensor operations with detailed shape information.
    ///
    /// # Arguments
    ///
    /// * `operation` - The tensor operation that failed.
    /// * `expected_shape` - The expected tensor shape.
    /// * `actual_shape` - The actual tensor shape.
    /// * `context` - Additional context about where the error occurred.
    /// * `error` - The underlying error that caused this error.
    ///
    /// # Returns
    ///
    /// An OCRError instance.
    pub fn tensor_operation_error(
        operation: &str,
        expected_shape: &[usize],
        actual_shape: &[usize],
        context: &str,
        error: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::TensorOperation {
            operation: operation.to_string(),
            expected_shape: expected_shape.to_vec(),
            actual_shape: actual_shape.to_vec(),
            context: context.to_string(),
            source: Box::new(error),
        }
    }

    /// Creates an OCRError for batch processing operations with detailed context.
    ///
    /// # Arguments
    ///
    /// * `stage` - The processing stage where the error occurred.
    /// * `batch_index` - The index of the batch item that failed.
    /// * `batch_size` - The total size of the batch.
    /// * `operation` - The operation that failed.
    /// * `error` - The underlying error that caused this error.
    ///
    /// # Returns
    ///
    /// An OCRError instance.
    pub fn batch_processing_error(
        stage: ProcessingStage,
        batch_index: usize,
        batch_size: usize,
        operation: &str,
        error: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Processing {
            kind: stage,
            context: format!(
                "Batch processing failed: operation '{operation}' failed on item {batch_index}/{batch_size}"
            ),
            source: Box::new(error),
        }
    }

    /// Creates an OCRError for pipeline stage operations with detailed context.
    ///
    /// # Arguments
    ///
    /// * `stage_name` - The name of the pipeline stage.
    /// * `stage_id` - The ID of the pipeline stage.
    /// * `input_count` - The number of input items.
    /// * `operation` - The operation that failed.
    /// * `error` - The underlying error that caused this error.
    ///
    /// # Returns
    ///
    /// An OCRError instance.
    pub fn pipeline_stage_error(
        stage_name: &str,
        stage_id: &str,
        input_count: usize,
        operation: &str,
        error: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::Processing {
            kind: ProcessingStage::PipelineExecution,
            context: format!(
                "Pipeline stage '{stage_name}' (id: {stage_id}) failed: operation '{operation}' on {input_count} items"
            ),
            source: Box::new(error),
        }
    }
}

/// Helper functions for creating consistent batch processing errors
impl OCRError {
    /// Creates a standardized error for batch item processing failures.
    ///
    /// This helper reduces code duplication and ensures consistent error formatting
    /// across different batch processing contexts (orientation, recognition, etc.).
    ///
    /// # Arguments
    ///
    /// * `stage_name` - The name of the processing stage (e.g., "orientation", "recognition")
    /// * `batch_context` - Additional context about the batch (e.g., batch ID, group name)
    /// * `item_index` - The index of the failed item within the batch
    /// * `total_items` - Total number of items in the batch (if known)
    /// * `operation` - The specific operation that failed
    /// * `error` - The underlying error
    ///
    /// # Returns
    ///
    /// A formatted OCRError with consistent batch processing context
    pub fn batch_item_error(
        stage_name: &str,
        batch_context: Option<&str>,
        item_index: usize,
        total_items: Option<usize>,
        operation: &str,
        error: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        let batch_info = match (batch_context, total_items) {
            (Some(context), Some(total)) => {
                format!(" in batch '{context}' (item {}/{total})", item_index + 1)
            }
            (Some(context), None) => format!(" in batch '{context}' (item {})", item_index + 1),
            (None, Some(total)) => format!(" (item {}/{total})", item_index + 1),
            (None, None) => format!(" (item {})", item_index + 1),
        };

        Self::Processing {
            kind: ProcessingStage::BatchProcessing,
            context: format!("{stage_name} processing failed{batch_info}: operation '{operation}'"),
            source: Box::new(error),
        }
    }

    /// Creates a standardized error message for batch processing failures.
    ///
    /// This helper creates formatted error messages without wrapping in OCRError,
    /// useful for logging and warning messages.
    ///
    /// # Arguments
    ///
    /// * `stage_name` - The name of the processing stage
    /// * `batch_context` - Additional context about the batch
    /// * `affected_indices` - Indices of items affected by the failure
    /// * `error` - The underlying error
    ///
    /// # Returns
    ///
    /// A formatted error message string
    pub fn format_batch_error_message(
        stage_name: &str,
        batch_context: &str,
        affected_indices: &[usize],
        error: &dyn std::error::Error,
    ) -> String {
        format!(
            "{stage_name} batch '{batch_context}' failed: {error} (affected indices: {affected_indices:?})"
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::errors::SimpleError;

    #[test]
    fn test_batch_item_error_with_full_context() {
        let underlying_error = SimpleError::new("test error");
        let error = OCRError::batch_item_error(
            "recognition",
            Some("batch_123"),
            2,
            Some(5),
            "predict",
            underlying_error,
        );

        match error {
            OCRError::Processing { context, .. } => {
                assert_eq!(
                    context,
                    "recognition processing failed in batch 'batch_123' (item 3/5): operation 'predict'"
                );
            }
            _ => panic!("Expected Processing error"),
        }
    }

    #[test]
    fn test_batch_item_error_minimal_context() {
        let underlying_error = SimpleError::new("test error");
        let error =
            OCRError::batch_item_error("orientation", None, 0, None, "process", underlying_error);

        match error {
            OCRError::Processing { context, .. } => {
                assert_eq!(
                    context,
                    "orientation processing failed (item 1): operation 'process'"
                );
            }
            _ => panic!("Expected Processing error"),
        }
    }

    #[test]
    fn test_format_batch_error_message() {
        let underlying_error = SimpleError::new("network timeout");
        let affected_indices = vec![1, 3, 5];

        let message = OCRError::format_batch_error_message(
            "text recognition",
            "group_aspect_1.2",
            &affected_indices,
            &underlying_error,
        );

        assert_eq!(
            message,
            "text recognition batch 'group_aspect_1.2' failed: network timeout (affected indices: [1, 3, 5])"
        );
    }
}
