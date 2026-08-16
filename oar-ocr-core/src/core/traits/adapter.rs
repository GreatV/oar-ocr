//! Model adapter trait definitions for the OCR pipeline.
//!
//! This module defines the `ModelAdapter` trait and related types that adapt
//! various model implementations to conform to task interfaces. Adapters handle
//! preprocessing, inference, and postprocessing for specific models.

use super::task::{Task, TaskType};
use crate::core::OCRError;
use std::fmt::Debug;

/// Information about a model adapter.
#[derive(Debug, Clone)]
pub struct AdapterInfo {
    /// Name of the model (e.g., "DB", "CRNN", "RT-DETR")
    pub model_name: String,
    /// Task type this adapter supports
    pub task_type: TaskType,
    /// Description of the model
    pub description: String,
}

impl AdapterInfo {
    /// Creates a new adapter info.
    pub fn new(
        model_name: impl Into<String>,
        task_type: TaskType,
        description: impl Into<String>,
    ) -> Self {
        Self {
            model_name: model_name.into(),
            task_type,
            description: description.into(),
        }
    }
}

/// Core trait for model adapters.
///
/// Adapters bridge the gap between task interfaces and concrete model implementations.
/// They handle model-specific preprocessing, inference, and postprocessing while
/// conforming to the task's typed input/output contract.
pub trait ModelAdapter: Send + Sync + Debug {
    /// The task type this adapter executes
    type Task: Task;

    /// Returns information about this adapter.
    fn info(&self) -> AdapterInfo;

    /// Executes the model on the given input.
    ///
    /// This method handles the complete pipeline:
    /// 1. Validate input
    /// 2. Preprocess
    /// 3. Run inference
    /// 4. Postprocess
    /// 5. Validate output
    ///
    /// # Arguments
    ///
    /// * `input` - The task input to process
    /// * `config` - Optional configuration for execution
    ///
    /// # Returns
    ///
    /// The task output or an error
    fn execute(
        &self,
        input: <Self::Task as Task>::Input,
        config: Option<&<Self::Task as Task>::Config>,
    ) -> Result<<Self::Task as Task>::Output, OCRError>;

    /// Returns whether this adapter can handle batched inputs efficiently.
    fn supports_batching(&self) -> bool {
        true // Most models support batching
    }

    /// Returns the recommended batch size for this adapter.
    fn recommended_batch_size(&self) -> usize {
        6 // Default from constants
    }
}

/// Builder trait for creating model adapters.
///
/// This trait defines the interface for building adapters with specific configurations.
pub trait AdapterBuilder: Sized {
    /// The configuration type for this builder
    type Config: Send + Sync + Debug + Clone;

    /// The adapter type that this builder creates
    type Adapter: ModelAdapter;

    /// Builds an adapter from a model source.
    ///
    /// # Arguments
    ///
    /// * `model_source` - Model file path or in-memory model bytes
    ///
    /// # Returns
    ///
    /// The built adapter or an error
    fn build(
        self,
        model_source: impl Into<crate::core::inference::ModelSource>,
    ) -> Result<Self::Adapter, OCRError>;

    /// Configures the builder with the given configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The configuration to use
    ///
    /// # Returns
    ///
    /// The configured builder
    fn with_config(self, config: Self::Config) -> Self;

    /// Returns the adapter type identifier.
    fn adapter_type(&self) -> &str;
}

/// Trait for adapter builders that support ONNX Runtime session configuration.
///
/// This trait is implemented by builders that can be configured with ORT session
/// settings like execution providers, thread count, and memory optimization.
pub trait OrtConfigurable: Sized {
    /// Configures the builder with ONNX Runtime session settings.
    fn with_ort_config(self, config: crate::core::config::OrtSessionConfig) -> Self;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adapter_info_creation() {
        let info = AdapterInfo::new(
            "DB",
            TaskType::TextDetection,
            "Differentiable Binarization text detector",
        );

        assert_eq!(info.model_name, "DB");
        assert_eq!(info.task_type, TaskType::TextDetection);
    }
}
