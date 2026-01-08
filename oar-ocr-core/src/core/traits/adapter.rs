//! Model adapter trait definitions for the OCR pipeline.
//!
//! This module defines the `ModelAdapter` trait and related types that adapt
//! various model implementations to conform to task interfaces. Adapters handle
//! preprocessing, inference, and postprocessing for specific models.

use super::task::{Task, TaskSchema, TaskType};
use crate::core::OCRError;
use std::fmt::Debug;
use std::path::Path;

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
/// conforming to the task's input/output schema.
pub trait ModelAdapter: Send + Sync + Debug {
    /// The task type this adapter executes
    type Task: Task;

    /// Returns information about this adapter.
    fn info(&self) -> AdapterInfo;

    /// Returns the schema that this adapter conforms to.
    fn schema(&self) -> TaskSchema {
        TaskSchema::new(
            self.info().task_type,
            vec!["image".to_string()], // Most adapters work with images
            vec!["result".to_string()],
        )
    }

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

    /// Validates that this adapter is compatible with the given task schema.
    ///
    /// # Arguments
    ///
    /// * `schema` - The schema to check compatibility with
    ///
    /// # Returns
    ///
    /// Result indicating success or incompatibility error
    fn validate_compatibility(&self, schema: &TaskSchema) -> Result<(), OCRError> {
        let adapter_schema = self.schema();
        if adapter_schema.task_type != schema.task_type {
            return Err(OCRError::ConfigError {
                message: format!(
                    "Adapter task type {:?} does not match required task type {:?}",
                    adapter_schema.task_type, schema.task_type
                ),
            });
        }
        Ok(())
    }

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

    /// Builds an adapter from a model file.
    ///
    /// # Arguments
    ///
    /// * `model_path` - Path to the model file (e.g., ONNX file)
    ///
    /// # Returns
    ///
    /// The built adapter or an error
    fn build(self, model_path: &Path) -> Result<Self::Adapter, OCRError>;

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

/// A wrapper that implements Task for an adapter's task type.
///
/// This allows adapters to be used polymorphically through the Task trait.
#[derive(Debug)]
pub struct AdapterTask<A: ModelAdapter> {
    adapter: A,
}

impl<A: ModelAdapter> AdapterTask<A> {
    /// Creates a new adapter task.
    pub fn new(adapter: A) -> Self {
        Self { adapter }
    }

    /// Returns a reference to the adapter.
    pub fn adapter(&self) -> &A {
        &self.adapter
    }
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

    #[test]
    fn test_schema_validation() {
        // This is a conceptual test - actual validation would be done with real adapters
        let schema = TaskSchema::new(
            TaskType::TextDetection,
            vec!["image".to_string()],
            vec!["text_boxes".to_string()],
        );

        assert_eq!(schema.task_type, TaskType::TextDetection);
        assert_eq!(schema.input_types, vec!["image".to_string()]);
    }
}
