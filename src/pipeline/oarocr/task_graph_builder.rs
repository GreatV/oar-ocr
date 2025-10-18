//! Task graph builder for constructing pipeline components from configuration.
//!
//! This module provides the TaskGraphBuilder which replaces the old ComponentBuilder,
//! instantiating TaskRunner<T> from configuration and organizing stage orchestration
//! around task interfaces.

use crate::core::traits::{TaskType, adapter::AdapterBuilder};
use crate::core::{ModelRegistry, OCRError};
use crate::models::classification::{
    DocOrientationAdapterBuilder, TextLineOrientationAdapterBuilder,
};
use crate::models::detection::DBTextDetectionAdapterBuilder;
use crate::models::recognition::CRNNTextRecognitionAdapterBuilder;
use crate::models::rectification::DoctrRectifierAdapterBuilder;
use crate::pipeline::oarocr::task_graph_config::{ModelBinding, TaskGraphConfig};
use std::sync::Arc;

/// Builder for constructing task graphs from configuration.
pub struct TaskGraphBuilder {
    /// The task graph configuration
    config: TaskGraphConfig,

    /// Model registry for adapter lookup
    registry: Arc<ModelRegistry>,
}

impl TaskGraphBuilder {
    /// Creates a new task graph builder.
    pub fn new(config: TaskGraphConfig) -> Self {
        Self {
            config,
            registry: Arc::new(ModelRegistry::new()),
        }
    }

    /// Creates a new task graph builder with an existing registry.
    pub fn with_registry(config: TaskGraphConfig, registry: Arc<ModelRegistry>) -> Self {
        Self { config, registry }
    }

    /// Validates the task graph configuration.
    pub fn validate(&self) -> Result<(), OCRError> {
        self.config.validate().map_err(|e| OCRError::ConfigError {
            message: format!("Task graph validation failed: {}", e),
        })
    }

    /// Builds and registers all model adapters from the configuration.
    pub fn build_adapters(&self) -> Result<(), OCRError> {
        // Iterate through all model bindings and create adapters
        for (name, binding) in &self.config.model_bindings {
            self.build_and_register_adapter(name, binding)?;
        }

        Ok(())
    }

    /// Builds and registers a single adapter.
    fn build_and_register_adapter(
        &self,
        name: &str,
        binding: &ModelBinding,
    ) -> Result<(), OCRError> {
        match binding.task_type {
            TaskType::TextDetection => {
                self.build_detection_adapter(name, binding)?;
            }
            TaskType::TextRecognition => {
                self.build_recognition_adapter(name, binding)?;
            }
            TaskType::DocumentOrientation => {
                self.build_doc_orientation_adapter(name, binding)?;
            }
            TaskType::TextLineOrientation => {
                self.build_text_line_orientation_adapter(name, binding)?;
            }
            TaskType::DocumentRectification => {
                self.build_rectification_adapter(name, binding)?;
            }
            TaskType::LayoutDetection => {
                return Err(OCRError::ConfigError {
                    message: format!(
                        "Layout detection adapter not yet implemented for model '{}'",
                        name
                    ),
                });
            }
        }

        Ok(())
    }

    /// Builds a text detection adapter.
    fn build_detection_adapter(&self, _name: &str, binding: &ModelBinding) -> Result<(), OCRError> {
        let mut builder = DBTextDetectionAdapterBuilder::new();

        // Apply configuration if provided
        if let Some(session_pool_size) = binding.session_pool_size {
            builder = builder.session_pool_size(session_pool_size);
        }

        // Build the adapter
        let adapter = builder.build(&binding.model_path)?;

        // Register in the registry
        self.registry.register(adapter)?;

        Ok(())
    }

    /// Builds a text recognition adapter.
    fn build_recognition_adapter(
        &self,
        _name: &str,
        binding: &ModelBinding,
    ) -> Result<(), OCRError> {
        // Get character dictionary path
        let char_dict_path =
            self.config
                .character_dict_path
                .as_ref()
                .ok_or_else(|| OCRError::ConfigError {
                    message: "Character dictionary path required for text recognition".to_string(),
                })?;

        // Load character dictionary
        let character_dict =
            Self::load_character_dict(char_dict_path.to_str().ok_or_else(|| {
                OCRError::ConfigError {
                    message: "Invalid character dictionary path".to_string(),
                }
            })?)?;

        let mut builder = CRNNTextRecognitionAdapterBuilder::new().character_dict(character_dict);

        // Apply configuration if provided
        if let Some(session_pool_size) = binding.session_pool_size {
            builder = builder.session_pool_size(session_pool_size);
        }

        // Build the adapter
        let adapter = builder.build(&binding.model_path)?;

        // Register in the registry
        self.registry.register(adapter)?;

        Ok(())
    }

    /// Builds a document orientation adapter.
    fn build_doc_orientation_adapter(
        &self,
        _name: &str,
        binding: &ModelBinding,
    ) -> Result<(), OCRError> {
        let mut builder = DocOrientationAdapterBuilder::new();

        // Apply configuration if provided
        if let Some(session_pool_size) = binding.session_pool_size {
            builder = builder.session_pool_size(session_pool_size);
        }

        // Build the adapter
        let adapter = builder.build(&binding.model_path)?;

        // Register in the registry
        self.registry.register(adapter)?;

        Ok(())
    }

    /// Builds a text line orientation adapter.
    fn build_text_line_orientation_adapter(
        &self,
        _name: &str,
        binding: &ModelBinding,
    ) -> Result<(), OCRError> {
        let mut builder = TextLineOrientationAdapterBuilder::new();

        // Apply configuration if provided
        if let Some(session_pool_size) = binding.session_pool_size {
            builder = builder.session_pool_size(session_pool_size);
        }

        // Build the adapter
        let adapter = builder.build(&binding.model_path)?;

        // Register in the registry
        self.registry.register(adapter)?;

        Ok(())
    }

    /// Builds a document rectification adapter.
    fn build_rectification_adapter(
        &self,
        _name: &str,
        binding: &ModelBinding,
    ) -> Result<(), OCRError> {
        let mut builder = DoctrRectifierAdapterBuilder::new();

        // Apply configuration if provided
        if let Some(session_pool_size) = binding.session_pool_size {
            builder = builder.session_pool_size(session_pool_size);
        }

        // Build the adapter
        let adapter = builder.build(&binding.model_path)?;

        // Register in the registry
        self.registry.register(adapter)?;

        Ok(())
    }

    /// Loads the character dictionary from a file.
    fn load_character_dict(dict_path: &str) -> Result<Vec<String>, OCRError> {
        let content = std::fs::read_to_string(dict_path).map_err(|e| OCRError::ConfigError {
            message: format!("Failed to load character dictionary from {dict_path}: {e}"),
        })?;

        Ok(content.lines().map(|line| line.to_string()).collect())
    }

    /// Returns the model registry.
    pub fn registry(&self) -> Arc<ModelRegistry> {
        self.registry.clone()
    }

    /// Returns the task graph configuration.
    pub fn config(&self) -> &TaskGraphConfig {
        &self.config
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_task_graph_builder_creation() {
        let config = TaskGraphConfig::new();
        let builder = TaskGraphBuilder::new(config);
        assert!(builder.validate().is_ok());
    }
}
