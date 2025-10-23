//! Task graph builder for constructing pipeline components from configuration.
//!
//! This module provides the TaskGraphBuilder which replaces the old ComponentBuilder,
//! instantiating TaskRunner<T> from configuration and organizing stage orchestration
//! around task interfaces.

use crate::core::traits::{TaskType, adapter::AdapterBuilder};
use crate::core::{ModelRegistry, OCRError};
use crate::domain::adapters::{
    DocumentOrientationAdapterBuilder, PPFormulaNetAdapterBuilder, SealTextDetectionAdapterBuilder,
    TableClassificationAdapterBuilder, TextDetectionAdapterBuilder,
    TextLineOrientationAdapterBuilder, TextRecognitionAdapterBuilder, UVDocRectifierAdapterBuilder,
    UniMERNetFormulaAdapterBuilder,
};
use crate::domain::tasks::FormulaRecognitionConfig;
use crate::pipeline::oarocr::task_graph_config::{ModelBinding, TaskGraphConfig};
use serde::Deserialize;
use std::path::PathBuf;
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
            TaskType::SealTextDetection => {
                self.build_seal_detection_adapter(name, binding)?;
            }
            TaskType::LayoutDetection => {
                return Err(OCRError::ConfigError {
                    message: format!(
                        "Layout detection adapter not yet implemented for model '{}'",
                        name
                    ),
                });
            }
            TaskType::TableCellDetection => {
                return Err(OCRError::ConfigError {
                    message: format!(
                        "Table cell detection adapter not yet implemented for model '{}'",
                        name
                    ),
                });
            }
            TaskType::TableClassification => {
                self.build_table_classification_adapter(name, binding)?;
            }
            TaskType::FormulaRecognition => {
                self.build_formula_adapter(name, binding)?;
            }
        }

        Ok(())
    }

    /// Builds a formula recognition adapter.
    fn build_formula_adapter(&self, name: &str, binding: &ModelBinding) -> Result<(), OCRError> {
        let settings: FormulaAdapterSettings = binding
            .config
            .as_ref()
            .map(|value| {
                serde_json::from_value(value.clone()).map_err(|err| OCRError::ConfigError {
                    message: format!(
                        "Invalid formula recognition config for model '{}': {}",
                        name, err
                    ),
                })
            })
            .transpose()?
            .unwrap_or_default();

        let variant_source = settings
            .model_type
            .clone()
            .unwrap_or_else(|| binding.model_name.clone());

        let variant = FormulaModelVariant::detect(&variant_source).ok_or_else(|| {
            OCRError::ConfigError {
                message: format!(
                    "Unsupported formula recognition model type '{}' for binding '{}'. \
                    Specify 'model_type' in the binding config (e.g., 'unimernet', 'pp-formulanet').",
                    variant_source, name
                ),
            }
        })?;

        let mut task_config = FormulaRecognitionConfig::default();
        if let Some(score_threshold) = settings.score_threshold {
            task_config.score_threshold = score_threshold;
        }
        if let Some(max_length) = settings.max_length {
            task_config.max_length = max_length;
        }

        let target_size = match (settings.target_width, settings.target_height) {
            (Some(width), Some(height)) => Some((width, height)),
            (Some(_), None) | (None, Some(_)) => {
                return Err(OCRError::ConfigError {
                    message: format!(
                        "Formula model '{}' must specify both 'target_width' and 'target_height' \
                         or neither.",
                        name
                    ),
                });
            }
            _ => None,
        };

        let tokenizer_path = settings.tokenizer_path.clone();
        let session_pool_size = binding.session_pool_size;

        match variant {
            FormulaModelVariant::UniMERNet => {
                let mut builder =
                    UniMERNetFormulaAdapterBuilder::new().with_config(task_config.clone());
                if let Some(session_pool_size) = session_pool_size {
                    builder = builder.session_pool_size(session_pool_size);
                }
                builder = builder.model_name(binding.model_name.clone());
                if let Some((width, height)) = target_size {
                    builder = builder.target_size(width, height);
                }
                if let Some(path) = tokenizer_path.as_ref() {
                    builder = builder.tokenizer_path(path.clone());
                }
                let adapter = builder.build(&binding.model_path)?;
                self.registry.register_with_id(name.to_string(), adapter)?;
            }
            FormulaModelVariant::PPFormulaNet => {
                let mut builder =
                    PPFormulaNetAdapterBuilder::new().with_config(task_config.clone());
                if let Some(session_pool_size) = session_pool_size {
                    builder = builder.session_pool_size(session_pool_size);
                }
                builder = builder.model_name(binding.model_name.clone());
                if let Some((width, height)) = target_size {
                    builder = builder.target_size(width, height);
                }
                if let Some(path) = tokenizer_path.as_ref() {
                    builder = builder.tokenizer_path(path.clone());
                }
                let adapter = builder.build(&binding.model_path)?;
                self.registry.register_with_id(name.to_string(), adapter)?;
            }
        }

        Ok(())
    }

    /// Builds a text detection adapter.
    fn build_detection_adapter(&self, name: &str, binding: &ModelBinding) -> Result<(), OCRError> {
        let mut builder = TextDetectionAdapterBuilder::new();

        // Apply configuration if provided
        if let Some(session_pool_size) = binding.session_pool_size {
            builder = builder.session_pool_size(session_pool_size);
        }

        // Build the adapter
        let adapter = builder.build(&binding.model_path)?;

        // Register in the registry using the binding name as identifier
        self.registry.register_with_id(name.to_string(), adapter)?;

        Ok(())
    }

    /// Builds a seal text detection adapter.
    fn build_seal_detection_adapter(
        &self,
        name: &str,
        binding: &ModelBinding,
    ) -> Result<(), OCRError> {
        let mut builder = SealTextDetectionAdapterBuilder::new();

        // Apply configuration if provided
        if let Some(session_pool_size) = binding.session_pool_size {
            builder = builder.session_pool_size(session_pool_size);
        }

        // Build the adapter
        let adapter = builder.build(&binding.model_path)?;

        // Register in the registry using the binding name as identifier
        self.registry.register_with_id(name.to_string(), adapter)?;

        Ok(())
    }

    /// Builds a text recognition adapter.
    fn build_recognition_adapter(
        &self,
        name: &str,
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

        let mut builder = TextRecognitionAdapterBuilder::new().character_dict(character_dict);

        // Apply configuration if provided
        if let Some(session_pool_size) = binding.session_pool_size {
            builder = builder.session_pool_size(session_pool_size);
        }

        // Build the adapter
        let adapter = builder.build(&binding.model_path)?;

        // Register in the registry using the binding name as identifier
        self.registry.register_with_id(name.to_string(), adapter)?;

        Ok(())
    }

    /// Builds a document orientation adapter.
    fn build_doc_orientation_adapter(
        &self,
        name: &str,
        binding: &ModelBinding,
    ) -> Result<(), OCRError> {
        let mut builder = DocumentOrientationAdapterBuilder::new();

        // Apply configuration if provided
        if let Some(session_pool_size) = binding.session_pool_size {
            builder = builder.session_pool_size(session_pool_size);
        }

        builder = builder.model_name(binding.model_name.clone());

        // Build the adapter
        let adapter = builder.build(&binding.model_path)?;

        // Register in the registry using the binding name as identifier
        self.registry.register_with_id(name.to_string(), adapter)?;

        Ok(())
    }

    /// Builds a text line orientation adapter.
    fn build_text_line_orientation_adapter(
        &self,
        name: &str,
        binding: &ModelBinding,
    ) -> Result<(), OCRError> {
        let mut builder = TextLineOrientationAdapterBuilder::new();

        // Apply configuration if provided
        if let Some(session_pool_size) = binding.session_pool_size {
            builder = builder.session_pool_size(session_pool_size);
        }

        builder = builder.model_name(binding.model_name.clone());

        // Build the adapter
        let adapter = builder.build(&binding.model_path)?;

        // Register in the registry using the binding name as identifier
        self.registry.register_with_id(name.to_string(), adapter)?;

        Ok(())
    }

    /// Builds a table classification adapter.
    fn build_table_classification_adapter(
        &self,
        name: &str,
        binding: &ModelBinding,
    ) -> Result<(), OCRError> {
        let mut builder = TableClassificationAdapterBuilder::new();

        // Apply configuration if provided
        if let Some(session_pool_size) = binding.session_pool_size {
            builder = builder.session_pool_size(session_pool_size);
        }

        builder = builder.model_name(binding.model_name.clone());

        // Build the adapter
        let adapter = builder.build(&binding.model_path)?;

        // Register in the registry using the binding name as identifier
        self.registry.register_with_id(name.to_string(), adapter)?;

        Ok(())
    }

    /// Builds a document rectification adapter.
    fn build_rectification_adapter(
        &self,
        name: &str,
        binding: &ModelBinding,
    ) -> Result<(), OCRError> {
        let mut builder = UVDocRectifierAdapterBuilder::new();

        // Apply configuration if provided
        if let Some(session_pool_size) = binding.session_pool_size {
            builder = builder.session_pool_size(session_pool_size);
        }

        builder = builder.model_name(binding.model_name.clone());

        // Build the adapter
        let adapter = builder.build(&binding.model_path)?;

        // Register in the registry using the binding name as identifier
        self.registry.register_with_id(name.to_string(), adapter)?;

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

/// Additional configuration for formula adapters parsed from model bindings.
#[derive(Debug, Default, Clone, Deserialize)]
struct FormulaAdapterSettings {
    #[serde(default)]
    model_type: Option<String>,
    #[serde(default)]
    tokenizer_path: Option<PathBuf>,
    #[serde(default)]
    target_width: Option<u32>,
    #[serde(default)]
    target_height: Option<u32>,
    #[serde(default)]
    score_threshold: Option<f32>,
    #[serde(default)]
    max_length: Option<usize>,
}

#[derive(Debug, Clone, Copy)]
enum FormulaModelVariant {
    UniMERNet,
    PPFormulaNet,
}

impl FormulaModelVariant {
    fn detect(source: &str) -> Option<Self> {
        let key = source.to_ascii_lowercase();

        if key.contains("unimernet") {
            Some(Self::UniMERNet)
        } else if key.contains("formula") && (key.contains("pp") || key.contains("net")) {
            Some(Self::PPFormulaNet)
        } else {
            None
        }
    }
}
