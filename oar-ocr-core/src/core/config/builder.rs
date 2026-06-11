//! Model inference configuration types and utilities.

use super::errors::{ConfigError, ConfigValidator};
use super::onnx::OrtSessionConfig;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Configuration for model inference and ONNX Runtime session setup.
///
/// This struct provides configuration options for building ONNX inference engines,
/// including runtime settings and model metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelInferenceConfig {
    /// The path to the model file (optional).
    pub model_path: Option<PathBuf>,
    /// The name of the model (optional).
    pub model_name: Option<String>,
    /// The batch size for processing (optional).
    pub batch_size: Option<usize>,
    /// Whether to enable logging (optional).
    pub enable_logging: Option<bool>,
    /// ONNX Runtime session configuration for this model (optional).
    #[serde(default)]
    pub ort_session: Option<OrtSessionConfig>,
}

impl ModelInferenceConfig {
    /// Creates a new config with all fields unset.
    pub fn new() -> Self {
        Self {
            model_path: None,
            model_name: None,
            batch_size: None,
            enable_logging: None,
            ort_session: None,
        }
    }

    /// Creates a config with the given model name and batch size, logging enabled.
    pub fn with_defaults(model_name: Option<String>, batch_size: Option<usize>) -> Self {
        Self {
            model_path: None,
            model_name,
            batch_size,
            enable_logging: Some(true),
            ort_session: None,
        }
    }

    /// Creates a config with the given model path, logging enabled.
    pub fn with_model_path(model_path: PathBuf) -> Self {
        Self {
            model_path: Some(model_path),
            model_name: None,
            batch_size: None,
            enable_logging: Some(true),
            ort_session: None,
        }
    }

    /// Sets the model path.
    pub fn model_path(mut self, model_path: impl Into<PathBuf>) -> Self {
        self.model_path = Some(model_path.into());
        self
    }

    /// Sets the model name.
    pub fn model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = Some(model_name.into());
        self
    }

    /// Sets the batch size.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Enables or disables logging.
    pub fn enable_logging(mut self, enable: bool) -> Self {
        self.enable_logging = Some(enable);
        self
    }

    /// Returns whether logging is enabled (defaults to true).
    pub fn get_enable_logging(&self) -> bool {
        self.enable_logging.unwrap_or(true)
    }

    /// Sets the ONNX Runtime session configuration.
    pub fn ort_session(mut self, cfg: OrtSessionConfig) -> Self {
        self.ort_session = Some(cfg);
        self
    }

    /// Validates the configuration.
    pub fn validate(&self) -> Result<(), ConfigError> {
        ConfigValidator::validate(self)
    }

    /// Merges in another config; its set fields override this one's.
    pub fn merge_with(mut self, other: &ModelInferenceConfig) -> Self {
        if other.model_path.is_some() {
            self.model_path = other.model_path.clone();
        }
        if other.model_name.is_some() {
            self.model_name = other.model_name.clone();
        }
        if other.batch_size.is_some() {
            self.batch_size = other.batch_size;
        }
        if other.enable_logging.is_some() {
            self.enable_logging = other.enable_logging;
        }
        if other.ort_session.is_some() {
            self.ort_session = other.ort_session.clone();
        }
        self
    }

    /// Effective batch size, defaulting to 1.
    pub fn get_batch_size(&self) -> usize {
        self.batch_size.unwrap_or(1)
    }

    /// Model name, defaulting to `"unnamed_model"`.
    pub fn get_model_name(&self) -> String {
        self.model_name
            .clone()
            .unwrap_or_else(|| "unnamed_model".to_string())
    }
}

impl ConfigValidator for ModelInferenceConfig {
    /// Validates the batch size (if set) and that the model path exists (if set).
    fn validate(&self) -> Result<(), ConfigError> {
        if let Some(batch_size) = self.batch_size {
            self.validate_batch_size_with_limits(batch_size, 1000)?;
        }

        if let Some(model_path) = &self.model_path {
            self.validate_model_path(model_path)?;
        }

        Ok(())
    }

    /// Returns the default configuration.
    fn get_defaults() -> Self {
        Self {
            model_path: None,
            model_name: Some("default_model".to_string()),
            batch_size: Some(32),
            enable_logging: Some(false),
            ort_session: None,
        }
    }
}

impl Default for ModelInferenceConfig {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_common_builder_config_builder_pattern() {
        let ort_cfg = OrtSessionConfig::default();
        let config = ModelInferenceConfig::new()
            .model_name("test_model")
            .batch_size(16)
            .enable_logging(true)
            .ort_session(ort_cfg);

        assert_eq!(config.model_name, Some("test_model".to_string()));
        assert_eq!(config.batch_size, Some(16));
        assert_eq!(config.enable_logging, Some(true));
        assert!(config.ort_session.is_some());
    }

    #[test]
    fn test_common_builder_config_merge() {
        let config1 = ModelInferenceConfig::new()
            .model_name("model1")
            .batch_size(8);
        let config2 = ModelInferenceConfig::new()
            .model_name("model2")
            .enable_logging(true);

        let merged = config1.merge_with(&config2);
        assert_eq!(merged.model_name, Some("model2".to_string()));
        assert_eq!(merged.batch_size, Some(8));
        assert_eq!(merged.enable_logging, Some(true));
    }

    #[test]
    fn test_common_builder_config_getters() {
        let ort_cfg = OrtSessionConfig::default();
        let config = ModelInferenceConfig::new()
            .model_name("test")
            .batch_size(16)
            .ort_session(ort_cfg);

        assert_eq!(config.get_model_name(), "test");
        assert_eq!(config.get_batch_size(), 16);
        assert!(config.get_enable_logging()); // Default is true
    }

    #[test]
    fn test_common_builder_config_validation() {
        let ort_cfg = OrtSessionConfig::default();
        let valid_config = ModelInferenceConfig::new()
            .batch_size(16)
            .ort_session(ort_cfg);
        assert!(valid_config.validate().is_ok());

        let invalid_batch_config = ModelInferenceConfig::new().batch_size(0);
        assert!(invalid_batch_config.validate().is_err());
    }
}
