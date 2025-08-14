//! Common builder configuration types and utilities.

use super::errors::{ConfigError, ConfigValidator};
use super::onnx::OrtSessionConfig;
use serde::{Deserialize, Serialize};
use std::path::PathBuf;

/// Common configuration for model builders.
///
/// This struct contains configuration options that are common across different
/// types of model builders in the OCR pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CommonBuilderConfig {
    /// The path to the model file (optional).
    pub model_path: Option<PathBuf>,
    /// The name of the model (optional).
    pub model_name: Option<String>,
    /// The batch size for processing (optional).
    pub batch_size: Option<usize>,
    /// Whether to enable logging (optional).
    pub enable_logging: Option<bool>,
    /// ONNX Runtime session configuration for this model (optional)
    #[serde(default)]
    pub ort_session: Option<OrtSessionConfig>,
    /// Size of the pinned session pool to allow concurrent predictions (>=1)
    /// If None, defaults to 1 (single session)
    #[serde(default)]
    pub session_pool_size: Option<usize>,
}

impl CommonBuilderConfig {
    /// Creates a new CommonBuilderConfig with default values.
    ///
    /// # Returns
    ///
    /// A new CommonBuilderConfig instance.
    pub fn new() -> Self {
        Self {
            model_path: None,
            model_name: None,
            batch_size: None,
            enable_logging: None,
            ort_session: None,
            session_pool_size: Some(1),
        }
    }

    /// Creates a new CommonBuilderConfig with default values for model name and batch size.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model (optional).
    /// * `batch_size` - The batch size for processing (optional).
    ///
    /// # Returns
    ///
    /// A new CommonBuilderConfig instance.
    pub fn with_defaults(model_name: Option<String>, batch_size: Option<usize>) -> Self {
        Self {
            model_path: None,
            model_name,
            batch_size,
            enable_logging: Some(true),
            ort_session: None,
            session_pool_size: Some(1),
        }
    }

    /// Creates a new CommonBuilderConfig with a model path.
    ///
    /// # Arguments
    ///
    /// * `model_path` - The path to the model file.
    ///
    /// # Returns
    ///
    /// A new CommonBuilderConfig instance.
    pub fn with_model_path(model_path: PathBuf) -> Self {
        Self {
            model_path: Some(model_path),
            model_name: None,
            batch_size: None,
            enable_logging: Some(true),
            ort_session: None,
            session_pool_size: Some(1),
        }
    }

    /// Sets the model path for the configuration.
    ///
    /// # Arguments
    ///
    /// * `model_path` - The path to the model file.
    ///
    /// # Returns
    ///
    /// The updated CommonBuilderConfig instance.
    pub fn model_path(mut self, model_path: impl Into<PathBuf>) -> Self {
        self.model_path = Some(model_path.into());
        self
    }

    /// Sets the model name for the configuration.
    ///
    /// # Arguments
    ///
    /// * `model_name` - The name of the model.
    ///
    /// # Returns
    ///
    /// The updated CommonBuilderConfig instance.
    pub fn model_name(mut self, model_name: impl Into<String>) -> Self {
        self.model_name = Some(model_name.into());
        self
    }

    /// Sets the batch size for the configuration.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - The batch size for processing.
    ///
    /// # Returns
    ///
    /// The updated CommonBuilderConfig instance.
    pub fn batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = Some(batch_size);
        self
    }

    /// Sets whether logging is enabled for the configuration.
    ///
    /// # Arguments
    ///
    /// * `enable` - Whether to enable logging.
    ///
    /// # Returns
    ///
    /// The updated CommonBuilderConfig instance.
    pub fn enable_logging(mut self, enable: bool) -> Self {
        self.enable_logging = Some(enable);
        self
    }

    /// Gets whether logging is enabled for the configuration.
    ///
    /// # Returns
    ///
    /// True if logging is enabled, false otherwise.
    pub fn get_enable_logging(&self) -> bool {
        self.enable_logging.unwrap_or(true)
    }

    /// Sets the ORT session configuration.
    ///
    /// # Arguments
    ///
    /// * `cfg` - The ONNX Runtime session configuration.
    ///
    /// # Returns
    ///
    /// The updated CommonBuilderConfig instance.
    pub fn ort_session(mut self, cfg: OrtSessionConfig) -> Self {
        self.ort_session = Some(cfg);
        self
    }

    /// Sets the session pool size used for concurrent predictions (>=1).
    ///
    /// # Arguments
    ///
    /// * `size` - The session pool size (minimum 1).
    ///
    /// # Returns
    ///
    /// The updated CommonBuilderConfig instance.
    pub fn session_pool_size(mut self, size: usize) -> Self {
        self.session_pool_size = Some(size);
        self
    }

    /// Validates the configuration.
    ///
    /// # Returns
    ///
    /// A Result indicating success or a ConfigError if validation fails.
    pub fn validate(&self) -> Result<(), ConfigError> {
        ConfigValidator::validate(self)
    }

    /// Merges this configuration with another configuration.
    ///
    /// Values from the other configuration will override values in this configuration
    /// if they are present in the other configuration.
    ///
    /// # Arguments
    ///
    /// * `other` - The other configuration to merge with.
    ///
    /// # Returns
    ///
    /// The updated CommonBuilderConfig instance.
    pub fn merge_with(mut self, other: &CommonBuilderConfig) -> Self {
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
        if other.session_pool_size.is_some() {
            self.session_pool_size = other.session_pool_size;
        }
        self
    }

    /// Gets the effective batch size.
    ///
    /// # Returns
    ///
    /// The batch size, or a default value if not set.
    pub fn get_batch_size(&self) -> usize {
        self.batch_size.unwrap_or(1)
    }

    /// Gets the effective session pool size.
    ///
    /// # Returns
    ///
    /// The session pool size, or a default value if not set.
    pub fn get_session_pool_size(&self) -> usize {
        self.session_pool_size.unwrap_or(1)
    }

    /// Gets the model name.
    ///
    /// # Returns
    ///
    /// The model name, or a default value if not set.
    pub fn get_model_name(&self) -> String {
        self.model_name
            .clone()
            .unwrap_or_else(|| "unnamed_model".to_string())
    }
}

impl ConfigValidator for CommonBuilderConfig {
    /// Validates the configuration.
    ///
    /// This method checks that the batch size is valid and that the model path exists
    /// if it is specified.
    ///
    /// # Returns
    ///
    /// A Result indicating success or a ConfigError if validation fails.
    fn validate(&self) -> Result<(), ConfigError> {
        if let Some(batch_size) = self.batch_size {
            self.validate_batch_size_with_limits(batch_size, 1000)?;
        }

        if let Some(model_path) = &self.model_path {
            self.validate_model_path(model_path)?;
        }

        if let Some(pool) = self.session_pool_size
            && pool == 0
        {
            return Err(ConfigError::InvalidConfig {
                message: "session_pool_size must be >= 1".to_string(),
            });
        }

        Ok(())
    }

    /// Returns the default configuration.
    ///
    /// # Returns
    ///
    /// The default CommonBuilderConfig instance.
    fn get_defaults() -> Self {
        Self {
            model_path: None,
            model_name: Some("default_model".to_string()),
            batch_size: Some(32),
            enable_logging: Some(false),
            ort_session: None,
            session_pool_size: Some(1),
        }
    }
}

impl Default for CommonBuilderConfig {
    /// This allows CommonBuilderConfig to be created with default values.
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_common_builder_config_builder_pattern() {
        let config = CommonBuilderConfig::new()
            .model_name("test_model")
            .batch_size(16)
            .enable_logging(true)
            .session_pool_size(4);

        assert_eq!(config.model_name, Some("test_model".to_string()));
        assert_eq!(config.batch_size, Some(16));
        assert_eq!(config.enable_logging, Some(true));
        assert_eq!(config.session_pool_size, Some(4));
    }

    #[test]
    fn test_common_builder_config_merge() {
        let config1 = CommonBuilderConfig::new()
            .model_name("model1")
            .batch_size(8);
        let config2 = CommonBuilderConfig::new()
            .model_name("model2")
            .enable_logging(true);

        let merged = config1.merge_with(&config2);
        assert_eq!(merged.model_name, Some("model2".to_string()));
        assert_eq!(merged.batch_size, Some(8));
        assert_eq!(merged.enable_logging, Some(true));
    }

    #[test]
    fn test_common_builder_config_getters() {
        let config = CommonBuilderConfig::new()
            .model_name("test")
            .batch_size(16)
            .session_pool_size(2);

        assert_eq!(config.get_model_name(), "test");
        assert_eq!(config.get_batch_size(), 16);
        assert_eq!(config.get_session_pool_size(), 2);
        assert!(config.get_enable_logging()); // Default is true
    }

    #[test]
    fn test_common_builder_config_validation() {
        let valid_config = CommonBuilderConfig::new()
            .batch_size(16)
            .session_pool_size(2);
        assert!(valid_config.validate().is_ok());

        let invalid_batch_config = CommonBuilderConfig::new().batch_size(0);
        assert!(invalid_batch_config.validate().is_err());

        let invalid_pool_config = CommonBuilderConfig::new().session_pool_size(0);
        assert!(invalid_pool_config.validate().is_err());
    }
}
