//! Configuration error types and validation traits.

use std::path::Path;
use thiserror::Error;

/// Errors that can occur during configuration validation.
///
/// This enum represents various errors that can occur when validating
/// configuration parameters in the OCR pipeline.
#[derive(Error, Debug)]
pub enum ConfigError {
    /// Error indicating that a batch size is invalid (must be greater than 0).
    #[error("batch size must be greater than 0")]
    InvalidBatchSize,

    /// Error indicating that a model path does not exist.
    #[error("model path does not exist: {path}")]
    ModelPathNotFound { path: std::path::PathBuf },

    /// Error indicating that a configuration is invalid.
    #[error("invalid configuration: {message}")]
    InvalidConfig { message: String },

    /// Error indicating that validation failed.
    #[error("validation failed: {message}")]
    ValidationFailed { message: String },

    /// Error indicating that a resource limit has been exceeded.
    #[error("resource limit exceeded: {message}")]
    ResourceLimitExceeded { message: String },
}

/// A trait for configuration types that can provide recommended defaults.
///
/// This trait complements ConfigValidator::get_defaults, allowing generic
/// code to talk about defaults without depending on validation details.
pub trait ConfigDefaults: Sized {
    /// Return the recommended defaults for this configuration type.
    fn defaults() -> Self;
}

// Blanket implementation: any ConfigValidator can provide defaults via get_defaults
impl<T: ConfigValidator> ConfigDefaults for T {
    fn defaults() -> Self {
        T::get_defaults()
    }
}

/// A trait for validating configuration parameters.
///
/// This trait provides methods for validating various configuration parameters
/// used in the OCR pipeline, such as batch sizes, model paths, and image dimensions.
pub trait ConfigValidator {
    /// Validates the configuration.
    ///
    /// This method should be implemented by types that need to validate their configuration.
    ///
    /// # Returns
    ///
    /// A Result indicating success or a ConfigError if validation fails.
    fn validate(&self) -> Result<(), ConfigError>;

    /// Returns the default configuration.
    ///
    /// This method should be implemented by types that have default configuration values.
    ///
    /// # Returns
    ///
    /// The default configuration.
    fn get_defaults() -> Self
    where
        Self: Sized;

    /// Validates a batch size.
    ///
    /// This method checks that the batch size is greater than 0.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - The batch size to validate.
    ///
    /// # Returns
    ///
    /// A Result indicating success or a ConfigError if validation fails.
    fn validate_batch_size(&self, batch_size: usize) -> Result<(), ConfigError> {
        if batch_size == 0 {
            Err(ConfigError::InvalidBatchSize)
        } else {
            Ok(())
        }
    }

    /// Validates a batch size against limits.
    ///
    /// This method checks that the batch size is greater than 0 and does not exceed
    /// the maximum allowed batch size.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - The batch size to validate.
    /// * `max_batch_size` - The maximum allowed batch size.
    ///
    /// # Returns
    ///
    /// A Result indicating success or a ConfigError if validation fails.
    fn validate_batch_size_with_limits(
        &self,
        batch_size: usize,
        max_batch_size: usize,
    ) -> Result<(), ConfigError> {
        if batch_size == 0 {
            return Err(ConfigError::InvalidBatchSize);
        }
        if batch_size > max_batch_size {
            return Err(ConfigError::ResourceLimitExceeded {
                message: format!(
                    "Batch size {} exceeds maximum allowed batch size {}",
                    batch_size, max_batch_size
                ),
            });
        }
        Ok(())
    }

    /// Validates a model path.
    ///
    /// This method checks that the model path exists and is a file.
    ///
    /// # Arguments
    ///
    /// * `path` - The path to validate.
    ///
    /// # Returns
    ///
    /// A Result indicating success or a ConfigError if validation fails.
    fn validate_model_path(&self, path: &Path) -> Result<(), ConfigError> {
        if !path.exists() {
            Err(ConfigError::ModelPathNotFound {
                path: path.to_path_buf(),
            })
        } else if !path.is_file() {
            Err(ConfigError::InvalidConfig {
                message: format!("Model path is not a file: {}", path.display()),
            })
        } else {
            Ok(())
        }
    }

    /// Validates image dimensions.
    ///
    /// This method checks that image dimensions are positive.
    ///
    /// # Arguments
    ///
    /// * `width` - The width to validate.
    /// * `height` - The height to validate.
    ///
    /// # Returns
    ///
    /// A Result indicating success or a ConfigError if validation fails.
    fn validate_image_dimensions(&self, width: u32, height: u32) -> Result<(), ConfigError> {
        if width == 0 || height == 0 {
            Err(ConfigError::InvalidConfig {
                message: "Image dimensions must be positive".to_string(),
            })
        } else {
            Ok(())
        }
    }

    /// Validates a confidence threshold.
    ///
    /// This method checks that the confidence threshold is between 0.0 and 1.0.
    ///
    /// # Arguments
    ///
    /// * `threshold` - The threshold to validate.
    ///
    /// # Returns
    ///
    /// A Result indicating success or a ConfigError if validation fails.
    fn validate_confidence_threshold(&self, threshold: f32) -> Result<(), ConfigError> {
        if !(0.0..=1.0).contains(&threshold) {
            Err(ConfigError::InvalidConfig {
                message: format!(
                    "Confidence threshold must be between 0.0 and 1.0, got {}",
                    threshold
                ),
            })
        } else {
            Ok(())
        }
    }

    /// Validates a memory limit.
    ///
    /// This method checks that the memory limit is reasonable.
    ///
    /// # Arguments
    ///
    /// * `limit_mb` - The memory limit in megabytes to validate.
    ///
    /// # Returns
    ///
    /// A Result indicating success or a ConfigError if validation fails.
    fn validate_memory_limit(&self, limit_mb: usize) -> Result<(), ConfigError> {
        const MAX_REASONABLE_MEMORY_MB: usize = 32 * 1024; // 32 GB

        if limit_mb > MAX_REASONABLE_MEMORY_MB {
            Err(ConfigError::ResourceLimitExceeded {
                message: format!(
                    "Memory limit {} MB exceeds reasonable maximum of {} MB",
                    limit_mb, MAX_REASONABLE_MEMORY_MB
                ),
            })
        } else {
            Ok(())
        }
    }

    /// Validates thread count.
    ///
    /// This method checks that the thread count is reasonable.
    ///
    /// # Arguments
    ///
    /// * `thread_count` - The thread count to validate.
    ///
    /// # Returns
    ///
    /// A Result indicating success or a ConfigError if validation fails.
    fn validate_thread_count(&self, thread_count: usize) -> Result<(), ConfigError> {
        const MAX_REASONABLE_THREADS: usize = 256;

        if thread_count == 0 {
            Err(ConfigError::InvalidConfig {
                message: "Thread count must be greater than 0".to_string(),
            })
        } else if thread_count > MAX_REASONABLE_THREADS {
            Err(ConfigError::ResourceLimitExceeded {
                message: format!(
                    "Thread count {} exceeds reasonable maximum of {}",
                    thread_count, MAX_REASONABLE_THREADS
                ),
            })
        } else {
            Ok(())
        }
    }

    /// Validates a float value is within a specified range.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to validate.
    /// * `min` - The minimum allowed value (inclusive).
    /// * `max` - The maximum allowed value (inclusive).
    /// * `field_name` - The name of the field being validated.
    ///
    /// # Returns
    ///
    /// A Result indicating success or a ConfigError if validation fails.
    fn validate_f32_range(
        &self,
        value: f32,
        min: f32,
        max: f32,
        field_name: &str,
    ) -> Result<(), ConfigError> {
        if value < min || value > max {
            Err(ConfigError::InvalidConfig {
                message: format!(
                    "{} must be between {} and {}, got {}",
                    field_name, min, max, value
                ),
            })
        } else {
            Ok(())
        }
    }

    /// Validates a float value is positive.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to validate.
    /// * `field_name` - The name of the field being validated.
    ///
    /// # Returns
    ///
    /// A Result indicating success or a ConfigError if validation fails.
    fn validate_positive_f32(&self, value: f32, field_name: &str) -> Result<(), ConfigError> {
        if value <= 0.0 {
            Err(ConfigError::InvalidConfig {
                message: format!("{} must be greater than 0, got {}", field_name, value),
            })
        } else {
            Ok(())
        }
    }

    /// Validates a usize value is positive.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to validate.
    /// * `field_name` - The name of the field being validated.
    ///
    /// # Returns
    ///
    /// A Result indicating success or a ConfigError if validation fails.
    fn validate_positive_usize(&self, value: usize, field_name: &str) -> Result<(), ConfigError> {
        if value == 0 {
            Err(ConfigError::InvalidConfig {
                message: format!("{} must be greater than 0, got {}", field_name, value),
            })
        } else {
            Ok(())
        }
    }
}

/// Provides a default implementation of ConfigValidator for any type.
///
/// This implementation provides basic validation methods that can be used
/// by any type that implements ConfigValidator.
pub struct DefaultValidator;

impl ConfigValidator for DefaultValidator {
    fn validate(&self) -> Result<(), ConfigError> {
        Ok(())
    }

    fn get_defaults() -> Self {
        DefaultValidator
    }
}

/// Extension trait for ConfigValidator that provides error wrapping utilities.
///
/// This trait extends ConfigValidator to provide convenient methods for wrapping
/// validation errors into OCRError types, reducing duplication across the codebase.
pub trait ConfigValidatorExt: ConfigValidator {
    /// Validates configuration and wraps any errors into OCRError::ConfigError.
    ///
    /// This method provides a convenient way to validate configuration and
    /// automatically wrap any validation errors into the appropriate OCRError type.
    /// This eliminates the repeated `config.validate().map_err(|e| OCRError::ConfigError { message: e.to_string() })`
    /// pattern found throughout the codebase.
    ///
    /// # Returns
    ///
    /// A Result indicating success or an OCRError if validation fails.
    fn validate_and_wrap_ocr_error(self) -> Result<Self, super::super::errors::OCRError>
    where
        Self: Sized,
    {
        self.validate()
            .map_err(|e| super::super::errors::OCRError::ConfigError {
                message: e.to_string(),
            })?;
        Ok(self)
    }

    /// Validates configuration and wraps any errors into a generic error.
    ///
    /// This method provides a convenient way to validate configuration when
    /// working with generic error types.
    ///
    /// # Returns
    ///
    /// A Result indicating success or a wrapped error if validation fails.
    fn validate_and_wrap_generic(self) -> Result<Self, Box<dyn std::error::Error + Send + Sync>>
    where
        Self: Sized,
    {
        self.validate()?;
        Ok(self)
    }
}

// Blanket implementation for all ConfigValidator types
impl<T: ConfigValidator> ConfigValidatorExt for T {}

impl From<ConfigError> for String {
    /// Converts a ConfigError to a String.
    ///
    /// This allows ConfigError to be converted to a String representation.
    fn from(error: ConfigError) -> Self {
        error.to_string()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    struct TestValidator;
    impl ConfigValidator for TestValidator {
        fn validate(&self) -> Result<(), ConfigError> {
            Ok(())
        }

        fn get_defaults() -> Self {
            TestValidator
        }
    }

    #[test]
    fn test_validate_batch_size() {
        let validator = TestValidator;
        assert!(validator.validate_batch_size(1).is_ok());
        assert!(validator.validate_batch_size(10).is_ok());
        assert!(validator.validate_batch_size(0).is_err());
    }

    #[test]
    fn test_validate_image_dimensions() {
        let validator = TestValidator;
        assert!(validator.validate_image_dimensions(100, 100).is_ok());
        assert!(validator.validate_image_dimensions(1, 1).is_ok());
        assert!(validator.validate_image_dimensions(0, 100).is_err());
        assert!(validator.validate_image_dimensions(100, 0).is_err());
        assert!(validator.validate_image_dimensions(0, 0).is_err());
    }

    #[test]
    fn test_validate_confidence_threshold() {
        let validator = TestValidator;
        assert!(validator.validate_confidence_threshold(0.0).is_ok());
        assert!(validator.validate_confidence_threshold(0.5).is_ok());
        assert!(validator.validate_confidence_threshold(1.0).is_ok());
        assert!(validator.validate_confidence_threshold(-0.1).is_err());
        assert!(validator.validate_confidence_threshold(1.1).is_err());
    }

    #[test]
    fn test_validate_memory_limit() {
        let validator = TestValidator;
        assert!(validator.validate_memory_limit(1024).is_ok());
        assert!(validator.validate_memory_limit(16 * 1024).is_ok());
        assert!(validator.validate_memory_limit(64 * 1024).is_err());
    }

    #[test]
    fn test_validate_thread_count() {
        let validator = TestValidator;
        assert!(validator.validate_thread_count(1).is_ok());
        assert!(validator.validate_thread_count(8).is_ok());
        assert!(validator.validate_thread_count(64).is_ok());
        assert!(validator.validate_thread_count(0).is_err());
        assert!(validator.validate_thread_count(512).is_err());
    }

    #[test]
    fn test_config_error_to_string() {
        let error = ConfigError::InvalidBatchSize;
        let error_string: String = error.into();
        assert_eq!(error_string, "batch size must be greater than 0");
    }
}
