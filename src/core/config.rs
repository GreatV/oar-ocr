//! Configuration utilities for the OCR pipeline.
//!
//! This module provides structures and functions for handling configuration
//! in the OCR pipeline, including error types, validation traits, and
//! configuration structures for various components.

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
                    "Batch size {} exceeds maximum allowed size {}",
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
    /// * `path` - The path to the model file.
    ///
    /// # Returns
    ///
    /// A Result indicating success or a ConfigError if validation fails.
    fn validate_model_path(&self, path: &Path) -> Result<(), ConfigError> {
        if !path.exists() {
            return Err(ConfigError::ModelPathNotFound {
                path: path.to_path_buf(),
            });
        }

        if !path.is_file() {
            return Err(ConfigError::InvalidConfig {
                message: format!(
                    "Model path must be a file, not a directory: {}",
                    path.display()
                ),
            });
        }

        Ok(())
    }

    /// Validates that a usize value is positive.
    ///
    /// This method checks that the value is greater than 0.
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
            return Err(ConfigError::InvalidConfig {
                message: format!("{} must be greater than 0", field_name),
            });
        }
        Ok(())
    }

    /// Validates that an f32 value is positive.
    ///
    /// This method checks that the value is greater than 0.0.
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
            return Err(ConfigError::InvalidConfig {
                message: format!("{} must be greater than 0.0", field_name),
            });
        }
        Ok(())
    }

    /// Validates that an f32 value is within a range.
    ///
    /// This method checks that the value is between the specified minimum and maximum values.
    ///
    /// # Arguments
    ///
    /// * `value` - The value to validate.
    /// * `min` - The minimum allowed value.
    /// * `max` - The maximum allowed value.
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
            return Err(ConfigError::InvalidConfig {
                message: format!(
                    "{} must be between {} and {}, got {}",
                    field_name, min, max, value
                ),
            });
        }
        Ok(())
    }

    /// Validates image dimensions.
    ///
    /// This method checks that the width and height are greater than 0 and do not exceed
    /// the maximum allowed dimensions.
    ///
    /// # Arguments
    ///
    /// * `width` - The width of the image.
    /// * `height` - The height of the image.
    /// * `field_name` - The name of the field being validated.
    ///
    /// # Returns
    ///
    /// A Result indicating success or a ConfigError if validation fails.
    fn validate_image_dimensions(
        &self,
        width: u32,
        height: u32,
        field_name: &str,
    ) -> Result<(), ConfigError> {
        if width == 0 || height == 0 {
            return Err(ConfigError::InvalidConfig {
                message: format!(
                    "{} dimensions must be greater than 0, got {}x{}",
                    field_name, width, height
                ),
            });
        }

        const MAX_DIMENSION: u32 = 8192;
        if width > MAX_DIMENSION || height > MAX_DIMENSION {
            return Err(ConfigError::ResourceLimitExceeded {
                message: format!(
                    "{} dimensions {}x{} exceed maximum allowed size {}x{}",
                    field_name, width, height, MAX_DIMENSION, MAX_DIMENSION
                ),
            });
        }

        Ok(())
    }
}

/// Implementation of `From<ConfigError>` for String.
///
/// This allows ConfigError to be converted to a String representation.
impl From<ConfigError> for String {
    fn from(error: ConfigError) -> Self {
        error.to_string()
    }
}

/// Enum representing different types of transforms that can be applied to images.
///
/// This enum defines the various transform types that can be used in the OCR pipeline.
#[derive(Debug, Clone, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
pub enum TransformType {
    /// Resize an image to a specified size.
    ResizeImage,
    /// Crop an image to a specified size.
    CropImage,
    /// Normalize an image using mean and standard deviation values.
    NormalizeImage,
    /// Convert an image from HWC (Height-Width-Channel) to CHW (Channel-Height-Width) format.
    ToCHWImage,
    /// Select the top-k elements from a list.
    Topk,
    /// Decode an image from a byte array.
    DecodeImage,
    /// Resize an image for recognition.
    RecResizeImg,
    /// Keep only specified keys in a data structure.
    KeepKeys,
    /// Encode multiple labels.
    MultiLabelEncode,
}

/// Enum representing configuration for different types of transforms.
///
/// This enum defines the configuration parameters for various transform types
/// that can be used in the OCR pipeline.
#[derive(Debug, serde::Serialize, serde::Deserialize)]
pub enum TransformConfig {
    /// Configuration for resizing an image.
    ResizeImage {
        /// The size to resize the shorter side to (optional).
        resize_short: Option<u32>,
        /// The target size as (width, height) (optional).
        size: Option<(u32, u32)>,
        /// The backend to use for resizing.
        backend: String,
        /// The interpolation method to use.
        interpolation: String,
    },
    /// Configuration for cropping an image.
    CropImage {
        /// The size to crop the image to.
        size: u32,
    },
    /// Configuration for normalizing an image.
    NormalizeImage {
        /// The mean values for normalization.
        mean: Vec<f32>,
        /// The standard deviation values for normalization.
        std: Vec<f32>,
        /// The scale factor for normalization.
        scale: f32,
        /// The order of the channels.
        order: String,
        /// The number of channels.
        channel_num: u32,
    },
    /// Configuration for converting an image from HWC to CHW format.
    ToCHWImage,
    /// Configuration for selecting the top-k elements.
    Topk {
        /// The number of top elements to select.
        topk: usize,
        /// The list of labels (optional).
        label_list: Option<Vec<String>>,
    },
    /// Configuration for decoding an image from a byte array.
    DecodeImage {
        /// Whether the channels should be first in the output.
        channel_first: bool,
        /// The image mode.
        img_mode: String,
    },
    /// Configuration for resizing an image for recognition.
    RecResizeImg {
        /// The shape of the image as [channels, height, width].
        image_shape: [usize; 3],
    },
    /// Configuration for keeping only specified keys.
    KeepKeys,
    /// Configuration for encoding multiple labels.
    MultiLabelEncode,
}

/// A registry for storing and managing transform configurations.
///
/// This struct provides a way to store and retrieve transform configurations
/// for different transform types used in the OCR pipeline.
#[derive(Debug)]
pub struct TransformRegistry {
    /// A vector of tuples containing transform types and their configurations.
    transforms: Vec<(TransformType, TransformConfig)>,
}

impl TransformRegistry {
    /// Creates a new empty TransformRegistry.
    ///
    /// # Returns
    ///
    /// A new TransformRegistry instance.
    pub fn new() -> Self {
        Self {
            transforms: Vec::new(),
        }
    }

    /// Creates a new TransformRegistry with the specified capacity.
    ///
    /// # Arguments
    ///
    /// * `capacity` - The initial capacity of the registry.
    ///
    /// # Returns
    ///
    /// A new TransformRegistry instance with the specified capacity.
    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            transforms: Vec::with_capacity(capacity),
        }
    }

    /// Registers a transform type and its configuration.
    ///
    /// If a configuration for the transform type already exists, it will be replaced.
    ///
    /// # Arguments
    ///
    /// * `transform_type` - The type of transform to register.
    /// * `config` - The configuration for the transform.
    pub fn register(&mut self, transform_type: TransformType, config: TransformConfig) {
        self.transforms.retain(|(t, _)| t != &transform_type);
        self.transforms.push((transform_type, config));
    }

    /// Gets the configuration for a transform type.
    ///
    /// # Arguments
    ///
    /// * `transform_type` - The type of transform to get the configuration for.
    ///
    /// # Returns
    ///
    /// An Option containing a reference to the configuration if it exists, or None if it doesn't.
    pub fn get(&self, transform_type: &TransformType) -> Option<&TransformConfig> {
        self.transforms
            .iter()
            .find(|(t, _)| t == transform_type)
            .map(|(_, config)| config)
    }

    /// Checks if the registry contains a configuration for a transform type.
    ///
    /// # Arguments
    ///
    /// * `transform_type` - The type of transform to check for.
    ///
    /// # Returns
    ///
    /// True if the registry contains a configuration for the transform type, false otherwise.
    pub fn contains(&self, transform_type: &TransformType) -> bool {
        self.transforms.iter().any(|(t, _)| t == transform_type)
    }

    /// Returns an iterator over the transform types and their configurations.
    ///
    /// # Returns
    ///
    /// An iterator over the transform types and their configurations.
    pub fn iter(&self) -> impl Iterator<Item = &(TransformType, TransformConfig)> {
        self.transforms.iter()
    }

    /// Creates a TransformRegistry from a vector of transform types and configurations.
    ///
    /// # Arguments
    ///
    /// * `transforms` - A vector of tuples containing transform types and their configurations.
    ///
    /// # Returns
    ///
    /// A new TransformRegistry instance.
    pub fn from_vec(transforms: Vec<(TransformType, TransformConfig)>) -> Self {
        Self { transforms }
    }
}

/// Implementation of IntoIterator for TransformRegistry.
///
/// This allows TransformRegistry to be used in for loops and other iterator contexts.
impl IntoIterator for TransformRegistry {
    type Item = (TransformType, TransformConfig);
    type IntoIter = std::vec::IntoIter<Self::Item>;

    fn into_iter(self) -> Self::IntoIter {
        self.transforms.into_iter()
    }
}

/// Implementation of IntoIterator for &TransformRegistry.
///
/// This allows &TransformRegistry to be used in for loops and other iterator contexts.
impl<'a> IntoIterator for &'a TransformRegistry {
    type Item = &'a (TransformType, TransformConfig);
    type IntoIter = std::slice::Iter<'a, (TransformType, TransformConfig)>;

    fn into_iter(self) -> Self::IntoIter {
        self.transforms.iter()
    }
}

/// Implementation of IntoIterator for &mut TransformRegistry.
///
/// This allows &mut TransformRegistry to be used in for loops and other iterator contexts.
impl<'a> IntoIterator for &'a mut TransformRegistry {
    type Item = &'a mut (TransformType, TransformConfig);
    type IntoIter = std::slice::IterMut<'a, (TransformType, TransformConfig)>;

    fn into_iter(self) -> Self::IntoIter {
        self.transforms.iter_mut()
    }
}

/// Implementation of Default for TransformRegistry.
///
/// This allows TransformRegistry to be created with default values.
impl Default for TransformRegistry {
    fn default() -> Self {
        Self::new()
    }
}

/// Configuration for building common components of the OCR pipeline.
///
/// This struct provides configuration options for building various components
/// of the OCR pipeline, such as models and batch processing parameters.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CommonBuilderConfig {
    /// The path to the model file (optional).
    pub model_path: Option<std::path::PathBuf>,
    /// The name of the model (optional).
    pub model_name: Option<String>,
    /// The batch size for processing (optional).
    pub batch_size: Option<usize>,
    /// Whether to enable logging (optional).
    pub enable_logging: Option<bool>,
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
    pub fn with_model_path(model_path: std::path::PathBuf) -> Self {
        Self {
            model_path: Some(model_path),
            model_name: None,
            batch_size: None,
            enable_logging: Some(true),
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
    pub fn model_path(mut self, model_path: impl Into<std::path::PathBuf>) -> Self {
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
        self
    }
}

/// Implementation of ConfigValidator for CommonBuilderConfig.
///
/// This allows CommonBuilderConfig to be validated using the ConfigValidator trait.
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
        }
    }
}

/// Implementation of Default for CommonBuilderConfig.
///
/// This allows CommonBuilderConfig to be created with default values.
impl Default for CommonBuilderConfig {
    fn default() -> Self {
        Self::new()
    }
}
