//! Extensible pipeline stage traits and types.
//!
//! This module provides the core traits and types for building extensible
//! pipeline stages that can be dynamically registered and executed.

use image::RgbImage;
use serde::{Deserialize, Serialize};
use std::any::Any;
use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::Arc;

use super::types::StageResult;
use crate::core::OCRError;

/// Unique identifier for a pipeline stage.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct StageId(pub String);

impl std::fmt::Display for StageId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl StageId {
    /// Create a new stage ID.
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Get the string representation of the stage ID.
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl From<&str> for StageId {
    fn from(s: &str) -> Self {
        Self(s.to_string())
    }
}

impl From<String> for StageId {
    fn from(s: String) -> Self {
        Self(s)
    }
}

/// Dependency specification for a pipeline stage.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum StageDependency {
    /// Stage must run after the specified stage
    After(StageId),
    /// Stage must run before the specified stage
    Before(StageId),
    /// Stage requires output from the specified stage
    Requires(StageId),
    /// Stage provides input to the specified stage
    Provides(StageId),
}

/// Context information available to pipeline stages during execution.
#[derive(Debug)]
pub struct StageContext {
    /// Current image being processed
    pub current_image: Arc<RgbImage>,
    /// Original input image (before any transformations)
    pub original_image: Arc<RgbImage>,
    /// Index of the current image in batch processing
    pub image_index: usize,
    /// Global pipeline configuration
    pub global_config: HashMap<String, serde_json::Value>,
    /// Results from previous stages
    pub stage_results: HashMap<StageId, Box<dyn Any + Send + Sync>>,
}

impl StageContext {
    /// Create a new stage context.
    pub fn new(
        current_image: Arc<RgbImage>,
        original_image: Arc<RgbImage>,
        image_index: usize,
    ) -> Self {
        Self {
            current_image,
            original_image,
            image_index,
            global_config: HashMap::new(),
            stage_results: HashMap::new(),
        }
    }

    /// Get a result from a previous stage.
    pub fn get_stage_result<T: 'static>(&self, stage_id: &StageId) -> Option<&T> {
        self.stage_results
            .get(stage_id)
            .and_then(|result| result.downcast_ref::<T>())
    }

    /// Set a result from a stage.
    pub fn set_stage_result<T: 'static + Send + Sync>(&mut self, stage_id: StageId, result: T) {
        self.stage_results.insert(stage_id, Box::new(result));
    }

    /// Get a global configuration value.
    pub fn get_config<T>(&self, key: &str) -> Option<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        self.global_config
            .get(key)
            .and_then(|value| serde_json::from_value(value.clone()).ok())
    }

    /// Set a global configuration value.
    pub fn set_config<T>(&mut self, key: String, value: T)
    where
        T: Serialize,
    {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.global_config.insert(key, json_value);
        }
    }
}

/// Data that flows between pipeline stages.
#[derive(Debug, Clone)]
pub struct StageData {
    /// The current processed image
    pub image: RgbImage,
    /// Metadata associated with the stage processing
    pub metadata: HashMap<String, serde_json::Value>,
}

impl StageData {
    /// Create new stage data with an image.
    pub fn new(image: RgbImage) -> Self {
        Self {
            image,
            metadata: HashMap::new(),
        }
    }

    /// Add metadata to the stage data.
    pub fn with_metadata<T: Serialize>(mut self, key: String, value: T) -> Self {
        if let Ok(json_value) = serde_json::to_value(value) {
            self.metadata.insert(key, json_value);
        }
        self
    }

    /// Get metadata from the stage data.
    pub fn get_metadata<T>(&self, key: &str) -> Option<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        self.metadata
            .get(key)
            .and_then(|value| serde_json::from_value(value.clone()).ok())
    }
}

/// Trait for extensible pipeline stages.
///
/// This trait defines the interface that all pipeline stages must implement
/// to participate in the extensible pipeline system.
///
/// # Default Contract
///
/// All pipeline stages must implement a **default contract** to ensure consistency,
/// reliability, and maintainability. This contract eliminates silent failures and
/// ensures all stages provide meaningful defaults and proper validation.
///
/// ## Contract Requirements:
///
/// 1. **Mandatory Default Configuration**: `default_config()` must return a valid `Self::Config`
/// 2. **ConfigValidator Implementation**: All config types must implement `ConfigValidator`
/// 3. **Validation Integration**: Stages must use ConfigValidator in `validate_config()`
/// 4. **Required Traits**: Config types must implement `ConfigValidator + Default + Clone + Debug + Send + Sync`
///
/// ## Example Implementation:
///
/// ```rust
/// use oar_ocr::pipeline::stages::{PipelineStage, StageId, StageContext, StageData, StageResult};
/// use oar_ocr::core::config::{ConfigValidator, ConfigError};
/// use oar_ocr::core::OCRError;
/// use serde::{Serialize, Deserialize};
///
/// #[derive(Debug, Clone, Serialize, Deserialize, Default)]
/// pub struct MyStageConfig {
///     pub threshold: f32,
///     pub enabled: bool,
/// }
///
/// impl ConfigValidator for MyStageConfig {
///     fn validate(&self) -> Result<(), ConfigError> {
///         if !(0.0..=1.0).contains(&self.threshold) {
///             return Err(ConfigError::InvalidConfig {
///                 message: "threshold must be between 0.0 and 1.0".to_string(),
///             });
///         }
///         Ok(())
///     }
///
///     fn get_defaults() -> Self {
///         Self { threshold: 0.5, enabled: true }
///     }
/// }
///
/// #[derive(Debug)]
/// pub struct MyStage;
///
/// impl PipelineStage for MyStage {
///     type Config = MyStageConfig;
///     type Result = String;
///
///     fn stage_id(&self) -> StageId { StageId::new("my_stage") }
///     fn stage_name(&self) -> &str { "My Stage" }
///
///     fn validate_config(&self, config: &Self::Config) -> Result<(), OCRError> {
///         config.validate().map_err(|e| OCRError::ConfigError {
///             message: format!("MyStageConfig validation failed: {}", e),
///         })
///     }
///
///     fn default_config(&self) -> Self::Config {
///         MyStageConfig::get_defaults()
///     }
///
///     fn process(
///         &self,
///         _context: &mut StageContext,
///         _data: StageData,
///         config: Option<&Self::Config>,
///     ) -> Result<StageResult<Self::Result>, OCRError> {
///         let config = config.cloned().unwrap_or_else(|| self.default_config());
///         self.validate_config(&config)?;
///         // Process with validated configuration...
///         # Ok(StageResult::new("result".to_string(), Default::default()))
///     }
/// }
/// ```
///
/// See [`DEFAULT_CONTRACT.md`](./DEFAULT_CONTRACT.md) for detailed documentation.
pub trait PipelineStage: Send + Sync + Debug {
    /// The configuration type for this stage.
    ///
    /// Must implement ConfigValidator to ensure validation is never skipped.
    type Config: Send + Sync + Debug + crate::core::config::ConfigValidator + Default;

    /// The result type produced by this stage.
    type Result: Send + Sync + Debug + 'static;

    /// Get the unique identifier for this stage.
    fn stage_id(&self) -> StageId;

    /// Get the human-readable name of this stage.
    fn stage_name(&self) -> &str;

    /// Get the dependencies for this stage.
    fn dependencies(&self) -> Vec<StageDependency> {
        Vec::new()
    }

    /// Check if this stage is enabled based on the context and configuration.
    fn is_enabled(&self, context: &StageContext, config: Option<&Self::Config>) -> bool {
        let _ = (context, config);
        true
    }

    /// Process the stage with the given context and configuration.
    ///
    /// # Arguments
    ///
    /// * `context` - The stage execution context
    /// * `data` - The input data for this stage
    /// * `config` - Optional stage-specific configuration
    ///
    /// # Returns
    ///
    /// A StageResult containing the processed data and metrics
    fn process(
        &self,
        context: &mut StageContext,
        data: StageData,
        config: Option<&Self::Config>,
    ) -> Result<StageResult<Self::Result>, OCRError>;

    /// Validate the stage configuration.
    fn validate_config(&self, config: &Self::Config) -> Result<(), OCRError> {
        let _ = config;
        Ok(())
    }

    /// Get default configuration for this stage.
    ///
    /// This method must return a valid configuration. The default contract
    /// ensures all stages provide meaningful defaults.
    fn default_config(&self) -> Self::Config {
        Self::Config::default()
    }
}
