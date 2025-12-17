//! Reusable builder configuration for model adapters.
//!
//! This module provides composable configuration components that can be used
//! across different adapter builders to eliminate code duplication.

use crate::core::config::{ConfigError, ConfigValidator, OrtSessionConfig};

/// Generic builder configuration that can be composed into adapter builders.
///
/// This struct holds common configuration options that are shared across all
/// adapter builders, including task configuration, session pool size, and
/// ONNX Runtime settings.
///
/// # Type Parameters
///
/// * `C` - The task-specific configuration type (e.g., `TextDetectionConfig`)
///
/// # Examples
///
/// ```rust,no_run
/// // pub struct MyAdapterBuilder {
/// //     config: AdapterBuilderConfig<MyTaskConfig>,
/// //     // ... other builder-specific fields
/// // }
///
/// // impl MyAdapterBuilder {
/// //     pub fn new() -> Self {
/// //         Self {
/// //             config: AdapterBuilderConfig::new(MyTaskConfig::default()),
/// //         }
/// //     }
///
/// //     pub fn with_config(mut self, config: MyTaskConfig) -> Self {
/// //         self.config = self.config.with_task_config(config);
/// //         self
/// //     }
///
/// //     pub fn session_pool_size(mut self, size: usize) -> Self {
/// //         self.config = self.config.with_session_pool_size(size);
/// //         self
/// //     }
/// // }
/// ```
#[derive(Debug, Clone)]
pub struct AdapterBuilderConfig<C> {
    /// Task-specific configuration
    pub task_config: C,
    /// Number of sessions in the ONNX Runtime session pool
    pub session_pool_size: usize,
    /// Optional ONNX Runtime session configuration
    pub ort_config: Option<OrtSessionConfig>,
}

impl<C> AdapterBuilderConfig<C> {
    /// Creates a new adapter builder configuration with default pool size.
    ///
    /// # Arguments
    ///
    /// * `task_config` - The task-specific configuration
    pub fn new(task_config: C) -> Self {
        Self {
            task_config,
            session_pool_size: 1,
            ort_config: None,
        }
    }

    /// Sets the task configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - The new task configuration
    pub fn with_task_config(mut self, config: C) -> Self {
        self.task_config = config;
        self
    }

    /// Sets the session pool size.
    ///
    /// The session pool size determines how many ONNX Runtime sessions are
    /// maintained for concurrent inference. A larger pool size can improve
    /// throughput for parallel requests but increases memory usage.
    ///
    /// # Arguments
    ///
    /// * `size` - Number of sessions in the pool (default: 1)
    pub fn with_session_pool_size(mut self, size: usize) -> Self {
        self.session_pool_size = size;
        self
    }

    /// Sets the ONNX Runtime session configuration.
    ///
    /// # Arguments
    ///
    /// * `config` - ONNX Runtime session configuration
    pub fn with_ort_config(mut self, config: OrtSessionConfig) -> Self {
        self.ort_config = Some(config);
        self
    }

    /// Gets a reference to the task configuration.
    pub fn task_config(&self) -> &C {
        &self.task_config
    }

    /// Gets the session pool size.
    pub fn session_pool_size(&self) -> usize {
        self.session_pool_size
    }

    /// Gets a reference to the ONNX Runtime configuration, if set.
    pub fn ort_config(&self) -> Option<&OrtSessionConfig> {
        self.ort_config.as_ref()
    }

    /// Consumes the config and returns the individual components.
    ///
    /// This is useful when building the final adapter.
    pub fn into_parts(self) -> (C, usize, Option<OrtSessionConfig>) {
        (self.task_config, self.session_pool_size, self.ort_config)
    }
}

impl<C: ConfigValidator> AdapterBuilderConfig<C> {
    /// Consumes the config and validates the task config and session pool size.
    ///
    /// Returns the parts if validation succeeds, otherwise returns a ConfigError.
    pub fn into_validated_parts(self) -> Result<(C, usize, Option<OrtSessionConfig>), ConfigError> {
        if self.session_pool_size == 0 {
            return Err(ConfigError::InvalidConfig {
                message: "session_pool_size must be at least 1".to_string(),
            });
        }

        self.task_config.validate()?;
        Ok((self.task_config, self.session_pool_size, self.ort_config))
    }
}

impl<C: Default> Default for AdapterBuilderConfig<C> {
    fn default() -> Self {
        Self::new(C::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::{ConfigError, ConfigValidator};

    #[derive(Debug, Clone, Default, PartialEq)]
    struct TestConfig {
        threshold: f32,
    }

    impl ConfigValidator for TestConfig {
        fn validate(&self) -> Result<(), ConfigError> {
            if !(0.0..=1.0).contains(&self.threshold) {
                return Err(ConfigError::InvalidConfig {
                    message: "threshold must be between 0 and 1".to_string(),
                });
            }
            Ok(())
        }

        fn get_defaults() -> Self {
            Self::default()
        }
    }

    #[test]
    fn test_new_with_defaults() {
        let config = AdapterBuilderConfig::new(TestConfig { threshold: 0.5 });
        assert_eq!(config.task_config.threshold, 0.5);
        assert_eq!(config.session_pool_size, 1);
        assert!(config.ort_config.is_none());
    }

    #[test]
    fn test_with_task_config() {
        let config = AdapterBuilderConfig::new(TestConfig { threshold: 0.5 })
            .with_task_config(TestConfig { threshold: 0.7 });
        assert_eq!(config.task_config.threshold, 0.7);
    }

    #[test]
    fn test_with_session_pool_size() {
        let config = AdapterBuilderConfig::new(TestConfig::default()).with_session_pool_size(4);
        assert_eq!(config.session_pool_size, 4);
    }

    #[test]
    fn test_with_ort_config() {
        let ort_config = OrtSessionConfig::default();
        let config =
            AdapterBuilderConfig::new(TestConfig::default()).with_ort_config(ort_config.clone());
        assert!(config.ort_config.is_some());
    }

    #[test]
    fn test_fluent_api_chaining() {
        let ort_config = OrtSessionConfig::default();
        let config = AdapterBuilderConfig::new(TestConfig { threshold: 0.5 })
            .with_task_config(TestConfig { threshold: 0.7 })
            .with_session_pool_size(4)
            .with_ort_config(ort_config);

        assert_eq!(config.task_config.threshold, 0.7);
        assert_eq!(config.session_pool_size, 4);
        assert!(config.ort_config.is_some());
    }

    #[test]
    fn test_getters() {
        let config =
            AdapterBuilderConfig::new(TestConfig { threshold: 0.5 }).with_session_pool_size(3);

        assert_eq!(config.task_config().threshold, 0.5);
        assert_eq!(config.session_pool_size(), 3);
        assert!(config.ort_config().is_none());
    }

    #[test]
    fn test_into_parts() {
        let ort_config = OrtSessionConfig::default();
        let config = AdapterBuilderConfig::new(TestConfig { threshold: 0.5 })
            .with_session_pool_size(2)
            .with_ort_config(ort_config);

        let (task_config, pool_size, ort) = config.into_parts();
        assert_eq!(task_config.threshold, 0.5);
        assert_eq!(pool_size, 2);
        assert!(ort.is_some());
    }

    #[test]
    fn test_into_validated_parts() {
        let config = AdapterBuilderConfig::new(TestConfig { threshold: 0.5 });
        assert!(config.clone().into_validated_parts().is_ok());

        let zero_pool = config.clone().with_session_pool_size(0);
        assert!(zero_pool.into_validated_parts().is_err());

        let invalid_config = AdapterBuilderConfig::new(TestConfig { threshold: 1.5 });
        assert!(invalid_config.into_validated_parts().is_err());
    }

    #[test]
    fn test_default_impl() {
        let config: AdapterBuilderConfig<TestConfig> = AdapterBuilderConfig::default();
        assert_eq!(config.task_config, TestConfig::default());
        assert_eq!(config.session_pool_size, 1);
        assert!(config.ort_config.is_none());
    }
}
