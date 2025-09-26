//! Configuration system for extensible pipeline stages.
//!
//! This module provides configuration structures and utilities for
//! managing extensible pipeline stage configurations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::extensible::StageId;
use crate::core::config::{ConfigError, ConfigValidator};

/// Configuration for the extensible pipeline system.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ExtensiblePipelineConfig {
    /// Whether to use the extensible pipeline system
    pub enabled: bool,
    /// Global pipeline settings
    pub global_settings: GlobalPipelineSettings,
    /// Stage-specific configurations
    pub stage_configs: HashMap<String, serde_json::Value>,
    /// Stage execution order (if not specified, dependencies will determine order)
    pub stage_order: Option<Vec<String>>,
    /// Stages to enable/disable
    pub enabled_stages: Option<Vec<String>>,
    pub disabled_stages: Option<Vec<String>>,
}

/// Global settings that apply to the entire pipeline.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GlobalPipelineSettings {
    /// Centralized parallel processing policy
    #[serde(default)]
    pub parallel_policy: crate::core::config::ParallelPolicy,
    /// Whether to continue processing if a stage fails
    pub continue_on_stage_failure: bool,
    /// Global timeout for pipeline execution (in seconds)
    pub pipeline_timeout_seconds: Option<u64>,
    /// Whether to collect detailed metrics for each stage
    pub collect_detailed_metrics: bool,
}

impl Default for GlobalPipelineSettings {
    fn default() -> Self {
        Self {
            parallel_policy: crate::core::config::ParallelPolicy::default(),
            continue_on_stage_failure: false,
            pipeline_timeout_seconds: None,
            collect_detailed_metrics: true,
        }
    }
}

impl GlobalPipelineSettings {
    /// Get the effective parallel policy
    pub fn effective_parallel_policy(&self) -> crate::core::config::ParallelPolicy {
        self.parallel_policy.clone()
    }
}

/// Utility functions for working with extensible pipeline configurations.
impl ExtensiblePipelineConfig {}

/// Utility functions for working with extensible pipeline configurations.
impl ExtensiblePipelineConfig {
    /// Get configuration for a specific stage.
    pub fn get_stage_config<T>(&self, stage_id: &str) -> Option<T>
    where
        T: for<'de> Deserialize<'de>,
    {
        self.stage_configs
            .get(stage_id)
            .and_then(|value| serde_json::from_value(value.clone()).ok())
    }

    /// Check if a stage is enabled.
    pub fn is_stage_enabled(&self, stage_id: &str) -> bool {
        // If enabled_stages is specified, only those stages are enabled
        if let Some(ref enabled) = self.enabled_stages {
            return enabled.contains(&stage_id.to_string());
        }

        // If disabled_stages is specified, check if this stage is disabled
        if let Some(ref disabled) = self.disabled_stages {
            return !disabled.contains(&stage_id.to_string());
        }

        // By default, all stages are enabled
        true
    }

    /// Get the configured stage execution order.
    pub fn get_stage_order(&self) -> Option<Vec<StageId>> {
        self.stage_order
            .as_ref()
            .map(|order| order.iter().map(|s| StageId::new(s.clone())).collect())
    }
}

/// Configuration validation utilities.
impl ExtensiblePipelineConfig {
    /// Validate the configuration for consistency and correctness.
    pub fn validate(&self) -> Result<(), ConfigError> {
        // Check for conflicting enabled/disabled stage settings
        if let (Some(enabled), Some(disabled)) = (&self.enabled_stages, &self.disabled_stages) {
            for stage in enabled {
                if disabled.contains(stage) {
                    return Err(ConfigError::ValidationFailed {
                        message: format!("Stage '{}' is both enabled and disabled", stage),
                    });
                }
            }
        }

        // Validate global settings
        if let Some(timeout) = self.global_settings.pipeline_timeout_seconds {
            #[allow(clippy::collapsible_if)]
            if timeout == 0 {
                return Err(ConfigError::InvalidConfig {
                    message: "Pipeline timeout must be greater than 0".to_string(),
                });
            }
        }

        // Validate parallel policy
        let effective_policy = self.global_settings.effective_parallel_policy();
        if let Some(threads) = effective_policy.max_threads
            && threads == 0
        {
            return Err(ConfigError::InvalidConfig {
                message: "Max parallel threads must be greater than 0".to_string(),
            });
        }

        // Validate stage configurations for basic JSON validity
        for (stage_id, config_value) in &self.stage_configs {
            if config_value.is_null() {
                return Err(ConfigError::InvalidConfig {
                    message: format!("Stage '{}' has null configuration", stage_id),
                });
            }
        }

        // Validate stage order references
        if let Some(ref stage_order) = self.stage_order {
            if stage_order.is_empty() {
                return Err(ConfigError::InvalidConfig {
                    message: "Stage order cannot be empty when specified".to_string(),
                });
            }

            // Check for duplicate stages in order
            let mut seen_stages = std::collections::HashSet::new();
            for stage_id in stage_order {
                if !seen_stages.insert(stage_id) {
                    return Err(ConfigError::ValidationFailed {
                        message: format!("Duplicate stage '{}' in execution order", stage_id),
                    });
                }
            }
        }

        Ok(())
    }
}

/// Implementation of ConfigValidator trait for ExtensiblePipelineConfig.
impl ConfigValidator for ExtensiblePipelineConfig {
    /// Validate the configuration for consistency and correctness.
    ///
    /// This implementation delegates to the existing validate method to maintain
    /// consistency with the structured error handling used throughout the codebase.
    fn validate(&self) -> Result<(), ConfigError> {
        self.validate()
    }

    /// Get default configuration.
    fn get_defaults() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::config::ConfigValidator;

    #[test]
    fn test_validate_conflicting_enabled_disabled_stages() {
        let config = ExtensiblePipelineConfig {
            enabled_stages: Some(vec!["stage1".to_string(), "stage2".to_string()]),
            disabled_stages: Some(vec!["stage2".to_string(), "stage3".to_string()]),
            ..Default::default()
        };

        let result = config.validate();
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(matches!(error, ConfigError::ValidationFailed { .. }));
        assert!(
            error
                .to_string()
                .contains("Stage 'stage2' is both enabled and disabled")
        );
    }

    #[test]
    fn test_validate_zero_pipeline_timeout() {
        let mut config = ExtensiblePipelineConfig::default();
        config.global_settings.pipeline_timeout_seconds = Some(0);

        let result = config.validate();
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(matches!(error, ConfigError::InvalidConfig { .. }));
        assert!(
            error
                .to_string()
                .contains("Pipeline timeout must be greater than 0")
        );
    }

    #[test]
    fn test_validate_zero_max_threads() {
        let mut config = ExtensiblePipelineConfig::default();
        config.global_settings.parallel_policy.max_threads = Some(0);

        let result = config.validate();
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(matches!(error, ConfigError::InvalidConfig { .. }));
        assert!(
            error
                .to_string()
                .contains("Max parallel threads must be greater than 0")
        );
    }

    #[test]
    fn test_validate_null_stage_config() {
        let mut config = ExtensiblePipelineConfig::default();
        config
            .stage_configs
            .insert("test_stage".to_string(), serde_json::Value::Null);

        let result = config.validate();
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(matches!(error, ConfigError::InvalidConfig { .. }));
        assert!(
            error
                .to_string()
                .contains("Stage 'test_stage' has null configuration")
        );
    }

    #[test]
    fn test_validate_empty_stage_order() {
        let config = ExtensiblePipelineConfig {
            stage_order: Some(vec![]),
            ..Default::default()
        };

        let result = config.validate();
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(matches!(error, ConfigError::InvalidConfig { .. }));
        assert!(
            error
                .to_string()
                .contains("Stage order cannot be empty when specified")
        );
    }

    #[test]
    fn test_validate_duplicate_stages_in_order() {
        let config = ExtensiblePipelineConfig {
            stage_order: Some(vec![
                "stage1".to_string(),
                "stage2".to_string(),
                "stage1".to_string(), // Duplicate
            ]),
            ..Default::default()
        };

        let result = config.validate();
        assert!(result.is_err());

        let error = result.unwrap_err();
        assert!(matches!(error, ConfigError::ValidationFailed { .. }));
        assert!(
            error
                .to_string()
                .contains("Duplicate stage 'stage1' in execution order")
        );
    }

    #[test]
    fn test_validate_valid_configuration() {
        let mut config = ExtensiblePipelineConfig {
            enabled_stages: Some(vec!["stage1".to_string(), "stage2".to_string()]),
            disabled_stages: Some(vec!["stage3".to_string()]),
            stage_order: Some(vec!["stage1".to_string(), "stage2".to_string()]),
            ..Default::default()
        };
        config.global_settings.pipeline_timeout_seconds = Some(60);
        config.global_settings.parallel_policy.max_threads = Some(4);
        config.stage_configs.insert(
            "test_stage".to_string(),
            serde_json::json!({"enabled": true}),
        );

        let result = config.validate();
        assert!(result.is_ok());
    }

    #[test]
    fn test_config_validator_trait_implementation() {
        let config = ExtensiblePipelineConfig::default();

        // Test that ConfigValidator trait methods work
        assert!(config.validate().is_ok());

        let defaults = ExtensiblePipelineConfig::get_defaults();
        assert!(defaults.validate().is_ok());
    }
}
