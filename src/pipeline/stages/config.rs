//! Configuration system for extensible pipeline stages.
//!
//! This module provides configuration structures and utilities for
//! managing extensible pipeline stage configurations.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

use super::extensible::StageId;

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
    pub parallel_policy: crate::pipeline::oarocr::ParallelPolicy,
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
            parallel_policy: crate::pipeline::oarocr::ParallelPolicy::default(),
            continue_on_stage_failure: false,
            pipeline_timeout_seconds: None,
            collect_detailed_metrics: true,
        }
    }
}

impl GlobalPipelineSettings {
    /// Get the effective parallel policy
    pub fn effective_parallel_policy(&self) -> crate::pipeline::oarocr::ParallelPolicy {
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
    pub fn validate(&self) -> Result<(), String> {
        // Check for conflicting enabled/disabled stage settings
        if let (Some(enabled), Some(disabled)) = (&self.enabled_stages, &self.disabled_stages) {
            for stage in enabled {
                if disabled.contains(stage) {
                    return Err(format!("Stage '{}' is both enabled and disabled", stage));
                }
            }
        }

        // Validate global settings
        if let Some(timeout) = self.global_settings.pipeline_timeout_seconds {
            #[allow(clippy::collapsible_if)]
            if timeout == 0 {
                return Err("Pipeline timeout must be greater than 0".to_string());
            }
        }

        // Validate parallel policy
        let effective_policy = self.global_settings.effective_parallel_policy();
        if let Some(threads) = effective_policy.max_threads
            && threads == 0
        {
            return Err("Max parallel threads must be greater than 0".to_string());
        }

        Ok(())
    }
}
