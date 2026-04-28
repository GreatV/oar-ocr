//! RKNN runtime configuration types.

use super::errors::{ConfigError, ConfigValidator};
use serde::{Deserialize, Serialize};

/// Selects which RK3588 NPU cores an RKNN context may use.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum RknnCoreMaskConfig {
    /// Let librknnrt choose the core.
    #[default]
    Auto,
    /// Pin to NPU core 0.
    Core0,
    /// Pin to NPU core 1.
    Core1,
    /// Pin to NPU core 2.
    Core2,
    /// Use cores 0 and 1.
    Core01,
    /// Use cores 0, 1, and 2.
    Core012,
}

/// Controls how RKNN input tensors are handed to `rknn_inputs_set`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[non_exhaustive]
pub enum RknnInputMode {
    /// Convert NCHW Array4 inputs to NHWC and let librknnrt run its normal
    /// input processing path.
    #[default]
    ToolkitLayout,
    /// Pass already-normalized NCHW bytes directly to the compiled graph.
    ///
    /// Use this only for RKNN models converted to consume normalized NCHW
    /// input directly. It avoids the per-call NCHW to NHWC copy.
    NormalizedNchwPassThrough,
}

/// Configuration for the RKNN backend.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[non_exhaustive]
pub struct RknnSessionConfig {
    /// Number of duplicated RKNN contexts to keep in the inference pool.
    ///
    /// `1` preserves single-context behavior. On RK3588, `3` is usually the
    /// useful upper bound because the SoC has three NPU cores.
    pub num_contexts: Option<usize>,
    /// Core mask applied to every context unless `per_context_core_masks` is set.
    pub core_mask: Option<RknnCoreMaskConfig>,
    /// Optional per-context core masks. If present, these override `core_mask`
    /// for the corresponding context index. If fewer masks are provided than
    /// contexts, remaining contexts fall back to `core_mask`.
    pub per_context_core_masks: Option<Vec<RknnCoreMaskConfig>>,
    /// Input handling mode for `rknn_inputs_set`.
    #[serde(default)]
    pub input_mode: RknnInputMode,
}

impl Default for RknnSessionConfig {
    fn default() -> Self {
        Self {
            num_contexts: Some(1),
            core_mask: Some(RknnCoreMaskConfig::Auto),
            per_context_core_masks: None,
            input_mode: RknnInputMode::ToolkitLayout,
        }
    }
}

impl RknnSessionConfig {
    /// Creates a conservative single-context RKNN configuration.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the number of contexts in the RKNN inference pool.
    pub fn with_num_contexts(mut self, num_contexts: usize) -> Self {
        self.num_contexts = Some(num_contexts);
        self
    }

    /// Sets the same core mask for every context.
    pub fn with_core_mask(mut self, core_mask: RknnCoreMaskConfig) -> Self {
        self.core_mask = Some(core_mask);
        self
    }

    /// Sets per-context core masks.
    pub fn with_per_context_core_masks(mut self, masks: Vec<RknnCoreMaskConfig>) -> Self {
        self.per_context_core_masks = Some(masks);
        self
    }

    /// Sets the RKNN input handling mode.
    pub fn with_input_mode(mut self, input_mode: RknnInputMode) -> Self {
        self.input_mode = input_mode;
        self
    }

    /// Effective context count. Call [`ConfigValidator::validate`] first to
    /// reject invalid zero values.
    pub fn effective_num_contexts(&self) -> usize {
        self.num_contexts.unwrap_or(1)
    }

    /// Returns the configured core mask for a context index, if any.
    pub fn core_mask_for_context(&self, index: usize) -> Option<RknnCoreMaskConfig> {
        self.per_context_core_masks
            .as_ref()
            .and_then(|masks| masks.get(index).copied())
            .or(self.core_mask)
    }
}

impl ConfigValidator for RknnSessionConfig {
    fn validate(&self) -> Result<(), ConfigError> {
        if self.num_contexts == Some(0) {
            return Err(ConfigError::InvalidConfig {
                message: "rknn_session.num_contexts must be greater than 0".to_string(),
            });
        }
        if let Some(masks) = &self.per_context_core_masks {
            let num_contexts = self.effective_num_contexts();
            if masks.len() > num_contexts {
                return Err(ConfigError::InvalidConfig {
                    message: format!(
                        "rknn_session.per_context_core_masks has {} entries but num_contexts is {}",
                        masks.len(),
                        num_contexts
                    ),
                });
            }
        }
        Ok(())
    }

    fn get_defaults() -> Self {
        Self::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn core_mask_for_context_prefers_per_context_mask() {
        let cfg = RknnSessionConfig::new()
            .with_core_mask(RknnCoreMaskConfig::Core012)
            .with_per_context_core_masks(vec![RknnCoreMaskConfig::Core0]);

        assert_eq!(
            cfg.core_mask_for_context(0),
            Some(RknnCoreMaskConfig::Core0)
        );
        assert_eq!(
            cfg.core_mask_for_context(1),
            Some(RknnCoreMaskConfig::Core012)
        );
    }

    #[test]
    fn core_mask_for_context_can_fall_through_to_none() {
        let cfg = RknnSessionConfig {
            core_mask: None,
            ..RknnSessionConfig::new()
        };

        assert_eq!(cfg.core_mask_for_context(0), None);
    }

    #[test]
    fn validate_rejects_zero_contexts() {
        let cfg = RknnSessionConfig::new().with_num_contexts(0);

        assert!(cfg.validate().is_err());
    }

    #[test]
    fn validate_rejects_extra_per_context_core_masks() {
        let cfg = RknnSessionConfig::new()
            .with_num_contexts(1)
            .with_per_context_core_masks(vec![
                RknnCoreMaskConfig::Core0,
                RknnCoreMaskConfig::Core1,
            ]);

        assert!(cfg.validate().is_err());
    }
}
