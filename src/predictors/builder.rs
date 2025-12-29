//! Shared utilities for predictor builders.
//!
//! The OCR crate exposes many task-specific predictor builders whose structure is
//! largely identical: hold onto a task configuration, optionally accept an
//! `OrtSessionConfig`, and provide builder-style setters.
//! This module centralises that shared logic so individual predictors only need to
//! focus on their task-specific parameters.

use crate::core::config::OrtSessionConfig;

/// Common state for predictor builders.
#[derive(Debug, Clone)]
pub struct PredictorBuilderState<C> {
    config: C,
    ort_config: Option<OrtSessionConfig>,
}

impl<C> PredictorBuilderState<C> {
    /// Creates a new builder state using the provided configuration.
    pub fn new(config: C) -> Self {
        Self {
            config,
            ort_config: None,
        }
    }

    /// Returns a mutable reference to the configuration for in-place updates.
    pub fn config_mut(&mut self) -> &mut C {
        &mut self.config
    }

    /// Overrides the stored configuration.
    pub fn set_config(&mut self, config: C) {
        self.config = config;
    }

    /// Overrides the stored OrtSessionConfig.
    pub fn set_ort_config(&mut self, config: OrtSessionConfig) {
        self.ort_config = Some(config);
    }

    /// Consumes the builder state and returns its parts.
    pub fn into_parts(self) -> (C, Option<OrtSessionConfig>) {
        (self.config, self.ort_config)
    }
}

/// Trait implemented by every predictor builder that uses `PredictorBuilderState`.
///
/// This trait provides default implementations for the common builder methods,
/// eliminating repeated code throughout the predictor modules.
pub trait TaskPredictorBuilder: Sized {
    /// Configuration type associated with the builder.
    type Config: Clone;

    /// Mutable accessor for the underlying builder state.
    fn state_mut(&mut self) -> &mut PredictorBuilderState<Self::Config>;

    /// Replaces the stored configuration.
    fn with_config(mut self, config: Self::Config) -> Self {
        self.state_mut().set_config(config);
        self
    }

    /// Stores the provided `OrtSessionConfig`.
    fn with_ort_config(mut self, config: OrtSessionConfig) -> Self {
        self.state_mut().set_ort_config(config);
        self
    }

    /// Conditionally stores the provided `OrtSessionConfig` if present.
    ///
    /// This is a convenience method that eliminates the common pattern:
    /// ```rust,no_run
    /// // if let Some(ort_config) = ort_config {
    /// //     builder = builder.with_ort_config(ort_config);
    /// // }
    /// ```
    ///
    /// Instead, you can simply write:
    /// ```rust,no_run
    /// // builder.with_optional_ort_config(ort_config)
    /// ```
    fn with_optional_ort_config(self, config: Option<OrtSessionConfig>) -> Self {
        if let Some(cfg) = config {
            self.with_ort_config(cfg)
        } else {
            self
        }
    }
}

/// Helper macro that wires up `TaskPredictorBuilder` plumbing and re-exports the
/// familiar `with_config`/`with_ort_config`/`with_optional_ort_config` inherent methods for a builder.
macro_rules! impl_task_predictor_builder {
    ($builder:ty, $config:ty) => {
        impl crate::predictors::builder::TaskPredictorBuilder for $builder {
            type Config = $config;

            fn state_mut(
                &mut self,
            ) -> &mut crate::predictors::builder::PredictorBuilderState<Self::Config> {
                &mut self.state
            }
        }

        impl $builder {
            /// Replace the full task configuration used by this builder.
            pub fn with_config(self, config: $config) -> Self {
                <Self as crate::predictors::builder::TaskPredictorBuilder>::with_config(
                    self, config,
                )
            }

            /// Configure ONNX Runtime session options.
            pub fn with_ort_config(self, config: crate::core::config::OrtSessionConfig) -> Self {
                <Self as crate::predictors::builder::TaskPredictorBuilder>::with_ort_config(
                    self, config,
                )
            }

            /// Conditionally configure ONNX Runtime session options if present.
            pub fn with_optional_ort_config(
                self,
                config: Option<crate::core::config::OrtSessionConfig>,
            ) -> Self {
                <Self as crate::predictors::builder::TaskPredictorBuilder>::with_optional_ort_config(
                    self, config,
                )
            }
        }
    };
}
