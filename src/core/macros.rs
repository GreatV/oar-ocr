//! Macros for the OCR pipeline.
//!
//! This module provides utility macros to reduce code duplication across
//! the OCR pipeline, particularly for builder patterns and metrics collection.

/// Macro to handle optional nested config initialization in builders.
///
/// This macro eliminates the repeated pattern of:
/// ```rust,no_run
/// // if self.config.field.is_none() {
/// //     self.config.field = Some(Type::new());
/// // }
/// ```
///
/// # Usage
///
/// ```rust,no_run
/// // Instead of:
/// // if self.config.orientation.is_none() {
/// //     self.config.orientation = Some(DocOrientationClassifierConfig::new());
/// // }
/// // if let Some(ref mut config) = self.config.orientation {
/// //     config.confidence_threshold = Some(threshold);
/// // }
///
/// // Use:
/// // with_nested!(self.config.orientation, DocOrientationClassifierConfig, config => {
/// //     config.confidence_threshold = Some(threshold);
/// // });
/// ```
#[macro_export]
macro_rules! with_nested {
    ($field:expr, $type:ty, $var:ident => $body:block) => {
        if $field.is_none() {
            $field = Some(<$type>::new());
        }
        if let Some(ref mut $var) = $field {
            $body
        }
    };
}

/// Macro to create pre-populated StageMetrics with common patterns.
///
/// This macro reduces duplication in metrics construction across stages.
///
/// # Usage
///
/// ```rust,no_run
/// // Instead of:
/// // StageMetrics::new(success_count, failure_count)
/// //     .with_processing_time(start_time.elapsed())
/// //     .with_info("stage", "cropping")
/// //     .with_info("batch_size", batch_size.to_string())
/// //     .with_info("parallel", parallel.to_string())
///
/// // Use:
/// // metrics!(success_count, failure_count, start_time; stage = "cropping", batch_size = batch_size, parallel = parallel)
/// // Or without timing:
/// // metrics!(success_count, failure_count; stage = "cropping", batch_size = batch_size)
/// ```
#[macro_export]
macro_rules! metrics {
    // With timing
    ($success:expr, $failure:expr, $start_time:expr; $($key:ident = $value:expr),*) => {
        {
            let mut metrics = $crate::pipeline::stages::StageMetrics::new($success, $failure);
            metrics = metrics.with_processing_time($start_time.elapsed());
            $(
                metrics = metrics.with_info(stringify!($key), $value.to_string());
            )*
            metrics
        }
    };
    // Without timing
    ($success:expr, $failure:expr; $($key:ident = $value:expr),*) => {
        {
            let mut metrics = $crate::pipeline::stages::StageMetrics::new($success, $failure);
            $(
                metrics = metrics.with_info(stringify!($key), $value.to_string());
            )*
            metrics
        }
    };
}

/// Comprehensive builder macro for generating common builder method patterns.
///
/// This macro generates multiple types of builder methods to reduce code duplication:
/// 1. Simple setters for direct field assignment
/// 2. Nested config setters using the `with_nested!` macro
/// 3. Enable/disable methods for optional features
/// 4. Dynamic batching configuration methods
///
/// # Usage
///
/// ```rust,no_run
/// // impl_complete_builder! {
/// //     builder: MyBuilder,
/// //     config_field: config,
///
/// //     // Simple setters
/// //     simple_setters: {
/// //         field_name: FieldType => "Documentation for the setter",
/// //     },
///
/// //     // Nested config setters
/// //     nested_setters: {
/// //         config_path: ConfigType => {
/// //             field_name: FieldType => "Documentation",
/// //         },
/// //     },
///
/// //     // Enable/disable methods
/// //     enable_methods: {
/// //         method_name => config_field: DefaultType => "Documentation",
/// //     },
/// // }
/// ```
#[macro_export]
macro_rules! impl_complete_builder {
    // Simple setters only
    (
        builder: $builder:ident,
        config_field: $config_field:ident,
        simple_setters: {
            $($simple_field:ident: $simple_type:ty => $simple_doc:literal),* $(,)?
        }
    ) => {
        impl $builder {
            $(
                #[doc = $simple_doc]
                pub fn $simple_field(mut self, value: $simple_type) -> Self {
                    self.$config_field.$simple_field = Some(value);
                    self
                }
            )*
        }
    };

    // Nested setters only
    (
        builder: $builder:ident,
        config_field: $config_field:ident,
        nested_setters: {
            $($nested_path:ident: $nested_type:ty => {
                $($nested_field:ident: $nested_field_type:ty => $nested_doc:literal),* $(,)?
            }),* $(,)?
        }
    ) => {
        impl $builder {
            $($(
                #[doc = $nested_doc]
                pub fn $nested_field(mut self, value: $nested_field_type) -> Self {
                    $crate::with_nested!(self.$config_field.$nested_path, $nested_type, config => {
                        config.$nested_field = Some(value);
                    });
                    self
                }
            )*)*
        }
    };

    // Enable methods only
    (
        builder: $builder:ident,
        config_field: $config_field:ident,
        enable_methods: {
            $($enable_method:ident => $enable_field:ident: $enable_type:ty => $enable_doc:literal),* $(,)?
        }
    ) => {
        impl $builder {
            $(
                #[doc = $enable_doc]
                pub fn $enable_method(mut self) -> Self {
                    self.$config_field.$enable_field = Some(<$enable_type>::default());
                    self
                }
            )*
        }
    };
}

/// Macro to implement `new()` and `with_common()` for config structs with per-module defaults.
#[macro_export]
macro_rules! impl_config_new_and_with_common {
    (
        $Config:ident,
        common_defaults: ($model_name_opt:expr, $batch_size_opt:expr),
        fields: { $( $field:ident : $default_expr:expr ),* $(,)? }
    ) => {
        impl $Config {
            /// Creates a new config instance with default values
            pub fn new() -> Self {
                Self {
                    common: $crate::core::config::builder::ModelInferenceConfig::with_defaults(
                        $model_name_opt, $batch_size_opt
                    ),
                    $( $field: $default_expr ),*
                }
            }
            /// Creates a new config instance using provided common configuration
            pub fn with_common(common: $crate::core::config::builder::ModelInferenceConfig) -> Self {
                Self {
                    common,
                    $( $field: $default_expr ),*
                }
            }
        }
    };
}

/// Macro to implement common builder methods for structs with a `ModelInferenceConfig` field.
#[macro_export]
macro_rules! impl_common_builder_methods {
    ($Builder:ident, $common_field:ident) => {
        impl $Builder {
            /// Sets the model path
            pub fn model_path(mut self, model_path: impl Into<std::path::PathBuf>) -> Self {
                self.$common_field = self.$common_field.model_path(model_path);
                self
            }
            /// Sets the model name
            pub fn model_name(mut self, model_name: impl Into<String>) -> Self {
                self.$common_field = self.$common_field.model_name(model_name);
                self
            }
            /// Sets the batch size
            pub fn batch_size(mut self, batch_size: usize) -> Self {
                self.$common_field = self.$common_field.batch_size(batch_size);
                self
            }
            /// Enables or disables logging
            pub fn enable_logging(mut self, enable: bool) -> Self {
                self.$common_field = self.$common_field.enable_logging(enable);
                self
            }
            /// Sets the ONNX Runtime session configuration
            pub fn ort_session(
                mut self,
                config: $crate::core::config::onnx::OrtSessionConfig,
            ) -> Self {
                self.$common_field = self.$common_field.ort_session(config);
                self
            }
            /// Sets the session pool size for concurrent predictions (>=1)
            pub fn session_pool_size(mut self, size: usize) -> Self {
                self.$common_field = self.$common_field.session_pool_size(size);
                self
            }
        }
    };
}

/// Macro to inject common builder methods into an existing `impl Builder` block.
/// Use this inside `impl YourBuilder { ... }` and pass the field name that holds
/// `ModelInferenceConfig` (e.g., `common`).
#[macro_export]
macro_rules! common_builder_methods {
    ($common_field:ident) => {
        /// Sets the model path
        pub fn model_path(mut self, model_path: impl Into<std::path::PathBuf>) -> Self {
            self.$common_field = self.$common_field.model_path(model_path);
            self
        }
        /// Sets the model name
        pub fn model_name(mut self, model_name: impl Into<String>) -> Self {
            self.$common_field = self.$common_field.model_name(model_name);
            self
        }
        /// Sets the batch size
        pub fn batch_size(mut self, batch_size: usize) -> Self {
            self.$common_field = self.$common_field.batch_size(batch_size);
            self
        }
        /// Enables or disables logging
        pub fn enable_logging(mut self, enable: bool) -> Self {
            self.$common_field = self.$common_field.enable_logging(enable);
            self
        }
        /// Sets the ONNX Runtime session configuration
        pub fn ort_session(mut self, config: $crate::core::config::onnx::OrtSessionConfig) -> Self {
            self.$common_field = self.$common_field.ort_session(config);
            self
        }
        /// Sets the session pool size for concurrent predictions (>=1)
        pub fn session_pool_size(mut self, size: usize) -> Self {
            self.$common_field = self.$common_field.session_pool_size(size);
            self
        }
    };
}

#[cfg(test)]
mod tests {

    // Test configuration structs
    #[derive(Debug, Default)]
    struct TestConfig {
        simple_field: Option<String>,
        nested_config: Option<NestedConfig>,
        enable_field: Option<EnabledFeature>,
    }

    #[derive(Debug, Default)]
    struct NestedConfig {
        nested_field: Option<i32>,
    }

    impl NestedConfig {
        fn new() -> Self {
            Self::default()
        }
    }

    #[derive(Debug, Default)]
    struct EnabledFeature {
        _enabled: bool,
    }

    // Test builder struct
    #[derive(Debug)]
    struct TestBuilder {
        config: TestConfig,
    }

    impl TestBuilder {
        fn new() -> Self {
            Self {
                config: TestConfig::default(),
            }
        }

        fn get_config(&self) -> &TestConfig {
            &self.config
        }
    }

    // Apply the macro to generate builder methods (separate calls for each type)
    impl_complete_builder! {
        builder: TestBuilder,
        config_field: config,
        simple_setters: {
            simple_field: String => "Sets a simple field value",
        }
    }

    impl_complete_builder! {
        builder: TestBuilder,
        config_field: config,
        nested_setters: {
            nested_config: NestedConfig => {
                nested_field: i32 => "Sets a nested field value",
            },
        }
    }

    impl_complete_builder! {
        builder: TestBuilder,
        config_field: config,
        enable_methods: {
            enable_feature => enable_field: EnabledFeature => "Enables a feature with default configuration",
        }
    }

    #[test]
    fn test_impl_complete_builder_nested_setter() {
        let builder = TestBuilder::new().nested_field(42);

        assert!(builder.get_config().nested_config.is_some());
        assert_eq!(
            builder
                .get_config()
                .nested_config
                .as_ref()
                .unwrap()
                .nested_field,
            Some(42)
        );
    }

    #[test]
    fn test_impl_complete_builder_enable_method() {
        let builder = TestBuilder::new().enable_feature();

        assert!(builder.get_config().enable_field.is_some());
    }

    #[test]
    fn test_impl_complete_builder_chaining() {
        let builder = TestBuilder::new()
            .simple_field("test".to_string())
            .nested_field(123)
            .enable_feature();

        let config = builder.get_config();
        assert_eq!(config.simple_field, Some("test".to_string()));
        assert!(config.nested_config.is_some());
        assert_eq!(
            config.nested_config.as_ref().unwrap().nested_field,
            Some(123)
        );
        assert!(config.enable_field.is_some());
    }
}
