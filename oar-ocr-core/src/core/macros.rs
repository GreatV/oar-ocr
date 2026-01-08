//! Macros for the OCR pipeline.
//!
//! This module provides utility macros to reduce code duplication across
//! the OCR pipeline, particularly for builder patterns and metrics collection.

/// Central task registry macro that defines all tasks in a single location.
///
/// This macro uses the "callback pattern" - it takes a callback macro name and
/// invokes it with the task registry data. Different consumers can process
/// the same data differently.
///
/// # Task Entry Format
///
/// Each task is defined as:
/// ```text
/// TaskName {
///     output: OutputType,    // fully qualified path
///     adapter: AdapterType,  // fully qualified path
///     constructor: constructor_name,
///     conversion: into_method_name,
///     doc: "Documentation string for IDE/rustdoc",
/// }
/// ```
///
/// The `TaskDefinition` trait provides task metadata (name, doc, empty()) for runtime.
/// The `doc` field in the registry is used for `#[doc]` attributes on enum variants.
#[macro_export]
macro_rules! with_task_registry {
    ($callback:path) => {
        $callback! {
            TextDetection {
                output: $crate::domain::tasks::TextDetectionOutput,
                adapter: $crate::domain::adapters::TextDetectionAdapter,
                constructor: text_detection,
                conversion: into_text_detection,
                doc: "Text detection - locating text regions in images",
            },
            TextRecognition {
                output: $crate::domain::tasks::TextRecognitionOutput,
                adapter: $crate::domain::adapters::TextRecognitionAdapter,
                constructor: text_recognition,
                conversion: into_text_recognition,
                doc: "Text recognition - converting text regions to strings",
            },
            DocumentOrientation {
                output: $crate::domain::tasks::DocumentOrientationOutput,
                adapter: $crate::domain::adapters::DocumentOrientationAdapter,
                constructor: document_orientation,
                conversion: into_document_orientation,
                doc: "Document orientation classification",
            },
            TextLineOrientation {
                output: $crate::domain::tasks::TextLineOrientationOutput,
                adapter: $crate::domain::adapters::TextLineOrientationAdapter,
                constructor: text_line_orientation,
                conversion: into_text_line_orientation,
                doc: "Text line orientation classification",
            },
            DocumentRectification {
                output: $crate::domain::tasks::DocumentRectificationOutput,
                adapter: $crate::domain::adapters::UVDocRectifierAdapter,
                constructor: document_rectification,
                conversion: into_document_rectification,
                doc: "Document rectification/unwarp",
            },
            LayoutDetection {
                output: $crate::domain::tasks::LayoutDetectionOutput,
                adapter: $crate::domain::adapters::LayoutDetectionAdapter,
                constructor: layout_detection,
                conversion: into_layout_detection,
                doc: "Layout detection/analysis",
            },
            TableCellDetection {
                output: $crate::domain::tasks::TableCellDetectionOutput,
                adapter: $crate::domain::adapters::TableCellDetectionAdapter,
                constructor: table_cell_detection,
                conversion: into_table_cell_detection,
                doc: "Table cell detection - locating cells within table regions",
            },
            FormulaRecognition {
                output: $crate::domain::tasks::FormulaRecognitionOutput,
                adapter: $crate::domain::adapters::FormulaRecognitionAdapter,
                constructor: formula_recognition,
                conversion: into_formula_recognition,
                doc: "Formula recognition - converting mathematical formulas to LaTeX",
            },
            SealTextDetection {
                output: $crate::domain::tasks::SealTextDetectionOutput,
                adapter: $crate::domain::adapters::SealTextDetectionAdapter,
                constructor: seal_text_detection,
                conversion: into_seal_text_detection,
                doc: "Seal text detection - locating text regions in seal/stamp images",
            },
            TableClassification {
                output: $crate::domain::tasks::TableClassificationOutput,
                adapter: $crate::domain::adapters::TableClassificationAdapter,
                constructor: table_classification,
                conversion: into_table_classification,
                doc: "Table classification - classifying table images as wired or wireless",
            },
            TableStructureRecognition {
                output: $crate::domain::tasks::TableStructureRecognitionOutput,
                adapter: $crate::domain::adapters::TableStructureRecognitionAdapter,
                constructor: table_structure_recognition,
                conversion: into_table_structure_recognition,
                doc: "Table structure recognition - recognizing table structure as HTML with bboxes",
            }
        }
    };
}

/// Generates the TaskType enum from the task registry.
///
/// Uses `TaskDefinition::TASK_NAME` for runtime metadata.
/// Uses `doc` field for `#[doc]` attributes on variants.
#[macro_export]
macro_rules! impl_task_type_enum {
    ($(
        $task:ident {
            output: $output:ty,
            adapter: $adapter:ty,
            constructor: $constructor:ident,
            conversion: $conversion:ident,
            doc: $doc:literal,
        }
    ),* $(,)?) => {
        /// Represents the type of OCR task being performed.
        #[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, serde::Serialize, serde::Deserialize)]
        pub enum TaskType {
            $(
                #[doc = $doc]
                $task,
            )*
        }

        impl TaskType {
            /// Returns a human-readable name for the task type.
            pub fn name(&self) -> &'static str {
                match self {
                    $(TaskType::$task => <$output as $crate::core::traits::TaskDefinition>::TASK_NAME,)*
                }
            }
        }
    };
}

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
    };
}

/// Internal helper macro that generates the common parts of adapter builders.
///
/// This macro generates:
/// - Builder struct definition
/// - `new()` constructor
/// - `base_adapter_info()` method
/// - Custom methods
/// - `Default` trait implementation
/// - `OrtConfigurable` trait implementation
///
/// It does NOT generate:
/// - `with_config()` inherent method (added separately for non-override variants)
/// - `AdapterBuilder` trait implementation (varies based on overrides)
#[doc(hidden)]
#[macro_export]
macro_rules! __impl_adapter_builder_common {
    (
        builder_name: $Builder:ident,
        adapter_name: $Adapter:ident,
        config_type: $Config:ty,
        adapter_type: $adapter_type_str:literal,
        adapter_desc: $adapter_desc:literal,
        task_type: $TaskType:ident,

        fields: {
            $($field_vis:vis $field_name:ident : $field_ty:ty = $field_default:expr),*
            $(,)?
        },

        methods: {
            $($method:item)*
        }
    ) => {
        /// Builder for [$Adapter].
        ///
        #[doc = $adapter_desc]
        #[derive(Debug)]
        pub struct $Builder {
            /// Common configuration shared across all adapters
            config: $crate::domain::adapters::builder_config::AdapterBuilderConfig<$Config>,
            $($field_vis $field_name : $field_ty),*
        }

        impl $Builder {
            /// Creates a new builder with default configuration.
            pub fn new() -> Self {
                Self {
                    config: $crate::domain::adapters::builder_config::AdapterBuilderConfig::default(),
                    $($field_name : $field_default),*
                }
            }

            /// Creates the base [`AdapterInfo`] for this adapter.
            ///
            /// This helper method constructs an [`AdapterInfo`] using the adapter's
            /// type, task type, and description from the macro.
            pub fn base_adapter_info() -> $crate::core::traits::adapter::AdapterInfo {
                $crate::core::traits::adapter::AdapterInfo::new(
                    $adapter_type_str,
                    $crate::core::traits::task::TaskType::$TaskType,
                    $adapter_desc,
                )
            }

            // Custom methods provided by the user
            $($method)*
        }

        impl Default for $Builder {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $crate::core::traits::OrtConfigurable for $Builder {
            fn with_ort_config(mut self, config: $crate::core::config::OrtSessionConfig) -> Self {
                self.config = self.config.with_ort_config(config);
                self
            }
        }
    };
}

/// Macro to implement common adapter builder boilerplate.
///
/// This macro generates the repetitive parts of adapter builders including the `build()` method.
/// Uses a callback pattern to work around Rust macro hygiene limitations.
///
/// Generates:
/// - Builder struct with `config` field plus custom fields
/// - `new()` constructor
/// - `with_config()` convenience method
/// - `Default` trait implementation
/// - `OrtConfigurable` trait implementation
/// - `AdapterBuilder` trait implementation (with callback-based `build()`)
///
/// # Syntax
///
/// ```rust,ignore
/// impl_adapter_builder! {
///     // Required: Type information
///     builder_name: MyAdapterBuilder,
///     adapter_name: MyAdapter,
///     config_type: MyConfig,
///     adapter_type: "MyAdapter",
///     adapter_desc: "Description",
///     task_type: MyTaskType,
///
///     // Optional: Custom fields
///     fields: {
///         pub custom_field: Option<String> = None,
///     },
///
///     // Optional: Custom methods
///     methods: {
///         pub fn custom_method(mut self, value: String) -> Self {
///             self.custom_field = Some(value);
///             self
///         }
///     }
///
///     // Required: Build closure (use |builder, model_path| { ... })
///     build: |builder, model_path| {
///         let (task_config, ort_config) = builder.config.into_validated_parts()?;
///         let model = apply_ort_config!(
///             SomeModelBuilder::new(),
///             ort_config
///         ).build(model_path)?;
///         Ok(MyAdapter::new(model, task_config))
///     }
/// }
/// ```
#[macro_export]
macro_rules! impl_adapter_builder {
    // Full variant with fields and methods (no overrides)
    (
        builder_name: $Builder:ident,
        adapter_name: $Adapter:ident,
        config_type: $Config:ty,
        adapter_type: $adapter_type_str:literal,
        adapter_desc: $adapter_desc:literal,
        task_type: $TaskType:ident,

        fields: {
            $($field_vis:vis $field_name:ident : $field_ty:ty = $field_default:expr),*
            $(,)?
        },

        methods: {
            $($method:item)*
        }

        build: $build_closure:expr,
    ) => {
        // Generate common parts (struct, new, base_adapter_info, Default, OrtConfigurable)
        $crate::__impl_adapter_builder_common! {
            builder_name: $Builder,
            adapter_name: $Adapter,
            config_type: $Config,
            adapter_type: $adapter_type_str,
            adapter_desc: $adapter_desc,
            task_type: $TaskType,

            fields: {
                $($field_vis $field_name : $field_ty = $field_default),*
            },

            methods: {
                /// Sets the task configuration.
                pub fn with_config(mut self, config: $Config) -> Self {
                    self.config = self.config.with_task_config(config);
                    self
                }

                $($method)*
            }
        }

        // Generate AdapterBuilder impl with standard methods
        impl $crate::core::traits::adapter::AdapterBuilder for $Builder {
            type Config = $Config;
            type Adapter = $Adapter;

            fn build(self, model_path: &std::path::Path) -> Result<Self::Adapter, $crate::core::OCRError> {
                let build_fn: fn(Self, &std::path::Path) -> Result<$Adapter, $crate::core::OCRError> = $build_closure;
                build_fn(self, model_path)
            }

            fn with_config(mut self, config: Self::Config) -> Self {
                self.config = self.config.with_task_config(config);
                self
            }

            fn adapter_type(&self) -> &str {
                $adapter_type_str
            }
        }
    };

    // Variant without custom fields
    (
        builder_name: $Builder:ident,
        adapter_name: $Adapter:ident,
        config_type: $Config:ty,
        adapter_type: $adapter_type_str:literal,
        adapter_desc: $adapter_desc:literal,
        task_type: $TaskType:ident,

        methods: {
            $($method:item)*
        }

        build: $build_closure:expr,
    ) => {
        impl_adapter_builder! {
            builder_name: $Builder,
            adapter_name: $Adapter,
            config_type: $Config,
            adapter_type: $adapter_type_str,
            adapter_desc: $adapter_desc,
            task_type: $TaskType,

            fields: {},

            methods: {
                $($method)*
            }

            build: $build_closure,
        }
    };

    // Variant without custom methods
    (
        builder_name: $Builder:ident,
        adapter_name: $Adapter:ident,
        config_type: $Config:ty,
        adapter_type: $adapter_type_str:literal,
        adapter_desc: $adapter_desc:literal,
        task_type: $TaskType:ident,

        fields: {
            $($field_vis:vis $field_name:ident : $field_ty:ty = $field_default:expr),*
            $(,)?
        }

        build: $build_closure:expr,
    ) => {
        impl_adapter_builder! {
            builder_name: $Builder,
            adapter_name: $Adapter,
            config_type: $Config,
            adapter_type: $adapter_type_str,
            adapter_desc: $adapter_desc,
            task_type: $TaskType,

            fields: {
                $($field_vis $field_name : $field_ty = $field_default),*
            },

            methods: {}

            build: $build_closure,
        }
    };

    // Minimal variant (no custom fields, no custom methods)
    (
        builder_name: $Builder:ident,
        adapter_name: $Adapter:ident,
        config_type: $Config:ty,
        adapter_type: $adapter_type_str:literal,
        adapter_desc: $adapter_desc:literal,
        task_type: $TaskType:ident

        build: $build_closure:expr,
    ) => {
        impl_adapter_builder! {
            builder_name: $Builder,
            adapter_name: $Adapter,
            config_type: $Config,
            adapter_type: $adapter_type_str,
            adapter_desc: $adapter_desc,
            task_type: $TaskType,

            fields: {},

            methods: {}

            build: $build_closure,
        }
    };

    // Variant with trait method overrides (for with_config, adapter_type)
    (
        builder_name: $Builder:ident,
        adapter_name: $Adapter:ident,
        config_type: $Config:ty,
        adapter_type: $adapter_type_str:literal,
        adapter_desc: $adapter_desc:literal,
        task_type: $TaskType:ident,

        fields: {
            $($field_vis:vis $field_name:ident : $field_ty:ty = $field_default:expr),*
            $(,)?
        },

        methods: {
            $($method:item)*
        }

        overrides: {
            with_config: $with_config_closure:expr,
            adapter_type: $adapter_type_closure:expr,
        }

        build: $build_closure:expr,
    ) => {
        // Generate common parts (struct, new, base_adapter_info, Default, OrtConfigurable)
        $crate::__impl_adapter_builder_common! {
            builder_name: $Builder,
            adapter_name: $Adapter,
            config_type: $Config,
            adapter_type: $adapter_type_str,
            adapter_desc: $adapter_desc,
            task_type: $TaskType,

            fields: {
                $($field_vis $field_name : $field_ty = $field_default),*
            },

            methods: {
                $($method)*
            }
        }

        // Generate AdapterBuilder impl with overridden methods
        impl $crate::core::traits::adapter::AdapterBuilder for $Builder {
            type Config = $Config;
            type Adapter = $Adapter;

            fn build(self, model_path: &std::path::Path) -> Result<Self::Adapter, $crate::core::OCRError> {
                let build_fn: fn(Self, &std::path::Path) -> Result<$Adapter, $crate::core::OCRError> = $build_closure;
                build_fn(self, model_path)
            }

            fn with_config(self, config: Self::Config) -> Self {
                let with_config_fn: fn(Self, Self::Config) -> Self = $with_config_closure;
                with_config_fn(self, config)
            }

            fn adapter_type(&self) -> &str {
                let adapter_type_fn: fn(&Self) -> &str = $adapter_type_closure;
                adapter_type_fn(self)
            }
        }
    };

    // Variant with only with_config override
    (
        builder_name: $Builder:ident,
        adapter_name: $Adapter:ident,
        config_type: $Config:ty,
        adapter_type: $adapter_type_str:literal,
        adapter_desc: $adapter_desc:literal,
        task_type: $TaskType:ident,

        fields: {
            $($field_vis:vis $field_name:ident : $field_ty:ty = $field_default:expr),*
            $(,)?
        },

        methods: {
            $($method:item)*
        }

        overrides: {
            with_config: $with_config_closure:expr,
        }

        build: $build_closure:expr,
    ) => {
        // Generate common parts (struct, new, base_adapter_info, Default, OrtConfigurable)
        $crate::__impl_adapter_builder_common! {
            builder_name: $Builder,
            adapter_name: $Adapter,
            config_type: $Config,
            adapter_type: $adapter_type_str,
            adapter_desc: $adapter_desc,
            task_type: $TaskType,

            fields: {
                $($field_vis $field_name : $field_ty = $field_default),*
            },

            methods: {
                $($method)*
            }
        }

        // Generate AdapterBuilder impl with with_config override
        impl $crate::core::traits::adapter::AdapterBuilder for $Builder {
            type Config = $Config;
            type Adapter = $Adapter;

            fn build(self, model_path: &std::path::Path) -> Result<Self::Adapter, $crate::core::OCRError> {
                let build_fn: fn(Self, &std::path::Path) -> Result<$Adapter, $crate::core::OCRError> = $build_closure;
                build_fn(self, model_path)
            }

            fn with_config(self, config: Self::Config) -> Self {
                let with_config_fn: fn(Self, Self::Config) -> Self = $with_config_closure;
                with_config_fn(self, config)
            }

            fn adapter_type(&self) -> &str {
                $adapter_type_str
            }
        }
    };
}

/// Macro to conditionally apply OrtSessionConfig to any builder that has `with_ort_config`.
///
/// This macro eliminates the repeated pattern:
/// ```rust,no_run
/// // let mut builder = SomeBuilder::new();
/// // if let Some(ort_config) = ort_config {
/// //     builder = builder.with_ort_config(ort_config);
/// // }
/// ```
///
/// Instead, use:
/// ```rust,no_run
/// // let builder = apply_ort_config!(SomeBuilder::new(), ort_config);
/// ```
///
/// # Usage
///
/// ```rust,no_run
/// // Works with any builder that has a `with_ort_config` method:
/// // let builder = apply_ort_config!(
/// //     DBModelBuilder::new()
/// //         .preprocess_config(config),
/// //     ort_config
/// // );
/// ```
#[macro_export]
macro_rules! apply_ort_config {
    ($builder:expr, $ort_config:expr) => {{
        let builder = $builder;
        if let Some(cfg) = $ort_config {
            builder.with_ort_config(cfg)
        } else {
            builder
        }
    }};
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
