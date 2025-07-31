//! Macros for the OCR pipeline.
//!
//! This module defines various macros that are used throughout the OCR pipeline
//! to reduce code duplication and provide common functionality for implementing
//! predictors, builders, and configuration validation.

/// Implements the Predictor trait for a generic type.
///
/// This macro generates implementations of the Predictor trait for a given type,
/// providing default implementations for the predict_single and predict_batch methods.
///
/// # Parameters
///
/// * `$type` - The type to implement the Predictor trait for.
///
/// # Example
///
/// ```rust,ignore
/// use oar_ocr::impl_predictor_from_generic;
/// # struct MyPredictor;
/// impl_predictor_from_generic!(MyPredictor);
/// ```
#[macro_export]
macro_rules! impl_predictor_from_generic {
    ($type:ty) => {
        impl $crate::core::traits::Predictor for $type {
            fn predict_single(
                &mut self,
                image_path: &std::path::Path,
            ) -> Result<
                $crate::core::predictions::PredictionResult<'static>,
                $crate::core::errors::OCRError,
            > {
                let path_str = image_path.to_string_lossy();
                let batch_data = $crate::core::batch::BatchData::from_shared_arc_paths(
                    vec![std::sync::Arc::from(path_str.as_ref())],
                    vec![0],
                );
                let result =
                    <Self as $crate::core::traits::BasePredictor>::process(self, batch_data)?;
                Ok(self.convert_to_prediction_result(result))
            }

            fn predict_batch(
                &mut self,
                image_paths: &[&std::path::Path],
            ) -> Result<
                Vec<$crate::core::predictions::PredictionResult<'static>>,
                $crate::core::errors::OCRError,
            > {
                let string_paths: Vec<String> = image_paths
                    .iter()
                    .map(|p| p.to_string_lossy().into_owned())
                    .collect();
                let batches = self.batch_sampler().sample(string_paths);
                let mut results = Vec::new();

                for batch in batches {
                    let result =
                        <Self as $crate::core::traits::BasePredictor>::process(self, batch)?;
                    results.push(self.convert_to_prediction_result(result));
                }

                Ok(results)
            }

            fn model_name(&self) -> &str {
                <Self as $crate::core::traits::BasePredictor>::model_name(self)
            }

            fn predictor_type_name(&self) -> &str {
                <Self as $crate::core::traits::BasePredictor>::predictor_type_name(self)
            }
        }
    };
}

/// Implements a builder pattern for a predictor.
///
/// This macro generates a builder struct and implementations for a predictor,
/// providing a fluent API for configuring and building the predictor.
///
/// # Parameters
///
/// * `$builder_name` - The name of the builder struct to generate.
/// * `$predictor_name` - The name of the predictor struct.
/// * `{ $( $field:ident : $t:ty ),* }` - A list of fields for the builder.
///
/// # Example
///
/// ```rust,ignore
/// use oar_ocr::impl_builder;
/// # struct MyBuilder { model_path: Option<String>, batch_size: Option<usize> }
/// # struct MyPredictor;
/// impl_builder!(MyBuilder, MyPredictor, { model_path: String, batch_size: usize });
/// ```
#[macro_export]
macro_rules! impl_builder {
    ($builder_name:ident, $predictor_name:ident, { $( $field:ident : $t:ty ),* $(,)? } ) => {
        impl $builder_name {
            pub fn new() -> Self {
                Self {
                    $( $field: None, )*
                }
            }

            $(
                pub fn $field(mut self, $field: $t) -> Self {
                    self.$field = Some($field);
                    self
                }
            )*

            pub fn build(self, model_path: impl AsRef<Path>) -> Result<$predictor_name, OCRError> {
                self.build_internal(model_path.as_ref())
            }
        }

        impl Default for $builder_name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}

/// Implements an enhanced builder pattern for a predictor with validation.
///
/// This macro generates a builder struct and implementations for a predictor,
/// providing a fluent API for configuring and building the predictor with validation.
///
/// # Parameters
///
/// * `$builder_name` - The name of the builder struct to generate.
/// * `$predictor_name` - The name of the predictor struct.
/// * `$error_type` - The type of error to return for validation errors.
/// * `$predictor_type` - A string expression representing the predictor type.
/// * `{ $( $field:ident : $t:ty ),* }` - A list of fields for the builder.
/// * `{ $( $val_field:ident => $validation:tt ),* }` - A list of fields and their validation rules.
///
/// # Example
///
/// ```rust,ignore
/// use oar_ocr::impl_enhanced_builder;
/// # struct MyBuilder { model_path: Option<String>, batch_size: Option<usize> }
/// # struct MyPredictor;
/// # struct MyError;
/// impl_enhanced_builder!(MyBuilder, MyPredictor, MyError, "my_predictor",
///     { model_path: String, batch_size: usize },
///     { batch_size => positive_int });
/// ```
#[macro_export]
macro_rules! impl_enhanced_builder {

    ($builder_name:ident, $predictor_name:ident, $error_type:ident, $predictor_type:expr,
     { $( $field:ident : $t:ty ),* $(,)? },
     { $( $val_field:ident => $validation:tt ),* $(,)? }) => {

        impl $builder_name {
            pub fn new() -> Self {
                Self {
                    $( $field: None, )*
                }
            }

            $(
                pub fn $field(mut self, $field: $t) -> Self {
                    self.$field = Some($field);
                    self
                }
            )*

            pub fn build(self, model_path: impl AsRef<Path>) -> Result<$predictor_name, $error_type> {

                self.validate()?;
                self.build_internal(model_path.as_ref())
            }


            pub fn validate(&self) -> Result<(), $error_type> {
                $(
                    $crate::impl_enhanced_builder!(@validate_field self, $val_field, self.$val_field, $validation, $error_type);
                )*
                Ok(())
            }


            #[allow(dead_code)]
            fn validate_batch_size(&self, batch_size: Option<usize>) -> Result<(), $error_type> {
                if let Some(size) = batch_size {
                    if size == 0 {
                        return Err($error_type::ConfigError {
                            message: "Batch size must be greater than 0".to_string(),
                        });
                    }
                }
                Ok(())
            }

            #[allow(dead_code)]
            fn validate_positive(&self, value: Option<f32>, field_name: &str) -> Result<(), $error_type> {
                if let Some(val) = value {
                    if val <= 0.0 {
                        return Err($error_type::ConfigError {
                            message: format!("{} must be greater than 0", field_name),
                        });
                    }
                }
                Ok(())
            }

            #[allow(dead_code)]
            fn validate_positive_int(&self, value: Option<usize>, field_name: &str) -> Result<(), $error_type> {
                if let Some(val) = value {
                    if val == 0 {
                        return Err($error_type::ConfigError {
                            message: format!("{} must be greater than 0", field_name),
                        });
                    }
                }
                Ok(())
            }
        }

        impl Default for $builder_name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $crate::core::traits::PredictorBuilder for $builder_name {
            type Predictor = $predictor_name;

            fn build_typed(self, model_path: &Path) -> Result<Self::Predictor, $crate::core::errors::OCRError> {
                self.build_internal(model_path)
            }

            fn build_predictor(self, model_path: &Path) -> Result<Box<dyn $crate::core::traits::Predictor>, $crate::core::errors::OCRError> {
                Ok(Box::new(self.build_internal(model_path)?))
            }

            fn predictor_type(&self) -> &str {
                $predictor_type
            }
        }
    };


    ($builder_name:ident, $predictor_name:ident, $error_type:ident, $predictor_type:expr, { $( $field:ident : $t:ty ),* $(,)? }) => {
        $crate::impl_enhanced_builder!($builder_name, $predictor_name, $error_type, $predictor_type, { $( $field: $t ),* }, {});
    };


    ($builder_name:ident, $predictor_name:ident, $error_type:ident, $predictor_type:expr,
     { $( $field:ident : $t:ty ),* $(,)? },
     { $( $val_field:ident => $validation:tt ),* $(,)? },
     defaults: { $( $default_field:ident = $default_value:expr ),* $(,)? }) => {

        impl $builder_name {
            pub fn new() -> Self {
                Self {
                    $( $field: None, )*
                }
            }

            $(
                pub fn $field(mut self, $field: $t) -> Self {
                    self.$field = Some($field);
                    self
                }
            )*


            $(
                pub fn $default_field(&self) -> $t {
                    self.$default_field.clone().unwrap_or($default_value)
                }
            )*

            pub fn build(self, model_path: impl AsRef<Path>) -> Result<$predictor_name, $error_type> {
                self.validate()?;
                self.build_internal(model_path.as_ref())
            }

            pub fn validate(&self) -> Result<(), $error_type> {
                $(
                    $crate::impl_enhanced_builder!(@validate_field self, $val_field, self.$val_field, $validation, $error_type);
                )*
                Ok(())
            }
        }

        impl Default for $builder_name {
            fn default() -> Self {
                Self::new()
            }
        }

        impl $crate::core::traits::PredictorBuilder for $builder_name {
            type Predictor = $predictor_name;

            fn build_typed(self, model_path: &Path) -> Result<Self::Predictor, $crate::core::errors::OCRError> {
                self.build_internal(model_path)
            }

            fn build_predictor(self, model_path: &Path) -> Result<Box<dyn $crate::core::traits::Predictor>, $crate::core::errors::OCRError> {
                Ok(Box::new(self.build_internal(model_path)?))
            }

            fn predictor_type(&self) -> &str {
                $predictor_type
            }
        }
    };


    (@validate_field $self:ident, $field:ident, $value:expr, positive, $error_type:ident) => {
        if let Some(val) = $value {
            if val <= 0.0 {
                return Err($error_type::ConfigError {
                    message: format!("{} must be greater than 0", stringify!($field)),
                });
            }
        }
    };

    (@validate_field $self:ident, $field:ident, $value:expr, positive_int, $error_type:ident) => {
        if let Some(val) = $value {
            if val == 0 {
                return Err($error_type::ConfigError {
                    message: format!("{} must be greater than 0", stringify!($field)),
                });
            }
        }
    };

    (@validate_field $self:ident, $field:ident, $value:expr, range($min:expr, $max:expr), $error_type:ident) => {
        if let Some(val) = $value {
            if val < $min || val > $max {
                return Err($error_type::ConfigError {
                    message: format!("{} must be between {} and {}", stringify!($field), $min, $max),
                });
            }
        }
    };
}

/// Implements configuration validation for a struct.
///
/// This macro generates validation methods for a configuration struct.
///
/// # Parameters
///
/// * `$config_name` - The name of the configuration struct.
/// * `$error_type` - The type of error to return for validation errors.
/// * `{ $( $field:ident : $t:ty ),* }` - A list of fields for the configuration.
///
/// # Example
///
/// ```rust,ignore
/// use oar_ocr::impl_config_validation;
/// # struct MyConfig { batch_size: usize, threshold: f32 }
/// # struct MyError;
/// impl_config_validation!(MyConfig, MyError, { batch_size: usize, threshold: f32 });
/// ```
#[macro_export]
macro_rules! impl_config_validation {
    ($config_name:ident, $error_type:ident, { $( $field:ident : $t:ty ),* $(,)? } ) => {
        impl $config_name {
            pub fn validate(&self) -> Result<(), $error_type> {
                $(
                    self.validate_$field()?;
                )*
                Ok(())
            }

            $(
                fn validate_$field(&self) -> Result<(), $error_type> {
                    Ok(())
                }
            )*
        }
    };
}

/// Implements a builder configuration struct.
///
/// This macro generates a configuration struct with optional fields and default values.
///
/// # Parameters
///
/// * `$config_name` - The name of the configuration struct to generate.
/// * `{ $( $field:ident : $t:ty = $default:expr ),* }` - A list of fields with their types and default values.
///
/// # Example
///
/// ```rust,ignore
/// use oar_ocr::impl_builder_config;
/// impl_builder_config!(MyConfig, { batch_size: usize = 32, threshold: f32 = 0.5 });
/// ```
#[macro_export]
macro_rules! impl_builder_config {
    ($config_name:ident, { $( $field:ident : $t:ty = $default:expr ),* $(,)? } ) => {
        #[derive(Debug, Clone)]
        pub struct $config_name {
            $( pub $field: Option<$t>, )*
        }

        impl $config_name {
            pub fn new() -> Self {
                Self {
                    $( $field: None, )*
                }
            }

            $(
                pub fn $field(mut self, value: $t) -> Self {
                    self.$field = Some(value);
                    self
                }
            )*


            $(
                pub fn $field(&self) -> $t {
                    self.$field.clone().unwrap_or($default)
                }
            )*
        }

        impl Default for $config_name {
            fn default() -> Self {
                Self::new()
            }
        }
    };
}
