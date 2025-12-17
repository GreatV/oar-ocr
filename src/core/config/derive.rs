//! Utilities for ConfigValidator trait implementation.
//!
//! This module provides macros and utilities to make implementing ConfigValidator
//! easier and more consistent across configuration types.

/// Macro to implement ConfigValidator with basic validation patterns.
///
/// This macro provides a convenient way to implement ConfigValidator for configuration
/// structs with common validation patterns.
///
/// # Example
///
/// ```rust,no_run
/// // use oar_ocr::core::config::{ConfigValidator, ConfigError};
/// // use oar_ocr::impl_config_validator;
/// // use serde::{Serialize, Deserialize};
///
/// // #[derive(Debug, Clone, Serialize, Deserialize, Default)]
/// // pub struct MyConfig {
/// //     pub confidence_threshold: f32,
/// //     pub batch_size: usize,
/// //     pub optional_threshold: Option<f32>,
/// // }
///
/// // impl_config_validator!(MyConfig {
/// //     confidence_threshold: range(0.0, 1.0),
/// //     batch_size: min(1),
/// //     optional_threshold: optional_range(0.0, 1.0),
/// // });
/// ```
#[macro_export]
macro_rules! impl_config_validator {
    ($type_name:ident { $($field:ident: $validator:ident $(($($args:tt)*))?),* $(,)? }) => {
        impl $crate::core::config::ConfigValidator for $type_name {
            fn validate(&self) -> Result<(), $crate::core::config::ConfigError> {
                $(
                    $crate::validate_field!(self, $field, $validator $(($($args)*))?);
                )*
                Ok(())
            }

            fn get_defaults() -> Self
            where
                Self: Sized,
            {
                Self::default()
            }
        }
    };
}

/// Helper macro for field validation.
#[macro_export]
macro_rules! validate_field {
    ($self:expr, $field:ident, range($min:expr, $max:expr)) => {
        if !($min..=$max).contains(&$self.$field) {
            return Err($crate::core::config::ConfigError::InvalidConfig {
                message: format!(
                    "{} must be between {} and {}",
                    stringify!($field),
                    $min,
                    $max
                ),
            });
        }
    };

    ($self:expr, $field:ident, min($min_val:expr)) => {
        if $self.$field < $min_val {
            return Err($crate::core::config::ConfigError::InvalidConfig {
                message: format!("{} must be at least {}", stringify!($field), $min_val),
            });
        }
    };

    ($self:expr, $field:ident, max($max_val:expr)) => {
        if $self.$field > $max_val {
            return Err($crate::core::config::ConfigError::InvalidConfig {
                message: format!("{} must be at most {}", stringify!($field), $max_val),
            });
        }
    };

    ($self:expr, $field:ident, optional_range($min:expr, $max:expr)) => {
        if let Some(value) = $self.$field {
            if !($min..=$max).contains(&value) {
                return Err($crate::core::config::ConfigError::InvalidConfig {
                    message: format!(
                        "{} must be between {} and {}",
                        stringify!($field),
                        $min,
                        $max
                    ),
                });
            }
        }
    };

    ($self:expr, $field:ident, path) => {
        $self.validate_model_path(&$self.$field)?;
    };

    ($self:expr, $field:ident, optional_path) => {
        if let Some(ref path) = $self.$field {
            $self.validate_model_path(path)?;
        }
    };
}
