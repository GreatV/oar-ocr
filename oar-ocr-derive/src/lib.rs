//! Procedural derive macros for oar-ocr.
//!
//! This crate provides derive macros to reduce boilerplate in the oar-ocr library.

use darling::{FromDeriveInput, FromField, FromMeta, ast};
use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, Expr, Type, parse_macro_input};

/// Parsed arguments for range validators: `range(min, max)` or `optional_range(min, max)`
#[derive(Debug, FromMeta)]
struct RangeArgs {
    min: Expr,
    max: Expr,
}

/// All supported validators that can be applied to a field.
#[derive(Debug, Default, FromMeta)]
struct Validators {
    /// `#[validate(range(min = expr, max = expr))]` - value must be in [min, max]
    #[darling(default)]
    range: Option<RangeArgs>,

    /// `#[validate(min = expr)]` - value must be >= expr
    #[darling(default)]
    min: Option<Expr>,

    /// `#[validate(max = expr)]` - value must be <= expr
    #[darling(default)]
    max: Option<Expr>,

    /// `#[validate(optional_range(min = expr, max = expr))]` - for Option<T> fields
    #[darling(default)]
    optional_range: Option<RangeArgs>,

    /// `#[validate(path)]` - validates path exists
    #[darling(default)]
    path: bool,

    /// `#[validate(optional_path)]` - validates path exists for Option<PathBuf>
    #[darling(default)]
    optional_path: bool,
}

/// A single field with its validation rules.
#[derive(Debug, FromField)]
#[darling(attributes(validate))]
struct ValidatedField {
    ident: Option<syn::Ident>,
    #[allow(dead_code)]
    ty: Type,
    #[darling(flatten)]
    validators: Validators,
}

/// The input struct for ConfigValidator derive.
#[derive(Debug, FromDeriveInput)]
#[darling(attributes(validate), supports(struct_named))]
struct ConfigValidatorInput {
    ident: syn::Ident,
    data: ast::Data<(), ValidatedField>,
}

/// Builder attribute: `#[builder(config = ConfigType)]`
#[derive(Debug, FromMeta)]
struct BuilderAttr {
    config: syn::Path,
}

/// A field in the builder struct.
#[derive(Debug, FromField)]
struct BuilderField {
    ident: Option<syn::Ident>,
    ty: Type,
}

/// The input struct for TaskPredictorBuilder derive.
#[derive(Debug, FromDeriveInput)]
#[darling(attributes(builder), supports(struct_named))]
struct TaskPredictorBuilderInput {
    ident: syn::Ident,
    data: ast::Data<(), BuilderField>,
    #[darling(flatten)]
    builder: BuilderAttr,
}

/// Derive macro for implementing ConfigValidator trait.
///
/// This macro generates a `ConfigValidator` implementation for configuration structs.
/// Validation rules are specified using the `#[validate(...)]` attribute on fields.
///
/// # Supported Validators
///
/// - `#[validate(range(min = value, max = value))]` - Validates that the field value is within [min, max]
/// - `#[validate(min = value)]` - Validates that the field value is at least `value`
/// - `#[validate(max = value)]` - Validates that the field value is at most `value`
/// - `#[validate(optional_range(min = value, max = value))]` - Like `range`, but for `Option<T>` fields
/// - `#[validate(path)]` - Validates that the path exists (for PathBuf fields)
/// - `#[validate(optional_path)]` - Like `path`, but for `Option<PathBuf>` fields
///
/// # Example
///
/// ```rust,ignore
/// use oar_ocr_derive::ConfigValidator;
///
/// #[derive(ConfigValidator, Default)]
/// pub struct TextDetectionConfig {
///     #[validate(range(min = 0.0, max = 1.0))]
///     pub score_threshold: f32,
///
///     #[validate(range(min = 0.0, max = 1.0))]
///     pub box_threshold: f32,
///
///     #[validate(min = 0.0)]
///     pub unclip_ratio: f32,
///
///     #[validate(min = 1)]
///     pub max_candidates: usize,
///
///     // Fields without #[validate] are not validated
///     pub limit_side_len: Option<u32>,
/// }
/// ```
#[proc_macro_derive(ConfigValidator, attributes(validate))]
pub fn derive_config_validator(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    ConfigValidatorInput::from_derive_input(&input)
        .map(|parsed| generate_config_validator(&parsed))
        .unwrap_or_else(|err| err.write_errors())
        .into()
}

fn generate_config_validator(input: &ConfigValidatorInput) -> proc_macro2::TokenStream {
    let name = &input.ident;

    let fields = input
        .data
        .as_ref()
        .take_struct()
        .expect("Only structs are supported");

    let validations: Vec<_> = fields
        .iter()
        .filter_map(|field| generate_field_validation(field))
        .collect();

    quote! {
        impl crate::core::config::ConfigValidator for #name {
            fn validate(&self) -> Result<(), crate::core::config::ConfigError> {
                #(#validations)*
                Ok(())
            }

            fn get_defaults() -> Self
            where
                Self: Sized,
            {
                Self::default()
            }
        }
    }
}

fn generate_field_validation(field: &ValidatedField) -> Option<proc_macro2::TokenStream> {
    let field_name = field.ident.as_ref()?;
    let field_name_str = field_name.to_string();
    let validators = &field.validators;

    let mut validations = Vec::new();

    // Range validation
    if let Some(range) = &validators.range {
        let min_expr = &range.min;
        let max_expr = &range.max;
        validations.push(quote! {
            if !(#min_expr..=#max_expr).contains(&self.#field_name) {
                return Err(crate::core::config::ConfigError::InvalidConfig {
                    message: format!(
                        "{} must be between {} and {}",
                        #field_name_str,
                        #min_expr,
                        #max_expr
                    ),
                });
            }
        });
    }

    // Min validation
    if let Some(min_expr) = &validators.min {
        validations.push(quote! {
            if self.#field_name < #min_expr {
                return Err(crate::core::config::ConfigError::InvalidConfig {
                    message: format!("{} must be at least {}", #field_name_str, #min_expr),
                });
            }
        });
    }

    // Max validation
    if let Some(max_expr) = &validators.max {
        validations.push(quote! {
            if self.#field_name > #max_expr {
                return Err(crate::core::config::ConfigError::InvalidConfig {
                    message: format!("{} must be at most {}", #field_name_str, #max_expr),
                });
            }
        });
    }

    // Optional range validation
    if let Some(range) = &validators.optional_range {
        let min_expr = &range.min;
        let max_expr = &range.max;
        validations.push(quote! {
            if let Some(value) = self.#field_name {
                if !(#min_expr..=#max_expr).contains(&value) {
                    return Err(crate::core::config::ConfigError::InvalidConfig {
                        message: format!(
                            "{} must be between {} and {}",
                            #field_name_str,
                            #min_expr,
                            #max_expr
                        ),
                    });
                }
            }
        });
    }

    // Path validation
    if validators.path {
        validations.push(quote! {
            self.validate_model_path(&self.#field_name)?;
        });
    }

    // Optional path validation
    if validators.optional_path {
        validations.push(quote! {
            if let Some(ref path) = self.#field_name {
                self.validate_model_path(path)?;
            }
        });
    }

    if validations.is_empty() {
        None
    } else {
        Some(quote! { #(#validations)* })
    }
}

/// Derive macro for implementing TaskPredictorBuilder trait.
///
/// This macro generates the `TaskPredictorBuilder` trait implementation and
/// common builder methods (`with_config`, `with_ort_config`).
///
/// # Requirements
///
/// - The struct must have a field named `state` of type `PredictorBuilderState<Config>`
/// - The config type must be specified using `#[builder(config = ConfigType)]`
///
/// # Example
///
/// ```rust,ignore
/// use oar_ocr_derive::TaskPredictorBuilder;
/// use oar_ocr::predictors::builder::PredictorBuilderState;
///
/// #[derive(TaskPredictorBuilder)]
/// #[builder(config = TextDetectionConfig)]
/// pub struct TextDetectionPredictorBuilder {
///     state: PredictorBuilderState<TextDetectionConfig>,
/// }
/// ```
#[proc_macro_derive(TaskPredictorBuilder, attributes(builder))]
pub fn derive_task_predictor_builder(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);

    TaskPredictorBuilderInput::from_derive_input(&input)
        .and_then(|parsed| generate_task_predictor_builder(&parsed))
        .unwrap_or_else(|err| err.write_errors())
        .into()
}

fn generate_task_predictor_builder(
    input: &TaskPredictorBuilderInput,
) -> darling::Result<proc_macro2::TokenStream> {
    let name = &input.ident;
    let config_type = &input.builder.config;

    // Verify the struct has a `state` field with correct type
    verify_state_field(input)?;

    Ok(quote! {
        impl crate::predictors::builder::TaskPredictorBuilder for #name {
            type Config = #config_type;

            fn state_mut(
                &mut self,
            ) -> &mut crate::predictors::builder::PredictorBuilderState<Self::Config> {
                &mut self.state
            }
        }

        impl #name {
            /// Replace the full task configuration used by this builder.
            pub fn with_config(self, config: #config_type) -> Self {
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
        }
    })
}

fn verify_state_field(input: &TaskPredictorBuilderInput) -> darling::Result<()> {
    let fields = input
        .data
        .as_ref()
        .take_struct()
        .expect("Only structs are supported");

    let state_field = fields
        .iter()
        .find(|f| f.ident.as_ref().is_some_and(|ident| ident == "state"));

    let state_field = match state_field {
        Some(field) => field,
        None => {
            return Err(darling::Error::custom(
                "Struct must have a `state` field of type PredictorBuilderState<Config>",
            ));
        }
    };

    // Verify the type is PredictorBuilderState<...>
    if !is_predictor_builder_state_type(&state_field.ty) {
        return Err(darling::Error::custom(
            "Field `state` must be of type PredictorBuilderState<Config>",
        )
        .with_span(&state_field.ty));
    }

    Ok(())
}

fn is_predictor_builder_state_type(ty: &Type) -> bool {
    let Type::Path(type_path) = ty else {
        return false;
    };

    let Some(last_segment) = type_path.path.segments.last() else {
        return false;
    };

    if last_segment.ident != "PredictorBuilderState" {
        return false;
    }

    // Verify it has exactly one generic argument
    matches!(
        &last_segment.arguments,
        syn::PathArguments::AngleBracketed(args) if args.args.len() == 1
    )
}
