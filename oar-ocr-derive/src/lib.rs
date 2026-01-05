//! Procedural derive macros for oar-ocr.
//!
//! This crate provides derive macros to reduce boilerplate in the oar-ocr library.

use proc_macro::TokenStream;
use quote::quote;
use syn::{DeriveInput, Expr, Field, Meta, Type, parse_macro_input};

/// Derive macro for implementing ConfigValidator trait.
///
/// This macro generates a `ConfigValidator` implementation for configuration structs.
/// Validation rules are specified using the `#[validate(...)]` attribute on fields.
///
/// # Supported Validators
///
/// - `#[validate(range(min, max))]` - Validates that the field value is within the inclusive range [min, max]
/// - `#[validate(min(value))]` - Validates that the field value is at least `value`
/// - `#[validate(max(value))]` - Validates that the field value is at most `value`
/// - `#[validate(optional_range(min, max))]` - Like `range`, but for `Option<T>` fields (only validates if Some)
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
///     #[validate(range(0.0, 1.0))]
///     pub score_threshold: f32,
///
///     #[validate(range(0.0, 1.0))]
///     pub box_threshold: f32,
///
///     #[validate(min(0.0))]
///     pub unclip_ratio: f32,
///
///     #[validate(min(1))]
///     pub max_candidates: usize,
///
///     // Fields without #[validate] are not validated
///     pub limit_side_len: Option<u32>,
/// }
/// ```
#[proc_macro_derive(ConfigValidator, attributes(validate))]
pub fn derive_config_validator(input: TokenStream) -> TokenStream {
    let input = parse_macro_input!(input as DeriveInput);
    impl_config_validator(&input)
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

fn impl_config_validator(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let name = &input.ident;

    let fields = match &input.data {
        syn::Data::Struct(data) => match &data.fields {
            syn::Fields::Named(fields) => &fields.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    input,
                    "ConfigValidator can only be derived for structs with named fields",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                input,
                "ConfigValidator can only be derived for structs",
            ));
        }
    };

    let validations = fields
        .iter()
        .filter_map(|field| generate_field_validation(field).transpose())
        .collect::<syn::Result<Vec<_>>>()?;

    Ok(quote! {
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
    })
}

fn generate_field_validation(field: &Field) -> syn::Result<Option<proc_macro2::TokenStream>> {
    let field_name = field
        .ident
        .as_ref()
        .ok_or_else(|| syn::Error::new_spanned(field, "Expected named field"))?;

    let field_name_str = field_name.to_string();

    let mut validations = Vec::new();

    for attr in &field.attrs {
        if !attr.path().is_ident("validate") {
            continue;
        }

        let meta = attr.parse_args::<Meta>()?;
        validations.push(generate_validation_code(
            field_name,
            &field_name_str,
            &meta,
            &field.ty,
        )?);
    }

    if validations.is_empty() {
        Ok(None)
    } else {
        Ok(Some(quote! { #(#validations)* }))
    }
}

fn generate_validation_code(
    field_name: &syn::Ident,
    field_name_str: &str,
    meta: &Meta,
    _field_ty: &Type,
) -> syn::Result<proc_macro2::TokenStream> {
    match meta {
        Meta::List(list) => {
            let validator_name = list
                .path
                .get_ident()
                .ok_or_else(|| syn::Error::new_spanned(&list.path, "Expected validator name"))?;
            let validator_str = validator_name.to_string();

            match validator_str.as_str() {
                "range" => {
                    let args = parse_two_args(&list.tokens)?;
                    let (min_expr, max_expr) = args;
                    Ok(quote! {
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
                    })
                }
                "min" => {
                    let min_expr = parse_one_arg(&list.tokens)?;
                    Ok(quote! {
                        if self.#field_name < #min_expr {
                            return Err(crate::core::config::ConfigError::InvalidConfig {
                                message: format!("{} must be at least {}", #field_name_str, #min_expr),
                            });
                        }
                    })
                }
                "max" => {
                    let max_expr = parse_one_arg(&list.tokens)?;
                    Ok(quote! {
                        if self.#field_name > #max_expr {
                            return Err(crate::core::config::ConfigError::InvalidConfig {
                                message: format!("{} must be at most {}", #field_name_str, #max_expr),
                            });
                        }
                    })
                }
                "optional_range" => {
                    let args = parse_two_args(&list.tokens)?;
                    let (min_expr, max_expr) = args;
                    Ok(quote! {
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
                    })
                }
                "path" => Ok(generate_path_validation(field_name, false)),
                "optional_path" => Ok(generate_path_validation(field_name, true)),
                other => Err(syn::Error::new_spanned(
                    validator_name,
                    format!("Unknown validator: {}", other),
                )),
            }
        }
        Meta::Path(path) => {
            let validator_name = path
                .get_ident()
                .ok_or_else(|| syn::Error::new_spanned(path, "Expected validator name"))?;
            let validator_str = validator_name.to_string();

            match validator_str.as_str() {
                "path" => Ok(generate_path_validation(field_name, false)),
                "optional_path" => Ok(generate_path_validation(field_name, true)),
                other => Err(syn::Error::new_spanned(
                    validator_name,
                    format!("Unknown validator without arguments: {}", other),
                )),
            }
        }
        _ => Err(syn::Error::new_spanned(meta, "Invalid validator format")),
    }
}

fn generate_path_validation(field_name: &syn::Ident, optional: bool) -> proc_macro2::TokenStream {
    if optional {
        quote! {
            if let Some(ref path) = self.#field_name {
                self.validate_model_path(path)?;
            }
        }
    } else {
        quote! {
            self.validate_model_path(&self.#field_name)?;
        }
    }
}

fn parse_one_arg(tokens: &proc_macro2::TokenStream) -> syn::Result<Expr> {
    syn::parse2(tokens.clone())
}

fn parse_two_args(tokens: &proc_macro2::TokenStream) -> syn::Result<(Expr, Expr)> {
    use syn::Token;
    use syn::parse::Parser;
    use syn::punctuated::Punctuated;

    let parser = Punctuated::<Expr, Token![,]>::parse_terminated;
    let args = parser.parse2(tokens.clone())?;
    let mut iter = args.into_iter();

    let first = iter
        .next()
        .ok_or_else(|| syn::Error::new_spanned(tokens, "Expected two arguments"))?;
    let second = iter
        .next()
        .ok_or_else(|| syn::Error::new_spanned(tokens, "Expected two arguments"))?;

    if iter.next().is_some() {
        return Err(syn::Error::new_spanned(
            tokens,
            "Expected exactly two arguments",
        ));
    }

    Ok((first, second))
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
    impl_task_predictor_builder(&input)
        .unwrap_or_else(|err| err.to_compile_error())
        .into()
}

fn impl_task_predictor_builder(input: &DeriveInput) -> syn::Result<proc_macro2::TokenStream> {
    let name = &input.ident;

    // Find the #[builder(config = Type)] attribute
    let config_type = find_builder_config_type(input)?;

    // Verify the struct has a `state` field
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

fn find_builder_config_type(input: &DeriveInput) -> syn::Result<Type> {
    for attr in &input.attrs {
        if !attr.path().is_ident("builder") {
            continue;
        }

        let meta = attr.parse_args::<Meta>()?;

        if let Meta::NameValue(nv) = meta
            && nv.path.is_ident("config")
        {
            if let Expr::Path(expr_path) = nv.value {
                return Ok(Type::Path(syn::TypePath {
                    qself: None,
                    path: expr_path.path,
                }));
            } else {
                return Err(syn::Error::new_spanned(
                    nv.value,
                    "Expected a type path, e.g., #[builder(config = MyConfigType)]",
                ));
            }
        }
    }

    Err(syn::Error::new_spanned(
        input,
        "Missing #[builder(config = ConfigType)] attribute",
    ))
}

fn verify_state_field(input: &DeriveInput) -> syn::Result<()> {
    let fields = match &input.data {
        syn::Data::Struct(data) => match &data.fields {
            syn::Fields::Named(fields) => &fields.named,
            _ => {
                return Err(syn::Error::new_spanned(
                    input,
                    "TaskPredictorBuilder can only be derived for structs with named fields",
                ));
            }
        },
        _ => {
            return Err(syn::Error::new_spanned(
                input,
                "TaskPredictorBuilder can only be derived for structs",
            ));
        }
    };

    let state_field = fields
        .iter()
        .find(|f| f.ident.as_ref().is_some_and(|ident| ident == "state"));

    let state_field = match state_field {
        Some(field) => field,
        None => {
            return Err(syn::Error::new_spanned(
                input,
                "Struct must have a `state` field of type PredictorBuilderState<Config>",
            ));
        }
    };

    // Verify the type is PredictorBuilderState<...>
    if !is_predictor_builder_state_type(&state_field.ty) {
        return Err(syn::Error::new_spanned(
            &state_field.ty,
            "Field `state` must be of type PredictorBuilderState<Config>",
        ));
    }

    Ok(())
}

fn is_predictor_builder_state_type(ty: &Type) -> bool {
    let Type::Path(type_path) = ty else {
        return false;
    };

    let last_segment = match type_path.path.segments.last() {
        Some(seg) => seg,
        None => return false,
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
