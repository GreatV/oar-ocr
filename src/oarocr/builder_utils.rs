//! Shared utilities for builder patterns in oarocr module.

use crate::core::OCRError;
use crate::core::config::OrtSessionConfig;
use crate::core::traits::OrtConfigurable;
use crate::core::traits::adapter::AdapterBuilder;
use std::path::PathBuf;

/// Builds an optional adapter from a model path using a builder factory.
///
/// This helper reduces boilerplate when building adapters that only need
/// a model path and optional ORT session configuration.
///
/// # Type Parameters
///
/// * `B` - A builder type that implements both `AdapterBuilder` and `OrtConfigurable`
///
/// # Arguments
///
/// * `model_path` - Optional path to the model file
/// * `ort_config` - Optional ORT session configuration
/// * `create_builder` - Factory function to create the builder
///
/// # Returns
///
/// * `Ok(Some(adapter))` if model_path is provided and build succeeds
/// * `Ok(None)` if model_path is None
/// * `Err(...)` if build fails
pub fn build_optional_adapter<B>(
    model_path: Option<&PathBuf>,
    ort_config: Option<&OrtSessionConfig>,
    create_builder: impl FnOnce() -> B,
) -> Result<Option<B::Adapter>, OCRError>
where
    B: AdapterBuilder + OrtConfigurable,
{
    let Some(model) = model_path else {
        return Ok(None);
    };

    let mut builder = create_builder();
    if let Some(config) = ort_config {
        builder = builder.with_ort_config(config.clone());
    }

    Ok(Some(builder.build(model)?))
}
