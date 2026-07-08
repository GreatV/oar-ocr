//! Shared utilities for builder patterns in oarocr module.

use oar_ocr_core::core::config::OrtSessionConfig;
use oar_ocr_core::core::traits::OrtConfigurable;
use oar_ocr_core::core::traits::adapter::AdapterBuilder;
use oar_ocr_core::core::{ModelSource, OCRError};
use std::path::{Path, PathBuf};

/// Resolves a model path through the auto-download cache when the
/// `auto-download` feature is enabled.
///
/// Behaviour:
///
/// - With `auto-download` on, delegates to
///   [`oar_ocr_core::core::download::resolve_path`]. Bare file names that
///   match the registry are fetched from ModelScope and verified against
///   the expected SHA-256; on-disk paths are returned unchanged.
/// - Without the feature, returns the input verbatim. The caller's normal
///   error path produces the usual "model not found" message.
pub fn resolve_model_path(path: &Path) -> Result<PathBuf, OCRError> {
    #[cfg(feature = "auto-download")]
    {
        oar_ocr_core::core::download::resolve_path(path)
    }
    #[cfg(not(feature = "auto-download"))]
    {
        Ok(path.to_path_buf())
    }
}

/// Resolves `Path` sources through the auto-download cache; in-memory
/// sources pass through untouched.
pub fn resolve_model_source(source: &ModelSource) -> Result<ModelSource, OCRError> {
    match source {
        ModelSource::Path(path) => Ok(ModelSource::Path(resolve_model_path(path)?)),
        memory @ ModelSource::Memory(_) => Ok(memory.clone()),
    }
}

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
    model_source: Option<&ModelSource>,
    ort_config: Option<&OrtSessionConfig>,
    create_builder: impl FnOnce() -> B,
) -> Result<Option<B::Adapter>, OCRError>
where
    B: AdapterBuilder + OrtConfigurable,
{
    let Some(source) = model_source else {
        return Ok(None);
    };
    let resolved = resolve_model_source(source)?;

    let mut builder = create_builder();
    if let Some(config) = ort_config {
        builder = builder.with_ort_config(config.clone());
    }

    Ok(Some(builder.build(resolved)?))
}
