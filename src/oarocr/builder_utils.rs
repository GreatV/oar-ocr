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

/// Resolves high-level pipeline batch sizes from the configured execution device.
///
/// ONNX Runtime CPU inference often loses throughput when large image tensors are
/// batched because each operator already uses the CPU thread pool. Accelerators
/// retain the adapter-specific defaults; explicit user values always win.
pub(crate) fn resolve_device_batch_sizes(
    ort_config: Option<&OrtSessionConfig>,
    image_batch_size: Option<usize>,
    region_batch_size: Option<usize>,
    cpu_image_batch_size: usize,
    cpu_region_batch_size: usize,
) -> (Option<usize>, Option<usize>) {
    let uses_accelerator = ort_config.is_some_and(OrtSessionConfig::has_accelerator_provider);
    if uses_accelerator {
        (image_batch_size, region_batch_size)
    } else {
        (
            image_batch_size.or(Some(cpu_image_batch_size)),
            region_batch_size.or(Some(cpu_region_batch_size)),
        )
    }
}

/// Selects the conservative CPU recognition batch for the configured model family.
///
/// PP-OCRv6 Tiny has substantially less per-item compute than the other bundled
/// recognizers and benefits from a wider outer batch on Windows CPUs. Larger
/// Small/Medium/Mobile models already occupy the ORT intra-op pool and regress
/// when the outer batch grows beyond four. Unknown and in-memory models therefore
/// stay on the conservative default; callers can always override it explicitly.
pub(crate) fn default_cpu_region_batch_size(
    model_source: Option<&ModelSource>,
    model_name: Option<&str>,
) -> usize {
    let source_name = model_source
        .and_then(ModelSource::as_path)
        .and_then(Path::file_stem)
        .and_then(|name| name.to_str());
    let is_tiny = model_name
        .into_iter()
        .chain(source_name)
        .any(|name| name.to_ascii_lowercase().contains("tiny"));

    if is_tiny { 16 } else { 4 }
}

#[cfg(test)]
mod tests {
    use super::*;
    use oar_ocr_core::core::config::OrtExecutionProvider;

    #[test]
    fn cpu_batch_defaults_are_applied_without_overriding_user_values() {
        assert_eq!(
            resolve_device_batch_sizes(None, None, None, 1, 4),
            (Some(1), Some(4))
        );
        assert_eq!(
            resolve_device_batch_sizes(None, Some(3), Some(7), 1, 4),
            (Some(3), Some(7))
        );
    }

    #[test]
    fn accelerator_keeps_adapter_batch_defaults() {
        let config = OrtSessionConfig::new().with_execution_providers(vec![
            OrtExecutionProvider::CUDA {
                device_id: Some(0),
                gpu_mem_limit: None,
                arena_extend_strategy: None,
                cudnn_conv_algo_search: None,
                cudnn_conv_use_max_workspace: None,
            },
            OrtExecutionProvider::CPU,
        ]);
        assert_eq!(
            resolve_device_batch_sizes(Some(&config), None, None, 1, 4),
            (None, None)
        );
    }

    #[test]
    fn cpu_recognition_batch_is_model_family_aware() {
        let tiny = ModelSource::from("models/pp-ocrv6_tiny_rec.onnx");
        let small = ModelSource::from("models/pp-ocrv6_small_rec.onnx");
        let memory = ModelSource::from(vec![0u8; 8]);

        assert_eq!(default_cpu_region_batch_size(Some(&tiny), None), 16);
        assert_eq!(default_cpu_region_batch_size(Some(&small), None), 4);
        assert_eq!(default_cpu_region_batch_size(Some(&memory), None), 4);
        assert_eq!(
            default_cpu_region_batch_size(Some(&small), Some("PP-OCRv6_tiny_rec")),
            16
        );
    }
}
