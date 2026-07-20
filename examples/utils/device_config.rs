//! Device configuration helper for examples.
//!
//! This module provides utilities for parsing device strings and creating
//! ONNX Runtime session configurations with appropriate execution providers.

#[cfg(any(feature = "cuda", feature = "directml", feature = "coreml"))]
use oar_ocr::core::config::OrtExecutionProvider;
use oar_ocr::core::config::OrtSessionConfig;
#[cfg(feature = "coreml")]
use oar_ocr::core::config::{
    OrtCoreMLComputeUnits, OrtCoreMLConfig, OrtCoreMLModelFormat, OrtCoreMLSpecializationStrategy,
};

/// Applies optional ONNX Runtime thread and execution-mode overrides.
///
/// A default CPU configuration is materialized only when an override is
/// present, preserving `None` for the common unconfigured CPU path.
#[allow(dead_code)] // This shared module is compiled by examples without tuning flags.
pub fn apply_ort_overrides(
    mut config: Option<OrtSessionConfig>,
    intra_threads: Option<usize>,
    inter_threads: Option<usize>,
    parallel_execution: bool,
) -> Result<Option<OrtSessionConfig>, Box<dyn std::error::Error>> {
    if intra_threads == Some(0) || inter_threads == Some(0) {
        return Err("ONNX Runtime thread counts must be at least one".into());
    }
    if intra_threads.is_none() && inter_threads.is_none() && !parallel_execution {
        return Ok(config);
    }

    let mut configured = config.take().unwrap_or_default();
    if let Some(threads) = intra_threads {
        configured = configured.with_intra_threads(threads);
    }
    if let Some(threads) = inter_threads {
        configured = configured.with_inter_threads(threads);
    }
    if parallel_execution {
        configured = configured.with_parallel_execution(true);
    }
    Ok(Some(configured))
}

/// Parses device string and creates OrtSessionConfig with appropriate execution providers.
///
/// # Supported formats
///
/// - `"cpu"` -> CPU execution provider (returns None as CPU is default)
/// - `"cuda"` or `"cuda:0"` -> CUDA execution provider with device ID
/// - `"directml"`, `"directml:0"`, or `"dml:0"` -> DirectML execution provider
/// - `"coreml"`, `"coreml:gpu"`, or `"coreml:ane"` -> CoreML MLProgram provider
/// - `"coreml-nn"` variants -> legacy CoreML NeuralNetwork representation
/// - `"coreml-static"` variants -> only offload models with static input shapes
///
/// # Arguments
///
/// * `device` - Device string to parse
///
/// # Returns
///
/// * `Ok(None)` - For CPU device (no special config needed)
/// * `Ok(Some(config))` - For CUDA device with appropriate configuration
/// * `Err(...)` - If device string is invalid or unsupported
///
/// # Examples
///
/// ```no_run
/// use oar_ocr::core::config::OrtSessionConfig;
///
/// // CPU device (default)
/// let config = parse_device_config("cpu")?;
/// assert!(config.is_none());
///
/// // CUDA device 0
/// let config = parse_device_config("cuda:0")?;
/// assert!(config.is_some());
/// ```
pub fn parse_device_config(
    device: &str,
) -> Result<Option<OrtSessionConfig>, Box<dyn std::error::Error>> {
    let device_lower = device.to_lowercase();

    if device_lower == "cpu" {
        // CPU is the default, no need for special config
        return Ok(None);
    }

    #[cfg(feature = "coreml")]
    if device_lower == "coreml"
        || device_lower.starts_with("coreml:")
        || device_lower == "coreml-nn"
        || device_lower.starts_with("coreml-nn:")
        || device_lower == "coreml-static"
        || device_lower.starts_with("coreml-static:")
    {
        let (kind, units) = device_lower
            .split_once(':')
            .unwrap_or((&device_lower, "all"));
        let compute_units = match units {
            "gpu" => OrtCoreMLComputeUnits::CPUAndGPU,
            "ane" => OrtCoreMLComputeUnits::CPUAndNeuralEngine,
            "cpu" => OrtCoreMLComputeUnits::CPUOnly,
            "all" => OrtCoreMLComputeUnits::All,
            _ => {
                return Err(format!(
                    "Invalid CoreML compute units: '{}'. Expected all, gpu, ane, or cpu",
                    units
                )
                .into());
            }
        };
        let model_format = match kind {
            "coreml-nn" => OrtCoreMLModelFormat::NeuralNetwork,
            "coreml" | "coreml-static" => OrtCoreMLModelFormat::MLProgram,
            _ => unreachable!("CoreML kind was validated above"),
        };
        let provider = OrtExecutionProvider::CoreML {
            ane_only: None,
            subgraphs: Some(false),
        };
        let coreml_config = OrtCoreMLConfig {
            compute_units: Some(compute_units),
            model_format: Some(model_format),
            static_input_shapes: Some(kind == "coreml-static"),
            specialization_strategy: Some(OrtCoreMLSpecializationStrategy::FastPrediction),
            allow_low_precision_accumulation_on_gpu: Some(true),
            profile_compute_plan: None,
            model_cache_dir: None,
        };
        return Ok(Some(
            OrtSessionConfig::new()
                .with_execution_providers(vec![provider])
                .with_coreml_config(coreml_config),
        ));
    }

    #[cfg(not(feature = "coreml"))]
    if device_lower.starts_with("coreml") {
        return Err(format!(
            "CoreML device '{}' requested but the coreml feature is not enabled. \
             Rebuild with --features=coreml to enable CoreML support.",
            device
        )
        .into());
    }

    #[cfg(feature = "cuda")]
    {
        if device_lower.starts_with("cuda") {
            let device_id = if device_lower == "cuda" {
                0
            } else if let Some(id_str) = device_lower.strip_prefix("cuda:") {
                id_str.parse::<i32>().map_err(|_| {
                    format!(
                        "Invalid CUDA device ID: {}. Expected format: 'cuda' or 'cuda:N'",
                        device
                    )
                })?
            } else {
                return Err(format!(
                    "Invalid device format: {}. Expected 'cuda' or 'cuda:N'",
                    device
                )
                .into());
            };

            let config = OrtSessionConfig::new().with_execution_providers(vec![
                OrtExecutionProvider::CUDA {
                    device_id: Some(device_id),
                    gpu_mem_limit: None,
                    arena_extend_strategy: None,
                    cudnn_conv_algo_search: None,
                    cudnn_conv_use_max_workspace: None,
                },
                OrtExecutionProvider::CPU, // Fallback to CPU
            ]);

            return Ok(Some(config));
        }
    }

    #[cfg(not(feature = "cuda"))]
    {
        if device_lower.starts_with("cuda") {
            return Err(format!(
                "CUDA device '{}' requested but CUDA feature is not enabled. \
                 Rebuild with --features=cuda to enable CUDA support.",
                device
            )
            .into());
        }
    }

    #[cfg(feature = "directml")]
    {
        if device_lower == "directml"
            || device_lower == "dml"
            || device_lower.starts_with("directml:")
            || device_lower.starts_with("dml:")
        {
            let id_str = device_lower
                .strip_prefix("directml:")
                .or_else(|| device_lower.strip_prefix("dml:"));
            let device_id = id_str
                .map(|id| {
                    id.parse::<i32>().map_err(|_| {
                        format!(
                            "Invalid DirectML device ID: {device}. Expected 'directml', 'directml:N', or 'dml:N'"
                        )
                    })
                })
                .transpose()?
                .unwrap_or(0);

            // The DirectML execution provider requires memory pattern
            // optimization to be disabled; leaving it at ONNX Runtime's
            // default (enabled) can fail session creation before any
            // inference runs.
            return Ok(Some(
                OrtSessionConfig::new()
                    .with_execution_providers(vec![
                        OrtExecutionProvider::DirectML {
                            device_id: Some(device_id),
                        },
                        OrtExecutionProvider::CPU,
                    ])
                    .with_memory_pattern(false),
            ));
        }
    }

    #[cfg(not(feature = "directml"))]
    {
        if device_lower == "directml"
            || device_lower == "dml"
            || device_lower.starts_with("directml:")
            || device_lower.starts_with("dml:")
        {
            return Err(format!(
                "DirectML device '{device}' requested but the DirectML feature is not enabled. \
                 Rebuild with --features=directml to enable DirectML support."
            )
            .into());
        }
    }

    let mut supported = vec!["cpu"];
    if cfg!(feature = "cuda") {
        supported.push("cuda");
        supported.push("cuda:N");
    }
    if cfg!(feature = "directml") {
        supported.push("directml");
        supported.push("directml:N");
    }
    if cfg!(feature = "coreml") {
        supported.push("coreml[:gpu|ane|cpu]");
        supported.push("coreml-nn[:gpu|ane|cpu]");
        supported.push("coreml-static[:gpu|ane|cpu]");
    }
    Err(format!(
        "Unsupported device: {device}. Supported devices: {}",
        supported.join(", ")
    )
    .into())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cpu_needs_no_explicit_session_config() {
        assert!(parse_device_config("cpu").unwrap().is_none());
    }

    #[cfg(feature = "directml")]
    #[test]
    fn parses_directml_alias_and_device_id() {
        for name in ["directml:2", "dml:2"] {
            let config = parse_device_config(name).unwrap().unwrap();
            assert_eq!(
                config.execution_providers,
                Some(vec![
                    OrtExecutionProvider::DirectML { device_id: Some(2) },
                    OrtExecutionProvider::CPU,
                ])
            );
            assert_eq!(
                config.enable_mem_pattern,
                Some(false),
                "DirectML requires memory pattern optimization to be disabled"
            );
        }
    }

    #[cfg(not(feature = "directml"))]
    #[test]
    fn directml_request_explains_required_feature() {
        let error = parse_device_config("directml").unwrap_err().to_string();
        assert!(error.contains("--features=directml"));
    }
}
