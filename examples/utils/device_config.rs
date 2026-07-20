//! Device configuration helper for examples.
//!
//! This module provides utilities for parsing device strings and creating
//! ONNX Runtime session configurations with appropriate execution providers.

#[cfg(any(feature = "cuda", feature = "coreml"))]
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

    Err(format!(
        "Unsupported device: {}. Supported devices: cpu{}{}",
        device,
        if cfg!(feature = "cuda") {
            ", cuda, cuda:N"
        } else {
            ""
        },
        if cfg!(feature = "coreml") {
            ", coreml[:gpu|ane|cpu], coreml-nn[:gpu|ane|cpu], coreml-static[:gpu|ane|cpu]"
        } else {
            ""
        }
    )
    .into())
}
