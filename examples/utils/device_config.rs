//! Device configuration helper for examples.
//!
//! This module provides utilities for parsing device strings and creating
//! ONNX Runtime session configurations with appropriate execution providers.

#[cfg(feature = "cuda")]
use oar_ocr::core::config::OrtExecutionProvider;
use oar_ocr::core::config::OrtSessionConfig;

/// Parses device string and creates OrtSessionConfig with appropriate execution providers.
///
/// # Supported formats
///
/// - `"cpu"` -> CPU execution provider (returns None as CPU is default)
/// - `"cuda"` or `"cuda:0"` -> CUDA execution provider with device ID
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
        "Unsupported device: {}. Supported devices: cpu{}",
        device,
        if cfg!(feature = "cuda") {
            ", cuda, cuda:N"
        } else {
            ""
        }
    )
    .into())
}
