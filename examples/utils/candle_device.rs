//! Candle device configuration helper for VL examples.

/// Parses device string and creates a Candle Device for VL models.
///
/// # Supported formats
///
/// - `"cpu"` -> CPU device
/// - `"cuda"` or `"gpu"` -> CUDA device 0
/// - `"cuda:N"` -> CUDA device N
pub fn parse_candle_device(
    device_str: &str,
) -> Result<candle_core::Device, Box<dyn std::error::Error>> {
    let device_str = device_str.to_lowercase();
    match device_str.as_str() {
        "cpu" => Ok(candle_core::Device::Cpu),
        "cuda" | "gpu" => {
            #[cfg(feature = "cuda")]
            {
                Ok(candle_core::Device::new_cuda(0)?)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err("CUDA support not enabled. Compile with --features cuda".into())
            }
        }
        s if s.starts_with("cuda:") => {
            #[cfg(feature = "cuda")]
            {
                let ordinal: usize = s
                    .strip_prefix("cuda:")
                    .unwrap()
                    .parse()
                    .map_err(|_| "Invalid CUDA device ordinal")?;
                Ok(candle_core::Device::new_cuda(ordinal)?)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err("CUDA support not enabled. Compile with --features cuda".into())
            }
        }
        _ => Err(format!(
            "Unknown device: {}. Use 'cpu', 'cuda', or 'cuda:N'",
            device_str
        )
        .into()),
    }
}
