//! Device parsing, dtype selection, and memory introspection.

use crate::api::error::Error;
use candle_core::{DType, Device, Tensor};

#[cfg(not(feature = "cuda"))]
fn cuda_not_enabled() -> Error {
    Error::config("CUDA support not enabled. Compile with --features cuda")
}

#[cfg(not(all(feature = "metal", target_os = "macos")))]
fn metal_not_enabled() -> Error {
    Error::config("Metal support not enabled. Compile on macOS with --features metal")
}

#[cfg(any(feature = "cuda", all(feature = "metal", target_os = "macos")))]
fn parse_with_ordinal(
    value: &str,
    prefix: &str,
    device_name: &str,
    creator: impl Fn(usize) -> candle_core::Result<Device>,
) -> Result<Device, Error> {
    let ordinal = value
        .strip_prefix(prefix)
        .ok_or_else(|| Error::config(format!("invalid {device_name} device string {value:?}")))?
        .parse::<usize>()
        .map_err(|_| Error::config(format!("invalid {device_name} ordinal in {value:?}")))?;
    creator(ordinal).map_err(|error| {
        Error::config(format!("failed to create {device_name} {ordinal}: {error}"))
    })
}

pub fn parse_device(device: &str) -> Result<Device, Error> {
    let device = device.to_lowercase();
    match device.as_str() {
        "cpu" => Ok(Device::Cpu),
        "cuda" | "gpu" => {
            #[cfg(feature = "cuda")]
            {
                Device::new_cuda(0).map_err(|error| {
                    Error::config(format!("failed to create CUDA device: {error}"))
                })
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(cuda_not_enabled())
            }
        }
        "metal" => {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                Device::new_metal(0).map_err(|error| {
                    Error::config(format!("failed to create Metal device: {error}"))
                })
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            {
                Err(metal_not_enabled())
            }
        }
        value if value.starts_with("cuda:") => {
            #[cfg(feature = "cuda")]
            {
                parse_with_ordinal(value, "cuda:", "CUDA", Device::new_cuda)
            }
            #[cfg(not(feature = "cuda"))]
            {
                Err(cuda_not_enabled())
            }
        }
        value if value.starts_with("metal:") => {
            #[cfg(all(feature = "metal", target_os = "macos"))]
            {
                parse_with_ordinal(value, "metal:", "Metal", Device::new_metal)
            }
            #[cfg(not(all(feature = "metal", target_os = "macos")))]
            {
                Err(metal_not_enabled())
            }
        }
        _ => Err(Error::config(format!(
            "unknown device {device:?}; use cpu, cuda, cuda:N, metal, or metal:N"
        ))),
    }
}

pub fn select_dtype(device: &Device) -> DType {
    if let Ok(value) = std::env::var("OAR_VL_DTYPE") {
        if let Some(dtype) = dtype_from_str(&value) {
            tracing::info!("using dtype {dtype:?} from OAR_VL_DTYPE={value}");
            return dtype;
        }
        tracing::warn!(
            "ignoring invalid OAR_VL_DTYPE value {value:?} (expected bf16, f16, or f32)"
        );
    }
    if !device.supports_bf16() {
        return DType::F32;
    }
    if bf16_works(device) {
        DType::BF16
    } else {
        tracing::warn!("BF16 probe failed; falling back to F16");
        DType::F16
    }
}

pub(crate) fn dtype_from_str(value: &str) -> Option<DType> {
    match value.to_lowercase().as_str() {
        "bf16" | "bfloat16" => Some(DType::BF16),
        "f16" | "fp16" | "float16" | "half" => Some(DType::F16),
        "f32" | "fp32" | "float32" => Some(DType::F32),
        _ => None,
    }
}

pub(crate) fn bf16_works(device: &Device) -> bool {
    let probe = || -> candle_core::Result<()> {
        let tensor = Tensor::ones((2, 2), DType::BF16, device)?;
        let tensor = tensor.matmul(&tensor)?;
        let tensor = (tensor * 2.0)?;
        tensor.to_dtype(DType::F32)?.sum_all()?.to_scalar::<f32>()?;
        Ok(())
    };
    match probe() {
        Ok(()) => true,
        Err(error) => {
            tracing::debug!("BF16 probe failed on {device:?}: {error}");
            false
        }
    }
}

pub fn free_device_memory(device: &Device) -> Option<usize> {
    #[cfg(feature = "cuda")]
    if let Device::Cuda(device) = device {
        return match device.cuda_stream().context().mem_get_info() {
            Ok((free, _)) => Some(free),
            Err(error) => {
                tracing::debug!("querying free CUDA memory failed: {error}");
                None
            }
        };
    }
    #[cfg(all(feature = "metal", target_os = "macos"))]
    if let Device::Metal(device) = device {
        let raw = device.metal_device();
        let budget = raw.recommended_max_working_set_size();
        let allocated = raw.current_allocated_size();
        return (budget > 0).then(|| budget.saturating_sub(allocated));
    }
    let _ = device;
    None
}
