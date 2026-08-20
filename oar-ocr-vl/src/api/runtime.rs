//! Explicit model runtime selection.

use candle_core::{DType, Device};

/// Compute dtype selection for model loading.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DTypePolicy {
    /// Probe the device and select the best supported dtype.
    #[default]
    Auto,
    /// Require an explicit compute dtype.
    Fixed(DType),
}

/// Device and compute policy used while constructing a model.
#[derive(Debug, Clone)]
pub struct RuntimeConfig {
    pub device: Device,
    pub dtype: DTypePolicy,
}

impl RuntimeConfig {
    pub fn new(device: Device) -> Self {
        Self {
            device,
            dtype: DTypePolicy::Auto,
        }
    }

    pub fn with_dtype(mut self, dtype: DType) -> Self {
        self.dtype = DTypePolicy::Fixed(dtype);
        self
    }

    #[allow(dead_code)]
    pub(crate) fn resolve(self) -> (Device, DType) {
        let dtype = match self.dtype {
            DTypePolicy::Auto => crate::runtime::device::select_dtype(&self.device),
            DTypePolicy::Fixed(dtype) => dtype,
        };
        (self.device, dtype)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fixed_dtype_overrides_automatic_selection() {
        let (device, dtype) = RuntimeConfig::new(Device::Cpu)
            .with_dtype(DType::F16)
            .resolve();
        assert!(device.is_cpu());
        assert_eq!(dtype, DType::F16);
    }
}
