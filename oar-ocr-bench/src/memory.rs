use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GpuMemory {
    pub(crate) name: String,
    pub(crate) baseline_bytes: u64,
    pub(crate) peak_bytes: u64,
}

impl GpuMemory {
    pub(crate) fn delta_bytes(&self) -> u64 {
        self.peak_bytes.saturating_sub(self.baseline_bytes)
    }
}

/// Peak resident memory of this process (Linux `VmHWM`).
pub(crate) fn host_peak_bytes() -> Option<u64> {
    #[cfg(target_os = "linux")]
    {
        parse_hwm(&std::fs::read_to_string("/proc/self/status").ok()?)
    }
    #[cfg(not(target_os = "linux"))]
    {
        None
    }
}

#[cfg(any(target_os = "linux", test))]
fn parse_hwm(status: &str) -> Option<u64> {
    let line = status.lines().find(|line| line.starts_with("VmHWM:"))?;
    line.split_whitespace()
        .nth(1)?
        .parse::<u64>()
        .ok()?
        .checked_mul(1024)
}

/// Samples device-wide used memory of one GPU until stopped.
///
/// The result is a sampled lower bound of the peak, and it includes any other
/// process on the GPU, so measurements need an otherwise idle device.
pub(crate) struct GpuSampler {
    #[cfg(feature = "nvml")]
    worker: Option<(
        std::thread::JoinHandle<Option<GpuMemory>>,
        std::sync::mpsc::Sender<()>,
    )>,
}

impl GpuSampler {
    pub(crate) fn start(device: &str) -> Self {
        #[cfg(feature = "nvml")]
        {
            let worker = device
                .strip_prefix("cuda:")
                .and_then(|index| index.parse().ok())
                .and_then(|index| spawn(index).ok());
            Self { worker }
        }
        #[cfg(not(feature = "nvml"))]
        {
            let _ = device;
            Self {}
        }
    }

    pub(crate) fn finish(self) -> Option<GpuMemory> {
        #[cfg(feature = "nvml")]
        {
            let (worker, stop) = self.worker?;
            let _ = stop.send(());
            worker.join().ok().flatten()
        }
        #[cfg(not(feature = "nvml"))]
        {
            None
        }
    }
}

#[cfg(feature = "nvml")]
fn spawn(
    index: u32,
) -> anyhow::Result<(
    std::thread::JoinHandle<Option<GpuMemory>>,
    std::sync::mpsc::Sender<()>,
)> {
    use anyhow::Context;
    use std::{sync::mpsc, time::Duration};
    let nvml = nvml_wrapper::Nvml::init()?;
    // A CUDA ordinal indexes the visible devices, which NVML does not see.
    let uuid = match std::env::var("CUDA_VISIBLE_DEVICES") {
        Ok(visible) => {
            let entry = visible
                .split(',')
                .nth(index as usize)
                .context("CUDA ordinal is not visible")?
                .trim()
                .to_string();
            if entry.starts_with("GPU-") || entry.starts_with("MIG-") {
                entry
            } else {
                nvml.device_by_index(entry.parse()?)?.uuid()?
            }
        }
        Err(_) => nvml.device_by_index(index)?.uuid()?,
    };
    let (baseline, name) = {
        let device = nvml.device_by_uuid(uuid.as_str())?;
        (device.memory_info()?.used, device.name()?)
    };
    let (stop, stopped) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        let mut memory = GpuMemory {
            name,
            baseline_bytes: baseline,
            peak_bytes: baseline,
        };
        loop {
            let done = !matches!(
                stopped.recv_timeout(Duration::from_millis(10)),
                Err(mpsc::RecvTimeoutError::Timeout)
            );
            let used = nvml
                .device_by_uuid(uuid.as_str())
                .ok()?
                .memory_info()
                .ok()?
                .used;
            memory.peak_bytes = memory.peak_bytes.max(used);
            if done {
                return Some(memory);
            }
        }
    });
    Ok((worker, stop))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_linux_high_water_mark() {
        assert_eq!(parse_hwm("Name: bench\nVmHWM:\t123 kB\n"), Some(125952));
        assert_eq!(parse_hwm("VmRSS: 5 kB"), None);
    }
}
