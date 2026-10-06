use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub(crate) struct GpuIdentity {
    pub(crate) name: String,
    pub(crate) uuid: String,
    pub(crate) driver: String,
}
#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct GpuMeasurement {
    pub(crate) device: GpuIdentity,
    pub(crate) baseline_bytes: u64,
    pub(crate) peak_bytes: u64,
    pub(crate) delta_bytes: u64,
    pub(crate) sampling_interval_ms: u64,
    pub(crate) samples: u64,
    pub(crate) other_processes: BTreeSet<u32>,
    pub(crate) co_tenant: bool,
    pub(crate) own_process_observed: bool,
    pub(crate) sampling_errors: Vec<String>,
    pub(crate) valid: bool,
}
impl GpuMeasurement {
    #[cfg(any(feature = "nvml", test))]
    fn record(&mut self, used: u64, processes: &[u32], pid: u32) {
        self.peak_bytes = self.peak_bytes.max(used);
        self.delta_bytes = self.peak_bytes.saturating_sub(self.baseline_bytes);
        self.samples += 1;
        for process in processes {
            if *process == pid {
                self.own_process_observed = true;
            } else {
                self.other_processes.insert(*process);
            }
        }
        self.co_tenant = !self.other_processes.is_empty();
    }
}

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

pub(crate) struct Monitor {
    #[cfg(feature = "nvml")]
    worker: Option<std::thread::JoinHandle<GpuMeasurement>>,
    #[cfg(feature = "nvml")]
    stop: Option<std::sync::mpsc::Sender<()>>,
    warning: Option<String>,
}
impl Monitor {
    pub(crate) fn start(device: &str, selector: Option<&str>, interval_ms: u64) -> Self {
        if !device.starts_with("cuda:") {
            return Self::unavailable(None);
        }
        #[cfg(feature = "nvml")]
        {
            match start_worker(selector, interval_ms) {
                Ok((worker, stop)) => Self {
                    worker: Some(worker),
                    stop: Some(stop),
                    warning: None,
                },
                Err(error) => {
                    Self::unavailable(Some(format!("NVML measurement unavailable: {error}")))
                }
            }
        }
        #[cfg(not(feature = "nvml"))]
        {
            let _ = (selector, interval_ms);
            Self::unavailable(Some(
                "GPU memory unavailable: compile with --features nvml".into(),
            ))
        }
    }
    fn unavailable(warning: Option<String>) -> Self {
        Self {
            #[cfg(feature = "nvml")]
            worker: None,
            #[cfg(feature = "nvml")]
            stop: None,
            warning,
        }
    }
    pub(crate) fn finish(mut self) -> (Option<GpuMeasurement>, Option<String>) {
        #[cfg(feature = "nvml")]
        {
            if let Some(stop) = self.stop.take() {
                let _ = stop.send(());
            }
            if let Some(worker) = self.worker.take() {
                return match worker.join() {
                    Ok(result) => (Some(result), self.warning.take()),
                    Err(_) => (None, Some("NVML sampling thread failed".into())),
                };
            }
        }
        (None, self.warning.take())
    }
}
impl Drop for Monitor {
    fn drop(&mut self) {
        #[cfg(feature = "nvml")]
        {
            if let Some(stop) = self.stop.take() {
                let _ = stop.send(());
            }
            if let Some(worker) = self.worker.take() {
                let _ = worker.join();
            }
        }
    }
}

#[cfg(feature = "nvml")]
fn start_worker(
    selector: Option<&str>,
    interval_ms: u64,
) -> anyhow::Result<(
    std::thread::JoinHandle<GpuMeasurement>,
    std::sync::mpsc::Sender<()>,
)> {
    use anyhow::Context;
    use nvml_wrapper::Nvml;
    use std::{sync::mpsc, time::Duration};
    let selector = selector
        .context("set nvml_device to a GPU UUID or index:N matching the CUDA device")?
        .to_string();
    let nvml = Nvml::init()?;
    let device = if let Some(index) = selector.strip_prefix("index:") {
        nvml.device_by_index(index.parse()?)?
    } else {
        nvml.device_by_uuid(selector.as_str())?
    };
    let identity = GpuIdentity {
        name: device.name()?,
        uuid: device.uuid()?,
        driver: nvml.sys_driver_version()?,
    };
    let baseline = device.memory_info()?.used;
    let pid = std::process::id();
    let processes = process_ids(&device);
    let mut measurement = GpuMeasurement {
        device: identity,
        baseline_bytes: baseline,
        peak_bytes: baseline,
        delta_bytes: 0,
        sampling_interval_ms: interval_ms,
        samples: 0,
        other_processes: BTreeSet::new(),
        co_tenant: false,
        own_process_observed: false,
        sampling_errors: Vec::new(),
        valid: false,
    };
    match processes {
        Ok(ids) => measurement.record(baseline, &ids, pid),
        Err(error) => {
            measurement.record(baseline, &[], pid);
            measurement.sampling_errors.push(error.to_string());
        }
    }
    let uuid = measurement.device.uuid.clone();
    let (stop, receive) = mpsc::channel();
    let worker = std::thread::spawn(move || {
        loop {
            let stopped = match receive.recv_timeout(Duration::from_millis(interval_ms)) {
                Ok(()) | Err(mpsc::RecvTimeoutError::Disconnected) => true,
                Err(mpsc::RecvTimeoutError::Timeout) => false,
            };
            let sample = || -> anyhow::Result<(u64, Vec<u32>)> {
                let device = nvml.device_by_uuid(uuid.as_str())?;
                Ok((device.memory_info()?.used, process_ids(&device)?))
            };
            match sample() {
                Ok((used, ids)) => measurement.record(used, &ids, pid),
                Err(error) => {
                    let message = error.to_string();
                    if !measurement.sampling_errors.contains(&message) {
                        measurement.sampling_errors.push(message);
                    }
                }
            }
            if stopped {
                break;
            }
        }
        measurement.valid = !measurement.co_tenant
            && measurement.own_process_observed
            && measurement.sampling_errors.is_empty();
        measurement
    });
    Ok((worker, stop))
}

#[cfg(feature = "nvml")]
fn process_ids(device: &nvml_wrapper::Device<'_>) -> anyhow::Result<Vec<u32>> {
    let mut ids: Vec<_> = device
        .running_compute_processes()?
        .into_iter()
        .map(|p| p.pid)
        .collect();
    ids.extend(
        device
            .running_graphics_processes()?
            .into_iter()
            .map(|p| p.pid),
    );
    ids.sort_unstable();
    ids.dedup();
    Ok(ids)
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn parses_linux_high_water_mark() {
        assert_eq!(parse_hwm("Name: bench\nVmHWM:\t123 kB\n"), Some(125952));
        assert_eq!(parse_hwm("VmRSS: 5 kB"), None);
    }
    #[test]
    fn peak_and_co_tenants_are_accumulated() {
        let mut m = GpuMeasurement {
            device: GpuIdentity {
                name: "mock".into(),
                uuid: "mock".into(),
                driver: "mock".into(),
            },
            baseline_bytes: 10,
            peak_bytes: 10,
            delta_bytes: 0,
            sampling_interval_ms: 10,
            samples: 0,
            other_processes: BTreeSet::new(),
            co_tenant: false,
            own_process_observed: false,
            sampling_errors: Vec::new(),
            valid: false,
        };
        m.record(15, &[7], 7);
        m.record(30, &[7, 9], 7);
        m.record(12, &[7], 7);
        assert_eq!(m.peak_bytes, 30);
        assert_eq!(m.delta_bytes, 20);
        assert!(m.co_tenant && m.own_process_observed);
        assert_eq!(m.other_processes, BTreeSet::from([9]));
    }
}
