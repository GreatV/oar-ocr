//! Process-wide ONNX Runtime thread-pool configuration.

use crate::core::errors::OCRError;

/// Options for sharing one ONNX Runtime thread pool across every session.
///
/// Multi-model OCR pipelines otherwise create a full intra-op pool for every
/// detector, recognizer, classifier, and layout model. Sharing a pool avoids
/// excess worker creation and cross-pool contention. The environment is
/// process-global, so [`commit`](Self::commit) must be called before any model
/// session is created.
#[derive(Debug, Clone, Default)]
pub struct OrtGlobalThreadPoolOptions {
    intra_threads: Option<usize>,
    inter_threads: Option<usize>,
    allow_spinning: Option<bool>,
}

impl OrtGlobalThreadPoolOptions {
    /// Creates options that retain ONNX Runtime's default thread counts.
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the process-wide intra-op thread count.
    pub fn with_intra_threads(mut self, threads: usize) -> Self {
        self.intra_threads = Some(threads);
        self
    }

    /// Sets the process-wide inter-op thread count.
    pub fn with_inter_threads(mut self, threads: usize) -> Self {
        self.inter_threads = Some(threads);
        self
    }

    /// Controls whether idle ONNX Runtime workers briefly spin for new work.
    pub fn with_spin_control(mut self, allow: bool) -> Self {
        self.allow_spinning = Some(allow);
        self
    }

    /// Commits the process-wide ONNX Runtime environment.
    ///
    /// Returns `false` when another caller already initialized the environment;
    /// in that case these options cannot take effect. Session-local thread counts
    /// should not be configured when a global pool is active.
    pub fn commit(self) -> Result<bool, OCRError> {
        let mut options = ort::environment::GlobalThreadPoolOptions::default();
        if let Some(threads) = self.intra_threads {
            options = options.with_intra_threads(threads).map_err(config_error)?;
        }
        if let Some(threads) = self.inter_threads {
            options = options.with_inter_threads(threads).map_err(config_error)?;
        }
        if let Some(allow) = self.allow_spinning {
            options = options.with_spin_control(allow).map_err(config_error)?;
        }

        Ok(ort::init().with_global_thread_pool(options).commit())
    }
}

fn config_error(error: ort::Error) -> OCRError {
    OCRError::ConfigError {
        message: format!("Failed to configure ONNX Runtime global thread pool: {error}"),
    }
}
