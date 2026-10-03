use super::{session, *};
use crate::core::config::ModelInferenceConfig;
use crate::core::inference::ModelSource;
use ort::logging::LogLevel;
use std::sync::Mutex;

impl OrtInfer {
    /// Creates a new OrtInfer instance with default ONNX Runtime settings and a single session.
    pub fn new(
        model_source: impl Into<ModelSource>,
        input_name: Option<&str>,
    ) -> Result<Self, OCRError> {
        let source = model_source.into();
        let session = session::load_session_with(
            source.clone(),
            |builder| Ok(builder.with_log_level(LogLevel::Error)?),
            Some("verify model path and compatibility with selected execution providers"),
        )?;
        let model_name = "unknown_model".to_string();

        Ok(OrtInfer {
            sessions: vec![Mutex::new(session)],
            next_idx: std::sync::atomic::AtomicUsize::new(0),
            input_name: input_name.unwrap_or("x").to_string(),
            model_path: source.display_path(),
            model_name,
            run_options: None,
        })
    }

    /// Creates a new OrtInfer instance from ModelInferenceConfig, applying ORT session
    /// configuration.
    pub fn from_config(
        common: &ModelInferenceConfig,
        model_source: impl Into<ModelSource>,
        input_name: Option<&str>,
    ) -> Result<Self, OCRError> {
        let source = model_source.into();

        // Workaround for a non-deterministic data race in ORT's CUDA EP that
        // corrupts arena buffers reused across `session.run()` calls.
        // Concretely, PP-FormulaNet's autoregressive Loop produces correct
        // tokens on the first run and pure garbage (max-trip-count) on every
        // subsequent run unless CUDA work is serialized at the driver level.
        Self::ensure_cuda_launch_blocking_if_needed(common);

        let session = session::load_session_with(
            source.clone(),
            |builder| {
                if let Some(cfg) = &common.ort_session {
                    Self::apply_ort_config(builder, cfg)
                } else {
                    Ok(builder.with_log_level(LogLevel::Error)?)
                }
            },
            Some("check device/EP configuration and model file"),
        )?;

        let model_name = common
            .model_name
            .clone()
            .unwrap_or_else(|| "unknown_model".to_string());
        let run_options = Self::arena_shrinkage_run_options(common, &session)?;

        Ok(OrtInfer {
            sessions: vec![Mutex::new(session)],
            next_idx: std::sync::atomic::AtomicUsize::new(0),
            input_name: input_name.unwrap_or("x").to_string(),
            model_path: source.display_path(),
            model_name,
            run_options,
        })
    }

    /// Builds the run options that return idle CUDA arena memory after each
    /// run, when [`OrtSessionConfig::arena_shrinkage`] is enabled and the
    /// session actually runs on a CUDA device.
    ///
    /// A failed CUDA provider registration falls back to CPU without an error,
    /// and ONNX Runtime rejects every run whose shrink list names a device the
    /// session has no arena for. So the device comes from the session's
    /// registered allocators, not just the requested configuration.
    ///
    /// [`OrtSessionConfig::arena_shrinkage`]: crate::core::config::OrtSessionConfig::arena_shrinkage
    fn arena_shrinkage_run_options(
        common: &ModelInferenceConfig,
        session: &ort::session::Session,
    ) -> Result<Option<ort::session::RunOptions>, OCRError> {
        let Some(device_id) = Self::arena_shrinkage_device(common) else {
            return Ok(None);
        };
        if !Self::session_has_cuda_allocator(session, device_id) {
            tracing::warn!(
                "CUDA arena shrinkage requested for gpu:{device_id}, but the session has no \
                 CUDA allocator there (CUDA provider not registered); continuing without it"
            );
            return Ok(None);
        }
        let device = format!("gpu:{device_id}");
        let build = || -> ort::Result<ort::session::RunOptions> {
            let mut options = ort::session::RunOptions::new()?;
            options.set("memory.enable_memory_arena_shrinkage", &device)?;
            Ok(options)
        };
        build().map(Some).map_err(|e| OCRError::ConfigError {
            message: format!("failed to enable CUDA arena shrinkage on {device}: {e}"),
        })
    }

    /// The CUDA device whose arena to shrink (the first CUDA execution
    /// provider's), or `None` when shrinkage is off or no CUDA provider is
    /// configured.
    pub(super) fn arena_shrinkage_device(common: &ModelInferenceConfig) -> Option<i32> {
        let cfg = common.ort_session.as_ref()?;
        if cfg.arena_shrinkage != Some(true) {
            return None;
        }
        cfg.execution_providers
            .as_ref()?
            .iter()
            .find_map(|ep| match ep {
                // Without the `cuda` feature no CUDA provider is registered, so
                // there is no CUDA arena to shrink.
                #[cfg(feature = "cuda")]
                crate::core::config::OrtExecutionProvider::CUDA { device_id, .. } => {
                    Some(device_id.unwrap_or(0))
                }
                _ => None,
            })
    }

    /// Whether the session holds an allocator for CUDA device `device_id`,
    /// i.e. the CUDA execution provider registered for that device.
    fn session_has_cuda_allocator(session: &ort::session::Session, device_id: i32) -> bool {
        use ort::memory::{AllocationDevice, Allocator, AllocatorType, MemoryInfo, MemoryType};
        MemoryInfo::new(
            AllocationDevice::CUDA,
            device_id,
            AllocatorType::Device,
            MemoryType::Default,
        )
        .and_then(|info| Allocator::new(session, info))
        .is_ok()
    }

    fn ensure_cuda_launch_blocking_if_needed(common: &ModelInferenceConfig) {
        use crate::core::config::OrtExecutionProvider;
        let model_name = common.model_name.as_deref().unwrap_or_default();
        let needs_formula_workaround = model_name.to_ascii_lowercase().contains("formulanet");
        if !needs_formula_workaround {
            return;
        }

        let wants_cuda = common
            .ort_session
            .as_ref()
            .and_then(|c| c.execution_providers.as_ref())
            .is_some_and(|eps| {
                eps.iter().any(|ep| {
                    matches!(
                        ep,
                        OrtExecutionProvider::CUDA { .. } | OrtExecutionProvider::TensorRT { .. }
                    )
                })
            });
        if !wants_cuda {
            return;
        }
        ensure_cuda_launch_blocking();
    }
}

/// Idempotently sets `CUDA_LAUNCH_BLOCKING=1`, respecting any value already
/// present in the environment.
///
/// MUST be called before the first CUDA session in the process is created: the
/// CUDA runtime reads this variable once, at context initialization, so setting
/// it after another CUDA session already exists has no effect. Pipelines that
/// build several CUDA models should therefore call this up front (before
/// building any adapter) rather than relying on a per-model trigger.
///
/// Works around onnxruntime#4829: PP-FormulaNet's autoregressive `Loop`
/// corrupts CUDA-EP arena buffers reused across `session.run()` calls when its
/// runs interleave with other models', producing garbage tokens. Serializing
/// CUDA launches avoids the race.
pub fn ensure_cuda_launch_blocking() {
    static SET_ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
    SET_ONCE.get_or_init(|| {
        if std::env::var_os("CUDA_LAUNCH_BLOCKING").is_none() {
            // SAFETY: set_var is not thread-safe in general, but OnceLock
            // serializes us, and callers must invoke this before any CUDA work.
            unsafe { std::env::set_var("CUDA_LAUNCH_BLOCKING", "1") };
            tracing::info!(
                "set CUDA_LAUNCH_BLOCKING=1 to work around onnxruntime#4829 (PP-FormulaNet CUDA Loop race)"
            );
        }
    });
}
