use super::{session, *};
use crate::core::config::ModelInferenceConfig;
use ort::logging::LogLevel;
use std::path::Path;
use std::sync::Mutex;

impl OrtInfer {
    /// Creates a new OrtInfer instance with default ONNX Runtime settings and a single session.
    pub fn new(model_path: impl AsRef<Path>, input_name: Option<&str>) -> Result<Self, OCRError> {
        let path = model_path.as_ref();
        let session = session::load_session_with(
            path,
            |builder| Ok(builder.with_log_level(LogLevel::Error)?),
            Some("verify model path and compatibility with selected execution providers"),
        )?;
        let model_name = "unknown_model".to_string();

        Ok(OrtInfer {
            sessions: vec![Mutex::new(session)],
            next_idx: std::sync::atomic::AtomicUsize::new(0),
            input_name: input_name.unwrap_or("x").to_string(),
            model_path: path.to_path_buf(),
            model_name,
        })
    }

    /// Creates a new OrtInfer instance from ModelInferenceConfig, applying ORT session
    /// configuration.
    pub fn from_config(
        common: &ModelInferenceConfig,
        model_path: impl AsRef<Path>,
        input_name: Option<&str>,
    ) -> Result<Self, OCRError> {
        let path = model_path.as_ref();

        // Workaround for a non-deterministic data race in ORT's CUDA EP that
        // corrupts arena buffers reused across `session.run()` calls.
        // Concretely, PP-FormulaNet's autoregressive Loop produces correct
        // tokens on the first run and pure garbage (max-trip-count) on every
        // subsequent run unless CUDA work is serialized at the driver level.
        Self::ensure_cuda_launch_blocking_if_needed(common);

        let session = session::load_session_with(
            path,
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

        Ok(OrtInfer {
            sessions: vec![Mutex::new(session)],
            next_idx: std::sync::atomic::AtomicUsize::new(0),
            input_name: input_name.unwrap_or("x").to_string(),
            model_path: path.to_path_buf(),
            model_name,
        })
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
        static SET_ONCE: std::sync::OnceLock<()> = std::sync::OnceLock::new();
        SET_ONCE.get_or_init(|| {
            if std::env::var_os("CUDA_LAUNCH_BLOCKING").is_none() {
                // SAFETY: set_var is not thread-safe in general, but OnceLock
                // serializes us, and we only run before any CUDA work in the
                // process (caller is about to build their first CUDA session).
                unsafe { std::env::set_var("CUDA_LAUNCH_BLOCKING", "1") };
                tracing::info!(
                    "ort_infer: CUDA EP detected; set CUDA_LAUNCH_BLOCKING=1 to work around onnxruntime#4829"
                );
            }
        });
    }
}
