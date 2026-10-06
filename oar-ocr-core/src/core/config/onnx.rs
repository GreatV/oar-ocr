//! ONNX Runtime configuration types and utilities.

use serde::{Deserialize, Serialize};

/// Graph optimization levels for ONNX Runtime.
///
/// This enum represents the different levels of graph optimization that can be applied
/// during ONNX Runtime session creation.
#[derive(Debug, Clone, Copy, Serialize, Deserialize, Default)]
pub enum OrtGraphOptimizationLevel {
    /// Disable all optimizations.
    DisableAll,
    /// Enable basic optimizations.
    #[default]
    Level1,
    /// Enable extended optimizations.
    Level2,
    /// Enable all optimizations.
    Level3,
    /// Enable all optimizations (alias for Level3).
    All,
}

/// CoreML hardware selection used by the ONNX Runtime CoreML execution provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OrtCoreMLComputeUnits {
    /// Let CoreML select from CPU, GPU, and Neural Engine.
    #[default]
    All,
    /// Restrict CoreML to CPU and GPU.
    CPUAndGPU,
    /// Restrict CoreML to CPU and Neural Engine.
    CPUAndNeuralEngine,
    /// Restrict CoreML to CPU.
    CPUOnly,
}

/// CoreML model representation created by ONNX Runtime.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OrtCoreMLModelFormat {
    /// The modern CoreML representation (macOS 12+), with broader operator support.
    #[default]
    MLProgram,
    /// The legacy CoreML neural-network representation.
    NeuralNetwork,
}

/// CoreML graph-specialization policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum OrtCoreMLSpecializationStrategy {
    /// CoreML's balanced default.
    #[default]
    Default,
    /// Prefer steady-state prediction latency over specialization time and size.
    FastPrediction,
}

/// Advanced CoreML execution-provider options.
///
/// These options live on [`OrtSessionConfig`] instead of adding fields to
/// [`OrtExecutionProvider::CoreML`], preserving source compatibility for code
/// that constructs or exhaustively matches the provider variant.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Default)]
pub struct OrtCoreMLConfig {
    /// Hardware units available to CoreML.
    pub compute_units: Option<OrtCoreMLComputeUnits>,
    /// CoreML model representation.
    pub model_format: Option<OrtCoreMLModelFormat>,
    /// Only claim nodes whose model inputs have static shapes.
    pub static_input_shapes: Option<bool>,
    /// CoreML graph-specialization policy.
    pub specialization_strategy: Option<OrtCoreMLSpecializationStrategy>,
    /// Permit FP16 accumulation on the GPU.
    pub allow_low_precision_accumulation_on_gpu: Option<bool>,
    /// Log CoreML's hardware assignment and estimated cost.
    pub profile_compute_plan: Option<bool>,
    /// Directory used to cache compiled CoreML models.
    pub model_cache_dir: Option<String>,
}

pub(crate) const COREML_CONFIG_ENTRY: &str = "oar.internal.coreml_config";
pub(crate) const AUTO_DEVICE_CONFIG_ENTRY: &str = "oar.internal.auto_device";

/// Identifies automatic candidates by provider kind and device ID, ignoring
/// tuning fields that model builders may adjust before resolution.
fn auto_signature(providers: &[OrtExecutionProvider]) -> String {
    let device = |id: &Option<i32>| id.map_or_else(String::new, |id| format!(":{id}"));
    providers
        .iter()
        .map(|provider| match provider {
            OrtExecutionProvider::CPU => "cpu".to_string(),
            OrtExecutionProvider::CUDA { device_id, .. } => format!("cuda{}", device(device_id)),
            OrtExecutionProvider::DirectML { device_id } => {
                format!("directml{}", device(device_id))
            }
            OrtExecutionProvider::OpenVINO { .. } => "openvino".to_string(),
            OrtExecutionProvider::TensorRT { device_id, .. } => {
                format!("tensorrt{}", device(device_id))
            }
            OrtExecutionProvider::CoreML { .. } => "coreml".to_string(),
            OrtExecutionProvider::WebGPU => "webgpu".to_string(),
        })
        .collect::<Vec<_>>()
        .join(",")
}

/// Execution providers for ONNX Runtime.
///
/// This enum represents the different execution providers that can be used
/// with ONNX Runtime for model inference.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Default)]
pub enum OrtExecutionProvider {
    /// CPU execution provider (always available)
    #[default]
    CPU,
    /// NVIDIA CUDA execution provider
    CUDA {
        /// CUDA device ID (default: 0)
        device_id: Option<i32>,
        /// Memory limit in bytes (optional)
        gpu_mem_limit: Option<usize>,
        /// Arena extend strategy: "NextPowerOfTwo" or "SameAsRequested"
        arena_extend_strategy: Option<String>,
        /// CUDNN convolution algorithm search: "Exhaustive", "Heuristic", or "Default"
        cudnn_conv_algo_search: Option<String>,
        /// CUDNN convolution use max workspace (default: true)
        cudnn_conv_use_max_workspace: Option<bool>,
    },
    /// DirectML execution provider (Windows only)
    DirectML {
        /// DirectML device ID (default: 0)
        device_id: Option<i32>,
    },
    /// OpenVINO execution provider
    OpenVINO {
        /// Device type (e.g., "CPU", "GPU", "MYRIAD")
        device_type: Option<String>,
        /// Number of threads (optional)
        num_threads: Option<usize>,
    },
    /// TensorRT execution provider
    TensorRT {
        /// TensorRT device ID (default: 0)
        device_id: Option<i32>,
        /// Maximum workspace size in bytes
        max_workspace_size: Option<usize>,
        /// Minimum subgraph size for TensorRT acceleration
        min_subgraph_size: Option<usize>,
        /// FP16 enable flag
        fp16_enable: Option<bool>,
        /// Enable use of timing cache to speed up builds
        timing_cache: Option<bool>,
        /// Set path for storing timing cache
        timing_cache_path: Option<String>,
        /// Force use of timing cache regardless of GPU match
        force_timing_cache: Option<bool>,
        /// Enable caching of TensorRT engines
        engine_cache: Option<bool>,
        /// Set path to store cached TensorRT engines
        engine_cache_path: Option<String>,
        /// Dump ep context model
        dump_ep_context_model: Option<bool>,
        /// The path of an embedded engine model
        ep_context_file_path: Option<String>,
    },
    /// CoreML execution provider (macOS/iOS only)
    CoreML {
        /// Use CPU and Apple Neural Engine compute units. Despite the
        /// historical name, unsupported nodes may still execute on CPU.
        ane_only: Option<bool>,
        /// Enable subgraphs
        subgraphs: Option<bool>,
    },
    /// WebGPU execution provider
    WebGPU,
}

/// Configuration for ONNX Runtime sessions.
///
/// This struct contains various configuration options for ONNX Runtime sessions,
/// including threading, memory management, and optimization settings.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct OrtSessionConfig {
    /// Number of threads used to parallelize execution within nodes
    pub intra_threads: Option<usize>,
    /// Number of threads used to parallelize execution across nodes
    pub inter_threads: Option<usize>,
    /// Enable parallel execution mode
    pub parallel_execution: Option<bool>,
    /// Graph optimization level
    pub optimization_level: Option<OrtGraphOptimizationLevel>,
    /// Execution providers in order of preference
    pub execution_providers: Option<Vec<OrtExecutionProvider>>,
    /// Enable memory pattern optimization
    pub enable_mem_pattern: Option<bool>,
    /// Log severity level (0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal)
    pub log_severity_level: Option<i32>,
    /// Log verbosity level
    pub log_verbosity_level: Option<i32>,
    /// Session configuration entries (key-value pairs)
    pub session_config_entries: Option<std::collections::HashMap<String, String>>,
    /// Return idle CUDA arena memory to the device after every run.
    ///
    /// Each ONNX Runtime session keeps its own CUDA memory arena, and arenas
    /// never give memory back on their own. A pipeline that holds many CUDA
    /// sessions fed variable-sized crops (layout, table, formula, OCR) grows
    /// every arena to its high-water mark, and the sum can exceed the GPU even
    /// though the models never need that much at once. Shrinkage releases the
    /// unused arena chunks at the end of each run, at a small per-run cost.
    /// Only takes effect with a CUDA execution provider.
    pub arena_shrinkage: Option<bool>,
}

impl OrtSessionConfig {
    /// Creates a new OrtSessionConfig with default values.
    pub fn new() -> Self {
        Self::default()
    }

    /// Select available compiled accelerators in CUDA(0), CoreML, DirectML(0)
    /// order, with CPU as the final fallback.
    ///
    /// Provider registration is probed during model or pipeline construction,
    /// before choosing pipeline batch defaults. Use [`Self::resolve_auto`] to
    /// resolve the selection explicitly. Unavailable accelerators are omitted.
    /// DirectML uses sequential execution and disables memory patterns.
    /// TensorRT, OpenVINO, and WebGPU require explicit configuration because of
    /// their initialization cost and compatibility requirements.
    pub fn auto() -> Self {
        let mut candidates = vec![
            #[cfg(feature = "cuda")]
            OrtExecutionProvider::CUDA {
                device_id: Some(0),
                gpu_mem_limit: None,
                arena_extend_strategy: None,
                cudnn_conv_algo_search: None,
                cudnn_conv_use_max_workspace: None,
            },
            #[cfg(all(feature = "coreml", any(target_os = "macos", target_os = "ios")))]
            OrtExecutionProvider::CoreML {
                ane_only: None,
                subgraphs: None,
            },
            #[cfg(all(feature = "directml", target_os = "windows"))]
            OrtExecutionProvider::DirectML { device_id: Some(0) },
        ];
        let has_candidates = !candidates.is_empty();
        candidates.push(OrtExecutionProvider::CPU);
        let config = Self::new().with_execution_providers(candidates);
        if has_candidates {
            config.with_pending_auto_selection()
        } else {
            config
        }
    }

    /// Marks the current provider list as automatic candidates.
    ///
    /// The marker records the candidates' kinds and device IDs, so replacing
    /// `execution_providers` directly cancels the pending selection.
    pub(crate) fn with_pending_auto_selection(self) -> Self {
        let signature = auto_signature(&self.get_execution_providers());
        self.add_config_entry(AUTO_DEVICE_CONFIG_ENTRY, signature)
    }

    /// Whether automatic execution-provider selection is still pending.
    pub fn has_pending_auto_selection(&self) -> bool {
        self.session_config_entries
            .as_ref()
            .and_then(|entries| entries.get(AUTO_DEVICE_CONFIG_ENTRY))
            .is_some_and(|value| *value == auto_signature(&self.get_execution_providers()))
    }

    /// Resolve automatic provider preferences into available providers.
    ///
    /// High-level pipelines call this before choosing batch sizes, and model
    /// builders call it before creating sessions. Probing can initialize device
    /// runtimes, so configure global runtime settings before calling this method.
    /// Explicit provider lists and caller-supplied session settings are preserved.
    pub fn resolve_auto(self) -> Self {
        self.resolve_auto_with_probe(crate::core::inference::OrtInfer::probe_execution_provider)
    }

    pub(crate) fn resolve_auto_with_probe(
        mut self,
        probe: impl FnMut(&OrtExecutionProvider) -> ort::Result<()>,
    ) -> Self {
        let pending = self.has_pending_auto_selection();
        self.clear_auto_selection();
        if !pending {
            return self;
        }
        let candidates = self
            .get_execution_providers()
            .into_iter()
            .filter(|provider| !matches!(provider, OrtExecutionProvider::CPU))
            .collect();
        let selected = Self::auto_with_probe(candidates, probe);
        let uses_directml = selected
            .get_execution_providers()
            .iter()
            .any(|provider| matches!(provider, OrtExecutionProvider::DirectML { .. }));
        self.execution_providers = selected.execution_providers;
        if uses_directml {
            // DirectML fails to initialize with parallel execution or memory
            // patterns, so its requirements override caller preferences.
            if self.parallel_execution == Some(true) || self.enable_mem_pattern == Some(true) {
                tracing::warn!(
                    "DirectML was selected automatically; disabling parallel execution and memory patterns"
                );
            }
            self.parallel_execution = Some(false);
            self.enable_mem_pattern = Some(false);
        }
        self
    }

    fn clear_auto_selection(&mut self) {
        if let Some(entries) = self.session_config_entries.as_mut() {
            entries.remove(AUTO_DEVICE_CONFIG_ENTRY);
            if entries.is_empty() {
                self.session_config_entries = None;
            }
        }
    }

    fn auto_with_probe(
        candidates: Vec<OrtExecutionProvider>,
        mut probe: impl FnMut(&OrtExecutionProvider) -> ort::Result<()>,
    ) -> Self {
        let mut providers = Vec::new();
        for provider in candidates {
            match probe(&provider) {
                Ok(()) => providers.push(provider),
                Err(error) => {
                    tracing::debug!(?provider, %error, "automatic execution provider selection failed")
                }
            }
        }
        let uses_directml = providers
            .iter()
            .any(|provider| matches!(provider, OrtExecutionProvider::DirectML { .. }));
        tracing::info!(provider = ?providers.first().unwrap_or(&OrtExecutionProvider::CPU), "automatically selected execution provider");
        providers.push(OrtExecutionProvider::CPU);
        let config = Self::new().with_execution_providers(providers);
        if uses_directml {
            config
                .with_parallel_execution(false)
                .with_memory_pattern(false)
        } else {
            config
        }
    }

    /// Sets the number of intra-op threads.
    pub fn with_intra_threads(mut self, threads: usize) -> Self {
        self.intra_threads = Some(threads);
        self
    }

    /// Sets the number of inter-op threads.
    pub fn with_inter_threads(mut self, threads: usize) -> Self {
        self.inter_threads = Some(threads);
        self
    }

    /// Enables or disables parallel execution.
    pub fn with_parallel_execution(mut self, enabled: bool) -> Self {
        self.parallel_execution = Some(enabled);
        self
    }

    /// Sets the graph optimization level.
    pub fn with_optimization_level(mut self, level: OrtGraphOptimizationLevel) -> Self {
        self.optimization_level = Some(level);
        self
    }

    /// Sets the execution providers, in order of preference.
    pub fn with_execution_providers(mut self, providers: Vec<OrtExecutionProvider>) -> Self {
        self.clear_auto_selection();
        self.execution_providers = Some(providers);
        self
    }

    /// Appends a single execution provider.
    pub fn add_execution_provider(mut self, provider: OrtExecutionProvider) -> Self {
        self.clear_auto_selection();
        if let Some(ref mut providers) = self.execution_providers {
            providers.push(provider);
        } else {
            self.execution_providers = Some(vec![provider]);
        }
        self
    }

    /// Enables or disables CUDA arena shrinkage after every run (see
    /// [`OrtSessionConfig::arena_shrinkage`]).
    pub fn with_arena_shrinkage(mut self, enable: bool) -> Self {
        self.arena_shrinkage = Some(enable);
        self
    }

    /// Enables or disables memory pattern optimization.
    pub fn with_memory_pattern(mut self, enable: bool) -> Self {
        self.enable_mem_pattern = Some(enable);
        self
    }

    /// Sets the log severity level (0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal).
    pub fn with_log_severity_level(mut self, level: i32) -> Self {
        self.log_severity_level = Some(level);
        self
    }

    /// Sets the log verbosity level.
    pub fn with_log_verbosity_level(mut self, level: i32) -> Self {
        self.log_verbosity_level = Some(level);
        self
    }

    /// Adds a session configuration entry.
    pub fn add_config_entry<K: Into<String>, V: Into<String>>(mut self, key: K, value: V) -> Self {
        if let Some(ref mut entries) = self.session_config_entries {
            entries.insert(key.into(), value.into());
        } else {
            let mut entries = std::collections::HashMap::new();
            entries.insert(key.into(), value.into());
            self.session_config_entries = Some(entries);
        }
        self
    }

    /// Sets advanced options for any CoreML execution provider in this session.
    pub fn with_coreml_config(mut self, config: OrtCoreMLConfig) -> Self {
        let value =
            serde_json::to_string(&config).expect("serializing OrtCoreMLConfig cannot fail");
        self.session_config_entries
            .get_or_insert_with(Default::default)
            .insert(COREML_CONFIG_ENTRY.to_owned(), value);
        self
    }

    pub(crate) fn coreml_config(&self) -> Result<Option<OrtCoreMLConfig>, serde_json::Error> {
        self.session_config_entries
            .as_ref()
            .and_then(|entries| entries.get(COREML_CONFIG_ENTRY))
            .map(|value| serde_json::from_str(value))
            .transpose()
    }

    /// Effective intra-op thread count, defaulting to available parallelism.
    pub fn get_intra_threads(&self) -> usize {
        self.intra_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| n.get())
                .unwrap_or(1)
        })
    }

    /// Effective inter-op thread count, defaulting to 1.
    pub fn get_inter_threads(&self) -> usize {
        self.inter_threads.unwrap_or(1)
    }

    /// Effective graph optimization level, defaulting to `OrtGraphOptimizationLevel::default()`.
    pub fn get_optimization_level(&self) -> OrtGraphOptimizationLevel {
        self.optimization_level.unwrap_or_default()
    }

    /// Configured execution providers, defaulting to CPU.
    pub fn get_execution_providers(&self) -> Vec<OrtExecutionProvider> {
        self.execution_providers
            .clone()
            .unwrap_or_else(|| vec![OrtExecutionProvider::CPU])
    }

    /// Returns whether an explicitly configured hardware accelerator is present.
    ///
    /// `execution_providers` is a preference-ordered list: ONNX Runtime lets
    /// each provider claim graph nodes in list order, and CPU can claim
    /// almost any node, so a CPU entry listed first effectively runs the
    /// session on CPU regardless of what accelerators follow it. Only the
    /// first provider therefore determines whether this is an accelerated
    /// configuration. No provider configuration, an empty provider list, and
    /// a CPU-first list (including CPU alone) all use CPU-oriented pipeline
    /// defaults; an accelerator listed first with a CPU fallback after it
    /// (CUDA, TensorRT, DirectML, OpenVINO, CoreML, or WebGPU) counts as
    /// accelerated.
    pub fn has_accelerator_provider(&self) -> bool {
        self.execution_providers
            .as_ref()
            .and_then(|providers| providers.first())
            .is_some_and(|provider| !matches!(provider, OrtExecutionProvider::CPU))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn auto_candidates() -> Vec<OrtExecutionProvider> {
        vec![
            OrtExecutionProvider::CUDA {
                device_id: Some(0),
                gpu_mem_limit: None,
                arena_extend_strategy: None,
                cudnn_conv_algo_search: None,
                cudnn_conv_use_max_workspace: None,
            },
            OrtExecutionProvider::CoreML {
                ane_only: None,
                subgraphs: None,
            },
            OrtExecutionProvider::DirectML { device_id: Some(0) },
        ]
    }

    #[cfg(not(any(feature = "cuda", feature = "coreml", feature = "directml")))]
    #[test]
    fn auto_without_accelerator_features_is_cpu_only() {
        let auto = OrtSessionConfig::auto();
        let cpu = OrtSessionConfig::new().with_execution_providers(vec![OrtExecutionProvider::CPU]);
        assert_eq!(
            serde_json::to_value(auto).unwrap(),
            serde_json::to_value(cpu).unwrap()
        );
    }

    #[test]
    fn auto_registration_failures_leave_a_cpu_only_configuration() {
        let candidates = auto_candidates();
        let mut attempted = Vec::new();
        let config = OrtSessionConfig::auto_with_probe(candidates.clone(), |provider| {
            attempted.push(provider.clone());
            Err(ort::Error::new("provider unavailable"))
        });
        assert_eq!(attempted, candidates);
        assert_eq!(
            config.get_execution_providers(),
            [OrtExecutionProvider::CPU]
        );
        assert!(!config.has_accelerator_provider());
        assert_eq!(config.parallel_execution, None);
        assert_eq!(config.enable_mem_pattern, None);
    }

    #[test]
    fn auto_retains_priority_and_configures_directml() {
        let candidates = auto_candidates();
        let config = OrtSessionConfig::auto_with_probe(candidates.clone(), |_| Ok(()));
        let mut expected = candidates;
        expected.push(OrtExecutionProvider::CPU);
        assert_eq!(config.get_execution_providers(), expected);
        assert!(config.has_accelerator_provider());
        assert_eq!(config.parallel_execution, Some(false));
        assert_eq!(config.enable_mem_pattern, Some(false));
    }

    #[test]
    fn auto_resolution_preserves_tuning_and_removes_internal_marker() {
        let mut candidates = auto_candidates();
        candidates.push(OrtExecutionProvider::CPU);
        let config = OrtSessionConfig::new()
            .with_execution_providers(candidates)
            .with_pending_auto_selection()
            .add_config_entry("session.dynamic_block_base", "4")
            .with_intra_threads(2)
            .with_parallel_execution(true)
            .with_memory_pattern(true);
        assert!(config.has_pending_auto_selection());
        let resolved = config.resolve_auto_with_probe(|_| Err(ort::Error::new("unavailable")));
        assert!(!resolved.has_pending_auto_selection());
        assert!(!resolved.has_accelerator_provider());
        let expected = OrtSessionConfig::new()
            .with_execution_providers(vec![OrtExecutionProvider::CPU])
            .add_config_entry("session.dynamic_block_base", "4")
            .with_intra_threads(2)
            .with_parallel_execution(true)
            .with_memory_pattern(true);
        assert_eq!(
            serde_json::to_value(resolved).unwrap(),
            serde_json::to_value(expected).unwrap()
        );
    }

    #[test]
    fn auto_resolved_directml_overrides_incompatible_caller_settings() {
        let mut candidates = auto_candidates();
        candidates.push(OrtExecutionProvider::CPU);
        let config = OrtSessionConfig::new()
            .with_execution_providers(candidates)
            .with_pending_auto_selection()
            .with_parallel_execution(true)
            .with_memory_pattern(true);
        let resolved = config.resolve_auto_with_probe(|provider| {
            if matches!(provider, OrtExecutionProvider::DirectML { .. }) {
                Ok(())
            } else {
                Err(ort::Error::new("unavailable"))
            }
        });
        assert_eq!(
            resolved.get_execution_providers(),
            [
                OrtExecutionProvider::DirectML { device_id: Some(0) },
                OrtExecutionProvider::CPU,
            ]
        );
        assert_eq!(resolved.parallel_execution, Some(false));
        assert_eq!(resolved.enable_mem_pattern, Some(false));
    }

    #[test]
    fn direct_provider_replacement_cancels_pending_auto_selection() {
        let mut candidates = auto_candidates();
        candidates.push(OrtExecutionProvider::CPU);
        let mut config = OrtSessionConfig::new()
            .with_execution_providers(candidates)
            .with_pending_auto_selection();
        assert!(config.has_pending_auto_selection());
        let explicit = vec![
            OrtExecutionProvider::OpenVINO {
                device_type: None,
                num_threads: None,
            },
            OrtExecutionProvider::CPU,
        ];
        config.execution_providers = Some(explicit.clone());
        assert!(!config.has_pending_auto_selection());
        let resolved =
            config.resolve_auto_with_probe(|_| panic!("explicit providers must not be probed"));
        assert_eq!(resolved.get_execution_providers(), explicit);
        assert!(resolved.session_config_entries.is_none());
    }

    #[test]
    fn tuning_a_candidate_keeps_auto_selection_pending() {
        let mut config = OrtSessionConfig::new()
            .with_execution_providers(auto_candidates())
            .with_pending_auto_selection();
        if let Some(OrtExecutionProvider::CUDA {
            arena_extend_strategy,
            ..
        }) = config
            .execution_providers
            .as_mut()
            .and_then(|eps| eps.first_mut())
        {
            *arena_extend_strategy = Some("SameAsRequested".to_string());
        }
        assert!(config.has_pending_auto_selection());
    }

    #[test]
    fn explicit_provider_resolution_never_probes_hardware() {
        let config = OrtSessionConfig::new().with_execution_providers(auto_candidates());
        let expected = serde_json::to_value(&config).unwrap();
        let resolved =
            config.resolve_auto_with_probe(|_| panic!("explicit preferences must not be probed"));
        assert_eq!(serde_json::to_value(resolved).unwrap(), expected);
    }

    #[test]
    fn explicit_provider_setters_clear_pending_auto_selection() {
        let pending = OrtSessionConfig::new()
            .with_execution_providers(auto_candidates())
            .with_pending_auto_selection();
        let explicit = pending
            .clone()
            .with_execution_providers(vec![OrtExecutionProvider::CPU]);
        assert!(!explicit.has_pending_auto_selection());
        assert_eq!(
            explicit.get_execution_providers(),
            [OrtExecutionProvider::CPU]
        );
        let appended = pending.add_execution_provider(OrtExecutionProvider::CPU);
        assert!(!appended.has_pending_auto_selection());
        let mut expected = auto_candidates();
        expected.push(OrtExecutionProvider::CPU);
        assert_eq!(appended.get_execution_providers(), expected);
    }

    #[test]
    fn auto_skips_an_unavailable_cuda_provider() {
        let config = OrtSessionConfig::auto_with_probe(auto_candidates(), |provider| {
            if matches!(provider, OrtExecutionProvider::CUDA { .. }) {
                Err(ort::Error::new("CUDA driver unavailable"))
            } else {
                Ok(())
            }
        });
        assert_eq!(
            config.get_execution_providers(),
            [
                OrtExecutionProvider::CoreML {
                    ane_only: None,
                    subgraphs: None
                },
                OrtExecutionProvider::DirectML { device_id: Some(0) },
                OrtExecutionProvider::CPU,
            ]
        );
    }

    #[test]
    fn test_ort_session_config_builder() {
        let config = OrtSessionConfig::new()
            .with_intra_threads(4)
            .with_inter_threads(2)
            .with_optimization_level(OrtGraphOptimizationLevel::Level2)
            .with_memory_pattern(true)
            .add_execution_provider(OrtExecutionProvider::CPU);

        assert_eq!(config.intra_threads, Some(4));
        assert_eq!(config.inter_threads, Some(2));
        assert!(matches!(
            config.optimization_level,
            Some(OrtGraphOptimizationLevel::Level2)
        ));
        assert_eq!(config.enable_mem_pattern, Some(true));
        assert!(config.execution_providers.is_some());
    }

    #[test]
    fn test_ort_session_config_getters() {
        let config = OrtSessionConfig::new()
            .with_intra_threads(8)
            .with_inter_threads(4)
            .with_optimization_level(OrtGraphOptimizationLevel::All);

        assert_eq!(config.get_intra_threads(), 8);
        assert_eq!(config.get_inter_threads(), 4);
        assert!(matches!(
            config.get_optimization_level(),
            OrtGraphOptimizationLevel::All
        ));
    }

    #[test]
    fn test_accelerator_provider_detection() {
        assert!(!OrtSessionConfig::new().has_accelerator_provider());
        assert!(
            !OrtSessionConfig::new()
                .with_execution_providers(vec![OrtExecutionProvider::CPU])
                .has_accelerator_provider()
        );
        assert!(
            OrtSessionConfig::new()
                .with_execution_providers(vec![
                    OrtExecutionProvider::DirectML { device_id: Some(0) },
                    OrtExecutionProvider::CPU,
                ])
                .has_accelerator_provider()
        );
        // CPU listed first claims nearly every node before the accelerator
        // gets a chance to, so this is a CPU-preferred configuration despite
        // the accelerator appearing later in the list.
        assert!(
            !OrtSessionConfig::new()
                .with_execution_providers(vec![
                    OrtExecutionProvider::CPU,
                    OrtExecutionProvider::DirectML { device_id: Some(0) },
                ])
                .has_accelerator_provider()
        );
    }

    #[test]
    fn coreml_provider_keeps_legacy_variant_shape() {
        let provider = OrtExecutionProvider::CoreML {
            ane_only: Some(true),
            subgraphs: Some(false),
        };
        let OrtExecutionProvider::CoreML {
            ane_only,
            subgraphs,
        } = provider
        else {
            unreachable!()
        };
        assert_eq!(ane_only, Some(true));
        assert_eq!(subgraphs, Some(false));
    }

    #[test]
    fn coreml_advanced_config_round_trips_through_session_config() {
        let expected = OrtCoreMLConfig {
            compute_units: Some(OrtCoreMLComputeUnits::CPUAndGPU),
            model_format: Some(OrtCoreMLModelFormat::MLProgram),
            static_input_shapes: Some(true),
            specialization_strategy: Some(OrtCoreMLSpecializationStrategy::FastPrediction),
            allow_low_precision_accumulation_on_gpu: Some(true),
            profile_compute_plan: None,
            model_cache_dir: Some("cache".to_owned()),
        };
        let config = OrtSessionConfig::new().with_coreml_config(expected.clone());
        assert_eq!(config.coreml_config().unwrap(), Some(expected));
    }
}
