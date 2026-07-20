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
}

impl OrtSessionConfig {
    /// Creates a new OrtSessionConfig with default values.
    pub fn new() -> Self {
        Self::default()
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
        self.execution_providers = Some(providers);
        self
    }

    /// Appends a single execution provider.
    pub fn add_execution_provider(mut self, provider: OrtExecutionProvider) -> Self {
        if let Some(ref mut providers) = self.execution_providers {
            providers.push(provider);
        } else {
            self.execution_providers = Some(vec![provider]);
        }
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
    /// No provider configuration, an empty provider list, and a CPU-only list
    /// all use CPU-oriented pipeline defaults. A CPU fallback after CUDA,
    /// TensorRT, DirectML, OpenVINO, CoreML, or WebGPU still counts as an
    /// accelerated configuration.
    pub fn has_accelerator_provider(&self) -> bool {
        self.execution_providers.as_ref().is_some_and(|providers| {
            providers
                .iter()
                .any(|provider| !matches!(provider, OrtExecutionProvider::CPU))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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
