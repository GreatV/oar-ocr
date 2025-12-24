use super::*;
use crate::core::config::{
    OrtExecutionProvider, OrtGraphOptimizationLevel as OG, OrtSessionConfig,
};
use ort::execution_providers::ExecutionProviderDispatch;
use ort::logging::LogLevel;
use ort::session::builder::{GraphOptimizationLevel as GOL, SessionBuilder};

impl OrtInfer {
    pub(super) fn apply_ort_config(
        mut builder: SessionBuilder,
        cfg: &OrtSessionConfig,
    ) -> Result<SessionBuilder, ort::Error> {
        if let Some(intra) = cfg.intra_threads {
            builder = builder.with_intra_threads(intra)?;
        }
        if let Some(inter) = cfg.inter_threads {
            builder = builder.with_inter_threads(inter)?;
        }
        if let Some(par) = cfg.parallel_execution {
            builder = builder.with_parallel_execution(par)?;
        }
        if let Some(level) = cfg.optimization_level {
            let mapped = match level {
                OG::DisableAll => GOL::Disable,
                OG::Level1 => GOL::Level1,
                OG::Level2 => GOL::Level2,
                OG::Level3 => GOL::Level3,
                // ONNX Runtime treats "All" optimizations as an alias for the
                // highest available level (Level3) in its public API, so we mirror
                // that behavior to stay aligned with upstream semantics.
                OG::All => GOL::Level3,
            };
            builder = builder.with_optimization_level(mapped)?;
        }
        if let Some(log_level) = cfg.log_severity_level {
            // Map log severity level to LogLevel
            // 0=Verbose, 1=Info, 2=Warning, 3=Error, 4=Fatal
            let logging_level = match log_level {
                0 => LogLevel::Verbose,
                1 => LogLevel::Info,
                2 => LogLevel::Warning,
                3 => LogLevel::Error,
                _ => LogLevel::Fatal,
            };
            builder = builder.with_log_level(logging_level)?;
        }
        if let Some(log_verbosity) = cfg.log_verbosity_level {
            builder = builder.with_log_verbosity(log_verbosity)?;
        }
        if let Some(enable) = cfg.enable_mem_pattern {
            builder = builder.with_memory_pattern(enable)?;
        }
        if let Some(entries) = &cfg.session_config_entries {
            for (key, value) in entries {
                builder = builder.with_config_entry(key, value)?;
            }
        }
        if let Some(eps) = &cfg.execution_providers {
            let providers = Self::build_execution_providers(eps)?;
            if !providers.is_empty() {
                builder = builder.with_execution_providers(providers)?;
            }
        }
        Ok(builder)
    }

    fn build_execution_providers(
        eps: &[OrtExecutionProvider],
    ) -> Result<Vec<ExecutionProviderDispatch>, ort::Error> {
        use crate::core::config::OrtExecutionProvider as EP;
        let mut providers = Vec::new();

        for ep in eps {
            match ep {
                EP::CPU => {
                    providers
                        .push(ort::execution_providers::CPUExecutionProvider::default().build());
                }
                #[cfg(feature = "cuda")]
                EP::CUDA {
                    device_id,
                    gpu_mem_limit,
                    arena_extend_strategy,
                    cudnn_conv_algo_search,
                    cudnn_conv_use_max_workspace,
                } => {
                    use ort::execution_providers::{
                        ArenaExtendStrategy, cuda::CuDNNConvAlgorithmSearch,
                    };
                    let mut cuda_provider =
                        ort::execution_providers::CUDAExecutionProvider::default();
                    if let Some(id) = device_id {
                        cuda_provider = cuda_provider.with_device_id(*id);
                    }
                    if let Some(limit) = gpu_mem_limit {
                        cuda_provider = cuda_provider.with_memory_limit(*limit);
                    }
                    if let Some(strategy) = arena_extend_strategy {
                        let strategy = match strategy.to_lowercase().as_str() {
                            "sameasrequested" | "same_as_requested" => {
                                ArenaExtendStrategy::SameAsRequested
                            }
                            _ => ArenaExtendStrategy::NextPowerOfTwo,
                        };
                        cuda_provider = cuda_provider.with_arena_extend_strategy(strategy);
                    }
                    if let Some(search) = cudnn_conv_algo_search {
                        let search = match search.to_lowercase().as_str() {
                            "heuristic" => CuDNNConvAlgorithmSearch::Heuristic,
                            "default" => CuDNNConvAlgorithmSearch::Default,
                            _ => CuDNNConvAlgorithmSearch::Exhaustive,
                        };
                        cuda_provider = cuda_provider.with_conv_algorithm_search(search);
                    }
                    if let Some(enable) = cudnn_conv_use_max_workspace {
                        cuda_provider = cuda_provider.with_conv_max_workspace(*enable);
                    }
                    providers.push(cuda_provider.build());
                }
                #[cfg(feature = "tensorrt")]
                EP::TensorRT {
                    device_id,
                    max_workspace_size,
                    min_subgraph_size,
                    fp16_enable,
                } => {
                    let mut trt_provider =
                        ort::execution_providers::TensorRTExecutionProvider::default();
                    if let Some(id) = device_id {
                        trt_provider = trt_provider.with_device_id(*id);
                    }
                    if let Some(workspace) = max_workspace_size {
                        trt_provider = trt_provider.with_max_workspace_size(*workspace);
                    }
                    if let Some(size) = min_subgraph_size {
                        trt_provider = trt_provider.with_min_subgraph_size(*size);
                    }
                    if let Some(fp16) = fp16_enable {
                        trt_provider = trt_provider.with_fp16(*fp16);
                    }
                    providers.push(trt_provider.build());
                }
                #[cfg(feature = "directml")]
                EP::DirectML { device_id } => {
                    let mut dml_provider =
                        ort::execution_providers::DirectMLExecutionProvider::default();
                    if let Some(id) = device_id {
                        dml_provider = dml_provider.with_device_id(*id);
                    }
                    providers.push(dml_provider.build());
                }
                #[cfg(feature = "coreml")]
                EP::CoreML {
                    ane_only,
                    subgraphs,
                } => {
                    use ort::execution_providers::coreml::CoreMLComputeUnits;
                    let mut coreml_provider =
                        ort::execution_providers::CoreMLExecutionProvider::default();
                    if let Some(true) = ane_only {
                        coreml_provider = coreml_provider
                            .with_compute_units(CoreMLComputeUnits::CPUAndNeuralEngine);
                    }
                    if let Some(sub) = subgraphs {
                        coreml_provider = coreml_provider.with_subgraphs(*sub);
                    }
                    providers.push(coreml_provider.build());
                }
                #[cfg(feature = "webgpu")]
                EP::WebGPU => {
                    providers
                        .push(ort::execution_providers::WebGPUExecutionProvider::default().build());
                }
                #[cfg(feature = "openvino")]
                EP::OpenVINO {
                    device_type,
                    num_threads,
                } => {
                    let mut openvino_provider =
                        ort::execution_providers::OpenVINOExecutionProvider::default();
                    if let Some(device) = device_type {
                        openvino_provider = openvino_provider.with_device_type(device.clone());
                    }
                    if let Some(threads) = num_threads {
                        openvino_provider = openvino_provider.with_num_threads(*threads);
                    }
                    providers.push(openvino_provider.build());
                }
                #[cfg(not(feature = "cuda"))]
                EP::CUDA { .. } => {
                    return Err(ort::Error::new(
                        "CUDA execution provider requested but cuda feature is not enabled",
                    ));
                }
                #[cfg(not(feature = "tensorrt"))]
                EP::TensorRT { .. } => {
                    return Err(ort::Error::new(
                        "TensorRT execution provider requested but tensorrt feature is not enabled",
                    ));
                }
                #[cfg(not(feature = "directml"))]
                EP::DirectML { .. } => {
                    return Err(ort::Error::new(
                        "DirectML execution provider requested but directml feature is not enabled",
                    ));
                }
                #[cfg(not(feature = "openvino"))]
                EP::OpenVINO { .. } => {
                    return Err(ort::Error::new(
                        "OpenVINO execution provider requested but openvino feature is not enabled",
                    ));
                }
                #[cfg(not(feature = "coreml"))]
                EP::CoreML { .. } => {
                    return Err(ort::Error::new(
                        "CoreML execution provider requested but coreml feature is not enabled",
                    ));
                }
                #[cfg(not(feature = "webgpu"))]
                EP::WebGPU => {
                    return Err(ort::Error::new(
                        "WebGPU execution provider requested but webgpu feature is not enabled",
                    ));
                }
            }
        }

        Ok(providers)
    }
}
