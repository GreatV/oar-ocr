// Single translation unit for the crate's custom CUDA kernels. Keeping all
// kernels in one PTX module lets existing callers continue loading
// `oar_vl_kernels.ptx` while model-specific implementations remain colocated
// with their Rust modules.
#include "runtime/cuda/dynamic_kv.cu"

#include "backbones/qwen3_5/gated_delta.cu"
