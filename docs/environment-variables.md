# Environment Variables

Runtime environment variables read by the oar-ocr crates. Project-specific variables use the `OAR_` prefix (`OAR_VL_` for the Vision-Language crate).

## Summary

| Variable | Crate | Default | Purpose |
|---|---|---|---|
| [`OAR_HOME`](#oar_home) | `oar-ocr-core` | `~/.oar` | Model cache directory for auto-download |
| [`OAR_VL_DTYPE`](#oar_vl_dtype) | `oar-ocr-vl` | auto | Compute dtype for VL models |
| [`OAR_VL_ATTN_FULL_SEQ_THRESHOLD`](#oar_vl_attn_full_seq_threshold) | `oar-ocr-vl` | `8192` | PaddleOCR-VL vision attention path selection |
| [`OAR_VL_DISABLE_FLASH_ATTN`](#vl-performance-and-debug-overrides) | `oar-ocr-vl` | unset | Disable CUDA FlashAttention |
| [`OAR_VL_DISABLE_GQA`](#vl-performance-and-debug-overrides) | `oar-ocr-vl` | unset | Use expanded-K/V GQA fallback |
| [`OAR_VL_DISABLE_CUDA_GRAPH`](#vl-performance-and-debug-overrides) | `oar-ocr-vl` | unset | Disable shared decoder CUDA graphs |
| [`OAR_VL_DISABLE_SPECULATIVE`](#vl-performance-and-debug-overrides) | `oar-ocr-vl` | unset | Disable GLM-OCR speculative decoding |
| [`OAR_PADDLEOCR_VL_DISABLE_CUDA_GRAPH`](#vl-performance-and-debug-overrides) | `oar-ocr-vl` | unset | Disable PaddleOCR-VL CUDA graphs |
| [`OAR_GLMOCR_DISABLE_MTP`](#vl-performance-and-debug-overrides) | `oar-ocr-vl` | unset | Disable GLM-OCR MTP |
| [`OAR_GLMOCR_DISABLE_CUDA_GRAPH`](#vl-performance-and-debug-overrides) | `oar-ocr-vl` | unset | Disable GLM-OCR CUDA graphs |
| [`OAR_HUNYUAN_DISABLE_CUDA_GRAPH`](#vl-performance-and-debug-overrides) | `oar-ocr-vl` | unset | Disable HunyuanOCR CUDA graphs |
| [`OAR_HUNYUAN_DISABLE_AR_CUDA_GRAPH`](#vl-performance-and-debug-overrides) | `oar-ocr-vl` | unset | Disable HunyuanOCR AR CUDA graph |
| [`OAR_MINERU_DISABLE_CUDA_GRAPH`](#vl-performance-and-debug-overrides) | `oar-ocr-vl` | unset | Disable MinerU CUDA graphs |
| [`OAR_MINERU_DISABLE_GPU_SAMPLING`](#vl-performance-and-debug-overrides) | `oar-ocr-vl` | unset | Disable MinerU GPU sampling |
| [`OAR_MINERU_DIFFUSION_DISABLE_GPU_SAMPLING`](#vl-performance-and-debug-overrides) | `oar-ocr-vl` | unset | Disable MinerU-Diffusion GPU sampling |
| [`CUDA_COMPUTE_CAP`](#cuda-build-overrides) | `oar-ocr-vl` build | auto | CUDA PTX target architecture |
| [`NVCC`](#cuda-build-overrides) | `oar-ocr-vl` build | `nvcc` | CUDA compiler path |
| [`CUDA_LAUNCH_BLOCKING`](#cuda_launch_blocking) | `oar-ocr-core` | set to `1` if unset | ONNX Runtime CUDA workaround |
| [`RUST_LOG`](#rust_log) | examples | `info` | Log filter |

## `OAR_HOME`

Directory where the `auto-download` feature caches model files (default `~/.oar`). Bare model file names passed to the builders are downloaded here and verified against their expected SHA-256. See [models.md](models.md#auto-download) for the exact path resolution rules.

```bash
OAR_HOME=/data/oar-models cargo run --release --example ocr -- doc.jpg
```

## `OAR_VL_DTYPE`

Overrides the automatic weight/compute dtype selection for Vision-Language models (PaddleOCR-VL, GLM-OCR, HunyuanOCR, MinerU2.5/Pro, and MinerU-Diffusion).

Accepted values (case-insensitive):

- `bf16` (alias `bfloat16`)
- `f16` (aliases `fp16`, `float16`, `half`)
- `f32` (aliases `fp32`, `float32`)

Without the override, devices that advertise BF16 support use it after a runtime kernel probe. If the probe fails (for example on pre-Ampere CUDA GPUs), inference falls back to F16. CPU and devices without advertised BF16 support use F32.

```bash
OAR_VL_DTYPE=f16 cargo run --release --features cuda,download-binaries -p oar-ocr-vl --example paddleocr_vl -- ...
```

## `OAR_VL_ATTN_FULL_SEQ_THRESHOLD`

Maximum sequence length (in vision patches) for which PaddleOCR-VL's vision attention may use the single-shot full-matrix path. Longer sequences use the numerically equivalent chunked path, which needs far less peak memory. Default `8192`. Set `0` to force chunked attention.

Independently of this threshold, the full path is skipped automatically whenever its softmax scratch would not fit in the free device memory, so low-VRAM GPUs normally don't need this variable.

```bash
OAR_VL_ATTN_FULL_SEQ_THRESHOLD=0 cargo run --release --features cuda,download-binaries -p oar-ocr-vl --example paddleocr_vl -- ...
```

## VL Performance and Debug Overrides

The following presence-based switches select portable fallbacks or disable optional accelerations. Setting a variable to any value enables the switch.

| Variable | Scope | Effect |
|---|---|---|
| `OAR_VL_DISABLE_FLASH_ATTN` | All CUDA VLM backends | Use the eager attention fallback instead of FlashAttention |
| `OAR_VL_DISABLE_GQA` | Backends using grouped-query attention | Expand K/V heads before eager attention instead of using the grouped implementation |
| `OAR_VL_DISABLE_CUDA_GRAPH` | PaddleOCR-VL, GLM-OCR, MinerU2.5/Pro | Disable decoder CUDA graph capture and replay |
| `OAR_VL_DISABLE_SPECULATIVE` | GLM-OCR | Disable MTP speculative decoding |
| `OAR_PADDLEOCR_VL_DISABLE_CUDA_GRAPH` | PaddleOCR-VL variants | Disable decoder CUDA graphs only for PaddleOCR-VL |
| `OAR_GLMOCR_DISABLE_MTP` | GLM-OCR | Do not load or use the MTP predictor |
| `OAR_GLMOCR_DISABLE_CUDA_GRAPH` | GLM-OCR | Disable autoregressive and MTP CUDA graphs |
| `OAR_HUNYUAN_DISABLE_CUDA_GRAPH` | HunyuanOCR | Disable target, autoregressive, and DFlash CUDA graphs |
| `OAR_HUNYUAN_DISABLE_AR_CUDA_GRAPH` | HunyuanOCR | Disable only the single-token autoregressive CUDA graph |
| `OAR_MINERU_DISABLE_CUDA_GRAPH` | MinerU2.5/Pro | Disable decoder CUDA graphs |
| `OAR_MINERU_DISABLE_GPU_SAMPLING` | MinerU2.5/Pro | Use the host sampling fallback instead of the CUDA greedy sampler |
| `OAR_MINERU_DIFFUSION_DISABLE_GPU_SAMPLING` | MinerU-Diffusion | Use the host sampling fallback instead of the CUDA sampler |

These switches are primarily useful for compatibility checks, debugging, and numerical comparisons. The accelerated paths remain enabled by default when the active device and dtype support them.

## CUDA Build Overrides

With the `cuda` feature enabled, `oar-ocr-vl` compiles its custom kernels to PTX. `CUDA_COMPUTE_CAP` overrides automatic GPU detection and accepts values such as `89`, `8.9`, `sm_89`, or `compute_89`. Compute capability 8.0 or newer is required. `NVCC` overrides the CUDA compiler executable.

```bash
CUDA_COMPUTE_CAP=89 NVCC=/usr/local/cuda/bin/nvcc \
    cargo build -p oar-ocr-vl --features cuda,download-binaries
```

## `CUDA_LAUNCH_BLOCKING`

Standard CUDA variable, listed here because the classic (ONNX Runtime) pipeline sets it to `1` at startup when it is not already set, to work around [onnxruntime#4829](https://github.com/microsoft/onnxruntime/issues/4829) (PP-FormulaNet's autoregressive `Loop` corrupting CUDA-EP arena buffers). Preset any value (e.g. `0`) to keep your own setting. Note this serializes CUDA kernel launches process-wide.

## `RUST_LOG`

Standard [`tracing_subscriber::EnvFilter`](https://docs.rs/tracing-subscriber) filter used by the examples, e.g. `RUST_LOG=oar_ocr_vl=debug` to see why the vision attention picked the chunked path.
