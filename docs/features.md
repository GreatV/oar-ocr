# Cargo Features

The root `oar-ocr` crate exposes optional Cargo features. Each feature is forwarded to `oar-ocr-core` under the same name.

The Cargo feature named `default` is not an additional capability. It is a feature set that enables `download-binaries` and `simd`.

## At a Glance

| Feature | Enabled by default | Category | Purpose |
|---|:---:|---|---|
| `simd` | Yes | CPU optimization | Accelerates CHW image normalization and CTC argmax decoding |
| `cuda` | No | Execution provider | Enables the ONNX Runtime CUDA provider for NVIDIA GPUs |
| `tensorrt` | No | Execution provider | Enables the ONNX Runtime TensorRT provider |
| `directml` | No | Execution provider | Enables the DirectML provider on Windows |
| `coreml` | No | Execution provider | Enables the Core ML provider on macOS and iOS |
| `webgpu` | No | Execution provider | Enables the WebGPU provider on supported ONNX Runtime targets |
| `openvino` | No | Execution provider | Enables the Intel OpenVINO provider |
| `download-binaries` | Yes | Build setup | Downloads a compatible ONNX Runtime distribution during the build |
| `auto-download` | No | Model management | Downloads registered OCR model files from ModelScope when first used |

## Default Configuration

The standard dependency declaration enables `download-binaries` and `simd`:

```bash
cargo add oar-ocr
```

This configuration uses the CPU execution provider unless an accelerator is enabled and selected through `OrtSessionConfig`. It also downloads the ONNX Runtime binaries required for linking and enables the optimized CPU preprocessing kernels.

Disable both defaults with:

```bash
cargo add oar-ocr --no-default-features
```

With default features disabled, CPU preprocessing uses scalar fallbacks. An ONNX Runtime installation must also be supplied before linking. See [`download-binaries`](#download-binaries) for the system-runtime setup.

## CPU Optimization

### `simd`

`simd` accelerates two hot CPU paths in `oar-ocr-core`:

- Per-pixel CHW image normalization
- Per-timestep CTC argmax decoding

The implementation uses `wide` and `multiversion` to select an available instruction set at runtime. Supported targets can use AVX2, SSE4.2, or NEON without compiling with `target-cpu=native`. The SIMD and scalar paths are tested for identical output.

This feature improves CPU pre-processing and post-processing only. It does not change the ONNX Runtime execution provider.

To keep automatic ONNX Runtime setup while disabling SIMD:

```bash
cargo add oar-ocr --no-default-features --features download-binaries
```

## Execution Provider Features

The execution provider features make their corresponding ONNX Runtime providers available to the crate. Enabling a feature does not select that provider automatically. Configure the provider explicitly and place the CPU provider last when a fallback is desired.

```rust
use oar_ocr::core::config::{OrtExecutionProvider, OrtSessionConfig};

let ort_config = OrtSessionConfig::new().with_execution_providers(vec![
    OrtExecutionProvider::CUDA {
        device_id: Some(0),
        gpu_mem_limit: None,
        arena_extend_strategy: None,
        cudnn_conv_algo_search: None,
        cudnn_conv_use_max_workspace: None,
    },
    OrtExecutionProvider::CPU,
]);
```

Pass the configuration to `OAROCRBuilder::ort_session` or `OARStructureBuilder::ort_session`. Requesting a provider without its matching Cargo feature returns a configuration error.

### `cuda`

Enables the NVIDIA CUDA execution provider. It is intended for supported Linux and Windows targets and requires a compatible NVIDIA driver, CUDA runtime, and cuDNN installation.

```bash
cargo add oar-ocr --features cuda
```

The CUDA provider supports device selection, a GPU memory limit, arena growth strategy, cuDNN convolution algorithm selection, and maximum-workspace control through `OrtExecutionProvider::CUDA`.

### `tensorrt`

Enables the NVIDIA TensorRT execution provider on supported Linux and Windows targets. The host must provide compatible TensorRT and CUDA runtimes.

TensorRT can fall back directly to CPU. For the usual TensorRT to CUDA to CPU provider chain, enable both accelerator features:

```bash
cargo add oar-ocr --features tensorrt,cuda
```

`OrtExecutionProvider::TensorRT` exposes workspace, FP16, timing-cache, engine-cache, and embedded-engine context options.

### `directml`

Enables the DirectML execution provider for Windows devices supported by DirectX 12. Select a device with the `device_id` field on `OrtExecutionProvider::DirectML`.

```bash
cargo add oar-ocr --features directml
```

### `coreml`

Enables the Core ML execution provider on macOS and iOS. Its configuration can prefer the Apple Neural Engine and can enable subgraph execution.

```bash
cargo add oar-ocr --features coreml
```

This feature controls the ONNX Runtime Core ML provider used by the classic pipeline. The `metal` feature in `oar-ocr-vl` is separate.

### `webgpu`

Enables the ONNX Runtime WebGPU execution provider. Target and binary support depends on the ONNX Runtime distribution used for the build.

```bash
cargo add oar-ocr --features webgpu
```

Select it with `OrtExecutionProvider::WebGPU`.

### `openvino`

Enables the Intel OpenVINO execution provider on supported Linux and Windows targets. `OrtExecutionProvider::OpenVINO` accepts a device type and an optional thread count.

```bash
cargo add oar-ocr --features openvino
```

## Runtime and Model Downloads

### `download-binaries`

`download-binaries` asks the pinned `ort` dependency to fetch a compatible ONNX Runtime distribution during the build. It is enabled by default and uses the platform certificate store for TLS.

This feature supplies the inference runtime. It does not download OCR model files.

For offline, enterprise, or custom ONNX Runtime builds, disable the default features and point `ORT_LIB_LOCATION` at the directory containing the ONNX Runtime libraries:

```bash
ORT_LIB_LOCATION=/opt/onnxruntime/lib \
    cargo build --no-default-features --features simd
```

If an execution provider feature is enabled, the supplied ONNX Runtime build must contain that provider and its runtime dependencies. Prebuilt availability also varies by target and provider combination.

### `auto-download`

`auto-download` fetches registered OCR model files from [`greatv/oar-ocr` on ModelScope](https://www.modelscope.cn/models/greatv/oar-ocr) when a builder receives a missing bare model name. Files are cached under `$OAR_HOME`, which defaults to `~/.oar`, and are verified against the bundled size and SHA-256 registry.

```bash
cargo add oar-ocr --features auto-download
```

Existing files and explicit paths remain under caller control. In-memory ONNX sources also bypass the download resolver. See the [model guide](models.md#auto-download-via-the-auto-download-feature) for the complete path-resolution and cache rules.

`auto-download` supplies model files at runtime. It does not install ONNX Runtime or any accelerator dependencies.

## Recommended Combinations

| Use case | Features |
|---|---|
| Standard CPU inference | Defaults only |
| CPU inference with automatic model downloads | `auto-download` |
| NVIDIA CUDA | `cuda` |
| NVIDIA CUDA with automatic model downloads | `cuda,auto-download` |
| NVIDIA TensorRT with CUDA and CPU fallbacks | `tensorrt,cuda` |
| Windows DirectML | `directml` |
| Apple Core ML | `coreml` |
| Intel OpenVINO | `openvino` |
| WebGPU on a supported target | `webgpu` |

Features listed in this table are added to the two default features unless `--no-default-features` is supplied.

## Selection Rules and Common Pitfalls

- Cargo features are additive. Another dependency in the graph can re-enable a feature that was disabled locally.
- CPU execution is always available and does not have a dedicated feature.
- Execution provider order matters. ONNX Runtime tries providers in the order supplied to `OrtSessionConfig`.
- Feature flags compile provider support but do not install GPU drivers, TensorRT, DirectML, Core ML, OpenVINO, or other system runtimes.
- Avoid `--all-features` for normal builds. Several providers are platform-specific and some provider combinations have no matching prebuilt ONNX Runtime distribution.
- The `oar-ocr-vl` crate has its own `cuda`, `metal`, and `download-binaries` features. They are independent of the root-crate features documented here.

Inspect the resolved feature graph with:

```bash
cargo tree -e features -i oar-ocr
```
