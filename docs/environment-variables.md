# Environment Variables

Runtime environment variables read by the oar-ocr crates and their build scripts.

## `OAR_HOME`

Directory where the `auto-download` feature caches model files (default `~/.oar`). Bare model file names passed to the builders are downloaded here and verified against their expected SHA-256. VL repository snapshots use the same cache root. See [models.md](models.md#auto-download) for classic model paths and the [VL README](../oar-ocr-vl/README.md) for checkpoint downloads.

```bash
OAR_HOME=/data/oar-models cargo run --release --example ocr -- doc.jpg
```

## VL runtime options

The VL crate reads no runtime environment variables. Select the compute dtype with `RuntimeConfig::with_dtype`, and switch a model to eager decoding with its `with_cuda_graphs(false)` builder. GLM-OCR's MTP head is on by default on CUDA (`GlmOcrLoadOptions::default().with_mtp(false)` turns it off); jina-ocr-v1's FastMTP is off by default (`JinaOcrLoadOptions::default().with_mtp(true)` turns it on). See the [VL README](../oar-ocr-vl/README.md) for details.

## CUDA Build Overrides

With the `cuda` feature enabled, `oar-ocr-vl` compiles its custom kernels, including OvisOCR2 Gated DeltaNet, to PTX. `CUDA_COMPUTE_CAP` overrides automatic GPU detection and accepts values such as `89`, `8.9`, `sm_89`, or `compute_89`. Compute capability 8.0 or newer is required. `NVCC` overrides the CUDA compiler executable.

```bash
CUDA_COMPUTE_CAP=89 NVCC=/usr/local/cuda/bin/nvcc \
    cargo build -p oar-ocr-vl --features cuda
```

## `CUDA_LAUNCH_BLOCKING`

Standard CUDA variable, listed here because the classic (ONNX Runtime) pipeline sets it to `1` at startup when it is not already set, to work around [onnxruntime#4829](https://github.com/microsoft/onnxruntime/issues/4829) (PP-FormulaNet's autoregressive `Loop` corrupting CUDA-EP arena buffers). Preset any value (e.g. `0`) to keep your own setting. Note this serializes CUDA kernel launches process-wide.

## `RUST_LOG`

Standard [`tracing_subscriber::EnvFilter`](https://docs.rs/tracing-subscriber) filter used by the examples, e.g. `RUST_LOG=oar_ocr_vl=debug`.
