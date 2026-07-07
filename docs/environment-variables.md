# Environment Variables

Runtime environment variables read by the oar-ocr crates. Project-specific
variables use the `OAR_` prefix (`OAR_VL_` for the Vision-Language crate).

## Summary

| Variable | Crate | Default | Purpose |
|---|---|---|---|
| [`OAR_HOME`](#oar_home) | `oar-ocr-core` | `~/.oar` | Model cache directory for auto-download |
| [`OAR_VL_DTYPE`](#oar_vl_dtype) | `oar-ocr-vl` | auto | Compute dtype for VL models |
| [`OAR_VL_ATTN_FULL_SEQ_THRESHOLD`](#oar_vl_attn_full_seq_threshold) | `oar-ocr-vl` | `8192` | PaddleOCR-VL vision attention path selection |
| [`CUDA_LAUNCH_BLOCKING`](#cuda_launch_blocking) | `oar-ocr-core` | set to `1` if unset | ONNX Runtime CUDA workaround |
| [`RUST_LOG`](#rust_log) | examples | `info` | Log filter |

## `OAR_HOME`

Directory where the `auto-download` feature caches model files
(default `~/.oar`). Bare model file names passed to the builders are
downloaded here and verified against their expected SHA-256. See
[models.md](models.md#auto-download-via-the-auto-download-feature) for the
exact path resolution rules.

```bash
OAR_HOME=/data/oar-models cargo run --release --example ocr -- doc.jpg
```

## `OAR_VL_DTYPE`

Overrides the automatic weight/compute dtype selection for Vision-Language
models (PaddleOCR-VL, GLM-OCR, HunyuanOCR, MinerU2.5).

Accepted values (case-insensitive):

- `bf16` (alias `bfloat16`)
- `f16` (aliases `fp16`, `float16`, `half`)
- `f32` (aliases `fp32`, `float32`)

Without the override, GPUs use BF16 when a runtime probe confirms kernel
support (CUDA compute capability >= 8.0), falling back to F16 otherwise
(e.g. GTX 10xx/16xx, RTX 20xx); CPU always uses F32.

```bash
OAR_VL_DTYPE=f16 cargo run --release --features cuda -p oar-ocr-vl --example paddleocr_vl -- ...
```

## `OAR_VL_ATTN_FULL_SEQ_THRESHOLD`

Maximum sequence length (in vision patches) for which PaddleOCR-VL's vision
attention may use the single-shot full-matrix path; longer sequences use the
numerically equivalent chunked path, which needs far less peak memory.
Default `8192`; set `0` to force chunked attention.

Independently of this threshold, the full path is skipped automatically
whenever its softmax scratch would not fit in the free device memory, so
low-VRAM GPUs normally don't need this variable.

```bash
OAR_VL_ATTN_FULL_SEQ_THRESHOLD=0 cargo run --release --features cuda -p oar-ocr-vl --example paddleocr_vl -- ...
```

## `CUDA_LAUNCH_BLOCKING`

Standard CUDA variable, listed here because the classic (ONNX Runtime)
pipeline sets it to `1` at startup when it is not already set, to work around
[onnxruntime#4829](https://github.com/microsoft/onnxruntime/issues/4829)
(PP-FormulaNet's autoregressive `Loop` corrupting CUDA-EP arena buffers).
Preset any value (e.g. `0`) to keep your own setting. Note this serializes
CUDA kernel launches process-wide.

## `RUST_LOG`

Standard [`tracing_subscriber::EnvFilter`](https://docs.rs/tracing-subscriber)
filter used by the examples, e.g. `RUST_LOG=oar_ocr_vl=debug` to see why the
vision attention picked the chunked path.
