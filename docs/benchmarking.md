# End-to-End Benchmarking

`oar-bench` measures complete OCR, structure, and VL page parsing runs. It is an
unpublished workspace binary, separate from the library's Criterion microbenchmarks.
Run commands from the repository root. It currently accepts explicit `cpu`,
`cuda:N`, and `metal` devices. For classic pipelines, `metal` selects CoreML;
for VL it selects Candle Metal. Accelerator cases require the matching feature
and platform. An unavailable explicitly requested accelerator fails the case,
rather than producing a CPU number labelled as GPU performance.

## Manifest

The supplied [manifest](../oar-ocr-bench/manifests/default.toml) includes classic
PP-OCRv6 and layout/table/OCR structure cases, all nine model-native PageParser
families, and external-layout PaddleOCR-VL, GLM-OCR, and TeleOCR cases. Classic
bare model filenames use the existing auto-download registry; populate the cache
before collecting baselines to exclude network downloads from loading time.
VL checkpoints are local directories under `models/<org>/<name>`.

A minimal CPU manifest is:

```toml
[inputs]
images = [".oar/images/general_ocr_001.png"]
# image_dirs = ["images"]       # recursive, sorted, image files only
# pdfs = ["documents/paper.pdf"]
# pdf_scale = 2.0               # pixels per PDF point, (0, 4]
max_pages = 1

[defaults]
device = "cpu"
warmup = 2
repetitions = 5

[defaults.options]
cpu_threads = 4

[[cases]]
name = "ocr-tiny"
kind = "ocr"
[cases.models]
detector = "pp-ocrv6_tiny_det.onnx"
recognizer = "pp-ocrv6_tiny_rec.onnx"
dictionary = "ppocrv6_tiny_dict.txt"
[cases.options]
batch_size = 1
region_batch_size = 4
```

Paths are relative to `--root` (default `.`); registered classic filenames remain
registry names. `images`, `image_dirs`, and `pdfs` can be combined. Files are
sorted and deduplicated; PDF pages follow document order. Every child decodes or
renders its own input set into resident RGB pages before loading models. Limit
large datasets with `max_pages`: the host peak includes resident inputs.

Cases override `device`, `warmup`, `repetitions`, and `[cases.options]` independently
of `[defaults]` and `[defaults.options]`. Case names must be unique. Unknown keys,
zero repetitions/batch sizes, and incomplete model configurations are errors.

- `ocr`: requires `models.detector`, `recognizer`, and `dictionary`.
- `structure`: requires `models.layout`; set `layout_name`, such as
  `PP-DocLayoutV3`, to identify its architecture. All three OCR models enable OCR.
  Optional `table_classifier`, `wired_table_structure`, `wireless_table_structure`,
  `wired_table_cells`, and `wireless_table_cells` add table processing; table
  structure models require `table_dictionary`. Optional `formula` also needs
  `formula_tokenizer` and `formula_type` (`pp_formulanet` or `unimernet`).
- `vl`: requires `model` and `model_path`. Model keys are `hpd-parsing`,
  `hunyuanocr`, `jina-ocr`, `mineru`, `mineru-diffusion`, `monkeyocrv2`, `ovisocr2`,
  `wevisdoc`, `xiaomi-ocr-0`, `paddleocr-vl`, `paddleocr-vl-1.5`, `paddleocr-vl-1.6`,
  `glmocr`, and `teleocr`. The last five require `layout_path` for PP-DocLayout.
  Versions sharing a loader are selected by their checkpoint directory.

`batch_size` controls classic image batches (default 1). VL PageParser processes
one page per call, so its `batch_size` must be 1. `region_batch_size` overrides
classic region batching and external-layout/MinerU scheduling; other native
PageParsers do not expose region batches. `max_tokens` overrides the VL generation
budget (MinerU-Diffusion uses it as `gen_length`); absent values retain model
defaults. `use_mtp` is HPD-specific; `diffusion_seed` is MinerU-Diffusion-specific.
`cpu_threads` (default 4) configures ORT and the child Rayon pool. Existing library
runtime behavior and model defaults apply; no new environment switches are needed.

## Run

```bash
CARGO_BUILD_JOBS=8 nice -n 10 cargo run --release -p oar-ocr-bench --bin oar-bench -- \
  run --manifest oar-ocr-bench/manifests/default.toml \
  --case ocr-tiny --device cpu --output benchmark-results/cpu-base.json
```

Repeat `--case` to select several cases; omit it to run all cases. `--device`
overrides the manifest's device for the run. Every case starts a fresh `run-case`
subprocess and exits before the next case starts, releasing model allocations and
GPU contexts. The parent writes one JSON report and prints a Markdown table.
An existing output filename is rejected; without `--output`, a timestamped report
is created under `benchmark-results/`. Failed cases remain in the report, and the
run exits nonzero if any case failed, was invalid, or had unstable outputs.

Use the same inputs, weights, model settings, build profile, and hardware for
comparisons. Run a release build; populate model caches and allow filesystem
caches to settle before recording baselines. Keep the machine otherwise idle.
Loading time is measured once per isolated process and includes model reads,
initialization, accelerator startup, and any uncached classic model downloads.
It is separate from the warmup and inference samples.

## Measurements

- Warmup runs cover the whole input set and are excluded from latency statistics.
  Each measured repetition covers the whole input set. Inference timing includes
  pipeline preprocessing, recognition, output rendering, and owned-input clones.
  VL calls end with a device synchronization. Decoding images, rendering PDFs,
  computing fingerprints, and writing JSON are outside inference timing.
- Classic batches report **amortized batch wall time per page**, not the response
  time of an individual page within a batch. `pages/s` is measured page count
  divided by total measured inference time. Percentiles use linear interpolation
  at `(n - 1) * p`; mean/p50/p95/min/max are in milliseconds.
- PageParser returns decoded documents without a common generated-token count.
  VL reports **Unicode output characters/s**, with `tokens_per_second: null` and
  an explicit `rate_basis`. It does not estimate tokens by re-tokenizing output.
  Markdown is preferred, followed by recognized block content, then raw protocol.
- Each page/repetition records the SHA-256 of decoded input pixels and rendered
  text. VL also hashes the complete PageDocument to catch layout/protocol changes.
  Non-fatal parser diagnostics invalidate the case; output variations across
  repetitions are flagged separately. These measurements do not score OCR accuracy.
- Linux host peak memory is `/proc/self/status`'s `VmHWM` in bytes for the entire
  child, including inputs, model loading, warmup, and measurement. Other platforms
  report `null`. The parent process is excluded.
- Environment metadata records runtime and build-time git revision/dirty state,
  build-time rustc version, release/debug profile, enabled harness features, CPU
  model, OS/architecture, timestamp, raw manifest and SHA-256, effective case
  configuration, and measured GPU names/UUIDs/drivers. CPU-only or unavailable
  NVML GPU metadata is `null`. Build and runtime revisions reveal stale binaries.

## GPU Memory and Isolation

`nvml` is optional and dynamically loads NVML. Missing libraries or unavailable
devices produce `gpu: null` plus a warning, without failing an otherwise valid
inference run. GPU memory/isolation is then **unverified**. Metal has no NVML
measurement.

For CUDA cases, set `nvml_device` to a physical GPU UUID (`GPU-...`) or `index:N`
and `nvml_interval_ms` (default 10) in options. NVML indices are not guaranteed to
match CUDA ordinals, particularly when visibility/order is changed. Choose the
selector matching the requested CUDA device; the report records its UUID.
The supplied manifest targets CUDA(0) and NVML index 0 for a single-GPU machine.

Sampling starts before model loading and includes warmup and inference. It
records the initial device-wide memory baseline, sampled peak, and
`peak - baseline` (saturating at zero), in bytes. This is a **sampled lower bound**,
not an allocator high-water mark; short-lived peaks between samples can be missed.
The final sample is taken before the model is dropped.

Use an **exclusive GPU window**. The sampler checks both compute and graphics
processes and accumulates every PID other than the worker's. A co-tenant, any
sampling/process-query error, or never observing the worker on the selected GPU
marks the measurement invalid. The report retains the values and reasons, but
compare does not treat their deltas as valid performance comparisons. Display
servers also count as co-tenants; an idle/headless GPU is preferable. Sampling
cannot detect processes that appear and disappear between polls.

Run the full GPU baseline after the weights/cache are ready:

```bash
CARGO_BUILD_JOBS=8 CUDAFORGE_THREADS=4 nice -n 10 cargo run --release \
  -p oar-ocr-bench --features cuda,nvml --bin oar-bench -- \
  run --manifest oar-ocr-bench/manifests/default.toml \
  --output benchmark-results/gpu-base.json
```

On macOS, use `--features metal` and `--device metal`; GPU memory fields remain
`null` and process isolation still comes from one child per case.

## Compare

```bash
CARGO_BUILD_JOBS=8 nice -n 10 cargo run --release -p oar-ocr-bench --bin oar-bench -- \
  compare benchmark-results/cpu-base.json benchmark-results/cpu-new.json --threshold 5%
```

The comparison table shows each case/metric's base, new value, percentage delta,
and status. Increasing latency/load/peak memory and decreasing throughput are
regressions only when the change **exceeds** the threshold (default 5%). GPU
baseline deltas are context, not regressions. Unavailable metrics stay marked
unavailable. Different repetition counts are allowed, but warmup, decoding,
batching, input fingerprints, and device configurations must match. Hardware or
build-profile mismatches, added/removed/failed cases, invalid samples, unstable
outputs, and changed output fingerprints are flagged and return exit code 1,
as do numerical regressions. Input/JSON/schema/CLI errors return exit code 2.
Git revisions may differ as expected when comparing implementations; dirty/build
revision metadata remains available for review.

The default workspace clippy/test commands compile this binary and run its
manifest, input ordering, statistics, memory-accumulator, and comparison unit tests.
They use no model weights or GPU. Optional `nvml` can also be checked and tested
independently without initializing hardware.
