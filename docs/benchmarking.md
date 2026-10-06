# End-to-End Benchmarking

`oar-bench` (the unpublished `oar-ocr-bench` workspace crate) times complete OCR,
structure, and VL page parsing on a fixed set of page images, and compares two
runs. Run it from the repository root.

## Run

Put the pages to measure in the gitignored `benchmark-inputs/` directory, or pass
image files and directories with repeated `--input` flags. Then:

```bash
cargo run --release -p oar-ocr-bench --features cuda,nvml --bin oar-bench -- \
  run --output benchmark-results/base.json
```

- `--manifest` selects the case list (default
  [`oar-ocr-bench/manifests/default.toml`](../oar-ocr-bench/manifests/default.toml),
  one case per model architecture on `cuda:0`).
- `--case <NAME>` (repeatable) runs a subset; `--device auto|cpu|cuda:N|metal`
  overrides every case's device.
- Classic model names are auto-download registry names; VL checkpoints are local
  directories under `models/<org>/<name>`. Download them beforehand so loading
  time does not include network transfers.

Each case runs in a fresh subprocess, so memory peaks are per case and GPU
memory is released between cases. The run prints a Markdown table and writes a
JSON report; it exits nonzero if any case failed.

An explicitly requested accelerator that cannot be initialized fails the case
instead of silently measuring a CPU fallback. `auto` records the device it
actually selected.

## Manifest

```toml
[inputs]
image_dirs = ["benchmark-inputs"]  # or images = ["page.png"]; optional max_pages

[defaults]
device = "cuda:0"
warmup = 2
repetitions = 5

[defaults.options]
cpu_threads = 4          # ORT intra-op threads and the Rayon pool
# batch_size = 1         # classic image batch
# region_batch_size = 4  # classic region batch; external-layout VL and MinerU
# max_tokens = 4096      # VL generation budget

[[cases]]
name = "ocr-tiny"
kind = "ocr"              # ocr | structure | vl
[cases.models]
detector = "pp-ocrv6_tiny_det.onnx"
recognizer = "pp-ocrv6_tiny_rec.onnx"
dictionary = "ppocrv6_tiny_dict.txt"
```

Cases may override `device`, `warmup`, `repetitions`, and `[cases.options]`.
`structure` cases take a `layout` model (with `layout_name` such as
`PP-DocLayoutV3`), optional OCR models, and optional table models. `vl` cases take
`model` and `model_path`; `paddleocr-vl`, `glmocr`, and `teleocr` also need a
PP-DocLayout `layout_path`.

## Measurements

| Field | Meaning |
|---|---|
| `load_ms` | Device setup and pipeline construction |
| `latency_ms` | Mean, p50, and p95 per page over all repetitions; batched pages share the batch time |
| `pages_per_second` | Measured pages divided by measured inference time |
| `output_chars_per_second` | VL only; PageParser exposes no generated-token count |
| `host_peak_bytes` | Linux `VmHWM` of the case process |
| `gpu` | With `nvml`: device-wide used memory before loading and its sampled peak |

Warmup runs are excluded. GPU memory is device-wide and sampled every 10 ms, so
use an otherwise idle GPU and treat the peak as a lower bound.

## Compare

```bash
cargo run --release -p oar-ocr-bench --bin oar-bench -- \
  compare benchmark-results/base.json benchmark-results/new.json --threshold 5%
```

For each case, latency, throughput, and memory changes worse than the threshold
are reported as regressions. Cases whose input pages or actual devices differ are
reported instead of compared. Environment differences (commit, CPU, features,
build profile) are printed as a note. The command exits nonzero on any regression,
missing case, or incomparable case.
