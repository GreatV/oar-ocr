# End-to-End Benchmarking

`oar-bench` (the unpublished `oar-ocr-bench` workspace crate) times complete OCR, structure, and VL page parsing on a fixed set of page images, and compares two runs. Run it from the repository root.

## Run

Put the pages to measure in the gitignored `benchmark-inputs/` directory, or pass image files and directories (top-level images only) with repeated `--input` flags. Then:

```bash
cargo run --release -p oar-ocr-bench --features cuda,nvml --bin oar-bench -- \
  run --output benchmark-results/base.json
```

- `--manifest` selects the case list (default [`oar-ocr-bench/manifests/default.toml`](../oar-ocr-bench/manifests/default.toml), one case per model architecture on `cuda:0`).
- `--case <NAME>` (repeatable) runs a subset; `--device auto|cpu|cuda:N|metal` overrides every case's device.
- Classic model names are auto-download registry names; VL checkpoints are local directories under `models/<org>/<name>`. Download them beforehand so loading time does not include network transfers.

Each case runs in a fresh subprocess, so memory peaks are per case and GPU memory is released between cases. The run prints a Markdown table and writes a JSON report; it exits nonzero if any case failed.

An explicitly requested accelerator that cannot be initialized fails the case instead of silently measuring a CPU fallback. `auto` records the device it actually selected.

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
# batch_size = 1         # pages per pipeline call; the library picks its own image batch
# region_batch_size = 4  # classic region batch; external-layout VL and MinerU
# max_tokens = 4096      # VL generation budget
# gpu_memory_budget = 4294967296  # classic GPU tuning hint in bytes

[[cases]]
name = "ocr-tiny"
kind = "ocr"              # ocr | structure | vl
[cases.models]
detector = "pp-ocrv6_tiny_det.onnx"
recognizer = "pp-ocrv6_tiny_rec.onnx"
dictionary = "ppocrv6_tiny_dict.txt"
```

Cases may override `device`, `warmup`, `repetitions`, and `[cases.options]`. `structure` cases take a `layout` model (with `layout_name` such as `PP-DocLayoutV3`), optional OCR models, and optional table models. `vl` cases require `model`, the checkpoint's Hugging Face repo ID (for example `PaddlePaddle/PaddleOCR-VL-1.5`), plus the local `model_path`; layout-composed models such as PaddleOCR-VL, GLM-OCR, and TeleOCR also need a PP-DocLayout `layout_path`. The ID never comes from the directory: most models are fine-tunes whose configs match their public base models, so the ID says what the directory holds.

## Official accuracy evaluation

Add `run --save-outputs predictions` to write the last measured repeat's text to `predictions/<case>/<image-stem>.md`, outside inference timing. VL saves the text used for chars/s, OCR saves recognized text, and structure saves Markdown. Use a fresh directory per run: a non-empty case directory is rejected, as are duplicate image stems within a case.

- [OmniDocBench](https://github.com/opendatalab/OmniDocBench): use matching v1.5 annotations and images with the `v1_5` evaluation branch. Each image needs a same-stem `.md`. Set the end2end config's `ground_truth.data_path` to the annotation JSON and `prediction.data_path` to `predictions/<case>`, then run the official `python pdf_validation.py --config configs/end2end.yaml`.
- [olmOCR-Bench](https://github.com/allenai/olmocr): render PDF pages to images first, naming them `<pdf-stem>_pg<1-based-page>_repeat1.png`. The scorer requires `bench_data/<method>/<pdf-relative-stem>_pg<page>_repeat1.md`; arrange saved files into the PDF's relative category directories before running the official `python -m olmocr.bench.benchmark --dir bench_data`. Bench saves flat case directories and does not reconstruct those category paths or rename pages.

## Measurements

| Field | Meaning |
|---|---|
| `load_ms` | Device setup and pipeline construction |
| `latency_ms` | Mean, p50, and p95 per page over all repetitions; a batched page takes the whole batch time |
| `pages_per_second` | Measured pages divided by measured inference time |
| `output_chars_per_second` | VL only; PageParser exposes no generated-token count |
| `host_peak_bytes` | Linux `VmHWM` of the case process |
| `gpu` | With `nvml`: device-wide used memory before loading and its sampled peak |

Warmup runs are excluded. A VL page with parser diagnostics, such as an exhausted token budget or a failed region, fails the case rather than being timed. GPU memory is device-wide and sampled every 10 ms, so use an otherwise idle GPU and treat the peak as a lower bound. The sampled GPU honors `CUDA_VISIBLE_DEVICES`; on multi-GPU hosts also set `CUDA_DEVICE_ORDER=PCI_BUS_ID` so CUDA ordinals follow NVML's order.

## Compare

```bash
cargo run --release -p oar-ocr-bench --bin oar-bench -- \
  compare benchmark-results/base.json benchmark-results/new.json --threshold 5%
```

For each case, latency and throughput (pages/s, and chars/s for VL) changes worse than the threshold are reported as regressions; memory peaks vary between identical runs and are shown as information only. Cases are matched by name, and inputs by page path, so keep the page images unchanged between the two runs. Cases whose configuration (other than device, warmup, and repetitions), pages, or actual devices differ are reported instead of compared; with `nvml`, the GPU model is part of the device, so the same `cuda:0` on different GPUs is not compared. Without `nvml`, compare runs from the same machine. Environment differences (commit, CPU, features, build profile, `OAR_*` overrides) are printed as a note. The command exits nonzero on any regression, a case missing or failed in either report, or an incomparable case.
