# OAR command-line tool

`oar` provides OCR, document structure analysis, and vision-language page parsing. Missing classic and VL models download automatically from ModelScope by default. The default device is `auto`, selecting compiled accelerators before CPU.

## Install

```bash
cargo install oar-ocr-cli
cargo install oar-ocr-cli --features cuda --force
```

On macOS, use `--features metal` for CoreML and Metal acceleration. Explicit accelerator requests need the corresponding feature; the default CPU installation works without a GPU.

## Use

```bash
oar ocr page.png
oar structure page.png -o documents
oar parse --model PaddlePaddle/PaddleOCR-VL-1.5 page.png -o parsed
```

OCR defaults to PP-OCRv6 Tiny for fast initial downloads, low memory use, and responsive CPU inference. `--det`, `--rec`, and `--dict` replace its model files or registered names. Structure uses PP-DocLayoutV3, the same OCR models, and the wired/wireless table models used by the benchmark's structure-v3 case.

Use `--format json` for structured results; OCR JSON includes text regions, pixel coordinates, and confidence scores. Text output goes to stdout without headers. `-o/--output DIR` instead writes `<image-stem>.md` or `.json` per image, and refuses to overwrite existing files. Multiple images on JSON stdout produce an array; a single image produces an object. Logs go to stderr at warn level; `-v` enables info progress.

`oar parse --list-models` lists the supported Hugging Face IDs without downloading anything. IDs are explicit, and downloading from ModelScope does not change their spelling. Use `--source huggingface` to choose that source, or `--model-dir DIR` to load a local checkpoint; local PaddleOCR-VL, GLM-OCR, and TeleOCR checkpoints also require `--layout-dir DIR`. Remote loading automatically downloads the required layout checkpoint unless `--layout-dir` is supplied. `--max-tokens N` limits VL generation and can produce partial output with warnings.

Inputs must be individual image files. Render PDFs to images first; directory recursion and service modes are not supported.
