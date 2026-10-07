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

# Earlier PP-OCR versions, e.g. PP-OCRv5 mobile
oar ocr --det pp-ocrv5_mobile_det.onnx --rec pp-ocrv5_mobile_rec.onnx --dict ppocrv5_dict.txt page.png
```

OCR uses the PP-OCRv6 builder preset, defaulting to Tiny for fast initial downloads, low memory use, and responsive CPU inference. Select another preset size with `--size small` or `--size medium`. Custom models require `--det`, `--rec`, and `--dict` together, accepting local files or registered names; they cannot be combined with `--size` and use generic detection defaults (the PaddleX values for PP-OCRv4/v5) rather than the PP-OCRv6 preset's tuned thresholds. Structure uses the PP-StructureV3 builder preset with PP-DocLayoutV3, PP-OCRv6 Tiny, and wired/wireless table models; the same three custom OCR options replace its OCR combination while retaining the layout and table preset.

`oar ocr` and `oar structure` accept a comma-separated `--device` list such as `cuda:0,cuda:1` and run one pipeline replica per entry, splitting page batches across them while keeping output in input order; a repeated entry like `cuda:0,cuda:0` runs two replicas on one GPU. Results are identical to a single device. `oar parse` uses a single device.

Use `--format json` for structured results. Structure and parse emit the shared [page JSON format](https://github.com/GreatV/oar-ocr/blob/main/docs/page-format.md), with pixel dimensions from the rendered page image. `page.index` is the 0-based position within the input file: images use `0`, and PDF page N uses `N-1`, including when selecting a subset of pages. The only CLI-specific addition to that format is top-level `source`, containing the input path. OCR keeps its text-region JSON with pixel coordinates and confidence scores. Text output goes to stdout without headers. `-o/--output DIR` instead writes `<image-stem>.md` or `.json` per image, and refuses to overwrite existing files. Multiple images on JSON stdout produce an array; a single image produces an object. Logs go to stderr at warn level; `-v` enables info progress.

`oar parse --list-models` lists the supported Hugging Face IDs without downloading anything. IDs are explicit, and downloading from ModelScope does not change their spelling. Use `--source huggingface` to choose that source, or `--model-dir DIR` to load a local checkpoint; local PaddleOCR-VL, GLM-OCR, and TeleOCR checkpoints also require `--layout-dir DIR`. Remote loading automatically downloads the required layout checkpoint unless `--layout-dir` is supplied. `--max-tokens N` limits VL generation and can produce partial output with warnings.

Inputs can be individual images or PDFs for all three commands. PDF detection uses the `%PDF-` content signature, regardless of the filename extension. Pages are rendered on a white background as needed: OCR and structure process batches of at most eight pages across inputs, while VL parsing processes one page at a time. Each batch is written before rendering the next, keeping rendered-image memory bounded. `--pages 1-3,5` selects 1-based PDF page numbers in document order, removing duplicates; without it, every page is processed. The selection applies to each PDF, and requesting a page beyond its page count is an error. `--dpi N` controls PDF rendering resolution; the default 144 DPI matches the examples' 2× scale. Both options leave image inputs unchanged. Directory recursion and service modes are not supported.

With `-o DIR`, PDFs produce `<stem>_p<N>.md` or `.json` for each selected page, retaining the original 1-based page numbers. Text and Markdown on stdout include `<!-- filename: page N -->` markers. PDF JSON on stdout is always an array of per-page objects, even when only one page is selected. Structure and parse use the page format's `page.index` plus `source`; OCR PDF results keep their existing `page_number` and `input_path` fields. Outputs are written incrementally; a later page error may leave earlier files or partial stdout output.

```bash
oar ocr document.pdf --pages 1-3,5 --dpi 144 -o pages
oar structure document.pdf --format json
oar parse --model PaddlePaddle/PaddleOCR-VL-1.5 document.pdf --pages 2
```
