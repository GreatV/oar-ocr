# OAR command-line tool

`oar` provides PP-OCR, Indra OCR, document structure analysis, and vision-language page parsing. Missing classic and VL models download automatically from ModelScope by default; Indra downloads directly from Hugging Face. The default device is `auto`, selecting compiled accelerators before CPU.

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

All four commands accept a comma-separated `--device` list such as `cuda:0,cuda:1` and run one pipeline replica per entry, splitting pages across them while keeping output in input order; a repeated entry like `cuda:0,cuda:0` runs two replicas on one GPU. For Indra, CPU and CUDA recognized text matched in testing, but near-tie reading-order or table-column decisions can differ on a few pages. Each `oar parse` replica loads its own copy of the VL model, so GPU memory grows with the number of replicas.

Use `--format json` for structured results. Structure, Indra, and parse emit the shared [page JSON format](https://github.com/GreatV/oar-ocr/blob/main/docs/page-format.md), with pixel dimensions from the rendered page image (Indra uses the upright page after orientation correction). `page.index` is the 0-based position within the input file: images use `0`, and PDF page N uses `N-1`, including when selecting a subset of pages. The only CLI-specific addition to that format is top-level `source`, containing the input path. OCR keeps its text-region JSON with pixel coordinates and confidence scores. Text output goes to stdout without headers. `-o/--output DIR` instead writes `<image-stem>.md` or `.json` per image, and refuses to overwrite existing files. Multiple images on JSON stdout produce an array; a single image produces an object. Logs go to stderr at warn level; `-v` enables info progress.

`oar parse --list-models` lists the supported Hugging Face IDs without downloading anything. IDs are explicit, and downloading from ModelScope does not change their spelling. Use `--source huggingface` to choose that source, or `--model-dir DIR` to load a local checkpoint; local PaddleOCR-VL, GLM-OCR, TeleOCR, and OXR checkpoints also require `--layout-dir DIR`. Remote loading automatically downloads the required layout checkpoint unless `--layout-dir` is supplied. `--max-tokens N` limits VL generation and can produce partial output with warnings.

`ATH-MaaS/OvisOCR3` selects the v3 Markdown page protocol without an external layout model. Structured parsing is available through the library's `OvisOcr3ParseOptions`; see the [OvisOCR3 guide](../oar-ocr-vl/README.md#ovisocr3).

Inputs can be individual images or PDFs for all four commands. PDF detection uses the `%PDF-` content signature, regardless of the filename extension. Pages are rendered on a white background as needed: OCR and structure process batches of at most eight pages across inputs, Indra defaults to max(8, 4 × workers) pages per replica, and VL parsing processes one page at a time. The ordered dispatch window bounds rendered-image memory across replicas. `--pages 1-3,5` selects 1-based PDF page numbers in document order, removing duplicates; without it, every page is processed. The selection applies to each PDF, and requesting a page beyond its page count is an error. `--dpi N` controls PDF rendering resolution; the default 144 DPI matches the examples' 2× scale. Both options leave image inputs unchanged. Directory recursion and service modes are not supported.

With `-o DIR`, PDFs produce `<stem>_p<N>.md` or `.json` for each selected page, retaining the original 1-based page numbers. Text and Markdown on stdout include `<!-- filename: page N -->` markers. PDF JSON on stdout is always an array of per-page objects, even when only one page is selected. Structure, Indra, and parse use the page format's `page.index` plus `source`; OCR PDF results keep their existing `page_number` and `input_path` fields. Outputs are written incrementally; a later page error may leave earlier files or partial stdout output.

```bash
oar ocr document.pdf --pages 1-3,5 --dpi 144 -o pages
oar structure document.pdf --format json
oar parse --model PaddlePaddle/PaddleOCR-VL-1.5 document.pdf --pages 2
```

## Indra OCR

```bash
oar indra page.png --device cpu
oar indra report.pdf --device cpu --workers 4 --batch-size 16 --pages 1-8 -o pages
oar indra page.png --device cuda:0 --format json
oar indra page.png --model-dir checkpoints/indra-ocr --device cpu --format text
```

Without `--model-dir`, Indra downloads `datalab-to/indra-ocr` at the pinned, parity-validated [revision `5993a956`](https://huggingface.co/datalab-to/indra-ocr/tree/5993a956d7ea1d43c0d40d409962bb0b033ac1e8) from Hugging Face into an immutable cached snapshot; it is not mirrored on ModelScope. Its weights use a modified OpenRAIL-M license. A local directory always takes precedence and is never filled in by downloading: it needs `meta.json`, `det.onnx`, `rec.onnx`, and `order.onnx`. Optional orientation, script recognizers, and language models are loaded when present. Builds without `auto-download` require `--model-dir`.

`--workers N` selects N CPU page workers per device-list entry, with separate model sessions and shared language models within each replica. ONNX thread counts use the library defaults: with multiple workers, unset intra/inter-op counts become one. CPU-only duplicate device entries multiply the worker count and memory use. Accelerator entries require `--workers 1`, including when `auto` selects an accelerator; use `--device cpu` for CPU workers. See the [Indra usage guide](../docs/usage.md#indraocrbuilder---indra-ocr) for measured throughput and memory use.

`--batch-size N` overrides the per-replica default of max(8, 4 × workers); batches smaller than the worker count can leave workers idle. Markdown is the default output, `--format text` joins lines in reading order, and `--format json` emits the shared page schema with `pipeline: "indra"`. Non-JSON files use the same `.md` extension as the other commands. Boxes and dimensions refer to the upright page.

`--keep-furniture` retains headers and footers in Markdown. `--table-heads` enables the optional reading-order table heads. `--math` reads math lines as inline LaTeX and additionally requires `math_meta.json`, `mathcls.onnx`, `math_enc.onnx`, `math_dec_init.onnx`, and `math_dec_step.onnx`. PDFs use the shared rasterizer, page selection, and DPI options.
