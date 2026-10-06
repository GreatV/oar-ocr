# oar-ocr-vl

Vision-Language models for document understanding in Rust.

This crate provides native Rust inference for document VLMs using [Candle](https://github.com/huggingface/candle), along with a document parsing pipeline for backends that work well with external layout detection.

## Supported Models

| Model | Parameters | Inference path |
|---|---:|---|
| [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | 0.9B | External-layout page parsing, text, table, and formula recognition |
| [HPD-Parsing](https://huggingface.co/PaddlePaddle/HPD-Parsing) | 1B | Model-native hierarchical full-page parsing with forked KV-prefix reuse and optional P-MTP |
| [HunyuanOCR 1.5 / 1.0](https://huggingface.co/tencent/HunyuanOCR) | 1B | Model-native prompt-driven parsing with optional DFlash decoding for 1.5 |
| [jina-ocr-v1](https://huggingface.co/jinaai/jina-ocr-v1) | 3B (570M active) | End-to-end page-to-Markdown parsing (SAM+CLIP DeepEncoder over a DeepSeek-V2 MoE decoder) |
| [MinerU-Diffusion-V1-0320](https://huggingface.co/opendatalab/MinerU-Diffusion-V1-0320-2.5B) | 2.5B | Block-diffusion OCR with two-step structured extraction or single-pass recognition |
| [MinerU2.5-2509](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B) | 1.2B | Model-native two-step layout detection and content extraction |
| [MinerU2.5-Pro-2605](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B) | 1.2B | Newer compatible checkpoint using the MinerU2.5 two-step pipeline |
| [MonkeyOCRv2-B-Parsing](https://huggingface.co/zenosai/MonkeyOCRv2-B-Parsing) | 0.7B | Higher-capacity ViT-B variant with the same parsing and recognition tasks |
| [MonkeyOCRv2-S-Parsing](https://huggingface.co/zenosai/MonkeyOCRv2-S-Parsing) | 0.6B | Model-native layout, end-to-end parsing, text, formula, and OTSL-table recognition |
| [OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) | 0.8B | Model-native full-page document-to-Markdown parsing |
| [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) | 0.9B | External-layout page parsing, text, table, formula, and chart recognition |
| [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) | 0.9B | PaddleOCR-VL tasks plus text spotting and seal recognition |
| [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | 0.9B | Region-aware refinement, drop-in compatible with the 1.5 loader |
| [TeleOCR](https://huggingface.co/XingChen-AGI/TeleOCR) | 1.2B | Qwen2.5-VL document parser with text, table (OTSL), formula, code, and layout tasks (formerly NaviDC-OCR) |
| [WeVisDoc-2B](https://huggingface.co/tencent/WeVisDoc-2B) / [4B](https://huggingface.co/tencent/WeVisDoc-4B) | 2B / 4B | Model-native full-page document-to-Markdown parsing (Qwen3-VL with DeepStack) |
| [Xiaomi-OCR-0](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0) | 0.8B | Model-native full-page document-to-Markdown parsing and text/table/formula/KIE region prompts (Qwen3.5) |
| [PP-DocLayoutV2](https://huggingface.co/PaddlePaddle/PP-DocLayoutV2_safetensors) / [V3](https://huggingface.co/PaddlePaddle/PP-DocLayoutV3_safetensors) | 54M / 33M | Layout detection and reading-order prediction, feeding `DocParser` |

See [`examples`](examples) for runnable examples.

## Document Parsing Pipeline

**DocParser** is a unified document parsing API for layout-first backends. It combines:

1. **Layout detection** to identify document regions and their reading order. `PpDocLayout` is a native Candle port of PP-DocLayoutV2/V3; any other detector can be plugged in through the `LayoutSource` trait.
2. **VL-based recognition** to extract content from each region

Use DocParser with PaddleOCR-VL, PaddleOCR-VL-1.5, PaddleOCR-VL-1.6, GLM-OCR, TeleOCR, jina-ocr-v1, MonkeyOCRv2, OvisOCR2, WeVisDoc, Xiaomi-OCR-0, HunyuanOCR, MinerU2.5/Pro, or MinerU-Diffusion for externally detected crops. HPD-Parsing currently supports only its model-native full-page protocol. For complete pages, prefer each model's native path where available: MonkeyOCRv2 `Layout`/`EndToEnd`, OvisOCR2, jina-ocr-v1, WeVisDoc, Xiaomi-OCR-0, and HPD-Parsing full-page parsing, HunyuanOCR full-page prompts, and the MinerU two-step extraction examples.

`LayoutPageParser` composes an owned layout source and recognition backend into `PageParser`, the same complete-page interface used by the model-native parsers:

```rust
use oar_ocr_vl::{
    DocParserConfig, LayoutPageParser, LayoutPageParserOptions, PageParser,
    PaddleOcrVl, PpDocLayout,
};
use oar_ocr_vl::utils::{image::load_image, parse_device};

let device = parse_device("cpu")?;
let layout = PpDocLayout::from_dir("PaddlePaddle/PP-DocLayoutV3_safetensors", device.clone())?;
let backend = PaddleOcrVl::from_dir("PaddlePaddle/PaddleOCR-VL-1.5", device)?;
let parser = LayoutPageParser::new(layout, backend).with_region_batch_size(2);
let image = load_image("document.jpg")?;
let page = parser.parse_page(&image, &LayoutPageParserOptions::default())?;
println!("{}", page.markdown.as_deref().unwrap_or_default());

// Override settings for one page; defaults use the parser's builder settings.
let options = LayoutPageParserOptions::default()
    .with_config(DocParserConfig { max_tokens: 8192, ..Default::default() })
    .with_region_batch_size(4);
let page = parser.parse_page(&image, &options)?;

// Use the structure entry point when pixel coordinates and metadata are needed.
let structure = parser.parse_structure(&image, &options)?;
```

Pass `&layout` and `&backend` instead to borrow existing models. `DocParser` remains available for callers that supply layout on every call and need `StructureResult`. The new page output includes normalized blocks, Markdown, and crop diagnostics. Use `LayoutPageParser::parse_structure` with the same options to obtain the original pixel coordinates, confidence, reading order, source metadata, tables, and formulas. Recognition failures still return an error, as in `DocParser`. The `doc_parser` example uses this new entry point.

## Unified Page Parsing

`AnyPageParser` wraps any of the crate's page parsers behind one `PageParser` implementation, so the model can be chosen at runtime — from a config file, CLI flag, or benchmark manifest — without per-model dispatch:

```rust
use oar_ocr_vl::{
    AnyPageParser, AnyPageParserOptions, LayoutPageParser, PageParser,
    PaddleOcrVl, PpDocLayout,
};

let parser = AnyPageParser::from(LayoutPageParser::new(layout, backend));
let page = parser.parse_page(&image, &AnyPageParserOptions::default())?;

// Override the shared knobs for one page; `None` knobs keep each model's own
// default and every model-specific option stays untouched.
let options = AnyPageParserOptions::default().with_max_new_tokens(8192);
let page = parser.parse_page(&image, &options)?;
```

Construct it from an already-loaded model with `From`: the model-native parsers (`HpdParsing`, `HunyuanOcr`, `JinaOcr`, `MinerU`, `MinerUDiffusion`, `MonkeyOcrV2`, `OvisOcr2`, `WeVisDoc`, `XiaomiOcr`) and `LayoutPageParser` compositions over `PpDocLayout` (`PaddleOcrVl`, `GlmOcr`, `TeleOcr`). `AnyPageParserOptions` carries the knobs every parser shares: `max_new_tokens` maps to each model's generation budget (including `MinerUDiffusion`'s `gen_length`), and `region_batch_size` applies to the region-batching parsers (`MinerU` and the layout-composed models); parsers without a matching concept ignore the knob.

`AnyPageParser::from_dir` loads a named parser from a model directory. The model is always named explicitly by its Hugging Face repo ID (`tencent/HunyuanOCR`, `PaddlePaddle/PaddleOCR-VL-1.5`, …) — most supported models are fine-tunes whose checkpoint configs match their public base models, so the directory cannot identify the model reliably on its own. Each model then loads through its own `from_dir` with its own defaults:

```rust
use candle_core::Device;
use oar_ocr_vl::{AnyPageParser, AnyPageParserLoadOptions, AnyPageParserModel};

// PaddleOCR-VL composes an external PP-DocLayout detector.
let options = AnyPageParserLoadOptions::default()
    .with_layout_dir("PaddlePaddle/PP-DocLayoutV3_safetensors");
let parser = AnyPageParser::from_dir_with_options(
    AnyPageParserModel::PaddleOcrVl1_5,
    "PaddlePaddle/PaddleOCR-VL-1.5",
    Device::Cpu,
    &options,
)?;

// Model-native parsers need no options; IDs also parse from strings
// (case-insensitively, like the Hub) for manifests and CLIs.
let parser = AnyPageParser::from_dir(AnyPageParserModel::HunyuanOcr, "tencent/HunyuanOCR", Device::Cpu)?;
let model: AnyPageParserModel = "tencent/hunyuanocr".parse()?;
```

The layout-composed models (PaddleOCR-VL, GLM-OCR, TeleOCR) require a PP-DocLayout directory in the load options; loading fails with a clear error when it is missing.

With the `auto-download` feature, `AnyPageParser::from_pretrained` downloads the checkpoint named by the model ID when it is not cached, then loads it through the same path:

```rust
use candle_core::Device;
use oar_ocr_vl::{
    AnyPageParser, AnyPageParserModel, AnyPageParserPretrainedOptions, DownloadSource,
};

# fn main() -> Result<(), Box<dyn std::error::Error>> {
let parser = AnyPageParser::from_pretrained(
    AnyPageParserModel::PaddleOcrVl1_5,
    Device::Cpu,
    &AnyPageParserPretrainedOptions::default(),
)?;
# let _ = parser;
# Ok(())
# }
```

ModelScope is the default source and Hugging Face is selectable with `with_source(DownloadSource::HuggingFace)`; `with_revision` pins a revision (the sources' defaults are `master` and `main`), which resolves to an immutable commit before anything downloads. Snapshots land under `$OAR_HOME/models/<org>/<name>/<commit>` (`$OAR_HOME` defaults to `~/.oar`, shared with oar-ocr-core): each file is verified against the SHA-256 the source API publishes (Hugging Face publishes it for LFS files), a snapshot is staged and published atomically, and published snapshots are never modified — a cached commit is reused without any network listing. The layout-composed models also download a PP-DocLayout checkpoint — `PaddlePaddle/PP-DocLayoutV3_safetensors` by default, overridable with `with_layout` or replaced by a local directory with `with_layout_dir`. ModelScope publishes GLM-OCR under `ZhipuAI/GLM-OCR` and that mirror is used automatically; HunyuanOCR and WeVisDoc are not on ModelScope and download from Hugging Face instead (logged once via `tracing`), with the cache still keyed by the model ID.

## Installation

This crate is self-contained: everything runs on Candle, it does not depend on `oar-ocr-core`, and **no build of it links ONNX Runtime**.

```bash
cargo add oar-ocr-vl
```

To enable GPU acceleration (CUDA), add the feature flag:

```bash
cargo add oar-ocr-vl --features cuda
```

To download checkpoints by model ID instead of managing local directories, enable `auto-download` (see [Unified Page Parsing](#unified-page-parsing)):

```bash
cargo add oar-ocr-vl --features auto-download
```

On macOS, enable Metal instead:

```bash
cargo add oar-ocr-vl --features metal
```

Metal inference uses Candle's fused SDPA kernels for supported attention shapes, including GQA without expanding K/V heads. For the best measured Apple Silicon throughput, explicitly set `OAR_VL_DTYPE=f16`. Use `OAR_VL_DISABLE_METAL_SDPA=1` for compatibility comparisons. The optional `OAR_VL_METAL_NATIVE_SOFTMAX=1` switch only changes the eager fallback; eager attention otherwise preserves the F32 softmax round trip.

The crate's custom CUDA kernels compile to PTX for the oldest GPU detected by `nvidia-smi` at build time. For headless, container, or cross-machine builds, set the target explicitly, for example `CUDA_COMPUTE_CAP=89 cargo build -p oar-ocr-vl --features cuda`. These kernels require compute capability 8.0 or newer.

## Usage

The snippets below use canonical model repository IDs for the checkpoints.

Examples default to `--device auto`: compiled CUDA(0), then Metal(0), then CPU, falling back on initialization failure. Library callers opt in with `auto_device()` or `utils::parse_device("auto")`; the existing dtype probe is unchanged. See [Automatic Device Selection](../docs/usage.md#automatic-device-selection) for details; use `--device cpu` to require CPU.

### PaddleOCR-VL

Use PaddleOCR-VL to recognize a specific aspect of an image (e.g., just the table or text).

```rust
use oar_ocr_vl::utils::image::load_image;
use oar_ocr_vl::{PaddleOcrVl, PaddleOcrVlTask};
use oar_ocr_vl::utils::parse_device;

let image = load_image("document.png")?;
let device = parse_device("cpu")?; // Or "cuda", "cuda:0", or "metal"

// Initialize model
let model = PaddleOcrVl::from_dir("PaddlePaddle/PaddleOCR-VL", device)?;

// Perform OCR. The API is batch-oriented, so pass one task per image.
let result = model
    .generate(&[image], &[PaddleOcrVlTask::Ocr], 256)?
    .into_iter()
    .next()
    .expect("one result")?;
println!("Result: {}", result);
```

PaddleOCR-VL-1.5 and PaddleOCR-VL-1.6 are loaded the same way, with additional tasks. PaddleOCR-VL-1.6 is plug-compatible with the 1.5 loader; point the same API at its checkpoint directory.

```rust
use oar_ocr_vl::utils::image::load_image;
use oar_ocr_vl::{PaddleOcrVl, PaddleOcrVlTask};
use oar_ocr_vl::utils::parse_device;

let image = load_image("seal.png")?;
let device = parse_device("cpu")?;
let model = PaddleOcrVl::from_dir("PaddlePaddle/PaddleOCR-VL-1.5", device)?;
let result = model
    .generate(&[image], &[PaddleOcrVlTask::Seal], 256)?
    .into_iter()
    .next()
    .expect("one result")?;
println!("Result: {}", result);
```

### PP-DocLayout

`PpDocLayout` detects layout regions and predicts their reading order, and implements `LayoutSource`, so it plugs straight into `DocParser` (see below). PP-DocLayoutV2 and PP-DocLayoutV3 load through the same API; the generation is read from `config.json`. Use the `_safetensors` repositories, which carry the `model.safetensors` weights this port loads.

PP-DocLayoutV2 applies the per-class thresholds from its `config.json`; PP-DocLayoutV3 uses a single threshold, adjustable with `with_score_threshold`.

### OvisOCR2

OvisOCR2 performs model-native full-page parsing without an external layout detector. `parse` applies the official prompt, image resizing, and post-processing and returns one Markdown document per page.

```rust
use oar_ocr_vl::utils::image::load_image;
use oar_ocr_vl::ovisocr2::DEFAULT_MAX_NEW_TOKENS;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::OvisOcr2;

let image = load_image("document.png")?;
let model = OvisOcr2::from_dir("ATH-MaaS/OvisOCR2", parse_device("cpu")?)?;
let markdown = model
    .parse(&[image], DEFAULT_MAX_NEW_TOKENS)?
    .into_iter()
    .next()
    .expect("one result")?;
println!("{markdown}");
```

The official runtime resizes RGB input with bicubic antialiasing to a 32-pixel-aligned area between `448²` and `2880²` pixels. Its fixed prompt requests reading-order Markdown, LaTeX formulas, HTML tables, and bounding-box `<img>` tags for visual regions. `parse` removes those visual-region blocks by default before applying truncated-repeat cleanup; call `parse_with_image_tags(..., true)` or `generate` to retain the references. The library does not create the referenced bounding-box crop files.

### Xiaomi-OCR-0

Xiaomi-OCR-0 performs model-native full-page parsing without an external layout detector, sharing its Qwen3.5 text tower with OvisOCR2. `parse` applies the official prompt and image preprocessing and returns one Markdown document per page with the official post-processing applied: truncated-repeat cleanup and OTSL table blocks converted to HTML.

```rust
use oar_ocr_vl::utils::image::load_image;
use oar_ocr_vl::xiaomi_ocr::DEFAULT_MAX_NEW_TOKENS;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::XiaomiOcr;

let image = load_image("document.png")?;
let model = XiaomiOcr::from_dir("SeerRay-Lab/Xiaomi-OCR-0", parse_device("cpu")?)?;
let markdown = model
    .parse(&[image], DEFAULT_MAX_NEW_TOKENS)?
    .into_iter()
    .next()
    .expect("one result")?;
println!("{markdown}");
```

The processor resizes RGB input with bicubic antialiasing to a 32-pixel-aligned area between the advertised bounds (`256²`–`4096²` pixels). The official prompts cover whole-page parsing, text regions, OTSL tables, LaTeX formulas, and key-information extraction (see the `xiaomi_ocr` module constants); as a `RecognitionBackend`, tables come back in OTSL and are converted to HTML by the pipeline, and chart regions are left unrecognized because the model defines no chart prompt.

### WeVisDoc

WeVisDoc (2B/4B) performs model-native full-page parsing on a Qwen3-VL backbone: the vision tower's intermediate features are injected into the decoder's first layers (DeepStack), and generation follows the official `wevisdoc/local.py` recipe — greedy decoding with the WeDocKit system prompt and a plain "Convert this document image to Markdown." instruction.

```rust
use oar_ocr_vl::utils::image::load_image;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::wevisdoc::DEFAULT_MAX_NEW_TOKENS;
use oar_ocr_vl::WeVisDoc;

let image = load_image("document.png")?;
let model = WeVisDoc::from_dir("Tencent/WeVisDoc-2B", parse_device("cpu")?)?;
let markdown = model
    .generate(&[image], DEFAULT_MAX_NEW_TOKENS)?
    .into_iter()
    .next()
    .expect("one result")?;
println!("{markdown}");
```

Pages are resized with the Qwen2-VL smart-resize rule to a 32-pixel-aligned area between `256²` and `4096²` pixels. Output is Markdown with LaTeX (`\\(...\\)`, `\\[...\\]`) formulas and HTML tables; the same `generate` path serves `DocParser` region crops.

### MonkeyOCRv2-S/B-Parsing

MonkeyOCRv2-S-Parsing and MonkeyOCRv2-B-Parsing use native Monkey ViT-S and ViT-B encoders, respectively, with the same Qwen3-0.6B decoder. The API reads either checkpoint's dimensions from its configuration and exposes the official full-page layout and end-to-end prompts as well as cropped text, formula, and OTSL-table recognition.

```rust
use oar_ocr_vl::utils::image::load_image;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::{MonkeyOcrV2, MonkeyOcrV2Task};

let image = load_image("document.png")?;
let model = MonkeyOcrV2::from_dir(
    "zenosai/MonkeyOCRv2-S-Parsing",
    parse_device("cuda:0")?,
)?;
let parsed = model
    .generate(&[image], &[MonkeyOcrV2Task::EndToEnd], 10_000)?
    .into_iter()
    .next()
    .expect("one result")?;
println!("{parsed}");
```

`EndToEnd` emits a reading-order list whose items contain normalized `bbox`, `label`, and `content` fields. `Layout` emits `bbox` and `label`; its preprocessing follows the official one-megapixel minimum used by the reference layout pass. `Text`, `Formula`, and `Table` can be used directly or through `RecognitionBackend`; table output is OTSL and is converted by `DocParser`.

### HPD-Parsing

HPD-Parsing performs full-page parsing with the official dynamic 448-pixel InternVL tiling path. Its parent branch emits layout and `<FORK>` markers; every marker immediately starts a content child from the matching parent KV prefix, and the child result is spliced back as `<CHILD>...`. The runtime advances all admitted parent/child requests as a continuous batch. Forked caches retain reference-counted, read-only prefix views and private writable tails; segmented attention consumes those views without copying the prefix K/V. P-MTP is enabled by default and drafts and verifies six future tokens in every active branch.

This is a model/runtime contract, not a training-free switch for arbitrary VLM checkpoints. A compatible model must have been trained to emit the `<FORK>`/`<CHILD>` protocol and must provide the matching P-MTP head. The decoder batching and fork-safe cache machinery is shared with the Qwen3-family text implementation, but other OCR VLMs do not acquire hierarchical outputs merely by loading HPD's head.

```rust
use oar_ocr_vl::utils::image::load_image;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::{HpdGenerationConfig, HpdParsing};

let image = load_image("document.png")?;
let model = HpdParsing::from_dir(
    "PaddlePaddle/HPD-Parsing",
    parse_device("cuda:0")?,
)?;
let parsed = model
    .parse(&[image], &HpdGenerationConfig::default())?
    .into_iter()
    .next()
    .expect("one result")?;
println!("{parsed}");
```

The returned text is the model-native reading-order `<BLOCK>type [bbox]<CHILD>content` stream. Set `use_mtp: false` in `HpdGenerationConfig` for ordinary greedy decoding. The native path uses the P-MTP weights embedded in the main checkpoint; the duplicate `P-MTP/model.safetensors` bundle is not required.

### DocParser

Parse an entire page into Markdown. This path is intended for external layout-first backends such as PaddleOCR-VL, PaddleOCR-VL-1.5, PaddleOCR-VL-1.6, and GLM-OCR.

```rust
use oar_ocr_vl::utils::image::load_image;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::{DocParser, PaddleOcrVl, PpDocLayout};

let device = parse_device("cpu")?;

let layout = PpDocLayout::from_dir("PaddlePaddle/PP-DocLayoutV3_safetensors", device.clone())?;
let vl = PaddleOcrVl::from_dir("PaddlePaddle/PaddleOCR-VL-1.5", device)?;
let parser = DocParser::new(&vl);

let result = parser.parse(&layout, load_image("page.jpg")?)?;
println!("{}", result.to_markdown());
```

### MinerU2.5 / MinerU2.5-Pro

```rust
use oar_ocr_vl::utils::image::load_image;
use oar_ocr_vl::{MinerU, MinerUParseOptions, PageParser};
use oar_ocr_vl::utils::parse_device;

let image = load_image("document.png")?;
let device = parse_device("cpu")?;
let model = MinerU::from_dir("opendatalab/MinerU2.5-2509-1.2B", device)?;
// For full documents, prefer the `mineru` example, which follows the
// model-native two-step pipeline: layout detection, then crop recognition.
let document = model.parse_page(&image, &MinerUParseOptions::default())?;
println!("{:#?}", document.blocks);
```

### TeleOCR

```rust
use oar_ocr_vl::utils::convert_otsl_to_html;
use oar_ocr_vl::utils::image::load_image;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::{TeleOcr, TeleOcrTask};

let image = load_image("table.png")?;
let device = parse_device("cpu")?;
let model = TeleOcr::from_dir("XingChen-AGI/TeleOCR", device)?;
// Tables come back as OTSL; formulas need `TeleOcrTask::postprocess`.
let raw = model
    .generate(&[image], &[TeleOcrTask::Table.prompt()], 4096)?
    .into_iter()
    .next()
    .expect("one result")?;
println!("{}", convert_otsl_to_html(raw.trim()));
```

### jina-ocr-v1

jina-ocr-v1 performs end-to-end page-to-Markdown parsing on a DeepSeek-OCR backbone: a SAM ViT-B + CLIP-L DeepEncoder produces visual tokens for a padded 1024px global view plus up to nine 640px dynamic tiles, and a 12-layer DeepSeek-V2 MoE decoder (64 routed + 2 shared experts, top-6, ~570M active parameters) generates greedily under the official sliding-window no-repeat-ngram guard.

```rust
use oar_ocr_vl::utils::image::load_image;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::jina_ocr::DEFAULT_MAX_NEW_TOKENS;
use oar_ocr_vl::JinaOcr;

let image = load_image("document.png")?;
let model = JinaOcr::from_dir("jinaai/jina-ocr-v1", parse_device("cpu")?)?;
let markdown = model
    .generate(&[image], DEFAULT_MAX_NEW_TOKENS)?
    .into_iter()
    .next()
    .expect("one result")?;
println!("{markdown}");
```

Output is Markdown with LaTeX formulas and HTML tables; the same `generate` path serves `DocParser` region crops. On CUDA the single-token decode step runs as a CUDA graph. The checkpoint's FastMTP head can add speculative decoding that is token-identical to plain autoregressive decoding in exact arithmetic (greedy verification plus the host-side no-repeat-ngram guard; in bf16 the verification block's kernel batching can flip near-tie picks, measured max |Δlogit| ≈ 0.25-0.56 on identical KV state), but it is **opt-in**: measured on the OmniDocBench demo pages (RTX 4090, bf16), adaptive MTP won on no page (median 3.7% slower than graphed plain decoding, +0.7% in total). Enable it with `OAR_JINAOCR_ENABLE_MTP=1` or `JinaOcrLoadOptions::with_mtp(true)` (`--mtp` in the example). Multi-image calls use a padded batch prefill and decode.

## Running Examples

The `oar-ocr-vl` crate includes several examples demonstrating its capabilities.

### DocParser

This example combines layout detection with a VLM for recognition. It supports PaddleOCR-VL, PaddleOCR-VL-1.5, PaddleOCR-VL-1.6, GLM-OCR, TeleOCR, and jina-ocr-v1.

```bash
cargo run --release -p oar-ocr-vl --features cuda --example doc_parser -- \
    --model-name paddleocr-vl-1.5 \
    --model-dir PaddlePaddle/PaddleOCR-VL-1.5 \
    --layout-dir PaddlePaddle/PP-DocLayoutV3_safetensors \
    --device cuda \
    document.jpg
```

The CLI example exposes the layout-first PaddleOCR-VL, GLM-OCR, TeleOCR, and jina-ocr-v1 paths. MonkeyOCRv2, OvisOCR2, Xiaomi-OCR-0, HunyuanOCR, and the MinerU models also implement `RecognitionBackend`; their dedicated examples remain the preferred complete-page paths. HPD-Parsing uses its model-native full-page protocol instead of `RecognitionBackend`.

### PaddleOCR-VL Direct Inference

Run the PaddleOCR-VL model directly on an image with a specific task prompt.

```bash
# OCR task
cargo run --release -p oar-ocr-vl --features cuda --example paddleocr_vl -- \
    --model-dir PaddlePaddle/PaddleOCR-VL \
    --device cuda \
    --task ocr \
    document.jpg

# Table task
cargo run --release -p oar-ocr-vl --features cuda --example paddleocr_vl -- \
    --model-dir PaddlePaddle/PaddleOCR-VL \
    --device cuda \
    --task table \
    table.jpg

# Text spotting with PaddleOCR-VL-1.5 or 1.6
cargo run --release -p oar-ocr-vl --features cuda --example paddleocr_vl -- \
    --model-dir PaddlePaddle/PaddleOCR-VL-1.5 \
    --device cuda \
    --task spotting \
    spotting.jpg

# Seal recognition with PaddleOCR-VL-1.5 or 1.6
cargo run --release -p oar-ocr-vl --features cuda --example paddleocr_vl -- \
    --model-dir PaddlePaddle/PaddleOCR-VL-1.6 \
    --device cuda \
    --task seal \
    seal.jpg
```

### HunyuanOCR 1.5 Direct Inference

```bash
cargo run --release -p oar-ocr-vl --features cuda --example hunyuanocr -- \
    --model-dir tencent/HunyuanOCR \
    --dflash-dir tencent/HunyuanOCR/dflash \
    --device cuda \
    --prompt "Detect and recognize text in the image, and output the text coordinates in a formatted manner." \
    document.jpg
```

The model repository root contains HunyuanOCR 1.5, which the loader detects automatically. To use the archived 1.0 checkpoint, pass its directory to `--model-dir`. `--dflash-dir` enables the official 15-token parallel draft path for 1.5. Omit it for ordinary autoregressive decoding. Library callers can use `HunyuanOcr::from_dirs(target_dir, dflash_dir, device)` or `HunyuanOcr::from_dir_with_dflash(model_dir, device)` when the draft is stored in the official `dflash/` subdirectory.

### GLM-OCR Direct Inference

```bash
cargo run --release -p oar-ocr-vl --features cuda --example glmocr -- \
    --model-dir zai-org/GLM-OCR \
    --device cuda \
    --prompt "Text Recognition:" \
    document.jpg
```

### OvisOCR2 Full-Page Parsing

The example accepts multiple page images. It uses the official prompt and defaults to 16,384 generated tokens per page. Add `--keep-image-tags` to retain the model's visual-region `<img>` blocks.

```bash
cargo run --release -p oar-ocr-vl --features cuda --example ovisocr2 -- \
    --model-dir ATH-MaaS/OvisOCR2 \
    --device cuda:0 \
    document-1.jpg document-2.jpg
```

### Xiaomi-OCR-0 Full-Page Parsing

The example accepts multiple page images. It uses the official prompt and defaults to 4,096 generated tokens per page; tables are converted from OTSL to HTML by the official post-processing.

```bash
cargo run --release -p oar-ocr-vl --features cuda --example xiaomi_ocr -- \
    --model-dir SeerRay-Lab/Xiaomi-OCR-0 \
    --device cuda:0 \
    document-1.jpg document-2.jpg
```

### MonkeyOCRv2-S/B-Parsing Direct Inference

Run the official end-to-end prompt over a complete page:

```bash
cargo run --release -p oar-ocr-vl --features cuda --example monkeyocrv2 -- \
    --model-dir zenosai/MonkeyOCRv2-S-Parsing \
    --device cuda:0 \
    --task end-to-end \
    document.jpg
```

Pass the ViT-B checkpoint directory to `--model-dir` to use that variant. Other task values are `layout`, `text`, `formula`, and `table`. Use `--prompt` to supply a custom instruction.

### HPD-Parsing Direct Inference

```bash
cargo run --release -p oar-ocr-vl --features cuda --example hpd_parsing -- \
    --model-dir PaddlePaddle/HPD-Parsing \
    --device cuda:0 \
    document.jpg
```

Use `--no-mtp` to compare with ordinary greedy decoding, `--speculative-tokens` to change the P-MTP draft length, `--max-active-branches` to bound the continuous batch, and `--prompt` to override `document parsing with fork.`. `--verbose` reports scheduler rounds, peak active branches, shared-prefix tokens, and P-MTP acceptance.

### MinerU2.5 and MinerU2.5-Pro Direct Inference

Model-native two-step document extraction (layout prompt + content extraction):

```bash
cargo run --release -p oar-ocr-vl --features cuda --example mineru -- \
    --model-dir opendatalab/MinerU2.5-2509-1.2B \
    --device cuda:0 \
    document.jpg
```

`MinerU2.5-Pro-2605` uses the same loader and example:

```bash
cargo run --release -p oar-ocr-vl --features cuda --example mineru -- \
    --model-dir opendatalab/MinerU2.5-Pro-2605-1.2B \
    --device cuda:0 \
    document.jpg
```

### MinerU-Diffusion-V1 Direct Inference

The default mode performs two-step structured extraction with block-diffusion decoding. Add `--single-pass` for flat full-page text recognition.

```bash
cargo run --release -p oar-ocr-vl --features cuda --example mineru_diffusion -- \
    --model-dir opendatalab/MinerU-Diffusion-V1-0320-2.5B \
    --device cuda:0 \
    document.jpg
```

### TeleOCR Direct Inference

Run the official per-task prompts; the layout tasks resize the input to 1036×1036 automatically. Pass `--raw` to skip OTSL/formula post-processing, or `--prompt` for a free-form instruction.

```bash
cargo run --release -p oar-ocr-vl --features cuda --example teleocr -- \
    --model-dir XingChen-AGI/TeleOCR \
    --device cuda:0 \
    --task table \
    XingChen-AGI/TeleOCR/assets/table.png
```
