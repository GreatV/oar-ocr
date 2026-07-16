# oar-ocr-vl

Vision-Language models for document understanding in Rust.

This crate provides native Rust inference for document VLMs using [Candle](https://github.com/huggingface/candle), along with a document parsing pipeline for backends that work well with external layout detection.

## Supported Models

| Model | Parameters | Inference path |
|---|---:|---|
| [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) | 0.9B | External-layout page parsing, text, table, formula, and chart recognition |
| [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) | 0.9B | PaddleOCR-VL tasks plus text spotting and seal recognition |
| [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | 0.9B | Region-aware refinement, drop-in compatible with the 1.5 loader |
| [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | 0.9B | External-layout page parsing, text, table, and formula recognition |
| [OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) | 0.8B | Model-native full-page document-to-Markdown parsing |
| [MonkeyOCRv2-S-Parsing](https://huggingface.co/zenosai/MonkeyOCRv2-S-Parsing) | 0.6B | Model-native layout, end-to-end parsing, text, formula, and OTSL-table recognition |
| [MonkeyOCRv2-B-Parsing](https://huggingface.co/zenosai/MonkeyOCRv2-B-Parsing) | 0.7B | Higher-capacity ViT-B variant with the same parsing and recognition tasks |
| [HunyuanOCR 1.5 / 1.0](https://huggingface.co/tencent/HunyuanOCR) | 1B | Model-native prompt-driven parsing with optional DFlash decoding for 1.5 |
| [MinerU2.5-2509](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B) | 1.2B | Model-native two-step layout detection and content extraction |
| [MinerU2.5-Pro-2605](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B) | 1.2B | Newer compatible checkpoint using the MinerU2.5 two-step pipeline |
| [MinerU-Diffusion-V1-0320](https://huggingface.co/opendatalab/MinerU-Diffusion-V1-0320-2.5B) | 2.5B | Block-diffusion OCR with two-step structured extraction or single-pass recognition |

See [`examples`](examples) for runnable examples.

## Document Parsing Pipeline

**DocParser** is a unified document parsing API for layout-first backends. It combines:

1. **Layout detection** (ONNX models like PP-DocLayoutV3) to identify document regions
2. **VL-based recognition** to extract content from each region

Use DocParser with PaddleOCR-VL, PaddleOCR-VL-1.5, PaddleOCR-VL-1.6, GLM-OCR, and optionally MonkeyOCRv2 for externally detected crops. MonkeyOCRv2's native `Layout` and `EndToEnd` tasks are preferable for complete pages. OvisOCR2 is an end-to-end full-page parser and deliberately does not use DocParser. HunyuanOCR should be used with its model-native full-page prompts. MinerU2.5, MinerU2.5-Pro, and MinerU-Diffusion should use their model-native two-step extraction examples.

## Installation

Add `oar-ocr-vl` with bundled ONNX Runtime binaries. The snippets below also import image-loading and layout helpers from `oar-ocr-core`, so declare it as a direct dependency:

```bash
cargo add oar-ocr-vl --features download-binaries
cargo add oar-ocr-core --no-default-features
```

The VL crate has no default features and depends on `oar-ocr-core`, which links ONNX Runtime. You may omit `download-binaries` only when a system ONNX Runtime installation is available through `ORT_LIB_PATH` or `ORT_LIB_LOCATION`.

```bash
cargo add oar-ocr-vl
cargo add oar-ocr-core --no-default-features
```

To enable GPU acceleration (CUDA), add the feature flag:

```bash
cargo add oar-ocr-vl --features cuda,download-binaries
```

On macOS, enable Metal instead:

```bash
cargo add oar-ocr-vl --features metal,download-binaries
```

The crate's custom CUDA kernels compile to PTX for the oldest GPU detected by `nvidia-smi` at build time. For headless, container, or cross-machine builds, set the target explicitly, for example `CUDA_COMPUTE_CAP=89 cargo build -p oar-ocr-vl --features cuda,download-binaries`. These kernels require compute capability 8.0 or newer.

## Usage

### PaddleOCR-VL

Use PaddleOCR-VL to recognize a specific aspect of an image (e.g., just the table or text).

```rust
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::{PaddleOcrVl, PaddleOcrVlTask};
use oar_ocr_vl::utils::parse_device;

let image = load_image("document.png")?;
let device = parse_device("cpu")?; // Or "cuda", "cuda:0", or "metal"

// Initialize model
let model = PaddleOcrVl::from_dir("PaddleOCR-VL", device)?;

// Perform OCR. The API is batch-oriented, so pass one task per image.
let result = model
    .generate(&[image], &[PaddleOcrVlTask::Ocr], 256)
    .into_iter()
    .next()
    .expect("one result")?;
println!("Result: {}", result);
```

PaddleOCR-VL-1.5 and PaddleOCR-VL-1.6 are loaded the same way, with additional tasks. PaddleOCR-VL-1.6 is plug-compatible with the 1.5 loader (`PaddleOcrVl::from_dir("PaddleOCR-VL-1.6", device)`):

```rust
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::{PaddleOcrVl, PaddleOcrVlTask};
use oar_ocr_vl::utils::parse_device;

let image = load_image("seal.png")?;
let device = parse_device("cpu")?;
let model = PaddleOcrVl::from_dir("PaddleOCR-VL-1.5", device)?;
let result = model
    .generate(&[image], &[PaddleOcrVlTask::Seal], 256)
    .into_iter()
    .next()
    .expect("one result")?;
println!("Result: {}", result);
```

### OvisOCR2

OvisOCR2 performs model-native full-page parsing without an external layout detector. `parse` applies the official prompt, image resizing, and post-processing and returns one Markdown document per page.

```rust
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::ovisocr2::DEFAULT_MAX_NEW_TOKENS;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::OvisOcr2;

let image = load_image("document.png")?;
let model = OvisOcr2::from_dir("models/OvisOCR2", parse_device("cpu")?)?;
let markdown = model
    .parse(&[image], DEFAULT_MAX_NEW_TOKENS)
    .into_iter()
    .next()
    .expect("one result")?;
println!("{markdown}");
```

The official runtime resizes RGB input with bicubic antialiasing to a 32-pixel-aligned area between `448²` and `2880²` pixels. Its fixed prompt requests reading-order Markdown, LaTeX formulas, HTML tables, and bounding-box `<img>` tags for visual regions. `parse` removes those visual-region blocks by default before applying truncated-repeat cleanup; call `parse_with_image_tags(..., true)` or `generate` to retain the references. The library does not create the referenced bounding-box crop files.

### MonkeyOCRv2-S/B-Parsing

MonkeyOCRv2-S-Parsing and MonkeyOCRv2-B-Parsing use native Monkey ViT-S and ViT-B encoders, respectively, with the same Qwen3-0.6B decoder. The API reads either checkpoint's dimensions from its configuration and exposes the official full-page layout and end-to-end prompts as well as cropped text, formula, and OTSL-table recognition.

```rust
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::{MonkeyOcrV2, MonkeyOcrV2Task};

let image = load_image("document.png")?;
let model = MonkeyOcrV2::from_dir(
    "MonkeyOCRv2-S-Parsing",
    parse_device("cuda:0")?,
)?;
let parsed = model
    .generate(&[image], &[MonkeyOcrV2Task::EndToEnd], 10_000)
    .into_iter()
    .next()
    .expect("one result")?;
println!("{parsed}");
```

`EndToEnd` emits a reading-order list whose items contain normalized `bbox`, `label`, and `content` fields. `Layout` emits `bbox` and `label`; its preprocessing follows the official one-megapixel minimum used by the reference layout pass. `Text`, `Formula`, and `Table` can be used directly or through `RecognitionBackend`; table output is OTSL and is converted by `DocParser`.

### DocParser

Parse an entire page into Markdown with a layout predictor. This path is intended for external layout-first backends such as PaddleOCR-VL, PaddleOCR-VL-1.5, PaddleOCR-VL-1.6, and GLM-OCR.

```rust
use oar_ocr_core::utils::load_image;
use oar_ocr_core::predictors::LayoutDetectionPredictor;
use oar_ocr_vl::{DocParser, PaddleOcrVl};
use oar_ocr_vl::utils::parse_device;

let device = parse_device("cpu")?;

// 1. Setup Layout Detector
let layout_predictor = LayoutDetectionPredictor::builder()
    .model_name("pp-doclayoutv3")
    .build("pp-doclayoutv3.onnx")?;

// 2. Setup a layout-first recognition backend
let vl = PaddleOcrVl::from_dir("models/PaddleOCR-VL-1.5", device.clone())?;
let parser = DocParser::new(&vl);

// 3. Parse Document
let image = load_image("page.jpg")?;
let result = parser.parse(&layout_predictor, image)?;

// 4. Output as Markdown
println!("{}", result.to_markdown());
```

### MinerU2.5 / MinerU2.5-Pro

```rust
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::MinerU;
use oar_ocr_vl::utils::parse_device;

let image = load_image("document.png")?;
let device = parse_device("cpu")?;
let model = MinerU::from_dir("models/MinerU2.5-2509-1.2B", device)?;
// For full documents, prefer the `mineru` example, which follows the
// model-native two-step pipeline: layout detection, then crop recognition.
let result = model
    .generate(&[image], &["\nText Recognition:"], 4096)
    .into_iter()
    .next()
    .expect("one result")?;
println!("Result: {}", result);
```

## Running Examples

The `oar-ocr-vl` crate includes several examples demonstrating its capabilities.

### DocParser

This example combines layout detection (ONNX) with a VLM for recognition. It supports PaddleOCR-VL, PaddleOCR-VL-1.5, PaddleOCR-VL-1.6, and GLM-OCR.

```bash
cargo run --release -p oar-ocr-vl --features cuda,download-binaries --example doc_parser -- \
    --model-name paddleocr-vl-1.5 \
    --model-dir models/PaddleOCR-VL-1.5 \
    --layout-model models/pp-doclayoutv3.onnx \
    --device cuda \
    document.jpg
```

OvisOCR2, HunyuanOCR, and the MinerU models are intentionally not exposed by this example because their reference-quality paths are model-native full-page parsing, prompt-driven full-page parsing, and model-native two-step extraction, respectively. MonkeyOCRv2 implements `RecognitionBackend`, but its dedicated example is the preferred complete-page path.

### PaddleOCR-VL Direct Inference

Run the PaddleOCR-VL model directly on an image with a specific task prompt.

```bash
# OCR task
cargo run --release -p oar-ocr-vl --features cuda,download-binaries --example paddleocr_vl -- \
    --model-dir models/PaddleOCR-VL \
    --device cuda \
    --task ocr \
    document.jpg

# Table task
cargo run --release -p oar-ocr-vl --features cuda,download-binaries --example paddleocr_vl -- \
    --model-dir models/PaddleOCR-VL \
    --device cuda \
    --task table \
    table.jpg

# Text spotting with PaddleOCR-VL-1.5 or 1.6
cargo run --release -p oar-ocr-vl --features cuda,download-binaries --example paddleocr_vl -- \
    --model-dir models/PaddleOCR-VL-1.5 \
    --device cuda \
    --task spotting \
    spotting.jpg

# Seal recognition with PaddleOCR-VL-1.5 or 1.6
cargo run --release -p oar-ocr-vl --features cuda,download-binaries --example paddleocr_vl -- \
    --model-dir models/PaddleOCR-VL-1.6 \
    --device cuda \
    --task seal \
    seal.jpg
```

### HunyuanOCR 1.5 Direct Inference

```bash
cargo run --release -p oar-ocr-vl --features cuda,download-binaries --example hunyuanocr -- \
    --model-dir models/HunyuanOCR \
    --dflash-dir models/HunyuanOCR/dflash \
    --device cuda \
    --prompt "Detect and recognize text in the image, and output the text coordinates in a formatted manner." \
    document.jpg
```

The model repository root contains HunyuanOCR 1.5. The loader detects it automatically. Use `--model-dir models/HunyuanOCR/v1.0` for the archived 1.0 checkpoint. `--dflash-dir` enables the official 15-token parallel draft path for 1.5. Omit it for ordinary autoregressive decoding. Library callers can use `HunyuanOcr::from_dirs(target_dir, dflash_dir, device)` or `HunyuanOcr::from_dir_with_dflash(model_dir, device)` when the draft is stored in the official `dflash/` subdirectory.

### GLM-OCR Direct Inference

```bash
cargo run --release -p oar-ocr-vl --features cuda,download-binaries --example glmocr -- \
    --model-dir models/GLM-OCR \
    --device cuda \
    --prompt "Text Recognition:" \
    document.jpg
```

### OvisOCR2 Full-Page Parsing

The example accepts multiple page images. It uses the official prompt and defaults to 16,384 generated tokens per page. Add `--keep-image-tags` to retain the model's visual-region `<img>` blocks.

```bash
cargo run --release -p oar-ocr-vl --features cuda,download-binaries --example ovisocr2 -- \
    --model-dir models/OvisOCR2 \
    --device cuda:0 \
    document-1.jpg document-2.jpg
```

### MonkeyOCRv2-S/B-Parsing Direct Inference

Run the official end-to-end prompt over a complete page:

```bash
cargo run --release -p oar-ocr-vl --features cuda,download-binaries --example monkeyocrv2 -- \
    --model-dir MonkeyOCRv2-S-Parsing \
    --device cuda:0 \
    --task end-to-end \
    document.jpg
```

Pass `MonkeyOCRv2-B-Parsing` to `--model-dir` to use the ViT-B checkpoint. Other task values are `layout`, `text`, `formula`, and `table`. Use `--prompt` to supply a custom instruction.

### MinerU2.5 and MinerU2.5-Pro Direct Inference

Model-native two-step document extraction (layout prompt + content extraction):

```bash
cargo run --release -p oar-ocr-vl --features cuda,download-binaries --example mineru -- \
    --model-dir models/MinerU2.5-2509-1.2B \
    --device cuda:0 \
    document.jpg
```

`MinerU2.5-Pro-2605` uses the same loader and example:

```bash
cargo run --release -p oar-ocr-vl --features cuda,download-binaries --example mineru -- \
    --model-dir models/MinerU2.5-Pro-2605-1.2B \
    --device cuda:0 \
    document.jpg
```

### MinerU-Diffusion-V1 Direct Inference

The default mode performs two-step structured extraction with block-diffusion decoding. Add `--single-pass` for flat full-page text recognition.

```bash
cargo run --release -p oar-ocr-vl --features cuda,download-binaries --example mineru_diffusion -- \
    --model-dir models/MinerU-Diffusion-V1-0320-2.5B \
    --device cuda:0 \
    document.jpg
```
