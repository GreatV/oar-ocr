# oar-ocr-vl

Vision-Language models for document understanding in Rust.

This crate provides native Rust inference for document VLMs using [Candle](https://github.com/huggingface/candle), along with a document parsing pipeline for backends that work well with external layout detection.

## Supported Models

| Model | Parameters | Description |
|-------|------------|-------------|
| [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) | 0.9B | SOTA document parsing VLM supporting 109 languages, text, tables, formulas, and 11 chart types |
| [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) | 0.9B | Next-gen PaddleOCR-VL with 94.5% on OmniDocBench v1.5, adds text spotting and seal recognition |
| [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | 1.0B | Region-aware refinement on top of PaddleOCR-VL-1.5; 96.33% on OmniDocBench v1.6 (SOTA), drop-in compatible with the 1.5 loader |
| [HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) | 1B | End-to-end OCR VLM for multilingual document parsing, text spotting, and information extraction |
| [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | 0.9B | #1 on OmniDocBench v1.5 (94.62), optimized for real-world scenarios with MTP loss and RL training |
| [MinerU2.5](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B) | 1.2B | Decoupled document parsing VLM with strong text, formula, and table recognition |

## Document Parsing Pipeline

**DocParser** is a unified document parsing API for layout-first backends. It combines:

1. **Layout detection** (ONNX models like PP-DocLayoutV3) to identify document regions
2. **VL-based recognition** to extract content from each region

Use DocParser with PaddleOCR-VL, PaddleOCR-VL-1.5, PaddleOCR-VL-1.6, and GLM-OCR. HunyuanOCR should be used with its model-native full-page prompts, and MinerU2.5 should use its model-native two-step extraction example.

## Installation

Add `oar-ocr-vl` to your project:

```bash
cargo add oar-ocr-vl
```

If you use ONNX-based helpers from `oar-ocr-core` and want ORT binaries to be fetched automatically during build, enable `download-binaries` explicitly:

```bash
cargo add oar-ocr-vl --features download-binaries
```

To enable GPU acceleration (CUDA), add the feature flag:

```bash
cargo add oar-ocr-vl --features cuda
```

### Hierarchical Speculative Decoding (HSD)

A training-free CUDA acceleration scheme for the VLMs listed above. A cheap pipeline drafter (layout + OCR) proposes per-region text candidates and the target VLM verifies them in batches via tree-attention. Each backbone exposes `generate_hsd*` methods alongside its baseline `generate`. Build with `--features hsd` (transitively enables `cuda`):

```bash
cargo add oar-ocr-vl --features hsd,download-binaries
```

See [`docs/hsd.md`](../docs/hsd.md) at the workspace root for the algorithm overview, `DsvConfig` / `HsdConfig` knobs, supported backbones, and Average-Acceptance-Length (AAL) guidance. End-to-end runnable examples live under `examples/hsd_*.rs`.

## Usage

### PaddleOCR-VL

Use PaddleOCR-VL to recognize a specific aspect of an image (e.g., just the table or text).

```rust
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::{PaddleOcrVl, PaddleOcrVlTask};

let image = load_image("document.png")?;
let device = candle_core::Device::Cpu; // Or Device::new_cuda(0)?

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

let image = load_image("seal.png")?;
let device = candle_core::Device::Cpu;
let model = PaddleOcrVl::from_dir("PaddleOCR-VL-1.5", device)?;
let result = model
    .generate(&[image], &[PaddleOcrVlTask::Seal], 256)
    .into_iter()
    .next()
    .expect("one result")?;
println!("Result: {}", result);
```

### DocParser

Parse an entire page into Markdown with a layout predictor. This path is intended for external layout-first backends such as PaddleOCR-VL, PaddleOCR-VL-1.5, PaddleOCR-VL-1.6, and GLM-OCR.

```rust
use oar_ocr_core::utils::load_image;
use oar_ocr_core::predictors::LayoutDetectionPredictor;
use oar_ocr_vl::{DocParser, PaddleOcrVl};

let device = candle_core::Device::Cpu;

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

### MinerU2.5

```rust
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::MinerU;

let image = load_image("document.png")?;
let device = candle_core::Device::Cpu;
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
cargo run --release --features cuda --example doc_parser -- \
    --model-name paddleocr-vl-1.5 \
    --model-dir models/PaddleOCR-VL-1.5 \
    --layout-model models/pp-doclayoutv3.onnx \
    --device cuda \
    document.jpg
```

HunyuanOCR and MinerU2.5 are intentionally not exposed by this example because their reference-quality paths are prompt-driven full-page parsing and model-native two-step extraction, respectively.

### PaddleOCR-VL (Direct Inference)

Run the PaddleOCR-VL model directly on an image with a specific task prompt.

```bash
# OCR task
cargo run --release --features cuda --example paddleocr_vl -- \
    --model-dir models/PaddleOCR-VL \
    --device cuda \
    --task ocr \
    document.jpg

# Table task
cargo run --release --features cuda --example paddleocr_vl -- \
    --model-dir models/PaddleOCR-VL \
    --device cuda \
    --task table \
    table.jpg

# Text spotting (PaddleOCR-VL-1.5)
cargo run --release --features cuda --example paddleocr_vl -- \
    --model-dir models/PaddleOCR-VL-1.5 \
    --device cuda \
    --task spotting \
    spotting.jpg

# Seal recognition (PaddleOCR-VL-1.5)
cargo run --release --features cuda --example paddleocr_vl -- \
    --model-dir models/PaddleOCR-VL-1.5 \
    --device cuda \
    --task seal \
    seal.jpg
```

### HunyuanOCR (Direct Inference)

```bash
cargo run --release --features cuda --example hunyuanocr -- \
    --model-dir models/HunyuanOCR \
    --device cuda \
    --prompt "Detect and recognize text in the image, and output the text coordinates in a formatted manner." \
    document.jpg
```

### GLM-OCR (Direct Inference)

```bash
cargo run --release --features cuda --example glmocr -- \
    --model-dir models/GLM-OCR \
    --device cuda \
    --prompt "Text Recognition:" \
    document.jpg
```

### MinerU2.5 (Direct Inference)

Model-native two-step document extraction (layout prompt + content extraction):

```bash
cargo run --release --features cuda --example mineru -- \
    --model-dir models/MinerU2.5-2509-1.2B \
    --device cuda:0 \
    document.jpg
```

### HSD (Hierarchical Speculative Decoding)

The shared `hsd_demo` example runs baseline and HSD back-to-back so you can
compare wall time and outputs. Select the target VLM via `--backend`:

```bash
# Single-page smoke test (HunyuanOCR backbone).
cargo run --release --features hsd,download-binaries --example hsd_demo -- \
    --backend hunyuanocr \
    --model-dir models/HunyuanOCR \
    --device cuda \
    --image document.jpg

# Quality + perf matrix over OmniDocBench-style inputs.
cargo run --release --features hsd,download-binaries --example hsd_omnidocbench -- --help
```

See [`docs/hsd.md`](../docs/hsd.md) for the full set of HSD knobs and the backbone-by-backbone capability matrix.
