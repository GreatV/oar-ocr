# oar-ocr-vl

Vision-Language models for document understanding in Rust.

This crate provides native Rust inference for document VLMs using [Candle](https://github.com/huggingface/candle), along with a two-stage document parsing pipeline.

## Supported Models

| Model | Parameters | Description |
|-------|------------|-------------|
| [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) | 0.9B | SOTA document parsing VLM supporting 109 languages, text, tables, formulas, and 11 chart types |
| [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) | 0.9B | Next-gen PaddleOCR-VL with 94.5% on OmniDocBench v1.5, adds text spotting and seal recognition |
| [UniRec](https://huggingface.co/topdu/unirec-0.1b) | 0.1B | Ultra-lightweight unified recognition for text, formulas, and tables (Chinese/English) |
| [HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) | 1B | End-to-end OCR VLM for multilingual document parsing, text spotting, and information extraction |
| [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | 0.9B | #1 on OmniDocBench v1.5 (94.62), optimized for real-world scenarios with MTP loss and RL training |
| [LightOnOCR-2](https://huggingface.co/lightonai/LightOnOCR-2-1B) | 1B | SOTA on OlmOCR-Bench, 9x smaller than competitors, processes 5.7 pages/s on H100 |
| [MinerU2.5](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B) | 1.2B | Decoupled document parsing VLM with strong text, formula, and table recognition |

## Document Parsing Pipeline

**DocParser** is a two-stage document parsing API that combines:

1. **Layout detection** (ONNX models like PP-DocLayoutV3) to identify document regions
2. **VL-based recognition** (any supported model above) to extract content from each region

This approach provides structured output with reading order preservation.

## Installation

Add `oar-ocr-vl` to your project:

```bash
cargo add oar-ocr-vl
```

To enable GPU acceleration (CUDA), add the feature flag:

```bash
cargo add oar-ocr-vl --features cuda
```

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

// Perform OCR
let result = model.generate(image, PaddleOcrVlTask::Ocr, 256)?;
println!("Result: {}", result);
```

PaddleOCR-VL-1.5 is loaded the same way, with additional tasks:

```rust
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::{PaddleOcrVl, PaddleOcrVlTask};

let image = load_image("seal.png")?;
let device = candle_core::Device::Cpu;
let model = PaddleOcrVl::from_dir("PaddleOCR-VL-1.5", device)?;
let result = model.generate(image, PaddleOcrVlTask::Seal, 256)?;
println!("Result: {}", result);
```

### UniRec

UniRec is a unified model that handles text, mathematical formulas, and table structures in a single pass without needing task-specific prompts.

```rust
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::UniRec;

let image = load_image("mixed_content.png")?;
let device = candle_core::Device::Cpu;

// Initialize model
let model = UniRec::from_dir("models/unirec-0.1b", device)?;

// Generate content (automatically handles text, formulas, etc.)
let result = model.generate(image, 512)?;
println!("Result: {}", result);
```

### DocParser

Combine layout detection with a VLM backend to parse an entire page into Markdown.

```rust
use oar_ocr_core::utils::load_image;
use oar_ocr_core::predictors::LayoutDetectionPredictor;
use oar_ocr_vl::{DocParser, UniRec};

let device = candle_core::Device::Cpu;

// 1. Setup Layout Detector
let layout_predictor = LayoutDetectionPredictor::builder()
    .model_name("pp-doclayoutv3")
    .build("pp-doclayoutv3.onnx")?;

// 2. Setup Recognition Backend (UniRec or PaddleOCR-VL)
let unirec = UniRec::from_dir("models/unirec-0.1b", device)?;
let parser = DocParser::new(&unirec);

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
let model = MinerU::from_dir("/path/to/MinerU2.5-2509-1.2B", device)?;
let result = model.generate(&[image], &["\nDocument Parsing:"], 4096);
println!("Result: {}", result[0].as_ref()?);
```

## Running Examples

The `oar-ocr-vl` crate includes several examples demonstrating its capabilities.

### DocParser (Two-Stage Pipeline)

This example combines layout detection (ONNX) with a VLM for recognition.

```bash
cargo run --release --features cuda --example doc_parser -- \
    --model-name unirec \
    --model-dir models/unirec-0.1b \
    --layout-model models/pp-doclayoutv3.onnx \
    --device cuda \
    document.jpg
```

### UniRec (Direct Inference)

Run the UniRec model directly on an image.

```bash
cargo run --release --features cuda --example unirec -- \
    --model-dir models/unirec-0.1b \
    --device cuda \
    formula.png
```

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

Two-step document extraction (layout detection + content extraction):

```bash
cargo run --release --features cuda --example mineru -- \
    --model-dir /path/to/MinerU2.5-2509-1.2B \
    --device cuda:0 \
    document.jpg
```
