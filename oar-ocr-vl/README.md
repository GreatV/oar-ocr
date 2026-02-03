# oar-ocr-vl

Vision-Language models for document understanding in Rust.

This crate provides PaddleOCR-VL, UniRec, HunyuanOCR, and LightOnOCR implementations using [Candle](https://github.com/huggingface/candle) for native Rust inference.

## Supported Models

### PaddleOCR-VL

[PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) is an ultra-compact (0.9B parameters) Vision-Language Model for document parsing, released by Baidu's PaddlePaddle team. It supports **109 languages** and excels in recognizing complex elements including text, tables, formulas, and 11 chart types.

### PaddleOCR-VL-1.5

[PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) is the next-generation 0.9B PaddleOCR-VL model with improved accuracy and support for **text spotting** and **seal recognition**. It is a drop-in replacement for PaddleOCR-VL when using `PaddleOcrVl::from_dir`, and adds `PaddleOcrVlTask::Spotting` and `PaddleOcrVlTask::Seal`.

### UniRec

[UniRec](https://github.com/Topdu/OpenOCR) is a unified recognition model with only **0.1B parameters**, developed by the FVL Laboratory at Fudan University. It is designed for high-accuracy and efficient recognition of plain text, mathematical formulas, and mixed content in both Chinese and English.

### HunyuanOCR

[HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) is a 1B parameter OCR expert VLM powered by Hunyuan's multimodal architecture. This crate provides native Rust inference for the `model_type=hunyuan_vl` checkpoint.

### LightOnOCR

[LightOnOCR-2](https://huggingface.co/lightonai/LightOnOCR-2-1B) is an efficient end-to-end OCR VLM for extracting clean text from document images without an external pipeline.

### DocParser

Two-stage document parsing API that combines layout detection (ONNX) with VL-based recognition, supporting UniRec, PaddleOCR-VL, HunyuanOCR, and LightOnOCR backends.

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
    --model-dir PaddleOCR-VL \
    --device cuda \
    --task ocr \
    document.jpg

# Table task
cargo run --release --features cuda --example paddleocr_vl -- \
    --model-dir PaddleOCR-VL \
    --device cuda \
    --task table \
    table.jpg

# Text spotting (PaddleOCR-VL-1.5)
cargo run --release --features cuda --example paddleocr_vl -- \
    --model-dir PaddleOCR-VL-1.5 \
    --device cuda \
    --task spotting \
    spotting.jpg

# Seal recognition (PaddleOCR-VL-1.5)
cargo run --release --features cuda --example paddleocr_vl -- \
    --model-dir PaddleOCR-VL-1.5 \
    --device cuda \
    --task seal \
    seal.jpg
```

### HunyuanOCR (Direct Inference)

```bash
cargo run --release --features cuda --example hunyuanocr -- \
    --model-dir ~/repos/HunyuanOCR \
    --device cuda \
    --prompt "Detect and recognize text in the image, and output the text coordinates in a formatted manner." \
    document.jpg
```
