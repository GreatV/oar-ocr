# oar-ocr-vl

Vision-Language models for document understanding in Rust.

This crate provides PaddleOCR-VL and UniRec implementations using [Candle](https://github.com/huggingface/candle) for native Rust inference.

## Supported Models

### PaddleOCR-VL

[PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) is an ultra-compact (0.9B parameters) Vision-Language Model for document parsing, released by Baidu's PaddlePaddle team. It supports **109 languages** and excels in recognizing complex elements including text, tables, formulas, and 11 chart types.

### UniRec

[UniRec](https://github.com/Topdu/OpenOCR) is a unified recognition model with only **0.1B parameters**, developed by the FVL Laboratory at Fudan University. It is designed for high-accuracy and efficient recognition of plain text, mathematical formulas, and mixed content in both Chinese and English.

### DocParser

Two-stage document parsing API that combines layout detection (ONNX) with VL-based recognition, supporting both UniRec and PaddleOCR-VL backends.

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
    .model_name("pp-doclayoutv2")
    .build("pp-doclayoutv2.onnx")?;

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
cargo run --release --example doc_parser -- \
    --model-name unirec \
    --model-dir models/unirec-0.1b \
    --layout-model models/pp-doclayoutv2.onnx \
    document.jpg
```

### UniRec (Direct Inference)

Run the UniRec model directly on an image.

```bash
cargo run --release --example unirec -- \
    --model-dir models/unirec-0.1b \
    formula.png
```

### PaddleOCR-VL (Direct Inference)

Run the PaddleOCR-VL model directly on an image with a specific task prompt.

```bash
# OCR task
cargo run --release --example paddleocr_vl -- \
    --model-dir PaddleOCR-VL \
    --task ocr \
    document.jpg

# Table task
cargo run --release --example paddleocr_vl -- \
    --model-dir PaddleOCR-VL \
    --task table \
    table.jpg
```
