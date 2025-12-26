# OAR (ONNXRuntime And Rust) OCR

![Crates.io Version](https://img.shields.io/crates/v/oar-ocr)
![Crates.io Downloads (recent)](https://img.shields.io/crates/dr/oar-ocr)
[![dependency status](https://deps.rs/repo/github/GreatV/oar-ocr/status.svg)](https://deps.rs/repo/github/GreatV/oar-ocr)
![GitHub License](https://img.shields.io/github/license/GreatV/oar-ocr)

A comprehensive OCR and document understanding library built in Rust with ONNX Runtime.

## Quick Start

### Installation

```bash
cargo add oar-ocr
```

With GPU support:

```bash
cargo add oar-ocr --features cuda
```

### Basic Usage

```rust
use oar_ocr::prelude::*;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ocr = OAROCRBuilder::new(
        "pp-ocrv5_mobile_det.onnx",
        "pp-ocrv5_mobile_rec.onnx",
        "ppocrv5_dict.txt",
    )
    .build()?;

    let image = load_image(Path::new("document.jpg"))?;
    let results = ocr.predict(vec![image])?;

    for text_region in &results[0].text_regions {
        if let Some((text, confidence)) = text_region.text_with_confidence() {
            println!("{} ({:.2})", text, confidence);
        }
    }

    Ok(())
}
```

### Document Structure Analysis

```rust
use oar_ocr::oarocr::OARStructureBuilder;

let structure = OARStructureBuilder::new("pp-doclayout_plus-l.onnx")
    .with_table_classification("pp-lcnet_x1_0_table_cls.onnx")
    .with_table_structure_recognition("slanet_plus.onnx", "wireless")
    .table_structure_dict_path("table_structure_dict_ch.txt")
    .with_ocr("pp-ocrv5_mobile_det.onnx", "pp-ocrv5_mobile_rec.onnx", "ppocrv5_dict.txt")
    .build()?;
```

## Documentation

- [**Usage Guide**](docs/usage.md) - Detailed API usage, builder patterns, GPU configuration
- [**Pre-trained Models**](docs/models.md) - Model download links and recommended configurations

## Examples

```bash
cargo run --example ocr -- --help
cargo run --example structure -- --help
```

See `examples/` directory for complete CLI examples.

### PaddleOCR-VL (Vision-Language)

[PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) is a Vision-Language model for advanced document understanding. It supports element-level OCR and layout-first document parsing. Our implementation uses [Candle](https://github.com/huggingface/candle) for inference. Download the model first:

```bash
huggingface-cli download PaddlePaddle/PaddleOCR-VL --local-dir PaddleOCR-VL
```

```bash
# Element-level OCR
cargo run --release --features paddleocr-vl,cuda --example paddleocr_vl -- --model-dir PaddleOCR-VL --task ocr document.jpg

# Table recognition (outputs HTML)
cargo run --release --features paddleocr-vl,cuda --example paddleocr_vl -- --model-dir PaddleOCR-VL --task table table.jpg

# Formula recognition (outputs LaTeX)
cargo run --release --features paddleocr-vl,cuda --example paddleocr_vl -- --model-dir PaddleOCR-VL --task formula formula.png

# Chart recognition
cargo run --release --features paddleocr-vl,cuda --example paddleocr_vl -- --model-dir PaddleOCR-VL --task chart chart.png

# Layout-first doc parsing (PP-DocLayoutV2 -> PaddleOCR-VL)
cargo run --release --features paddleocr-vl,cuda --example paddleocr_vl -- --model-dir PaddleOCR-VL --layout-model pp-doclayoutv2.onnx document.jpg
```

## Acknowledgments

This project builds upon the excellent work of several open-source projects:

- **[ort](https://github.com/pykeio/ort)**: Rust bindings for ONNX Runtime by pykeio. This crate provides the Rust interface to ONNX Runtime that powers the efficient inference engine in this OCR library.

- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)**: Baidu's awesome multilingual OCR toolkits based on PaddlePaddle. This project utilizes PaddleOCR's pre-trained models, which provide excellent accuracy and performance for text detection and recognition across multiple languages.

- **[Candle](https://github.com/huggingface/candle)**: A minimalist ML framework for Rust by Hugging Face. We use Candle to implement Vision-Language model inference.
