# OAR (ONNXRuntime And Rust) OCR

![Crates.io Version](https://img.shields.io/crates/v/oar-ocr)
![Crates.io Downloads (recent)](https://img.shields.io/crates/dr/oar-ocr)
[![dependency status](https://deps.rs/repo/github/GreatV/oar-ocr/status.svg)](https://deps.rs/repo/github/GreatV/oar-ocr)
![GitHub License](https://img.shields.io/github/license/GreatV/oar-ocr)

A comprehensive OCR (Optical Character Recognition) library, built in Rust with ONNX Runtime for efficient inference.

## Quick Start

### Installation

Add OAR OCR to your project's `Cargo.toml`:

```bash
cargo add oar-ocr
```

Or manually add it to your `Cargo.toml`:

```toml
[dependencies]
oar-ocr = "0.1"
```

### Basic Usage

Here's a simple example of how to use OAR OCR to extract text from an image:

```rust
use oar_ocr::prelude::*;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build OCR pipeline with required models
    let mut ocr = OAROCRBuilder::new(
        "detection_model.onnx".to_string(),
        "recognition_model.onnx".to_string(),
        "char_dict.txt".to_string(),
    ).build()?;

    // Process an image
    let result = ocr.predict(Path::new("document.jpg"))?;

    // Print extracted text with confidence scores
    for (text, score) in result.rec_texts.iter().zip(result.rec_scores.iter()) {
        println!("Text: {} (confidence: {:.2})", text, score);
    }

    Ok(())
}
```

This example creates an OCR pipeline using pre-trained models for text detection and recognition. The pipeline processes the input image and returns the recognized text along with confidence scores.

### Running Examples

The library includes several examples to help you get started:

```bash
# Complete OCR pipeline example
cargo run --example oarocr_pipeline -- \
    --text-detection-model path/to/detection_model.onnx \
    --text-recognition-model path/to/recognition_model.onnx \
    --char-dict path/to/char_dict.txt \
    --output-dir ./visualizations \
    --font-path path/to/font.ttf \
    image1.jpg image2.png

# Text recognition example
cargo run --example text_recognition -- \
    --model path/to/recognition_model.onnx \
    --char-dict path/to/char_dict.txt \
    text_crop1.jpg text_crop2.jpg
```

## Pre-trained Models

OAR OCR provides several pre-trained models for different OCR tasks. Download them from the [GitHub Releases](https://github.com/GreatV/oar-ocr/releases/tag/v0.1.0) page.

### Text Detection Models

Choose between mobile and server variants based on your needs:

- **Mobile**: Smaller, faster models suitable for real-time applications
- **Server**: Larger, more accurate models for high-precision requirements

| Model Version | Model Type | File Name | Size | Download Link |
|---------------|------------|-----------|------|---------------|
| PPOCRv4 | Mobile | `ppocrv4_mobile_det.onnx` | 4.8MB | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv4_mobile_det.onnx) |
| PPOCRv4 | Server | `ppocrv4_server_det.onnx` | 113.2MB | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv4_server_det.onnx) |
| PPOCRv5 | Mobile | `ppocrv5_mobile_det.onnx` | 4.8MB | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_mobile_det.onnx) |
| PPOCRv5 | Server | `ppocrv5_server_det.onnx` | 87.7MB | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_server_det.onnx) |

### Text Recognition Models

Recognition models are available in multiple versions and languages:

#### Chinese/General Models

| Model Version | Model Type | File Name | Size | Download Link |
|---------------|------------|-----------|------|---------------|
| PPOCRv4 | Mobile | `ppocrv4_mobile_rec.onnx` | 10.8MB | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv4_mobile_rec.onnx) |
| PPOCRv4 | Server | `ppocrv4_server_rec.onnx` | 90.4MB | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv4_server_rec.onnx) |
| PPOCRv4 | Server (Document) | `ppocrv4_server_rec_doc.onnx` | 94.7MB | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv4_server_rec_doc.onnx) |
| PPOCRv5 | Mobile | `ppocrv5_mobile_rec.onnx` | 16.5MB | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_mobile_rec.onnx) |
| PPOCRv5 | Server | `ppocrv5_server_rec.onnx` | 84.1MB | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_server_rec.onnx) |

#### Language-Specific Models

| Language | Model Version | File Name | Size | Download Link |
|----------|---------------|-----------|------|---------------|
| English | PPOCRv4 | `en_ppocrv4_mobile_rec.onnx` | 7.7MB | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/en_ppocrv4_mobile_rec.onnx) |
| Eastern Slavic | PPOCRv5 | `eslav_ppocrv5_mobile_rec.onnx` | 7.9MB | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/eslav_ppocrv5_mobile_rec.onnx) |
| Korean | PPOCRv5 | `korean_ppocrv5_mobile_rec.onnx` | 13.4MB | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/korean_ppocrv5_mobile_rec.onnx) |
| Latin | PPOCRv5 | `latin_ppocrv5_mobile_rec.onnx` | 7.9MB | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/latin_ppocrv5_mobile_rec.onnx) |

### Character Dictionaries

Character dictionaries are required for text recognition models. Choose the appropriate dictionary for your models:

#### General Dictionaries

| Dictionary | File Name | Description | Download Link |
|------------|-----------|-------------|---------------|
| PPOCRv4 Document | `ppocrv4_doc_dict.txt` | For PPOCRv4 document models | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv4_doc_dict.txt) |
| PPOCRv5 | `ppocrv5_dict.txt` | For PPOCRv5 models | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_dict.txt) |
| PPOCR Keys v1 | `ppocr_keys_v1.txt` | For older PPOCR models | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocr_keys_v1.txt) |

#### Language-Specific Dictionaries

| Language | File Name | Description | Download Link |
|----------|-----------|-------------|---------------|
| English | `en_dict.txt` | For PPOCRv4 English recognition models | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/en_dict.txt) |
| Eastern Slavic | `ppocrv5_eslav_dict.txt` | For PPOCRv5 Eastern Slavic models | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_eslav_dict.txt) |
| Korean | `ppocrv5_korean_dict.txt` | For PPOCRv5 Korean models | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_korean_dict.txt) |
| Latin | `ppocrv5_latin_dict.txt` | For PPOCRv5 Latin script models | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_latin_dict.txt) |

### Optional Models

These models provide additional functionality for specialized use cases:

| Model Type | File Name | Size | Description | Download Link |
|------------|-----------|------|-------------|---------------|
| Document Orientation | `pplcnet_x1_0_doc_ori.onnx` | 6.7MB | Detect document rotation | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/pplcnet_x1_0_doc_ori.onnx) |
| Text Line Orientation (Light) | `pplcnet_x0_25_textline_ori.onnx` | 988KB | Detect text line orientation | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/pplcnet_x0_25_textline_ori.onnx) |
| Text Line Orientation (Standard) | `pplcnet_x1_0_textline_ori.onnx` | 6.7MB | Detect text line orientation | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/pplcnet_x1_0_textline_ori.onnx) |
| Document Rectification | `uvdoc.onnx` | 31.6MB | Fix perspective distortion | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/uvdoc.onnx) |

## Acknowledgments

This project builds upon the excellent work of several open-source projects:

- **[ort](https://github.com/pykeio/ort)**: Rust bindings for ONNX Runtime by pykeio. This crate provides the Rust interface to ONNX Runtime that powers the efficient inference engine in this OCR library.

- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)**: Baidu's awesome multilingual OCR toolkits based on PaddlePaddle. This project utilizes PaddleOCR's pre-trained models, which provide excellent accuracy and performance for text detection and recognition across multiple languages.
