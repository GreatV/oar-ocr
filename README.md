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
oar-ocr = "0.2"
```

### Basic Usage

Here's a simple example of how to use OAR OCR to extract text from an image:

```rust
use oar_ocr::prelude::*;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build OCR pipeline with required models
    let ocr = OAROCRBuilder::new(
        "detection_model.onnx".to_string(),
        "recognition_model.onnx".to_string(),
        "char_dict.txt".to_string(),
    ).build()?;

    // Process a single image
    let results = ocr.predict(&[Path::new("document.jpg")])?;
    let result = &results[0];

    // Print extracted text with confidence scores
    for (text, score) in result.rec_texts.iter().zip(result.rec_scores.iter()) {
        println!("Text: {} (confidence: {:.2})", text, score);
    }

    // Process multiple images at once
    let results = ocr.predict(&[
        Path::new("document1.jpg"),
        Path::new("document2.jpg"),
        Path::new("document3.jpg"),
    ])?;

    for result in results {
        println!("Image {}: {} text regions found", result.index, result.text_boxes.len());
        for (text, score) in result.rec_texts.iter().zip(result.rec_scores.iter()) {
            println!("  Text: {} (confidence: {:.2})", text, score);
        }
    }

    Ok(())
}
```

This example creates an OCR pipeline using pre-trained models for text detection and recognition. The pipeline processes the input image and returns the recognized text along with confidence scores.

## Pre-trained Models

OAR OCR provides several pre-trained models for different OCR tasks. Download them from the [GitHub Releases](https://github.com/GreatV/oar-ocr/releases) page.

### Text Detection Models

Choose between mobile and server variants based on your needs:

- **Mobile**: Smaller, faster models suitable for real-time applications
- **Server**: Larger, more accurate models for high-precision requirements

| Model Type | Version | Language/Category | File Name | Size | Description | Download Link |
|------------|---------|-------------------|-----------|------|-------------|---------------|
| Text Detection | PPOCRv4 | Mobile | `ppocrv4_mobile_det.onnx` | 4.8MB | Mobile variant for real-time applications | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv4_mobile_det.onnx) |
| Text Detection | PPOCRv4 | Server | `ppocrv4_server_det.onnx` | 113.2MB | Server variant for high-precision requirements | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv4_server_det.onnx) |
| Text Detection | PPOCRv5 | Mobile | `ppocrv5_mobile_det.onnx` | 4.8MB | Mobile variant for real-time applications | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_mobile_det.onnx) |
| Text Detection | PPOCRv5 | Server | `ppocrv5_server_det.onnx` | 87.7MB | Server variant for high-precision requirements | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_server_det.onnx) |

### Text Recognition Models

Recognition models are available in multiple versions and languages:

#### Chinese/General Models

| Model Type | Version | Language/Category | File Name | Size | Description | Download Link |
|------------|---------|-------------------|-----------|------|-------------|---------------|
| Text Recognition | PPOCRv4 | Chinese/General | `ppocrv4_mobile_rec.onnx` | 10.8MB | Mobile variant | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv4_mobile_rec.onnx) |
| Text Recognition | PPOCRv4 | Chinese/General | `ppocrv4_server_rec.onnx` | 90.4MB | Server variant | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv4_server_rec.onnx) |
| Text Recognition | PPOCRv4 | Chinese/General | `ppocrv4_server_rec_doc.onnx` | 94.7MB | Server variant for document text | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv4_server_rec_doc.onnx) |
| Text Recognition | PPOCRv5 | Chinese/General | `ppocrv5_mobile_rec.onnx` | 16.5MB | Mobile variant | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_mobile_rec.onnx) |
| Text Recognition | PPOCRv5 | Chinese/General | `ppocrv5_server_rec.onnx` | 84.1MB | Server variant | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_server_rec.onnx) |

#### Language-Specific Models

| Model Type | Version | Language/Category | File Name | Size | Description | Download Link |
|------------|---------|-------------------|-----------|------|-------------|---------------|
| Text Recognition | PPOCRv4 | English | `en_ppocrv4_mobile_rec.onnx` | 7.7MB | Language-specific model | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/en_ppocrv4_mobile_rec.onnx) |
| Text Recognition | PPOCRv5 | Eastern Slavic | `eslav_ppocrv5_mobile_rec.onnx` | 7.9MB | Language-specific model | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/eslav_ppocrv5_mobile_rec.onnx) |
| Text Recognition | PPOCRv5 | Korean | `korean_ppocrv5_mobile_rec.onnx` | 13.4MB | Language-specific model | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/korean_ppocrv5_mobile_rec.onnx) |
| Text Recognition | PPOCRv5 | Latin | `latin_ppocrv5_mobile_rec.onnx` | 7.9MB | Language-specific model | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/latin_ppocrv5_mobile_rec.onnx) |

### Character Dictionaries

Character dictionaries are required for text recognition models. Choose the appropriate dictionary for your models:

#### General Dictionaries

| Model Type | Version | Language/Category | File Name | Size | Description | Download Link |
|------------|---------|-------------------|-----------|------|-------------|---------------|
| Character Dictionary | PPOCRv4 | Document | `ppocrv4_doc_dict.txt` | N/A | For PPOCRv4 document models | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv4_doc_dict.txt) |
| Character Dictionary | PPOCRv5 | General | `ppocrv5_dict.txt` | N/A | For PPOCRv5 models | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_dict.txt) |
| Character Dictionary | PPOCR Keys v1 | General | `ppocr_keys_v1.txt` | N/A | For older PPOCR models | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocr_keys_v1.txt) |

#### Language-Specific Dictionaries

| Model Type | Version | Language/Category | File Name | Size | Description | Download Link |
|------------|---------|-------------------|-----------|------|-------------|---------------|
| Character Dictionary | PPOCRv4/PPOCRv5 | English | `en_dict.txt` | N/A | For PPOCRv4 English recognition models | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/en_dict.txt) |
| Character Dictionary | PPOCRv5 | Eastern Slavic | `ppocrv5_eslav_dict.txt` | N/A | For PPOCRv5 Eastern Slavic models | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_eslav_dict.txt) |
| Character Dictionary | PPOCRv5 | Korean | `ppocrv5_korean_dict.txt` | N/A | For PPOCRv5 Korean models | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_korean_dict.txt) |
| Character Dictionary | PPOCRv5 | Latin | `ppocrv5_latin_dict.txt` | N/A | For PPOCRv5 Latin script models | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_latin_dict.txt) |

### Optional Models

These models provide additional functionality for specialized use cases:

| Model Type | Version | Language/Category | File Name | Size | Description | Download Link |
|------------|---------|-------------------|-----------|------|-------------|---------------|
| Document Orientation | PPLCNet | N/A | `pplcnet_x1_0_doc_ori.onnx` | 6.7MB | Detect document rotation | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/pplcnet_x1_0_doc_ori.onnx) |
| Text Line Orientation | PPLCNet | Light | `pplcnet_x0_25_textline_ori.onnx` | 988KB | Detect text line orientation | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/pplcnet_x0_25_textline_ori.onnx) |
| Text Line Orientation | PPLCNet | Standard | `pplcnet_x1_0_textline_ori.onnx` | 6.7MB | Detect text line orientation | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/pplcnet_x1_0_textline_ori.onnx) |
| Document Rectification | UVDoc | N/A | `uvdoc.onnx` | 31.6MB | Fix perspective distortion | [Download](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/uvdoc.onnx) |

## Acknowledgments

This project builds upon the excellent work of several open-source projects:

- **[ort](https://github.com/pykeio/ort)**: Rust bindings for ONNX Runtime by pykeio. This crate provides the Rust interface to ONNX Runtime that powers the efficient inference engine in this OCR library.

- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)**: Baidu's awesome multilingual OCR toolkits based on PaddlePaddle. This project utilizes PaddleOCR's pre-trained models, which provide excellent accuracy and performance for text detection and recognition across multiple languages.
