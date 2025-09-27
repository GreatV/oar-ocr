# OAR (ONNXRuntime And Rust) OCR

![Crates.io Version](https://img.shields.io/crates/v/oar-ocr)
![Crates.io Downloads (recent)](https://img.shields.io/crates/dr/oar-ocr)
[![dependency status](https://deps.rs/repo/github/GreatV/oar-ocr/status.svg)](https://deps.rs/repo/github/GreatV/oar-ocr)
![GitHub License](https://img.shields.io/github/license/GreatV/oar-ocr)

A comprehensive OCR (Optical Character Recognition) library, built in Rust with ONNX Runtime for efficient inference.

## Quick Start

### Installation

Add OAROCR to your project's `Cargo.toml`:

```bash
cargo add oar-ocr
```

For CUDA support, add with the `cuda` feature:

```bash
cargo add oar-ocr --features cuda
```

Or manually add it to your `Cargo.toml`:

```toml
[dependencies]
oar-ocr = "0.2"

# For CUDA support
oar-ocr = { version = "0.2", features = ["cuda"] }
```

### Basic Usage

Here's a simple example of how to use OAROCR to extract text from an image:

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
    let image = oar_ocr::utils::load_image(Path::new("document.jpg"))?;
    let results = ocr.predict(&[image])?;
    let result = &results[0];

    // Print extracted text with confidence scores using the modern TextRegion API
    for text_region in &result.text_regions {
        if let (Some(text), Some(confidence)) = (&text_region.text, text_region.confidence) {
            println!("Text: {} (confidence: {:.2})", text, confidence);
        }
    }

    // Process multiple images at once
    let images = oar_ocr::utils::load_images(&[
        Path::new("document1.jpg"),
        Path::new("document2.jpg"),
        Path::new("document3.jpg"),
    ])?;
    let results = ocr.predict(&images)?;

    for result in results {
        println!("Image {}: {} text regions found", result.index, result.text_regions.len());
        for text_region in &result.text_regions {
            if let (Some(text), Some(confidence)) = (&text_region.text, text_region.confidence) {
                println!("  Text: {} (confidence: {:.2})", text, confidence);
            }
        }
    }

    Ok(())
}
```

This example creates an OCR pipeline using pre-trained models for text detection and recognition. The pipeline processes the input image and returns structured `TextRegion` objects containing the recognized text, confidence scores, and bounding boxes for each detected text region.

### Using CUDA for GPU Acceleration

For better performance, you can enable CUDA support to run inference on GPU:

```rust
use oar_ocr::prelude::*;
use oar_ocr::core::config::onnx::{OrtSessionConfig, OrtExecutionProvider};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure CUDA execution provider for GPU acceleration
    let ort_config = OrtSessionConfig::new()
        .with_execution_providers(vec![
            OrtExecutionProvider::CUDA {
                device_id: Some(0),  // Use GPU 0
                gpu_mem_limit: None,
                arena_extend_strategy: None,
                cudnn_conv_algo_search: None,
                do_copy_in_default_stream: None,
                cudnn_conv_use_max_workspace: None,
            },
            OrtExecutionProvider::CPU,  // Fallback to CPU if CUDA fails
        ]);

    // Build OCR pipeline with CUDA support
    let ocr = OAROCRBuilder::new(
        "detection_model.onnx".to_string(),
        "recognition_model.onnx".to_string(),
        "char_dict.txt".to_string(),
    )
    .global_ort_session(ort_config)  // Apply CUDA config to all components
    .build()?;

    // Process images (same as CPU example)
    let image = oar_ocr::utils::load_image(Path::new("document.jpg"))?;
    let results = ocr.predict(&[image])?;
    let result = &results[0];

    // Extract text from results
    for text_region in &result.text_regions {
        if let (Some(text), Some(confidence)) = (&text_region.text, text_region.confidence) {
            println!("Text: {} (confidence: {:.2})", text, confidence);
        }
    }

    Ok(())
}
```

**Note**: To use CUDA support, you need to:

1. Install oar-ocr with CUDA feature: `cargo add oar-ocr --features cuda`
2. Have CUDA toolkit and cuDNN installed on your system
3. Ensure your ONNX models are compatible with CUDA execution

## Pre-trained Models

OAROCR provides several pre-trained models for different OCR tasks. Download them from the [GitHub Releases](https://github.com/GreatV/oar-ocr/releases) page.

### Text Detection Models

Choose between mobile and server variants based on your needs:

- **Mobile**: Smaller, faster models suitable for real-time applications
- **Server**: Larger, more accurate models for high-precision requirements

| Model Type     | Version  | Category | Model File                                                                                                      | Size    | Description                                    |
|----------------|----------|----------|-----------------------------------------------------------------------------------------------------------------|---------|------------------------------------------------|
| Text Detection | PP-OCRv4 | Mobile   | [`ppocrv4_mobile_det.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv4_mobile_det.onnx) | 4.8MB   | Mobile variant for real-time applications      |
| Text Detection | PP-OCRv4 | Server   | [`ppocrv4_server_det.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv4_server_det.onnx) | 113.2MB | Server variant for high-precision requirements |
| Text Detection | PP-OCRv5 | Mobile   | [`ppocrv5_mobile_det.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_mobile_det.onnx) | 4.8MB   | Mobile variant for real-time applications      |
| Text Detection | PP-OCRv5 | Server   | [`ppocrv5_server_det.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_server_det.onnx) | 87.7MB  | Server variant for high-precision requirements |

### Text Recognition Models

Recognition models are available in multiple versions and languages:

#### Chinese/General Models

| Model Type       | Version  | Language/Category | Model File                                                                                                              | Size   | Description                      |
|------------------|----------|-------------------|-------------------------------------------------------------------------------------------------------------------------|--------|----------------------------------|
| Text Recognition | PP-OCRv4 | Chinese/General   | [`ppocrv4_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv4_mobile_rec.onnx)         | 10.8MB | Mobile variant                   |
| Text Recognition | PP-OCRv4 | Chinese/General   | [`ppocrv4_server_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv4_server_rec.onnx)         | 90.4MB | Server variant                   |
| Text Recognition | PP-OCRv4 | Chinese/General   | [`ppocrv4_server_rec_doc.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv4_server_rec_doc.onnx) | 94.7MB | Server variant for document text |
| Text Recognition | PP-OCRv5 | Chinese/General   | [`ppocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_mobile_rec.onnx)         | 16.5MB | Mobile variant                   |
| Text Recognition | PP-OCRv5 | Chinese/General   | [`ppocrv5_server_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_server_rec.onnx)         | 84.1MB | Server variant                   |

#### Language-Specific Models

| Model Type       | Version  | Language       | Model File                                                                                                                    | Size   | Description             |
|------------------|----------|----------------|-------------------------------------------------------------------------------------------------------------------------------|--------|-------------------------|
| Text Recognition | PP-OCRv4 | English        | [`en_ppocrv4_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/en_ppocrv4_mobile_rec.onnx)         | 7.7MB  | Language-specific model |
| Text Recognition | PP-OCRv5 | Eastern Slavic | [`eslav_ppocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/eslav_ppocrv5_mobile_rec.onnx)   | 7.9MB  | Language-specific model |
| Text Recognition | PP-OCRv5 | Korean         | [`korean_ppocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/korean_ppocrv5_mobile_rec.onnx) | 13.4MB | Language-specific model |
| Text Recognition | PP-OCRv5 | Latin          | [`latin_ppocrv5_mobile_rec.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/latin_ppocrv5_mobile_rec.onnx)   | 7.9MB  | Language-specific model |

### Character Dictionaries

Character dictionaries are required for text recognition models. Choose the appropriate dictionary for your models:

#### General Dictionaries

| File Type            | Version        | Category | Model File                                                                                                | Size | Description                  |
|----------------------|----------------|----------|-----------------------------------------------------------------------------------------------------------|------|------------------------------|
| Character Dictionary | PP-OCRv4       | Document | [`ppocrv4_doc_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv4_doc_dict.txt) | -    | For PP-OCRv4 document models |
| Character Dictionary | PP-OCRv5       | General  | [`ppocrv5_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_dict.txt)         | -    | For PP-OCRv5 models          |
| Character Dictionary | PP-OCR Keys v1 | General  | [`ppocr_keys_v1.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocr_keys_v1.txt)       | -    | For older PP-OCR models      |

#### Language-Specific Dictionaries

| File Type            | Version  | Language       | Model File                                                                                                      | Size | Description                             |
|----------------------|----------|----------------|-----------------------------------------------------------------------------------------------------------------|------|-----------------------------------------|
| Character Dictionary | PP-OCRv4 | English        | [`en_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/en_dict.txt)                         | -    | For PP-OCRv4 English recognition models |
| Character Dictionary | PP-OCRv5 | Eastern Slavic | [`ppocrv5_eslav_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_eslav_dict.txt)   | -    | For PP-OCRv5 Eastern Slavic models      |
| Character Dictionary | PP-OCRv5 | Korean         | [`ppocrv5_korean_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_korean_dict.txt) | -    | For PP-OCRv5 Korean models              |
| Character Dictionary | PP-OCRv5 | Latin          | [`ppocrv5_latin_dict.txt`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/ppocrv5_latin_dict.txt)   | -    | For PP-OCRv5 Latin script models        |

### Optional Models

These models provide additional functionality for specialized use cases:

| Model Type             | Version | Category | Model File                                                                                                                      | Size   | Description                  |
|------------------------|---------|----------|---------------------------------------------------------------------------------------------------------------------------------|--------|------------------------------|
| Document Orientation   | PPLCNet | -        | [`pplcnet_x1_0_doc_ori.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/pplcnet_x1_0_doc_ori.onnx)             | 6.7MB  | Detect document rotation     |
| Text Line Orientation  | PPLCNet | Light    | [`pplcnet_x0_25_textline_ori.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/pplcnet_x0_25_textline_ori.onnx) | 988KB  | Detect text line orientation |
| Text Line Orientation  | PPLCNet | Standard | [`pplcnet_x1_0_textline_ori.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/pplcnet_x1_0_textline_ori.onnx)   | 6.7MB  | Detect text line orientation |
| Document Rectification | UVDoc   | -        | [`uvdoc.onnx`](https://github.com/GreatV/oar-ocr/releases/download/v0.1.0/uvdoc.onnx)                                           | 31.6MB | Fix perspective distortion   |

## Acknowledgments

This project builds upon the excellent work of several open-source projects:

- **[ort](https://github.com/pykeio/ort)**: Rust bindings for ONNX Runtime by pykeio. This crate provides the Rust interface to ONNX Runtime that powers the efficient inference engine in this OCR library.

- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)**: Baidu's awesome multilingual OCR toolkits based on PaddlePaddle. This project utilizes PaddleOCR's pre-trained models, which provide excellent accuracy and performance for text detection and recognition across multiple languages.
