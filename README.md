# OAR (ONNXRuntime And Rust) OCR

![Crates.io Version](https://img.shields.io/crates/v/oar-ocr)
![Crates.io Downloads (recent)](https://img.shields.io/crates/dr/oar-ocr)
[![dependency status](https://deps.rs/repo/github/GreatV/oar-ocr/status.svg)](https://deps.rs/repo/github/GreatV/oar-ocr)
![GitHub License](https://img.shields.io/github/license/GreatV/oar-ocr)

A comprehensive OCR (Optical Character Recognition) and document understanding library, built in Rust with ONNX Runtime for efficient inference.

## Features

- End-to-end OCR pipeline (text detection → text recognition)
- Optional preprocessing: document orientation, text-line orientation, UVDoc rectification
- Document structure analysis (PP-StructureV3-style): layout, regions, tables, formulas, seals
- Typed configs for each task/model (serde-friendly)
- ONNX Runtime execution providers (CPU by default; CUDA/TensorRT/DirectML/CoreML/OpenVINO/WebGPU via features)
- Optional visualization helpers (feature `visualization`)

## Quick Start

### Installation

Add OAROCR to your project's `Cargo.toml`:

```bash
cargo add oar-ocr
```

Enable ONNX Runtime execution providers via crate features:

- `cuda`, `tensorrt`, `directml`, `coreml`, `openvino`, `webgpu`

For example, for CUDA support:

```bash
cargo add oar-ocr --features cuda
```

For visualization utilities (used by examples):

```bash
cargo add oar-ocr --features visualization
```

Or manually add it to your `Cargo.toml`:

```toml
[dependencies]
oar-ocr = "0.3"

# Example: CUDA + visualization
oar-ocr = { version = "0.3", features = ["cuda", "visualization"] }

# Other execution providers:
# oar-ocr = { version = "0.3", features = ["tensorrt"] }
# oar-ocr = { version = "0.3", features = ["directml"] }
# oar-ocr = { version = "0.3", features = ["coreml"] }
# oar-ocr = { version = "0.3", features = ["openvino"] }
# oar-ocr = { version = "0.3", features = ["webgpu"] }
```

### Basic Usage

Here's a simple example of how to use OAROCR to extract text from an image:

```rust
use oar_ocr::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build OCR pipeline with required models
    let ocr = OAROCRBuilder::new(
        "detection_model.onnx",
        "recognition_model.onnx",
        "char_dict.txt",
    )
    .build()?;

    // Process a single image
    let image = load_image("document.jpg")?;
    let results = ocr.predict(vec![image])?;
    let result = &results[0];

    // Print extracted text with confidence scores using the modern TextRegion API
    for text_region in &result.text_regions {
        if let Some((text, confidence)) = text_region.text_with_confidence() {
            println!("Text: {} (confidence: {:.2})", text, confidence);
        }
    }

    // Process multiple images at once
    let images = load_images(&[
        "document1.jpg",
        "document2.jpg",
        "document3.jpg",
    ])?;
    let results = ocr.predict(images)?;

    for result in results {
        println!("Image {}: {} text regions found", result.index, result.text_regions.len());
        for text_region in &result.text_regions {
            if let Some((text, confidence)) = text_region.text_with_confidence() {
                println!("  Text: {} (confidence: {:.2})", text, confidence);
            }
        }
    }

    Ok(())
}
```

This example creates an OCR pipeline using pre-trained models for text detection and recognition. The pipeline processes the input image and returns structured `TextRegion` objects containing the recognized text, confidence scores, and bounding boxes for each detected text region.

### High-Level Builder APIs

OAROCR provides two high-level builder APIs for easy pipeline construction:

#### OAROCRBuilder - Text Recognition Pipeline

The `OAROCRBuilder` provides a fluent API for building OCR pipelines with optional components:

```rust
use oar_ocr::oarocr::OAROCRBuilder;

// Basic OCR pipeline
let ocr = OAROCRBuilder::new(
    "models/det.onnx",
    "models/rec.onnx",
    "models/dict.txt"
)
.build()?;

// OCR with optional components
let ocr = OAROCRBuilder::new(
    "models/det.onnx",
    "models/rec.onnx",
    "models/dict.txt"
)
.with_document_image_orientation_classification("models/doc_orient.onnx")
.with_text_line_orientation_classification("models/line_orient.onnx")
.with_document_image_rectification("models/rectify.onnx")
.image_batch_size(4)
.region_batch_size(64)
.build()?;
```

Useful options:

- `.text_type("seal")` - optimized pipeline defaults for curved seal/stamp text
- `.return_word_box(true)` - enable word-level boxes from recognition output

#### OARStructureBuilder - Document Structure Analysis

The `OARStructureBuilder` enables document structure analysis with layout detection, table recognition, and formula extraction:

```rust
use oar_ocr::oarocr::OARStructureBuilder;

// Basic layout detection
let structure = OARStructureBuilder::new("models/layout.onnx")
    .build()?;

// Full document structure analysis with table and formula recognition
let structure = OARStructureBuilder::new("models/layout.onnx")
    .with_table_classification("models/table_cls.onnx")
    .with_table_cell_detection("models/table_cell.onnx", "wired")
    .with_table_structure_recognition("models/table_struct.onnx", "wired")
    .table_structure_dict_path("models/table_structure_dict_ch.txt")
    .with_formula_recognition("models/formula.onnx", "models/tokenizer.json", "pp_formulanet")
    .build()?;

// Structure analysis with integrated OCR
let structure = OARStructureBuilder::new("models/layout.onnx")
    .with_table_classification("models/table_cls.onnx")
    .with_ocr("models/det.onnx", "models/rec.onnx", "models/dict.txt")
    .build()?;
```

Both builders support:

- **Configuration**: Set task configs via typed structs (serde-friendly)
- **Batch/Concurrency**: Tune session pools via `image_batch_size` / `region_batch_size`
- **ONNX Runtime Settings**: Apply a shared `OrtSessionConfig` via `.ort_session(...)`
- **Validation**: Automatic validation with detailed errors

### Examples

This repository includes runnable CLI examples under `examples/` (they require model files). Use `--help` to see all options:

```bash
cargo run --example ocr -- --help
cargo run --example structure -- --help
```

### Using CUDA for GPU Acceleration

For better performance, you can enable CUDA support to run inference on GPU:

```rust
use oar_ocr::prelude::*;
use oar_ocr::core::config::{OrtSessionConfig, OrtExecutionProvider};

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
        "detection_model.onnx",
        "recognition_model.onnx",
        "char_dict.txt",
    )
    .ort_session(ort_config)  // Apply ORT config to all components
    .build()?;

    // Process images (same as CPU example)
    let image = load_image("document.jpg")?;
    let results = ocr.predict(vec![image])?;
    let result = &results[0];

    // Extract text from results
    for text_region in &result.text_regions {
        if let Some((text, confidence)) = text_region.text_with_confidence() {
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
4. (Optional) Use other execution providers via `tensorrt`, `directml`, `coreml`, `openvino`, `webgpu` features

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

### Document Structure Models

These models are typically used with `OARStructureBuilder` (layout, tables, formulas, seals). File names below match the presets used by the builders and examples; download them from the Releases page as needed.

| Component                  | Suggested Model File(s)                         | Notes |
|---------------------------|--------------------------------------------------|-------|
| Layout Detection          | `pp-doclayout_plus-l.onnx`                       | PP-DocLayout_plus-L (default preset) |
| Region Detection          | `pp-docblocklayout.onnx`                         | PP-DocBlockLayout, for hierarchical ordering |
| Table Classification      | `pp-lcnet_x1_0_table_cls.onnx`                   | Wired vs wireless table type |
| Table Cell Detection      | `rt-detr-l_wired_table_cell_det.onnx`            | Wired tables (RT-DETR) |
| Table Cell Detection      | `rt-detr-l_wireless_table_cell_det.onnx`         | Wireless tables (RT-DETR) |
| Table Structure Recognition | `slanext_wired.onnx`, `slanet_plus.onnx`       | Wired / wireless structure recognition |
| Table Structure Dictionary | `table_structure_dict_ch.txt`                  | Required when enabling table structure recognition |
| Formula Recognition       | `pp-formulanet_plus-l.onnx`, `unimernet.onnx`    | `with_formula_recognition(..., tokenizer.json, model_type)` |
| Formula Tokenizer         | `unimernet_tokenizer.json`                      | Must match the selected formula model |
| Seal Text Detection       | `pp-ocrv4_server_seal_det.onnx`                  | Seal/stamp text detection |

## Acknowledgments

This project builds upon the excellent work of several open-source projects:

- **[ort](https://github.com/pykeio/ort)**: Rust bindings for ONNX Runtime by pykeio. This crate provides the Rust interface to ONNX Runtime that powers the efficient inference engine in this OCR library.

- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)**: Baidu's awesome multilingual OCR toolkits based on PaddlePaddle. This project utilizes PaddleOCR's pre-trained models, which provide excellent accuracy and performance for text detection and recognition across multiple languages.
