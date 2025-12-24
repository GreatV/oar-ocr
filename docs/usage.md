# Usage Guide

This guide covers the detailed usage of OAROCR for text recognition and document structure analysis.

## Basic OCR Pipeline

### Simple Usage

```rust
use oar_ocr::prelude::*;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Build OCR pipeline with required models
    let ocr = OAROCRBuilder::new(
        "pp-ocrv5_mobile_det.onnx",
        "pp-ocrv5_mobile_rec.onnx",
        "ppocrv5_dict.txt",
    )
    .build()?;

    // Process a single image
    let image = load_image(Path::new("document.jpg"))?;
    let results = ocr.predict(vec![image])?;
    let result = &results[0];

    // Print extracted text with confidence scores
    for text_region in &result.text_regions {
        if let Some((text, confidence)) = text_region.text_with_confidence() {
            println!("Text: {} (confidence: {:.2})", text, confidence);
        }
    }

    Ok(())
}
```

### Batch Processing

```rust
// Process multiple images at once (accepts &str paths)
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
```

## Builder APIs

OAROCR provides two high-level builder APIs for easy pipeline construction.

### OAROCRBuilder - Text Recognition Pipeline

The `OAROCRBuilder` provides a fluent API for building OCR pipelines with optional components:

```rust
use oar_ocr::oarocr::OAROCRBuilder;

// Basic OCR pipeline
let ocr = OAROCRBuilder::new(
    "pp-ocrv5_mobile_det.onnx",
    "pp-ocrv5_mobile_rec.onnx",
    "ppocrv5_dict.txt",
)
.build()?;

// OCR with optional preprocessing
let ocr = OAROCRBuilder::new(
    "pp-ocrv5_mobile_det.onnx",
    "pp-ocrv5_mobile_rec.onnx",
    "ppocrv5_dict.txt",
)
.with_document_image_orientation_classification("pp-lcnet_x1_0_doc_ori.onnx")
.with_text_line_orientation_classification("pp-lcnet_x1_0_textline_ori.onnx")
.with_document_image_rectification("uvdoc.onnx")
.image_batch_size(4)
.region_batch_size(64)
.build()?;
```

#### Available Options

| Method | Description |
|--------|-------------|
| `.with_document_image_orientation_classification(path)` | Add document orientation detection |
| `.with_text_line_orientation_classification(path)` | Add text line orientation detection |
| `.with_document_image_rectification(path)` | Add document rectification (UVDoc) |
| `.text_type("seal")` | Optimize pipeline for curved seal/stamp text |
| `.return_word_box(true)` | Enable word-level bounding boxes |
| `.image_batch_size(n)` | Set batch size for image processing |
| `.region_batch_size(n)` | Set batch size for region processing |
| `.ort_session(config)` | Apply ONNX Runtime configuration |

### OARStructureBuilder - Document Structure Analysis

The `OARStructureBuilder` enables document structure analysis with layout detection, table recognition, and formula extraction:

```rust
use oar_ocr::oarocr::OARStructureBuilder;

// Basic layout detection
let structure = OARStructureBuilder::new("picodet-l_layout_17cls.onnx")
    .build()?;

// Full document structure analysis
let structure = OARStructureBuilder::new("picodet-l_layout_17cls.onnx")
    .with_table_classification("pp-lcnet_x1_0_table_cls.onnx")
    .with_table_cell_detection("rt-detr-l_wired_table_cell_det.onnx", "wired")
    .with_table_structure_recognition("slanext_wired.onnx", "wired")
    .table_structure_dict_path("table_structure_dict_ch.txt")
    .with_formula_recognition("pp-formulanet-l.onnx", "unimernet_tokenizer.json", "pp_formulanet")
    .build()?;

// Structure analysis with integrated OCR
let structure = OARStructureBuilder::new("picodet-l_layout_17cls.onnx")
    .with_table_classification("pp-lcnet_x1_0_table_cls.onnx")
    .with_ocr("pp-ocrv5_mobile_det.onnx", "pp-ocrv5_mobile_rec.onnx", "ppocrv5_dict.txt")
    .build()?;
```

#### Available Options

| Method | Description |
|--------|-------------|
| `.with_table_classification(path)` | Add wired/wireless table classification |
| `.with_table_cell_detection(path, type)` | Add table cell detection |
| `.with_table_structure_recognition(path, type)` | Add table structure recognition |
| `.table_structure_dict_path(path)` | Set table structure dictionary |
| `.with_formula_recognition(model, tokenizer, type)` | Add formula recognition |
| `.with_ocr(det, rec, dict)` | Add integrated OCR pipeline |
| `.with_seal_detection(path)` | Add seal/stamp text detection |
| `.image_batch_size(n)` | Set batch size for image processing |
| `.region_batch_size(n)` | Set batch size for region processing |
| `.ort_session(config)` | Apply ONNX Runtime configuration |

## GPU Acceleration

### CUDA

Enable CUDA support for GPU inference:

```rust
use oar_ocr::prelude::*;
use oar_ocr::core::config::{OrtSessionConfig, OrtExecutionProvider};
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Configure CUDA execution provider
    let ort_config = OrtSessionConfig::new()
        .with_execution_providers(vec![
            OrtExecutionProvider::CUDA {
                device_id: Some(0),
                gpu_mem_limit: None,
                arena_extend_strategy: None,
                cudnn_conv_algo_search: None,
                cudnn_conv_use_max_workspace: None,
            },
            OrtExecutionProvider::CPU,  // Fallback
        ]);

    // Build OCR pipeline with CUDA
    let ocr = OAROCRBuilder::new(
        "pp-ocrv5_mobile_det.onnx",
        "pp-ocrv5_mobile_rec.onnx",
        "ppocrv5_dict.txt",
    )
    .ort_session(ort_config)
    .build()?;

    // Use as normal
    let image = load_image(Path::new("document.jpg"))?;
    let results = ocr.predict(vec![image])?;

    Ok(())
}
```

**Requirements:**

1. Install with CUDA feature: `cargo add oar-ocr --features cuda`
2. CUDA toolkit and cuDNN installed on your system
3. ONNX models compatible with CUDA execution

### Other Execution Providers

OAROCR supports multiple execution providers via feature flags:

| Feature | Provider | Platform |
|---------|----------|----------|
| `cuda` | NVIDIA CUDA | Linux, Windows |
| `tensorrt` | NVIDIA TensorRT | Linux, Windows |
| `directml` | DirectML | Windows |
| `coreml` | Core ML | macOS, iOS |
| `openvino` | Intel OpenVINO | Linux, Windows |
| `webgpu` | WebGPU | Cross-platform |

Example with TensorRT:

```rust
let ort_config = OrtSessionConfig::new()
    .with_execution_providers(vec![
        OrtExecutionProvider::TensorRT {
            device_id: Some(0),
            max_workspace_size: None,
            min_subgraph_size: None,
            fp16_enable: None,
        },
        OrtExecutionProvider::CUDA {
            device_id: Some(0),
            gpu_mem_limit: None,
            arena_extend_strategy: None,
            cudnn_conv_algo_search: None,
            cudnn_conv_use_max_workspace: None,
        },
        OrtExecutionProvider::CPU,
    ]);
```

## Configuration Options

### OrtSessionConfig

Control ONNX Runtime session behavior:

```rust
use oar_ocr::core::config::{OrtSessionConfig, OrtExecutionProvider};

let config = OrtSessionConfig::new()
    .with_execution_providers(vec![OrtExecutionProvider::CPU])
    .with_intra_threads(4)
    .with_inter_threads(2);
```

### Task-Specific Configs

Each task has its own configuration struct that can be customized:

```rust
use oar_ocr::domain::TextDetectionConfig;

let det_config = TextDetectionConfig {
    score_threshold: 0.3,
    box_threshold: 0.6,
    unclip_ratio: 1.5,
    max_candidates: 1000,
    ..Default::default()
};
```
