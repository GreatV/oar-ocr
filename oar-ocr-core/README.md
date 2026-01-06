# oar-ocr-core

Foundational types, model abstractions, and task-specific predictors for the OAR OCR library.

`oar-ocr-core` is the engine room of the OAR OCR ecosystem. It provides the core traits and implementations for high-performance OCR pipelines, featuring ONNX-based inference, specialized image processing, and a decoupled architecture designed for extensibility and speed.

## Architecture

This crate implements a **three-layer architecture** to ensure modularity and maintainability:

1. **Models**: Low-level wrappers around ONNX Runtime sessions, handling raw tensor input/output.
2. **Adapters**: Traits and implementations that bridge raw model outputs to domain-specific types, handling pre- and post-processing logic.
3. **Tasks**: Semantic contracts that define what a predictor does (e.g., "Text Detection"), ensuring consistent APIs across different model implementations.

## Installation

Add `oar-ocr-core` to your project:

```bash
cargo add oar-ocr-core
```

### Feature Flags

| Feature | Description |
| :--- | :--- |
| `cuda` | Enable NVIDIA CUDA execution provider |
| `tensorrt` | Enable NVIDIA TensorRT execution provider |
| `directml` | Enable DirectML execution provider (Windows) |
| `coreml` | Enable Core ML execution provider (macOS/iOS) |
| `openvino` | Enable Intel OpenVINO execution provider |
| `webgpu` | Enable WebGPU execution provider |
| `visualization` | Enable drawing utilities for debugging |
| `download-binaries` | Automatically download ONNX Runtime binaries (default) |

## Quick Start

### Text Detection

Detect text regions in an image using a DBNet-based model:

```rust
use oar_ocr_core::predictors::TextDetectionPredictor;
use oar_ocr_core::utils::load_image;

// 1. Initialize the predictor
let predictor = TextDetectionPredictor::builder()
    .build("pp-ocrv5_mobile_det.onnx")?;

// 2. Load and process
let image = load_image("document.jpg")?;
let results = predictor.predict(vec![image])?;

// 3. Access results (detections for the first image)
for det in &results.detections[0] {
    println!("Box: {:?}, Score: {:.2}", det.bbox, det.score);
}
```

### Text Recognition

Recognize text from cropped image regions:

```rust
use oar_ocr_core::predictors::TextRecognitionPredictor;
use oar_ocr_core::utils::load_image;

let predictor = TextRecognitionPredictor::builder()
    .dict_path("ppocrv5_dict.txt")
    .build("pp-ocrv5_mobile_rec.onnx")?;

let image = load_image("text_line.jpg")?;
let results = predictor.predict(vec![image])?;

// Recognition returns results per input image
for (text, score) in results.texts.iter().zip(&results.scores) {
    println!("Text: {}, Confidence: {:.2}", text, score);
}
```

### Layout Analysis

Analyze the structure of a document to identify titles, tables, and figures:

```rust
use oar_ocr_core::predictors::LayoutDetectionPredictor;
use oar_ocr_core::domain::LayoutDetectionConfig;
use oar_ocr_core::utils::load_image;

let predictor = LayoutDetectionPredictor::builder()
    .model_name("pp-doclayoutv2")
    .with_config(LayoutDetectionConfig::with_pp_doclayoutv2_defaults())
    .build("pp-doclayoutv2.onnx")?;

let image = load_image("page.jpg")?;
let results = predictor.predict(vec![image])?;

for element in &results.elements[0] {
    println!("Type: {}, Score: {:.2}", element.element_type, element.score);
}
```

## Available Predictors

| Predictor | Description |
| :--- | :--- |
| `TextDetectionPredictor` | Locates text regions (polygons) in images. |
| `TextRecognitionPredictor` | Converts text regions into strings. |
| `LayoutDetectionPredictor` | Identifies semantic elements (Title, Table, Figure). |
| `TableStructureRecognitionPredictor` | Extracts HTML/Markdown structure from tables. |
| `TableCellDetectionPredictor` | Locates individual cells within a table. |
| `FormulaRecognitionPredictor` | Converts math formulas to LaTeX. |
| `DocumentOrientationPredictor` | Detects and corrects document rotation. |
| `DocumentRectificationPredictor` | Unwarps perspective or curved document images. |
| `SealTextDetectionPredictor` | Specialized detection for curved official stamps. |
