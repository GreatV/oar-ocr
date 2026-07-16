# OAR-OCR

[![Crates.io Version](https://img.shields.io/crates/v/oar-ocr)](https://crates.io/crates/oar-ocr)
![Crates.io Downloads (recent)](https://img.shields.io/crates/dr/oar-ocr)
[![dependency status](https://deps.rs/repo/github/GreatV/oar-ocr/status.svg)](https://deps.rs/repo/github/GreatV/oar-ocr)
![GitHub License](https://img.shields.io/github/license/GreatV/oar-ocr)

A native Rust toolkit for OCR, document layout analysis, and vision-language document understanding.

## Highlights

- End-to-end text detection and recognition with PP-OCR models, including PP-OCRv6.
- Document structure analysis for layout, tables, formulas, seals, orientation, and rectification.
- Native Candle inference for compact document VLMs through the `oar-ocr-vl` crate.
- CPU and GPU execution, model auto-download, and in-memory ONNX model loading.

## Quick Start

### Installation

```bash
cargo add oar-ocr
```

The default build enables ONNX Runtime binary downloads and SIMD acceleration. Add only the optional capabilities needed by your application. For example:

```bash
cargo add oar-ocr --features cuda,auto-download
```

This keeps the default `download-binaries` and `simd` features enabled, makes the ONNX Runtime CUDA execution provider available for selection, and downloads missing registered model files from ModelScope into `$OAR_HOME` when they are first used.

See the [Cargo feature guide](docs/features.md) for all available features and the [model guide](docs/models.md#auto-download) for model download and cache behavior.

Builders also accept raw ONNX bytes such as `include_bytes!`, allowing models to be embedded in a single binary. See [Loading Models from Memory](docs/usage.md#loading-models-from-memory).

### OCR Pipeline

With `auto-download`, pass registered model names directly. Otherwise, replace them with local paths.

```rust
use oar_ocr::domain::tasks::TextDetectionConfig;
use oar_ocr::prelude::*;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ocr = OAROCRBuilder::new(
        "pp-ocrv6_tiny_det.onnx",
        "pp-ocrv6_tiny_rec.onnx",
        "ppocrv6_tiny_dict.txt",
    )
    .text_detection_config(TextDetectionConfig {
        score_threshold: 0.2,
        box_threshold: 0.45,
        unclip_ratio: 1.4,
        max_candidates: 3000,
        ..Default::default()
    })
    .build()?;

    let image = load_image(Path::new("document.jpg"))?;
    let results = ocr.predict(vec![image])?;

    for region in &results[0].text_regions {
        if let Some((text, confidence)) = region.text_with_confidence() {
            println!("{text} ({confidence:.2})");
        }
    }

    Ok(())
}
```

### Document Structure Analysis

```rust
use oar_ocr::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let structure = OARStructureBuilder::new("pp-doclayout_plus-l.onnx")
        .with_table_classification("pp-lcnet_x1_0_table_cls.onnx")
        .with_table_structure_recognition("slanet_plus.onnx", "wireless")
        .table_structure_dict_path("table_structure_dict_ch.txt")
        .with_ocr(
            "pp-ocrv5_mobile_det.onnx",
            "pp-ocrv5_mobile_rec.onnx",
            "ppocrv5_dict.txt",
        )
        .build()?;

    let result = structure.predict("document.jpg")?;
    println!("{}", result.to_markdown());

    Ok(())
}
```

## Supported Models

The classic pipeline runs ONNX models through ONNX Runtime and supports the following model families. See the [pre-trained model guide](docs/models.md) for exact checkpoints, dictionaries, download links, and auto-download names.

### Classic ONNX Models

| Task | Supported model families |
|---|---|
| Text detection | PP-OCRv4, PP-OCRv5, and PP-OCRv6 DB detectors |
| Text recognition | PP-OCRv3, PP-OCRv4, PP-OCRv5, PP-OCRv6, SVTRv2, and RepSVTR CTC recognizers |
| Document preprocessing | PP-LCNet document orientation, PP-LCNet text-line orientation, and UVDoc rectification |
| Layout detection | PicoDet, RT-DETR-H, PP-DocLayout S/M/L, PP-DocLayout Plus-L, PP-DocLayoutV2/V3, and PP-DocBlockLayout |
| Table analysis | PP-LCNet table classification, RT-DETR-L cell detection, and SLANet, SLANet+, and SLANeXt structure recognition |
| Formula recognition | PP-FormulaNet, PP-FormulaNet Plus, and UniMERNet |
| Seal text detection | PP-OCRv4 mobile and server seal detectors |

Available text-recognition checkpoints cover Chinese, Traditional Chinese, English, Arabic, Cyrillic, Devanagari, Greek, Eastern Slavic, Japanese, Georgian, Korean, Latin, Tamil, Telugu, and Thai scripts or languages.

### Vision-Language Models

The [`oar-ocr-vl`](oar-ocr-vl/README.md) crate provides native [Candle](https://github.com/huggingface/candle) inference for compact document VLMs on CPU, CUDA, and Metal.

| Model | Parameters | Capabilities |
|---|---:|---|
| [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) | 0.9B | Page parsing, text, table, formula, and chart recognition |
| [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) | 0.9B | PaddleOCR-VL tasks plus text spotting and seal recognition |
| [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | 0.9B | Region-aware page parsing and task-specific recognition |
| [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | 0.9B | Page parsing, text, table, and formula recognition |
| [OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) | 0.8B | Model-native full-page document-to-Markdown parsing |
| [MonkeyOCRv2-S-Parsing](https://huggingface.co/zenosai/MonkeyOCRv2-S-Parsing) | 0.6B | Model-native layout, end-to-end parsing, text, formula, and OTSL-table recognition |
| [HunyuanOCR 1.5 / 1.0](https://huggingface.co/tencent/HunyuanOCR) | 1B | Prompt-driven full-page parsing, text spotting, table, formula, and chart recognition, with optional DFlash decoding for 1.5 |
| [MinerU2.5-2509](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B) | 1.2B | Model-native two-step layout detection and content extraction |
| [MinerU2.5-Pro-2605](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B) | 1.2B | Newer MinerU2.5 checkpoint using the same two-step pipeline |
| [MinerU-Diffusion-V1-0320](https://huggingface.co/opendatalab/MinerU-Diffusion-V1-0320-2.5B) | 2.5B | Block-diffusion OCR with structured two-step extraction or single-pass text recognition |

PaddleOCR-VL variants and GLM-OCR integrate with the external-layout [`DocParser`](oar-ocr-vl/README.md#document-parsing-pipeline). OvisOCR2 and MonkeyOCRv2-S-Parsing instead provide model-native full-page paths through dedicated examples. HunyuanOCR and the MinerU models also use their model-native parsing pipelines.

See the [`oar-ocr-vl` guide](oar-ocr-vl/README.md) for setup and [`oar-ocr-vl/examples`](oar-ocr-vl/examples) for runnable examples.

## Documentation

- [Usage guide](docs/usage.md) — APIs, builder patterns, accelerators, and model loading
- [Cargo features](docs/features.md) — defaults, execution providers, and feature combinations
- [Pre-trained models](docs/models.md) — model files, dictionaries, and auto-download behavior
- [Environment variables](docs/environment-variables.md) — runtime and performance overrides
- [FAQ](docs/FAQ.md) — common build and runtime issues

## Examples

See the [usage guide](docs/usage.md) for other pipeline configurations and APIs. Complete classic-pipeline examples live in [`examples`](examples), while VLM examples live in [`oar-ocr-vl/examples`](oar-ocr-vl/examples).

## Acknowledgments

This project builds upon the excellent work of several open-source projects:

- **[ort](https://github.com/pykeio/ort)**: Rust bindings for ONNX Runtime by pykeio. This crate provides the Rust interface to ONNX Runtime that powers the efficient inference engine in this OCR library.

- **[PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)**: Baidu's awesome multilingual OCR toolkits based on PaddlePaddle. This project utilizes PaddleOCR's pre-trained models, which provide excellent accuracy and performance for text detection and recognition across multiple languages.

- **[Candle](https://github.com/huggingface/candle)**: A minimalist ML framework for Rust by Hugging Face. We use Candle to implement Vision-Language model inference.
