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
- Any supported VLM loads by its Hugging Face model ID and downloads on first use.
- One documented [page JSON format](docs/page-format.md) from both the classic and vision-language pipelines.
- The `oar` command-line tool for OCR, structure analysis, and VLM parsing of images and PDFs.

## Quick Start

### Command line

Install the standalone tool with `cargo install oar-ocr-cli` (or add `--features cuda`). Inputs can be images or PDFs. Models download automatically; `auto` selects an available compiled device.

```bash
oar ocr page.png
oar structure page.png -o documents
oar parse --model PaddlePaddle/PaddleOCR-VL-1.5 page.png
oar structure report.pdf --pages 1-3 --format json
```

See the [CLI guide](https://github.com/GreatV/oar-ocr/blob/main/oar-ocr-cli/README.md) for JSON output, model overrides, and local checkpoints.

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

The `pp_ocrv6` preset configures a PP-OCRv6 pipeline by model size, filling in the model names, the matching dictionary, and the official detection thresholds. With `auto-download`, the names resolve through the model registry; without it they resolve as local paths, so nothing changes for offline setups.

```rust
use oar_ocr::prelude::*;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ocr = OAROCRBuilder::pp_ocrv6(PpOcrV6Size::Small).build()?;

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

The preset's detection thresholds are defaults, so a `text_type` like `seal` still applies its own detection settings, and an explicit `text_detection_config` overrides everything. Sizes pick the models, the dictionary, and PaddleOCR's per-size box threshold (0.4 for Tiny, 0.45 for Small and Medium): `PpOcrV6Size::Tiny` runs the fastest pair over its reduced dictionary.

```rust
use oar_ocr::prelude::*;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let ocr = OAROCRBuilder::pp_ocrv6(PpOcrV6Size::Tiny).build()?;

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

The `pp_structurev3` preset configures the PP-StructureV3-style stack: PP-DocLayoutV3 layout, PP-OCRv6 Tiny text recognition, table classification, SLANeXt wired and SLANet+ wireless table structure, wired cell detection, and the table dictionary.

```rust
use oar_ocr::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let structure = OARStructureBuilder::pp_structurev3().build()?;

    let result = structure.predict("document.jpg")?;
    println!("{}", result.to_markdown());

    Ok(())
}
```

The exact chain the preset expands to, for swapping individual models:

```rust
use oar_ocr::prelude::*;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let structure = OARStructureBuilder::new("pp-doclayoutv3.onnx")
        .layout_model_name("PP-DocLayoutV3")
        .with_ocr(
            "pp-ocrv6_tiny_det.onnx",
            "pp-ocrv6_tiny_rec.onnx",
            "ppocrv6_tiny_dict.txt",
        )
        .with_table_classification("pp-lcnet_x1_0_table_cls.onnx")
        .with_wired_table_structure("slanext_wired.onnx")
        .with_wireless_table_structure("slanet_plus.onnx")
        .with_wired_table_cell_detection("rt-detr-l_wired_table_cell_det.onnx")
        .table_structure_dict_path("table_structure_dict_ch.txt")
        .build()?;

    let result = structure.predict("document.jpg")?;
    println!("{}", result.to_markdown());

    Ok(())
}
```

### Vision-Language Page Parsing

With `cargo add oar-ocr-vl --features auto-download` (plus `cargo add image` for decoding), any supported VLM loads by its Hugging Face model ID; the checkpoint downloads from ModelScope (or Hugging Face) on first use and is cached under `~/.oar`.

```rust
use oar_ocr_vl::{
    AnyPageParser, AnyPageParserModel, AnyPageParserOptions, AnyPageParserPretrainedOptions,
    PageParser, auto_device,
};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let parser = AnyPageParser::from_pretrained(
        AnyPageParserModel::PaddleOcrVl1_5,
        auto_device(),
        &AnyPageParserPretrainedOptions::default(),
    )?;

    let image = image::open("document.jpg")?.to_rgb8();
    let page = parser.parse_page(&image, &AnyPageParserOptions::default())?;
    println!("{}", page.markdown.unwrap_or_default());

    Ok(())
}
```

Both pipelines export the same [page JSON](docs/page-format.md): `page.to_json(width, height)` here, and `StructureResult::to_json(width, height)` for the classic structure pipeline.

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
| [GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) | 0.9B | Page parsing, text, table, and formula recognition |
| [HPD-Parsing](https://huggingface.co/PaddlePaddle/HPD-Parsing) | 1B | Hierarchical full-page parsing with continuously batched content branches, zero-copy shared-prefix KV, and per-branch P-MTP |
| [HunyuanOCR 1.5 / 1.0](https://huggingface.co/tencent/HunyuanOCR) | 1B | Prompt-driven full-page parsing, text spotting, table, formula, and chart recognition, with optional DFlash decoding for 1.5 |
| [jina-ocr-v1](https://huggingface.co/jinaai/jina-ocr-v1) | 3B (570M active) | End-to-end page-to-Markdown parsing (SAM+CLIP DeepEncoder over a DeepSeek-V2 MoE decoder) |
| [MinerU-Diffusion-V1-0320](https://huggingface.co/opendatalab/MinerU-Diffusion-V1-0320-2.5B) | 2.5B | Block-diffusion OCR with structured two-step extraction or single-pass text recognition |
| [MinerU2.5-2509](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B) | 1.2B | Model-native two-step layout detection and content extraction |
| [MinerU2.5-Pro-2605](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B) | 1.2B | Newer MinerU2.5 checkpoint using the same two-step pipeline |
| [MonkeyOCRv2-B-Parsing](https://huggingface.co/zenosai/MonkeyOCRv2-B-Parsing) | 0.7B | Higher-capacity ViT-B variant with the same parsing and recognition tasks |
| [MonkeyOCRv2-S-Parsing](https://huggingface.co/zenosai/MonkeyOCRv2-S-Parsing) | 0.6B | Model-native layout, end-to-end parsing, text, formula, and OTSL-table recognition |
| [OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) | 0.8B | Model-native full-page document-to-Markdown parsing |
| [PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) | 0.9B | Page parsing, text, table, formula, and chart recognition |
| [PaddleOCR-VL-1.5](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5) | 0.9B | PaddleOCR-VL tasks plus text spotting and seal recognition |
| [PaddleOCR-VL-1.6](https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6) | 0.9B | Region-aware page parsing and task-specific recognition |
| [TeleOCR](https://huggingface.co/XingChen-AGI/TeleOCR) | 1.2B | Text, table (OTSL), formula, code, and layout recognition for digital and camera-captured documents (formerly NaviDC-OCR) |
| [WeVisDoc-2B](https://huggingface.co/tencent/WeVisDoc-2B) / [4B](https://huggingface.co/tencent/WeVisDoc-4B) | 2B / 4B | Model-native full-page document-to-Markdown parsing (Qwen3-VL with DeepStack) |
| [Xiaomi-OCR-0](https://huggingface.co/SeerRay-Lab/Xiaomi-OCR-0) | 0.8B | Model-native full-page document-to-Markdown parsing (Qwen3.5; tables as OTSL converted to HTML) |

PaddleOCR-VL variants, GLM-OCR, TeleOCR, jina-ocr-v1, and WeVisDoc integrate with the external-layout [`DocParser`](oar-ocr-vl/README.md#document-parsing-pipeline). OvisOCR2, WeVisDoc, Xiaomi-OCR-0, HPD-Parsing, and the MonkeyOCRv2 S/B parsing models also provide model-native full-page paths through dedicated examples. HunyuanOCR and the MinerU models also use their model-native parsing pipelines.

See the [`oar-ocr-vl` guide](oar-ocr-vl/README.md) for setup and [`oar-ocr-vl/examples`](oar-ocr-vl/examples) for runnable examples.

## Documentation

- [Usage guide](docs/usage.md) — APIs, builder patterns, accelerators, and model loading
- [Page JSON format](docs/page-format.md) — the shared output schema of both pipelines
- [Benchmarking](docs/benchmarking.md) — reproducible pipeline baselines and comparisons
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
