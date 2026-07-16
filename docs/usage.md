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
| `.character_dict_content(text)` | Provide the character dictionary as an in-memory string |

#### Loading Models from Memory

Everywhere a builder accepts a model path it also accepts raw ONNX bytes (`Vec<u8>`, `&'static [u8]`, or `Arc<[u8]>`), so models can be embedded into the binary or decrypted at runtime without touching the filesystem:

```rust
use oar_ocr::oarocr::OAROCRBuilder;

static DET_MODEL: &[u8] = include_bytes!("../models/pp-ocrv6_tiny_det.onnx");
static REC_MODEL: &[u8] = include_bytes!("../models/pp-ocrv6_tiny_rec.onnx");
static DICT: &str = include_str!("../models/ppocrv6_tiny_dict.txt");

let ocr = OAROCRBuilder::new(DET_MODEL, REC_MODEL, "")
    .character_dict_content(DICT)
    .build()?;
```

In-memory sources skip auto-download resolution, and models that reference external-data sidecar files cannot be loaded this way. The same applies to the per-task predictors (`TextDetectionPredictorBuilder::build(...)` etc.), `OARStructureBuilder` model setters, and `AdapterBuilder::build(...)`.

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
    .with_formula_recognition("pp-formulanet-l.onnx", "pp-formulanet-tokenizer.json", "pp_formulanet")
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
| `.formula_recognition_config(config)` | Set formula score threshold, max length, and batch size |
| `.formula_ort_session(config)` | Apply ONNX Runtime configuration only to formula recognition |
| `.with_ocr(det, rec, dict)` | Add integrated OCR pipeline |
| `.with_seal_text_detection(path)` | Add seal/stamp text detection |
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

See the [Cargo feature guide](features.md) for the root-crate features, default behavior, platform requirements, and recommended combinations.

| Feature | Provider | Platform |
|---------|----------|----------|
| `cuda` | NVIDIA CUDA | Linux, Windows |
| `tensorrt` | NVIDIA TensorRT | Linux, Windows |
| `directml` | DirectML | Windows |
| `coreml` | Core ML | macOS, iOS |
| `webgpu` | WebGPU | Supported ONNX Runtime targets |
| `openvino` | Intel OpenVINO | Linux, Windows |

Example with TensorRT:

```rust
let ort_config = OrtSessionConfig::new()
    .with_execution_providers(vec![
        OrtExecutionProvider::TensorRT {
            device_id: Some(0),
            max_workspace_size: None,
            min_subgraph_size: None,
            fp16_enable: None,
            timing_cache: None,
            timing_cache_path: None,
            force_timing_cache: None,
            engine_cache: None,
            engine_cache_path: None,
            dump_ep_context_model: None,
            ep_context_file_path: None,
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

## PaddleOCR-VL

[PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) is a 0.9B document Vision-Language Model from the PaddlePaddle team. It supports 109 languages and task-specific recognition for text, tables, formulas, and charts.

This functionality is available in the separate `oar-ocr-vl` crate, using [Candle](https://github.com/huggingface/candle) for native Rust inference.

PaddleOCR-VL-1.5 and PaddleOCR-VL-1.6 are drop-in replacements via `PaddleOcrVl::from_dir`. Both add **text spotting** and **seal recognition** to the original model's task set.

### Installation

Add the VL crate and the core crate used by the snippets below to your `Cargo.toml`. `download-binaries` supplies ONNX Runtime for the core helpers unless you link a system installation with `ORT_LIB_PATH` or `ORT_LIB_LOCATION`.

```toml
[dependencies]
oar-ocr-core = { version = "0.8", default-features = false }
oar-ocr-vl = { version = "0.8", features = ["download-binaries"] }
```

For GPU acceleration, enable CUDA:

```toml
[dependencies]
oar-ocr-core = { version = "0.8", default-features = false }
oar-ocr-vl = { version = "0.8", features = ["cuda", "download-binaries"] }
```

On macOS, use the `metal` feature instead.

### Downloading the Model

Download the PaddleOCR-VL model from Hugging Face:

```bash
# Recommended git download
git lfs install
git clone https://huggingface.co/PaddlePaddle/PaddleOCR-VL

# PaddleOCR-VL-1.5
git clone https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5

# PaddleOCR-VL-1.6
git clone https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.6

# Or using hf
hf download PaddlePaddle/PaddleOCR-VL --local-dir PaddleOCR-VL
hf download PaddlePaddle/PaddleOCR-VL-1.5 --local-dir PaddleOCR-VL-1.5
hf download PaddlePaddle/PaddleOCR-VL-1.6 --local-dir PaddleOCR-VL-1.6
```

### Basic Usage

```rust,no_run
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::{PaddleOcrVl, PaddleOcrVlTask};
use oar_ocr_vl::utils::parse_device;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = load_image(Path::new("document.png"))?;
    let device = parse_device("cpu")?; // or "cuda", "cuda:0", "metal"
    let vl = PaddleOcrVl::from_dir("PaddleOCR-VL", device)?;

    // Element-level OCR. The API is batch-oriented, so pass one task per image.
    let result = vl
        .generate(&[image], &[PaddleOcrVlTask::Ocr], 256)
        .into_iter()
        .next()
        .expect("one result")?;
    println!("{result}");

    Ok(())
}
```

PaddleOCR-VL-1.5 and PaddleOCR-VL-1.6 use the same API:

```rust,no_run
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::{PaddleOcrVl, PaddleOcrVlTask};
use oar_ocr_vl::utils::parse_device;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = load_image(Path::new("seal.png"))?;
    let device = parse_device("cpu")?;
    // Use "PaddleOCR-VL-1.5" here to load the 1.5 checkpoint instead.
    let vl = PaddleOcrVl::from_dir("PaddleOCR-VL-1.6", device)?;

    let result = vl
        .generate(&[image], &[PaddleOcrVlTask::Seal], 256)
        .into_iter()
        .next()
        .expect("one result")?;
    println!("{result}");

    Ok(())
}
```

### Running the Example

```bash
cargo run -p oar-ocr-vl --features cuda,download-binaries --example paddleocr_vl -- \
    -m PaddleOCR-VL --device cuda --task ocr document.jpg

cargo run -p oar-ocr-vl --features cuda,download-binaries --example paddleocr_vl -- \
    -m PaddleOCR-VL-1.5 --device cuda --task spotting spotting.jpg

cargo run -p oar-ocr-vl --features cuda,download-binaries --example paddleocr_vl -- \
    -m PaddleOCR-VL-1.6 --device cuda --task seal seal.jpg
```

### Supported Tasks

| Task | Description | Output Format |
|------|-------------|---------------|
| `PaddleOcrVlTask::Ocr` | Text recognition | Plain text |
| `PaddleOcrVlTask::Table` | Table structure recognition | HTML |
| `PaddleOcrVlTask::Formula` | Mathematical formula recognition | LaTeX |
| `PaddleOcrVlTask::Chart` | Chart understanding | Structured text |
| `PaddleOcrVlTask::Spotting` | Text spotting (localization + recognition) | Structured text |
| `PaddleOcrVlTask::Seal` | Seal recognition | Plain text |

## OvisOCR2

[OvisOCR2](https://huggingface.co/ATH-MaaS/OvisOCR2) is a 0.8B end-to-end page parser in the `oar-ocr-vl` crate. It turns each complete page into one Markdown document using its model-native path. It does not use an external layout detector and is not a `DocParser` backend.

### Downloading the Model

```bash
git lfs install
git clone https://huggingface.co/ATH-MaaS/OvisOCR2

# Or using hf
hf download ATH-MaaS/OvisOCR2 --local-dir OvisOCR2
```

### Official Input and Output Contract

The library uses the official prompt internally; callers do not need to provide it. The leading newline below is part of the prompt:

```text

Extract all readable content from the image in natural human reading order and output the result as a single Markdown document. For charts or images, represent them using an HTML image tag: <img src="images/bbox_{left}_{top}_{right}_{bottom}.jpg" />, where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as HTML: <table>...</table>. Transcribe all other text as standard Markdown. Preserve the original text without translation or paraphrasing.
```

Input pages are converted to RGB, resized with bicubic antialiasing, and aligned to the model's 32-pixel factor. The official runtime constrains image area to `448²` through `2880²` pixels. Output is reading-order Markdown with LaTeX formulas and HTML tables. Visual regions may be emitted as `<img src="images/bbox_left_top_right_bottom.jpg" />`, with coordinates scaled to `[0, 1000)`.

`OvisOcr2::parse` applies truncated-repeat cleanup and removes standalone visual-region image-tag blocks, matching the default official post-processing. Use `parse_with_image_tags(..., true)` or `generate` when those tags must be retained.

### Basic Usage

```rust,no_run
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::ovisocr2::DEFAULT_MAX_NEW_TOKENS;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::OvisOcr2;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = load_image("document.jpg")?;
    let model = OvisOcr2::from_dir("OvisOCR2", parse_device("cpu")?)?;
    let markdown = model
        .parse(&[image], DEFAULT_MAX_NEW_TOKENS)
        .into_iter()
        .next()
        .expect("one result")?;
    println!("{markdown}");
    Ok(())
}
```

The API is batch-oriented: pass multiple page images to `parse` and receive one result per page in the same order. The upstream generation limit, exported as `DEFAULT_MAX_NEW_TOKENS`, is 16,384.

### Running the Example

```bash
cargo run -p oar-ocr-vl --features cuda,download-binaries --example ovisocr2 -- \
    --model-dir OvisOCR2 \
    --device cuda:0 \
    document-1.jpg document-2.jpg
```

Add `--keep-image-tags` to retain visual-region image-tag references, or use `--max-tokens` to override the 16,384-token default. The example does not create the referenced bounding-box crop files.

## MonkeyOCRv2-S-Parsing

[MonkeyOCRv2-S-Parsing](https://huggingface.co/zenosai/MonkeyOCRv2-S-Parsing) is a 0.6B document parser with a Monkey ViT-S vision encoder and Qwen3-0.6B decoder. The native Candle implementation supports the checkpoint's five official tasks: `Layout`, `EndToEnd`, `Text`, `Formula`, and `Table`.

### Downloading the Model

```bash
git lfs install
git clone https://huggingface.co/zenosai/MonkeyOCRv2-S-Parsing

# Or using hf
hf download zenosai/MonkeyOCRv2-S-Parsing --local-dir MonkeyOCRv2-S-Parsing
```

### Basic Usage

```rust,no_run
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::{MonkeyOcrV2, MonkeyOcrV2Task};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = load_image("document.jpg")?;
    let model = MonkeyOcrV2::from_dir(
        "MonkeyOCRv2-S-Parsing",
        parse_device("cuda:0")?,
    )?;
    let output = model
        .generate(&[image], &[MonkeyOcrV2Task::EndToEnd], 10_000)
        .into_iter()
        .next()
        .expect("one result")?;
    println!("{output}");
    Ok(())
}
```

`EndToEnd` returns a Python/JSON-like reading-order list of `{bbox, label, content}` items; box coordinates are normalized to `[0, 1000]`. `Layout` returns `{bbox, label}` items and applies the official layout-pass minimum area of 1,003,520 pixels. `Table` emits OTSL, while `Formula` emits LaTeX. `generate_with_prompts` is available for custom instructions.

### Running the Example

```bash
cargo run -p oar-ocr-vl --features cuda,download-binaries --example monkeyocrv2 -- \
    --model-dir MonkeyOCRv2-S-Parsing \
    --device cuda:0 \
    --task end-to-end \
    document.jpg
```

The other `--task` values are `layout`, `text`, `formula`, and `table`. The model also implements `RecognitionBackend` for external-layout crop recognition, although its native tasks are the preferred full-page path.

## HunyuanOCR 1.5

[HunyuanOCR 1.5](https://huggingface.co/tencent/HunyuanOCR) is a lightweight OCR expert VLM. It is available in the `oar-ocr-vl` crate and supports prompt-driven image-to-text OCR. `HunyuanOcr::from_dir` automatically detects 1.5 at the model repository root and remains compatible with archived 1.0 weights under `v1.0/`.

HunyuanOCR 1.5 inputs use the checkpoint's 16M-pixel budget (up to a 4K square input). The 2048 value in `vision_config.max_image_size` describes the learned positional-embedding base grid. Larger input grids are interpolated, as in the official implementation.

### Downloading the Model

```bash
git lfs install
git clone https://huggingface.co/tencent/HunyuanOCR

# Or using hf
hf download tencent/HunyuanOCR --local-dir HunyuanOCR
```

The download places 1.5 weights directly in `HunyuanOCR/` and its optional DFlash draft in `HunyuanOCR/dflash/`. To use 1.0 instead, pass `HunyuanOCR/v1.0` as the model directory.

### Basic Usage

```rust,no_run
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::HunyuanOcr;
use oar_ocr_vl::utils::parse_device;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = load_image("document.jpg")?;
    let device = parse_device("cpu")?; // or "cuda", "cuda:0", "metal"

    // Repository root = HunyuanOCR 1.5; `HunyuanOCR/v1.0` also works.
    let model = HunyuanOcr::from_dir("HunyuanOCR", device)?;

    let prompt = "Detect and recognize text in the image, and output the text coordinates in a formatted manner.";
    let text = model
        .generate(&[image], &[prompt], 1024)
        .into_iter()
        .next()
        .expect("one result")?;
    println!("{text}");

    Ok(())
}
```

HunyuanOCR 1.5 can load the official DFlash draft for speculative decoding:

```rust,no_run
use oar_ocr_vl::HunyuanOcr;
use oar_ocr_vl::utils::parse_device;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = HunyuanOcr::from_dir_with_dflash(
        "HunyuanOCR",
        parse_device("cuda:0")?,
    )?;
    assert!(model.dflash_enabled());
    Ok(())
}
```

### Running the Example

```bash
cargo run -p oar-ocr-vl --features cuda,download-binaries --example hunyuanocr -- \
    --model-dir HunyuanOCR \
    --dflash-dir HunyuanOCR/dflash \
    --device cuda \
    --prompt "Detect and recognize text in the image, and output the text coordinates in a formatted manner." \
    document.jpg
```

### Application-oriented Prompts

Prompts from the upstream HunyuanOCR README:

| Task | English | Chinese |
|------|---------|---------|
| **Spotting** | Detect and recognize text in the image, and output the text coordinates in a formatted manner. | 检测并识别图片中的文字，将文本坐标格式化输出。 |
| **Parsing** | • Identify the formula in the image and represent it using LaTeX format.<br><br>• Parse the table in the image into HTML.<br><br>• Parse the chart in the image; use Mermaid format for flowcharts and Markdown for other charts.<br><br>• Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in HTML format, formulas in the document should be represented using LaTeX format, and the parsing should be organized according to the reading order. | • 识别图片中的公式，用 LaTeX 格式表示。<br><br>• 把图中的表格解析为 HTML。<br><br>• 解析图中的图表，对于流程图使用 Mermaid 格式表示，其他图表使用 Markdown 格式表示。<br><br>• 提取文档图片中正文的所有信息用 markdown 格式表示，其中页眉、页脚部分忽略，表格用 html 格式表达，文档中公式用 latex 格式表示，按照阅读顺序组织进行解析。 |
| **Information Extraction** | • Output the value of Key.<br><br>• Extract the content of the fields: ['key1','key2', ...] from the image and return it in JSON format.<br><br>• Extract the subtitles from the image. | • 输出 Key 的值。<br><br>• 提取图片中的: ['key1','key2', ...] 的字段内容，并按照 JSON 格式返回。<br><br>• 提取图片中的字幕。 |
| **Translation** | First extract the text, then translate the text content into English. If it is a document, ignore the header and footer. Formulas should be represented in LaTeX format, and tables should be represented in HTML format. | 先提取文字，再将文字内容翻译为英文。若是文档，则其中页眉、页脚忽略。公式用latex格式表示，表格用html格式表示。 |

## GLM-OCR

[GLM-OCR](https://huggingface.co/zai-org/GLM-OCR) is an OCR expert VLM in the `oar-ocr-vl` crate. It uses prompt-driven image-to-text generation and can be used directly or as a `DocParser` backend.

### Downloading the Model

```bash
git lfs install
git clone https://huggingface.co/zai-org/GLM-OCR

# Or using hf
hf download zai-org/GLM-OCR --local-dir GLM-OCR
```

### Basic Usage

```rust,no_run
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::GlmOcr;
use oar_ocr_vl::utils::parse_device;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = load_image("document.jpg")?;
    let device = parse_device("cpu")?; // or "cuda", "cuda:0", "metal"

    let model = GlmOcr::from_dir("GLM-OCR", device)?;
    let prompt = "Text Recognition:";
    let text = model
        .generate(&[image], &[prompt], 1024)
        .into_iter()
        .next()
        .expect("one result")?;
    println!("{text}");

    Ok(())
}
```

### Running the Example

```bash
cargo run -p oar-ocr-vl --features cuda,download-binaries --example glmocr -- \
    --model-dir GLM-OCR \
    --device cuda \
    --prompt "Text Recognition:" \
    document.jpg
```

## MinerU2.5 and MinerU2.5-Pro

[MinerU2.5](https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B) and [MinerU2.5-Pro](https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B) are document parsing VLMs supported by the same `MinerU` loader. For full-page documents, use their model-native two-step pipeline rather than forcing them through `DocParser`.

### Downloading the Model

```bash
git lfs install
git clone https://huggingface.co/opendatalab/MinerU2.5-2509-1.2B
git clone https://huggingface.co/opendatalab/MinerU2.5-Pro-2605-1.2B

# Or using hf
hf download opendatalab/MinerU2.5-2509-1.2B --local-dir MinerU2.5-2509-1.2B
hf download opendatalab/MinerU2.5-Pro-2605-1.2B --local-dir MinerU2.5-Pro-2605-1.2B
```

### Basic Usage

```rust,no_run
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::MinerU;
use oar_ocr_vl::utils::parse_device;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = load_image("document.jpg")?;
    let device = parse_device("cpu")?; // or "cuda", "cuda:0", "metal"

    // MinerU2.5-Pro-2605-1.2B is loaded through the same API.
    let model = MinerU::from_dir("MinerU2.5-2509-1.2B", device)?;
    let prompt = "\nText Recognition:";
    let text = model
        .generate(&[image], &[prompt], 1024)
        .into_iter()
        .next()
        .expect("one result")?;
    println!("{text}");

    Ok(())
}
```

### Running the Example

```bash
cargo run -p oar-ocr-vl --features cuda,download-binaries --example mineru -- \
    --model-dir MinerU2.5-2509-1.2B \
    --device cuda \
    document.jpg

cargo run -p oar-ocr-vl --features cuda,download-binaries --example mineru -- \
    --model-dir MinerU2.5-Pro-2605-1.2B \
    --device cuda \
    document.jpg
```

## MinerU-Diffusion-V1

[MinerU-Diffusion-V1](https://huggingface.co/opendatalab/MinerU-Diffusion-V1-0320-2.5B) replaces autoregressive text generation with block-diffusion decoding. The `mineru_diffusion` example defaults to MinerU-style two-step structured extraction and also provides `--single-pass` for flat full-page text recognition.

### Downloading the Model

```bash
git lfs install
git clone https://huggingface.co/opendatalab/MinerU-Diffusion-V1-0320-2.5B

# Or using hf
hf download opendatalab/MinerU-Diffusion-V1-0320-2.5B \
    --local-dir MinerU-Diffusion-V1-0320-2.5B
```

### Basic Usage

```rust,no_run
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::mineru_diffusion::DEFAULT_PROMPT;
use oar_ocr_vl::utils::parse_device;
use oar_ocr_vl::{DiffusionGenerationConfig, MinerUDiffusion};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = load_image("document.jpg")?;
    let model = MinerUDiffusion::from_dir(
        "MinerU-Diffusion-V1-0320-2.5B",
        parse_device("cuda:0")?,
    )?;
    let text = model.generate(
        &image,
        DEFAULT_PROMPT,
        &DiffusionGenerationConfig::default(),
    )?;
    println!("{text}");
    Ok(())
}
```

### Running the Example

```bash
cargo run -p oar-ocr-vl --features cuda,download-binaries \
    --example mineru_diffusion -- \
    --model-dir MinerU-Diffusion-V1-0320-2.5B \
    --device cuda:0 \
    document.jpg
```

## DocParser

DocParser provides a unified API for external layout-first document parsing with VL-based recognition. The `doc_parser` example supports PaddleOCR-VL, PaddleOCR-VL-1.5, PaddleOCR-VL-1.6, and GLM-OCR. MonkeyOCRv2 also implements `RecognitionBackend`, but uses its dedicated model-native example for complete pages.

Use `parse(&layout, image)` with an ONNX layout detector. OvisOCR2 deliberately does not implement `RecognitionBackend`; use its full-page `OvisOcr2::parse` API instead. The library also implements `RecognitionBackend` for MonkeyOCRv2, HunyuanOCR, and MinerU2.5/Pro, but they are intentionally not exposed by the CLI example because their reference-quality paths are model-native parsing. MinerU-Diffusion uses its dedicated example.

### Basic Usage

```rust
use oar_ocr_core::utils::load_image;
use oar_ocr_core::predictors::LayoutDetectionPredictor;
use oar_ocr_vl::{DocParser, GlmOcr, PaddleOcrVl};
use oar_ocr_vl::utils::parse_device;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let device = parse_device("cpu")?;

    // Initialize layout detector
    let layout = LayoutDetectionPredictor::builder()
        .model_name("pp-doclayoutv3")
        .build("pp-doclayoutv3.onnx")?;

    // Load document image
    let image = load_image(Path::new("document.jpg"))?;

    // Option 1: Using PaddleOCR-VL
    let paddleocr_vl = PaddleOcrVl::from_dir("PaddleOCR-VL", device.clone())?;
    let parser = DocParser::new(&paddleocr_vl);
    let result = parser.parse(&layout, image.clone())?;
    println!("{}", result.to_markdown());

    // Option 2: Using PaddleOCR-VL-1.5
    let paddleocr_vl_15 = PaddleOcrVl::from_dir("PaddleOCR-VL-1.5", device.clone())?;
    let parser = DocParser::new(&paddleocr_vl_15);
    let result = parser.parse(&layout, image.clone())?;
    println!("{}", result.to_markdown());

    // Option 3: Using PaddleOCR-VL-1.6
    let paddleocr_vl_16 = PaddleOcrVl::from_dir("PaddleOCR-VL-1.6", device.clone())?;
    let parser = DocParser::new(&paddleocr_vl_16);
    let result = parser.parse(&layout, image.clone())?;
    println!("{}", result.to_markdown());

    // Option 4: Using GLM-OCR with external layout
    let glmocr = GlmOcr::from_dir("GLM-OCR", device)?;
    let parser = DocParser::new(&glmocr);
    let result = parser.parse(&layout, image)?;
    println!("{}", result.to_markdown());

    Ok(())
}
```

### Running the Example

```bash
# Using PaddleOCR-VL
cargo run -p oar-ocr-vl --features cuda,download-binaries --example doc_parser -- \
    --model-name paddleocr-vl \
    --model-dir PaddleOCR-VL \
    --layout-model models/pp-doclayoutv3.onnx \
    --device cuda \
    document.jpg

# Using PaddleOCR-VL-1.5
cargo run -p oar-ocr-vl --features cuda,download-binaries --example doc_parser -- \
    --model-name paddleocr-vl-1.5 \
    --model-dir PaddleOCR-VL-1.5 \
    --layout-model models/pp-doclayoutv3.onnx \
    --device cuda \
    document.jpg

# Using PaddleOCR-VL-1.6
cargo run -p oar-ocr-vl --features cuda,download-binaries --example doc_parser -- \
    --model-name paddleocr-vl-1.6 \
    --model-dir PaddleOCR-VL-1.6 \
    --layout-model models/pp-doclayoutv3.onnx \
    --device cuda \
    document.jpg

# Using GLM-OCR with layout
cargo run -p oar-ocr-vl --features cuda,download-binaries --example doc_parser -- \
    --model-name glmocr \
    --model-dir GLM-OCR \
    --layout-model models/pp-doclayoutv3.onnx \
    --device cuda \
    document.jpg

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
