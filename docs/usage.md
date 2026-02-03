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

## PaddleOCR-VL (Vision-Language)

[PaddleOCR-VL](https://huggingface.co/PaddlePaddle/PaddleOCR-VL) is an ultra-compact (0.9B parameters) Vision-Language Model for document parsing, released by Baidu's PaddlePaddle team. It supports **109 languages** and excels in recognizing complex elements including text, tables, formulas, and 11 chart types. The model achieves SOTA performance in both page-level document parsing and element-level recognition while maintaining minimal resource consumption.

This functionality is available in the separate `oar-ocr-vl` crate, using [Candle](https://github.com/huggingface/candle) for native Rust inference.

PaddleOCR-VL-1.5 is also supported as a drop-in replacement via `PaddleOcrVl::from_dir`, and adds **text spotting** and **seal recognition** tasks.

### Installation

Add the VL crate to your `Cargo.toml`:

```toml
[dependencies]
oar-ocr-vl = "0.6"
```

For GPU acceleration, enable CUDA:

```toml
[dependencies]
oar-ocr-vl = { version = "0.6", features = ["cuda"] }
```

### Downloading the Model

Download the PaddleOCR-VL model from Hugging Face:

```bash
# Using git (recommended)
git lfs install
git clone https://huggingface.co/PaddlePaddle/PaddleOCR-VL

# PaddleOCR-VL-1.5
git clone https://huggingface.co/PaddlePaddle/PaddleOCR-VL-1.5

# Or using hf
hf download PaddlePaddle/PaddleOCR-VL --local-dir PaddleOCR-VL
hf download PaddlePaddle/PaddleOCR-VL-1.5 --local-dir PaddleOCR-VL-1.5
```

### Basic Usage

```rust,no_run
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::{PaddleOcrVl, PaddleOcrVlTask};
use oar_ocr_vl::utils::parse_device;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = load_image(Path::new("document.png"))?;
    let device = parse_device("cpu")?;  // or "cuda", "cuda:0"
    let vl = PaddleOcrVl::from_dir("PaddleOCR-VL", device)?;

    // Element-level OCR
    let result = vl.generate(image, PaddleOcrVlTask::Ocr, 256)?;
    println!("{result}");

    Ok(())
}
```

PaddleOCR-VL-1.5 uses the same API:

```rust,no_run
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::{PaddleOcrVl, PaddleOcrVlTask};
use oar_ocr_vl::utils::parse_device;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = load_image(Path::new("seal.png"))?;
    let device = parse_device("cpu")?;
    let vl = PaddleOcrVl::from_dir("PaddleOCR-VL-1.5", device)?;

    let result = vl.generate(image, PaddleOcrVlTask::Seal, 256)?;
    println!("{result}");

    Ok(())
}
```

### Running the Example

```bash
cargo run -p oar-ocr-vl --features cuda --example paddleocr_vl -- \
    -m PaddleOCR-VL --device cuda --task ocr document.jpg

cargo run -p oar-ocr-vl --features cuda --example paddleocr_vl -- \
    -m PaddleOCR-VL-1.5 --device cuda --task spotting spotting.jpg
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

## UniRec

[UniRec](https://github.com/Topdu/OpenOCR) is a unified recognition model with only **0.1B parameters**, developed by the FVL Laboratory at Fudan University as part of the OpenOCR project. It is designed for high-accuracy and efficient recognition of plain text (words, lines, paragraphs), mathematical formulas (single-line, multi-line), and mixed content in both Chinese and English. Despite its small size, it achieves performance comparable to or better than much larger vision-language models. It's also available in the `oar-ocr-vl` crate.

### Downloading the Model

```bash
hf download Topdu/UniRec-0.1B --local-dir models/unirec-0.1b
```

### Basic Usage

```rust
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::UniRec;
use oar_ocr_vl::utils::parse_device;
use std::path::Path;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = load_image(Path::new("formula.png"))?;
    let device = parse_device("cpu")?;  // or "cuda", "cuda:0"

    // Load UniRec model
    let model = UniRec::from_dir("models/unirec-0.1b", device)?;

    // Generate recognition result
    let result = model.generate(image, 512)?;
    println!("{result}");

    Ok(())
}
```

### Running the Example

```bash
cargo run -p oar-ocr-vl --features cuda --example unirec -- \
    -m models/unirec-0.1b --device cuda formula.jpg
```

## HunyuanOCR

[HunyuanOCR](https://huggingface.co/tencent/HunyuanOCR) is a 1B parameter OCR expert VLM powered by Hunyuan's multimodal architecture. It's available in the `oar-ocr-vl` crate and supports prompt-driven image-to-text OCR.

Note: inputs are automatically resized to satisfy the model's image/token limits (e.g., max side length 2048).

### Downloading the Model

```bash
git lfs install
git clone https://huggingface.co/tencent/HunyuanOCR

# Or using hf
hf download tencent/HunyuanOCR --local-dir HunyuanOCR
```

### Basic Usage

```rust,no_run
use oar_ocr_core::utils::load_image;
use oar_ocr_vl::HunyuanOcr;
use oar_ocr_vl::utils::parse_device;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let image = load_image("document.jpg")?;
    let device = parse_device("cpu")?; // or "cuda", "cuda:0"

    let model = HunyuanOcr::from_dir("HunyuanOCR", device)?;

    let prompt = "Detect and recognize text in the image, and output the text coordinates in a formatted manner.";
    let text = model.generate(image, prompt, 1024)?;
    println!("{text}");

    Ok(())
}
```

### Running the Example

```bash
cargo run -p oar-ocr-vl --features cuda --example hunyuanocr -- \
    --model-dir HunyuanOCR \
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

## DocParser

DocParser provides a unified API for two-stage document parsing that combines layout detection with VL-based recognition. It supports UniRec and PaddleOCR-VL (including PaddleOCR-VL-1.5) as recognition backends.

### Basic Usage

```rust
use oar_ocr_core::utils::load_image;
use oar_ocr_core::predictors::LayoutDetectionPredictor;
use oar_ocr_vl::{DocParser, DocParserConfig, UniRec, PaddleOcrVl};
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

    // Option 1: Using UniRec (lighter, faster)
    let unirec = UniRec::from_dir("models/unirec-0.1b", device.clone())?;
    let parser = DocParser::with_config(&unirec, DocParserConfig::default());
    let result = parser.parse(&layout, image.clone())?;
    println!("{}", result.to_markdown());

    // Option 2: Using PaddleOCR-VL (heavier, more accurate)
    let paddleocr_vl = PaddleOcrVl::from_dir("PaddleOCR-VL", device)?;
    let parser = DocParser::new(&paddleocr_vl);
    let result = parser.parse(&layout, image.clone())?;
    println!("{}", result.to_markdown());

    // Option 3: Using PaddleOCR-VL-1.5 (next-gen, more accurate)
    let paddleocr_vl_15 = PaddleOcrVl::from_dir("PaddleOCR-VL-1.5", device)?;
    let parser = DocParser::new(&paddleocr_vl_15);
    let result = parser.parse(&layout, image)?;
    println!("{}", result.to_markdown());

    Ok(())
}
```

### Running the Example

```bash
# Using UniRec (default, lighter)
cargo run -p oar-ocr-vl --features cuda --example doc_parser -- \
    --model-name unirec \
    --model-dir models/unirec-0.1b \
    --layout-model models/pp-doclayoutv3.onnx \
    --device cuda \
    document.jpg

# Using PaddleOCR-VL (heavier, more accurate)
cargo run -p oar-ocr-vl --features cuda --example doc_parser -- \
    --model-name paddleocr-vl \
    --model-dir PaddleOCR-VL \
    --layout-model models/pp-doclayoutv3.onnx \
    --device cuda \
    document.jpg

# Using PaddleOCR-VL-1.5 (next-gen, more accurate)
cargo run -p oar-ocr-vl --features cuda --example doc_parser -- \
    --model-name paddleocr-vl-1.5 \
    --model-dir PaddleOCR-VL-1.5 \
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
