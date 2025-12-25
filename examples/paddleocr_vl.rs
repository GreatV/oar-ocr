//! PaddleOCR-VL Example
//!
//! This example demonstrates running PaddleOCR-VL (Vision-Language OCR) using Candle.
//!
//! # Usage
//!
//! ```bash
//! cargo run --release --features paddleocr-vl --example paddleocr_vl -- \
//!   --model-dir PaddleOCR-VL \
//!   --layout-model pp-doclayoutv2.onnx \
//!   --task ocr \
//!   --max-new-tokens 256 \
//!   path/to/image.png
//! ```

#[cfg(feature = "paddleocr-vl")]
mod enabled {
    use clap::Parser;
    use oar_ocr::predictors::LayoutDetectionPredictor;
    use oar_ocr::utils::load_image;
    use oar_ocr::vl::{
        PaddleOcrVl, PaddleOcrVlDocParserConfig, PaddleOcrVlTask, parse_document_with_layout_config,
    };
    use std::path::PathBuf;

    #[derive(Parser, Debug)]
    #[command(name = "paddleocr_vl")]
    #[command(about = "Run PaddleOCR-VL (Vision-Language OCR) on an image")]
    struct Args {
        /// Path to the PaddleOCR-VL model directory (containing config/tokenizer/model.safetensors)
        #[arg(long, default_value = "PaddleOCR-VL")]
        model_dir: PathBuf,

        /// Task prompt: ocr | table | chart | formula
        #[arg(long, default_value = "ocr")]
        task: String,

        /// Optional PP-DocLayoutV2 ONNX model for layout-first document parsing.
        #[arg(long)]
        layout_model: Option<PathBuf>,

        /// Layout model name for label mapping (default: pp-doclayoutv2).
        #[arg(long, default_value = "pp-doclayoutv2")]
        layout_model_name: String,

        /// Maximum number of generated tokens
        #[arg(long, default_value_t = 256)]
        max_new_tokens: usize,

        /// Input image path
        #[arg(value_name = "IMAGE")]
        image: PathBuf,
    }

    pub fn run() -> Result<(), Box<dyn std::error::Error>> {
        let args = Args::parse();

        let image = load_image(&args.image)?;
        let task = match args.task.as_str() {
            "ocr" => PaddleOcrVlTask::Ocr,
            "table" => PaddleOcrVlTask::Table,
            "chart" => PaddleOcrVlTask::Chart,
            "formula" => PaddleOcrVlTask::Formula,
            other => return Err(format!("unknown task '{other}'").into()),
        };

        let device = if cfg!(any(feature = "cuda")) {
            candle_core::Device::new_cuda(0).unwrap_or(candle_core::Device::Cpu)
        } else {
            candle_core::Device::Cpu
        };
        let vl = PaddleOcrVl::from_dir(&args.model_dir, device)?;

        if let Some(layout_model) = args.layout_model {
            let layout = LayoutDetectionPredictor::builder()
                .model_name(args.layout_model_name)
                .build(layout_model)?;
            let cfg = PaddleOcrVlDocParserConfig::default();
            let result = parse_document_with_layout_config(
                &vl,
                &layout,
                args.image.to_string_lossy().to_string(),
                0,
                image,
                &cfg,
                args.max_new_tokens,
            )?;
            println!("{}", result.to_markdown());
        } else {
            let out = vl.generate(image, task, args.max_new_tokens)?;
            println!("{out}");
        }

        Ok(())
    }
}

#[cfg(feature = "paddleocr-vl")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    enabled::run()
}

#[cfg(not(feature = "paddleocr-vl"))]
fn main() {
    eprintln!("This example requires `--features paddleocr-vl`.");
}
