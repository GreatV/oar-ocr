//! # OAR OCR VL
//!
//! Vision-Language models for document understanding, integrating with oar-ocr-core.
//!
//! ## Modules
//!
//! - `paddleocr_vl` - PaddleOCR-VL for OCR, table, formula, chart, spotting, and seal recognition
//! - `hunyuanocr` - HunyuanOCR 1.5 / 1.0 OCR expert VLM
//! - `glmocr` - GLM-OCR OCR expert VLM
//! - `mineru` - MinerU2.5 document parsing VLM (Qwen2-VL backbone)
//! - `mineru_diffusion` - MinerU-Diffusion-V1 block-diffusion document OCR (Qwen2-VL vision + SDAR decoder)
//! - `doc_parser` - Unified document parsing with pluggable recognition backends
//! - `utils` - Device parsing, candle helpers, markdown, OTSL conversion
//! - `attention` - Unified attention shared by all models
//!
//! GPU acceleration is gated behind the `cuda` feature. Parse device strings
//! with [`utils::parse_device`]:
//!
//! ```no_run
//! use oar_ocr_vl::utils::parse_device;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let device = parse_device("cuda:0")?;
//! # let _ = device;
//! # Ok(())
//! # }
//! ```

// Core model modules
pub mod doc_parser;
pub mod glmocr;
pub mod hunyuanocr;
pub mod mineru;
pub mod mineru_diffusion;
pub mod paddleocr_vl;
pub mod utils;

// Shared attention implementation
pub mod attention;

// `TrimmableKvCache` backs the KV cache used by every model's attention
// forward path.
pub(crate) mod kv_trim;

// Re-exports for convenience
pub use paddleocr_vl::{
    PaddleOcrVl, PaddleOcrVlConfig, PaddleOcrVlImageProcessorConfig, PaddleOcrVlTask,
};

pub use glmocr::GlmOcr;
pub use hunyuanocr::{DFlashConfig, DFlashTargetConfig, HunyuanOcr, HunyuanOcrVersion};
pub use mineru::MinerU;
pub use mineru_diffusion::{DiffusionGenerationConfig, MinerUDiffusion};

pub use doc_parser::{DocParser, DocParserConfig, RecognitionBackend, RecognitionTask};
