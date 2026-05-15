//! # OAR OCR VL
//!
//! Vision-Language models for document understanding.
//!
//! This crate provides Vision-Language models that integrate with oar-ocr-core
//! for advanced document processing tasks.
//!
//! ## Module Structure
//!
//! - `paddleocr_vl` - PaddleOCR-VL for OCR, table, formula, chart, spotting, and seal recognition
//! - `hunyuanocr` - HunyuanOCR OCR expert VLM
//! - `glmocr` - GLM-OCR OCR expert VLM
//! - `mineru` - MinerU2.5 document parsing VLM (Qwen2-VL backbone)
//! - `doc_parser` - Unified document parsing with pluggable recognition backends
//! - `utils` - Utility functions (device parsing, candle helpers, markdown, OTSL conversion)
//! - `attention` - Unified attention implementation shared by all models
//! - `hsd` - Hierarchical Speculative Decoding (DSV-on-OAR engineering of paper
//!   arXiv:2602.12957 §3.2); gated behind the `hsd` feature, which requires
//!   `cuda` for tree-attention and KV-cache gather. See the module docs for
//!   the two-stage flow, prefix-tree batching, and per-backend integration.
//!
//! ## Features
//!
//! - `cuda` - Enable CUDA support for GPU acceleration
//! - `hsd` - Enable Hierarchical Speculative Decoding (implies `cuda`)
//!
//! ## Device Configuration
//!
//! Use [`utils::parse_device`] to parse device strings:
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
pub mod paddleocr_vl;
pub mod utils;

// Shared attention implementation
pub mod attention;

// `TrimmableKvCache` backs the KV cache used by every model's attention
// forward path, so it must remain accessible regardless of the `hsd` feature.
// The source lives under `hsd/kv_trim.rs` and is also re-exported from
// `crate::hsd` when that module is compiled in.
#[path = "hsd/kv_trim.rs"]
pub(crate) mod kv_trim;

// Hierarchical Speculative Decoding (requires CUDA-backed Candle for KV-cache
// gather and tree-attention; gated behind the `hsd` cargo feature).
#[cfg(feature = "hsd")]
pub mod hsd;

// Re-exports for convenience
pub use paddleocr_vl::{
    PaddleOcrVl, PaddleOcrVlConfig, PaddleOcrVlImageProcessorConfig, PaddleOcrVlTask,
};

pub use glmocr::GlmOcr;
#[cfg(feature = "hsd")]
pub use hunyuanocr::HunyuanHsdPrompts;
pub use hunyuanocr::HunyuanOcr;
pub use mineru::MinerU;

pub use doc_parser::{DocParser, DocParserConfig, RecognitionBackend, RecognitionTask};
