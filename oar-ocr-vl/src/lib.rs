//! # OAR OCR VL
//!
//! Vision-Language models for document understanding.
//!
//! This crate provides Vision-Language models that integrate with oar-ocr-core
//! for advanced document processing tasks.
//!
//! ## Module Structure
//!
//! - `paddleocr_vl` - PaddleOCR-VL for OCR, table, formula, chart recognition
//! - `unirec` - UniRec unified text/formula/table recognition
//! - `doc_parser` - Unified document parsing with pluggable recognition backends
//! - `utils` - Utility functions (device parsing, candle helpers, markdown, OTSL conversion)
//!
//! ## Features
//!
//! - `cuda` - Enable CUDA support for GPU acceleration
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

pub mod doc_parser;
pub mod hunyuanocr;
pub mod paddleocr_vl;
pub mod unirec;
pub mod utils;

// Re-exports for convenience
pub use paddleocr_vl::{
    PaddleOcrVl, PaddleOcrVlConfig, PaddleOcrVlImageProcessorConfig, PaddleOcrVlTask,
};

pub use unirec::UniRec;

pub use hunyuanocr::HunyuanOcr;

pub use doc_parser::{DocParser, DocParserConfig, RecognitionBackend, RecognitionTask};
