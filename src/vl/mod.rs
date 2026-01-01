//! Vision-Language modules (optional).
//!
//! This module contains integrations that are not part of the core ONNX-based
//! OCR pipeline (e.g. large VLMs). These are behind feature flags to keep
//! default builds lightweight.
//!
//! ## Module Structure
//!
//! - `paddleocr_vl` - PaddleOCR-VL for OCR, table, formula, chart recognition
//! - `unirec` - UniRec unified text/formula/table recognition
//! - `doc_parser` - Unified document parsing with pluggable recognition backends
//! - `utils` - Utility functions (candle helpers, markdown, OTSL conversion)

#[cfg(feature = "vl")]
pub mod paddleocr_vl;

#[cfg(feature = "vl")]
pub mod unirec;

#[cfg(feature = "vl")]
pub mod utils;

#[cfg(feature = "vl")]
pub mod doc_parser;

// Re-exports for convenience
#[cfg(feature = "vl")]
pub use paddleocr_vl::{
    PaddleOcrVl, PaddleOcrVlConfig, PaddleOcrVlImageProcessorConfig, PaddleOcrVlTask,
};

#[cfg(feature = "vl")]
pub use unirec::UniRec;

#[cfg(feature = "vl")]
pub use doc_parser::{DocParser, DocParserConfig, RecognitionBackend, RecognitionTask};
