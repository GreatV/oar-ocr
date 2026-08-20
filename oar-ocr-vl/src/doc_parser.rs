//! Backwards-compatible document-parser exports.

pub use crate::pipeline::doc_parser::*;

/// Backwards-compatible parser alias specialized for PaddleOCR-VL.
pub type PaddleOcrVlDocParser<'a> = crate::pipeline::doc_parser::DocParser<'a, crate::PaddleOcrVl>;
