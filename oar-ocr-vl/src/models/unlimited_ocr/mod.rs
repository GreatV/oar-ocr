//! Native Unlimited-OCR-2 inference: the DeepSeek-OCR DeepEncoder (SAM-B and
//! CLIP-L) and a DeepSeek-V2 MoE decoder with DASH block-sparse attention.
//!
//! One request parses one or many pages: each page contributes a `<PAGE>`
//! marker and its image tokens, and the output carries one `<PAGE>` per
//! input page. A single page is encoded with dynamic 640-pixel tiles plus a
//! 1024-pixel global view, several pages with the global view only, as in
//! the reference `infer`. Decoding is greedy with the reference page-count
//! guard and the official no-repeat n-gram guard; no CUDA graph is used
//! because the routed blocks change per step. The output is a layout-block
//! protocol, not Markdown; [`to_markdown`] converts it.

mod config;
mod dash;
mod markdown;
mod model;
mod parser;
pub mod processing;

pub use markdown::to_markdown;
pub use model::{GenerationTrace, UnlimitedOcr, UnlimitedOcrGenerationOptions};
pub use parser::UnlimitedOcrParseOptions;
pub use processing::{UnlimitedOcrPage, UnlimitedOcrViewMode};

fn infer(error: candle_core::Error) -> crate::Error {
    crate::runtime::errors::candle_to_ocr_inference("Unlimited-OCR", "tensor operation", error)
}
