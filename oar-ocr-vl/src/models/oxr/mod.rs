//! Native OXR-1.0 inference (SigLIP2 NaViT visual tokenizer and Qwen3).
//!
//! Image preprocessing defaults to the upstream vLLM processor profile:
//! min_pixels=200704 and max_pixels=921600. Generation is greedy and requests execute sequentially.
//!
//! Weights use [SoMark's OXR License](https://huggingface.co/SoMarkAI/OXR-1.0/blob/main/LICENSE):
//! Apache 2.0 plus commercial thresholds, an attribution duty, and a ban on
//! competing hosted APIs. Users must accept these terms before using the weights.
//! Page parsing uses PP-DocLayout and the crate's shared document pipeline.
//! OXR predicts page reading order after detection; invalid predictions keep
//! the detector order and produce a page diagnostic.

mod adapter;
mod config;
mod model;
pub mod processing;
mod reading_order;
mod vision;

pub use model::{GenerationTrace, Oxr, OxrGenerationOptions, OxrTask};

fn infer(error: candle_core::Error) -> crate::Error {
    crate::runtime::errors::candle_to_ocr_inference("OXR", "tensor operation", error)
}
