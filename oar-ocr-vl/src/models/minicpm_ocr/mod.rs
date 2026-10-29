//! Native MiniCPM-OCR 1.0 inference (SigLIP vision tower, 2x2 merger and a
//! Qwen3.5 text decoder).
//!
//! Preprocessing follows the Transformers `MiniCPMOcrImageProcessor`
//! defaults: an overview image plus up to nine 448-pixel slices, every
//! slice upscaled to the slice budget, and 4x visual-token downsampling.
//! Generation is greedy, like the official `minicpmvocr` SDK. Page parsing
//! uses PP-DocLayout regions and the crate's shared document pipeline, as
//! the SDK does with PP-DocLayoutV3.

mod adapter;
mod config;
mod model;
pub mod processing;
mod vision;

pub use model::{GenerationTrace, MiniCpmOcr, MiniCpmOcrGenerationOptions, MiniCpmOcrTask};
pub use processing::{MiniCpmOcrImageInputs, MiniCpmOcrSlice};

fn infer(error: candle_core::Error) -> crate::Error {
    crate::runtime::errors::candle_to_ocr_inference("MiniCPM-OCR", "tensor operation", error)
}
