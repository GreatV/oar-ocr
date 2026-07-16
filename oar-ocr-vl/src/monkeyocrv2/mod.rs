//! MonkeyOCRv2-S-Parsing document VLM.
//!
//! The checkpoint combines a MonkeyOCRv2 ViT-S encoder with a Qwen3-0.6B
//! causal decoder. This module provides native Candle inference for the
//! official layout, end-to-end, text, formula, and OTSL-table prompts.

mod config;
mod model;
mod processing;
mod vision;

pub use config::{MonkeyOcrV2Config, MonkeyOcrV2ImageProcessorConfig, MonkeyOcrV2VisionConfig};
pub use model::{DEFAULT_MAX_NEW_TOKENS, LAYOUT_MIN_PIXELS, MonkeyOcrV2, MonkeyOcrV2Task};
