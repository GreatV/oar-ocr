//! HunyuanOCR (HunYuanVL) Vision-Language model.
//!
//! This module provides native Rust inference for the `tencent/HunyuanOCR` model
//! (config `model_type=hunyuan_vl`) using Candle.

mod config;
mod llm;
mod model;
mod processing;
mod vision;

pub use config::{
    HunyuanOcrConfig, HunyuanOcrImageProcessorConfig, HunyuanOcrRopeScaling, HunyuanOcrVisionConfig,
};
pub use model::HunyuanOcr;
