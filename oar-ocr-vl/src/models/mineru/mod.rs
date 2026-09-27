//! MinerU2.5 Vision-Language model implementation (Qwen2-VL backbone).
//!
//! [`MinerU::from_dir`] supports both `MinerU2.5-2509-1.2B` and
//! `MinerU2.5-Pro-2605-1.2B`. For full-page documents, use the model-native
//! two-step layout-detection and crop-recognition pipeline demonstrated by the
//! `mineru` example.

mod adapter;
mod config;
mod model;
mod parser;
pub(crate) use crate::backbones::qwen_vl_processing as processing;
pub(crate) use crate::backbones::qwen2_vl as vision;

pub use config::{
    MinerUConfig, MinerURopeScaling, MinerUTextConfig, Qwen2VlVisionConfig,
    QwenVlImageProcessorConfig, QwenVlImageSize,
};
pub use model::MinerU;
pub use parser::MinerUParseOptions;
