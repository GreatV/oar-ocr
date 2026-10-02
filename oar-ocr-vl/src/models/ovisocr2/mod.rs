//! OvisOCR2 document-to-Markdown model support.
//!
//! The Qwen3.5 text tower (decoder and Gated DeltaNet kernel) lives in
//! `backbones::qwen3_5`, shared with Xiaomi-OCR-0.

mod adapter;
mod config;
mod model;
mod parser;
pub(crate) mod processing;
pub(crate) mod vision;

pub use config::{
    OVIS_OCR2_MAX_PIXELS, OVIS_OCR2_MIN_PIXELS, OvisOcr2Config, OvisOcr2ImageProcessorConfig,
    OvisOcr2ImageProcessorSize, OvisOcr2RopeParameters, OvisOcr2TextConfig, OvisOcr2VisionConfig,
};
pub use model::{
    DEFAULT_MAX_NEW_TOKENS, DEFAULT_PROMPT, OvisOcr2, clean_truncated_repeats,
    filter_visual_image_tags,
};
pub use parser::OvisOcr2ParseOptions;
