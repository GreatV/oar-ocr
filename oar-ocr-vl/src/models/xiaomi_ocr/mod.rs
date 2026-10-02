//! Xiaomi-OCR-0 document-to-Markdown model support.
//!
//! [`XiaomiOcr`] (SeerRay-Lab/Xiaomi-OCR-0) is a 0.8B Qwen3.5 VLM whose text
//! tower is shared with OvisOCR2 through `backbones::qwen3_5`. The
//! model-specific pieces are the official task prompts, the nested
//! `processor_config.json` with its advertised pixel bounds, and the
//! whole-page post-processing that converts OTSL table blocks to HTML.

mod adapter;
mod config;
mod model;
mod parser;
pub(crate) mod processing;

pub use config::{
    Qwen35RopeParameters, Qwen35TextConfig, XiaomiOcrConfig, XiaomiOcrProcessorConfig,
    XiaomiOcrVisionConfig,
};
pub use model::{
    DEFAULT_MAX_NEW_TOKENS, DEFAULT_PROMPT, FORMULA_REGION_PROMPT, KIE_PROMPT, TABLE_REGION_PROMPT,
    TEXT_REGION_PROMPT, XiaomiOcr, convert_markdown_otsl_tables, finalize_markdown,
    key_information_extraction_prompt,
};
pub use parser::XiaomiOcrParseOptions;
