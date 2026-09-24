//! jina-ocr-v1 document-to-Markdown model support.
//!
//! Native Rust inference for the Jina jina-ocr-v1 checkpoint: a SAM ViT-B +
//! CLIP-L dual encoder feeding a 12-layer DeepSeek-V2 MoE decoder through a
//! linear projector. Pages are encoded as a padded 1024px global view plus
//! (when the page is larger than 640px) up to nine 640px tiles
//! (`crop_mode`), and generation follows the official greedy recipe with a
//! sliding-window no-repeat-ngram guard. The FastMTP draft head implements
//! lossless three-token speculation (greedy verification keeps the sequence
//! identical; opt-in via `OAR_JINAOCR_ENABLE_MTP`) with a CUDA-graph-captured
//! draft step; multi-page calls run a padded batch prefill and decode.

mod adapter;
mod config;
mod model;
mod mtp;
mod parser;
pub mod processing;

pub use config::{DeepSeekV2TextConfig, JinaOcrConfig};
pub use model::{DEFAULT_MAX_NEW_TOKENS, JinaOcr};
pub use parser::JinaOcrParseOptions;
pub use processing::DEFAULT_OCR_PROMPT;
