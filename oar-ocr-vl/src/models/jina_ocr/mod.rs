//! jina-ocr-v1 document-to-Markdown model support.
//!
//! Native Rust inference for the Jina jina-ocr-v1 checkpoint: a SAM ViT-B +
//! CLIP-L dual encoder feeding a 12-layer DeepSeek-V2 MoE decoder through a
//! linear projector. Pages are encoded as a padded 1024px global view plus
//! (when the page is larger than 640px) up to nine 640px tiles
//! (`crop_mode`), and generation follows the official greedy recipe with a
//! sliding-window no-repeat-ngram guard. On CUDA the decode step and the
//! FastMTP draft's speculation run as CUDA graphs (on-device MoE routing and
//! greedy selection keep them graph-safe), with lossless greedy verification;
//! multi-page calls run a padded batch prefill and decode.

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
