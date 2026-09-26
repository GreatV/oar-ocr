//! jina-ocr-v1 document-to-Markdown model support.
//!
//! Native Rust inference for the Jina jina-ocr-v1 checkpoint: a SAM ViT-B +
//! CLIP-L dual encoder feeding a 12-layer DeepSeek-V2 MoE decoder through a
//! linear projector. Pages are encoded as a padded 1024px global view plus
//! (when the page is larger than 640px) up to nine 640px tiles
//! (`crop_mode`), and generation follows the official greedy recipe with a
//! sliding-window no-repeat-ngram guard. On CUDA the decode step runs as a
//! CUDA graph (on-device MoE routing and greedy selection keep it
//! graph-safe). The checkpoint's FastMTP draft head can additionally run
//! speculative decoding (token-identical in exact arithmetic; bf16 kernel
//! noise can flip near-tie picks — see the model module docs), but it is
//! opt-in
//! ([`JinaOcrLoadOptions::with_mtp`] or `OAR_JINAOCR_ENABLE_MTP`): on the
//! OmniDocBench demo pages (RTX 4090, bf16) adaptive MTP lost to graphed
//! plain decoding on 17 of 18 pages. Multi-page calls run a padded batch
//! prefill and decode.

mod adapter;
mod config;
mod model;
mod mtp;
mod parser;
pub mod processing;

pub use config::{DeepSeekV2TextConfig, JinaOcrConfig};
pub use model::{DEFAULT_MAX_NEW_TOKENS, JinaOcr, JinaOcrLoadOptions};
pub use parser::JinaOcrParseOptions;
pub use processing::DEFAULT_OCR_PROMPT;
