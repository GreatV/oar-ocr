//! WeVisDoc document-to-Markdown model support.
//!
//! Native Rust inference for the Tencent WeVisDoc checkpoints (Qwen3-VL
//! backbone, 2B and 4B). WeVisDoc turns a page image into structured Markdown
//! with LaTeX formulas and HTML tables, following the official
//! `wevisdoc/local.py` inference recipe (greedy decoding, WeDocKit system
//! prompt, 8192-token default budget).

mod adapter;
mod config;
mod model;
mod parser;
pub mod processing;

pub use config::{Qwen3VlRopeScaling, Qwen3VlTextConfig, Qwen3VlVisionConfig, WeVisDocConfig};
pub use model::{
    DEFAULT_MAX_NEW_TOKENS, DEFAULT_PROMPT, DEFAULT_SYSTEM_PROMPT, WeVisDoc, build_prompt,
};
pub use parser::WeVisDocParseOptions;
