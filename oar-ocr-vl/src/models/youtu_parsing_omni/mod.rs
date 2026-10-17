//! Native Youtu-Parsing-Omni image and document parsing.
//!
//! Loads `tencent/Youtu-Parsing-Omni` checkpoints directly with Candle. The
//! image path uses the shared omni encoder and a multi-head latent-attention
//! decoder. Audio and video inputs are outside this image-based API.

mod config;
mod layers;
mod model;
mod parser;
mod processing;
#[cfg(test)]
mod tests;
mod text;
mod vision;

pub use model::YoutuParsingOmni;
pub use parser::YoutuParsingOmniParseOptions;

/// Official [document parsing instruction](https://github.com/TencentCloudADP/youtu-parsing/blob/main/youtu_parsing_omni/prompts/youtu_parsing_omni.json), including the OmniSchema JSON format.
pub const DEFAULT_PROMPT: &str = include_str!("document_prompt.txt");
/// Official greedy generation budget.
pub const DEFAULT_MAX_NEW_TOKENS: usize = 32_768;
