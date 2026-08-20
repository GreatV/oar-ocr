use crate::error::Error;
use serde::Deserialize;
use std::path::Path;

fn default_text_hidden_act() -> String {
    "silu".to_string()
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct MinerURopeScaling {
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub mrope_section: Vec<usize>,
}

/// Subset of the nested `text_config` block emitted by newer transformers
/// (>= 4.52) checkpoints such as `MinerU2.5-Pro-2605`. These checkpoints share
/// the Qwen2-VL backbone and the exact same weight layout as `MinerU2.5-2509`,
/// but relocate a handful of text-tower fields out of the config root and into
/// `text_config`. We only need `tie_word_embeddings` from it: the Pro config
/// omits the field at the root (so it would default to `false`), yet the
/// checkpoint ties the LM head to the input embeddings and ships no
/// `lm_head.weight` tensor. Resolving the effective flag from either location
/// keeps both the 2509 and Pro layouts loadable through the same path.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct MinerUTextConfig {
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

pub use crate::backbones::qwen2_vl::MinerUVisionConfig;

#[derive(Debug, Clone, Deserialize)]
pub struct MinerUConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default)]
    pub attention_dropout: f64,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub max_window_layers: usize,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default = "default_text_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub vision_token_id: u32,
    pub image_token_id: u32,
    pub video_token_id: u32,
    #[serde(default)]
    pub rope_scaling: MinerURopeScaling,
    pub vision_config: MinerUVisionConfig,
    /// Nested text-tower config present in newer transformers checkpoints
    /// (e.g. `MinerU2.5-Pro-2605`). Absent on the original 2509 layout.
    #[serde(default)]
    pub text_config: MinerUTextConfig,
}

impl MinerUConfig {
    /// Effective `tie_word_embeddings` flag, honouring both the legacy root
    /// field (2509) and the newer nested `text_config` field (Pro-2605).
    pub fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings || self.text_config.tie_word_embeddings
    }

    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, Error> {
        crate::utils::load_json_config(path, "MinerU2.5", "config.json")
    }

    pub fn head_dim(&self) -> Result<usize, Error> {
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(Error::Config {
                message: format!(
                    "MinerU2.5: hidden_size {} not divisible by num_attention_heads {}",
                    self.hidden_size, self.num_attention_heads
                ),
            });
        }
        Ok(self.hidden_size / self.num_attention_heads)
    }
}

pub use crate::backbones::qwen_vl_processing::{MinerUImageProcessorConfig, MinerUImageSize};
