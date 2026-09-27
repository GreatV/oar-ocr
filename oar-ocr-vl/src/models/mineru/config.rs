use crate::backbones::qwen2_vl::Qwen2VlTextConfig;
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

pub use crate::backbones::qwen2_vl::Qwen2VlVisionConfig;

/// Deprecated alias kept for the published `0.9.x` API surface.
#[deprecated(since = "0.9.3", note = "use Qwen2VlVisionConfig")]
pub type MinerUVisionConfig = Qwen2VlVisionConfig;

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
    pub vision_config: Qwen2VlVisionConfig,
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

    /// Convert to the shared Qwen2 text-tower configuration. MinerU2.5
    /// keeps the q/k/v projection biases and has no per-head q/k norm.
    pub fn qwen2_vl_text_config(&self) -> Result<Qwen2VlTextConfig, Error> {
        Ok(Qwen2VlTextConfig {
            model_name: "MinerU2.5",
            vocab_size: self.vocab_size,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_hidden_layers: self.num_hidden_layers,
            num_attention_heads: self.num_attention_heads,
            num_key_value_heads: self.num_key_value_heads,
            rms_norm_eps: self.rms_norm_eps,
            rope_theta: self.rope_theta,
            max_position_embeddings: self.max_position_embeddings,
            head_dim: self.head_dim()?,
            mrope_section: self.rope_scaling.mrope_section.clone(),
            attention_bias: true,
            qk_head_norm: false,
            graph_disable_env: "OAR_MINERU_DISABLE_CUDA_GRAPH",
            decode_cache_len: 16_384,
        })
    }
}

pub use crate::backbones::qwen_vl_processing::{QwenVlImageProcessorConfig, QwenVlImageSize};

/// Deprecated alias kept for the published `0.9.x` API surface.
#[deprecated(since = "0.9.3", note = "use QwenVlImageProcessorConfig")]
pub type MinerUImageProcessorConfig = QwenVlImageProcessorConfig;

/// Deprecated alias kept for the published `0.9.x` API surface.
#[deprecated(since = "0.9.3", note = "use QwenVlImageSize")]
pub type MinerUImageSize = QwenVlImageSize;
