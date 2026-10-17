use crate::Error;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub(super) struct RopeParameters {
    pub rope_theta: f64,
    pub rope_type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct TextConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub vocab_size: usize,
    pub max_position_embeddings: usize,
    pub head_dim: usize,
    pub q_lora_rank: Option<usize>,
    pub kv_lora_rank: usize,
    pub qk_head_dim: usize,
    pub qk_nope_head_dim: usize,
    pub qk_rope_head_dim: usize,
    pub v_head_dim: usize,
    pub rope_interleave: bool,
    pub rope_parameters: RopeParameters,
    pub rms_norm_eps: f64,
    pub hidden_act: String,
    pub attention_bias: bool,
    pub mlp_bias: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct OmniConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub num_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub merger_hidden_size: usize,
    pub out_hidden_size: usize,
    pub rms_norm_eps: f64,
    pub hidden_act: String,
    pub attention_bias: bool,
    pub rope_theta: f64,
    pub rope_theta_m: f64,
    pub rope_m_dim: usize,
    pub video_omni_chunked_mthw_rope: bool,
    pub video_omni_interleaved_mthw_rope: bool,
    pub use_sliding_window: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub(super) struct Config {
    pub model_type: String,
    pub architectures: Vec<String>,
    pub text_config: TextConfig,
    pub omni_config: OmniConfig,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl Config {
    pub fn load(path: &Path) -> Result<Self, Error> {
        let cfg: Self = serde_json::from_slice(&std::fs::read(path)?)
            .map_err(|e| Error::config(format!("invalid Youtu-Parsing-Omni config: {e}")))?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn validate(&self) -> Result<(), Error> {
        let text = &self.text_config;
        let omni = &self.omni_config;
        if self.model_type != "youtu_vita"
            || !self
                .architectures
                .iter()
                .any(|a| a == "YoutuVITAForCausalLM")
        {
            return Err(Error::config(
                "expected the YoutuVITAForCausalLM checkpoint",
            ));
        }
        if [
            text.hidden_size,
            text.intermediate_size,
            text.num_hidden_layers,
            text.num_attention_heads,
            text.vocab_size,
            text.max_position_embeddings,
            text.kv_lora_rank,
            text.qk_rope_head_dim,
            text.v_head_dim,
            omni.hidden_size,
            omni.intermediate_size,
            omni.num_hidden_layers,
            omni.num_attention_heads,
            omni.num_key_value_heads,
            omni.head_dim,
            omni.patch_size,
            omni.spatial_merge_size,
            omni.merger_hidden_size,
        ]
        .contains(&0)
            || text.q_lora_rank == Some(0)
            || text.qk_head_dim != text.qk_nope_head_dim + text.qk_rope_head_dim
            || text.head_dim != text.qk_rope_head_dim
            || !text.head_dim.is_multiple_of(2)
            || text.qk_head_dim < text.v_head_dim
            || !omni.head_dim.is_multiple_of(4)
            || !omni
                .num_attention_heads
                .is_multiple_of(omni.num_key_value_heads)
            || omni.num_channels != 3
            || omni.out_hidden_size != text.hidden_size
        {
            return Err(Error::config(
                "invalid Youtu-Parsing-Omni encoder/decoder dimensions",
            ));
        }
        if text.hidden_act != "silu"
            || omni.hidden_act != "silu"
            || text.attention_bias
            || text.mlp_bias
            || omni.attention_bias
            || text.rope_parameters.rope_type != "default"
            || omni.video_omni_interleaved_mthw_rope
            || omni.use_sliding_window
            || (omni.video_omni_chunked_mthw_rope && omni.rope_m_dim + 3 > omni.head_dim / 2)
        {
            return Err(Error::config(
                "unsupported Youtu-Parsing-Omni attention, MLP, or rotary configuration",
            ));
        }
        if [
            text.rope_parameters.rope_theta,
            omni.rope_theta,
            omni.rope_theta_m,
            text.rms_norm_eps,
            omni.rms_norm_eps,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(Error::config(
                "Youtu-Parsing-Omni rotary bases and norm epsilon must be finite and positive",
            ));
        }
        Ok(())
    }
}
