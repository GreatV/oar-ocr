use crate::Error;
use crate::backbones::sdar::SdarConfig;
use serde::Deserialize;

#[derive(Debug, Deserialize)]
pub(super) struct Config {
    model_type: String,
    pub llm_config: TextConfig,
    pub vit_config: VisionConfig,
    pub visual_vocab_size: usize,
    hidden_size: usize,
}

#[derive(Debug, Deserialize)]
pub(super) struct TextConfig {
    model_type: String,
    #[serde(flatten)]
    pub decoder: SdarConfig,
    rope_scaling: Option<serde_json::Value>,
    #[serde(default)]
    use_sliding_window: bool,
    layer_types: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
pub(super) struct VisionConfig {
    model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_stride: usize,
    pub layer_norm_eps: f64,
    num_channels: usize,
    temporal_patch_size: usize,
    num_patches: i64,
    hidden_act: String,
    preserve_original_pe: bool,
    use_rope: bool,
    fullatt_block_indexes: Option<serde_json::Value>,
}

impl Config {
    pub fn validate(&self) -> Result<(), Error> {
        let text = &self.llm_config;
        let t = &text.decoder;
        let v = &self.vit_config;
        if v.fullatt_block_indexes.is_some() {
            return Err(Error::config(
                "OXR requires fullatt_block_indexes=null (full vision attention); windowed checkpoints are unsupported",
            ));
        }
        if self.model_type != "oxr" || text.model_type != "qwen3" || v.model_type != "siglip2_navit"
        {
            return Err(Error::config(
                "OXR requires oxr / qwen3 / siglip2_navit configurations",
            ));
        }
        if t.vocab_size == 0
            || t.hidden_size == 0
            || t.intermediate_size == 0
            || t.num_hidden_layers == 0
            || t.num_attention_heads == 0
            || t.num_key_value_heads == 0
            || t.max_position_embeddings == 0
            || self.visual_vocab_size <= 4
            || self.hidden_size != t.hidden_size
        {
            return Err(Error::config(
                "OXR has invalid text/visual vocabulary dimensions",
            ));
        }
        if t.head_dim()? == 0
            || !t.head_dim()?.is_multiple_of(2)
            || !t.num_attention_heads.is_multiple_of(t.num_key_value_heads)
            || !t.tie_word_embeddings
            || t.attention_bias
            || t.hidden_act != "silu"
            || text.rope_scaling.is_some()
            || text.use_sliding_window
            || text.layer_types.as_ref().is_some_and(|layers| {
                layers.len() != t.num_hidden_layers
                    || layers.iter().any(|layer| layer != "full_attention")
            })
        {
            return Err(Error::config(
                "OXR requires a tied dense Qwen3 decoder with ordinary RoPE and full causal attention",
            ));
        }
        if v.hidden_size == 0
            || v.intermediate_size == 0
            || v.num_hidden_layers == 0
            || v.num_attention_heads == 0
            || !v.hidden_size.is_multiple_of(v.num_attention_heads)
            || !(v.hidden_size / v.num_attention_heads).is_multiple_of(4)
            || v.patch_size != 16
            || v.hidden_stride != 2
            || v.temporal_patch_size != 1
            || v.num_channels != 3
            || v.num_patches != -1
            || v.image_size == 0
            || !v.image_size.is_multiple_of(v.patch_size)
            || !v.preserve_original_pe
            || !v.use_rope
            || v.hidden_act != "gelu_pytorch_tanh"
        {
            return Err(Error::config("unsupported OXR SigLIP2 NaViT configuration"));
        }
        for value in [t.rms_norm_eps, t.rope_theta, v.layer_norm_eps] {
            if !value.is_finite() || value <= 0.0 {
                return Err(Error::config(
                    "OXR normalization epsilon and RoPE theta must be finite and positive",
                ));
            }
        }
        Ok(())
    }
}

#[derive(Debug, Deserialize)]
pub(super) struct ImageConfig {
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
    pub rescale_factor: f64,
    resample: u32,
    do_resize: bool,
    do_rescale: bool,
    do_normalize: bool,
}

impl ImageConfig {
    pub fn validate(&self) -> Result<(), Error> {
        if self.resample != 2 || !self.do_resize || !self.do_rescale || !self.do_normalize {
            return Err(Error::config(
                "OXR requires Pillow bilinear resize, rescaling and normalization",
            ));
        }
        if !self.rescale_factor.is_finite()
            || self.rescale_factor <= 0.0
            || self.image_mean.iter().any(|v| !v.is_finite())
            || self.image_std.iter().any(|v| !v.is_finite() || *v <= 0.0)
        {
            return Err(Error::config(
                "OXR image normalization values must be finite with positive scales",
            ));
        }
        Ok(())
    }
}

#[derive(Deserialize)]
pub(super) struct GenerationConfig {
    pub eos_token_id: Vec<u32>,
}
