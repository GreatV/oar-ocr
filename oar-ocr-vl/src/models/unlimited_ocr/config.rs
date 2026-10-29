//! Checkpoint configuration for Unlimited-OCR-2.

use crate::Error;
use crate::backbones::deep_encoder::{ClipConfig, SamConfig};
use crate::backbones::deepseek_v2::DeepSeekV2TextConfig;
use crate::runtime::checkpoint::load_json_config;
use serde::Deserialize;
use std::path::Path;

/// Root `config.json` of an `unlimited-ocr` checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct UnlimitedOcrConfig {
    pub model_type: String,
    #[serde(flatten)]
    pub text: DeepSeekV2TextConfig,
    pub projector_config: ProjectorConfig,
    pub vision_config: VisionConfig,
    pub sparse_attention_config: SparseAttentionConfig,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProjectorConfig {
    pub input_dim: usize,
    pub n_embed: usize,
    pub projector_type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub image_size: usize,
    pub width: VisionWidths,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionWidths {
    #[serde(rename = "clip-l-14-224")]
    pub clip_l: ClipWidth,
    pub sam_vit_b: SamWidth,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClipWidth {
    pub heads: usize,
    pub image_size: usize,
    pub layers: usize,
    pub patch_size: usize,
    pub width: usize,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SamWidth {
    pub downsample_channels: Vec<usize>,
    pub global_attn_indexes: Vec<usize>,
    pub heads: usize,
    pub layers: usize,
    pub width: usize,
}

/// The DASH sparse-attention settings read at inference.
#[derive(Debug, Clone, Deserialize)]
pub struct SparseAttentionConfig {
    pub use_sparse_attention: bool,
    pub image_token_id: u32,
    pub page_token_id: u32,
    pub sparse_block_size: usize,
    pub sparse_local_block: usize,
    pub sparse_index_dim: usize,
    pub sparse_attention_freq: Vec<usize>,
    pub dash_topk_blocks: usize,
    pub dash_scale_per_layer: Vec<f32>,
    pub dash_floor_per_layer: Vec<f32>,
    pub dash_sigma: f32,
    pub dash_stage0: String,
    pub dash_image_only: bool,
    pub image_bucket_constraint: bool,
    pub sparse_main_rope_mode: String,
    pub sparse_rope_mode: String,
    pub sparse_main_rope_tok_freq_mode: String,
    pub sparse_rope_tok_theta: f32,
    pub sparse_rope_page_theta: f32,
    pub sparse_rope_page_pairs: usize,
    pub sparse_rope_page_mod: usize,
    #[serde(default)]
    pub sparse_rope_page_span: usize,
}

impl UnlimitedOcrConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, Error> {
        let config: Self = load_json_config(path, "Unlimited-OCR", "config.json")?;
        config.validate()?;
        Ok(config)
    }

    fn validate(&self) -> Result<(), Error> {
        let text = &self.text;
        // The checkpoint spells out v_head_dim for its plain MHA heads; the
        // shared validation wants the MLA fields zeroed.
        let mut mha = text.clone();
        if mha.v_head_dim == mha.head_dim()? {
            mha.v_head_dim = 0;
        }
        mha.validate()?;
        if self.model_type != "unlimited-ocr" {
            return Err(Error::config(format!(
                "Unlimited-OCR expected model_type 'unlimited-ocr', got '{}'",
                self.model_type
            )));
        }
        if text.use_mla || text.tie_word_embeddings || text.attention_bias {
            return Err(Error::config(
                "Unlimited-OCR expects MHA attention without bias and an untied lm_head",
            ));
        }
        if self.projector_config.projector_type != "linear"
            || self.projector_config.n_embed != text.hidden_size
        {
            return Err(Error::config(
                "Unlimited-OCR expects a linear projector into the text hidden size",
            ));
        }
        let sparse = &self.sparse_attention_config;
        let layers = text.num_hidden_layers;
        let head_dim = text.head_dim()?;
        // Only the inference path the checkpoint ships is implemented; any
        // other router or RoPE setting would silently compute something else.
        let supported = sparse.use_sparse_attention
            && sparse.dash_stage0 == "head_cls"
            && sparse.dash_image_only
            && sparse.image_bucket_constraint
            && sparse.sparse_main_rope_mode == "page2d"
            && sparse.sparse_rope_mode == "page2d"
            && sparse.sparse_main_rope_tok_freq_mode == "resample"
            && sparse.sparse_rope_page_span == 0
            && sparse.sparse_index_dim == head_dim
            && sparse.sparse_rope_page_pairs > 0
            && sparse.sparse_rope_page_pairs < head_dim / 2
            && sparse.sparse_block_size > 0
            && sparse.sparse_local_block > 0
            && sparse.dash_topk_blocks > 0
            && sparse.dash_sigma > 0.0
            && sparse.sparse_attention_freq.len() == layers
            && sparse.sparse_attention_freq.iter().all(|&freq| freq == 1)
            && sparse.dash_scale_per_layer.len() == layers
            && sparse.dash_floor_per_layer.len() == layers;
        if !supported {
            return Err(Error::config(
                "Unlimited-OCR supports only the shipped DASH configuration (head_cls router, \
                 page2d RoPE, image-only page-aware candidates on every layer)",
            ));
        }
        if text.num_key_value_heads != text.num_attention_heads {
            return Err(Error::config(
                "Unlimited-OCR expects multi-head attention (num_key_value_heads == num_attention_heads)",
            ));
        }
        Ok(())
    }

    /// SAM-B + CLIP-L geometry of the DeepEncoder.
    pub fn vision_configs(&self) -> Result<(SamConfig, ClipConfig), Error> {
        let sam = &self.vision_config.width.sam_vit_b;
        let clip = &self.vision_config.width.clip_l;
        let out_channels =
            sam.downsample_channels.last().copied().ok_or_else(|| {
                Error::config("Unlimited-OCR SAM config lacks downsample_channels")
            })?;
        if clip.width + out_channels != self.projector_config.input_dim {
            return Err(Error::config(
                "Unlimited-OCR projector input_dim does not match CLIP + SAM widths",
            ));
        }
        Ok((
            SamConfig {
                depth: sam.layers,
                width: sam.width,
                heads: sam.heads,
                patch_size: 16,
                global_attn_indexes: sam.global_attn_indexes.clone(),
                window_size: 14,
                pretrained_grid_side: self.vision_config.image_size / 16,
                out_channels,
            },
            ClipConfig {
                depth: clip.layers,
                width: clip.width,
                heads: clip.heads,
                ffn_hidden_size: 4096,
                layer_norm_eps: 1e-5,
                pretrained_grid_side: clip.image_size / clip.patch_size,
            },
        ))
    }
}
