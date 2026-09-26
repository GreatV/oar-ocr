use crate::error::Error;
use serde::Deserialize;
use std::path::Path;

// Re-exported from the defining submodule so the binding keeps its `pub`
// visibility (a `pub(crate) use` hop would downgrade it and make a public
// re-export impossible).
pub use crate::backbones::deep_encoder::{ClipConfig, SamConfig};
pub use crate::backbones::deepseek_v2::DeepSeekV2TextConfig;

/// Root `config.json` of a jina-ocr-v1 (DeepSeek-OCR) checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct JinaOcrConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    #[serde(flatten)]
    pub text: DeepSeekV2TextConfig,
    pub image_token_index: u32,
    pub projector_config: ProjectorConfig,
    pub vision_config: VisionConfig,
    #[serde(default)]
    pub num_nextn_predict_layers: Option<usize>,
    /// Reference `modeling_deepseekocr.py` always concatenates mrope views as
    /// [local, global, separator] and never reads this flag; accept only the
    /// value our implementation matches so a "tail" checkpoint fails loudly
    /// instead of silently reordering positions.
    #[serde(default)]
    pub global_view_pos: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ProjectorConfig {
    pub input_dim: usize,
    pub n_embed: usize,
    pub projector_type: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct SamWidthConfig {
    pub heads: usize,
    pub layers: usize,
    pub width: usize,
    #[serde(default)]
    pub global_attn_indexes: Vec<usize>,
    #[serde(default)]
    pub downsample_channels: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct ClipWidthConfig {
    pub heads: usize,
    pub layers: usize,
    pub patch_size: usize,
    pub width: usize,
    #[serde(default)]
    pub image_size: Option<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionConfig {
    pub model_type: String,
    pub model_name: String,
    pub image_size: usize,
    pub mlp_ratio: f64,
    pub width: VisionWidths,
}

#[derive(Debug, Clone, Deserialize)]
pub struct VisionWidths {
    #[serde(rename = "clip-l-14-224")]
    pub clip_l: ClipWidthConfig,
    #[serde(rename = "sam_vit_b")]
    pub sam_vit_b: SamWidthConfig,
}

impl VisionConfig {
    /// SAM ViT patch grid side for `size`-pixel inputs (`size / 16`).
    pub fn sam_grid_side(&self, size: usize) -> Result<usize, Error> {
        let patch = 16usize;
        if size == 0 || !size.is_multiple_of(patch) {
            return Err(Error::Config {
                message: format!(
                    "JinaOCR vision input size {size} must be a positive multiple of the SAM patch size {patch}"
                ),
            });
        }
        Ok(size / patch)
    }
}

impl JinaOcrConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, Error> {
        let cfg: Self =
            crate::runtime::checkpoint::load_json_config(path, "JinaOCR", "config.json")?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn head_dim(&self) -> Result<usize, Error> {
        self.text.head_dim()
    }

    /// SAM-B + CLIP-L geometry from the checkpoint's vision config.
    pub fn vision_configs(&self) -> Result<(SamConfig, ClipConfig), Error> {
        let sam = &self.vision_config.width.sam_vit_b;
        let clip = &self.vision_config.width.clip_l;
        Ok((
            SamConfig {
                depth: sam.layers,
                width: sam.width,
                heads: sam.heads,
                patch_size: 16,
                global_attn_indexes: sam.global_attn_indexes.clone(),
                window_size: 14,
                // SAM ViT-B pretrains on 1024×1024 with patch 16.
                pretrained_grid_side: self
                    .vision_config
                    .sam_grid_side(self.vision_config.image_size)?,
                out_channels: sam.downsample_channels.last().copied().ok_or_else(|| {
                    Error::Config {
                        message: "JinaOCR SAM config is missing downsample_channels".to_string(),
                    }
                })?,
            },
            ClipConfig {
                depth: clip.layers,
                width: clip.width,
                heads: clip.heads,
                ffn_hidden_size: 4096,
                layer_norm_eps: 1e-5,
                pretrained_grid_side: clip.image_size.map_or(16, |size| size / clip.patch_size),
            },
        ))
    }

    pub fn validate(&self) -> Result<(), Error> {
        if let Some(view_pos) = &self.global_view_pos
            && view_pos != "head"
        {
            return Err(Error::Config {
                message: format!(
                    "JinaOCR expects global_view_pos 'head' (the reference fixes [local, global, sep] order), got '{view_pos}'"
                ),
            });
        }
        if self.model_type != "deepseek_vl_v2" {
            return Err(Error::Config {
                message: format!(
                    "JinaOCR expected model_type 'deepseek_vl_v2', got '{}'",
                    self.model_type
                ),
            });
        }
        if let Some(architecture) = self.architectures.first()
            && architecture != "DeepseekOCRForCausalLM"
        {
            return Err(Error::Config {
                message: format!(
                    "JinaOCR expected architecture 'DeepseekOCRForCausalLM', got '{architecture}'"
                ),
            });
        }
        self.text.validate()?;
        // Vision tower geometry: every field that later appears as a divisor
        // or a shape is validated here so a bad checkpoint fails at load time
        // with a config error instead of a division-by-zero panic.
        let sam = &self.vision_config.width.sam_vit_b;
        let clip = &self.vision_config.width.clip_l;
        for (name, heads, width) in [
            ("sam_vit_b", sam.heads, sam.width),
            ("clip-l", clip.heads, clip.width),
        ] {
            if heads == 0 || width == 0 || !width.is_multiple_of(heads) {
                return Err(Error::Config {
                    message: format!(
                        "JinaOCR vision {name}: width {width} must be non-zero and divide evenly into {heads} heads"
                    ),
                });
            }
        }
        for (name, layers) in [("sam_vit_b", sam.layers), ("clip-l", clip.layers)] {
            if layers == 0 {
                return Err(Error::Config {
                    message: format!("JinaOCR vision {name}: layers must be non-zero"),
                });
            }
        }
        if clip.patch_size == 0 {
            return Err(Error::Config {
                message: "JinaOCR vision clip-l: patch_size must be non-zero".to_string(),
            });
        }
        if let Some(size) = clip.image_size
            && (size == 0 || !size.is_multiple_of(clip.patch_size))
        {
            return Err(Error::Config {
                message: format!(
                    "JinaOCR vision clip-l: image_size {size} must be a positive multiple of patch_size {}",
                    clip.patch_size
                ),
            });
        }
        // SAM pretrained grid: image_size must be a positive multiple of the
        // SAM patch size (16).
        self.vision_config
            .sam_grid_side(self.vision_config.image_size)?;
        for (name, token_id) in [
            ("bos_token_id", self.text.bos_token_id),
            ("eos_token_id", self.text.eos_token_id),
            ("pad_token_id", self.text.pad_token_id),
            ("image_token_index", self.image_token_index),
        ] {
            if token_id as usize >= self.text.vocab_size {
                return Err(Error::Config {
                    message: format!(
                        "JinaOCR {name} {token_id} is outside vocab_size {}",
                        self.text.vocab_size
                    ),
                });
            }
        }
        // FastMTP draft weights ship in the checkpoint; this decoder-only port
        // ignores them exactly like the reference transformers path.
        if self.num_nextn_predict_layers.unwrap_or(0) > 1 {
            return Err(Error::Config {
                message: "JinaOCR supports at most one MTP draft layer".to_string(),
            });
        }
        if self.text.tie_word_embeddings {
            return Err(Error::Config {
                message: "JinaOCR expects a separate lm_head, not tied embeddings".to_string(),
            });
        }
        if self.projector_config.projector_type != "linear" {
            return Err(Error::Config {
                message: format!(
                    "JinaOCR expected projector_type 'linear', got '{}'",
                    self.projector_config.projector_type
                ),
            });
        }
        if self.projector_config.n_embed != self.text.hidden_size {
            return Err(Error::Config {
                message: format!(
                    "JinaOCR projector n_embed ({}) must equal hidden_size ({})",
                    self.projector_config.n_embed, self.text.hidden_size
                ),
            });
        }
        if self.vision_config.model_type != "vision" {
            return Err(Error::Config {
                message: format!(
                    "JinaOCR expected vision model_type 'vision', got '{}'",
                    self.vision_config.model_type
                ),
            });
        }
        if self.vision_config.width.sam_vit_b.width != 768
            || self.vision_config.width.clip_l.width != 1024
            || self.vision_config.width.clip_l.patch_size != 14
        {
            return Err(Error::Config {
                message: "JinaOCR expects the SAM ViT-B + CLIP-L encoder pair".to_string(),
            });
        }
        // SAM's channel-doubling downsampler must land on the projector input.
        let (sam_cfg, clip_cfg) = self.vision_configs()?;
        let input_dim = clip_cfg.width + sam_cfg.out_channels;
        if self.projector_config.input_dim != input_dim {
            return Err(Error::Config {
                message: format!(
                    "JinaOCR projector input_dim must be {input_dim} (CLIP-L {} + SAM {}), got {}",
                    clip_cfg.width, sam_cfg.out_channels, self.projector_config.input_dim
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const CONFIG: &str = r#"
    {
      "architectures": ["DeepseekOCRForCausalLM"],
      "attention_bias": false,
      "attention_dropout": 0.0,
      "bos_token_id": 0,
      "candidate_resolutions": [[1024, 1024]],
      "dtype": "bfloat16",
      "eos_token_id": 1,
      "ep_size": 1,
      "first_k_dense_replace": 1,
      "global_view_pos": "head",
      "hidden_act": "silu",
      "hidden_size": 1280,
      "image_token_index": 128815,
      "initializer_range": 0.02,
      "intermediate_size": 6848,
      "kv_lora_rank": null,
      "language_config": {},
      "lm_head": true,
      "max_position_embeddings": 32768,
      "model_type": "deepseek_vl_v2",
      "moe_intermediate_size": 896,
      "moe_layer_freq": 1,
      "n_group": 1,
      "n_routed_experts": 64,
      "n_shared_experts": 2,
      "num_attention_heads": 10,
      "num_experts_per_tok": 6,
      "num_hidden_layers": 12,
      "num_key_value_heads": 10,
      "num_nextn_predict_layers": 1,
      "norm_topk_prob": false,
      "pad_token_id": 2,
      "pretraining_tp": 1,
      "projector_config": {"input_dim": 2048, "model_type": "mlp_projector", "n_embed": 1280, "projector_type": "linear"},
      "q_lora_rank": null,
      "qk_nope_head_dim": 0,
      "qk_rope_head_dim": 0,
      "rms_norm_eps": 1e-06,
      "rope_theta": 1000000,
      "routed_scaling_factor": 1.0,
      "scoring_func": "softmax",
      "tie_word_embeddings": false,
      "tile_tag": "2D",
      "topk_group": 1,
      "topk_method": "greedy",
      "use_cache": true,
      "use_mla": false,
      "v_head_dim": 0,
      "vision_config": {
        "image_size": 1024,
        "mlp_ratio": 3.7362,
        "model_name": "deeplip_b_l",
        "model_type": "vision",
        "width": {
          "clip-l-14-224": {"heads": 16, "image_size": 224, "layers": 24, "patch_size": 14, "width": 1024},
          "sam_vit_b": {"downsample_channels": [512, 1024], "global_attn_indexes": [2, 5, 8, 11], "heads": 12, "layers": 12, "width": 768}
        }
      },
      "vocab_size": 129280
    }
    "#;

    #[test]
    fn parses_official_checkpoint_shape() {
        let cfg: JinaOcrConfig = serde_json::from_str(CONFIG).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.text.num_hidden_layers, 12);
        assert_eq!(cfg.text.first_k_dense_replace, 1);
        assert_eq!(cfg.text.n_routed_experts, 64);
        assert_eq!(cfg.text.n_shared_experts, 2);
        assert_eq!(cfg.text.num_experts_per_tok, 6);
        assert_eq!(cfg.text.moe_intermediate_size, 896);
        assert_eq!(cfg.head_dim().unwrap(), 128);
        assert_eq!(cfg.text.rope_theta, 1_000_000.0);
        assert_eq!(cfg.image_token_index, 128815);
        assert_eq!(cfg.vision_config.sam_grid_side(1024).unwrap(), 64);
        assert_eq!(cfg.vision_config.sam_grid_side(640).unwrap(), 40);
        assert!(cfg.vision_config.sam_grid_side(641).is_err());
        let (sam, clip) = cfg.vision_configs().unwrap();
        assert_eq!(sam.out_channels, 1024);
        assert_eq!(sam.pretrained_grid_side, 64);
        assert_eq!(sam.window_size, 14);
        assert_eq!(clip.ffn_hidden_size, 4096);
        assert_eq!(clip.pretrained_grid_side, 16);
    }

    #[test]
    fn rejects_mla_checkpoints() {
        let mut cfg: JinaOcrConfig = serde_json::from_str(CONFIG).unwrap();
        cfg.text.use_mla = true;
        assert!(cfg.validate().unwrap_err().to_string().contains("use_mla"));
    }

    #[test]
    fn rejects_unsupported_router_variants() {
        let mut cfg: JinaOcrConfig = serde_json::from_str(CONFIG).unwrap();
        cfg.text.norm_topk_prob = true;
        assert!(cfg.validate().unwrap_err().to_string().contains("router"));
    }

    #[test]
    fn rejects_gqa_checkpoints() {
        let mut cfg: JinaOcrConfig = serde_json::from_str(CONFIG).unwrap();
        cfg.text.num_key_value_heads = 5;
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("multi-head attention")
        );
    }

    #[test]
    fn rejects_bias_and_mla_shaped_checkpoints() {
        // attention_bias/mlp_bias would be silently dropped by the no-bias
        // loaders; MLA shape fields are plain-MHA-only.
        let mut cfg: JinaOcrConfig = serde_json::from_str(CONFIG).unwrap();
        cfg.text.attention_bias = true;
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("attention_bias")
        );

        let mut cfg: JinaOcrConfig = serde_json::from_str(CONFIG).unwrap();
        cfg.text.mlp_bias = true;
        assert!(cfg.validate().unwrap_err().to_string().contains("mlp_bias"));

        for mutate in [
            (|c: &mut JinaOcrConfig| c.text.q_lora_rank = Some(1536)) as fn(&mut JinaOcrConfig),
            (|c: &mut JinaOcrConfig| c.text.kv_lora_rank = Some(512)) as fn(&mut JinaOcrConfig),
            (|c: &mut JinaOcrConfig| c.text.qk_nope_head_dim = 128) as fn(&mut JinaOcrConfig),
            (|c: &mut JinaOcrConfig| c.text.qk_rope_head_dim = 64) as fn(&mut JinaOcrConfig),
            (|c: &mut JinaOcrConfig| c.text.v_head_dim = 128) as fn(&mut JinaOcrConfig),
        ] {
            let mut cfg: JinaOcrConfig = serde_json::from_str(CONFIG).unwrap();
            mutate(&mut cfg);
            let err = cfg.validate().unwrap_err().to_string();
            assert!(err.contains("MLA"), "unexpected error: {err}");
        }
    }

    #[test]
    fn rejects_zero_heads_and_undividable_vision_widths() {
        let mut cfg: JinaOcrConfig = serde_json::from_str(CONFIG).unwrap();
        cfg.vision_config.width.sam_vit_b.heads = 0;
        let err = cfg.validate().unwrap_err().to_string();
        assert!(err.contains("sam_vit_b"), "unexpected error: {err}");

        let mut cfg: JinaOcrConfig = serde_json::from_str(CONFIG).unwrap();
        cfg.vision_config.width.clip_l.heads = 0;
        assert!(cfg.validate().unwrap_err().to_string().contains("clip-l"));

        let mut cfg: JinaOcrConfig = serde_json::from_str(CONFIG).unwrap();
        cfg.vision_config.width.sam_vit_b.heads = 7; // 768 % 7 != 0
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn rejects_zero_vision_layers_patch_size_and_bad_image_size() {
        let mut cfg: JinaOcrConfig = serde_json::from_str(CONFIG).unwrap();
        cfg.vision_config.width.sam_vit_b.layers = 0;
        assert!(cfg.validate().unwrap_err().to_string().contains("layers"));

        let mut cfg: JinaOcrConfig = serde_json::from_str(CONFIG).unwrap();
        cfg.vision_config.width.clip_l.patch_size = 0;
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("patch_size")
        );

        // clip image_size not divisible by its patch size
        let mut cfg: JinaOcrConfig = serde_json::from_str(CONFIG).unwrap();
        cfg.vision_config.width.clip_l.image_size = Some(225);
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("image_size")
        );

        // SAM image_size not a multiple of the 16px patch
        let mut cfg: JinaOcrConfig = serde_json::from_str(CONFIG).unwrap();
        cfg.vision_config.image_size = 641;
        assert!(cfg.validate().is_err());
        cfg.vision_config.image_size = 0;
        assert!(cfg.validate().is_err());
    }
}
