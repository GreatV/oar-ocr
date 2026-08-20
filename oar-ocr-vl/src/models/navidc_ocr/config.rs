use crate::error::Error;
use serde::Deserialize;
use std::path::Path;

fn default_text_hidden_act() -> String {
    "silu".to_string()
}

/// `rope_scaling` block. Only `mrope_section` is consumed — the `type` /
/// `rope_type` keys (NaviDC-OCR ships both, value `"default"`) need no
/// handling, and undeclared keys are ignored by serde by default.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct NaviDcRopeScaling {
    #[serde(default)]
    pub mrope_section: Vec<usize>,
}

/// Subset of the nested `text_config` block emitted by newer transformers
/// (>= 4.52) checkpoints such as NaviDC-OCR. The root config carries nearly
/// all text-tower fields; only `pad_token_id`, `tie_word_embeddings`,
/// `layer_types`, and a redundant `head_dim`/`eos_token_id` live here. Each
/// accessor on [`NaviDcConfig`] resolves the effective value from both spots.
#[derive(Debug, Clone, Default, Deserialize)]
pub struct NaviDcTextConfig {
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    #[serde(default)]
    pub layer_types: Vec<String>,
}

pub use crate::backbones::qwen25_vl::NaviDcVisionConfig;

/// Root `config.json` for NaviDC-OCR (`Qwen2_5_VLForConditionalGeneration`).
#[derive(Debug, Clone, Deserialize)]
pub struct NaviDcConfig {
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
    /// Explicit attention head dim (128). When present it overrides
    /// `hidden_size / num_attention_heads` (which would be 64 here).
    #[serde(default)]
    pub head_dim: Option<usize>,
    #[serde(default = "default_text_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub bos_token_id: u32,
    #[serde(default)]
    pub eos_token_id: u32,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub vision_token_id: u32,
    pub image_token_id: u32,
    #[serde(default)]
    pub video_token_id: u32,
    #[serde(default)]
    pub rope_scaling: NaviDcRopeScaling,
    pub vision_config: NaviDcVisionConfig,
    #[serde(default)]
    pub text_config: NaviDcTextConfig,
}

impl NaviDcConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, Error> {
        crate::utils::load_json_config(path, "NaviDC-OCR", "config.json")
    }

    /// Effective attention head dim: the explicit `head_dim` field when set
    /// (NaviDC-OCR ships 128, larger than `hidden/heads` = 64), falling back
    /// to `hidden_size / num_attention_heads`.
    pub fn head_dim(&self) -> Result<usize, Error> {
        if let Some(head_dim) = self.head_dim.or(self.text_config.head_dim) {
            return Ok(head_dim);
        }
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(Error::Config {
                message: format!(
                    "NaviDC-OCR: hidden_size {} not divisible by num_attention_heads {} and no explicit head_dim",
                    self.hidden_size, self.num_attention_heads
                ),
            });
        }
        Ok(self.hidden_size / self.num_attention_heads)
    }

    /// Effective `tie_word_embeddings` flag, honouring both the root field
    /// and the nested `text_config` field (where NaviDC-OCR sets it true).
    pub fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings || self.text_config.tie_word_embeddings
    }

    /// Effective pad token id; NaviDC-OCR only declares it in `text_config`.
    pub fn effective_pad_token_id(&self) -> Option<u32> {
        self.pad_token_id.or(self.text_config.pad_token_id)
    }

    /// Effective eos token id; the root field wins when present.
    pub fn effective_eos_token_id(&self) -> u32 {
        if self.eos_token_id != 0 {
            self.eos_token_id
        } else {
            self.text_config.eos_token_id.unwrap_or(0)
        }
    }

    pub fn mrope_section(&self) -> Result<&[usize], Error> {
        if self.rope_scaling.mrope_section.is_empty() {
            return Err(Error::Config {
                message: "NaviDC-OCR: rope_scaling.mrope_section is required".to_string(),
            });
        }
        Ok(&self.rope_scaling.mrope_section)
    }

    pub fn validate(&self) -> Result<(), Error> {
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(Error::Config {
                message: format!(
                    "NaviDC-OCR: num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                    self.num_attention_heads, self.num_key_value_heads
                ),
            });
        }

        let head_dim = self.head_dim()?;
        let section_sum: usize = self.mrope_section()?.iter().sum();
        if section_sum * 2 != head_dim {
            return Err(Error::Config {
                message: format!(
                    "NaviDC-OCR: mrope_section sums to {section_sum}; doubled ({}) must equal head_dim {head_dim}",
                    section_sum * 2
                ),
            });
        }

        if self
            .text_config
            .layer_types
            .iter()
            .any(|layer| layer == "sliding_attention")
        {
            return Err(Error::Config {
                message: "NaviDC-OCR: sliding_attention layers are not supported".to_string(),
            });
        }

        let vision = &self.vision_config;
        vision.head_dim()?;
        for &index in &vision.fullatt_block_indexes {
            if index >= vision.depth {
                return Err(Error::Config {
                    message: format!(
                        "NaviDC-OCR: fullatt_block_indexes entry {index} >= vision depth {}",
                        vision.depth
                    ),
                });
            }
        }
        let window_tokens = vision
            .spatial_merge_size
            .checked_mul(vision.patch_size)
            .ok_or_else(|| Error::Config {
                message: "NaviDC-OCR: vision spatial_merge_size * patch_size overflow".to_string(),
            })?;
        if window_tokens == 0 || !vision.window_size.is_multiple_of(window_tokens) {
            return Err(Error::Config {
                message: format!(
                    "NaviDC-OCR: window_size ({}) must be a multiple of spatial_merge_size * patch_size ({window_tokens})",
                    vision.window_size
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) fn test_config_json() -> String {
    fixture_json_inner()
}

/// Minimal config matching the real StarDoc-AI/NaviDC-OCR layout: text
/// fields at the root AND in `text_config`, both `in_chans` spellings, and
/// both `type`/`rope_type` rope keys.
#[cfg(test)]
fn fixture_json_inner() -> String {
    r#"{
          "architectures": ["Qwen2_5_VLForConditionalGeneration"],
          "bos_token_id": 151643,
          "eos_token_id": 151645,
          "head_dim": 128,
          "hidden_act": "silu",
          "hidden_size": 1024,
          "intermediate_size": 3072,
          "num_attention_heads": 16,
          "num_hidden_layers": 28,
          "num_key_value_heads": 8,
          "rms_norm_eps": 1e-06,
          "rope_scaling": {
            "mrope_section": [16, 24, 24],
            "rope_type": "default",
            "type": "default"
          },
          "rope_theta": 1000000.0,
          "vocab_size": 151936,
          "image_token_id": 151655,
          "video_token_id": 151656,
          "vision_start_token_id": 151652,
          "vision_end_token_id": 151653,
          "vision_token_id": 151654,
          "max_position_embeddings": 128000,
          "text_config": {
            "head_dim": 128,
            "tie_word_embeddings": true,
            "pad_token_id": 151643,
            "eos_token_id": 151645,
            "layer_types": ["full_attention"]
          },
          "vision_config": {
            "depth": 32,
            "fullatt_block_indexes": [7, 15, 23, 31],
            "hidden_act": "silu",
            "hidden_size": 1280,
            "in_chans": 3,
            "in_channels": 3,
            "intermediate_size": 3420,
            "num_heads": 16,
            "out_hidden_size": 1024,
            "patch_size": 14,
            "spatial_merge_size": 2,
            "temporal_patch_size": 2,
            "window_size": 112
          }
        }"#
    .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse_fixture() -> NaviDcConfig {
        serde_json::from_str(&test_config_json()).expect("fixture config must parse")
    }

    #[test]
    fn parses_navidc_checkpoint_layout() {
        let cfg = parse_fixture();
        assert_eq!(cfg.head_dim().unwrap(), 128);
        assert_eq!(cfg.vocab_size, 151936);
        assert_eq!(cfg.effective_eos_token_id(), 151645);
        assert!(cfg.tie_word_embeddings());
        assert_eq!(cfg.effective_pad_token_id(), Some(151643));
        assert_eq!(cfg.mrope_section().unwrap(), &[16, 24, 24]);
        assert_eq!(cfg.vision_config.in_channels(), 3);
        assert_eq!(cfg.vision_config.head_dim().unwrap(), 80);
        assert_eq!(cfg.vision_config.fullatt_block_indexes, vec![7, 15, 23, 31]);
        cfg.validate().unwrap();
    }

    #[test]
    fn head_dim_falls_back_to_hidden_over_heads() {
        let mut cfg = parse_fixture();
        cfg.head_dim = None;
        cfg.text_config.head_dim = None;
        assert_eq!(cfg.head_dim().unwrap(), 1024 / 16);
    }

    #[test]
    fn head_dim_fallback_requires_divisibility() {
        let mut cfg = parse_fixture();
        cfg.head_dim = None;
        cfg.text_config.head_dim = None;
        cfg.num_attention_heads = 15;
        assert!(cfg.head_dim().is_err());
    }

    #[test]
    fn nested_tie_word_embeddings_alone_is_honoured() {
        let mut cfg = parse_fixture();
        assert!(!cfg.tie_word_embeddings); // root absent in the fixture
        assert!(cfg.tie_word_embeddings()); // nested text_config wins
        cfg.text_config.tie_word_embeddings = false;
        assert!(!cfg.tie_word_embeddings());
    }

    #[test]
    fn mrope_section_must_match_head_dim() {
        let mut cfg = parse_fixture();
        cfg.rope_scaling.mrope_section = vec![16, 24, 23];
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("mrope_section"), "{err}");
    }

    #[test]
    fn sliding_attention_layers_are_rejected() {
        let mut cfg = parse_fixture();
        cfg.text_config.layer_types = vec!["sliding_attention".to_string()];
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("sliding_attention"), "{err}");
    }

    #[test]
    fn window_size_must_align_with_window_tokens() {
        let mut cfg = parse_fixture();
        cfg.vision_config.window_size = 100;
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("window_size"), "{err}");
    }

    #[test]
    fn fullatt_index_beyond_depth_is_rejected() {
        let mut cfg = parse_fixture();
        cfg.vision_config.fullatt_block_indexes = vec![32];
        let err = cfg.validate().unwrap_err();
        assert!(err.to_string().contains("fullatt_block_indexes"), "{err}");
    }
}
