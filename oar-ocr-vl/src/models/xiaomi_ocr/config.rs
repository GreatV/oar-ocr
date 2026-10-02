use crate::backbones::qwen_vl_processing::QwenVlImageProcessorConfig;
use crate::backbones::qwen3_vl::vision::Qwen3VlVisionConfig;
use crate::error::Error;
use candle_nn::Activation;
use serde::Deserialize;
use std::path::Path;

// Re-exported from the shared Qwen3.5 backbone so the bindings keep their
// `pub` visibility (Xiaomi-OCR-0's `text_config` schema is byte-identical to
// OvisOCR2's).
pub use crate::backbones::qwen3_5::text::{Qwen35RopeParameters, Qwen35TextConfig};

/// Root `config.json` of a Xiaomi-OCR-0 checkpoint (`Qwen3_5ForConditionalGeneration`).
#[derive(Debug, Clone, Deserialize)]
pub struct XiaomiOcrConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub text_config: Qwen35TextConfig,
    pub vision_config: XiaomiOcrVisionConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    /// Top-level `<|im_end|>` id (`248046`); the text tower repeats the
    /// Qwen-base `<|endoftext|>` id under `text_config.eos_token_id`.
    #[serde(default)]
    pub eos_token_id: Option<u32>,
    /// Top-level `<|endoftext|>` id (`248044`).
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    #[serde(default)]
    pub bos_token_id: Option<u32>,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub transformers_version: Option<String>,
}

impl XiaomiOcrConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, Error> {
        let cfg: Self = crate::utils::load_json_config(path, "Xiaomi-OCR-0", "config.json")?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings || self.text_config.tie_word_embeddings
    }

    pub fn validate(&self) -> Result<(), Error> {
        if self.model_type != "qwen3_5" {
            return Err(Error::Config {
                message: format!(
                    "Xiaomi-OCR-0 expected model_type 'qwen3_5', got '{}'",
                    self.model_type
                ),
            });
        }
        if let Some(architecture) = self.architectures.first()
            && architecture != "Qwen3_5ForConditionalGeneration"
        {
            return Err(Error::Config {
                message: format!(
                    "Xiaomi-OCR-0 expected architecture 'Qwen3_5ForConditionalGeneration', got '{architecture}'"
                ),
            });
        }
        self.text_config.validate_for("Xiaomi-OCR-0")?;
        self.vision_config.validate()?;
        if !self.tie_word_embeddings() {
            return Err(Error::Config {
                message: "Xiaomi-OCR-0 requires tied token embeddings".to_string(),
            });
        }
        for (name, token_id) in [
            ("image_token_id", Some(self.image_token_id)),
            ("video_token_id", Some(self.video_token_id)),
            ("vision_start_token_id", Some(self.vision_start_token_id)),
            ("vision_end_token_id", Some(self.vision_end_token_id)),
            ("eos_token_id", self.eos_token_id),
            ("pad_token_id", self.pad_token_id),
        ] {
            let Some(token_id) = token_id else {
                continue;
            };
            if token_id as usize >= self.text_config.vocab_size {
                return Err(Error::Config {
                    message: format!(
                        "Xiaomi-OCR-0 {name} {token_id} is outside vocab_size {}",
                        self.text_config.vocab_size
                    ),
                });
            }
        }
        if self.vision_config.out_hidden_size != self.text_config.hidden_size {
            return Err(Error::Config {
                message: format!(
                    "Xiaomi-OCR-0 vision out_hidden_size ({}) must equal text hidden_size ({})",
                    self.vision_config.out_hidden_size, self.text_config.hidden_size
                ),
            });
        }
        Ok(())
    }
}

/// `processor_config.json` of a Xiaomi-OCR-0 checkpoint.
///
/// Xiaomi ships the Qwen3-VL processor layout, where the image-processor
/// settings are nested under `image_processor` (OvisOCR2 flattens them into
/// `preprocessor_config.json` instead). The nested object parses into the
/// shared [`QwenVlImageProcessorConfig`], so the advertised
/// `size.shortest_edge`/`longest_edge` pixel-area bounds are used as-is —
/// unlike OvisOCR2, whose official runtime overrides them.
#[derive(Debug, Clone, Deserialize)]
pub struct XiaomiOcrProcessorConfig {
    pub image_processor: QwenVlImageProcessorConfig,
    #[serde(default)]
    pub processor_class: Option<String>,
}

impl XiaomiOcrProcessorConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, Error> {
        let cfg: Self =
            crate::utils::load_json_config(path, "Xiaomi-OCR-0", "processor_config.json")?;
        cfg.validate()?;
        Ok(cfg)
    }

    /// Advertised pixel-area bounds (`256²..4096²` for the released
    /// checkpoint), straight from the processor metadata.
    pub fn pixel_bounds(&self) -> Result<(u32, u32), Error> {
        self.image_processor.pixel_bounds()
    }

    pub fn validate(&self) -> Result<(), Error> {
        self.image_processor.validate()
    }

    /// The image processor must agree with the vision tower on how pixels
    /// become patches.
    pub fn validate_vision_compatibility(
        &self,
        vision_cfg: &XiaomiOcrVisionConfig,
    ) -> Result<(), Error> {
        let image_processor = &self.image_processor;
        if image_processor.patch_size != vision_cfg.patch_size {
            return Err(Error::Config {
                message: format!(
                    "Xiaomi-OCR-0 patch_size mismatch: processor {} != vision {}",
                    image_processor.patch_size, vision_cfg.patch_size
                ),
            });
        }
        if image_processor.temporal_patch_size != vision_cfg.temporal_patch_size {
            return Err(Error::Config {
                message: format!(
                    "Xiaomi-OCR-0 temporal_patch_size mismatch: processor {} != vision {}",
                    image_processor.temporal_patch_size, vision_cfg.temporal_patch_size
                ),
            });
        }
        if image_processor.merge_size != vision_cfg.spatial_merge_size {
            return Err(Error::Config {
                message: format!(
                    "Xiaomi-OCR-0 merge_size mismatch: processor {} != vision {}",
                    image_processor.merge_size, vision_cfg.spatial_merge_size
                ),
            });
        }
        if vision_cfg.in_channels != 3 {
            return Err(Error::Config {
                message: format!(
                    "Xiaomi-OCR-0 image preprocessing supports three RGB channels, got {}",
                    vision_cfg.in_channels
                ),
            });
        }
        Ok(())
    }
}

/// Vision-tower configuration of a Xiaomi-OCR-0 checkpoint.
///
/// The tower is architecturally a Qwen3-VL tower without DeepStack taps;
/// `transformers` 5.x tags it `qwen3_5_vision` (the released OvisOCR2
/// checkpoint tags the identical tower `qwen3_5`), so only the `model_type`
/// check differs from the OvisOCR2 binding.
#[derive(Debug, Clone, Deserialize)]
pub struct XiaomiOcrVisionConfig {
    pub model_type: String,
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    pub in_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub out_hidden_size: usize,
    pub num_position_embeddings: usize,
    pub hidden_act: Activation,
    #[serde(default)]
    pub initializer_range: f64,
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<usize>,
}

impl XiaomiOcrVisionConfig {
    pub fn head_dim(&self) -> Result<usize, Error> {
        if self.num_heads == 0 || !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(Error::Config {
                message: format!(
                    "Xiaomi-OCR-0 vision hidden_size {} must be divisible by num_heads {}",
                    self.hidden_size, self.num_heads
                ),
            });
        }
        Ok(self.hidden_size / self.num_heads)
    }

    pub fn position_grid_size(&self) -> Result<usize, Error> {
        let side = (self.num_position_embeddings as f64).sqrt() as usize;
        if side == 0 || side * side != self.num_position_embeddings {
            return Err(Error::Config {
                message: format!(
                    "Xiaomi-OCR-0 num_position_embeddings must be a non-zero square, got {}",
                    self.num_position_embeddings
                ),
            });
        }
        Ok(side)
    }

    /// Backbone configuration for the shared Qwen3-VL vision tower.
    ///
    /// Field-by-field on purpose: a new field on either struct must fail to
    /// compile here rather than be silently dropped.
    pub(crate) fn to_qwen3_vl(&self) -> Qwen3VlVisionConfig {
        Qwen3VlVisionConfig {
            model_type: "qwen3_vl".to_string(),
            depth: self.depth,
            hidden_size: self.hidden_size,
            intermediate_size: self.intermediate_size,
            num_heads: self.num_heads,
            in_channels: self.in_channels,
            patch_size: self.patch_size,
            spatial_merge_size: self.spatial_merge_size,
            temporal_patch_size: self.temporal_patch_size,
            out_hidden_size: self.out_hidden_size,
            num_position_embeddings: self.num_position_embeddings,
            hidden_act: self.hidden_act,
            deepstack_visual_indexes: self.deepstack_visual_indexes.clone(),
        }
    }

    pub fn validate(&self) -> Result<(), Error> {
        if self.model_type != "qwen3_5_vision" {
            return Err(Error::Config {
                message: format!(
                    "Xiaomi-OCR-0 expected vision model_type 'qwen3_5_vision', got '{}'",
                    self.model_type
                ),
            });
        }
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.in_channels == 0
            || self.patch_size == 0
            || self.spatial_merge_size == 0
            || self.temporal_patch_size == 0
            || self.out_hidden_size == 0
        {
            return Err(Error::Config {
                message: "Xiaomi-OCR-0 vision dimensions must be non-zero".to_string(),
            });
        }
        let head_dim = self.head_dim()?;
        if !head_dim.is_multiple_of(4) {
            return Err(Error::Config {
                message: format!(
                    "Xiaomi-OCR-0 vision head_dim must be divisible by 4 for 2D RoPE, got {head_dim}"
                ),
            });
        }
        self.position_grid_size()?;
        if !self.deepstack_visual_indexes.is_empty() {
            return Err(Error::Config {
                message: "Xiaomi-OCR-0 deepstack vision features are not supported".to_string(),
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
      "architectures": ["Qwen3_5ForConditionalGeneration"],
      "bos_token_id": null,
      "dtype": "bfloat16",
      "eos_token_id": 248046,
      "hidden_size": 1024,
      "image_token_id": 248056,
      "model_type": "qwen3_5",
      "pad_token_id": 248044,
      "rope_theta": 10000000,
      "text_config": {
        "attention_bias": false,
        "attention_dropout": 0.0,
        "attn_output_gate": true,
        "bos_token_id": null,
        "dtype": "bfloat16",
        "eos_token_id": 248044,
        "full_attention_interval": 4,
        "head_dim": 256,
        "hidden_act": "silu",
        "hidden_size": 1024,
        "initializer_range": 0.02,
        "intermediate_size": 3584,
        "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
        "linear_conv_kernel_dim": 4,
        "linear_key_head_dim": 128,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_value_head_dim": 128,
        "max_position_embeddings": 262144,
        "mlp_only_layers": [],
        "model_type": "qwen3_5_text",
        "mtp_num_hidden_layers": 1,
        "mtp_use_dedicated_embeddings": false,
        "num_attention_heads": 8,
        "num_hidden_layers": 4,
        "num_key_value_heads": 2,
        "rms_norm_eps": 1e-6,
        "tie_word_embeddings": true,
        "use_cache": true,
        "vocab_size": 248320,
        "mamba_ssm_dtype": "float32",
        "rope_parameters": {
          "mrope_interleaved": true,
          "mrope_section": [11, 11, 10],
          "rope_type": "default",
          "rope_theta": 10000000,
          "partial_rotary_factor": 0.25
        }
      },
      "tie_word_embeddings": true,
      "transformers_version": "5.8.1",
      "video_token_id": 248057,
      "vision_config": {
        "deepstack_visual_indexes": [],
        "depth": 12,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 768,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 3072,
        "model_type": "qwen3_5_vision",
        "num_heads": 12,
        "num_position_embeddings": 2304,
        "out_hidden_size": 1024,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2
      },
      "vision_end_token_id": 248054,
      "vision_start_token_id": 248053
    }
    "#;

    const PROCESSOR: &str = r#"
    {
      "image_processor": {
        "do_convert_rgb": true,
        "do_normalize": true,
        "do_rescale": true,
        "do_resize": true,
        "image_mean": [0.5, 0.5, 0.5],
        "image_processor_type": "Qwen2VLImageProcessor",
        "image_std": [0.5, 0.5, 0.5],
        "merge_size": 2,
        "patch_size": 16,
        "resample": 3,
        "rescale_factor": 0.00392156862745098,
        "size": {"longest_edge": 16777216, "shortest_edge": 65536},
        "temporal_patch_size": 2
      },
      "processor_class": "Qwen3VLProcessor",
      "video_processor": {"fps": 2}
    }
    "#;

    #[test]
    fn parses_local_checkpoint_shape() {
        let cfg: XiaomiOcrConfig = serde_json::from_str(CONFIG).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.text_config.layer_types.len(), 4);
        assert_eq!(cfg.text_config.eos_token_id, 248_044);
        assert_eq!(cfg.eos_token_id, Some(248_046));
        assert_eq!(cfg.pad_token_id, Some(248_044));
        assert!(cfg.bos_token_id.is_none());
        assert_eq!(cfg.vision_config.position_grid_size().unwrap(), 48);
        assert_eq!(cfg.vision_config.head_dim().unwrap(), 64);
        assert!(cfg.tie_word_embeddings());
        assert_eq!(cfg.text_config.hidden_act, Activation::Silu);
        assert_eq!(cfg.vision_config.hidden_act, Activation::GeluPytorchTanh);
    }

    #[test]
    fn parses_nested_processor_config() {
        let cfg: XiaomiOcrProcessorConfig = serde_json::from_str(PROCESSOR).unwrap();
        cfg.validate().unwrap();
        assert_eq!(cfg.image_processor.patch_size, 16);
        assert_eq!(cfg.image_processor.merge_size, 2);
        assert_eq!(cfg.image_processor.resample, Some(3));
        // The advertised pixel-area bounds are used as-is.
        assert_eq!(cfg.pixel_bounds().unwrap(), (65536, 16_777_216));
    }

    #[test]
    fn rejects_wrong_model_type() {
        let mut cfg: XiaomiOcrConfig = serde_json::from_str(CONFIG).unwrap();
        cfg.model_type = "qwen3_vl".to_string();
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("model_type")
        );
    }

    #[test]
    fn rejects_wrong_architecture() {
        let mut cfg: XiaomiOcrConfig = serde_json::from_str(CONFIG).unwrap();
        cfg.architectures = vec!["Qwen3VLForConditionalGeneration".to_string()];
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("Qwen3_5ForConditionalGeneration")
        );
    }

    #[test]
    fn rejects_untied_checkpoint() {
        let mut cfg: XiaomiOcrConfig = serde_json::from_str(CONFIG).unwrap();
        cfg.tie_word_embeddings = false;
        cfg.text_config.tie_word_embeddings = false;
        assert!(cfg.validate().unwrap_err().to_string().contains("tied"));
    }

    #[test]
    fn rejects_out_of_vocab_top_level_tokens() {
        let mut cfg: XiaomiOcrConfig = serde_json::from_str(CONFIG).unwrap();
        cfg.eos_token_id = Some(248_320);
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("eos_token_id")
        );
    }

    #[test]
    fn rejects_processor_vision_mismatch() {
        let processor: XiaomiOcrProcessorConfig = serde_json::from_str(PROCESSOR).unwrap();
        let mut vision: XiaomiOcrVisionConfig = serde_json::from_str::<XiaomiOcrConfig>(CONFIG)
            .unwrap()
            .vision_config;
        vision.patch_size = 14;
        let error = processor
            .validate_vision_compatibility(&vision)
            .unwrap_err();
        assert!(error.to_string().contains("patch_size mismatch"));
    }

    #[test]
    fn vision_config_maps_onto_the_shared_qwen3_vl_tower() {
        let cfg: XiaomiOcrConfig = serde_json::from_str(CONFIG).unwrap();
        let backbone = cfg.vision_config.to_qwen3_vl();
        assert_eq!(backbone.model_type, "qwen3_vl");
        assert_eq!(backbone.depth, cfg.vision_config.depth);
        assert_eq!(backbone.hidden_size, cfg.vision_config.hidden_size);
        assert!(backbone.deepstack_visual_indexes.is_empty());
        backbone.validate().unwrap();
    }

    /// Silence the unused re-export warning when only aliases are consumed.
    #[test]
    fn text_config_alias_matches_backbone_type() {
        let cfg: XiaomiOcrConfig = serde_json::from_str(CONFIG).unwrap();
        let _: &Qwen35TextConfig = &cfg.text_config;
        let _: &Qwen35RopeParameters = &cfg.text_config.rope_parameters;
    }
}
