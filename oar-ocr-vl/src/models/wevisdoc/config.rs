use crate::error::Error;
use serde::Deserialize;
use std::path::Path;

// Re-exported from the defining submodules so the bindings keep their `pub`
// visibility (a `pub(crate) use` hop would downgrade them and make a public
// re-export impossible).
pub use crate::backbones::qwen3_vl::text::{Qwen3VlRopeScaling, Qwen3VlTextConfig};
pub use crate::backbones::qwen3_vl::vision::Qwen3VlVisionConfig;

/// Root `config.json` of a WeVisDoc (Qwen3-VL) checkpoint.
#[derive(Debug, Clone, Deserialize)]
pub struct WeVisDocConfig {
    pub architectures: Vec<String>,
    pub model_type: String,
    pub text_config: Qwen3VlTextConfig,
    pub vision_config: Qwen3VlVisionConfig,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub transformers_version: Option<String>,
}

impl WeVisDocConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, Error> {
        let cfg: Self =
            crate::runtime::checkpoint::load_json_config(path, "WeVisDoc", "config.json")?;
        cfg.validate()?;
        Ok(cfg)
    }

    pub fn tie_word_embeddings(&self) -> bool {
        self.tie_word_embeddings || self.text_config.tie_word_embeddings
    }

    pub fn validate(&self) -> Result<(), Error> {
        if self.model_type != "qwen3_vl" {
            return Err(Error::Config {
                message: format!(
                    "WeVisDoc expected model_type 'qwen3_vl', got '{}'",
                    self.model_type
                ),
            });
        }
        if let Some(architecture) = self.architectures.first()
            && architecture != "Qwen3VLForConditionalGeneration"
        {
            return Err(Error::Config {
                message: format!(
                    "WeVisDoc expected architecture 'Qwen3VLForConditionalGeneration', got '{architecture}'"
                ),
            });
        }
        self.text_config.validate()?;
        self.vision_config.validate()?;
        if !self.tie_word_embeddings() {
            return Err(Error::Config {
                message: "WeVisDoc requires tied token embeddings".to_string(),
            });
        }
        for (name, token_id) in [
            ("image_token_id", self.image_token_id),
            ("video_token_id", self.video_token_id),
            ("vision_start_token_id", self.vision_start_token_id),
            ("vision_end_token_id", self.vision_end_token_id),
        ] {
            if token_id as usize >= self.text_config.vocab_size {
                return Err(Error::Config {
                    message: format!(
                        "WeVisDoc {name} {token_id} is outside vocab_size {}",
                        self.text_config.vocab_size
                    ),
                });
            }
        }
        if self.vision_config.out_hidden_size != self.text_config.hidden_size {
            return Err(Error::Config {
                message: format!(
                    "WeVisDoc vision out_hidden_size ({}) must equal text hidden_size ({})",
                    self.vision_config.out_hidden_size, self.text_config.hidden_size
                ),
            });
        }
        Ok(())
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub(crate) const CONFIG: &str = r#"
    {
      "architectures": ["Qwen3VLForConditionalGeneration"],
      "image_token_id": 151655,
      "model_type": "qwen3_vl",
      "text_config": {
        "attention_bias": false,
        "attention_dropout": 0.0,
        "bos_token_id": 151643,
        "dtype": "bfloat16",
        "eos_token_id": 151645,
        "head_dim": 128,
        "hidden_act": "silu",
        "hidden_size": 2048,
        "initializer_range": 0.02,
        "intermediate_size": 6144,
        "max_position_embeddings": 262144,
        "model_type": "qwen3_vl_text",
        "num_attention_heads": 16,
        "num_hidden_layers": 28,
        "num_key_value_heads": 8,
        "rms_norm_eps": 1e-06,
        "rope_scaling": {
          "mrope_interleaved": true,
          "mrope_section": [24, 20, 20],
          "rope_type": "default"
        },
        "rope_theta": 5000000,
        "tie_word_embeddings": true,
        "use_cache": true,
        "vocab_size": 151936
      },
      "tie_word_embeddings": true,
      "transformers_version": "4.57.0.dev0",
      "video_token_id": 151656,
      "vision_config": {
        "deepstack_visual_indexes": [5, 11, 17],
        "depth": 24,
        "hidden_act": "gelu_pytorch_tanh",
        "hidden_size": 1024,
        "in_channels": 3,
        "initializer_range": 0.02,
        "intermediate_size": 4096,
        "model_type": "qwen3_vl",
        "num_heads": 16,
        "num_position_embeddings": 2304,
        "out_hidden_size": 2048,
        "patch_size": 16,
        "spatial_merge_size": 2,
        "temporal_patch_size": 2
      },
      "vision_end_token_id": 151653,
      "vision_start_token_id": 151652
    }
    "#;

    #[test]
    fn parses_official_checkpoint_shape() {
        let cfg = official_config();
        cfg.validate().unwrap();
        assert_eq!(cfg.text_config.num_hidden_layers, 28);
        assert_eq!(cfg.text_config.num_attention_heads, 16);
        assert_eq!(cfg.text_config.num_key_value_heads, 8);
        assert_eq!(cfg.text_config.head_dim, 128);
        assert_eq!(cfg.text_config.rope_theta(), 5_000_000.0);
        assert_eq!(cfg.text_config.mrope_section(), &[24, 20, 20]);
        assert!(cfg.text_config.mrope_interleaved());
        assert!(cfg.tie_word_embeddings());
        assert_eq!(cfg.vision_config.deepstack_visual_indexes, [5, 11, 17]);
        assert_eq!(cfg.vision_config.position_grid_size().unwrap(), 48);
        assert_eq!(cfg.vision_config.head_dim().unwrap(), 64);
        assert_eq!(
            cfg.vision_config.hidden_act,
            candle_nn::Activation::GeluPytorchTanh
        );
    }

    #[test]
    fn rejects_wrong_model_type() {
        let mut cfg = official_config();
        cfg.model_type = "qwen2_5_vl".to_string();
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("model_type")
        );
    }

    #[test]
    fn rejects_untied_checkpoint() {
        let mut cfg = official_config();
        cfg.tie_word_embeddings = false;
        cfg.text_config.tie_word_embeddings = false;
        assert!(cfg.validate().unwrap_err().to_string().contains("tied"));
    }

    #[test]
    fn rejects_hidden_size_mismatch_between_tower_and_decoder() {
        let mut cfg = official_config();
        cfg.vision_config.out_hidden_size = 1024;
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("out_hidden_size")
        );
    }

    #[test]
    fn rejects_unknown_architecture() {
        let mut cfg = official_config();
        cfg.architectures = vec!["GPT2LMHeadModel".to_string()];
        assert!(
            cfg.validate()
                .unwrap_err()
                .to_string()
                .contains("Qwen3VLForConditionalGeneration")
        );
    }

    pub(crate) fn official_config() -> WeVisDocConfig {
        serde_json::from_str(CONFIG).unwrap()
    }
}
