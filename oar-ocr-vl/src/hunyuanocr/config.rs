use oar_ocr_core::core::OCRError;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct HunyuanOcrRopeScaling {
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub xdrope_section: Vec<usize>,
    #[serde(default)]
    pub factor: Option<f64>,
    #[serde(default)]
    pub alpha: Option<f64>,
    #[serde(default)]
    pub beta_fast: Option<usize>,
    #[serde(default)]
    pub beta_slow: Option<usize>,
    #[serde(default)]
    pub mscale: Option<f64>,
    #[serde(default)]
    pub mscale_all_dim: Option<f64>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HunyuanOcrVisionConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub rms_norm_eps: f64,
    pub hidden_act: String,
    pub add_patchemb_bias: bool,
    pub cat_extra_token: usize,
    pub max_vit_seq_len: usize,
    pub max_image_size: usize,
    pub img_max_token_num: usize,
    pub interpolate_mode: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HunyuanOcrConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,

    pub bos_token_id: u32,
    pub eos_token_id: u32,
    pub eod_token_id: u32,
    pub pad_id: u32,

    pub image_start_token_id: u32,
    pub image_end_token_id: u32,
    pub image_token_id: u32,
    pub image_newline_token_id: u32,

    pub use_qk_norm: bool,
    pub rope_scaling: HunyuanOcrRopeScaling,
    pub vision_config: HunyuanOcrVisionConfig,
}

impl HunyuanOcrConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, OCRError> {
        let contents = std::fs::read_to_string(path)?;
        serde_json::from_str(&contents).map_err(|e| OCRError::ConfigError {
            message: format!("failed to parse HunyuanOCR config.json: {e}"),
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct HunyuanOcrImageProcessorConfig {
    pub min_pixels: u32,
    pub max_pixels: u32,
    pub patch_size: usize,
    #[serde(default)]
    pub resample: Option<u32>,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
}

impl HunyuanOcrImageProcessorConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, OCRError> {
        let contents = std::fs::read_to_string(path)?;
        serde_json::from_str(&contents).map_err(|e| OCRError::ConfigError {
            message: format!("failed to parse HunyuanOCR preprocessor_config.json: {e}"),
        })
    }

    pub fn validate(&self) -> Result<(), OCRError> {
        if self.image_mean.len() != 3 || self.image_std.len() != 3 {
            return Err(OCRError::ConfigError {
                message: format!(
                    "HunyuanOCR image_mean/std must have length 3, got mean={} std={}",
                    self.image_mean.len(),
                    self.image_std.len()
                ),
            });
        }
        if self.patch_size == 0 || self.merge_size == 0 || self.temporal_patch_size == 0 {
            return Err(OCRError::ConfigError {
                message: "HunyuanOCR patch_size/merge_size/temporal_patch_size must be > 0"
                    .to_string(),
            });
        }
        Ok(())
    }
}
