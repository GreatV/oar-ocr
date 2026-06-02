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
        crate::utils::load_json_config(path, "HunyuanOCR", "config.json")
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
        crate::utils::load_json_config(path, "HunyuanOCR", "preprocessor_config.json")
    }

    pub fn validate(&self) -> Result<(), OCRError> {
        crate::utils::validate_image_mean_std("HunyuanOCR", &self.image_mean, &self.image_std)?;
        crate::utils::validate_patch_merge_temporal(
            "HunyuanOCR",
            self.patch_size,
            self.merge_size,
            self.temporal_patch_size,
        )
    }
}
