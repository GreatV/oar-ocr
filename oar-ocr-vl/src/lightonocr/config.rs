use candle_nn::Activation;
use oar_ocr_core::core::OCRError;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct LightOnOcrTextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub attention_bias: bool,
    #[serde(default)]
    pub attention_dropout: f32,
    pub hidden_act: Activation,
    pub max_position_embeddings: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub max_window_layers: usize,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub use_qk_norm: bool,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LightOnOcrVisionConfig {
    pub hidden_size: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub rope_theta: f64,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub hidden_act: Activation,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LightOnOcrConfig {
    pub text_config: LightOnOcrTextConfig,
    pub vision_config: LightOnOcrVisionConfig,
    pub image_token_id: u32,
    pub eos_token_id: u32,
    pub pad_token_id: u32,
    pub spatial_merge_size: usize,
    pub projector_hidden_act: Activation,
    pub multimodal_projector_bias: bool,
    #[serde(default)]
    pub vision_feature_layer: i32,
}

impl LightOnOcrConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, OCRError> {
        let contents = std::fs::read_to_string(path)?;
        serde_json::from_str(&contents).map_err(|e| OCRError::ConfigError {
            message: format!("failed to parse LightOnOCR config.json: {e}"),
        })
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct LightOnOcrImageProcessorSize {
    pub longest_edge: u32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct LightOnOcrImageProcessorConfig {
    pub do_resize: bool,
    pub do_rescale: bool,
    pub do_normalize: bool,
    pub do_convert_rgb: bool,
    pub rescale_factor: f32,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
    pub patch_size: usize,
    #[serde(default)]
    pub resample: Option<u32>,
    pub size: LightOnOcrImageProcessorSize,
}

impl LightOnOcrImageProcessorConfig {
    pub fn validate(&self) -> Result<(), OCRError> {
        if self.image_mean.len() != 3 || self.image_std.len() != 3 {
            return Err(OCRError::ConfigError {
                message: format!(
                    "LightOnOCR image_mean/std must have length 3, got mean={} std={}",
                    self.image_mean.len(),
                    self.image_std.len()
                ),
            });
        }
        if self.patch_size == 0 {
            return Err(OCRError::ConfigError {
                message: "LightOnOCR patch_size must be > 0".to_string(),
            });
        }
        if self.size.longest_edge == 0 {
            return Err(OCRError::ConfigError {
                message: "LightOnOCR longest_edge must be > 0".to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct LightOnOcrProcessorConfig {
    pub image_processor: LightOnOcrImageProcessorConfig,
    #[serde(default)]
    pub spatial_merge_size: Option<usize>,
    #[serde(default)]
    pub patch_size: Option<usize>,
}

impl LightOnOcrProcessorConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, OCRError> {
        let contents = std::fs::read_to_string(path)?;
        serde_json::from_str(&contents).map_err(|e| OCRError::ConfigError {
            message: format!("failed to parse LightOnOCR processor_config.json: {e}"),
        })
    }
}
