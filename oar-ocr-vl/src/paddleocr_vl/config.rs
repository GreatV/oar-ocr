use oar_ocr_core::core::OCRError;
use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize)]
pub struct PaddleOcrVlImageProcessorConfig {
    pub do_resize: bool,
    pub do_rescale: bool,
    pub do_normalize: bool,
    pub do_convert_rgb: bool,
    pub rescale_factor: f32,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
    pub min_pixels: u32,
    pub max_pixels: u32,
    #[serde(default)]
    pub resample: Option<u32>,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
}

impl PaddleOcrVlImageProcessorConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, OCRError> {
        let contents = std::fs::read_to_string(path)?;
        serde_json::from_str(&contents).map_err(|e| OCRError::ConfigError {
            message: format!("failed to parse PaddleOCR-VL preprocessor_config.json: {e}"),
        })
    }

    pub fn validate(&self) -> Result<(), OCRError> {
        if self.image_mean.len() != 3 || self.image_std.len() != 3 {
            return Err(OCRError::ConfigError {
                message: format!(
                    "PaddleOCR-VL image_mean/std must have length 3, got mean={} std={}",
                    self.image_mean.len(),
                    self.image_std.len()
                ),
            });
        }
        if self.patch_size == 0 || self.merge_size == 0 || self.temporal_patch_size == 0 {
            return Err(OCRError::ConfigError {
                message: "PaddleOCR-VL patch_size/merge_size/temporal_patch_size must be > 0"
                    .to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct PaddleOcrVlRopeScaling {
    pub mrope_section: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PaddleOcrVlVisionConfig {
    pub image_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub tokens_per_second: usize,
    pub layer_norm_eps: f64,
    pub hidden_act: String,
}

#[derive(Debug, Clone, Deserialize)]
pub struct PaddleOcrVlConfig {
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub image_token_id: u32,
    pub video_token_id: u32,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub hidden_act: String,
    pub use_bias: bool,
    pub rope_scaling: PaddleOcrVlRopeScaling,
    pub vision_config: PaddleOcrVlVisionConfig,
}

impl PaddleOcrVlConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, OCRError> {
        let contents = std::fs::read_to_string(path)?;
        serde_json::from_str(&contents).map_err(|e| OCRError::ConfigError {
            message: format!("failed to parse PaddleOCR-VL config.json: {e}"),
        })
    }
}
