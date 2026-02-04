use oar_ocr_core::core::OCRError;
use serde::Deserialize;
use std::path::Path;

fn default_vision_hidden_act() -> String {
    "quick_gelu".to_string()
}

fn default_text_hidden_act() -> String {
    "silu".to_string()
}

#[derive(Debug, Clone, Default, Deserialize)]
pub struct MinerURopeScaling {
    #[serde(default)]
    pub r#type: Option<String>,
    #[serde(default)]
    pub mrope_section: Vec<usize>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MinerUVisionConfig {
    pub depth: usize,
    pub embed_dim: usize,
    pub hidden_size: usize,
    #[serde(default = "default_vision_hidden_act")]
    pub hidden_act: String,
    pub mlp_ratio: f64,
    pub num_heads: usize,
    #[serde(alias = "in_chans", alias = "in_channels")]
    pub in_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    #[serde(default)]
    pub spatial_patch_size: Option<usize>,
    pub temporal_patch_size: usize,
}

fn default_true() -> bool {
    true
}

impl MinerUVisionConfig {
    pub fn mlp_hidden_dim(&self) -> usize {
        (self.embed_dim as f64 * self.mlp_ratio).round() as usize
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct MinerUConfig {
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
    #[serde(default)]
    pub sliding_window: Option<usize>,
    #[serde(default)]
    pub max_window_layers: usize,
    #[serde(default)]
    pub use_sliding_window: bool,
    #[serde(default = "default_text_hidden_act")]
    pub hidden_act: String,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub bos_token_id: u32,
    pub eos_token_id: u32,
    #[serde(default)]
    pub pad_token_id: Option<u32>,
    pub vision_start_token_id: u32,
    pub vision_end_token_id: u32,
    pub vision_token_id: u32,
    pub image_token_id: u32,
    pub video_token_id: u32,
    #[serde(default)]
    pub rope_scaling: MinerURopeScaling,
    pub vision_config: MinerUVisionConfig,
}

impl MinerUConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, OCRError> {
        let contents = std::fs::read_to_string(path)?;
        serde_json::from_str(&contents).map_err(|e| OCRError::ConfigError {
            message: format!("failed to parse MinerU2.5 config.json: {e}"),
        })
    }

    pub fn head_dim(&self) -> Result<usize, OCRError> {
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(OCRError::ConfigError {
                message: format!(
                    "MinerU2.5: hidden_size {} not divisible by num_attention_heads {}",
                    self.hidden_size, self.num_attention_heads
                ),
            });
        }
        Ok(self.hidden_size / self.num_attention_heads)
    }
}

#[derive(Debug, Clone, Deserialize)]
pub struct MinerUImageProcessorConfig {
    #[serde(default)]
    pub min_pixels: Option<u32>,
    #[serde(default)]
    pub max_pixels: Option<u32>,
    #[serde(default)]
    pub size: Option<MinerUImageSize>,
    #[serde(default = "default_true")]
    pub do_resize: bool,
    #[serde(default = "default_true")]
    pub do_rescale: bool,
    #[serde(default = "default_true")]
    pub do_normalize: bool,
    #[serde(default = "default_true")]
    pub do_convert_rgb: bool,
    pub patch_size: usize,
    pub temporal_patch_size: usize,
    pub merge_size: usize,
    pub image_mean: Vec<f32>,
    pub image_std: Vec<f32>,
    #[serde(default)]
    pub resample: Option<u32>,
    #[serde(default = "default_rescale_factor")]
    pub rescale_factor: f32,
}

#[derive(Debug, Clone, Deserialize)]
pub struct MinerUImageSize {
    pub shortest_edge: u32,
    pub longest_edge: u32,
}

fn default_rescale_factor() -> f32 {
    1.0 / 255.0
}

impl MinerUImageProcessorConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, OCRError> {
        let contents = std::fs::read_to_string(path)?;
        serde_json::from_str(&contents).map_err(|e| OCRError::ConfigError {
            message: format!("failed to parse MinerU2.5 preprocessor_config.json: {e}"),
        })
    }

    pub fn pixel_bounds(&self) -> Result<(u32, u32), OCRError> {
        if let Some(size) = &self.size {
            if size.shortest_edge == 0 || size.longest_edge == 0 {
                return Err(OCRError::ConfigError {
                    message: "MinerU2.5 size.shortest_edge/longest_edge must be > 0".to_string(),
                });
            }
            return Ok((size.shortest_edge, size.longest_edge));
        }

        match (self.min_pixels, self.max_pixels) {
            (Some(min_pixels), Some(max_pixels)) => Ok((min_pixels, max_pixels)),
            _ => Err(OCRError::ConfigError {
                message: "MinerU2.5 preprocessor_config missing size or min/max pixels".to_string(),
            }),
        }
    }

    pub fn validate(&self) -> Result<(), OCRError> {
        if self.do_normalize {
            if self.image_mean.len() != 3 || self.image_std.len() != 3 {
                return Err(OCRError::ConfigError {
                    message: format!(
                        "MinerU2.5 image_mean/std must have length 3, got mean={} std={}",
                        self.image_mean.len(),
                        self.image_std.len()
                    ),
                });
            }
            if self.image_std.contains(&0.0) {
                return Err(OCRError::ConfigError {
                    message: "MinerU2.5 image_std values must be non-zero".to_string(),
                });
            }
        }
        if self.patch_size == 0 || self.merge_size == 0 || self.temporal_patch_size == 0 {
            return Err(OCRError::ConfigError {
                message: "MinerU2.5 patch_size/merge_size/temporal_patch_size must be > 0"
                    .to_string(),
            });
        }
        if self.do_resize {
            let (min_pixels, max_pixels) = self.pixel_bounds()?;
            if min_pixels == 0 || max_pixels == 0 {
                return Err(OCRError::ConfigError {
                    message: "MinerU2.5 min/max pixels must be > 0".to_string(),
                });
            }
            if min_pixels > max_pixels {
                return Err(OCRError::ConfigError {
                    message: format!(
                        "MinerU2.5 min_pixels ({min_pixels}) must be <= max_pixels ({max_pixels})"
                    ),
                });
            }
        }
        if self.do_rescale && self.rescale_factor <= 0.0 {
            return Err(OCRError::ConfigError {
                message: "MinerU2.5 rescale_factor must be > 0".to_string(),
            });
        }
        Ok(())
    }
}
