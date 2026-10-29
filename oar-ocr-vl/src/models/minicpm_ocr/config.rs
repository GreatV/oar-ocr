//! Checkpoint configuration for MiniCPM-OCR.

use crate::Error;
use crate::backbones::qwen3_5::text::Qwen35TextConfig;
use crate::runtime::checkpoint::load_json_config;
use serde::Deserialize;
use std::path::Path;

/// Top-level `config.json` of a `minicpm_ocr` checkpoint.
#[derive(Debug, Clone)]
pub struct MiniCpmOcrConfig {
    pub image_token_id: u32,
    pub vision_config: MiniCpmOcrVisionConfig,
    pub text_config: Qwen35TextConfig,
}

#[derive(Debug, Deserialize)]
struct RawConfig {
    model_type: String,
    image_token_id: u32,
    #[serde(default = "default_downsample_mode")]
    downsample_mode: String,
    #[serde(default = "default_merge_kernel_size")]
    merge_kernel_size: [usize; 2],
    #[serde(default = "default_merger_times")]
    merger_times: usize,
    vision_config: MiniCpmOcrVisionConfig,
    text_config: serde_json::Value,
}

/// SigLIP vision tower configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct MiniCpmOcrVisionConfig {
    pub model_type: String,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub hidden_act: String,
    pub layer_norm_eps: f64,
}

/// `preprocessor_config.json` slicing and normalization settings.
#[derive(Debug, Clone, Deserialize)]
pub struct MiniCpmOcrImageProcessorConfig {
    pub max_slice_nums: usize,
    pub scale_resolution: u32,
    pub patch_size: u32,
    pub slice_mode: bool,
    #[serde(default = "default_true")]
    pub use_image_id: bool,
    #[serde(default = "default_downsample_mode")]
    pub downsample_mode: String,
    pub image_mean: [f32; 3],
    pub image_std: [f32; 3],
}

#[derive(Debug, Deserialize)]
pub(super) struct GenerationConfig {
    pub eos_token_id: Vec<u32>,
}

fn default_downsample_mode() -> String {
    "4x".to_string()
}

fn default_merge_kernel_size() -> [usize; 2] {
    [2, 2]
}

fn default_merger_times() -> usize {
    1
}

fn default_true() -> bool {
    true
}

/// MRoPE split that degenerates to the plain 1-D rotary embedding the
/// checkpoint uses: every text position carries the same index on all three
/// axes, so the axis assignment of each frequency never changes the result.
fn one_dimensional_mrope_section(text: &serde_json::Value) -> Option<[usize; 3]> {
    let head_dim = text.get("head_dim")?.as_u64()? as f64;
    let partial = text
        .pointer("/rope_parameters/partial_rotary_factor")
        .and_then(serde_json::Value::as_f64)
        .unwrap_or(1.0);
    let half = (head_dim * partial) as usize / 2;
    let third = half / 3;
    Some([half - 2 * third, third, third])
}

impl MiniCpmOcrConfig {
    pub fn from_path(path: impl AsRef<Path>, eos_token_id: u32) -> Result<Self, Error> {
        let raw: RawConfig = load_json_config(path, "MiniCPM-OCR", "config.json")?;
        if raw.model_type != "minicpm_ocr" {
            return Err(Error::config(format!(
                "MiniCPM-OCR expected model_type 'minicpm_ocr', got '{}'",
                raw.model_type
            )));
        }
        if raw.downsample_mode != "4x" || raw.merge_kernel_size != [2, 2] || raw.merger_times != 1 {
            return Err(Error::config(
                "MiniCPM-OCR supports only downsample_mode '4x' with one 2x2 merge",
            ));
        }
        let mut text = raw.text_config;
        let section = one_dimensional_mrope_section(&text)
            .ok_or_else(|| Error::config("MiniCPM-OCR text_config lacks head_dim"))?;
        let rope = text
            .get_mut("rope_parameters")
            .and_then(serde_json::Value::as_object_mut)
            .ok_or_else(|| Error::config("MiniCPM-OCR text_config lacks rope_parameters"))?;
        if rope.contains_key("mrope_section") {
            return Err(Error::config(
                "MiniCPM-OCR expects 1-D text RoPE without mrope_section",
            ));
        }
        rope.insert("mrope_section".into(), serde_json::json!(section));
        rope.insert("mrope_interleaved".into(), serde_json::json!(true));
        let object = text
            .as_object_mut()
            .ok_or_else(|| Error::config("MiniCPM-OCR text_config must be an object"))?;
        object
            .entry("eos_token_id")
            .or_insert(serde_json::json!(eos_token_id));
        let text_config: Qwen35TextConfig = serde_json::from_value(text)
            .map_err(|e| Error::config(format!("MiniCPM-OCR text_config: {e}")))?;
        text_config.validate_for("MiniCPM-OCR")?;
        if !text_config.tie_word_embeddings {
            return Err(Error::config("MiniCPM-OCR requires tied word embeddings"));
        }
        if raw.image_token_id as usize >= text_config.vocab_size {
            return Err(Error::config(
                "MiniCPM-OCR image_token_id is outside the vocabulary",
            ));
        }
        raw.vision_config.validate()?;
        Ok(Self {
            image_token_id: raw.image_token_id,
            vision_config: raw.vision_config,
            text_config,
        })
    }
}

impl MiniCpmOcrVisionConfig {
    fn validate(&self) -> Result<(), Error> {
        if self.model_type != "minicpm_ocr_vision"
            || self.hidden_act != "gelu_pytorch_tanh"
            || self.num_channels != 3
        {
            return Err(Error::config(
                "MiniCPM-OCR requires a 3-channel minicpm_ocr_vision tower with gelu_pytorch_tanh",
            ));
        }
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || !self.hidden_size.is_multiple_of(self.num_attention_heads)
            || self.patch_size == 0
            || self.image_size < self.patch_size
        {
            return Err(Error::config("MiniCPM-OCR vision dimensions are invalid"));
        }
        Ok(())
    }

    /// Side length of the learned position-embedding grid.
    pub fn num_patches_per_side(&self) -> usize {
        self.image_size / self.patch_size
    }
}

impl MiniCpmOcrImageProcessorConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, Error> {
        let config: Self = load_json_config(path, "MiniCPM-OCR", "preprocessor_config.json")?;
        if config.max_slice_nums == 0
            || config.scale_resolution == 0
            || config.patch_size == 0
            || config.downsample_mode != "4x"
            || config
                .image_std
                .iter()
                .any(|&std| !std.is_finite() || std == 0.0)
        {
            return Err(Error::config(
                "MiniCPM-OCR preprocessor_config.json is invalid or not in '4x' mode",
            ));
        }
        Ok(config)
    }
}
