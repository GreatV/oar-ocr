use crate::error::Error;
use serde::Deserialize;
use std::path::Path;

/// Checkpoint generation supported by the HunyuanOCR backend.
///
/// The upstream config does not expose a dedicated version field. V1.5 uses
/// the newer nested `text_config` schema, while the archived V1.0 checkpoint
/// keeps all text fields at the top level.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HunyuanOcrVersion {
    V1,
    V1_5,
}

impl std::fmt::Display for HunyuanOcrVersion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::V1 => f.write_str("1.0"),
            Self::V1_5 => f.write_str("1.5"),
        }
    }
}

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

// The published vision checkpoint keeps a patch bias and one unused positional
// embedding slot; modern Transformers configs omit these fixed properties.
fn default_patch_bias() -> bool {
    true
}
fn default_extra_tokens() -> usize {
    1
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
    #[serde(default = "default_patch_bias")]
    pub add_patchemb_bias: bool,
    #[serde(default = "default_extra_tokens")]
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

    /// Present in HunyuanOCR 1.5's Transformers config. Kept for version
    /// detection; loading normalizes nested text fields into the legacy view.
    #[serde(default)]
    pub text_config: Option<serde_json::Value>,
}

impl HunyuanOcrConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, Error> {
        let mut value: serde_json::Value =
            crate::utils::load_json_config(path, "HunyuanOCR", "config.json")?;
        // Keep the public flat configuration compatible with both checkpoint
        // schemas. Explicit legacy root fields take precedence over nested ones.
        let root = value
            .as_object_mut()
            .ok_or_else(|| Error::config("HunyuanOCR config.json must be an object"))?;
        if let Some(text) = root
            .get("text_config")
            .and_then(serde_json::Value::as_object)
            .cloned()
        {
            for (key, field) in text {
                root.entry(key).or_insert(field);
            }
        }
        for (old, new) in [
            ("pad_id", "pad_token_id"),
            ("image_start_token_id", "im_start_id"),
            ("image_end_token_id", "im_end_id"),
            ("image_newline_token_id", "im_newline_id"),
            ("rope_scaling", "rope_parameters"),
        ] {
            if !root.contains_key(old)
                && let Some(field) = root.get(new).cloned()
            {
                root.insert(old.to_string(), field);
            }
        }
        if !root.contains_key("rope_theta")
            && let Some(theta) = root
                .get("rope_scaling")
                .and_then(|rope| rope.get("rope_theta"))
                .cloned()
        {
            root.insert("rope_theta".to_string(), theta);
        }
        serde_json::from_value(value).map_err(|error| {
            Error::config(format!("failed to parse HunyuanOCR config.json: {error}"))
        })
    }

    pub fn version(&self) -> HunyuanOcrVersion {
        if self
            .text_config
            .as_ref()
            .and_then(serde_json::Value::as_object)
            .is_some()
        {
            HunyuanOcrVersion::V1_5
        } else {
            HunyuanOcrVersion::V1
        }
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
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, Error> {
        crate::utils::load_json_config(path, "HunyuanOCR", "preprocessor_config.json")
    }

    pub fn validate(&self) -> Result<(), Error> {
        crate::utils::validate_image_mean_std("HunyuanOCR", &self.image_mean, &self.image_std)?;
        crate::utils::validate_patch_merge_temporal(
            "HunyuanOCR",
            self.patch_size,
            self.merge_size,
            self.temporal_patch_size,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn loads_nested_and_legacy_checkpoint_schemas() {
        // Official HunyuanOCR config at 47644ecc4fc854efa4f505155158831f36773ee4.
        let modern = include_str!("fixtures/nested_config.json");
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("config.json");
        std::fs::write(&path, modern).unwrap();
        let cfg = HunyuanOcrConfig::from_path(&path).unwrap();
        assert_eq!(cfg.version(), HunyuanOcrVersion::V1_5);
        assert_eq!(
            (cfg.hidden_size, cfg.pad_id, cfg.rope_theta),
            (1024, 120002, 10000.0)
        );
        assert_eq!(cfg.rope_scaling.xdrope_section, [16; 4]);
        assert!(cfg.vision_config.add_patchemb_bias);
        assert_eq!(cfg.vision_config.cat_extra_token, 1);
        // Legacy root fields must survive even when nested values differ.
        let mut legacy: serde_json::Value = serde_json::from_str(modern).unwrap();
        let text = legacy["text_config"].as_object().unwrap().clone();
        let root = legacy.as_object_mut().unwrap();
        root.extend(text);
        root.insert("hidden_size".into(), 2048.into());
        root.insert("pad_id".into(), 42.into());
        std::fs::write(&path, serde_json::to_vec(&legacy).unwrap()).unwrap();
        let cfg = HunyuanOcrConfig::from_path(&path).unwrap();
        assert_eq!(cfg.version(), HunyuanOcrVersion::V1_5);
        assert_eq!((cfg.hidden_size, cfg.pad_id), (2048, 42));
        legacy.as_object_mut().unwrap().remove("text_config");
        std::fs::write(&path, serde_json::to_vec(&legacy).unwrap()).unwrap();
        assert_eq!(
            HunyuanOcrConfig::from_path(path).unwrap().version(),
            HunyuanOcrVersion::V1
        );
    }
}
