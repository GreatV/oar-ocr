//! UniRec model configuration.

use serde::{Deserialize, Serialize};
use std::path::Path;

use crate::core::OCRError;

/// UniRec model configuration.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniRecConfig {
    /// Model dimensionality (d_model)
    #[serde(default = "default_d_model")]
    pub d_model: usize,

    /// Vocabulary size
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,

    /// Number of decoder layers
    #[serde(default = "default_decoder_layers")]
    pub decoder_layers: usize,

    /// Number of decoder attention heads
    #[serde(default = "default_decoder_attention_heads")]
    pub decoder_attention_heads: usize,

    /// Decoder FFN dimension
    #[serde(default = "default_decoder_ffn_dim")]
    pub decoder_ffn_dim: usize,

    /// Encoder sequence length (vision features)
    #[serde(default = "default_encoder_seq_len")]
    pub encoder_seq_len: usize,

    /// Input image height
    #[serde(default = "default_input_height")]
    pub input_height: usize,

    /// Input image width
    #[serde(default = "default_input_width")]
    pub input_width: usize,

    /// Maximum sequence length for generation
    #[serde(default = "default_max_length")]
    pub max_length: usize,

    /// BOS token id
    #[serde(default = "default_bos_token_id")]
    pub bos_token_id: u32,

    /// EOS token id
    #[serde(default = "default_eos_token_id")]
    pub eos_token_id: u32,

    /// PAD token id
    #[serde(default = "default_pad_token_id")]
    pub pad_token_id: u32,

    /// Decoder start token id
    #[serde(default = "default_decoder_start_token_id")]
    pub decoder_start_token_id: u32,

    /// Dropout rate
    #[serde(default = "default_dropout")]
    pub dropout: f64,

    /// Attention dropout rate
    #[serde(default = "default_attention_dropout")]
    pub attention_dropout: f64,

    /// Whether to scale embeddings
    #[serde(default = "default_scale_embedding")]
    pub scale_embedding: bool,

    // FocalSVTR encoder config
    /// Vision encoder embed dimension (base)
    #[serde(default = "default_encoder_embed_dim")]
    pub encoder_embed_dim: usize,

    /// Vision encoder layer depths
    #[serde(default = "default_encoder_depths")]
    pub encoder_depths: Vec<usize>,

    /// Vision encoder focal levels
    #[serde(default = "default_focal_levels")]
    pub focal_levels: Vec<usize>,

    /// Vision encoder focal windows
    #[serde(default = "default_focal_windows")]
    pub focal_windows: Vec<usize>,

    /// Vision encoder max kernel heights
    #[serde(default = "default_max_khs")]
    pub max_khs: Vec<usize>,

    /// Vision encoder subsampling kernels
    #[serde(default = "default_sub_k")]
    pub sub_k: Vec<(usize, usize)>,
}

fn default_d_model() -> usize {
    768
}

fn default_vocab_size() -> usize {
    56371
}

fn default_decoder_layers() -> usize {
    6
}

fn default_decoder_attention_heads() -> usize {
    12
}

fn default_decoder_ffn_dim() -> usize {
    3072
}

fn default_encoder_seq_len() -> usize {
    1320
}

fn default_input_height() -> usize {
    1408 // max_height from Python max_side[1]
}

fn default_input_width() -> usize {
    960 // max_width from Python max_side[0]
}

fn default_max_length() -> usize {
    2048
}

fn default_bos_token_id() -> u32 {
    0
}

fn default_eos_token_id() -> u32 {
    2
}

fn default_pad_token_id() -> u32 {
    1
}

fn default_decoder_start_token_id() -> u32 {
    0
}

fn default_dropout() -> f64 {
    0.0
}

fn default_attention_dropout() -> f64 {
    0.0
}

fn default_scale_embedding() -> bool {
    true
}

fn default_encoder_embed_dim() -> usize {
    96
}

fn default_encoder_depths() -> Vec<usize> {
    vec![2, 2, 9, 2]
}

fn default_focal_levels() -> Vec<usize> {
    vec![3, 3, 3, 3]
}

fn default_focal_windows() -> Vec<usize> {
    vec![3, 3, 3, 3]
}

fn default_max_khs() -> Vec<usize> {
    vec![7, 3, 3, 3]
}

fn default_sub_k() -> Vec<(usize, usize)> {
    vec![(2, 2), (2, 2), (2, 2), (0, 0)] // (0,0) means no downsampling for last stage
}

impl Default for UniRecConfig {
    fn default() -> Self {
        Self {
            d_model: default_d_model(),
            vocab_size: default_vocab_size(),
            decoder_layers: default_decoder_layers(),
            decoder_attention_heads: default_decoder_attention_heads(),
            decoder_ffn_dim: default_decoder_ffn_dim(),
            encoder_seq_len: default_encoder_seq_len(),
            input_height: default_input_height(),
            input_width: default_input_width(),
            max_length: default_max_length(),
            bos_token_id: default_bos_token_id(),
            eos_token_id: default_eos_token_id(),
            pad_token_id: default_pad_token_id(),
            decoder_start_token_id: default_decoder_start_token_id(),
            dropout: default_dropout(),
            attention_dropout: default_attention_dropout(),
            scale_embedding: default_scale_embedding(),
            encoder_embed_dim: default_encoder_embed_dim(),
            encoder_depths: default_encoder_depths(),
            focal_levels: default_focal_levels(),
            focal_windows: default_focal_windows(),
            max_khs: default_max_khs(),
            sub_k: default_sub_k(),
        }
    }
}

impl UniRecConfig {
    /// Load configuration from a JSON file.
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, OCRError> {
        let path = path.as_ref();
        let content = std::fs::read_to_string(path).map_err(|e| OCRError::ConfigError {
            message: format!("Failed to read UniRec config from {:?}: {}", path, e),
        })?;
        serde_json::from_str(&content).map_err(|e| OCRError::ConfigError {
            message: format!("Failed to parse UniRec config: {}", e),
        })
    }

    /// Get the number of attention heads per key-value head.
    /// For standard multi-head attention, this is 1.
    pub fn num_key_value_heads(&self) -> usize {
        self.decoder_attention_heads
    }

    /// Get the dimension per attention head.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.decoder_attention_heads
    }
}
