//! Qwen3 text decoder shared by `qwen3_vl` checkpoints.
//!
//! Ported from `modeling_qwen3_vl.py` (transformers 4.57): per-head q/k
//! RMSNorm, untied GQA projections without bias, SwiGLU MLP, and **interleaved
//! MRoPE** — rotary frequencies cycle temporal/height/width in steps of three
//! (`THTHW...TT`) instead of the contiguous sections used by Qwen2.5-VL.
//!
//! DeepStack injection mirrors `Qwen3VLTextModel.forward`: the visual features
//! tapped from the tower are added to the image-token hidden states of the
//! first `deepstack.len()` decoder layers, prefill only.

use crate::error::Error;
use crate::runtime::attention::{
    RotaryEmbedding, flash_attention, scaled_dot_product_attention_gqa,
};
use crate::runtime::cache::TrimmableKvCache;
use crate::runtime::errors::candle_to_ocr_inference;
use crate::runtime::tensor::rotate_half;
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{
    Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_no_bias, rms_norm,
};
use std::cell::RefCell;

const MODEL_NAME: &str = "Qwen3-VL";

fn default_rms_norm_eps() -> f64 {
    1e-6
}

fn default_rope_theta() -> f64 {
    5_000_000.0
}

fn default_rope_type() -> String {
    "default".to_string()
}

/// Nested `rope_scaling` block of a Qwen3-VL text config.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen3VlRopeScaling {
    #[serde(default = "default_rope_type")]
    pub rope_type: String,
    #[serde(default)]
    pub mrope_interleaved: bool,
    #[serde(default)]
    pub mrope_section: Vec<usize>,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
}

impl Default for Qwen3VlRopeScaling {
    fn default() -> Self {
        Self {
            rope_type: default_rope_type(),
            mrope_interleaved: false,
            mrope_section: Vec::new(),
            rope_theta: default_rope_theta(),
        }
    }
}

/// Text-decoder configuration shared by Qwen3-VL checkpoints.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen3VlTextConfig {
    pub model_type: String,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub max_position_embeddings: usize,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default)]
    pub rope_scaling: Qwen3VlRopeScaling,
    pub eos_token_id: u32,
    #[serde(default)]
    pub attention_bias: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
}

impl Qwen3VlTextConfig {
    pub fn mrope_section(&self) -> &[usize] {
        &self.rope_scaling.mrope_section
    }

    pub fn mrope_interleaved(&self) -> bool {
        self.rope_scaling.mrope_interleaved
    }

    pub fn rope_theta(&self) -> f64 {
        self.rope_scaling.rope_theta
    }

    pub fn validate(&self) -> Result<(), Error> {
        if self.model_type != "qwen3_vl_text" {
            return Err(Error::Config {
                message: format!(
                    "{MODEL_NAME} expected text model_type 'qwen3_vl_text', got '{}'",
                    self.model_type
                ),
            });
        }
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.vocab_size == 0
            || self.num_hidden_layers == 0
            || self.num_attention_heads == 0
            || self.num_key_value_heads == 0
            || self.head_dim == 0
            || self.max_position_embeddings == 0
        {
            return Err(Error::Config {
                message: format!("{MODEL_NAME} text dimensions must be non-zero"),
            });
        }
        if self.attention_bias {
            return Err(Error::Config {
                message: format!("{MODEL_NAME} attention_bias=true is not supported"),
            });
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(Error::Config {
                message: format!(
                    "{MODEL_NAME} num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                    self.num_attention_heads, self.num_key_value_heads
                ),
            });
        }
        if self.rope_scaling.rope_type != "default" {
            return Err(Error::Config {
                message: format!(
                    "{MODEL_NAME} unsupported rope_type '{}'",
                    self.rope_scaling.rope_type
                ),
            });
        }
        if !self.mrope_interleaved() {
            return Err(Error::Config {
                message: format!("{MODEL_NAME} requires interleaved MRoPE"),
            });
        }
        let section = self.mrope_section();
        if section.len() != 3 || section.contains(&0) {
            return Err(Error::Config {
                message: format!("{MODEL_NAME} mrope_section must contain three non-zero entries"),
            });
        }
        let half = self.head_dim / 2;
        if !self.head_dim.is_multiple_of(2) || section.iter().sum::<usize>() != half {
            return Err(Error::Config {
                message: format!(
                    "{MODEL_NAME} mrope_section {section:?} must sum to head_dim/2 ({half})"
                ),
            });
        }
        if !self.rope_theta().is_finite() || self.rope_theta() <= 0.0 {
            return Err(Error::Config {
                message: format!(
                    "{MODEL_NAME} rope_theta must be finite and positive, got {}",
                    self.rope_theta()
                ),
            });
        }
        if self.eos_token_id as usize >= self.vocab_size {
            return Err(Error::Config {
                message: format!(
                    "{MODEL_NAME} eos_token_id {} is outside vocab_size {}",
                    self.eos_token_id, self.vocab_size
                ),
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
struct Qwen3Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
    kv_cache: RefCell<TrimmableKvCache>,
}

impl Qwen3Attention {
    fn load(cfg: &Qwen3VlTextConfig, vb: VarBuilder) -> Result<Self, Error> {
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            return Err(Error::Config {
                message: format!(
                    "{MODEL_NAME} num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                    cfg.num_attention_heads, cfg.num_key_value_heads
                ),
            });
        }
        let q_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.head_dim,
            vb.pp("q_proj"),
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load q_proj", e))?;
        let k_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            vb.pp("k_proj"),
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load k_proj", e))?;
        let v_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            vb.pp("v_proj"),
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load v_proj", e))?;
        let o_proj = linear_no_bias(
            cfg.num_attention_heads * cfg.head_dim,
            cfg.hidden_size,
            vb.pp("o_proj"),
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load o_proj", e))?;
        let q_norm = rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load q_norm", e))?;
        let k_norm = rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load k_norm", e))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            scaling: 1.0 / (cfg.head_dim as f64).sqrt(),
            kv_cache: RefCell::new(TrimmableKvCache::new(2, cfg.max_position_embeddings)),
        })
    }

    fn apply_rope(&self, tensor: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor, Error> {
        let rotary_dim = cos
            .dim(D::Minus1)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "rotary dimension", e))?;
        if rotary_dim == self.head_dim {
            let cos = cos
                .unsqueeze(1)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "cos layout", e))?;
            let sin = sin
                .unsqueeze(1)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "sin layout", e))?;
            let rotated = rotate_half(tensor)?;
            tensor
                .broadcast_mul(&cos)
                .and_then(|lhs| rotated.broadcast_mul(&sin).and_then(|rhs| &lhs + &rhs))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "apply RoPE", e))
        } else {
            let rotary = tensor
                .narrow(D::Minus1, 0, rotary_dim)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "rotary slice", e))?;
            let pass = tensor
                .narrow(D::Minus1, rotary_dim, self.head_dim - rotary_dim)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "pass-through slice", e))?;
            let cos = cos
                .unsqueeze(1)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "cos layout", e))?;
            let sin = sin
                .unsqueeze(1)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "sin layout", e))?;
            let rotated = rotate_half(&rotary)?;
            let embedded = rotary
                .broadcast_mul(&cos)
                .and_then(|lhs| rotated.broadcast_mul(&sin).and_then(|rhs| &lhs + &rhs))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "apply RoPE", e))?;
            Tensor::cat(&[&embedded, &pass], D::Minus1)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "RoPE output", e))
        }
    }

    fn forward(&self, hidden_states: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor, Error> {
        let (batch, seq_len, _) = hidden_states
            .dims3()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention input", e))?;
        let q = self
            .q_proj
            .forward(hidden_states)
            .and_then(|x| x.reshape((batch, seq_len, self.num_heads, self.head_dim)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "query projection", e))?;
        let q = self
            .q_norm
            .forward(&q)?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "query layout", e))?;
        let k = self
            .k_proj
            .forward(hidden_states)
            .and_then(|x| x.reshape((batch, seq_len, self.num_kv_heads, self.head_dim)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "key projection", e))?;
        let k = self
            .k_norm
            .forward(&k)?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "key layout", e))?;
        let v = self
            .v_proj
            .forward(hidden_states)
            .and_then(|x| x.reshape((batch, seq_len, self.num_kv_heads, self.head_dim)))
            .and_then(|x| x.transpose(1, 2))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "value projection", e))?;
        let q = self
            .apply_rope(&q, cos, sin)?
            .contiguous()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "query contiguous", e))?;
        let k = self
            .apply_rope(&k, cos, sin)?
            .contiguous()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "key contiguous", e))?;
        let v = v
            .contiguous()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "value contiguous", e))?;
        let (k, v) = self
            .kv_cache
            .borrow_mut()
            .append(&k, &v)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "KV cache", e))?;

        let output = match flash_attention(&q, &k, &v, self.scaling, seq_len > 1)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "flash attention", e))?
        {
            Some(output) => output,
            None => scaled_dot_product_attention_gqa(
                &q,
                &k,
                &v,
                None,
                self.scaling,
                true,
                self.num_kv_groups,
            )
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "grouped-query attention", e))?,
        };
        let output = output
            .transpose(1, 2)
            .and_then(|x| x.reshape((batch, seq_len, self.num_heads * self.head_dim)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention output layout", e))?;
        self.o_proj
            .forward(&output)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "output projection", e))
    }

    fn clear_cache(&self) {
        self.kv_cache.borrow_mut().reset();
    }
}

#[derive(Debug)]
struct Qwen3Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl Qwen3Mlp {
    fn load(cfg: &Qwen3VlTextConfig, vb: VarBuilder) -> Result<Self, Error> {
        let gate_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load MLP gate_proj", e))?;
        let up_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load MLP up_proj", e))?;
        let down_proj = linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load MLP down_proj", e))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, Error> {
        let gate = self
            .gate_proj
            .forward(xs)
            .and_then(|gate| candle_nn::ops::silu(&gate))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MLP gate", e))?;
        let up = self
            .up_proj
            .forward(xs)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MLP up", e))?;
        self.down_proj
            .forward(
                &(&gate * &up)
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MLP gate product", e))?,
            )
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MLP down", e))
    }
}

#[derive(Debug)]
struct DecoderLayer {
    attention: Qwen3Attention,
    mlp: Qwen3Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn load(cfg: &Qwen3VlTextConfig, vb: VarBuilder) -> Result<Self, Error> {
        Ok(Self {
            attention: Qwen3Attention::load(cfg, vb.pp("self_attn"))?,
            mlp: Qwen3Mlp::load(cfg, vb.pp("mlp"))?,
            input_layernorm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load input_layernorm", e))?,
            post_attention_layernorm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load post_attention_layernorm", e))?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor, Error> {
        let residual = hidden_states.clone();
        let normalized = self.input_layernorm.forward(hidden_states)?;
        let mixed = self.attention.forward(&normalized, cos, sin)?;
        let hidden_states = (&residual + &mixed)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention residual", e))?;
        let residual = hidden_states.clone();
        let normalized = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp = self.mlp.forward(&normalized)?;
        (&residual + &mlp).map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MLP residual", e))
    }

    fn clear_cache(&self) {
        self.attention.clear_cache();
    }
}

/// Interleaved-MRoPE axis per rotary dimension: frequencies cycle T/H/W in
/// steps of three, keeping the tail (past `h_limit`/`w_limit`) on the
/// temporal axis. Mirrors `Qwen3VLTextRotaryEmbedding.apply_interleaved_mrope`.
pub(crate) fn interleaved_axis_ids(rotary_dim: usize, mrope_section: &[usize]) -> Vec<u32> {
    let half = rotary_dim / 2;
    let h_limit = mrope_section[1] * 3;
    let w_limit = mrope_section[2] * 3;
    (0..rotary_dim)
        .map(|dimension| {
            let freq_idx = dimension % half;
            if freq_idx % 3 == 1 && freq_idx < h_limit {
                1
            } else if freq_idx % 3 == 2 && freq_idx < w_limit {
                2
            } else {
                0
            }
        })
        .collect()
}

/// Visual features tapped from the vision tower, to be added to the
/// image-token hidden states of the first `embeds.len()` decoder layers.
/// `image_span` is the `(start, len)` of the contiguous image-token span.
#[derive(Debug, Clone)]
pub(crate) struct DeepstackVisualEmbeds {
    pub image_span: (usize, usize),
    pub embeds: Vec<Tensor>,
}

#[derive(Debug, Clone)]
struct TextRotaryEmbedding {
    rotary: RotaryEmbedding,
    axis_ids: Tensor,
}

impl TextRotaryEmbedding {
    fn new(cfg: &Qwen3VlTextConfig, device: &Device) -> Result<Self, Error> {
        let rotary = RotaryEmbedding::new_multi_axis(cfg.head_dim, cfg.rope_theta(), 3, device)?;
        let axis_ids = Tensor::from_vec(
            interleaved_axis_ids(cfg.head_dim, cfg.mrope_section()),
            (1, 1, cfg.head_dim, 1),
            device,
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "create mRoPE axis map", e))?;
        Ok(Self { rotary, axis_ids })
    }

    fn forward(&self, position_ids: &Tensor, dtype: DType) -> Result<(Tensor, Tensor), Error> {
        let (cos, sin) = self.rotary.forward_multi_axis(position_ids, dtype)?;
        Ok((self.select_axes(&cos)?, self.select_axes(&sin)?))
    }

    /// Reduce `(3, batch, seq, head_dim)` to `(batch, seq, head_dim)` by
    /// gathering each dimension's angle from its interleaved axis.
    fn select_axes(&self, values: &Tensor) -> Result<Tensor, Error> {
        let (_, batch, seq_len, rotary_dim) = values
            .dims4()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "mRoPE tensor shape", e))?;
        let values = values
            .permute((1, 2, 3, 0))
            .and_then(|values| values.contiguous())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "mRoPE axis layout", e))?;
        let axis_ids = self
            .axis_ids
            .expand((batch, seq_len, rotary_dim, 1))
            .and_then(|axis_ids| axis_ids.contiguous())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "expand mRoPE axis map", e))?;
        values
            .gather(&axis_ids, 3)
            .and_then(|values| values.squeeze(3))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "select mRoPE axes", e))
    }
}

pub(crate) struct Qwen3VlTextModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    rotary_emb: TextRotaryEmbedding,
}

impl Qwen3VlTextModel {
    pub(crate) fn load(cfg: &Qwen3VlTextConfig, vb: VarBuilder) -> Result<Self, Error> {
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load token embeddings", e))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for index in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::load(cfg, vb.pp(format!("layers.{index}")))?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load final norm", e))?;
        let rotary_emb = TextRotaryEmbedding::new(cfg, vb.device())?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
        })
    }

    pub(crate) fn embed(&self, input_ids: &Tensor) -> Result<Tensor, Error> {
        self.embed_tokens
            .forward(input_ids)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "token embedding", e))
    }

    /// The token embedding matrix, used as the tied output projection.
    pub(crate) fn token_embedding_weight(&self) -> Tensor {
        self.embed_tokens.embeddings().clone()
    }

    pub(crate) fn forward(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        deepstack: Option<&DeepstackVisualEmbeds>,
    ) -> Result<Tensor, Error> {
        let (cos, sin) = self
            .rotary_emb
            .forward(position_ids, inputs_embeds.dtype())?;
        let mut hidden_states = inputs_embeds.clone();
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, &cos, &sin)?;
            if let Some(deepstack) = deepstack
                && layer_index < deepstack.embeds.len()
            {
                hidden_states = add_deepstack(hidden_states, deepstack, layer_index)?;
            }
        }
        self.norm
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "final norm", e))
    }

    pub(crate) fn clear_cache(&self) {
        for layer in &self.layers {
            layer.clear_cache();
        }
    }
}

/// Add one DeepStack feature map to the image-token span of `hidden_states`
/// (`(1, seq, hidden)` batch layout).
fn add_deepstack(
    hidden_states: Tensor,
    deepstack: &DeepstackVisualEmbeds,
    layer_index: usize,
) -> Result<Tensor, Error> {
    let (batch, seq_len, hidden_size) = hidden_states
        .dims3()
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "deepstack input", e))?;
    let (start, len) = deepstack.image_span;
    if batch != 1 || start + len > seq_len {
        return Err(Error::InvalidInput {
            message: format!(
                "{MODEL_NAME} deepstack span {start}..{} outside sequence (batch {batch}, len {seq_len})",
                start + len
            ),
        });
    }
    let embeds = deepstack.embeds[layer_index]
        .to_dtype(hidden_states.dtype())
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "deepstack feature layout", e))?;
    if embeds
        .dim(0)
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "deepstack feature length", e))?
        != len
    {
        return Err(Error::InvalidInput {
            message: format!(
                "{MODEL_NAME} deepstack feature length {} != image span {len}",
                deepstack.embeds[layer_index].dim(0).unwrap_or(0)
            ),
        });
    }
    let prefix = if start == 0 {
        Tensor::zeros(
            (1, 0, hidden_size),
            hidden_states.dtype(),
            hidden_states.device(),
        )
    } else {
        hidden_states.narrow(1, 0, start)
    }
    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "deepstack prefix", e))?;
    let image_hidden = hidden_states
        .narrow(1, start, len)
        .and_then(|image| image + &embeds.unsqueeze(0)?)
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "deepstack add", e))?;
    let suffix = if start + len == seq_len {
        Tensor::zeros(
            (1, 0, hidden_size),
            hidden_states.dtype(),
            hidden_states.device(),
        )
    } else {
        hidden_states.narrow(1, start + len, seq_len - start - len)
    }
    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "deepstack suffix", e))?;
    Tensor::cat(&[&prefix, &image_hidden, &suffix], 1)
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "deepstack splice", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn interleaved_mrope_axes_follow_t_h_w_cycle_with_temporal_tail() {
        // head_dim 64, sections [11, 11, 10]: dims cycle T,H,W for the first
        // 3*10=30... following the reference: H occupies idx 1,4,..,<11*3 and
        // W occupies 2,5,..,<10*3; everything else stays T.
        let ids = interleaved_axis_ids(64, &[11, 11, 10]);
        assert_eq!(ids.len(), 64);
        assert_eq!(ids[0], 0);
        assert_eq!(ids[1], 1);
        assert_eq!(ids[2], 2);
        assert_eq!(ids[3], 0);
        // h_limit = 33, w_limit = 30: idx 31 (1 mod 3, < 33) is still H; idx
        // 32 (2 mod 3, >= 30) falls back to T.
        assert_eq!(ids[29], 2);
        assert_eq!(ids[31], 1);
        assert_eq!(ids[32], 0);
        // Second half mirrors the first via `freq_idx = d % half`.
        assert_eq!(ids[32 + 1], 1);
        assert_eq!(ids[32 + 29], 2);
        // Last frequency (31 < h_limit) is still height.
        assert_eq!(ids[63], 1);
    }

    #[test]
    fn wevisdoc_mrope_sections_select_expected_axes() {
        // [24, 20, 20]: frequencies 0..59 cycle T,H,W (20 each); 60..63 stay
        // temporal, and the second head half mirrors the first.
        let ids = interleaved_axis_ids(128, &[24, 20, 20]);
        assert_eq!(&ids[..4], &[0, 1, 2, 0]);
        assert_eq!(ids[59], 2);
        assert_eq!(&ids[60..64], &[0; 4]);
        assert_eq!(&ids[124..128], &[0; 4]);
        assert_eq!(ids[64], 0);
        assert_eq!(ids[65], 1);
        assert_eq!(ids[123], 2);
    }

    #[test]
    fn config_rejects_section_sum_mismatch() {
        let mut cfg = valid_tiny_config();
        cfg.rope_scaling.mrope_section = vec![8, 8, 8];
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rejects_non_interleaved_rope() {
        let mut cfg = valid_tiny_config();
        cfg.rope_scaling.mrope_interleaved = false;
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn config_rejects_unknown_rope_type() {
        let mut cfg = valid_tiny_config();
        cfg.rope_scaling.rope_type = "linear".to_string();
        assert!(cfg.validate().is_err());
    }

    fn valid_tiny_config() -> Qwen3VlTextConfig {
        Qwen3VlTextConfig {
            model_type: "qwen3_vl_text".to_string(),
            vocab_size: 100,
            hidden_size: 64,
            intermediate_size: 32,
            num_hidden_layers: 1,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 32,
            max_position_embeddings: 1024,
            rms_norm_eps: 1e-6,
            rope_scaling: super::Qwen3VlRopeScaling {
                rope_type: "default".to_string(),
                mrope_interleaved: true,
                mrope_section: vec![6, 5, 5],
                rope_theta: 1_000_000.0,
            },
            eos_token_id: 1,
            attention_bias: false,
            tie_word_embeddings: true,
        }
    }

    #[test]
    fn deepstack_add_splices_only_the_image_span() -> Result<(), Error> {
        let device = Device::Cpu;
        let hidden = Tensor::from_vec(
            (0..24).map(|v| v as f32).collect::<Vec<_>>(),
            (1, 6, 4),
            &device,
        )?;
        let embeds = Tensor::from_vec(vec![100f32; 8], (2, 4), &device)?;
        let deepstack = DeepstackVisualEmbeds {
            image_span: (2, 2),
            embeds: vec![embeds],
        };
        let out = add_deepstack(hidden, &deepstack, 0)?;
        let values = out.flatten_all()?.to_vec1::<f32>()?;
        // Positions 2 and 3 (dims 8..16) gain 100.
        assert_eq!(
            &values[8..16],
            &[108.0, 109.0, 110.0, 111.0, 112.0, 113.0, 114.0, 115.0]
        );
        assert_eq!(&values[..8], &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]);
        assert_eq!(
            &values[16..],
            &[16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0]
        );
        Ok(())
    }

    #[test]
    fn deepstack_add_rejects_span_outside_sequence() -> Result<(), Error> {
        let device = Device::Cpu;
        let hidden = Tensor::zeros((1, 4, 4), DType::F32, &device)?;
        let embeds = Tensor::zeros((2, 4), DType::F32, &device)?;
        let deepstack = DeepstackVisualEmbeds {
            image_span: (3, 2),
            embeds: vec![embeds],
        };
        assert!(add_deepstack(hidden, &deepstack, 0).is_err());
        Ok(())
    }
}
