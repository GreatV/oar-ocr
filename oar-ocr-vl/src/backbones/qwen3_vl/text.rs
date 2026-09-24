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
#[cfg(feature = "cuda")]
use crate::runtime::cuda::dynamic_kv::DynamicKvAppend;
#[cfg(feature = "cuda")]
use crate::runtime::decoder_graph::{
    CudaGraphDrainGuard, CudaGraphKvLengths, SingleTokenDecoderCudaGraph, cuda_graph_error,
    decoder_attention_is_causal, decoder_cache_capacity, sync_graph_tensor,
};
use crate::runtime::errors::candle_to_ocr_inference;
use crate::runtime::tensor::rotate_half;
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{
    Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_no_bias, rms_norm,
};
use std::cell::RefCell;

const MODEL_NAME: &str = "Qwen3-VL";

/// Upper bound for graph-backed decode KV buckets. The graph's masked
/// attention scans the whole bucket every step, so this trades generation
/// headroom before falling back to eager decoding against per-step scan cost.
#[cfg(feature = "cuda")]
const WEVISDOC_DECODE_CACHE_LEN: usize = 8_192;

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

    fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, Error> {
        let (batch, seq_len, _) = hidden_states
            .dims3()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention input", e))?;
        let (q, k, v) = self.project_qkv(hidden_states, cos, sin)?;
        let (k, v) = self
            .kv_cache
            .borrow_mut()
            .append(&k, &v)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "KV cache", e))?;
        let output = self.attend(&q, &k, &v, batch, seq_len, attention_mask)?;
        self.project_output(&output, batch, seq_len)
    }

    fn project_qkv(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor), Error> {
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
        Ok((q, k, v))
    }

    fn attend(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        batch: usize,
        seq_len: usize,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, Error> {
        // Single-token decoding keeps to the eager gemm kernels: FA2 tiles
        // q into 128-row blocks, so a one-query step lights up only
        // `heads` blocks (12% of the SMs here) and reads the KV cache at
        // a fraction of the achievable bandwidth.
        let flash = if batch == 1 && seq_len > 1 {
            flash_attention(q, k, v, self.scaling, seq_len > 1)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "flash attention", e))?
        } else {
            None
        };
        match flash {
            Some(output) => Ok(output),
            None => {
                // Masked batch attention materializes (B, heads, q, kv)
                // scores eagerly; page-scale prefills chunk the query axis so
                // the transient stays bounded.
                const MASKED_ATTN_CHUNK: usize = 1024;
                if attention_mask.is_some() && seq_len > MASKED_ATTN_CHUNK {
                    let mut chunks = Vec::with_capacity(seq_len.div_ceil(MASKED_ATTN_CHUNK));
                    let mut start = 0usize;
                    while start < seq_len {
                        let len = (seq_len - start).min(MASKED_ATTN_CHUNK);
                        let q_chunk = q.narrow(2, start, len)?;
                        let mask_chunk = attention_mask
                            .map(|mask| mask.narrow(2, start, len))
                            .transpose()?;
                        chunks.push(
                            scaled_dot_product_attention_gqa(
                                &q_chunk,
                                k,
                                v,
                                mask_chunk.as_ref(),
                                self.scaling,
                                false,
                                self.num_kv_groups,
                            )
                            .map_err(|e| {
                                candle_to_ocr_inference(MODEL_NAME, "grouped-query attention", e)
                            })?,
                        );
                        start += len;
                    }
                    let refs: Vec<&Tensor> = chunks.iter().collect();
                    return Tensor::cat(&refs, 2)
                        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention chunks", e));
                }
                scaled_dot_product_attention_gqa(
                    q,
                    k,
                    v,
                    attention_mask,
                    self.scaling,
                    attention_mask.is_none(),
                    self.num_kv_groups,
                )
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "grouped-query attention", e))
            }
        }
    }

    fn project_output(
        &self,
        output: &Tensor,
        batch: usize,
        seq_len: usize,
    ) -> Result<Tensor, Error> {
        let output = output
            .transpose(1, 2)
            .and_then(|x| x.reshape((batch, seq_len, self.num_heads * self.head_dim)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention output layout", e))?;
        self.o_proj
            .forward(&output)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "output projection", e))
    }

    #[cfg(feature = "cuda")]
    fn prepare_dynamic_cache(&self, query_len: usize, cache_len: usize) -> Result<(), Error> {
        let template = Tensor::zeros(
            (1, self.num_kv_heads, query_len, self.head_dim),
            self.q_proj.weight().dtype(),
            self.q_proj.weight().device(),
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "dynamic KV template", e))?;
        self.kv_cache
            .borrow_mut()
            .initialize_storage_with_capacity(&template, cache_len)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "initialize dynamic KV", e))
    }

    /// CUDA-graph decode step: appends into fixed-capacity storage and runs
    /// masked attention over `[0, kv_len]`. `kv_positions` is the constant
    /// `(1, 1, cache_len)` index row (built before capture — `arange`
    /// uploads from host, which capture forbids).
    #[cfg(feature = "cuda")]
    fn forward_dynamic(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        query_lengths: &Tensor,
        kv_lengths: &Tensor,
        kv_positions: &Tensor,
    ) -> Result<Tensor, Error> {
        let (batch, query_len, _) = hidden_states
            .dims3()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "dynamic attention input", e))?;
        if batch != 1 {
            return Err(Error::Config {
                message: format!(
                    "{MODEL_NAME} CUDA-graph attention requires batch size 1, got {batch}"
                ),
            });
        }
        let (q, k, v) = self.project_qkv(hidden_states, cos, sin)?;
        let cache = self.kv_cache.borrow();
        let cache_len = cache.storage_capacity();
        let (cache_k, cache_v) = cache.storage().ok_or_else(|| Error::Config {
            message: format!("{MODEL_NAME} dynamic KV storage is not initialized"),
        })?;
        drop(cache);
        let append = DynamicKvAppend {
            query_len,
            cache_len,
        };
        cache_k
            .inplace_op3(&k, kv_lengths, &append)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "dynamic key cache append", e))?;
        cache_v
            .inplace_op3(&v, kv_lengths, &append)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "dynamic value cache append", e))?;

        // Attention over the fixed-capacity storage with a device-side
        // additive mask derived from `kv_lengths`: FA2's varlen kernel runs
        // one 128-row query block per head on this shape (a handful of the
        // GPU's SMs) and reads the cache far below bandwidth, while the
        // eager gemm kernels stay fast. Masked positions get a very negative
        // score, so stale storage beyond the live length contributes exactly
        // zero after the softmax — identical math to the narrowed eager path.
        let kv_bound = kv_lengths
            .i(1..)
            .and_then(|bound| bound.reshape((1, 1, 1)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "dynamic KV bound", e))?;
        let live = kv_positions.broadcast_lt(&kv_bound)?;
        // live -> 0, dead -> -1e9 via a scalar affine (host scalars ride
        // as kernel parameters, so nothing uploads during graph capture).
        let mask = live
            .to_dtype(hidden_states.dtype())?
            .affine(1e9, -1e9)?
            .unsqueeze(1)?;
        let attn = scaled_dot_product_attention_gqa(
            &q,
            &cache_k,
            &cache_v,
            Some(&mask),
            self.scaling,
            false,
            self.num_kv_groups,
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "dynamic masked attention", e))?;
        let _ = query_lengths;
        self.project_output(&attn, batch, query_len)
    }

    #[cfg(feature = "cuda")]
    fn kv_cache_len(&self) -> usize {
        self.kv_cache.borrow().current_seq_len()
    }

    #[cfg(feature = "cuda")]
    fn set_kv_cache_len(&self, len: usize) -> Result<(), Error> {
        self.kv_cache
            .borrow_mut()
            .set_current_len(len)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "set dynamic KV length", e))
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

    fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, Error> {
        let residual = hidden_states.clone();
        let normalized = self.input_layernorm.forward(hidden_states)?;
        let mixed = self
            .attention
            .forward(&normalized, cos, sin, attention_mask)?;
        let hidden_states = (&residual + &mixed)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention residual", e))?;
        let residual = hidden_states.clone();
        let normalized = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp = self.mlp.forward(&normalized)?;
        (&residual + &mlp).map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MLP residual", e))
    }

    #[cfg(feature = "cuda")]
    fn forward_dynamic(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        query_lengths: &Tensor,
        kv_lengths: &Tensor,
        kv_positions: &Tensor,
    ) -> Result<Tensor, Error> {
        let residual = hidden_states.clone();
        let normalized = self.input_layernorm.forward(hidden_states)?;
        let mixed = self.attention.forward_dynamic(
            &normalized,
            cos,
            sin,
            query_lengths,
            kv_lengths,
            kv_positions,
        )?;
        let hidden_states = (&residual + &mixed)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention residual", e))?;
        let residual = hidden_states.clone();
        let normalized = self.post_attention_layernorm.forward(&hidden_states)?;
        let mlp = self.mlp.forward(&normalized)?;
        (&residual + &mlp).map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MLP residual", e))
    }

    #[cfg(feature = "cuda")]
    fn prepare_dynamic_cache(&self, query_len: usize, cache_len: usize) -> Result<(), Error> {
        self.attention.prepare_dynamic_cache(query_len, cache_len)
    }

    #[cfg(feature = "cuda")]
    fn set_kv_cache_len(&self, len: usize) -> Result<(), Error> {
        self.attention.set_kv_cache_len(len)
    }

    #[cfg(feature = "cuda")]
    fn kv_cache_len(&self) -> usize {
        self.attention.kv_cache_len()
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
/// `image_spans` carries one contiguous image-token `(start, len)` span per
/// batch row; `embeds[layer]` concatenates the rows' feature maps in row
/// order, so rows may have different span lengths (different image grids).
#[derive(Debug, Clone)]
pub(crate) struct DeepstackVisualEmbeds {
    pub image_spans: Vec<(usize, usize)>,
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
    #[cfg(feature = "cuda")]
    decode_graph: RefCell<Option<SingleTokenDecoderCudaGraph>>,
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    rotary_emb: TextRotaryEmbedding,
    // Must stay the last field: it drops last and drains CUDA errors the
    // other fields' frees may stash (see CudaGraphDrainGuard).
    #[cfg(feature = "cuda")]
    _drain_guard: CudaGraphDrainGuard,
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

        #[cfg(feature = "cuda")]
        let _drain_guard = CudaGraphDrainGuard::new(vb.device());
        Ok(Self {
            #[cfg(feature = "cuda")]
            decode_graph: RefCell::new(None),
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            #[cfg(feature = "cuda")]
            _drain_guard,
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
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, Error> {
        let (cos, sin) = self
            .rotary_emb
            .forward(position_ids, inputs_embeds.dtype())?;
        let mut hidden_states = inputs_embeds.clone();
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, &cos, &sin, attention_mask)?;
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

    #[cfg(feature = "cuda")]
    fn forward_dynamic(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        query_lengths: &Tensor,
        kv_lengths: &Tensor,
        kv_positions: &Tensor,
    ) -> Result<Tensor, Error> {
        let (cos, sin) = self
            .rotary_emb
            .forward(position_ids, inputs_embeds.dtype())?;
        let mut hidden_states = inputs_embeds.clone();
        for layer in &self.layers {
            hidden_states = layer.forward_dynamic(
                &hidden_states,
                &cos,
                &sin,
                query_lengths,
                kv_lengths,
                kv_positions,
            )?;
        }
        self.norm
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "final norm", e))
    }

    fn project_logits(&self, hidden_states: &Tensor, lm_head: &Linear) -> Result<Tensor, Error> {
        lm_head
            .forward(hidden_states)
            .and_then(|logits| logits.i((0, 0, ..)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "decode LM head", e))
    }

    /// One decode step at `position_ids`; batch rows beyond the first
    /// require `attention_mask`.
    pub(crate) fn forward_decode_logits(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        lm_head: &Linear,
    ) -> Result<Tensor, Error> {
        #[cfg(feature = "cuda")]
        if attention_mask.is_none() {
            let kv_len = self.kv_cache_len().saturating_add(1);
            if let Some(logits) = self.replay_cuda_graph(inputs_embeds, position_ids, kv_len)? {
                return Ok(logits);
            }
        }
        let hidden = self.forward(inputs_embeds, position_ids, None, attention_mask)?;
        self.project_logits(&hidden, lm_head)
    }

    /// Capture the batch-1 single-token decode graph when eligible.
    pub(crate) fn prepare_ar_cuda_graph(
        &self,
        prompt_len: usize,
        max_new_tokens: usize,
        lm_head: &Linear,
    ) -> Result<(), Error> {
        if std::env::var_os("OAR_VL_DISABLE_CUDA_GRAPH").is_some()
            || std::env::var_os("OAR_WEVISDOC_DISABLE_CUDA_GRAPH").is_some()
        {
            #[cfg(feature = "cuda")]
            self.invalidate_cuda_graph();
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        if self.embed_tokens.embeddings().device().is_cuda()
            && matches!(
                self.embed_tokens.embeddings().dtype(),
                DType::BF16 | DType::F16
            )
        {
            let Some(cache_len) =
                decoder_cache_capacity(prompt_len, max_new_tokens, WEVISDOC_DECODE_CACHE_LEN)
            else {
                self.invalidate_cuda_graph();
                return Ok(());
            };
            let required = prompt_len
                .saturating_add(max_new_tokens)
                .min(WEVISDOC_DECODE_CACHE_LEN);
            let reusable = self
                .decode_graph
                .borrow()
                .as_ref()
                .is_some_and(|graph| graph.cache_len >= required);
            if reusable {
                return Ok(());
            }
            self.invalidate_cuda_graph();
            self.capture_cuda_graph(cache_len, lm_head)?;
        }
        let _ = prompt_len;
        let _ = max_new_tokens;
        let _ = lm_head;
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn capture_cuda_graph(&self, cache_len: usize, lm_head: &Linear) -> Result<(), Error> {
        use candle_core::cuda_backend::cudarc::driver::sys::{
            CUgraphInstantiate_flags_enum, CUstreamCaptureMode_enum,
        };

        if self.decode_graph.borrow().is_some() {
            return Ok(());
        }
        let Device::Cuda(cuda) = self.embed_tokens.embeddings().device() else {
            return Ok(());
        };
        let query_len = 1;
        for layer in &self.layers {
            layer.prepare_dynamic_cache(query_len, cache_len)?;
        }
        let hidden_size = self
            .embed_tokens
            .embeddings()
            .dim(1)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "graph hidden size", e))?;
        let device = self.embed_tokens.embeddings().device().clone();
        let hidden_input = Tensor::zeros(
            (1, query_len, hidden_size),
            self.embed_tokens.embeddings().dtype(),
            &device,
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "graph hidden input", e))?;
        let position_input = Tensor::zeros((3, 1, query_len), DType::I64, &device)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "graph position input", e))?;
        let query_lengths = Tensor::new(&[0u32, query_len as u32], &device)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "graph query lengths", e))?;
        let kv_lengths = CudaGraphKvLengths::new(query_len, &device)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "graph KV lengths", e))?;
        let kv_positions =
            Tensor::arange(0u32, cache_len as u32, &device)?.reshape((1, 1, cache_len))?;
        let stream = cuda.cuda_stream();
        let _htod_cache = cuda.enable_cuda_graph_htod_cache();

        let warm = self.forward_dynamic(
            &hidden_input,
            &position_input,
            &query_lengths,
            kv_lengths.tensor(),
            &kv_positions,
        )?;
        let warm_logits = self.project_logits(&warm, lm_head)?;
        sync_graph_tensor(MODEL_NAME, &warm_logits, "warm decoder CUDA graph")?;
        // Allocate the output buffer before capture so it belongs to the
        // regular stream-ordered pool; a capture-time allocation lives in the
        // graph's private pool and can never be returned to the allocator
        // safely. Prime the copy so the captured run sees a warm kernel.
        let logits_output = Tensor::zeros_like(&warm_logits)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "graph logits output", e))?;
        logits_output
            .slice_set(&warm_logits, 0, 0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "prime graph logits copy", e))?;

        stream
            .begin_capture(CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_GLOBAL)
            .map_err(|e| cuda_graph_error(MODEL_NAME, "begin decoder CUDA graph capture", e))?;
        let captured_output: Result<(), Error> = (|| {
            let hidden = self.forward_dynamic(
                &hidden_input,
                &position_input,
                &query_lengths,
                kv_lengths.tensor(),
                &kv_positions,
            )?;
            let logits = self.project_logits(&hidden, lm_head)?;
            logits_output
                .slice_set(&logits, 0, 0)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "record graph logits copy", e))
        })();
        if let Err(error) = captured_output {
            let _ = stream.end_capture(
                CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            return Err(error);
        }
        let graph = stream
            .end_capture(
                CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .map_err(|e| cuda_graph_error(MODEL_NAME, "end decoder CUDA graph capture", e))?
            .ok_or_else(|| Error::Config {
                message: format!("{MODEL_NAME} decoder capture returned no graph"),
            })?;
        graph
            .launch()
            .map_err(|e| cuda_graph_error(MODEL_NAME, "warm decoder CUDA graph", e))?;
        sync_graph_tensor(MODEL_NAME, &logits_output, "sync decoder CUDA graph")?;
        self.clear_cache();
        *self.decode_graph.borrow_mut() = Some(SingleTokenDecoderCudaGraph {
            graph,
            hidden_input,
            position_input,
            _query_lengths: query_lengths,
            kv_lengths,
            logits_output,
            cache_len,
        });
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn replay_cuda_graph(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        kv_len: usize,
    ) -> Result<Option<Tensor>, Error> {
        let captured_ref = self.decode_graph.borrow();
        let Some(captured) = captured_ref.as_ref() else {
            return Ok(None);
        };
        if kv_len > captured.cache_len {
            drop(captured_ref);
            self.invalidate_cuda_graph();
            return Ok(None);
        }
        if inputs_embeds.shape() != captured.hidden_input.shape()
            || position_ids.shape() != captured.position_input.shape()
        {
            return Ok(None);
        }
        captured
            .hidden_input
            .slice_set(inputs_embeds, 0, 0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "copy graph hidden", e))?;
        captured
            .position_input
            .slice_set(position_ids, 0, 0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "copy graph positions", e))?;
        captured
            .kv_lengths
            .update(kv_len)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "update graph KV lengths", e))?;
        captured
            .graph
            .launch()
            .map_err(|e| cuda_graph_error(MODEL_NAME, "launch decoder CUDA graph", e))?;
        for layer in &self.layers {
            layer.set_kv_cache_len(kv_len)?;
        }
        Ok(Some(captured.logits_output.clone()))
    }

    #[cfg(feature = "cuda")]
    fn invalidate_cuda_graph(&self) {
        if let Some(graph) = self.decode_graph.borrow_mut().take() {
            graph.dispose();
        }
    }

    pub(crate) fn invalidate_ar_cuda_graph(&self) {
        #[cfg(feature = "cuda")]
        self.invalidate_cuda_graph();
    }

    /// Whether the decode graph is currently captured — lets the GPU
    /// self-check assert the capture really ran (it is bf16/f16-gated).
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    pub(crate) fn decode_graph_captured(&self) -> bool {
        self.decode_graph.borrow().is_some()
    }

    #[cfg(feature = "cuda")]
    fn kv_cache_len(&self) -> usize {
        let len = self.layers.first().map_or(0, |layer| layer.kv_cache_len());
        debug_assert!(self.layers.iter().all(|layer| layer.kv_cache_len() == len));
        len
    }

    pub(crate) fn clear_cache(&self) {
        for layer in &self.layers {
            layer.clear_cache();
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for Qwen3VlTextModel {
    fn drop(&mut self) {
        // A cached graph must go through dispose: plainly dropping it returns
        // graph-bound buffers to the allocator and poisons it.
        self.invalidate_cuda_graph();
    }
}

/// Add one DeepStack feature map to each row's image-token span of
/// `hidden_states` (`(batch, seq, hidden)` layout). Rows share the span
/// lengths may differ across rows (different image grids); starts shift with
/// left padding.
fn add_deepstack(
    hidden_states: Tensor,
    deepstack: &DeepstackVisualEmbeds,
    layer_index: usize,
) -> Result<Tensor, Error> {
    let (batch, seq_len, hidden_size) = hidden_states
        .dims3()
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "deepstack input", e))?;
    if batch != deepstack.image_spans.len() {
        return Err(Error::InvalidInput {
            message: format!(
                "{MODEL_NAME} deepstack spans cover {} rows, hidden states have {batch}",
                deepstack.image_spans.len()
            ),
        });
    }
    let total_features: usize = deepstack.image_spans.iter().map(|&(_, len)| len).sum();
    let embeds = deepstack.embeds[layer_index]
        .to_dtype(hidden_states.dtype())
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "deepstack feature layout", e))?;
    let embeds_len = embeds
        .dim(0)
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "deepstack feature length", e))?;
    if embeds_len != total_features {
        return Err(Error::InvalidInput {
            message: format!(
                "{MODEL_NAME} deepstack feature length {embeds_len} != total span {total_features}"
            ),
        });
    }
    let mut rows = Vec::with_capacity(batch);
    let mut feature_offset = 0usize;
    for (row, &(start, len)) in deepstack.image_spans.iter().enumerate() {
        if len == 0 || start + len > seq_len {
            return Err(Error::InvalidInput {
                message: format!(
                    "{MODEL_NAME} deepstack span {start}..{} outside row {row} (len {seq_len})",
                    start + len
                ),
            });
        }
        let row_features = embeds
            .narrow(0, feature_offset, len)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "deepstack row features", e))?;
        feature_offset += len;
        let row_hidden = hidden_states.i(row)?;
        let prefix = if start == 0 {
            Tensor::zeros(
                (0, hidden_size),
                hidden_states.dtype(),
                hidden_states.device(),
            )
        } else {
            row_hidden.narrow(0, 0, start)
        }
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "deepstack prefix", e))?;
        let image_hidden = row_hidden
            .narrow(0, start, len)
            .and_then(|image| image + row_features)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "deepstack add", e))?;
        let suffix = if start + len == seq_len {
            Tensor::zeros(
                (0, hidden_size),
                hidden_states.dtype(),
                hidden_states.device(),
            )
        } else {
            row_hidden.narrow(0, start + len, seq_len - start - len)
        }
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "deepstack suffix", e))?;
        rows.push(
            Tensor::cat(&[&prefix, &image_hidden, &suffix], 0)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "deepstack splice", e))?,
        );
    }
    let refs: Vec<&Tensor> = rows.iter().collect();
    Tensor::stack(&refs, 0)
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "deepstack batch stack", e))
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

    /// Random-weight equivalence of the padded batch forward against
    /// per-sequence forwards: two rows with different lengths, left-padded,
    /// per-row MRoPE positions, batch DeepStack spans, and the combined
    /// causal+padding mask must produce identical last-token logits to
    /// running each row alone.
    #[test]
    fn batched_forward_matches_single_sequences_with_deepstack() -> Result<(), Error> {
        use crate::runtime::attention::{
            combine_masks, create_causal_mask, create_left_padding_mask,
        };
        let device = Device::Cpu;
        let cfg = valid_tiny_config();
        let vb = random_varbuilder(&cfg, &device);
        let model = Qwen3VlTextModel::load(&cfg, vb.pp("model"))?;
        let lm_head = candle_nn::Linear::new(
            vb.get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")?,
            None,
        );

        // Two sequences with image spans at different offsets.
        let hidden_size = cfg.hidden_size;
        let seq_a = 12usize;
        let seq_b = 9usize;
        let span = 4usize;
        let build_row = |seq_len: usize, image_start: usize| -> Result<(Tensor, Tensor), Error> {
            let embeds = Tensor::from_vec(
                (0..seq_len * hidden_size)
                    .map(|i| (i % 17) as f32 / 17.0)
                    .collect(),
                (1, seq_len, hidden_size),
                &device,
            )?;
            let mut axes = [Vec::new(), Vec::new(), Vec::new()];
            for position in 0..seq_len as i64 {
                for axis in &mut axes {
                    axis.push(position);
                }
            }
            let data: Vec<i64> = axes.into_iter().flatten().collect();
            let position_ids = Tensor::from_vec(data, (3, 1, seq_len), &device)?;
            let _ = image_start;
            Ok((embeds, position_ids))
        };
        let (embeds_a, pos_a) = build_row(seq_a, 4)?;
        let (embeds_b, pos_b) = build_row(seq_b, 3)?;
        let lens = [seq_a, seq_b];
        let max_len = *lens.iter().max().unwrap();

        let feature = |tokens: usize| -> Tensor {
            Tensor::from_vec(
                (0..tokens * hidden_size)
                    .map(|i| ((i * 7) % 23) as f32 / 23.0)
                    .collect(),
                (tokens, hidden_size),
                &device,
            )
            .unwrap()
        };
        let single_deepstack = |image_start: usize, pad_len: usize| DeepstackVisualEmbeds {
            image_spans: vec![(image_start + pad_len, span)],
            embeds: vec![feature(span)],
        };

        // Batched: left-pad both rows, shift spans, combined mask.
        let mut embeds_rows = Vec::new();
        let mut position_rows = Vec::new();
        let mut spans = Vec::new();
        for (embeds, positions, image_start) in
            [(&embeds_a, &pos_a, 4usize), (&embeds_b, &pos_b, 3usize)]
        {
            let pad_len = max_len - positions.dim(2)?;
            let pad = Tensor::zeros((1, pad_len, hidden_size), DType::F32, &device)?;
            embeds_rows.push(Tensor::cat(&[&pad, embeds], 1)?);
            let pad_pos = Tensor::zeros((3, 1, pad_len), pos_a.dtype(), &device)?;
            position_rows.push(Tensor::cat(&[&pad_pos, positions], 2)?);
            spans.push((image_start + pad_len, span));
        }
        let batch_embeds = Tensor::cat(&embeds_rows.iter().collect::<Vec<_>>(), 0)?;
        let batch_positions = Tensor::cat(&position_rows.iter().collect::<Vec<_>>(), 1)?;
        let deepstack = DeepstackVisualEmbeds {
            image_spans: spans,
            embeds: vec![feature(2 * span)],
        };
        let causal = create_causal_mask(max_len, max_len, DType::F32, &device)?;
        let padding = create_left_padding_mask(&lens, max_len, DType::F32, &device)?;
        let mask = combine_masks(&causal, &padding)?;

        model.clear_cache();
        let batched = model.forward(
            &batch_embeds,
            &batch_positions,
            Some(&deepstack),
            Some(&mask),
        )?;
        let batched_logits = lm_head.forward(&batched.i((.., max_len - 1, ..))?.contiguous()?)?;

        for (row, (embeds, positions, image_start)) in
            [(&embeds_a, &pos_a, 4usize), (&embeds_b, &pos_b, 3usize)]
                .into_iter()
                .enumerate()
        {
            let single_deepstack = single_deepstack(image_start, 0);
            model.clear_cache();
            let single = model.forward(embeds, positions, Some(&single_deepstack), None)?;
            let seq_len = positions.dim(2)?;
            let last = single.i((0, seq_len - 1, ..))?.contiguous()?.unsqueeze(0)?;
            let single_logits = lm_head.forward(&last)?;
            let a = batched_logits.i(row)?;
            let b = single_logits.i(0)?;
            let diff = (&a - &b)?.abs()?.max_all()?.to_scalar::<f32>()?;
            assert!(diff < 1e-4, "row {row}: batched vs single delta {diff}");
        }
        Ok(())
    }

    /// GPU self-check: a bf16 random-weight model on CUDA must actually
    /// capture the decode graph (prepare succeeds, decode steps replay it),
    /// and graph replay must be token-identical to eager decoding on the
    /// same model. Skips without a CUDA device; opt in with
    /// `OAR_WEVISDOC_GPU_SELFTEST=1`.
    #[test]
    fn cuda_decode_graph_captures_and_matches_eager() {
        #[cfg(feature = "cuda")]
        {
            use candle_nn::Linear;
            if std::env::var_os("OAR_WEVISDOC_GPU_SELFTEST").is_none() {
                eprintln!("skipping: OAR_WEVISDOC_GPU_SELFTEST is not set");
                return;
            }
            let Ok(device) = Device::new_cuda(0) else {
                eprintln!("skipping: no CUDA device");
                return;
            };
            let cfg = valid_tiny_config();
            // The decode graph is bf16/f16-gated: build in bf16 so the
            // test really exercises capture and replay.
            let vb = random_varbuilder_typed(&cfg, &device, DType::BF16);
            let model = Qwen3VlTextModel::load(&cfg, vb.pp("model")).unwrap();
            let lm_head = Linear::new(
                vb.get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")
                    .unwrap(),
                None,
            );

            let ids: Vec<u32> = (4..36).map(|i| 10 + i % 60).collect();
            let seq_len = ids.len();
            let token_ids = Tensor::from_vec(ids.clone(), (1, seq_len), &device).unwrap();
            let embeds = model.embed(&token_ids).unwrap();
            let positions = {
                let base = Tensor::arange(0i64, seq_len as i64, &device)
                    .unwrap()
                    .reshape((1, 1, seq_len))
                    .unwrap();
                let mut data = Vec::with_capacity(3 * seq_len);
                for _ in 0..3 {
                    data.extend(base.flatten_all().unwrap().to_vec1::<i64>().unwrap());
                }
                Tensor::from_vec(data, (3, 1, seq_len), &device).unwrap()
            };

            // Eager baseline: greedy decode without the graph.
            model.clear_cache();
            let hidden = model.forward(&embeds, &positions, None, None).unwrap();
            let mut logits = lm_head
                .forward(
                    &hidden
                        .i((0, seq_len - 1, ..))
                        .unwrap()
                        .unsqueeze(0)
                        .unwrap(),
                )
                .unwrap()
                .squeeze(0)
                .unwrap();
            let mut eager = Vec::new();
            for step in 0..12 {
                let scores = logits
                    .to_dtype(DType::F32)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let mut best = 0usize;
                let mut best_value = f32::NEG_INFINITY;
                for (i, &v) in scores.iter().enumerate() {
                    if v > best_value {
                        best_value = v;
                        best = i;
                    }
                }
                if best as u32 == cfg.eos_token_id {
                    break;
                }
                eager.push(best as u32);
                let token = Tensor::from_vec(vec![best as u32], (1, 1), &device).unwrap();
                let embed = model.embed(&token).unwrap();
                let pos =
                    Tensor::from_vec(vec![(seq_len + step) as i64; 3], (3, 1, 1), &device).unwrap();
                logits = model
                    .forward_decode_logits(&embed, &pos, None, &lm_head)
                    .unwrap();
            }

            // Graph path: capture, assert it happened, replay the same steps.
            model.clear_cache();
            model.prepare_ar_cuda_graph(seq_len, 12, &lm_head).unwrap();
            assert!(
                model.decode_graph_captured(),
                "decode graph did not capture (dtype gate?)"
            );
            let hidden = model.forward(&embeds, &positions, None, None).unwrap();
            let mut logits = lm_head
                .forward(
                    &hidden
                        .i((0, seq_len - 1, ..))
                        .unwrap()
                        .unsqueeze(0)
                        .unwrap(),
                )
                .unwrap()
                .squeeze(0)
                .unwrap();
            let mut graphed = Vec::new();
            for step in 0..12 {
                let scores = logits
                    .to_dtype(DType::F32)
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                let mut best = 0usize;
                let mut best_value = f32::NEG_INFINITY;
                for (i, &v) in scores.iter().enumerate() {
                    if v > best_value {
                        best_value = v;
                        best = i;
                    }
                }
                if best as u32 == cfg.eos_token_id {
                    break;
                }
                graphed.push(best as u32);
                let token = Tensor::from_vec(vec![best as u32], (1, 1), &device).unwrap();
                let embed = model.embed(&token).unwrap();
                let pos =
                    Tensor::from_vec(vec![(seq_len + step) as i64; 3], (3, 1, 1), &device).unwrap();
                logits = model
                    .forward_decode_logits(&embed, &pos, None, &lm_head)
                    .unwrap();
            }
            assert_eq!(
                graphed, eager,
                "graph-replayed decoding must match eager decoding"
            );
        }
        #[cfg(not(feature = "cuda"))]
        eprintln!("skipping: built without the cuda feature");
    }

    fn random_varbuilder(cfg: &Qwen3VlTextConfig, device: &Device) -> VarBuilder<'static> {
        random_varbuilder_typed(cfg, device, DType::F32)
    }

    fn random_varbuilder_typed(
        cfg: &Qwen3VlTextConfig,
        device: &Device,
        dtype: DType,
    ) -> VarBuilder<'static> {
        use candle_nn::VarBuilder;
        let mut tensors = std::collections::HashMap::new();
        let h = cfg.hidden_size;
        let put = |tensors: &mut std::collections::HashMap<String, Tensor>,
                   name: String,
                   shape: Vec<usize>| {
            let len: usize = shape.iter().product();
            let data: Vec<f32> = (0..len)
                .map(|i| {
                    let x = (i as u32).wrapping_mul(2_654_435_761) % 10_001;
                    (x as f32 / 10_000.0 - 0.5) * 0.1
                })
                .collect();
            let tensor = Tensor::from_vec(data, shape, device)
                .unwrap()
                .to_dtype(dtype)
                .unwrap();
            tensors.insert(name, tensor);
        };
        put(
            &mut tensors,
            "model.embed_tokens.weight".into(),
            vec![cfg.vocab_size, h],
        );
        put(&mut tensors, "model.norm.weight".into(), vec![h]);
        put(
            &mut tensors,
            "lm_head.weight".into(),
            vec![cfg.vocab_size, h],
        );
        let prefix = "model.layers.0";
        for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
            let out = if proj == "q_proj" {
                cfg.num_attention_heads * cfg.head_dim
            } else if proj == "o_proj" {
                cfg.hidden_size
            } else {
                cfg.num_key_value_heads * cfg.head_dim
            };
            put(
                &mut tensors,
                format!("{prefix}.self_attn.{proj}.weight"),
                vec![out, h],
            );
        }
        put(
            &mut tensors,
            format!("{prefix}.self_attn.q_norm.weight"),
            vec![cfg.head_dim],
        );
        put(
            &mut tensors,
            format!("{prefix}.self_attn.k_norm.weight"),
            vec![cfg.head_dim],
        );
        for norm in ["input_layernorm", "post_attention_layernorm"] {
            put(&mut tensors, format!("{prefix}.{norm}.weight"), vec![h]);
        }
        for proj in ["gate_proj", "up_proj"] {
            put(
                &mut tensors,
                format!("{prefix}.mlp.{proj}.weight"),
                vec![cfg.intermediate_size, h],
            );
        }
        put(
            &mut tensors,
            format!("{prefix}.mlp.down_proj.weight"),
            vec![h, cfg.intermediate_size],
        );
        VarBuilder::from_tensors(tensors, dtype, device)
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
            image_spans: vec![(2, 2)],
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
            image_spans: vec![(3, 2)],
            embeds: vec![embeds],
        };
        assert!(add_deepstack(hidden, &deepstack, 0).is_err());
        Ok(())
    }
}
