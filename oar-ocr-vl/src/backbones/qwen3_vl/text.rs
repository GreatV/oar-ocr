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
#[cfg(feature = "cuda")]
use crate::runtime::attention::masked_score;
use crate::runtime::attention::{
    RotaryEmbedding, create_causal_mask, flash_attention, scaled_dot_product_attention_gqa,
};
use crate::runtime::cache::TrimmableKvCache;
#[cfg(feature = "cuda")]
use crate::runtime::cuda::dynamic_kv::{DynamicBatchKvAppend, DynamicKvAppend};
#[cfg(feature = "cuda")]
use crate::runtime::decoder_graph::{
    BatchDecodeRows, BatchDecoderCudaGraph, CudaGraphDrainGuard, CudaGraphKvLengths,
    CudaGraphPerRowU32, SingleTokenDecoderCudaGraph, cuda_graph_error, decoder_cache_capacity,
    drain_cuda_context_errors, drop_and_drain, next_decode_bucket, prompt_decode_bucket,
    sync_graph_tensor,
};
use crate::runtime::errors::candle_to_ocr_inference;
use crate::runtime::tensor::rotate_half;
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{
    Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_no_bias, rms_norm,
};
use std::cell::RefCell;

const MODEL_NAME: &str = "Qwen3-VL";

/// Upper bound for graph-backed decode KV buckets. Captured buckets start
/// just past the prompt and double as the generation grows; this ceiling
/// keeps the graph's masked attention from scanning unboundedly, with
/// longer generations falling back to eager decoding against per-step
/// scan cost.
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
        row_spans: Option<&[(usize, usize)]>,
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
        let output = self.attend(&q, &k, &v, attention_mask, row_spans)?;
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
        attention_mask: Option<&Tensor>,
        row_spans: Option<&[(usize, usize)]>,
    ) -> Result<Tensor, Error> {
        if let Some(spans) = row_spans
            && let Some(output) = self.attend_rows(q, k, v, spans)?
        {
            return Ok(output);
        }
        self.attend_masked(q, k, v, attention_mask)
    }

    fn attend_masked(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, Error> {
        let (batch, seq_len) = q
            .dims4()
            .map(|(batch, _, seq_len, _)| (batch, seq_len))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention shape", e))?;
        // Single-row sequences stay on flash, matching the eager reference
        // implementation's numerics exactly: FA2's accumulation order
        // differs from the gemm kernels, and swapping them mid-family
        // flips near-tie argmax decisions on format-heavy pages. The
        // captured decode graph is unaffected (it runs its own masked
        // kernel over the fixed-capacity storage).
        let flash = if batch == 1 {
            flash_attention(q, k, v, self.scaling, seq_len > 1)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "flash attention", e))?
        } else {
            None
        };
        match flash {
            Some(output) => Ok(output),
            None => {
                // Masked batch attention materializes (B, heads, q, kv)
                // scores eagerly; page-scale prefills chunk the query axis
                // so the transient stays bounded. Mask-less single-row
                // causal prefill chunks the same way — a 16K-token page
                // would otherwise materialize the full score matrix at
                // once — building each chunk's causal mask at the chunk's
                // query offset.
                const MASKED_ATTN_CHUNK: usize = 1024;
                let chunked = attention_mask.is_some() && seq_len > MASKED_ATTN_CHUNK
                    || (attention_mask.is_none() && batch == 1 && seq_len > MASKED_ATTN_CHUNK);
                if chunked {
                    return attention_masked_chunked(
                        q,
                        k,
                        v,
                        attention_mask,
                        self.scaling,
                        self.num_kv_groups,
                    );
                }
                attention_masked_single(q, k, v, attention_mask, self.scaling, self.num_kv_groups)
            }
        }
    }

    /// Left-padded batch prefill: every row attends only within its own
    /// `[start, start + len)` span, so run the same flash kernel the
    /// single-row prefill uses per row instead of materializing
    /// `(batch, heads, seq, seq)` scores under an additive mask. Pad
    /// positions receive zero attention output; nothing downstream reads
    /// them (their KV columns stay masked during decode).
    fn attend_rows(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        spans: &[(usize, usize)],
    ) -> Result<Option<Tensor>, Error> {
        let (batch, seq_len) = q
            .dims4()
            .map(|(batch, _, seq_len, _)| (batch, seq_len))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention shape", e))?;
        if spans.len() != batch {
            return Err(Error::Config {
                message: format!(
                    "{MODEL_NAME} row spans ({}) do not cover the batch ({batch})",
                    spans.len()
                ),
            });
        }
        let mut rows = Vec::with_capacity(batch);
        for (row, &(start, len)) in spans.iter().enumerate() {
            if len == 0 || start + len > seq_len {
                return Err(Error::Config {
                    message: format!(
                        "{MODEL_NAME} row span ({start}, {len}) outside the padded length {seq_len}"
                    ),
                });
            }
            let narrow_row = |t: &Tensor| -> Result<Tensor, Error> {
                t.narrow(0, row, 1)
                    .and_then(|t| t.narrow(2, start, len))
                    .and_then(|t| t.contiguous())
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "row attention slice", e))
            };
            let (q_row, k_row, v_row) = (narrow_row(q)?, narrow_row(k)?, narrow_row(v)?);
            // Flash unavailable (CPU or unsupported dtype): the caller's
            // padding mask takes over on the eager path.
            let Some(output) = flash_attention(&q_row, &k_row, &v_row, self.scaling, len > 1)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "row flash attention", e))?
            else {
                return Ok(None);
            };
            let padded = output
                .pad_with_zeros(2, start, seq_len - len - start)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "row attention padding", e))?;
            rows.push(padded);
        }
        let refs: Vec<&Tensor> = rows.iter().collect();
        let output = Tensor::cat(&refs, 0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "row attention output", e))?;
        Ok(Some(output))
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

    /// Number of KV heads, for sizing the batched storage template.
    #[cfg(feature = "cuda")]
    fn attention_num_kv_heads(&self) -> usize {
        self.num_kv_heads
    }

    /// Head width, for sizing the batched storage template.
    #[cfg(feature = "cuda")]
    fn attention_head_dim(&self) -> usize {
        self.head_dim
    }

    #[cfg(feature = "cuda")]
    fn prepare_dynamic_cache(&self, query_len: usize, cache_len: usize) -> Result<(), Error> {
        self.prepare_dynamic_cache_batch(1, query_len, self.num_kv_heads, self.head_dim, cache_len)
    }

    #[cfg(feature = "cuda")]
    fn prepare_dynamic_cache_batch(
        &self,
        batch: usize,
        query_len: usize,
        kv_heads: usize,
        head_dim: usize,
        cache_len: usize,
    ) -> Result<(), Error> {
        let template = Tensor::zeros(
            (batch, kv_heads, query_len, head_dim),
            self.q_proj.weight().dtype(),
            self.q_proj.weight().device(),
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "dynamic KV template", e))?;
        self.kv_cache
            .borrow_mut()
            .initialize_storage_with_capacity(&template, cache_len)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "initialize dynamic KV", e))
    }

    /// Free the fixed-capacity KV storage, restoring the organically
    /// grown eager cache form. Used when a graph capture fails: the fixed
    /// buckets were preallocated for a decode that will never run.
    #[cfg(feature = "cuda")]
    fn release_dynamic_cache_batch(&self) {
        let device = self.q_proj.weight().device().clone();
        if let Some((k, v)) = self.kv_cache.borrow_mut().take_fixed_storage() {
            drop_and_drain(k, &device);
            drop_and_drain(v, &device);
        }
    }

    /// Grow the fixed storage, preserving appended history. The template
    /// matches `prepare_dynamic_cache_batch`'s, so a capture at the new
    /// capacity afterwards sees the storage as already reusable instead of
    /// re-initializing it and discarding that history.
    #[cfg(feature = "cuda")]
    fn grow_dynamic_cache_batch(
        &self,
        batch: usize,
        query_len: usize,
        kv_heads: usize,
        head_dim: usize,
        cache_len: usize,
    ) -> Result<(), Error> {
        let template = Tensor::zeros(
            (batch, kv_heads, query_len, head_dim),
            self.q_proj.weight().dtype(),
            self.q_proj.weight().device(),
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "dynamic KV template", e))?;
        self.kv_cache
            .borrow_mut()
            .grow_fixed_storage(&template, cache_len)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "grow dynamic KV", e))
    }

    /// Batched CUDA-graph decode step: every row appends at its own
    /// device-side offset and attends over `[pad_row, start_row + query_len)`
    /// inside the shared fixed-capacity storage. `kv_positions` is a
    /// capture-time constant (its values depend only on the pinned
    /// `cache_len`); `row_starts` and `pad_bounds` are rewritten before
    /// every replay, so no input survives from a previous batch.
    #[cfg(feature = "cuda")]
    fn forward_dynamic_batch(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        row_starts: &Tensor,
        kv_positions: &Tensor,
        pad_bounds: &Tensor,
    ) -> Result<Tensor, Error> {
        let (batch, query_len, _) = hidden_states
            .dims3()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "batched dynamic input", e))?;
        let (q, k, v) = self.project_qkv(hidden_states, cos, sin)?;
        let cache = self.kv_cache.borrow();
        let cache_len = cache.storage_capacity();
        let (cache_k, cache_v) = cache.storage().ok_or_else(|| Error::Config {
            message: format!("{MODEL_NAME} batched dynamic KV storage is not initialized"),
        })?;
        drop(cache);
        let append = DynamicBatchKvAppend {
            query_len,
            batch,
            cache_len,
        };
        cache_k
            .inplace_op3(&k, row_starts, &append)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "batched key cache append", e))?;
        cache_v
            .inplace_op3(&v, row_starts, &append)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "batched value cache append", e))?;

        // Live span per row: [pad_row, start_row + query_len). Both bounds
        // are device tensors, so nothing uploads during replay.
        let ends = row_starts
            .reshape((batch, 1, 1))?
            .affine(1.0, query_len as f64)?
            .unsqueeze(2)?;
        let after_pad = kv_positions.broadcast_ge(pad_bounds)?;
        let before_end = kv_positions.broadcast_lt(&ends)?;
        let live = after_pad.broadcast_mul(&before_end)?;
        // masked_score stays finite in F16: 1e9 saturates to -inf there,
        // and 0 * inf - inf is exactly the all-NaN row this mask exists to
        // prevent.
        let fill = masked_score(hidden_states.dtype());
        let mask = live.to_dtype(hidden_states.dtype())?.affine(-fill, fill)?;
        let attn = scaled_dot_product_attention_gqa(
            &q,
            &cache_k,
            &cache_v,
            Some(&mask),
            self.scaling,
            false,
            self.num_kv_groups,
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "batched dynamic attention", e))?;
        self.project_output(&attn, batch, query_len)
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
        let fill = masked_score(hidden_states.dtype());
        let mask = live
            .to_dtype(hidden_states.dtype())?
            .affine(-fill, fill)?
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

/// Test-only injection point: after which layer's KV allocation a capture
/// should fail. Unset means only the post-allocation full injection fires.
#[cfg(all(test, feature = "cuda"))]
fn injected_capture_failure_after_layer(index: usize) -> bool {
    std::env::var_os("OAR_WEVISDOC_FAIL_CAPTURE_AFTER_LAYER")
        .and_then(|value| value.into_string().ok())
        .and_then(|value| value.parse::<usize>().ok())
        .is_some_and(|target| index + 1 == target)
}

/// Masked attention in one pass; `is_causal` follows the mask's absence so
/// mask-less callers get the kernel's causal flag.
fn attention_masked_single(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attention_mask: Option<&Tensor>,
    scaling: f64,
    num_kv_groups: usize,
) -> Result<Tensor, Error> {
    scaled_dot_product_attention_gqa(
        q,
        k,
        v,
        attention_mask,
        scaling,
        attention_mask.is_none(),
        num_kv_groups,
    )
    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "grouped-query attention", e))
}

/// Query-chunked masked attention. The caller's mask is narrowed along the
/// query axis per chunk; a mask-less single-row causal prefill builds each
/// chunk's causal mask at the chunk's query offset instead, so page-scale
/// sequences never materialize the full (heads, seq, seq) score matrix.
fn attention_masked_chunked(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    attention_mask: Option<&Tensor>,
    scaling: f64,
    num_kv_groups: usize,
) -> Result<Tensor, Error> {
    const MASKED_ATTN_CHUNK: usize = 1024;
    let (batch, seq_len) = q
        .dims4()
        .map(|(batch, _, seq_len, _)| (batch, seq_len))
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention shape", e))?;
    if attention_mask.is_none() && batch != 1 {
        return Err(Error::Config {
            message: format!(
                "{MODEL_NAME} causal chunked attention requires batch size 1, got {batch}"
            ),
        });
    }
    let mut chunks = Vec::with_capacity(seq_len.div_ceil(MASKED_ATTN_CHUNK));
    let mut start = 0usize;
    while start < seq_len {
        let len = (seq_len - start).min(MASKED_ATTN_CHUNK);
        let q_chunk = q.narrow(2, start, len)?;
        let mask_chunk = match attention_mask {
            Some(mask) => Some(mask.narrow(2, start, len)?),
            None => {
                // Causal single-row prefill: the query chunk may only see
                // its prefix, so narrow K/V to `start + len` — the dropped
                // columns sit behind the causal mask (-inf) and contribute
                // exactly zero, keeping the result bit-identical.
                let visible = start + len;
                Some(create_causal_mask(len, visible, q.dtype(), q.device())?)
            }
        };
        // The caller's mask already hides columns; keep the full K/V.
        // The causal path narrows K/V to the visible prefix — the dropped
        // columns sit behind the causal mask (-inf) and contribute exactly
        // zero, so the result stays bit-identical.
        let (k_chunk, v_chunk) = if attention_mask.is_some() {
            (k, v)
        } else {
            (&k.narrow(2, 0, start + len)?, &v.narrow(2, 0, start + len)?)
        };
        chunks.push(scaled_dot_product_attention_gqa(
            &q_chunk,
            k_chunk,
            v_chunk,
            mask_chunk.as_ref(),
            scaling,
            false,
            num_kv_groups,
        )?);
        start += len;
    }
    let refs: Vec<&Tensor> = chunks.iter().collect();
    Tensor::cat(&refs, 2).map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention chunks", e))
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
        row_spans: Option<&[(usize, usize)]>,
    ) -> Result<Tensor, Error> {
        let residual = hidden_states.clone();
        let normalized = self.input_layernorm.forward(hidden_states)?;
        let mixed = self
            .attention
            .forward(&normalized, cos, sin, attention_mask, row_spans)?;
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

    /// Number of KV heads, for sizing the batched storage template.
    #[cfg(feature = "cuda")]
    fn attention_num_kv_heads(&self) -> usize {
        self.attention.attention_num_kv_heads()
    }

    /// Head width, for sizing the batched storage template.
    #[cfg(feature = "cuda")]
    fn attention_head_dim(&self) -> usize {
        self.attention.attention_head_dim()
    }

    #[cfg(feature = "cuda")]
    fn prepare_dynamic_cache_batch(
        &self,
        batch: usize,
        query_len: usize,
        kv_heads: usize,
        head_dim: usize,
        cache_len: usize,
    ) -> Result<(), Error> {
        self.attention
            .prepare_dynamic_cache_batch(batch, query_len, kv_heads, head_dim, cache_len)
    }

    #[cfg(feature = "cuda")]
    fn grow_dynamic_cache_batch(
        &self,
        batch: usize,
        query_len: usize,
        kv_heads: usize,
        head_dim: usize,
        cache_len: usize,
    ) -> Result<(), Error> {
        self.attention
            .grow_dynamic_cache_batch(batch, query_len, kv_heads, head_dim, cache_len)
    }

    #[cfg(feature = "cuda")]
    fn release_dynamic_cache_batch(&self) {
        self.attention.release_dynamic_cache_batch()
    }

    #[cfg(feature = "cuda")]
    fn forward_dynamic_batch(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        row_starts: &Tensor,
        kv_positions: &Tensor,
        pad_bounds: &Tensor,
    ) -> Result<Tensor, Error> {
        let residual = hidden_states.clone();
        let normalized = self.input_layernorm.forward(hidden_states)?;
        let mixed = self.attention.forward_dynamic_batch(
            &normalized,
            cos,
            sin,
            row_starts,
            kv_positions,
            pad_bounds,
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
    #[cfg(feature = "cuda")]
    batch_decode_graph: RefCell<Option<BatchDecoderCudaGraph>>,
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
    /// Double the single-row decode bucket, preserving appended history.
    #[cfg(feature = "cuda")]
    fn grow_dynamic_cache(&self, query_len: usize, cache_len: usize) -> Result<(), Error> {
        self.grow_dynamic_cache_batch(1, query_len, cache_len)
    }

    /// Free graphs and fixed-capacity KV storage that cannot serve the
    /// incoming request: a single-row request cannot reuse a batch graph
    /// (and vice versa), and a batch request of a different width cannot
    /// reuse the captured batch graph. Their preallocated buckets would
    /// otherwise sit in memory through image preprocessing and vision
    /// encoding — the next prepare only replaces them after both — and
    /// competing allocations can OOM there. Compatible storage stays.
    #[cfg(feature = "cuda")]
    pub(crate) fn release_incompatible_fixed_storage(&self, request_batch: Option<usize>) {
        let incompatible = match request_batch {
            None => self.batch_decode_graph.borrow().is_some(),
            Some(width) => match self.batch_decode_graph.borrow().as_ref() {
                Some(graph) => graph.batch != width,
                None => self.decode_graph.borrow().is_some(),
            },
        };
        if incompatible {
            self.recover_failed_capture();
        }
    }

    /// Tear down everything a failed capture left behind — both graphs
    /// (never installed, but cheap to drop) and every layer's
    /// preallocated fixed-capacity KV storage — so the eager fallback runs
    /// with the memory it needs.
    #[cfg(feature = "cuda")]
    pub(crate) fn recover_failed_capture(&self) {
        self.invalidate_cuda_graph();
        self.invalidate_batch_cuda_graph();
        #[cfg(test)]
        {
            let output = std::process::Command::new("nvidia-smi")
                .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
                .output();
            if let Ok(out) = output {
                eprintln!(
                    "DBGM2 at-failure used={}MiB",
                    String::from_utf8_lossy(&out.stdout).trim()
                );
            }
        }
        #[cfg(test)]
        if std::env::var_os("OAR_WEVISDOC_SKIP_RELEASE").is_some() {
            // Test-only control: skip the storage release so the memory
            // assertions can prove they catch a missing release.
            return;
        }
        self.release_dynamic_caches();
    }

    /// Free every layer's fixed-capacity KV storage after a failed graph
    /// capture, before the eager fallback runs: the fallback exists for
    /// low-memory situations, and the preallocated buckets would only
    /// starve it.
    #[cfg(feature = "cuda")]
    fn release_dynamic_caches(&self) {
        for layer in &self.layers {
            layer.release_dynamic_cache_batch();
        }
        drain_cuda_context_errors(self.embed_tokens.embeddings().device());
    }

    /// Double the batched decode bucket, preserving appended history.
    #[cfg(feature = "cuda")]
    fn grow_dynamic_cache_batch(
        &self,
        batch: usize,
        query_len: usize,
        cache_len: usize,
    ) -> Result<(), Error> {
        let kv_heads = self
            .layers
            .first()
            .map(|layer| layer.attention_num_kv_heads())
            .unwrap_or(1);
        let head_dim = self
            .layers
            .first()
            .map(|layer| layer.attention_head_dim())
            .unwrap_or(1);
        for layer in &self.layers {
            layer.grow_dynamic_cache_batch(batch, query_len, kv_heads, head_dim, cache_len)?;
        }
        Ok(())
    }

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
            #[cfg(feature = "cuda")]
            batch_decode_graph: RefCell::new(None),
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
        row_spans: Option<&[(usize, usize)]>,
    ) -> Result<Tensor, Error> {
        let (cos, sin) = self
            .rotary_emb
            .forward(position_ids, inputs_embeds.dtype())?;
        let mut hidden_states = inputs_embeds.clone();
        for (layer_index, layer) in self.layers.iter().enumerate() {
            hidden_states = layer.forward(&hidden_states, &cos, &sin, attention_mask, row_spans)?;
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

    /// Batched graph decode step: rotary comes from `position_ids` in-graph,
    /// per-row offsets from `row_starts`.
    #[cfg(feature = "cuda")]
    fn forward_dynamic_batch(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        row_starts: &Tensor,
        kv_positions: &Tensor,
        pad_bounds: &Tensor,
    ) -> Result<Tensor, Error> {
        let (cos, sin) = self
            .rotary_emb
            .forward(position_ids, inputs_embeds.dtype())?;
        let mut hidden_states = inputs_embeds.clone();
        for layer in &self.layers {
            hidden_states = layer.forward_dynamic_batch(
                &hidden_states,
                &cos,
                &sin,
                row_starts,
                kv_positions,
                pad_bounds,
            )?;
        }
        self.norm
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "final norm", e))
    }

    /// Capture the batched decode graph for one batch width. The layout
    /// change replaces the per-layer KV storage, so any single-row graph is
    /// invalidated first. Region decoding (`ladder`) starts the bucket just
    /// past the prompt and grows it on demand; page decoding pins the
    /// legacy declared-maximum bucket, reproducing the reference numerics.
    #[cfg(feature = "cuda")]
    pub(crate) fn prepare_batch_ar_cuda_graph(
        &self,
        batch: usize,
        prompt_len: usize,
        max_new_tokens: usize,
        pad_lens: &[usize],
        lm_head: &Linear,
        ladder: bool,
    ) -> Result<(), Error> {
        if std::env::var_os("OAR_VL_DISABLE_CUDA_GRAPH").is_some()
            || std::env::var_os("OAR_WEVISDOC_DISABLE_CUDA_GRAPH").is_some()
        {
            self.invalidate_cuda_graph();
            self.invalidate_batch_cuda_graph();
            return Ok(());
        }
        if self.embed_tokens.embeddings().device().is_cuda()
            && matches!(
                self.embed_tokens.embeddings().dtype(),
                DType::BF16 | DType::F16
            )
        {
            let Some(cache_len) = (if ladder {
                prompt_decode_bucket(prompt_len, WEVISDOC_DECODE_CACHE_LEN)
            } else {
                decoder_cache_capacity(prompt_len, max_new_tokens, WEVISDOC_DECODE_CACHE_LEN)
            }) else {
                // The following batch prefill reinitializes the shared KV
                // storage, so the single-row graph must go with the batch
                // graph — same reasoning as the single-row fallback below.
                self.invalidate_cuda_graph();
                self.invalidate_batch_cuda_graph();
                return Ok(());
            };
            // Reuse is sound only because every batch-dependent graph input
            // is rewritten before replay: hidden states, positions, row
            // starts, and the pad bounds behind the attention mask. What
            // stays baked in is a function of (batch, cache_len) alone.
            // The bucket must match exactly: a wider graph would scan past
            // the padding mask for the whole generation, which is the cost
            // the ladder exists to avoid.
            let reusable = self
                .batch_decode_graph
                .borrow()
                .as_ref()
                .is_some_and(|graph| graph.batch == batch && graph.cache_len == cache_len);
            if reusable {
                return Ok(());
            }
            self.invalidate_cuda_graph();
            self.invalidate_batch_cuda_graph();
            if let Err(error) = self.capture_batch_cuda_graph(
                batch,
                cache_len,
                WEVISDOC_DECODE_CACHE_LEN,
                pad_lens,
                lm_head,
                0,
            ) {
                tracing::warn!(
                    "{MODEL_NAME} batch graph capture failed: {error}; continuing eager"
                );
                self.recover_failed_capture();
            }
        }
        let _ = (prompt_len, max_new_tokens, pad_lens, lm_head, ladder);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn capture_batch_cuda_graph(
        &self,
        batch: usize,
        cache_len: usize,
        ceiling: usize,
        pad_lens: &[usize],
        lm_head: &Linear,
        append_slot: usize,
    ) -> Result<(), Error> {
        use candle_core::cuda_backend::cudarc::driver::sys::{
            CUgraphInstantiate_flags_enum, CUstreamCaptureMode_enum,
        };

        if self.batch_decode_graph.borrow().is_some() {
            return Ok(());
        }
        if pad_lens.len() != batch {
            return Err(Error::Config {
                message: format!(
                    "{MODEL_NAME} batch graph needs {batch} padding lengths, got {}",
                    pad_lens.len()
                ),
            });
        }
        let Device::Cuda(cuda) = self.embed_tokens.embeddings().device() else {
            return Ok(());
        };
        if append_slot >= cache_len {
            return Err(Error::Config {
                message: format!(
                    "{MODEL_NAME} batch graph append slot {append_slot} outside bucket {cache_len}"
                ),
            });
        }
        let query_len = 1;
        let kv_heads = self
            .layers
            .first()
            .map(|layer| layer.attention_num_kv_heads())
            .unwrap_or(1);
        let head_dim = self
            .layers
            .first()
            .map(|layer| layer.attention_head_dim())
            .unwrap_or(1);
        for (index, layer) in self.layers.iter().enumerate() {
            layer.prepare_dynamic_cache_batch(batch, query_len, kv_heads, head_dim, cache_len)?;
            #[cfg(test)]
            if injected_capture_failure_after_layer(index) {
                return Err(Error::Config {
                    message: "injected capture failure (test)".to_string(),
                });
            }
        }
        #[cfg(test)]
        if std::env::var_os("OAR_WEVISDOC_FAIL_CAPTURE").is_some() {
            return Err(Error::Config {
                message: "injected capture failure (test)".to_string(),
            });
        }
        let hidden_size = self
            .embed_tokens
            .embeddings()
            .dim(1)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "batch graph hidden size", e))?;
        let device = self.embed_tokens.embeddings().device().clone();
        let hidden_input = Tensor::zeros(
            (batch, query_len, hidden_size),
            self.embed_tokens.embeddings().dtype(),
            &device,
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "batch graph hidden input", e))?;
        let position_input = Tensor::zeros((3, batch, query_len), DType::I64, &device)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "batch graph position input", e))?;
        let row_starts = CudaGraphPerRowU32::new(&[batch], &device)
            .map_err(|e| cuda_graph_error(MODEL_NAME, "batch graph row starts", e))?;
        // Warm and captured runs append at `append_slot`: 0 for a fresh
        // capture, the live history length when a grown bucket is
        // re-captured mid-generation, so the warmup never overwrites
        // preserved KV entries.
        row_starts
            .update(&vec![append_slot as u32; batch])
            .map_err(|e| cuda_graph_error(MODEL_NAME, "seed batch graph row starts", e))?;
        let kv_positions =
            Tensor::arange(0u32, cache_len as u32, &device)?.reshape((1, 1, 1, cache_len))?;
        // Padded-span bounds live in a pinned-backed device buffer so every
        // batch rewrites them before replay; a capture-time constant here is
        // what let a reused graph mask with the previous batch's pads.
        let pad_bounds = CudaGraphPerRowU32::new(&[batch, 1, 1, 1], &device)
            .map_err(|e| cuda_graph_error(MODEL_NAME, "batch graph pad bounds", e))?;
        let pads: Vec<u32> = pad_lens
            .iter()
            .map(|&pad| {
                u32::try_from(pad).map_err(|_| Error::Config {
                    message: format!("{MODEL_NAME} batch graph pad {pad} exceeds u32"),
                })
            })
            .collect::<Result<Vec<u32>, Error>>()?;
        pad_bounds
            .update(&pads)
            .map_err(|e| cuda_graph_error(MODEL_NAME, "seed batch graph pad bounds", e))?;
        let stream = cuda.cuda_stream();
        let _htod_cache = cuda.enable_cuda_graph_htod_cache();

        let warm = self.forward_dynamic_batch(
            &hidden_input,
            &position_input,
            row_starts.tensor(),
            &kv_positions,
            pad_bounds.tensor(),
        )?;
        let warm_logits = self.project_logits_batch(&warm, lm_head)?;
        sync_graph_tensor(MODEL_NAME, &warm_logits, "warm batch decoder CUDA graph")?;
        let logits_output = Tensor::zeros_like(&warm_logits)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "batch graph logits output", e))?;
        logits_output
            .slice_set(&warm_logits, 0, 0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "prime batch logits copy", e))?;

        stream
            .begin_capture(CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_GLOBAL)
            .map_err(|e| cuda_graph_error(MODEL_NAME, "begin batch decoder capture", e))?;
        let captured_output: Result<(), Error> = (|| {
            let hidden = self.forward_dynamic_batch(
                &hidden_input,
                &position_input,
                row_starts.tensor(),
                &kv_positions,
                pad_bounds.tensor(),
            )?;
            let logits = self.project_logits_batch(&hidden, lm_head)?;
            logits_output
                .slice_set(&logits, 0, 0)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "record batch logits copy", e))
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
            .map_err(|e| cuda_graph_error(MODEL_NAME, "end batch decoder capture", e))?
            .ok_or_else(|| Error::Config {
                message: format!("{MODEL_NAME} batch decoder capture returned no graph"),
            })?;
        graph
            .launch()
            .map_err(|e| cuda_graph_error(MODEL_NAME, "warm batch decoder graph", e))?;
        sync_graph_tensor(MODEL_NAME, &logits_output, "sync batch decoder graph")?;
        self.clear_cache();
        *self.batch_decode_graph.borrow_mut() = Some(BatchDecoderCudaGraph {
            graph,
            hidden_input,
            position_input,
            row_starts,
            pad_bounds,
            logits_output,
            retained_inputs: vec![kv_positions],
            batch,
            cache_len,
            ceiling,
        });
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn replay_batch_cuda_graph(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        rows: BatchDecodeRows<'_>,
        max_kv_len: usize,
        lm_head: &Linear,
    ) -> Result<Option<Tensor>, Error> {
        if self.batch_decode_graph.borrow().is_none() {
            return Ok(None);
        }
        let overflow = {
            let captured_ref = self.batch_decode_graph.borrow();
            captured_ref
                .as_ref()
                .is_some_and(|captured| max_kv_len > captured.cache_len)
        };
        if overflow {
            let (batch, cache_len, ceiling) = {
                let captured_ref = self.batch_decode_graph.borrow();
                let captured = captured_ref.as_ref().expect("overflow implies Some");
                (captured.batch, captured.cache_len, captured.ceiling)
            };
            let Some(next) = next_decode_bucket(cache_len, ceiling) else {
                // Ladder ceiling: the rest of this generation decodes eager,
                // whose append path grows the storage organically.
                self.invalidate_cuda_graph();
                self.invalidate_batch_cuda_graph();
                return Ok(None);
            };
            // Grow the fixed bucket and re-capture. Appended history is
            // preserved, and the re-captured graph warms up appending at
            // the live end of the sequence (`max_kv_len - 1`), so the
            // warmup cannot overwrite real KV entries. Graphs are disposed
            // before the storage move: both hold pointers into it.
            self.invalidate_cuda_graph();
            self.invalidate_batch_cuda_graph();
            let pads: Vec<usize> = rows.pad_lens.iter().map(|&pad| pad as usize).collect();
            self.grow_dynamic_cache_batch(batch, 1, next)?;
            if let Err(error) =
                self.capture_batch_cuda_graph(batch, next, ceiling, &pads, lm_head, max_kv_len - 1)
            {
                tracing::warn!(
                    "{MODEL_NAME} batch graph re-capture at bucket {next} failed: {error}; continuing eager"
                );
                self.invalidate_cuda_graph();
                self.invalidate_batch_cuda_graph();
                return Ok(None);
            }
        }
        let captured_ref = self.batch_decode_graph.borrow();
        let Some(captured) = captured_ref.as_ref() else {
            return Ok(None);
        };
        if inputs_embeds.shape() != captured.hidden_input.shape()
            || position_ids.shape() != captured.position_input.shape()
        {
            return Ok(None);
        }
        captured
            .hidden_input
            .slice_set(inputs_embeds, 0, 0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "copy batch graph hidden", e))?;
        captured
            .position_input
            .slice_set(position_ids, 0, 0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "copy batch graph positions", e))?;
        captured
            .row_starts
            .update(rows.row_starts)
            .map_err(|e| cuda_graph_error(MODEL_NAME, "update batch graph row starts", e))?;
        captured
            .pad_bounds
            .update(rows.pad_lens)
            .map_err(|e| cuda_graph_error(MODEL_NAME, "update batch graph pad bounds", e))?;
        captured
            .graph
            .launch()
            .map_err(|e| cuda_graph_error(MODEL_NAME, "launch batch decoder graph", e))?;
        for layer in &self.layers {
            layer.set_kv_cache_len(max_kv_len)?;
        }
        Ok(Some(captured.logits_output.clone()))
    }

    /// One batched decode step at `position_ids`; returns `(batch, vocab)`
    /// logits. `rows` carries the batch's per-row write offsets and
    /// left-padding lengths — the graph mask is refreshed from them before
    /// replay. Falls back to the eager masked path when no graph fits.
    #[cfg(feature = "cuda")]
    pub(crate) fn forward_decode_logits_batch(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        rows: BatchDecodeRows<'_>,
        max_kv_len: usize,
        attention_mask: Option<&Tensor>,
        lm_head: &Linear,
    ) -> Result<Tensor, Error> {
        // The captured graph masks padding internally, so replay takes
        // priority; the caller's mask only serves the eager fallback.
        if let Some(logits) =
            self.replay_batch_cuda_graph(inputs_embeds, position_ids, rows, max_kv_len, lm_head)?
        {
            return Ok(logits);
        }
        let hidden = self.forward(inputs_embeds, position_ids, None, attention_mask, None)?;
        self.project_logits_batch(&hidden, lm_head)
    }

    #[cfg(feature = "cuda")]
    fn project_logits_batch(
        &self,
        hidden_states: &Tensor,
        lm_head: &Linear,
    ) -> Result<Tensor, Error> {
        lm_head
            .forward(hidden_states)
            .and_then(|logits| logits.squeeze(1))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "batch decode LM head", e))
    }

    #[cfg(feature = "cuda")]
    fn invalidate_batch_cuda_graph(&self) {
        if let Some(graph) = self.batch_decode_graph.borrow_mut().take() {
            graph.dispose();
        }
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
            if let Some(logits) =
                self.replay_cuda_graph(inputs_embeds, position_ids, kv_len, lm_head)?
            {
                return Ok(logits);
            }
        }
        let hidden = self.forward(inputs_embeds, position_ids, None, attention_mask, None)?;
        self.project_logits(&hidden, lm_head)
    }

    /// Capture the batch-1 single-token decode graph when eligible.
    /// Region decoding (`ladder`) starts the bucket just past the prompt
    /// and grows it on demand; page decoding pins the legacy
    /// declared-maximum bucket, reproducing the reference numerics.
    pub(crate) fn prepare_ar_cuda_graph(
        &self,
        prompt_len: usize,
        max_new_tokens: usize,
        lm_head: &Linear,
        ladder: bool,
    ) -> Result<(), Error> {
        if std::env::var_os("OAR_VL_DISABLE_CUDA_GRAPH").is_some()
            || std::env::var_os("OAR_WEVISDOC_DISABLE_CUDA_GRAPH").is_some()
        {
            // The gate removes every graph, not just this path's: a live
            // batch graph would keep replaying against KV storage the
            // single-row layout no longer matches.
            #[cfg(feature = "cuda")]
            {
                self.invalidate_cuda_graph();
                self.invalidate_batch_cuda_graph();
            }
            return Ok(());
        }
        #[cfg(feature = "cuda")]
        if self.embed_tokens.embeddings().device().is_cuda()
            && matches!(
                self.embed_tokens.embeddings().dtype(),
                DType::BF16 | DType::F16
            )
        {
            let Some(cache_len) = (if ladder {
                prompt_decode_bucket(prompt_len, WEVISDOC_DECODE_CACHE_LEN)
            } else {
                decoder_cache_capacity(prompt_len, max_new_tokens, WEVISDOC_DECODE_CACHE_LEN)
            }) else {
                // Eager fallback: the prompt alone does not fit the largest
                // bucket, so no graph may stay alive over it.
                self.invalidate_cuda_graph();
                self.invalidate_batch_cuda_graph();
                return Ok(());
            };
            // Reuse within one doubling: region sweeps visit nearby prompt
            // sizes, and re-capturing per region costs more than the wider
            // scan. Anything further out re-captures, keeping the scan
            // proportional to the prompt.
            let reusable = self.decode_graph.borrow().as_ref().is_some_and(|graph| {
                graph.cache_len >= cache_len && graph.cache_len < cache_len * 2
            });
            if reusable {
                return Ok(());
            }
            self.invalidate_cuda_graph();
            self.invalidate_batch_cuda_graph();
            if let Err(error) =
                self.capture_cuda_graph(cache_len, WEVISDOC_DECODE_CACHE_LEN, lm_head, 0)
            {
                // The graph is only an optimization: a failed capture (a KV
                // preallocation that outgrew free memory, say) continues
                // eager right here, so no caller can forget the fallback.
                tracing::warn!(
                    "{MODEL_NAME} decoder graph capture failed: {error}; continuing eager"
                );
                self.recover_failed_capture();
            }
        }
        let _ = (prompt_len, max_new_tokens, lm_head, ladder);
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn capture_cuda_graph(
        &self,
        cache_len: usize,
        ceiling: usize,
        lm_head: &Linear,
        append_slot: usize,
    ) -> Result<(), Error> {
        use candle_core::cuda_backend::cudarc::driver::sys::{
            CUgraphInstantiate_flags_enum, CUstreamCaptureMode_enum,
        };

        if self.decode_graph.borrow().is_some() {
            return Ok(());
        }
        let Device::Cuda(cuda) = self.embed_tokens.embeddings().device() else {
            return Ok(());
        };
        if append_slot >= cache_len {
            return Err(Error::Config {
                message: format!(
                    "{MODEL_NAME} decoder graph append slot {append_slot} outside bucket {cache_len}"
                ),
            });
        }
        let query_len = 1;
        for (index, layer) in self.layers.iter().enumerate() {
            layer.prepare_dynamic_cache(query_len, cache_len)?;
            #[cfg(test)]
            if injected_capture_failure_after_layer(index) {
                // Fires after this layer's KV allocation: the fallback must
                // release a partially allocated set of fixed buckets.
                return Err(Error::Config {
                    message: "injected capture failure (test)".to_string(),
                });
            }
        }
        #[cfg(test)]
        if std::env::var_os("OAR_WEVISDOC_FAIL_CAPTURE").is_some() {
            return Err(Error::Config {
                message: "injected capture failure (test)".to_string(),
            });
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
        // Warm and captured runs append after the live history: slot
        // `append_slot` (0 for a fresh capture, the preserved length when a
        // grown bucket is re-captured mid-generation), so the warmup never
        // overwrites real KV entries. The append kernel derives the slot
        // from the cumulative END, hence `append_slot + query_len`.
        let query_lengths = Tensor::new(&[0u32, query_len as u32], &device)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "graph query lengths", e))?;
        let kv_lengths = CudaGraphKvLengths::new(append_slot + query_len, &device)
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
            retained_inputs: vec![kv_positions],
            cache_len,
            ceiling,
        });
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn replay_cuda_graph(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        kv_len: usize,
        lm_head: &Linear,
    ) -> Result<Option<Tensor>, Error> {
        if self.decode_graph.borrow().is_none() {
            return Ok(None);
        }
        let overflow = {
            let captured_ref = self.decode_graph.borrow();
            captured_ref
                .as_ref()
                .is_some_and(|captured| kv_len > captured.cache_len)
        };
        if overflow {
            let (cache_len, ceiling) = {
                let captured_ref = self.decode_graph.borrow();
                captured_ref
                    .as_ref()
                    .map(|captured| (captured.cache_len, captured.ceiling))
                    .expect("overflow implies Some")
            };
            let Some(next) = next_decode_bucket(cache_len, ceiling) else {
                // Ladder ceiling: the rest of this generation decodes eager,
                // whose append path grows the storage organically.
                self.invalidate_cuda_graph();
                self.invalidate_batch_cuda_graph();
                return Ok(None);
            };
            // Grow the fixed bucket and re-capture; appended history is
            // preserved and the warmup appends at the live end of the
            // sequence. Graphs are disposed before the storage move.
            self.invalidate_cuda_graph();
            self.invalidate_batch_cuda_graph();
            self.grow_dynamic_cache(1, next)?;
            if let Err(error) = self.capture_cuda_graph(next, ceiling, lm_head, kv_len - 1) {
                tracing::warn!(
                    "{MODEL_NAME} decoder graph re-capture at bucket {next} failed: {error}; continuing eager"
                );
                self.invalidate_cuda_graph();
                self.invalidate_batch_cuda_graph();
                return Ok(None);
            }
        }
        let captured_ref = self.decode_graph.borrow();
        let Some(captured) = captured_ref.as_ref() else {
            return Ok(None);
        };
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

    /// Whether the batched decode graph is currently captured — asserted
    /// by the GPU self-checks.
    #[cfg(all(test, feature = "cuda"))]
    pub(crate) fn batch_decode_graph_captured(&self) -> bool {
        self.batch_decode_graph.borrow().is_some()
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
        // graph-bound buffers to the allocator and poisons it. Both the
        // single-row and the batched graph may be alive at the end.
        self.invalidate_cuda_graph();
        self.invalidate_batch_cuda_graph();
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
        let row_spans = [(max_len - seq_a, seq_a), (max_len - seq_b, seq_b)];
        let batched = model.forward(
            &batch_embeds,
            &batch_positions,
            Some(&deepstack),
            Some(&mask),
            Some(&row_spans),
        )?;
        let batched_logits = lm_head.forward(&batched.i((.., max_len - 1, ..))?.contiguous()?)?;

        for (row, (embeds, positions, image_start)) in
            [(&embeds_a, &pos_a, 4usize), (&embeds_b, &pos_b, 3usize)]
                .into_iter()
                .enumerate()
        {
            let single_deepstack = single_deepstack(image_start, 0);
            model.clear_cache();
            let single = model.forward(embeds, positions, Some(&single_deepstack), None, None)?;
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
            let mut cfg = valid_tiny_config();
            // Real-model attention dims: the tiny shapes take different
            // kernels than production decode.
            cfg.hidden_size = 2048;
            cfg.intermediate_size = 6144;
            cfg.num_attention_heads = 16;
            cfg.num_key_value_heads = 8;
            cfg.head_dim = 128;
            cfg.num_hidden_layers = 28;
            cfg.vocab_size = 32768;
            // The decode graph is bf16/f16-gated: build in bf16 so the
            // test really exercises capture and replay. Both instances
            // share the weight tensors, so the audit scenarios stay
            // within one allocation of model memory.
            let tensors = random_var_map(&cfg, &device, DType::BF16);
            let make_vb = || VarBuilder::from_tensors(tensors.clone(), DType::BF16, &device);
            let model = Qwen3VlTextModel::load(&cfg, make_vb().pp("model")).unwrap();
            let lm_head = Linear::new(
                make_vb()
                    .get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")
                    .unwrap(),
                None,
            );

            // A prompt over 4096 tokens forces the 8192 capacity bucket —
            // the size class the real model decodes from. The short prompt
            // lands in the 1024 bucket.
            let long: Vec<u32> = (0..4200).map(|i| 10 + i % 60).collect();
            let short: Vec<u32> = (0..600).map(|i| 10 + i % 60).collect();

            // First capture at the small bucket.
            assert_eq!(
                greedy_eager(&model, &lm_head, &short, 8),
                greedy_graphed(&model, &lm_head, &short, 8, true),
                "small-bucket graph decode must match eager"
            );
            assert!(model.decode_graph_captured());

            // Growing past the captured capacity forces a re-capture in the
            // same process — the dangling-read bug reproduced here.
            assert_eq!(
                greedy_eager(&model, &lm_head, &long, 8),
                greedy_graphed(&model, &lm_head, &long, 8, true),
                "re-captured graph decode must match eager"
            );

            // The larger graph now covers the small prompt again: the
            // ladder shrinks the bucket back to the prompt's own.
            assert_eq!(
                greedy_eager(&model, &lm_head, &short, 8),
                greedy_graphed(&model, &lm_head, &short, 8, true),
                "shrunk-bucket graph decode must match eager"
            );

            // A generation that crosses the captured bucket mid-decode:
            // the ladder doubles it, preserves the KV history, and decode
            // must still match eager token for token. Prompt 100 tokens ->
            // 128 bucket; 130 steps run to KV 230, growing 128 -> 256.
            let medium: Vec<u32> = (0..100).map(|i| 10 + i % 60).collect();
            assert_eq!(
                greedy_eager(&model, &lm_head, &medium, 130),
                greedy_graphed(&model, &lm_head, &medium, 130, true),
                "ladder-grown graph decode must match eager"
            );

            // A second instance capturing while the first graph is alive —
            // the other reproduction of the dangling-read bug.
            let second = Qwen3VlTextModel::load(&cfg, make_vb().pp("model")).unwrap();
            assert_eq!(
                greedy_eager(&second, &lm_head, &long, 8),
                greedy_graphed(&second, &lm_head, &long, 8, true),
                "second-instance graph decode must match eager"
            );
        }
        #[cfg(not(feature = "cuda"))]
        eprintln!("skipping: built without the cuda feature");
    }

    /// Batched decode graph vs the eager masked path: two left-padded rows
    /// of different lengths, checked token-for-token. Opt in with
    /// `OAR_WEVISDOC_GPU_SELFTEST=1`.
    /// Chunked causal attention must agree with the one-pass form to
    /// float epsilon: only the softmax reduction grouping changes.
    #[test]
    fn causal_chunked_attention_matches_single_pass() {
        let device = Device::Cpu;
        let (heads, seq, head_dim, kv_heads) = (4usize, 2560usize, 32usize, 2usize);
        let q = Tensor::randn(0f32, 1f32, (1, heads, seq, head_dim), &device).unwrap();
        let k = Tensor::randn(0f32, 1f32, (1, kv_heads, seq, head_dim), &device).unwrap();
        let v = Tensor::randn(0f32, 1f32, (1, kv_heads, seq, head_dim), &device).unwrap();
        let scaling = 1.0 / (head_dim as f64).sqrt();
        let single = attention_masked_single(&q, &k, &v, None, scaling, 2).unwrap();
        let chunked = attention_masked_chunked(&q, &k, &v, None, scaling, 2).unwrap();
        let a = single.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let b = chunked.flatten_all().unwrap().to_vec1::<f32>().unwrap();
        let worst = a
            .iter()
            .zip(b.iter())
            .map(|(x, y)| (x - y).abs())
            .fold(0.0f32, f32::max);
        eprintln!("causal chunk vs single max|delta| = {worst:e}");
        assert!(
            worst < 1e-3,
            "chunked causal attention diverged: max|delta| = {worst}"
        );
    }

    // The CUDA async allocator hides pool-internal reuse from nvidia-smi;
    // trim the pool to its live allocations before every reading or the
    // phases never move the needle.
    #[cfg(all(test, feature = "cuda"))]
    fn trim_pool(model: &Qwen3VlTextModel) {
        let Device::Cuda(cuda) = model.embed_tokens.embeddings().device() else {
            return;
        };
        cuda.cuda_stream().synchronize().unwrap();
        let ordinal = cuda.cuda_stream().context().ordinal();
        let mut pool: candle_core::cuda_backend::cudarc::driver::sys::CUmemoryPool =
            std::ptr::null_mut();
        unsafe {
            use candle_core::cuda_backend::cudarc::driver::sys;
            sys::cuDeviceGetDefaultMemPool(&mut pool, ordinal as i32);
            sys::cuMemPoolTrimTo(pool, 0);
        }
    }

    #[cfg(all(test, feature = "cuda"))]
    fn smi_used() -> u64 {
        let output = std::process::Command::new("nvidia-smi")
            .args(["--query-gpu=memory.used", "--format=csv,noheader,nounits"])
            .output();
        match output {
            Ok(out) => String::from_utf8_lossy(&out.stdout)
                .trim()
                .lines()
                .next()
                .and_then(|line| line.trim().parse().ok())
                .unwrap_or(0),
            Err(_) => 0,
        }
    }

    #[cfg(all(test, feature = "cuda"))]
    fn measured(model: &Qwen3VlTextModel) -> u64 {
        let Device::Cuda(cuda) = model.embed_tokens.embeddings().device() else {
            return 0;
        };
        cuda.cuda_stream().synchronize().unwrap();
        trim_pool(model);
        smi_used()
    }

    /// An injected capture failure — fired after the fixed KV buckets are
    /// already allocated — must be absorbed by the production prepare call
    /// itself: no error may escape, both graphs stay uncaptured, the
    /// preallocated storage is freed, and eager decoding continues and
    /// matches eager output. A control with the storage release skipped
    /// proves the memory assertion can catch a missing release.
    #[test]
    fn cuda_capture_failure_falls_back_to_eager() {
        #[cfg(feature = "cuda")]
        {
            if std::env::var_os("OAR_WEVISDOC_GPU_SELFTEST").is_none() {
                eprintln!("skipping: OAR_WEVISDOC_GPU_SELFTEST is not set");
                return;
            }
            let Ok(device) = Device::new_cuda(0) else {
                eprintln!("skipping: no CUDA device");
                return;
            };
            use crate::runtime::attention::{
                combine_masks, create_causal_mask, create_generation_mask_if_needed,
                create_left_padding_mask,
            };
            // Production-shape depth: 28 layers at the 8192 bucket hold
            // ~940 MiB of fixed KV, far past the pool's slack.
            let mut cfg = valid_tiny_config();
            cfg.hidden_size = 2048;
            cfg.intermediate_size = 6144;
            cfg.num_attention_heads = 16;
            cfg.num_key_value_heads = 8;
            cfg.head_dim = 128;
            cfg.num_hidden_layers = 28;
            cfg.vocab_size = 32768;
            let tensors = random_var_map(&cfg, &device, DType::BF16);
            let vb = VarBuilder::from_tensors(tensors, DType::BF16, &device);
            let model = Qwen3VlTextModel::load(&cfg, vb.pp("model")).unwrap();
            let lm_head = Linear::new(
                vb.get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")
                    .unwrap(),
                None,
            );
            let ids = (0..600).map(|i| 10 + i % 60).collect::<Vec<u32>>();

            model.clear_cache();
            let _ = model
                .forward(
                    &Tensor::zeros((1, 8, cfg.hidden_size), DType::BF16, &device).unwrap(),
                    &Tensor::zeros((3, 1, 8), DType::I64, &device).unwrap(),
                    None,
                    None,
                    Some(&[(0usize, 8)]),
                )
                .unwrap();
            let baseline = measured(&model);
            eprintln!("DBGM1 baseline={baseline}MiB");
            assert!(baseline > 0, "nvidia-smi unavailable; cannot measure");

            // SAFETY: the self-test runner is single-threaded, so the
            // process-global environment is owned by this test.
            unsafe { std::env::set_var("OAR_WEVISDOC_FAIL_CAPTURE", "1") };

            // Production single-row entry: succeeds even though the capture
            // failed; no graph may survive.
            model
                .prepare_ar_cuda_graph(600, 8192, &lm_head, false)
                .unwrap();
            let after_single = measured(&model);
            eprintln!("DBGM1 after-single-fallback={after_single}MiB");
            assert!(
                after_single.saturating_sub(baseline) <= 16,
                "single-row fallback leaked {} MiB",
                after_single.saturating_sub(baseline)
            );
            assert!(!model.decode_graph_captured());

            // Partial allocation: fail after the 24th of 28 layers.
            unsafe {
                std::env::set_var("OAR_WEVISDOC_FAIL_CAPTURE_AFTER_LAYER", "24");
            }
            model
                .prepare_ar_cuda_graph(600, 8192, &lm_head, false)
                .unwrap();
            let after_partial = measured(&model);
            eprintln!("DBGM1 after-partial-fallback={after_partial}MiB");
            assert!(after_partial.saturating_sub(baseline) <= 16);
            unsafe {
                std::env::remove_var("OAR_WEVISDOC_FAIL_CAPTURE_AFTER_LAYER");
            }

            // Production batch entry with all layers allocated.
            model
                .prepare_batch_ar_cuda_graph(2, 600, 8192, &[0, 10], &lm_head, false)
                .unwrap();
            let after_batch = measured(&model);
            eprintln!("DBGM1 after-batch-fallback={after_batch}MiB");
            assert!(after_batch.saturating_sub(baseline) <= 16);
            assert!(!model.batch_decode_graph_captured());
            assert!(!model.decode_graph_captured());

            // Control: with the storage release skipped, the preallocated
            // buckets stay resident — proving the assertion above catches
            // a missing release. (The DBGM2 line printed by the recovery
            // is the at-failure reading, taken before the release runs.)
            unsafe { std::env::set_var("OAR_WEVISDOC_SKIP_RELEASE", "1") };
            model
                .prepare_batch_ar_cuda_graph(2, 600, 8192, &[0, 10], &lm_head, false)
                .unwrap();
            let no_release = measured(&model);
            eprintln!("DBGM1 control (release skipped)={no_release}MiB");
            assert!(
                no_release.saturating_sub(baseline) >= 300,
                "the memory assertion failed to catch a missing release"
            );
            unsafe {
                std::env::remove_var("OAR_WEVISDOC_SKIP_RELEASE");
                std::env::remove_var("OAR_WEVISDOC_FAIL_CAPTURE");
            }
            model.recover_failed_capture();
            let settled = measured(&model);
            eprintln!("DBGM1 settled after control={settled}MiB");
            assert!(settled.saturating_sub(baseline) <= 16);

            // With the injection cleared, decoding captures normally again
            // and matches eager.
            let eager = greedy_eager(&model, &lm_head, &ids, 8);
            let graphed = greedy_graphed(&model, &lm_head, &ids, 8, true);
            assert_eq!(graphed, eager, "decode after recovery must match eager");

            drop(model);
            drop(lm_head);
            let probe = Tensor::randn(0f32, 1f32, (64, 64), &device).unwrap();
            let probe = (&probe * &probe).unwrap().sum_all().unwrap();
            let _ = probe.to_scalar::<f32>().unwrap();
        }
        #[cfg(not(feature = "cuda"))]
        eprintln!("skipping: built without the cuda feature");
    }

    /// A batch-shaped graph left behind by region decoding must be freed
    /// before a single-page request starts (and vice versa), or its fixed
    /// KV buckets compete with vision encoding for memory. Compatible
    /// storage must survive.
    #[test]
    fn cuda_incompatible_fixed_storage_is_released_before_the_next_request() {
        #[cfg(feature = "cuda")]
        {
            if std::env::var_os("OAR_WEVISDOC_GPU_SELFTEST").is_none() {
                eprintln!("skipping: OAR_WEVISDOC_GPU_SELFTEST is not set");
                return;
            }
            let Ok(device) = Device::new_cuda(0) else {
                eprintln!("skipping: no CUDA device");
                return;
            };
            let mut cfg = valid_tiny_config();
            cfg.hidden_size = 2048;
            cfg.intermediate_size = 6144;
            cfg.num_attention_heads = 16;
            cfg.num_key_value_heads = 8;
            cfg.head_dim = 128;
            cfg.num_hidden_layers = 28;
            cfg.vocab_size = 32768;
            let tensors = random_var_map(&cfg, &device, DType::BF16);
            let vb = VarBuilder::from_tensors(tensors, DType::BF16, &device);
            let model = Qwen3VlTextModel::load(&cfg, vb.pp("model")).unwrap();
            let lm_head = Linear::new(
                vb.get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")
                    .unwrap(),
                None,
            );
            let ids = (0..600).map(|i| 10 + i % 60).collect::<Vec<u32>>();

            model.clear_cache();
            let _ = model
                .forward(
                    &Tensor::zeros((1, 8, cfg.hidden_size), DType::BF16, &device).unwrap(),
                    &Tensor::zeros((3, 1, 8), DType::I64, &device).unwrap(),
                    None,
                    None,
                    Some(&[(0usize, 8)]),
                )
                .unwrap();
            let baseline = measured(&model);
            eprintln!("DBGM3 baseline={baseline}MiB");
            assert!(baseline > 0, "nvidia-smi unavailable; cannot measure");

            // Region-style batch request captures a batch graph.
            model
                .prepare_batch_ar_cuda_graph(2, 600, 8192, &[0, 10], &lm_head, false)
                .unwrap();
            assert!(model.batch_decode_graph_captured());

            // A single-page request arrives: its entry frees the batch
            // graph and buckets before vision encoding.
            model.release_incompatible_fixed_storage(None);
            let after = measured(&model);
            eprintln!("DBGM3 after-single-entry-release={after}MiB");
            assert!(
                after.saturating_sub(baseline) <= 16,
                "batch storage survived the single-page entry: {} MiB",
                after.saturating_sub(baseline)
            );
            assert!(!model.batch_decode_graph_captured());
            assert!(!model.decode_graph_captured());

            // Single-page decoding captures a single-row graph.
            model
                .prepare_ar_cuda_graph(600, 8192, &lm_head, false)
                .unwrap();
            assert!(model.decode_graph_captured());

            // A region-style batch request arrives: its entry frees the
            // single-row graph and buckets.
            model.release_incompatible_fixed_storage(Some(2));
            let after = measured(&model);
            eprintln!("DBGM3 after-batch-entry-release={after}MiB");
            assert!(
                after.saturating_sub(baseline) <= 16,
                "single-row storage survived the batch entry: {} MiB",
                after.saturating_sub(baseline)
            );
            assert!(!model.decode_graph_captured());
            assert!(!model.batch_decode_graph_captured());

            // Compatible storage survives a same-width request.
            model
                .prepare_batch_ar_cuda_graph(2, 600, 8192, &[0, 10], &lm_head, false)
                .unwrap();
            assert!(model.batch_decode_graph_captured());
            model.release_incompatible_fixed_storage(Some(2));
            assert!(
                model.batch_decode_graph_captured(),
                "compatible batch storage must not be released"
            );

            // Decoding still matches eager after all of this.
            let eager = greedy_eager(&model, &lm_head, &ids, 8);
            let graphed = greedy_graphed(&model, &lm_head, &ids, 8, true);
            assert_eq!(graphed, eager, "decode must match eager afterwards");
        }
        #[cfg(not(feature = "cuda"))]
        eprintln!("skipping: built without the cuda feature");
    }

    /// F16 graphs must produce finite logits: the attention fill has to
    /// stay inside F16's range, or every masked row collapses to NaN.
    /// Both graph flavors decode and match F16 eager token for token.
    #[test]
    fn cuda_f16_graph_decode_stays_finite_and_matches_eager() {
        #[cfg(feature = "cuda")]
        {
            if std::env::var_os("OAR_WEVISDOC_GPU_SELFTEST").is_none() {
                eprintln!("skipping: OAR_WEVISDOC_GPU_SELFTEST is not set");
                return;
            }
            let Ok(device) = Device::new_cuda(0) else {
                eprintln!("skipping: no CUDA device");
                return;
            };
            use crate::runtime::attention::{
                combine_masks, create_causal_mask, create_generation_mask_if_needed,
                create_left_padding_mask,
            };
            let mut cfg = valid_tiny_config();
            cfg.hidden_size = 2048;
            cfg.intermediate_size = 6144;
            cfg.num_attention_heads = 16;
            cfg.num_key_value_heads = 8;
            cfg.head_dim = 128;
            cfg.num_hidden_layers = 28;
            cfg.vocab_size = 32768;
            // F16 model: the mask fill must be finite in this dtype.
            let tensors = random_var_map(&cfg, &device, DType::F16);
            let vb = VarBuilder::from_tensors(tensors, DType::F16, &device);
            let model = Qwen3VlTextModel::load(&cfg, vb.pp("model")).unwrap();
            let lm_head = Linear::new(
                vb.get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")
                    .unwrap(),
                None,
            );
            let ids = (0..600).map(|i| 10 + i % 60).collect::<Vec<u32>>();

            // Single-row graph: finite logits and greedy agreement.
            let eager = greedy_eager(&model, &lm_head, &ids, 8);
            let graphed = greedy_graphed(&model, &lm_head, &ids, 8, true);
            eprintln!("DBGF16 single eager={eager:?}");
            eprintln!("DBGF16 single graphed={graphed:?}");
            assert_eq!(graphed, eager, "F16 single-row graph must match eager");

            // Batch graph: decode through it and assert every logit is
            // finite (NaN would flow from the masked attention rows).
            let seq_lens = vec![540usize, 520usize];
            let pads = vec![0usize, 20usize];
            let pad_starts: Vec<u32> = pads.iter().map(|&pad| pad as u32).collect();
            let embeds = Tensor::randn(0f32, 1f32, (2, 540, cfg.hidden_size), &device)
                .unwrap()
                .to_dtype(DType::F16)
                .unwrap();
            let positions = Tensor::zeros((3, 2, 540), DType::I64, &device).unwrap();
            let rows = vec![540usize, 520usize];
            model.clear_cache();
            model
                .prepare_batch_ar_cuda_graph(
                    rows.len(),
                    *seq_lens.iter().max().unwrap(),
                    8,
                    &pads,
                    &lm_head,
                    true,
                )
                .unwrap();
            assert!(model.batch_decode_graph_captured());
            let causal = create_causal_mask(540, 540, DType::F16, &device).unwrap();
            let padding = create_left_padding_mask(&seq_lens, 540, DType::F16, &device).unwrap();
            let prefill_mask = combine_masks(&causal, &padding).unwrap();
            let hidden = model
                .forward(&embeds, &positions, None, Some(&prefill_mask), None)
                .unwrap();
            let mut logits = lm_head
                .forward(&hidden.i((.., 539, ..)).unwrap().contiguous().unwrap())
                .unwrap();
            for step in 0..4 {
                let scores = logits
                    .to_dtype(DType::F32)
                    .unwrap()
                    .to_vec2::<f32>()
                    .unwrap();
                for (row, score) in scores.iter().enumerate() {
                    let nonfinite = score.iter().filter(|v| !v.is_finite()).count();
                    let lo = score.iter().cloned().fold(f32::INFINITY, f32::min);
                    let hi = score.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                    eprintln!(
                        "DBGF16 batch step {step} row {row}: len {} non-finite {nonfinite} range [{lo:.3}, {hi:.3}]",
                        score.len()
                    );
                    assert!(
                        nonfinite == 0,
                        "F16 batch graph produced non-finite logits at step {step} row {row}"
                    );
                }
                let tokens: Vec<u32> = (0..rows.len())
                    .map(|row| {
                        let t = logits.i(row).unwrap();
                        argmax_of(&t) as u32
                    })
                    .collect();
                eprintln!("DBGF16 batch step {step} picked {tokens:?}");
                let kv_len = 540 + step + 1;
                let row_starts = vec![(kv_len - 1) as u32; rows.len()];
                let ids_t = Tensor::from_vec(tokens.clone(), (rows.len(), 1), &device).unwrap();
                let embed = model.embed(&ids_t).unwrap();
                let mut pos_data = Vec::with_capacity(3 * rows.len());
                for &seq_len in &seq_lens {
                    for _ in 0..3 {
                        pos_data.push((seq_len + step) as i64);
                    }
                }
                let pos = Tensor::from_vec(pos_data, (3, rows.len(), 1), &device).unwrap();
                let gen_mask =
                    create_generation_mask_if_needed(&pads, kv_len, DType::F16, &device).unwrap();
                logits = model
                    .forward_decode_logits_batch(
                        &embed,
                        &pos,
                        BatchDecodeRows {
                            row_starts: &row_starts,
                            pad_lens: &pad_starts,
                        },
                        kv_len,
                        gen_mask.as_ref(),
                        &lm_head,
                    )
                    .unwrap();
            }
            drop(model);
            drop(lm_head);
        }
        #[cfg(not(feature = "cuda"))]
        eprintln!("skipping: built without the cuda feature");
    }

    #[test]
    fn cuda_batch_decode_graph_matches_masked_eager() {
        #[cfg(feature = "cuda")]
        {
            use crate::runtime::attention::{
                combine_masks, create_causal_mask, create_generation_mask_if_needed,
                create_left_padding_mask,
            };
            use candle_nn::Linear;
            if std::env::var_os("OAR_WEVISDOC_GPU_SELFTEST").is_none() {
                eprintln!("skipping: OAR_WEVISDOC_GPU_SELFTEST is not set");
                return;
            }
            let Ok(device) = Device::new_cuda(0) else {
                eprintln!("skipping: no CUDA device");
                return;
            };
            let mut cfg = valid_tiny_config();
            cfg.hidden_size = 2048;
            cfg.intermediate_size = 6144;
            cfg.num_attention_heads = 16;
            cfg.num_key_value_heads = 8;
            cfg.head_dim = 128;
            cfg.num_hidden_layers = 4;
            cfg.vocab_size = 32768;
            let tensors = random_var_map(&cfg, &device, DType::BF16);
            let vb = VarBuilder::from_tensors(tensors, DType::BF16, &device);
            let model = Qwen3VlTextModel::load(&cfg, vb.pp("model")).unwrap();
            let lm_head = Linear::new(
                vb.get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")
                    .unwrap(),
                None,
            );

            let make_ids =
                |len: usize| -> Vec<u32> { (0..len).map(|i| 10 + i as u32 % 60).collect() };
            let steps = 6usize;

            fn decode_step(row: usize, logits: &Tensor) -> u32 {
                let scores = logits.i(row).unwrap();
                argmax_of(&scores) as u32
            }

            // Left-padded embeds and positions for one batch, plus its
            // per-row sequence and pad lengths.
            let build_padded_batch =
                |rows: &[Vec<u32>]| -> (Tensor, Tensor, Vec<usize>, Vec<usize>) {
                    let seq_lens: Vec<usize> = rows.iter().map(|r| r.len()).collect();
                    let max_seq = *seq_lens.iter().max().unwrap();
                    let pads: Vec<usize> = seq_lens.iter().map(|&len| max_seq - len).collect();
                    let mut embed_rows = Vec::new();
                    let mut position_rows = Vec::new();
                    for row in rows {
                        let ids = Tensor::from_vec(row.clone(), (1, row.len()), &device).unwrap();
                        let embeds = model.embed(&ids).unwrap();
                        let pad = Tensor::zeros(
                            (1, max_seq - row.len(), cfg.hidden_size),
                            DType::BF16,
                            &device,
                        )
                        .unwrap();
                        embed_rows.push(Tensor::cat(&[&pad, &embeds], 1).unwrap());
                        let base = Tensor::arange(0i64, row.len() as i64, &device)
                            .unwrap()
                            .reshape((1, 1, row.len()))
                            .unwrap();
                        let mut data = Vec::with_capacity(3 * row.len());
                        for _ in 0..3 {
                            data.extend(base.flatten_all().unwrap().to_vec1::<i64>().unwrap());
                        }
                        let positions = Tensor::from_vec(data, (3, 1, row.len()), &device).unwrap();
                        let pad_pos =
                            Tensor::zeros((3, 1, max_seq - row.len()), DType::I64, &device)
                                .unwrap();
                        position_rows.push(Tensor::cat(&[&pad_pos, &positions], 2).unwrap());
                    }
                    let embeds = Tensor::cat(&embed_rows.iter().collect::<Vec<_>>(), 0).unwrap();
                    let positions =
                        Tensor::cat(&position_rows.iter().collect::<Vec<_>>(), 1).unwrap();
                    (embeds, positions, seq_lens, pads)
                };

            // One batch through the eager masked path: masked prefill plus
            // `steps` masked decode steps, one token row per step.
            let run_eager_batch = |rows: &[Vec<u32>], steps: usize| -> Vec<Vec<u32>> {
                let (embeds, positions, seq_lens, pads) = build_padded_batch(rows);
                let batch = rows.len();
                let max_seq = *seq_lens.iter().max().unwrap();
                let causal = create_causal_mask(max_seq, max_seq, DType::BF16, &device).unwrap();
                let padding =
                    create_left_padding_mask(&seq_lens, max_seq, DType::BF16, &device).unwrap();
                let prefill_mask = combine_masks(&causal, &padding).unwrap();

                model.clear_cache();
                let hidden = model
                    .forward(&embeds, &positions, None, Some(&prefill_mask), None)
                    .unwrap();
                let mut logits = lm_head
                    .forward(
                        &hidden
                            .i((.., max_seq - 1, ..))
                            .unwrap()
                            .contiguous()
                            .unwrap(),
                    )
                    .unwrap();
                let mut tokens_per_step = Vec::new();
                for step in 0..steps {
                    let mut tokens = Vec::new();
                    for row in 0..batch {
                        tokens.push(decode_step(row, &logits));
                    }
                    tokens_per_step.push(tokens.clone());
                    let kv_len = max_seq + step + 1;
                    let ids = Tensor::from_vec(tokens.clone(), (batch, 1), &device).unwrap();
                    let embed = model.embed(&ids).unwrap();
                    let mut pos_data = Vec::with_capacity(3 * batch);
                    for _ in 0..3 {
                        for &seq_len in &seq_lens {
                            pos_data.push((seq_len + step) as i64);
                        }
                    }
                    let pos = Tensor::from_vec(pos_data, (3, batch, 1), &device).unwrap();
                    let gen_mask =
                        create_generation_mask_if_needed(&pads, kv_len, DType::BF16, &device)
                            .unwrap();
                    let hidden = model
                        .forward(&embed, &pos, None, gen_mask.as_ref(), None)
                        .unwrap();
                    logits = lm_head.forward(&hidden).unwrap();
                }
                tokens_per_step
            };

            // The same batch through the production path: identical prefill,
            // decode steps replay the captured batch graph.
            let run_graphed_batch = |rows: &[Vec<u32>], steps: usize| -> Vec<Vec<u32>> {
                let (embeds, positions, seq_lens, pads) = build_padded_batch(rows);
                let batch = rows.len();
                let max_seq = *seq_lens.iter().max().unwrap();
                let pad_starts: Vec<u32> = pads.iter().map(|&pad| pad as u32).collect();
                let causal = create_causal_mask(max_seq, max_seq, DType::BF16, &device).unwrap();
                let padding =
                    create_left_padding_mask(&seq_lens, max_seq, DType::BF16, &device).unwrap();
                let prefill_mask = combine_masks(&causal, &padding).unwrap();

                model.clear_cache();
                model
                    .prepare_batch_ar_cuda_graph(batch, max_seq, steps, &pads, &lm_head, true)
                    .unwrap();
                assert!(
                    model.batch_decode_graph_captured(),
                    "batch decode graph did not capture (dtype gate?)"
                );
                let hidden = model
                    .forward(&embeds, &positions, None, Some(&prefill_mask), None)
                    .unwrap();
                let mut logits = lm_head
                    .forward(
                        &hidden
                            .i((.., max_seq - 1, ..))
                            .unwrap()
                            .contiguous()
                            .unwrap(),
                    )
                    .unwrap();
                let mut tokens_per_step = Vec::new();
                for step in 0..steps {
                    let mut tokens = Vec::new();
                    for row in 0..batch {
                        tokens.push(decode_step(row, &logits));
                    }
                    tokens_per_step.push(tokens.clone());
                    let kv_len = max_seq + step + 1;
                    let row_starts = vec![(kv_len - 1) as u32; batch];
                    let ids = Tensor::from_vec(tokens.clone(), (batch, 1), &device).unwrap();
                    let embed = model.embed(&ids).unwrap();
                    let mut pos_data = Vec::with_capacity(3 * batch);
                    for &seq_len in &seq_lens {
                        for _ in 0..3 {
                            pos_data.push((seq_len + step) as i64);
                        }
                    }
                    let pos = Tensor::from_vec(pos_data, (3, batch, 1), &device).unwrap();
                    let gen_mask =
                        create_generation_mask_if_needed(&pads, kv_len, DType::BF16, &device)
                            .unwrap();
                    logits = model
                        .forward_decode_logits_batch(
                            &embed,
                            &pos,
                            BatchDecodeRows {
                                row_starts: &row_starts,
                                pad_lens: &pad_starts,
                            },
                            kv_len,
                            gen_mask.as_ref(),
                            &lm_head,
                        )
                        .unwrap();
                }
                tokens_per_step
            };

            // First batch: captures the batch graph.
            let batch_a: Vec<Vec<u32>> = vec![make_ids(200), make_ids(120)];
            assert_eq!(
                run_graphed_batch(&batch_a, steps),
                run_eager_batch(&batch_a, steps),
                "first batch graph decode must match the eager masked path"
            );

            // Second batch: same width, different row lengths — different
            // pads and a different prefill width, so the graph is reused.
            // Every batch-dependent input must be rewritten before replay;
            // a stale pad mask reads the previous batch's padding bounds and
            // derails the decode (the bug this guards).
            let batch_b: Vec<Vec<u32>> = vec![make_ids(140), make_ids(180)];
            assert_eq!(
                run_graphed_batch(&batch_b, steps),
                run_eager_batch(&batch_b, steps),
                "reused graph decode must match eager for a different batch shape"
            );

            // Third batch: a new width forces a re-capture; it must match too.
            let batch_c: Vec<Vec<u32>> = vec![make_ids(150), make_ids(130), make_ids(110)];
            assert_eq!(
                run_graphed_batch(&batch_c, steps),
                run_eager_batch(&batch_c, steps),
                "re-captured graph decode must match eager after a width change"
            );

            // Fourth batch: a smaller prompt bucket shrinks the graph, then
            // a long generation crosses the bucket mid-decode — the ladder
            // doubles it, preserves the KV history, and must still match
            // eager token for token.
            let batch_d: Vec<Vec<u32>> = vec![make_ids(100), make_ids(90)];
            assert_eq!(
                run_graphed_batch(&batch_d, 40),
                run_eager_batch(&batch_d, 40),
                "ladder-grown graph decode must match eager"
            );

            // Teacher forcing along the graph's own token sequence: replay
            // the generated tokens through the eager path and bound the
            // per-step |delta logit| on row 0. Argmax agreement alone can
            // hide compensating drift; this catches a stale mask or a
            // missed rewrite while it is still a rounding artifact.
            {
                let rows: Vec<Vec<u32>> = vec![make_ids(100), make_ids(90)];
                let probe_steps = 40usize;
                let (embeds, positions, seq_lens, pads) = build_padded_batch(&rows);
                let probe_batch = rows.len();
                let max_seq = *seq_lens.iter().max().unwrap();
                let pad_starts: Vec<u32> = pads.iter().map(|&pad| pad as u32).collect();
                let causal = create_causal_mask(max_seq, max_seq, DType::BF16, &device).unwrap();
                let padding =
                    create_left_padding_mask(&seq_lens, max_seq, DType::BF16, &device).unwrap();
                let prefill_mask = combine_masks(&causal, &padding).unwrap();

                // Graphed pass, recording row 0's logits every step.
                model.clear_cache();
                model
                    .prepare_batch_ar_cuda_graph(probe_batch, max_seq, steps, &pads, &lm_head, true)
                    .unwrap();
                let hidden = model
                    .forward(&embeds, &positions, None, Some(&prefill_mask), None)
                    .unwrap();
                let mut logits = lm_head
                    .forward(
                        &hidden
                            .i((.., max_seq - 1, ..))
                            .unwrap()
                            .contiguous()
                            .unwrap(),
                    )
                    .unwrap();
                let mut graphed_tokens: Vec<Vec<u32>> = Vec::new();
                let mut graphed_row0: Vec<Vec<f32>> = Vec::new();
                for step in 0..probe_steps {
                    graphed_row0.push(
                        logits
                            .i(0)
                            .unwrap()
                            .to_dtype(DType::F32)
                            .unwrap()
                            .flatten_all()
                            .unwrap()
                            .to_vec1::<f32>()
                            .unwrap(),
                    );
                    let mut tokens = Vec::new();
                    for row in 0..probe_batch {
                        tokens.push(decode_step(row, &logits));
                    }
                    graphed_tokens.push(tokens.clone());
                    let kv_len = max_seq + step + 1;
                    let row_starts = vec![(kv_len - 1) as u32; probe_batch];
                    let ids = Tensor::from_vec(tokens, (probe_batch, 1), &device).unwrap();
                    let embed = model.embed(&ids).unwrap();
                    let mut pos_data = Vec::with_capacity(3 * probe_batch);
                    for &seq_len in &seq_lens {
                        for _ in 0..3 {
                            pos_data.push((seq_len + step) as i64);
                        }
                    }
                    let pos = Tensor::from_vec(pos_data, (3, probe_batch, 1), &device).unwrap();
                    let gen_mask =
                        create_generation_mask_if_needed(&pads, kv_len, DType::BF16, &device)
                            .unwrap();
                    logits = model
                        .forward_decode_logits_batch(
                            &embed,
                            &pos,
                            BatchDecodeRows {
                                row_starts: &row_starts,
                                pad_lens: &pad_starts,
                            },
                            kv_len,
                            gen_mask.as_ref(),
                            &lm_head,
                        )
                        .unwrap();
                }

                // Eager replay on the recorded tokens.
                model.clear_cache();
                let hidden = model
                    .forward(&embeds, &positions, None, Some(&prefill_mask), None)
                    .unwrap();
                let mut logits = lm_head
                    .forward(
                        &hidden
                            .i((.., max_seq - 1, ..))
                            .unwrap()
                            .contiguous()
                            .unwrap(),
                    )
                    .unwrap();
                let mut worst_delta = 0.0f32;
                for step in 0..probe_steps {
                    let eager0 = logits
                        .i(0)
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap()
                        .flatten_all()
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                    let delta = graphed_row0[step]
                        .iter()
                        .zip(eager0.iter())
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0f32, f32::max);
                    worst_delta = worst_delta.max(delta);
                    let tokens = graphed_tokens[step].clone();
                    let kv_len = max_seq + step + 1;
                    let ids = Tensor::from_vec(tokens, (probe_batch, 1), &device).unwrap();
                    let embed = model.embed(&ids).unwrap();
                    let mut pos_data = Vec::with_capacity(3 * probe_batch);
                    for _ in 0..3 {
                        for &seq_len in &seq_lens {
                            pos_data.push((seq_len + step) as i64);
                        }
                    }
                    let pos = Tensor::from_vec(pos_data, (3, probe_batch, 1), &device).unwrap();
                    let gen_mask =
                        create_generation_mask_if_needed(&pads, kv_len, DType::BF16, &device)
                            .unwrap();
                    let hidden = model
                        .forward(&embed, &pos, None, gen_mask.as_ref(), None)
                        .unwrap();
                    logits = lm_head.forward(&hidden).unwrap();
                }
                assert!(
                    worst_delta < 2.0,
                    "teacher-forced |delta logit| reached {worst_delta}, \
                     the graph path is not tracking eager"
                );
            }

            // A single-row request whose prompt exceeds the bucket limit
            // falls back to eager: both graphs must be dropped, because the
            // long decode would outgrow their captured KV storage. The
            // fallback decode still has to match eager token for token.
            let huge: Vec<u32> = (0..8300).map(|i| 10 + i as u32 % 60).collect();
            model.clear_cache();
            model
                .prepare_ar_cuda_graph(huge.len(), 4, &lm_head, true)
                .unwrap();
            assert!(
                !model.decode_graph_captured(),
                "eager fallback must drop the single-row graph"
            );
            assert!(
                !model.batch_decode_graph_captured(),
                "eager fallback must drop the batch graph"
            );
            assert_eq!(
                greedy_eager(&model, &lm_head, &huge, 4),
                greedy_graphed(&model, &lm_head, &huge, 4, false),
                "long-prompt eager fallback decode must still match eager"
            );

            // After the batch graph ran and the fallback dropped both
            // graphs, a fresh single-row capture must work again at the
            // production bucket and match eager token for token.
            let long = make_ids(4200);
            assert_eq!(
                greedy_eager(&model, &lm_head, &long, 8),
                greedy_graphed(&model, &lm_head, &long, 8, true),
                "capture after the eager fallback must match eager"
            );

            // The reverse fallback: with the single-row graph alive, an
            // over-limit batch request stays eager and its prefill
            // reinitializes the KV storage — the fallback must drop BOTH
            // graphs, or the next single-row replay reads freed memory.
            model.clear_cache();
            model.prepare_ar_cuda_graph(600, 8, &lm_head, true).unwrap();
            assert!(model.decode_graph_captured());
            model
                .prepare_batch_ar_cuda_graph(2, 8300, steps, &[0, 10], &lm_head, true)
                .unwrap();
            assert!(
                !model.decode_graph_captured(),
                "over-limit batch fallback must drop the single-row graph"
            );
            assert!(
                !model.batch_decode_graph_captured(),
                "over-limit batch fallback must drop the batch graph"
            );
            let after = make_ids(600);
            assert_eq!(
                greedy_eager(&model, &lm_head, &after, 8),
                greedy_graphed(&model, &lm_head, &after, 8, true),
                "single-row decode after the batch eager fallback must match eager"
            );

            // Dropping a model with a live batch graph must dispose it
            // cleanly; the allocator stays healthy for later work.
            drop(model);
            drop(lm_head);
            let probe = Tensor::randn(0f32, 1f32, (64, 64), &device).unwrap();
            let probe = (&probe * &probe).unwrap().sum_all().unwrap();
            let _ = probe.to_scalar::<f32>().unwrap();
        }
        #[cfg(not(feature = "cuda"))]
        eprintln!("skipping: built without the cuda feature");
    }

    /// Greedy decode driven by plain `forward` calls: never captures or
    /// replays the decode graph, so it is the eager reference.
    #[cfg(feature = "cuda")]
    fn greedy_eager(
        model: &Qwen3VlTextModel,
        lm_head: &candle_nn::Linear,
        ids: &[u32],
        steps: usize,
    ) -> Vec<u32> {
        let device = model.embed_tokens.embeddings().device();
        let seq_len = ids.len();
        let token_ids = Tensor::from_vec(ids.to_vec(), (1, seq_len), device).unwrap();
        let embeds = model.embed(&token_ids).unwrap();
        let positions = text_position_ids_range(seq_len, device);
        model.clear_cache();
        let hidden = model
            .forward(&embeds, &positions, None, None, None)
            .unwrap();
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
        let mut out = Vec::new();
        for step in 0..steps {
            let best = argmax_of(&logits);
            out.push(best as u32);
            let token = Tensor::from_vec(vec![best as u32], (1, 1), device).unwrap();
            let embed = model.embed(&token).unwrap();
            let pos =
                Tensor::from_vec(vec![(seq_len + step) as i64; 3], (3, 1, 1), device).unwrap();
            let next = model.forward(&embed, &pos, None, None, None).unwrap();
            logits = lm_head
                .forward(&next.i((0, 0, ..)).unwrap().unsqueeze(0).unwrap())
                .unwrap()
                .squeeze(0)
                .unwrap();
        }
        out
    }

    /// Greedy decode through `prepare_ar_cuda_graph` + `forward_decode_logits`,
    /// i.e. exactly the production graph path. `expect_capture` asserts the
    /// capture state the request deserves: prompts beyond the bucket limit
    /// stay eager, everything else must capture.
    #[cfg(feature = "cuda")]
    fn greedy_graphed(
        model: &Qwen3VlTextModel,
        lm_head: &candle_nn::Linear,
        ids: &[u32],
        steps: usize,
        expect_capture: bool,
    ) -> Vec<u32> {
        let device = model.embed_tokens.embeddings().device();
        let seq_len = ids.len();
        let token_ids = Tensor::from_vec(ids.to_vec(), (1, seq_len), device).unwrap();
        let embeds = model.embed(&token_ids).unwrap();
        let positions = text_position_ids_range(seq_len, device);
        model.clear_cache();
        model
            .prepare_ar_cuda_graph(seq_len, steps, lm_head, true)
            .unwrap();
        assert_eq!(
            model.decode_graph_captured(),
            expect_capture,
            "single-row graph capture state does not match the request"
        );
        let hidden = model
            .forward(&embeds, &positions, None, None, None)
            .unwrap();
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
        let mut out = Vec::new();
        for step in 0..steps {
            let best = argmax_of(&logits);
            out.push(best as u32);
            let token = Tensor::from_vec(vec![best as u32], (1, 1), device).unwrap();
            let embed = model.embed(&token).unwrap();
            let pos =
                Tensor::from_vec(vec![(seq_len + step) as i64; 3], (3, 1, 1), device).unwrap();
            logits = model
                .forward_decode_logits(&embed, &pos, None, lm_head)
                .unwrap();
        }
        out
    }

    #[cfg(feature = "cuda")]
    fn text_position_ids_range(seq_len: usize, device: &Device) -> Tensor {
        let base = Tensor::arange(0i64, seq_len as i64, device)
            .unwrap()
            .reshape((1, 1, seq_len))
            .unwrap();
        let mut data = Vec::with_capacity(3 * seq_len);
        for _ in 0..3 {
            data.extend(base.flatten_all().unwrap().to_vec1::<i64>().unwrap());
        }
        Tensor::from_vec(data, (3, 1, seq_len), device).unwrap()
    }

    #[cfg(feature = "cuda")]
    fn argmax_of(logits: &Tensor) -> usize {
        let scores = logits
            .flatten_all()
            .unwrap()
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
        best
    }

    fn random_varbuilder(cfg: &Qwen3VlTextConfig, device: &Device) -> VarBuilder<'static> {
        random_varbuilder_typed(cfg, device, DType::F32)
    }

    fn random_varbuilder_typed(
        cfg: &Qwen3VlTextConfig,
        device: &Device,
        dtype: DType,
    ) -> VarBuilder<'static> {
        let tensors = random_var_map(cfg, device, dtype);
        VarBuilder::from_tensors(tensors, dtype, device)
    }

    fn random_var_map(
        cfg: &Qwen3VlTextConfig,
        device: &Device,
        dtype: DType,
    ) -> std::collections::HashMap<String, Tensor> {
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
        for layer in 0..cfg.num_hidden_layers {
            let prefix = format!("model.layers.{layer}");
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
        }
        tensors
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
