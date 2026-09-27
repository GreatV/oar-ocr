//! Qwen2-family text decoder shared by the MinerU2.5 and NaviDC-OCR towers.
//!
//! Both checkpoints carry the same Qwen2/Qwen2.5 decoder stack (mrope
//! attention over a TrimmableKvCache); they differ only in knobs expressed
//! by [`Qwen2VlTextConfig`]: whether the attention projections carry a bias
//! and whether each head is normalised by `q_norm`/`k_norm` (RMSNorm over
//! `head_dim`, applied after the projection and before RoPE). The CUDA
//! decode-graph lifecycle (single-bucket capture, capacity-based reuse,
//! eager fallback) is written once here for both.

use crate::attention::{
    RotaryEmbedding, flash_attention, scaled_dot_product_attention_gqa, select_rope_sections,
};
use crate::error::Error;
use crate::runtime::cache::TrimmableKvCache;
#[cfg(feature = "cuda")]
use crate::runtime::cuda::dynamic_kv::DynamicKvAppend;
#[cfg(feature = "cuda")]
use crate::runtime::decoder_graph::decoder_cache_capacity;
#[cfg(feature = "cuda")]
use crate::runtime::decoder_graph::{
    CudaGraphDrainGuard, CudaGraphInputs, CudaGraphKvLengths, DecoderCudaGraph,
    capture_decoder_graph, cuda_graph_error, decoder_attention_is_causal,
};
use crate::runtime::errors::{candle_to_ocr_inference, candle_to_ocr_processing};
use crate::runtime::tensor::rotate_half;
#[cfg(feature = "cuda")]
use candle_core::DType;
use candle_core::{IndexOp, Tensor};
use candle_nn::{
    Embedding, Linear, Module, VarBuilder, embedding, linear, linear_no_bias, rms_norm,
};
use std::cell::RefCell;
use std::sync::Arc;

/// Text-decoder configuration for the shared Qwen2-family tower. Each model
/// converts its checkpoint config into this struct; the knob fields carry
/// the per-model differences so the decoder code itself stays shared.
#[derive(Debug, Clone)]
pub struct Qwen2VlTextConfig {
    /// Model name prefixing every error message (byte-identical to the
    /// per-model towers this replaces).
    pub model_name: &'static str,
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    /// Effective attention head dim, computed by the caller (MinerU2.5
    /// derives it from `hidden_size / num_attention_heads`; NaviDC-OCR
    /// prefers its explicit config field).
    pub head_dim: usize,
    pub mrope_section: Vec<usize>,
    /// Whether the q/k/v projections load a bias term.
    pub attention_bias: bool,
    /// Whether each head is normalised by `self_attn.q_norm`/`k_norm`
    /// (RMSNorm over `head_dim`) before RoPE.
    pub qk_head_norm: bool,
    /// Extra env var (besides `OAR_VL_DISABLE_CUDA_GRAPH`) that disables
    /// the decode graph for this model.
    pub graph_disable_env: &'static str,
    /// Upper bound on the decode-graph KV bucket.
    pub decode_cache_len: usize,
}

fn apply_multimodal_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mrope_section: &[usize],
    model_name: &str,
) -> Result<(Tensor, Tensor), Error> {
    let cos = select_rope_sections(cos, mrope_section, 3)?;
    let sin = select_rope_sections(sin, mrope_section, 3)?;

    let q_mul = q.broadcast_mul(&cos).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            format!("{model_name}: mrope q*cos failed"),
            e,
        )
    })?;
    let q_half = rotate_half(q)?;
    let q_half_mul = q_half.broadcast_mul(&sin).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            format!("{model_name}: mrope rotate_half(q)*sin failed"),
            e,
        )
    })?;
    let q_rot = (&q_mul + &q_half_mul).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            format!("{model_name}: mrope apply on q failed"),
            e,
        )
    })?;

    let k_mul = k.broadcast_mul(&cos).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            format!("{model_name}: mrope k*cos failed"),
            e,
        )
    })?;
    let k_half = rotate_half(k)?;
    let k_half_mul = k_half.broadcast_mul(&sin).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            format!("{model_name}: mrope rotate_half(k)*sin failed"),
            e,
        )
    })?;
    let k_rot = (&k_mul + &k_half_mul).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            format!("{model_name}: mrope apply on k failed"),
            e,
        )
    })?;

    Ok((q_rot, k_rot))
}

#[derive(Debug, Clone)]
struct Qwen2VlMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    model_name: &'static str,
}

impl Qwen2VlMlp {
    fn load(cfg: &Qwen2VlTextConfig, vb: VarBuilder) -> Result<Self, Error> {
        let gate_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("mlp.gate_proj"),
        )
        .map_err(|e| candle_to_ocr_inference(cfg.model_name, "load gate_proj", e))?;
        let up_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp.up_proj"))
            .map_err(|e| candle_to_ocr_inference(cfg.model_name, "load up_proj", e))?;
        let down_proj = linear_no_bias(
            cfg.intermediate_size,
            cfg.hidden_size,
            vb.pp("mlp.down_proj"),
        )
        .map_err(|e| candle_to_ocr_inference(cfg.model_name, "load down_proj", e))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            model_name: cfg.model_name,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, Error> {
        let gate = self
            .gate_proj
            .forward(xs)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "mlp gate_proj", e))?;
        let gate = candle_nn::ops::silu(&gate)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "mlp silu", e))?;
        let up = self
            .up_proj
            .forward(xs)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "mlp up_proj", e))?;
        let prod = (&gate * &up)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "mlp gate*up", e))?;
        self.down_proj
            .forward(&prod)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "mlp down_proj", e))
    }
}

#[derive(Debug)]
struct Qwen2VlAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    /// Per-head RMSNorm applied to Q after projection, before RoPE.
    q_norm: Option<candle_nn::RmsNorm>,
    /// Per-head RMSNorm applied to K after projection, before RoPE.
    k_norm: Option<candle_nn::RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
    mrope_section: Vec<usize>,
    model_name: &'static str,
    kv_cache: RefCell<TrimmableKvCache>,
}

impl Qwen2VlAttention {
    fn load(cfg: &Qwen2VlTextConfig, vb: VarBuilder) -> Result<Self, Error> {
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            return Err(Error::Config {
                message: format!(
                    "{}: num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                    cfg.model_name, cfg.num_attention_heads, cfg.num_key_value_heads
                ),
            });
        }
        let head_dim = cfg.head_dim;
        // NaviDC-OCR (and Qwen2.5 generally) drops the projection biases;
        // MinerU2.5 keeps them on q/k/v. `o_proj` never carries one.
        let (q_proj, k_proj, v_proj) = if cfg.attention_bias {
            (
                linear(
                    cfg.hidden_size,
                    cfg.num_attention_heads * head_dim,
                    vb.pp("self_attn.q_proj"),
                )
                .map_err(|e| candle_to_ocr_inference(cfg.model_name, "load q_proj", e))?,
                linear(
                    cfg.hidden_size,
                    cfg.num_key_value_heads * head_dim,
                    vb.pp("self_attn.k_proj"),
                )
                .map_err(|e| candle_to_ocr_inference(cfg.model_name, "load k_proj", e))?,
                linear(
                    cfg.hidden_size,
                    cfg.num_key_value_heads * head_dim,
                    vb.pp("self_attn.v_proj"),
                )
                .map_err(|e| candle_to_ocr_inference(cfg.model_name, "load v_proj", e))?,
            )
        } else {
            (
                linear_no_bias(
                    cfg.hidden_size,
                    cfg.num_attention_heads * head_dim,
                    vb.pp("self_attn.q_proj"),
                )
                .map_err(|e| candle_to_ocr_inference(cfg.model_name, "load q_proj", e))?,
                linear_no_bias(
                    cfg.hidden_size,
                    cfg.num_key_value_heads * head_dim,
                    vb.pp("self_attn.k_proj"),
                )
                .map_err(|e| candle_to_ocr_inference(cfg.model_name, "load k_proj", e))?,
                linear_no_bias(
                    cfg.hidden_size,
                    cfg.num_key_value_heads * head_dim,
                    vb.pp("self_attn.v_proj"),
                )
                .map_err(|e| candle_to_ocr_inference(cfg.model_name, "load v_proj", e))?,
            )
        };
        let o_proj = linear_no_bias(
            cfg.num_attention_heads * head_dim,
            cfg.hidden_size,
            vb.pp("self_attn.o_proj"),
        )
        .map_err(|e| candle_to_ocr_inference(cfg.model_name, "load o_proj", e))?;
        let q_norm = if cfg.qk_head_norm {
            Some(
                rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("self_attn.q_norm"))
                    .map_err(|e| candle_to_ocr_inference(cfg.model_name, "load q_norm", e))?,
            )
        } else {
            None
        };
        let k_norm = if cfg.qk_head_norm {
            Some(
                rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("self_attn.k_norm"))
                    .map_err(|e| candle_to_ocr_inference(cfg.model_name, "load k_norm", e))?,
            )
        } else {
            None
        };

        // Trim/gather-capable KV cache.
        let kv_cache = TrimmableKvCache::new(2, cfg.max_position_embeddings.max(8192));

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
            head_dim,
            scaling: (head_dim as f64).powf(-0.5),
            mrope_section: cfg.mrope_section.clone(),
            model_name: cfg.model_name,
            kv_cache: RefCell::new(kv_cache),
        })
    }

    fn project_qkv(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<(Tensor, Tensor, Tensor), Error> {
        let (b, seq_len, _) = hidden_states
            .dims3()
            .map_err(|e| candle_to_ocr_inference(self.model_name, "attn hidden_states dims3", e))?;

        // Qwen2.5 normalises each head on the (b, s, h, head_dim) view before
        // the transpose and RoPE application.
        let q = self
            .q_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "attn q_proj", e))?
            .reshape((b, seq_len, self.num_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference(self.model_name, "attn q reshape", e))?;
        let q = match &self.q_norm {
            Some(q_norm) => q_norm
                .forward(&q)
                .map_err(|e| candle_to_ocr_inference(self.model_name, "attn q_norm", e))?,
            None => q,
        };
        let q = q
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "attn q transpose", e))?;

        let k = self
            .k_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "attn k_proj", e))?
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference(self.model_name, "attn k reshape", e))?;
        let k = match &self.k_norm {
            Some(k_norm) => k_norm
                .forward(&k)
                .map_err(|e| candle_to_ocr_inference(self.model_name, "attn k_norm", e))?,
            None => k,
        };
        let k = k
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "attn k transpose", e))?;

        let v = self
            .v_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "attn v_proj", e))?
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference(self.model_name, "attn v reshape", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "attn v transpose", e))?;

        let (q, k) = apply_multimodal_rotary_pos_emb(
            &q,
            &k,
            cos,
            sin,
            &self.mrope_section,
            self.model_name,
        )?;
        let k = k
            .contiguous()
            .map_err(|e| candle_to_ocr_inference(self.model_name, "attn k contiguous", e))?;
        let v = v
            .contiguous()
            .map_err(|e| candle_to_ocr_inference(self.model_name, "attn v contiguous", e))?;

        Ok((q, k, v))
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, Error> {
        let (b, seq_len, _) = hidden_states
            .dims3()
            .map_err(|e| candle_to_ocr_inference(self.model_name, "attn hidden_states dims3", e))?;
        let (q, k, v) = self.project_qkv(hidden_states, cos, sin)?;

        let (k, v) = self
            .kv_cache
            .borrow_mut()
            .append(&k, &v)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "attn kv_cache append", e))?;
        let is_causal = attention_mask.is_none();
        let flash = if b == 1 {
            flash_attention(&q, &k, &v, self.scaling, seq_len > 1)
                .map_err(|e| candle_to_ocr_inference(self.model_name, "flash attention", e))?
        } else {
            None
        };
        let attn_output = match flash {
            Some(attn) => attn,
            None => scaled_dot_product_attention_gqa(
                &q,
                &k,
                &v,
                attention_mask,
                self.scaling,
                is_causal,
                self.num_kv_groups,
            )
            .map_err(|e| candle_to_ocr_inference(self.model_name, "grouped-query attention", e))?,
        };
        self.project_attention_output(&attn_output, b, seq_len)
    }

    fn project_attention_output(
        &self,
        attn_output: &Tensor,
        batch: usize,
        seq_len: usize,
    ) -> Result<Tensor, Error> {
        let attn_output = attn_output
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "attn output transpose", e))?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))
            .map_err(|e| candle_to_ocr_inference(self.model_name, "attn output reshape", e))?;

        self.o_proj
            .forward(&attn_output)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "attn o_proj", e))
    }

    #[cfg(feature = "cuda")]
    fn prepare_dynamic_cache(&self, query_len: usize, cache_len: usize) -> Result<(), Error> {
        let template = Tensor::zeros(
            (1, self.num_kv_heads, query_len, self.head_dim),
            self.k_proj.weight().dtype(),
            self.k_proj.weight().device(),
        )
        .map_err(|e| candle_to_ocr_inference(self.model_name, "dynamic KV template", e))?;
        self.kv_cache
            .borrow_mut()
            .initialize_storage_with_capacity(&template, cache_len)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "initialize dynamic KV", e))
    }

    #[cfg(feature = "cuda")]
    fn forward_dynamic(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        query_lengths: &Tensor,
        kv_lengths: &Tensor,
    ) -> Result<Tensor, Error> {
        let (batch, query_len, _) = hidden_states.dims3().map_err(|e| {
            candle_to_ocr_inference(self.model_name, "dynamic attention hidden shape", e)
        })?;
        if batch != 1 {
            return Err(Error::Config {
                message: format!(
                    "{} CUDA-graph attention requires batch size 1",
                    self.model_name
                ),
            });
        }
        let (q, k, v) = self.project_qkv(hidden_states, cos, sin)?;
        let cache = self.kv_cache.borrow();
        let cache_len = cache.storage_capacity();
        let (cache_k, cache_v) = cache.storage().ok_or_else(|| Error::Config {
            message: format!("{} dynamic KV storage is not initialized", self.model_name),
        })?;
        drop(cache);
        let append = DynamicKvAppend {
            query_len,
            cache_len,
        };
        cache_k
            .inplace_op3(&k, kv_lengths, &append)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "dynamic key cache append", e))?;
        cache_v.inplace_op3(&v, kv_lengths, &append).map_err(|e| {
            candle_to_ocr_inference(self.model_name, "dynamic value cache append", e)
        })?;

        let q = q
            .squeeze(0)
            .and_then(|q| q.transpose(0, 1))
            .map_err(|e| candle_to_ocr_inference(self.model_name, "dynamic Q layout", e))?;
        let cache_k = cache_k
            .squeeze(0)
            .and_then(|k| k.transpose(0, 1))
            .map_err(|e| candle_to_ocr_inference(self.model_name, "dynamic K layout", e))?;
        let cache_v = cache_v
            .squeeze(0)
            .and_then(|v| v.transpose(0, 1))
            .map_err(|e| candle_to_ocr_inference(self.model_name, "dynamic V layout", e))?;
        let attn = candle_flash_attn::flash_attn_varlen(
            &q,
            &cache_k,
            &cache_v,
            query_lengths,
            kv_lengths,
            query_len,
            cache_len,
            self.scaling as f32,
            decoder_attention_is_causal(query_len),
        )
        .map_err(|e| candle_to_ocr_inference(self.model_name, "dynamic flash attention", e))?
        .transpose(0, 1)
        .and_then(|attn| attn.unsqueeze(0))
        .map_err(|e| candle_to_ocr_inference(self.model_name, "dynamic attention layout", e))?;
        self.project_attention_output(&attn, batch, query_len)
    }

    fn clear_kv_cache(&self) {
        self.kv_cache.borrow_mut().reset();
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
            .map_err(|e| candle_to_ocr_inference(self.model_name, "set dynamic KV length", e))
    }
}

pub struct Qwen2VlDecoderLayer {
    self_attn: Qwen2VlAttention,
    mlp: Qwen2VlMlp,
    input_layernorm: candle_nn::RmsNorm,
    post_attention_layernorm: candle_nn::RmsNorm,
    model_name: &'static str,
}

impl Qwen2VlDecoderLayer {
    fn load(cfg: &Qwen2VlTextConfig, vb: VarBuilder) -> Result<Self, Error> {
        let self_attn = Qwen2VlAttention::load(cfg, vb.clone())?;
        let mlp = Qwen2VlMlp::load(cfg, vb.clone())?;
        let input_layernorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))
            .map_err(|e| candle_to_ocr_inference(cfg.model_name, "load input_layernorm", e))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )
        .map_err(|e| candle_to_ocr_inference(cfg.model_name, "load post_attention_layernorm", e))?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            model_name: cfg.model_name,
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
        let hidden_states = self
            .input_layernorm
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "input_layernorm", e))?;
        let hidden_states = self
            .self_attn
            .forward(&hidden_states, cos, sin, attention_mask)?;
        let hidden_states = (&residual + &hidden_states).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                format!(
                    "{model_name}: attn residual add failed",
                    model_name = self.model_name
                ),
                e,
            )
        })?;

        let residual = hidden_states.clone();
        let hidden_states = self
            .post_attention_layernorm
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "post_attention_layernorm", e))?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        (&residual + &hidden_states).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                format!(
                    "{model_name}: mlp residual add failed",
                    model_name = self.model_name
                ),
                e,
            )
        })
    }

    #[cfg(feature = "cuda")]
    fn forward_dynamic(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        query_lengths: &Tensor,
        kv_lengths: &Tensor,
    ) -> Result<Tensor, Error> {
        let residual = hidden_states.clone();
        let hidden_states = self
            .input_layernorm
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "input_layernorm", e))?;
        let hidden_states =
            self.self_attn
                .forward_dynamic(&hidden_states, cos, sin, query_lengths, kv_lengths)?;
        let hidden_states = (&residual + &hidden_states).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                format!(
                    "{model_name}: attn residual add failed",
                    model_name = self.model_name
                ),
                e,
            )
        })?;

        let residual = hidden_states.clone();
        let hidden_states = self
            .post_attention_layernorm
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "post_attention_layernorm", e))?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        (&residual + &hidden_states).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                format!(
                    "{model_name}: mlp residual add failed",
                    model_name = self.model_name
                ),
                e,
            )
        })
    }

    fn clear_kv_cache(&self) {
        self.self_attn.clear_kv_cache();
    }

    #[cfg(feature = "cuda")]
    fn kv_cache_len(&self) -> usize {
        self.self_attn.kv_cache_len()
    }

    #[cfg(feature = "cuda")]
    fn set_kv_cache_len(&self, len: usize) -> Result<(), Error> {
        self.self_attn.set_kv_cache_len(len)
    }
}

pub struct Qwen2VlTextModel {
    #[cfg(feature = "cuda")]
    decode_graph: RefCell<Option<DecoderCudaGraph<CudaGraphInputs>>>,
    embed_tokens: Embedding,
    layers: Vec<Qwen2VlDecoderLayer>,
    norm: candle_nn::RmsNorm,
    rotary_emb: Arc<RotaryEmbedding>,
    model_name: &'static str,
    graph_disable_env: &'static str,
    #[cfg(feature = "cuda")]
    decode_cache_len: usize,
    // Must stay the last field: it drops last and drains CUDA errors the
    // other fields' frees may stash (see CudaGraphDrainGuard).
    #[cfg(feature = "cuda")]
    _drain_guard: CudaGraphDrainGuard,
}

impl Qwen2VlTextModel {
    pub fn load(cfg: &Qwen2VlTextConfig, vb: VarBuilder) -> Result<Self, Error> {
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))
            .map_err(|e| candle_to_ocr_inference(cfg.model_name, "load embed_tokens", e))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer_vb = vb.pp(format!("layers.{i}"));
            layers.push(Qwen2VlDecoderLayer::load(cfg, layer_vb)?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))
            .map_err(|e| candle_to_ocr_inference(cfg.model_name, "load norm", e))?;
        let rotary_emb = Arc::new(RotaryEmbedding::new_multi_axis(
            cfg.head_dim,
            cfg.rope_theta,
            3,
            vb.device(),
        )?);

        #[cfg(feature = "cuda")]
        let _drain_guard = CudaGraphDrainGuard::new(vb.device());
        Ok(Self {
            #[cfg(feature = "cuda")]
            decode_graph: RefCell::new(None),
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            model_name: cfg.model_name,
            graph_disable_env: cfg.graph_disable_env,
            #[cfg(feature = "cuda")]
            decode_cache_len: cfg.decode_cache_len,
            #[cfg(feature = "cuda")]
            _drain_guard,
        })
    }

    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor, Error> {
        self.embed_tokens
            .forward(input_ids)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "embed forward", e))
    }

    pub fn forward(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, Error> {
        let (cos, sin) = self
            .rotary_emb
            .forward_multi_axis(position_ids, inputs_embeds.dtype())?;

        let mut hidden_states = inputs_embeds.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &cos, &sin, attention_mask)?;
        }
        self.norm
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "norm forward", e))
    }

    fn project_logits(&self, hidden_states: &Tensor, lm_head: &Linear) -> Result<Tensor, Error> {
        lm_head
            .forward(hidden_states)
            .and_then(|logits| logits.i((0, 0, ..)))
            .map_err(|e| candle_to_ocr_inference(self.model_name, "decode LM head", e))
    }

    pub(crate) fn forward_decode_logits(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        lm_head: &Linear,
    ) -> Result<Tensor, Error> {
        #[cfg(feature = "cuda")]
        {
            let kv_len = self.kv_cache_len().saturating_add(1);
            if let Some(logits) = self.replay_cuda_graph(inputs_embeds, position_ids, kv_len)? {
                return Ok(logits);
            }
        }
        let hidden = self.forward(inputs_embeds, position_ids, attention_mask)?;
        self.project_logits(&hidden, lm_head)
    }

    #[cfg(feature = "cuda")]
    fn forward_dynamic(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        query_lengths: &Tensor,
        kv_lengths: &Tensor,
    ) -> Result<Tensor, Error> {
        let (cos, sin) = self
            .rotary_emb
            .forward_multi_axis(position_ids, inputs_embeds.dtype())?;
        let mut hidden_states = inputs_embeds.clone();
        for layer in &self.layers {
            hidden_states =
                layer.forward_dynamic(&hidden_states, &cos, &sin, query_lengths, kv_lengths)?;
        }
        self.norm
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "dynamic norm", e))
    }

    pub(crate) fn prepare_ar_cuda_graph(
        &self,
        prompt_len: usize,
        max_new_tokens: usize,
        lm_head: &Linear,
    ) -> Result<(), Error> {
        if std::env::var_os("OAR_VL_DISABLE_CUDA_GRAPH").is_some()
            || std::env::var_os(self.graph_disable_env).is_some()
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
                decoder_cache_capacity(prompt_len, max_new_tokens, self.decode_cache_len)
            else {
                self.invalidate_cuda_graph();
                return Ok(());
            };
            let required = prompt_len
                .saturating_add(max_new_tokens)
                .min(self.decode_cache_len);
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
        if self.decode_graph.borrow().is_some() {
            return Ok(());
        }
        let device = self.embed_tokens.embeddings().device();
        if !device.is_cuda() {
            return Ok(());
        }
        let query_len = 1;
        for layer in &self.layers {
            layer
                .self_attn
                .prepare_dynamic_cache(query_len, cache_len)?;
        }
        let hidden_size = self
            .embed_tokens
            .embeddings()
            .dim(1)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "graph hidden size", e))?;
        let device = self.embed_tokens.embeddings().device();
        let inputs = CudaGraphInputs {
            hidden: Tensor::zeros(
                (1, query_len, hidden_size),
                self.embed_tokens.embeddings().dtype(),
                device,
            )
            .map_err(|e| candle_to_ocr_inference(self.model_name, "graph hidden input", e))?,
            positions: Tensor::zeros((3, 1, query_len), DType::I64, device)
                .map_err(|e| candle_to_ocr_inference(self.model_name, "graph position input", e))?,
            query_lengths: Tensor::new(&[0u32, query_len as u32], device)
                .map_err(|e| candle_to_ocr_inference(self.model_name, "graph query lengths", e))?,
            kv_lengths: CudaGraphKvLengths::new(query_len, device)
                .map_err(|e| candle_to_ocr_inference(self.model_name, "graph KV lengths", e))?,
            lm_head: lm_head.clone(),
        };
        let graph = capture_decoder_graph(
            device,
            self.model_name,
            self,
            inputs,
            Self::decode_graph_body,
            cache_len,
        )?;
        self.clear_kv_cache();
        *self.decode_graph.borrow_mut() = Some(graph);
        Ok(())
    }

    /// The captured decode step: a bare `fn` so the captured region can
    /// only read model-owned weights and the registered inputs.
    #[cfg(feature = "cuda")]
    fn decode_graph_body(this: &Self, inputs: &CudaGraphInputs) -> Result<Vec<Tensor>, Error> {
        let hidden = this.forward_dynamic(
            &inputs.hidden,
            &inputs.positions,
            &inputs.query_lengths,
            inputs.kv_lengths.tensor(),
        )?;
        let logits = this.project_logits(&hidden, &inputs.lm_head)?;
        Ok(vec![logits])
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
        if inputs_embeds.shape() != captured.inputs.hidden.shape()
            || position_ids.shape() != captured.inputs.positions.shape()
        {
            return Ok(None);
        }
        captured
            .inputs
            .hidden
            .slice_set(inputs_embeds, 0, 0)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "copy graph hidden", e))?;
        captured
            .inputs
            .positions
            .slice_set(position_ids, 0, 0)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "copy graph positions", e))?;
        captured
            .inputs
            .kv_lengths
            .update(kv_len)
            .map_err(|e| candle_to_ocr_inference(self.model_name, "update graph KV lengths", e))?;
        captured
            .graph
            .launch()
            .map_err(|e| cuda_graph_error(self.model_name, "launch decoder CUDA graph", e))?;
        for layer in &self.layers {
            layer.set_kv_cache_len(kv_len)?;
        }
        // A borrowed alias is safe because both callers (the MinerU2.5 and
        // NaviDC-OCR decode loops) consume the logits with
        // `select_next_token` and drop them before the next replay
        // overwrites this buffer.
        Ok(Some(captured.outputs[0].clone()))
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

    #[cfg(feature = "cuda")]
    fn kv_cache_len(&self) -> usize {
        let len = self.layers.first().map_or(0, |layer| layer.kv_cache_len());
        debug_assert!(self.layers.iter().all(|layer| layer.kv_cache_len() == len));
        len
    }

    pub fn token_embedding_weight(&self) -> Tensor {
        self.embed_tokens.embeddings().clone()
    }

    pub fn clear_kv_cache(&self) {
        for layer in &self.layers {
            layer.clear_kv_cache();
        }
    }

    /// Whether the decode graph is currently captured — asserted by the GPU
    /// self-checks.
    #[cfg(all(test, feature = "cuda"))]
    fn decode_graph_captured(&self) -> bool {
        self.decode_graph.borrow().is_some()
    }
}

#[cfg(feature = "cuda")]
impl Drop for Qwen2VlTextModel {
    fn drop(&mut self) {
        // A cached graph must go through dispose: plainly dropping it returns
        // graph-bound buffers to the allocator and poisons it.
        self.invalidate_cuda_graph();
    }
}

#[cfg(test)]
mod tests {
    use super::{Qwen2VlTextConfig, Qwen2VlTextModel};
    #[cfg(feature = "cuda")]
    use candle_core::IndexOp;
    use candle_core::{DType, Device, Tensor};
    #[cfg(feature = "cuda")]
    use candle_nn::Module;
    use candle_nn::VarBuilder;

    /// One knob combination: the MinerU2.5 tuning (bias, no head norm) and
    /// the NaviDC-OCR tuning (no bias, q_norm/k_norm).
    #[derive(Clone, Copy)]
    struct Tuning {
        name: &'static str,
        attention_bias: bool,
        qk_head_norm: bool,
    }

    const MINERU_TUNING: Tuning = Tuning {
        name: "MinerU2.5",
        attention_bias: true,
        qk_head_norm: false,
    };

    const NAVIDC_TUNING: Tuning = Tuning {
        name: "NaviDC-OCR",
        attention_bias: false,
        qk_head_norm: true,
    };

    /// CPU-runnable config small enough for a fast unit test: two layers,
    /// four heads of width eight.
    fn unit_config(tuning: Tuning) -> Qwen2VlTextConfig {
        Qwen2VlTextConfig {
            model_name: tuning.name,
            vocab_size: 64,
            hidden_size: 32,
            intermediate_size: 64,
            num_hidden_layers: 2,
            num_attention_heads: 4,
            num_key_value_heads: 2,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 512,
            head_dim: 8,
            mrope_section: vec![2, 1, 1],
            attention_bias: tuning.attention_bias,
            qk_head_norm: tuning.qk_head_norm,
            graph_disable_env: "OAR_UNIT_TEST_DISABLE_CUDA_GRAPH",
            decode_cache_len: 512,
        }
    }

    /// The full error text including every wrapped cause, so assertions can
    /// match the candle message naming a missing weight.
    fn error_chain(err: &dyn std::error::Error) -> String {
        let mut text = err.to_string();
        let mut source = err.source();
        while let Some(err) = source {
            text.push_str(": ");
            text.push_str(&err.to_string());
            source = err.source();
        }
        text
    }

    /// Weight map holding exactly the names a tuning loads: bias vectors on
    /// q/k/v only when `attention_bias`, `q_norm`/`k_norm` only when
    /// `qk_head_norm`.
    fn unit_var_map(
        cfg: &Qwen2VlTextConfig,
        include_bias: bool,
        include_qk_norm: bool,
    ) -> std::collections::HashMap<String, Tensor> {
        let device = Device::Cpu;
        let mut tensors = std::collections::HashMap::new();
        let zeros = |shape: Vec<usize>| Tensor::zeros(shape, DType::F32, &device).unwrap();
        tensors.insert(
            "embed_tokens.weight".into(),
            zeros(vec![cfg.vocab_size, cfg.hidden_size]),
        );
        tensors.insert("norm.weight".into(), zeros(vec![cfg.hidden_size]));
        for layer in 0..cfg.num_hidden_layers {
            let prefix = format!("layers.{layer}");
            for proj in ["q_proj", "k_proj", "v_proj"] {
                let rows = if proj == "q_proj" {
                    cfg.num_attention_heads
                } else {
                    cfg.num_key_value_heads
                } * cfg.head_dim;
                tensors.insert(
                    format!("{prefix}.self_attn.{proj}.weight"),
                    zeros(vec![rows, cfg.hidden_size]),
                );
                if include_bias {
                    tensors.insert(format!("{prefix}.self_attn.{proj}.bias"), zeros(vec![rows]));
                }
            }
            tensors.insert(
                format!("{prefix}.self_attn.o_proj.weight"),
                zeros(vec![
                    cfg.hidden_size,
                    cfg.num_attention_heads * cfg.head_dim,
                ]),
            );
            if include_qk_norm {
                for norm in ["q_norm", "k_norm"] {
                    tensors.insert(
                        format!("{prefix}.self_attn.{norm}.weight"),
                        zeros(vec![cfg.head_dim]),
                    );
                }
            }
            for norm in ["input_layernorm", "post_attention_layernorm"] {
                tensors.insert(
                    format!("{prefix}.{norm}.weight"),
                    zeros(vec![cfg.hidden_size]),
                );
            }
            for proj in ["gate_proj", "up_proj"] {
                tensors.insert(
                    format!("{prefix}.mlp.{proj}.weight"),
                    zeros(vec![cfg.intermediate_size, cfg.hidden_size]),
                );
            }
            tensors.insert(
                format!("{prefix}.mlp.down_proj.weight"),
                zeros(vec![cfg.hidden_size, cfg.intermediate_size]),
            );
        }
        tensors
    }

    /// Both knob combinations load exactly their own weight names and run a
    /// forward pass; the missing-name cases prove the bias and head-norm
    /// loads are really gated by the config.
    #[test]
    fn knobs_gate_loaded_weight_names_and_forward_runs() {
        let device = Device::Cpu;
        for tuning in [MINERU_TUNING, NAVIDC_TUNING] {
            let cfg = unit_config(tuning);
            let tensors = unit_var_map(&cfg, tuning.attention_bias, tuning.qk_head_norm);
            let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
            let model = Qwen2VlTextModel::load(&cfg, vb).unwrap();
            let ids = Tensor::zeros((1, 4), DType::U32, &device).unwrap();
            let embeds = model.embed(&ids).unwrap();
            let axis = Tensor::arange(0i64, 4, &device).unwrap();
            let positions = Tensor::cat(&[&axis, &axis, &axis], 0)
                .unwrap()
                .reshape((3, 1, 4))
                .unwrap();
            let hidden = model.forward(&embeds, &positions, None).unwrap();
            assert_eq!(hidden.dims(), &[1, 4, cfg.hidden_size]);
        }

        // The MinerU2.5 tuning (bias) must reject a checkpoint whose q/k/v
        // biases are missing, and the NaviDC-OCR tuning (qk_head_norm) one
        // whose q_norm/k_norm are missing; the error names the exact
        // weight that failed to load, so the cases cannot pass by failing
        // somewhere earlier.
        let bias_cfg = unit_config(MINERU_TUNING);
        let tensors = unit_var_map(&bias_cfg, false, false);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let err = match Qwen2VlTextModel::load(&bias_cfg, vb) {
            Ok(_) => panic!("bias-negative case loaded a checkpoint missing q_proj.bias"),
            Err(err) => err,
        };
        let chain = error_chain(&err);
        assert!(
            chain.contains("q_proj.bias"),
            "bias-negative case failed with unexpected error: {chain}"
        );

        let qk_cfg = unit_config(NAVIDC_TUNING);
        let tensors = unit_var_map(&qk_cfg, false, false);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let err = match Qwen2VlTextModel::load(&qk_cfg, vb) {
            Ok(_) => panic!("qk-norm-negative case loaded a checkpoint missing q_norm.weight"),
            Err(err) => err,
        };
        let chain = error_chain(&err);
        assert!(
            chain.contains("q_norm.weight"),
            "qk-norm-negative case failed with unexpected error: {chain}"
        );
    }

    /// GPU self-check for the decode graph lifecycle, in BF16 so the graph
    /// captures: a short prompt captures a small bucket, a longer prompt
    /// forces a same-process re-capture, the larger graph then covers the
    /// short prompt again, and a second instance captures while the first
    /// graph is alive. Every graphed run must match plain eager decoding.
    /// Skips without a CUDA device; opt in with
    /// `OAR_MINERU_GPU_SELFTEST=1` / `OAR_NAVIDC_GPU_SELFTEST=1`.
    #[test]
    fn cuda_decode_graph_recaptures_and_matches_eager_for_both_tunings() {
        #[cfg(feature = "cuda")]
        {
            use candle_nn::Linear;
            let tuning = if std::env::var_os("OAR_MINERU_GPU_SELFTEST").is_some() {
                Some(MINERU_TUNING)
            } else if std::env::var_os("OAR_NAVIDC_GPU_SELFTEST").is_some() {
                Some(NAVIDC_TUNING)
            } else {
                None
            };
            let Some(tuning) = tuning else {
                eprintln!(
                    "skipping: neither OAR_MINERU_GPU_SELFTEST nor OAR_NAVIDC_GPU_SELFTEST is set"
                );
                return;
            };
            let Ok(device) = Device::new_cuda(0) else {
                eprintln!("skipping: no CUDA device");
                return;
            };
            let cfg = gpu_selftest_config(tuning);
            let tensors = gpu_selftest_var_map(&cfg, &device);
            let make_vb = || VarBuilder::from_tensors(tensors.clone(), DType::BF16, &device);
            let model = Qwen2VlTextModel::load(&cfg, make_vb()).unwrap();
            let lm_head = Linear::new(
                make_vb()
                    .get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")
                    .unwrap(),
                None,
            );

            let long: Vec<u32> = (0..4200).map(|i| 10 + i % 60).collect();
            let short: Vec<u32> = (0..600).map(|i| 10 + i % 60).collect();

            // Small bucket first.
            assert_eq!(
                greedy_eager(&model, &lm_head, &short, 8),
                greedy_graphed(&model, &lm_head, &short, 8),
                "small-bucket graph decode must match eager"
            );
            assert!(model.decode_graph_captured());

            // Growth past the captured bucket forces a re-capture in the
            // same process — the dangling-read failure mode.
            assert_eq!(
                greedy_eager(&model, &lm_head, &long, 8),
                greedy_graphed(&model, &lm_head, &long, 8),
                "re-captured graph decode must match eager"
            );

            // The larger graph now covers the short prompt again (reuse).
            assert_eq!(
                greedy_eager(&model, &lm_head, &short, 8),
                greedy_graphed(&model, &lm_head, &short, 8),
                "reused graph decode must match eager"
            );

            // A second instance capturing while the first graph is alive.
            let second = Qwen2VlTextModel::load(&cfg, make_vb()).unwrap();
            assert_eq!(
                greedy_eager(&second, &lm_head, &long, 8),
                greedy_graphed(&second, &lm_head, &long, 8),
                "second-instance graph decode must match eager"
            );
        }
        #[cfg(not(feature = "cuda"))]
        eprintln!("skipping: built without the cuda feature");
    }

    /// Text-tower shape mirroring the NaviDC-OCR production config (head
    /// dim 128, mrope 16/24/24, four layers) at reduced width.
    #[cfg(feature = "cuda")]
    fn gpu_selftest_config(tuning: Tuning) -> Qwen2VlTextConfig {
        Qwen2VlTextConfig {
            model_name: tuning.name,
            vocab_size: 32768,
            hidden_size: 2048,
            intermediate_size: 6144,
            num_hidden_layers: 4,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 262_144,
            head_dim: 128,
            mrope_section: vec![16, 24, 24],
            attention_bias: tuning.attention_bias,
            qk_head_norm: tuning.qk_head_norm,
            graph_disable_env: "OAR_VL_DISABLE_CUDA_GRAPH",
            decode_cache_len: 16_384,
        }
    }

    #[cfg(feature = "cuda")]
    fn gpu_selftest_var_map(
        cfg: &Qwen2VlTextConfig,
        device: &Device,
    ) -> std::collections::HashMap<String, Tensor> {
        let mut tensors = std::collections::HashMap::new();
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim;
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
                .to_dtype(DType::BF16)
                .unwrap();
            tensors.insert(name, tensor);
        };
        put(
            &mut tensors,
            "embed_tokens.weight".into(),
            vec![cfg.vocab_size, h],
        );
        put(&mut tensors, "norm.weight".into(), vec![h]);
        put(
            &mut tensors,
            "lm_head.weight".into(),
            vec![cfg.vocab_size, h],
        );
        for layer in 0..cfg.num_hidden_layers {
            let prefix = format!("layers.{layer}");
            for (proj, heads) in [
                ("q_proj", cfg.num_attention_heads),
                ("k_proj", cfg.num_key_value_heads),
                ("v_proj", cfg.num_key_value_heads),
            ] {
                let rows = heads * head_dim;
                put(
                    &mut tensors,
                    format!("{prefix}.self_attn.{proj}.weight"),
                    vec![rows, h],
                );
                if cfg.attention_bias {
                    put(
                        &mut tensors,
                        format!("{prefix}.self_attn.{proj}.bias"),
                        vec![rows],
                    );
                }
            }
            put(
                &mut tensors,
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![h, cfg.num_attention_heads * head_dim],
            );
            if cfg.qk_head_norm {
                for norm in ["q_norm", "k_norm"] {
                    put(
                        &mut tensors,
                        format!("{prefix}.self_attn.{norm}.weight"),
                        vec![head_dim],
                    );
                }
            }
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

    /// Greedy decode driven by plain `forward` calls: never captures or
    /// replays the decode graph, so it is the eager reference.
    #[cfg(feature = "cuda")]
    fn greedy_eager(
        model: &Qwen2VlTextModel,
        lm_head: &candle_nn::Linear,
        ids: &[u32],
        steps: usize,
    ) -> Vec<u32> {
        let device = model.embed_tokens.embeddings().device();
        let seq_len = ids.len();
        let token_ids = Tensor::from_vec(ids.to_vec(), (1, seq_len), device).unwrap();
        let embeds = model.embed(&token_ids).unwrap();
        let positions = text_position_ids(seq_len, device);
        model.clear_kv_cache();
        let mut logits = step_logits(model, lm_head, &embeds, &positions, seq_len);
        let mut out = Vec::new();
        for step in 0..steps {
            let best = argmax_of(&logits);
            out.push(best as u32);
            let embed = embed_token(model, device, best as u32);
            let pos = decode_position(seq_len + step, device);
            let hidden = model.forward(&embed, &pos, None).unwrap();
            logits = project(lm_head, &hidden);
        }
        out
    }

    /// Greedy decode through `prepare_ar_cuda_graph` +
    /// `forward_decode_logits`, i.e. exactly the production graph path.
    #[cfg(feature = "cuda")]
    fn greedy_graphed(
        model: &Qwen2VlTextModel,
        lm_head: &candle_nn::Linear,
        ids: &[u32],
        steps: usize,
    ) -> Vec<u32> {
        let device = model.embed_tokens.embeddings().device();
        let seq_len = ids.len();
        let token_ids = Tensor::from_vec(ids.to_vec(), (1, seq_len), device).unwrap();
        let embeds = model.embed(&token_ids).unwrap();
        let positions = text_position_ids(seq_len, device);
        model.clear_kv_cache();
        model
            .prepare_ar_cuda_graph(seq_len, steps, lm_head)
            .unwrap();
        assert!(
            model.decode_graph_captured(),
            "decode graph did not capture (dtype gate?)"
        );
        let mut logits = step_logits(model, lm_head, &embeds, &positions, seq_len);
        let mut out = Vec::new();
        for step in 0..steps {
            let best = argmax_of(&logits);
            out.push(best as u32);
            let embed = embed_token(model, device, best as u32);
            let pos = decode_position(seq_len + step, device);
            logits = model
                .forward_decode_logits(&embed, &pos, None, lm_head)
                .unwrap();
        }
        out
    }

    #[cfg(feature = "cuda")]
    fn step_logits(
        model: &Qwen2VlTextModel,
        lm_head: &candle_nn::Linear,
        embeds: &Tensor,
        positions: &Tensor,
        seq_len: usize,
    ) -> Tensor {
        let hidden = model.forward(embeds, positions, None).unwrap();
        let last = hidden
            .i((0, seq_len - 1, ..))
            .unwrap()
            .contiguous()
            .unwrap();
        lm_head.forward(&last.unsqueeze(0).unwrap()).unwrap()
    }

    #[cfg(feature = "cuda")]
    fn project(lm_head: &candle_nn::Linear, hidden: &Tensor) -> Tensor {
        let last = hidden.i((0, 0, ..)).unwrap().contiguous().unwrap();
        lm_head.forward(&last.unsqueeze(0).unwrap()).unwrap()
    }

    #[cfg(feature = "cuda")]
    fn embed_token(model: &Qwen2VlTextModel, device: &Device, token: u32) -> Tensor {
        let ids = Tensor::from_vec(vec![token], (1, 1), device).unwrap();
        model.embed(&ids).unwrap()
    }

    #[cfg(feature = "cuda")]
    fn decode_position(position: usize, device: &Device) -> Tensor {
        Tensor::from_vec(vec![position as i64; 3], (3, 1, 1), device).unwrap()
    }

    #[cfg(feature = "cuda")]
    fn text_position_ids(seq_len: usize, device: &Device) -> Tensor {
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
}
