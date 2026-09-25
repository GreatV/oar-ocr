//! Qwen2.5 text tower for NaviDC-OCR.
//!
//! Structured after `mineru::text` (the in-repo Qwen2-VL decoder) with the two
//! Qwen2.5 differences: attention projections carry **no bias** and each head
//! is normalised by `q_norm`/`k_norm` (RMSNorm over `head_dim`, applied after
//! the projection and before RoPE, see `modeling_naviocr.py` lines 679-686),
//! and `head_dim` comes from the explicit config field (128) rather than
//! `hidden_size / num_attention_heads` (64).

use super::config::NaviDcConfig;
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
use crate::utils::{candle_to_ocr_inference, candle_to_ocr_processing, rotate_half};
#[cfg(feature = "cuda")]
use candle_core::DType;
use candle_core::{IndexOp, Tensor};
use candle_nn::{Embedding, Linear, Module, VarBuilder, embedding, linear_no_bias, rms_norm};
use std::cell::RefCell;
use std::sync::Arc;

#[cfg(feature = "cuda")]
const NAVIDC_DECODE_CACHE_LEN: usize = 16_384;

fn apply_multimodal_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mrope_section: &[usize],
) -> Result<(Tensor, Tensor), Error> {
    let cos = select_rope_sections(cos, mrope_section, 3)?;
    let sin = select_rope_sections(sin, mrope_section, 3)?;

    let q_mul = q.broadcast_mul(&cos).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            "NaviDC-OCR: mrope q*cos failed",
            e,
        )
    })?;
    let q_half = rotate_half(q)?;
    let q_half_mul = q_half.broadcast_mul(&sin).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            "NaviDC-OCR: mrope rotate_half(q)*sin failed",
            e,
        )
    })?;
    let q_rot = (&q_mul + &q_half_mul).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            "NaviDC-OCR: mrope apply on q failed",
            e,
        )
    })?;

    let k_mul = k.broadcast_mul(&cos).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            "NaviDC-OCR: mrope k*cos failed",
            e,
        )
    })?;
    let k_half = rotate_half(k)?;
    let k_half_mul = k_half.broadcast_mul(&sin).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            "NaviDC-OCR: mrope rotate_half(k)*sin failed",
            e,
        )
    })?;
    let k_rot = (&k_mul + &k_half_mul).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            "NaviDC-OCR: mrope apply on k failed",
            e,
        )
    })?;

    Ok((q_rot, k_rot))
}

#[derive(Debug, Clone)]
struct NaviDcMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl NaviDcMlp {
    fn load(cfg: &NaviDcConfig, vb: VarBuilder) -> Result<Self, Error> {
        let gate_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("mlp.gate_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load gate_proj", e))?;
        let up_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp.up_proj"))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load up_proj", e))?;
        let down_proj = linear_no_bias(
            cfg.intermediate_size,
            cfg.hidden_size,
            vb.pp("mlp.down_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load down_proj", e))?;
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
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "mlp gate_proj", e))?;
        let gate = candle_nn::ops::silu(&gate)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "mlp silu", e))?;
        let up = self
            .up_proj
            .forward(xs)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "mlp up_proj", e))?;
        let prod =
            (&gate * &up).map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "mlp gate*up", e))?;
        self.down_proj
            .forward(&prod)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "mlp down_proj", e))
    }
}

#[derive(Debug)]
struct NaviDcAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    /// Per-head RMSNorm applied to Q after projection, before RoPE.
    q_norm: candle_nn::RmsNorm,
    /// Per-head RMSNorm applied to K after projection, before RoPE.
    k_norm: candle_nn::RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
    mrope_section: Vec<usize>,
    kv_cache: RefCell<TrimmableKvCache>,
}

impl NaviDcAttention {
    fn load(cfg: &NaviDcConfig, vb: VarBuilder) -> Result<Self, Error> {
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            return Err(Error::Config {
                message: format!(
                    "NaviDC-OCR: num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                    cfg.num_attention_heads, cfg.num_key_value_heads
                ),
            });
        }
        let head_dim = cfg.head_dim()?;
        let q_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_attention_heads * head_dim,
            vb.pp("self_attn.q_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load q_proj", e))?;
        let k_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("self_attn.k_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load k_proj", e))?;
        let v_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("self_attn.v_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load v_proj", e))?;
        let o_proj = linear_no_bias(
            cfg.num_attention_heads * head_dim,
            cfg.hidden_size,
            vb.pp("self_attn.o_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load o_proj", e))?;
        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("self_attn.q_norm"))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load q_norm", e))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("self_attn.k_norm"))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load k_norm", e))?;

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
            mrope_section: cfg.rope_scaling.mrope_section.clone(),
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
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "attn hidden_states dims3", e))?;

        // Qwen2.5 normalises each head on the (b, s, h, head_dim) view before
        // the transpose and RoPE application.
        let q = self
            .q_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "attn q_proj", e))?
            .reshape((b, seq_len, self.num_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "attn q reshape", e))?;
        let q = self
            .q_norm
            .forward(&q)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "attn q_norm", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "attn q transpose", e))?;

        let k = self
            .k_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "attn k_proj", e))?
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "attn k reshape", e))?;
        let k = self
            .k_norm
            .forward(&k)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "attn k_norm", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "attn k transpose", e))?;

        let v = self
            .v_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "attn v_proj", e))?
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "attn v reshape", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "attn v transpose", e))?;

        let (q, k) = apply_multimodal_rotary_pos_emb(&q, &k, cos, sin, &self.mrope_section)?;
        let k = k
            .contiguous()
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "attn k contiguous", e))?;
        let v = v
            .contiguous()
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "attn v contiguous", e))?;

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
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "attn hidden_states dims3", e))?;
        let (q, k, v) = self.project_qkv(hidden_states, cos, sin)?;

        let (k, v) = self
            .kv_cache
            .borrow_mut()
            .append(&k, &v)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "attn kv_cache append", e))?;
        let is_causal = attention_mask.is_none();
        let flash = if b == 1 {
            flash_attention(&q, &k, &v, self.scaling, seq_len > 1)
                .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "flash attention", e))?
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
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "grouped-query attention", e))?,
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
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "attn output transpose", e))?
            .reshape((batch, seq_len, self.num_heads * self.head_dim))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "attn output reshape", e))?;

        self.o_proj
            .forward(&attn_output)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "attn o_proj", e))
    }

    #[cfg(feature = "cuda")]
    fn prepare_dynamic_cache(&self, query_len: usize, cache_len: usize) -> Result<(), Error> {
        let template = Tensor::zeros(
            (1, self.num_kv_heads, query_len, self.head_dim),
            self.k_proj.weight().dtype(),
            self.k_proj.weight().device(),
        )
        .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "dynamic KV template", e))?;
        self.kv_cache
            .borrow_mut()
            .initialize_storage_with_capacity(&template, cache_len)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "initialize dynamic KV", e))
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
            candle_to_ocr_inference("NaviDC-OCR", "dynamic attention hidden shape", e)
        })?;
        if batch != 1 {
            return Err(Error::Config {
                message: "NaviDC-OCR CUDA-graph attention requires batch size 1".to_string(),
            });
        }
        let (q, k, v) = self.project_qkv(hidden_states, cos, sin)?;
        let cache = self.kv_cache.borrow();
        let cache_len = cache.storage_capacity();
        let (cache_k, cache_v) = cache.storage().ok_or_else(|| Error::Config {
            message: "NaviDC-OCR dynamic KV storage is not initialized".to_string(),
        })?;
        drop(cache);
        let append = DynamicKvAppend {
            query_len,
            cache_len,
        };
        cache_k
            .inplace_op3(&k, kv_lengths, &append)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "dynamic key cache append", e))?;
        cache_v
            .inplace_op3(&v, kv_lengths, &append)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "dynamic value cache append", e))?;

        let q = q
            .squeeze(0)
            .and_then(|q| q.transpose(0, 1))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "dynamic Q layout", e))?;
        let cache_k = cache_k
            .squeeze(0)
            .and_then(|k| k.transpose(0, 1))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "dynamic K layout", e))?;
        let cache_v = cache_v
            .squeeze(0)
            .and_then(|v| v.transpose(0, 1))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "dynamic V layout", e))?;
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
        .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "dynamic flash attention", e))?
        .transpose(0, 1)
        .and_then(|attn| attn.unsqueeze(0))
        .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "dynamic attention layout", e))?;
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
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "set dynamic KV length", e))
    }
}

pub struct NaviDcDecoderLayer {
    self_attn: NaviDcAttention,
    mlp: NaviDcMlp,
    input_layernorm: candle_nn::RmsNorm,
    post_attention_layernorm: candle_nn::RmsNorm,
}

impl NaviDcDecoderLayer {
    fn load(cfg: &NaviDcConfig, vb: VarBuilder) -> Result<Self, Error> {
        let self_attn = NaviDcAttention::load(cfg, vb.clone())?;
        let mlp = NaviDcMlp::load(cfg, vb.clone())?;
        let input_layernorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load input_layernorm", e))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )
        .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load post_attention_layernorm", e))?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
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
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "input_layernorm", e))?;
        let hidden_states = self
            .self_attn
            .forward(&hidden_states, cos, sin, attention_mask)?;
        let hidden_states = (&residual + &hidden_states).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: attn residual add failed",
                e,
            )
        })?;

        let residual = hidden_states.clone();
        let hidden_states = self
            .post_attention_layernorm
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "post_attention_layernorm", e))?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        (&residual + &hidden_states).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: mlp residual add failed",
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
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "input_layernorm", e))?;
        let hidden_states =
            self.self_attn
                .forward_dynamic(&hidden_states, cos, sin, query_lengths, kv_lengths)?;
        let hidden_states = (&residual + &hidden_states).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: attn residual add failed",
                e,
            )
        })?;

        let residual = hidden_states.clone();
        let hidden_states = self
            .post_attention_layernorm
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "post_attention_layernorm", e))?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        (&residual + &hidden_states).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: mlp residual add failed",
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

pub struct NaviDcTextModel {
    #[cfg(feature = "cuda")]
    decode_graph: RefCell<Option<DecoderCudaGraph<CudaGraphInputs>>>,
    embed_tokens: Embedding,
    layers: Vec<NaviDcDecoderLayer>,
    norm: candle_nn::RmsNorm,
    rotary_emb: Arc<RotaryEmbedding>,
    // Must stay the last field: it drops last and drains CUDA errors the
    // other fields' frees may stash (see CudaGraphDrainGuard).
    #[cfg(feature = "cuda")]
    _drain_guard: CudaGraphDrainGuard,
}

impl NaviDcTextModel {
    pub fn load(cfg: &NaviDcConfig, vb: VarBuilder) -> Result<Self, Error> {
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load embed_tokens", e))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer_vb = vb.pp(format!("layers.{i}"));
            layers.push(NaviDcDecoderLayer::load(cfg, layer_vb)?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load norm", e))?;
        let head_dim = cfg.head_dim()?;
        let rotary_emb = Arc::new(RotaryEmbedding::new_multi_axis(
            head_dim,
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
            #[cfg(feature = "cuda")]
            _drain_guard,
        })
    }

    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor, Error> {
        self.embed_tokens
            .forward(input_ids)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "embed forward", e))
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
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "norm forward", e))
    }

    fn project_logits(&self, hidden_states: &Tensor, lm_head: &Linear) -> Result<Tensor, Error> {
        lm_head
            .forward(hidden_states)
            .and_then(|logits| logits.i((0, 0, ..)))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "decode LM head", e))
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
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "dynamic norm", e))
    }

    pub(crate) fn prepare_ar_cuda_graph(
        &self,
        prompt_len: usize,
        max_new_tokens: usize,
        lm_head: &Linear,
    ) -> Result<(), Error> {
        if std::env::var_os("OAR_VL_DISABLE_CUDA_GRAPH").is_some()
            || std::env::var_os("OAR_NAVIDC_DISABLE_CUDA_GRAPH").is_some()
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
                decoder_cache_capacity(prompt_len, max_new_tokens, NAVIDC_DECODE_CACHE_LEN)
            else {
                self.invalidate_cuda_graph();
                return Ok(());
            };
            let required = prompt_len
                .saturating_add(max_new_tokens)
                .min(NAVIDC_DECODE_CACHE_LEN);
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
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "graph hidden size", e))?;
        let device = self.embed_tokens.embeddings().device();
        let inputs = CudaGraphInputs {
            hidden: Tensor::zeros(
                (1, query_len, hidden_size),
                self.embed_tokens.embeddings().dtype(),
                device,
            )
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "graph hidden input", e))?,
            positions: Tensor::zeros((3, 1, query_len), DType::I64, device)
                .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "graph position input", e))?,
            query_lengths: Tensor::new(&[0u32, query_len as u32], device)
                .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "graph query lengths", e))?,
            kv_lengths: CudaGraphKvLengths::new(query_len, device)
                .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "graph KV lengths", e))?,
            lm_head: lm_head.clone(),
        };
        let graph = capture_decoder_graph(
            device,
            "NaviDC-OCR",
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
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "copy graph hidden", e))?;
        captured
            .inputs
            .positions
            .slice_set(position_ids, 0, 0)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "copy graph positions", e))?;
        captured
            .inputs
            .kv_lengths
            .update(kv_len)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "update graph KV lengths", e))?;
        captured
            .graph
            .launch()
            .map_err(|e| cuda_graph_error("NaviDC-OCR", "launch decoder CUDA graph", e))?;
        for layer in &self.layers {
            layer.set_kv_cache_len(kv_len)?;
        }
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
impl Drop for NaviDcTextModel {
    fn drop(&mut self) {
        // A cached graph must go through dispose: plainly dropping it returns
        // graph-bound buffers to the allocator and poisons it.
        self.invalidate_cuda_graph();
    }
}

#[cfg(test)]
mod tests {
    #[cfg(feature = "cuda")]
    use super::super::config::NaviDcConfig;
    #[cfg(feature = "cuda")]
    use super::*;

    /// GPU self-check for the decode graph lifecycle, in BF16 so the graph
    /// captures: a short prompt captures a small bucket, a longer prompt
    /// forces a same-process re-capture, the larger graph then covers the
    /// short prompt again, and a second instance captures while the first
    /// graph is alive. Every graphed run must match plain eager decoding.
    /// Skips without a CUDA device; opt in with
    /// `OAR_NAVIDC_GPU_SELFTEST=1`.
    #[test]
    fn cuda_decode_graph_recaptures_and_matches_eager() {
        #[cfg(feature = "cuda")]
        {
            use candle_nn::Linear;
            if std::env::var_os("OAR_NAVIDC_GPU_SELFTEST").is_none() {
                eprintln!("skipping: OAR_NAVIDC_GPU_SELFTEST is not set");
                return;
            }
            let Ok(device) = Device::new_cuda(0) else {
                eprintln!("skipping: no CUDA device");
                return;
            };
            let cfg = production_config();
            let tensors = random_var_map(&cfg, &device, DType::BF16);
            let make_vb =
                || candle_nn::VarBuilder::from_tensors(tensors.clone(), DType::BF16, &device);
            let model = NaviDcTextModel::load(&cfg, make_vb().pp("model")).unwrap();
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
            let second = NaviDcTextModel::load(&cfg, make_vb().pp("model")).unwrap();
            assert_eq!(
                greedy_eager(&second, &lm_head, &long, 8),
                greedy_graphed(&second, &lm_head, &long, 8),
                "second-instance graph decode must match eager"
            );
        }
        #[cfg(not(feature = "cuda"))]
        eprintln!("skipping: built without the cuda feature");
    }

    #[cfg(feature = "cuda")]
    fn production_config() -> NaviDcConfig {
        NaviDcConfig {
            vocab_size: 32768,
            hidden_size: 2048,
            intermediate_size: 6144,
            num_hidden_layers: 4,
            num_attention_heads: 16,
            num_key_value_heads: 8,
            attention_dropout: 0.0,
            rms_norm_eps: 1e-6,
            rope_theta: 1_000_000.0,
            max_position_embeddings: 262_144,
            head_dim: Some(128),
            hidden_act: "silu".to_string(),
            tie_word_embeddings: true,
            bos_token_id: 0,
            eos_token_id: 1,
            pad_token_id: None,
            vision_start_token_id: 2,
            vision_end_token_id: 3,
            vision_token_id: 4,
            image_token_id: 5,
        }
    }

    #[cfg(feature = "cuda")]
    fn random_var_map(
        cfg: &NaviDcConfig,
        device: &Device,
        dtype: DType,
    ) -> std::collections::HashMap<String, Tensor> {
        let mut tensors = std::collections::HashMap::new();
        let h = cfg.hidden_size;
        let head_dim = cfg.head_dim.unwrap();
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
            put(
                &mut tensors,
                format!("{prefix}.self_attn.q_proj.weight"),
                vec![cfg.num_attention_heads * head_dim, h],
            );
            put(
                &mut tensors,
                format!("{prefix}.self_attn.k_proj.weight"),
                vec![cfg.num_key_value_heads * head_dim, h],
            );
            put(
                &mut tensors,
                format!("{prefix}.self_attn.v_proj.weight"),
                vec![cfg.num_key_value_heads * head_dim, h],
            );
            put(
                &mut tensors,
                format!("{prefix}.self_attn.o_proj.weight"),
                vec![h, cfg.num_attention_heads * head_dim],
            );
            put(
                &mut tensors,
                format!("{prefix}.self_attn.q_norm.weight"),
                vec![head_dim],
            );
            put(
                &mut tensors,
                format!("{prefix}.self_attn.k_norm.weight"),
                vec![head_dim],
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

    /// Greedy decode driven by plain `forward` calls: never captures or
    /// replays the decode graph, so it is the eager reference.
    #[cfg(feature = "cuda")]
    fn greedy_eager(
        model: &NaviDcTextModel,
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
            logits = project(model, lm_head, &hidden);
        }
        out
    }

    /// Greedy decode through `prepare_ar_cuda_graph` +
    /// `forward_decode_logits`, i.e. exactly the production graph path.
    #[cfg(feature = "cuda")]
    fn greedy_graphed(
        model: &NaviDcTextModel,
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
        model: &NaviDcTextModel,
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
    fn project(model: &NaviDcTextModel, lm_head: &candle_nn::Linear, hidden: &Tensor) -> Tensor {
        let last = hidden.i((0, 0, ..)).unwrap().contiguous().unwrap();
        lm_head.forward(&last.unsqueeze(0).unwrap()).unwrap()
    }

    #[cfg(feature = "cuda")]
    fn embed_token(model: &NaviDcTextModel, device: &Device, token: u32) -> Tensor {
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
