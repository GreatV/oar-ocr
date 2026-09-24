//! DeepSeek-V2 MoE decoder shared by checkpoints with the
//! `deepseek_vl_v2`-family architecture.
//!
//! Ported from `modeling_deepseekv2.py` (the `use_mla=false` path): plain
//! multi-head attention without biases, standard Llama-style RoPE on plain
//! sequential positions, and a mixture-of-experts feed-forward — a dense MLP
//! on the first `first_k_dense_replace` layers, then `n_routed_experts`
//! routed plus `n_shared_experts` shared experts with greedy top-k softmax
//! routing and no weight normalization. Model crates own tokenization,
//! prompt assembly, and generation; the backbone only maps tensors to
//! tensors.
//!
//! The router reads its top-k back to the host (like the reference
//! `moe_infer` dispatch), so the MoE layers cannot run inside a CUDA graph;
//! graph-backed decoding therefore belongs to dense heads built on this
//! backbone (see the FastMTP draft block below, whose fixed-capacity KV and
//! varlen attention stay fully on-device).

use crate::error::Error;
use crate::runtime::attention::{RotaryEmbedding, flash_attention, scaled_dot_product_attention};
use crate::runtime::cache::TrimmableKvCache;
use crate::runtime::errors::candle_to_ocr_inference;
use crate::runtime::tensor::rotate_half;
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{
    Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_no_bias, rms_norm,
};
use std::cell::RefCell;

#[cfg(feature = "cuda")]
use crate::runtime::cuda::dynamic_kv::DynamicKvAppend;
#[cfg(feature = "cuda")]
use crate::runtime::decoder_graph::{
    CudaGraphDrainGuard, CudaGraphKvLengths, cuda_graph_error, decoder_attention_is_causal,
    decoder_cache_capacity, drop_and_drain, report_stashed_cuda_error, sync_graph_tensor,
};
#[cfg(feature = "cuda")]
use candle_core::Device;

const MODEL_NAME: &str = "DeepSeek-V2";

/// Upper bound on `(tokens, top-k)` pairs routed through the fused `moe_gemm`
/// kernels on the eager path: their token sort is a shared-memory bitonic
/// sort, which stops fitting past a few thousand pairs (measured: 5400 pairs
/// = 32KB fits, 10800 pairs = 64KB exceeds the 48KB default limit).
#[cfg(feature = "cuda")]
const FUSED_MOE_MAX_PAIRS: usize = 4096;

/// Upper bound for graph-backed decode KV buckets, mirroring the context the
/// checkpoints are actually used with.
#[cfg(feature = "cuda")]
const DECODE_CACHE_LEN: usize = 16_384;

fn graphs_disabled() -> bool {
    std::env::var_os("OAR_VL_DISABLE_CUDA_GRAPH").is_some()
        || std::env::var_os("OAR_DEEPSEEK_V2_DISABLE_CUDA_GRAPH").is_some()
}

/// Captured single-token decode step that exports both the logits and the
/// fed token's hidden state, so adaptive speculation can collect the states
/// it needs for a later re-sync without leaving graph replay.
#[cfg(feature = "cuda")]
struct DecodeCudaGraph {
    // The graph owns device pointers into all tensors below; dispose via
    // `dispose` so capture-touched buffers are never returned to the
    // stream-ordered allocator (see SingleTokenDecoderCudaGraph::dispose).
    graph: candle_core::cuda_backend::cudarc::driver::CudaGraph,
    hidden_input: Tensor,
    position_input: Tensor,
    _query_lengths: Tensor,
    kv_lengths: CudaGraphKvLengths,
    logits_output: Tensor,
    hidden_output: Tensor,
    cache_len: usize,
}

#[cfg(feature = "cuda")]
impl DecodeCudaGraph {
    fn dispose(self) {
        let Self {
            graph,
            hidden_input,
            position_input,
            _query_lengths,
            kv_lengths,
            logits_output,
            hidden_output,
            cache_len: _,
        } = self;
        let device = hidden_input.device().clone();
        report_stashed_cuda_error(&device, "decoder CUDA graph disposal");
        drop_and_drain(graph, &device);
        drop_and_drain(hidden_output, &device);
        drop_and_drain(logits_output, &device);
        drop_and_drain(kv_lengths, &device);
        drop_and_drain(_query_lengths, &device);
        drop_and_drain(position_input, &device);
        drop_and_drain(hidden_input, &device);
    }
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for DecodeCudaGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("DecodeCudaGraph")
            .field("cache_len", &self.cache_len)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "cuda")]
struct VerificationCudaGraph {
    // The graph owns device pointers into all tensors below; dispose via
    // `dispose` so capture-touched buffers are never returned to the
    // stream-ordered allocator (see SingleTokenDecoderCudaGraph::dispose).
    graph: candle_core::cuda_backend::cudarc::driver::CudaGraph,
    hidden_input: Tensor,
    position_input: Tensor,
    _query_lengths: Tensor,
    kv_lengths: CudaGraphKvLengths,
    hidden_output: Tensor,
    logits_output: Tensor,
    cache_len: usize,
    query_len: usize,
}

#[cfg(feature = "cuda")]
impl VerificationCudaGraph {
    fn dispose(self) {
        let Self {
            graph,
            hidden_input,
            position_input,
            _query_lengths,
            kv_lengths,
            hidden_output,
            logits_output,
            cache_len: _,
            query_len: _,
        } = self;
        let device = hidden_input.device().clone();
        report_stashed_cuda_error(&device, "CUDA graph disposal");
        drop_and_drain(graph, &device);
        drop_and_drain(logits_output, &device);
        drop_and_drain(hidden_output, &device);
        drop_and_drain(kv_lengths, &device);
        drop_and_drain(_query_lengths, &device);
        drop_and_drain(position_input, &device);
        drop_and_drain(hidden_input, &device);
    }
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for VerificationCudaGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("VerificationCudaGraph")
            .field("query_len", &self.query_len)
            .field("cache_len", &self.cache_len)
            .finish_non_exhaustive()
    }
}

#[cfg(feature = "cuda")]
fn decode_capacity(prompt_len: usize, max_new_tokens: usize) -> Option<usize> {
    decoder_cache_capacity(prompt_len, max_new_tokens, DECODE_CACHE_LEN)
}

fn default_scoring_func() -> String {
    "softmax".to_string()
}

fn default_topk_method() -> String {
    "greedy".to_string()
}

fn default_one() -> usize {
    1
}

fn default_one_f32() -> f32 {
    1.0
}

/// Text-decoder configuration shared by DeepSeek-V2 MoE checkpoints (the
/// flattened decoder fields of the root `config.json`).
#[derive(Debug, Clone, serde::Deserialize)]
pub struct DeepSeekV2TextConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub max_position_embeddings: usize,
    pub eos_token_id: u32,
    pub bos_token_id: u32,
    pub pad_token_id: u32,
    pub first_k_dense_replace: usize,
    pub moe_layer_freq: usize,
    pub n_routed_experts: usize,
    pub n_shared_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    #[serde(default = "default_scoring_func")]
    pub scoring_func: String,
    #[serde(default = "default_topk_method")]
    pub topk_method: String,
    #[serde(default)]
    pub norm_topk_prob: bool,
    #[serde(default = "default_one_f32")]
    pub routed_scaling_factor: f32,
    #[serde(default = "default_one")]
    pub n_group: usize,
    #[serde(default = "default_one")]
    pub topk_group: usize,
    #[serde(default)]
    pub use_mla: bool,
    #[serde(default)]
    pub tie_word_embeddings: bool,
    #[serde(default)]
    pub attention_bias: bool,
}

impl DeepSeekV2TextConfig {
    pub fn head_dim(&self) -> Result<usize, Error> {
        if !self.hidden_size.is_multiple_of(self.num_attention_heads) {
            return Err(Error::Config {
                message: format!(
                    "DeepSeek-V2 hidden_size {} must be divisible by num_attention_heads {}",
                    self.hidden_size, self.num_attention_heads
                ),
            });
        }
        Ok(self.hidden_size / self.num_attention_heads)
    }

    /// Invariants this port relies on; `hidden_act` is fixed to SiLU like the
    /// reference checkpoint family.
    pub fn validate(&self) -> Result<(), Error> {
        if self.use_mla {
            return Err(Error::Config {
                message: "DeepSeek-V2 backbone requires use_mla=false (plain MHA decoder)"
                    .to_string(),
            });
        }
        if self.topk_method != "greedy"
            || self.scoring_func != "softmax"
            || self.norm_topk_prob
            || self.routed_scaling_factor != 1.0
            || self.n_group != 1
            || self.topk_group != 1
        {
            return Err(Error::Config {
                message: "DeepSeek-V2 backbone supports the greedy softmax MoE router without normalization only"
                    .to_string(),
            });
        }
        if self.num_experts_per_tok == 0
            || self.n_routed_experts == 0
            || self.n_shared_experts == 0
            || self.num_experts_per_tok > self.n_routed_experts
        {
            return Err(Error::Config {
                message: "DeepSeek-V2 MoE expert counts must be non-zero with top-k <= experts"
                    .to_string(),
            });
        }
        if self.first_k_dense_replace == 0 || self.moe_layer_freq == 0 {
            return Err(Error::Config {
                message: "DeepSeek-V2 first_k_dense_replace and moe_layer_freq must be non-zero"
                    .to_string(),
            });
        }
        for (name, count) in [
            ("hidden_size", self.hidden_size),
            ("intermediate_size", self.intermediate_size),
            ("vocab_size", self.vocab_size),
            ("num_hidden_layers", self.num_hidden_layers),
            ("num_attention_heads", self.num_attention_heads),
            ("num_key_value_heads", self.num_key_value_heads),
            ("max_position_embeddings", self.max_position_embeddings),
        ] {
            if count == 0 {
                return Err(Error::Config {
                    message: format!("DeepSeek-V2 {name} must be non-zero"),
                });
            }
        }
        if self.num_key_value_heads != self.num_attention_heads {
            return Err(Error::Config {
                message: format!(
                    "DeepSeek-V2 decoder is multi-head attention; num_key_value_heads ({}) must equal num_attention_heads ({})",
                    self.num_key_value_heads, self.num_attention_heads
                ),
            });
        }
        self.head_dim()?;
        if !self.rope_theta.is_finite() || self.rope_theta <= 0.0 {
            return Err(Error::Config {
                message: format!(
                    "DeepSeek-V2 rope_theta must be finite and positive, got {}",
                    self.rope_theta
                ),
            });
        }
        Ok(())
    }
}

#[derive(Debug)]
struct DeepSeekV2Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    heads: usize,
    head_dim: usize,
    scaling: f64,
    kv_cache: RefCell<TrimmableKvCache>,
}

impl DeepSeekV2Attention {
    fn load(cfg: &DeepSeekV2TextConfig, vb: VarBuilder) -> Result<Self, Error> {
        let head_dim = cfg.head_dim()?;
        let q_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_attention_heads * head_dim,
            vb.pp("q_proj"),
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load q_proj", e))?;
        let k_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("k_proj"),
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load k_proj", e))?;
        let v_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("v_proj"),
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load v_proj", e))?;
        let o_proj = linear_no_bias(
            cfg.num_attention_heads * head_dim,
            cfg.hidden_size,
            vb.pp("o_proj"),
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load o_proj", e))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            heads: cfg.num_attention_heads,
            head_dim,
            scaling: 1.0 / (head_dim as f64).sqrt(),
            kv_cache: RefCell::new(TrimmableKvCache::new(2, cfg.max_position_embeddings)),
        })
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
        let apply_rope = |tensor: &Tensor| -> Result<Tensor, Error> {
            let rotated = rotate_half(tensor)?;
            tensor
                .broadcast_mul(cos)
                .and_then(|lhs| rotated.broadcast_mul(sin).and_then(|rhs| &lhs + &rhs))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "apply RoPE", e))
        };
        let project = |linear: &Linear| -> Result<Tensor, Error> {
            linear
                .forward(hidden_states)
                .and_then(|x| x.reshape((batch, seq_len, self.heads, self.head_dim)))
                .and_then(|x| x.transpose(1, 2))
                .and_then(|x| x.contiguous())
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention projection", e))
        };
        let q = apply_rope(&project(&self.q_proj)?)?;
        let k = apply_rope(&project(&self.k_proj)?)?;
        let v = project(&self.v_proj)?;
        Ok((q, k, v))
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

        let flash = if batch == 1 {
            flash_attention(&q, &k, &v, self.scaling, seq_len > 1)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "flash attention", e))?
        } else {
            None
        };
        let output = match flash {
            Some(output) => output,
            None => scaled_dot_product_attention(
                &q,
                &k,
                &v,
                attention_mask,
                self.scaling,
                attention_mask.is_none(),
            )
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention", e))?,
        };
        self.project_attention_output(&output, batch, seq_len)
    }

    fn project_attention_output(
        &self,
        attn_output: &Tensor,
        batch: usize,
        seq_len: usize,
    ) -> Result<Tensor, Error> {
        let attn_output = attn_output
            .transpose(1, 2)
            .and_then(|x| x.reshape((batch, seq_len, self.heads * self.head_dim)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention output layout", e))?;
        self.o_proj
            .forward(&attn_output)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "output projection", e))
    }

    #[cfg(feature = "cuda")]
    fn prepare_dynamic_cache(&self, query_len: usize, cache_len: usize) -> Result<(), Error> {
        let template = Tensor::zeros(
            (1, self.heads, query_len, self.head_dim),
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
    /// varlen flash attention over `[0, kv_len]`. Dense blocks only — the
    /// MoE layers cannot run under graph capture (see the module docs).
    #[cfg(feature = "cuda")]
    fn forward_dynamic(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        query_lengths: &Tensor,
        kv_lengths: &Tensor,
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

        let q = q
            .squeeze(0)
            .and_then(|q| q.transpose(0, 1))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "dynamic Q layout", e))?;
        let cache_k = cache_k
            .squeeze(0)
            .and_then(|k| k.transpose(0, 1))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "dynamic K layout", e))?;
        let cache_v = cache_v
            .squeeze(0)
            .and_then(|v| v.transpose(0, 1))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "dynamic V layout", e))?;
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
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "dynamic flash attention", e))?
        .transpose(0, 1)
        .and_then(|attn| attn.unsqueeze(0))
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "dynamic attention layout", e))?;
        self.project_attention_output(&attn, batch, query_len)
    }

    fn clear_cache(&self) {
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
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "set dynamic KV length", e))
    }

    fn trim_kv_cache(&self, len: usize) -> Result<(), Error> {
        self.kv_cache
            .borrow_mut()
            .trim_to(len)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "trim KV cache", e))
    }
}

#[derive(Debug)]
struct DenseMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl DenseMlp {
    fn load(hidden_size: usize, intermediate_size: usize, vb: VarBuilder) -> Result<Self, Error> {
        Ok(Self {
            gate_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("gate_proj"))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load gate_proj", e))?,
            up_proj: linear_no_bias(hidden_size, intermediate_size, vb.pp("up_proj"))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load up_proj", e))?,
            down_proj: linear_no_bias(intermediate_size, hidden_size, vb.pp("down_proj"))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load down_proj", e))?,
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
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MLP product", e))?,
            )
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MLP down", e))
    }
}

/// `DeepseekV2MoE`: greedy top-k softmax routing over the routed experts plus
/// the always-on shared expert block. The gate runs in f32 like the
/// reference; routing stays on-device (top-k by argsort, one small router-id
/// readback on the generic path, fused `moe_gemm` kernels on CUDA
/// half-precision) so the layer is CUDA-graph safe and never streams expert
/// outputs through the host.
#[derive(Debug)]
struct MoeFeedForward {
    gate: Tensor,
    /// Stacked `[gate; up]` expert weights, `(E, 2I, H)`.
    gate_up_w: Tensor,
    /// Stacked down-projection weights, `(E, H, I)`.
    down_w: Tensor,
    shared_experts: DenseMlp,
    top_k: usize,
    n_experts: usize,
    intermediate: usize,
}

impl MoeFeedForward {
    fn load(cfg: &DeepSeekV2TextConfig, vb: VarBuilder) -> Result<Self, Error> {
        let gate = vb
            .pp("gate")
            .get((cfg.n_routed_experts, cfg.hidden_size), "weight")
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load MoE gate", e))?;
        let experts_vb = vb.pp("experts");
        let mut gate_up: Vec<Tensor> = Vec::with_capacity(cfg.n_routed_experts);
        let mut down: Vec<Tensor> = Vec::with_capacity(cfg.n_routed_experts);
        for index in 0..cfg.n_routed_experts {
            let expert = experts_vb.pp(index);
            let gate_proj = expert
                .get(
                    (cfg.moe_intermediate_size, cfg.hidden_size),
                    "gate_proj.weight",
                )
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load expert gate_proj", e))?;
            let up_proj = expert
                .get(
                    (cfg.moe_intermediate_size, cfg.hidden_size),
                    "up_proj.weight",
                )
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load expert up_proj", e))?;
            let down_proj = expert
                .get(
                    (cfg.hidden_size, cfg.moe_intermediate_size),
                    "down_proj.weight",
                )
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load expert down_proj", e))?;
            gate_up.push(
                Tensor::cat(&[&gate_proj, &up_proj], 0)
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "stack expert gate_up", e))?,
            );
            down.push(down_proj);
        }
        let refs: Vec<&Tensor> = gate_up.iter().collect();
        let gate_up_w = Tensor::stack(&refs, 0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "stack MoE gate_up", e))?;
        let refs: Vec<&Tensor> = down.iter().collect();
        let down_w = Tensor::stack(&refs, 0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "stack MoE down", e))?;
        let shared_experts = DenseMlp::load(
            cfg.hidden_size,
            cfg.moe_intermediate_size * cfg.n_shared_experts,
            vb.pp("shared_experts"),
        )?;
        Ok(Self {
            gate,
            gate_up_w,
            down_w,
            shared_experts,
            top_k: cfg.num_experts_per_tok,
            n_experts: cfg.n_routed_experts,
            intermediate: cfg.moe_intermediate_size,
        })
    }

    /// Router scores and the on-device greedy top-k (descending, unnormalized
    /// — `norm_topk_prob` is false and `routed_scaling_factor` is 1).
    fn route(&self, flat: &Tensor) -> Result<(Tensor, Tensor), Error> {
        let logits = flat
            .to_dtype(DType::F32)
            .and_then(|x| x.matmul(&self.gate.to_dtype(DType::F32)?.t()?))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE router matmul", e))?;
        let scores = candle_nn::ops::softmax_last_dim(&logits)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE router softmax", e))?;
        let topk_ids = scores
            .arg_sort_last_dim(false)
            .and_then(|order| order.narrow(candle_core::D::Minus1, 0, self.top_k))
            .and_then(|ids| ids.contiguous())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE router top-k", e))?;
        let topk_weights = scores
            .gather(&topk_ids, candle_core::D::Minus1)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE router gather", e))?;
        Ok((topk_ids, topk_weights))
    }

    /// Fused `moe_gemm` path for CUDA half-precision tensors (the kernel only
    /// accepts f16/bf16).
    #[cfg(feature = "cuda")]
    fn forward_fused(
        &self,
        flat: &Tensor,
        topk_ids: &Tensor,
        topk_weights: &Tensor,
        is_prefill: bool,
    ) -> Result<Tensor, Error> {
        use candle_nn::moe::moe_gemm;

        let (tokens, hidden) = flat.dims2()?;
        let (expert_ids, sorted_token_ids) = topk_ids
            .flatten_all()
            .and_then(|flat| flat.sort_last_dim(true))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE token sort", e))?;
        let gate_up = moe_gemm(
            flat,
            &self.gate_up_w,
            &None,
            &sorted_token_ids,
            &expert_ids,
            self.top_k,
            is_prefill,
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "fused MoE gate_up gemm", e))?;
        let gate = gate_up
            .narrow(candle_core::D::Minus1, 0, self.intermediate)
            .and_then(|g| g.contiguous())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "fused MoE gate split", e))?;
        let up = gate_up
            .narrow(candle_core::D::Minus1, self.intermediate, self.intermediate)
            .and_then(|u| u.contiguous())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "fused MoE up split", e))?;
        let gate = candle_nn::ops::silu(&gate)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "fused MoE silu", e))?;
        let down_inputs = (&gate * &up)
            .and_then(|x| x.reshape(((), self.intermediate)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "fused MoE down input", e))?;
        let routed = moe_gemm(
            &down_inputs,
            &self.down_w,
            &Some(topk_weights.clone()),
            &sorted_token_ids,
            &expert_ids,
            self.top_k,
            is_prefill,
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "fused MoE down gemm", e))?
        .reshape((tokens, (), hidden))
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "fused MoE down reshape", e))?
        .to_dtype(DType::F32)
        .and_then(|y| y.sum(candle_core::D::Minus2))
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "fused MoE combine", e))?;
        Ok(routed)
    }

    /// Generic device-resident path: one small router-id readback groups
    /// (token, slot) pairs by expert, each expert runs one dense batch, and
    /// the f32 combine uses `index_add` in expert order — bit-identical to
    /// the reference `moe_infer` accumulation.
    fn forward_indexed(
        &self,
        flat: &Tensor,
        topk_ids: &Tensor,
        topk_weights: &Tensor,
    ) -> Result<Tensor, Error> {
        let (tokens, hidden) = flat
            .dims2()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE input shape", e))?;
        let ids: Vec<u32> = topk_ids
            .flatten_all()
            .and_then(|ids| ids.to_vec1::<u32>())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "read router ids", e))?;
        let weights_flat = topk_weights
            .reshape((tokens * self.top_k,))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "flatten router weights", e))?;
        let device = flat.device();

        let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); self.n_experts];
        for (slot, &id) in ids.iter().enumerate() {
            by_expert[id as usize].push((slot / self.top_k, slot));
        }

        let mut combined = Tensor::zeros((tokens, hidden), DType::F32, device)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE combine buffer", e))?;
        for (expert, slots) in by_expert.iter().enumerate() {
            if slots.is_empty() {
                continue;
            }
            let rows: Vec<u32> = slots.iter().map(|&(token, _)| token as u32).collect();
            let slot_ids: Vec<u32> = slots.iter().map(|&(_, slot)| slot as u32).collect();
            let row_index = Tensor::from_vec(rows, slots.len(), device)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE row index", e))?;
            let slot_index = Tensor::from_vec(slot_ids, slots.len(), device)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE slot index", e))?;
            let routed = flat
                .index_select(&row_index, 0)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE gather rows", e))?;
            let gate_up_w = self
                .gate_up_w
                .i((expert, ..))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE expert weights", e))?;
            let gate_w = gate_up_w
                .narrow(0, 0, self.intermediate)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE gate weight", e))?;
            let up_w = gate_up_w
                .narrow(0, self.intermediate, self.intermediate)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE up weight", e))?;
            let gate = routed
                .matmul(&gate_w.t()?)
                .and_then(|g| candle_nn::ops::silu(&g))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE expert gate", e))?;
            let up = routed
                .matmul(&up_w.t()?)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE expert up", e))?;
            let down_w = self
                .down_w
                .i((expert, ..))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE expert down weight", e))?;
            let expert_out = (&gate * &up)?
                .matmul(&down_w.t()?)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE expert down", e))?;
            let weights = weights_flat
                .index_select(&slot_index, 0)
                .and_then(|w| w.reshape((slots.len(), 1)))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE slot weights", e))?;
            let weighted = expert_out
                .to_dtype(DType::F32)?
                .broadcast_mul(&weights)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE weight outputs", e))?;
            combined = combined
                .index_add(&row_index, &weighted, 0)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE combine", e))?;
        }
        Ok(combined)
    }

    /// Graph-captured decode/verification path: identical math to
    /// [`Self::forward`] via the fused `moe_gemm` kernels, fully on-device.
    /// Only called with the handful of tokens a decode step or verification
    /// block produces — `sort_last_dim` over `tokens * top_k` elements uses
    /// shared memory that stops fitting beyond a few thousand pairs.
    #[cfg(feature = "cuda")]
    fn forward_dynamic(&self, xs: &Tensor) -> Result<Tensor, Error> {
        let (batch, seq, hidden) = xs
            .dims3()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE input shape", e))?;
        let tokens = batch * seq;
        let flat = xs
            .reshape((tokens, hidden))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE flatten", e))?;
        let (topk_ids, topk_weights) = self.route(&flat)?;
        let routed = self.forward_fused(&flat, &topk_ids, &topk_weights, false)?;
        let shared = self.shared_experts.forward(&flat)?;
        let combined = routed
            .to_dtype(xs.dtype())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE routed cast", e))?;
        (&combined + &shared)?
            .reshape((batch, seq, hidden))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE output", e))
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, Error> {
        let (batch, seq, hidden) = xs
            .dims3()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE input shape", e))?;
        let tokens = batch * seq;
        let flat = xs
            .reshape((tokens, hidden))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE flatten", e))?;
        let (topk_ids, topk_weights) = self.route(&flat)?;
        // Small CUDA half-precision batches (a decode step or a short batch
        // step) take the fused kernels — no host round-trip per layer. The
        // token-sort inside `moe_gemm` needs shared memory that stops
        // fitting past a few thousand (token, slot) pairs, so prefill-scale
        // inputs stay on the indexed path.
        #[cfg(feature = "cuda")]
        let routed = if flat.device().is_cuda()
            && matches!(flat.dtype(), DType::BF16 | DType::F16)
            && tokens * self.top_k <= FUSED_MOE_MAX_PAIRS
        {
            self.forward_fused(&flat, &topk_ids, &topk_weights, false)?
        } else {
            self.forward_indexed(&flat, &topk_ids, &topk_weights)?
        };
        #[cfg(not(feature = "cuda"))]
        let routed = self.forward_indexed(&flat, &topk_ids, &topk_weights)?;

        let shared = self.shared_experts.forward(&flat)?;
        let combined = routed
            .to_dtype(xs.dtype())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE routed cast", e))?;
        (&combined + &shared)?
            .reshape((batch, seq, hidden))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE output", e))
    }
}

#[derive(Debug)]
struct DecoderLayer {
    attention: DeepSeekV2Attention,
    mlp: FeedForward,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

#[derive(Debug)]
enum FeedForward {
    Dense(DenseMlp),
    Moe(MoeFeedForward),
}

impl DecoderLayer {
    fn load(cfg: &DeepSeekV2TextConfig, index: usize, vb: VarBuilder) -> Result<Self, Error> {
        let is_moe = cfg.n_routed_experts > 0
            && index >= cfg.first_k_dense_replace
            && index.is_multiple_of(cfg.moe_layer_freq);
        let mlp = if is_moe {
            FeedForward::Moe(MoeFeedForward::load(cfg, vb.pp("mlp"))?)
        } else {
            FeedForward::Dense(DenseMlp::load(
                cfg.hidden_size,
                cfg.intermediate_size,
                vb.pp("mlp"),
            )?)
        };
        Ok(Self {
            attention: DeepSeekV2Attention::load(cfg, vb.pp("self_attn"))?,
            mlp,
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
        let mlp = self.forward_mlp(&normalized)?;
        (&residual + &mlp).map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MLP residual", e))
    }

    fn forward_mlp(&self, normalized: &Tensor) -> Result<Tensor, Error> {
        match &self.mlp {
            FeedForward::Dense(mlp) => mlp.forward(normalized),
            FeedForward::Moe(moe) => moe.forward(normalized),
        }
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
        let normalized = self.input_layernorm.forward(hidden_states)?;
        let mixed =
            self.attention
                .forward_dynamic(&normalized, cos, sin, query_lengths, kv_lengths)?;
        let hidden_states = (&residual + &mixed)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention residual", e))?;
        let residual = hidden_states.clone();
        let normalized = self.post_attention_layernorm.forward(&hidden_states)?;
        // Graph capture forbids host reads, so the MoE layers take the fused
        // on-device kernels here (decode and verification blocks only — a
        // few tokens at a time).
        let mlp = match &self.mlp {
            FeedForward::Dense(mlp) => mlp.forward(&normalized)?,
            FeedForward::Moe(moe) => moe.forward_dynamic(&normalized)?,
        };
        (&residual + &mlp).map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MLP residual", e))
    }

    fn clear_cache(&self) {
        self.attention.clear_cache();
    }

    #[cfg(feature = "cuda")]
    fn prepare_dynamic_cache(&self, query_len: usize, cache_len: usize) -> Result<(), Error> {
        self.attention.prepare_dynamic_cache(query_len, cache_len)
    }

    #[cfg(feature = "cuda")]
    fn kv_cache_len(&self) -> usize {
        self.attention.kv_cache_len()
    }

    #[cfg(feature = "cuda")]
    fn set_kv_cache_len(&self, len: usize) -> Result<(), Error> {
        self.attention.set_kv_cache_len(len)
    }

    fn trim_kv_cache(&self, len: usize) -> Result<(), Error> {
        self.attention.trim_kv_cache(len)
    }

    /// Snapshot the live KV contents (`((k, v, len),)` triples per layer
    /// dimension), so a mid-generation graph capture can restore them after
    /// its warmup runs disturb the shared storage.
    #[cfg(feature = "cuda")]
    fn save_kv_cache(&self) -> Result<(Tensor, Tensor, usize), Error> {
        let cache = self.attention.kv_cache.borrow();
        let len = cache.current_seq_len();
        let Some((storage_k, storage_v)) = cache.storage() else {
            let device = storage_k_device(&cache);
            return Ok((Tensor::new(0f32, &device)?, Tensor::new(0f32, &device)?, 0));
        };
        let k = storage_k
            .narrow(2, 0, len)
            .and_then(|k| k.contiguous())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "snapshot KV keys", e))?;
        let v = storage_v
            .narrow(2, 0, len)
            .and_then(|v| v.contiguous())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "snapshot KV values", e))?;
        drop(cache);
        Ok((k, v, len))
    }

    #[cfg(feature = "cuda")]
    fn restore_kv_cache(&self, (k, v, len): &(Tensor, Tensor, usize)) -> Result<(), Error> {
        if *len == 0 {
            self.attention.clear_cache();
            return Ok(());
        }
        {
            let cache = self.attention.kv_cache.borrow_mut();
            let (storage_k, storage_v) = cache.storage().ok_or_else(|| Error::Config {
                message: format!("{MODEL_NAME} KV storage missing during restore"),
            })?;
            storage_k
                .slice_set(k, 2, 0)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "restore KV keys", e))?;
            storage_v
                .slice_set(v, 2, 0)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "restore KV values", e))?;
        }
        self.attention.set_kv_cache_len(*len)
    }
}

/// Device of a cache's storage for empty snapshots.
#[cfg(feature = "cuda")]
fn storage_k_device(cache: &TrimmableKvCache) -> Device {
    cache
        .storage()
        .map(|(k, _)| k.device().clone())
        .unwrap_or(Device::Cpu)
}

pub(crate) struct DeepSeekV2TextModel {
    #[cfg(feature = "cuda")]
    decode_graph: RefCell<Option<DecodeCudaGraph>>,
    #[cfg(feature = "cuda")]
    verification_graph: RefCell<Option<VerificationCudaGraph>>,
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    rotary: RotaryEmbedding,
    // Must stay the last field: it drops last and drains CUDA errors the
    // other fields' frees may stash (see CudaGraphDrainGuard).
    #[cfg(feature = "cuda")]
    _drain_guard: CudaGraphDrainGuard,
}

impl DeepSeekV2TextModel {
    pub(crate) fn load(cfg: &DeepSeekV2TextConfig, vb: VarBuilder) -> Result<Self, Error> {
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load token embeddings", e))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for index in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::load(
                cfg,
                index,
                vb.pp(format!("layers.{index}")),
            )?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load final norm", e))?;
        let rotary = RotaryEmbedding::new_dynamic(cfg.head_dim()?, cfg.rope_theta, vb.device())?;

        #[cfg(feature = "cuda")]
        let _drain_guard = CudaGraphDrainGuard::new(vb.device());
        Ok(Self {
            #[cfg(feature = "cuda")]
            decode_graph: RefCell::new(None),
            #[cfg(feature = "cuda")]
            verification_graph: RefCell::new(None),
            embed_tokens,
            layers,
            norm,
            rotary,
            #[cfg(feature = "cuda")]
            _drain_guard,
        })
    }

    pub(crate) fn embed(&self, input_ids: &Tensor) -> Result<Tensor, Error> {
        self.embed_tokens
            .forward(input_ids)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "token embedding", e))
    }

    pub(crate) fn token_embedding_weight(&self) -> Tensor {
        self.embed_tokens.embeddings().clone()
    }

    /// The final RMSNorm weight, shared zero-copy with speculative draft
    /// heads (`mtp_share_norm`).
    pub(crate) fn final_norm_weight(&self) -> Tensor {
        self.norm.weight().clone()
    }

    /// `(B, seq, hidden)` final hidden states (post final norm) at the plain
    /// sequential positions given by `position_ids` (`(1, B, seq)`).
    pub(crate) fn forward(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, Error> {
        let (cos, sin) = self
            .rotary
            .forward_multi_axis(position_ids, inputs_embeds.dtype())?;
        // (1, batch, seq, head_dim) -> (batch, 1, seq, head_dim): broadcast
        // over heads in the attention projections.
        let cos = cos.squeeze(0)?.unsqueeze(1)?;
        let sin = sin.squeeze(0)?.unsqueeze(1)?;
        let mut hidden_states = inputs_embeds.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &cos, &sin, attention_mask)?;
        }
        self.norm
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "final norm", e))
    }

    /// Logits of the single last position, for step-wise greedy decoding.
    fn project_logits(&self, hidden_states: &Tensor, lm_head: &Linear) -> Result<Tensor, Error> {
        lm_head
            .forward(hidden_states)
            .and_then(|logits| logits.i((0, 0, ..)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "decode LM head", e))
    }

    /// Per-position logits over a whole query block, for greedy verification.
    fn project_all_logits(
        &self,
        hidden_states: &Tensor,
        lm_head: &Linear,
    ) -> Result<Tensor, Error> {
        lm_head
            .forward(hidden_states)
            .and_then(|logits| logits.squeeze(0))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "verification LM head", e))
    }

    /// One eager or graph-replayed decode step at `position_ids`; batch rows
    /// beyond the first require `attention_mask` (graphs are batch-1 only).
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
            if let Some((logits, _)) =
                self.replay_cuda_graph(inputs_embeds, position_ids, kv_len)?
            {
                return Ok(logits);
            }
        }
        let hidden = self.forward(inputs_embeds, position_ids, attention_mask)?;
        self.project_logits(&hidden, lm_head)
    }

    /// Decode step that also returns the fed token's post-final-norm hidden
    /// state — the state a speculative draft head needs to re-sync after a
    /// pause. Graph replay exports both when available.
    pub(crate) fn forward_decode_logits_and_hidden(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        lm_head: &Linear,
    ) -> Result<(Tensor, Tensor), Error> {
        #[cfg(feature = "cuda")]
        {
            let kv_len = self.kv_cache_len().saturating_add(1);
            if let Some(output) = self.replay_cuda_graph(inputs_embeds, position_ids, kv_len)? {
                return Ok(output);
            }
        }
        let hidden = self.forward(inputs_embeds, position_ids, None)?;
        let logits = self.project_logits(&hidden, lm_head)?;
        let token_hidden = hidden
            .i((0, 0, ..))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "decode hidden", e))?;
        Ok((logits, token_hidden))
    }

    /// Verify a fixed block of speculative tokens in one causal target pass.
    ///
    /// Returns the target's post-final-norm hidden states (which the FastMTP
    /// draft consumes on its next sync pass) and the block's per-position
    /// logits; the caller applies greedy decoding logit processors host-side
    /// before argmaxing, so speculation cannot change the official recipe.
    pub(crate) fn forward_verification_tokens(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        lm_head: &Linear,
    ) -> Result<(Tensor, Tensor), Error> {
        #[cfg(feature = "cuda")]
        let query_len = inputs_embeds
            .dim(1)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "verification query length", e))?;
        #[cfg(feature = "cuda")]
        {
            let kv_len = self.kv_cache_len().saturating_add(query_len);
            if let Some(output) =
                self.replay_verification_cuda_graph(inputs_embeds, position_ids, kv_len)?
            {
                return Ok(output);
            }
        }
        let hidden = self.forward(inputs_embeds, position_ids, None)?;
        let logits = self.project_all_logits(&hidden, lm_head)?;
        Ok((hidden, logits))
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
            .rotary
            .forward_multi_axis(position_ids, inputs_embeds.dtype())?;
        let cos = cos.squeeze(0)?.unsqueeze(1)?;
        let sin = sin.squeeze(0)?.unsqueeze(1)?;
        let mut hidden_states = inputs_embeds.clone();
        for layer in &self.layers {
            hidden_states =
                layer.forward_dynamic(&hidden_states, &cos, &sin, query_lengths, kv_lengths)?;
        }
        self.norm
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "final norm", e))
    }

    /// Capture the batch-1 single-token decode graph when eligible.
    pub(crate) fn prepare_ar_cuda_graph(
        &self,
        prompt_len: usize,
        max_new_tokens: usize,
        lm_head: &Linear,
    ) -> Result<(), Error> {
        if graphs_disabled() {
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
            let Some(cache_len) = decode_capacity(prompt_len, max_new_tokens) else {
                self.invalidate_cuda_graph();
                return Ok(());
            };
            let required = prompt_len
                .saturating_add(max_new_tokens)
                .min(DECODE_CACHE_LEN);
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

    /// Capture the fixed-width MTP verification block; returns the shared
    /// cache bucket the draft model should capture against.
    #[cfg(feature = "cuda")]
    pub(crate) fn prepare_verification_cuda_graph(
        &self,
        prompt_len: usize,
        max_new_tokens: usize,
        query_len: usize,
        lm_head: &Linear,
    ) -> Result<Option<usize>, Error> {
        if query_len == 0 || graphs_disabled() {
            self.invalidate_cuda_graph();
            return Ok(None);
        }
        if !self.embed_tokens.embeddings().device().is_cuda()
            || !matches!(
                self.embed_tokens.embeddings().dtype(),
                DType::BF16 | DType::F16
            )
        {
            self.invalidate_cuda_graph();
            return Ok(None);
        }

        // Verification can temporarily place a complete query block in KV
        // beyond the user-visible output limit, so reserve one extra block.
        let Some(cache_len) = decode_capacity(prompt_len, max_new_tokens.saturating_add(query_len))
        else {
            self.invalidate_cuda_graph();
            return Ok(None);
        };
        let required = prompt_len
            .saturating_add(max_new_tokens)
            .saturating_add(query_len)
            .min(DECODE_CACHE_LEN);
        let retained_cache_len = self
            .verification_graph
            .borrow()
            .as_ref()
            .filter(|graph| graph.query_len == query_len && graph.cache_len >= required)
            .map(|graph| graph.cache_len);
        if let Some(retained_cache_len) = retained_cache_len {
            // The draft graph shares this capacity contract. Returning the
            // newly computed (possibly smaller) bucket would force it to
            // recapture even though the retained target graph is reusable.
            return Ok(Some(retained_cache_len));
        }

        self.invalidate_cuda_graph();
        self.capture_verification_cuda_graph(cache_len, query_len, lm_head)?;
        Ok(Some(cache_len))
    }

    /// Capture the decode graph against an existing fixed-capacity bucket
    /// (e.g. one a verification graph already holds, so the shared storage is
    /// reused instead of reallocated). No-op when a decode graph exists.
    #[cfg(feature = "cuda")]
    pub(crate) fn capture_ar_cuda_graph_with_capacity(
        &self,
        cache_len: usize,
        lm_head: &Linear,
    ) -> Result<(), Error> {
        self.capture_cuda_graph(cache_len, lm_head)
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
        let device = self.embed_tokens.embeddings().device().clone();
        let query_len = 1;
        for layer in &self.layers {
            layer.prepare_dynamic_cache(query_len, cache_len)?;
        }
        let hidden_size = self
            .embed_tokens
            .embeddings()
            .dim(1)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "graph hidden size", e))?;
        let hidden_input = Tensor::zeros(
            (1, query_len, hidden_size),
            self.embed_tokens.embeddings().dtype(),
            &device,
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "graph hidden input", e))?;
        let position_input = Tensor::zeros((1, 1, query_len), DType::U32, &device)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "graph position input", e))?;
        let query_lengths = Tensor::new(&[0u32, query_len as u32], &device)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "graph query lengths", e))?;
        let kv_lengths = CudaGraphKvLengths::new(query_len, &device)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "graph KV lengths", e))?;
        let stream = cuda.cuda_stream();
        let _htod_cache = cuda.enable_cuda_graph_htod_cache();

        let warm = self.forward_dynamic(
            &hidden_input,
            &position_input,
            &query_lengths,
            kv_lengths.tensor(),
        )?;
        let warm_logits = self.project_logits(&warm, lm_head)?;
        sync_graph_tensor(MODEL_NAME, &warm_logits, "warm decoder CUDA graph")?;
        // Allocate the output buffers before capture so they belong to the
        // regular stream-ordered pool; a capture-time allocation lives in the
        // graph's private pool and can never be returned to the allocator
        // safely. Prime the copies so the captured run sees warm kernels.
        let logits_output = Tensor::zeros_like(&warm_logits)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "graph logits output", e))?;
        logits_output
            .slice_set(&warm_logits, 0, 0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "prime graph logits copy", e))?;
        let warm_hidden = warm
            .i((0, 0, ..))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "warm decoder hidden", e))?;
        let hidden_output = Tensor::zeros_like(&warm_hidden)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "graph hidden output", e))?;
        hidden_output
            .slice_set(&warm_hidden, 0, 0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "prime graph hidden copy", e))?;

        stream
            .begin_capture(CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_GLOBAL)
            .map_err(|e| cuda_graph_error(MODEL_NAME, "begin decoder CUDA graph capture", e))?;
        let captured_output: Result<(), Error> = (|| {
            let hidden = self.forward_dynamic(
                &hidden_input,
                &position_input,
                &query_lengths,
                kv_lengths.tensor(),
            )?;
            let token_hidden = hidden
                .i((0, 0, ..))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "graph hidden slice", e))?;
            let logits = self.project_logits(&hidden, lm_head)?;
            logits_output
                .slice_set(&logits, 0, 0)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "record graph logits copy", e))?;
            hidden_output
                .slice_set(&token_hidden, 0, 0)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "record graph hidden copy", e))
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
        self.clear_kv_cache();
        *self.decode_graph.borrow_mut() = Some(DecodeCudaGraph {
            graph,
            hidden_input,
            position_input,
            _query_lengths: query_lengths,
            kv_lengths,
            logits_output,
            hidden_output,
            cache_len,
        });
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn capture_verification_cuda_graph(
        &self,
        cache_len: usize,
        query_len: usize,
        lm_head: &Linear,
    ) -> Result<(), Error> {
        use candle_core::cuda_backend::cudarc::driver::sys::{
            CUgraphInstantiate_flags_enum, CUstreamCaptureMode_enum,
        };

        let Device::Cuda(cuda) = self.embed_tokens.embeddings().device() else {
            return Ok(());
        };
        let device = self.embed_tokens.embeddings().device().clone();
        for layer in &self.layers {
            layer.prepare_dynamic_cache(query_len, cache_len)?;
        }
        let hidden_size = self
            .embed_tokens
            .embeddings()
            .dim(1)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "graph hidden size", e))?;
        let hidden_input = Tensor::zeros(
            (1, query_len, hidden_size),
            self.embed_tokens.embeddings().dtype(),
            &device,
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "verification graph input", e))?;
        let position_input = Tensor::zeros((1, 1, query_len), DType::U32, &device)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "verification graph positions", e))?;
        let query_lengths = Tensor::new(&[0u32, query_len as u32], &device)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "verification query lengths", e))?;
        let kv_lengths = CudaGraphKvLengths::new(query_len, &device)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "verification KV lengths", e))?;
        let stream = cuda.cuda_stream();
        let _htod_cache = cuda.enable_cuda_graph_htod_cache();

        let warm = self.forward_dynamic(
            &hidden_input,
            &position_input,
            &query_lengths,
            kv_lengths.tensor(),
        )?;
        let warm_logits = self.project_all_logits(&warm, lm_head)?;
        sync_graph_tensor(MODEL_NAME, &warm_logits, "warm verification CUDA graph")?;
        // Allocate the output buffers before capture so they belong to the
        // regular stream-ordered pool; a capture-time allocation lives in the
        // graph's private pool and can never be returned to the allocator
        // safely. Prime the copies so the captured run sees warm kernels.
        let hidden_output = Tensor::zeros_like(&warm)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "verification hidden output", e))?;
        let logits_output = Tensor::zeros_like(&warm_logits)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "verification logits output", e))?;
        hidden_output.slice_set(&warm, 0, 0).map_err(|e| {
            candle_to_ocr_inference(MODEL_NAME, "prime verification hidden copy", e)
        })?;
        logits_output.slice_set(&warm_logits, 0, 0).map_err(|e| {
            candle_to_ocr_inference(MODEL_NAME, "prime verification logits copy", e)
        })?;

        stream
            .begin_capture(CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_GLOBAL)
            .map_err(|e| cuda_graph_error(MODEL_NAME, "begin verification graph capture", e))?;
        let captured_output: Result<(), Error> = (|| {
            let hidden = self.forward_dynamic(
                &hidden_input,
                &position_input,
                &query_lengths,
                kv_lengths.tensor(),
            )?;
            let logits = self.project_all_logits(&hidden, lm_head)?;
            hidden_output.slice_set(&hidden, 0, 0).map_err(|e| {
                candle_to_ocr_inference(MODEL_NAME, "record verification hidden copy", e)
            })?;
            logits_output.slice_set(&logits, 0, 0).map_err(|e| {
                candle_to_ocr_inference(MODEL_NAME, "record verification logits copy", e)
            })
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
            .map_err(|e| cuda_graph_error(MODEL_NAME, "end verification graph capture", e))?
            .ok_or_else(|| Error::Config {
                message: format!("{MODEL_NAME} verification capture returned no graph"),
            })?;
        graph
            .launch()
            .map_err(|e| cuda_graph_error(MODEL_NAME, "warm verification CUDA graph", e))?;
        sync_graph_tensor(MODEL_NAME, &logits_output, "sync verification CUDA graph")?;
        self.clear_kv_cache();
        *self.verification_graph.borrow_mut() = Some(VerificationCudaGraph {
            graph,
            hidden_input,
            position_input,
            _query_lengths: query_lengths,
            kv_lengths,
            hidden_output,
            logits_output,
            cache_len,
            query_len,
        });
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn replay_cuda_graph(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        kv_len: usize,
    ) -> Result<Option<(Tensor, Tensor)>, Error> {
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
        Ok(Some((
            captured.logits_output.clone(),
            captured.hidden_output.clone(),
        )))
    }

    #[cfg(feature = "cuda")]
    fn replay_verification_cuda_graph(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        kv_len: usize,
    ) -> Result<Option<(Tensor, Tensor)>, Error> {
        let captured_ref = self.verification_graph.borrow();
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
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "copy verification hidden", e))?;
        captured
            .position_input
            .slice_set(position_ids, 0, 0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "copy verification positions", e))?;
        captured.kv_lengths.update(kv_len).map_err(|e| {
            candle_to_ocr_inference(MODEL_NAME, "update verification KV lengths", e)
        })?;
        captured
            .graph
            .launch()
            .map_err(|e| cuda_graph_error(MODEL_NAME, "launch verification CUDA graph", e))?;
        for layer in &self.layers {
            layer.set_kv_cache_len(kv_len)?;
        }
        Ok(Some((
            captured.hidden_output.clone(),
            captured.logits_output.clone(),
        )))
    }

    #[cfg(feature = "cuda")]
    fn invalidate_cuda_graph(&self) {
        if let Some(graph) = self.decode_graph.borrow_mut().take() {
            graph.dispose();
        }
        if let Some(graph) = self.verification_graph.borrow_mut().take() {
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

    pub(crate) fn trim_kv_cache(&self, len: usize) -> Result<(), Error> {
        for layer in &self.layers {
            layer.trim_kv_cache(len)?;
        }
        Ok(())
    }

    /// Snapshot every layer's live KV contents. Mid-generation graph captures
    /// reuse the fixed storage but their warmup runs overwrite the first
    /// positions and reset the logical length; this saves what must be
    /// restored afterwards.
    #[cfg(feature = "cuda")]
    pub(crate) fn save_kv_cache(&self) -> Result<Vec<(Tensor, Tensor, usize)>, Error> {
        let mut saved = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            saved.push(layer.save_kv_cache()?);
        }
        Ok(saved)
    }

    /// Restore a [`Self::save_kv_cache`] snapshot into the live storage.
    #[cfg(feature = "cuda")]
    pub(crate) fn restore_kv_cache(&self, saved: &[(Tensor, Tensor, usize)]) -> Result<(), Error> {
        for (layer, snapshot) in self.layers.iter().zip(saved) {
            layer.restore_kv_cache(snapshot)?;
        }
        Ok(())
    }

    pub(crate) fn clear_kv_cache(&self) {
        for layer in &self.layers {
            layer.clear_cache();
        }
    }
}

#[cfg(feature = "cuda")]
impl Drop for DeepSeekV2TextModel {
    fn drop(&mut self) {
        // A cached graph must go through dispose: plainly dropping it returns
        // graph-bound buffers to the allocator and poisons it.
        self.invalidate_cuda_graph();
    }
}

/// Reusable dense decoder block for speculative draft layers.
///
/// The FastMTP draft reuses the target's attention/MLP structure with a dense
/// feed-forward (`n_routed_experts` forced to zero); weights load from the
/// `mtp_block` subtree of a draft head.
#[derive(Debug)]
pub(crate) struct DeepSeekV2MtpBlock(DecoderLayer);

impl DeepSeekV2MtpBlock {
    pub(crate) fn load(cfg: &DeepSeekV2TextConfig, vb: VarBuilder) -> Result<Self, Error> {
        let mut dense_cfg = cfg.clone();
        dense_cfg.n_routed_experts = 0;
        Ok(Self(DecoderLayer::load(&dense_cfg, 0, vb)?))
    }

    pub(crate) fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> Result<Tensor, Error> {
        self.0.forward(hidden_states, cos, sin, None)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn forward_dynamic(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        query_lengths: &Tensor,
        kv_lengths: &Tensor,
    ) -> Result<Tensor, Error> {
        self.0
            .forward_dynamic(hidden_states, cos, sin, query_lengths, kv_lengths)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn prepare_dynamic_cache(
        &self,
        query_len: usize,
        cache_len: usize,
    ) -> Result<(), Error> {
        self.0.prepare_dynamic_cache(query_len, cache_len)
    }

    pub(crate) fn trim_kv_cache(&self, len: usize) -> Result<(), Error> {
        self.0.trim_kv_cache(len)
    }

    pub(crate) fn clear_kv_cache(&self) {
        self.0.clear_cache();
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn set_kv_cache_len(&self, len: usize) -> Result<(), Error> {
        self.0.set_kv_cache_len(len)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn kv_cache_len(&self) -> usize {
        self.0.kv_cache_len()
    }
}
