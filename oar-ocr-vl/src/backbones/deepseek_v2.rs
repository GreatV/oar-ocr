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
use crate::runtime::decoder_graph::decoder_attention_is_causal;
#[cfg(feature = "cuda")]
use candle_core::Device;

const MODEL_NAME: &str = "DeepSeek-V2";

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
/// the always-on shared expert block. The gate runs in f32 and the expert
/// outputs are combined in the gate's f32 weights, mirroring `moe_infer`.
#[derive(Debug)]
struct MoeFeedForward {
    gate: Tensor,
    experts: Vec<DenseMlp>,
    shared_experts: DenseMlp,
    top_k: usize,
}

impl MoeFeedForward {
    fn load(cfg: &DeepSeekV2TextConfig, vb: VarBuilder) -> Result<Self, Error> {
        let gate = vb
            .pp("gate")
            .get((cfg.n_routed_experts, cfg.hidden_size), "weight")
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load MoE gate", e))?;
        let experts = (0..cfg.n_routed_experts)
            .map(|i| {
                DenseMlp::load(
                    cfg.hidden_size,
                    cfg.moe_intermediate_size,
                    vb.pp("experts").pp(i),
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let shared_experts = DenseMlp::load(
            cfg.hidden_size,
            cfg.moe_intermediate_size * cfg.n_shared_experts,
            vb.pp("shared_experts"),
        )?;
        Ok(Self {
            gate,
            experts,
            shared_experts,
            top_k: cfg.num_experts_per_tok,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, Error> {
        let (batch, seq, hidden) = xs
            .dims3()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE input shape", e))?;
        let tokens = batch * seq;
        let flat = xs.reshape((tokens, hidden))?;

        // Router: f32 logits, softmax scores, greedy top-k sorted by weight
        // (like `torch.topk`), weights unnormalized (`routed_scaling_factor` 1).
        let logits = flat
            .to_dtype(DType::F32)?
            .matmul(&self.gate.to_dtype(DType::F32)?.t()?)?;
        let scores = candle_nn::ops::softmax_last_dim(&logits)?;
        let (topk_ids, topk_weights) = topk_weights(&scores, self.top_k)?;
        let ids: Vec<u32> = topk_ids
            .flatten_all()?
            .to_vec1::<u32>()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "read router ids", e))?;
        let weights: Vec<f32> = topk_weights
            .flatten_all()?
            .to_vec1::<f32>()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "read router weights", e))?;

        // Group selected (token, slot) pairs by expert so each expert runs one
        // dense batch, then combine in the router's f32 like `moe_infer`.
        let mut by_expert: Vec<Vec<(usize, usize)>> = vec![Vec::new(); self.experts.len()];
        for (slot, &id) in ids.iter().enumerate() {
            by_expert[id as usize].push((slot / self.top_k, slot));
        }

        let mut combined = vec![0f32; tokens * hidden];
        for (expert_index, slots) in by_expert.iter().enumerate() {
            if slots.is_empty() {
                continue;
            }
            let rows: Vec<u32> = slots.iter().map(|&(token, _)| token as u32).collect();
            let row_index = Tensor::from_vec(rows, slots.len(), xs.device())?;
            let routed = flat.index_select(&row_index, 0)?;
            let expert_out = self.experts[expert_index]
                .forward(&routed)?
                .to_dtype(DType::F32)?
                .to_vec2::<f32>()
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "read expert output", e))?;
            for (row, &(token, slot)) in slots.iter().enumerate() {
                let weight = weights[slot];
                let base_in = token * hidden;
                for dim in 0..hidden {
                    combined[base_in + dim] += weight * expert_out[row][dim];
                }
            }
        }
        let combined =
            Tensor::from_vec(combined, (tokens, hidden), xs.device())?.to_dtype(xs.dtype())?;
        let shared = self.shared_experts.forward(&flat)?;
        (&combined + &shared)?
            .reshape((batch, seq, hidden))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MoE output", e))
    }
}

/// Sorted top-k over softmax scores: `(ids (tokens, k) u32, weights (tokens,
/// k) f32)`. Ties break toward the lower expert id.
fn topk_weights(scores: &Tensor, k: usize) -> Result<(Tensor, Tensor), Error> {
    let (tokens, experts) = scores.dims2()?;
    let data = scores
        .flatten_all()?
        .to_vec1::<f32>()
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "read router scores", e))?;
    let mut ids = Vec::with_capacity(tokens * k);
    let mut weights = Vec::with_capacity(tokens * k);
    for token in 0..tokens {
        let row = &data[token * experts..(token + 1) * experts];
        let mut order: Vec<u32> = (0..experts as u32).collect();
        order.sort_by(|&a, &b| {
            row[b as usize]
                .partial_cmp(&row[a as usize])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.cmp(&b))
        });
        order.truncate(k);
        for &id in &order {
            ids.push(id);
            weights.push(row[id as usize]);
        }
    }
    let device = scores.device();
    Ok((
        Tensor::from_vec(ids, (tokens, k), device)?,
        Tensor::from_vec(weights, (tokens, k), device)?,
    ))
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
        let mlp = self.forward_mlp(&normalized)?;
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
}

pub(crate) struct DeepSeekV2TextModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: RmsNorm,
    rotary: RotaryEmbedding,
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
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary,
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

    fn prepare_rope(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor), Error> {
        let (cos, sin) = self
            .rotary
            .forward_multi_axis(position_ids, inputs_embeds.dtype())?;
        // (1, batch, seq, head_dim) -> (batch, 1, seq, head_dim): broadcast
        // over heads in the attention projections.
        let cos = cos
            .squeeze(0)
            .and_then(|c| c.unsqueeze(1))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "RoPE cos layout", e))?;
        let sin = sin
            .squeeze(0)
            .and_then(|s| s.unsqueeze(1))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "RoPE sin layout", e))?;
        Ok((cos, sin))
    }

    /// `(B, seq, hidden)` final hidden states (post final norm) at the plain
    /// sequential positions given by `position_ids` (`(1, B, seq)`).
    pub(crate) fn forward(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, Error> {
        let (cos, sin) = self.prepare_rope(inputs_embeds, position_ids)?;
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

    /// One decode step at `position_ids`; batch rows beyond the first
    /// require `attention_mask`.
    pub(crate) fn forward_decode_logits(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        attention_mask: Option<&Tensor>,
        lm_head: &Linear,
    ) -> Result<Tensor, Error> {
        let hidden = self.forward(inputs_embeds, position_ids, attention_mask)?;
        self.project_logits(&hidden, lm_head)
    }

    /// Verify a fixed block of speculative tokens in one causal target pass.
    ///
    /// Returns the target's post-final-norm hidden states (which the FastMTP
    /// draft consumes on its next sync pass) and the block's per-position
    /// logits; the caller applies greedy-decoding logit processors host-side
    /// before argmaxing, so speculation cannot change the official recipe.
    pub(crate) fn forward_verification_tokens(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        lm_head: &Linear,
    ) -> Result<(Tensor, Tensor), Error> {
        let hidden = self.forward(inputs_embeds, position_ids, None)?;
        let logits = lm_head
            .forward(&hidden)
            .and_then(|l| l.squeeze(0))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "verification LM head", e))?;
        Ok((hidden, logits))
    }

    pub(crate) fn trim_kv_cache(&self, len: usize) -> Result<(), Error> {
        for layer in &self.layers {
            layer.trim_kv_cache(len)?;
        }
        Ok(())
    }

    pub(crate) fn clear_kv_cache(&self) {
        for layer in &self.layers {
            layer.clear_cache();
        }
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
