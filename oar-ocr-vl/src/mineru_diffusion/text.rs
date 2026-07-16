//! SDAR block-diffusion text decoder.
//!
//! Architecturally a Qwen2-style decoder (RMSNorm, SwiGLU MLP, GQA, RoPE) with
//! two departures that matter for block diffusion:
//!
//! * **QK-norm** — per-head RMSNorm is applied to queries and keys before RoPE.
//! * **Non-causal attention** — there is no built-in causal mask. Causality is
//!   imposed at *block* granularity by the caller: the prompt is prefilled with
//!   an explicit block-causal mask and each generation block attends fully to
//!   the committed KV cache plus itself (no mask needed).
//!
//! The [`SdarKvCache`] supports the diffusion access pattern: denoising passes
//! read the committed cache and transiently append the in-flight block's KV
//! without storing it (`store_kv = false`); once a block is fully decoded it is
//! committed once (`store_kv = true`).

use super::config::SdarConfig;
use crate::attention::{RotaryEmbedding, flash_attention, scaled_dot_product_attention_gqa};
use crate::utils::{candle_to_ocr_inference, candle_to_ocr_processing, rotate_half};
use candle_core::{DType, Device, Tensor};
use candle_nn::{
    Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear, linear_no_bias, rms_norm,
};
use oar_ocr_core::core::OCRError;

fn proc_err(msg: &'static str, e: candle_core::Error) -> OCRError {
    candle_to_ocr_processing(
        oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
        msg,
        e,
    )
}

/// Per-layer committed KV (rope already applied to keys).
/// Shape `(batch=1, num_kv_heads, len, head_dim)`.
#[derive(Debug, Default, Clone)]
struct LayerKvCache {
    k: Option<Tensor>,
    v: Option<Tensor>,
}

#[derive(Debug, Clone, Copy)]
struct AttentionMode {
    store_kv: bool,
    is_causal: bool,
}

/// Committed KV cache across all decoder layers.
#[derive(Debug, Clone)]
pub struct SdarKvCache {
    layers: Vec<LayerKvCache>,
}

impl SdarKvCache {
    pub fn new(num_layers: usize) -> Self {
        Self {
            layers: vec![LayerKvCache::default(); num_layers],
        }
    }
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor, OCRError> {
    let a = x
        .broadcast_mul(cos)
        .map_err(|e| proc_err("SDAR rope x*cos", e))?;
    let b = rotate_half(x)?
        .broadcast_mul(sin)
        .map_err(|e| proc_err("SDAR rope rotate_half*sin", e))?;
    (a + b).map_err(|e| proc_err("SDAR rope sum", e))
}

#[derive(Debug)]
struct SdarAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl SdarAttention {
    fn load(cfg: &SdarConfig, vb: VarBuilder) -> Result<Self, OCRError> {
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            return Err(OCRError::ConfigError {
                message: format!(
                    "MinerU-Diffusion: num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                    cfg.num_attention_heads, cfg.num_key_value_heads
                ),
            });
        }
        let head_dim = cfg.head_dim()?;
        let q_dim = cfg.num_attention_heads * head_dim;
        let kv_dim = cfg.num_key_value_heads * head_dim;
        let mk = |i: usize, o: usize, name: &str, vb: VarBuilder| -> Result<Linear, OCRError> {
            if cfg.attention_bias {
                linear(i, o, vb)
            } else {
                linear_no_bias(i, o, vb)
            }
            .map_err(|e| candle_to_ocr_inference("MinerU-Diffusion", name, e))
        };
        let q_proj = mk(cfg.hidden_size, q_dim, "load q_proj", vb.pp("q_proj"))?;
        let k_proj = mk(cfg.hidden_size, kv_dim, "load k_proj", vb.pp("k_proj"))?;
        let v_proj = mk(cfg.hidden_size, kv_dim, "load v_proj", vb.pp("v_proj"))?;
        let o_proj = mk(q_dim, cfg.hidden_size, "load o_proj", vb.pp("o_proj"))?;
        let q_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))
            .map_err(|e| candle_to_ocr_inference("MinerU-Diffusion", "load q_norm", e))?;
        let k_norm = rms_norm(head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))
            .map_err(|e| candle_to_ocr_inference("MinerU-Diffusion", "load k_norm", e))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
        cache: &mut LayerKvCache,
        mode: AttentionMode,
    ) -> Result<Tensor, OCRError> {
        let (b, s, _) = hidden
            .dims3()
            .map_err(|e| candle_to_ocr_inference("MinerU-Diffusion", "attn dims", e))?;

        let project =
            |proj: &Linear, heads: usize, norm: Option<&RmsNorm>| -> Result<Tensor, OCRError> {
                let x = proj
                    .forward(hidden)
                    .map_err(|e| candle_to_ocr_inference("MinerU-Diffusion", "attn proj", e))?;
                let x = x
                    .reshape((b, s, heads, self.head_dim))
                    .map_err(|e| proc_err("SDAR attn reshape", e))?;
                let x = match norm {
                    Some(n) => n.forward(&x).map_err(|e| {
                        candle_to_ocr_inference("MinerU-Diffusion", "attn qk-norm", e)
                    })?,
                    None => x,
                };
                x.transpose(1, 2)
                    .map_err(|e| proc_err("SDAR attn transpose", e))?
                    .contiguous()
                    .map_err(|e| proc_err("SDAR attn contiguous", e))
            };

        let q = project(&self.q_proj, self.num_heads, Some(&self.q_norm))?;
        let k = project(&self.k_proj, self.num_kv_heads, Some(&self.k_norm))?;
        let v = project(&self.v_proj, self.num_kv_heads, None)?;

        let q = apply_rope(&q, cos, sin)?;
        let k = apply_rope(&k, cos, sin)?;

        // Concatenate the committed cache (already rope-applied) with the
        // current segment's KV. Only persist when committing the block.
        let (full_k, full_v) = match (&cache.k, &cache.v) {
            (Some(ck), Some(cv)) => {
                let fk = Tensor::cat(&[ck, &k], 2).map_err(|e| proc_err("SDAR kv cat k", e))?;
                let fv = Tensor::cat(&[cv, &v], 2).map_err(|e| proc_err("SDAR kv cat v", e))?;
                (fk, fv)
            }
            _ => (k, v),
        };
        if mode.store_kv {
            cache.k = Some(full_k.clone());
            cache.v = Some(full_v.clone());
        }

        let n_rep = self.num_heads / self.num_kv_heads;
        let flash = if mask.is_none() {
            flash_attention(&q, &full_k, &full_v, self.scale, mode.is_causal)
                .map_err(|e| candle_to_ocr_inference("MinerU-Diffusion", "flash attention", e))?
        } else {
            None
        };
        let attn = match flash {
            Some(attn) => attn,
            None => scaled_dot_product_attention_gqa(
                &q,
                &full_k,
                &full_v,
                mask,
                self.scale,
                mode.is_causal,
                n_rep,
            )
            .map_err(|e| {
                candle_to_ocr_inference("MinerU-Diffusion", "grouped-query attention", e)
            })?,
        };
        let attn = attn
            .transpose(1, 2)
            .map_err(|e| proc_err("SDAR attn out transpose", e))?
            .reshape((b, s, self.num_heads * self.head_dim))
            .map_err(|e| proc_err("SDAR attn out reshape", e))?;
        self.o_proj
            .forward(&attn)
            .map_err(|e| candle_to_ocr_inference("MinerU-Diffusion", "o_proj", e))
    }
}

#[derive(Debug)]
struct SdarMlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl SdarMlp {
    fn load(cfg: &SdarConfig, vb: VarBuilder) -> Result<Self, OCRError> {
        let gate = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))
            .map_err(|e| candle_to_ocr_inference("MinerU-Diffusion", "load gate_proj", e))?;
        let up = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))
            .map_err(|e| candle_to_ocr_inference("MinerU-Diffusion", "load up_proj", e))?;
        let down = linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))
            .map_err(|e| candle_to_ocr_inference("MinerU-Diffusion", "load down_proj", e))?;
        Ok(Self { gate, up, down })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, OCRError> {
        let gate = self
            .gate
            .forward(x)
            .map_err(|e| candle_to_ocr_inference("MinerU-Diffusion", "gate_proj", e))?;
        let gate = candle_nn::ops::silu(&gate).map_err(|e| proc_err("SDAR silu", e))?;
        let up = self
            .up
            .forward(x)
            .map_err(|e| candle_to_ocr_inference("MinerU-Diffusion", "up_proj", e))?;
        let prod = (gate * up).map_err(|e| proc_err("SDAR mlp mul", e))?;
        self.down
            .forward(&prod)
            .map_err(|e| candle_to_ocr_inference("MinerU-Diffusion", "down_proj", e))
    }
}

#[derive(Debug)]
struct SdarLayer {
    input_layernorm: RmsNorm,
    self_attn: SdarAttention,
    post_attention_layernorm: RmsNorm,
    mlp: SdarMlp,
}

impl SdarLayer {
    fn load(cfg: &SdarConfig, vb: VarBuilder) -> Result<Self, OCRError> {
        let input_layernorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))
            .map_err(|e| candle_to_ocr_inference("MinerU-Diffusion", "load input_layernorm", e))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )
        .map_err(|e| candle_to_ocr_inference("MinerU-Diffusion", "load post_attn_ln", e))?;
        let self_attn = SdarAttention::load(cfg, vb.pp("self_attn"))?;
        let mlp = SdarMlp::load(cfg, vb.pp("mlp"))?;
        Ok(Self {
            input_layernorm,
            self_attn,
            post_attention_layernorm,
            mlp,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        mask: Option<&Tensor>,
        cache: &mut LayerKvCache,
        mode: AttentionMode,
    ) -> Result<Tensor, OCRError> {
        let normed = self
            .input_layernorm
            .forward(hidden)
            .map_err(|e| candle_to_ocr_inference("MinerU-Diffusion", "input_layernorm", e))?;
        let attn = self
            .self_attn
            .forward(&normed, cos, sin, mask, cache, mode)?;
        let hidden = (hidden + attn).map_err(|e| proc_err("SDAR attn residual", e))?;
        let normed = self
            .post_attention_layernorm
            .forward(&hidden)
            .map_err(|e| candle_to_ocr_inference("MinerU-Diffusion", "post_attn_ln", e))?;
        let mlp = self.mlp.forward(&normed)?;
        (hidden + mlp).map_err(|e| proc_err("SDAR mlp residual", e))
    }
}

/// SDAR decoder stack + tied/untied LM head.
pub struct SdarModel {
    embed_tokens: Embedding,
    layers: Vec<SdarLayer>,
    norm: RmsNorm,
    lm_head: Linear,
    rotary: RotaryEmbedding,
    dtype: DType,
    device: Device,
}

impl SdarModel {
    /// `vb` should point at `language_model` (so `vb.pp("model")` is the
    /// decoder and `vb.pp("lm_head")` the output projection).
    pub fn load(cfg: &SdarConfig, vb: VarBuilder, dtype: DType) -> Result<Self, OCRError> {
        let model_vb = vb.pp("model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, model_vb.pp("embed_tokens"))
            .map_err(|e| candle_to_ocr_inference("MinerU-Diffusion", "load embed_tokens", e))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(SdarLayer::load(cfg, model_vb.pp(format!("layers.{i}")))?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, model_vb.pp("norm"))
            .map_err(|e| candle_to_ocr_inference("MinerU-Diffusion", "load norm", e))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))
                .map_err(|e| candle_to_ocr_inference("MinerU-Diffusion", "load lm_head", e))?
        };
        let head_dim = cfg.head_dim()?;
        let rotary = RotaryEmbedding::new_dynamic(head_dim, cfg.rope_theta, vb.device())?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            lm_head,
            rotary,
            dtype,
            device: vb.device().clone(),
        })
    }

    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }

    /// Embed token ids `(1, S)` into `(1, S, hidden)`.
    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor, OCRError> {
        self.embed_tokens
            .forward(input_ids)
            .map_err(|e| candle_to_ocr_inference("MinerU-Diffusion", "embed", e))
    }

    fn rope_for(&self, positions: &[i64]) -> Result<(Tensor, Tensor), OCRError> {
        let s = positions.len();
        let pos = Tensor::from_vec(positions.to_vec(), (1, 1, s), &self.device)
            .map_err(|e| proc_err("SDAR position_ids tensor", e))?;
        self.rotary.forward_multi_axis(&pos, self.dtype)
    }

    /// Run the decoder over `inputs_embeds` `(1, S, hidden)` at the given
    /// absolute `positions`, returning the post-norm hidden states `(1, S, hidden)`.
    /// `mask` is an additive attention bias `(1, 1, S, kv_len)` or `None` for
    /// full attention. When `store_kv` is set, each layer commits its KV.
    pub fn forward(
        &self,
        inputs_embeds: &Tensor,
        positions: &[i64],
        mask: Option<&Tensor>,
        cache: &mut SdarKvCache,
        store_kv: bool,
    ) -> Result<Tensor, OCRError> {
        self.forward_inner(inputs_embeds, positions, mask, cache, store_kv, false)
    }

    /// Run the same Qwen3 decoder in ordinary autoregressive causal mode.
    ///
    /// SDAR itself uses block-level masks and therefore calls [`Self::forward`]
    /// with non-causal attention. Multimodal models that share its Qwen3 +
    /// QK-norm backbone can use this path for flash-attention prefill and
    /// token-by-token decoding without duplicating the decoder stack.
    pub(crate) fn forward_causal(
        &self,
        inputs_embeds: &Tensor,
        positions: &[i64],
        cache: &mut SdarKvCache,
        store_kv: bool,
    ) -> Result<Tensor, OCRError> {
        self.forward_inner(inputs_embeds, positions, None, cache, store_kv, true)
    }

    fn forward_inner(
        &self,
        inputs_embeds: &Tensor,
        positions: &[i64],
        mask: Option<&Tensor>,
        cache: &mut SdarKvCache,
        store_kv: bool,
        is_causal: bool,
    ) -> Result<Tensor, OCRError> {
        let (cos, sin) = self.rope_for(positions)?;
        let mut hidden = inputs_embeds.clone();
        let mode = AttentionMode {
            store_kv,
            is_causal,
        };
        for (layer, layer_cache) in self.layers.iter().zip(cache.layers.iter_mut()) {
            hidden = layer.forward(&hidden, &cos, &sin, mask, layer_cache, mode)?;
        }
        self.norm
            .forward(&hidden)
            .map_err(|e| candle_to_ocr_inference("MinerU-Diffusion", "final norm", e))
    }

    /// Project hidden states `(1, S, hidden)` to logits `(1, S, vocab)`.
    pub fn lm_logits(&self, hidden: &Tensor) -> Result<Tensor, OCRError> {
        self.lm_head
            .forward(hidden)
            .map_err(|e| candle_to_ocr_inference("MinerU-Diffusion", "lm_head", e))
    }
}
