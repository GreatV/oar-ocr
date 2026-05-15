use super::config::HunyuanOcrConfig;
use crate::attention::{
    RotaryEmbedding, repeat_kv, scaled_dot_product_attention, select_rope_sections,
};
#[cfg(feature = "hsd")]
use crate::hsd::TrimmableKvCache;
#[cfg(not(feature = "hsd"))]
use crate::kv_trim::TrimmableKvCache;
use crate::utils::{candle_to_ocr_inference, candle_to_ocr_processing, rotate_half};
use candle_core::Tensor;
use candle_nn::Module;
use oar_ocr_core::core::OCRError;
use std::cell::RefCell;

/// Apply XDRoPE to `(q, k)` using already-section-mixed F32 `cos`/`sin`.
///
/// The section-mix (`select_rope_sections`) and the F32 cast of cos/sin are
/// layer-invariant — only q/k change between layers — so the caller
/// ([`HunyuanLlm::forward`]) hoists those steps out of the layer loop and
/// hands us the prepared tensors. Each layer then only pays the q/k F32 cast
/// and the actual rotary multiply.
fn apply_xdrope_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor), OCRError> {
    // Match upstream HF (`apply_rotary_pos_emb_xdrope`): apply the rotary
    // mix in F32, then cast q/k back to the original dtype.
    use candle_core::DType;
    let origin_dtype = q.dtype();
    let q_f32 = q
        .to_dtype(DType::F32)
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "xdrope q to_dtype f32", e))?;
    let k_f32 = k
        .to_dtype(DType::F32)
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "xdrope k to_dtype f32", e))?;

    let q_mul = q_f32.broadcast_mul(cos).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "HunyuanOCR: xdrope q*cos failed",
            e,
        )
    })?;
    let q_half = rotate_half(&q_f32)?;
    let q_half_mul = q_half.broadcast_mul(sin).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "HunyuanOCR: xdrope rotate_half(q)*sin failed",
            e,
        )
    })?;
    let q_rot = (&q_mul + &q_half_mul)
        .map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: xdrope apply on q failed",
                e,
            )
        })?
        .to_dtype(origin_dtype)
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "xdrope q to_dtype back", e))?;

    let k_mul = k_f32.broadcast_mul(cos).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "HunyuanOCR: xdrope k*cos failed",
            e,
        )
    })?;
    let k_half = rotate_half(&k_f32)?;
    let k_half_mul = k_half.broadcast_mul(sin).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "HunyuanOCR: xdrope rotate_half(k)*sin failed",
            e,
        )
    })?;
    let k_rot = (&k_mul + &k_half_mul)
        .map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: xdrope apply on k failed",
                e,
            )
        })?
        .to_dtype(origin_dtype)
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "xdrope k to_dtype back", e))?;
    Ok((q_rot, k_rot))
}

#[derive(Debug, Clone)]
struct HunyuanMlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl HunyuanMlp {
    fn load(cfg: &HunyuanOcrConfig, vb: candle_nn::VarBuilder) -> Result<Self, OCRError> {
        let gate_proj =
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load gate_proj", e))?;
        let up_proj =
            candle_nn::linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load up_proj", e))?;
        let down_proj =
            candle_nn::linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load down_proj", e))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, OCRError> {
        let gate = self
            .gate_proj
            .forward(xs)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "mlp gate_proj", e))?;
        let gate = candle_nn::ops::silu(&gate)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "mlp silu", e))?;
        let up = self
            .up_proj
            .forward(xs)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "mlp up_proj", e))?;
        let prod =
            (&gate * &up).map_err(|e| candle_to_ocr_inference("HunyuanOCR", "mlp gate*up", e))?;
        self.down_proj
            .forward(&prod)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "mlp down_proj", e))
    }
}

#[derive(Debug)]
struct HunyuanAttention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    query_layernorm: Option<candle_nn::RmsNorm>,
    key_layernorm: Option<candle_nn::RmsNorm>,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
    kv_cache: RefCell<TrimmableKvCache>,
}

impl HunyuanAttention {
    fn load(cfg: &HunyuanOcrConfig, vb: candle_nn::VarBuilder) -> Result<Self, OCRError> {
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            return Err(OCRError::ConfigError {
                message: format!(
                    "HunyuanOCR: num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                    cfg.num_attention_heads, cfg.num_key_value_heads
                ),
            });
        }

        let q_proj = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.head_dim,
            vb.pp("q_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load q_proj", e))?;
        let k_proj = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            vb.pp("k_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load k_proj", e))?;
        let v_proj = candle_nn::linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            vb.pp("v_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load v_proj", e))?;
        let o_proj = candle_nn::linear_no_bias(
            cfg.num_attention_heads * cfg.head_dim,
            cfg.hidden_size,
            vb.pp("o_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load o_proj", e))?;

        let (query_layernorm, key_layernorm) = if cfg.use_qk_norm {
            let q_ln =
                candle_nn::rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("query_layernorm"))
                    .map_err(|e| {
                        candle_to_ocr_inference("HunyuanOCR", "load query_layernorm", e)
                    })?;
            let k_ln = candle_nn::rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("key_layernorm"))
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load key_layernorm", e))?;
            (Some(q_ln), Some(k_ln))
        } else {
            (None, None)
        };

        // Cat-along-seq KV cache. Capacity 16384 covers ~1000-2000 vision tokens
        // plus the longest realistic generation. Same growth strategy as
        // candle_nn::kv_cache::KvCache (Tensor::cat per append). Trim/gather
        // support is required by HSD's tree-verification path.
        let kv_cache = TrimmableKvCache::new(2, 16384);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            query_layernorm,
            key_layernorm,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            scaling: (cfg.head_dim as f64).powf(-0.5),
            kv_cache: RefCell::new(kv_cache),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        causal_mask: Option<&Tensor>,
    ) -> Result<Tensor, OCRError> {
        let (b, seq_len, _) = hidden_states
            .dims3()
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "attn hidden_states dims3", e))?;

        let q = self
            .q_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "attn q_proj", e))?
            .reshape((b, seq_len, self.num_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "attn q reshape", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "attn q transpose", e))?;

        let k = self
            .k_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "attn k_proj", e))?
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "attn k reshape", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "attn k transpose", e))?;

        let v = self
            .v_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "attn v_proj", e))?
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "attn v reshape", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "attn v transpose", e))?;

        let (q, k) = apply_xdrope_rotary_pos_emb(&q, &k, cos, sin)?;

        // Match upstream HunyuanVL: apply XDRoPE first, then Q/K RMSNorm.
        // The learned RMSNorm weight is per head dimension, so it does not
        // commute with the rotary half-dimension mixing.
        let q = match &self.query_layernorm {
            Some(ln) => ln
                .forward(&q)
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "attn query_layernorm", e))?,
            None => q,
        };
        let k = match &self.key_layernorm {
            Some(ln) => ln
                .forward(&k)
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "attn key_layernorm", e))?,
            None => k,
        };

        let q = q
            .contiguous()
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "attn q contiguous", e))?;
        let k = k
            .contiguous()
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "attn k contiguous", e))?;
        let v = v
            .contiguous()
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "attn v contiguous", e))?;

        let (k_all, v_all) = self
            .kv_cache
            .borrow_mut()
            .append(&k, &v)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "append kv_cache", e))?;

        let key_states = repeat_kv(&k_all, self.num_kv_groups)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "repeat_kv key", e))?;
        let value_states = repeat_kv(&v_all, self.num_kv_groups)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "repeat_kv value", e))?;

        let key_states = key_states
            .contiguous()
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "attn key_states contiguous", e))?;
        let value_states = value_states.contiguous().map_err(|e| {
            candle_to_ocr_inference("HunyuanOCR", "attn value_states contiguous", e)
        })?;

        // Use unified attention implementation (BF16 Q·K, F32 softmax, BF16 A·V).
        // The main Hunyuan-specific numerical requirement is above: upstream
        // applies XDRoPE in F32 before Q/K RMSNorm.
        let attn_output = scaled_dot_product_attention(
            &q,
            &key_states,
            &value_states,
            causal_mask,
            self.scaling,
            false, // not causal - mask is explicit
        )
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "scaled_dot_product_attention", e))?;

        let attn_output = attn_output
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "attn out transpose", e))?
            .reshape((b, seq_len, self.num_heads * self.head_dim))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "attn out reshape", e))?;

        self.o_proj
            .forward(&attn_output)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "attn o_proj", e))
    }

    fn clear_kv_cache(&self) {
        self.kv_cache.borrow_mut().reset();
    }

    #[cfg(feature = "hsd")]
    fn current_kv_len(&self) -> usize {
        self.kv_cache.borrow().current_seq_len()
    }

    #[cfg(feature = "hsd")]
    fn keep_kv_indices(&self, indices: &[u32]) -> Result<(), OCRError> {
        self.kv_cache
            .borrow_mut()
            .keep_indices(indices)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "keep_kv_indices", e))
    }
}

#[derive(Debug)]
struct HunyuanDecoderLayer {
    self_attn: HunyuanAttention,
    mlp: HunyuanMlp,
    input_layernorm: candle_nn::RmsNorm,
    post_attention_layernorm: candle_nn::RmsNorm,
}

impl HunyuanDecoderLayer {
    fn load(cfg: &HunyuanOcrConfig, vb: candle_nn::VarBuilder) -> Result<Self, OCRError> {
        let self_attn = HunyuanAttention::load(cfg, vb.pp("self_attn"))?;
        let mlp = HunyuanMlp::load(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load input_layernorm", e))?;
        let post_attention_layernorm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load post_attention_layernorm", e))?;
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
        causal_mask: Option<&Tensor>,
    ) -> Result<Tensor, OCRError> {
        let residual = hidden_states;
        let hidden_states = self
            .input_layernorm
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "llm input_layernorm", e))?;
        let attn_out = self
            .self_attn
            .forward(&hidden_states, cos, sin, causal_mask)?;
        let hidden_states = (residual + &attn_out)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "llm attn residual add", e))?;

        let residual = hidden_states.clone();
        let hidden_states = self
            .post_attention_layernorm
            .forward(&hidden_states)
            .map_err(|e| {
                candle_to_ocr_inference("HunyuanOCR", "llm post_attention_layernorm", e)
            })?;
        let mlp_out = self.mlp.forward(&hidden_states)?;
        (&residual + &mlp_out)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "llm mlp residual add", e))
    }

    fn clear_kv_cache(&self) {
        self.self_attn.clear_kv_cache();
    }

    #[cfg(feature = "hsd")]
    fn keep_kv_indices(&self, indices: &[u32]) -> Result<(), OCRError> {
        self.self_attn.keep_kv_indices(indices)
    }
}

#[derive(Debug)]
pub struct HunyuanLlm {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<HunyuanDecoderLayer>,
    norm: candle_nn::RmsNorm,
    rotary: RotaryEmbedding,
    /// XDRoPE section sizes (`config.rope_scaling.xdrope_section`). Used once
    /// per `forward` to section-mix the rotary `cos`/`sin` tensors before they
    /// fan out to every layer — see [`apply_xdrope_rotary_pos_emb`].
    xdrope_section: Vec<usize>,
}

impl HunyuanLlm {
    pub fn load(cfg: &HunyuanOcrConfig, vb: candle_nn::VarBuilder) -> Result<Self, OCRError> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load embed_tokens", e))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(HunyuanDecoderLayer::load(
                cfg,
                vb.pp(format!("layers.{i}")),
            )?);
        }
        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load final norm", e))?;

        let rope_theta = match cfg.rope_scaling.alpha {
            Some(alpha) if alpha != 0.0 => {
                cfg.rope_theta * alpha.powf(cfg.head_dim as f64 / (cfg.head_dim as f64 - 2.0))
            }
            _ => cfg.rope_theta,
        };
        let rotary = RotaryEmbedding::new_multi_axis(cfg.head_dim, rope_theta, 4, vb.device())?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary,
            xdrope_section: cfg.rope_scaling.xdrope_section.clone(),
        })
    }

    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor, OCRError> {
        self.embed_tokens
            .forward(input_ids)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "embed tokens", e))
    }

    pub fn token_embedding_weight(&self) -> Tensor {
        self.embed_tokens.embeddings().clone()
    }

    pub fn forward(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        causal_mask: Option<&Tensor>,
    ) -> Result<Tensor, OCRError> {
        use candle_core::DType;

        let (cos, sin) = self
            .rotary
            .forward_multi_axis(position_ids, inputs_embeds.dtype())?;

        // XDRoPE section-mix + F32 cast: the result is layer-invariant
        // (depends only on position_ids and xdrope_section), so do it once
        // here instead of per-layer inside `apply_xdrope_rotary_pos_emb`.
        // Saves ~num_layers × (2 select_rope_sections + 2 to_dtype) ops per
        // forward.
        let cos = select_rope_sections(&cos, &self.xdrope_section, 4)?
            .to_dtype(DType::F32)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "xdrope cos to_dtype f32", e))?;
        let sin = select_rope_sections(&sin, &self.xdrope_section, 4)?
            .to_dtype(DType::F32)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "xdrope sin to_dtype f32", e))?;

        let mut hidden_states = inputs_embeds.clone();
        for layer in self.layers.iter() {
            hidden_states = layer.forward(&hidden_states, &cos, &sin, causal_mask)?;
        }
        let hidden_states = self
            .norm
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "llm final norm", e))?;
        Ok(hidden_states)
    }

    pub fn clear_kv_cache(&self) {
        for layer in &self.layers {
            layer.clear_kv_cache();
        }
    }

    /// Current sequence length held in the KV cache. All layers stay in sync,
    /// so we read it from layer 0.
    #[cfg(feature = "hsd")]
    pub fn current_kv_len(&self) -> usize {
        self.layers
            .first()
            .map(|l| l.self_attn.current_kv_len())
            .unwrap_or(0)
    }

    /// Gather every layer's KV cache to keep only the supplied positions
    /// (in order). Used by HSD after tree-attention verification to retain the
    /// accepted-path KV entries and drop the rejected-tree positions.
    #[cfg(feature = "hsd")]
    pub fn keep_kv_indices(&self, indices: &[u32]) -> Result<(), OCRError> {
        for layer in &self.layers {
            layer.keep_kv_indices(indices)?;
        }
        Ok(())
    }
}
