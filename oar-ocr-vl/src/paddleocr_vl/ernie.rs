use super::config::PaddleOcrVlConfig;
use crate::attention::{
    RotaryEmbedding, repeat_kv, scaled_dot_product_attention, select_rope_sections,
};
use crate::utils::{candle_to_ocr_inference, candle_to_ocr_processing, rotate_half};
use candle_core::Tensor;
use candle_nn::{Module, kv_cache::KvCache};
use oar_ocr_core::core::OCRError;
use std::cell::RefCell;

fn apply_multimodal_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mrope_section: &[usize],
) -> Result<(Tensor, Tensor), OCRError> {
    // MRoPE uses 3 position dimensions
    let cos = select_rope_sections(cos, mrope_section, 3)?;
    let sin = select_rope_sections(sin, mrope_section, 3)?;
    let q_mul = q.broadcast_mul(&cos).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "PaddleOCR-VL: mrope q*cos failed",
            e,
        )
    })?;
    let q_half = rotate_half(q)?;
    let q_half_mul = q_half.broadcast_mul(&sin).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "PaddleOCR-VL: mrope rotate_half(q)*sin failed",
            e,
        )
    })?;
    let q_rot = (&q_mul + &q_half_mul).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "PaddleOCR-VL: mrope apply on q failed",
            e,
        )
    })?;

    let k_mul = k.broadcast_mul(&cos).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "PaddleOCR-VL: mrope k*cos failed",
            e,
        )
    })?;
    let k_half = rotate_half(k)?;
    let k_half_mul = k_half.broadcast_mul(&sin).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "PaddleOCR-VL: mrope rotate_half(k)*sin failed",
            e,
        )
    })?;
    let k_rot = (&k_mul + &k_half_mul).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "PaddleOCR-VL: mrope apply on k failed",
            e,
        )
    })?;
    Ok((q_rot, k_rot))
}

#[derive(Debug, Clone)]
struct Ernie4_5Mlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl Ernie4_5Mlp {
    fn load(cfg: &PaddleOcrVlConfig, vb: candle_nn::VarBuilder) -> Result<Self, OCRError> {
        let gate_proj = candle_nn::linear_b(
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.use_bias,
            vb.pp("gate_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load gate_proj", e))?;
        let up_proj = candle_nn::linear_b(
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.use_bias,
            vb.pp("up_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load up_proj", e))?;
        let down_proj = candle_nn::linear_b(
            cfg.intermediate_size,
            cfg.hidden_size,
            cfg.use_bias,
            vb.pp("down_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load down_proj", e))?;
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
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "mlp gate_proj", e))?;
        let gate = candle_nn::ops::silu(&gate)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "mlp silu", e))?;
        let up = self
            .up_proj
            .forward(xs)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "mlp up_proj", e))?;
        let prod =
            (&gate * &up).map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "mlp gate*up", e))?;
        self.down_proj
            .forward(&prod)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "mlp down_proj", e))
    }
}

#[derive(Debug)]
struct Ernie4_5Attention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
    mrope_section: Vec<usize>,
    kv_cache: RefCell<KvCache>,
}

impl Ernie4_5Attention {
    fn load(cfg: &PaddleOcrVlConfig, vb: candle_nn::VarBuilder) -> Result<Self, OCRError> {
        let q_proj = candle_nn::linear_b(
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.head_dim,
            cfg.use_bias,
            vb.pp("q_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load q_proj", e))?;
        let k_proj = candle_nn::linear_b(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            cfg.use_bias,
            vb.pp("k_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load k_proj", e))?;
        let v_proj = candle_nn::linear_b(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            cfg.use_bias,
            vb.pp("v_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load v_proj", e))?;
        let o_proj = candle_nn::linear_b(
            cfg.num_attention_heads * cfg.head_dim,
            cfg.hidden_size,
            cfg.use_bias,
            vb.pp("o_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load o_proj", e))?;

        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            return Err(OCRError::ConfigError {
                message: format!(
                    "PaddleOCR-VL: num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                    cfg.num_attention_heads, cfg.num_key_value_heads
                ),
            });
        }

        // Create KvCache with dim=2 for seq_len dimension, use 8192 as reasonable max length
        // Create KvCache with dim=2 for seq_len dimension
        // Pre-allocate enough space to avoid O(N) reallocation during generation
        // Conservative estimate: vision tokens + max_generation_tokens
        // Typical: ~1000-2000 vision tokens + 4096 generation tokens = ~6000-8000 total
        // Use 16384 to handle worst case without reallocation
        let kv_cache = KvCache::new(2, 16384);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            scaling: (cfg.head_dim as f64).powf(-0.5),
            mrope_section: cfg.rope_scaling.mrope_section.clone(),
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
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn hidden_states dims3", e))?;

        let q = self
            .q_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn q_proj", e))?
            .reshape((b, seq_len, self.num_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn q reshape", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn q transpose", e))?;

        let k = self
            .k_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn k_proj", e))?
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn k reshape", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn k transpose", e))?;

        let v = self
            .v_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn v_proj", e))?
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn v reshape", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn v transpose", e))?;

        let (q, k) = apply_multimodal_rotary_pos_emb(&q, &k, cos, sin, &self.mrope_section)?;

        let q = q
            .contiguous()
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn q contiguous", e))?;
        let k = k
            .contiguous()
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn k contiguous", e))?;
        let v = v
            .contiguous()
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn v contiguous", e))?;

        let (k_all, v_all) = self
            .kv_cache
            .borrow_mut()
            .append(&k, &v)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "append kv_cache", e))?;

        let key_states = repeat_kv(&k_all, self.num_kv_groups)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "repeat_kv key", e))?;
        let value_states = repeat_kv(&v_all, self.num_kv_groups)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "repeat_kv value", e))?;

        let key_states = key_states.contiguous().map_err(|e| {
            candle_to_ocr_inference("PaddleOCR-VL", "attn key_states contiguous", e)
        })?;
        let value_states = value_states.contiguous().map_err(|e| {
            candle_to_ocr_inference("PaddleOCR-VL", "attn value_states contiguous", e)
        })?;

        // Use unified attention implementation
        let attn_output = scaled_dot_product_attention(
            &q,
            &key_states,
            &value_states,
            causal_mask,
            self.scaling,
            false, // not causal - mask is explicit
        )
        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "scaled_dot_product_attention", e))?;

        let attn_output = attn_output
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn out transpose", e))?
            .reshape((b, seq_len, self.num_heads * self.head_dim))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn out reshape", e))?;

        self.o_proj
            .forward(&attn_output)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn o_proj", e))
    }

    fn clear_kv_cache(&self) {
        self.kv_cache.borrow_mut().reset();
    }
}

#[derive(Debug)]
struct Ernie4_5DecoderLayer {
    self_attn: Ernie4_5Attention,
    mlp: Ernie4_5Mlp,
    input_layernorm: candle_nn::RmsNorm,
    post_attention_layernorm: candle_nn::RmsNorm,
}

impl Ernie4_5DecoderLayer {
    fn load(cfg: &PaddleOcrVlConfig, vb: candle_nn::VarBuilder) -> Result<Self, OCRError> {
        let self_attn = Ernie4_5Attention::load(cfg, vb.pp("self_attn"))?;
        let mlp = Ernie4_5Mlp::load(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))
                .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load input_layernorm", e))?;
        let post_attention_layernorm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )
        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load post_attention_layernorm", e))?;
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
        let residual = hidden_states.clone();
        let hidden_states = self
            .input_layernorm
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "llm input_layernorm", e))?;
        let attn_out = self
            .self_attn
            .forward(&hidden_states, cos, sin, causal_mask)?;
        let hidden_states = (&residual + &attn_out)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "llm attn residual add", e))?;

        let residual = hidden_states.clone();
        let hidden_states = self
            .post_attention_layernorm
            .forward(&hidden_states)
            .map_err(|e| {
                candle_to_ocr_inference("PaddleOCR-VL", "llm post_attention_layernorm", e)
            })?;
        let mlp_out = self.mlp.forward(&hidden_states)?;
        (&residual + &mlp_out)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "llm mlp residual add", e))
    }

    fn clear_kv_cache(&self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug)]
pub struct Ernie4_5Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Ernie4_5DecoderLayer>,
    norm: candle_nn::RmsNorm,
    rotary: RotaryEmbedding,
}

impl Ernie4_5Model {
    pub fn load(cfg: &PaddleOcrVlConfig, vb: candle_nn::VarBuilder) -> Result<Self, OCRError> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))
                .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load embed_tokens", e))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Ernie4_5DecoderLayer::load(
                cfg,
                vb.pp(format!("layers.{i}")),
            )?);
        }
        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load final norm", e))?;
        let rotary = RotaryEmbedding::new_multi_axis(cfg.head_dim, cfg.rope_theta, 3, vb.device())?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary,
        })
    }

    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor, OCRError> {
        self.embed_tokens
            .forward(input_ids)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "embed tokens", e))
    }

    pub fn forward(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        causal_mask: Option<&Tensor>,
    ) -> Result<Tensor, OCRError> {
        let (cos, sin) = self
            .rotary
            .forward_multi_axis(position_ids, inputs_embeds.dtype())?;

        let mut hidden_states = inputs_embeds.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &cos, &sin, causal_mask)?;
        }
        let hidden_states = self
            .norm
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "llm final norm", e))?;
        Ok(hidden_states)
    }

    pub fn clear_kv_cache(&self) {
        for layer in &self.layers {
            layer.clear_kv_cache();
        }
    }
}
