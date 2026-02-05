use super::config::MinerUConfig;
use crate::attention::{
    RotaryEmbedding, repeat_kv, scaled_dot_product_attention, select_rope_sections,
};
use crate::utils::{candle_to_ocr_inference, candle_to_ocr_processing, rotate_half};
use candle_core::Tensor;
use candle_nn::{
    Embedding, Linear, Module, VarBuilder, embedding, kv_cache::KvCache, linear, linear_no_bias,
    rms_norm,
};
use oar_ocr_core::core::OCRError;
use std::cell::RefCell;
use std::sync::Arc;

fn apply_multimodal_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mrope_section: &[usize],
) -> Result<(Tensor, Tensor), OCRError> {
    let cos = select_rope_sections(cos, mrope_section, 3)?;
    let sin = select_rope_sections(sin, mrope_section, 3)?;

    let q_mul = q.broadcast_mul(&cos).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "MinerU2.5: mrope q*cos failed",
            e,
        )
    })?;
    let q_half = rotate_half(q)?;
    let q_half_mul = q_half.broadcast_mul(&sin).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "MinerU2.5: mrope rotate_half(q)*sin failed",
            e,
        )
    })?;
    let q_rot = (&q_mul + &q_half_mul).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "MinerU2.5: mrope apply on q failed",
            e,
        )
    })?;

    let k_mul = k.broadcast_mul(&cos).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "MinerU2.5: mrope k*cos failed",
            e,
        )
    })?;
    let k_half = rotate_half(k)?;
    let k_half_mul = k_half.broadcast_mul(&sin).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "MinerU2.5: mrope rotate_half(k)*sin failed",
            e,
        )
    })?;
    let k_rot = (&k_mul + &k_half_mul).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "MinerU2.5: mrope apply on k failed",
            e,
        )
    })?;

    Ok((q_rot, k_rot))
}

#[derive(Debug, Clone)]
struct MinerUMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl MinerUMlp {
    fn load(cfg: &MinerUConfig, vb: VarBuilder) -> Result<Self, OCRError> {
        let gate_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("mlp.gate_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("MinerU2.5", "load gate_proj", e))?;
        let up_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp.up_proj"))
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "load up_proj", e))?;
        let down_proj = linear_no_bias(
            cfg.intermediate_size,
            cfg.hidden_size,
            vb.pp("mlp.down_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("MinerU2.5", "load down_proj", e))?;
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
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "mlp gate_proj", e))?;
        let gate = candle_nn::ops::silu(&gate)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "mlp silu", e))?;
        let up = self
            .up_proj
            .forward(xs)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "mlp up_proj", e))?;
        let prod =
            (&gate * &up).map_err(|e| candle_to_ocr_inference("MinerU2.5", "mlp gate*up", e))?;
        self.down_proj
            .forward(&prod)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "mlp down_proj", e))
    }
}

#[derive(Debug)]
struct MinerUAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
    mrope_section: Vec<usize>,
    kv_cache: RefCell<KvCache>,
}

impl MinerUAttention {
    fn load(cfg: &MinerUConfig, vb: VarBuilder) -> Result<Self, OCRError> {
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            return Err(OCRError::ConfigError {
                message: format!(
                    "MinerU2.5: num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                    cfg.num_attention_heads, cfg.num_key_value_heads
                ),
            });
        }
        let head_dim = cfg.head_dim()?;
        let q_proj = linear(
            cfg.hidden_size,
            cfg.num_attention_heads * head_dim,
            vb.pp("self_attn.q_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("MinerU2.5", "load q_proj", e))?;
        let k_proj = linear(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("self_attn.k_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("MinerU2.5", "load k_proj", e))?;
        let v_proj = linear(
            cfg.hidden_size,
            cfg.num_key_value_heads * head_dim,
            vb.pp("self_attn.v_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("MinerU2.5", "load v_proj", e))?;
        let o_proj = linear_no_bias(
            cfg.num_attention_heads * head_dim,
            cfg.hidden_size,
            vb.pp("self_attn.o_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("MinerU2.5", "load o_proj", e))?;

        let kv_cache = KvCache::new(2, cfg.max_position_embeddings.max(8192));

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim,
            scaling: (head_dim as f64).powf(-0.5),
            mrope_section: cfg.rope_scaling.mrope_section.clone(),
            kv_cache: RefCell::new(kv_cache),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, OCRError> {
        let (b, seq_len, _) = hidden_states
            .dims3()
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "attn hidden_states dims3", e))?;

        let q = self
            .q_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "attn q_proj", e))?
            .reshape((b, seq_len, self.num_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "attn q reshape", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "attn q transpose", e))?;

        let k = self
            .k_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "attn k_proj", e))?
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "attn k reshape", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "attn k transpose", e))?;

        let v = self
            .v_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "attn v_proj", e))?
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "attn v reshape", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "attn v transpose", e))?;

        let (q, k) = apply_multimodal_rotary_pos_emb(&q, &k, cos, sin, &self.mrope_section)?;
        let k = k
            .contiguous()
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "attn k contiguous", e))?;
        let v = v
            .contiguous()
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "attn v contiguous", e))?;

        let (k, v) = self
            .kv_cache
            .borrow_mut()
            .append(&k, &v)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "attn kv_cache append", e))?;
        let k = repeat_kv(&k, self.num_kv_groups)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "repeat kv k", e))?;
        let v = repeat_kv(&v, self.num_kv_groups)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "repeat kv v", e))?;

        let is_causal = attention_mask.is_none();
        let attn_output =
            scaled_dot_product_attention(&q, &k, &v, attention_mask, self.scaling, is_causal)
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "attn scaled_dot_product", e))?;
        let attn_output = attn_output
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "attn output transpose", e))?
            .reshape((b, seq_len, self.num_heads * self.head_dim))
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "attn output reshape", e))?;

        self.o_proj
            .forward(&attn_output)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "attn o_proj", e))
    }

    fn clear_kv_cache(&self) {
        self.kv_cache.borrow_mut().reset();
    }
}

pub struct MinerUDecoderLayer {
    self_attn: MinerUAttention,
    mlp: MinerUMlp,
    input_layernorm: candle_nn::RmsNorm,
    post_attention_layernorm: candle_nn::RmsNorm,
}

impl MinerUDecoderLayer {
    fn load(cfg: &MinerUConfig, vb: VarBuilder) -> Result<Self, OCRError> {
        let self_attn = MinerUAttention::load(cfg, vb.clone())?;
        let mlp = MinerUMlp::load(cfg, vb.clone())?;
        let input_layernorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "load input_layernorm", e))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )
        .map_err(|e| candle_to_ocr_inference("MinerU2.5", "load post_attention_layernorm", e))?;
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
    ) -> Result<Tensor, OCRError> {
        let residual = hidden_states.clone();
        let hidden_states = self
            .input_layernorm
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "input_layernorm", e))?;
        let hidden_states = self
            .self_attn
            .forward(&hidden_states, cos, sin, attention_mask)?;
        let hidden_states = (&residual + &hidden_states).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "MinerU2.5: attn residual add failed",
                e,
            )
        })?;

        let residual = hidden_states.clone();
        let hidden_states = self
            .post_attention_layernorm
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "post_attention_layernorm", e))?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        (&residual + &hidden_states).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "MinerU2.5: mlp residual add failed",
                e,
            )
        })
    }

    fn clear_kv_cache(&self) {
        self.self_attn.clear_kv_cache();
    }
}

pub struct MinerUTextModel {
    embed_tokens: Embedding,
    layers: Vec<MinerUDecoderLayer>,
    norm: candle_nn::RmsNorm,
    rotary_emb: Arc<RotaryEmbedding>,
}

impl MinerUTextModel {
    pub fn load(cfg: &MinerUConfig, vb: VarBuilder) -> Result<Self, OCRError> {
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "load embed_tokens", e))?;

        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer_vb = vb.pp(format!("layers.{i}"));
            layers.push(MinerUDecoderLayer::load(cfg, layer_vb)?);
        }

        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "load norm", e))?;
        let head_dim = cfg.head_dim()?;
        let rotary_emb = Arc::new(RotaryEmbedding::new_multi_axis(
            head_dim,
            cfg.rope_theta,
            3,
            vb.device(),
        )?);

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
        })
    }

    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor, OCRError> {
        self.embed_tokens
            .forward(input_ids)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "embed forward", e))
    }

    pub fn forward(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, OCRError> {
        let (cos, sin) = self
            .rotary_emb
            .forward_multi_axis(position_ids, inputs_embeds.dtype())?;

        let mut hidden_states = inputs_embeds.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &cos, &sin, attention_mask)?;
        }
        self.norm
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "norm forward", e))
    }

    pub fn token_embedding_weight(&self) -> Tensor {
        self.embed_tokens.embeddings().clone()
    }

    pub fn clear_kv_cache(&self) {
        for layer in &self.layers {
            layer.clear_kv_cache();
        }
    }
}
