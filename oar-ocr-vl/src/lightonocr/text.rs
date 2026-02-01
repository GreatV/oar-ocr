use std::cell::RefCell;
use std::sync::Arc;

use candle_core::{IndexOp, Result, Tensor};
use candle_nn::{
    Activation, Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, kv_cache::KvCache,
    linear_b, rms_norm,
};

use super::config::LightOnOcrTextConfig;
use crate::attention::{RotaryEmbedding, repeat_kv, scaled_dot_product_attention};

struct Mlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: Activation,
}

impl Mlp {
    fn new(cfg: &LightOnOcrTextConfig, vb: VarBuilder) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let intermediate_sz = cfg.intermediate_size;
        let gate_proj = linear_b(hidden_sz, intermediate_sz, false, vb.pp("gate_proj"))?;
        let up_proj = linear_b(hidden_sz, intermediate_sz, false, vb.pp("up_proj"))?;
        let down_proj = linear_b(intermediate_sz, hidden_sz, false, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let lhs = self.gate_proj.forward(xs)?.apply(&self.act_fn)?;
        let rhs = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(lhs * rhs)?)
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    head_dim: usize,
    rotary_emb: Arc<RotaryEmbedding>,
    n_kv_groups: usize,
    softmax_scale: f64,
    kv_cache: RefCell<KvCache>,
}

impl Attention {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &LightOnOcrTextConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let hidden_sz = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let q_proj = linear_b(
            hidden_sz,
            num_heads * cfg.head_dim,
            cfg.attention_bias,
            vb.pp("q_proj"),
        )?;
        let k_proj = linear_b(
            hidden_sz,
            num_kv_heads * cfg.head_dim,
            cfg.attention_bias,
            vb.pp("k_proj"),
        )?;
        let v_proj = linear_b(
            hidden_sz,
            num_kv_heads * cfg.head_dim,
            cfg.attention_bias,
            vb.pp("v_proj"),
        )?;
        let o_proj = linear_b(
            num_heads * cfg.head_dim,
            hidden_sz,
            cfg.attention_bias,
            vb.pp("o_proj"),
        )?;
        let q_norm = rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;

        // Create KvCache with dim=2 for seq_len dimension
        // Cap initial capacity to avoid excessive memory allocation
        let kv_cache_capacity = cfg.max_position_embeddings.min(8192);
        let kv_cache = KvCache::new(2, kv_cache_capacity);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads,
            num_kv_heads,
            head_dim: cfg.head_dim,
            rotary_emb,
            n_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            softmax_scale: 1.0 / (cfg.head_dim as f64).sqrt(),
            kv_cache: RefCell::new(kv_cache),
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
    ) -> Result<Tensor> {
        let (b_sz, q_len, _) = xs.dims3()?;
        let mut q = self.q_proj.forward(xs)?;
        let mut k = self.k_proj.forward(xs)?;
        let mut v = self.v_proj.forward(xs)?;

        q = q
            .reshape((b_sz, q_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        k = k
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        v = v
            .reshape((b_sz, q_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        q = q.apply(&self.q_norm)?;
        k = k.apply(&self.k_norm)?;

        (q, k) = self.rotary_emb.apply_rotary_emb(&q, &k, seqlen_offsets)?;

        let q = q.contiguous()?;
        let k = k.contiguous()?;
        let v = v.contiguous()?;

        let (k, v) = self.kv_cache.borrow_mut().append(&k, &v)?;

        let k = repeat_kv(&k, self.n_kv_groups)?.contiguous()?;
        let v = repeat_kv(&v, self.n_kv_groups)?.contiguous()?;

        // Use unified attention implementation
        let is_causal = attention_mask.is_none();
        let attn_output = scaled_dot_product_attention(
            &q,
            &k,
            &v,
            attention_mask,
            self.softmax_scale,
            is_causal,
        )?;
        let attn_output = attn_output.transpose(1, 2)?.reshape((b_sz, q_len, ()))?;

        self.o_proj.forward(&attn_output)
    }

    fn clear_kv_cache(&self) {
        self.kv_cache.borrow_mut().reset();
    }
}

pub struct DecoderLayer {
    self_attn: Attention,
    mlp: Mlp,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl DecoderLayer {
    fn new(
        rotary_emb: Arc<RotaryEmbedding>,
        cfg: &LightOnOcrTextConfig,
        vb: VarBuilder,
    ) -> Result<Self> {
        let self_attn = Attention::new(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = Mlp::new(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
    ) -> Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self
            .self_attn
            .forward(&xs, attention_mask, seqlen_offsets)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = self
            .mlp
            .forward(&xs.apply(&self.post_attention_layernorm)?)?;
        residual + xs
    }

    fn clear_kv_cache(&self) {
        self.self_attn.clear_kv_cache();
    }
}

pub struct LightOnOcrTextModel {
    embed_tokens: Embedding,
    norm: RmsNorm,
    layers: Vec<DecoderLayer>,
    lm_head: Linear,
}

impl LightOnOcrTextModel {
    /// Maximum positions to precompute for RoPE embeddings.
    /// Capped to avoid excessive memory usage when config specifies very large values.
    const MAX_PRECOMPUTED_POSITIONS: usize = 8192;

    pub fn new(cfg: &LightOnOcrTextConfig, vb: VarBuilder) -> Result<Self> {
        let vb_m = vb.pp("model").pp("language_model");
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;

        // Cap precomputed RoPE size to avoid memory waste from large max_position_embeddings
        let rope_max_pos = cfg
            .max_position_embeddings
            .min(Self::MAX_PRECOMPUTED_POSITIONS);
        let rotary_emb = Arc::new(RotaryEmbedding::new_precomputed(
            cfg.rope_theta as f32,
            cfg.head_dim,
            rope_max_pos,
            vb.device(),
            vb_m.dtype(),
        )?);

        let vb_l = vb_m.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for layer_idx in 0..cfg.num_hidden_layers {
            layers.push(DecoderLayer::new(
                rotary_emb.clone(),
                cfg,
                vb_l.pp(layer_idx),
            )?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(embed_tokens.embeddings().clone(), None)
        } else {
            linear_b(cfg.hidden_size, cfg.vocab_size, false, vb.pp("lm_head"))?
        };

        Ok(Self {
            embed_tokens,
            norm,
            layers,
            lm_head,
        })
    }

    pub fn embed_tokens(&self, input_ids: &Tensor) -> Result<Tensor> {
        self.embed_tokens.forward(input_ids)
    }

    pub fn forward_embeds(
        &self,
        mut xs: Tensor,
        attention_mask: Option<&Tensor>,
        seqlen_offsets: &[usize],
    ) -> Result<Tensor> {
        let (_, seq_len, _) = xs.dims3()?;

        for layer in &self.layers {
            let mask = attention_mask
                .map(|m| m.to_device(xs.device()))
                .transpose()?;
            xs = layer.forward(&xs, mask.as_ref(), seqlen_offsets)?;
        }

        xs = xs.apply(&self.norm)?;

        self.lm_head
            .forward(&xs)?
            .i((.., seq_len - 1, ..))?
            .contiguous()
    }

    pub fn clear_kv_cache(&self) {
        for layer in &self.layers {
            layer.clear_kv_cache();
        }
    }
}
