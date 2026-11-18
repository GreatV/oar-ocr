//! Prefix-causal mBART decoder with tied embeddings and grouped-query attention.
use crate::runtime::{
    attention::{
        create_causal_mask, flash_attention, row_flash_attention_available,
        scaled_dot_product_attention_gqa,
    },
    cache::TrimmableKvCache,
};
use candle_core::{Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder, embedding, layer_norm, linear};

const HIDDEN: usize = 1024;
struct Layer {
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    norm: LayerNorm,
    ffn_norm: LayerNorm,
    up: Linear,
    down: Linear,
}
impl Layer {
    fn load(vb: VarBuilder) -> Result<Self> {
        let attn = vb.pp("self_attn");
        Ok(Self {
            q: linear(HIDDEN, HIDDEN, attn.pp("q_proj"))?,
            k: linear(HIDDEN, 512, attn.pp("k_proj"))?,
            v: linear(HIDDEN, 512, attn.pp("v_proj"))?,
            out: linear(HIDDEN, HIDDEN, attn.pp("out_proj"))?,
            norm: layer_norm(HIDDEN, 1e-5, vb.pp("self_attn_layer_norm"))?,
            ffn_norm: layer_norm(HIDDEN, 1e-5, vb.pp("final_layer_norm"))?,
            up: linear(HIDDEN, 4096, vb.pp("fc1"))?,
            down: linear(4096, HIDDEN, vb.pp("fc2"))?,
        })
    }
    fn forward(
        &self,
        x: &Tensor,
        cache: &mut TrimmableKvCache,
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let norm = x.apply(&self.norm)?;
        let (b, n, _) = x.dims3()?;
        let project = |layer: &Linear, heads| {
            layer
                .forward(&norm)?
                .reshape((b, n, heads, 64))?
                .transpose(1, 2)?
                .contiguous()
        };
        let q = project(&self.q, 16)?;
        let (k, v) = cache.append(&project(&self.k, 8)?, &project(&self.v, 8)?)?;
        let causal_prefill = n > 1 && mask.is_none();
        let ctx = if causal_prefill && let Some(context) = flash_attention(&q, &k, &v, 0.125, true)?
        {
            context
        } else {
            scaled_dot_product_attention_gqa(&q, &k, &v, mask, 0.125, causal_prefill, 2)?
        };
        let ctx = ctx.transpose(1, 2)?.reshape((b, n, HIDDEN))?;
        let hidden = (x + ctx.apply(&self.out)?)?;
        let update = hidden
            .apply(&self.ffn_norm)?
            .apply(&self.up)?
            .gelu_erf()?
            .apply(&self.down)?;
        hidden + update
    }
}
pub(super) struct Decoder {
    tokens: Embedding,
    norm_in: LayerNorm,
    layers: Vec<Layer>,
    norm_out: LayerNorm,
    lm_head: Linear,
}
impl Decoder {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let tokens = embedding(72512, HIDDEN, vb.pp("embed_tokens"))?;
        Ok(Self {
            lm_head: Linear::new(tokens.embeddings().clone(), None),
            tokens,
            norm_in: layer_norm(HIDDEN, 1e-5, vb.pp("layernorm_embedding"))?,
            norm_out: layer_norm(HIDDEN, 1e-5, vb.pp("layer_norm"))?,
            layers: (0..10)
                .map(|i| Layer::load(vb.pp("layers").pp(i)))
                .collect::<Result<_>>()?,
        })
    }
    pub fn cache(&self, capacity: usize) -> Vec<TrimmableKvCache> {
        (0..self.layers.len())
            .map(|_| TrimmableKvCache::new(2, capacity))
            .collect()
    }
    pub fn forward(
        &self,
        ids: &Tensor,
        prefix: Option<&Tensor>,
        cache: &mut [TrimmableKvCache],
    ) -> Result<Tensor> {
        let text = (self.tokens.forward(ids)? * 32.)?;
        let embeddings = match prefix {
            Some(vision) => Tensor::cat(&[vision, &text], 1)?,
            None => text,
        };
        let n = embeddings.dim(1)?;
        let past = cache[0].current_seq_len();
        let flash_prefill =
            past == 0 && row_flash_attention_available(embeddings.device(), embeddings.dtype());
        let mask = if n > 1 && !flash_prefill {
            Some(create_causal_mask(
                n,
                n + past,
                embeddings.dtype(),
                embeddings.device(),
            )?)
        } else {
            None
        };
        let mut hidden = embeddings.apply(&self.norm_in)?;
        for (layer, kv) in self.layers.iter().zip(cache.iter_mut()) {
            hidden = layer.forward(&hidden, kv, mask.as_ref())?;
        }
        hidden
            .narrow(1, n - 1, 1)?
            .squeeze(1)?
            .apply(&self.norm_out)?
            .apply(&self.lm_head)?
            .squeeze(0)
    }
}
