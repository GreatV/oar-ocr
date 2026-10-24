use super::config::TextConfig;
use super::layers::{Mlp, cos_sin, rotary};
use crate::runtime::attention::{
    ATTENTION_CHUNK_SCRATCH_BUDGET, attention_query_chunk, flash_attention,
    scaled_dot_product_attention,
};
use crate::runtime::cache::TrimmableKvCache;
use candle_core::{DType, Result, Tensor};
use candle_nn::{
    Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_no_bias, rms_norm,
};
use std::cell::RefCell;

struct Attention {
    query_a: Option<(Linear, RmsNorm)>,
    query_b: Linear,
    kv_a: Linear,
    kv_norm: RmsNorm,
    kv_b: Linear,
    key_weight: Tensor,
    value_weight: Tensor,
    output: Linear,
    cfg: TextConfig,
    cache: RefCell<TrimmableKvCache>,
}

impl Attention {
    fn load(cfg: &TextConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let query_a = cfg
            .q_lora_rank
            .map(|rank| {
                Ok::<_, candle_core::Error>((
                    linear_no_bias(cfg.hidden_size, rank, vb.pp("q_a_proj"))?,
                    rms_norm(rank, 1e-6, vb.pp("q_a_layernorm"))?,
                ))
            })
            .transpose()?;
        let query_b = linear_no_bias(
            cfg.q_lora_rank.unwrap_or(cfg.hidden_size),
            cfg.num_attention_heads * cfg.qk_head_dim,
            vb.pp(if query_a.is_some() {
                "q_b_proj"
            } else {
                "q_proj"
            }),
        )?;
        let kv_b = linear_no_bias(
            cfg.kv_lora_rank,
            cfg.num_attention_heads * (cfg.qk_nope_head_dim + cfg.v_head_dim),
            vb.pp("kv_b_proj"),
        )?;
        let weights = kv_b.weight().reshape((
            cfg.num_attention_heads,
            cfg.qk_nope_head_dim + cfg.v_head_dim,
            cfg.kv_lora_rank,
        ))?;
        Ok(Self {
            query_a,
            query_b,
            kv_a: linear_no_bias(
                cfg.hidden_size,
                cfg.kv_lora_rank + cfg.qk_rope_head_dim,
                vb.pp("kv_a_proj_with_mqa"),
            )?,
            kv_norm: rms_norm(cfg.kv_lora_rank, 1e-6, vb.pp("kv_a_layernorm"))?,
            key_weight: weights
                .narrow(1, 0, cfg.qk_nope_head_dim)?
                .to_dtype(DType::F32)?
                .contiguous()?
                .unsqueeze(0)?,
            value_weight: weights
                .narrow(1, cfg.qk_nope_head_dim, cfg.v_head_dim)?
                .transpose(1, 2)?
                .to_dtype(DType::F32)?
                .contiguous()?
                .unsqueeze(0)?,
            kv_b,
            output: linear_no_bias(
                cfg.num_attention_heads * cfg.v_head_dim,
                cfg.hidden_size,
                vb.pp("o_proj"),
            )?,
            cfg: cfg.clone(),
            cache: RefCell::new(TrimmableKvCache::new(2, cfg.max_position_embeddings)),
        })
    }

    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let c = &self.cfg;
        let (batch, len, _) = xs.dims3()?;
        let queries = match &self.query_a {
            Some((projection, norm)) => norm.forward(&projection.forward(xs)?)?,
            None => xs.clone(),
        };
        let queries = self
            .query_b
            .forward(&queries)?
            .reshape((batch, len, c.num_attention_heads, c.qk_head_dim))?
            .transpose(1, 2)?;
        let q_pass = queries.narrow(3, 0, c.qk_nope_head_dim)?.contiguous()?;
        let q_rot = rotary(
            &queries.narrow(3, c.qk_nope_head_dim, c.qk_rope_head_dim)?,
            cos,
            sin,
            c.rope_interleave,
            false,
        )?;
        let compressed = self.kv_a.forward(xs)?;
        let latent = self
            .kv_norm
            .forward(&compressed.narrow(2, 0, c.kv_lora_rank)?.contiguous()?)?
            .unsqueeze(1)?;
        let k_rot = rotary(
            &compressed
                .narrow(2, c.kv_lora_rank, c.qk_rope_head_dim)?
                .unsqueeze(1)?,
            cos,
            sin,
            c.rope_interleave,
            false,
        )?;
        let (k_rot, latent) = self.append_cache(&k_rot, &latent)?;
        let out = if len == 1 {
            self.decode(&q_pass, &q_rot, &k_rot, &latent)?
        } else {
            // The expanded prefill projections retain their original dtype.
            self.prefill(
                &q_pass,
                &q_rot,
                &k_rot.to_dtype(xs.dtype())?,
                &latent.to_dtype(xs.dtype())?,
            )?
        };
        self.output.forward(
            &out.transpose(1, 2)?
                .reshape((batch, len, c.num_attention_heads * c.v_head_dim))?
                .contiguous()?,
        )
    }

    fn append_cache(&self, k_rot: &Tensor, latent: &Tensor) -> Result<(Tensor, Tensor)> {
        // Decode consumes F32 keys and latents. Convert only newly appended
        // values, after their native-dtype projection/normalization/RoPE.
        self.cache.borrow_mut().append(
            &k_rot.contiguous()?.to_dtype(DType::F32)?,
            &latent.contiguous()?.to_dtype(DType::F32)?,
        )
    }

    fn decode(
        &self,
        q_pass: &Tensor,
        q_rot: &Tensor,
        k_rot: &Tensor,
        latent: &Tensor,
    ) -> Result<Tensor> {
        let (batch, heads, _, _) = q_pass.dims4()?;
        // Absorb the key/value up-projections; cache one latent and one rotary key per token.
        // Keep the reassociated products in F32 to avoid additional half-precision rounding.
        let q_latent = q_pass
            .to_dtype(DType::F32)?
            .broadcast_matmul(&self.key_weight)?
            .reshape((batch, heads, self.cfg.kv_lora_rank))?;
        let latent = latent.squeeze(1)?.contiguous()?;
        let scores = (q_latent.matmul(&latent.transpose(1, 2)?)?
            + q_rot
                .to_dtype(DType::F32)?
                .reshape((batch, heads, self.cfg.qk_rope_head_dim))?
                .matmul(&k_rot.squeeze(1)?.transpose(1, 2)?)?)?;
        let scores = (scores * (self.cfg.qk_head_dim as f64).powf(-0.5))?;
        let probabilities = candle_nn::ops::softmax_last_dim(&scores)?;
        probabilities
            .matmul(&latent)?
            .unsqueeze(2)?
            .broadcast_matmul(&self.value_weight)?
            .to_dtype(q_pass.dtype())
    }

    fn prefill(
        &self,
        q_pass: &Tensor,
        q_rot: &Tensor,
        k_rot: &Tensor,
        latent: &Tensor,
    ) -> Result<Tensor> {
        let c = &self.cfg;
        let (batch, _, kv_len, _) = latent.dims4()?;
        let expanded = self
            .kv_b
            .forward(&latent.squeeze(1)?.contiguous()?)?
            .reshape((
                batch,
                kv_len,
                c.num_attention_heads,
                c.qk_nope_head_dim + c.v_head_dim,
            ))?
            .transpose(1, 2)?;
        let keys = Tensor::cat(
            &[
                &expanded.narrow(3, 0, c.qk_nope_head_dim)?,
                &k_rot.broadcast_as((batch, c.num_attention_heads, kv_len, c.qk_rope_head_dim))?,
            ],
            3,
        )?
        .contiguous()?;
        let queries = Tensor::cat(&[q_pass, q_rot], 3)?.contiguous()?;
        let values = expanded
            .narrow(3, c.qk_nope_head_dim, c.v_head_dim)?
            .contiguous()?;
        let scale = (c.qk_head_dim as f64).powf(-0.5);
        let padded_values = if c.v_head_dim != c.qk_head_dim {
            Tensor::cat(
                &[
                    &values,
                    &Tensor::zeros(
                        (
                            batch,
                            c.num_attention_heads,
                            kv_len,
                            c.qk_head_dim - c.v_head_dim,
                        ),
                        values.dtype(),
                        values.device(),
                    )?,
                ],
                3,
            )?
        } else {
            values.clone()
        };
        if let Some(out) = flash_attention(&queries, &keys, &padded_values, scale, true)? {
            return out.narrow(3, 0, c.v_head_dim)?.contiguous();
        }
        let len = queries.dim(2)?;
        let past = kv_len - len;
        let chunk = attention_query_chunk(
            c.num_attention_heads,
            kv_len,
            ATTENTION_CHUNK_SCRATCH_BUDGET,
        )
        .min(256);
        let mut outputs = Vec::new();
        for start in (0..len).step_by(chunk) {
            let count = (len - start).min(chunk);
            let end = past + start + count;
            outputs.push(scaled_dot_product_attention(
                &queries.narrow(2, start, count)?,
                &keys.narrow(2, 0, end)?,
                &values.narrow(2, 0, end)?,
                None,
                scale,
                true,
            )?);
        }
        Tensor::cat(&outputs, 2)
    }
}

struct Layer {
    attention: Attention,
    input_norm: RmsNorm,
    post_norm: RmsNorm,
    mlp: Mlp,
}

impl Layer {
    fn load(cfg: &TextConfig, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            attention: Attention::load(cfg, vb.pp("self_attn"))?,
            input_norm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_norm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            mlp: Mlp::load(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"))?,
        })
    }

    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let xs = (xs
            + self
                .attention
                .forward(&self.input_norm.forward(xs)?, cos, sin)?)?;
        &xs + self.mlp.forward(&self.post_norm.forward(&xs)?)?
    }
}

pub(super) struct TextModel {
    pub embedding: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    cfg: TextConfig,
}

impl TextModel {
    pub fn load(cfg: &TextConfig, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            embedding: embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))?,
            layers: (0..cfg.num_hidden_layers)
                .map(|i| Layer::load(cfg, vb.pp(format!("layers.{i}"))))
                .collect::<Result<_>>()?,
            norm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?,
            cfg: cfg.clone(),
        })
    }

    pub fn forward(&self, xs: &Tensor, offset: usize) -> Result<Tensor> {
        let len = xs.dim(1)?;
        let half = self.cfg.qk_rope_head_dim / 2;
        let mut angles = Vec::with_capacity(len * half);
        for pos in offset..offset + len {
            for i in 0..half {
                angles.push(
                    pos as f32
                        * (self
                            .cfg
                            .rope_parameters
                            .rope_theta
                            .powf(-(i as f64) / half as f64) as f32),
                );
            }
        }
        let (cos, sin) = cos_sin(angles, len, half, xs.dtype(), xs.device())?;
        let mut xs = xs.clone();
        for layer in &self.layers {
            xs = layer.forward(&xs, &cos, &sin)?;
        }
        self.norm.forward(&xs)
    }

    pub fn clear_cache(&self) {
        for layer in &self.layers {
            layer.attention.cache.borrow_mut().reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    #[test]
    fn incremental_f32_cache_matches_full_cache_cast() -> Result<()> {
        let reference: serde_json::Value =
            serde_json::from_str(include_str!("fixtures/reference.json")).unwrap();
        let cfg: TextConfig =
            serde_json::from_value(reference["config"]["text_config"].clone()).unwrap();
        let device = Device::Cpu;
        let vb = VarBuilder::from_slice_safetensors(
            include_bytes!("fixtures/tiny.safetensors"),
            DType::F32,
            &device,
        )?;
        let attention = Attention::load(&cfg, vb.pp("model.language_model.layers.0.self_attn"))?;
        for dtype in [DType::F32, DType::BF16] {
            for batch in [1, 2] {
                attention.cache.borrow_mut().reset();
                let mut native = TrimmableKvCache::new(2, 32);
                let tensor = |shape: (usize, usize, usize, usize), offset: usize| {
                    let count = shape.0 * shape.1 * shape.2 * shape.3;
                    Tensor::from_vec(
                        (offset..offset + count)
                            .map(|i| (i as f32 * 0.17).sin())
                            .collect(),
                        shape,
                        &device,
                    )?
                    .to_dtype(dtype)
                };
                for len in [3, 1, 1, 2] {
                    let k = tensor((batch, 1, len, cfg.qk_rope_head_dim), len)?;
                    let v = tensor((batch, 1, len, cfg.kv_lora_rank), len + 7)?;
                    let (old_k, old_v) = native.append(&k, &v)?;
                    let (new_k, new_v) = attention.append_cache(&k, &v)?;
                    let q = tensor(
                        (batch, cfg.num_attention_heads, 1, cfg.qk_nope_head_dim),
                        11,
                    )?;
                    let qr = tensor(
                        (batch, cfg.num_attention_heads, 1, cfg.qk_rope_head_dim),
                        19,
                    )?;
                    // Old decode widens the entire native cache on every step.
                    let old = attention.decode(
                        &q,
                        &qr,
                        &old_k.to_dtype(DType::F32)?,
                        &old_v.to_dtype(DType::F32)?,
                    )?;
                    let new = attention.decode(&q, &qr, &new_k, &new_v)?;
                    let bits = |t: Tensor| -> Result<Vec<u32>> {
                        Ok(t.to_dtype(DType::F32)?
                            .flatten_all()?
                            .to_vec1::<f32>()?
                            .into_iter()
                            .map(f32::to_bits)
                            .collect())
                    };
                    assert_eq!(bits(new)?, bits(old)?);
                }
            }
        }
        Ok(())
    }
}
