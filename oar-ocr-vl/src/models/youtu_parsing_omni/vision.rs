use super::config::OmniConfig;
use super::layers::{Mlp, cos_sin, rotary};
use crate::runtime::attention::{
    ATTENTION_CHUNK_SCRATCH_BUDGET, attention_query_chunk, flash_attention,
    scaled_dot_product_attention_gqa,
};
use candle_core::{DType, Result, Tensor};
use candle_nn::{Linear, Module, RmsNorm, VarBuilder, linear, linear_no_bias, rms_norm};

struct Layer {
    input_norm: RmsNorm,
    post_norm: RmsNorm,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    q: Linear,
    k: Linear,
    v: Linear,
    out: Linear,
    mlp: Mlp,
    cfg: OmniConfig,
}

impl Layer {
    fn load(cfg: &OmniConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let attn = vb.pp("self_attn");
        Ok(Self {
            input_norm: rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?,
            post_norm: rms_norm(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
            q_norm: rms_norm(cfg.head_dim, cfg.rms_norm_eps, attn.pp("q_norm"))?,
            k_norm: rms_norm(cfg.head_dim, cfg.rms_norm_eps, attn.pp("k_norm"))?,
            q: linear_no_bias(
                cfg.hidden_size,
                cfg.num_attention_heads * cfg.head_dim,
                attn.pp("q_proj"),
            )?,
            k: linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                attn.pp("k_proj"),
            )?,
            v: linear_no_bias(
                cfg.hidden_size,
                cfg.num_key_value_heads * cfg.head_dim,
                attn.pp("v_proj"),
            )?,
            out: linear_no_bias(
                cfg.num_attention_heads * cfg.head_dim,
                cfg.hidden_size,
                attn.pp("o_proj"),
            )?,
            mlp: Mlp::load(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp"))?,
            cfg: cfg.clone(),
        })
    }

    fn forward(&self, xs: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor> {
        let c = &self.cfg;
        let len = xs.dim(0)?;
        let normalized = self.input_norm.forward(xs)?;
        let queries = self
            .q_norm
            .forward(&self.q.forward(&normalized)?.reshape((
                len,
                c.num_attention_heads,
                c.head_dim,
            ))?)?
            .transpose(0, 1)?
            .unsqueeze(0)?;
        let keys = self
            .k_norm
            .forward(&self.k.forward(&normalized)?.reshape((
                len,
                c.num_key_value_heads,
                c.head_dim,
            ))?)?
            .transpose(0, 1)?
            .unsqueeze(0)?;
        let queries = rotary(&queries, cos, sin, false, true)?.contiguous()?;
        let keys = rotary(&keys, cos, sin, false, true)?.contiguous()?;
        let values = self
            .v
            .forward(&normalized)?
            .reshape((len, c.num_key_value_heads, c.head_dim))?
            .transpose(0, 1)?
            .unsqueeze(0)?
            .contiguous()?;
        let scale = (c.head_dim as f64).powf(-0.5);
        let attended = if let Some(out) = flash_attention(&queries, &keys, &values, scale, false)? {
            out
        } else {
            let chunk =
                attention_query_chunk(c.num_attention_heads, len, ATTENTION_CHUNK_SCRATCH_BUDGET)
                    .min(256);
            let mut outputs = Vec::new();
            for start in (0..len).step_by(chunk) {
                outputs.push(scaled_dot_product_attention_gqa(
                    &queries.narrow(2, start, (len - start).min(chunk))?,
                    &keys,
                    &values,
                    None,
                    scale,
                    false,
                    c.num_attention_heads / c.num_key_value_heads,
                )?);
            }
            Tensor::cat(&outputs, 2)?
        };
        let xs = (xs
            + self.out.forward(
                &attended
                    .squeeze(0)?
                    .transpose(0, 1)?
                    .reshape((len, c.num_attention_heads * c.head_dim))?
                    .contiguous()?,
            )?)?;
        &xs + self.mlp.forward(&self.post_norm.forward(&xs)?)?
    }
}

pub(super) struct VisionModel {
    embedding: Linear,
    layers: Vec<Layer>,
    norm: RmsNorm,
    fc1: Linear,
    fc2: Linear,
    cfg: OmniConfig,
}

impl VisionModel {
    pub fn load(cfg: &OmniConfig, vb: VarBuilder<'_>) -> Result<Self> {
        let merged = cfg.hidden_size * cfg.spatial_merge_size.pow(2);
        Ok(Self {
            embedding: linear(
                cfg.num_channels * cfg.patch_size.pow(2),
                cfg.hidden_size,
                vb.pp("vision_embeddings.patch_embedding"),
            )?,
            layers: (0..cfg.num_hidden_layers)
                .map(|i| Layer::load(cfg, vb.pp(format!("encoder.layers.{i}"))))
                .collect::<Result<_>>()?,
            // torch.nn.RMSNorm upcasts half inputs and uses the F32 epsilon
            // when eps=None, including for BF16 checkpoint weights.
            norm: rms_norm(merged, f64::from(f32::EPSILON), vb.pp("vision_merger.norm"))?,
            fc1: linear_no_bias(
                merged,
                cfg.merger_hidden_size,
                vb.pp("vision_merger.linear_fc1"),
            )?,
            fc2: linear_no_bias(
                cfg.merger_hidden_size,
                cfg.out_hidden_size,
                vb.pp("vision_merger.linear_fc2"),
            )?,
            cfg: cfg.clone(),
        })
    }

    pub fn forward(&self, patches: &Tensor, grid_h: usize, grid_w: usize) -> Result<Tensor> {
        let c = &self.cfg;
        let angles = vision_angles(c, grid_h, grid_w);
        let (cos, sin) = cos_sin(
            angles,
            grid_h * grid_w,
            c.head_dim / 2,
            DType::F32,
            patches.device(),
        )?;
        let mut xs = self.embedding.forward(patches)?;
        for layer in &self.layers {
            xs = layer.forward(&xs, &cos, &sin)?;
        }
        let merged = c.hidden_size * c.spatial_merge_size.pow(2);
        let xs = self.norm.forward(&xs.reshape(((), merged))?)?;
        self.fc2.forward(&self.fc1.forward(&xs)?.gelu_erf()?)
    }
}

fn vision_angles(c: &OmniConfig, height: usize, width: usize) -> Vec<f32> {
    let half = c.head_dim / 2;
    let mut angles = Vec::with_capacity(height * width * half);
    let merge = c.spatial_merge_size;
    for hb in 0..height / merge {
        for wb in 0..width / merge {
            for hi in 0..merge {
                for wi in 0..merge {
                    let h = (hb * merge + hi) as f32;
                    let w = (wb * merge + wi) as f32;
                    if c.video_omni_chunked_mthw_rope {
                        let axis = (half - c.rope_m_dim) / 3;
                        angles.extend(std::iter::repeat_n(0.0, c.rope_m_dim + axis));
                        angles.extend(
                            (0..axis)
                                .map(|i| h * c.rope_theta.powf(-(i as f64) / axis as f64) as f32),
                        );
                        let remaining = half - c.rope_m_dim - axis * 2;
                        angles.extend(
                            (0..remaining).map(|i| {
                                w * c.rope_theta.powf(-(i as f64) / remaining as f64) as f32
                            }),
                        );
                    } else {
                        for pos in [h, w] {
                            angles.extend((0..half / 2).map(|i| {
                                pos * c.rope_theta.powf(-(i as f64) / (half / 2) as f64) as f32
                            }));
                        }
                    }
                }
            }
        }
    }
    angles
}

#[cfg(test)]
mod tests {
    use super::super::config::Config;
    use super::*;
    use candle_core::Device;

    #[test]
    fn bf16_merger_matches_pytorch_default_epsilon() -> Result<()> {
        let reference: serde_json::Value =
            serde_json::from_str(include_str!("fixtures/reference.json")).unwrap();
        let config: Config = serde_json::from_value(reference["config"].clone()).unwrap();
        let device = Device::Cpu;
        let vb = VarBuilder::from_slice_safetensors(
            include_bytes!("fixtures/tiny.safetensors"),
            DType::BF16,
            &device,
        )?;
        let vision = VisionModel::load(&config.omni_config, vb.pp("model.omni_model"))?;
        // Small activations distinguish the F32 default from BF16's much larger epsilon.
        let input = Tensor::full(half::bf16::from_f32(0.002), (1, 32), &device)?;
        let actual = vision
            .norm
            .forward(&input)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;
        let expected = reference["merger_bf16"].as_array().unwrap();
        assert_eq!(actual.len(), expected.len());
        for (actual, expected) in actual.iter().zip(expected) {
            assert!((actual - expected.as_f64().unwrap() as f32).abs() < 0.016);
        }
        Ok(())
    }
}
