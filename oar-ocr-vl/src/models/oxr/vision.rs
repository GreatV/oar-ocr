//! Patch encoder and probability-weighted visual codebook for OXR.

use super::config::{Config, VisionConfig};
use super::infer;
use super::processing::{MERGE_SIZE, PATCH_SIZE};
use crate::Error;
use crate::runtime::attention::{
    ATTENTION_CHUNK_SCRATCH_BUDGET, attention_query_chunk, chunked_vision_attention,
    flash_attention,
};
use crate::runtime::tensor::apply_rotary_f32;
use candle_core::{DType, Device, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder, layer_norm, linear, linear_no_bias};
use rayon::prelude::*;

const GROUP_AREA: usize = MERGE_SIZE * MERGE_SIZE;
const RESERVED_CODES: usize = 4;

/// Coordinates of a patch stream whose consecutive groups cover 2x2 patches.
/// Both learned positions and RoPE use this same ordering.
fn patch_coordinates(height: usize, width: usize) -> impl Iterator<Item = (usize, usize)> {
    let groups_per_row = width / MERGE_SIZE;
    (0..height * width).map(move |index| {
        let group = index / GROUP_AREA;
        let local = index % GROUP_AREA;
        (
            (group / groups_per_row) * MERGE_SIZE + local / MERGE_SIZE,
            (group % groups_per_row) * MERGE_SIZE + local % MERGE_SIZE,
        )
    })
}

struct PatchAttention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    heads: usize,
    head_dim: usize,
}

impl PatchAttention {
    fn load(config: &VisionConfig, weights: VarBuilder) -> candle_core::Result<Self> {
        let projection =
            |name: &str| linear(config.hidden_size, config.hidden_size, weights.pp(name));
        Ok(Self {
            query: projection("q_proj")?,
            key: projection("k_proj")?,
            value: projection("v_proj")?,
            output: projection("out_proj")?,
            heads: config.num_attention_heads,
            head_dim: config.hidden_size / config.num_attention_heads,
        })
    }

    fn forward(&self, input: &Tensor, rotary: &(Tensor, Tensor)) -> Result<Tensor, Error> {
        let length = input.dim(0).map_err(infer)?;
        let project = |layer: &Linear| -> Result<Tensor, Error> {
            layer
                .forward(input)
                .and_then(|x| x.reshape((length, self.heads, self.head_dim)))
                .map_err(infer)
        };
        let to_heads = |x: Tensor| -> Result<Tensor, Error> {
            x.transpose(0, 1)
                .and_then(|x| x.unsqueeze(0))
                .and_then(|x| x.contiguous())
                .map_err(infer)
        };
        let (query, key) = apply_rotary_f32(
            &project(&self.query)?,
            &project(&self.key)?,
            &rotary.0,
            &rotary.1,
            "OXR",
        )?;
        let query = to_heads(query)?;
        let key = to_heads(key)?;
        let value = to_heads(project(&self.value)?)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let context = match flash_attention(&query, &key, &value, scale, false).map_err(infer)? {
            Some(context) => context,
            None => {
                let chunk =
                    attention_query_chunk(self.heads, length, ATTENTION_CHUNK_SCRATCH_BUDGET)
                        .min(256);
                chunked_vision_attention(&query, &key, &value, scale, chunk).map_err(infer)?
            }
        };
        let context = context
            .transpose(1, 2)
            .and_then(|x| x.reshape((length, self.heads * self.head_dim)))
            .map_err(infer)?;
        self.output.forward(&context).map_err(infer)
    }
}

struct VisionLayer {
    attention_norm: LayerNorm,
    attention: PatchAttention,
    mlp_norm: LayerNorm,
    expand: Linear,
    contract: Linear,
}

impl VisionLayer {
    fn load(config: &VisionConfig, weights: VarBuilder) -> candle_core::Result<Self> {
        Ok(Self {
            attention_norm: layer_norm(
                config.hidden_size,
                config.layer_norm_eps,
                weights.pp("layer_norm1"),
            )?,
            attention: PatchAttention::load(config, weights.pp("self_attn"))?,
            mlp_norm: layer_norm(
                config.hidden_size,
                config.layer_norm_eps,
                weights.pp("layer_norm2"),
            )?,
            expand: linear(
                config.hidden_size,
                config.intermediate_size,
                weights.pp("mlp.fc1"),
            )?,
            contract: linear(
                config.intermediate_size,
                config.hidden_size,
                weights.pp("mlp.fc2"),
            )?,
        })
    }

    fn forward(&self, input: &Tensor, rotary: &(Tensor, Tensor)) -> Result<Tensor, Error> {
        let query = self.attention_norm.forward(input).map_err(infer)?;
        let residual = (input + self.attention.forward(&query, rotary)?).map_err(infer)?;
        let update = self
            .mlp_norm
            .forward(&residual)
            .and_then(|x| self.expand.forward(&x))
            .and_then(|x| x.gelu())
            .and_then(|x| self.contract.forward(&x))
            .map_err(infer)?;
        (residual + update).map_err(infer)
    }
}

/// Softmax probabilities select an expectation over the codebook rows. The
/// four reserved codes are excluded from that distribution, so no zero-padding
/// tensor is needed. Image delimiters use the first two reserved rows directly.
struct VisualCodebook {
    logits: Linear,
    normalization: LayerNorm,
    entries: Tensor,
    image_boundaries: Tensor,
}

impl VisualCodebook {
    fn load(config: &Config, weights: VarBuilder) -> candle_core::Result<Self> {
        let codes = config.visual_vocab_size - RESERVED_CODES;
        let table = weights.get(
            (
                config.visual_vocab_size,
                config.llm_config.decoder.hidden_size,
            ),
            "vte.weight",
        )?;
        Ok(Self {
            logits: linear_no_bias(
                config.vit_config.hidden_size * GROUP_AREA,
                codes,
                weights.pp("visual_tokenizer.head.0"),
            )?,
            normalization: layer_norm(codes, 1e-5, weights.pp("visual_tokenizer.head.1"))?,
            entries: table.narrow(0, 0, codes)?,
            image_boundaries: table.narrow(0, codes, 2)?,
        })
    }

    fn embed(&self, patch_features: Tensor) -> Result<ImageEmbedding, Error> {
        let groups = patch_features.dim(0).map_err(infer)? / GROUP_AREA;
        let grouped = patch_features.reshape((groups, ())).map_err(infer)?;
        let mut chunks = Vec::new();
        // Keep the vocabulary-wide scratch bounded independently of image size.
        for start in (0..groups).step_by(128) {
            let rows = grouped
                .narrow(0, start, (groups - start).min(128))
                .map_err(infer)?;
            let scores = self
                .logits
                .forward(&rows)
                .and_then(|x| self.normalization.forward(&x))
                .map_err(infer)?;
            let probabilities =
                candle_nn::ops::softmax_last_dim(&scores.to_dtype(DType::F32).map_err(infer)?)
                    .and_then(|x| x.to_dtype(scores.dtype()))
                    .map_err(infer)?;
            chunks.push(probabilities.matmul(&self.entries).map_err(infer)?);
        }
        Ok(ImageEmbedding {
            tokens: Tensor::cat(&chunks, 0).map_err(infer)?,
            boundaries: self.image_boundaries.clone(),
        })
    }
}

pub(super) struct ImageEmbedding {
    pub tokens: Tensor,
    pub boundaries: Tensor,
}

pub(super) struct ImageEncoder {
    patch_projection: Linear,
    learned_positions: Vec<f32>,
    learned_side: usize,
    head_dim: usize,
    layers: Vec<VisionLayer>,
    codebook: VisualCodebook,
}

impl ImageEncoder {
    pub fn load(config: &Config, weights: VarBuilder) -> Result<Self, Error> {
        let vision = &config.vit_config;
        let tower = weights.pp("visual_tokenizer.vit.vision_model");
        let embeddings = tower.pp("embeddings");
        let patch_weights = embeddings
            .get(
                (vision.hidden_size, 3, PATCH_SIZE, PATCH_SIZE),
                "patch_embedding.weight",
            )
            .and_then(|x| x.reshape((vision.hidden_size, 3 * PATCH_SIZE * PATCH_SIZE)))
            .map_err(infer)?;
        let patch_bias = embeddings
            .get(vision.hidden_size, "patch_embedding.bias")
            .map_err(infer)?;
        let learned_side = vision.image_size / PATCH_SIZE;
        let learned_positions = embeddings
            .get(
                (learned_side * learned_side, vision.hidden_size),
                "position_embedding.weight",
            )
            .and_then(|x| x.to_dtype(DType::F32))
            .and_then(|x| x.flatten_all())
            .and_then(|x| x.to_vec1::<f32>())
            .map_err(infer)?;
        let layers = (0..vision.num_hidden_layers)
            .map(|index| VisionLayer::load(vision, tower.pp(format!("encoder.layers.{index}"))))
            .collect::<candle_core::Result<Vec<_>>>()
            .map_err(infer)?;
        Ok(Self {
            patch_projection: Linear::new(patch_weights, Some(patch_bias)),
            learned_positions,
            learned_side,
            head_dim: vision.hidden_size / vision.num_attention_heads,
            layers,
            codebook: VisualCodebook::load(config, weights).map_err(infer)?,
        })
    }

    pub fn encode(&self, patches: &Tensor, grid: [usize; 3]) -> Result<ImageEmbedding, Error> {
        let [_, height, width] = grid;
        let positions = interpolate_positions(
            &self.learned_positions,
            self.learned_side,
            height,
            width,
            patches.device(),
            patches.dtype(),
        )?;
        let projected = self.patch_projection.forward(patches).map_err(infer)?;
        let mut state = (projected + positions).map_err(infer)?;
        let rotary = rotary_grid(height, width, self.head_dim, patches.device())?;
        // This architecture uses full attention throughout. Permuting tokens
        // together with their positions would leave the encoder unchanged.
        for layer in &self.layers {
            state = layer.forward(&state, &rotary)?;
        }
        // The visual vocabulary consumes the final residual state, without
        // the vision tower's unused output LayerNorm.
        self.codebook.embed(state)
    }
}

fn rotary_grid(
    height: usize,
    width: usize,
    head_dim: usize,
    device: &Device,
) -> Result<(Tensor, Tensor), Error> {
    let axis_dim = head_dim / 2;
    let inverse_frequencies: Vec<f32> = (0..axis_dim)
        .step_by(2)
        .map(|index| 1.0 / 10000f32.powf(index as f32 / axis_dim as f32))
        .collect();
    let mut angles = Vec::with_capacity(height * width * head_dim);
    for (row, column) in patch_coordinates(height, width) {
        let start = angles.len();
        for coordinate in [row, column] {
            angles.extend(
                inverse_frequencies
                    .iter()
                    .map(|frequency| coordinate as f32 * frequency),
            );
        }
        angles.extend_from_within(start..);
    }
    let angles = Tensor::from_vec(angles, (height * width, 1, head_dim), device).map_err(infer)?;
    Ok((angles.cos().map_err(infer)?, angles.sin().map_err(infer)?))
}

// Bicubic learned-position interpolation with half-pixel coordinates, clamped
// edges and coefficient -0.75. Pixel resampling uses the separate Pillow engine.
fn cubic_weights(t: f32) -> [f32; 4] {
    let a = -0.75;
    let inner = |x: f32| ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    let outer = |x: f32| ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    [outer(t + 1.0), inner(t), inner(1.0 - t), outer(2.0 - t)]
}

fn interpolate_positions(
    positions: &[f32],
    side: usize,
    height: usize,
    width: usize,
    device: &Device,
    dtype: DType,
) -> Result<Tensor, Error> {
    let dim = positions.len() / (side * side);
    let coordinates: Vec<_> = patch_coordinates(height, width).collect();
    let mut output = vec![0f32; height * width * dim];
    output
        .par_chunks_mut(dim)
        .zip(coordinates)
        .for_each(|(values, (row, column))| {
            let y = (side as f32 / height as f32) * (row as f32 + 0.5) - 0.5;
            let x = (side as f32 / width as f32) * (column as f32 + 0.5) - 0.5;
            let (iy, ix) = (y.floor() as isize, x.floor() as isize);
            let wy = cubic_weights(y - y.floor());
            let wx = cubic_weights(x - x.floor());
            // The sixteen source locations are shared by every channel. Keep
            // channels adjacent in the table and preserve each scalar sum's order.
            let offsets: [[usize; 4]; 4] = std::array::from_fn(|dy| {
                let yy = (iy + dy as isize - 1).clamp(0, side as isize - 1) as usize;
                std::array::from_fn(|dx| {
                    let xx = (ix + dx as isize - 1).clamp(0, side as isize - 1) as usize;
                    (yy * side + xx) * dim
                })
            });
            for (channel, out) in values.iter_mut().enumerate() {
                let mut value = 0f32;
                for (dy, &weight_y) in wy.iter().enumerate() {
                    let mut row = 0f32;
                    for (dx, &weight_x) in wx.iter().enumerate() {
                        row += positions[offsets[dy][dx] + channel] * weight_x;
                    }
                    value += row * weight_y;
                }
                *out = value;
            }
        });
    Tensor::from_vec(output, (height * width, dim), device)
        .and_then(|x| x.to_dtype(dtype))
        .map_err(infer)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bicubic_positions_use_half_pixel_coordinates_and_merge_order() {
        let table = vec![1f32, 2.0, 3.0, 4.0];
        let values = interpolate_positions(&table, 2, 4, 4, &Device::Cpu, DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(&values[..4], &[0.68359375, 1.015625, 1.3476563, 1.6796875]);
        let identity = interpolate_positions(&table, 2, 2, 2, &Device::Cpu, DType::F32)
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap();
        assert_eq!(identity, [1.0, 2.0, 3.0, 4.0]);

        // Original channel-major scalar loop, including edge clamping and
        // the exact horizontal-then-vertical accumulation order.
        fn original(channels: &[Vec<f32>], side: usize, height: usize, width: usize) -> Vec<f32> {
            let mut output = Vec::new();
            for (row, column) in patch_coordinates(height, width) {
                let y = (side as f32 / height as f32) * (row as f32 + 0.5) - 0.5;
                let x = (side as f32 / width as f32) * (column as f32 + 0.5) - 0.5;
                let (iy, ix) = (y.floor() as isize, x.floor() as isize);
                let wy = cubic_weights(y - y.floor());
                let wx = cubic_weights(x - x.floor());
                for channel in channels {
                    let mut value = 0f32;
                    for (dy, &weight_y) in wy.iter().enumerate() {
                        let yy = (iy + dy as isize - 1).clamp(0, side as isize - 1) as usize;
                        let mut row = 0f32;
                        for (dx, &weight_x) in wx.iter().enumerate() {
                            let xx = (ix + dx as isize - 1).clamp(0, side as isize - 1) as usize;
                            row += channel[yy * side + xx] * weight_x;
                        }
                        value += row * weight_y;
                    }
                    output.push(value);
                }
            }
            output
        }
        for side in [2, 3, 5] {
            for dim in [1, 3, 17] {
                let table: Vec<f32> = (0..side * side * dim)
                    .map(|i| ((i * 13 % 101) as f32 - 40.0) * 0.03125)
                    .collect();
                let channels: Vec<Vec<f32>> = (0..dim)
                    .map(|channel| {
                        table
                            .chunks_exact(dim)
                            .map(|pixel| pixel[channel])
                            .collect()
                    })
                    .collect();
                for (height, width) in [(2, 2), (4, 6), (6, 2), (2, 8), (8, 4)] {
                    let expected = original(&channels, side, height, width);
                    let actual = interpolate_positions(
                        &table,
                        side,
                        height,
                        width,
                        &Device::Cpu,
                        DType::F32,
                    )
                    .unwrap()
                    .flatten_all()
                    .unwrap()
                    .to_vec1::<f32>()
                    .unwrap();
                    assert_eq!(actual.len(), expected.len());
                    assert!(
                        actual
                            .iter()
                            .zip(expected)
                            .all(|(a, b)| a.to_bits() == b.to_bits())
                    );
                }
            }
        }
    }
}
