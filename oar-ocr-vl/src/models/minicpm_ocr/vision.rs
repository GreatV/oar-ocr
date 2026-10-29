//! SigLIP vision tower and 2x2 merger MLP of MiniCPM-OCR (`4x` mode).
//!
//! The intermediate window-attention merger (`vit_merger`) only runs in the
//! `16x` mode and is not loaded.

use super::config::MiniCpmOcrVisionConfig;
use super::infer;
use super::processing::{MERGE_SIZE, MiniCpmOcrSlice};
use crate::Error;
use crate::runtime::attention::{
    ATTENTION_CHUNK_SCRATCH_BUDGET, attention_query_chunk, chunked_vision_attention,
    flash_attention,
};
use candle_core::{D, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder, embedding, layer_norm, linear};
use std::collections::BTreeMap;

struct Attention {
    query: Linear,
    key: Linear,
    value: Linear,
    output: Linear,
    heads: usize,
    head_dim: usize,
}

impl Attention {
    fn load(config: &MiniCpmOcrVisionConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let projection = |name: &str| linear(config.hidden_size, config.hidden_size, vb.pp(name));
        Ok(Self {
            query: projection("q_proj")?,
            key: projection("k_proj")?,
            value: projection("v_proj")?,
            output: projection("out_proj")?,
            heads: config.num_attention_heads,
            head_dim: config.hidden_size / config.num_attention_heads,
        })
    }

    /// Full bidirectional attention within each batch item `(B, L, C)`.
    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        let (batch, length, _) = input.dims3().map_err(infer)?;
        let to_heads = |layer: &Linear| -> Result<Tensor, Error> {
            layer
                .forward(input)
                .and_then(|x| x.reshape((batch, length, self.heads, self.head_dim)))
                .and_then(|x| x.transpose(1, 2))
                .and_then(|x| x.contiguous())
                .map_err(infer)
        };
        let (query, key, value) = (
            to_heads(&self.query)?,
            to_heads(&self.key)?,
            to_heads(&self.value)?,
        );
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let context = match flash_attention(&query, &key, &value, scale, false).map_err(infer)? {
            Some(context) => context,
            None => {
                let chunk = attention_query_chunk(
                    batch * self.heads,
                    length,
                    ATTENTION_CHUNK_SCRATCH_BUDGET,
                )
                .min(256);
                chunked_vision_attention(&query, &key, &value, scale, chunk).map_err(infer)?
            }
        };
        let context = context
            .transpose(1, 2)
            .and_then(|x| x.reshape((batch, length, self.heads * self.head_dim)))
            .map_err(infer)?;
        self.output.forward(&context).map_err(infer)
    }
}

struct EncoderLayer {
    layer_norm1: LayerNorm,
    attention: Attention,
    layer_norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
}

impl EncoderLayer {
    fn load(config: &MiniCpmOcrVisionConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let norm = |name: &str| layer_norm(config.hidden_size, config.layer_norm_eps, vb.pp(name));
        Ok(Self {
            layer_norm1: norm("layer_norm1")?,
            attention: Attention::load(config, vb.pp("self_attn"))?,
            layer_norm2: norm("layer_norm2")?,
            fc1: linear(
                config.hidden_size,
                config.intermediate_size,
                vb.pp("mlp.fc1"),
            )?,
            fc2: linear(
                config.intermediate_size,
                config.hidden_size,
                vb.pp("mlp.fc2"),
            )?,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, Error> {
        let normed = self.layer_norm1.forward(input).map_err(infer)?;
        let hidden = (input + self.attention.forward(&normed)?).map_err(infer)?;
        let update = self
            .layer_norm2
            .forward(&hidden)
            .and_then(|x| self.fc1.forward(&x))
            // gelu_pytorch_tanh
            .and_then(|x| x.gelu())
            .and_then(|x| self.fc2.forward(&x))
            .map_err(infer)?;
        (hidden + update).map_err(infer)
    }
}

/// Patch-merging MLP: LayerNorm, Linear, exact GELU, Linear.
struct Merger {
    pre_norm: LayerNorm,
    linear_1: Linear,
    linear_2: Linear,
}

impl Merger {
    fn load(hidden: usize, text_hidden: usize, vb: VarBuilder) -> candle_core::Result<Self> {
        let merged = hidden * MERGE_SIZE * MERGE_SIZE;
        Ok(Self {
            pre_norm: layer_norm(merged, 1e-6, vb.pp("pre_norm"))?,
            linear_1: linear(merged, merged, vb.pp("linear_1"))?,
            linear_2: linear(merged, text_hidden, vb.pp("linear_2"))?,
        })
    }

    /// Merge each 2x2 patch block of a `(L, C)` raster grid into one token.
    fn forward(&self, hidden: &Tensor, grid_hw: (usize, usize)) -> Result<Tensor, Error> {
        let (height, width) = grid_hw;
        let channels = hidden.dim(D::Minus1).map_err(infer)?;
        let (merged_h, merged_w) = (height / MERGE_SIZE, width / MERGE_SIZE);
        hidden
            .reshape((merged_h, MERGE_SIZE, merged_w, MERGE_SIZE, channels))
            .and_then(|x| x.permute((0, 2, 1, 3, 4)))
            .and_then(|x| x.reshape((merged_h * merged_w, MERGE_SIZE * MERGE_SIZE * channels)))
            .and_then(|x| self.pre_norm.forward(&x))
            .and_then(|x| self.linear_1.forward(&x))
            .and_then(|x| x.gelu_erf())
            .and_then(|x| self.linear_2.forward(&x))
            .map_err(infer)
    }
}

pub(super) struct VisionTower {
    patch_embedding: Linear,
    position_embedding: Embedding,
    layers: Vec<EncoderLayer>,
    post_layernorm: LayerNorm,
    merger: Merger,
    patches_per_side: usize,
}

impl VisionTower {
    /// `vb` is the checkpoint root `model` prefix.
    pub fn load(
        config: &MiniCpmOcrVisionConfig,
        text_hidden: usize,
        vb: VarBuilder,
    ) -> Result<Self, Error> {
        let tower = vb.pp("vision_tower");
        let hidden = config.hidden_size;
        let patch_dim = config.num_channels * config.patch_size * config.patch_size;
        let embeddings = tower.pp("embeddings");
        // The stride-p, kernel-p convolution is one matmul over flattened
        // `[channel][row][column]` patches.
        let weight = embeddings
            .get(
                (
                    hidden,
                    config.num_channels,
                    config.patch_size,
                    config.patch_size,
                ),
                "patch_embedding.weight",
            )
            .and_then(|w| w.reshape((hidden, patch_dim)))
            .map_err(infer)?;
        let bias = embeddings
            .get(hidden, "patch_embedding.bias")
            .map_err(infer)?;
        let patches_per_side = config.num_patches_per_side();
        let position_embedding = embedding(
            patches_per_side * patches_per_side,
            hidden,
            embeddings.pp("position_embedding"),
        )
        .map_err(infer)?;
        let layers = (0..config.num_hidden_layers)
            .map(|index| EncoderLayer::load(config, tower.pp(format!("encoder.layers.{index}"))))
            .collect::<candle_core::Result<Vec<_>>>()
            .map_err(infer)?;
        let post_layernorm =
            layer_norm(hidden, config.layer_norm_eps, tower.pp("post_layernorm")).map_err(infer)?;
        let merger = Merger::load(hidden, text_hidden, vb.pp("merger.mlp.0")).map_err(infer)?;
        Ok(Self {
            patch_embedding: Linear::new(weight, Some(bias)),
            position_embedding,
            layers,
            post_layernorm,
            merger,
            patches_per_side,
        })
    }

    /// Encode every slice and return the merged visual tokens of all
    /// slices concatenated in input order, `(tokens, text_hidden)`.
    pub fn forward(&self, slices: &[MiniCpmOcrSlice]) -> Result<Tensor, Error> {
        // Slices of one shape share a batch; attention never crosses slices.
        let mut groups: BTreeMap<(usize, usize), Vec<usize>> = BTreeMap::new();
        for (index, slice) in slices.iter().enumerate() {
            groups.entry(slice.grid_hw).or_default().push(index);
        }
        let mut merged: Vec<Option<Tensor>> = vec![None; slices.len()];
        for (grid_hw, indices) in groups {
            let pixels = indices
                .iter()
                .map(|&index| &slices[index].pixel_values)
                .collect::<Vec<_>>();
            let pixels = Tensor::stack(&pixels, 0).map_err(infer)?;
            let positions = Tensor::new(
                nearest_position_ids(grid_hw, self.patches_per_side).as_slice(),
                pixels.device(),
            )
            .map_err(infer)?;
            let mut hidden = self
                .patch_embedding
                .forward(&pixels)
                .and_then(|x| {
                    x.broadcast_add(&self.position_embedding.forward(&positions)?.unsqueeze(0)?)
                })
                .map_err(infer)?;
            for layer in &self.layers {
                hidden = layer.forward(&hidden)?;
            }
            let hidden = self.post_layernorm.forward(&hidden).map_err(infer)?;
            for (batch_index, &index) in indices.iter().enumerate() {
                let item = hidden.get(batch_index).map_err(infer)?;
                merged[index] = Some(self.merger.forward(&item, grid_hw)?);
            }
        }
        let merged = merged.into_iter().flatten().collect::<Vec<_>>();
        Tensor::cat(&merged, 0).map_err(infer)
    }
}

/// Nearest-bucket indices into the learned `side x side` position table,
/// mirroring `get_vision_nearest_position_ids`: `torch.arange` boundaries
/// in float32 and `bucketize(right=True)` over `i / height` and `j / width`.
fn nearest_position_ids(grid_hw: (usize, usize), side: usize) -> Vec<u32> {
    let step = 1.0f64 / side as f64;
    let count = ((1.0 - step) / step).ceil() as usize;
    let boundaries: Vec<f32> = (0..count)
        .map(|i| (step + i as f64 * step) as f32)
        .collect();
    let bucket = |index: usize, length: usize| {
        let coordinate = index as f32 / length as f32;
        boundaries.partition_point(|&boundary| boundary <= coordinate)
    };
    let (height, width) = grid_hw;
    let rows: Vec<usize> = (0..height).map(|i| bucket(i, height)).collect();
    let columns: Vec<usize> = (0..width).map(|j| bucket(j, width)).collect();
    rows.iter()
        .flat_map(|&row| {
            columns
                .iter()
                .map(move |&column| (row * side + column) as u32)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn position_ids_bucket_onto_the_learned_grid() {
        // A full 70x70 grid maps one-to-one; a 2x3 grid samples its rows and
        // columns at fractions 0, 1/2 and 0, 1/3, 2/3 of the table.
        let full = nearest_position_ids((70, 70), 70);
        assert_eq!(full, (0..4900).collect::<Vec<u32>>());
        assert_eq!(
            nearest_position_ids((2, 3), 70),
            vec![0, 23, 46, 35 * 70, 35 * 70 + 23, 35 * 70 + 46]
        );
        // Index sums from the torch reference for odd grids, where float32
        // rounding decides the boundary buckets.
        let sum = |grid| {
            nearest_position_ids(grid, 70)
                .iter()
                .map(|&id| id as u64)
                .sum::<u64>()
        };
        assert_eq!(sum((37, 53)), 4_674_198);
        assert_eq!(sum((46, 26)), 2_866_158);
    }
}
