//! Qwen-VL spatial patch merger shared by the Qwen2-VL and Qwen3-VL towers,
//! MonkeyOCRv2 and the MinerU-Diffusion abstractor.
//!
//! Groups of `merge_size²` consecutive patches (the merge-grouped order the
//! patchifiers emit) are concatenated and projected by
//! `Linear -> GELU (erf) -> Linear`, with an eps-1e-6 LayerNorm applied
//! either per patch before the merge (`pre-shuffle`) or over the merged rows
//! (`post-shuffle`, Qwen3-VL DeepStack taps).

use crate::error::Error;
use crate::runtime::errors::candle_to_ocr_inference;
use candle_core::Tensor;
use candle_nn::{LayerNorm, LayerNormConfig, Linear, Module, VarBuilder, layer_norm, linear};

/// Checkpoint key names of the merger's norm and two MLP layers.
#[derive(Debug, Clone, Copy)]
pub(crate) struct MergerKeys {
    pub norm: &'static str,
    pub fc1: &'static str,
    pub fc2: &'static str,
}

impl MergerKeys {
    /// Qwen2-VL `PatchMerger`: `ln_q`, `mlp.0`, `mlp.2`.
    pub(crate) const QWEN2_VL: Self = Self {
        norm: "ln_q",
        fc1: "mlp.0",
        fc2: "mlp.2",
    };
    /// Qwen3-VL `VisionPatchMerger`: `norm`, `linear_fc1`, `linear_fc2`.
    pub(crate) const QWEN3_VL: Self = Self {
        norm: "norm",
        fc1: "linear_fc1",
        fc2: "linear_fc2",
    };
}

#[derive(Debug, Clone)]
pub(crate) struct SpatialPatchMerger {
    norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    merge_group: usize,
    merged_dim: usize,
    postshuffle_norm: bool,
    model_name: &'static str,
}

impl SpatialPatchMerger {
    /// `vb` points at the merger directory itself; `hidden_size` is the
    /// per-patch width and `out_size` the projected (LLM) width.
    pub(crate) fn load(
        vb: VarBuilder,
        keys: MergerKeys,
        hidden_size: usize,
        merge_size: usize,
        out_size: usize,
        postshuffle_norm: bool,
        model_name: &'static str,
    ) -> Result<Self, Error> {
        let merge_group = merge_size * merge_size;
        let merged_dim = hidden_size * merge_group;
        let norm_size = if postshuffle_norm {
            merged_dim
        } else {
            hidden_size
        };
        let norm_cfg = LayerNormConfig {
            eps: 1e-6,
            ..Default::default()
        };
        let norm = layer_norm(norm_size, norm_cfg, vb.pp(keys.norm))
            .map_err(|e| candle_to_ocr_inference(model_name, "load merger norm", e))?;
        let fc1 = linear(merged_dim, merged_dim, vb.pp(keys.fc1))
            .map_err(|e| candle_to_ocr_inference(model_name, "load merger fc1", e))?;
        let fc2 = linear(merged_dim, out_size, vb.pp(keys.fc2))
            .map_err(|e| candle_to_ocr_inference(model_name, "load merger fc2", e))?;
        Ok(Self {
            norm,
            fc1,
            fc2,
            merge_group,
            merged_dim,
            postshuffle_norm,
            model_name,
        })
    }

    /// `(num_patches, hidden_size)` -> `(num_patches / merge_size², out_size)`.
    pub(crate) fn forward(&self, hidden_states: &Tensor) -> Result<Tensor, Error> {
        let model_name = self.model_name;
        let num_patches = hidden_states
            .dim(0)
            .map_err(|e| candle_to_ocr_inference(model_name, "merger patch count", e))?;
        if !num_patches.is_multiple_of(self.merge_group) {
            return Err(Error::InvalidInput {
                message: format!(
                    "{model_name}: merger expects a patch count divisible by {}, got {num_patches}",
                    self.merge_group
                ),
            });
        }
        let merged_shape = (num_patches / self.merge_group, self.merged_dim);
        let merged = if self.postshuffle_norm {
            hidden_states
                .reshape(merged_shape)
                .and_then(|merged| self.norm.forward(&merged))
        } else {
            self.norm
                .forward(hidden_states)
                .and_then(|normed| normed.reshape(merged_shape))
        }
        .map_err(|e| candle_to_ocr_inference(model_name, "merger norm", e))?;
        let hidden = self
            .fc1
            .forward(&merged)
            .and_then(|hidden| hidden.gelu_erf())
            .map_err(|e| candle_to_ocr_inference(model_name, "merger fc1", e))?;
        self.fc2
            .forward(&hidden)
            .map_err(|e| candle_to_ocr_inference(model_name, "merger fc2", e))
    }
}
