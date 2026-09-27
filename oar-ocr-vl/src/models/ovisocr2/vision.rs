//! OvisOCR2 vision tower: a thin adapter over the shared Qwen3-VL vision
//! backbone (`backbones::qwen3_vl::vision`).
//!
//! The OvisOCR2 (`qwen3_5`) checkpoint's vision tower is architecturally a
//! Qwen3-VL tower with an empty `deepstack_visual_indexes` list — same patch
//! embed, learned position grid bilinearly interpolated the same way, same
//! LayerNorm blocks, same pre-shuffle merger — so the tower itself lives in
//! the backbone and this module only keeps the OvisOCR2-specific input
//! validation and config conversion.

use super::config::OvisOcr2VisionConfig;
use crate::backbones::qwen3_vl::Qwen3VlVisionModel;
use crate::error::Error;
use crate::utils::candle_to_ocr_inference;
use candle_core::Tensor;
use candle_nn::VarBuilder;

/// OvisOCR2 vision tower: [`Qwen3VlVisionModel`] plus the checkpoint's
/// patch-count validation.
pub struct OvisOcr2VisionModel {
    inner: Qwen3VlVisionModel,
}

impl OvisOcr2VisionModel {
    pub fn load(cfg: &OvisOcr2VisionConfig, vb: VarBuilder) -> Result<Self, Error> {
        cfg.validate()?;
        let inner = Qwen3VlVisionModel::load(&cfg.to_qwen3_vl(), vb)?;
        Ok(Self { inner })
    }

    pub fn forward(
        &self,
        pixel_values: &Tensor,
        grid_thw: (usize, usize, usize),
    ) -> Result<Tensor, Error> {
        validate_patch_count(pixel_values, grid_thw)?;
        // The config rejects DeepStack taps, so the backbone always returns
        // an empty feature list here.
        let (merged, deepstack) = self.inner.forward(pixel_values, &[grid_thw])?;
        debug_assert!(
            deepstack.is_empty(),
            "OvisOCR2 vision config rejects DeepStack taps"
        );
        Ok(merged)
    }
}

/// OvisOCR2-specific input check: the backbone narrows `pixel_values` to the
/// grid's patch count, so a mismatch must be rejected up front (the grid
/// itself is validated by the backbone's merge-group coordinate builder).
fn validate_patch_count(
    pixel_values: &Tensor,
    grid_thw: (usize, usize, usize),
) -> Result<(), Error> {
    let (grid_t, grid_h, grid_w) = grid_thw;
    let num_patches = grid_t
        .checked_mul(grid_h)
        .and_then(|value| value.checked_mul(grid_w))
        .ok_or_else(|| Error::InvalidInput {
            message: "OvisOCR2 vision patch count overflow".to_string(),
        })?;
    if pixel_values
        .dim(0)
        .map_err(|e| candle_to_ocr_inference("OvisOCR2", "read vision pixel_values length", e))?
        != num_patches
    {
        return Err(Error::InvalidInput {
            message: format!(
                "OvisOCR2 pixel_values patch count ({}) does not match grid {grid_thw:?} ({num_patches})",
                pixel_values.dims().first().copied().unwrap_or(0)
            ),
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};
    use candle_nn::Activation;
    use std::collections::HashMap;

    fn tiny_config() -> OvisOcr2VisionConfig {
        OvisOcr2VisionConfig {
            model_type: "qwen3_5".to_string(),
            depth: 2,
            hidden_size: 4,
            intermediate_size: 8,
            num_heads: 1,
            in_channels: 3,
            patch_size: 1,
            spatial_merge_size: 1,
            temporal_patch_size: 1,
            out_hidden_size: 4,
            num_position_embeddings: 4,
            hidden_act: Activation::GeluPytorchTanh,
            initializer_range: 0.02,
            deepstack_visual_indexes: Vec::new(),
        }
    }

    #[test]
    fn converts_to_backbone_config_field_by_field() {
        let cfg = tiny_config();
        let backbone = cfg.to_qwen3_vl();
        assert_eq!(backbone.model_type, "qwen3_vl");
        assert_eq!(backbone.depth, cfg.depth);
        assert_eq!(backbone.hidden_size, cfg.hidden_size);
        assert_eq!(backbone.intermediate_size, cfg.intermediate_size);
        assert_eq!(backbone.num_heads, cfg.num_heads);
        assert_eq!(backbone.in_channels, cfg.in_channels);
        assert_eq!(backbone.patch_size, cfg.patch_size);
        assert_eq!(backbone.spatial_merge_size, cfg.spatial_merge_size);
        assert_eq!(backbone.temporal_patch_size, cfg.temporal_patch_size);
        assert_eq!(backbone.out_hidden_size, cfg.out_hidden_size);
        assert_eq!(
            backbone.num_position_embeddings,
            cfg.num_position_embeddings
        );
        assert_eq!(backbone.hidden_act, cfg.hidden_act);
        assert_eq!(
            backbone.deepstack_visual_indexes,
            cfg.deepstack_visual_indexes
        );
        backbone.validate().unwrap();
    }

    /// Mirrors the tower construction in `model.rs`
    /// (`OvisOcr2VisionModel::load(&cfg.vision_config, vb.pp("model").pp("visual"))`),
    /// so the test exercises the production load path including the config
    /// conversion.
    #[test]
    fn loads_and_forwards_through_the_model_load_path() -> Result<(), Error> {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let prefix = "model.visual";
        let mut tensors = HashMap::new();
        tensors.insert(
            format!("{prefix}.patch_embed.proj.weight"),
            Tensor::zeros((4, 3, 1, 1, 1), DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.patch_embed.proj.bias"),
            Tensor::zeros(4, DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.pos_embed.weight"),
            Tensor::zeros((4, 4), DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.merger.norm.weight"),
            Tensor::ones(4, DType::F32, &device).unwrap(),
        );
        tensors.insert(
            format!("{prefix}.merger.norm.bias"),
            Tensor::zeros(4, DType::F32, &device).unwrap(),
        );
        for name in ["linear_fc1", "linear_fc2"] {
            tensors.insert(
                format!("{prefix}.merger.{name}.weight"),
                Tensor::zeros((4, 4), DType::F32, &device).unwrap(),
            );
            tensors.insert(
                format!("{prefix}.merger.{name}.bias"),
                Tensor::zeros(4, DType::F32, &device).unwrap(),
            );
        }
        for layer in 0..cfg.depth {
            let block = format!("{prefix}.blocks.{layer}");
            tensors.insert(
                format!("{block}.norm1.weight"),
                Tensor::ones(4, DType::F32, &device).unwrap(),
            );
            tensors.insert(
                format!("{block}.norm1.bias"),
                Tensor::zeros(4, DType::F32, &device).unwrap(),
            );
            tensors.insert(
                format!("{block}.norm2.weight"),
                Tensor::ones(4, DType::F32, &device).unwrap(),
            );
            tensors.insert(
                format!("{block}.norm2.bias"),
                Tensor::zeros(4, DType::F32, &device).unwrap(),
            );
            tensors.insert(
                format!("{block}.attn.qkv.weight"),
                Tensor::zeros((12, 4), DType::F32, &device).unwrap(),
            );
            tensors.insert(
                format!("{block}.attn.qkv.bias"),
                Tensor::zeros(12, DType::F32, &device).unwrap(),
            );
            tensors.insert(
                format!("{block}.attn.proj.weight"),
                Tensor::zeros((4, 4), DType::F32, &device).unwrap(),
            );
            tensors.insert(
                format!("{block}.attn.proj.bias"),
                Tensor::zeros(4, DType::F32, &device).unwrap(),
            );
            tensors.insert(
                format!("{block}.mlp.linear_fc1.weight"),
                Tensor::zeros((8, 4), DType::F32, &device).unwrap(),
            );
            tensors.insert(
                format!("{block}.mlp.linear_fc1.bias"),
                Tensor::zeros(8, DType::F32, &device).unwrap(),
            );
            tensors.insert(
                format!("{block}.mlp.linear_fc2.weight"),
                Tensor::zeros((4, 8), DType::F32, &device).unwrap(),
            );
            tensors.insert(
                format!("{block}.mlp.linear_fc2.bias"),
                Tensor::zeros(4, DType::F32, &device).unwrap(),
            );
        }
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = OvisOcr2VisionModel::load(&cfg, vb.pp("model").pp("visual"))?;
        let output = model.forward(&Tensor::zeros((4, 3), DType::F32, &device)?, (1, 2, 2))?;
        assert_eq!(output.dims(), &[4, 4]);
        Ok(())
    }

    #[test]
    fn rejects_patch_count_mismatch() {
        let device = Device::Cpu;
        let error = validate_patch_count(
            &Tensor::zeros((3, 3), DType::F32, &device).unwrap(),
            (1, 2, 2),
        )
        .unwrap_err();
        assert!(error.to_string().contains("does not match grid"));
        validate_patch_count(
            &Tensor::zeros((4, 3), DType::F32, &device).unwrap(),
            (1, 2, 2),
        )
        .unwrap();
    }
}
