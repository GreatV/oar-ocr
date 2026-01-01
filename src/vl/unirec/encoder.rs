//! FocalSVTR visual encoder implementation.
//!
//! FocalSVTR combines Focal Modulation Networks with SVTR-style patch embedding
//! for vision encoding in OCR tasks.

use candle_core::{Module, Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Dropout, LayerNorm, Linear, VarBuilder};

use super::config::UniRecConfig;
use crate::core::OCRError;
use crate::vl::utils::candle_to_ocr_inference;

/// Drop path (stochastic depth) for regularization.
#[derive(Debug, Clone)]
struct DropPath {
    drop_prob: f64,
}

impl DropPath {
    fn new(drop_prob: f64) -> Self {
        Self { drop_prob }
    }
}

impl Module for DropPath {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        if self.drop_prob == 0.0 {
            return Ok(x.clone());
        }
        // During inference, we don't apply drop path
        Ok(x.clone())
    }
}

/// Convolution + BatchNorm + Activation layer.
#[derive(Debug, Clone)]
struct ConvBNLayer {
    conv: Conv2d,
    bn_weight: Tensor,
    bn_bias: Tensor,
    bn_running_mean: Tensor,
    bn_running_var: Tensor,
    eps: f64,
}

impl ConvBNLayer {
    fn load(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        stride: usize,
        padding: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride,
            padding,
            ..Default::default()
        };
        // Conv layer has no bias in the original model
        let conv = candle_nn::conv2d_no_bias(
            in_channels,
            out_channels,
            kernel_size,
            conv_cfg,
            vb.pp("conv"),
        )?;

        // Load BatchNorm parameters
        let norm_vb = vb.pp("norm");
        let bn_weight = norm_vb.get(out_channels, "weight")?;
        let bn_bias = norm_vb.get(out_channels, "bias")?;
        let bn_running_mean = norm_vb.get(out_channels, "running_mean")?;
        let bn_running_var = norm_vb.get(out_channels, "running_var")?;

        Ok(Self {
            conv,
            bn_weight,
            bn_bias,
            bn_running_mean,
            bn_running_var,
            eps: 1e-5,
        })
    }
}

impl Module for ConvBNLayer {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.conv.forward(x)?;
        // Apply batch norm in inference mode: y = (x - mean) / sqrt(var + eps) * weight + bias
        let (_, c, _, _) = x.dims4()?;

        // Reshape stats for broadcasting: (C,) -> (1, C, 1, 1)
        let mean = self.bn_running_mean.reshape((1, c, 1, 1))?;
        let var = self.bn_running_var.reshape((1, c, 1, 1))?;
        let weight = self.bn_weight.reshape((1, c, 1, 1))?;
        let bias = self.bn_bias.reshape((1, c, 1, 1))?;

        // Normalize
        let x = x.broadcast_sub(&mean)?;
        let std = (var + self.eps)?.sqrt()?;
        let x = x.broadcast_div(&std)?;
        let x = x.broadcast_mul(&weight)?;
        let x = x.broadcast_add(&bias)?;

        // GELU activation
        x.gelu()
    }
}

/// MLP block with two linear layers.
#[derive(Debug, Clone)]
struct Mlp {
    fc1: Linear,
    fc2: Linear,
    drop: Dropout,
}

impl Mlp {
    fn load(in_features: usize, hidden_features: usize, drop: f64, vb: VarBuilder) -> Result<Self> {
        let fc1 = candle_nn::linear(in_features, hidden_features, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(hidden_features, in_features, vb.pp("fc2"))?;
        let drop = Dropout::new(drop as f32);
        Ok(Self { fc1, fc2, drop })
    }
}

impl Module for Mlp {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let x = self.fc1.forward(x)?;
        let x = x.gelu()?;
        let x = self.drop.forward(&x, false)?;
        let x = self.fc2.forward(&x)?;
        self.drop.forward(&x, false)
    }
}

/// Focal Modulation block for multi-scale context aggregation.
#[derive(Debug)]
struct FocalModulation {
    #[allow(dead_code)]
    dim: usize,
    focal_level: usize,
    f_proj: Linear,
    h_conv: Conv2d,
    proj: Linear,
    focal_layers: Vec<Conv2d>,
}

impl FocalModulation {
    fn load(dim: usize, focal_window: usize, focal_level: usize, vb: VarBuilder) -> Result<Self> {
        // f projects to 2*dim + (focal_level + 1) for gating
        let f_proj = candle_nn::linear(dim, 2 * dim + focal_level + 1, vb.pp("f"))?;

        // h is a 1x1 conv for modulator
        let h_conv = candle_nn::conv2d(dim, dim, 1, Default::default(), vb.pp("h"))?;

        // Output projection
        let proj = candle_nn::linear(dim, dim, vb.pp("proj"))?;

        // Focal convolution layers at different scales
        let focal_factor = 2;
        let mut focal_layers = Vec::new();
        for k in 0..focal_level {
            let kernel_size = focal_factor * k + focal_window;
            // Padding should be (kernel_size - 1) / 2 to preserve spatial dimensions
            let padding = kernel_size / 2;

            // Depthwise conv with groups=dim
            let cfg = Conv2dConfig {
                padding,
                groups: dim,
                ..Default::default()
            };
            // Weight key is focal_layers.{k}.0.weight (the .0 is from Sequential)
            let conv = candle_nn::conv2d_no_bias(
                dim,
                dim,
                kernel_size,
                cfg,
                vb.pp(format!("focal_layers.{}.0", k)),
            )?;
            focal_layers.push(conv);
        }

        Ok(Self {
            dim,
            focal_level,
            f_proj,
            h_conv,
            proj,
            focal_layers,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (B, H, W, C)
        let (b, h, w, c) = x.dims4()?;

        // Pre linear projection: (B, H, W, C) -> (B, H, W, 2*C + focal_level + 1)
        let projected = self.f_proj.forward(x)?;
        // Permute to (B, 2*C + focal_level + 1, H, W) and make contiguous for CUDA
        let projected = projected.permute((0, 3, 1, 2))?.contiguous()?;

        // Split into q, ctx, gates
        let q = projected.narrow(1, 0, c)?.contiguous()?;
        let mut ctx = projected.narrow(1, c, c)?.contiguous()?;
        let gates = projected.narrow(1, 2 * c, self.focal_level + 1)?;

        // Context aggregation with focal convolutions
        let mut ctx_all = Tensor::zeros((b, c, h, w), x.dtype(), x.device())?;
        for l in 0..self.focal_level {
            ctx = self.focal_layers[l].forward(&ctx)?;
            ctx = ctx.gelu()?;
            let gate = gates.narrow(1, l, 1)?;
            let weighted = ctx.broadcast_mul(&gate)?;
            ctx_all = (&ctx_all + &weighted)?;
        }

        // Global context
        let ctx_global = ctx.mean_keepdim(2)?.mean_keepdim(3)?;
        let ctx_global = ctx_global.gelu()?;
        let gate_global = gates.narrow(1, self.focal_level, 1)?;
        let weighted_global = ctx_global.broadcast_mul(&gate_global)?;
        ctx_all = (&ctx_all + &weighted_global)?;

        // Focal modulation
        let modulator = self.h_conv.forward(&ctx_all)?;
        let x_out = (&q * &modulator)?;

        // Permute back to (B, H, W, C) and make contiguous for linear projection
        let x_out = x_out.permute((0, 2, 3, 1))?.contiguous()?;

        // Post linear projection
        self.proj.forward(&x_out)
    }
}

/// Focal Network Block.
#[derive(Debug)]
struct FocalNetBlock {
    norm1: LayerNorm,
    modulation: FocalModulation,
    drop_path: DropPath,
    norm2: LayerNorm,
    mlp: Mlp,
}

impl FocalNetBlock {
    fn load(
        dim: usize,
        mlp_ratio: f64,
        drop: f64,
        drop_path: f64,
        focal_level: usize,
        focal_window: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let norm1 = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm1"))?;
        let modulation =
            FocalModulation::load(dim, focal_window, focal_level, vb.pp("modulation"))?;
        let drop_path = DropPath::new(drop_path);
        let norm2 = candle_nn::layer_norm(dim, 1e-5, vb.pp("norm2"))?;
        let mlp_hidden_dim = (dim as f64 * mlp_ratio) as usize;
        let mlp = Mlp::load(dim, mlp_hidden_dim, drop, vb.pp("mlp"))?;

        Ok(Self {
            norm1,
            modulation,
            drop_path,
            norm2,
            mlp,
        })
    }

    fn forward(&self, x: &Tensor, h: usize, w: usize) -> Result<Tensor> {
        let (b, _l, c) = x.dims3()?;
        let shortcut = x.clone();

        // Focal modulation
        let x = self
            .norm1
            .forward(x)
            .map_err(|e| candle_core::Error::Msg(format!("norm1 failed: {}", e)))?;
        let x = x.reshape((b, h, w, c)).map_err(|e| {
            candle_core::Error::Msg(format!("reshape for modulation failed: {}", e))
        })?;
        let x = self
            .modulation
            .forward(&x)
            .map_err(|e| candle_core::Error::Msg(format!("modulation failed: {}", e)))?;
        let x = x.reshape((b, h * w, c)).map_err(|e| {
            candle_core::Error::Msg(format!("reshape after modulation failed: {}", e))
        })?;

        // Residual connection with drop path
        let x = self.drop_path.forward(&x)?;
        let x = (&shortcut + &x)?;

        // MLP
        let mlp_out = self
            .norm2
            .forward(&x)
            .map_err(|e| candle_core::Error::Msg(format!("norm2 failed: {}", e)))?;
        let mlp_out = self
            .mlp
            .forward(&mlp_out)
            .map_err(|e| candle_core::Error::Msg(format!("mlp failed: {}", e)))?;
        let mlp_out = self.drop_path.forward(&mlp_out)?;
        &x + &mlp_out
    }
}

/// Patch embedding layer.
#[derive(Debug)]
struct PatchEmbed {
    proj: Conv2d,
    norm: Option<LayerNorm>,
}

impl PatchEmbed {
    fn load(
        in_chans: usize,
        embed_dim: usize,
        patch_size: (usize, usize),
        use_conv_embed: bool,
        is_stem: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let (kernel_size, stride, padding) = if use_conv_embed {
            if is_stem { (7, 4, 2) } else { (3, 2, 1) }
        } else {
            (
                patch_size.0.max(patch_size.1),
                patch_size.0.max(patch_size.1),
                0,
            )
        };

        let cfg = Conv2dConfig {
            stride,
            padding,
            ..Default::default()
        };
        let proj = candle_nn::conv2d(in_chans, embed_dim, kernel_size, cfg, vb.pp("proj"))?;
        let norm = candle_nn::layer_norm(embed_dim, 1e-5, vb.pp("norm")).ok();

        Ok(Self { proj, norm })
    }

    fn forward(&self, x: &Tensor) -> Result<(Tensor, usize, usize)> {
        let x = self.proj.forward(x)?;
        let (_, _, h, w) = x.dims4()?;
        // Flatten to (B, H*W, C)
        let x = x.flatten(2, 3)?.permute((0, 2, 1))?;
        let x = if let Some(ref norm) = self.norm {
            norm.forward(&x)?
        } else {
            x
        };
        Ok((x, h, w))
    }
}

/// Configuration for BasicLayer.
struct BasicLayerConfig<'a> {
    dim: usize,
    out_dim: Option<usize>,
    depth: usize,
    mlp_ratio: f64,
    drop: f64,
    drop_path: &'a [f64],
    focal_level: usize,
    focal_window: usize,
    downsample_kernel: Option<(usize, usize)>,
}

/// Basic layer containing multiple FocalNetBlocks.
#[derive(Debug)]
struct BasicLayer {
    blocks: Vec<FocalNetBlock>,
    downsample: Option<PatchEmbed>,
}

impl BasicLayer {
    fn load(cfg: BasicLayerConfig<'_>, vb: VarBuilder) -> Result<Self> {
        let mut blocks = Vec::new();
        for i in 0..cfg.depth {
            let block = FocalNetBlock::load(
                cfg.dim,
                cfg.mlp_ratio,
                cfg.drop,
                cfg.drop_path.get(i).copied().unwrap_or(0.0),
                cfg.focal_level,
                cfg.focal_window,
                vb.pp(format!("blocks.{}", i)),
            )?;
            blocks.push(block);
        }

        let downsample = if let (Some(out_d), Some((kh, kw))) = (cfg.out_dim, cfg.downsample_kernel)
        {
            if kh > 0 && kw > 0 {
                Some(PatchEmbed::load(
                    cfg.dim,
                    out_d,
                    (kh, kw),
                    false,
                    false,
                    vb.pp("downsample"),
                )?)
            } else {
                None
            }
        } else {
            None
        };

        Ok(Self { blocks, downsample })
    }

    fn forward(&self, x: &Tensor, h: usize, w: usize) -> Result<(Tensor, usize, usize)> {
        let mut x = x.clone();
        let (mut out_h, mut out_w) = (h, w);

        for (i, block) in self.blocks.iter().enumerate() {
            x = block
                .forward(&x, out_h, out_w)
                .map_err(|e| candle_core::Error::Msg(format!("block.{} failed: {}", i, e)))?;
        }

        if let Some(ref downsample) = self.downsample {
            let (b, _, c) = x.dims3()?;
            // Reshape to (B, C, H, W) for downsampling
            let x_2d = x.permute((0, 2, 1))?.reshape((b, c, out_h, out_w))?;
            let (x_new, new_h, new_w) = downsample.forward(&x_2d)?;
            x = x_new;
            out_h = new_h;
            out_w = new_w;
        }

        Ok((x, out_h, out_w))
    }
}

/// FocalSVTR visual encoder.
#[derive(Debug)]
pub struct FocalSVTR {
    patch_embed_0: ConvBNLayer,
    patch_embed_1: ConvBNLayer,
    pos_drop: Dropout,
    layers: Vec<BasicLayer>,
    #[allow(dead_code)]
    num_features: usize,
    vision_fc: Linear,
}

impl FocalSVTR {
    /// Load FocalSVTR encoder from weights.
    pub fn load(cfg: &UniRecConfig, vb: VarBuilder) -> std::result::Result<Self, OCRError> {
        // Calculate embed dimensions for each stage
        let base_dim = cfg.encoder_embed_dim;
        let embed_dims: Vec<usize> = (0..cfg.encoder_depths.len())
            .map(|i| base_dim * (1 << i))
            .collect();
        let num_features = *embed_dims.last().unwrap_or(&base_dim);

        // Patch embedding (two ConvBNLayers)
        let patch_embed_0 = ConvBNLayer::load(
            3,
            embed_dims[0] / 2,
            3,
            2,
            1,
            vb.pp("vision_encoder.patch_embed.0"),
        )
        .map_err(|e| candle_to_ocr_inference("FocalSVTR", "load patch_embed.0", e))?;

        let patch_embed_1 = ConvBNLayer::load(
            embed_dims[0] / 2,
            embed_dims[0],
            3,
            2,
            1,
            vb.pp("vision_encoder.patch_embed.1"),
        )
        .map_err(|e| candle_to_ocr_inference("FocalSVTR", "load patch_embed.1", e))?;

        let pos_drop = Dropout::new(0.0);

        // Calculate drop path rates
        let total_depth: usize = cfg.encoder_depths.iter().sum();
        let drop_path_rate = 0.1;
        let dpr: Vec<f64> = (0..total_depth)
            .map(|i| drop_path_rate * (i as f64) / (total_depth as f64 - 1.0).max(1.0))
            .collect();

        // Build layers
        let mut layers = Vec::new();
        let num_layers = cfg.encoder_depths.len();
        let mut depth_offset = 0;

        for i in 0..num_layers {
            let depth = cfg.encoder_depths[i];
            let layer_dpr = &dpr[depth_offset..depth_offset + depth];
            depth_offset += depth;

            let out_dim = if i < num_layers - 1 {
                Some(embed_dims[i + 1])
            } else {
                None
            };

            let downsample_kernel = if i < num_layers - 1 {
                let (kh, kw) = cfg.sub_k[i];
                if kh > 0 && kw > 0 {
                    Some((kh, kw))
                } else {
                    None
                }
            } else {
                None
            };

            let focal_level = cfg.focal_levels.get(i).copied().unwrap_or(3);
            let focal_window = cfg.focal_windows.get(i).copied().unwrap_or(3);

            let layer = BasicLayer::load(
                BasicLayerConfig {
                    dim: embed_dims[i],
                    out_dim,
                    depth,
                    mlp_ratio: 4.0,
                    drop: 0.0,
                    drop_path: layer_dpr,
                    focal_level,
                    focal_window,
                    downsample_kernel,
                },
                vb.pp(format!("vision_encoder.layers.{}", i)),
            )
            .map_err(|e| candle_to_ocr_inference("FocalSVTR", format!("load layer.{}", i), e))?;

            layers.push(layer);
        }

        // Vision FC layer to project to d_model
        let vision_fc = candle_nn::linear(num_features, cfg.d_model, vb.pp("vision_fc"))
            .map_err(|e| candle_to_ocr_inference("FocalSVTR", "load vision_fc", e))?;

        Ok(Self {
            patch_embed_0,
            patch_embed_1,
            pos_drop,
            layers,
            num_features,
            vision_fc,
        })
    }

    /// Forward pass through the encoder.
    pub fn forward(&self, x: &Tensor) -> std::result::Result<Tensor, OCRError> {
        // Patch embedding
        let x = self
            .patch_embed_0
            .forward(x)
            .map_err(|e| candle_to_ocr_inference("FocalSVTR", "patch_embed_0", e))?;
        let x = self
            .patch_embed_1
            .forward(&x)
            .map_err(|e| candle_to_ocr_inference("FocalSVTR", "patch_embed_1", e))?;

        let (_, _, h, w) = x
            .dims4()
            .map_err(|e| candle_to_ocr_inference("FocalSVTR", "get dims", e))?;

        // Flatten to (B, H*W, C)
        let x = x
            .flatten(2, 3)
            .map_err(|e| candle_to_ocr_inference("FocalSVTR", "flatten", e))?
            .permute((0, 2, 1))
            .map_err(|e| candle_to_ocr_inference("FocalSVTR", "permute", e))?;

        // Position dropout
        let mut x = self
            .pos_drop
            .forward(&x, false)
            .map_err(|e| candle_to_ocr_inference("FocalSVTR", "pos_drop", e))?;
        let (mut out_h, mut out_w) = (h, w);

        // Process through layers
        for (i, layer) in self.layers.iter().enumerate() {
            let result = layer.forward(&x, out_h, out_w);
            match result {
                Ok((new_x, new_h, new_w)) => {
                    x = new_x;
                    out_h = new_h;
                    out_w = new_w;
                }
                Err(e) => {
                    return Err(candle_to_ocr_inference(
                        "FocalSVTR",
                        format!("layer.{}: {}", i, e),
                        candle_core::Error::Msg(format!("{}", e)),
                    ));
                }
            }
        }

        // Vision FC projection
        self.vision_fc
            .forward(&x)
            .map_err(|e| candle_to_ocr_inference("FocalSVTR", "vision_fc", e))
    }
}
