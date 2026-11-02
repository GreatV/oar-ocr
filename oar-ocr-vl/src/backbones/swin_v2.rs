//! SwinV2 window attention, residual blocks, and patch rearrangement.
//!
//! Architecture follows the Apache-2.0 Swin Transformer Object Detection code
//! embedded by MonkeyOCRv2 (Microsoft and MonkeyOCRv2 authors).

use candle_core::{D, Tensor};
use candle_nn::{LayerNorm, Linear, Module, VarBuilder, layer_norm, linear, linear_no_bias};

type Result<T> = candle_core::Result<T>;

#[derive(Debug)]
pub(crate) struct SwinV2Block {
    qkv: Linear,
    projection: Linear,
    norm1: LayerNorm,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    bias: Tensor,
    scale: Tensor,
    dim: usize,
    heads: usize,
    window: usize,
    shift: usize,
}

impl SwinV2Block {
    pub(crate) fn load(
        dim: usize,
        heads: usize,
        window: usize,
        shift: usize,
        pretrained_window: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let attn = vb.pp("attn");
        let q_bias = attn.get(dim, "q_bias")?;
        let v_bias = attn.get(dim, "v_bias")?;
        let zeros = q_bias.zeros_like()?;
        let qkv = Linear::new(
            attn.get((3 * dim, dim), "qkv.weight")?,
            Some(Tensor::cat(&[&q_bias, &zeros, &v_bias], 0)?),
        );
        let extent = 2 * window - 1;
        let mut coordinates = Vec::with_capacity(extent * extent * 2);
        let denominator = if pretrained_window > 0 {
            pretrained_window - 1
        } else {
            window - 1
        };
        for y in 0..extent {
            for x in 0..extent {
                for value in [
                    y as f32 - (window - 1) as f32,
                    x as f32 - (window - 1) as f32,
                ] {
                    let value = value / denominator as f32 * 8.;
                    coordinates.push(value.signum() * (value.abs() + 1.).log2() / 3.);
                }
            }
        }
        // Checkpoints retain the coordinate scale used to train each stage.
        let coordinates = if attn.contains_tensor("relative_coords_table") {
            attn.get((1, extent, extent, 2), "relative_coords_table")?
                .reshape((extent * extent, 2))?
        } else {
            Tensor::from_vec(coordinates, (extent * extent, 2), vb.device())?
        };
        let cpb = attn.pp("cpb_mlp");
        let table = linear_no_bias(512, heads, cpb.pp(2))?
            .forward(&linear(2, 512, cpb.pp(0))?.forward(&coordinates)?.relu()?)?;
        let tokens = window * window;
        let mut indices = Vec::with_capacity(tokens * tokens);
        for a in 0..tokens {
            for b in 0..tokens {
                let dy = (a / window + window - 1) as i32 - (b / window) as i32;
                let dx = (a % window + window - 1) as i32 - (b % window) as i32;
                indices.push((dy * extent as i32 + dx) as u32);
            }
        }
        let index = Tensor::from_vec(indices, tokens * tokens, vb.device())?;
        let bias = table
            .index_select(&index, 0)?
            .reshape((tokens, tokens, heads))?
            .permute((2, 0, 1))?
            .contiguous()?;
        let bias = (candle_nn::ops::sigmoid(&bias)? * 16.)?;
        Ok(Self {
            qkv,
            projection: linear(dim, dim, attn.pp("proj"))?,
            norm1: layer_norm(dim, 1e-5, vb.pp("norm1"))?,
            norm2: layer_norm(dim, 1e-5, vb.pp("norm2"))?,
            fc1: linear(dim, dim * 4, vb.pp("mlp.fc1"))?,
            fc2: linear(dim * 4, dim, vb.pp("mlp.fc2"))?,
            bias,
            scale: attn
                .get((heads, 1, 1), "logit_scale")?
                .clamp(f32::NEG_INFINITY, 100f32.ln())?
                .exp()?,
            dim,
            heads,
            window,
            shift,
        })
    }

    pub(crate) fn forward(&self, input: &Tensor, height: usize, width: usize) -> Result<Tensor> {
        let batch = input.dim(0)?;
        let hp = height.div_ceil(self.window) * self.window;
        let wp = width.div_ceil(self.window) * self.window;
        let mut x = input
            .reshape((batch, height, width, self.dim))?
            .pad_with_zeros(1, 0, hp - height)?
            .pad_with_zeros(2, 0, wp - width)?;
        if self.shift > 0 {
            x = x
                .roll(-(self.shift as i32), 1)?
                .roll(-(self.shift as i32), 2)?;
        }
        let windows = partition(&x, self.window)?;
        let n = self.window * self.window;
        let windows_count = windows.dim(0)?;
        let head_dim = self.dim / self.heads;
        let qkv = self
            .qkv
            .forward(&windows)?
            .reshape((windows_count, n, 3, self.heads, head_dim))?
            .permute((2, 0, 3, 1, 4))?;
        let q = qkv.narrow(0, 0, 1)?.squeeze(0)?.contiguous()?;
        let k = qkv.narrow(0, 1, 1)?.squeeze(0)?.contiguous()?;
        let v = qkv.narrow(0, 2, 1)?.squeeze(0)?.contiguous()?;
        let normalize = |v: &Tensor| {
            v.broadcast_div(
                &v.sqr()?
                    .sum_keepdim(D::Minus1)?
                    .sqrt()?
                    .clamp(1e-12f32, f32::INFINITY)?,
            )
        };
        let mut scores = normalize(&q)?
            .matmul(&normalize(&k)?.transpose(2, 3)?)?
            .broadcast_mul(&self.scale)?
            .broadcast_add(&self.bias)?;
        if self.shift > 0 {
            let mask = shift_mask(hp, wp, self.window, self.shift, input.device())?;
            let count = mask.dim(0)?;
            scores = scores
                .reshape((batch, count, self.heads, n, n))?
                .broadcast_add(&mask.unsqueeze(0)?.unsqueeze(2)?)?
                .reshape((windows_count, self.heads, n, n))?;
        }
        let attention = candle_nn::ops::softmax_last_dim(&scores)?;
        let values = attention
            .matmul(&v)?
            .transpose(1, 2)?
            .contiguous()?
            .reshape((windows_count, n, self.dim))?;
        let values = self.projection.forward(&values)?;
        let mut x = reverse(&values, batch, hp, wp, self.window)?;
        if self.shift > 0 {
            x = x.roll(self.shift as i32, 1)?.roll(self.shift as i32, 2)?;
        }
        let x = x
            .narrow(1, 0, height)?
            .narrow(2, 0, width)?
            .contiguous()?
            .reshape((batch, height * width, self.dim))?;
        let x = (input + self.norm1.forward(&x)?)?;
        let mlp = self.fc2.forward(&self.fc1.forward(&x)?.gelu_erf()?)?;
        &x + self.norm2.forward(&mlp)?
    }
}

fn partition(x: &Tensor, window: usize) -> Result<Tensor> {
    let (b, h, w, c) = x.dims4()?;
    x.reshape((b, h / window, window, w / window, window, c))?
        .permute((0, 1, 3, 2, 4, 5))?
        .contiguous()?
        .reshape((b * (h / window) * (w / window), window * window, c))
}

fn reverse(x: &Tensor, batch: usize, height: usize, width: usize, window: usize) -> Result<Tensor> {
    let channels = x.dim(2)?;
    x.reshape((
        batch,
        height / window,
        width / window,
        window,
        window,
        channels,
    ))?
    .permute((0, 1, 3, 2, 4, 5))?
    .contiguous()?
    .reshape((batch, height, width, channels))
}

fn shift_mask(
    height: usize,
    width: usize,
    window: usize,
    shift: usize,
    device: &candle_core::Device,
) -> Result<Tensor> {
    let region = |coordinate: usize, length: usize| {
        if coordinate < length - window {
            0
        } else if coordinate < length - shift {
            1
        } else {
            2
        }
    };
    let mut mask = Vec::with_capacity((height / window) * (width / window) * window.pow(4));
    for row in 0..height / window {
        for column in 0..width / window {
            let labels: Vec<_> = (0..window * window)
                .map(|i| {
                    region(row * window + i / window, height) * 3
                        + region(column * window + i % window, width)
                })
                .collect();
            for a in &labels {
                for b in &labels {
                    mask.push(if a == b { 0f32 } else { -100. });
                }
            }
        }
    }
    Tensor::from_vec(
        mask,
        (
            (height / window) * (width / window),
            window * window,
            window * window,
        ),
        device,
    )
}

#[derive(Debug)]
pub(crate) struct PatchMerge {
    reduction: Linear,
    norm: LayerNorm,
}
impl PatchMerge {
    pub(crate) fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            reduction: linear_no_bias(4 * dim, 2 * dim, vb.pp("reduction"))?,
            norm: layer_norm(2 * dim, 1e-5, vb.pp("norm"))?,
        })
    }
    pub(crate) fn forward(&self, x: &Tensor, height: usize, width: usize) -> Result<Tensor> {
        let (batch, _, dim) = x.dims3()?;
        let x = x
            .reshape((batch, height / 2, 2, width / 2, 2, dim))?
            .permute((0, 1, 3, 4, 2, 5))?
            .contiguous()?
            .reshape((batch, height * width / 4, 4 * dim))?;
        self.norm.forward(&self.reduction.forward(&x)?)
    }
}

#[derive(Debug)]
pub(crate) struct PatchSplit {
    projection: Linear,
    norm: LayerNorm,
}
impl PatchSplit {
    pub(crate) fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            projection: linear_no_bias(dim / 4, dim / 2, vb.pp("upsample"))?,
            norm: layer_norm(dim / 2, 1e-5, vb.pp("norm"))?,
        })
    }
    pub(crate) fn forward(&self, x: &Tensor, height: usize, width: usize) -> Result<Tensor> {
        let (batch, _, dim) = x.dims3()?;
        let order = Tensor::from_slice(&[0u32, 2, 1, 3], 4, x.device())?;
        let x = x
            .reshape((batch, height, width, 4, dim / 4))?
            .index_select(&order, 3)?
            .reshape((batch, height, width, 2, 2, dim / 4))?
            .permute((0, 1, 3, 2, 4, 5))?
            .contiguous()?
            .reshape((batch, height * width * 4, dim / 4))?;
        self.norm.forward(&self.projection.forward(&x)?)
    }
}

#[derive(Debug)]
pub(crate) struct SwinV2Stage {
    blocks: Vec<SwinV2Block>,
}
impl SwinV2Stage {
    pub(crate) fn load(
        dim: usize,
        heads: usize,
        depth: usize,
        window: usize,
        pretrained_window: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let blocks = (0..depth)
            .map(|i| {
                SwinV2Block::load(
                    dim,
                    heads,
                    window,
                    if i % 2 == 0 { 0 } else { window / 2 },
                    pretrained_window,
                    vb.pp("blocks").pp(i),
                )
            })
            .collect::<Result<_>>()?;
        Ok(Self { blocks })
    }
    pub(crate) fn forward(&self, x: &Tensor, height: usize, width: usize) -> Result<Tensor> {
        let mut x = x.clone();
        for block in &self.blocks {
            x = block.forward(&x, height, width)?;
        }
        Ok(x)
    }
}

#[derive(Debug)]
pub(crate) struct SwinV2Encoder {
    patch: candle_nn::Conv2d,
    patch_norm: LayerNorm,
    stages: Vec<SwinV2Stage>,
    merging: Vec<PatchMerge>,
    norms: Vec<LayerNorm>,
}
impl SwinV2Encoder {
    pub(crate) fn load(
        dim: usize,
        depths: &[usize],
        heads: &[usize],
        window: usize,
        pretrained_window: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut stages = Vec::new();
        let mut merging = Vec::new();
        let mut norms = Vec::new();
        for (i, &depth) in depths.iter().enumerate() {
            let width = dim << i;
            let layer = vb.pp("layers").pp(i);
            stages.push(SwinV2Stage::load(
                width,
                heads[i],
                depth,
                window,
                pretrained_window,
                layer.clone(),
            )?);
            if i + 1 < depths.len() {
                merging.push(PatchMerge::load(width, layer.pp("downsample"))?);
            }
            norms.push(layer_norm(width, 1e-5, vb.pp(format!("norm{i}")))?);
        }
        Ok(Self {
            patch: candle_nn::conv2d(
                3,
                dim,
                4,
                candle_nn::Conv2dConfig {
                    stride: 4,
                    ..Default::default()
                },
                vb.pp("patch_embed.proj"),
            )?,
            patch_norm: layer_norm(dim, 1e-5, vb.pp("patch_embed.norm"))?,
            stages,
            merging,
            norms,
        })
    }
    pub(crate) fn forward(&self, input: &Tensor) -> Result<Vec<Tensor>> {
        let x = self.patch.forward(input)?;
        let (batch, dim, mut height, mut width) = x.dims4()?;
        let mut x = self
            .patch_norm
            .forward(&x.flatten_from(2)?.transpose(1, 2)?.contiguous()?)?;
        let mut features = Vec::new();
        for (i, stage) in self.stages.iter().enumerate() {
            x = stage.forward(&x, height, width)?;
            features.push(
                self.norms[i]
                    .forward(&x)?
                    .reshape((batch, height, width, dim << i))?
                    .permute((0, 3, 1, 2))?
                    .contiguous()?,
            );
            if let Some(merge) = self.merging.get(i) {
                x = merge.forward(&x, height, width)?;
                height /= 2;
                width /= 2;
            }
        }
        Ok(features)
    }
}
