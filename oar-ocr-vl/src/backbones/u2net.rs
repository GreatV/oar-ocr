//! U²-Net residual U-blocks and its compact mask network.
//!
//! Follows the Apache-2.0 U²-Net implementation embedded by MonkeyOCRv2,
//! by Xuebin Qin and the MonkeyOCRv2 authors.

use candle_core::{Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, Module, VarBuilder};

use crate::runtime::layers::ConvNormRelu;

fn rebn(input: usize, output: usize, dilation: usize, vb: VarBuilder) -> Result<ConvNormRelu> {
    ConvNormRelu::load(input, output, 1, dilation, vb.pp("conv_s1"), vb.pp("bn_s1"))
}

pub(crate) fn resize_like(x: &Tensor, target: &Tensor) -> Result<Tensor> {
    let (_, _, height, width) = target.dims4()?;
    x.upsample_bilinear2d(height, width, false)
}

fn pool(x: &Tensor) -> Result<Tensor> {
    let (b, c, h, w) = x.dims4()?;
    let mut x = x.clone();
    if h % 2 != 0 {
        let pad = Tensor::full(f32::NEG_INFINITY, (b, c, 1, w), x.device())?;
        x = Tensor::cat(&[&x, &pad], 2)?;
    }
    if w % 2 != 0 {
        let pad = Tensor::full(f32::NEG_INFINITY, (b, c, x.dim(2)?, 1), x.device())?;
        x = Tensor::cat(&[&x, &pad], 3)?;
    }
    x.max_pool2d(2)
}

#[derive(Debug)]
pub(crate) struct ResidualUBlock {
    input: ConvNormRelu,
    encoder: Vec<ConvNormRelu>,
    bottom: ConvNormRelu,
    decoder: Vec<ConvNormRelu>,
    dilated: bool,
}
impl ResidualUBlock {
    pub(crate) fn load(
        input: usize,
        middle: usize,
        output: usize,
        depth: usize,
        dilated: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let mut encoder = Vec::new();
        let mut decoder = Vec::new();
        for i in 1..depth {
            let dilation = if dilated { 1 << (i - 1) } else { 1 };
            encoder.push(rebn(
                if i == 1 { output } else { middle },
                middle,
                dilation,
                vb.pp(format!("rebnconv{i}")),
            )?);
            decoder.push(rebn(
                middle * 2,
                if i == 1 { output } else { middle },
                dilation,
                vb.pp(format!("rebnconv{i}d")),
            )?);
        }
        Ok(Self {
            input: rebn(input, output, 1, vb.pp("rebnconvin"))?,
            encoder,
            bottom: rebn(
                middle,
                middle,
                if dilated { 1 << (depth - 1) } else { 2 },
                vb.pp(format!("rebnconv{depth}")),
            )?,
            decoder,
            dilated,
        })
    }
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let residual = self.input.forward(x)?;
        let mut features = Vec::with_capacity(self.encoder.len());
        let mut x = residual.clone();
        for (i, layer) in self.encoder.iter().enumerate() {
            x = layer.forward(&x)?;
            features.push(x.clone());
            if !self.dilated && i + 1 < self.encoder.len() {
                x = pool(&x)?;
            }
        }
        x = self.bottom.forward(&x)?;
        for (i, (layer, feature)) in self.decoder.iter().zip(&features).enumerate().rev() {
            x = layer.forward(&Tensor::cat(&[&x, feature], 1)?)?;
            if !self.dilated && i > 0 {
                x = resize_like(&x, &features[i - 1])?;
            }
        }
        x + residual
    }
}

#[derive(Debug)]
pub(crate) struct U2NetP {
    stages: Vec<ResidualUBlock>,
    decoder: Vec<ResidualUBlock>,
    sides: Vec<Conv2d>,
    fuse: Conv2d,
}
impl U2NetP {
    pub(crate) fn load(vb: VarBuilder) -> Result<Self> {
        let recipes = [
            (7, false),
            (6, false),
            (5, false),
            (4, false),
            (4, true),
            (4, true),
        ];
        let mut stages = Vec::new();
        let mut decoder = Vec::new();
        let mut sides = Vec::new();
        for (i, (depth, dilated)) in recipes.into_iter().enumerate() {
            stages.push(ResidualUBlock::load(
                if i == 0 { 3 } else { 64 },
                16,
                64,
                depth,
                dilated,
                vb.pp(format!("stage{}", i + 1)),
            )?);
            if i < 5 {
                decoder.push(ResidualUBlock::load(
                    128,
                    16,
                    64,
                    depth,
                    dilated,
                    vb.pp(format!("stage{}d", i + 1)),
                )?);
            }
            sides.push(candle_nn::conv2d(
                64,
                1,
                3,
                Conv2dConfig {
                    padding: 1,
                    ..Default::default()
                },
                vb.pp(format!("side{}", i + 1)),
            )?);
        }
        Ok(Self {
            stages,
            decoder,
            sides,
            fuse: candle_nn::conv2d(6, 1, 1, Default::default(), vb.pp("outconv"))?,
        })
    }
    pub(crate) fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let mut encoded = Vec::with_capacity(6);
        let mut x = input.clone();
        for (i, stage) in self.stages.iter().enumerate() {
            x = stage.forward(&x)?;
            encoded.push(x.clone());
            if i < 5 {
                x = pool(&x)?;
            }
        }
        let mut maps = vec![resize_like(&self.sides[5].forward(&x)?, input)?];
        for i in (0..5).rev() {
            x = resize_like(&x, &encoded[i])?;
            x = self.decoder[i].forward(&Tensor::cat(&[&x, &encoded[i]], 1)?)?;
            maps.push(resize_like(&self.sides[i].forward(&x)?, input)?);
        }
        maps.reverse();
        candle_nn::ops::sigmoid(&self.fuse.forward(&Tensor::cat(&maps, 1)?)?)
    }
}
