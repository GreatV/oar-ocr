//! ViTEraser point-grid head from MonkeyOCRv2's Apache-2.0 preprocessor.

use crate::backbones::swin_v2::{PatchSplit, SwinV2Encoder, SwinV2Stage};
use crate::runtime::layers::ConvNormRelu;
use candle_core::{Result, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, Linear, Module, VarBuilder};

#[derive(Debug)]
struct Lateral {
    layers: Vec<Conv2d>,
}
impl Lateral {
    fn load(dim: usize, vb: VarBuilder) -> Result<Self> {
        let mut layers = Vec::new();
        for (i, (input, output, kernel)) in [
            (dim, dim, 1),
            (dim, dim * 2, 3),
            (dim * 2, dim * 2, 3),
            (dim * 2, dim, 1),
        ]
        .into_iter()
        .enumerate()
        {
            layers.push(candle_nn::conv2d(
                input,
                output,
                kernel,
                Conv2dConfig {
                    padding: kernel / 2,
                    ..Default::default()
                },
                vb.pp(i),
            )?);
        }
        Ok(Self { layers })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mut x = x.clone();
        for layer in &self.layers {
            x = layer.forward(&x)?;
        }
        Ok(x)
    }
}

#[derive(Debug)]
pub(super) struct PointGrid {
    encoder: SwinV2Encoder,
    embed: Linear,
    decoder: Vec<SwinV2Stage>,
    splits: Vec<PatchSplit>,
    lateral: Vec<Lateral>,
    norm: LayerNorm,
    head: Vec<ConvNormRelu>,
    output: Conv2d,
}
impl PointGrid {
    pub(super) fn load(vb: VarBuilder) -> Result<Self> {
        let encoder = SwinV2Encoder::load(
            96,
            &[2, 2, 6, 2],
            &[3, 6, 12, 24],
            16,
            8,
            vb.pp("encoder.backbone"),
        )?;
        let dec = vb.pp("decoder");
        let mut decoder = Vec::new();
        let mut splits = Vec::new();
        let mut lateral = Vec::new();
        for (i, (depth, heads)) in [(2, 24), (6, 12), (2, 6), (2, 3), (2, 2)]
            .into_iter()
            .enumerate()
        {
            let dim = 768 >> i;
            let layer = dec.pp("layers").pp(i);
            decoder.push(SwinV2Stage::load(dim, heads, depth, 16, 8, layer.clone())?);
            splits.push(PatchSplit::load(dim, layer.pp("upsample"))?);
            if i < 3 {
                lateral.push(Lateral::load(
                    dim / 2,
                    dec.pp("lateral_connection_list").pp(i),
                )?);
            }
        }
        let mut head = Vec::new();
        for (i, (input, output, stride, dilation)) in [
            (24, 128, 2, 1),
            (128, 256, 2, 1),
            (256, 256, 1, 2),
            (256, 192, 2, 1),
        ]
        .into_iter()
        .enumerate()
        {
            head.push(ConvNormRelu::load(
                input,
                output,
                stride,
                dilation,
                dec.pp("pred_conv").pp(3 * i),
                dec.pp("pred_conv").pp(3 * i + 1),
            )?);
        }
        Ok(Self {
            encoder,
            embed: candle_nn::linear(768, 768, vb.pp("pixel_embed"))?,
            decoder,
            splits,
            lateral,
            norm: candle_nn::layer_norm(24, 1e-5, dec.pp("norm4"))?,
            head,
            output: candle_nn::conv2d(
                192,
                2,
                3,
                Conv2dConfig {
                    stride: 2,
                    padding: 1,
                    ..Default::default()
                },
                dec.pp("pred_conv.12"),
            )?,
        })
    }
    pub(super) fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let features = self.encoder.forward(input)?;
        let last = features
            .last()
            .ok_or_else(|| candle_core::Error::Msg("SwinV2 encoder returned no levels".into()))?;
        let (batch, _, mut height, mut width) = last.dims4()?;
        let mut x = self
            .embed
            .forward(&last.permute((0, 2, 3, 1))?.contiguous()?)?
            .reshape((batch, height * width, 768))?;
        for (i, (stage, split)) in self.decoder.iter().zip(&self.splits).enumerate() {
            x = stage.forward(&x, height, width)?;
            x = split.forward(&x, height, width)?;
            height *= 2;
            width *= 2;
            if let Some(lateral) = self.lateral.get(i) {
                let skip = lateral
                    .forward(&features[features.len() - i - 2])?
                    .flatten_from(2)?
                    .transpose(1, 2)?;
                x = (&x + skip)?;
            }
        }
        let mut x = self
            .norm
            .forward(&x)?
            .reshape((batch, height, width, 24))?
            .permute((0, 3, 1, 2))?
            .contiguous()?;
        for layer in &self.head {
            x = layer.forward(&x)?;
        }
        self.output.forward(&x)
    }
}
