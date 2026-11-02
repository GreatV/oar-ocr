//! Shared convolution and evaluation normalization layers.

use candle_core::{Result, Tensor};
use candle_nn::{BatchNorm, Conv2d, Conv2dConfig, Module, ModuleT, VarBuilder};

#[derive(Debug)]
pub(crate) struct ConvNormRelu {
    conv: Conv2d,
    norm: BatchNorm,
}
impl ConvNormRelu {
    pub(crate) fn load(
        input: usize,
        output: usize,
        stride: usize,
        dilation: usize,
        conv: VarBuilder,
        norm: VarBuilder,
    ) -> Result<Self> {
        Ok(Self {
            conv: candle_nn::conv2d(
                input,
                output,
                3,
                Conv2dConfig {
                    stride,
                    padding: dilation,
                    dilation,
                    ..Default::default()
                },
                conv,
            )?,
            norm: candle_nn::batch_norm(output, 1e-5, norm)?,
        })
    }
    pub(crate) fn forward(&self, x: &Tensor) -> Result<Tensor> {
        self.norm.forward_t(&self.conv.forward(x)?, false)?.relu()
    }
}
