//! Feed-forward residual shared by pre-normalized vision transformer blocks.
use candle_core::{Result, Tensor};
use candle_nn::{Activation, LayerNorm, Linear, Module};

pub(crate) fn mlp_residual(
    hidden: &Tensor,
    norm: &LayerNorm,
    up: &Linear,
    activation: Activation,
    down: &Linear,
) -> Result<Tensor> {
    let features = norm
        .forward(hidden)?
        .apply(up)?
        .apply(&activation)?
        .apply(down)?;
    hidden + features
}
