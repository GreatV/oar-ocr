use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};

pub(super) struct Mlp {
    gate: Linear,
    up: Linear,
    down: Linear,
}

impl Mlp {
    pub fn load(hidden: usize, intermediate: usize, vb: VarBuilder<'_>) -> Result<Self> {
        Ok(Self {
            gate: linear_no_bias(hidden, intermediate, vb.pp("gate_proj"))?,
            up: linear_no_bias(hidden, intermediate, vb.pp("up_proj"))?,
            down: linear_no_bias(intermediate, hidden, vb.pp("down_proj"))?,
        })
    }

    pub fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        self.down
            .forward(&(candle_nn::ops::silu(&self.gate.forward(xs)?)? * self.up.forward(xs)?)?)
    }
}

pub(super) fn rotary(
    xs: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    interleave: bool,
    upcast: bool,
) -> Result<Tensor> {
    let dtype = xs.dtype();
    let mut xs = if upcast {
        xs.to_dtype(DType::F32)?
    } else {
        xs.clone()
    };
    let dim = xs.dim(D::Minus1)?;
    if interleave {
        let mut shape = xs.dims().to_vec();
        shape.pop();
        shape.extend([dim / 2, 2]);
        xs = xs
            .reshape(shape)?
            .transpose(D::Minus1, D::Minus2)?
            .flatten(D::Minus2, D::Minus1)?;
    }
    let first = xs.narrow(D::Minus1, 0, dim / 2)?;
    let second = xs.narrow(D::Minus1, dim / 2, dim / 2)?.neg()?;
    let rotated = Tensor::cat(&[&second, &first], D::Minus1)?;
    (xs.broadcast_mul(cos)? + rotated.broadcast_mul(sin)?)?.to_dtype(dtype)
}

pub(super) fn cos_sin(
    angles: Vec<f32>,
    len: usize,
    half_dim: usize,
    dtype: DType,
    device: &Device,
) -> Result<(Tensor, Tensor)> {
    let angles = Tensor::from_vec(angles, (len, half_dim), device)?;
    let angles = Tensor::cat(&[&angles, &angles], 1)?.reshape((1, 1, len, half_dim * 2))?;
    Ok((
        angles.cos()?.to_dtype(dtype)?,
        angles.sin()?.to_dtype(dtype)?,
    ))
}
