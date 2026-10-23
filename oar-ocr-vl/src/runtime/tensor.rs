//! Shared tensor and rotary-embedding helpers.

use crate::api::error::{Error, ProcessingStage};
use crate::runtime::errors::candle_to_ocr_processing;
use candle_core::{D, DType, Device, Tensor};

pub fn vision_inv_freq(
    dim: usize,
    theta: f64,
    model_name: &str,
    device: &Device,
) -> Result<Tensor, Error> {
    let inv_freq: Vec<f32> = (0..dim)
        .step_by(2)
        .map(|index| (1f64 / theta.powf(index as f64 / dim as f64)) as f32)
        .collect();
    Tensor::from_vec(inv_freq, (dim / 2,), device).map_err(|error| {
        candle_to_ocr_processing(
            ProcessingStage::TensorOperation,
            format!("{model_name}: vision inv_freq tensor failed"),
            error,
        )
    })
}

pub fn rotate_half(tensor: &Tensor) -> Result<Tensor, Error> {
    let dimension = tensor.dim(D::Minus1).map_err(|error| {
        candle_to_ocr_processing(ProcessingStage::TensorOperation, "rotate_half dim", error)
    })?;
    let half = dimension / 2;
    let first = tensor.narrow(D::Minus1, 0, half).map_err(|error| {
        candle_to_ocr_processing(ProcessingStage::TensorOperation, "rotate_half first", error)
    })?;
    let second = tensor
        .narrow(D::Minus1, half, dimension - half)
        .map_err(|error| {
            candle_to_ocr_processing(
                ProcessingStage::TensorOperation,
                "rotate_half second",
                error,
            )
        })?;
    let negative_second = second.neg().map_err(|error| {
        candle_to_ocr_processing(
            ProcessingStage::TensorOperation,
            "rotate_half negate",
            error,
        )
    })?;
    Tensor::cat(&[&negative_second, &first], D::Minus1).map_err(|error| {
        candle_to_ocr_processing(
            ProcessingStage::TensorOperation,
            "rotate_half concatenate",
            error,
        )
    })
}

fn rotary_err(model_name: &str, what: &str, error: candle_core::Error) -> Error {
    candle_to_ocr_processing(
        ProcessingStage::TensorOperation,
        format!("{model_name}: rotary {what} failed"),
        error,
    )
}

/// `x * cos + rotate_half(x) * sin` for `q` and `k`, computed in F32 and cast
/// back to each input's dtype: the shared F32 rotary of the Qwen-VL vision
/// towers (`apply_rotary_pos_emb_vision`) and HunyuanOCR's XDRoPE
/// (`apply_rotary_pos_emb_xdrope`).
///
/// `cos`/`sin` must already broadcast against `q`/`k`; they are cast to F32
/// too (a no-op when the caller prepared them in F32).
pub(crate) fn apply_rotary_f32(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    model_name: &str,
) -> Result<(Tensor, Tensor), Error> {
    let cos = cos
        .to_dtype(DType::F32)
        .map_err(|e| rotary_err(model_name, "cos cast", e))?;
    let sin = sin
        .to_dtype(DType::F32)
        .map_err(|e| rotary_err(model_name, "sin cast", e))?;
    let apply = |x: &Tensor, name: &str| -> Result<Tensor, Error> {
        let dtype = x.dtype();
        let x = x
            .to_dtype(DType::F32)
            .map_err(|e| rotary_err(model_name, &format!("{name} cast"), e))?;
        let rotated = rotate_half(&x)?;
        x.broadcast_mul(&cos)
            .and_then(|lhs| lhs.broadcast_add(&rotated.broadcast_mul(&sin)?))
            .and_then(|embedded| embedded.to_dtype(dtype))
            .map_err(|e| rotary_err(model_name, &format!("{name} apply"), e))
    };
    Ok((apply(q, "q")?, apply(k, "k")?))
}

/// One-axis inverse-frequency table for the 2D vision RoPE
/// (`VisionRotaryEmbedding(head_dim // 2)` in the Qwen-VL references).
#[derive(Debug, Clone)]
pub(crate) struct VisionRotaryEmbedding {
    inv_freq: Tensor,
    model_name: &'static str,
}

impl VisionRotaryEmbedding {
    pub(crate) fn new(
        dim: usize,
        theta: f64,
        model_name: &'static str,
        device: &Device,
    ) -> Result<Self, Error> {
        let inv_freq = vision_inv_freq(dim, theta, model_name, device)?;
        Ok(Self {
            inv_freq,
            model_name,
        })
    }

    /// F32 `(seqlen, dim / 2)` table of `position * inv_freq`.
    pub(crate) fn forward(&self, seqlen: usize, device: &Device) -> Result<Tensor, Error> {
        // arange is unsupported on Metal.
        crate::runtime::attention::on_compute_device(device, |compute_device| {
            let seq = Tensor::arange(0u32, seqlen as u32, compute_device)?.to_dtype(DType::F32)?;
            let inv = self
                .inv_freq
                .to_device(compute_device)?
                .to_dtype(DType::F32)?;
            seq.unsqueeze(1)?.matmul(&inv.unsqueeze(0)?)
        })
        .map_err(|e| rotary_err(self.model_name, "vision frequency table", e))
    }
}

/// Per-patch `(h, w)` rotary position ids in the merge-grouped order the
/// Qwen-VL patchifiers emit: merge blocks in raster order `(hb, wb)`, then
/// the `(h_inner, w_inner)` offsets inside each block, repeated per frame.
pub(crate) fn merge_grouped_positions(
    grid_thw: (usize, usize, usize),
    merge_size: usize,
    model_name: &str,
) -> Result<(Vec<u32>, Vec<u32>), Error> {
    let (grid_t, grid_h, grid_w) = grid_thw;
    if grid_t == 0
        || grid_h == 0
        || grid_w == 0
        || merge_size == 0
        || !grid_h.is_multiple_of(merge_size)
        || !grid_w.is_multiple_of(merge_size)
    {
        return Err(Error::InvalidInput {
            message: format!(
                "{model_name} invalid merge-grouped grid: grid={grid_thw:?}, merge={merge_size}"
            ),
        });
    }
    let num_patches = grid_t
        .checked_mul(grid_h)
        .and_then(|value| value.checked_mul(grid_w))
        .filter(|&value| u32::try_from(value).is_ok())
        .ok_or_else(|| Error::InvalidInput {
            message: format!("{model_name} merge-grouped patch count overflow"),
        })?;
    let mut hpos = Vec::with_capacity(num_patches);
    let mut wpos = Vec::with_capacity(num_patches);
    for _ in 0..grid_t {
        for hb in 0..grid_h / merge_size {
            for wb in 0..grid_w / merge_size {
                for h_inner in 0..merge_size {
                    for w_inner in 0..merge_size {
                        hpos.push((hb * merge_size + h_inner) as u32);
                        wpos.push((wb * merge_size + w_inner) as u32);
                    }
                }
            }
        }
    }
    Ok((hpos, wpos))
}

/// `(cos, sin)` of shape `(patches, head_dim)` for the 2D vision RoPE:
/// gather the frequency table at each patch's `(h, w)` id, concatenate the
/// two axes, then double for `rotate_half` (`emb = cat(rot, rot)`).
pub(crate) fn vision_rotary_cos_sin(
    frequency_table: &Tensor,
    hpos: &[u32],
    wpos: &[u32],
    model_name: &str,
    device: &Device,
) -> Result<(Tensor, Tensor), Error> {
    let gather = |ids: &[u32]| -> Result<Tensor, Error> {
        Tensor::from_slice(ids, ids.len(), device)
            .and_then(|ids| frequency_table.index_select(&ids, 0))
            .map_err(|e| rotary_err(model_name, "vision position gather", e))
    };
    let rotary = Tensor::cat(&[&gather(hpos)?, &gather(wpos)?], D::Minus1)
        .map_err(|e| rotary_err(model_name, "vision axes cat", e))?;
    let embedding = Tensor::cat(&[&rotary, &rotary], D::Minus1)
        .map_err(|e| rotary_err(model_name, "vision emb cat", e))?;
    let cos = embedding
        .cos()
        .map_err(|e| rotary_err(model_name, "vision cos", e))?;
    let sin = embedding
        .sin()
        .map_err(|e| rotary_err(model_name, "vision sin", e))?;
    Ok((cos, sin))
}

/// `x * cos + rotate_half(x) * sin` in the input dtype (no upcast).
/// `cos`/`sin` must broadcast against `x`.
pub(crate) fn rope_native(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    model_name: &str,
) -> Result<Tensor, Error> {
    let rotated = rotate_half(x)?;
    x.broadcast_mul(cos)
        .and_then(|lhs| rotated.broadcast_mul(sin).and_then(|rhs| &lhs + &rhs))
        .map_err(|e| rotary_err(model_name, "apply", e))
}

/// [`rope_native`] over the leading `cos.dim(-1)` channels of `x`; the
/// remaining channels pass through unchanged (partial rotary).
pub(crate) fn rope_partial(
    x: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    model_name: &str,
) -> Result<Tensor, Error> {
    let rotary_dim = cos
        .dim(D::Minus1)
        .map_err(|e| rotary_err(model_name, "rotary dimension", e))?;
    let head_dim = x
        .dim(D::Minus1)
        .map_err(|e| rotary_err(model_name, "head dimension", e))?;
    if rotary_dim == head_dim {
        return rope_native(x, cos, sin, model_name);
    }
    let rotary = x
        .narrow(D::Minus1, 0, rotary_dim)
        .map_err(|e| rotary_err(model_name, "rotary slice", e))?;
    let pass = x
        .narrow(D::Minus1, rotary_dim, head_dim - rotary_dim)
        .map_err(|e| rotary_err(model_name, "pass-through slice", e))?;
    let embedded = rope_native(&rotary, cos, sin, model_name)?;
    Tensor::cat(&[&embedded, &pass], D::Minus1).map_err(|e| rotary_err(model_name, "output", e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn merge_grouped_positions_match_patchifier_order() {
        // 2x3 merge blocks in raster order, each listing its four inner
        // patches in (h_inner, w_inner) order (HF's permute(0, 2, 1, 3)).
        let (hpos, wpos) = merge_grouped_positions((1, 4, 6), 2, "test").unwrap();
        assert_eq!(
            hpos,
            [
                0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3, 2, 2, 3, 3
            ]
        );
        assert_eq!(
            wpos,
            [
                0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5, 0, 1, 0, 1, 2, 3, 2, 3, 4, 5, 4, 5
            ]
        );
        assert!(merge_grouped_positions((1, 3, 4), 2, "test").is_err());
        assert!(merge_grouped_positions((0, 4, 4), 2, "test").is_err());
    }
}
