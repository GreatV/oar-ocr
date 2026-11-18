//! Content-normalized patch positions from the OpenMDW Nemotron export.
use super::processing::Page;
use candle_core::{DType, Device, Result, Tensor};

const SOURCE: usize = 128;
fn resize_weights(out: usize, average: bool, device: &Device) -> Result<Tensor> {
    let mut table = vec![0f32; out * SOURCE];
    let scale = SOURCE as f32 / out as f32;
    for (i, row) in table.chunks_mut(SOURCE).enumerate() {
        if average {
            let start = i * SOURCE / out;
            let end = ((i + 1) * SOURCE).div_ceil(out);
            row[start..end].fill(1. / (end - start) as f32);
        } else {
            let center = (i as f32 + 0.5) * scale;
            let radius = scale.max(1.);
            for (j, v) in row.iter_mut().enumerate() {
                *v = (1. - ((j as f32 + 0.5 - center) / radius).abs()).max(0.);
            }
            let sum: f32 = row.iter().sum();
            for v in row {
                *v /= sum;
            }
        }
    }
    Tensor::from_vec(table, (out, SOURCE), device)
}

fn sample_weights(patches: usize, content: usize, device: &Device) -> Result<Tensor> {
    let visible = content.div_ceil(16);
    let mut table = vec![0f32; patches * visible];
    for (i, row) in table.chunks_mut(visible).enumerate() {
        let start = i * 16;
        let extent = content.saturating_sub(start).min(16);
        let midpoint = (start as f32 + 0.5 * extent as f32).min(content as f32) / content as f32;
        let grid = 2. * midpoint - 1.;
        let x = ((grid + 1.) * visible as f32 / 2. - 0.5).clamp(0., (visible - 1) as f32);
        let left = x.floor() as usize;
        let fraction = x - left as f32;
        row[left] += 1. - fraction;
        row[(left + 1).min(visible - 1)] += fraction;
    }
    Tensor::from_vec(table, (patches, visible), device)
}

/// Separable interpolation of a row-major HWC field. Each matrix fits in at most 128×128 elements.
fn resample(grid: &Tensor, y: &Tensor, x: &Tensor) -> Result<Tensor> {
    let (h, w, c) = grid.dims3()?;
    let (oh, ow) = (y.dim(0)?, x.dim(0)?);
    let by_x = x
        .matmul(&grid.transpose(0, 1)?.contiguous()?.reshape((w, h * c))?)?
        .reshape((ow, h, c))?;
    y.matmul(&by_x.transpose(0, 1)?.contiguous()?.reshape((h, ow * c))?)?
        .reshape((oh, ow, c))
}
pub(super) fn positions(field: &Tensor, page: &Page) -> Result<Tensor> {
    let dev = field.device();
    let (rh, cw) = (page.height.div_ceil(16), page.width.div_ceil(16));
    let pool = rh.min(cw) == 1;
    let grid = field
        .to_dtype(DType::F32)?
        .reshape((SOURCE, SOURCE, 1152))?;
    let visible = resample(
        &grid,
        &resize_weights(rh, pool, dev)?,
        &resize_weights(cw, pool, dev)?,
    )?;
    resample(
        &visible,
        &sample_weights(page.rows, page.height, dev)?,
        &sample_weights(page.cols, page.width, dev)?,
    )?
    .reshape((1, page.rows * page.cols, 1152))?
    .to_dtype(page.patches.dtype())
}
