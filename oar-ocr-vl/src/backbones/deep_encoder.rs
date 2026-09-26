//! DeepEncoder dual vision tower shared by DeepSeek-OCR-style checkpoints.
//!
//! Ported from `deepencoder.py`: SAM ViT-B and CLIP-L share one spatial grid —
//! SAM processes the image (patch 16, windowed attention with decomposed
//! relative position embeddings, neck plus two stride-2 convolutions down to
//! `grid/4 × grid/4` at 1024 channels) and its output *is* CLIP's patch
//! embedding: CLIP adds its class token and (interpolated) position embedding
//! on top, runs 24 quick-GELU blocks, and the class token is dropped.
//! `[CLIP features | SAM features]` then pass through a single linear
//! projector to the decoder width.

use crate::error::Error;
use crate::runtime::attention::scaled_dot_product_attention;
use crate::runtime::errors::{candle_to_ocr_inference, candle_to_ocr_processing};
use candle_core::{DType, IndexOp, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder, layer_norm,
    linear,
};

const MODEL_NAME: &str = "DeepEncoder";

fn quick_gelu(xs: &Tensor) -> Result<Tensor, Error> {
    let scaled =
        (xs * 1.702f64).map_err(|e| candle_to_ocr_inference(MODEL_NAME, "quick_gelu scale", e))?;
    let sig = candle_nn::ops::sigmoid(&scaled)
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "quick_gelu sigmoid", e))?;
    (xs * sig).map_err(|e| candle_to_ocr_inference(MODEL_NAME, "quick_gelu multiply", e))
}

#[derive(Debug, Clone)]
pub struct SamConfig {
    pub depth: usize,
    pub width: usize,
    pub heads: usize,
    pub patch_size: usize,
    /// Blocks using global attention; the rest attend within windows.
    pub global_attn_indexes: Vec<usize>,
    /// Window side for the windowed blocks (`window_size` in the reference).
    pub window_size: usize,
    /// Side of the patch grid the rel-pos tables were pretrained for.
    pub pretrained_grid_side: usize,
    /// Output channels of the final stride-2 downsampling convolution.
    pub out_channels: usize,
}

impl SamConfig {
    fn mlp_hidden(&self) -> usize {
        (self.width as f64 * 4.0).round() as usize
    }
}

#[derive(Debug, Clone)]
pub struct ClipConfig {
    pub depth: usize,
    pub width: usize,
    pub heads: usize,
    pub ffn_hidden_size: usize,
    pub layer_norm_eps: f64,
    /// Side of the position-embedding grid at pretraining (224/14 = 16).
    pub pretrained_grid_side: usize,
}

/// Torch-compatible antialias bicubic resize of a `(1, C, H, W)` tensor to
/// `(out_h, out_w)`, following `F.interpolate(..., mode="bicubic",
/// antialias=True, align_corners=False)`: a normalized cubic-convolution
/// kernel (a = -0.5) whose support widens with the downscale factor. Runs in
/// f32 on the CPU (the inputs are tiny position-embedding grids).
pub(crate) fn bicubic_antialias_resize(
    input: &Tensor,
    out_h: usize,
    out_w: usize,
) -> Result<Tensor, Error> {
    let (dims, data) = input
        .to_dtype(DType::F32)
        .and_then(|input| {
            let dims = input.dims().to_vec();
            let data = input.flatten_all()?.to_vec1::<f32>()?;
            Ok((dims, data))
        })
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "read bicubic input", e))?;
    let [_, channels, in_h, in_w] = dims[..] else {
        return Err(Error::InvalidInput {
            message: format!("{MODEL_NAME} bicubic resize expects NCHW, got {dims:?}"),
        });
    };
    if in_h == 0 || in_w == 0 || out_h == 0 || out_w == 0 {
        return Err(Error::InvalidInput {
            message: format!(
                "{MODEL_NAME} bicubic resize needs non-empty sizes, got {dims:?} -> {out_h}x{out_w}"
            ),
        });
    }
    let scale_h = in_h as f64 / out_h as f64;
    let scale_w = in_w as f64 / out_w as f64;

    // Horizontal pass: (C, in_h, in_w) -> (C, in_h, out_w).
    let mut tmp = vec![0f32; channels * in_h * out_w];
    for c in 0..channels {
        for y in 0..in_h {
            let row = &data[c * in_h * in_w + y * in_w..(c * in_h + y + 1) * in_w];
            for x in 0..out_w {
                let center = scale_w * (x as f64 + 0.5) - 0.5;
                let (value, total) = bicubic_sample_1d(row, center, scale_w);
                tmp[c * in_h * out_w + y * out_w + x] = (value / total) as f32;
            }
        }
    }
    // Vertical pass: (C, in_h, out_w) -> (C, out_h, out_w).
    let mut out = vec![0f32; channels * out_h * out_w];
    for c in 0..channels {
        for x in 0..out_w {
            let column: Vec<f32> = (0..in_h)
                .map(|src| tmp[c * in_h * out_w + src * out_w + x])
                .collect();
            for y in 0..out_h {
                let center = scale_h * (y as f64 + 0.5) - 0.5;
                let (value, total) = bicubic_sample_1d(&column, center, scale_h);
                out[c * out_h * out_w + y * out_w + x] = (value / total) as f32;
            }
        }
    }
    Tensor::from_vec(out, (1, channels, out_h, out_w), input.device())
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "build bicubic output", e))
}

/// One-dimensional normalized antialias cubic sampling: returns
/// `(weighted_sum, weight_sum)` over the source window, clamped at the
/// borders (replicate padding, like torch's `get_value_bounded`).
fn bicubic_sample_1d(samples: &[f32], center: f64, scale: f64) -> (f64, f64) {
    const RADIUS: f64 = 2.0;
    let support = if scale > 1.0 { RADIUS * scale } else { RADIUS };
    let low = ((center - support).floor() as isize).max(0) as usize;
    let high = ((center + support).ceil() as isize).min(samples.len() as isize - 1) as usize;
    let (mut acc, mut total) = (0f64, 0f64);
    for (offset, &sample) in samples[low..=high].iter().enumerate() {
        let distance = (low + offset) as f64 - center;
        let weight = cubic_convolution(distance / scale);
        acc += weight * sample as f64;
        total += weight;
    }
    (acc, total.max(f64::MIN_POSITIVE))
}

/// Cubic convolution kernel with `a = -0.5` (Keys), as used by torch.
fn cubic_convolution(x: f64) -> f64 {
    const A: f64 = -0.5;
    let ax = x.abs();
    if ax <= 1.0 {
        ((A + 2.0) * ax - (A + 3.0)) * ax * ax + 1.0
    } else if ax < 2.0 {
        ((A * ax - 5.0 * A) * ax + 8.0 * A) * ax - 4.0 * A
    } else {
        0.0
    }
}

/// `F.interpolate(..., mode="linear")` along the row axis of a `(rows, dim)`
/// table (`align_corners=False`), used by `get_rel_pos` when the attention
/// grid differs from SAM pretraining.
fn linear_interp_rows(table: &Tensor, target_rows: usize) -> Result<Tensor, Error> {
    let (rows, dim) = table
        .dims2()
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "rel pos table shape", e))?;
    if rows == target_rows || rows == 1 {
        return Ok(table.clone());
    }
    let data = table
        .to_dtype(DType::F32)
        .and_then(|t| t.flatten_all()?.to_vec1::<f32>())
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "rel pos table read", e))?;
    let scale = rows as f64 / target_rows as f64;
    let mut out = vec![0f32; target_rows * dim];
    for i in 0..target_rows {
        let src_index = (scale * (i as f64 + 0.5) - 0.5).clamp(0.0, (rows - 1) as f64);
        let i0 = src_index.floor() as usize;
        let i1 = (i0 + 1).min(rows - 1);
        let frac = src_index - i0 as f64;
        for d in 0..dim {
            let a = data[i0 * dim + d] as f64;
            let b = data[i1 * dim + d] as f64;
            out[i * dim + d] = (a + (b - a) * frac) as f32;
        }
    }
    Tensor::from_vec(out, (target_rows, dim), table.device())
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "rel pos table interp", e))
}

/// Build the per-query-row relative table `Rh[ri, rj] = full[ri - rj + kh - 1]`
/// (`get_rel_pos`'s index arithmetic), shape `(query_side, key_side, dim)`.
fn gather_rel_table(
    full: &Tensor,
    (query_side, key_side): (usize, usize),
) -> Result<Tensor, Error> {
    let device = full.device();
    let mut indices = Vec::with_capacity(query_side * key_side);
    for ri in 0..query_side {
        for rj in 0..key_side {
            indices.push((ri + key_side - rj - 1) as u32);
        }
    }
    full.index_select(
        &Tensor::from_vec(indices, query_side * key_side, device)?,
        0,
    )
    .and_then(|t| t.reshape((query_side, key_side, t.dim(1)?)))
    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "gather rel pos table", e))
}

// ---------------------------------------------------------------------------
// SAM ViT-B
// ---------------------------------------------------------------------------

struct SamAttention {
    qkv: Linear,
    proj: Linear,
    rel_pos_h: Tensor,
    rel_pos_w: Tensor,
    heads: usize,
    head_dim: usize,
}

impl SamAttention {
    fn load(cfg: &SamConfig, global_attention: bool, vb: VarBuilder) -> Result<Self, Error> {
        let qkv = linear(cfg.width, cfg.width * 3, vb.pp("attn").pp("qkv"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load SAM qkv", e))?;
        let proj = linear(cfg.width, cfg.width, vb.pp("attn").pp("proj"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load SAM proj", e))?;
        // Global blocks pretrained on the full 64×64 grid, windowed blocks on
        // 14×14 windows — the rel-pos table rows differ accordingly.
        let rel_side = if global_attention {
            cfg.pretrained_grid_side
        } else {
            cfg.window_size
        };
        let rel_rows = 2 * rel_side - 1;
        let rel_pos_h = vb
            .get((rel_rows, cfg.width / cfg.heads), "attn.rel_pos_h")
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load SAM rel_pos_h", e))?;
        let rel_pos_w = vb
            .get((rel_rows, cfg.width / cfg.heads), "attn.rel_pos_w")
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load SAM rel_pos_w", e))?;
        Ok(Self {
            qkv,
            proj,
            rel_pos_h,
            rel_pos_w,
            heads: cfg.heads,
            head_dim: cfg.width / cfg.heads,
        })
    }

    /// `hidden_states`: `(H, W, width)` for one image (or window); attention
    /// with the decomposed rel-pos bias. Scores run per head in query chunks
    /// so the `S×S` bias is never materialized whole.
    fn forward(&self, hidden_states: &Tensor, grid: (usize, usize)) -> Result<Tensor, Error> {
        let (side_h, side_w, width) = hidden_states
            .dims3()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "SAM attention input", e))?;
        let seq = side_h * side_w;
        let flat = hidden_states
            .reshape((seq, width))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "SAM attention flatten", e))?;
        let qkv = self
            .qkv
            .forward(&flat)
            .and_then(|qkv| qkv.reshape((seq, 3, self.heads, self.head_dim)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "SAM qkv projection", e))?;
        let q = qkv.i((.., 0, .., ..))?;
        let k = qkv.i((.., 1, .., ..))?;
        let v = qkv.i((.., 2, .., ..))?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();

        let mut outputs = Vec::with_capacity(self.heads);
        for head in 0..self.heads {
            // Per-head slices are strided views; CUDA matmul needs contiguous
            // operands, unlike the CPU kernels.
            let q_head = q.i((.., head, ..))?.contiguous()?;
            let k_head = k.i((.., head, ..))?.contiguous()?;
            let v_head = v.i((.., head, ..))?.contiguous()?;
            outputs.push(self.attend_head(&q_head, &k_head, &v_head, scale, grid)?);
        }
        let refs: Vec<&Tensor> = outputs.iter().collect();
        let attn = Tensor::cat(&refs, 1)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "SAM heads cat", e))?;
        self.proj
            .forward(&attn)
            .and_then(|out| out.reshape((side_h, side_w, width)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "SAM proj", e))
    }

    fn attend_head(
        &self,
        q: &Tensor,
        k: &Tensor,
        v: &Tensor,
        scale: f64,
        grid: (usize, usize),
    ) -> Result<Tensor, Error> {
        let seq = q
            .dim(0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "SAM attend shape", e))?;
        let (query_h, query_w) = grid;
        let key_side = query_h.max(query_w);
        // `get_rel_pos`: interpolate the (2N-1, dim) table only when the grid
        // differs from SAM pretraining, then gather the per-query-row table.
        let target_rows = 2 * key_side - 1;
        let rel_h = if self.rel_pos_h.dim(0)? != target_rows {
            linear_interp_rows(&self.rel_pos_h, target_rows)?
        } else {
            self.rel_pos_h.clone()
        };
        let rel_w = if self.rel_pos_w.dim(0)? != target_rows {
            linear_interp_rows(&self.rel_pos_w, target_rows)?
        } else {
            self.rel_pos_w.clone()
        };
        let table_h = gather_rel_table(&rel_h.to_dtype(DType::F32)?, (query_h, query_h))?;
        let table_w = gather_rel_table(&rel_w.to_dtype(DType::F32)?, (query_w, query_w))?;

        let k_t = k
            .t()?
            .contiguous()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "SAM key layout", e))?;
        let mut out_rows: Vec<Tensor> = Vec::with_capacity(seq.div_ceil(REL_POS_ATTN_CHUNK));
        let mut start = 0usize;
        while start < seq {
            let len = (seq - start).min(REL_POS_ATTN_CHUNK);
            let q_chunk = q.narrow(0, start, len)?;
            let bias = self.rel_pos_bias_chunk(
                &q_chunk,
                start,
                len,
                (query_h, query_w),
                &table_h,
                &table_w,
            )?;
            let scores = q_chunk
                .matmul(&k_t)?
                .affine(scale, 0.0)?
                // The rel-pos bias is accumulated in f32; bring the scores
                // up before the add so bf16 runs stay dtype-consistent.
                .to_dtype(DType::F32)?
                .broadcast_add(&bias)?;
            let weights = candle_nn::ops::softmax_last_dim(&scores)?.to_dtype(q.dtype())?;
            out_rows.push(
                weights
                    .matmul(v)
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "SAM rel-pos attend", e))?,
            );
            start += len;
        }
        let refs: Vec<&Tensor> = out_rows.iter().collect();
        Tensor::cat(&refs, 0).map_err(|e| candle_to_ocr_inference(MODEL_NAME, "SAM attend cat", e))
    }

    /// Rel-pos bias for one contiguous query chunk of a `query_h×query_w`
    /// grid: `(len, query_h*query_w)` where
    /// `bias[i, r*query_w + c] = q_i·rel_h[ri, r, :] + q_i·rel_w[ci, c, :]`.
    fn rel_pos_bias_chunk(
        &self,
        q_chunk: &Tensor,
        chunk_start: usize,
        len: usize,
        (query_h, query_w): (usize, usize),
        table_h: &Tensor,
        table_w: &Tensor,
    ) -> Result<Tensor, Error> {
        let device = q_chunk.device();
        let q32 = q_chunk.to_dtype(DType::F32)?;
        let gather_axis = |table: &Tensor, coords: Vec<u32>| -> Result<Tensor, Error> {
            // table: (query_side, key_side, dim) gathered per query coordinate.
            let idx: Vec<u32> = (0..len).map(|i| coords[chunk_start + i]).collect();
            let selected = table.index_select(&Tensor::from_vec(idx, len, device)?, 0)?; // (len, side, dim)
            q32.unsqueeze(1)? // (len, 1, dim)
                .matmul(&selected.transpose(1, 2)?) // (len, 1, side)
                .and_then(|t| t.squeeze(1))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "rel pos dot", e))
        };
        let query_rows: Vec<u32> = (0..query_h * query_w)
            .map(|i| (i / query_w) as u32)
            .collect();
        let query_cols: Vec<u32> = (0..query_h * query_w)
            .map(|i| (i % query_w) as u32)
            .collect();
        let rel_h_q = gather_axis(table_h, query_rows)?; // (len, query_h)
        let rel_w_q = gather_axis(table_w, query_cols)?; // (len, query_w)
        rel_h_q
            .unsqueeze(2)?
            .broadcast_add(&rel_w_q.unsqueeze(1)?)?
            .reshape((len, query_h * query_w))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "rel pos bias", e))
    }
}

const REL_POS_ATTN_CHUNK: usize = 256;

struct SamBlock {
    norm1: LayerNorm,
    attention: SamAttention,
    norm2: LayerNorm,
    mlp_lin1: Linear,
    mlp_lin2: Linear,
    window_size: usize,
}

impl SamBlock {
    fn load(cfg: &SamConfig, global_attention: bool, vb: VarBuilder) -> Result<Self, Error> {
        let norm_cfg = LayerNormConfig {
            eps: 1e-6,
            ..Default::default()
        };
        let window_size = if global_attention { 0 } else { cfg.window_size };
        Ok(Self {
            norm1: layer_norm(cfg.width, norm_cfg, vb.pp("norm1"))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load SAM norm1", e))?,
            attention: SamAttention::load(cfg, global_attention, vb.clone())?,
            norm2: layer_norm(cfg.width, norm_cfg, vb.pp("norm2"))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load SAM norm2", e))?,
            mlp_lin1: linear(cfg.width, cfg.mlp_hidden(), vb.pp("mlp").pp("lin1"))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load SAM mlp lin1", e))?,
            mlp_lin2: linear(cfg.mlp_hidden(), cfg.width, vb.pp("mlp").pp("lin2"))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load SAM mlp lin2", e))?,
            window_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let (side_h, side_w, _) = x
            .dims3()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "SAM block input", e))?;
        let normed = self
            .norm1
            .forward(x)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "SAM norm1", e))?;
        let (attended, padded) = if self.window_size > 0 {
            let (windows, padded) = window_partition(&normed, side_h, side_w, self.window_size)?;
            let num_windows = windows.dim(0)?;
            let channels = windows.dim(3)?;
            let mut outs = Vec::with_capacity(num_windows);
            for w in 0..num_windows {
                let window = windows.i(w..w + 1)?.squeeze(0)?;
                outs.push(
                    self.attention
                        .forward(&window, (self.window_size, self.window_size))?,
                );
            }
            let refs: Vec<&Tensor> = outs.iter().collect();
            let stacked = Tensor::cat(&refs, 0)
                .and_then(|t| {
                    t.reshape((num_windows, self.window_size, self.window_size, channels))
                })
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "SAM windows cat", e))?;
            (stacked, Some(padded))
        } else {
            (self.attention.forward(&normed, (side_h, side_w))?, None)
        };
        let attended = match padded {
            Some((pad_h, pad_w)) => {
                window_unpartition(&attended, self.window_size, pad_h, pad_w, side_h, side_w)?
            }
            None => attended,
        };
        let x = (x + attended).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                format!("{MODEL_NAME}: SAM attention residual"),
                e,
            )
        })?;
        let normed = self
            .norm2
            .forward(&x)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "SAM norm2", e))?;
        let hidden = self
            .mlp_lin1
            .forward(&normed)
            .and_then(|h| h.gelu_erf())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "SAM mlp", e))?;
        let out = self
            .mlp_lin2
            .forward(&hidden)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "SAM mlp out", e))?;
        (x + out).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                format!("{MODEL_NAME}: SAM MLP residual"),
                e,
            )
        })
    }
}

/// Partition `(H, W, C)` into windows of `size`, zero-padding as needed.
/// Returns the windows plus the padded side lengths.
fn window_partition(
    x: &Tensor,
    side_h: usize,
    side_w: usize,
    size: usize,
) -> Result<(Tensor, (usize, usize)), Error> {
    let pad_h = (size - side_h % size) % size;
    let pad_w = (size - side_w % size) % size;
    let x = if pad_h > 0 {
        x.pad_with_zeros(0, 0, pad_h)
    } else {
        Ok(x.clone())
    }
    .and_then(|x| {
        if pad_w > 0 {
            x.pad_with_zeros(1, 0, pad_w)
        } else {
            Ok(x)
        }
    })
    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "SAM window pad", e))?;
    let padded_h = side_h + pad_h;
    let padded_w = side_w + pad_w;
    let channels = x.dim(2)?;
    let windows = x
        .reshape((padded_h / size, size, padded_w / size, size, channels))
        .and_then(|x| x.permute((0, 2, 1, 3, 4)))
        .and_then(|x| x.reshape(((padded_h / size) * (padded_w / size), size, size, x.dim(4)?)))
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "SAM window partition", e))?;
    Ok((windows, (padded_h, padded_w)))
}

#[allow(clippy::too_many_arguments)]
fn window_unpartition(
    windows: &Tensor,
    size: usize,
    padded_h: usize,
    padded_w: usize,
    side_h: usize,
    side_w: usize,
) -> Result<Tensor, Error> {
    let channels = windows.dim(3)?;
    let x = windows
        .reshape((padded_h / size, padded_w / size, size, size, channels))
        .and_then(|x| x.permute((0, 2, 1, 3, 4)))
        .and_then(|x| x.reshape((padded_h, padded_w, channels)))
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "SAM window unpartition", e))?;
    if padded_h != side_h || padded_w != side_w {
        x.i((0..side_h, 0..side_w))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "SAM window crop", e))
    } else {
        Ok(x)
    }
}

/// Bias-free SAM convolution (`nn.Conv2d(..., bias=False)`).
fn sam_conv(
    kernel: usize,
    in_channels: usize,
    out_channels: usize,
    padding: usize,
    stride: usize,
    vb: VarBuilder,
    context: &str,
) -> Result<Conv2d, Error> {
    let weight = vb
        .get((out_channels, in_channels, kernel, kernel), "weight")
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, context, e))?;
    Ok(Conv2d::new(
        weight,
        None,
        Conv2dConfig {
            padding,
            stride,
            ..Default::default()
        },
    ))
}

/// SAM's channel LayerNorm (`LayerNorm2d`): statistics over dim 1 of
/// `(B, C, H, W)`.
fn layer_norm_2d(x: &Tensor, weight: &Tensor, bias: &Tensor, eps: f64) -> Result<Tensor, Error> {
    let mean = x.mean_keepdim(1)?;
    let centered = x.broadcast_sub(&mean)?;
    let var = centered.sqr()?.mean_keepdim(1)?;
    let normalized = centered.broadcast_div(&(var + eps)?.sqrt()?)?;
    normalized
        .broadcast_mul(&weight.reshape((1, weight.elem_count(), 1, 1))?)?
        .broadcast_add(&bias.reshape((1, bias.elem_count(), 1, 1))?)
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "SAM layer norm 2d", e))
}

pub struct SamVisionModel {
    patch_embed_weight: Tensor,
    patch_embed_bias: Tensor,
    pos_embed: Tensor,
    blocks: Vec<SamBlock>,
    neck_conv1: Conv2d,
    neck_norm1: (Tensor, Tensor),
    neck_conv2: Conv2d,
    neck_norm2: (Tensor, Tensor),
    net_2: Conv2d,
    net_3: Conv2d,
    patch_size: usize,
    pretrained_grid_side: usize,
}

impl SamVisionModel {
    pub fn load(cfg: &SamConfig, vb: VarBuilder) -> Result<Self, Error> {
        let patch_dim = 3 * cfg.patch_size * cfg.patch_size;
        let patch_embed_weight = vb
            .pp("patch_embed")
            .pp("proj")
            .get((cfg.width, 3, cfg.patch_size, cfg.patch_size), "weight")
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load SAM patch weight", e))?
            .reshape((cfg.width, patch_dim))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "reshape SAM patch weight", e))?;
        let patch_embed_bias = vb
            .pp("patch_embed")
            .pp("proj")
            .get(cfg.width, "bias")
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load SAM patch bias", e))?;
        let pos_embed = vb
            .get(
                (
                    1,
                    cfg.pretrained_grid_side,
                    cfg.pretrained_grid_side,
                    cfg.width,
                ),
                "pos_embed",
            )
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load SAM pos_embed", e))?;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(SamBlock::load(
                cfg,
                cfg.global_attn_indexes.contains(&i),
                vb.pp("blocks").pp(i),
            )?);
        }
        let neck_conv1 = sam_conv(
            1,
            cfg.width,
            256,
            0,
            1,
            vb.pp("neck").pp(0),
            "SAM neck conv1",
        )?;
        let neck_norm1 = (
            vb.get(256, "neck.1.weight")
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load SAM neck norm1 w", e))?,
            vb.get(256, "neck.1.bias")
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load SAM neck norm1 b", e))?,
        );
        let neck_conv2 = sam_conv(3, 256, 256, 1, 1, vb.pp("neck").pp(2), "SAM neck conv2")?;
        let neck_norm2 = (
            vb.get(256, "neck.3.weight")
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load SAM neck norm2 w", e))?,
            vb.get(256, "neck.3.bias")
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load SAM neck norm2 b", e))?,
        );
        let net_2 = sam_conv(3, 256, 512, 1, 2, vb.pp("net_2"), "SAM net_2")?;
        let net_3 = sam_conv(3, 512, cfg.out_channels, 1, 2, vb.pp("net_3"), "SAM net_3")?;
        Ok(Self {
            patch_embed_weight,
            patch_embed_bias,
            pos_embed,
            blocks,
            neck_conv1,
            neck_norm1,
            neck_conv2,
            neck_norm2,
            net_2,
            net_3,
            patch_size: cfg.patch_size,
            pretrained_grid_side: cfg.pretrained_grid_side,
        })
    }

    /// `(1, 3, H, W)` -> `(1, out_channels, H/4/patch, W/4/patch)` SAM
    /// features, the `ImageEncoderViT` output layout. The windowed blocks
    /// operate on one image at a time.
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor, Error> {
        let (batch, _, height, width) = pixel_values
            .dims4()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "SAM input shape", e))?;
        if batch != 1 {
            return Err(Error::InvalidInput {
                message: format!(
                    "{MODEL_NAME} SAM tower processes one image at a time, got batch {batch}"
                ),
            });
        }
        let grid_h = height / self.patch_size;
        let grid_w = width / self.patch_size;
        let patches = conv_patchify(pixel_values, self.patch_size)?;
        let width_out = self.patch_embed_weight.dim(0)?;
        let x = patches
            .flatten(0, 1)?
            .matmul(&self.patch_embed_weight.t()?.to_dtype(patches.dtype())?)?
            .broadcast_add(&self.patch_embed_bias)?;
        let x = x.reshape((grid_h, grid_w, width_out))?;

        // `get_abs_pos_sam`: bicubic-interpolate the pretrained grid (f32).
        let pos_embed =
            if self.pretrained_grid_side == grid_h && self.pretrained_grid_side == grid_w {
                self.pos_embed.squeeze(0)?
            } else {
                bicubic_antialias_resize(&self.pos_embed.permute((0, 3, 1, 2))?, grid_h, grid_w)?
                    .permute((0, 2, 3, 1))?
                    .squeeze(0)?
            }
            .to_dtype(x.dtype())?;
        let x = x.broadcast_add(&pos_embed)?;

        let mut x = x;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }

        // Neck + downsampling convs: (B, C, H, W) layout.
        let x = x.permute((2, 0, 1))?.unsqueeze(0)?.contiguous()?;
        let x = self.neck_conv1.forward(&x)?;
        let x = layer_norm_2d(&x, &self.neck_norm1.0, &self.neck_norm1.1, 1e-6)?;
        let x = self.neck_conv2.forward(&x)?;
        let x = layer_norm_2d(&x, &self.neck_norm2.0, &self.neck_norm2.1, 1e-6)?;
        let x = self.net_2.forward(&x)?;
        self.net_3
            .forward(&x)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "SAM downsampler", e))
    }
}

/// NCHW image -> `(B, (H/p)*(W/p), 3*p*p)` raster-ordered patches (the layout
/// a stride-`p` Conv2d sees before its reshape).
fn conv_patchify(pixel_values: &Tensor, p: usize) -> Result<Tensor, Error> {
    let (batch, channels, height, width) = pixel_values
        .dims4()
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "patchify input shape", e))?;
    let grid_h = height / p;
    let grid_w = width / p;
    pixel_values
        .reshape((batch, channels, grid_h, p, grid_w, p))
        .and_then(|x| x.permute((0, 2, 4, 1, 3, 5)))
        .and_then(|x| x.reshape((batch, grid_h * grid_w, channels * p * p)))
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "patchify", e))
}

// ---------------------------------------------------------------------------
// CLIP-L
// ---------------------------------------------------------------------------

struct ClipBlock {
    norm1: LayerNorm,
    qkv: Linear,
    out_proj: Linear,
    norm2: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    heads: usize,
    head_dim: usize,
}

impl ClipBlock {
    fn load(cfg: &ClipConfig, vb: VarBuilder) -> Result<Self, Error> {
        let norm_cfg = LayerNormConfig {
            eps: cfg.layer_norm_eps,
            ..Default::default()
        };
        Ok(Self {
            norm1: layer_norm(cfg.width, norm_cfg, vb.pp("layer_norm1"))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load CLIP norm1", e))?,
            qkv: linear(cfg.width, cfg.width * 3, vb.pp("self_attn").pp("qkv_proj"))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load CLIP qkv_proj", e))?,
            out_proj: linear(cfg.width, cfg.width, vb.pp("self_attn").pp("out_proj"))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load CLIP out_proj", e))?,
            norm2: layer_norm(cfg.width, norm_cfg, vb.pp("layer_norm2"))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load CLIP norm2", e))?,
            fc1: linear(cfg.width, cfg.ffn_hidden_size, vb.pp("mlp").pp("fc1"))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load CLIP fc1", e))?,
            fc2: linear(cfg.ffn_hidden_size, cfg.width, vb.pp("mlp").pp("fc2"))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load CLIP fc2", e))?,
            heads: cfg.heads,
            head_dim: cfg.width / cfg.heads,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let (batch, seq, width) = x
            .dims3()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "CLIP block input", e))?;
        let normed = self
            .norm1
            .forward(x)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "CLIP norm1", e))?;
        let qkv = self
            .qkv
            .forward(&normed)
            .and_then(|qkv| qkv.reshape((batch, seq, 3, self.heads, self.head_dim)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "CLIP qkv", e))?;
        let q = qkv.i((.., .., 0, .., ..))?;
        let k = qkv.i((.., .., 1, .., ..))?;
        let v = qkv.i((.., .., 2, .., ..))?;
        let attn = clip_attention(&q, &k, &v)?;
        let attn = attn
            .reshape((batch, seq, width))
            .and_then(|a| self.out_proj.forward(&a))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "CLIP out_proj", e))?;
        let x = (x + attn).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                format!("{MODEL_NAME}: CLIP attention residual"),
                e,
            )
        })?;
        let normed = self
            .norm2
            .forward(&x)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "CLIP norm2", e))?;
        let hidden = self
            .fc1
            .forward(&normed)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "CLIP fc1", e))?;
        let hidden = quick_gelu(&hidden)?;
        let out = self
            .fc2
            .forward(&hidden)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "CLIP fc2", e))?;
        (x + out).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                format!("{MODEL_NAME}: CLIP MLP residual"),
                e,
            )
        })
    }
}

/// Non-causal batched attention over `(B, seq, heads, head_dim)` slices.
fn clip_attention(q: &Tensor, k: &Tensor, v: &Tensor) -> Result<Tensor, Error> {
    let (batch, seq, heads, head_dim) = q
        .dims4()
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "CLIP attention shape", e))?;
    let q4 = q.permute((0, 2, 1, 3))?.contiguous()?;
    let k4 = k.permute((0, 2, 1, 3))?.contiguous()?;
    let v4 = v.permute((0, 2, 1, 3))?.contiguous()?;
    let out =
        scaled_dot_product_attention(&q4, &k4, &v4, None, 1.0 / (head_dim as f64).sqrt(), false)?;
    out.permute((0, 2, 1, 3))?
        .reshape((batch, seq, heads * head_dim))
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "CLIP attention", e))
}

pub struct ClipVisionModel {
    class_embedding: Tensor,
    position_embedding: Tensor,
    pre_layer_norm: LayerNorm,
    blocks: Vec<ClipBlock>,
    pretrained_grid_side: usize,
}

impl ClipVisionModel {
    pub fn load(cfg: &ClipConfig, vb: VarBuilder) -> Result<Self, Error> {
        let class_embedding = vb
            .pp("embeddings")
            .get(cfg.width, "class_embedding")
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load CLIP class embedding", e))?;
        let position_embedding = vb
            .pp("embeddings")
            .get(
                (cfg.pretrained_positions(), cfg.width),
                "position_embedding.weight",
            )
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load CLIP position embedding", e))?;
        let norm_cfg = LayerNormConfig {
            eps: cfg.layer_norm_eps,
            ..Default::default()
        };
        let pre_layer_norm = layer_norm(cfg.width, norm_cfg, vb.pp("pre_layrnorm"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load CLIP pre layernorm", e))?;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            blocks.push(ClipBlock::load(
                cfg,
                vb.pp("transformer").pp("layers").pp(i),
            )?);
        }
        Ok(Self {
            class_embedding,
            position_embedding,
            pre_layer_norm,
            blocks,
            pretrained_grid_side: cfg.pretrained_grid_side,
        })
    }

    /// `patch_embeds`: SAM features on the shared grid, `(B, C, g, g)`.
    /// Returns `(B, g² + 1, width)`.
    pub fn forward(&self, patch_embeds: &Tensor) -> Result<Tensor, Error> {
        let (batch, _, grid_h, grid_w) = patch_embeds
            .dims4()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "CLIP input shape", e))?;
        let seq = grid_h * grid_w;
        let patches = patch_embeds
            .reshape((batch, patch_embeds.dim(1)?, seq))?
            .transpose(1, 2)?
            .contiguous()?;
        let class = self
            .class_embedding
            .broadcast_as((batch, 1, self.class_embedding.dim(0)?))?;
        let embeddings = Tensor::cat(&[&class, &patches], 1)?;

        // `get_abs_pos`: interpolate the grid part, keep the class row.
        let positions =
            if grid_h == self.pretrained_grid_side && grid_w == self.pretrained_grid_side {
                self.position_embedding.unsqueeze(0)?
            } else {
                let (rows, dim) = self.position_embedding.dims2()?;
                let cls = self.position_embedding.i((0..1, ..))?;
                let grid = self.position_embedding.i((1..rows, ..))?;
                let side = (rows - 1).isqrt();
                let resized = bicubic_antialias_resize(
                    &grid.reshape((1, side, side, dim))?.permute((0, 3, 1, 2))?,
                    grid_h,
                    grid_w,
                )?;
                let resized = resized
                    .permute((0, 2, 3, 1))?
                    .reshape((seq, dim))?
                    // The bicubic resampler works in f32; match the class
                    // row before concatenating.
                    .to_dtype(self.position_embedding.dtype())?;
                Tensor::cat(&[&cls, &resized], 0)?
                    .unsqueeze(0)?
                    .contiguous()?
            };
        let x = embeddings.broadcast_add(&positions.to_dtype(embeddings.dtype())?)?;
        let mut x = self
            .pre_layer_norm
            .forward(&x)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "CLIP pre layernorm", e))?;
        for block in &self.blocks {
            x = block.forward(&x)?;
        }
        Ok(x)
    }
}

impl ClipConfig {
    /// Rows of the shipped position-embedding table: `(224/14)² + 1`.
    fn pretrained_positions(&self) -> usize {
        self.pretrained_grid_side * self.pretrained_grid_side + 1
    }
}

// ---------------------------------------------------------------------------
// Combined encoder
// ---------------------------------------------------------------------------

/// SAM ViT-B + CLIP-L + linear projector (`DeepseekOCRModel`'s vision path).
pub struct DeepEncoder {
    sam: SamVisionModel,
    clip: ClipVisionModel,
    projector: Linear,
}

impl DeepEncoder {
    pub fn load(
        sam_cfg: &SamConfig,
        clip_cfg: &ClipConfig,
        n_embed: usize,
        vb: VarBuilder,
    ) -> Result<Self, Error> {
        let sam = SamVisionModel::load(sam_cfg, vb.pp("sam_model"))?;
        let clip = ClipVisionModel::load(clip_cfg, vb.pp("vision_model"))?;
        let input_dim = clip_cfg.width + sam_cfg.out_channels;
        let projector = linear(input_dim, n_embed, vb.pp("projector").pp("layers"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load projector", e))?;
        Ok(Self {
            sam,
            clip,
            projector,
        })
    }

    /// `(B, 3, H, W)` -> `(B, (H/4/p) * (W/4/p), n_embed)` visual features.
    /// The windowed SAM tower runs one image at a time (the reference batches
    /// same-size views instead, which is numerically equivalent).
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor, Error> {
        let batch = pixel_values
            .dims4()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "encoder input shape", e))?
            .0;
        let mut outputs = Vec::with_capacity(batch);
        for index in 0..batch {
            let image = pixel_values
                .i(index..index + 1)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "slice encoder input", e))?;
            outputs.push(self.forward_one(&image)?);
        }
        let refs: Vec<&Tensor> = outputs.iter().collect();
        Tensor::cat(&refs, 0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "encoder batch cat", e))
    }

    /// One image: `(1, 3, H, W)` -> `(1, (H/4/p) * (W/4/p), n_embed)`.
    fn forward_one(&self, pixel_values: &Tensor) -> Result<Tensor, Error> {
        let sam_e = self.sam.forward(pixel_values)?;
        let vision_e = self.clip.forward(&sam_e)?;
        // Drop the class token, flatten SAM channels-last, concat [CLIP | SAM]
        // and project.
        let vision_e = vision_e.i((.., 1.., ..))?;
        let sam_e = sam_e.flatten(2, 3)?.transpose(1, 2)?.contiguous()?;
        let concat = Tensor::cat(&[&vision_e, &sam_e], 2)?;
        self.projector
            .forward(&concat)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "vision projector", e))
    }
}
