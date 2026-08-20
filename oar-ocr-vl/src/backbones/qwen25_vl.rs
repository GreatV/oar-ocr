//! Qwen2.5-VL vision tower for NaviDC-OCR.
//!
//! Structured after `mineru::vision` (the in-repo Qwen2-VL tower: same
//! conv3d patch embed, 2D rotary embedding, and patch merger), with the
//! Qwen2.5 differences: RMSNorm block norms (not LayerNorm), a biased SwiGLU
//! MLP (not fc1/fc2), and **windowed attention** — tokens are permuted into
//! `window_size` windows and every block except `fullatt_block_indexes`
//! attends within its window only (see `modeling_naviocr.py` lines 337-465).
//!
//! Each image is processed independently, mirroring the per-image
//! `cu_seqlens` segmentation of the reference implementation.

use crate::attention::{
    VISION_CHUNKED_ATTN_CHUNK_SIZE, VISION_CHUNKED_ATTN_SEQ_THRESHOLD, chunked_vision_attention,
    on_compute_device, scaled_dot_product_attention,
};
use crate::error::Error;
use crate::runtime::errors::{candle_to_ocr_inference, candle_to_ocr_processing};
use crate::runtime::tensor::{rotate_half, vision_inv_freq};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear, rms_norm};
use serde::Deserialize;

fn default_hidden_act() -> String {
    "silu".to_string()
}

/// Shared Qwen2.5-VL windowed vision-tower configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct NaviDcVisionConfig {
    pub depth: usize,
    pub hidden_size: usize,
    pub out_hidden_size: usize,
    pub num_heads: usize,
    pub intermediate_size: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    #[serde(default, rename = "in_chans")]
    in_chans: Option<usize>,
    #[serde(default, rename = "in_channels")]
    in_channels_alias: Option<usize>,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    #[serde(default)]
    pub spatial_patch_size: Option<usize>,
    pub temporal_patch_size: usize,
    pub window_size: usize,
    #[serde(default)]
    pub fullatt_block_indexes: Vec<usize>,
}

impl NaviDcVisionConfig {
    pub fn in_channels(&self) -> usize {
        self.in_chans.or(self.in_channels_alias).unwrap_or(3)
    }

    pub fn head_dim(&self) -> Result<usize, Error> {
        if !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(Error::config(format!(
                "Qwen2.5-VL vision hidden_size {} not divisible by num_heads {}",
                self.hidden_size, self.num_heads
            )));
        }
        Ok(self.hidden_size / self.num_heads)
    }
}

fn apply_rotary_pos_emb_vision(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor), Error> {
    let orig_q_dtype = q.dtype();
    let orig_k_dtype = k.dtype();
    let q = q.to_dtype(DType::F32).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            "NaviDC-OCR: vision q cast failed",
            e,
        )
    })?;
    let k = k.to_dtype(DType::F32).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            "NaviDC-OCR: vision k cast failed",
            e,
        )
    })?;
    let cos = cos
        .unsqueeze(1)
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: vision cos unsqueeze failed",
                e,
            )
        })?
        .to_dtype(DType::F32)
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: vision cos cast failed",
                e,
            )
        })?;
    let sin = sin
        .unsqueeze(1)
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: vision sin unsqueeze failed",
                e,
            )
        })?
        .to_dtype(DType::F32)
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: vision sin cast failed",
                e,
            )
        })?;

    let q_rot = rotate_half(&q)?;
    let q_embed = q
        .broadcast_mul(&cos)
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: vision q*cos failed",
                e,
            )
        })?
        .broadcast_add(&q_rot.broadcast_mul(&sin).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: vision rotate_half(q)*sin failed",
                e,
            )
        })?)
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: vision q rope add failed",
                e,
            )
        })?;

    let k_rot = rotate_half(&k)?;
    let k_embed = k
        .broadcast_mul(&cos)
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: vision k*cos failed",
                e,
            )
        })?
        .broadcast_add(&k_rot.broadcast_mul(&sin).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: vision rotate_half(k)*sin failed",
                e,
            )
        })?)
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: vision k rope add failed",
                e,
            )
        })?;

    let q_embed = q_embed.to_dtype(orig_q_dtype).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            "NaviDC-OCR: vision q_embed cast back failed",
            e,
        )
    })?;
    let k_embed = k_embed.to_dtype(orig_k_dtype).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            "NaviDC-OCR: vision k_embed cast back failed",
            e,
        )
    })?;

    Ok((q_embed, k_embed))
}

#[derive(Debug, Clone)]
struct VisionRotaryEmbedding {
    inv_freq: Tensor,
    dim: usize,
}

impl VisionRotaryEmbedding {
    fn new(dim: usize, theta: f64, device: &Device) -> Result<Self, Error> {
        let inv_freq = vision_inv_freq(dim, theta, "NaviDC-OCR", device)?;
        Ok(Self { inv_freq, dim })
    }

    fn forward(&self, seqlen: usize, device: &Device) -> Result<Tensor, Error> {
        // Use on_compute_device to handle Metal's lack of support for arange
        on_compute_device(device, |compute_device| {
            let seq = Tensor::arange(0u32, seqlen as u32, compute_device)?.to_dtype(DType::F32)?;
            let inv = self
                .inv_freq
                .to_device(compute_device)?
                .to_dtype(DType::F32)?;
            seq.unsqueeze(1)?.matmul(&inv.unsqueeze(0)?)
        })
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: vision rope forward failed",
                e,
            )
        })
    }

    fn dim(&self) -> usize {
        self.dim
    }
}

#[derive(Debug, Clone)]
struct PatchEmbed {
    weight: Tensor,
}

impl PatchEmbed {
    fn load(cfg: &NaviDcVisionConfig, vb: VarBuilder) -> Result<Self, Error> {
        let patch_dim =
            cfg.in_channels() * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
        let weight = match vb.get((cfg.hidden_size, patch_dim), "patch_embed.proj.weight") {
            Ok(weight) => weight,
            Err(_) => {
                let weight = vb
                    .get(
                        (
                            cfg.hidden_size,
                            cfg.in_channels(),
                            cfg.temporal_patch_size,
                            cfg.patch_size,
                            cfg.patch_size,
                        ),
                        "patch_embed.proj.weight",
                    )
                    .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load patch_embed", e))?;
                weight
                    .reshape((cfg.hidden_size, patch_dim))
                    .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "reshape patch_embed", e))?
            }
        };
        Ok(Self { weight })
    }

    fn forward(&self, patches: &Tensor) -> Result<Tensor, Error> {
        let weight_t = self.weight.transpose(0, 1).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: patch_embed weight transpose failed",
                e,
            )
        })?;
        let patches = patches.to_dtype(self.weight.dtype()).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: patch_embed input cast failed",
                e,
            )
        })?;
        patches
            .matmul(&weight_t)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "patch_embed matmul", e))
    }
}

/// Window partitioning for one batch of images: the permutation into window
/// order plus the cumulative segment lengths the windowed blocks attend
/// within. Ported from `get_window_index` in `modeling_naviocr.py`
/// (lines 366-405) together with the `torch.unique_consecutive` on
/// `cu_window_seqlens` (line 426).
#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct WindowPlan {
    /// `window_index[out_unit] = source_unit`: merge-unit permutation into
    /// window-major order.
    pub(crate) window_index: Vec<usize>,
    /// Cumulative window segment lengths in **patch** units (each entry is
    /// `4 * merged_tokens` for `spatial_merge_size = 2`); consecutive pairs
    /// bound the segments a windowed block attends over.
    pub(crate) cu_window_seqlens: Vec<usize>,
    /// Inverse of `window_index`: `reverse_index[i]` is the window-order slot
    /// holding source unit `i`.
    pub(crate) reverse_index: Vec<usize>,
}

pub(crate) fn window_plan(
    grid_thw: &[(usize, usize, usize)],
    spatial_merge_size: usize,
    patch_size: usize,
    window_size: usize,
) -> Result<WindowPlan, Error> {
    let merge_unit = spatial_merge_size
        .checked_mul(spatial_merge_size)
        .filter(|&unit| unit > 0)
        .ok_or_else(|| Error::Config {
            message: "NaviDC-OCR: spatial_merge_size must be >= 1".to_string(),
        })?;
    let window_tokens = spatial_merge_size
        .checked_mul(patch_size)
        .filter(|&t| t > 0)
        .ok_or_else(|| Error::Config {
            message: "NaviDC-OCR: spatial_merge_size * patch_size overflow".to_string(),
        })?;
    if window_size == 0 || !window_size.is_multiple_of(window_tokens) {
        return Err(Error::Config {
            message: format!(
                "NaviDC-OCR: window_size ({window_size}) must be a positive multiple of spatial_merge_size * patch_size ({window_tokens})"
            ),
        });
    }
    let window_side = window_size / window_tokens;

    let total_units: usize = grid_thw
        .iter()
        .map(|&(t, h, w)| (h / spatial_merge_size) * (w / spatial_merge_size) * t)
        .sum();

    let mut window_index: Vec<usize> = Vec::with_capacity(total_units);
    let mut cu_window_seqlens: Vec<usize> = vec![0];
    let mut base = 0usize;

    for &(t, h, w) in grid_thw {
        let llm_h = h / spatial_merge_size;
        let llm_w = w / spatial_merge_size;
        // The reference pads each dimension up to a whole number of windows;
        // a dimension that already divides evenly gets a full extra (empty)
        // window row/column, which unique_consecutive later drops. Emitting a
        // boundary only for non-empty windows reproduces that exactly.
        let pad_h = window_side - llm_h % window_side;
        let pad_w = window_side - llm_w % window_side;
        let num_windows_h = (llm_h + pad_h) / window_side;
        let num_windows_w = (llm_w + pad_w) / window_side;

        for tt in 0..t {
            for wh in 0..num_windows_h {
                for ww in 0..num_windows_w {
                    let mut count = 0usize;
                    for ih in 0..window_side {
                        for iw in 0..window_side {
                            let row = wh * window_side + ih;
                            let col = ww * window_side + iw;
                            if row < llm_h && col < llm_w {
                                window_index.push(base + tt * llm_h * llm_w + row * llm_w + col);
                                count += 1;
                            }
                        }
                    }
                    if count > 0 {
                        let last = *cu_window_seqlens.last().expect("starts with 0");
                        cu_window_seqlens.push(last + count * merge_unit);
                    }
                }
            }
        }
        base += t * llm_h * llm_w;
    }

    let mut reverse_index = vec![0usize; window_index.len()];
    for (slot, &source) in window_index.iter().enumerate() {
        reverse_index[source] = slot;
    }

    Ok(WindowPlan {
        window_index,
        cu_window_seqlens,
        reverse_index,
    })
}

/// (h, w) rotary position ids for every patch of one image, in the
/// merge-grouped raster order produced by the patchifier (merge block rows:
/// `hb, wb, h_inner, w_inner`). All patches of a merge unit do NOT share a
/// position: the id is the raw patch coordinate.
fn vision_position_ids(t: usize, h: usize, w: usize, merge: usize) -> (Vec<i64>, Vec<i64>) {
    let mut hpos = Vec::with_capacity(t * h * w);
    let mut wpos = Vec::with_capacity(t * h * w);
    for _ in 0..t {
        for hb in 0..(h / merge) {
            for wb in 0..(w / merge) {
                for h_inner in 0..merge {
                    for w_inner in 0..merge {
                        hpos.push((hb * merge + h_inner) as i64);
                        wpos.push((wb * merge + w_inner) as i64);
                    }
                }
            }
        }
    }
    (hpos, wpos)
}

/// Build `(cos, sin)` of shape `(patches, head_dim)` from per-patch `(h, w)`
/// position ids, mirroring `rot_pos_emb` + `emb = cat(rot, rot)` in the
/// reference: gather the frequency table at each coordinate, concatenate the
/// axes, then double for rotate_half.
fn gather_vision_pos_emb(
    rotary_full: &Tensor,
    freq_dim: usize,
    hpos: &[i64],
    wpos: &[i64],
    device: &Device,
) -> Result<(Tensor, Tensor), Error> {
    let num_patches = hpos.len();
    let make_index = |values: &[i64]| -> Result<Tensor, Error> {
        let index =
            Tensor::from_vec(values.to_vec(), (num_patches, 1usize), device).map_err(|e| {
                candle_to_ocr_processing(
                    crate::error::ProcessingStage::TensorOperation,
                    "NaviDC-OCR: vision pos tensor failed",
                    e,
                )
            })?;
        index
            .broadcast_as((num_patches, freq_dim))
            .and_then(|index| index.contiguous())
            .map_err(|e| {
                candle_to_ocr_processing(
                    crate::error::ProcessingStage::TensorOperation,
                    "NaviDC-OCR: vision pos broadcast failed",
                    e,
                )
            })
    };
    let h_index = make_index(hpos)?;
    let w_index = make_index(wpos)?;

    let freqs_h = rotary_full
        .gather(&h_index, 0)
        .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision gather h", e))?;
    let freqs_w = rotary_full
        .gather(&w_index, 0)
        .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision gather w", e))?;
    let rotary = Tensor::cat(&[&freqs_h, &freqs_w], candle_core::D::Minus1)
        .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision pos cat", e))?;
    let emb = Tensor::cat(&[&rotary, &rotary], candle_core::D::Minus1)
        .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision emb cat", e))?;
    let cos = emb
        .cos()
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: vision cos failed",
                e,
            )
        })?
        .contiguous()
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: vision cos contiguous failed",
                e,
            )
        })?;
    let sin = emb
        .sin()
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: vision sin failed",
                e,
            )
        })?
        .contiguous()
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: vision sin contiguous failed",
                e,
            )
        })?;
    Ok((cos, sin))
}

/// Non-causal attention over one contiguous segment. Segments longer than
/// [`VISION_CHUNKED_ATTN_SEQ_THRESHOLD`] fall back to query chunking to cap
/// the attention-score memory (full-attention blocks on large pages).
fn attend_segment(q: &Tensor, k: &Tensor, v: &Tensor, scale: f64) -> Result<Tensor, Error> {
    let seq_len = q
        .dim(2)
        .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision segment dim", e))?;
    if seq_len > VISION_CHUNKED_ATTN_SEQ_THRESHOLD {
        chunked_vision_attention(q, k, v, scale, VISION_CHUNKED_ATTN_CHUNK_SIZE)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision chunked attention", e))
    } else {
        scaled_dot_product_attention(q, k, v, None, scale, false)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision segment attention", e))
    }
}

#[derive(Debug, Clone)]
struct NaviDcVisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl NaviDcVisionAttention {
    fn load(cfg: &NaviDcVisionConfig, vb: VarBuilder) -> Result<Self, Error> {
        let qkv = linear(cfg.hidden_size, cfg.hidden_size * 3, vb.pp("attn.qkv"))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load vision qkv", e))?;
        let proj = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("attn.proj"))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load vision proj", e))?;
        let head_dim = cfg.head_dim()?;
        Ok(Self {
            qkv,
            proj,
            num_heads: cfg.num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    /// Attend over `segments` (start, len) of the window-ordered sequence.
    /// Windowed blocks pass the per-window segments; full-attention blocks
    /// pass a single whole-image segment.
    fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        segments: &[(usize, usize)],
    ) -> Result<Tensor, Error> {
        let seq_len = hidden_states
            .dim(0)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision hidden_states dim", e))?;
        let qkv = self
            .qkv
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision qkv", e))?;
        let qkv = qkv
            .reshape((seq_len, 3, self.num_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision qkv reshape", e))?;
        let q = qkv
            .i((.., 0, .., ..))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision q slice", e))?;
        let k = qkv
            .i((.., 1, .., ..))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision k slice", e))?;
        let v = qkv
            .i((.., 2, .., ..))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision v slice", e))?;

        let (q, k) = apply_rotary_pos_emb_vision(&q, &k, cos, sin)?;

        let q = q
            .transpose(0, 1)
            .and_then(|q| q.unsqueeze(0))
            .and_then(|q| q.contiguous())
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision q layout", e))?;
        let k = k
            .transpose(0, 1)
            .and_then(|k| k.unsqueeze(0))
            .and_then(|k| k.contiguous())
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision k layout", e))?;
        let v = v
            .transpose(0, 1)
            .and_then(|v| v.unsqueeze(0))
            .and_then(|v| v.contiguous())
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision v layout", e))?;

        let mut outputs = Vec::with_capacity(segments.len());
        for &(start, len) in segments {
            if len == 0 {
                continue;
            }
            let q_seg = q
                .narrow(2, start, len)
                .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision segment narrow", e))?;
            let k_seg = k
                .narrow(2, start, len)
                .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision segment narrow", e))?;
            let v_seg = v
                .narrow(2, start, len)
                .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision segment narrow", e))?;
            outputs.push(attend_segment(&q_seg, &k_seg, &v_seg, self.scale)?);
        }
        let attn = if outputs.len() == 1 {
            outputs.pop().expect("one output was just pushed")
        } else {
            let refs: Vec<&Tensor> = outputs.iter().collect();
            Tensor::cat(&refs, 2)
                .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision segments cat", e))?
        };

        let attn = attn
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision attn transpose", e))?
            .reshape((seq_len, self.num_heads * self.head_dim))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision attn reshape", e))?;
        self.proj
            .forward(&attn)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision proj", e))
    }
}

#[derive(Debug, Clone)]
struct NaviDcVisionMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl NaviDcVisionMlp {
    fn load(cfg: &NaviDcVisionConfig, vb: VarBuilder) -> Result<Self, Error> {
        let gate_proj = linear(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("mlp.gate_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load vision gate_proj", e))?;
        let up_proj = linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("mlp.up_proj"))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load vision up_proj", e))?;
        let down_proj = linear(
            cfg.intermediate_size,
            cfg.hidden_size,
            vb.pp("mlp.down_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load vision down_proj", e))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, Error> {
        let gate = self
            .gate_proj
            .forward(xs)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision gate_proj", e))?;
        let gate = candle_nn::ops::silu(&gate)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision silu", e))?;
        let up = self
            .up_proj
            .forward(xs)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision up_proj", e))?;
        let prod = (&gate * &up)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision gate*up", e))?;
        self.down_proj
            .forward(&prod)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision down_proj", e))
    }
}

#[derive(Debug, Clone)]
struct NaviDcVisionBlock {
    norm1: candle_nn::RmsNorm,
    norm2: candle_nn::RmsNorm,
    attn: NaviDcVisionAttention,
    mlp: NaviDcVisionMlp,
}

impl NaviDcVisionBlock {
    fn load(cfg: &NaviDcVisionConfig, vb: VarBuilder) -> Result<Self, Error> {
        // The reference hardcodes eps=1e-6 for the vision RMSNorms
        // (`Qwen2RMSNorm(config.hidden_size, eps=1e-6)`).
        let norm1 = rms_norm(cfg.hidden_size, 1e-6, vb.pp("norm1"))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load vision norm1", e))?;
        let norm2 = rms_norm(cfg.hidden_size, 1e-6, vb.pp("norm2"))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load vision norm2", e))?;
        let attn = NaviDcVisionAttention::load(cfg, vb.clone())?;
        let mlp = NaviDcVisionMlp::load(cfg, vb)?;
        Ok(Self {
            norm1,
            norm2,
            attn,
            mlp,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        segments: &[(usize, usize)],
    ) -> Result<Tensor, Error> {
        let normed = self
            .norm1
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision norm1 forward", e))?;
        let attn_out = self.attn.forward(&normed, cos, sin, segments)?;
        let hidden_states = (hidden_states + attn_out).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: vision attn residual failed",
                e,
            )
        })?;

        let normed = self
            .norm2
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision norm2 forward", e))?;
        let mlp_out = self.mlp.forward(&normed)?;
        (hidden_states + mlp_out).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: vision mlp residual failed",
                e,
            )
        })
    }
}

#[derive(Debug, Clone)]
struct NaviDcPatchMerger {
    ln_q: candle_nn::RmsNorm,
    mlp1: Linear,
    mlp2: Linear,
    merge_size: usize,
    hidden_size: usize,
}

impl NaviDcPatchMerger {
    fn load(cfg: &NaviDcVisionConfig, vb: VarBuilder) -> Result<Self, Error> {
        let ln_q = rms_norm(cfg.hidden_size, 1e-6, vb.pp("merger.ln_q"))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load merger ln_q", e))?;
        let hidden_size = cfg.hidden_size * cfg.spatial_merge_size * cfg.spatial_merge_size;
        let mlp1 = linear(hidden_size, hidden_size, vb.pp("merger.mlp.0"))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load merger mlp1", e))?;
        let mlp2 = linear(hidden_size, cfg.out_hidden_size, vb.pp("merger.mlp.2"))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "load merger mlp2", e))?;
        Ok(Self {
            ln_q,
            mlp1,
            mlp2,
            merge_size: cfg.spatial_merge_size,
            hidden_size,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor, Error> {
        let num_patches = x
            .dim(0)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "merger dim", e))?;
        let group = self.merge_size * self.merge_size;
        if num_patches % group != 0 {
            return Err(Error::InvalidInput {
                message: format!(
                    "NaviDC-OCR: merger expects num_patches divisible by {}, got {}",
                    group, num_patches
                ),
            });
        }
        let x = self
            .ln_q
            .forward(x)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "merger ln_q", e))?;
        let x = x
            .reshape((num_patches / group, self.hidden_size))
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "merger reshape", e))?;
        let x = self
            .mlp1
            .forward(&x)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "merger mlp1", e))?;
        let x = x.gelu_erf().map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "NaviDC-OCR: merger gelu failed",
                e,
            )
        })?;
        self.mlp2
            .forward(&x)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "merger mlp2", e))
    }
}

pub struct NaviDcVisionModel {
    patch_embed: PatchEmbed,
    blocks: Vec<NaviDcVisionBlock>,
    merger: NaviDcPatchMerger,
    rotary_pos_emb: VisionRotaryEmbedding,
    spatial_merge_size: usize,
    patch_size: usize,
    window_size: usize,
    fullatt_block_indexes: Vec<usize>,
}

impl NaviDcVisionModel {
    pub fn load(cfg: &NaviDcVisionConfig, vb: VarBuilder) -> Result<Self, Error> {
        let patch_embed = PatchEmbed::load(cfg, vb.clone())?;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let block_vb = vb.pp(format!("blocks.{i}"));
            blocks.push(NaviDcVisionBlock::load(cfg, block_vb)?);
        }
        let merger = NaviDcPatchMerger::load(cfg, vb.clone())?;
        let head_dim = cfg.head_dim()?;
        if head_dim % 2 != 0 {
            return Err(Error::Config {
                message: format!(
                    "NaviDC-OCR: vision head_dim {head_dim} must be even for rotary embeddings"
                ),
            });
        }
        // Matches Qwen2_5_VisionRotaryEmbedding(head_dim // 2): 20
        // frequencies per axis for head_dim 80.
        let rotary_pos_emb = VisionRotaryEmbedding::new(head_dim / 2, 10000.0, vb.device())?;
        Ok(Self {
            patch_embed,
            blocks,
            merger,
            rotary_pos_emb,
            spatial_merge_size: cfg.spatial_merge_size,
            patch_size: cfg.patch_size,
            window_size: cfg.window_size,
            fullatt_block_indexes: cfg.fullatt_block_indexes.clone(),
        })
    }

    /// Run the vision tower over a batch of images and return the merged
    /// image embeddings (one row per merged token, in raster order).
    pub fn forward(
        &self,
        pixel_values: &Tensor,
        grid_thw: &[(usize, usize, usize)],
    ) -> Result<Tensor, Error> {
        let device = pixel_values.device();
        let max_grid = grid_thw
            .iter()
            .map(|(_, h, w)| (*h).max(*w))
            .max()
            .unwrap_or(0);
        let rotary_full = self.rotary_pos_emb.forward(max_grid, device)?;
        let freq_dim = self.rotary_pos_emb.dim() / 2;
        let merge_unit = self.spatial_merge_size * self.spatial_merge_size;

        let mut outputs: Vec<Tensor> = Vec::with_capacity(grid_thw.len());
        let mut offset = 0usize;
        for &(t, h, w) in grid_thw {
            let num_patches = t * h * w;
            let patches = pixel_values
                .narrow(0, offset, num_patches)
                .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision narrow patches", e))?;
            offset += num_patches;

            let mut hidden = self.patch_embed.forward(&patches)?;

            let (hpos, wpos) = vision_position_ids(t, h, w, self.spatial_merge_size);
            let plan = window_plan(
                &[(t, h, w)],
                self.spatial_merge_size,
                self.patch_size,
                self.window_size,
            )?;

            // Permute the position ids and the hidden states into window
            // order at merge-unit granularity; rope ids are built directly in
            // window order so the attention rows line up with the segments.
            let num_units = num_patches / merge_unit;
            if plan.window_index.len() != num_units {
                return Err(Error::InvalidInput {
                    message: format!(
                        "NaviDC-OCR: window plan covers {} units, expected {num_units}",
                        plan.window_index.len()
                    ),
                });
            }
            let mut hpos_window = Vec::with_capacity(num_patches);
            let mut wpos_window = Vec::with_capacity(num_patches);
            for &unit in &plan.window_index {
                for i in 0..merge_unit {
                    let patch = unit * merge_unit + i;
                    hpos_window.push(hpos[patch]);
                    wpos_window.push(wpos[patch]);
                }
            }
            let (cos, sin) =
                gather_vision_pos_emb(&rotary_full, freq_dim, &hpos_window, &wpos_window, device)?;

            let index = Tensor::from_vec(
                plan.window_index.iter().map(|&u| u as i64).collect(),
                (num_units,),
                device,
            )
            .map_err(|e| {
                candle_to_ocr_processing(
                    crate::error::ProcessingStage::TensorOperation,
                    "NaviDC-OCR: vision window index tensor failed",
                    e,
                )
            })?;
            hidden = hidden
                .reshape((num_units, merge_unit, ()))
                .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision reshape units", e))?
                .index_select(&index, 0)
                .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision window permute", e))?
                .reshape((num_patches, ()))
                .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision reshape patches", e))?;

            let window_segments: Vec<(usize, usize)> = plan
                .cu_window_seqlens
                .windows(2)
                .map(|pair| (pair[0], pair[1] - pair[0]))
                .collect();
            let fullatt_segment = [(0usize, num_patches)];
            for (layer, block) in self.blocks.iter().enumerate() {
                let segments: &[(usize, usize)] = if self.fullatt_block_indexes.contains(&layer) {
                    &fullatt_segment
                } else {
                    &window_segments
                };
                hidden = block.forward(&hidden, &cos, &sin, segments)?;
            }

            // The merger groups whole merge units, which stay contiguous
            // under the window permutation, so it runs in window order; the
            // inverse permutation then restores raster order per merged token
            // (mirroring `argsort(window_index)` in the reference).
            let merged = self.merger.forward(&hidden)?;
            let reverse = Tensor::from_vec(
                plan.reverse_index.iter().map(|&u| u as i64).collect(),
                (num_units,),
                device,
            )
            .map_err(|e| {
                candle_to_ocr_processing(
                    crate::error::ProcessingStage::TensorOperation,
                    "NaviDC-OCR: vision reverse index tensor failed",
                    e,
                )
            })?;
            let restored = merged
                .index_select(&reverse, 0)
                .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision reverse permute", e))?;
            outputs.push(restored);
        }

        let refs: Vec<&Tensor> = outputs.iter().collect();
        Tensor::cat(&refs, 0)
            .map_err(|e| candle_to_ocr_inference("NaviDC-OCR", "vision outputs cat", e))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // merge=2, patch=14, window=112 → window side = 112/2/14 = 4 merged
    // tokens, 4 patches per merged unit (matches HF's
    // `window_size // spatial_merge_size // patch_size`).
    const MERGE: usize = 2;
    const PATCH: usize = 14;
    const WINDOW: usize = 112;

    #[test]
    fn window_plan_single_exact_window() {
        // 8×8 patches → 4×4 merged = exactly one window (16 units = 64 patches)
        let plan = window_plan(&[(1, 8, 8)], MERGE, PATCH, WINDOW).unwrap();
        assert_eq!(plan.window_index, (0..16).collect::<Vec<_>>());
        assert_eq!(plan.cu_window_seqlens, vec![0, 64]);
        assert_eq!(plan.reverse_index, (0..16).collect::<Vec<_>>());
    }

    #[test]
    fn window_plan_partial_edge_windows() {
        // 28×28 patches → 14×14 merged, window side 4 → 4×4 windows; the
        // last row/column of windows covers only 2 merged cells each way.
        // Segment lengths in patches: rows 0-2 full (64,64,64,32), row 3
        // partial (32,32,32,16).
        let plan = window_plan(&[(1, 28, 28)], MERGE, PATCH, WINDOW).unwrap();
        assert_eq!(
            plan.cu_window_seqlens,
            vec![
                0, 64, 128, 192, 224, 288, 352, 416, 448, 512, 576, 640, 672, 704, 736, 768, 784
            ]
        );
        // First window: merged rows 0-3 × cols 0-3 of the 14-wide grid.
        let first: Vec<usize> = (0..4)
            .flat_map(|r| (0..4).map(move |c| r * 14 + c))
            .collect();
        assert_eq!(&plan.window_index[..16], &first[..]);
        // Second window (cols 4-7): another 16 units.
        let second: Vec<usize> = (0..4)
            .flat_map(|r| (4..8).map(move |c| r * 14 + c))
            .collect();
        assert_eq!(&plan.window_index[16..32], &second[..]);
    }

    #[test]
    fn window_plan_exact_multiple_drops_empty_tail_windows() {
        // 32×32 patches → 16×16 merged: already a multiple of the 4-window
        // side, so the reference's padding would add a full empty window
        // row/column that unique_consecutive drops — leaving 16 full windows.
        let plan = window_plan(&[(1, 32, 32)], MERGE, PATCH, WINDOW).unwrap();
        let expected: Vec<usize> = (0..=16).map(|i| i * 64).collect();
        assert_eq!(plan.cu_window_seqlens, expected);
        assert_eq!(plan.window_index.len(), 256);
    }

    #[test]
    fn window_plan_grid_smaller_than_one_window() {
        // 4×4 patches → 2×2 merged: one partial window of 4 units (16 patches)
        let plan = window_plan(&[(1, 4, 4)], MERGE, PATCH, WINDOW).unwrap();
        assert_eq!(plan.window_index, vec![0, 1, 2, 3]);
        assert_eq!(plan.cu_window_seqlens, vec![0, 16]);
    }

    #[test]
    fn window_plan_multiple_images_accumulate_offsets() {
        let plan = window_plan(&[(1, 16, 16), (1, 8, 8)], MERGE, PATCH, WINDOW).unwrap();
        // First image (8×8 merged): four 4×4 windows of 64 patches. Second
        // image (4×4 merged): one window, ids offset by 64 units.
        assert_eq!(plan.window_index[64..], (64..80).collect::<Vec<_>>());
        assert_eq!(plan.cu_window_seqlens, vec![0, 64, 128, 192, 256, 320]);
    }

    #[test]
    fn window_plan_permutations_are_mutual_inverses() {
        let plan = window_plan(&[(1, 28, 28), (1, 20, 36)], MERGE, PATCH, WINDOW).unwrap();
        for (slot, &source) in plan.window_index.iter().enumerate() {
            assert_eq!(plan.window_index[plan.reverse_index[source]], source);
            assert_eq!(plan.reverse_index[source], slot);
        }
        let total: usize = [(1usize, 28usize, 28usize), (1, 20, 36)]
            .iter()
            .map(|&(t, h, w)| (h / MERGE) * (w / MERGE) * t)
            .sum();
        assert_eq!(plan.window_index.len(), total);
        assert_eq!(*plan.cu_window_seqlens.last().unwrap(), total * 4);
    }

    #[test]
    fn window_plan_rejects_misaligned_window_size() {
        let err = window_plan(&[(1, 16, 16)], MERGE, PATCH, 100).unwrap_err();
        assert!(err.to_string().contains("window_size"), "{err}");
    }

    #[test]
    fn vision_position_ids_follow_merge_grouped_raster_order() {
        let (hpos, wpos) = vision_position_ids(1, 4, 4, 2);
        // Merge blocks of 2×2 patches come block-raster-first (hb, wb), and
        // within a block the four patches enumerate the inner 2×2 offsets in
        // (h_inner, w_inner) order — the layout of HF's permute(0, 2, 1, 3).
        assert_eq!(hpos, vec![0, 0, 1, 1, 0, 0, 1, 1, 2, 2, 3, 3, 2, 2, 3, 3]);
        assert_eq!(wpos, vec![0, 1, 0, 1, 2, 3, 2, 3, 0, 1, 0, 1, 2, 3, 2, 3]);
    }
}
