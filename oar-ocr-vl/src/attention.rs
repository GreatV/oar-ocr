//! Unified attention implementation for all VLM models.
//!
//! This module provides shared attention and rotary embedding implementations
//! to ensure consistent behavior across PaddleOCR-VL, HunyuanOCR, GLM-OCR,
//! and MinerU2.5 models.
//!
//! ## Benefits
//!
//! - Single place for attention and RoPE optimizations
//! - Consistent mask handling across models
//! - Shared KV cache logic
//! - Support for multi-axis RoPE variants (MRoPE, XDRoPE)
//! - Easier testing and maintenance
//!
//! ## Usage
//!
//! ```ignore
//! use oar_ocr_vl::attention::{scaled_dot_product_attention, create_causal_mask, RotaryEmbedding};
//!
//! // Standard attention
//! let output = scaled_dot_product_attention(&q, &k, &v, mask, scale, is_causal)?;
//!
//! // Create causal mask for autoregressive decoding
//! let mask = create_causal_mask(seq_len, kv_len, dtype, device)?;
//!
//! // Multi-axis RoPE (for PaddleOCR-VL, HunyuanOCR)
//! let rope = RotaryEmbedding::new_multi_axis(head_dim, rope_theta, num_dims, device)?;
//! let (cos, sin) = rope.forward_multi_axis(&position_ids, dtype)?;
//! ```

use crate::utils::candle_to_ocr_processing;
use candle_core::{D, DType, Device, IndexOp, Result, Tensor};
use oar_ocr_core::core::errors::OCRError;

/// Helper function to handle Metal device computation.
///
/// Metal backend doesn't support certain operations (arange, broadcast_*, etc.).
/// This helper executes operations on CPU for Metal devices, then transfers the
/// result back to Metal.
///
/// # Arguments
/// * `device` - Target device
/// * `f` - Closure that creates the tensor on the compute device
///
/// # Returns
/// Tensor on the target device
pub(crate) fn on_compute_device<F>(device: &Device, f: F) -> Result<Tensor>
where
    F: FnOnce(&Device) -> Result<Tensor>,
{
    if device.is_metal() {
        // Operations unsupported on Metal are run on the CPU...
        let cpu_device = Device::Cpu;
        let tensor_on_cpu = f(&cpu_device)?;
        // ...and the result is moved back to the Metal device.
        tensor_on_cpu.to_device(device)
    } else {
        // For other devices, run directly.
        f(device)
    }
}

/// Scaled dot-product attention.
///
/// Computes attention as: softmax(Q @ K^T * scale) @ V
///
/// # Arguments
/// * `q` - Query tensor: (batch, heads, seq_q, head_dim)
/// * `k` - Key tensor: (batch, heads, seq_kv, head_dim)
/// * `v` - Value tensor: (batch, heads, seq_kv, head_dim)
/// * `mask` - Optional attention mask to add before softmax
/// * `scale` - Scaling factor (typically 1/sqrt(head_dim))
/// * `is_causal` - If true and mask is None, creates a causal mask
///
/// # Returns
/// Output tensor: (batch, heads, seq_q, head_dim)
pub fn scaled_dot_product_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
    is_causal: bool,
) -> Result<Tensor> {
    // Q @ K^T
    let attn_weights = q.matmul(&k.transpose(2, 3)?)?;

    // Scale
    let attn_weights = (attn_weights * scale)?;

    // Apply mask
    let attn_weights = match mask {
        Some(m) => attn_weights.broadcast_add(m)?,
        None if is_causal => {
            let seq_len = attn_weights.dim(2)?;
            let kv_len = attn_weights.dim(3)?;
            let causal_mask =
                create_causal_mask(seq_len, kv_len, attn_weights.dtype(), q.device())?;
            attn_weights.broadcast_add(&causal_mask)?
        }
        None => attn_weights,
    };

    // Softmax: cast to F32 for numerical stability, then cast back
    let input_dtype = attn_weights.dtype();
    let attn_weights = attn_weights.to_dtype(DType::F32)?;
    let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
    let attn_weights = attn_weights.to_dtype(input_dtype)?;

    // Attention @ V
    attn_weights.matmul(v)
}

/// Create a causal (lower-triangular) attention mask.
///
/// Returns a mask where position i can only attend to positions <= i.
/// The mask contains 0 for allowed positions and -inf for masked positions.
///
/// # Arguments
/// * `seq_len` - Query sequence length
/// * `kv_len` - Key/Value sequence length
/// * `dtype` - Data type for the mask tensor
/// * `device` - Device for the mask tensor
///
/// # Returns
/// Mask tensor of shape (1, 1, seq_len, kv_len)
pub fn create_causal_mask(
    seq_len: usize,
    kv_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    on_compute_device(device, |compute_device| {
        let row_idx = Tensor::arange(0u32, seq_len as u32, compute_device)?
            .reshape((seq_len, 1))?
            .to_dtype(dtype)?;
        let col_idx = Tensor::arange(0u32, kv_len as u32, compute_device)?
            .reshape((1, kv_len))?
            .to_dtype(dtype)?;

        let offset = (kv_len.saturating_sub(seq_len)) as f64;
        // Condition: col <= row + offset
        // col - offset <= row
        let diff = col_idx.broadcast_sub(&Tensor::new(offset, compute_device)?.to_dtype(dtype)?)?;
        let mask_cond = diff.broadcast_le(&row_idx)?;

        let zero = Tensor::new(0f32, compute_device)?
            .to_dtype(dtype)?
            .broadcast_as(mask_cond.shape())?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, compute_device)?
            .to_dtype(dtype)?
            .broadcast_as(mask_cond.shape())?;

        mask_cond
            .where_cond(&zero, &neg_inf)?
            .reshape((1, 1, seq_len, kv_len))
    })
}

/// Create a padding mask from sequence lengths.
///
/// # Arguments
/// * `seq_lens` - Sequence lengths for each batch item
/// * `max_len` - Maximum sequence length
/// * `dtype` - Data type for the mask tensor
/// * `device` - Device for the mask tensor
///
/// # Returns
/// Mask tensor of shape (batch, 1, 1, max_len)
pub fn create_padding_mask(
    seq_lens: &[usize],
    max_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let batch_size = seq_lens.len();

    on_compute_device(device, |compute_device| {
        // (B, 1, 1, 1)
        let lens_tensor = Tensor::from_vec(
            seq_lens.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            (batch_size, 1, 1, 1),
            compute_device,
        )?
        .to_dtype(dtype)?;

        // (1, 1, 1, max_len)
        let pos_tensor = Tensor::arange(0u32, max_len as u32, compute_device)?
            .reshape((1, 1, 1, max_len))?
            .to_dtype(dtype)?;

        // Mask: pos < len -> 0, else -inf
        let mask_cond = pos_tensor.broadcast_lt(&lens_tensor)?;

        let zero = Tensor::new(0f32, compute_device)?
            .to_dtype(dtype)?
            .broadcast_as(mask_cond.shape())?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, compute_device)?
            .to_dtype(dtype)?
            .broadcast_as(mask_cond.shape())?;

        mask_cond.where_cond(&zero, &neg_inf)
    })
}

/// Combine causal and padding masks.
///
/// # Arguments
/// * `causal_mask` - Causal mask (1, 1, seq_len, kv_len)
/// * `padding_mask` - Padding mask (batch, 1, 1, kv_len)
///
/// # Returns
/// Combined mask (batch, 1, seq_len, kv_len)
pub fn combine_masks(causal_mask: &Tensor, padding_mask: &Tensor) -> Result<Tensor> {
    causal_mask.broadcast_add(padding_mask)
}

/// Create a tree-ancestry attention mask for HSD verification (paper §3.2 / Fig. 2c).
///
/// Each candidate token in the prefix tree may attend only to (a) the accepted
/// prefix already in the KV cache and (b) tokens along its own ancestor path
/// (including itself). Sibling subtrees and descendants are masked out.
///
/// # Arguments
/// * `parents` - Parent indices for each candidate node, in packed order. `None`
///   means the node is a direct child of the (implicit) root. Length = `N`.
/// * `prefix_kv_len` - Number of accepted-prefix tokens already in the KV cache.
/// * `dtype` - Data type for the mask.
/// * `device` - Target device.
///
/// # Returns
/// Mask of shape `(1, 1, N, prefix_kv_len + N)` where allowed positions are 0
/// and forbidden positions are `-inf`.
///
/// # Performance note (paper parity)
///
/// The paper §4.3 runs verification through PyTorch's
/// [FlexAttention](https://pytorch.org/blog/flexattention/), which compiles the
/// tree-ancestry predicate directly into a fused attention kernel. The mask
/// never materialises as a tensor — it's a function evaluated per-thread inside
/// the FlashAttention-style kernel, so the asymptotic memory cost is `O(N)`
/// rather than `O((prefix_kv_len + N) * N)` and the attention forward is
/// compute-bound.
///
/// Candle has no FlexAttention equivalent today, so this helper materialises
/// the full `(1, 1, N, prefix_kv_len + N)` mask tensor and feeds it into the
/// model's standard attention path (`F.scaled_dot_product_attention` analog).
/// At the typical HSD scales (`N ≤ 32` candidates after capping, `prefix_kv_len`
/// up to ~16k for a page) the materialised mask is only a few MiB so memory is
/// not the constraint, but the **separate mask kernel + standard attention
/// kernel** combination is measurably slower than a fused FlexAttention path —
/// this is one source of the gap between the paper's SR_e2e and the
/// OAR-realised SR_e2e at equal AAL. Fixing it requires either:
///
/// 1. A candle-side fused tree-attention kernel (custom CUDA kernel), or
/// 2. An upstream FlexAttention-equivalent in candle that we can call from
///    the model's attention path with a tree-ancestry predicate.
///
/// Neither is in-scope for this crate; we accept the kernel-level overhead as
/// the documented gap. Acceptance length (AAL) is unaffected — only wall-clock
/// SR_decode / SR_e2e.
pub fn create_tree_attention_mask(
    parents: &[Option<usize>],
    prefix_kv_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let n = parents.len();
    let total_kv = prefix_kv_len + n;

    if n == 0 {
        return on_compute_device(device, |compute_device| {
            Tensor::zeros((1, 1, 0, total_kv), dtype, compute_device)
        });
    }

    // Host-side mask buffer. Initialised to -inf, then we punch holes for
    // (a) the accepted prefix (whole left block) and (b) each node's ancestor
    // chain (sparse hits in the right block).
    //
    // Earlier revisions materialised an O(N²) `Vec<Vec<bool>>` ancestor matrix
    // up front; that's redundant because we only consume each ancestor set
    // once. Walking the parent chain inline during the buffer fill removes:
    // - the N² bool allocation (up to 32² = 1 KiB, irrelevant for big-O but
    //   one extra heap call per HSD step), and
    // - the per-row O(N) "is this an ancestor?" check (now a tight
    //   parent-pointer walk of length = node depth, typically ≤ 8 for HSD
    //   trees of width 32).
    //
    // The accepted-prefix block uses `slice::fill` (memset under the hood)
    // instead of a per-cell loop, which is the dominant cost on long pages —
    // for a 16k-token prefix and 32 candidates that's ~512K float writes per
    // verify step, and memset is 5-10× faster than the per-element write.
    let mut buf = vec![f32::NEG_INFINITY; n * total_kv];
    for i in 0..n {
        let row_off = i * total_kv;
        // (a) Allow attending to the entire accepted prefix — single memset.
        if prefix_kv_len > 0 {
            buf[row_off..row_off + prefix_kv_len].fill(0.0);
        }
        // (b) Allow attending to this node + its ancestors. Walk parent
        //     pointers in place; no auxiliary bitset.
        let mut cur = Some(i);
        while let Some(j) = cur {
            buf[row_off + prefix_kv_len + j] = 0.0;
            cur = parents[j];
        }
    }

    on_compute_device(device, move |compute_device| {
        Tensor::from_vec(buf, (1, 1, n, total_kv), compute_device)?.to_dtype(dtype)
    })
}

/// Create a left-padding mask for batched sequences.
///
/// Left-padding aligns sequences at the right edge, which is standard for
/// autoregressive generation. The mask marks left-padded positions as -inf.
///
/// # Arguments
/// * `seq_lens` - Actual sequence lengths for each batch item
/// * `max_len` - Maximum (padded) sequence length
/// * `dtype` - Data type for the mask tensor
/// * `device` - Device for the mask tensor
///
/// # Returns
/// Mask tensor of shape (batch, 1, 1, max_len) where:
/// - Left-padded positions (j < max_len - seq_len) are -inf
/// - Valid positions are 0
///
/// # Example
/// ```text
/// // seq_lens = [3, 5], max_len = 5
/// // Produces masks:
/// // Item 0: [-inf, -inf, 0, 0, 0]  (3 valid tokens, 2 padding)
/// // Item 1: [0, 0, 0, 0, 0]        (5 valid tokens, 0 padding)
/// ```
pub fn create_left_padding_mask(
    seq_lens: &[usize],
    max_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let batch_size = seq_lens.len();

    on_compute_device(device, |compute_device| {
        // pad_len = max_len - len
        // lens: (B, 1, 1, 1)
        let lens_tensor = Tensor::from_vec(
            seq_lens.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            (batch_size, 1, 1, 1),
            compute_device,
        )?
        .to_dtype(dtype)?;

        let max_len_t = Tensor::new(max_len as u32, compute_device)?.to_dtype(dtype)?;
        let pad_len = max_len_t.broadcast_sub(&lens_tensor)?; // (B, 1, 1, 1)

        // pos: (1, 1, 1, max_len)
        let pos_tensor = Tensor::arange(0u32, max_len as u32, compute_device)?
            .reshape((1, 1, 1, max_len))?
            .to_dtype(dtype)?;

        // Mask: pos < pad_len -> -inf, else 0
        let mask_cond = pos_tensor.broadcast_lt(&pad_len)?;

        let zero = Tensor::new(0f32, compute_device)?
            .to_dtype(dtype)?
            .broadcast_as(mask_cond.shape())?;
        let neg_inf = Tensor::new(f32::NEG_INFINITY, compute_device)?
            .to_dtype(dtype)?
            .broadcast_as(mask_cond.shape())?;

        // if pos < pad_len (padded region), return -inf
        mask_cond.where_cond(&neg_inf, &zero)
    })
}

/// Builds the per-step attention mask for autoregressive decoding with a
/// left-padded batch.
///
/// During decode each query is a single new token that may attend to the
/// entire KV cache (length `kv_len`). The leading `pad_lens[i]` positions of
/// row `i` are left-padding and must be masked out, otherwise the new token
/// attends to padding KV and the output is corrupted for batches with unequal
/// prompt lengths. Returns a `(batch, 1, 1, kv_len)` additive mask (`0` for
/// attendable positions, a large negative value for padding).
///
/// For a batch of size 1 (or any batch with no padding) every `pad_len` is 0
/// and the mask is all-zeros, i.e. a no-op.
pub fn create_generation_mask(
    pad_lens: &[usize],
    kv_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let batch_size = pad_lens.len();

    on_compute_device(device, |compute_device| {
        // pad_lens as tensor: (batch, 1, 1, 1)
        let pad_lens_tensor = Tensor::from_vec(
            pad_lens.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            (batch_size, 1, 1, 1),
            compute_device,
        )?
        .to_dtype(dtype)?;

        // Position indices: (1, 1, 1, kv_len)
        let pos_tensor = Tensor::arange(0u32, kv_len as u32, compute_device)?
            .reshape((1, 1, 1, kv_len))?
            .to_dtype(dtype)?;

        // Mask condition: pos < pad_len -> masked (large negative value)
        let mask_cond = pos_tensor.broadcast_lt(&pad_lens_tensor)?;

        let zero = Tensor::new(0f32, compute_device)?
            .to_dtype(dtype)?
            .broadcast_as(mask_cond.shape())?;
        // Use large negative value instead of -inf to avoid potential numerical issues
        let mask_value = Tensor::new(-1e9_f32, compute_device)?
            .to_dtype(dtype)?
            .broadcast_as(mask_cond.shape())?;

        mask_cond.where_cond(&mask_value, &zero)
    })
}

// ============================================================================
// Rotary Positional Embedding (RoPE)
// ============================================================================

/// Unified Rotary Positional Embedding implementation.
///
/// Supports multiple RoPE variants:
/// - **Dynamic RoPE**: On-the-fly computation from position IDs (single-axis)
/// - **Multi-axis RoPE (MRoPE)**: 3-axis encoding for text/height/width (PaddleOCR-VL)
/// - **Extended Dimension RoPE (XDRoPE)**: Configurable num_dims (HunyuanOCR)
///
/// ## Architecture
///
/// All variants share a single `Dynamic` representation parameterized by
/// `num_dims`; the constructor functions pick the right `num_dims` for each
/// model family.
#[derive(Debug, Clone)]
pub enum RotaryEmbedding {
    /// Dynamic computation from inverse frequencies (used by PaddleOCR-VL, HunyuanOCR).
    /// Supports multi-axis position encoding.
    Dynamic {
        inv_freq: Tensor,
        /// Number of position dimensions (1 for standard, 3 for MRoPE/XDRoPE)
        num_dims: usize,
    },
}

impl RotaryEmbedding {
    /// Create a dynamic RoPE with on-the-fly computation (standard single-axis).
    ///
    /// This computes embeddings dynamically from position IDs, suitable for
    /// variable-length sequences.
    ///
    /// # Arguments
    /// * `head_dim` - Dimension of each attention head (must be even)
    /// * `rope_theta` - Base frequency (typically 10000.0)
    /// * `device` - Device for tensor allocation
    ///
    /// # Returns
    /// RotaryEmbedding in Dynamic mode with num_dims=1
    pub fn new_dynamic(
        head_dim: usize,
        rope_theta: f64,
        device: &Device,
    ) -> std::result::Result<Self, OCRError> {
        Self::new_multi_axis(head_dim, rope_theta, 1, device)
    }

    /// Create a multi-axis RoPE for position encoding (MRoPE/XDRoPE).
    ///
    /// Supports multiple position dimensions for complex position encoding:
    /// - num_dims=1: Standard RoPE (single position)
    /// - num_dims=3: MRoPE (text position, height, width)
    ///
    /// # Arguments
    /// * `head_dim` - Dimension of each attention head (must be even)
    /// * `rope_theta` - Base frequency (typically 10000.0)
    /// * `num_dims` - Number of position dimensions
    /// * `device` - Device for tensor allocation
    ///
    /// # Returns
    /// RotaryEmbedding in Dynamic mode with specified num_dims
    pub fn new_multi_axis(
        head_dim: usize,
        rope_theta: f64,
        num_dims: usize,
        device: &Device,
    ) -> std::result::Result<Self, OCRError> {
        if !head_dim.is_multiple_of(2) {
            return Err(OCRError::ConfigError {
                message: format!("RotaryEmbedding: head_dim must be even, got {head_dim}"),
            });
        }
        let half = head_dim / 2;
        let mut inv_freq = Vec::with_capacity(half);
        for i in (0..head_dim).step_by(2) {
            let v = 1f64 / rope_theta.powf(i as f64 / head_dim as f64);
            inv_freq.push(v as f32);
        }
        let inv_freq = Tensor::from_vec(inv_freq, (half,), device).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "RotaryEmbedding: failed to create inv_freq tensor",
                e,
            )
        })?;
        Ok(Self::Dynamic { inv_freq, num_dims })
    }

    /// Forward pass for multi-axis RoPE.
    ///
    /// Computes cos/sin from position IDs dynamically. Supports multi-dimensional
    /// position encoding.
    ///
    /// # Arguments
    /// * `position_ids` - Position tensor, shape: (num_dims, batch, seq)
    /// * `dtype` - Target data type for output
    ///
    /// # Returns
    /// Tuple of (cos, sin) tensors, shape: (num_dims, batch, seq, head_dim)
    pub fn forward_multi_axis(
        &self,
        position_ids: &Tensor,
        dtype: DType,
    ) -> std::result::Result<(Tensor, Tensor), OCRError> {
        match self {
            Self::Dynamic { inv_freq, num_dims } => {
                let dims = position_ids.dims();
                if dims.len() != 3 || dims[0] != *num_dims {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "RotaryEmbedding: expected position_ids shape ({}, B, S), got {:?}",
                            num_dims, dims
                        ),
                    });
                }

                let position_ids = position_ids.to_dtype(DType::F32).map_err(|e| {
                    candle_to_ocr_processing(
                        oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                        "RotaryEmbedding: position_ids cast to f32 failed",
                        e,
                    )
                })?;

                let inv_len = inv_freq.dims1().map_err(|e| {
                    candle_to_ocr_processing(
                        oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                        "RotaryEmbedding: inv_freq dims1 failed",
                        e,
                    )
                })?;
                let inv = inv_freq
                    .reshape((1usize, 1usize, 1usize, inv_len))
                    .map_err(|e| {
                        candle_to_ocr_processing(
                            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                            "RotaryEmbedding: inv_freq reshape failed",
                            e,
                        )
                    })?;

                let freqs = position_ids
                    .unsqueeze(3)
                    .map_err(|e| {
                        candle_to_ocr_processing(
                            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                            "RotaryEmbedding: position_ids unsqueeze failed",
                            e,
                        )
                    })?
                    .broadcast_mul(&inv)
                    .map_err(|e| {
                        candle_to_ocr_processing(
                            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                            "RotaryEmbedding: rotary freqs multiply failed",
                            e,
                        )
                    })?;

                let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1).map_err(|e| {
                    candle_to_ocr_processing(
                        oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                        "RotaryEmbedding: rotary emb cat failed",
                        e,
                    )
                })?;

                let cos = emb
                    .cos()
                    .map_err(|e| {
                        candle_to_ocr_processing(
                            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                            "RotaryEmbedding: rotary cos failed",
                            e,
                        )
                    })?
                    .to_dtype(dtype)
                    .map_err(|e| {
                        candle_to_ocr_processing(
                            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                            "RotaryEmbedding: rotary cos cast failed",
                            e,
                        )
                    })?;

                let sin = emb
                    .sin()
                    .map_err(|e| {
                        candle_to_ocr_processing(
                            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                            "RotaryEmbedding: rotary sin failed",
                            e,
                        )
                    })?
                    .to_dtype(dtype)
                    .map_err(|e| {
                        candle_to_ocr_processing(
                            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                            "RotaryEmbedding: rotary sin cast failed",
                            e,
                        )
                    })?;
                Ok((cos, sin))
            }
        }
    }
}

/// Repeat KV heads for Grouped Query Attention (GQA).
///
/// When num_heads > num_kv_heads, the KV heads need to be repeated
/// to match the number of query heads.
///
/// # Arguments
/// * `x` - Input tensor: (batch, num_kv_heads, seq, head_dim)
/// * `n_rep` - Number of times to repeat each KV head
///
/// # Returns
/// Output tensor: (batch, num_kv_heads * n_rep, seq, head_dim)
pub fn repeat_kv(x: &Tensor, n_rep: usize) -> Result<Tensor> {
    if n_rep == 1 {
        return Ok(x.clone());
    }
    let (batch, num_kv_heads, seq_len, head_dim) = x.dims4()?;
    x.unsqueeze(2)?
        .expand((batch, num_kv_heads, n_rep, seq_len, head_dim))?
        .reshape((batch, num_kv_heads * n_rep, seq_len, head_dim))
}

/// Select and combine RoPE sections for multi-axis position encoding.
///
/// This is a unified implementation for both MRoPE (3-axis) and XDRoPE (4-axis).
/// It selects different position dimensions for different head_dim sections,
/// enabling the model to encode spatial (height, width) and temporal positions separately.
///
/// # Arguments
/// * `cos_or_sin` - Input tensor: (num_dims, batch, seq, head_dim)
/// * `rope_section` - Section sizes, must sum to head_dim/2
/// * `num_dims` - Number of position dimensions (3 for MRoPE, 4 for XDRoPE)
///
/// # Returns
/// Output tensor: (batch, 1, seq, head_dim)
///
/// # Example
/// For MRoPE with rope_section=[16, 24, 24] and head_dim=128:
/// - Sections are doubled: [16, 24, 24, 16, 24, 24]
/// - Each section picks from dim (i % 3): [dim0, dim1, dim2, dim0, dim1, dim2]
/// - Results are concatenated along head_dim axis
pub fn select_rope_sections(
    cos_or_sin: &Tensor,
    rope_section: &[usize],
    num_dims: usize,
) -> std::result::Result<Tensor, OCRError> {
    if rope_section.is_empty() {
        return Err(OCRError::ConfigError {
            message: "rope_section is empty".to_string(),
        });
    }

    let dims = cos_or_sin.dims();
    let head_dim = dims.get(3).copied().unwrap_or(0);
    let section_sum: usize = rope_section.iter().sum();
    if section_sum * 2 != head_dim {
        return Err(OCRError::ConfigError {
            message: format!(
                "rope_section sum ({}) * 2 != head_dim ({})",
                section_sum, head_dim
            ),
        });
    }

    let actual_dims = dims.first().copied().unwrap_or(0);
    if actual_dims != num_dims {
        return Err(OCRError::InvalidInput {
            message: format!(
                "rope tensor has {} dims, expected {}",
                actual_dims, num_dims
            ),
        });
    }

    // Double the sections: [a, b, c] -> [a, b, c, a, b, c]
    let doubled_sections: Vec<usize> = rope_section
        .iter()
        .chain(rope_section.iter())
        .copied()
        .collect();

    let mut offset = 0usize;
    let mut chunks: Vec<Tensor> = Vec::with_capacity(doubled_sections.len());
    for (i, &sec) in doubled_sections.iter().enumerate() {
        let next = offset + sec;
        let seg = cos_or_sin.i((.., .., .., offset..next)).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                format!(
                    "rope slice failed at chunk {} (offset {}..{})",
                    i, offset, next
                ),
                e,
            )
        })?;
        let picked = seg.i((i % num_dims, .., .., ..)).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                format!("rope pick failed at chunk {} (dim {})", i, i % num_dims),
                e,
            )
        })?;
        chunks.push(picked);
        offset = next;
    }

    let refs: Vec<&Tensor> = chunks.iter().collect();
    let cat = Tensor::cat(&refs, D::Minus1).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "rope cat failed",
            e,
        )
    })?;
    cat.unsqueeze(1).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "rope unsqueeze failed",
            e,
        )
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scaled_dot_product_attention() -> Result<()> {
        let device = Device::Cpu;

        // Create simple Q, K, V tensors
        let q = Tensor::randn(0f32, 1., (1, 4, 8, 64), &device)?;
        let k = Tensor::randn(0f32, 1., (1, 4, 8, 64), &device)?;
        let v = Tensor::randn(0f32, 1., (1, 4, 8, 64), &device)?;

        let scale = 1.0 / (64f64).sqrt();

        // Without mask
        let out = scaled_dot_product_attention(&q, &k, &v, None, scale, false)?;
        assert_eq!(out.dims(), &[1, 4, 8, 64]);

        // With causal mask
        let out = scaled_dot_product_attention(&q, &k, &v, None, scale, true)?;
        assert_eq!(out.dims(), &[1, 4, 8, 64]);

        Ok(())
    }

    #[test]
    fn test_causal_mask() -> Result<()> {
        let device = Device::Cpu;
        let mask = create_causal_mask(4, 4, DType::F32, &device)?;

        assert_eq!(mask.dims(), &[1, 1, 4, 4]);

        // Check mask values
        let mask_data: Vec<f32> = mask.flatten_all()?.to_vec1()?;

        // Row 0: can only attend to position 0
        assert_eq!(mask_data[0], 0.0);
        assert!(mask_data[1].is_infinite() && mask_data[1] < 0.0);

        // Row 3: can attend to all positions
        assert_eq!(mask_data[12], 0.0);
        assert_eq!(mask_data[13], 0.0);
        assert_eq!(mask_data[14], 0.0);
        assert_eq!(mask_data[15], 0.0);

        Ok(())
    }

    #[test]
    fn test_causal_mask_with_kv_cache() -> Result<()> {
        let device = Device::Cpu;

        // Simulating decode step: seq_len=1, kv_len=5 (4 cached + 1 new)
        let mask = create_causal_mask(1, 5, DType::F32, &device)?;
        assert_eq!(mask.dims(), &[1, 1, 1, 5]);

        // Should be able to attend to all positions (including cached)
        let mask_data: Vec<f32> = mask.flatten_all()?.to_vec1()?;
        for &v in &mask_data {
            assert_eq!(v, 0.0);
        }

        Ok(())
    }

    #[test]
    fn test_tree_attention_mask_shape() -> Result<()> {
        let device = Device::Cpu;
        // chain: 0 -> 1 -> 2  (linear), prefix_kv_len = 2
        let parents = vec![None, Some(0), Some(1)];
        let mask = create_tree_attention_mask(&parents, 2, DType::F32, &device)?;
        assert_eq!(mask.dims(), &[1, 1, 3, 5]);
        let m: Vec<f32> = mask.flatten_all()?.to_vec1()?;
        // Row 0: prefix [0,1] = 0; cand cols [0]=self -> 0; [1,2]=-inf
        assert_eq!(m[0], 0.0);
        assert_eq!(m[1], 0.0);
        assert_eq!(m[2], 0.0);
        assert!(m[3].is_infinite());
        assert!(m[4].is_infinite());
        // Row 1: prefix=0; cand cols [0,1]=ancestors -> 0; [2]=-inf
        assert_eq!(m[5], 0.0);
        assert_eq!(m[5 + 1], 0.0);
        assert_eq!(m[5 + 2], 0.0);
        assert_eq!(m[5 + 3], 0.0);
        assert!(m[5 + 4].is_infinite());
        // Row 2: prefix=0; cand cols [0,1,2]=full path -> all 0
        for col in 0..5 {
            assert_eq!(m[10 + col], 0.0);
        }
        Ok(())
    }

    #[test]
    fn test_tree_attention_mask_branching() -> Result<()> {
        let device = Device::Cpu;
        // tree:
        //     root
        //      |
        //   0 (tok)
        //    /   \
        //   1     2
        let parents = vec![None, Some(0), Some(0)];
        let mask = create_tree_attention_mask(&parents, 0, DType::F32, &device)?;
        assert_eq!(mask.dims(), &[1, 1, 3, 3]);
        let m: Vec<f32> = mask.flatten_all()?.to_vec1()?;
        // Node 1 should NOT attend to node 2 (its sibling).
        // Layout: row*3 + col
        // Row 1 (node 1): col 0 = parent (allowed), col 1 = self (allowed), col 2 = sibling (FORBIDDEN)
        assert_eq!(m[3], 0.0);
        assert_eq!(m[3 + 1], 0.0);
        assert!(m[3 + 2].is_infinite());
        // Row 2 (node 2): col 0 = parent, col 1 = sibling, col 2 = self
        assert_eq!(m[6], 0.0);
        assert!(m[6 + 1].is_infinite());
        assert_eq!(m[6 + 2], 0.0);
        Ok(())
    }

    #[test]
    fn test_tree_attention_mask_changes_sibling_logits() -> Result<()> {
        let device = Device::Cpu;
        let parents = vec![None, Some(0), Some(0)];
        let tree_mask = create_tree_attention_mask(&parents, 0, DType::F32, &device)?;
        let no_op_mask = Tensor::zeros((1, 1, 3, 3), DType::F32, &device)?;

        let q = Tensor::from_vec(
            vec![
                0.0f32, 0.0, // node 0
                1.0, 0.0, // node 1 strongly matches sibling key
                0.0, 0.0, // node 2
            ],
            (1, 1, 3, 2),
            &device,
        )?;
        let k = Tensor::from_vec(
            vec![
                0.0f32, 0.0, // node 0
                0.0, 0.0, // node 1
                10.0, 0.0, // node 2 sibling
            ],
            (1, 1, 3, 2),
            &device,
        )?;
        let v = Tensor::from_vec(
            vec![
                0.0f32, 0.0, // node 0
                0.0, 0.0, // node 1
                100.0, 0.0, // node 2 sibling
            ],
            (1, 1, 3, 2),
            &device,
        )?;

        let masked = scaled_dot_product_attention(&q, &k, &v, Some(&tree_mask), 1.0, false)?;
        let unmasked = scaled_dot_product_attention(&q, &k, &v, Some(&no_op_mask), 1.0, false)?;

        let masked_node_1: Vec<f32> = masked.i((0, 0, 1, ..))?.to_vec1()?;
        let unmasked_node_1: Vec<f32> = unmasked.i((0, 0, 1, ..))?.to_vec1()?;
        assert!(
            unmasked_node_1[0] - masked_node_1[0] > 90.0,
            "sibling unmask should materially change node logits: masked={masked_node_1:?}, unmasked={unmasked_node_1:?}"
        );

        Ok(())
    }

    #[test]
    fn test_tree_attention_mask_empty() -> Result<()> {
        let device = Device::Cpu;
        let mask = create_tree_attention_mask(&[], 4, DType::F32, &device)?;
        assert_eq!(mask.dims(), &[1, 1, 0, 4]);
        Ok(())
    }

    /// Reference implementation kept from the pre-optimization revision
    /// (the `Vec<Vec<bool>>` ancestor-bitset version). Lets the parity test
    /// compare the optimized parent-walk path against the original
    /// O(N²)-bitset path across a battery of randomly shaped trees.
    fn create_tree_attention_mask_reference(
        parents: &[Option<usize>],
        prefix_kv_len: usize,
        dtype: DType,
        device: &Device,
    ) -> Result<Tensor> {
        let n = parents.len();
        let total_kv = prefix_kv_len + n;
        if n == 0 {
            return Tensor::zeros((1, 1, 0, total_kv), dtype, device);
        }
        let mut ancestors: Vec<Vec<bool>> = vec![vec![false; n]; n];
        for (i, row) in ancestors.iter_mut().enumerate() {
            let mut cur = Some(i);
            while let Some(j) = cur {
                row[j] = true;
                cur = parents[j];
            }
        }
        let mut buf = vec![f32::NEG_INFINITY; n * total_kv];
        for (row_buf, ancestor_row) in buf.chunks_mut(total_kv).zip(ancestors.iter()) {
            row_buf[..prefix_kv_len].fill(0.0);
            for (j, &is_anc) in ancestor_row.iter().enumerate() {
                if is_anc {
                    row_buf[prefix_kv_len + j] = 0.0;
                }
            }
        }
        Tensor::from_vec(buf, (1, 1, n, total_kv), device)?.to_dtype(dtype)
    }

    #[test]
    fn test_tree_attention_mask_matches_reference_on_varied_shapes() -> Result<()> {
        let device = Device::Cpu;

        // Deterministic pseudo-random tree shapes (we hand-pick parents so the
        // test is reproducible without a seeded RNG dependency).
        let shapes: Vec<(Vec<Option<usize>>, usize)> = vec![
            // Linear chain, no prefix.
            (vec![None, Some(0), Some(1), Some(2)], 0),
            // Linear chain, with prefix.
            (vec![None, Some(0), Some(1)], 7),
            // Branching tree, with prefix.
            (
                vec![
                    None,
                    Some(0),
                    Some(0),
                    Some(1),
                    Some(1),
                    Some(2),
                    Some(2),
                    Some(2),
                ],
                12,
            ),
            // Multi-root forest, no prefix.
            (vec![None, None, Some(0), Some(1), Some(2)], 0),
            // Single root, deep+wide, with prefix matching a realistic HSD
            // size: ~16 candidates, 4096-token prefix.
            (
                {
                    let mut p = vec![None];
                    for i in 1..16 {
                        // alternate root-children and chains, like an
                        // expanded prefix-tree from many drafts.
                        p.push(if i % 4 == 0 { None } else { Some(i - 1) });
                    }
                    p
                },
                4096,
            ),
        ];

        for (parents, prefix_kv_len) in shapes {
            let opt = create_tree_attention_mask(&parents, prefix_kv_len, DType::F32, &device)?;
            let reference =
                create_tree_attention_mask_reference(&parents, prefix_kv_len, DType::F32, &device)?;
            assert_eq!(
                opt.dims(),
                reference.dims(),
                "shape mismatch for parents={:?}",
                parents
            );
            let opt_vals: Vec<f32> = opt.flatten_all()?.to_vec1()?;
            let ref_vals: Vec<f32> = reference.flatten_all()?.to_vec1()?;
            assert_eq!(opt_vals.len(), ref_vals.len());
            for (i, (&a, &b)) in opt_vals.iter().zip(ref_vals.iter()).enumerate() {
                let same = match (a.is_infinite(), b.is_infinite()) {
                    (true, true) => a.is_sign_negative() == b.is_sign_negative(),
                    (false, false) => (a - b).abs() < 1e-6,
                    _ => false,
                };
                assert!(
                    same,
                    "mismatch at idx {i} for parents={:?} prefix={}: opt={a} ref={b}",
                    parents, prefix_kv_len
                );
            }
        }

        Ok(())
    }

    #[test]
    fn test_repeat_kv() -> Result<()> {
        let device = Device::Cpu;
        let x = Tensor::randn(0f32, 1., (1, 4, 8, 64), &device)?;

        // n_rep = 1, should return same tensor
        let out = repeat_kv(&x, 1)?;
        assert_eq!(out.dims(), &[1, 4, 8, 64]);

        // n_rep = 2, should double heads
        let out = repeat_kv(&x, 2)?;
        assert_eq!(out.dims(), &[1, 8, 8, 64]);

        Ok(())
    }

    #[test]
    fn test_padding_mask() -> Result<()> {
        let device = Device::Cpu;
        let seq_lens = vec![3, 5, 2];
        let max_len = 5;

        let mask = create_padding_mask(&seq_lens, max_len, DType::F32, &device)?;
        assert_eq!(mask.dims(), &[3, 1, 1, 5]);

        let mask_data: Vec<f32> = mask.flatten_all()?.to_vec1()?;

        // Batch 0: len=3, positions 0-2 valid, 3-4 masked
        assert_eq!(mask_data[0], 0.0);
        assert_eq!(mask_data[1], 0.0);
        assert_eq!(mask_data[2], 0.0);
        assert!(mask_data[3].is_infinite());
        assert!(mask_data[4].is_infinite());

        // Batch 1: len=5, all valid
        assert_eq!(mask_data[5], 0.0);
        assert_eq!(mask_data[6], 0.0);
        assert_eq!(mask_data[7], 0.0);
        assert_eq!(mask_data[8], 0.0);
        assert_eq!(mask_data[9], 0.0);

        Ok(())
    }

    #[test]
    fn test_left_padding_mask() -> Result<()> {
        let device = Device::Cpu;
        let seq_lens = vec![3, 5, 2];
        let max_len = 5;

        let mask = create_left_padding_mask(&seq_lens, max_len, DType::F32, &device)?;
        assert_eq!(mask.dims(), &[3, 1, 1, 5]);

        let mask_data: Vec<f32> = mask.flatten_all()?.to_vec1()?;

        // Batch 0: len=3, left-padded by 2 -> positions 0-1 masked, 2-4 valid
        assert!(mask_data[0].is_infinite());
        assert!(mask_data[1].is_infinite());
        assert_eq!(mask_data[2], 0.0);
        assert_eq!(mask_data[3], 0.0);
        assert_eq!(mask_data[4], 0.0);

        // Batch 1: len=5, no padding, all valid
        assert_eq!(mask_data[5], 0.0);
        assert_eq!(mask_data[6], 0.0);
        assert_eq!(mask_data[7], 0.0);
        assert_eq!(mask_data[8], 0.0);
        assert_eq!(mask_data[9], 0.0);

        // Batch 2: len=2, left-padded by 3 -> positions 0-2 masked, 3-4 valid
        assert!(mask_data[10].is_infinite());
        assert!(mask_data[11].is_infinite());
        assert!(mask_data[12].is_infinite());
        assert_eq!(mask_data[13], 0.0);
        assert_eq!(mask_data[14], 0.0);

        Ok(())
    }

    // ========================================================================
    // RoPE Tests
    // ========================================================================

    #[test]
    fn test_rotary_embedding_dynamic_single_axis() -> std::result::Result<(), OCRError> {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new_dynamic(64, 10000.0, &device)?;

        // Create position IDs: (1, batch, seq)
        let position_ids = Tensor::arange(0u32, 8u32, &device)
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "Failed to create position_ids",
                    e,
                )
            })?
            .reshape((1, 1, 8))
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "Failed to reshape position_ids",
                    e,
                )
            })?;

        let (cos, sin) = rope.forward_multi_axis(&position_ids, DType::F32)?;
        assert_eq!(cos.dims(), &[1, 1, 8, 64]); // (num_dims, batch, seq, head_dim)
        assert_eq!(sin.dims(), &[1, 1, 8, 64]);

        Ok(())
    }

    #[test]
    fn test_rotary_embedding_multi_axis() -> std::result::Result<(), OCRError> {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new_multi_axis(128, 10000.0, 3, &device)?;

        // Create 3-axis position IDs: (3, batch, seq)
        let position_ids = Tensor::zeros((3, 2, 16), DType::U32, &device).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "Failed to create position_ids",
                e,
            )
        })?;

        let (cos, sin) = rope.forward_multi_axis(&position_ids, DType::F16)?;
        assert_eq!(cos.dims(), &[3, 2, 16, 128]); // (num_dims, batch, seq, head_dim)
        assert_eq!(sin.dims(), &[3, 2, 16, 128]);
        assert_eq!(cos.dtype(), DType::F16);

        Ok(())
    }

    #[test]
    fn test_rotary_embedding_invalid_head_dim() {
        let device = Device::Cpu;
        // Odd head_dim should fail
        let result = RotaryEmbedding::new_multi_axis(63, 10000.0, 1, &device);
        assert!(result.is_err());
        if let Err(OCRError::ConfigError { message }) = result {
            assert!(message.contains("must be even"));
        } else {
            panic!("Expected ConfigError");
        }
    }

    #[test]
    fn test_rotary_embedding_wrong_position_ids_shape() -> std::result::Result<(), OCRError> {
        let device = Device::Cpu;
        let rope = RotaryEmbedding::new_multi_axis(64, 10000.0, 3, &device)?;

        // Wrong shape: (2, batch, seq) instead of (3, batch, seq)
        let position_ids = Tensor::zeros((2, 2, 16), DType::U32, &device).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "Failed to create position_ids",
                e,
            )
        })?;

        let result = rope.forward_multi_axis(&position_ids, DType::F32);
        assert!(result.is_err());
        if let Err(OCRError::InvalidInput { message }) = result {
            assert!(message.contains("expected position_ids shape (3"));
        } else {
            panic!("Expected InvalidInput error");
        }

        Ok(())
    }
}
