//! Unified attention and rotary embedding shared by all VLM models
//! (PaddleOCR-VL, HunyuanOCR, GLM-OCR, MinerU2.5), for consistent mask handling,
//! KV-cache logic, and multi-axis RoPE (MRoPE, XDRoPE).
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

/// Scaled dot-product attention for grouped-query attention without expanding
/// K/V heads. Query heads that share one KV head are folded into the matrix row
/// dimension, preserving the usual head order in the returned tensor.
pub fn scaled_dot_product_attention_gqa(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    mask: Option<&Tensor>,
    scale: f64,
    is_causal: bool,
    num_kv_groups: usize,
) -> Result<Tensor> {
    if num_kv_groups == 1 {
        return scaled_dot_product_attention(q, k, v, mask, scale, is_causal);
    }

    let (batch, num_heads, query_len, head_dim) = q.dims4()?;
    let (k_batch, num_kv_heads, kv_len, k_head_dim) = k.dims4()?;
    let (v_batch, v_heads, v_len, v_head_dim) = v.dims4()?;
    if batch != k_batch
        || batch != v_batch
        || num_heads != num_kv_heads * num_kv_groups
        || num_kv_heads != v_heads
        || kv_len != v_len
        || head_dim != k_head_dim
        || head_dim != v_head_dim
    {
        candle_core::bail!(
            "invalid GQA shapes q={:?}, k={:?}, v={:?}, groups={num_kv_groups}",
            q.dims(),
            k.dims(),
            v.dims()
        )
    }

    let grouped_batch = batch * num_kv_heads;
    let grouped_queries = num_kv_groups * query_len;
    let grouped_q = q.reshape((grouped_batch, grouped_queries, head_dim))?;
    let grouped_k = k
        .reshape((grouped_batch, kv_len, head_dim))?
        .transpose(1, 2)?;
    let mut weights =
        (grouped_q.matmul(&grouped_k)? * scale)?.reshape((batch, num_heads, query_len, kv_len))?;

    weights = match mask {
        Some(mask) => weights.broadcast_add(mask)?,
        None if is_causal => {
            let causal = create_causal_mask(query_len, kv_len, weights.dtype(), q.device())?;
            weights.broadcast_add(&causal)?
        }
        None => weights,
    };

    let weight_dtype = weights.dtype();
    let weights = candle_nn::ops::softmax_last_dim(&weights.to_dtype(DType::F32)?)?
        .to_dtype(weight_dtype)?
        .reshape((grouped_batch, grouped_queries, kv_len))?;
    let grouped_v = v.reshape((grouped_batch, kv_len, head_dim))?;
    weights
        .matmul(&grouped_v)?
        .reshape((batch, num_heads, query_len, head_dim))
}

/// Run CUDA FlashAttention v2 for Q/K/V tensors in `(batch, heads, seq,
/// head_dim)` layout. Returns `None` on non-CUDA devices so callers can retain
/// their portable eager fallback.
pub fn flash_attention(
    q: &Tensor,
    k: &Tensor,
    v: &Tensor,
    scale: f64,
    causal: bool,
) -> Result<Option<Tensor>> {
    #[cfg(feature = "cuda")]
    if q.device().is_cuda() {
        // The CUDA kernel consumes (batch, seq, heads, head_dim) and natively
        // supports GQA when K/V have fewer heads than Q.
        let q = q.transpose(1, 2)?;
        let k = k.transpose(1, 2)?;
        let v = v.transpose(1, 2)?;
        let output = candle_flash_attn::flash_attn(&q, &k, &v, scale as f32, causal)?;
        return Ok(Some(output.transpose(1, 2)?));
    }

    let _ = (q, k, v, scale, causal);
    Ok(None)
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
        let row_idx =
            Tensor::arange(0u32, seq_len as u32, compute_device)?.reshape((seq_len, 1))?;
        let col_idx = Tensor::arange(0u32, kv_len as u32, compute_device)?.reshape((1, kv_len))?;

        let offset = kv_len.saturating_sub(seq_len) as u32;
        // Condition: col <= row + offset
        // Keep this comparison in integer space. BF16 cannot distinguish
        // adjacent absolute positions once a document context grows beyond
        // 256 tokens, which would let verification queries see future draft
        // tokens and invalidate speculative decoding.
        let row_limit = row_idx.broadcast_add(&Tensor::new(offset, compute_device)?)?;
        let mask_cond = col_idx.broadcast_le(&row_limit)?;

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

/// Create a left-padding mask for batched sequences (right-aligned, standard for
/// autoregressive generation).
///
/// Returns a `(batch, 1, 1, max_len)` mask where left-padded positions
/// (`j < max_len - seq_len`) are `-inf` and valid positions are `0`. With
/// `seq_lens = [3, 5]` and `max_len = 5`, item 0 is `[-inf, -inf, 0, 0, 0]`.
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

/// Builds the per-step decode attention mask for a left-padded batch.
///
/// Masks out the leading `pad_lens[i]` padding positions of each row so the new
/// token never attends to padding KV (which would corrupt unequal-length
/// batches). Returns a `(batch, 1, 1, kv_len)` additive mask (`0` attendable, a
/// large negative for padding); a no-op when there is no padding.
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

        let zero = Tensor::zeros(mask_cond.shape(), dtype, compute_device)?;
        // Use large negative value instead of -inf to avoid potential numerical issues
        let mask_value =
            Tensor::full(-1e9_f32, mask_cond.shape(), compute_device)?.to_dtype(dtype)?;

        mask_cond.where_cond(&mask_value, &zero)
    })
}

// Rotary Positional Embedding (RoPE)

/// Unified Rotary Positional Embedding supporting single-axis RoPE, MRoPE
/// (3-axis text/height/width, PaddleOCR-VL), and XDRoPE (configurable `num_dims`,
/// HunyuanOCR). All variants share one `Dynamic` representation parameterized by
/// `num_dims`; constructors pick the right value per model family.
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
    /// Create a dynamic single-axis RoPE computed on-the-fly from position IDs
    /// (suitable for variable-length sequences).
    ///
    /// `head_dim` must be even; `rope_theta` is the base frequency (typically 10000.0).
    pub fn new_dynamic(
        head_dim: usize,
        rope_theta: f64,
        device: &Device,
    ) -> std::result::Result<Self, OCRError> {
        Self::new_multi_axis(head_dim, rope_theta, 1, device)
    }

    /// Create a multi-axis RoPE (MRoPE/XDRoPE) with `num_dims` position
    /// dimensions (1 = standard, 3 = MRoPE text/height/width).
    ///
    /// `head_dim` must be even; `rope_theta` is the base frequency (typically 10000.0).
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

/// Select and combine RoPE sections for multi-axis encoding (MRoPE 3-axis,
/// XDRoPE 4-axis): different `head_dim` sections take different position
/// dimensions, encoding spatial (height, width) and temporal positions separately.
///
/// `cos_or_sin` is `(num_dims, batch, seq, head_dim)`, `rope_section` sums to
/// `head_dim/2`; returns `(batch, 1, seq, head_dim)`.
///
/// # Example
/// MRoPE with `rope_section=[16, 24, 24]`, `head_dim=128`: sections double to
/// `[16, 24, 24, 16, 24, 24]`, each picking dim `i % 3`, concatenated along `head_dim`.
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
    fn test_grouped_query_attention_matches_repeated_kv() -> Result<()> {
        let device = Device::Cpu;
        let q = Tensor::randn(0f32, 1., (1, 4, 3, 8), &device)?;
        let k = Tensor::randn(0f32, 1., (1, 2, 5, 8), &device)?;
        let v = Tensor::randn(0f32, 1., (1, 2, 5, 8), &device)?;
        let mask = create_causal_mask(3, 5, DType::F32, &device)?;
        let scale = 1.0 / (8f64).sqrt();

        let repeated = scaled_dot_product_attention(
            &q,
            &repeat_kv(&k, 2)?,
            &repeat_kv(&v, 2)?,
            Some(&mask),
            scale,
            false,
        )?;
        let grouped = scaled_dot_product_attention_gqa(&q, &k, &v, Some(&mask), scale, false, 2)?;
        let repeated = repeated.flatten_all()?.to_vec1::<f32>()?;
        let grouped = grouped.flatten_all()?.to_vec1::<f32>()?;
        assert!(
            repeated
                .iter()
                .zip(grouped)
                .all(|(left, right)| (left - right).abs() < 1e-5)
        );
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
    fn test_bf16_causal_mask_preserves_adjacent_positions_in_long_context() -> Result<()> {
        let device = Device::Cpu;
        let query_len = 16;
        let kv_len = 2048;
        let context_len = kv_len - query_len;
        let mask = create_causal_mask(query_len, kv_len, DType::BF16, &device)?
            .to_dtype(DType::F32)?
            .flatten_all()?
            .to_vec1::<f32>()?;

        for row in 0..query_len {
            let start = row * kv_len;
            let last_visible = context_len + row;
            assert_eq!(mask[start + last_visible], 0.0);
            if last_visible + 1 < kv_len {
                assert!(mask[start + last_visible + 1].is_infinite());
                assert!(mask[start + last_visible + 1].is_sign_negative());
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

    // RoPE Tests

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
