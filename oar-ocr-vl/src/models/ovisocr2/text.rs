//! Qwen3.5 text decoder used by OvisOCR2.
//!
//! Qwen3.5 alternates three Gated DeltaNet layers with one full-attention
//! layer. Its decoder RMSNorm checkpoints are zero-centred (`1 + weight`),
//! while Gated DeltaNet's internal gated RMSNorm is conventionally centred at
//! one. Its multimodal RoPE frequencies are interleaved T/H/W.

use super::config::OvisOcr2TextConfig;
use super::gated_delta::gated_delta_rule;
use crate::attention::{RotaryEmbedding, flash_attention, scaled_dot_product_attention_gqa};
use crate::error::Error;
#[cfg(feature = "cuda")]
use crate::runtime::attention::masked_score;
use crate::runtime::cache::TrimmableKvCache;
#[cfg(feature = "cuda")]
use crate::runtime::cuda::dynamic_kv::DynamicKvAppend;
#[cfg(feature = "cuda")]
use crate::runtime::decoder_graph::{
    CudaGraphDrainGuard, CudaGraphKvLengths, DecoderCudaGraph, DecoderGraphInputs,
    capture_decoder_graph, cuda_graph_error, drain_cuda_context_errors, drop_and_drain,
    next_decode_bucket, prompt_decode_bucket,
};
use crate::utils::{candle_to_ocr_inference, rotate_half};
#[cfg(feature = "cuda")]
use candle_core::IndexOp;
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{
    Conv1d, Conv1dConfig, Embedding, Linear, Module, RmsNorm, VarBuilder, embedding,
    linear_no_bias, rms_norm,
};
use std::cell::RefCell;

const MODEL_NAME: &str = "OvisOCR2";

#[derive(Debug, Clone)]
struct AdditiveRmsNorm {
    weight: Tensor,
    eps: f64,
}

impl AdditiveRmsNorm {
    fn load(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self, Error> {
        let weight = vb
            .get(dim, "weight")
            .and_then(|weight| weight.to_dtype(DType::F32))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load additive RMSNorm", e))?;
        Ok(Self { weight, eps })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, Error> {
        let dtype = xs.dtype();
        let xs = xs
            .to_dtype(DType::F32)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "RMSNorm input cast", e))?;
        let variance = xs
            .sqr()
            .and_then(|xs| xs.mean_keepdim(D::Minus1))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "RMSNorm variance", e))?;
        let normalized = xs
            .broadcast_div(
                &(variance + self.eps)
                    .and_then(|variance| variance.sqrt())
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "RMSNorm rsqrt", e))?,
            )
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "RMSNorm normalize", e))?;
        let scale = (&self.weight + 1.0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "RMSNorm scale", e))?;
        normalized
            .broadcast_mul(&scale)
            .and_then(|xs| xs.to_dtype(dtype))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "RMSNorm output", e))
    }
}

#[derive(Debug, Clone)]
struct OvisMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl OvisMlp {
    fn load(cfg: &OvisOcr2TextConfig, vb: VarBuilder) -> Result<Self, Error> {
        let gate_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load MLP gate_proj", e))?;
        let up_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load MLP up_proj", e))?;
        let down_proj = linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load MLP down_proj", e))?;
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
            .and_then(|gate| candle_nn::ops::silu(&gate))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MLP gate", e))?;
        let up = self
            .up_proj
            .forward(xs)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MLP up", e))?;
        self.down_proj
            .forward(
                &(&gate * &up)
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MLP gate product", e))?,
            )
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "MLP down", e))
    }
}

#[derive(Debug)]
struct GatedDeltaNet {
    in_proj_qkv: Linear,
    in_proj_z: Linear,
    in_proj_b: Linear,
    in_proj_a: Linear,
    conv1d: Conv1d,
    decode_conv_weight: Tensor,
    dt_bias: Tensor,
    neg_a: Tensor,
    norm: RmsNorm,
    out_proj: Linear,
    num_key_heads: usize,
    num_value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    conv_kernel_size: usize,
    conv_state: RefCell<Option<Tensor>>,
    recurrent_state: RefCell<Option<Tensor>>,
}

fn cached_depthwise_conv_step(
    state: &Tensor,
    mixed: &Tensor,
    weight: &Tensor,
    kernel_size: usize,
) -> candle_core::Result<(Tensor, Tensor)> {
    let tail = state.narrow(2, 1, kernel_size - 1)?;
    let new_state = Tensor::cat(&[&tail, mixed], 2)?;
    let output = match mixed.dtype() {
        DType::BF16 | DType::F16 => new_state
            .to_dtype(DType::F32)?
            .broadcast_mul(weight)?
            .sum_keepdim(2)?
            .to_dtype(mixed.dtype())?,
        _ => new_state.broadcast_mul(weight)?.sum_keepdim(2)?,
    };
    Ok((output, new_state))
}

/// Store the next Gated DeltaNet state, writing in place when the slot
/// already holds a buffer of the same shape. Decode steps therefore keep one
/// fixed allocation per state, which the CUDA-graph capture can safely hold
/// a pointer to, and an eager fallback after a replay reads the values the
/// graph last wrote — one buffer, no synchronization between the two paths.
///
/// The slot must hold a contiguous tensor from the very first write:
/// `slice_set` requires both sides contiguous, and producers such as
/// `Tensor::narrow` hand in non-contiguous views. Replacing the buffer on a
/// later non-contiguous write is not an option — a captured graph keeps a
/// pointer to this exact allocation — so contiguity is enforced here, at
/// the funnel through which every state write passes.
fn store_state(
    slot: &RefCell<Option<Tensor>>,
    new_state: Tensor,
    context: &'static str,
) -> Result<(), Error> {
    // A no-op clone when `new_state` is already contiguous.
    let new_state = new_state
        .contiguous()
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, context, e))?;
    let mut borrow = slot.borrow_mut();
    if let Some(existing) = borrow.as_ref()
        && existing.shape() == new_state.shape()
    {
        // Guaranteed contiguous: the slot is only ever filled above.
        debug_assert!(existing.is_contiguous());
        existing
            .slice_set(&new_state, 0, 0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, context, e))?;
        return Ok(());
    }
    *borrow = Some(new_state);
    Ok(())
}

impl GatedDeltaNet {
    fn load(cfg: &OvisOcr2TextConfig, vb: VarBuilder) -> Result<Self, Error> {
        let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
        let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        if !cfg
            .linear_num_value_heads
            .is_multiple_of(cfg.linear_num_key_heads)
        {
            return Err(Error::Config {
                message: format!(
                    "OvisOCR2: linear_num_value_heads ({}) must be divisible by linear_num_key_heads ({})",
                    cfg.linear_num_value_heads, cfg.linear_num_key_heads
                ),
            });
        }
        if cfg.linear_key_head_dim != cfg.linear_value_head_dim {
            return Err(Error::Config {
                message: format!(
                    "OvisOCR2 currently requires equal Gated DeltaNet key/value head dims, got {}/{}",
                    cfg.linear_key_head_dim, cfg.linear_value_head_dim
                ),
            });
        }

        let in_proj_qkv = linear_no_bias(cfg.hidden_size, conv_dim, vb.pp("in_proj_qkv"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load GDN in_proj_qkv", e))?;
        let in_proj_z = linear_no_bias(cfg.hidden_size, value_dim, vb.pp("in_proj_z"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load GDN in_proj_z", e))?;
        let in_proj_b = linear_no_bias(
            cfg.hidden_size,
            cfg.linear_num_value_heads,
            vb.pp("in_proj_b"),
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load GDN in_proj_b", e))?;
        let in_proj_a = linear_no_bias(
            cfg.hidden_size,
            cfg.linear_num_value_heads,
            vb.pp("in_proj_a"),
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load GDN in_proj_a", e))?;
        let conv_weight = vb
            .get((conv_dim, 1, cfg.linear_conv_kernel_dim), "conv1d.weight")
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load GDN conv1d", e))?;
        let decode_conv_weight = conv_weight
            .to_dtype(DType::F32)
            .and_then(|weight| weight.squeeze(1))
            .and_then(|weight| weight.unsqueeze(0))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load GDN decode conv weight", e))?;
        let conv1d = Conv1d::new(
            conv_weight,
            None,
            Conv1dConfig {
                padding: cfg.linear_conv_kernel_dim.saturating_sub(1),
                groups: conv_dim,
                ..Default::default()
            },
        );
        let dt_bias = vb
            .get(cfg.linear_num_value_heads, "dt_bias")
            .and_then(|x| x.to_dtype(DType::F32))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load GDN dt_bias", e))?;
        let neg_a = vb
            .get(cfg.linear_num_value_heads, "A_log")
            .and_then(|x| x.to_dtype(DType::F32))
            .and_then(|x| x.exp())
            .and_then(|x| x.neg())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load GDN A_log", e))?;
        // Qwen3_5RMSNormGated initializes this weight to one and applies a
        // plain RMSNorm before the SiLU gate. It intentionally differs from
        // the zero-centred AdditiveRmsNorm used by the decoder layers.
        let norm = rms_norm(cfg.linear_value_head_dim, cfg.rms_norm_eps, vb.pp("norm"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load GDN norm", e))?;
        let out_proj = linear_no_bias(value_dim, cfg.hidden_size, vb.pp("out_proj"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load GDN out_proj", e))?;

        Ok(Self {
            in_proj_qkv,
            in_proj_z,
            in_proj_b,
            in_proj_a,
            conv1d,
            decode_conv_weight,
            dt_bias,
            neg_a,
            norm,
            out_proj,
            num_key_heads: cfg.linear_num_key_heads,
            num_value_heads: cfg.linear_num_value_heads,
            key_head_dim: cfg.linear_key_head_dim,
            value_head_dim: cfg.linear_value_head_dim,
            conv_kernel_size: cfg.linear_conv_kernel_dim,
            conv_state: RefCell::new(None),
            recurrent_state: RefCell::new(None),
        })
    }

    fn causal_conv(&self, mixed: &Tensor) -> Result<Tensor, Error> {
        let (batch, channels, seq_len) = mixed
            .dims3()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN convolution input", e))?;
        let previous = self.conv_state.borrow().clone();
        let (output, new_state) = match previous.as_ref() {
            None => {
                let output = self
                    .conv1d
                    .forward(mixed)
                    .and_then(|output| output.narrow(2, 0, seq_len))
                    .map_err(|e| {
                        candle_to_ocr_inference(MODEL_NAME, "GDN causal convolution", e)
                    })?;
                let new_state = if seq_len >= self.conv_kernel_size {
                    mixed.narrow(2, seq_len - self.conv_kernel_size, self.conv_kernel_size)
                } else {
                    let padding = Tensor::zeros(
                        (batch, channels, self.conv_kernel_size - seq_len),
                        mixed.dtype(),
                        mixed.device(),
                    )
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "pad GDN conv state", e))?;
                    Tensor::cat(&[&padding, mixed], 2)
                }
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN update conv state", e))?;
                (output, new_state)
            }
            Some(state) if seq_len == 1 => {
                // A grouped Conv1d launch with one group per channel is very
                // expensive for autoregressive decoding. For a single token,
                // the same depthwise convolution is just a weighted sum of the
                // shifted cache and the new projection.
                cached_depthwise_conv_step(
                    state,
                    mixed,
                    &self.decode_conv_weight,
                    self.conv_kernel_size,
                )
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN decode convolution", e))?
            }
            Some(state) => {
                let joined = Tensor::cat(&[state, mixed], 2).map_err(|e| {
                    candle_to_ocr_inference(MODEL_NAME, "join GDN convolution context", e)
                })?;
                let conv = Conv1d::new(
                    self.conv1d.weight().clone(),
                    None,
                    Conv1dConfig {
                        groups: channels,
                        ..Default::default()
                    },
                );
                let output = conv
                    .forward(&joined)
                    .and_then(|output| output.narrow(2, 1, seq_len))
                    .map_err(|e| {
                        candle_to_ocr_inference(MODEL_NAME, "GDN causal convolution", e)
                    })?;
                let context_len = joined.dim(2).map_err(|e| {
                    candle_to_ocr_inference(MODEL_NAME, "GDN convolution cache length", e)
                })?;
                let new_state = joined
                    .narrow(
                        2,
                        context_len - self.conv_kernel_size,
                        self.conv_kernel_size,
                    )
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN update conv state", e))?;
                (output, new_state)
            }
        };
        store_state(&self.conv_state, new_state, "GDN store conv state")?;

        candle_nn::ops::silu(&output)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN convolution SiLU", e))
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor, Error> {
        let (batch, seq_len, _) = hidden_states
            .dims3()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN input shape", e))?;
        let mixed = self
            .in_proj_qkv
            .forward(hidden_states)
            .and_then(|x| x.transpose(1, 2))
            .and_then(|x| x.contiguous())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN qkv projection", e))?;
        let mixed = self
            .causal_conv(&mixed)?
            .transpose(1, 2)
            .and_then(|x| x.contiguous())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN qkv layout", e))?;

        let key_dim = self.num_key_heads * self.key_head_dim;
        let value_dim = self.num_value_heads * self.value_head_dim;
        let query = mixed
            .narrow(D::Minus1, 0, key_dim)
            .and_then(|x| x.reshape((batch, seq_len, self.num_key_heads, self.key_head_dim)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN query", e))?;
        let key = mixed
            .narrow(D::Minus1, key_dim, key_dim)
            .and_then(|x| x.reshape((batch, seq_len, self.num_key_heads, self.key_head_dim)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN key", e))?;
        let value = mixed
            .narrow(D::Minus1, key_dim * 2, value_dim)
            .and_then(|x| x.reshape((batch, seq_len, self.num_value_heads, self.value_head_dim)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN value", e))?;

        let repeat = self.num_value_heads / self.num_key_heads;
        let query = if repeat == 1 {
            query
        } else {
            query
                .unsqueeze(3)
                .and_then(|x| x.repeat((1, 1, 1, repeat, 1)))
                .and_then(|x| x.reshape((batch, seq_len, self.num_value_heads, self.key_head_dim)))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN repeat query heads", e))?
        };
        let key = if repeat == 1 {
            key
        } else {
            key.unsqueeze(3)
                .and_then(|x| x.repeat((1, 1, 1, repeat, 1)))
                .and_then(|x| x.reshape((batch, seq_len, self.num_value_heads, self.key_head_dim)))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN repeat key heads", e))?
        };
        let packed_qkv = Tensor::cat(&[&query, &key, &value], D::Minus1)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN pack qkv", e))?;

        let beta = self
            .in_proj_b
            .forward(hidden_states)
            .and_then(|x| candle_nn::ops::sigmoid(&x))
            .and_then(|x| x.to_dtype(DType::F32))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN beta", e))?;
        let a = self
            .in_proj_a
            .forward(hidden_states)
            .and_then(|x| x.to_dtype(DType::F32))
            .and_then(|x| x.broadcast_add(&self.dt_bias))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN decay projection", e))?;
        // Stable equivalent of log(1 + exp(a)), matching torch softplus
        // without overflowing for large positive decay logits.
        let softplus = a
            .relu()
            .and_then(|positive| {
                a.abs()
                    .and_then(|magnitude| magnitude.neg())
                    .and_then(|negative| negative.exp())
                    .and_then(|correction| correction + 1.0)
                    .and_then(|correction| correction.log())
                    .and_then(|correction| positive + correction)
            })
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN softplus", e))?;
        let g = softplus
            .broadcast_mul(&self.neg_a)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN decay", e))?;
        let gb = Tensor::stack(&[&g, &beta], D::Minus1)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN pack decay/beta", e))?;

        let initial_state = match self.recurrent_state.borrow().as_ref() {
            Some(state) => state.clone(),
            None => Tensor::zeros(
                (
                    batch,
                    self.num_value_heads,
                    self.key_head_dim,
                    self.value_head_dim,
                ),
                DType::F32,
                hidden_states.device(),
            )
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN initial state", e))?,
        };
        let (core, final_state) = gated_delta_rule(&packed_qkv, &gb, &initial_state)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN recurrence", e))?;
        store_state(
            &self.recurrent_state,
            final_state,
            "GDN store recurrent state",
        )?;

        let z = self
            .in_proj_z
            .forward(hidden_states)
            .and_then(|x| x.reshape((batch, seq_len, self.num_value_heads, self.value_head_dim)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN z projection", e))?;
        let core = self
            .norm
            .forward(&core)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN output norm", e))?;
        let gate = z
            .to_dtype(DType::F32)
            .and_then(|z| candle_nn::ops::silu(&z))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN output gate", e))?;
        let core = core
            .to_dtype(DType::F32)
            .and_then(|core| core.broadcast_mul(&gate))
            .and_then(|core| core.to_dtype(hidden_states.dtype()))
            .and_then(|core| core.reshape((batch, seq_len, value_dim)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN gated output", e))?;
        self.out_proj
            .forward(&core)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "GDN output projection", e))
    }

    /// Release the state buffers. Under CUDA a captured decode graph has
    /// referenced them (it writes them in place during replay), so they go
    /// through the drain path rather than a plain drop — unconditionally,
    /// since the drain is a no-op when nothing was ever captured.
    #[cfg(feature = "cuda")]
    fn clear_cache(&self) {
        for slot in [&self.conv_state, &self.recurrent_state] {
            if let Some(state) = slot.borrow_mut().take() {
                let device = state.device().clone();
                drop_and_drain(state, &device);
            }
        }
    }

    #[cfg(not(feature = "cuda"))]
    fn clear_cache(&self) {
        *self.conv_state.borrow_mut() = None;
        *self.recurrent_state.borrow_mut() = None;
    }
}

#[derive(Debug)]
struct FullAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: AdditiveRmsNorm,
    k_norm: AdditiveRmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
    kv_cache: RefCell<TrimmableKvCache>,
}

impl FullAttention {
    fn load(cfg: &OvisOcr2TextConfig, vb: VarBuilder) -> Result<Self, Error> {
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            return Err(Error::Config {
                message: format!(
                    "OvisOCR2: num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                    cfg.num_attention_heads, cfg.num_key_value_heads
                ),
            });
        }
        let q_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.head_dim * 2,
            vb.pp("q_proj"),
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load attention q_proj", e))?;
        let k_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            vb.pp("k_proj"),
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load attention k_proj", e))?;
        let v_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            vb.pp("v_proj"),
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load attention v_proj", e))?;
        let o_proj = linear_no_bias(
            cfg.num_attention_heads * cfg.head_dim,
            cfg.hidden_size,
            vb.pp("o_proj"),
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load attention o_proj", e))?;
        let q_norm = AdditiveRmsNorm::load(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))?;
        let k_norm = AdditiveRmsNorm::load(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            scaling: 1.0 / (cfg.head_dim as f64).sqrt(),
            kv_cache: RefCell::new(TrimmableKvCache::new(2, cfg.max_position_embeddings)),
        })
    }

    fn apply_rope(&self, tensor: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor, Error> {
        let rotary_dim = cos
            .dim(D::Minus1)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention rotary dimension", e))?;
        let rotary = tensor
            .narrow(D::Minus1, 0, rotary_dim)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention rotary slice", e))?;
        let pass = tensor
            .narrow(D::Minus1, rotary_dim, self.head_dim - rotary_dim)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention pass slice", e))?;
        let cos = cos
            .unsqueeze(1)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention cos layout", e))?;
        let sin = sin
            .unsqueeze(1)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention sin layout", e))?;
        let rotated = rotate_half(&rotary)?;
        let embedded = (rotary
            .broadcast_mul(&cos)
            .and_then(|lhs| rotated.broadcast_mul(&sin).and_then(|rhs| &lhs + &rhs)))
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "apply attention RoPE", e))?;
        Tensor::cat(&[&embedded, &pass], D::Minus1)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention RoPE output", e))
    }

    fn forward(&self, hidden_states: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor, Error> {
        let (batch, seq_len, _) = hidden_states
            .dims3()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention input", e))?;
        let qg = self
            .q_proj
            .forward(hidden_states)
            .and_then(|x| x.reshape((batch, seq_len, self.num_heads, self.head_dim * 2)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention q/g projection", e))?;
        let q = qg
            .narrow(D::Minus1, 0, self.head_dim)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention q slice", e))?;
        let gate = qg
            .narrow(D::Minus1, self.head_dim, self.head_dim)
            .and_then(|x| x.reshape((batch, seq_len, self.num_heads * self.head_dim)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention gate slice", e))?;
        let q = self
            .q_norm
            .forward(&q)?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention q layout", e))?;
        let k = self
            .k_proj
            .forward(hidden_states)
            .and_then(|x| x.reshape((batch, seq_len, self.num_kv_heads, self.head_dim)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention k projection", e))?;
        let k = self
            .k_norm
            .forward(&k)?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention k layout", e))?;
        let v = self
            .v_proj
            .forward(hidden_states)
            .and_then(|x| x.reshape((batch, seq_len, self.num_kv_heads, self.head_dim)))
            .and_then(|x| x.transpose(1, 2))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention v projection", e))?;
        let q = self
            .apply_rope(&q, cos, sin)?
            .contiguous()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention q contiguous", e))?;
        let k = self
            .apply_rope(&k, cos, sin)?
            .contiguous()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention k contiguous", e))?;
        let v = v
            .contiguous()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention v contiguous", e))?;
        let (k, v) = self
            .kv_cache
            .borrow_mut()
            .append(&k, &v)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention KV cache", e))?;

        let output = match flash_attention(&q, &k, &v, self.scaling, seq_len > 1)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "flash attention", e))?
        {
            Some(output) => output,
            None => scaled_dot_product_attention_gqa(
                &q,
                &k,
                &v,
                None,
                self.scaling,
                true,
                self.num_kv_groups,
            )
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "grouped-query attention", e))?,
        };
        let output = output
            .transpose(1, 2)
            .and_then(|x| x.reshape((batch, seq_len, self.num_heads * self.head_dim)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention output layout", e))?;
        let gate = candle_nn::ops::sigmoid(&gate)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention output gate", e))?;
        self.o_proj
            .forward(
                &(&output * &gate).map_err(|e| {
                    candle_to_ocr_inference(MODEL_NAME, "attention gated output", e)
                })?,
            )
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention output projection", e))
    }

    fn clear_cache(&self) {
        self.kv_cache.borrow_mut().reset();
    }

    /// Bring the fixed-capacity KV storage to `cache_len`, preserving the
    /// appended history. Runs after the (eager) prefill, before capture.
    #[cfg(feature = "cuda")]
    fn prepare_dynamic_cache(&self, cache_len: usize) -> Result<(), Error> {
        let template = Tensor::zeros(
            (1, self.num_kv_heads, 1, self.head_dim),
            self.q_proj.weight().dtype(),
            self.q_proj.weight().device(),
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "dynamic KV template", e))?;
        let released = self
            .kv_cache
            .borrow_mut()
            .grow_fixed_storage(&template, cache_len)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "prepare dynamic KV", e))?;
        // The replaced bucket was referenced by the graph disposed before
        // this growth: release it through the drain path, not a plain drop.
        if let Some((k, v)) = released {
            let device = template.device();
            drop_and_drain(k, device);
            drop_and_drain(v, device);
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn kv_cache_len(&self) -> usize {
        self.kv_cache.borrow().current_seq_len()
    }

    #[cfg(feature = "cuda")]
    fn set_kv_cache_len(&self, len: usize) -> Result<(), Error> {
        self.kv_cache
            .borrow_mut()
            .set_current_len(len)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "set dynamic KV length", e))
    }

    /// Shrink the fixed KV bucket back to the organic eager form after a
    /// failed capture, preserving the live history and freeing the spare
    /// capacity next to a context drain.
    #[cfg(feature = "cuda")]
    fn release_dynamic_cache(&self, device: &Device) -> Result<(), Error> {
        match self
            .kv_cache
            .borrow_mut()
            .shrink_fixed_storage_preserving_history()
        {
            Ok(Some((k, v))) => {
                drop_and_drain(k, device);
                drop_and_drain(v, device);
            }
            Ok(None) => {}
            // A failed shrink (OOM) is a hard error: keeping the
            // graph-referenced bucket would let a later organic append
            // plain-drop it, so the page fails instead.
            Err(error) => {
                return Err(candle_to_ocr_inference(
                    MODEL_NAME,
                    "shrink KV bucket",
                    error,
                ));
            }
        }
        Ok(())
    }

    /// Layout of the KV storage, `(batch, capacity)`, while any storage
    /// exists — after a shrink the organic storage reports the live length.
    #[cfg(all(test, feature = "cuda"))]
    fn fixed_storage_layout(&self) -> Option<(usize, usize)> {
        self.kv_cache.borrow().fixed_storage_layout()
    }

    /// Drop the fixed KV storage entirely between pages: every page
    /// re-captures from scratch, so retaining the bucket would only save
    /// one allocation — while a plain drop of the graph-referenced storage
    /// can poison the context. Take it out and release through the drain
    /// path instead.
    #[cfg(feature = "cuda")]
    fn drop_fixed_storage(&self, device: &Device) {
        if let Some((k, v)) = self.kv_cache.borrow_mut().take_fixed_storage() {
            drop_and_drain(k, device);
            drop_and_drain(v, device);
        }
    }

    /// CUDA-graph decode step: appends into fixed-capacity storage and runs
    /// masked attention over `[0, kv_len)`. Mirrors the eager prologue op for
    /// op; only the cache append and the attention call differ (the graph
    /// needs static shapes, so it reads the full bucket with a device-side
    /// mask instead of narrowing to the live length and calling flash
    /// attention). `kv_positions` is the constant `(1, 1, cache_len)` index
    /// row built before capture.
    #[cfg(feature = "cuda")]
    fn forward_dynamic(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        kv_lengths: &Tensor,
        kv_positions: &Tensor,
    ) -> Result<Tensor, Error> {
        let (batch, seq_len, _) = hidden_states
            .dims3()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "dynamic attention input", e))?;
        if batch != 1 {
            return Err(Error::Config {
                message: format!(
                    "{MODEL_NAME} CUDA-graph attention requires batch size 1, got {batch}"
                ),
            });
        }
        let qg = self
            .q_proj
            .forward(hidden_states)
            .and_then(|x| x.reshape((batch, seq_len, self.num_heads, self.head_dim * 2)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention q/g projection", e))?;
        let q = qg
            .narrow(D::Minus1, 0, self.head_dim)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention q slice", e))?;
        let gate = qg
            .narrow(D::Minus1, self.head_dim, self.head_dim)
            .and_then(|x| x.reshape((batch, seq_len, self.num_heads * self.head_dim)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention gate slice", e))?;
        let q = self
            .q_norm
            .forward(&q)?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention q layout", e))?;
        let k = self
            .k_proj
            .forward(hidden_states)
            .and_then(|x| x.reshape((batch, seq_len, self.num_kv_heads, self.head_dim)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention k projection", e))?;
        let k = self
            .k_norm
            .forward(&k)?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention k layout", e))?;
        let v = self
            .v_proj
            .forward(hidden_states)
            .and_then(|x| x.reshape((batch, seq_len, self.num_kv_heads, self.head_dim)))
            .and_then(|x| x.transpose(1, 2))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention v projection", e))?;
        let q = self
            .apply_rope(&q, cos, sin)?
            .contiguous()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention q contiguous", e))?;
        let k = self
            .apply_rope(&k, cos, sin)?
            .contiguous()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention k contiguous", e))?;
        let v = v
            .contiguous()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention v contiguous", e))?;

        let cache = self.kv_cache.borrow();
        let cache_len = cache.storage_capacity();
        let (cache_k, cache_v) = cache.storage().ok_or_else(|| Error::Config {
            message: format!("{MODEL_NAME} dynamic KV storage is not initialized"),
        })?;
        drop(cache);
        let append = DynamicKvAppend {
            query_len: seq_len,
            cache_len,
        };
        cache_k
            .inplace_op3(&k, kv_lengths, &append)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "dynamic key cache append", e))?;
        cache_v
            .inplace_op3(&v, kv_lengths, &append)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "dynamic value cache append", e))?;

        // Attention over the fixed-capacity storage with a device-side
        // additive mask derived from `kv_lengths`; masked positions get a
        // very negative score, so stale storage beyond the live length
        // contributes exactly zero after the softmax.
        let kv_bound = kv_lengths
            .i(1..)
            .and_then(|bound| bound.reshape((1, 1, 1)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "dynamic KV bound", e))?;
        let live = kv_positions.broadcast_lt(&kv_bound)?;
        let fill = masked_score(hidden_states.dtype());
        let mask = live
            .to_dtype(hidden_states.dtype())?
            .affine(-fill, fill)?
            .unsqueeze(1)?;
        let output = scaled_dot_product_attention_gqa(
            &q,
            &cache_k,
            &cache_v,
            Some(&mask),
            self.scaling,
            false,
            self.num_kv_groups,
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "dynamic masked attention", e))?;
        let output = output
            .transpose(1, 2)
            .and_then(|x| x.reshape((batch, seq_len, self.num_heads * self.head_dim)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention output layout", e))?;
        let gate = candle_nn::ops::sigmoid(&gate)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention output gate", e))?;
        self.o_proj
            .forward(
                &(&output * &gate).map_err(|e| {
                    candle_to_ocr_inference(MODEL_NAME, "attention gated output", e)
                })?,
            )
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "attention output projection", e))
    }
}

#[derive(Debug)]
enum TokenMixer {
    Linear(GatedDeltaNet),
    Full(FullAttention),
}

#[derive(Debug)]
struct DecoderLayer {
    mixer: TokenMixer,
    mlp: OvisMlp,
    input_layernorm: AdditiveRmsNorm,
    post_attention_layernorm: AdditiveRmsNorm,
}

impl DecoderLayer {
    fn load(cfg: &OvisOcr2TextConfig, layer_type: &str, vb: VarBuilder) -> Result<Self, Error> {
        let mixer = match layer_type {
            "linear_attention" => {
                TokenMixer::Linear(GatedDeltaNet::load(cfg, vb.pp("linear_attn"))?)
            }
            "full_attention" => TokenMixer::Full(FullAttention::load(cfg, vb.pp("self_attn"))?),
            other => {
                return Err(Error::Config {
                    message: format!("OvisOCR2: unsupported decoder layer type '{other}'"),
                });
            }
        };
        Ok(Self {
            mixer,
            mlp: OvisMlp::load(cfg, vb.pp("mlp"))?,
            input_layernorm: AdditiveRmsNorm::load(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("input_layernorm"),
            )?,
            post_attention_layernorm: AdditiveRmsNorm::load(
                cfg.hidden_size,
                cfg.rms_norm_eps,
                vb.pp("post_attention_layernorm"),
            )?,
        })
    }

    fn forward(&self, hidden_states: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor, Error> {
        let residual = hidden_states.clone();
        let normalized = self.input_layernorm.forward(hidden_states)?;
        let mixed = match &self.mixer {
            TokenMixer::Linear(layer) => layer.forward(&normalized)?,
            TokenMixer::Full(layer) => layer.forward(&normalized, cos, sin)?,
        };
        let hidden_states = (&residual + &mixed)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "decoder mixer residual", e))?;
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        (&residual + &hidden_states)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "decoder MLP residual", e))
    }

    fn clear_cache(&self) {
        match &self.mixer {
            TokenMixer::Linear(layer) => layer.clear_cache(),
            TokenMixer::Full(layer) => layer.clear_cache(),
        }
    }

    /// CUDA-graph decode step: linear-attention layers run their regular
    /// forward (their states live in fixed buffers updated in place), only
    /// full-attention layers need the dynamic KV path.
    #[cfg(feature = "cuda")]
    fn forward_dynamic(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        kv_lengths: &Tensor,
        kv_positions: &Tensor,
    ) -> Result<Tensor, Error> {
        let residual = hidden_states.clone();
        let normalized = self.input_layernorm.forward(hidden_states)?;
        let mixed = match &self.mixer {
            TokenMixer::Linear(layer) => layer.forward(&normalized)?,
            TokenMixer::Full(layer) => {
                layer.forward_dynamic(&normalized, cos, sin, kv_lengths, kv_positions)?
            }
        };
        let hidden_states = (&residual + &mixed)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "decoder mixer residual", e))?;
        let residual = hidden_states.clone();
        let hidden_states = self.post_attention_layernorm.forward(&hidden_states)?;
        let hidden_states = self.mlp.forward(&hidden_states)?;
        (&residual + &hidden_states)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "decoder MLP residual", e))
    }

    #[cfg(feature = "cuda")]
    fn prepare_dynamic_cache(&self, cache_len: usize) -> Result<(), Error> {
        match &self.mixer {
            TokenMixer::Linear(_) => Ok(()),
            TokenMixer::Full(layer) => layer.prepare_dynamic_cache(cache_len),
        }
    }

    /// Live KV length of a full-attention layer; `None` for linear layers.
    #[cfg(feature = "cuda")]
    fn full_kv_cache_len(&self) -> Option<usize> {
        match &self.mixer {
            TokenMixer::Linear(_) => None,
            TokenMixer::Full(layer) => Some(layer.kv_cache_len()),
        }
    }

    #[cfg(feature = "cuda")]
    fn set_full_kv_cache_len(&self, len: usize) -> Result<(), Error> {
        match &self.mixer {
            TokenMixer::Linear(_) => Ok(()),
            TokenMixer::Full(layer) => layer.set_kv_cache_len(len),
        }
    }

    #[cfg(feature = "cuda")]
    fn release_dynamic_cache(&self, device: &Device) -> Result<(), Error> {
        if let TokenMixer::Full(layer) = &self.mixer {
            layer.release_dynamic_cache(device)?;
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn drop_fixed_storage(&self, device: &Device) {
        if let TokenMixer::Full(layer) = &self.mixer {
            layer.drop_fixed_storage(device);
        }
    }

    #[cfg(all(test, feature = "cuda"))]
    fn fixed_storage_layout(&self) -> Option<(usize, usize)> {
        match &self.mixer {
            TokenMixer::Linear(_) => None,
            TokenMixer::Full(layer) => layer.fixed_storage_layout(),
        }
    }
}

#[derive(Debug, Clone)]
struct TextRotaryEmbedding {
    rotary: RotaryEmbedding,
    axis_ids: Tensor,
}

impl TextRotaryEmbedding {
    fn new(cfg: &OvisOcr2TextConfig, device: &Device) -> Result<Self, Error> {
        let rotary_dim = (cfg.head_dim as f64 * cfg.rope_parameters.partial_rotary_factor) as usize;
        if rotary_dim == 0 || !rotary_dim.is_multiple_of(2) {
            return Err(Error::Config {
                message: format!("OvisOCR2: invalid rotary dimension {rotary_dim}"),
            });
        }
        let half = rotary_dim / 2;
        if cfg.rope_parameters.mrope_section.iter().sum::<usize>() != half {
            return Err(Error::Config {
                message: format!(
                    "OvisOCR2: mrope_section {:?} must sum to rotary_dim/2 ({half})",
                    cfg.rope_parameters.mrope_section
                ),
            });
        }
        let rotary =
            RotaryEmbedding::new_multi_axis(rotary_dim, cfg.rope_parameters.rope_theta, 3, device)?;
        let axis_ids = Tensor::from_vec(
            interleaved_axis_ids(rotary_dim, &cfg.rope_parameters.mrope_section),
            (1, 1, rotary_dim, 1),
            device,
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "create mRoPE axis map", e))?;
        Ok(Self { rotary, axis_ids })
    }

    fn forward(&self, position_ids: &Tensor, dtype: DType) -> Result<(Tensor, Tensor), Error> {
        let (cos, sin) = self.rotary.forward_multi_axis(position_ids, dtype)?;
        Ok((self.select_axes(&cos)?, self.select_axes(&sin)?))
    }

    fn select_axes(&self, values: &Tensor) -> Result<Tensor, Error> {
        let (_, batch, seq_len, rotary_dim) = values
            .dims4()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "mRoPE tensor shape", e))?;
        let values = values
            .permute((1, 2, 3, 0))
            .and_then(|values| values.contiguous())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "mRoPE axis layout", e))?;
        let axis_ids = self
            .axis_ids
            .expand((batch, seq_len, rotary_dim, 1))
            .and_then(|axis_ids| axis_ids.contiguous())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "expand mRoPE axis map", e))?;
        values
            .gather(&axis_ids, 3)
            .and_then(|values| values.squeeze(3))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "select mRoPE axes", e))
    }
}

fn interleaved_axis_ids(rotary_dim: usize, mrope_section: &[usize]) -> Vec<u32> {
    let half = rotary_dim / 2;
    let h_limit = mrope_section[1] * 3;
    let w_limit = mrope_section[2] * 3;
    (0..rotary_dim)
        .map(|dimension| {
            let freq_idx = dimension % half;
            if freq_idx % 3 == 1 && freq_idx < h_limit {
                1
            } else if freq_idx % 3 == 2 && freq_idx < w_limit {
                2
            } else {
                0
            }
        })
        .collect()
}

pub(crate) struct OvisOcr2TextModel {
    embed_tokens: Embedding,
    layers: Vec<DecoderLayer>,
    norm: AdditiveRmsNorm,
    rotary_emb: TextRotaryEmbedding,
    #[cfg(feature = "cuda")]
    decode_graph: RefCell<Option<DecoderCudaGraph<OvisDecodeGraphInputs>>>,
    /// Bucket a lazy capture will use at the first decode step, set by
    /// `prepare_decode_graph`. Capture waits until a second token is
    /// actually needed: a generation whose first token stops it never pays
    /// for a graph it would never replay.
    #[cfg(feature = "cuda")]
    pending_capture_bucket: RefCell<Option<usize>>,
    /// Test-only failure injection; production builds compile this out.
    #[cfg(all(test, feature = "cuda"))]
    hooks: TestHooks,
    /// Test-only counter for `TestHooks::fail_body_on_call`.
    #[cfg(all(test, feature = "cuda"))]
    body_calls: std::cell::Cell<usize>,
    /// Declared last so it drops last: the model's `Drop` disposes the
    /// cached graph, and this guard drains whatever the remaining fields'
    /// frees stash on the CUDA context afterwards.
    #[cfg(feature = "cuda")]
    #[allow(dead_code)]
    drain_guard: CudaGraphDrainGuard,
}

/// Largest KV bucket a captured decode graph covers; sized to the official
/// generation limit so a full-length decode never leaves the graph path.
#[cfg(feature = "cuda")]
const OVISOCR2_DECODE_CACHE_LEN: usize = 16_384;

/// Inputs the decode graph captures, named and typed. The bundle owns every
/// tensor the captured region reads that no model field holds, so nothing
/// outside it can dangle under a live graph.
#[cfg(feature = "cuda")]
struct OvisDecodeGraphInputs {
    hidden: Tensor,
    positions: Tensor,
    kv_lengths: CudaGraphKvLengths,
    /// Static [0, cache_len) slot positions; a capture-time constant the
    /// attention mask compares against, retained here for the graph's
    /// lifetime.
    kv_positions: Tensor,
    /// The LM head read inside the captured region.
    lm_head: Linear,
}

#[cfg(feature = "cuda")]
impl DecoderGraphInputs for OvisDecodeGraphInputs {
    fn dispose(self, device: &Device) {
        let Self {
            hidden,
            positions,
            kv_lengths,
            kv_positions,
            lm_head,
        } = self;
        drop_and_drain(kv_lengths, device);
        drop_and_drain(kv_positions, device);
        drop_and_drain(positions, device);
        drop_and_drain(hidden, device);
        drop_and_drain(lm_head, device);
    }
}

/// Test probe: how many times a captured decode graph has replayed.
#[cfg(all(test, feature = "cuda"))]
static DECODE_GRAPH_REPLAYS: std::sync::atomic::AtomicUsize =
    std::sync::atomic::AtomicUsize::new(0);

/// Test-only capture failure injection, mirroring the qwen3_vl hooks.
/// Every field defaults to off and production builds compile none of this
/// in.
#[cfg(all(test, feature = "cuda"))]
#[derive(Default)]
pub(crate) struct TestHooks {
    /// Fail the decode-graph capture right after the KV buckets are
    /// allocated, leaving them resident for the recovery to shrink.
    pub fail_capture: bool,
    /// Fail the decode-graph body on its nth invocation (1 = warmup, 2 =
    /// capture, exercising the shared helper's drained teardown).
    pub fail_body_on_call: Option<usize>,
    /// Fail the bucket allocation right after 0-based layer `n`'s
    /// `prepare_dynamic_cache`, leaving a partially allocated bucket set
    /// behind (e.g. an OOM on a later layer).
    pub fail_prepare_after_layer: Option<usize>,
    /// Keep a failed capture's buckets instead of shrinking them (the
    /// control that proves the layout assertions can fail).
    pub skip_release: bool,
    /// Pin the decode bucket ceiling small so tests can reach the ladder
    /// ceiling quickly.
    pub decode_cache_ceiling: Option<usize>,
}

impl OvisOcr2TextModel {
    pub(crate) fn load(cfg: &OvisOcr2TextConfig, vb: VarBuilder) -> Result<Self, Error> {
        if cfg.layer_types.len() != cfg.num_hidden_layers {
            return Err(Error::Config {
                message: format!(
                    "OvisOCR2: layer_types has {} entries, expected {}",
                    cfg.layer_types.len(),
                    cfg.num_hidden_layers
                ),
            });
        }
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load token embeddings", e))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for (index, layer_type) in cfg.layer_types.iter().enumerate() {
            layers.push(DecoderLayer::load(
                cfg,
                layer_type,
                vb.pp(format!("layers.{index}")),
            )?);
        }
        let norm = AdditiveRmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))?;
        let rotary_emb = TextRotaryEmbedding::new(cfg, vb.device())?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
            #[cfg(feature = "cuda")]
            decode_graph: RefCell::new(None),
            #[cfg(feature = "cuda")]
            pending_capture_bucket: RefCell::new(None),
            #[cfg(all(test, feature = "cuda"))]
            hooks: TestHooks::default(),
            #[cfg(all(test, feature = "cuda"))]
            body_calls: std::cell::Cell::new(0),
            #[cfg(feature = "cuda")]
            drain_guard: CudaGraphDrainGuard::new(vb.device()),
        })
    }

    pub(crate) fn embed(&self, input_ids: &Tensor) -> Result<Tensor, Error> {
        self.embed_tokens
            .forward(input_ids)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "token embedding", e))
    }

    pub(crate) fn token_embedding_weight(&self) -> Tensor {
        self.embed_tokens.embeddings().clone()
    }

    pub(crate) fn forward(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
    ) -> Result<Tensor, Error> {
        let (cos, sin) = self
            .rotary_emb
            .forward(position_ids, inputs_embeds.dtype())?;
        let mut hidden_states = inputs_embeds.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &cos, &sin)?;
        }
        self.norm.forward(&hidden_states)
    }

    pub(crate) fn clear_cache(&self) {
        // The graph holds raw pointers into the linear-attention state
        // buffers and the fixed KV storage; it must be disposed before
        // either is released here.
        #[cfg(feature = "cuda")]
        {
            self.invalidate_decode_graph();
            self.pending_capture_bucket.borrow_mut().take();
            // Free the fixed KV buckets too: reset() only clears the
            // logical length, and a longer prompt on the next page would
            // make the prefill's append plain-drop this graph-referenced
            // storage. Every page re-captures anyway, so retention only
            // saved one allocation.
            let device = self.embed_tokens.embeddings().device().clone();
            for layer in &self.layers {
                layer.drop_fixed_storage(&device);
            }
            drain_cuda_context_errors(&device);
        }
        for layer in &self.layers {
            layer.clear_cache();
        }
    }

    /// Record the decode bucket after a prefill; the capture itself is lazy
    /// and happens at the first decode step (see `decode_step_graph`), so a
    /// generation stopped by its very first token never captures. Buckets
    /// follow the shared ladder; anything ineligible stays eager.
    #[cfg(feature = "cuda")]
    pub(crate) fn prepare_decode_graph(
        &self,
        prompt_len: usize,
        max_new_tokens: usize,
    ) -> Result<(), Error> {
        // A single-token generation takes no decode step at all (the one
        // token comes from the prefill's own logits), so a captured graph
        // would sit unused: run it eager.
        if max_new_tokens <= 1 {
            return Ok(());
        }
        if std::env::var_os("OAR_VL_DISABLE_CUDA_GRAPH").is_some()
            || std::env::var_os("OAR_OVISOCR2_DISABLE_CUDA_GRAPH").is_some()
        {
            self.invalidate_decode_graph();
            self.pending_capture_bucket.borrow_mut().take();
            return Ok(());
        }
        let embeddings = self.embed_tokens.embeddings();
        if !embeddings.device().is_cuda() || !matches!(embeddings.dtype(), DType::BF16 | DType::F16)
        {
            return Ok(());
        }
        // Start the ladder at the prompt, not at the declared budget: the
        // default budget would pin the largest bucket up front and make the
        // masked attention scan 16K slots from the first step. The replay
        // path re-captures into bigger buckets as the sequence grows.
        let Some(cache_len) = prompt_decode_bucket(prompt_len, self.decode_cache_ceiling()) else {
            // The prompt alone does not fit the largest bucket; no graph may
            // stay alive over it.
            self.invalidate_decode_graph();
            self.pending_capture_bucket.borrow_mut().take();
            return Ok(());
        };
        // `clear_cache` disposes the graph at the start of every page, so a
        // live graph never reaches this point: each page captures once.
        self.invalidate_decode_graph();
        *self.pending_capture_bucket.borrow_mut() = Some(cache_len);
        Ok(())
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn prepare_decode_graph(
        &self,
        _prompt_len: usize,
        _max_new_tokens: usize,
    ) -> Result<(), Error> {
        Ok(())
    }

    /// Capture once the prefill has run: the full-attention KV history moves
    /// into fixed storage (contents preserved), the linear-attention states
    /// already live in fixed buffers thanks to the in-place decode updates.
    /// The capture helper runs the body three times (warmup, capture, warm
    /// launch), which would advance the recurrent states with scratch input;
    /// the states are snapshotted first and restored afterwards. The KV
    /// warmup appends only into the scratch slot at `prompt_len`, so the
    /// history is never touched.
    /// Everything up to and including the capture is a soft failure
    /// (`Ok(false)`): the caller shrinks the preallocated buckets and
    /// continues eager — a bucket-allocation OOM must not fail the page.
    /// Once the states are snapshotted they are restored before the soft
    /// return, and only a rollback failure is a hard error: continuing
    /// eager on half-restored states would silently corrupt the output.
    #[cfg(feature = "cuda")]
    fn capture_decode_graph(
        &self,
        cache_len: usize,
        prompt_len: usize,
        lm_head: &Linear,
    ) -> Result<bool, Error> {
        let prepared: Result<(), Error> =
            self.layers
                .iter()
                .enumerate()
                .try_for_each(|(_index, layer)| {
                    layer.prepare_dynamic_cache(cache_len)?;
                    #[cfg(all(test, feature = "cuda"))]
                    if self
                        .hooks
                        .fail_prepare_after_layer
                        .is_some_and(|target| _index + 1 == target)
                    {
                        return Err(Error::Config {
                            message: "injected prepare failure (test)".to_string(),
                        });
                    }
                    Ok(())
                });
        if let Err(error) = prepared {
            tracing::warn!(
                "{MODEL_NAME} decoder graph bucket preparation failed: {error}; continuing eager"
            );
            return Ok(false);
        }
        #[cfg(all(test, feature = "cuda"))]
        if self.hooks.fail_capture {
            // Injected failure with the buckets already allocated, exercising
            // the caller's recovery path.
            return Ok(false);
        }
        let snapshots = match self.snapshot_linear_states() {
            Ok(snapshots) => snapshots,
            Err(error) => {
                tracing::warn!(
                    "{MODEL_NAME} decoder graph state snapshot failed: {error}; continuing eager"
                );
                return Ok(false);
            }
        };
        let embeddings = self.embed_tokens.embeddings();
        let device = embeddings.device().clone();
        let inputs: Result<OvisDecodeGraphInputs, Error> = (|| {
            let hidden_size = embeddings
                .dim(1)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "graph hidden size", e))?;
            Ok(OvisDecodeGraphInputs {
                hidden: Tensor::zeros((1, 1, hidden_size), embeddings.dtype(), &device)
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "graph hidden input", e))?,
                positions: Tensor::zeros((3, 1, 1), DType::I64, &device)
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "graph position input", e))?,
                // The append kernel derives the write slot from the
                // cumulative END, hence prompt_len + one decode step.
                kv_lengths: CudaGraphKvLengths::new(prompt_len + 1, &device)
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "graph KV lengths", e))?,
                kv_positions: Tensor::arange(0u32, cache_len as u32, &device)?
                    .reshape((1, 1, cache_len))
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "graph KV positions", e))?,
                lm_head: lm_head.clone(),
            })
        })();
        let inputs = match inputs {
            Ok(inputs) => inputs,
            Err(error) => {
                // The snapshot is already taken: roll the states back before
                // the soft return (a failed rollback is the hard error).
                self.restore_linear_states(snapshots)?;
                tracing::warn!(
                    "{MODEL_NAME} decoder graph input allocation failed: {error}; continuing eager"
                );
                return Ok(false);
            }
        };
        let captured = capture_decoder_graph(
            &device,
            MODEL_NAME,
            self,
            inputs,
            Self::decode_graph_body,
            cache_len,
        );
        let graph = match captured {
            Ok(graph) => graph,
            Err(capture_error) => {
                // The warmup may have advanced the linear-attention states
                // with scratch input: put the prefill's values back into the
                // fixed buffers so the eager fallback reads them intact.
                self.restore_linear_states(snapshots)?;
                tracing::warn!(
                    "{MODEL_NAME} decoder graph capture failed: {capture_error}; continuing eager"
                );
                return Ok(false);
            }
        };
        if let Err(error) = self.restore_linear_states(snapshots) {
            // A plainly dropped graph returns graph-bound buffers to the
            // allocator and poisons it: dispose before propagating.
            graph.dispose();
            return Err(error);
        }
        tracing::info!("{MODEL_NAME} decoder graph captured: bucket={cache_len}");
        *self.decode_graph.borrow_mut() = Some(graph);
        Ok(true)
    }

    /// Captured region of the decode graph: one decode step plus the LM
    /// head, reading only the registered bundle and model-owned state.
    #[cfg(feature = "cuda")]
    fn decode_graph_body(
        this: &Self,
        inputs: &OvisDecodeGraphInputs,
    ) -> Result<Vec<Tensor>, Error> {
        #[cfg(all(test, feature = "cuda"))]
        {
            let call = this.body_calls.get() + 1;
            this.body_calls.set(call);
            if this.hooks.fail_body_on_call.is_some_and(|n| call == n) {
                return Err(Error::Config {
                    message: "injected capture-body failure (test)".to_string(),
                });
            }
        }
        let (cos, sin) = this
            .rotary_emb
            .forward(&inputs.positions, inputs.hidden.dtype())?;
        let mut hidden_states = inputs.hidden.clone();
        for layer in &this.layers {
            hidden_states = layer.forward_dynamic(
                &hidden_states,
                &cos,
                &sin,
                inputs.kv_lengths.tensor(),
                &inputs.kv_positions,
            )?;
        }
        let hidden = this.norm.forward(&hidden_states)?;
        // Match the eager LM head's GEMM shape exactly: the eager path
        // selects the last row to (H) and unsqueezes to (1, H) before the
        // projection, so the graph does the same rather than projecting the
        // (1, 1, H) tensor (a different cublas shape can round differently).
        let logits = inputs
            .lm_head
            .forward(
                &hidden
                    .i((0, 0, ..))
                    .and_then(|last| last.unsqueeze(0))
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "graph LM head input", e))?,
            )
            .and_then(|logits| logits.squeeze(0))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "decode LM head", e))?;
        Ok(vec![logits])
    }

    /// Replay the captured graph for one decode step. Returns `None` when no
    /// graph fits this step (not captured, shape mismatch, or past the
    /// ladder ceiling) and the caller should run the eager path. The first
    /// call after `prepare_decode_graph` captures on demand: by then the
    /// generation loop has emitted a non-stop token, so the graph is known
    /// to be needed.
    #[cfg(feature = "cuda")]
    pub(crate) fn decode_step_graph(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        lm_head: &Linear,
    ) -> Result<Option<Tensor>, Error> {
        if self.decode_graph.borrow().is_none() {
            let Some(cache_len) = self.pending_capture_bucket.borrow_mut().take() else {
                return Ok(None);
            };
            let prompt_len = self.kv_cache_len();
            // A soft capture failure already logged itself; release the
            // preallocated buckets so the eager fallback and later pages do
            // not carry them. A rollback failure is a hard error and
            // propagates.
            if !self.capture_decode_graph(cache_len, prompt_len, lm_head)? {
                self.recover_failed_capture()?;
                return Ok(None);
            }
        }
        let kv_len = self.kv_cache_len().saturating_add(1);
        let overflow = self
            .decode_graph
            .borrow()
            .as_ref()
            .is_some_and(|graph| kv_len > graph.cache_len);
        if overflow {
            let cache_len = self
                .decode_graph
                .borrow()
                .as_ref()
                .map(|graph| graph.cache_len)
                .expect("overflow implies Some");
            let Some(next) = next_decode_bucket(cache_len, self.decode_cache_ceiling()) else {
                // Ladder ceiling: the rest of this generation decodes eager.
                // The retiring graph referenced the fixed buckets: shrink
                // them back to organic storage now, so the eager append
                // path's later growth never plain-drops graph-referenced
                // memory.
                self.invalidate_decode_graph();
                self.release_dynamic_caches()?;
                return Ok(None);
            };
            // Grow the fixed buckets and re-capture; the appended history is
            // preserved and the warmup appends at the live end.
            self.invalidate_decode_graph();
            for layer in &self.layers {
                if let Err(error) = layer.prepare_dynamic_cache(next) {
                    tracing::warn!(
                        "{MODEL_NAME} decoder KV growth to bucket {next} failed: {error}; continuing eager"
                    );
                    self.recover_failed_capture()?;
                    return Ok(None);
                }
            }
            if !self.capture_decode_graph(next, kv_len - 1, lm_head)? {
                self.recover_failed_capture()?;
                return Ok(None);
            }
            tracing::info!("{MODEL_NAME} decoder graph ladder: bucket {cache_len} -> {next}");
        }
        let captured_ref = self.decode_graph.borrow();
        let Some(captured) = captured_ref.as_ref() else {
            return Ok(None);
        };
        if inputs_embeds.shape() != captured.inputs.hidden.shape()
            || position_ids.shape() != captured.inputs.positions.shape()
        {
            return Ok(None);
        }
        captured
            .inputs
            .hidden
            .slice_set(inputs_embeds, 0, 0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "copy graph hidden", e))?;
        captured
            .inputs
            .positions
            .slice_set(position_ids, 0, 0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "copy graph positions", e))?;
        captured
            .inputs
            .kv_lengths
            .update(kv_len)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "update graph KV lengths", e))?;
        captured
            .graph
            .launch()
            .map_err(|e| cuda_graph_error(MODEL_NAME, "launch decoder CUDA graph", e))?;
        for layer in &self.layers {
            layer.set_full_kv_cache_len(kv_len)?;
        }
        #[cfg(test)]
        DECODE_GRAPH_REPLAYS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        // Owned copy: a later replay overwrites the captured output buffer,
        // and callers may hold the logits past it.
        Ok(Some(captured.outputs[0].copy().map_err(|e| {
            candle_to_ocr_inference(MODEL_NAME, "copy graph logits", e)
        })?))
    }

    #[cfg(not(feature = "cuda"))]
    pub(crate) fn decode_step_graph(
        &self,
        _inputs_embeds: &Tensor,
        _position_ids: &Tensor,
        _lm_head: &Linear,
    ) -> Result<Option<Tensor>, Error> {
        Ok(None)
    }

    #[cfg(feature = "cuda")]
    fn invalidate_decode_graph(&self) {
        if let Some(graph) = self.decode_graph.borrow_mut().take() {
            graph.dispose();
        }
    }

    /// Shrink every full-attention layer's fixed KV bucket back to the
    /// organic eager form, preserving the live history. Runs after a failed
    /// capture and at the ladder ceiling: the retiring (or never installed)
    /// graph referenced those buckets, and the organic growth that follows
    /// must never plain-drop graph-referenced storage.
    #[cfg(feature = "cuda")]
    fn release_dynamic_caches(&self) -> Result<(), Error> {
        let device = self.embed_tokens.embeddings().device().clone();
        for layer in &self.layers {
            layer.release_dynamic_cache(&device)?;
        }
        drain_cuda_context_errors(&device);
        Ok(())
    }

    /// Tear down what a failed capture left behind — the graph (never
    /// installed, but cheap to drop) and every full-attention layer's
    /// preallocated KV bucket, shrunk back to the organic eager form with
    /// the prompt history preserved. The eager fallback exists for
    /// low-memory situations; buckets a decode will never use would only
    /// starve it and the pages after it.
    #[cfg(feature = "cuda")]
    fn recover_failed_capture(&self) -> Result<(), Error> {
        self.invalidate_decode_graph();
        self.pending_capture_bucket.borrow_mut().take();
        #[cfg(all(test, feature = "cuda"))]
        if self.hooks.skip_release {
            // Test-only control: keep the buckets so the layout assertions
            // can prove they catch a missing release.
            return Ok(());
        }
        self.release_dynamic_caches()
    }

    /// Decode bucket ceiling; tests may pin it small to reach the ladder
    /// ceiling quickly.
    #[cfg(feature = "cuda")]
    fn decode_cache_ceiling(&self) -> usize {
        #[cfg(all(test, feature = "cuda"))]
        if let Some(ceiling) = self.hooks.decode_cache_ceiling {
            return ceiling;
        }
        OVISOCR2_DECODE_CACHE_LEN
    }

    /// Live KV length of the full-attention layers (all share one length).
    #[cfg(feature = "cuda")]
    fn kv_cache_len(&self) -> usize {
        let len = self
            .layers
            .iter()
            .find_map(|layer| layer.full_kv_cache_len())
            .unwrap_or(0);
        debug_assert!(
            self.layers
                .iter()
                .filter_map(|layer| layer.full_kv_cache_len())
                .all(|layer_len| layer_len == len)
        );
        len
    }

    /// Value snapshots of every linear-attention layer's conv and recurrent
    /// states, taken before a capture so the warmup's scratch steps can be
    /// rolled back into the same fixed buffers.
    #[cfg(feature = "cuda")]
    fn snapshot_linear_states(&self) -> Result<Vec<(Tensor, Tensor)>, Error> {
        let mut snapshots = Vec::new();
        for layer in &self.layers {
            let TokenMixer::Linear(layer) = &layer.mixer else {
                continue;
            };
            let conv = layer.conv_state.borrow().as_ref().map(|state| {
                state
                    .copy()
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "snapshot conv state", e))
            });
            let recurrent = layer.recurrent_state.borrow().as_ref().map(|state| {
                state
                    .copy()
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "snapshot recurrent state", e))
            });
            match (conv, recurrent) {
                (Some(conv), Some(recurrent)) => snapshots.push((conv?, recurrent?)),
                // No prefill has run: nothing to preserve, and the capture
                // starts the states from zeros like the eager path would.
                (None, None) => {}
                _ => {
                    return Err(Error::Config {
                        message: format!(
                            "{MODEL_NAME} partial Gated DeltaNet state at graph capture"
                        ),
                    });
                }
            }
        }
        Ok(snapshots)
    }

    #[cfg(feature = "cuda")]
    fn restore_linear_states(&self, snapshots: Vec<(Tensor, Tensor)>) -> Result<(), Error> {
        let mut snapshots = snapshots.into_iter();
        for layer in &self.layers {
            let TokenMixer::Linear(layer) = &layer.mixer else {
                continue;
            };
            let Some((conv, recurrent)) = snapshots.next() else {
                continue;
            };
            store_state(&layer.conv_state, conv, "restore conv state")?;
            store_state(&layer.recurrent_state, recurrent, "restore recurrent state")?;
        }
        Ok(())
    }

    /// Whether the decode graph is currently captured — asserted by the
    /// CUDA graph test.
    #[cfg(all(test, feature = "cuda"))]
    pub(crate) fn decode_graph_captured(&self) -> bool {
        self.decode_graph.borrow().is_some()
    }
}

#[cfg(feature = "cuda")]
impl Drop for OvisOcr2TextModel {
    fn drop(&mut self) {
        // A cached graph must go through dispose: plainly dropping it
        // returns graph-bound buffers to the allocator and poisons it.
        self.invalidate_decode_graph();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::IndexOp;

    #[test]
    fn decode_state_updates_write_in_place() {
        let device = Device::Cpu;
        let slot = RefCell::new(Some(Tensor::zeros((1, 2, 4), DType::F32, &device).unwrap()));
        let id_before = slot.borrow().as_ref().unwrap().id();
        store_state(
            &slot,
            Tensor::ones((1, 2, 4), DType::F32, &device).unwrap(),
            "test store",
        )
        .unwrap();
        {
            let borrow = slot.borrow();
            let state = borrow.as_ref().unwrap();
            // Same buffer, new values: a captured graph keeps pointing at it.
            assert_eq!(state.id(), id_before);
            assert_eq!(
                state.flatten_all().unwrap().to_vec1::<f32>().unwrap(),
                vec![1.0; 8]
            );
        }
        // A shape change (a fresh prefill) replaces the buffer instead.
        store_state(
            &slot,
            Tensor::zeros((2, 2, 4), DType::F32, &device).unwrap(),
            "test store",
        )
        .unwrap();
        assert_eq!(slot.borrow().as_ref().unwrap().dims(), &[2, 2, 4]);
    }

    /// Regression test for the production eager path: prefill stores the GDN
    /// conv state as a non-contiguous `narrow` view, and every following
    /// decode step must still be able to store the next state in place.
    /// Exercises a tiny three-layer model (linear/full/linear attention)
    /// through the same `forward` the generation loop drives, on CPU — no
    /// CUDA graph involved. Runs in well under 50ms.
    #[test]
    fn prefill_then_decode_steps_store_states() -> Result<(), Error> {
        let device = Device::Cpu;
        let cfg = tiny_graph_config();
        cfg.validate()?;
        let tensors = tiny_graph_tensors(&cfg, &device);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = OvisOcr2TextModel::load(&cfg, vb)?;

        // Prefill four tokens through the production forward.
        let prompt = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let prompt_positions = Tensor::from_vec(
            vec![0i64, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            (3, 1, 4),
            &device,
        )
        .unwrap();
        let embeds = model.embed(&prompt)?;
        model.forward(&embeds, &prompt_positions)?;

        // Three decode steps through the same production forward.
        let token = Tensor::from_vec(vec![7u32], (1, 1), &device).unwrap();
        for step in 0..3 {
            let position = 4 + step as i64;
            let positions = Tensor::from_vec(vec![position; 3], (3, 1, 1), &device).unwrap();
            let embed = model.embed(&token)?;
            model.forward(&embed, &positions)?;
        }
        Ok(())
    }

    #[test]
    fn cached_depthwise_step_matches_grouped_convolution() -> candle_core::Result<()> {
        let state = Tensor::from_vec(
            vec![1f32, 2., 3., 4., 5., 6., 7., 8.],
            (1, 2, 4),
            &Device::Cpu,
        )?;
        let mixed = Tensor::from_vec(vec![9f32, 10.], (1, 2, 1), &Device::Cpu)?;
        let weight = Tensor::from_vec(
            vec![0.1f32, 0.2, 0.3, 0.4, -0.5, 0.25, 0.75, 1.0],
            (2, 1, 4),
            &Device::Cpu,
        )?;
        let decode_weight = weight.squeeze(1)?.unsqueeze(0)?;
        let (actual, new_state) = cached_depthwise_conv_step(&state, &mixed, &decode_weight, 4)?;

        let joined = Tensor::cat(&[&state, &mixed], 2)?;
        let conv = Conv1d::new(
            weight,
            None,
            Conv1dConfig {
                groups: 2,
                ..Default::default()
            },
        );
        let expected = conv.forward(&joined)?.narrow(2, 1, 1)?;
        assert_eq!(actual.to_vec3::<f32>()?, expected.to_vec3::<f32>()?);
        assert_eq!(
            new_state.to_vec3::<f32>()?,
            joined.narrow(2, 1, 4)?.to_vec3::<f32>()?
        );
        Ok(())
    }

    #[test]
    fn cached_depthwise_step_matches_low_precision_convolution() -> candle_core::Result<()> {
        for dtype in [DType::BF16, DType::F16] {
            let state = Tensor::from_vec(
                vec![1f32, 2., 3., 4., 5., 6., 7., 8.],
                (1, 2, 4),
                &Device::Cpu,
            )?
            .to_dtype(dtype)?;
            let mixed =
                Tensor::from_vec(vec![9f32, 10.], (1, 2, 1), &Device::Cpu)?.to_dtype(dtype)?;
            let weight = Tensor::from_vec(
                vec![0.1f32, 0.2, 0.3, 0.4, -0.5, 0.25, 0.75, 1.0],
                (2, 1, 4),
                &Device::Cpu,
            )?
            .to_dtype(dtype)?;
            let decode_weight = weight.to_dtype(DType::F32)?.squeeze(1)?.unsqueeze(0)?;
            let (actual, _) = cached_depthwise_conv_step(&state, &mixed, &decode_weight, 4)?;
            let expected = match dtype {
                DType::BF16 => vec![vec![vec![5.59375f32], vec![14.75]]],
                DType::F16 => vec![vec![vec![5.597_656_3f32], vec![14.75]]],
                _ => unreachable!(),
            };
            assert_eq!(actual.dtype(), dtype);
            assert_eq!(
                actual.to_dtype(DType::F32)?.to_vec3::<f32>()?,
                expected,
                "cached convolution mismatch for {dtype:?}"
            );
        }
        Ok(())
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cached_depthwise_step_matches_cuda_low_precision_convolution() -> candle_core::Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        for dtype in [DType::BF16, DType::F16] {
            let state =
                Tensor::from_vec(vec![1f32, 2., 3., 4., 5., 6., 7., 8.], (1, 2, 4), &device)?
                    .to_dtype(dtype)?;
            let mixed = Tensor::from_vec(vec![9f32, 10.], (1, 2, 1), &device)?.to_dtype(dtype)?;
            let weight = Tensor::from_vec(
                vec![0.1f32, 0.2, 0.3, 0.4, -0.5, 0.25, 0.75, 1.0],
                (2, 1, 4),
                &device,
            )?
            .to_dtype(dtype)?;
            let decode_weight = weight.to_dtype(DType::F32)?.squeeze(1)?.unsqueeze(0)?;
            let (actual, _) = cached_depthwise_conv_step(&state, &mixed, &decode_weight, 4)?;

            let joined = Tensor::cat(&[&state, &mixed], 2)?;
            let conv = Conv1d::new(
                weight,
                None,
                Conv1dConfig {
                    groups: 2,
                    ..Default::default()
                },
            );
            let expected = conv.forward(&joined)?.narrow(2, 1, 1)?;
            assert_eq!(actual.dtype(), dtype);
            assert_eq!(
                actual
                    .to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?
                    .to_vec3::<f32>()?,
                expected
                    .to_dtype(DType::F32)?
                    .to_device(&Device::Cpu)?
                    .to_vec3::<f32>()?,
                "cached convolution mismatch for {dtype:?}"
            );
        }
        Ok(())
    }

    #[test]
    fn qwen35_mrope_axes_are_interleaved() {
        let rotary_dim = 64;
        let sections = [11, 11, 10];
        let rope = TextRotaryEmbedding {
            rotary: RotaryEmbedding::new_multi_axis(rotary_dim, 1.0, 3, &Device::Cpu).unwrap(),
            axis_ids: Tensor::from_vec(
                interleaved_axis_ids(rotary_dim, &sections),
                (1, 1, rotary_dim, 1),
                &Device::Cpu,
            )
            .unwrap(),
        };
        let ids = Tensor::from_vec(
            [vec![10i64; 32], vec![20i64; 32], vec![30i64; 32]].concat(),
            (3, 1, 32),
            &Device::Cpu,
        )
        .unwrap();
        let (cos, _) = rope.forward(&ids, DType::F32).unwrap();
        let values = cos.i((0, 0)).unwrap().to_vec1::<f32>().unwrap();
        assert!((values[0] - 10f32.cos()).abs() < 1e-6);
        assert!((values[1] - 20f32.cos()).abs() < 1e-6);
        assert!((values[2] - 30f32.cos()).abs() < 1e-6);
        assert!((values[31] - 20f32.cos()).abs() < 1e-6);
    }

    fn tiny_graph_config() -> OvisOcr2TextConfig {
        OvisOcr2TextConfig {
            model_type: "qwen3_5_text".to_string(),
            vocab_size: 128,
            hidden_size: 64,
            intermediate_size: 128,
            num_hidden_layers: 3,
            num_attention_heads: 2,
            num_key_value_heads: 1,
            head_dim: 32,
            hidden_act: candle_nn::Activation::Silu,
            max_position_embeddings: 256,
            rms_norm_eps: 1e-6,
            rope_parameters: super::super::config::OvisOcr2RopeParameters {
                rope_type: "default".to_string(),
                mrope_section: vec![3, 3, 2],
                mrope_interleaved: true,
                rope_theta: 10_000.0,
                partial_rotary_factor: 0.5,
            },
            layer_types: vec![
                "linear_attention".to_string(),
                "full_attention".to_string(),
                "linear_attention".to_string(),
            ],
            linear_conv_kernel_dim: 3,
            linear_key_head_dim: 8,
            linear_value_head_dim: 8,
            linear_num_key_heads: 2,
            linear_num_value_heads: 2,
            eos_token_id: 5,
            attention_bias: false,
            attention_dropout: 0.0,
            attn_output_gate: true,
            initializer_range: 0.02,
            full_attention_interval: 0,
            mlp_only_layers: Vec::new(),
            mtp_num_hidden_layers: 0,
            mtp_use_dedicated_embeddings: false,
            tie_word_embeddings: true,
            use_cache: true,
            dtype: None,
            mamba_ssm_dtype: None,
        }
    }

    fn tiny_graph_tensors(
        cfg: &OvisOcr2TextConfig,
        device: &Device,
    ) -> std::collections::HashMap<String, Tensor> {
        let mut tensors = std::collections::HashMap::new();
        // Deterministic stand-in for randn (xorshift64*, uniform on
        // [-0.1, 0.1)): the graph/eager comparisons must not flake on
        // unseeded draws.
        let rng = std::cell::Cell::new(0x9E3779B97F4A7C15u64);
        let det = |shape: &[usize]| -> Tensor {
            let len: usize = shape.iter().product();
            let mut state = rng.get();
            let values: Vec<f32> = (0..len)
                .map(|_| {
                    state ^= state >> 12;
                    state ^= state << 25;
                    state ^= state >> 27;
                    state = state.wrapping_mul(0x2545F4914F6CDD1D);
                    let u = (state >> 40) as f32 / (1u64 << 24) as f32;
                    (u - 0.5) * 0.2
                })
                .collect();
            rng.set(state);
            Tensor::from_vec(values, shape.to_vec(), device).unwrap()
        };
        let randn = |rows: usize, cols: usize| det(&[rows, cols]);
        let hidden = cfg.hidden_size;
        let key_dim = cfg.linear_num_key_heads * cfg.linear_key_head_dim;
        let value_dim = cfg.linear_num_value_heads * cfg.linear_value_head_dim;
        let conv_dim = key_dim * 2 + value_dim;
        tensors.insert(
            "embed_tokens.weight".to_string(),
            randn(cfg.vocab_size, hidden),
        );
        tensors.insert("norm.weight".to_string(), det(&[hidden]));
        for (index, layer_type) in cfg.layer_types.iter().enumerate() {
            let prefix = format!("layers.{index}");
            if layer_type == "linear_attention" {
                let attn = format!("{prefix}.linear_attn");
                tensors.insert(
                    format!("{attn}.in_proj_qkv.weight"),
                    randn(conv_dim, hidden),
                );
                tensors.insert(format!("{attn}.in_proj_z.weight"), randn(value_dim, hidden));
                tensors.insert(
                    format!("{attn}.in_proj_b.weight"),
                    randn(cfg.linear_num_value_heads, hidden),
                );
                tensors.insert(
                    format!("{attn}.in_proj_a.weight"),
                    randn(cfg.linear_num_value_heads, hidden),
                );
                tensors.insert(
                    format!("{attn}.conv1d.weight"),
                    det(&[conv_dim, 1, cfg.linear_conv_kernel_dim]),
                );
                tensors.insert(
                    format!("{attn}.dt_bias"),
                    det(&[cfg.linear_num_value_heads]),
                );
                tensors.insert(format!("{attn}.A_log"), det(&[cfg.linear_num_value_heads]));
                tensors.insert(
                    format!("{attn}.norm.weight"),
                    Tensor::ones(cfg.linear_value_head_dim, DType::F32, device).unwrap(),
                );
                tensors.insert(format!("{attn}.out_proj.weight"), randn(hidden, value_dim));
            } else {
                let attn = format!("{prefix}.self_attn");
                tensors.insert(
                    format!("{attn}.q_proj.weight"),
                    randn(cfg.num_attention_heads * cfg.head_dim * 2, hidden),
                );
                tensors.insert(
                    format!("{attn}.k_proj.weight"),
                    randn(cfg.num_key_value_heads * cfg.head_dim, hidden),
                );
                tensors.insert(
                    format!("{attn}.v_proj.weight"),
                    randn(cfg.num_key_value_heads * cfg.head_dim, hidden),
                );
                tensors.insert(
                    format!("{attn}.o_proj.weight"),
                    randn(hidden, cfg.num_attention_heads * cfg.head_dim),
                );
                tensors.insert(format!("{attn}.q_norm.weight"), det(&[cfg.head_dim]));
                tensors.insert(format!("{attn}.k_norm.weight"), det(&[cfg.head_dim]));
            }
            tensors.insert(format!("{prefix}.input_layernorm.weight"), det(&[hidden]));
            tensors.insert(
                format!("{prefix}.post_attention_layernorm.weight"),
                det(&[hidden]),
            );
            tensors.insert(
                format!("{prefix}.mlp.gate_proj.weight"),
                randn(cfg.intermediate_size, hidden),
            );
            tensors.insert(
                format!("{prefix}.mlp.up_proj.weight"),
                randn(cfg.intermediate_size, hidden),
            );
            tensors.insert(
                format!("{prefix}.mlp.down_proj.weight"),
                randn(hidden, cfg.intermediate_size),
            );
        }
        tensors
    }

    /// Proves the decode CUDA graph is captured and replayed through the
    /// same entry points the generation loop in `model.rs` calls
    /// (`prepare_decode_graph` after prefill, `decode_step_graph` per
    /// token): the capture flag and the replay probe must both move, and
    /// the replayed logits must match an identical eager model. A tiny
    /// three-layer model keeps capture plus two replays around a second;
    /// without a CUDA device the test is a no-op.
    #[cfg(feature = "cuda")]
    #[test]
    fn decode_graph_captures_and_replays() -> Result<(), Error> {
        // Serialize with the other GPU self-tests: concurrent CUDA-graph
        // captures on one device are not safe.
        let _gpu = crate::backbones::qwen3_vl::text::GPU_SELFTEST_LOCK.lock();
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let cfg = tiny_graph_config();
        cfg.validate()?;
        let tensors = tiny_graph_tensors(&cfg, &device);

        let vb = VarBuilder::from_tensors(tensors.clone(), DType::BF16, &device);
        let graphed = OvisOcr2TextModel::load(&cfg, vb)?;
        // Tied LM head, built exactly as `OvisOcr2::from_dir` builds it.
        let lm_head = Linear::new(graphed.token_embedding_weight(), None);

        // Prefill four tokens, then let the first decode step capture — the
        // production ordering. Preparing alone must not capture: a page
        // stopped by its first token never pays for a graph.
        let prompt = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let prompt_positions = Tensor::from_vec(
            vec![0i64, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            (3, 1, 4),
            &device,
        )
        .unwrap();
        let embeds = graphed.embed(&prompt)?;
        graphed.forward(&embeds, &prompt_positions)?;
        graphed.prepare_decode_graph(4, 16)?;
        assert!(
            !graphed.decode_graph_captured(),
            "capture is lazy until the first decode step"
        );
        DECODE_GRAPH_REPLAYS.store(0, std::sync::atomic::Ordering::Relaxed);

        let token = Tensor::from_vec(vec![7u32], (1, 1), &device).unwrap();
        let pos4 = Tensor::from_vec(vec![4i64; 3], (3, 1, 1), &device).unwrap();
        let embed = graphed.embed(&token)?;
        let logits = graphed.decode_step_graph(&embed, &pos4, &lm_head)?;
        let logits_graph = logits.expect("the captured graph should serve the step");
        assert!(
            graphed.decode_graph_captured(),
            "the first decode step must capture the graph"
        );
        assert_eq!(logits_graph.dims(), &[cfg.vocab_size]);
        let embed = graphed.embed(&token)?;
        let pos5 = Tensor::from_vec(vec![5i64; 3], (3, 1, 1), &device).unwrap();
        graphed
            .decode_step_graph(&embed, &pos5, &lm_head)?
            .expect("the captured graph should serve the second step");
        assert_eq!(
            DECODE_GRAPH_REPLAYS.load(std::sync::atomic::Ordering::Relaxed),
            2,
            "decode steps must replay the graph, not fall back to eager"
        );

        // Eager reference: an identical model without a captured graph.
        let vb = VarBuilder::from_tensors(tensors, DType::BF16, &device);
        let eager = OvisOcr2TextModel::load(&cfg, vb)?;
        let embeds = eager.embed(&prompt)?;
        eager.forward(&embeds, &prompt_positions)?;
        let hidden = eager.forward(&eager.embed(&token)?, &pos4)?;
        let logits_eager = Linear::new(eager.token_embedding_weight(), None)
            .forward(
                &hidden
                    .i((0, 0, ..))
                    .and_then(|last| last.unsqueeze(0))
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "eager reference input", e))?,
            )
            .and_then(|logits| logits.squeeze(0))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "eager reference logits", e))?;
        // The graph attends with masked SDPA while the eager path uses
        // flash attention; same math to bf16 resolution.
        let graph_f32 = logits_graph.to_dtype(DType::F32).unwrap();
        let eager_f32 = logits_eager.to_dtype(DType::F32).unwrap();
        let worst = (graph_f32 - eager_f32)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .fold(0f32, f32::max);
        assert!(
            worst < 0.05,
            "graph logits diverged from eager: max|delta| = {worst}"
        );
        Ok(())
    }

    /// Scenario outcome: the model, the eager step's logits, and the KV
    /// storage layouts read right after the recovery.
    #[cfg(feature = "cuda")]
    type FailedCaptureOutcome = (OvisOcr2TextModel, Tensor, Vec<(usize, usize)>);

    /// Runs a prefill plus one lazy-capture decode step under the given hook
    /// configuration, then one eager step. Returns the model, the eager
    /// step's logits, and the KV storage layouts read right after the
    /// recovery (a further eager step grows the organic storage again and
    /// would mask the shrink).
    #[cfg(feature = "cuda")]
    fn failed_capture_scenario(
        cfg: &OvisOcr2TextConfig,
        tensors: &std::collections::HashMap<String, Tensor>,
        device: &Device,
        configure: fn(&mut TestHooks),
    ) -> Result<FailedCaptureOutcome, Error> {
        let prompt = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), device).unwrap();
        let prompt_positions = Tensor::from_vec(
            vec![0i64, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            (3, 1, 4),
            device,
        )
        .unwrap();
        let token = Tensor::from_vec(vec![7u32], (1, 1), device).unwrap();
        let pos4 = Tensor::from_vec(vec![4i64; 3], (3, 1, 1), device).unwrap();
        let vb = VarBuilder::from_tensors(tensors.clone(), DType::BF16, device);
        let mut model = OvisOcr2TextModel::load(cfg, vb)?;
        configure(&mut model.hooks);
        let lm_head = Linear::new(model.token_embedding_weight(), None);
        let embeds = model.embed(&prompt)?;
        model.forward(&embeds, &prompt_positions)?;
        model.prepare_decode_graph(4, 16)?;
        // The first decode step triggers the lazy capture; the hook fails it.
        let embed = model.embed(&token)?;
        let served = model.decode_step_graph(&embed, &pos4, &lm_head)?;
        assert!(served.is_none(), "a failed capture must fall back eager");
        assert!(!model.decode_graph_captured());
        let layouts = model
            .layers
            .iter()
            .filter_map(|layer| layer.fixed_storage_layout())
            .collect();
        // Eager decode continues on the preserved prompt history.
        let hidden = model.forward(&embed, &pos4)?;
        let logits = lm_head
            .forward(
                &hidden
                    .i((0, 0, ..))
                    .and_then(|last| last.unsqueeze(0))
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "eager input", e))?,
            )
            .and_then(|logits| logits.squeeze(0))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "eager logits", e))?;
        Ok((model, logits, layouts))
    }

    /// One prefill plus eager step on a clean model that never tried to
    /// capture: the reference the recovered model's output must match.
    #[cfg(feature = "cuda")]
    fn eager_reference_logits(
        cfg: &OvisOcr2TextConfig,
        tensors: std::collections::HashMap<String, Tensor>,
        device: &Device,
    ) -> Result<Tensor, Error> {
        let prompt = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), device).unwrap();
        let prompt_positions = Tensor::from_vec(
            vec![0i64, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            (3, 1, 4),
            device,
        )
        .unwrap();
        let pos4 = Tensor::from_vec(vec![4i64; 3], (3, 1, 1), device).unwrap();
        let vb = VarBuilder::from_tensors(tensors, DType::BF16, device);
        let reference = OvisOcr2TextModel::load(cfg, vb)?;
        let lm_head = Linear::new(reference.token_embedding_weight(), None);
        let embeds = reference.embed(&prompt)?;
        reference.forward(&embeds, &prompt_positions)?;
        let embed = reference.embed(&Tensor::from_vec(vec![7u32], (1, 1), device).unwrap())?;
        let hidden = reference.forward(&embed, &pos4)?;
        lm_head
            .forward(
                &hidden
                    .i((0, 0, ..))
                    .and_then(|last| last.unsqueeze(0))
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "reference input", e))?,
            )
            .and_then(|logits| logits.squeeze(0))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "reference logits", e))
    }

    /// max |a-b| in F32.
    #[cfg(feature = "cuda")]
    fn max_abs_delta(a: &Tensor, b: &Tensor) -> f32 {
        let a = a.to_dtype(DType::F32).unwrap();
        let b = b.to_dtype(DType::F32).unwrap();
        (a - b)
            .unwrap()
            .abs()
            .unwrap()
            .flatten_all()
            .unwrap()
            .to_vec1::<f32>()
            .unwrap()
            .into_iter()
            .fold(0f32, f32::max)
    }

    /// A failed capture must shrink the preallocated KV buckets back to the
    /// organic eager form — preserving the prompt history — so the eager
    /// fallback and later pages do not carry the spare capacity. The
    /// `skip_release` control proves the layout assertions catch a missing
    /// release. Without a CUDA device the test is a no-op.
    #[cfg(feature = "cuda")]
    #[test]
    fn failed_capture_shrinks_buckets_and_eager_decode_survives() -> Result<(), Error> {
        // Serialize with the other GPU self-tests: see GPU_SELFTEST_LOCK.
        let _gpu = crate::backbones::qwen3_vl::text::GPU_SELFTEST_LOCK.lock();
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let cfg = tiny_graph_config();
        cfg.validate()?;
        let tensors = tiny_graph_tensors(&cfg, &device);

        // prompt_decode_bucket(4) is 8: before the fix the bucket stayed at
        // (1, 8) through the eager fallback.
        let (_recovered, logits, layouts) =
            failed_capture_scenario(&cfg, &tensors, &device, |hooks| {
                hooks.fail_capture = true;
            })?;
        assert_eq!(
            layouts,
            vec![(1, 4)],
            "buckets must shrink to the live prompt KV, got {layouts:?}"
        );

        // The control: skipping the release keeps the preallocated bucket,
        // proving the assertion above can fail.
        let (_kept, _, layouts) = failed_capture_scenario(&cfg, &tensors, &device, |hooks| {
            hooks.fail_capture = true;
            hooks.skip_release = true;
        })?;
        assert_eq!(
            layouts,
            vec![(1, 8)],
            "control kept its bucket: {layouts:?}"
        );

        // The preserved history must produce the same logits as a clean
        // eager model that never tried to capture.
        let reference = eager_reference_logits(&cfg, tensors, &device)?;
        let worst = max_abs_delta(&logits, &reference);
        assert!(
            worst < 0.05,
            "eager decode after a failed capture diverged: max|delta| = {worst}"
        );
        Ok(())
    }

    /// A failure while allocating the buckets (e.g. an OOM on a large
    /// bucket) is likewise soft: the page decodes eager on the preserved
    /// history and the partially allocated buckets shrink. Without a CUDA
    /// device the test is a no-op.
    #[cfg(feature = "cuda")]
    #[test]
    fn failed_bucket_allocation_shrinks_and_decodes_eager() -> Result<(), Error> {
        // Serialize with the other GPU self-tests: see GPU_SELFTEST_LOCK.
        let _gpu = crate::backbones::qwen3_vl::text::GPU_SELFTEST_LOCK.lock();
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let cfg = tiny_graph_config();
        cfg.validate()?;
        let tensors = tiny_graph_tensors(&cfg, &device);

        // Fail right after layer 1 (the full-attention layer) allocated its
        // bucket, leaving the set partially resident.
        let (_recovered, logits, layouts) =
            failed_capture_scenario(&cfg, &tensors, &device, |hooks| {
                hooks.fail_prepare_after_layer = Some(2);
            })?;
        assert_eq!(
            layouts,
            vec![(1, 4)],
            "partial buckets must shrink to the live prompt KV, got {layouts:?}"
        );
        let reference = eager_reference_logits(&cfg, tensors, &device)?;
        let worst = max_abs_delta(&logits, &reference);
        assert!(
            worst < 0.05,
            "eager decode after a failed bucket allocation diverged: max|delta| = {worst}"
        );
        Ok(())
    }

    /// A failure inside the capture body must leave the CUDA context clean:
    /// the shared helper releases the capture-referenced buffers through the
    /// drain path, so the eager fallback reads no stale
    /// CUDA_ERROR_INVALID_VALUE. Without a CUDA device the test is a no-op.
    #[cfg(feature = "cuda")]
    #[test]
    fn failed_capture_body_drains_and_eager_decode_survives() -> Result<(), Error> {
        // Serialize with the other GPU self-tests: see GPU_SELFTEST_LOCK.
        let _gpu = crate::backbones::qwen3_vl::text::GPU_SELFTEST_LOCK.lock();
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let cfg = tiny_graph_config();
        cfg.validate()?;
        let tensors = tiny_graph_tensors(&cfg, &device);
        let prompt = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let prompt_positions = Tensor::from_vec(
            vec![0i64, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            (3, 1, 4),
            &device,
        )
        .unwrap();
        let token = Tensor::from_vec(vec![7u32], (1, 1), &device).unwrap();
        let pos4 = Tensor::from_vec(vec![4i64; 3], (3, 1, 1), &device).unwrap();

        let vb = VarBuilder::from_tensors(tensors.clone(), DType::BF16, &device);
        let mut model = OvisOcr2TextModel::load(&cfg, vb)?;
        // The body's second invocation is the capture itself.
        model.hooks.fail_body_on_call = Some(2);
        let lm_head = Linear::new(model.token_embedding_weight(), None);
        let embeds = model.embed(&prompt)?;
        model.forward(&embeds, &prompt_positions)?;
        model.prepare_decode_graph(4, 16)?;
        let embed = model.embed(&token)?;
        let served = model.decode_step_graph(&embed, &pos4, &lm_head)?;
        assert!(served.is_none(), "a failed capture must fall back eager");

        // The failed capture's output buffers and input bundle were
        // referenced by the capture; plain-dropping them would stash an
        // INVALID_VALUE on the context.
        let Device::Cuda(cuda) = &device else {
            unreachable!()
        };
        let status = cuda.cuda_stream().context().check_err();
        assert!(
            status.is_ok(),
            "stashed CUDA error after failed capture: {status:?}"
        );

        // Fresh allocations and the eager fallback both work afterwards.
        Tensor::zeros((256, 256), DType::BF16, &device)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "post-failure alloc", e))?;
        let hidden = model.forward(&embed, &pos4)?;
        let logits = lm_head
            .forward(
                &hidden
                    .i((0, 0, ..))
                    .and_then(|last| last.unsqueeze(0))
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "eager input", e))?,
            )
            .and_then(|logits| logits.squeeze(0))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "eager logits", e))?;
        let reference = eager_reference_logits(&cfg, tensors, &device)?;
        let worst = max_abs_delta(&logits, &reference);
        assert!(
            worst < 0.05,
            "eager decode after a failed capture body diverged: max|delta| = {worst}"
        );
        Ok(())
    }
    /// Crossing the ladder ceiling retires the graph: the buckets it
    /// referenced must shrink back to organic storage right then, so the
    /// eager tail's organic growth never plain-drops graph-referenced
    /// memory. Ceiling pinned to 8 via the test hook; the prompt is 7
    /// tokens, so ONE replayed step fills the bucket — keeping the numeric
    /// comparison at a single shared step. Longer compounding horizons
    /// flaked under concurrent GPU load on this random-weight model (cause
    /// not located; one candidate mechanism is cuBLAS picking different
    /// algorithms when memory/workspace is tight). Without a CUDA device
    /// the test is a no-op.
    #[cfg(feature = "cuda")]
    #[test]
    fn ceiling_retirement_shrinks_buckets_and_decodes_eager() -> Result<(), Error> {
        // Serialize with the other GPU self-tests: see GPU_SELFTEST_LOCK.
        let _gpu = crate::backbones::qwen3_vl::text::GPU_SELFTEST_LOCK.lock();
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let cfg = tiny_graph_config();
        cfg.validate()?;
        let tensors = tiny_graph_tensors(&cfg, &device);
        let prompt = Tensor::from_vec((1u32..8).collect::<Vec<_>>(), (1, 7), &device).unwrap();
        let prompt_positions = Tensor::from_vec(
            vec![
                0i64, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6, 0, 1, 2, 3, 4, 5, 6,
            ],
            (3, 1, 7),
            &device,
        )
        .unwrap();
        let token = Tensor::from_vec(vec![7u32], (1, 1), &device).unwrap();
        let position_at = |p: i64| Tensor::from_vec(vec![p; 3], (3, 1, 1), &device).unwrap();

        // prompt_decode_bucket(7, 8) = 8: one replayed step fills the
        // bucket (KV 7 -> 8); the second step retires the graph.
        let vb = VarBuilder::from_tensors(tensors.clone(), DType::BF16, &device);
        let mut model = OvisOcr2TextModel::load(&cfg, vb)?;
        model.hooks.decode_cache_ceiling = Some(8);
        let lm_head = Linear::new(model.token_embedding_weight(), None);
        let embeds = model.embed(&prompt)?;
        model.forward(&embeds, &prompt_positions)?;
        model.prepare_decode_graph(7, 16)?;
        let embed = model.embed(&token)?;
        let served = model.decode_step_graph(&embed, &position_at(7), &lm_head)?;
        assert!(served.is_some(), "the first step should replay the graph");
        assert!(model.decode_graph_captured());

        let embed = model.embed(&token)?;
        let served = model.decode_step_graph(&embed, &position_at(8), &lm_head)?;
        assert!(served.is_none(), "past the ceiling the step decodes eager");
        assert!(!model.decode_graph_captured());
        // Right after retirement — before the eager append grows the
        // storage organically — the bucket hugs the live length. (At the
        // ceiling the bucket is exactly full, so capacity == cur_len holds
        // either way; the replacement guarantee itself is covered by the
        // cache-level shrink tests.)
        let layouts: Vec<_> = model
            .layers
            .iter()
            .filter_map(|layer| layer.fixed_storage_layout())
            .collect();
        assert_eq!(
            layouts,
            vec![(1, 8)],
            "buckets must shrink to the live KV at the ceiling, got {layouts:?}"
        );
        // The retired path continues eager on the shrunk storage.
        let hidden = model.forward(&model.embed(&token)?, &position_at(8))?;
        let logits = lm_head
            .forward(
                &hidden
                    .i((0, 0, ..))
                    .and_then(|last| last.unsqueeze(0))
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "eager input", e))?,
            )
            .and_then(|logits| logits.squeeze(0))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "eager logits", e))?;

        // Reference: a model whose ceiling stays high replays the second
        // step too (the ladder re-captures at bucket 32). Both sides share
        // exactly one graph step, and the second step is a single
        // eager-vs-graph comparison — the same horizon the
        // decode_graph_captures_and_replays tolerance covers.
        let vb = VarBuilder::from_tensors(tensors, DType::BF16, &device);
        let reference = OvisOcr2TextModel::load(&cfg, vb)?;
        let lm_head_ref = Linear::new(reference.token_embedding_weight(), None);
        let embeds = reference.embed(&prompt)?;
        reference.forward(&embeds, &prompt_positions)?;
        reference.prepare_decode_graph(7, 16)?;
        let embed = reference.embed(&token)?;
        assert!(
            reference
                .decode_step_graph(&embed, &position_at(7), &lm_head_ref)?
                .is_some()
        );
        let embed = reference.embed(&token)?;
        let reference_logits = reference
            .decode_step_graph(&embed, &position_at(8), &lm_head_ref)?
            .expect("the ladder re-captures and serves the second step");
        let worst = max_abs_delta(&logits, &reference_logits);
        assert!(
            worst < 0.05,
            "ceiling-retired eager decode diverged from the graphed path: max|delta| = {worst}"
        );
        Ok(())
    }

    /// A shrink failure at the ladder ceiling is a hard error: the GPU is
    /// out of memory by then, and continuing eager would let a later
    /// organic append plain-drop the graph-referenced bucket. Without a
    /// CUDA device the test is a no-op.
    #[cfg(feature = "cuda")]
    #[test]
    fn ceiling_shrink_failure_fails_the_page() -> Result<(), Error> {
        // Serialize with the other GPU self-tests: see GPU_SELFTEST_LOCK.
        let _gpu = crate::backbones::qwen3_vl::text::GPU_SELFTEST_LOCK.lock();
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let cfg = tiny_graph_config();
        cfg.validate()?;
        let tensors = tiny_graph_tensors(&cfg, &device);
        let prompt = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let prompt_positions = Tensor::from_vec(
            vec![0i64, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            (3, 1, 4),
            &device,
        )
        .unwrap();
        let token = Tensor::from_vec(vec![7u32], (1, 1), &device).unwrap();
        let position_at = |p: i64| Tensor::from_vec(vec![p; 3], (3, 1, 1), &device).unwrap();

        let vb = VarBuilder::from_tensors(tensors, DType::BF16, &device);
        let mut model = OvisOcr2TextModel::load(&cfg, vb)?;
        model.hooks.decode_cache_ceiling = Some(8);
        let lm_head = Linear::new(model.token_embedding_weight(), None);
        let embeds = model.embed(&prompt)?;
        model.forward(&embeds, &prompt_positions)?;
        model.prepare_decode_graph(4, 16)?;
        for step in 0..4 {
            let embed = model.embed(&token)?;
            let served = model.decode_step_graph(&embed, &position_at(4 + step), &lm_head)?;
            assert!(served.is_some(), "step {step} should replay the graph");
        }

        // The ceiling step's shrink fails (injected at the V copy): the
        // page must fail, not continue eager on the graph-referenced bucket.
        crate::runtime::cache::FAIL_SHRINK_V_COPY.with(|flag| flag.set(true));
        let embed = model.embed(&token)?;
        let result = model.decode_step_graph(&embed, &position_at(8), &lm_head);
        crate::runtime::cache::FAIL_SHRINK_V_COPY.with(|flag| flag.set(false));
        assert!(
            result.is_err(),
            "a shrink failure at the ceiling must fail the page"
        );
        Ok(())
    }

    /// Between pages, clear_cache must free the fixed KV buckets (drained),
    /// not just the logical length: the next page's longer prompt would
    /// otherwise make the prefill append plain-drop graph-referenced
    /// storage. Without a CUDA device the test is a no-op.
    #[cfg(feature = "cuda")]
    #[test]
    fn clear_cache_releases_fixed_buckets() -> Result<(), Error> {
        // Serialize with the other GPU self-tests: see GPU_SELFTEST_LOCK.
        let _gpu = crate::backbones::qwen3_vl::text::GPU_SELFTEST_LOCK.lock();
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let cfg = tiny_graph_config();
        cfg.validate()?;
        let tensors = tiny_graph_tensors(&cfg, &device);
        let prompt = Tensor::from_vec(vec![1u32, 2, 3, 4], (1, 4), &device).unwrap();
        let prompt_positions = Tensor::from_vec(
            vec![0i64, 1, 2, 3, 0, 1, 2, 3, 0, 1, 2, 3],
            (3, 1, 4),
            &device,
        )
        .unwrap();
        let token = Tensor::from_vec(vec![7u32], (1, 1), &device).unwrap();
        let pos4 = Tensor::from_vec(vec![4i64; 3], (3, 1, 1), &device).unwrap();

        let vb = VarBuilder::from_tensors(tensors, DType::BF16, &device);
        let model = OvisOcr2TextModel::load(&cfg, vb)?;
        let lm_head = Linear::new(model.token_embedding_weight(), None);
        let embeds = model.embed(&prompt)?;
        model.forward(&embeds, &prompt_positions)?;
        model.prepare_decode_graph(4, 16)?;
        // Two decode steps: the first captures, so the buckets are
        // graph-referenced now.
        let embed = model.embed(&token)?;
        assert!(model.decode_step_graph(&embed, &pos4, &lm_head)?.is_some());
        let layouts: Vec<_> = model
            .layers
            .iter()
            .filter_map(|layer| layer.fixed_storage_layout())
            .collect();
        assert_eq!(layouts, vec![(1, 8)], "bucket live while decoding");

        model.clear_cache();
        let layouts: Vec<_> = model
            .layers
            .iter()
            .filter_map(|layer| layer.fixed_storage_layout())
            .collect();
        assert!(
            layouts.is_empty(),
            "clear_cache must free the fixed buckets, got {layouts:?}"
        );
        Ok(())
    }
}
