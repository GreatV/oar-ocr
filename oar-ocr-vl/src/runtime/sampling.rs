//! Greedy token selection and signed repetition penalties.
use std::collections::HashSet;

pub(crate) fn argmax_token(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &val) in logits.iter().enumerate() {
        if val.is_nan() {
            continue;
        }
        if val > best_val {
            best_val = val;
            best_idx = idx;
        }
    }
    best_idx as u32
}

pub(crate) fn apply_repetition_penalty(logits: &mut [f32], history: &[u32], penalty: f32) {
    if penalty <= 1.0 {
        return;
    }
    let mut seen = HashSet::new();
    for &token in history {
        if !seen.insert(token) {
            continue;
        }
        let idx = token as usize;
        if idx >= logits.len() {
            continue;
        }
        let val = logits[idx];
        logits[idx] = if val < 0.0 {
            val * penalty
        } else {
            val / penalty
        };
    }
}

pub(crate) struct RepetitionHistory {
    present: Vec<bool>,
    pub(crate) unique: Vec<u32>,
}

impl RepetitionHistory {
    pub(crate) fn from_tokens(vocab_size: usize, tokens: &[u32]) -> Self {
        let mut history = Self {
            present: vec![false; vocab_size],
            unique: Vec::new(),
        };
        for &token in tokens {
            history.insert(token);
        }
        history
    }

    pub(crate) fn insert(&mut self, token: u32) {
        if let Some(present) = self.present.get_mut(token as usize)
            && !*present
        {
            *present = true;
            self.unique.push(token);
        }
    }
}

pub(crate) fn apply_unique_repetition_penalty(logits: &mut [f32], unique: &[u32], penalty: f32) {
    if penalty <= 1.0 {
        return;
    }
    for &token in unique {
        if let Some(value) = logits.get_mut(token as usize) {
            *value = if *value < 0.0 {
                *value * penalty
            } else {
                *value / penalty
            };
        }
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn select_penalized_greedy_token_cuda(
    model: &str,
    logits: &candle_core::Tensor,
    unique_history: &[u32],
    penalty: f32,
) -> Result<u32, crate::Error> {
    use crate::runtime::{
        cuda::{ArgmaxFirstF32, dynamic_kv::RepetitionPenaltyF32},
        errors::candle_to_ocr_inference,
    };
    use candle_core::{DType, IndexOp, Tensor};
    let adjusted = logits
        .reshape((1, logits.elem_count()))
        .and_then(|logits| {
            // In-place processing must not modify an F32 decoder output or a
            // graph buffer. The BF16 cast already creates separate storage.
            if logits.dtype() == DType::F32 {
                logits.copy()
            } else {
                logits.to_dtype(DType::F32)
            }
        })
        .and_then(|logits| logits.contiguous())
        .map_err(|e| candle_to_ocr_inference(model, "copy F32 repetition logits", e))?;
    if !unique_history.is_empty() {
        let ids = Tensor::new(unique_history, logits.device())
            .and_then(|ids| ids.unsqueeze(0))
            .map_err(|e| candle_to_ocr_inference(model, "upload unique repetition ids", e))?;
        adjusted
            .inplace_op2(&ids, &RepetitionPenaltyF32 { penalty })
            .map_err(|e| candle_to_ocr_inference(model, "GPU repetition penalty", e))?;
    }
    // The CPU scan returns id 0 if every value is NaN or -inf. Replacing
    // NaNs with -inf preserves that case as well as ordinary lowest-id ties.
    let not_nan = adjusted
        .eq(&adjusted)
        .map_err(|e| candle_to_ocr_inference(model, "penalized NaN mask", e))?;
    let neg_inf = Tensor::new(f32::NEG_INFINITY, logits.device())
        .and_then(|value| value.broadcast_as(adjusted.dims()))
        .map_err(|e| candle_to_ocr_inference(model, "penalized NaN replacement", e))?;
    not_nan
        .where_cond(&adjusted, &neg_inf)
        .and_then(|logits| logits.apply_op1_no_bwd(&ArgmaxFirstF32))
        .and_then(|tokens| tokens.i(0))
        .and_then(|token| token.to_scalar::<u32>())
        .map_err(|e| candle_to_ocr_inference(model, "GPU penalized argmax", e))
}
