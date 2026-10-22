//! Per-step logit traces for numerical alignment against upstream references.

use candle_core::{DType, Tensor};

/// Greedy generation with per-step top-3 logits for alignment tests.
#[derive(Debug, Clone)]
pub struct GenerationTrace {
    pub tokens: Vec<u32>,
    pub step_top: Vec<[(u32, f32); 3]>,
}

/// The three highest `(token, logit)` pairs of one decoding step; on ties
/// the lower token id ranks first, like the greedy argmax.
pub(crate) fn top3(logits: &Tensor) -> candle_core::Result<[(u32, f32); 3]> {
    let scores = logits
        .flatten_all()?
        .to_dtype(DType::F32)?
        .to_vec1::<f32>()?;
    let mut best: [(usize, f32); 3] = [(0, f32::NEG_INFINITY); 3];
    for (index, &value) in scores.iter().enumerate() {
        if value > best[2].1 {
            if value > best[1].1 {
                if value > best[0].1 {
                    best[2] = best[1];
                    best[1] = best[0];
                    best[0] = (index, value);
                } else {
                    best[2] = best[1];
                    best[1] = (index, value);
                }
            } else {
                best[2] = (index, value);
            }
        }
    }
    Ok(best.map(|(index, value)| (index as u32, value)))
}
