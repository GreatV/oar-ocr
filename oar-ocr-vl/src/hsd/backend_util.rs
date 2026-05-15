//! Mechanical helpers shared by every `SpecBackend` implementation.
//!
//! Each VLM backbone (GLM-OCR, MinerU, PaddleOCR-VL, HunyuanOCR) implements
//! [`super::verify::SpecBackend`] over its own text model and uses its own
//! lm_head / rep-penalty conventions. Those parts are intentionally per-backend.
//! What is *not* per-backend is the index/position-id arithmetic — that lives
//! here so the backends only have to spell out the truly model-specific work.

use candle_core::{Device, Result as CandleResult, Tensor};

use super::prefix_tree::PrefixTree;

/// Build the 3D position-id tensor for a single-token decode step.
///
/// Returns a tensor of shape `(axes, 1, 1)` filled with `kv_len + rope_delta`.
/// `axes` is the number of MRoPE axes the backbone uses (3 for GLM-OCR /
/// MinerU / PaddleOCR-VL, 4 for HunyuanOCR).
pub fn step_pos_ids(
    axes: usize,
    kv_len: usize,
    rope_delta: i64,
    device: &Device,
) -> CandleResult<Tensor> {
    let pos = kv_len as i64 + rope_delta;
    let data = vec![pos; axes];
    Tensor::from_vec(data, (axes, 1usize, 1usize), device)
}

/// Build the 3D position-id tensor for a tree-verification forward pass.
///
/// Returns a tensor of shape `(axes, 1, num_nodes)` where node `i` is placed
/// at logical position `prefix_kv + rope_delta + tree.depths[i] - 1` along
/// every MRoPE axis (depth-1 = first newly-generated token).
pub fn tree_pos_ids(
    axes: usize,
    prefix_kv: usize,
    rope_delta: i64,
    tree: &PrefixTree,
    device: &Device,
) -> CandleResult<Tensor> {
    let n = tree.num_nodes();
    let mut pos_data: Vec<i64> = Vec::with_capacity(axes * n);
    for _axis in 0..axes {
        for d in &tree.depths {
            pos_data.push(prefix_kv as i64 + rope_delta + (*d as i64) - 1);
        }
    }
    Tensor::from_vec(pos_data, (axes, 1usize, n), device)
}

/// Build the KV-cache `keep_indices` vector for `commit_verify`.
///
/// The cache keeps `[0, prefix_kv)` (the accepted history) followed by the
/// path-node positions `prefix_kv + p` for each `p` in `accepted_path`.
pub fn commit_keep_indices(prefix_kv: usize, accepted_path: &[usize]) -> Vec<u32> {
    let mut indices: Vec<u32> = Vec::with_capacity(prefix_kv + accepted_path.len());
    for i in 0..prefix_kv {
        indices.push(i as u32);
    }
    for &p in accepted_path {
        indices.push((prefix_kv + p) as u32);
    }
    indices
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hsd::matching::Candidate;
    use crate::hsd::prefix_tree::build_prefix_tree;

    fn candidate(tokens: Vec<u32>) -> Candidate {
        Candidate {
            draft_idx: 0,
            suffix_start: 0,
            tokens,
        }
    }

    #[test]
    fn step_pos_ids_three_axis() -> CandleResult<()> {
        let t = step_pos_ids(3, 10, 4, &Device::Cpu)?;
        assert_eq!(t.dims(), &[3, 1, 1]);
        let v: Vec<i64> = t.flatten_all()?.to_vec1()?;
        assert_eq!(v, vec![14, 14, 14]);
        Ok(())
    }

    #[test]
    fn step_pos_ids_four_axis_zero_delta() -> CandleResult<()> {
        let t = step_pos_ids(4, 7, 0, &Device::Cpu)?;
        assert_eq!(t.dims(), &[4, 1, 1]);
        let v: Vec<i64> = t.flatten_all()?.to_vec1()?;
        assert_eq!(v, vec![7, 7, 7, 7]);
        Ok(())
    }

    #[test]
    fn tree_pos_ids_shape_and_values() -> CandleResult<()> {
        // Build a small tree: root → 10 → 11; root → 20.
        let cands = vec![candidate(vec![10u32, 11u32]), candidate(vec![20u32])];
        let tree = build_prefix_tree(&cands);
        let t = tree_pos_ids(3, 5, 2, &tree, &Device::Cpu)?;
        let n = tree.num_nodes();
        assert_eq!(t.dims(), &[3, 1, n]);
        let v: Vec<i64> = t.flatten_all()?.to_vec1()?;
        // First axis should equal prefix_kv + rope_delta + depth - 1.
        for axis in 0..3 {
            for (i, &d) in tree.depths.iter().enumerate() {
                let expected = 5 + 2 + (d as i64) - 1;
                assert_eq!(v[axis * n + i], expected);
            }
        }
        Ok(())
    }

    #[test]
    fn commit_indices_layout() {
        let indices = commit_keep_indices(3, &[0, 2, 5]);
        assert_eq!(indices, vec![0, 1, 2, 3, 5, 8]);
    }

    #[test]
    fn commit_indices_empty_path() {
        let indices = commit_keep_indices(4, &[]);
        assert_eq!(indices, vec![0, 1, 2, 3]);
    }
}
