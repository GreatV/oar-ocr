//! KV-cache wrapper supporting head-only trimming and gather.
//!
//! `candle_nn::kv_cache::KvCache` exposes `append` / `reset` but no way to
//! roll back the cache to an earlier sequence length, nor to keep an
//! arbitrary subset of positions. HSD's verifier needs both: after a
//! tree-attention forward pass we may accept fewer tokens than were
//! appended, and the unaccepted suffix must be discarded so the next forward
//! pass sees a clean prefix.
//!
//! ## Implementation note
//!
//! Append uses `Tensor::cat`, matching the public-API behaviour of
//! `candle_nn::KvCache` before its preallocation rewrite. We tried a
//! preallocation + `slice_set` strategy (see git history) for parity with
//! `candle_nn::kv_cache::Cache`, but the resulting K/V values caused HSD
//! acceptance to collapse on the same workloads where the cat-based
//! implementation was correct, despite kv-only unit tests passing. The cause
//! turned out to be subtle and we reverted; rare per-page slowdowns observed
//! on long benchmarks remain an open issue tracked separately.

use candle_core::{Result, Tensor};

/// Append-and-trim KV cache for use during HSD verification.
///
/// `Clone` mirrors `candle_nn::kv_cache::KvCache::Clone`: it produces a
/// shallow copy that shares the same underlying `Tensor` storage. Cheap; only
/// useful for structures that need to derive `Clone` (e.g. GLM-OCR's text
/// model, which is held by value in multiple places).
#[derive(Debug, Clone)]
pub struct TrimmableKvCache {
    /// Concatenation axis (typically `2` for the seq dim of `(B, H, T, D)` tensors).
    cat_dim: usize,
    /// Cached keys, shape `(B, H, cur_len, D)`. `None` until first `append`.
    k: Option<Tensor>,
    v: Option<Tensor>,
    cur_len: usize,
    /// Configured sequence-length capacity retained for parity with
    /// `candle_nn::kv_cache::KvCache::new`. This wrapper does not enforce it.
    max_len: usize,
}

impl TrimmableKvCache {
    pub fn new(cat_dim: usize, max_len: usize) -> Self {
        Self {
            cat_dim,
            k: None,
            v: None,
            cur_len: 0,
            max_len,
        }
    }

    /// Append `(k_new, v_new)` to the cache and return the concatenated
    /// `(K_all, V_all)` tensors that the attention path will consume.
    ///
    /// Cat-based growth (not preallocated). We tried a `slice_set` /
    /// preallocated-buffer rewrite to match `candle_nn::kv_cache::Cache` —
    /// nsys profiling had pointed to per-step cat as a candidate bottleneck
    /// on long-output pages. The rewrite passed unit tests, didn't change
    /// per-page wall time on either fast or slow images, and *regressed*
    /// HSD acceptance length (AAL collapsed from ~22 → ~15 on 30 v1.5
    /// pages). Reverted: the wall-time outliers we were chasing are
    /// dominated by candle's per-op CPU dispatch on long decode loops, not
    /// by KV-cache copy overhead.
    pub fn append(&mut self, k_new: &Tensor, v_new: &Tensor) -> Result<(Tensor, Tensor)> {
        let new_len = k_new.dim(self.cat_dim)?;
        let (k_all, v_all) = match (self.k.as_ref(), self.v.as_ref()) {
            (None, None) => (k_new.clone(), v_new.clone()),
            (Some(k_old), Some(v_old)) => {
                let k = Tensor::cat(&[k_old, k_new], self.cat_dim)?.contiguous()?;
                let v = Tensor::cat(&[v_old, v_new], self.cat_dim)?.contiguous()?;
                (k, v)
            }
            _ => unreachable!("k/v must be set together"),
        };
        self.k = Some(k_all.clone());
        self.v = Some(v_all.clone());
        self.cur_len += new_len;
        Ok((k_all, v_all))
    }

    /// Drop everything at sequence indices `>= len`. No-op if `len >= cur_len`.
    pub fn trim_to(&mut self, len: usize) -> Result<()> {
        if len >= self.cur_len {
            return Ok(());
        }
        if len == 0 {
            self.reset();
            return Ok(());
        }
        let k = self
            .k
            .as_ref()
            .expect("cache populated when cur_len > 0")
            .narrow(self.cat_dim, 0, len)?
            .contiguous()?;
        let v = self
            .v
            .as_ref()
            .expect("cache populated when cur_len > 0")
            .narrow(self.cat_dim, 0, len)?
            .contiguous()?;
        self.k = Some(k);
        self.v = Some(v);
        self.cur_len = len;
        Ok(())
    }

    /// Gather the cache to keep only the supplied positions, in the supplied
    /// order. This is the operation HSD performs after a tree-attention
    /// verification pass: keep `[0..prefix_kv_len)` (the accepted history)
    /// then append the path-node positions.
    ///
    /// Each index must be `< current_seq_len()`. Indices may be repeated, but
    /// in normal HSD use they are distinct.
    pub fn keep_indices(&mut self, indices: &[u32]) -> Result<()> {
        if indices.is_empty() {
            self.reset();
            return Ok(());
        }
        for &i in indices {
            if (i as usize) >= self.cur_len {
                return Err(candle_core::Error::Msg(format!(
                    "TrimmableKvCache::keep_indices: index {} out of bounds (cur_len={})",
                    i, self.cur_len
                )));
            }
        }
        let (k, v) = match (self.k.as_ref(), self.v.as_ref()) {
            (Some(k), Some(v)) => (k, v),
            _ => {
                return Err(candle_core::Error::Msg(
                    "TrimmableKvCache::keep_indices on empty cache".into(),
                ));
            }
        };
        if indices.iter().enumerate().all(|(i, &x)| x as usize == i) {
            return self.trim_to(indices.len());
        }
        let device = k.device();
        let idx_t = Tensor::from_vec(indices.to_vec(), (indices.len(),), device)?;
        let new_k = k.index_select(&idx_t, self.cat_dim)?.contiguous()?;
        let new_v = v.index_select(&idx_t, self.cat_dim)?.contiguous()?;
        self.k = Some(new_k);
        self.v = Some(new_v);
        self.cur_len = indices.len();
        Ok(())
    }

    pub fn current_seq_len(&self) -> usize {
        self.cur_len
    }

    pub fn max_seq_len(&self) -> usize {
        self.max_len
    }

    pub fn reset(&mut self) {
        self.k = None;
        self.v = None;
        self.cur_len = 0;
    }

    /// Borrow the current K cache, if any.
    pub fn k(&self) -> Option<&Tensor> {
        self.k.as_ref()
    }

    /// Borrow the current V cache, if any.
    pub fn v(&self) -> Option<&Tensor> {
        self.v.as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::{DType, Device};

    fn dev() -> Device {
        Device::Cpu
    }

    #[test]
    fn append_grows_and_returns_full_cache() -> Result<()> {
        let mut c = TrimmableKvCache::new(2, 64);
        let a = Tensor::zeros((1, 2, 3, 4), DType::F32, &dev())?;
        let b = Tensor::ones((1, 2, 5, 4), DType::F32, &dev())?;
        let (k1, _) = c.append(&a, &a)?;
        assert_eq!(k1.dims(), &[1, 2, 3, 4]);
        assert_eq!(c.current_seq_len(), 3);
        let (k2, _) = c.append(&b, &b)?;
        assert_eq!(k2.dims(), &[1, 2, 8, 4]);
        assert_eq!(c.current_seq_len(), 8);
        Ok(())
    }

    #[test]
    fn trim_to_shorter_drops_tail() -> Result<()> {
        let mut c = TrimmableKvCache::new(2, 64);
        let t = Tensor::zeros((1, 2, 6, 4), DType::F32, &dev())?;
        c.append(&t, &t)?;
        c.trim_to(4)?;
        assert_eq!(c.current_seq_len(), 4);
        assert_eq!(c.k().unwrap().dims(), &[1, 2, 4, 4]);
        Ok(())
    }

    #[test]
    fn trim_to_zero_resets() -> Result<()> {
        let mut c = TrimmableKvCache::new(2, 64);
        let t = Tensor::zeros((1, 2, 6, 4), DType::F32, &dev())?;
        c.append(&t, &t)?;
        c.trim_to(0)?;
        assert_eq!(c.current_seq_len(), 0);
        assert!(c.k().is_none());
        assert!(c.v().is_none());
        Ok(())
    }

    #[test]
    fn trim_to_longer_is_noop() -> Result<()> {
        let mut c = TrimmableKvCache::new(2, 64);
        let t = Tensor::zeros((1, 2, 3, 4), DType::F32, &dev())?;
        c.append(&t, &t)?;
        c.trim_to(10)?;
        assert_eq!(c.current_seq_len(), 3);
        Ok(())
    }

    #[test]
    fn reset_then_append_works() -> Result<()> {
        let mut c = TrimmableKvCache::new(2, 64);
        let t = Tensor::zeros((1, 2, 3, 4), DType::F32, &dev())?;
        c.append(&t, &t)?;
        c.reset();
        let s = Tensor::ones((1, 2, 2, 4), DType::F32, &dev())?;
        let (k, _) = c.append(&s, &s)?;
        assert_eq!(k.dims(), &[1, 2, 2, 4]);
        assert_eq!(c.current_seq_len(), 2);
        Ok(())
    }

    #[test]
    fn trim_then_append_concats_correctly() -> Result<()> {
        let mut c = TrimmableKvCache::new(2, 64);
        let a = Tensor::zeros((1, 2, 6, 4), DType::F32, &dev())?;
        let b = Tensor::ones((1, 2, 3, 4), DType::F32, &dev())?;
        c.append(&a, &a)?;
        c.trim_to(4)?;
        let (k, _) = c.append(&b, &b)?;
        assert_eq!(k.dims(), &[1, 2, 7, 4]);
        assert_eq!(c.current_seq_len(), 7);
        Ok(())
    }

    #[test]
    fn empty_then_trim_is_noop() -> Result<()> {
        let mut c = TrimmableKvCache::new(2, 64);
        c.trim_to(5)?;
        assert_eq!(c.current_seq_len(), 0);
        Ok(())
    }

    /// Build a deterministic cache where K[..., t, 0] == t (so we can verify
    /// the gathered ordering after `keep_indices`).
    fn build_indexed_cache(len: usize) -> Result<TrimmableKvCache> {
        let mut c = TrimmableKvCache::new(2, 128);
        for t in 0..len {
            let k = Tensor::from_vec(vec![t as f32, 0.0, 0.0, 0.0], (1, 1, 1, 4), &dev())?;
            c.append(&k, &k)?;
        }
        Ok(c)
    }

    #[test]
    fn keep_indices_gathers_in_order() -> Result<()> {
        let mut c = build_indexed_cache(8)?;
        c.keep_indices(&[0, 1, 3, 5])?;
        assert_eq!(c.current_seq_len(), 4);
        let k = c.k().unwrap();
        let raw: Vec<f32> = k.flatten_all()?.to_vec1()?;
        assert_eq!(raw[0], 0.0);
        assert_eq!(raw[4], 1.0);
        assert_eq!(raw[8], 3.0);
        assert_eq!(raw[12], 5.0);
        Ok(())
    }

    #[test]
    fn keep_indices_prefix_uses_trim_fast_path() -> Result<()> {
        let mut c = build_indexed_cache(6)?;
        c.keep_indices(&[0, 1, 2])?;
        assert_eq!(c.current_seq_len(), 3);
        Ok(())
    }

    #[test]
    fn keep_indices_empty_resets() -> Result<()> {
        let mut c = build_indexed_cache(3)?;
        c.keep_indices(&[])?;
        assert_eq!(c.current_seq_len(), 0);
        assert!(c.k().is_none());
        Ok(())
    }

    #[test]
    fn keep_indices_out_of_bounds_errors() {
        let mut c = build_indexed_cache(3).unwrap();
        let err = c.keep_indices(&[0, 5]).unwrap_err().to_string();
        assert!(err.contains("out of bounds"), "unexpected error: {err}");
    }

    #[test]
    fn keep_indices_then_append_works() -> Result<()> {
        let mut c = build_indexed_cache(5)?;
        c.keep_indices(&[1, 3])?;
        let extra = Tensor::from_vec(vec![99.0f32, 0.0, 0.0, 0.0], (1, 1, 1, 4), &dev())?;
        c.append(&extra, &extra)?;
        assert_eq!(c.current_seq_len(), 3);
        let raw: Vec<f32> = c.k().unwrap().flatten_all()?.to_vec1()?;
        assert_eq!(raw[0], 1.0);
        assert_eq!(raw[4], 3.0);
        assert_eq!(raw[8], 99.0);
        Ok(())
    }
}
