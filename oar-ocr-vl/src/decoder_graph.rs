//! Shared single-token decoder CUDA-graph plumbing.
//!
//! Model-specific decoder code owns capture and replay because each model has
//! a different layer stack. This module centralizes the storage lifetime and
//! cache-bucket rules that must remain identical across those implementations.

/// Select a bounded power-of-two KV-cache bucket for single-token decoding.
///
/// Returning `None` means the request should stay on the eager path. A request
/// whose declared maximum exceeds `limit` may still use the largest bucket;
/// replay drops the graph and falls back to eager before the cache grows past
/// that bucket.
#[cfg(any(feature = "cuda", test))]
pub(crate) fn decoder_cache_capacity(
    prompt_len: usize,
    max_new_tokens: usize,
    limit: usize,
) -> Option<usize> {
    if max_new_tokens == 0 || prompt_len >= limit || limit == 0 {
        return None;
    }
    let required = prompt_len.saturating_add(max_new_tokens).min(limit);
    Some(required.max(1).next_power_of_two().min(limit))
}

/// Match eager decoder attention: a single query has no future token to mask,
/// while verification blocks must remain causal within the block.
#[cfg(any(feature = "cuda", test))]
pub(crate) const fn decoder_attention_is_causal(query_len: usize) -> bool {
    query_len > 1
}

#[cfg(feature = "cuda")]
use crate::utils::candle_to_ocr_inference;
#[cfg(feature = "cuda")]
use candle_core::{CpuStorage, DType, Device, InplaceOp1, Layout, Tensor};
#[cfg(feature = "cuda")]
use oar_ocr_core::core::OCRError;
#[cfg(feature = "cuda")]
use std::cell::RefCell;

#[cfg(feature = "cuda")]
pub(crate) fn cuda_graph_error(
    model_name: &str,
    context: impl Into<String>,
    source: impl std::error::Error + Send + Sync + 'static,
) -> OCRError {
    OCRError::Inference {
        model_name: model_name.to_string(),
        context: context.into(),
        source: Box::new(source),
    }
}

#[cfg(feature = "cuda")]
pub(crate) fn sync_graph_tensor(
    model_name: &str,
    tensor: &Tensor,
    operation: &'static str,
) -> Result<(), OCRError> {
    tensor
        .flatten_all()
        .and_then(|x| x.narrow(0, 0, 1))
        .and_then(|x| x.to_dtype(DType::F32))
        .and_then(|x| x.to_vec1::<f32>())
        .map(|_| ())
        .map_err(|e| candle_to_ocr_inference(model_name, operation, e))
}

/// Persistent device/host pair used to update `[0, kv_len]` before replay.
///
/// Keeping both allocations alive avoids constructing a temporary CUDA tensor
/// on every generated token. The page-locked host buffer also lets the tiny
/// copy remain ordered on the decoder stream without a whole-stream sync.
#[cfg(feature = "cuda")]
pub(crate) struct CudaGraphKvLengths {
    tensor: Tensor,
    host: RefCell<candle_core::cuda_backend::cudarc::driver::PinnedHostSlice<u32>>,
}

#[cfg(feature = "cuda")]
struct CopyPinnedKvLengths<'a> {
    source: &'a candle_core::cuda_backend::cudarc::driver::PinnedHostSlice<u32>,
}

#[cfg(feature = "cuda")]
impl InplaceOp1 for CopyPinnedKvLengths<'_> {
    fn name(&self) -> &'static str {
        "copy-pinned-cuda-graph-kv-lengths"
    }

    fn cpu_fwd(&self, _storage: &mut CpuStorage, _layout: &Layout) -> candle_core::Result<()> {
        candle_core::bail!("CUDA-graph KV lengths are CUDA-only")
    }

    fn cuda_fwd(
        &self,
        storage: &mut candle_core::CudaStorage,
        layout: &Layout,
    ) -> candle_core::Result<()> {
        let Some((start, end)) = layout.contiguous_offsets() else {
            candle_core::bail!("CUDA-graph KV lengths must be contiguous")
        };
        if end.saturating_sub(start) != 2 {
            candle_core::bail!("CUDA-graph KV lengths must contain two u32 values")
        }
        let device = storage.device.clone();
        let destination = storage.as_cuda_slice_mut::<u32>()?;
        let mut destination = destination.slice_mut(start..end);
        device.memcpy_htod(self.source, &mut destination)
    }
}

#[cfg(feature = "cuda")]
impl CudaGraphKvLengths {
    pub(crate) fn new(initial_kv_len: usize, device: &Device) -> candle_core::Result<Self> {
        use candle_core::cuda_backend::WrapErr;

        let Device::Cuda(cuda) = device else {
            candle_core::bail!("CUDA-graph KV lengths require a CUDA device")
        };
        let initial_kv_len = u32::try_from(initial_kv_len)
            .map_err(|_| candle_core::Error::Msg("KV length exceeds u32".to_string()))?;
        let tensor = Tensor::new(&[0u32, initial_kv_len], device)?;
        let stream = cuda.cuda_stream();
        // SAFETY: both u32 elements are initialized immediately below before
        // the page-locked allocation can be read or copied.
        let mut host = unsafe { stream.context().alloc_pinned::<u32>(2) }.w()?;
        let host_ptr = host.as_mut_ptr().w()?;
        // SAFETY: `host` owns two properly aligned u32 slots.
        unsafe {
            host_ptr.write(0);
            host_ptr.add(1).write(initial_kv_len);
        }
        Ok(Self {
            tensor,
            host: RefCell::new(host),
        })
    }

    pub(crate) fn tensor(&self) -> &Tensor {
        &self.tensor
    }

    pub(crate) fn update(&self, kv_len: usize) -> candle_core::Result<()> {
        use candle_core::cuda_backend::WrapErr;

        let kv_len = u32::try_from(kv_len)
            .map_err(|_| candle_core::Error::Msg("KV length exceeds u32".to_string()))?;
        let mut host = self.host.borrow_mut();
        let host_ptr = host.as_mut_ptr().w()?;
        // SAFETY: `host` owns two properly aligned u32 slots, and waiting in
        // `as_mut_ptr` makes the previous asynchronous copy safe to overwrite.
        unsafe {
            host_ptr.write(0);
            host_ptr.add(1).write(kv_len);
        }
        self.tensor
            .inplace_op1(&CopyPinnedKvLengths { source: &host })
    }
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for CudaGraphKvLengths {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaGraphKvLengths").finish_non_exhaustive()
    }
}

/// Captured storage for a batch-1, query-length-1 decoder graph.
///
/// The graph owns raw pointers into every tensor below, the model's fixed KV
/// storage, its weights, and the external LM head. Keep `graph` first so it is
/// destroyed before the tensors when this value is dropped.
#[cfg(feature = "cuda")]
pub(crate) struct SingleTokenDecoderCudaGraph {
    pub(crate) graph: candle_core::cuda_backend::cudarc::driver::CudaGraph,
    pub(crate) hidden_input: Tensor,
    pub(crate) position_input: Tensor,
    pub(crate) _query_lengths: Tensor,
    pub(crate) kv_lengths: CudaGraphKvLengths,
    pub(crate) logits_output: Tensor,
    pub(crate) cache_len: usize,
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for SingleTokenDecoderCudaGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SingleTokenDecoderCudaGraph")
            .field("hidden", &self.hidden_input.shape())
            .field("cache_len", &self.cache_len)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::{decoder_attention_is_causal, decoder_cache_capacity};

    #[test]
    fn cache_capacity_uses_bounded_power_of_two_buckets() {
        const LIMIT: usize = 16_384;
        assert_eq!(decoder_cache_capacity(1500, 256, LIMIT), Some(2048));
        assert_eq!(decoder_cache_capacity(2000, 4096, LIMIT), Some(8192));
        assert_eq!(decoder_cache_capacity(10_000, 20_000, LIMIT), Some(LIMIT));
        assert_eq!(decoder_cache_capacity(100, 0, LIMIT), None);
        assert_eq!(decoder_cache_capacity(LIMIT, 1, LIMIT), None);
        assert_eq!(decoder_cache_capacity(1, 1, 0), None);
    }

    #[test]
    fn single_token_decode_is_not_causal_but_verification_blocks_are() {
        assert!(!decoder_attention_is_causal(1));
        assert!(decoder_attention_is_causal(2));
        assert!(decoder_attention_is_causal(16));
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn persistent_kv_lengths_update_device_storage() -> candle_core::Result<()> {
        use super::CudaGraphKvLengths;
        use candle_core::Device;

        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        let lengths = CudaGraphKvLengths::new(1, &device)?;
        lengths.update(12_345)?;
        assert_eq!(lengths.tensor().to_vec1::<u32>()?, [0, 12_345]);
        Ok(())
    }
}
