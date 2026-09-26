//! jina-ocr-v1 FastMTP draft head used for lossless greedy speculation.
//!
//! The checkpoint ships one trained dense MTP layer (`mtp_module.heads.0`,
//! `mtp_share_embedding_weights`/`mtp_share_lm_head`/`mtp_share_norm` all
//! true, so the embedding table, final norm, and LM head are the target
//! model's own tensors). The layer is reused recurrently to propose
//! `mtp_num_speculative_steps` tokens; greedy verification in the caller
//! accepts only the token-equality prefix, so the emitted sequence stays
//! identical to plain greedy decoding. The head is loaded only on explicit
//! request (`JinaOcrLoadOptions::with_mtp` / `OAR_JINAOCR_ENABLE_MTP`):
//! measured on the OmniDocBench demo pages (RTX 4090, bf16), adaptive MTP
//! lost to graphed plain decoding on 17 of 18 pages.

use crate::backbones::deepseek_v2::{DeepSeekV2MtpBlock, DeepSeekV2TextConfig};
use crate::error::Error;
use crate::runtime::attention::RotaryEmbedding;
#[cfg(feature = "cuda")]
use crate::runtime::decoder_graph::{
    CudaGraphKvLengths, cuda_graph_error, drop_and_drain, report_stashed_cuda_error,
    sync_graph_tensor,
};
use crate::runtime::errors::candle_to_ocr_inference;
use candle_core::Tensor;
#[cfg(feature = "cuda")]
use candle_core::{DType, Device};
use candle_nn::{Embedding, Linear, Module, RmsNorm, VarBuilder, linear_no_bias, rms_norm};
#[cfg(feature = "cuda")]
use std::cell::RefCell;

#[cfg(feature = "cuda")]
struct MtpCudaGraph {
    // The graph owns device pointers into all tensors below; dispose via
    // `dispose` so capture-touched buffers are never returned to the
    // stream-ordered allocator (see SingleTokenDecoderCudaGraph::dispose).
    graph: candle_core::cuda_backend::cudarc::driver::CudaGraph,
    token_input: Tensor,
    previous_hidden_input: Tensor,
    position_input: Tensor,
    _query_lengths: Tensor,
    kv_lengths: CudaGraphKvLengths,
    hidden_output: Tensor,
    token_output: Tensor,
    cache_len: usize,
}

#[cfg(feature = "cuda")]
impl MtpCudaGraph {
    fn dispose(self) {
        let Self {
            graph,
            token_input,
            previous_hidden_input,
            position_input,
            _query_lengths,
            kv_lengths,
            hidden_output,
            token_output,
            cache_len: _,
        } = self;
        let device = token_input.device().clone();
        report_stashed_cuda_error(&device, "CUDA graph disposal");
        drop_and_drain(graph, &device);
        drop_and_drain(token_output, &device);
        drop_and_drain(hidden_output, &device);
        drop_and_drain(kv_lengths, &device);
        drop_and_drain(_query_lengths, &device);
        drop_and_drain(position_input, &device);
        drop_and_drain(previous_hidden_input, &device);
        drop_and_drain(token_input, &device);
    }
}

#[cfg(feature = "cuda")]
impl std::fmt::Debug for MtpCudaGraph {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("JinaOcrMtpGraph")
            .field("cache_len", &self.cache_len)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
pub(crate) struct JinaOcrMtp {
    #[cfg(feature = "cuda")]
    graph: RefCell<Option<MtpCudaGraph>>,
    embed_tokens: Embedding,
    enorm: RmsNorm,
    hnorm: RmsNorm,
    eh_proj: Linear,
    block: DeepSeekV2MtpBlock,
    shared_norm: RmsNorm,
    shared_head: Linear,
    rotary: RotaryEmbedding,
}

impl JinaOcrMtp {
    /// Load the FastMTP head. The embedding table, final norm, and LM head
    /// tensors are shared with the target model (zero-copy clones).
    pub(crate) fn load(
        text_cfg: &DeepSeekV2TextConfig,
        embed_weight: Tensor,
        norm_weight: Tensor,
        head_weight: Tensor,
        vb: VarBuilder,
    ) -> Result<Self, Error> {
        let vb = vb.pp("mtp_module").pp("heads").pp(0);
        let embed_tokens = Embedding::new(embed_weight, text_cfg.hidden_size);
        let shared_norm = RmsNorm::new(norm_weight, text_cfg.rms_norm_eps);
        let shared_head = Linear::new(head_weight, None);
        let enorm = rms_norm(text_cfg.hidden_size, text_cfg.rms_norm_eps, vb.pp("enorm"))
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "load MTP enorm", e))?;
        let hnorm = rms_norm(text_cfg.hidden_size, text_cfg.rms_norm_eps, vb.pp("hnorm"))
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "load MTP hnorm", e))?;
        let eh_proj = linear_no_bias(
            text_cfg.hidden_size * 2,
            text_cfg.hidden_size,
            vb.pp("eh_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("JinaOCR", "load MTP eh_proj", e))?;
        let block = DeepSeekV2MtpBlock::load(text_cfg, vb.pp("mtp_block"))?;
        let rotary =
            RotaryEmbedding::new_dynamic(text_cfg.head_dim()?, text_cfg.rope_theta, vb.device())?;
        Ok(Self {
            #[cfg(feature = "cuda")]
            graph: RefCell::new(None),
            embed_tokens,
            enorm,
            hnorm,
            eh_proj,
            block,
            shared_norm,
            shared_head,
            rotary,
        })
    }

    /// `fuse_inputs`: embed the input ids (zeroing the position-0 embedding —
    /// the draft never needs the very first token), norm both streams, and
    /// project the concatenation.
    fn fuse_inputs(
        &self,
        input_ids: &Tensor,
        previous_hidden_states: &Tensor,
        mask_first_position: bool,
    ) -> Result<Tensor, Error> {
        let mut inputs_embeds = self
            .embed_tokens
            .forward(input_ids)
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP embedding", e))?;
        if mask_first_position {
            let seq_len = inputs_embeds
                .dim(1)
                .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP sequence length", e))?;
            let hidden_size = inputs_embeds
                .dim(2)
                .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP hidden size", e))?;
            let zero = Tensor::zeros(
                (1, 1, hidden_size),
                inputs_embeds.dtype(),
                inputs_embeds.device(),
            )
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP zero first embedding", e))?;
            inputs_embeds = if seq_len == 1 {
                zero
            } else {
                let tail = inputs_embeds
                    .narrow(1, 1, seq_len - 1)
                    .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP embedding tail", e))?;
                Tensor::cat(&[&zero, &tail], 1)
                    .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP masked embeddings", e))?
            };
        }

        let inputs_embeds = self
            .enorm
            .forward(&inputs_embeds)
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP enorm", e))?;
        let previous_hidden_states = self
            .hnorm
            .forward(previous_hidden_states)
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP hnorm", e))?;
        let fused = Tensor::cat(
            &[&inputs_embeds, &previous_hidden_states],
            candle_core::D::Minus1,
        )
        .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP concatenate inputs", e))?;
        self.eh_proj
            .forward(&fused)
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP eh_proj", e))
    }

    /// Shared-norm hidden states and their greedy proposals for one span. The
    /// returned hidden states feed the next recurrent step (FastMTP trains
    /// with this post-norm feedback).
    /// Shared-head greedy proposal for a hidden-state block, `(rows,)` u32.
    fn tokens_from_hidden(&self, hidden_states: &Tensor) -> Result<Tensor, Error> {
        self.shared_head
            .forward(hidden_states)
            .and_then(|logits| logits.squeeze(0))
            .and_then(|logits| logits.argmax(candle_core::D::Minus1))
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP shared head argmax", e))
    }

    fn forward_tokens_inner(
        &self,
        input_ids: &Tensor,
        previous_hidden_states: &Tensor,
        position_ids: &Tensor,
        mask_first_position: bool,
        want_proposals: bool,
    ) -> Result<(Tensor, Option<Tensor>), Error> {
        let hidden_states =
            self.fuse_inputs(input_ids, previous_hidden_states, mask_first_position)?;
        let (cos, sin) = self.rope_tables(position_ids, &hidden_states)?;
        let hidden_states = self.block.forward(&hidden_states, &cos, &sin)?;
        let hidden_states = self
            .shared_norm
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP shared norm", e))?;
        if !want_proposals {
            return Ok((hidden_states, None));
        }
        let tokens = self
            .shared_head
            .forward(&hidden_states)
            .and_then(|logits| logits.squeeze(0))
            .and_then(|logits| logits.argmax(candle_core::D::Minus1))
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP shared head argmax", e))?;
        Ok((hidden_states, Some(tokens)))
    }

    fn forward_tokens(
        &self,
        input_id: &Tensor,
        previous_hidden_state: &Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor), Error> {
        let (hidden_states, tokens) =
            self.forward_tokens_inner(input_id, previous_hidden_state, position_ids, false, true)?;
        Ok((hidden_states, tokens.expect("want_proposals")))
    }

    #[cfg(feature = "cuda")]
    fn forward_dynamic(
        &self,
        input_ids: &Tensor,
        previous_hidden_states: &Tensor,
        position_ids: &Tensor,
        query_lengths: &Tensor,
        kv_lengths: &Tensor,
    ) -> Result<(Tensor, Tensor), Error> {
        let hidden_states = self.fuse_inputs(input_ids, previous_hidden_states, false)?;
        let (cos, sin) = self.rope_tables(position_ids, &hidden_states)?;
        let hidden_states =
            self.block
                .forward_dynamic(&hidden_states, &cos, &sin, query_lengths, kv_lengths)?;
        let hidden_states = self
            .shared_norm
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP shared norm", e))?;
        let tokens = self
            .shared_head
            .forward(&hidden_states)
            .and_then(|logits| logits.squeeze(0))
            .and_then(|logits| logits.argmax(candle_core::D::Minus1))
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP shared head argmax", e))?;
        Ok((hidden_states, tokens))
    }

    /// `(1, batch, seq, head_dim)` -> `(batch, 1, seq, head_dim)` tables for
    /// broadcasting over heads.
    fn rope_tables(
        &self,
        position_ids: &Tensor,
        hidden_states: &Tensor,
    ) -> Result<(Tensor, Tensor), Error> {
        let (cos, sin) = self
            .rotary
            .forward_multi_axis(position_ids, hidden_states.dtype())?;
        let cos = cos
            .squeeze(0)
            .and_then(|c| c.unsqueeze(1))
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP RoPE cos layout", e))?;
        let sin = sin
            .squeeze(0)
            .and_then(|s| s.unsqueeze(1))
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP RoPE sin layout", e))?;
        Ok((cos, sin))
    }

    /// Synchronize the draft with a target span and return the span's hidden
    /// states plus the greedy proposal for the span's LAST position (the only
    /// one callers consume — scoring the whole span through the shared head
    /// would dominate every sync's cost).
    pub(crate) fn sync_target_span(
        &self,
        shifted_input_ids: &Tensor,
        target_hidden_states: &Tensor,
        position_ids: &Tensor,
        mask_first_position: bool,
    ) -> Result<(Tensor, Tensor), Error> {
        let (hidden_states, _) = self.forward_tokens_inner(
            shifted_input_ids,
            target_hidden_states,
            position_ids,
            mask_first_position,
            false,
        )?;
        let seq_len = hidden_states
            .dim(1)
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP span length", e))?;
        let last = hidden_states
            .narrow(1, seq_len - 1, 1)
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP last hidden", e))?;
        let token = self.tokens_from_hidden(&last)?;
        Ok((hidden_states, token))
    }

    /// Recurrently propose one more token from the preceding MTP hidden state.
    pub(crate) fn predict_single(
        &self,
        input_id: &Tensor,
        previous_hidden_state: &Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor), Error> {
        #[cfg(feature = "cuda")]
        {
            let kv_len = self.kv_cache_len().saturating_add(1);
            if let Some(output) =
                self.replay_cuda_graph(input_id, previous_hidden_state, position_ids, kv_len)?
            {
                return Ok(output);
            }
        }
        self.forward_tokens(input_id, previous_hidden_state, position_ids)
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn prepare_cuda_graph(&self, cache_len: usize) -> Result<(), Error> {
        if std::env::var_os("OAR_VL_DISABLE_CUDA_GRAPH").is_some()
            || std::env::var_os("OAR_JINAOCR_DISABLE_CUDA_GRAPH").is_some()
        {
            self.invalidate_cuda_graph();
            return Ok(());
        }
        if !self.embed_tokens.embeddings().device().is_cuda()
            || !matches!(
                self.embed_tokens.embeddings().dtype(),
                DType::BF16 | DType::F16
            )
        {
            self.invalidate_cuda_graph();
            return Ok(());
        }
        let reusable = self
            .graph
            .borrow()
            .as_ref()
            .is_some_and(|graph| graph.cache_len == cache_len);
        if reusable {
            return Ok(());
        }
        self.invalidate_cuda_graph();
        self.capture_cuda_graph(cache_len)
    }

    #[cfg(feature = "cuda")]
    fn capture_cuda_graph(&self, cache_len: usize) -> Result<(), Error> {
        use candle_core::cuda_backend::cudarc::driver::sys::{
            CUgraphInstantiate_flags_enum, CUstreamCaptureMode_enum,
        };

        let Device::Cuda(cuda) = self.embed_tokens.embeddings().device() else {
            return Ok(());
        };
        let query_len = 1;
        self.block.prepare_dynamic_cache(query_len, cache_len)?;
        let hidden_size = self
            .embed_tokens
            .embeddings()
            .dim(1)
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP graph hidden size", e))?;
        let device = self.embed_tokens.embeddings().device().clone();
        let token_input = Tensor::zeros((1, 1), DType::U32, &device)
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP graph token", e))?;
        let previous_hidden_input = Tensor::zeros(
            (1, 1, hidden_size),
            self.embed_tokens.embeddings().dtype(),
            &device,
        )
        .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP graph hidden input", e))?;
        let position_input = Tensor::zeros((1, 1, 1), DType::U32, &device)
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP graph positions", e))?;
        let query_lengths = Tensor::new(&[0u32, 1u32], &device)
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP query lengths", e))?;
        let kv_lengths = CudaGraphKvLengths::new(query_len, &device)
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP KV lengths", e))?;
        let stream = cuda.cuda_stream();
        let _htod_cache = cuda.enable_cuda_graph_htod_cache();

        let (warm_hidden, warm_token) = self.forward_dynamic(
            &token_input,
            &previous_hidden_input,
            &position_input,
            &query_lengths,
            kv_lengths.tensor(),
        )?;
        sync_graph_tensor("JinaOCR", &warm_token, "warm MTP CUDA graph")?;
        // Allocate the output buffers before capture so they belong to the
        // regular stream-ordered pool; a capture-time allocation lives in the
        // graph's private pool and can never be returned to the allocator
        // safely. Prime the copies so the captured run sees warm kernels.
        let hidden_output = Tensor::zeros_like(&warm_hidden)
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP hidden output", e))?;
        let token_output = Tensor::zeros_like(&warm_token)
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "MTP token output", e))?;
        hidden_output
            .slice_set(&warm_hidden, 0, 0)
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "prime MTP hidden copy", e))?;
        token_output
            .slice_set(&warm_token, 0, 0)
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "prime MTP token copy", e))?;

        stream
            .begin_capture(CUstreamCaptureMode_enum::CU_STREAM_CAPTURE_MODE_GLOBAL)
            .map_err(|e| cuda_graph_error("JinaOCR", "begin MTP CUDA graph capture", e))?;
        let captured_output: Result<(), Error> = (|| {
            let (hidden, token) = self.forward_dynamic(
                &token_input,
                &previous_hidden_input,
                &position_input,
                &query_lengths,
                kv_lengths.tensor(),
            )?;
            hidden_output
                .slice_set(&hidden, 0, 0)
                .map_err(|e| candle_to_ocr_inference("JinaOCR", "record MTP hidden copy", e))?;
            token_output
                .slice_set(&token, 0, 0)
                .map_err(|e| candle_to_ocr_inference("JinaOCR", "record MTP token copy", e))
        })();
        if let Err(error) = captured_output {
            let _ = stream.end_capture(
                CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            );
            return Err(error);
        }
        let graph = stream
            .end_capture(
                CUgraphInstantiate_flags_enum::CUDA_GRAPH_INSTANTIATE_FLAG_AUTO_FREE_ON_LAUNCH,
            )
            .map_err(|e| cuda_graph_error("JinaOCR", "end MTP CUDA graph capture", e))?
            .ok_or_else(|| Error::Config {
                message: "JinaOCR MTP capture returned no graph".to_string(),
            })?;
        graph
            .launch()
            .map_err(|e| cuda_graph_error("JinaOCR", "warm MTP CUDA graph", e))?;
        sync_graph_tensor("JinaOCR", &token_output, "sync MTP CUDA graph")?;
        self.clear_kv_cache();
        *self.graph.borrow_mut() = Some(MtpCudaGraph {
            graph,
            token_input,
            previous_hidden_input,
            position_input,
            _query_lengths: query_lengths,
            kv_lengths,
            hidden_output,
            token_output,
            cache_len,
        });
        Ok(())
    }

    #[cfg(feature = "cuda")]
    fn replay_cuda_graph(
        &self,
        input_id: &Tensor,
        previous_hidden_state: &Tensor,
        position_ids: &Tensor,
        kv_len: usize,
    ) -> Result<Option<(Tensor, Tensor)>, Error> {
        let captured_ref = self.graph.borrow();
        let Some(captured) = captured_ref.as_ref() else {
            return Ok(None);
        };
        if kv_len > captured.cache_len {
            drop(captured_ref);
            self.invalidate_cuda_graph();
            return Ok(None);
        }
        if input_id.shape() != captured.token_input.shape()
            || previous_hidden_state.shape() != captured.previous_hidden_input.shape()
            || position_ids.shape() != captured.position_input.shape()
        {
            return Ok(None);
        }
        captured
            .token_input
            .slice_set(input_id, 0, 0)
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "copy MTP token", e))?;
        captured
            .previous_hidden_input
            .slice_set(previous_hidden_state, 0, 0)
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "copy MTP hidden", e))?;
        captured
            .position_input
            .slice_set(position_ids, 0, 0)
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "copy MTP positions", e))?;
        captured
            .kv_lengths
            .update(kv_len)
            .map_err(|e| candle_to_ocr_inference("JinaOCR", "update MTP KV lengths", e))?;
        captured
            .graph
            .launch()
            .map_err(|e| cuda_graph_error("JinaOCR", "launch MTP CUDA graph", e))?;
        self.block.set_kv_cache_len(kv_len)?;
        Ok(Some((
            captured.hidden_output.clone(),
            captured.token_output.clone(),
        )))
    }

    #[cfg(feature = "cuda")]
    fn invalidate_cuda_graph(&self) {
        if let Some(graph) = self.graph.borrow_mut().take() {
            graph.dispose();
        }
    }

    #[cfg(feature = "cuda")]
    pub(crate) fn disable_cuda_graph(&self) {
        self.invalidate_cuda_graph();
    }

    pub(crate) fn trim_kv_cache(&self, len: usize) -> Result<(), Error> {
        self.block.trim_kv_cache(len)
    }

    pub(crate) fn clear_kv_cache(&self) {
        self.block.clear_kv_cache();
    }

    #[cfg(feature = "cuda")]
    fn kv_cache_len(&self) -> usize {
        self.block.kv_cache_len()
    }
}

impl Drop for JinaOcrMtp {
    fn drop(&mut self) {
        // A cached graph must go through dispose: plainly dropping it returns
        // graph-bound buffers to the allocator and poisons it.
        #[cfg(feature = "cuda")]
        self.invalidate_cuda_graph();
    }
}
