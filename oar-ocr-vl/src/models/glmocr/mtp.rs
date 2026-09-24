//! GLM-OCR multi-token predictor used for lossless greedy speculation.
//!
//! The checkpoint stores one trained MTP layer after the 16 target decoder
//! layers. It is recurrently reused to propose several tokens; the caller
//! chooses the verification width independently of the checkpoint layer.

use super::config::GlmOcrTextConfig;
use super::text::{GlmOcrTextDecoderLayer, GlmOcrTextRotaryEmbedding};
use crate::error::Error;
use crate::runtime::decoder_graph::{
    CudaGraphDrainGuard, CudaGraphKvLengths, DecoderCudaGraph, capture_decoder_graph,
    cuda_graph_error,
};
use crate::utils::candle_to_ocr_inference;
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{
    Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_no_bias, rms_norm,
};
use std::cell::RefCell;

#[derive(Debug)]
pub(super) struct GlmOcrMtpModel {
    graph: RefCell<Option<DecoderCudaGraph<MtpGraphInputs>>>,
    embed_tokens: Embedding,
    enorm: RmsNorm,
    hnorm: RmsNorm,
    eh_proj: Linear,
    layer: GlmOcrTextDecoderLayer,
    shared_norm: RmsNorm,
    shared_head: Linear,
    rotary_emb: GlmOcrTextRotaryEmbedding,
    // Must stay the last field: it drops last and drains CUDA errors the
    // other fields' frees may stash (see CudaGraphDrainGuard).
    _drain_guard: CudaGraphDrainGuard,
}

/// Inputs of the MTP draft graph: the sampled token, the previous step's
/// hidden state, its positions, and the shared length tensors.
#[cfg(feature = "cuda")]
struct MtpGraphInputs {
    token: Tensor,
    previous_hidden: Tensor,
    positions: Tensor,
    query_lengths: Tensor,
    kv_lengths: CudaGraphKvLengths,
}

impl GlmOcrMtpModel {
    pub(super) fn load(
        cfg: &GlmOcrTextConfig,
        layer_index: usize,
        vb_language_model: VarBuilder,
    ) -> Result<Self, Error> {
        let vb = vb_language_model.pp("layers").pp(layer_index);
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "load MTP embeddings", e))?;
        let enorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("enorm"))
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "load MTP embedding norm", e))?;
        let hnorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("hnorm"))
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "load MTP hidden norm", e))?;
        let eh_proj = linear_no_bias(cfg.hidden_size * 2, cfg.hidden_size, vb.pp("eh_proj"))
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "load MTP fusion projection", e))?;
        let layer = GlmOcrTextDecoderLayer::load(cfg, vb.clone())?;
        let shared_norm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("shared_head").pp("norm"),
        )
        .map_err(|e| candle_to_ocr_inference("GLM-OCR", "load MTP shared norm", e))?;
        let shared_head = linear_no_bias(
            cfg.hidden_size,
            cfg.vocab_size,
            vb.pp("shared_head").pp("head"),
        )
        .map_err(|e| candle_to_ocr_inference("GLM-OCR", "load MTP shared head", e))?;
        let rotary_emb = GlmOcrTextRotaryEmbedding::new(cfg, vb.device())?;

        let _drain_guard = CudaGraphDrainGuard::new(vb_language_model.device());
        Ok(Self {
            graph: RefCell::new(None),
            embed_tokens,
            enorm,
            hnorm,
            eh_proj,
            layer,
            shared_norm,
            shared_head,
            rotary_emb,
            _drain_guard,
        })
    }

    fn fuse_inputs(
        &self,
        input_ids: &Tensor,
        previous_hidden_states: &Tensor,
        mask_first_position: bool,
    ) -> Result<Tensor, Error> {
        let mut inputs_embeds = self
            .embed_tokens
            .forward(input_ids)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "MTP embedding", e))?;
        if mask_first_position {
            let seq_len = inputs_embeds
                .dim(1)
                .map_err(|e| candle_to_ocr_inference("GLM-OCR", "MTP sequence length", e))?;
            let hidden_size = inputs_embeds
                .dim(2)
                .map_err(|e| candle_to_ocr_inference("GLM-OCR", "MTP hidden size", e))?;
            let zero = Tensor::zeros(
                (1, 1, hidden_size),
                inputs_embeds.dtype(),
                inputs_embeds.device(),
            )
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "MTP zero first embedding", e))?;
            inputs_embeds = if seq_len == 1 {
                zero
            } else {
                let tail = inputs_embeds
                    .narrow(1, 1, seq_len - 1)
                    .map_err(|e| candle_to_ocr_inference("GLM-OCR", "MTP embedding tail", e))?;
                Tensor::cat(&[&zero, &tail], 1)
                    .map_err(|e| candle_to_ocr_inference("GLM-OCR", "MTP masked embeddings", e))?
            };
        }

        let inputs_embeds = self
            .enorm
            .forward(&inputs_embeds)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "MTP embedding norm", e))?;
        let previous_hidden_states = self
            .hnorm
            .forward(previous_hidden_states)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "MTP hidden norm", e))?;
        let fused = Tensor::cat(&[&inputs_embeds, &previous_hidden_states], D::Minus1)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "MTP concatenate inputs", e))?;
        self.eh_proj
            .forward(&fused)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "MTP fusion projection", e))
    }

    fn tokens_from_hidden(&self, hidden_states: &Tensor) -> Result<Tensor, Error> {
        let hidden_states = self
            .shared_norm
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "MTP shared norm", e))?;
        self.shared_head
            .forward(&hidden_states)
            .and_then(|logits| logits.squeeze(0))
            .and_then(|logits| logits.argmax(D::Minus1))
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "MTP shared head argmax", e))
    }

    fn forward_tokens(
        &self,
        input_ids: &Tensor,
        previous_hidden_states: &Tensor,
        position_ids: &Tensor,
        mask_first_position: bool,
    ) -> Result<(Tensor, Tensor), Error> {
        let hidden_states =
            self.fuse_inputs(input_ids, previous_hidden_states, mask_first_position)?;
        let (cos, sin) = self.rotary_emb.forward(&hidden_states, position_ids)?;
        let hidden_states = self.layer.forward(&hidden_states, &cos, &sin, None)?;
        let tokens = self.tokens_from_hidden(&hidden_states)?;
        Ok((hidden_states, tokens))
    }

    fn forward_dynamic(
        &self,
        input_ids: &Tensor,
        previous_hidden_states: &Tensor,
        position_ids: &Tensor,
        query_lengths: &Tensor,
        kv_lengths: &Tensor,
    ) -> Result<(Tensor, Tensor), Error> {
        let hidden_states = self.fuse_inputs(input_ids, previous_hidden_states, false)?;
        let (cos, sin) = self.rotary_emb.forward(&hidden_states, position_ids)?;
        let hidden_states =
            self.layer
                .forward_dynamic(&hidden_states, &cos, &sin, query_lengths, kv_lengths)?;
        let tokens = self.tokens_from_hidden(&hidden_states)?;
        Ok((hidden_states, tokens))
    }

    /// Synchronize MTP with a target-model span and return its first proposal.
    pub(super) fn sync_target_span(
        &self,
        shifted_input_ids: &Tensor,
        target_hidden_states: &Tensor,
        position_ids: &Tensor,
        mask_first_position: bool,
    ) -> Result<(Tensor, Tensor), Error> {
        self.forward_tokens(
            shifted_input_ids,
            target_hidden_states,
            position_ids,
            mask_first_position,
        )
    }

    /// Recurrently propose one more token from the preceding MTP hidden state.
    pub(super) fn predict_single(
        &self,
        input_id: &Tensor,
        previous_hidden_state: &Tensor,
        position_ids: &Tensor,
    ) -> Result<(Tensor, Tensor), Error> {
        let kv_len = self.kv_cache_len().saturating_add(1);
        if let Some(output) =
            self.replay_cuda_graph(input_id, previous_hidden_state, position_ids, kv_len)?
        {
            return Ok(output);
        }
        self.forward_tokens(input_id, previous_hidden_state, position_ids, false)
    }

    pub(super) fn prepare_cuda_graph(&self, cache_len: usize) -> Result<(), Error> {
        if std::env::var_os("OAR_VL_DISABLE_CUDA_GRAPH").is_some()
            || std::env::var_os("OAR_GLMOCR_DISABLE_CUDA_GRAPH").is_some()
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

    fn capture_cuda_graph(&self, cache_len: usize) -> Result<(), Error> {
        let device = self.embed_tokens.embeddings().device();
        if !device.is_cuda() {
            return Ok(());
        }
        let query_len = 1;
        self.layer.prepare_dynamic_cache(query_len, cache_len)?;
        let hidden_size = self
            .embed_tokens
            .embeddings()
            .dim(1)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "MTP graph hidden size", e))?;
        let device = self.embed_tokens.embeddings().device();
        let inputs = MtpGraphInputs {
            token: Tensor::zeros((1, 1), DType::U32, device)
                .map_err(|e| candle_to_ocr_inference("GLM-OCR", "MTP graph token", e))?,
            previous_hidden: Tensor::zeros(
                (1, 1, hidden_size),
                self.embed_tokens.embeddings().dtype(),
                device,
            )
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "MTP graph hidden input", e))?,
            positions: Tensor::zeros((3, 1, 1), DType::I64, device)
                .map_err(|e| candle_to_ocr_inference("GLM-OCR", "MTP graph positions", e))?,
            query_lengths: Tensor::new(&[0u32, 1u32], device)
                .map_err(|e| candle_to_ocr_inference("GLM-OCR", "MTP query lengths", e))?,
            kv_lengths: CudaGraphKvLengths::new(query_len, device)
                .map_err(|e| candle_to_ocr_inference("GLM-OCR", "MTP KV lengths", e))?,
        };
        let graph =
            capture_decoder_graph(device, "GLM-OCR", self, inputs, Self::graph_body, cache_len)?;
        self.clear_kv_cache();
        *self.graph.borrow_mut() = Some(graph);
        Ok(())
    }

    /// The captured MTP step: next hidden state plus its greedy token, as a
    /// bare `fn` over the registered inputs.
    #[cfg(feature = "cuda")]
    fn graph_body(this: &Self, inputs: &MtpGraphInputs) -> Result<Vec<Tensor>, Error> {
        let (hidden, token) = this.forward_dynamic(
            &inputs.token,
            &inputs.previous_hidden,
            &inputs.positions,
            &inputs.query_lengths,
            inputs.kv_lengths.tensor(),
        )?;
        Ok(vec![hidden, token])
    }

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
        if input_id.shape() != captured.inputs.token.shape()
            || previous_hidden_state.shape() != captured.inputs.previous_hidden.shape()
            || position_ids.shape() != captured.inputs.positions.shape()
        {
            return Ok(None);
        }
        captured
            .inputs
            .token
            .slice_set(input_id, 0, 0)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "copy MTP token", e))?;
        captured
            .inputs
            .previous_hidden
            .slice_set(previous_hidden_state, 0, 0)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "copy MTP hidden", e))?;
        captured
            .inputs
            .positions
            .slice_set(position_ids, 0, 0)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "copy MTP positions", e))?;
        captured
            .inputs
            .kv_lengths
            .update(kv_len)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "update MTP KV lengths", e))?;
        captured
            .graph
            .launch()
            .map_err(|e| cuda_graph_error("GLM-OCR", "launch MTP CUDA graph", e))?;
        self.layer.set_kv_cache_len(kv_len)?;
        Ok(Some((
            captured.outputs[0].clone(),
            captured.outputs[1].clone(),
        )))
    }

    fn invalidate_cuda_graph(&self) {
        if let Some(graph) = self.graph.borrow_mut().take() {
            graph.dispose();
        }
    }

    pub(super) fn disable_cuda_graph(&self) {
        self.invalidate_cuda_graph();
    }

    pub(super) fn trim_kv_cache(&self, len: usize) -> Result<(), Error> {
        self.layer.trim_kv_cache(len)
    }

    pub(super) fn clear_kv_cache(&self) {
        self.layer.clear_kv_cache();
    }

    pub(super) fn kv_cache_len(&self) -> usize {
        self.layer.kv_cache_len()
    }
}

impl Drop for GlmOcrMtpModel {
    fn drop(&mut self) {
        // A cached graph must go through dispose: plainly dropping it returns
        // graph-bound buffers to the allocator and poisons it.
        self.invalidate_cuda_graph();
    }
}
