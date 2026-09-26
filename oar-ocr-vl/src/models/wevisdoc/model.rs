//! WeVisDoc (Qwen3-VL) model implementation.
//!
//! WeVisDoc-2B/4B are Qwen3-VL checkpoints fine-tuned for end-to-end page
//! parsing. Generation follows `wevisdoc/local.py` from the official
//! repository: greedy decoding (`do_sample=False`) with a stop on either EOS
//! id from `generation_config.json` (or the tokenizer's `<|im_end|>`), the
//! WeDocKit-derived system prompt, and the plain user instruction. The
//! `generation_config.json` sampling fields are ignored on purpose.

use super::config::WeVisDocConfig;
use super::processing::{
    WeVisDocImageInputs, load_image_processor_config, preprocess_image,
    validate_processor_vision_compatibility,
};
use crate::backbones::qwen3_vl::{DeepstackVisualEmbeds, Qwen3VlTextModel, Qwen3VlVisionModel};
use crate::error::Error;
use crate::runtime::attention::{
    create_generation_mask_if_needed, decode_position_buffer, row_flash_attention_available,
};
use crate::runtime::checkpoint::{collect_safetensors, load_optional_json_config};
#[cfg(feature = "cuda")]
use crate::runtime::cuda::{ArgmaxFirstBf16, ArgmaxFirstF32};
use crate::runtime::errors::{candle_to_ocr_inference, candle_to_ocr_processing};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use image::RgbImage;
use serde::Deserialize;
use std::path::Path;
use tokenizers::Tokenizer;

const MODEL_NAME: &str = "WeVisDoc";

/// Official WeVisDoc user instruction (`DEFAULT_PROMPT` in
/// `wevisdoc/prompts.py`).
pub const DEFAULT_PROMPT: &str = "Convert this document image to Markdown.";

/// Official WeVisDoc system prompt (`DEFAULT_SYSTEM_PROMPT` in
/// `wevisdoc/prompts.py`, adapted from WeDocKit).
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are an AI assistant specialized in converting PDF images to Markdown format. Please follow these instructions for the conversion:\n\n1. Text Processing:\n- Accurately recognize all text content in the PDF image without guessing or inferring.\n- Convert the recognized text into Markdown format.\n- Maintain the original document structure, including headings, paragraphs, lists, etc.\n\n2. Mathematical Formula Processing:\n- Convert all mathematical formulas to LaTeX format.\n- Enclose inline formulas with \\( \\). For example: This is an inline formula \\( E = mc^2 \\)\n- Enclose block formulas with \\[ \\]. For example: \\[ \\frac{-b \\pm \\sqrt{b^2 - 4ac}}{2a} \\]\n\n3. Table Processing:\n- Convert tables to HTML format.\n- Wrap the entire table with <table> and </table>.\n\n4. Figure Handling:\n- Ignore figures content in the PDF image. Do not attempt to describe or convert images.\n\n5. Output Format:\n- Ensure the output Markdown document has a clear structure with appropriate line breaks between elements.\n- For complex layouts, try to maintain the original document's structure and format as closely as possible.\n\nPlease strictly follow these guidelines to ensure accuracy and consistency in the conversion. Your task is to accurately convert the content of the PDF image into Markdown format without adding any extra explanations or comments.\n";

/// Upstream `--max-tokens` default (8192) in `wevisdoc/local.py`.
pub const DEFAULT_MAX_NEW_TOKENS: usize = 8_192;

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum WeVisDocEosTokenId {
    Single(u32),
    Multi(Vec<u32>),
}

#[derive(Debug, Deserialize)]
struct WeVisDocGenerationConfig {
    #[serde(default)]
    eos_token_id: Option<WeVisDocEosTokenId>,
}

/// End-to-end WeVisDoc page parser backed by Qwen3-VL.
pub struct WeVisDoc {
    device: Device,
    dtype: DType,
    cfg: WeVisDocConfig,
    image_cfg: crate::backbones::qwen_vl_processing::MinerUImageProcessorConfig,
    tokenizer: Tokenizer,
    text: Qwen3VlTextModel,
    vision: Qwen3VlVisionModel,
    lm_head: Linear,
    stop_token_ids: Vec<u32>,
    image_token_id: u32,
}

struct TextCacheGuard<'a>(&'a Qwen3VlTextModel);

impl Drop for TextCacheGuard<'_> {
    fn drop(&mut self) {
        self.0.clear_cache();
    }
}

impl WeVisDoc {
    /// Load a WeVisDoc Hugging Face model directory.
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, Error> {
        Self::from_dir_with_runtime(model_dir, crate::RuntimeConfig::new(device))
    }

    pub fn from_dir_with_runtime(
        model_dir: impl AsRef<Path>,
        runtime: crate::RuntimeConfig,
    ) -> Result<Self, Error> {
        let (device, dtype) = runtime.resolve();
        let model_dir = model_dir.as_ref();
        let cfg = WeVisDocConfig::from_path(model_dir.join("config.json"))?;
        let image_cfg = load_image_processor_config(model_dir.join("preprocessor_config.json"))?;
        validate_processor_vision_compatibility(&image_cfg, &cfg.vision_config)?;
        let tokenizer =
            Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| Error::Config {
                message: format!("failed to load WeVisDoc tokenizer.json: {e}"),
            })?;
        require_token_id(&tokenizer, "<|image_pad|>", Some(cfg.image_token_id))?;
        require_token_id(
            &tokenizer,
            "<|vision_start|>",
            Some(cfg.vision_start_token_id),
        )?;
        require_token_id(&tokenizer, "<|vision_end|>", Some(cfg.vision_end_token_id))?;
        require_token_id(&tokenizer, "<|im_start|>", None)?;
        let tokenizer_eos = require_token_id(&tokenizer, "<|im_end|>", None)?;

        let weight_files = collect_safetensors(model_dir, MODEL_NAME)?;
        // SAFETY: The model files must remain unchanged while their mmap is in use.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_files, dtype, &device)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load safetensors", e))?
        };
        let text = Qwen3VlTextModel::load(&cfg.text_config, vb.pp("model").pp("language_model"))?;
        let vision = Qwen3VlVisionModel::load(&cfg.vision_config, vb.pp("model").pp("visual"))?;
        // The output projection reuses the token embeddings when tied
        // (WeVisDoc-2B) and otherwise loads the checkpoint's lm_head
        // (WeVisDoc-4B).
        let lm_head = if cfg.tie_word_embeddings() {
            Linear::new(text.token_embedding_weight(), None)
        } else {
            let weight = vb
                .get(
                    (cfg.text_config.vocab_size, cfg.text_config.hidden_size),
                    "lm_head.weight",
                )
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load lm_head", e))?;
            Linear::new(weight, None)
        };

        let generation_cfg: Option<WeVisDocGenerationConfig> = load_optional_json_config(
            model_dir.join("generation_config.json"),
            MODEL_NAME,
            "generation_config.json",
        )?;
        let mut stop_token_ids = match generation_cfg.and_then(|cfg| cfg.eos_token_id) {
            Some(WeVisDocEosTokenId::Single(id)) => vec![id],
            Some(WeVisDocEosTokenId::Multi(ids)) => ids,
            None => Vec::new(),
        };
        stop_token_ids.push(tokenizer_eos);
        stop_token_ids.sort_unstable();
        stop_token_ids.dedup();

        let image_token_id = cfg.image_token_id;
        Ok(Self {
            device,
            dtype,
            cfg,
            image_cfg,
            tokenizer,
            text,
            vision,
            lm_head,
            stop_token_ids,
            image_token_id,
        })
    }

    /// Generate model-native Markdown for each page.
    pub fn generate(
        &self,
        images: &[RgbImage],
        max_new_tokens: usize,
    ) -> crate::error::BatchResult<String> {
        Ok(self
            .generate_tokens(images, max_new_tokens)?
            .into_iter()
            .map(|result| result.and_then(|tokens| self.decode_tokens(&tokens)))
            .collect())
    }

    /// Generate raw token ids for each input page. A single page runs through
    /// the CUDA-graph decode fast path; larger batches run a padded batch
    /// prefill and decode (the graph is single-sequence only).
    pub fn generate_tokens(
        &self,
        images: &[RgbImage],
        max_new_tokens: usize,
    ) -> crate::error::BatchResult<Vec<u32>> {
        self.generate_tokens_impl(images, max_new_tokens, LoopGuard::Off)
    }

    /// Region-recognition entry: identical to [`Self::generate_tokens`]
    /// except decoding uses the region bucket ladder and the degenerate
    /// loop guard. Only region crops run this way — full-page parsing must
    /// reproduce the reference decoding, where legitimate repeated
    /// structures (empty table rows, dot leaders, repeated headers) would
    /// risk being cut, and the wider reference bucket keeps the numerics
    /// bit-identical.
    pub(crate) fn generate_tokens_for_regions(
        &self,
        images: &[RgbImage],
        max_new_tokens: usize,
        loop_guard: LoopGuard,
    ) -> crate::error::BatchResult<Vec<u32>> {
        self.generate_tokens_impl(images, max_new_tokens, loop_guard)
    }

    fn generate_tokens_impl(
        &self,
        images: &[RgbImage],
        max_new_tokens: usize,
        loop_guard: LoopGuard,
    ) -> crate::error::BatchResult<Vec<u32>> {
        // Without the per-row flash path, batched prefill would have to
        // materialize a quadratic (B,1,S,S) mask — pages near the token
        // limit already need gigabytes for it. Running the pages through
        // the single-page decoder instead keeps every row bit-identical to
        // the reference single-page decoding at single-page memory cost.
        if !row_flash_attention_available(&self.device, self.dtype) {
            return Ok(images
                .iter()
                .map(|image| {
                    self.generate_one(image, max_new_tokens, loop_guard)
                        .map(|(tokens, _)| tokens)
                })
                .collect());
        }
        if images.len() <= 1 {
            return Ok(images
                .iter()
                .map(|image| {
                    self.generate_one(image, max_new_tokens, loop_guard)
                        .map(|(tokens, _)| tokens)
                })
                .collect());
        }
        self.generate_batch_tokens(images, max_new_tokens, loop_guard)
            .map(|results| results.into_iter().map(Ok).collect())
    }

    /// Generate one page's tokens plus whether decoding stopped on an EOS
    /// token. `false` means the token budget ran out first — the official
    /// `wevisdoc/local.py` treats that as a truncation error.
    pub(crate) fn generate_one(
        &self,
        image: &RgbImage,
        max_new_tokens: usize,
        loop_guard: LoopGuard,
    ) -> Result<(Vec<u32>, bool), Error> {
        self.text.clear_cache();
        let _cache_guard = TextCacheGuard(&self.text);
        if max_new_tokens == 0 {
            return Ok((Vec::new(), false));
        }
        let context_limit = self.cfg.text_config.max_position_embeddings;
        let image_inputs = preprocess_image(
            image,
            &self.image_cfg,
            &self.cfg.vision_config,
            &self.device,
            self.dtype,
        )?;
        let prompt = build_prompt(image_inputs.num_image_tokens, DEFAULT_SYSTEM_PROMPT);
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| Error::InvalidInput {
                message: format!("WeVisDoc: tokenizer encode failed: {e}"),
            })?;
        let input_ids = encoding.get_ids().to_vec();
        if input_ids.is_empty() {
            return Err(Error::InvalidInput {
                message: "WeVisDoc: prompt tokenization produced no tokens".to_string(),
            });
        }
        validate_generation_length(input_ids.len(), max_new_tokens, context_limit)?;

        let (inputs_embeds, deepstack) = self.prepare_inputs(&input_ids, &image_inputs)?;
        let (position_ids, rope_delta) = build_position_ids(
            &input_ids,
            image_inputs.grid_thw,
            self.cfg.vision_config.spatial_merge_size,
            self.cfg.vision_start_token_id,
            self.image_token_id,
            &self.device,
        )?;
        self.text.prepare_ar_cuda_graph(
            input_ids.len(),
            max_new_tokens,
            &self.lm_head,
            loop_guard != LoopGuard::Off,
        )?;
        let hidden =
            self.text
                .forward(&inputs_embeds, &position_ids, Some(&deepstack), None, None)?;
        let prompt_len = input_ids.len();
        let last_hidden = hidden
            .i((0, prompt_len - 1, ..))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "select prompt hidden", e))?;
        let mut logits = self.logits_from_hidden(&last_hidden)?;
        let mut generated = Vec::new();
        generated
            .try_reserve_exact(max_new_tokens)
            .map_err(|e| Error::InvalidInput {
                message: format!("WeVisDoc cannot reserve output for {max_new_tokens} tokens: {e}"),
            })?;

        for step in 0..max_new_tokens {
            let token = select_greedy_token(&logits)?;
            if self.stop_token_ids.contains(&token) {
                return Ok((generated, true));
            }
            generated.push(token);
            // A greedy loop never reaches EOS; cutting the repeated cycles
            // (one is kept) both bounds the output and saves the rest of
            // the token budget. Region crops only — see
            // `generate_tokens_with_loop_guard`.
            match loop_guard_action(loop_guard, &generated, max_new_tokens) {
                Some(LoopAction::TrimStop { period, repeats }) => {
                    generated.truncate(generated.len() - (repeats - 1) * period);
                    return Ok((generated, false));
                }
                Some(LoopAction::Stop) => return Ok((generated, false)),
                None => {}
            }
            if step + 1 == max_new_tokens {
                break;
            }

            let token_ids = Tensor::from_vec(vec![token], (1, 1), &self.device).map_err(|e| {
                candle_to_ocr_processing(
                    crate::error::ProcessingStage::TensorOperation,
                    format!("{MODEL_NAME}: create decode token"),
                    e,
                )
            })?;
            let token_embed = self.text.embed(&token_ids)?;
            let position = prompt_len as i64 + step as i64 + rope_delta;
            let position_ids = text_position_ids(position, &self.device)?;
            logits = self.text.forward_decode_logits(
                &token_embed,
                &position_ids,
                None,
                &self.lm_head,
            )?;
        }
        Ok((generated, false))
    }

    /// Padded batch generation: the same tokens as per-page generation, with
    /// one prefill and one decode step for the whole batch. Unequal prompt
    /// lengths are left-padded; prefill and decode masks hide the padded KV
    /// positions, DeepStack spans and per-row MRoPE positions shift with the
    /// padding, and each sequence stops at its own EOS.
    fn generate_batch_tokens(
        &self,
        images: &[RgbImage],
        max_new_tokens: usize,
        loop_guard: LoopGuard,
    ) -> Result<Vec<Vec<u32>>, Error> {
        let batch_size = images.len();
        let context_limit = self.cfg.text_config.max_position_embeddings;
        if max_new_tokens == 0 {
            return Ok(vec![Vec::new(); batch_size]);
        }

        let mut rows: Vec<BatchPrompt> = Vec::with_capacity(batch_size);
        for image in images {
            let image_inputs = preprocess_image(
                image,
                &self.image_cfg,
                &self.cfg.vision_config,
                &self.device,
                self.dtype,
            )?;
            let prompt = build_prompt(image_inputs.num_image_tokens, DEFAULT_SYSTEM_PROMPT);
            let encoding =
                self.tokenizer
                    .encode(prompt, false)
                    .map_err(|e| Error::InvalidInput {
                        message: format!("WeVisDoc: tokenizer encode failed: {e}"),
                    })?;
            let input_ids = encoding.get_ids().to_vec();
            if input_ids.is_empty() {
                return Err(Error::InvalidInput {
                    message: "WeVisDoc: prompt tokenization produced no tokens".to_string(),
                });
            }
            validate_generation_length(input_ids.len(), max_new_tokens, context_limit)?;
            let (inputs_embeds, deepstack) = self.prepare_inputs(&input_ids, &image_inputs)?;
            let (position_ids, rope_delta) = build_position_ids(
                &input_ids,
                image_inputs.grid_thw,
                self.cfg.vision_config.spatial_merge_size,
                self.cfg.vision_start_token_id,
                self.image_token_id,
                &self.device,
            )?;
            rows.push(BatchPrompt {
                input_ids,
                inputs_embeds,
                deepstack,
                position_ids,
                rope_delta,
            });
        }

        let seq_lens: Vec<usize> = rows.iter().map(|row| row.input_ids.len()).collect();
        let max_seq_len = *seq_lens.iter().max().ok_or_else(|| Error::InvalidInput {
            message: "WeVisDoc: empty batch is not supported".to_string(),
        })?;

        // Left-pad embeds and positions; DeepStack spans shift with the pad.
        let mut embeds_rows = Vec::with_capacity(batch_size);
        let mut position_rows = Vec::with_capacity(batch_size);
        let mut spans = Vec::with_capacity(batch_size);
        for (row, &seq_len) in rows.iter().zip(&seq_lens) {
            let pad_len = max_seq_len - seq_len;
            let embeds = if pad_len > 0 {
                let pad = Tensor::zeros(
                    (1, pad_len, self.cfg.text_config.hidden_size),
                    row.inputs_embeds.dtype(),
                    &self.device,
                )
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "create pad", e))?;
                Tensor::cat(&[&pad, &row.inputs_embeds], 1)
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "cat pad", e))?
            } else {
                row.inputs_embeds.clone()
            };
            embeds_rows.push(embeds);
            let mut positions = row.position_ids.clone();
            if pad_len > 0 {
                let pad = Tensor::zeros((3, 1, pad_len), row.position_ids.dtype(), &self.device)
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "create pad positions", e))?;
                positions = Tensor::cat(&[&pad, &positions], 2)
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "cat pad positions", e))?;
            }
            position_rows.push(positions);
            spans.push((
                row.deepstack.image_spans[0].0 + pad_len,
                row.deepstack.image_spans[0].1,
            ));
        }
        // Concatenate the per-row DeepStack feature maps in row order; rows
        // may carry different image grids, so span lengths differ.
        let mut deepstack_embeds = Vec::new();
        for layer in 0..rows[0].deepstack.embeds.len() {
            let mut parts = Vec::with_capacity(batch_size);
            for row in &rows {
                parts.push(row.deepstack.embeds[layer].clone());
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            deepstack_embeds.push(
                Tensor::cat(&refs, 0).map_err(|e| {
                    candle_to_ocr_inference(MODEL_NAME, "stack deepstack features", e)
                })?,
            );
        }
        let deepstack = DeepstackVisualEmbeds {
            image_spans: spans,
            embeds: deepstack_embeds,
        };

        let embeds_refs: Vec<&Tensor> = embeds_rows.iter().collect();
        let inputs_embeds = Tensor::cat(&embeds_refs, 0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "stack embeds", e))?;
        let position_refs: Vec<&Tensor> = position_rows.iter().collect();
        let position_ids = Tensor::cat(&position_refs, 1)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "stack positions", e))?;
        // Batched decoding only runs on devices with per-row flash, whose
        // prefill attends each row's real span directly — no quadratic
        // (B,1,S,S) mask is ever materialized. Other devices take the
        // per-page sequential path above.

        // Per-row real-token spans: left-padded rows attend only within
        // their own span, which lets the prefill use the flash kernel per
        // row instead of materialized masked scores.
        let row_spans: Vec<(usize, usize)> = seq_lens
            .iter()
            .map(|&len| (max_seq_len - len, len))
            .collect();

        self.text.clear_cache();
        // The batched prepare reuses the captured graph when the batch
        // width and prompt bucket match, and otherwise re-captures, dropping
        // any single-row graph that points at (1, H, C, D) storage first —
        // so the prefill writes straight into the graph's fixed capacity.
        // The bucket starts just past the prompt and grows with the
        // generation.
        #[cfg(feature = "cuda")]
        {
            let pads: Vec<usize> = (0..batch_size)
                .map(|row| max_seq_len - seq_lens[row])
                .collect();
            if let Err(error) = self.text.prepare_batch_ar_cuda_graph(
                batch_size,
                max_seq_len,
                max_new_tokens,
                &pads,
                &self.lm_head,
                loop_guard != LoopGuard::Off,
            ) {
                tracing::warn!(
                    "{MODEL_NAME} batch graph capture failed: {error}; continuing eager"
                );
                self.text.recover_failed_capture();
            }
        }
        let hidden = self.text.forward(
            &inputs_embeds,
            &position_ids,
            Some(&deepstack),
            None,
            Some(&row_spans),
        )?;
        let last_hidden = hidden
            .i((.., max_seq_len - 1, ..))
            .and_then(|h| h.contiguous())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "select last hidden", e))?;
        let mut logits = self
            .lm_head
            .forward(&last_hidden)
            .and_then(|l| l.squeeze(1))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "batch lm_head", e))?;
        // Per-row greedy pick on the device: only the chosen token ids
        // cross back to the host each step, never the vocab-wide logits.
        let mut tokens = argmax_rows(&logits)?;

        let mut generated: Vec<Vec<u32>> = vec![Vec::new(); batch_size];
        let mut finished = vec![false; batch_size];
        let mut positions: Vec<i64> = seq_lens
            .iter()
            .zip(rows.iter().map(|row| row.rope_delta))
            .map(|(&len, delta)| len as i64 + delta)
            .collect();
        let pad_lens: Vec<usize> = seq_lens.iter().map(|&len| max_seq_len - len).collect();
        // The graph mask is refreshed from these every step, so a reused
        // graph never reads the previous batch's padding bounds.
        #[cfg(feature = "cuda")]
        let pad_starts: Vec<u32> = pad_lens.iter().map(|&pad| pad as u32).collect();
        let mut kv_len = max_seq_len;

        for step in 0..max_new_tokens {
            if finished.iter().all(|&f| f) {
                break;
            }
            let mut next_tokens: Vec<u32> = Vec::with_capacity(batch_size);
            for row in 0..batch_size {
                if finished[row] {
                    next_tokens.push(0);
                    continue;
                }
                let token = tokens[row];
                if self.stop_token_ids.contains(&token) {
                    finished[row] = true;
                } else {
                    generated[row].push(token);
                    // Same loop guard as single-row generation: cut the
                    // repeated cycles (one is kept) and retire the row so
                    // the batch stops paying for its decode. Region crops
                    // only — see `generate_tokens_with_loop_guard`.
                    match loop_guard_action(loop_guard, &generated[row], max_new_tokens) {
                        Some(LoopAction::TrimStop { period, repeats }) => {
                            let keep = generated[row].len() - (repeats - 1) * period;
                            generated[row].truncate(keep);
                            finished[row] = true;
                        }
                        Some(LoopAction::Stop) => finished[row] = true,
                        None => {}
                    }
                }
                next_tokens.push(token);
            }
            if finished.iter().all(|&f| f) {
                break;
            }
            if step + 1 == max_new_tokens {
                break;
            }

            let decode_ids = Tensor::from_vec(next_tokens, (batch_size, 1), &self.device)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "create decode tokens", e))?;
            let embeds = self.text.embed(&decode_ids)?;
            let pos_data = decode_position_buffer(&positions, 3);
            let pos = Tensor::from_vec(pos_data, (3, batch_size, 1), &self.device)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "create decode positions", e))?;
            kv_len += 1;
            // The padded prefill filled [0, max_seq) for every row, so each
            // new token lands at the same storage offset in its own row.
            #[cfg(feature = "cuda")]
            let row_starts = vec![(kv_len - 1) as u32; batch_size];
            let gen_mask =
                create_generation_mask_if_needed(&pad_lens, kv_len, self.dtype, &self.device)
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "create decode mask", e))?;
            #[cfg(feature = "cuda")]
            let next_logits = self.text.forward_decode_logits_batch(
                &embeds,
                &pos,
                crate::runtime::decoder_graph::BatchDecodeRows {
                    row_starts: &row_starts,
                    pad_lens: &pad_starts,
                },
                kv_len,
                gen_mask.as_ref(),
                &self.lm_head,
            )?;
            #[cfg(not(feature = "cuda"))]
            let next_logits = {
                let hidden = self
                    .text
                    .forward(&embeds, &pos, None, gen_mask.as_ref(), None)?;
                self.lm_head
                    .forward(&hidden)
                    .and_then(|l| l.squeeze(1))
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "batch decode lm_head", e))?
            };
            logits = next_logits;
            tokens = argmax_rows(&logits)?;
            for (row, position) in positions.iter_mut().enumerate() {
                if !finished[row] {
                    *position += 1;
                }
            }
        }
        Ok(generated)
    }

    /// Embed the token ids and splice in the vision embeddings, returning the
    /// DeepStack taps alongside (empty when the checkpoint has none).
    fn prepare_inputs(
        &self,
        input_ids: &[u32],
        image_inputs: &WeVisDocImageInputs,
    ) -> Result<(Tensor, DeepstackVisualEmbeds), Error> {
        let seq_len = input_ids.len();
        let token_ids = Tensor::from_vec(input_ids.to_vec(), (1, seq_len), &self.device)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "create prompt token ids", e))?;
        let embeds = self.text.embed(&token_ids)?;
        let (image_embeds, deepstack_features) = self
            .vision
            .forward(&image_inputs.pixel_values, &[image_inputs.grid_thw])?;
        let image_embeds = image_embeds
            .to_dtype(self.dtype)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "cast image embeddings", e))?;

        let image_positions: Vec<usize> = input_ids
            .iter()
            .enumerate()
            .filter_map(|(index, &token)| (token == self.image_token_id).then_some(index))
            .collect();
        let image_len = image_embeds
            .dim(0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "image embedding length", e))?;
        if image_positions.is_empty() || image_positions.len() != image_len {
            return Err(Error::InvalidInput {
                message: format!(
                    "WeVisDoc: image placeholder count ({}) != image embedding count ({image_len})",
                    image_positions.len()
                ),
            });
        }
        let start = image_positions[0];
        if image_positions
            .iter()
            .enumerate()
            .any(|(offset, &position)| position != start + offset)
        {
            return Err(Error::InvalidInput {
                message: "WeVisDoc: image placeholder tokens must be contiguous".to_string(),
            });
        }
        let end = start + image_positions.len();
        let hidden_size = embeds
            .dim(2)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "embedding hidden size", e))?;
        let prefix = if start == 0 {
            Tensor::zeros((1, 0, hidden_size), embeds.dtype(), embeds.device())
        } else {
            embeds.narrow(1, 0, start)
        }
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "embedding prefix", e))?;
        let suffix = if end == seq_len {
            Tensor::zeros((1, 0, hidden_size), embeds.dtype(), embeds.device())
        } else {
            embeds.narrow(1, end, seq_len - end)
        }
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "embedding suffix", e))?;
        let image_embeds = image_embeds
            .unsqueeze(0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "image embedding batch", e))?;
        let inputs_embeds = Tensor::cat(&[&prefix, &image_embeds, &suffix], 1)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "merge multimodal embeddings", e))?;
        let deepstack = DeepstackVisualEmbeds {
            image_spans: vec![(start, image_len)],
            embeds: deepstack_features,
        };
        Ok((inputs_embeds, deepstack))
    }

    fn logits_from_hidden(&self, hidden: &Tensor) -> Result<Tensor, Error> {
        self.lm_head
            .forward(
                &hidden
                    .unsqueeze(0)
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "LM head input", e))?,
            )
            .and_then(|logits| logits.squeeze(0))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "language model head", e))
    }

    /// Decode generated token ids.
    pub fn decode_tokens(&self, tokens: &[u32]) -> Result<String, Error> {
        self.tokenizer
            .decode(tokens, true)
            .map(|text| text.trim().to_string())
            .map_err(|e| Error::InvalidInput {
                message: format!("WeVisDoc: tokenizer decode failed: {e}"),
            })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn config(&self) -> &WeVisDocConfig {
        &self.cfg
    }

    pub fn image_processor_config(
        &self,
    ) -> &crate::backbones::qwen_vl_processing::MinerUImageProcessorConfig {
        &self.image_cfg
    }
}

/// One page's prepared prompt for batch generation.
struct BatchPrompt {
    input_ids: Vec<u32>,
    inputs_embeds: Tensor,
    deepstack: DeepstackVisualEmbeds,
    position_ids: Tensor,
    rope_delta: i64,
}

/// Greedy token per row, computed on the device: one `batch`-wide readback
/// instead of transferring the full logits matrix to the host every step.
fn argmax_rows(logits: &Tensor) -> Result<Vec<u32>, Error> {
    #[cfg(feature = "cuda")]
    if logits.device().is_cuda() && matches!(logits.dtype(), DType::BF16 | DType::F16 | DType::F32)
    {
        // F16 logits convert to F32 on the device first: the argmax kernel
        // reads F32, and only the picked token ids ever cross back to host.
        let flat = logits
            .reshape((logits.dim(0)?, logits.dim(1)?))
            .and_then(|l| l.contiguous())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "reshape batch logits", e))?;
        let picked = match flat.dtype() {
            DType::BF16 => flat.apply_op1_no_bwd(&ArgmaxFirstBf16),
            DType::F16 => flat.to_dtype(DType::F32)?.apply_op1_no_bwd(&ArgmaxFirstF32),
            DType::F32 => flat.apply_op1_no_bwd(&ArgmaxFirstF32),
            _ => unreachable!("dtype checked above"),
        }
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "batch GPU argmax", e))?;
        return picked
            .to_vec1::<u32>()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "read batch tokens", e));
    }
    let scores = logits
        .to_dtype(DType::F32)
        .and_then(|l| l.to_vec2::<f32>())
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "read batch logits", e))?;
    scores
        .iter()
        .map(|row| {
            let mut best = 0usize;
            let mut best_value = f32::NEG_INFINITY;
            for (i, &v) in row.iter().enumerate() {
                if v > best_value {
                    best_value = v;
                    best = i;
                }
            }
            Ok(best as u32)
        })
        .collect()
}

fn require_token_id(
    tokenizer: &Tokenizer,
    token: &str,
    expected: Option<u32>,
) -> Result<u32, Error> {
    let token_id = tokenizer.token_to_id(token).ok_or_else(|| Error::Config {
        message: format!("WeVisDoc tokenizer is missing required token {token:?}"),
    })?;
    if let Some(expected) = expected
        && token_id != expected
    {
        return Err(Error::Config {
            message: format!(
                "WeVisDoc token {token:?} id mismatch: tokenizer {token_id} != config {expected}"
            ),
        });
    }
    Ok(token_id)
}

fn validate_generation_length(
    prompt_len: usize,
    max_new_tokens: usize,
    context_limit: usize,
) -> Result<(), Error> {
    let requested = prompt_len
        .checked_add(max_new_tokens)
        .ok_or_else(|| Error::InvalidInput {
            message: "WeVisDoc requested sequence length overflows usize".to_string(),
        })?;
    if requested > context_limit {
        return Err(Error::InvalidInput {
            message: format!(
                "WeVisDoc prompt ({prompt_len}) plus max_new_tokens ({max_new_tokens}) exceeds context limit {context_limit}"
            ),
        });
    }
    Ok(())
}

/// Render the official chat template for one image plus an instruction:
/// system turn, user turn with `<|vision_start|>` pads `<|vision_end|>` and
/// the instruction, then the assistant generation prompt.
pub fn build_prompt(num_image_tokens: usize, system_prompt: &str) -> String {
    let mut prompt =
        String::with_capacity(system_prompt.len() + num_image_tokens * "<|image_pad|>".len() + 128);
    if !system_prompt.is_empty() {
        prompt.push_str("<|im_start|>system\n");
        prompt.push_str(system_prompt);
        prompt.push_str("<|im_end|>\n");
    }
    prompt.push_str("<|im_start|>user\n<|vision_start|>");
    for _ in 0..num_image_tokens {
        prompt.push_str("<|image_pad|>");
    }
    prompt.push_str("<|vision_end|>");
    prompt.push_str(DEFAULT_PROMPT);
    prompt.push_str("<|im_end|>\n<|im_start|>assistant\n");
    prompt
}

/// Multimodal position ids for a single-image prompt, mirroring
/// `Qwen3VLModel.get_rope_index`: text keeps contiguous ids, the vision span
/// uses `(t, h, w)` indices offset by the running maximum, and the following
/// text continues from the span maximum.
fn build_position_ids(
    input_ids: &[u32],
    grid_thw: (usize, usize, usize),
    spatial_merge_size: usize,
    vision_start_token_id: u32,
    image_token_id: u32,
    device: &Device,
) -> Result<(Tensor, i64), Error> {
    // Locate the image span the way `get_rope_index` does: the tokens right
    // after each `<|vision_start|>` decide image vs video, and the span is the
    // contiguous run of image placeholder tokens.
    let image_start = input_ids
        .iter()
        .position(|&token| token == image_token_id)
        .ok_or_else(|| Error::InvalidInput {
            message: "WeVisDoc: image token missing from prompt".to_string(),
        })?;
    if image_start == 0 || input_ids[image_start - 1] != vision_start_token_id {
        return Err(Error::InvalidInput {
            message: "WeVisDoc: image tokens must follow <|vision_start|>".to_string(),
        });
    }
    let image_len = input_ids[image_start..]
        .iter()
        .take_while(|&&token| token == image_token_id)
        .count();
    if input_ids[image_start + image_len..].contains(&image_token_id) {
        return Err(Error::InvalidInput {
            message: "WeVisDoc: non-contiguous image token span".to_string(),
        });
    }

    let (grid_t, grid_h, grid_w) = grid_thw;
    if spatial_merge_size == 0
        || !grid_h.is_multiple_of(spatial_merge_size)
        || !grid_w.is_multiple_of(spatial_merge_size)
    {
        return Err(Error::Config {
            message: format!(
                "WeVisDoc: invalid image grid {grid_thw:?} for merge size {spatial_merge_size}"
            ),
        });
    }
    let llm_h = grid_h / spatial_merge_size;
    let llm_w = grid_w / spatial_merge_size;
    if image_len != grid_t * llm_h * llm_w {
        return Err(Error::InvalidInput {
            message: format!(
                "WeVisDoc: image token count {image_len} != merged grid token count {}",
                grid_t * llm_h * llm_w
            ),
        });
    }

    let seq_len = input_ids.len();
    let mut axes = [
        Vec::with_capacity(seq_len),
        Vec::with_capacity(seq_len),
        Vec::with_capacity(seq_len),
    ];
    for position in 0..image_start as i64 {
        for axis in &mut axes {
            axis.push(position);
        }
    }
    let vision_start = image_start as i64;
    for temporal in 0..grid_t {
        for row in 0..llm_h {
            for col in 0..llm_w {
                axes[0].push(vision_start + temporal as i64);
                axes[1].push(vision_start + row as i64);
                axes[2].push(vision_start + col as i64);
            }
        }
    }
    let text_start = vision_start + llm_h.max(llm_w) as i64;
    for (offset, _) in (image_start + image_len..seq_len).enumerate() {
        let current = text_start + offset as i64;
        for axis in &mut axes {
            axis.push(current);
        }
    }
    let max_position = axes
        .iter()
        .flat_map(|axis| axis.iter())
        .copied()
        .max()
        .unwrap_or(0);
    let rope_delta = max_position + 1 - seq_len as i64;
    let data: Vec<i64> = axes.into_iter().flatten().collect();
    let tensor = Tensor::from_vec(data, (3, 1, seq_len), device).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            format!("{MODEL_NAME}: create multimodal position ids"),
            e,
        )
    })?;
    Ok((tensor, rope_delta))
}

fn text_position_ids(position: i64, device: &Device) -> Result<Tensor, Error> {
    Tensor::from_vec(vec![position; 3], (3, 1, 1), device).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            format!("{MODEL_NAME}: create decode position ids"),
            e,
        )
    })
}

/// Longest token cycle at the tail of `tokens`, reported as
/// `(period, repeats)`, when the repetition is strong enough to be a decode
/// loop instead of real content. Two loop shapes are recognized:
///
/// * exact cycles — every repeated unit identical — which cover collapsed
///   single-token runs and repeated sentences; they need `EXACT_TAIL_UNITS
///   × period` tokens of looping tail, at least `MIN_REPEATS` repeats.
///   The tail budget keeps short-period markup (empty rows, dots) safe
///   while long sentence cycles still trip after a handful of repeats;
/// * near-cycles — units differing in at most two token slots, the same
///   slots every time — which cover counter loops like an incrementing
///   year. These need a period of at least `MIN_NEAR_PERIOD` (short
///   periods are where legitimate enumeration markup lives) and a longer
///   looping tail than the exact rule.
///
/// The repetition floors sit well above what legitimate content repeats:
/// empty table rows, dot leaders, and repeated headers stay in the single
/// digits of cycles, while the observed decode loops run tens of cycles.
fn trailing_decode_loop(tokens: &[u32]) -> Option<(usize, usize)> {
    const MAX_PERIOD: usize = 128;
    const MIN_TAIL: usize = 64;
    const MIN_REPEATS: usize = 4;
    const EXACT_TAIL_UNITS: usize = 96;
    const MIN_NEAR_PERIOD: usize = 8;
    const MIN_NEAR_REPEATS: usize = 8;
    const NEAR_TAIL_UNITS: usize = 128;
    const MAX_NEAR_DIFF: usize = 2;
    let len = tokens.len();
    if len < MIN_TAIL {
        return None;
    }
    // Longest period first: a loop that also matches a longer cycle keeps
    // one full cycle of the pattern instead of a fragment of it.
    for period in (1..=MAX_PERIOD.min(len / MIN_REPEATS)).rev() {
        let unit = &tokens[len - period..];
        // Slots where the last two units disagree — the counter positions.
        let near_slots: Vec<usize> = if period >= MIN_NEAR_PERIOD {
            let previous = &tokens[len - 2 * period..len - period];
            unit.iter()
                .zip(previous.iter())
                .enumerate()
                .filter_map(|(slot, (a, b))| (a != b).then_some(slot))
                .collect()
        } else {
            Vec::new()
        };
        if near_slots.len() > MAX_NEAR_DIFF {
            continue;
        }
        let mut repeats = 1;
        let near = !near_slots.is_empty();
        while (repeats + 1) * period <= len {
            let candidate = &tokens[len - (repeats + 1) * period..len - repeats * period];
            let matches = if near {
                // The unit must agree with the reference everywhere outside
                // the counter slots.
                candidate.len() == unit.len()
                    && candidate
                        .iter()
                        .zip(unit.iter())
                        .enumerate()
                        .all(|(slot, (a, b))| a == b || near_slots.contains(&slot))
            } else {
                candidate == unit
            };
            if !matches {
                break;
            }
            repeats += 1;
        }
        let needed = if near {
            NEAR_TAIL_UNITS.div_ceil(period).max(MIN_NEAR_REPEATS)
        } else {
            EXACT_TAIL_UNITS.div_ceil(period).max(MIN_REPEATS)
        };
        if repeats >= needed {
            return Some((period, repeats));
        }
    }
    None
}

/// How aggressively region decoding guards against degenerate loops.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum LoopGuard {
    /// Page decoding: no guard, byte-reproduces the reference decoding.
    Off,
    /// Text-like regions: stop and trim as soon as the tail cycles.
    Standard,
    /// Structured regions (tables, formulas): legitimate output repeats
    /// rows and patterns, so only intervene when the token budget is
    /// nearly exhausted — stop, but keep everything generated so far.
    Conservative,
}

/// How far the generation may run in [`LoopGuard::Conservative`] before a
/// detected loop is acted on.
const CONSERVATIVE_MARGIN: usize = 256;

/// The action the guard takes for `generated` under `mode`, if any.
/// `budget` is the region's `max_new_tokens`.
fn loop_guard_action(mode: LoopGuard, generated: &[u32], budget: usize) -> Option<LoopAction> {
    match mode {
        LoopGuard::Off => None,
        LoopGuard::Conservative => {
            let near_budget = budget.saturating_sub(generated.len()) <= CONSERVATIVE_MARGIN;
            near_budget
                .then(|| trailing_decode_loop(generated))
                .flatten()
                .map(|_| LoopAction::Stop)
        }
        LoopGuard::Standard => trailing_decode_loop(generated)
            .map(|(period, repeats)| LoopAction::TrimStop { period, repeats }),
    }
}

/// What to do with a detected loop.
#[derive(Debug, PartialEq, Eq)]
enum LoopAction {
    /// Drop the repeated cycles (one stays) and stop the sequence.
    TrimStop { period: usize, repeats: usize },
    /// Stop without touching the generated tokens.
    Stop,
}

fn select_greedy_token(logits: &Tensor) -> Result<u32, Error> {
    #[cfg(feature = "cuda")]
    if logits.device().is_cuda() && matches!(logits.dtype(), DType::BF16 | DType::F16 | DType::F32)
    {
        let vocab_size = logits.elem_count();
        let logits = logits
            .reshape((1, vocab_size))
            .and_then(|l| l.contiguous())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "reshape GPU logits", e))?;
        let tokens = match logits.dtype() {
            DType::BF16 => logits.apply_op1_no_bwd(&ArgmaxFirstBf16),
            DType::F16 => logits
                .to_dtype(DType::F32)?
                .apply_op1_no_bwd(&ArgmaxFirstF32),
            DType::F32 => logits.apply_op1_no_bwd(&ArgmaxFirstF32),
            _ => unreachable!("dtype checked above"),
        }
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "stable GPU argmax", e))?;
        return tokens
            .i(0)
            .and_then(|token| token.to_scalar::<u32>())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "copy selected token", e));
    }

    logits
        .argmax(candle_core::D::Minus1)
        .and_then(|token| token.to_scalar::<u32>())
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                format!("{MODEL_NAME}: greedy argmax"),
                e,
            )
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn official_prompt_matches_the_chat_template_rendering() {
        let prompt = build_prompt(2, DEFAULT_SYSTEM_PROMPT);
        assert!(prompt.starts_with("<|im_start|>system\nYou are an AI assistant specialized"));
        assert!(prompt.contains("explanations or comments.\n<|im_end|>\n<|im_start|>user\n"));
        assert!(prompt.contains(
            "<|vision_start|><|image_pad|><|image_pad|><|vision_end|>Convert this document image to Markdown.<|im_end|>\n<|im_start|>assistant\n"
        ));
        assert!(prompt.ends_with("<|im_start|>assistant\n"));
    }

    #[test]
    fn empty_system_prompt_omits_the_system_turn() {
        let prompt = build_prompt(1, "");
        assert!(prompt.starts_with("<|im_start|>user\n<|vision_start|>"));
    }

    #[test]
    fn golden_position_ids_match_the_python_reference() {
        // Golden values dumped from transformers 4.57.1 `get_rope_index`:
        // grid (1, 32, 122) -> llm grid 16x61, prompt len 1273, delta -915.
        let mut ids = vec![1u32; 1273];
        // 283 text tokens, <|vision_start|>, 976 image pads, <|vision_end|>.
        ids[283] = 151_652;
        for token in &mut ids[284..284 + 976] {
            *token = 151_655;
        }
        ids[284 + 976] = 151_653;
        let (positions, delta) =
            build_position_ids(&ids, (1, 32, 122), 2, 151_652, 151_655, &Device::Cpu).unwrap();
        assert_eq!(delta, -915);
        let positions = positions.to_vec3::<i64>().unwrap();
        // Text before the span (including <|vision_start|>): contiguous.
        assert_eq!(positions[0][0][0], 0);
        assert_eq!(positions[2][0][283], 283);
        // Vision span base equals the first image pad's sequence index.
        assert_eq!(positions[0][0][284], 284);
        assert_eq!(positions[1][0][284], 284);
        assert_eq!(positions[2][0][284], 284);
        assert_eq!(positions[0][0][700], 284);
        assert_eq!(positions[1][0][700], 290);
        assert_eq!(positions[2][0][700], 334);
        // Tail text continues from the span maximum (284 + 61 - 1 = 344).
        assert_eq!(positions[0][0][1260], 345);
        assert_eq!(positions[0][0][1261], 346);
        assert_eq!(positions[1][0][1265], 350);
        assert_eq!(positions[2][0][1272], 357);
    }

    #[test]
    fn position_ids_reject_misplaced_image_tokens() {
        let ids = vec![151_655u32; 4];
        assert!(build_position_ids(&ids, (1, 4, 2), 2, 151_652, 151_655, &Device::Cpu).is_err());
    }

    #[test]
    fn short_or_acyclic_tails_are_not_loops() {
        assert_eq!(trailing_decode_loop(&[]), None);
        assert_eq!(trailing_decode_loop(&[1, 2, 3, 1, 2, 3]), None);

        // Two repetitions of a sentence-length cycle: normal prose.
        let mut prose = vec![10u32; 20];
        prose.extend_from_within(..20);
        assert_eq!(trailing_decode_loop(&prose), None);

        // Long acyclic tail (LCG noise never closes an exact cycle).
        let mut state = 12345u64;
        let noise: Vec<u32> = (0..600)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as u32
            })
            .collect();
        assert_eq!(trailing_decode_loop(&noise), None);
    }

    #[test]
    fn non_flash_devices_take_the_sequential_batch_path() {
        // CPU never has flash; the batch entry must dispatch those devices
        // to the per-page sequential path (see generate_tokens_impl).
        assert!(!row_flash_attention_available(&Device::Cpu, DType::BF16));
        assert!(!row_flash_attention_available(&Device::Cpu, DType::F32));
    }

    #[test]
    fn loop_guard_respects_task_modes() {
        let budget = 4096usize;
        // A table region: ten identical long rows (>= 30 tokens each) plus
        // the closing tag — legitimate structure, not a loop.
        let row: Vec<u32> = (300..340).collect();
        let mut table = Vec::new();
        for _ in 0..10 {
            table.extend_from_slice(&row);
        }
        table.extend_from_slice(&[500, 501]);
        assert_eq!(
            loop_guard_action(LoopGuard::Conservative, &table, budget),
            None,
            "table rows far from the budget must not be touched"
        );

        // A genuine dead loop only acts once the budget is nearly spent,
        // and even then keeps everything generated so far.
        let mut dead = Vec::new();
        for _ in 0..120 {
            dead.extend_from_slice(&[7u32, 8, 9]);
        }
        assert_eq!(
            loop_guard_action(LoopGuard::Conservative, &dead, budget),
            None
        );
        match loop_guard_action(LoopGuard::Conservative, &dead, dead.len() + 128) {
            Some(LoopAction::Stop) => {}
            other => panic!("expected Stop near the budget, got {other:?}"),
        }

        // Text-like regions keep today's behavior: trim cycles and stop.
        match loop_guard_action(LoopGuard::Standard, &dead, budget) {
            Some(LoopAction::TrimStop { .. }) => {}
            other => panic!("expected TrimStop under Standard, got {other:?}"),
        }

        // Page decoding never guards.
        assert_eq!(loop_guard_action(LoopGuard::Off, &dead, budget), None);
    }

    #[test]
    fn cuda_f16_argmax_matches_f32_batch_and_single() {
        #[cfg(feature = "cuda")]
        {
            let Ok(device) = Device::new_cuda(0) else {
                eprintln!("skipping: no CUDA device");
                return;
            };
            let vocab = 4096usize;
            // Values are f16-exact so both tensors hold the same numbers;
            // the last row ends in an exact tie resolved to first index.
            let rows: Vec<Vec<f32>> = vec![
                {
                    let mut r = vec![0.25f32; vocab];
                    r[17] = 3.5;
                    r
                },
                {
                    let mut r = vec![-1.0f32; vocab];
                    r[7] = 1.5;
                    r[4000] = 2.25;
                    r
                },
                {
                    let mut r = vec![0.75f32; vocab];
                    r[9] = 4.0;
                    r[13] = 4.0;
                    r
                },
            ];
            let flat: Vec<f32> = rows.concat();
            let t32 = Tensor::from_vec(flat, (rows.len(), vocab), &device).unwrap();
            let t16 = t32.to_dtype(DType::F16).unwrap();
            assert_eq!(t16.dtype(), DType::F16);

            let picked32 = argmax_rows(&t32).unwrap();
            let picked16 = argmax_rows(&t16).unwrap();
            assert_eq!(picked16, picked32, "batch argmax must agree across dtypes");
            assert_eq!(picked16, vec![17u32, 4000, 9]);

            let single32 = t32.i(2).unwrap();
            let single16 = t16.i(2).unwrap();
            assert_eq!(
                select_greedy_token(&single16).unwrap(),
                select_greedy_token(&single32).unwrap(),
                "single-row argmax must agree across dtypes"
            );
            assert_eq!(select_greedy_token(&single16).unwrap(), 9);
        }
        #[cfg(not(feature = "cuda"))]
        eprintln!("skipping: built without the cuda feature");
    }

    #[test]
    fn legitimate_repeated_structures_do_not_trip_the_guard() {
        // Empty table rows (a spacer table in a region crop): a handful of
        // identical rows is normal content, not a decode loop.
        let row = [11u32, 12, 13, 14, 15, 16];
        let mut table = vec![10u32];
        for _ in 0..6 {
            table.extend_from_slice(&row);
        }
        table.extend_from_slice(&[17, 18]);
        assert_eq!(trailing_decode_loop(&table), None);

        // Dot leaders: 「标题……40」 — the run closes with a page number.
        let mut toc = vec![20u32, 21, 22];
        for _ in 0..40 {
            toc.extend_from_slice(&[23, 24]);
        }
        toc.extend_from_slice(&[25, 26]);
        assert_eq!(trailing_decode_loop(&toc), None);

        // A bare single-token dot run just under the exact floor.
        let dots = vec![7u32; 90];
        assert_eq!(trailing_decode_loop(&dots), None);
    }

    #[test]
    fn degenerate_cycles_are_detected_with_the_longest_period() {
        // Single-token loop (the 3993x "。" failure mode). A constant run
        // is periodic under every divisor, so the reported period varies —
        // what matters is that it trips and the trim keeps one cycle.
        let run = vec![5u32; 200];
        let (period, repeats) = trailing_decode_loop(&run).expect("constant run is a loop");
        let keep = run.len() - (repeats - 1) * period;
        assert!(keep >= period && keep < run.len());
        assert!(run[..keep].iter().all(|&t| t == 5));

        // A 20-token sentence repeated 12 times. The reported period is
        // the longest qualifying cycle (a multiple of the sentence), and
        // the trim keeps whole cycles of that period.
        let sentence: Vec<u32> = (100..120).collect();
        let mut looped = Vec::new();
        for _ in 0..12 {
            looped.extend_from_slice(&sentence);
        }
        let (period, repeats) = trailing_decode_loop(&looped).expect("sentence cycle detected");
        assert_eq!(period % 20, 0);
        let trimmed = looped.len() - (repeats - 1) * period;
        assert_eq!(trimmed % period, 0);
        assert!(looped[..trimmed].ends_with(&sentence));

        // A long sentence-level exact cycle (the newspaper-page runaway:
        // ~68 tokens per cycle) repeated 9 times.
        let long_sentence: Vec<u32> = (200..268).collect();
        let mut long_loop = Vec::new();
        for _ in 0..9 {
            long_loop.extend_from_slice(&long_sentence);
        }
        let (period, repeats) =
            trailing_decode_loop(&long_loop).expect("long sentence cycle detected");
        assert!(period >= 68);
        let keep = long_loop.len() - (repeats - 1) * period;
        assert_eq!(&long_loop[..keep], &long_sentence);

        // Widely varying units are not a near-cycle: real prose survives.
        let mut state = 987_654_321u64;
        let prose: Vec<u32> = (0..300)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as u32
            })
            .collect();
        assert_eq!(trailing_decode_loop(&prose), None);
    }

    #[test]
    fn counter_loops_with_one_moving_slot_are_loops() {
        // The newspaper-page failure mode: an otherwise fixed sentence
        // whose year token increments every cycle.
        let mut counter = Vec::new();
        for year in 2015..2055u32 {
            counter.extend_from_slice(&[10, 11, 12, 13, 14, 15, 16, year, 18, 19, 20]);
        }
        let (period, repeats) = trailing_decode_loop(&counter).expect("counter loop detected");
        let keep = counter.len() - (repeats - 1) * period;
        assert!(keep >= period && keep < counter.len());
        // The kept prefix is exactly whole units of the reported cycle.
        assert_eq!(keep % period, 0);

        // Two moving slots still qualify.
        let mut two = Vec::new();
        for year in 2015..2065u32 {
            two.extend_from_slice(&[1, 2, 3, year, 5, 6, 7, 8, year + 1, 10, 11, 12, 13]);
        }
        assert!(trailing_decode_loop(&two).is_some());

        // Widely varying units are not a near-cycle: real prose survives.
        let mut state = 987_654_321u64;
        let prose: Vec<u32> = (0..300)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                (state >> 33) as u32
            })
            .collect();
        assert_eq!(trailing_decode_loop(&prose), None);
    }

    #[test]
    fn generation_length_is_checked_without_overflow() {
        validate_generation_length(775, 8_192, 262_144).unwrap();
        assert!(validate_generation_length(775, 262_000, 262_144).is_err());
        assert!(validate_generation_length(1, usize::MAX, usize::MAX).is_err());
    }

    #[test]
    fn greedy_argmax_prefers_the_first_tied_token() {
        let logits = Tensor::from_vec(vec![1f32, 3., 3., 2.], 4, &Device::Cpu).unwrap();
        assert_eq!(select_greedy_token(&logits).unwrap(), 1);
    }
}
