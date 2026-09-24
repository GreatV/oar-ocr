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
    combine_masks, create_causal_mask, create_generation_mask_if_needed, create_left_padding_mask,
    decode_position_buffer,
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
        // WeVisDoc ties the language-model output projection to token
        // embeddings (`tie_word_embeddings`, checked against the config).
        let lm_head = Linear::new(text.token_embedding_weight(), None);

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
        if images.len() <= 1 {
            return Ok(images
                .iter()
                .map(|image| {
                    self.generate_one(image, max_new_tokens)
                        .map(|(tokens, _)| tokens)
                })
                .collect());
        }
        self.generate_batch_tokens(images, max_new_tokens)
            .map(|results| results.into_iter().map(Ok).collect())
    }

    /// Generate one page's tokens plus whether decoding stopped on an EOS
    /// token. `false` means the token budget ran out first — the official
    /// `wevisdoc/local.py` treats that as a truncation error.
    pub(crate) fn generate_one(
        &self,
        image: &RgbImage,
        max_new_tokens: usize,
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
        self.text
            .prepare_ar_cuda_graph(input_ids.len(), max_new_tokens, &self.lm_head)?;
        let hidden = self
            .text
            .forward(&inputs_embeds, &position_ids, Some(&deepstack), None)?;
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
        let mask = if batch_size > 1 {
            let causal = create_causal_mask(max_seq_len, max_seq_len, self.dtype, &self.device)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "create causal mask", e))?;
            let padding =
                create_left_padding_mask(&seq_lens, max_seq_len, self.dtype, &self.device)
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "create padding mask", e))?;
            Some(
                combine_masks(&causal, &padding)
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "combine masks", e))?,
            )
        } else {
            None
        };

        self.text.clear_cache();
        // Batch prefill replaces the batch-1 KV backing storage; drop any
        // captured graph before those raw pointers become stale.
        self.text.invalidate_ar_cuda_graph();
        let hidden = self.text.forward(
            &inputs_embeds,
            &position_ids,
            Some(&deepstack),
            mask.as_ref(),
        )?;
        let last_hidden = hidden
            .i((.., max_seq_len - 1, ..))
            .and_then(|h| h.contiguous())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "select last hidden", e))?;
        let mut logits_rows = self
            .lm_head
            .forward(&last_hidden)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "batch lm_head", e))?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "read batch logits", e))?;

        let mut generated: Vec<Vec<u32>> = vec![Vec::new(); batch_size];
        let mut finished = vec![false; batch_size];
        let mut positions: Vec<i64> = seq_lens
            .iter()
            .zip(rows.iter().map(|row| row.rope_delta))
            .map(|(&len, delta)| len as i64 + delta)
            .collect();
        let pad_lens: Vec<usize> = seq_lens.iter().map(|&len| max_seq_len - len).collect();
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
                let token = argmax_token(&logits_rows[row])?;
                if self.stop_token_ids.contains(&token) {
                    finished[row] = true;
                } else {
                    generated[row].push(token);
                }
                next_tokens.push(token);
            }
            if finished.iter().all(|&f| f) {
                break;
            }
            if step + 1 == max_new_tokens {
                break;
            }

            let tokens = Tensor::from_vec(next_tokens, (batch_size, 1), &self.device)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "create decode tokens", e))?;
            let embeds = self.text.embed(&tokens)?;
            let pos_data = decode_position_buffer(&positions, 3);
            let pos = Tensor::from_vec(pos_data, (3, batch_size, 1), &self.device)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "create decode positions", e))?;
            kv_len += 1;
            let gen_mask =
                create_generation_mask_if_needed(&pad_lens, kv_len, self.dtype, &self.device)
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "create decode mask", e))?;
            let hidden = self.text.forward(&embeds, &pos, None, gen_mask.as_ref())?;
            logits_rows = self
                .lm_head
                .forward(&hidden)
                .and_then(|l| l.squeeze(1))
                .and_then(|l| l.to_dtype(DType::F32))
                .and_then(|l| l.to_vec2::<f32>())
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "batch decode lm_head", e))?;
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

/// Greedy argmax over a host score row (first index wins ties).
fn argmax_token(scores: &[f32]) -> Result<u32, Error> {
    let mut best = 0usize;
    let mut best_value = f32::NEG_INFINITY;
    for (index, &value) in scores.iter().enumerate() {
        if value > best_value {
            best_value = value;
            best = index;
        }
    }
    Ok(best as u32)
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

fn select_greedy_token(logits: &Tensor) -> Result<u32, Error> {
    #[cfg(feature = "cuda")]
    if logits.device().is_cuda() && matches!(logits.dtype(), DType::BF16 | DType::F32) {
        let vocab_size = logits.elem_count();
        let logits = logits
            .reshape((1, vocab_size))
            .and_then(|logits| logits.contiguous())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "reshape GPU logits", e))?;
        let tokens = match logits.dtype() {
            DType::BF16 => logits.apply_op1_no_bwd(&ArgmaxFirstBf16),
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
