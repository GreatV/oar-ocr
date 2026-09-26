//! jina-ocr-v1 (DeepSeek-OCR) model implementation.
//!
//! Native Rust inference for the Jina jina-ocr-v1 checkpoint (SAM ViT-B +
//! CLIP-L encoder over a 12-layer DeepSeek-V2 MoE decoder). Generation follows
//! the official `example.py` transformers recipe: greedy decoding with a
//! sliding-window no-repeat-ngram guard (35-grams within a 1024-token window,
//! `<td>`/`</td>` whitelisted). On CUDA the decode step runs inside a CUDA
//! graph, and the trained FastMTP draft head (`mtp_module`) proposes
//! three-token blocks whose greedy verification preserves the plain-greedy
//! sequence; the n-gram ban is applied to the verification logits host-side
//! so speculation cannot change the official recipe.

use super::config::JinaOcrConfig;
use super::mtp::JinaOcrMtp;
use super::processing::{
    DEFAULT_OCR_PROMPT, JinaOcrImageInputs, JinaOcrProcessorConfig, image_tokens, preprocess_image,
};
use crate::backbones::deep_encoder::DeepEncoder;
use crate::backbones::deepseek_v2::DeepSeekV2TextModel;
use crate::error::Error;
use crate::runtime::attention::{
    combine_masks, create_causal_mask, create_generation_mask_if_needed, create_left_padding_mask,
};
use crate::runtime::checkpoint::collect_safetensors;
use crate::runtime::errors::{candle_to_ocr_inference, candle_to_ocr_processing};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};
use image::RgbImage;
use std::path::Path;
use tokenizers::Tokenizer;

const MODEL_NAME: &str = "JinaOCR";

/// Official `--max-new-tokens` default in `example.py`.
pub const DEFAULT_MAX_NEW_TOKENS: usize = 4_096;

/// `SlidingWindowNoRepeatNgramProcessor` defaults from the reference
/// `generate()` override.
const NGRAM_SIZE: usize = 35;
const NGRAM_WINDOW: usize = 1024;
/// `<td>` / `</td>` are exempt from the n-gram ban.
const NGRAM_WHITELIST: [u32; 2] = [128_821, 128_822];

/// `mtp_num_speculative_steps`: FastMTP draft tokens per verification block.
const MTP_DRAFT_TOKENS: usize = 3;
const MTP_QUERY_LEN: usize = MTP_DRAFT_TOKENS + 1;
/// Below this budget speculation costs more than plain decoding: the
/// per-page setup (initial draft sync plus graph captures, ~80ms on an RTX
/// 4090) dominates short generations.
const MTP_MIN_NEW_TOKENS: usize = 384;

struct TextCacheGuard<'a>(&'a DeepSeekV2TextModel);

impl Drop for TextCacheGuard<'_> {
    fn drop(&mut self) {
        self.0.clear_kv_cache();
    }
}

/// Greedy-generation result with per-step numerical evidence (see
/// [`JinaOcr::generate_traced`]).
#[derive(Debug)]
pub struct GenerationTrace {
    /// Generated token ids, without the closing EOS.
    pub tokens: Vec<u32>,
    /// Whether decoding stopped on EOS (`false` = the budget ran out).
    pub hit_eos: bool,
    /// Raw top-3 `(token, logit)` pairs (pre-ngram-ban) per decoding step,
    /// including the step that produced the EOS. Only filled by
    /// [`JinaOcr::generate_traced`].
    pub step_top: Vec<[(u32, f32); 3]>,
}

/// End-to-end jina-ocr-v1 page parser.
pub struct JinaOcr {
    device: Device,
    dtype: DType,
    cfg: JinaOcrConfig,
    processor_cfg: JinaOcrProcessorConfig,
    tokenizer: Tokenizer,
    text: DeepSeekV2TextModel,
    vision: DeepEncoder,
    mtp: Option<JinaOcrMtp>,
    lm_head: Linear,
    image_newline: Tensor,
    view_separator: Tensor,
    image_token_id: u32,
    eos_token_ids: Vec<u32>,
}

impl JinaOcr {
    /// Load a jina-ocr-v1 Hugging Face model directory.
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, Error> {
        Self::from_dir_with_runtime(model_dir, crate::RuntimeConfig::new(device))
    }

    pub fn from_dir_with_runtime(
        model_dir: impl AsRef<Path>,
        runtime: crate::RuntimeConfig,
    ) -> Result<Self, Error> {
        let (device, dtype) = runtime.resolve();
        let model_dir = model_dir.as_ref();
        let cfg = JinaOcrConfig::from_path(model_dir.join("config.json"))?;
        let processor_cfg =
            JinaOcrProcessorConfig::from_path(model_dir.join("processor_config.json"))?;
        let tokenizer =
            Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| Error::Config {
                message: format!("failed to load JinaOCR tokenizer.json: {e}"),
            })?;
        let image_token_id = require_token_id(&tokenizer, "<image>", Some(cfg.image_token_index))?;
        let eos_string = "<｜end▁of▁sentence｜>";
        let tokenizer_eos = require_token_id(&tokenizer, eos_string, Some(cfg.text.eos_token_id))?;
        let eos_token_ids = vec![cfg.text.eos_token_id, tokenizer_eos];

        let weight_files = collect_safetensors(model_dir, MODEL_NAME)?;
        // SAFETY: The model files must remain unchanged while their mmap is in use.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_files, dtype, &device)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load safetensors", e))?
        };
        let (sam_cfg, clip_cfg) = cfg.vision_configs()?;
        let vision = DeepEncoder::load(&sam_cfg, &clip_cfg, cfg.text.hidden_size, vb.pp("model"))?;
        let text = DeepSeekV2TextModel::load(&cfg.text, vb.pp("model"))?;
        let lm_head = linear_no_bias(cfg.text.hidden_size, cfg.text.vocab_size, vb.pp("lm_head"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load lm_head", e))?;
        let image_newline = vb
            .pp("model")
            .get(cfg.text.hidden_size, "image_newline")
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load image_newline", e))?;
        let view_separator = vb
            .pp("model")
            .get(cfg.text.hidden_size, "view_seperator")
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load view_seperator", e))?;
        let mtp = if cfg.num_nextn_predict_layers.unwrap_or(0) >= 1
            && std::env::var_os("OAR_JINAOCR_DISABLE_MTP").is_none()
        {
            Some(JinaOcrMtp::load(
                &cfg.text,
                text.token_embedding_weight(),
                text.final_norm_weight(),
                lm_head.weight().clone(),
                vb,
            )?)
        } else {
            None
        };

        Ok(Self {
            device,
            dtype,
            cfg,
            processor_cfg,
            tokenizer,
            text,
            vision,
            mtp,
            lm_head,
            image_newline,
            view_separator,
            image_token_id,
            eos_token_ids,
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
    /// the MTP/graph fast path; larger batches run a padded batch prefill and
    /// decode (speculation and decode graphs are single-sequence only).
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
        let results = self.generate_batch_tokens(images, max_new_tokens)?;
        Ok(results.into_iter().map(Ok).collect())
    }

    /// Generate one page's tokens plus whether decoding stopped on EOS
    /// (`false` = the budget ran out first, the official n-gram-bounded
    /// truncation signal).
    pub(crate) fn generate_one(
        &self,
        image: &RgbImage,
        max_new_tokens: usize,
    ) -> Result<(Vec<u32>, bool), Error> {
        let prompt = self.prepare_prompt(image, max_new_tokens)?;
        self.text.clear_kv_cache();
        let _cache_guard = TextCacheGuard(&self.text);
        if self.mtp_enabled(max_new_tokens) {
            let hidden = self
                .text
                .forward(&prompt.inputs_embeds, &prompt.position_ids, None)?;
            let (tokens, hit_eos) = self.engine().mtp_tokens(
                &prompt.input_ids,
                &hidden,
                max_new_tokens,
                &AdaptiveSpec::default(),
            )?;
            Ok((tokens, hit_eos))
        } else {
            let hidden = self
                .text
                .forward(&prompt.inputs_embeds, &prompt.position_ids, None)?;
            let trace =
                self.engine()
                    .ar_tokens(&prompt.input_ids, &hidden, max_new_tokens, false)?;
            Ok((trace.tokens, trace.hit_eos))
        }
    }

    /// Same greedy generation with the raw (pre-ngram-ban) top-3 `(token,
    /// logit)` pairs of every decoding step attached — the
    /// numerical-alignment evidence compared against the transformers
    /// reference by the environment-gated alignment test. Always plain
    /// autoregressive decoding, never speculative.
    #[doc(hidden)]
    pub fn generate_traced(
        &self,
        image: &RgbImage,
        max_new_tokens: usize,
    ) -> Result<GenerationTrace, Error> {
        let prompt = self.prepare_prompt(image, max_new_tokens)?;
        self.text.clear_kv_cache();
        let _cache_guard = TextCacheGuard(&self.text);
        self.text
            .prepare_ar_cuda_graph(prompt.input_ids.len(), max_new_tokens, &self.lm_head)?;
        let hidden = self
            .text
            .forward(&prompt.inputs_embeds, &prompt.position_ids, None)?;
        self.engine()
            .ar_tokens(&prompt.input_ids, &hidden, max_new_tokens, true)
    }

    /// Padded batch generation: the same tokens as per-image generation, with
    /// one prefill and one decode step for the whole batch. Unequal prompt
    /// lengths are left-padded; prefill and decode masks hide the padded KV
    /// positions, and each sequence stops at its own EOS.
    fn generate_batch_tokens(
        &self,
        images: &[RgbImage],
        max_new_tokens: usize,
    ) -> Result<Vec<Vec<u32>>, Error> {
        let batch_size = images.len();
        let mut prompts: Vec<PreparedPrompt> = Vec::with_capacity(batch_size);
        for image in images {
            prompts.push(self.prepare_prompt(image, max_new_tokens)?);
        }
        let seq_lens: Vec<usize> = prompts.iter().map(|p| p.input_ids.len()).collect();
        let Some(&max_seq_len) = seq_lens.iter().max() else {
            return Err(Error::InvalidInput {
                message: "JinaOCR: empty batch is not supported".to_string(),
            });
        };

        // Left-pad every sequence to the batch maximum.
        let mut embeds_rows = Vec::with_capacity(batch_size);
        let mut position_rows = Vec::with_capacity(batch_size);
        for (prompt, &seq_len) in prompts.iter().zip(&seq_lens) {
            let pad_len = max_seq_len - seq_len;
            let embeds = if pad_len > 0 {
                let pad = Tensor::zeros(
                    (1, pad_len, self.cfg.text.hidden_size),
                    prompt.inputs_embeds.dtype(),
                    &self.device,
                )
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "create pad", e))?;
                Tensor::cat(&[&pad, &prompt.inputs_embeds], 1)
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "cat pad", e))?
            } else {
                prompt.inputs_embeds.clone()
            };
            embeds_rows.push(embeds);
            let mut positions =
                Tensor::arange(0u32, seq_len as u32, &self.device)?.reshape((1, 1, seq_len))?;
            if pad_len > 0 {
                let pad = Tensor::zeros((1, 1, pad_len), DType::U32, &self.device)?;
                positions = Tensor::cat(&[&pad, &positions], 2)?;
            }
            position_rows.push(positions);
        }
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

        self.text.clear_kv_cache();
        // Batch prefill replaces the batch-1 KV backing storage; drop any
        // captured graph before those raw pointers become stale.
        self.text.invalidate_ar_cuda_graph();
        let hidden = self
            .text
            .forward(&inputs_embeds, &position_ids, mask.as_ref())?;
        let last_hidden = hidden
            .i((.., max_seq_len - 1, ..))
            .and_then(|h| h.contiguous())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "get last hidden", e))?;
        let mut logits_rows = self
            .lm_head
            .forward(&last_hidden)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "batch lm_head", e))?
            .to_dtype(DType::F32)?
            .to_vec2::<f32>()
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "read batch logits", e))?;

        let mut generated: Vec<Vec<u32>> = vec![Vec::new(); batch_size];
        let mut finished = vec![false; batch_size];
        let mut positions: Vec<u32> = seq_lens.iter().map(|&len| len as u32).collect();
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
                let prompt_ids = &prompts[row].input_ids;
                let mut scores = std::mem::take(&mut logits_rows[row]);
                let mut history = Vec::with_capacity(prompt_ids.len() + generated[row].len());
                history.extend_from_slice(prompt_ids);
                history.extend_from_slice(&generated[row]);
                apply_no_repeat_ngram(&history, &mut scores);
                let token = argmax(&scores)?;
                if self.eos_token_ids.contains(&token) {
                    finished[row] = true;
                } else {
                    generated[row].push(token);
                }
                next_tokens.push(token);
                logits_rows[row] = scores;
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
            let pos = Tensor::from_vec(positions.clone(), (1, batch_size, 1), &self.device)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "create decode positions", e))?;
            kv_len += 1;
            let gen_mask =
                create_generation_mask_if_needed(&pad_lens, kv_len, self.dtype, &self.device)
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "create decode mask", e))?;
            let hidden = self.text.forward(&embeds, &pos, gen_mask.as_ref())?;
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

    fn engine(&self) -> GreedyEngine<'_> {
        GreedyEngine {
            model_name: MODEL_NAME,
            text: &self.text,
            lm_head: &self.lm_head,
            mtp: self.mtp.as_ref(),
            eos_token_ids: &self.eos_token_ids,
            device: &self.device,
        }
    }

    /// FastMTP speculation is on by default where the draft head is loaded
    /// (CUDA): greedy verification keeps the output token-identical to plain
    /// autoregressive decoding. Disable with `OAR_JINAOCR_DISABLE_MTP` (skip
    /// loading the head) or `OAR_VL_DISABLE_SPECULATIVE`.
    fn mtp_enabled(&self, max_new_tokens: usize) -> bool {
        self.mtp.is_some()
            && self.device.is_cuda()
            && max_new_tokens >= MTP_MIN_NEW_TOKENS
            && std::env::var_os("OAR_VL_DISABLE_SPECULATIVE").is_none()
    }

    /// Capture the FastMTP draft graph over a fixed-capacity KV bucket. The
    /// dense draft block stays fully on-device, so it is the one piece of the
    /// speculative loop that graph capture can accelerate (the target's MoE
    /// layers route through the host and must stay eager).
    /// Tokenize the fixed OCR prompt for one page and splice the visual
    /// features into the image placeholder positions.
    fn prepare_prompt(
        &self,
        image: &RgbImage,
        max_new_tokens: usize,
    ) -> Result<PreparedPrompt, Error> {
        let image_inputs = preprocess_image(image, &self.processor_cfg, &self.device, self.dtype)?;
        // JINA_OCR_CHAT_TEMPLATE for one user turn: "<|User|>:\n" + image +
        // "\n" + instruction, then "\n<|Assistant|>:\n" as the generation
        // prompt. No BOS: the processor encodes without special tokens.
        let prompt = format!("<|User|>:\n<image>\n{DEFAULT_OCR_PROMPT}\n<|Assistant|>:\n");
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| Error::InvalidInput {
                message: format!("JinaOCR: tokenizer encode failed: {e}"),
            })?;
        let encoded = encoding.get_ids().to_vec();
        let image_count = encoded
            .iter()
            .filter(|&&t| t == self.image_token_id)
            .count();
        if image_count != 1 {
            return Err(Error::InvalidInput {
                message: format!(
                    "JinaOCR: prompt must contain exactly one <image> placeholder, found {image_count}"
                ),
            });
        }
        let tokens_layout = image_tokens(self.image_token_id, &image_inputs);
        let mut input_ids = Vec::with_capacity(encoded.len() - 1 + tokens_layout.len());
        for &token in &encoded {
            if token == self.image_token_id {
                input_ids.extend_from_slice(&tokens_layout);
            } else {
                input_ids.push(token);
            }
        }
        if input_ids.len() + max_new_tokens > self.cfg.text.max_position_embeddings {
            return Err(Error::InvalidInput {
                message: format!(
                    "JinaOCR prompt ({}) plus max_new_tokens ({max_new_tokens}) exceeds context limit {}",
                    input_ids.len(),
                    self.cfg.text.max_position_embeddings
                ),
            });
        }
        let inputs_embeds = self.prepare_inputs(&input_ids, &image_inputs)?;
        let seq_len = input_ids.len();
        let position_ids =
            Tensor::arange(0u32, seq_len as u32, &self.device)?.reshape((1, 1, seq_len))?;
        Ok(PreparedPrompt {
            input_ids,
            inputs_embeds,
            position_ids,
        })
    }

    /// Embed the token ids and splice `[local tiles | global | separator]`
    /// visual features into the placeholder positions (`compute_inputs_embeds`
    /// with `tile_tag="2D"`).
    fn prepare_inputs(
        &self,
        input_ids: &[u32],
        image_inputs: &JinaOcrImageInputs,
    ) -> Result<Tensor, Error> {
        let seq_len = input_ids.len();
        let token_ids = Tensor::from_vec(input_ids.to_vec(), (1, seq_len), &self.device)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "create prompt token ids", e))?;
        let embeds = self.text.embed(&token_ids)?;

        let visual = self.assemble_visual_features(image_inputs)?;
        let visual_len = visual.dim(0)?;
        let placeholder_positions: Vec<usize> = input_ids
            .iter()
            .enumerate()
            .filter_map(|(index, &token)| (token == self.image_token_id).then_some(index))
            .collect();
        if placeholder_positions.len() != visual_len {
            return Err(Error::InvalidInput {
                message: format!(
                    "JinaOCR: visual token mismatch — encoder produced {visual_len}, mask has {} positions",
                    placeholder_positions.len()
                ),
            });
        }
        let embeds = embeds.squeeze(0)?;
        for (row, &position) in placeholder_positions.iter().enumerate() {
            let feature = visual.i(row..row + 1)?;
            embeds.slice_set(&feature, 0, position)?;
        }
        embeds
            .unsqueeze(0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "merge visual features", e))
    }

    /// `[local | global | view separator]` features for one image,
    /// `(tokens, hidden)`.
    fn assemble_visual_features(&self, inputs: &JinaOcrImageInputs) -> Result<Tensor, Error> {
        let global = self.vision.forward(&inputs.global_view)?.squeeze(0)?;
        let (global_tokens, dim) = global.dims2()?;
        let grid_side = global_tokens.isqrt();
        let global_grid = global.reshape((grid_side, grid_side, dim))?;
        let newline = self.image_newline.reshape((1, 1, dim))?;
        let global_rows = Tensor::cat(
            &[&global_grid, &newline.broadcast_as((grid_side, 1, dim))?],
            1,
        )?
        .reshape((grid_side * (grid_side + 1), dim))?;

        let (width_tiles, height_tiles) = inputs.tile_grid;
        let has_tiles = width_tiles > 1 || height_tiles > 1;
        let local_rows = if has_tiles {
            let tiles = self.vision.forward(&inputs.tiles)?;
            let (tile_count, tokens_per_tile, _) = tiles.dims3()?;
            let tile_side = tokens_per_tile.isqrt();
            debug_assert_eq!(tile_count, width_tiles * height_tiles);
            let tiles = tiles.reshape((height_tiles, width_tiles, tile_side, tile_side, dim))?;
            // (H, q, W, q, D) -> (H*q, W*q, D): height-major patch rows.
            let mosaic = tiles.permute((0, 2, 1, 3, 4))?.contiguous()?;
            let rows = height_tiles * tile_side;
            let cols = width_tiles * tile_side;
            let mosaic = mosaic.reshape((rows, cols, dim))?;
            Tensor::cat(&[&mosaic, &newline.broadcast_as((rows, 1, dim))?], 1)?
                .reshape((rows * (cols + 1), dim))?
        } else {
            Tensor::zeros((0usize, dim), global.dtype(), global.device())?
        };

        let separator = self.view_separator.unsqueeze(0)?;
        Tensor::cat(&[&local_rows, &global_rows, &separator], 0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "visual features cat", e))
    }

    /// Decode generated ids the way `decode_ocr` does: keep special tokens in
    /// the decode, then strip the known special strings.
    pub fn decode_tokens(&self, tokens: &[u32]) -> Result<String, Error> {
        let text = self
            .tokenizer
            .decode(tokens, false)
            .map_err(|e| Error::InvalidInput {
                message: format!("JinaOCR: tokenizer decode failed: {e}"),
            })?;
        let mut text = text;
        for special in [
            "<｜end▁of▁sentence｜>",
            "<｜▁pad▁｜>",
            "<｜begin▁of▁sentence｜>",
            "<image>",
        ] {
            text = text.replace(special, "");
        }
        Ok(text.trim().to_string())
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn config(&self) -> &JinaOcrConfig {
        &self.cfg
    }

    pub fn processor_config(&self) -> &JinaOcrProcessorConfig {
        &self.processor_cfg
    }
}

/// Everything [`JinaOcr::prepare_prompt`] produces for one page.
pub(crate) struct PreparedPrompt {
    pub(crate) input_ids: Vec<u32>,
    pub(crate) inputs_embeds: Tensor,
    pub(crate) position_ids: Tensor,
}

/// Tuning for the adaptive speculation controller. Speculation is paused
/// when a page's recent acceptance sits below the break-even point and
/// probed again periodically; the switch is pure scheduling — MTP and plain
/// greedy decoding are token-identical, so pausing never changes the output.
#[derive(Debug, Clone)]
pub(crate) struct AdaptiveSpec {
    /// Verification rounds in the sliding acceptance window (default 16).
    pub window_rounds: usize,
    /// Mean committed tokens per round below which speculation pauses.
    /// Measured on RTX 4090 (bf16, 1024-token pages, six-page sweep with
    /// sub-0.3% run noise): acceptance ~1.5 lost 3-11%, ~1.9 sat within
    /// ±2%, and 2.4+ won about 10%. The real crossing is close to 2.0, so
    /// pages between 1.7 and 1.9 gain nothing from speculation — pausing
    /// them costs nothing and removes the downside.
    pub break_even: f64,
    /// Plain-decode tokens before the first speculation probe (default
    /// 256). A probe rebuilds the draft state with one dense-layer pass over
    /// the committed prefix (~tens of milliseconds); each consecutive
    /// fallback quadruples the next interval so pages that keep rejecting
    /// stay in plain decoding.
    pub probe_interval: usize,
    /// Capture the decode graph at the first probe (default true;
    /// `OAR_JINAOCR_DISABLE_LAZY_DECODE_GRAPH` forces it off regardless).
    /// Exposed for the GPU self-check, which A/B-compares the two settings.
    pub lazy_decode_graph: bool,
}

impl Default for AdaptiveSpec {
    fn default() -> Self {
        Self {
            window_rounds: 16,
            break_even: 1.9,
            probe_interval: 256,
            lazy_decode_graph: true,
        }
    }
}

/// The greedy engine shared by the autoregressive, traced, and speculative
/// paths. Held separately from [`JinaOcr`] so random-weight unit tests can
/// drive it without a vision tower or tokenizer.
pub(crate) struct GreedyEngine<'a> {
    pub(crate) model_name: &'static str,
    pub(crate) text: &'a DeepSeekV2TextModel,
    pub(crate) lm_head: &'a Linear,
    pub(crate) mtp: Option<&'a JinaOcrMtp>,
    pub(crate) eos_token_ids: &'a [u32],
    pub(crate) device: &'a Device,
}

impl GreedyEngine<'_> {
    /// Plain autoregressive greedy decoding over a prefilled prompt;
    /// `want_trace` attaches the raw top-3 logits of every step.
    pub(crate) fn ar_tokens(
        &self,
        prompt_ids: &[u32],
        prompt_hidden: &Tensor,
        max_new_tokens: usize,
        want_trace: bool,
    ) -> Result<GenerationTrace, Error> {
        let prompt_len = prompt_ids.len();
        let last_hidden = prompt_hidden
            .i((0, prompt_len - 1, ..))
            .map_err(|e| candle_to_ocr_inference(self.model_name, "select prompt hidden", e))?;
        let mut logits = self
            .lm_head
            .forward(&last_hidden.unsqueeze(0)?)
            .and_then(|l| l.squeeze(0))
            .map_err(|e| {
                candle_to_ocr_inference(self.model_name, "prompt language model head", e)
            })?;
        let mut trace = GenerationTrace {
            tokens: Vec::with_capacity(max_new_tokens),
            hit_eos: false,
            step_top: Vec::with_capacity(if want_trace { max_new_tokens } else { 0 }),
        };
        let mut history = prompt_ids.to_vec();

        for step in 0..max_new_tokens {
            let token = if want_trace {
                let mut scores = logits
                    .to_dtype(DType::F32)
                    .and_then(|l| l.to_vec1::<f32>())
                    .map_err(|e| {
                        candle_to_ocr_inference(self.model_name, "read decode scores", e)
                    })?;
                trace.step_top.push(top3(&scores));
                apply_no_repeat_ngram(&history, &mut scores);
                argmax(&scores)?
            } else {
                select_greedy_token(&logits, &history)?
            };
            if self.eos_token_ids.contains(&token) {
                trace.hit_eos = true;
                return Ok(trace);
            }
            trace.tokens.push(token);
            history.push(token);
            if step + 1 == max_new_tokens {
                break;
            }

            let token_ids = Tensor::from_vec(vec![token], (1, 1), self.device).map_err(|e| {
                candle_to_ocr_processing(
                    crate::error::ProcessingStage::TensorOperation,
                    format!("{}: create decode token", self.model_name),
                    e,
                )
            })?;
            let embeds = self.text.embed(&token_ids)?;
            let position = Tensor::arange(
                (prompt_len + step) as u32,
                (prompt_len + step + 1) as u32,
                self.device,
            )?
            .reshape((1, 1, 1))?;
            logits = self
                .text
                .forward_decode_logits(&embeds, &position, None, self.lm_head)?;
        }
        Ok(trace)
    }

    /// Speculative greedy decoding with the FastMTP draft head. Greedy
    /// verification accepts only the token-equality prefix, so the emitted
    /// sequence matches [`Self::ar_tokens`] exactly; the n-gram ban is
    /// applied to the verification logits host-side, sequentially over the
    /// accepted block (positions past a rejection are computed over the
    /// wrong prefix and discarded by the accept loop).
    pub(crate) fn mtp_tokens(
        &self,
        prompt_ids: &[u32],
        prompt_hidden: &Tensor,
        max_new_tokens: usize,
        adaptive: &AdaptiveSpec,
    ) -> Result<(Vec<u32>, bool), Error> {
        let mtp = self.mtp.ok_or_else(|| Error::Config {
            message: format!("{}: MTP draft head is not loaded", self.model_name),
        })?;
        let prompt_len = prompt_ids.len();
        let name = self.model_name;

        // Initial certain token: the greedy pick over the prompt's last
        // position, n-gram ban included.
        let last_hidden = prompt_hidden
            .i((0, prompt_len - 1, ..))
            .map_err(|e| candle_to_ocr_inference(name, "MTP prompt hidden", e))?;
        let prompt_logits = self
            .lm_head
            .forward(&last_hidden.unsqueeze(0)?)
            .and_then(|l| l.squeeze(0))
            .map_err(|e| candle_to_ocr_inference(name, "MTP prompt logits", e))?;
        let mut current = select_greedy_token(&prompt_logits, prompt_ids)?;
        if self.eos_token_ids.contains(&current) {
            return Ok((Vec::new(), true));
        }

        // Speculation starts paused: pages that never reach the first probe
        // (short outputs, or the budget is small) run plain decoding with
        // zero speculative setup, and the first probe pays the only full
        // draft sync of the page.
        let mut drafts: Vec<u32> = Vec::new();

        let mut generated: Vec<u32> = Vec::with_capacity(max_new_tokens);
        let mut history = prompt_ids.to_vec();
        let mut base = prompt_len;
        let mut position = prompt_len as u32;
        let mut hit_eos = false;
        let mut rounds = 0usize;
        let mut accepted_drafts = 0usize;
        let mut accepted_by_position = [0usize; MTP_DRAFT_TOKENS];
        // Hidden states of the committed generated tokens (positions
        // `prompt_len..base`), kept so a speculation probe can rebuild the
        // draft state after a pause.
        let mut committed_hiddens: Vec<Tensor> = Vec::with_capacity(max_new_tokens);
        // Adaptive controller: speculate while the recent acceptance window
        // holds up, fall back to plain decoding when it does not, and probe
        // again every `probe_interval` committed tokens.
        let mut window: std::collections::VecDeque<usize> =
            std::collections::VecDeque::with_capacity(adaptive.window_rounds.max(1));
        let mut cooldown_remaining = adaptive.probe_interval;
        let mut next_probe_interval = adaptive.probe_interval;
        let mut graphs_ready = false;

        loop {
            if self.eos_token_ids.contains(&current) {
                hit_eos = true;
                break;
            }
            generated.push(current);
            history.push(current);
            if generated.len() == max_new_tokens {
                break;
            }

            if drafts.is_empty() && cooldown_remaining == 0 {
                // Budget-exhaustion safety: the cooldown path always fills
                // `drafts` before it expires.
                return Err(Error::Config {
                    message: format!("{name}: speculation has neither drafts nor a cooldown"),
                });
            }

            if cooldown_remaining > 0 {
                // Plain greedy step (target KV is exactly the committed
                // prefix). The hidden state is kept for the next probe.
                let token_ids = Tensor::from_vec(vec![current], (1, 1), self.device)
                    .map_err(|e| candle_to_ocr_inference(name, "cooldown decode token", e))?;
                let embeds = self.text.embed(&token_ids)?;
                let pos =
                    Tensor::arange(position, position + 1, self.device)?.reshape((1, 1, 1))?;
                let (logits, hidden) =
                    self.text
                        .forward_decode_logits_and_hidden(&embeds, &pos, self.lm_head)?;
                committed_hiddens.push(hidden.unsqueeze(0)?.unsqueeze(0)?);
                current = select_greedy_token(&logits, &history)?;
                base += 1;
                position += 1;
                cooldown_remaining -= 1;
                if cooldown_remaining == 0 {
                    // First probe pays the page's only speculative setup:
                    // graph capture plus a full draft sync.
                    if !graphs_ready {
                        self.prepare_speculation_graphs(
                            prompt_ids.len(),
                            max_new_tokens,
                            generated.len(),
                            adaptive.lazy_decode_graph,
                        )?;
                        graphs_ready = true;
                    }
                    drafts = self.rebuild_drafts(
                        mtp,
                        prompt_hidden,
                        &history,
                        current,
                        &committed_hiddens,
                        base,
                    )?;
                    window.clear();
                }
                continue;
            }

            // Verify [current, drafts..] in one causal target pass.
            let mut query_ids = Vec::with_capacity(MTP_QUERY_LEN);
            query_ids.push(current);
            query_ids.extend_from_slice(&drafts);
            let query = Tensor::from_vec(query_ids.clone(), (1, MTP_QUERY_LEN), self.device)
                .map_err(|e| candle_to_ocr_inference(name, "MTP verification ids", e))?;
            let embeds = self.text.embed(&query)?;
            let query_positions =
                Tensor::arange(position, position + MTP_QUERY_LEN as u32, self.device)?
                    .reshape((1, 1, MTP_QUERY_LEN))?;
            let (block_hidden, block_logits) =
                self.text
                    .forward_verification_tokens(&embeds, &query_positions, self.lm_head)?;
            // Greedy verification, ban applied sequentially over the block.
            let mut block_tokens = [0u32; MTP_QUERY_LEN];
            let mut block_history = history.clone();
            for slot in 0..MTP_QUERY_LEN {
                let row = block_logits.i(slot)?;
                block_tokens[slot] = select_greedy_token(&row, &block_history)?;
                block_history.push(query_ids[slot]);
            }

            let mut accepted = 0usize;
            let mut stop = false;
            rounds += 1;
            while accepted < MTP_DRAFT_TOKENS && drafts[accepted] == block_tokens[accepted] {
                let token = drafts[accepted];
                accepted += 1;
                accepted_drafts += 1;
                accepted_by_position[accepted - 1] += 1;
                if self.eos_token_ids.contains(&token) {
                    hit_eos = true;
                    stop = true;
                    break;
                }
                generated.push(token);
                history.push(token);
                if generated.len() == max_new_tokens {
                    stop = true;
                    break;
                }
            }
            if stop {
                break;
            }

            let next_token = block_tokens[accepted];
            let keep = accepted + 1;

            // The verification block wrote MTP_QUERY_LEN KV entries; keep
            // only the certain current token plus its accepted draft prefix,
            // and roll the recurrent MTP tail back to the same base.
            self.text.trim_kv_cache(base + keep)?;
            mtp.trim_kv_cache(base)?;
            let round_hiddens = block_hidden
                .narrow(1, 0, keep)
                .map_err(|e| candle_to_ocr_inference(name, "MTP committed hiddens", e))?;
            for row in 0..keep {
                committed_hiddens.push(round_hiddens.i((0, row))?.unsqueeze(0)?.unsqueeze(0)?);
            }
            base += keep;
            position += keep as u32;
            current = next_token;

            if window.is_empty() {
                // Fresh window (first round or just re-entered after a probe).
                next_probe_interval = adaptive.probe_interval;
            }
            window.push_back(keep);
            if window.len() > adaptive.window_rounds {
                window.pop_front();
            }
            if window.len() == adaptive.window_rounds {
                let mean: f64 =
                    window.iter().map(|&keep| keep as f64).sum::<f64>() / window.len() as f64;
                if mean < adaptive.break_even {
                    tracing::debug!(
                        committed = generated.len(),
                        mean,
                        break_even = adaptive.break_even,
                        next_cooldown = next_probe_interval,
                        "MTP fallback to plain decoding"
                    );
                    cooldown_remaining = next_probe_interval;
                    // Exponential backoff: a page that keeps rejecting
                    // speculation spends progressively more of itself in
                    // plain decoding.
                    next_probe_interval = next_probe_interval.saturating_mul(4);
                    window.clear();
                }
            }

            // Re-sync the draft with the accepted target span.
            let mut sync_ids = query_ids[1..keep].to_vec();
            sync_ids.push(current);
            let sync = Tensor::from_vec(sync_ids, (1, keep), self.device)
                .map_err(|e| candle_to_ocr_inference(name, "MTP sync ids", e))?;
            let sync_hidden = block_hidden
                .narrow(1, 0, keep)
                .map_err(|e| candle_to_ocr_inference(name, "MTP sync target hidden", e))?;
            let sync_positions = Tensor::arange(position - keep as u32, position, self.device)?
                .reshape((1, 1, keep))?;
            let (span_hidden, span_tokens) =
                mtp.sync_target_span(&sync, &sync_hidden, &sync_positions, false)?;
            drafts = self.complete_mtp_drafts(mtp, &span_hidden, &span_tokens, position)?;
        }
        if rounds > 0 {
            tracing::debug!(
                rounds,
                accepted_drafts,
                plain_decode_tokens = generated.len().saturating_sub(accepted_drafts + rounds),
                ?accepted_by_position,
                mean_acceptance_length = 1.0 + accepted_drafts as f64 / rounds as f64,
                "JinaOCR MTP acceptance"
            );
        }
        Ok((generated, hit_eos))
    }

    /// Capture the target verification and draft graphs (CUDA only; a no-op
    /// elsewhere). Instrumented with `tracing::debug!` at each stage and the
    /// committed-token position it happens at, so divergences can be checked
    /// against the nearest KV restore. `OAR_JINAOCR_DISABLE_LAZY_DECODE_GRAPH`
    /// keeps cooldown decoding eager (the verification graph and the
    /// snapshot/restore cycle still run) for A/B numerics experiments.
    pub(crate) fn prepare_speculation_graphs(
        &self,
        prompt_len: usize,
        max_new_tokens: usize,
        committed: usize,
        lazy_decode_graph: bool,
    ) -> Result<(), Error> {
        #[cfg(feature = "cuda")]
        {
            let mtp = self.mtp.ok_or_else(|| Error::Config {
                message: format!("{}: MTP draft head is not loaded", self.model_name),
            })?;
            tracing::debug!(
                committed,
                prompt_len,
                max_new_tokens,
                "MTP probe: begin setup"
            );
            // Captures reuse the fixed KV storage but their warmup runs
            // overwrite the leading positions and reset the logical length —
            // snapshot and restore the live cache around them.
            let saved = self.text.save_kv_cache()?;
            tracing::debug!(committed, "MTP probe: KV snapshot taken");
            if let Some(cache_len) = self.text.prepare_verification_cuda_graph(
                prompt_len,
                max_new_tokens,
                MTP_QUERY_LEN,
                self.lm_head,
            )? {
                // Cooldown decoding runs through the graph too: capture the
                // decode graph against the verification bucket so the shared
                // KV storage is reused, not reallocated.
                if let Err(error) = self
                    .text
                    .capture_ar_cuda_graph_with_capacity(cache_len, self.lm_head)
                {
                    tracing::warn!("JinaOCR decode graph capture failed: {error}");
                }
                if let Err(error) = mtp.prepare_cuda_graph(cache_len) {
                    tracing::warn!("JinaOCR MTP graph capture failed: {error}");
                    mtp.disable_cuda_graph();
                }
            } else {
                mtp.disable_cuda_graph();
            }
            self.text.restore_kv_cache(&saved)?;
            tracing::debug!(committed, "MTP probe: KV restored, setup done");
        }
        let _ = (prompt_len, max_new_tokens, committed, lazy_decode_graph);
        Ok(())
    }

    /// Rebuild the draft state over the whole committed prefix — the same
    /// synchronization the initial pass does, fed with the prompt hidden
    /// states plus the committed generated hidden states saved along the way.
    fn rebuild_drafts(
        &self,
        mtp: &JinaOcrMtp,
        prompt_hidden: &Tensor,
        history: &[u32],
        current: u32,
        committed_hiddens: &[Tensor],
        base: usize,
    ) -> Result<Vec<u32>, Error> {
        let name = self.model_name;
        let prompt_len = prompt_hidden
            .dim(1)
            .map_err(|e| candle_to_ocr_inference(name, "MTP rebuild prompt length", e))?;
        debug_assert_eq!(committed_hiddens.len(), base - prompt_len);
        mtp.clear_kv_cache();
        let hidden_full = Tensor::cat(
            &{
                let mut parts: Vec<&Tensor> = vec![prompt_hidden];
                parts.extend(committed_hiddens.iter().map(|t| t as &Tensor));
                parts
            },
            1,
        )?;
        let mut shifted_ids = history[1..].to_vec();
        shifted_ids.push(current);
        debug_assert_eq!(shifted_ids.len(), base);
        let shifted = Tensor::from_vec(shifted_ids, (1, base), self.device)
            .map_err(|e| candle_to_ocr_inference(name, "MTP rebuild ids", e))?;
        let positions = Tensor::arange(0u32, base as u32, self.device)?.reshape((1, 1, base))?;
        let (span_hidden, span_tokens) =
            mtp.sync_target_span(&shifted, &hidden_full, &positions, true)?;
        self.complete_mtp_drafts(mtp, &span_hidden, &span_tokens, base as u32)
    }

    /// Recurrently complete the draft block from a synchronized span: the
    /// span's last proposal is the first draft, each further draft re-feeds
    /// the preceding MTP hidden state.
    fn complete_mtp_drafts(
        &self,
        mtp: &JinaOcrMtp,
        first_hidden: &Tensor,
        first_tokens: &Tensor,
        position_after_span: u32,
    ) -> Result<Vec<u32>, Error> {
        let name = self.model_name;
        let seq_len = first_hidden
            .dim(1)
            .map_err(|e| candle_to_ocr_inference(name, "MTP span length", e))?;
        let mut hidden = first_hidden
            .narrow(1, seq_len - 1, 1)
            .map_err(|e| candle_to_ocr_inference(name, "MTP final hidden state", e))?;
        let token_count = first_tokens
            .dim(0)
            .map_err(|e| candle_to_ocr_inference(name, "MTP proposal count", e))?;
        let mut token = first_tokens
            .narrow(0, token_count - 1, 1)
            .and_then(|token| token.reshape((1, 1)))
            .map_err(|e| candle_to_ocr_inference(name, "MTP first proposal", e))?;
        let mut draft_tensors = Vec::with_capacity(MTP_DRAFT_TOKENS);

        while draft_tensors.len() < MTP_DRAFT_TOKENS {
            // CUDA-graph replay overwrites its captured output storage. Keep
            // each proposal in independent storage before launching the next
            // recurrent step, otherwise earlier draft handles would silently
            // observe the newest token.
            draft_tensors.push(
                token
                    .copy()
                    .map_err(|e| candle_to_ocr_inference(name, "save MTP proposal", e))?,
            );
            if draft_tensors.len() == MTP_DRAFT_TOKENS {
                break;
            }
            let position = Tensor::arange(
                position_after_span + draft_tensors.len() as u32 - 1,
                position_after_span + draft_tensors.len() as u32,
                self.device,
            )?
            .reshape((1, 1, 1))?;
            let (next_hidden, next_token) = mtp.predict_single(&token, &hidden, &position)?;
            hidden = next_hidden;
            token = next_token
                .reshape((1, 1))
                .map_err(|e| candle_to_ocr_inference(name, "MTP proposal shape", e))?;
        }

        let refs: Vec<&Tensor> = draft_tensors.iter().collect();
        Tensor::cat(&refs, 1)
            .and_then(|drafts| drafts.flatten_all())
            .and_then(|drafts| drafts.to_vec1::<u32>())
            .map_err(|e| candle_to_ocr_inference(name, "copy MTP proposals", e))
    }
}

fn require_token_id(
    tokenizer: &Tokenizer,
    token: &str,
    expected: Option<u32>,
) -> Result<u32, Error> {
    let token_id = tokenizer.token_to_id(token).ok_or_else(|| Error::Config {
        message: format!("JinaOCR tokenizer is missing required token {token:?}"),
    })?;
    if let Some(expected) = expected
        && token_id != expected
    {
        return Err(Error::Config {
            message: format!(
                "JinaOCR token {token:?} id mismatch: tokenizer {token_id} != config {expected}"
            ),
        });
    }
    Ok(token_id)
}

/// `SlidingWindowNoRepeatNgramProcessor`: the token ids that would complete
/// an n-gram already seen within the last `NGRAM_WINDOW` tokens of the
/// running sequence, minus the whitelist.
fn ngram_banned_tokens(sequence: &[u32]) -> Vec<u32> {
    if NGRAM_SIZE == 0 {
        return Vec::new();
    }
    let len = sequence.len();
    if len < NGRAM_SIZE {
        return Vec::new();
    }
    let search_start = len.saturating_sub(NGRAM_WINDOW);
    let search_end = len - NGRAM_SIZE + 1;
    if search_end <= search_start {
        return Vec::new();
    }
    let prefix = &sequence[len - NGRAM_SIZE + 1..];
    let mut banned = Vec::new();
    for start in search_start..search_end {
        if sequence[start..start + NGRAM_SIZE - 1] == *prefix {
            let next = sequence[start + NGRAM_SIZE - 1];
            if !NGRAM_WHITELIST.contains(&next) {
                banned.push(next);
            }
        }
    }
    banned
}

/// Host-side application of [`ngram_banned_tokens`] to a score row.
fn apply_no_repeat_ngram(sequence: &[u32], scores: &mut [f32]) {
    for token in ngram_banned_tokens(sequence) {
        scores[token as usize] = f32::NEG_INFINITY;
    }
}

/// Greedy pick under the n-gram ban without streaming the vocabulary through
/// the host: the banned set depends only on the token history, so a sparse
/// device-side mask plus a stable device argmax reads back a single token.
/// Semantics match `apply_no_repeat_ngram` + `argmax` exactly.
fn select_greedy_token(logits: &Tensor, history: &[u32]) -> Result<u32, Error> {
    if logits.device().is_cuda() {
        #[cfg(feature = "cuda")]
        {
            use crate::runtime::cuda::{ArgmaxFirstBf16, ArgmaxFirstF32, MaskTokenIds};
            let vocab = logits.elem_count();
            let row = logits
                .reshape((1, vocab))
                .and_then(|l| l.contiguous())
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "reshape greedy logits", e))?;
            let banned = ngram_banned_tokens(history);
            let row = if banned.is_empty() {
                row
            } else {
                let ids = Tensor::new(banned, logits.device())
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "upload banned ids", e))?;
                row.inplace_op2(&ids, &MaskTokenIds)
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "mask banned ids", e))?;
                row
            };
            // F16 logits upcast to F32 first: the argmax is breadth-one,
            // so the cast costs one vocab-wide copy.
            let row = if row.dtype() == DType::F16 {
                row.to_dtype(DType::F32)
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "upcast f16 logits", e))?
            } else {
                row
            };
            let token = match row.dtype() {
                DType::BF16 => row.apply_op1_no_bwd(&ArgmaxFirstBf16),
                DType::F32 => row.apply_op1_no_bwd(&ArgmaxFirstF32),
                dtype => {
                    return Err(Error::Config {
                        message: format!("{MODEL_NAME}: unsupported greedy logits dtype {dtype:?}"),
                    });
                }
            }
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "stable GPU argmax", e))?;
            return token
                .i(0)
                .and_then(|t| t.to_scalar::<u32>())
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "copy greedy token", e));
        }
        #[allow(unreachable_code)]
        return Err(Error::Config {
            message: format!("{MODEL_NAME}: CUDA logits on a non-CUDA build"),
        });
    }
    let mut scores = logits
        .to_dtype(DType::F32)
        .and_then(|l| l.to_vec1::<f32>())
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "read greedy scores", e))?;
    apply_no_repeat_ngram(history, &mut scores);
    argmax(&scores)
}

fn argmax(scores: &[f32]) -> Result<u32, Error> {
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

/// Top-3 `(token, logit)` pairs of a raw score row, ties toward the lower
/// token id like `argmax`.
fn top3(scores: &[f32]) -> [(u32, f32); 3] {
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
    best.map(|(index, value)| (index as u32, value))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ngram_ban_blocks_repeats_inside_the_window() {
        let sequence = [10u32; 40];
        let mut scores = vec![1f32; 1000];
        apply_no_repeat_ngram(&sequence, &mut scores);
        assert_eq!(scores[10], f32::NEG_INFINITY);
        // Unrelated tokens stay intact.
        assert_eq!(scores[8], 1.0);
    }

    #[test]
    fn ngram_ban_whitelists_table_tags() {
        // 34 sevens then a <td> token: the next 7 would complete a 35-gram.
        let mut sequence = vec![7u32; 34];
        sequence.push(128_821);
        let mut scores = vec![1f32; 200_000];
        apply_no_repeat_ngram(&sequence, &mut scores);
        assert_eq!(scores[7], 1.0);
    }

    #[test]
    fn ngram_ban_ignores_short_sequences() {
        let mut scores = vec![1f32; 1000];
        apply_no_repeat_ngram(&[10u32; 20], &mut scores);
        assert_eq!(scores[7], 1.0);
    }

    #[test]
    fn ngram_window_limits_the_search() {
        // A repeat outside the last `NGRAM_WINDOW` tokens must not be banned.
        let mut sequence = vec![9u32; 40];
        sequence.extend(std::iter::repeat_n(8u32, NGRAM_WINDOW));
        sequence.extend(std::iter::repeat_n(7u32, 40));
        let mut scores = vec![1f32; 1000];
        apply_no_repeat_ngram(&sequence, &mut scores);
        assert_eq!(scores[8], 1.0);
    }

    #[test]
    fn greedy_argmax_prefers_the_first_tied_token() {
        assert_eq!(argmax(&[1.0, 3.0, 3.0, 2.0]).unwrap(), 1);
    }

    #[test]
    fn top3_orders_desc_and_breaks_ties_downward() {
        let top = top3(&[1.0, 5.0, 2.0, 5.0, 4.0]);
        assert_eq!(top[0], (1, 5.0));
        // The tied 5.0 keeps arrival order; token 4 fills rank two.
        assert_eq!(top[1], (3, 5.0));
        assert_eq!(top[2], (4, 4.0));
    }

    #[test]
    fn top3_handles_short_rows() {
        let top = top3(&[7.0]);
        assert_eq!(top[0], (0, 7.0));
        assert_eq!(top[1], (0, f32::NEG_INFINITY));
    }

    // Random-weight equivalence tests for the batch and speculative paths.
    mod engine_tests {
        use super::super::*;
        use crate::backbones::deepseek_v2::DeepSeekV2TextConfig;
        use candle_core::{DType, Device};
        use candle_nn::VarBuilder;
        use std::collections::HashMap;

        fn tiny_config() -> DeepSeekV2TextConfig {
            DeepSeekV2TextConfig {
                vocab_size: 64,
                hidden_size: 32,
                intermediate_size: 48,
                num_hidden_layers: 2,
                num_attention_heads: 4,
                num_key_value_heads: 4,
                rms_norm_eps: 1e-5,
                rope_theta: 10_000.0,
                max_position_embeddings: 512,
                eos_token_id: 3,
                bos_token_id: 0,
                pad_token_id: 2,
                first_k_dense_replace: 1,
                moe_layer_freq: 1,
                n_routed_experts: 4,
                n_shared_experts: 1,
                num_experts_per_tok: 2,
                moe_intermediate_size: 16,
                scoring_func: "softmax".to_string(),
                topk_method: "greedy".to_string(),
                norm_topk_prob: false,
                routed_scaling_factor: 1.0,
                n_group: 1,
                topk_group: 1,
                use_mla: false,
                tie_word_embeddings: false,
                attention_bias: false,
            }
        }

        fn random_varbuilder(
            cfg: &DeepSeekV2TextConfig,
            device: &Device,
            with_mtp: bool,
        ) -> VarBuilder<'static> {
            random_varbuilder_typed(cfg, device, with_mtp, DType::F32)
        }

        fn random_varbuilder_typed(
            cfg: &DeepSeekV2TextConfig,
            device: &Device,
            with_mtp: bool,
            dtype: DType,
        ) -> VarBuilder<'static> {
            let mut tensors: HashMap<String, Tensor> = HashMap::new();
            let h = cfg.hidden_size;
            let put = |tensors: &mut HashMap<String, Tensor>, name: String, shape: Vec<usize>| {
                let len: usize = shape.iter().product();
                let data: Vec<f32> = (0..len)
                    .map(|i| {
                        // Deterministic pseudo-random in [-0.05, 0.05].
                        let x = (i as u32).wrapping_mul(2_654_435_761) % 10_001;
                        (x as f32 / 10_000.0 - 0.5) * 0.1
                    })
                    .collect();
                let tensor = Tensor::from_vec(data, shape, device)
                    .unwrap()
                    .to_dtype(dtype)
                    .unwrap();
                tensors.insert(name, tensor);
            };
            put(
                &mut tensors,
                "model.embed_tokens.weight".into(),
                vec![cfg.vocab_size, h],
            );
            put(&mut tensors, "model.norm.weight".into(), vec![h]);
            put(
                &mut tensors,
                "lm_head.weight".into(),
                vec![cfg.vocab_size, h],
            );
            for layer in 0..cfg.num_hidden_layers {
                let prefix = format!("model.layers.{layer}");
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    put(
                        &mut tensors,
                        format!("{prefix}.self_attn.{proj}.weight"),
                        vec![h, h],
                    );
                }
                put(
                    &mut tensors,
                    format!("{prefix}.input_layernorm.weight"),
                    vec![h],
                );
                put(
                    &mut tensors,
                    format!("{prefix}.post_attention_layernorm.weight"),
                    vec![h],
                );
                if layer < cfg.first_k_dense_replace {
                    for proj in ["gate_proj", "up_proj"] {
                        put(
                            &mut tensors,
                            format!("{prefix}.mlp.{proj}.weight"),
                            vec![cfg.intermediate_size, h],
                        );
                    }
                    put(
                        &mut tensors,
                        format!("{prefix}.mlp.down_proj.weight"),
                        vec![h, cfg.intermediate_size],
                    );
                } else {
                    put(
                        &mut tensors,
                        format!("{prefix}.mlp.gate.weight"),
                        vec![cfg.n_routed_experts, h],
                    );
                    for expert in 0..cfg.n_routed_experts {
                        for proj in ["gate_proj", "up_proj"] {
                            put(
                                &mut tensors,
                                format!("{prefix}.mlp.experts.{expert}.{proj}.weight"),
                                vec![cfg.moe_intermediate_size, h],
                            );
                        }
                        put(
                            &mut tensors,
                            format!("{prefix}.mlp.experts.{expert}.down_proj.weight"),
                            vec![h, cfg.moe_intermediate_size],
                        );
                    }
                    for proj in ["gate_proj", "up_proj"] {
                        put(
                            &mut tensors,
                            format!("{prefix}.mlp.shared_experts.{proj}.weight"),
                            vec![cfg.moe_intermediate_size * cfg.n_shared_experts, h],
                        );
                    }
                    put(
                        &mut tensors,
                        format!("{prefix}.mlp.shared_experts.down_proj.weight"),
                        vec![h, cfg.moe_intermediate_size * cfg.n_shared_experts],
                    );
                }
            }
            if with_mtp {
                let prefix = "mtp_module.heads.0";
                put(&mut tensors, format!("{prefix}.enorm.weight"), vec![h]);
                put(&mut tensors, format!("{prefix}.hnorm.weight"), vec![h]);
                put(
                    &mut tensors,
                    format!("{prefix}.eh_proj.weight"),
                    vec![h, h * 2],
                );
                for norm in ["input_layernorm", "post_attention_layernorm"] {
                    put(
                        &mut tensors,
                        format!("{prefix}.mtp_block.{norm}.weight"),
                        vec![h],
                    );
                }
                for proj in ["q_proj", "k_proj", "v_proj", "o_proj"] {
                    put(
                        &mut tensors,
                        format!("{prefix}.mtp_block.self_attn.{proj}.weight"),
                        vec![h, h],
                    );
                }
                for proj in ["gate_proj", "up_proj"] {
                    put(
                        &mut tensors,
                        format!("{prefix}.mtp_block.mlp.{proj}.weight"),
                        vec![cfg.intermediate_size, h],
                    );
                }
                put(
                    &mut tensors,
                    format!("{prefix}.mtp_block.mlp.down_proj.weight"),
                    vec![h, cfg.intermediate_size],
                );
            }
            VarBuilder::from_tensors(tensors, dtype, device)
        }

        struct TinyModel {
            text: DeepSeekV2TextModel,
            mtp: Option<JinaOcrMtp>,
            lm_head: Linear,
            eos: Vec<u32>,
        }

        #[cfg(feature = "cuda")]
        fn build_tiny_on(device: &Device) -> (TinyModel, Device) {
            let cfg = tiny_config();
            let vb = random_varbuilder(&cfg, device, true);
            let text = DeepSeekV2TextModel::load(&cfg, vb.pp("model")).unwrap();
            let lm_head = Linear::new(
                vb.get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")
                    .unwrap(),
                None,
            );
            let mtp = Some(
                JinaOcrMtp::load(
                    &cfg,
                    text.token_embedding_weight(),
                    text.final_norm_weight(),
                    lm_head.weight().clone(),
                    vb.clone(),
                )
                .unwrap(),
            );
            (
                TinyModel {
                    text,
                    mtp,
                    lm_head,
                    eos: vec![3],
                },
                device.clone(),
            )
        }

        fn build_tiny(with_mtp: bool) -> (TinyModel, Device) {
            let device = Device::Cpu;
            let cfg = tiny_config();
            let vb = random_varbuilder(&cfg, &device, with_mtp);
            let text = DeepSeekV2TextModel::load(&cfg, vb.pp("model")).unwrap();
            let lm_head = Linear::new(
                vb.get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")
                    .unwrap(),
                None,
            );
            let mtp = if with_mtp {
                Some(
                    JinaOcrMtp::load(
                        &cfg,
                        text.token_embedding_weight(),
                        text.final_norm_weight(),
                        lm_head.weight().clone(),
                        vb.clone(),
                    )
                    .unwrap(),
                )
            } else {
                None
            };
            (
                TinyModel {
                    text,
                    mtp,
                    lm_head,
                    eos: vec![3],
                },
                device,
            )
        }

        fn prepared_prompt(model: &TinyModel, device: &Device, ids: &[u32]) -> PreparedPrompt {
            let token_ids = Tensor::from_vec(ids.to_vec(), (1, ids.len()), device).unwrap();
            let inputs_embeds = model.text.embed(&token_ids).unwrap();
            let position_ids = Tensor::arange(0u32, ids.len() as u32, device)
                .unwrap()
                .reshape((1, 1, ids.len()))
                .unwrap();
            PreparedPrompt {
                input_ids: ids.to_vec(),
                inputs_embeds,
                position_ids,
            }
        }

        fn engine<'a>(model: &'a TinyModel, device: &'a Device) -> GreedyEngine<'a> {
            GreedyEngine {
                model_name: "JinaOCR-test",
                text: &model.text,
                lm_head: &model.lm_head,
                mtp: model.mtp.as_ref(),
                eos_token_ids: &model.eos,
                device,
            }
        }

        #[test]
        fn mtp_matches_plain_greedy_token_for_token() {
            let (model, device) = build_tiny(true);
            let ids: Vec<u32> = (4..40).map(|i| 8 + i % 50).collect();
            let prompt = prepared_prompt(&model, &device, &ids);
            let max_new_tokens = 48;

            model.text.clear_kv_cache();
            let hidden = model
                .text
                .forward(&prompt.inputs_embeds, &prompt.position_ids, None)
                .unwrap();
            let plain = engine(&model, &device)
                .ar_tokens(&prompt.input_ids, &hidden, max_new_tokens, false)
                .unwrap();

            model.text.clear_kv_cache();
            if let Some(mtp) = model.mtp.as_ref() {
                mtp.clear_kv_cache();
            }
            let hidden = model
                .text
                .forward(&prompt.inputs_embeds, &prompt.position_ids, None)
                .unwrap();
            let (speculative, hit_eos) = engine(&model, &device)
                .mtp_tokens(
                    &prompt.input_ids,
                    &hidden,
                    max_new_tokens,
                    &AdaptiveSpec::default(),
                )
                .unwrap();

            assert_eq!(speculative, plain.tokens, "MTP output must match greedy");
            assert_eq!(hit_eos, plain.hit_eos);
        }

        #[test]
        fn adaptive_fallback_and_reprobe_match_plain_greedy() {
            let (model, device) = build_tiny(true);
            let ids: Vec<u32> = (4..40).map(|i| 8 + i % 50).collect();
            let prompt = prepared_prompt(&model, &device, &ids);
            let max_new_tokens = 40;

            model.text.clear_kv_cache();
            let hidden = model
                .text
                .forward(&prompt.inputs_embeds, &prompt.position_ids, None)
                .unwrap();
            let plain = engine(&model, &device)
                .ar_tokens(&prompt.input_ids, &hidden, max_new_tokens, false)
                .unwrap();

            // Random weights accept almost nothing, so the controller must
            // fall back after two rounds, cool down for three tokens, probe,
            // fall back again — several full cycles across the page.
            let adaptive = AdaptiveSpec {
                window_rounds: 2,
                break_even: 10.0,
                probe_interval: 3,
                lazy_decode_graph: true,
            };
            model.text.clear_kv_cache();
            if let Some(mtp) = model.mtp.as_ref() {
                mtp.clear_kv_cache();
            }
            let hidden = model
                .text
                .forward(&prompt.inputs_embeds, &prompt.position_ids, None)
                .unwrap();
            let (speculative, hit_eos) = engine(&model, &device)
                .mtp_tokens(&prompt.input_ids, &hidden, max_new_tokens, &adaptive)
                .unwrap();

            assert_eq!(
                speculative, plain.tokens,
                "adaptive switching output must match greedy"
            );
            assert_eq!(hit_eos, plain.hit_eos);
        }

        /// GPU self-check for the mid-generation KV snapshot/restore cycle,
        /// run in BF16 so the CUDA graphs (f16/bf16-gated) actually capture:
        /// (1) element-wise KV equality across the production capture flow —
        /// prefill, snapshot, the same graph captures the engine performs
        /// (whose warmups overwrite the storage prefix and reset the length),
        /// restore — with the captures asserted to have happened; (2) token
        /// identity through forced probe → capture → restore → fallback →
        /// re-probe cycles, with and without the lazy decode graph (the only
        /// scheduled difference; the verification capture, snapshot, and
        /// restore run in both). Skips without a CUDA device; opt in with
        /// `OAR_JINAOCR_GPU_SELFTEST=1`.
        #[test]
        fn cuda_kv_snapshot_restore_preserves_state() {
            #[cfg(feature = "cuda")]
            {
                use crate::backbones::deepseek_v2::DeepSeekV2TextModel;
                if std::env::var_os("OAR_JINAOCR_GPU_SELFTEST").is_none() {
                    eprintln!("skipping: OAR_JINAOCR_GPU_SELFTEST is not set");
                    return;
                }
                let Ok(device) = candle_core::Device::new_cuda(0) else {
                    eprintln!("skipping: no CUDA device");
                    return;
                };
                let ids: Vec<u32> = (4..40).map(|i| 8 + i % 50).collect();

                let build = |lazy_decode_graph: bool| {
                    let cfg = tiny_config();
                    let vb = random_varbuilder_typed(&cfg, &device, true, DType::BF16);
                    let text = DeepSeekV2TextModel::load(&cfg, vb.pp("model")).unwrap();
                    let lm_head = Linear::new(
                        vb.get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")
                            .unwrap(),
                        None,
                    );
                    let mtp = JinaOcrMtp::load(
                        &cfg,
                        text.token_embedding_weight(),
                        text.final_norm_weight(),
                        lm_head.weight().clone(),
                        vb.clone(),
                    )
                    .unwrap();
                    let model = TinyModel {
                        text,
                        mtp: Some(mtp),
                        lm_head,
                        eos: vec![3],
                    };
                    let prompt = prepared_prompt(&model, &device, &ids);
                    (model, prompt, lazy_decode_graph)
                };

                // (1) Element-wise snapshot/restore across the real capture
                //     flow, in bf16 so the graphs capture.
                let (model, prompt, _) = build(true);
                model.text.clear_kv_cache();
                let hidden = model
                    .text
                    .forward(&prompt.inputs_embeds, &prompt.position_ids, None)
                    .unwrap();
                let before = model.text.save_kv_cache().unwrap();
                engine(&model, &device)
                    .prepare_speculation_graphs(ids.len(), 48, ids.len(), true)
                    .unwrap();
                let (decode, verification) = model.text.graphs_captured();
                assert!(decode, "decode graph did not capture (dtype gate?)");
                assert!(
                    verification,
                    "verification graph did not capture (dtype gate?)"
                );
                model.text.restore_kv_cache(&before).unwrap();
                let after = model.text.save_kv_cache().unwrap();
                for (layer, (saved, got)) in before.iter().zip(&after).enumerate() {
                    assert_eq!(saved.2, got.2, "layer {layer} length mismatch");
                    let saved_k = saved
                        .0
                        .flatten_all()
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                    let got_k = got
                        .0
                        .flatten_all()
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                    let saved_v = saved
                        .1
                        .flatten_all()
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                    let got_v = got
                        .1
                        .flatten_all()
                        .unwrap()
                        .to_dtype(DType::F32)
                        .unwrap()
                        .to_vec1::<f32>()
                        .unwrap();
                    assert_eq!(saved_k, got_k, "layer {layer} keys differ after restore");
                    assert_eq!(saved_v, got_v, "layer {layer} values differ after restore");
                }

                // Baseline: uninterrupted plain decoding (eager, graphs off
                // via the schedule) for reference.
                let forced = AdaptiveSpec {
                    window_rounds: 2,
                    break_even: 10.0,
                    probe_interval: 4,
                    lazy_decode_graph: false,
                };
                model.text.clear_kv_cache();
                let hidden = model
                    .text
                    .forward(&prompt.inputs_embeds, &prompt.position_ids, None)
                    .unwrap();
                let plain = engine(&model, &device)
                    .ar_tokens(&prompt.input_ids, &hidden, 40, false)
                    .unwrap();

                // (2) Forced probe → capture → restore → fallback → re-probe
                //     cycles, A/B on the lazy decode graph only. Both runs
                //     capture the verification graph mid-generation and both
                //     snapshot/restore around it, so token equality shows the
                //     restore (and the decode-graph replay at T=1) preserve
                //     the greedy sequence.
                let mut results = Vec::new();
                for lazy in [true, false] {
                    let (model, prompt, _) = build(lazy);
                    model.text.clear_kv_cache();
                    let hidden = model
                        .text
                        .forward(&prompt.inputs_embeds, &prompt.position_ids, None)
                        .unwrap();
                    let adaptive = AdaptiveSpec {
                        lazy_decode_graph: lazy,
                        ..forced
                    };
                    let (tokens, hit_eos) = engine(&model, &device)
                        .mtp_tokens(&prompt.input_ids, &hidden, 40, &adaptive)
                        .unwrap();
                    if lazy {
                        let (decode, verification) = model.text.graphs_captured();
                        assert!(
                            decode && verification,
                            "forced-cycle run did not capture both graphs"
                        );
                    }
                    results.push((tokens, hit_eos));
                }
                assert_eq!(
                    results[0].0, plain.tokens,
                    "probe/restore cycles with the decode graph must match greedy"
                );
                assert_eq!(results[0].0, results[1].0, "lazy decode graph A/B mismatch");
                assert_eq!(results[0].1, plain.hit_eos);
                assert_eq!(results[0].1, results[1].1);
            }
            #[cfg(not(feature = "cuda"))]
            eprintln!("skipping: built without the cuda feature");
        }

        /// GPU self-check for graph re-capture within one process, in BF16 so
        /// the graphs capture: a short prompt captures a small bucket, a
        /// longer prompt forces a re-capture, the larger graph then covers
        /// the short prompt again, and finally a second instance captures
        /// while the first graphs are alive. Every graphed run must produce
        /// the same tokens as the same model on the graphs-off schedule.
        /// Skips without a CUDA device; opt in with
        /// `OAR_JINAOCR_GPU_SELFTEST=1`.
        #[test]
        fn cuda_graph_recaptures_and_second_instances_match() {
            #[cfg(feature = "cuda")]
            {
                use crate::backbones::deepseek_v2::DeepSeekV2TextModel;
                if std::env::var_os("OAR_JINAOCR_GPU_SELFTEST").is_none() {
                    eprintln!("skipping: OAR_JINAOCR_GPU_SELFTEST is not set");
                    return;
                }
                let Ok(device) = candle_core::Device::new_cuda(0) else {
                    eprintln!("skipping: no CUDA device");
                    return;
                };
                let short: Vec<u32> = (4..36).map(|i| 8 + i % 50).collect();
                let long: Vec<u32> = (0..600).map(|i| 8 + i % 50).collect();

                let build_bf16 = || {
                    let cfg = tiny_config();
                    let vb = random_varbuilder_typed(&cfg, &device, true, DType::BF16);
                    let text = DeepSeekV2TextModel::load(&cfg, vb.pp("model")).unwrap();
                    let lm_head = Linear::new(
                        vb.get((cfg.vocab_size, cfg.hidden_size), "lm_head.weight")
                            .unwrap(),
                        None,
                    );
                    let mtp = JinaOcrMtp::load(
                        &cfg,
                        text.token_embedding_weight(),
                        text.final_norm_weight(),
                        lm_head.weight().clone(),
                        vb.clone(),
                    )
                    .unwrap();
                    TinyModel {
                        text,
                        mtp: Some(mtp),
                        lm_head,
                        eos: vec![3],
                    }
                };
                let run = |model: &TinyModel, ids: &[u32], lazy: bool| -> Vec<u32> {
                    let prompt = prepared_prompt(model, &device, ids);
                    model.text.clear_kv_cache();
                    let hidden = model
                        .text
                        .forward(&prompt.inputs_embeds, &prompt.position_ids, None)
                        .unwrap();
                    let adaptive = AdaptiveSpec {
                        window_rounds: 2,
                        break_even: 10.0,
                        probe_interval: 4,
                        lazy_decode_graph: lazy,
                    };
                    engine(model, &device)
                        .mtp_tokens(&prompt.input_ids, &hidden, 24, &adaptive)
                        .unwrap()
                        .0
                };

                let model = build_bf16();
                // Small bucket first.
                let ref_short = run(&model, &short, false);
                assert_eq!(
                    run(&model, &short, true),
                    ref_short,
                    "small-bucket graphed run must match the graphs-off schedule"
                );
                let (decode, verification) = model.text.graphs_captured();
                assert!(
                    decode && verification,
                    "small-bucket run did not capture both graphs"
                );

                // Growth past the captured bucket forces a re-capture in the
                // same process.
                let ref_long = run(&model, &long, false);
                assert_eq!(
                    run(&model, &long, true),
                    ref_long,
                    "re-captured graphed run must match the graphs-off schedule"
                );

                // The larger graph now covers the short prompt again.
                assert_eq!(
                    run(&model, &short, true),
                    ref_short,
                    "reused graphed run must match the graphs-off schedule"
                );

                // A second instance capturing while the first graphs live.
                let second = build_bf16();
                let ref_second = run(&second, &long, false);
                assert_eq!(
                    run(&second, &long, true),
                    ref_second,
                    "second-instance graphed run must match the graphs-off schedule"
                );
            }
            #[cfg(not(feature = "cuda"))]
            eprintln!("skipping: built without the cuda feature");
        }

        #[test]
        fn batched_forward_matches_single_sequences() {
            let (model, device) = build_tiny(false);
            let seq_a: Vec<u32> = (4..24).map(|i| 8 + i % 50).collect();
            let seq_b: Vec<u32> = (4..16).map(|i| 20 + i % 40).collect();
            let lens = [seq_a.len(), seq_b.len()];
            let max_len = *lens.iter().max().unwrap();

            // Batched, left-padded prefill with combined mask.
            let mut embeds_rows = Vec::new();
            let mut position_rows = Vec::new();
            for seq in [&seq_a, &seq_b] {
                let prompt = prepared_prompt(&model, &device, seq);
                let pad_len = max_len - seq.len();
                let pad = Tensor::zeros((1, pad_len, 32), DType::F32, &device).unwrap();
                let embeds = Tensor::cat(&[&pad, &prompt.inputs_embeds], 1).unwrap();
                let pad_pos = Tensor::zeros((1, 1, pad_len), DType::U32, &device).unwrap();
                let positions = Tensor::cat(&[&pad_pos, &prompt.position_ids], 2).unwrap();
                embeds_rows.push(embeds);
                position_rows.push(positions);
            }
            let embeds = Tensor::cat(&embeds_rows.iter().collect::<Vec<_>>(), 0).unwrap();
            let positions = Tensor::cat(&position_rows.iter().collect::<Vec<_>>(), 1).unwrap();
            let causal = create_causal_mask(max_len, max_len, DType::F32, &device).unwrap();
            let padding = create_left_padding_mask(&lens, max_len, DType::F32, &device).unwrap();
            let mask = combine_masks(&causal, &padding).unwrap();

            model.text.clear_kv_cache();
            let batched = model
                .text
                .forward(&embeds, &positions, Some(&mask))
                .unwrap();
            let batched_logits = model
                .lm_head
                .forward(
                    &batched
                        .i((.., max_len - 1, ..))
                        .unwrap()
                        .contiguous()
                        .unwrap(),
                )
                .unwrap();

            // Each row alone, no padding and no mask.
            for (row, seq) in [&seq_a, &seq_b].iter().enumerate() {
                let prompt = prepared_prompt(&model, &device, seq);
                model.text.clear_kv_cache();
                let single = model
                    .text
                    .forward(&prompt.inputs_embeds, &prompt.position_ids, None)
                    .unwrap();
                let single_logits = model
                    .lm_head
                    .forward(
                        &single
                            .i((0, seq.len() - 1, ..))
                            .unwrap()
                            .unsqueeze(0)
                            .unwrap(),
                    )
                    .unwrap();
                let a = batched_logits.i(row).unwrap();
                let b = single_logits.i(0).unwrap();
                let diff = (&a - &b).unwrap().abs().unwrap().max_all().unwrap();
                let diff = diff.to_scalar::<f32>().unwrap();
                assert!(
                    diff < 1e-4,
                    "row {row}: batched vs single logit delta {diff}"
                );
            }
        }
    }
}
