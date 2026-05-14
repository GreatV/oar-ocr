use super::config::{MinerUConfig, MinerUImageProcessorConfig};
use super::processing::preprocess_images;
use super::text::MinerUTextModel;
use super::vision::MinerUVisionModel;
#[cfg(feature = "hsd")]
use crate::attention::create_tree_attention_mask;
use crate::attention::{
    combine_masks, create_causal_mask, create_left_padding_mask, on_compute_device,
};
#[cfg(feature = "hsd")]
use crate::hsd::backend_util::{commit_keep_indices, step_pos_ids, tree_pos_ids};
#[cfg(feature = "hsd")]
use crate::hsd::drafting::{
    TargetDraftAdapter, bbox_xyxy, crop_region_image, format_verified_region, map_layout_kind,
    region_markdown_for, region_markdowns_for, structure_result_to_layout_elements,
};
#[cfg(feature = "hsd")]
use crate::hsd::prefix_tree::PrefixTree;
#[cfg(feature = "hsd")]
use crate::hsd::types::{AcceptStats, Draft, HsdConfig, HsdStats, RegionStageStats};
#[cfg(feature = "hsd")]
use crate::hsd::verify::{SpecBackend, spec_decode};
use crate::utils::{candle_to_ocr_inference, candle_to_ocr_processing};
#[cfg(feature = "hsd")]
use candle_core::{D, Result as CandleResult};
use candle_core::{DType, Device, IndexOp, Tensor};
#[cfg(feature = "hsd")]
use candle_nn::ops as cnn_ops;
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};
use image::RgbImage;
use oar_ocr_core::core::OCRError;
use oar_ocr_core::domain::structure::LayoutElementType;
#[cfg(feature = "hsd")]
use oar_ocr_core::domain::structure::{LayoutElement, StructureResult};
use rand::distr::weighted::WeightedIndex;
use rand::prelude::*;
use serde::Deserialize;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::path::Path;
#[cfg(feature = "hsd")]
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

/// Canonical MinerU2.5 per-element prompts as defined by the official
/// `mineru_vl_utils` package (`DEFAULT_PROMPTS` in `mineru_client.py`).
///
/// MinerU's `two_step_extract` flow first runs a layout pass, then routes each
/// cropped region to a per-type recognizer with the matching prompt. Outside
/// of `two_step_extract`, callers can still mix and match: a single
/// `Text Recognition:` prompt fed an entire page yields a generic markdown
/// output (the non-standard usage we previously defaulted to in
/// `hsd_omnidocbench`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[cfg_attr(not(feature = "hsd"), allow(dead_code))]
pub enum MinerUTaskPrompt {
    /// `\nText Recognition:` — default for body text, titles, paragraphs,
    /// lists, captions, references, footnotes, page numbers, etc.
    Text,
    /// `\nFormula Recognition:` — display formulas (`Formula`,
    /// `FormulaNumber`).
    Formula,
    /// `\nTable Recognition:` — tables.
    Table,
    /// `\nImage Analysis:` — figure / image / chart blocks.
    ImageAnalysis,
    /// `\nLayout Detection:` — full-page layout dump (only used by
    /// `two_step_extract` Stage 0, not HSD verify). Kept for completeness
    /// with the official `mineru_vl_utils` prompt set so callers can drive
    /// the layout pass externally if they choose.
    #[allow(dead_code)]
    LayoutDetection,
}

#[cfg_attr(not(feature = "hsd"), allow(dead_code))]
impl MinerUTaskPrompt {
    /// Canonical prompt string (with the leading `\n` that MinerU's
    /// `two_step_extract` builds via its chat-template wrapper).
    pub fn prompt(self) -> &'static str {
        match self {
            Self::Text => "\nText Recognition:",
            Self::Formula => "\nFormula Recognition:",
            Self::Table => "\nTable Recognition:",
            Self::ImageAnalysis => "\nImage Analysis:",
            Self::LayoutDetection => "\nLayout Detection:",
        }
    }

    /// Map an OAR `LayoutElementType` to the MinerU element prompt that best
    /// matches its content kind. Mirrors the heuristic the official `mineru_vl_utils`
    /// client uses when picking a per-block prompt (text-like → `[default]`,
    /// table → `table`, equation → `equation`, image/chart → `image`).
    pub fn for_layout(t: LayoutElementType) -> Self {
        use LayoutElementType::*;
        match t {
            Table => Self::Table,
            Formula | FormulaNumber => Self::Formula,
            Image | Chart | Seal | HeaderImage | FooterImage => Self::ImageAnalysis,
            _ => Self::Text,
        }
    }
}

pub struct MinerU {
    device: Device,
    dtype: DType,
    cfg: MinerUConfig,
    image_cfg: MinerUImageProcessorConfig,
    tokenizer: Tokenizer,
    text: MinerUTextModel,
    vision: MinerUVisionModel,
    lm_head: Linear,
    image_token_id: u32,
    eos_token_ids: Vec<u32>,
    skip_token_ids: HashSet<u32>,
    vision_start_token_id: u32,
    video_token_id: u32,
    spatial_merge_size: usize,
    repetition_penalty: f32,
    no_repeat_ngram_size: usize,
    do_sample: bool,
    temperature: f32,
    top_p: f32,
    top_k: usize,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MinerUEosTokenId {
    Single(u32),
    Multi(Vec<u32>),
}

#[derive(Debug, Deserialize)]
struct MinerUGenerationConfig {
    #[serde(default)]
    do_sample: Option<bool>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    no_repeat_ngram_size: Option<usize>,
    #[serde(default)]
    eos_token_id: Option<MinerUEosTokenId>,
    #[serde(default)]
    pad_token_id: Option<u32>,
}

impl MinerU {
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, OCRError> {
        let model_dir = model_dir.as_ref();
        let cfg = MinerUConfig::from_path(model_dir.join("config.json"))?;
        let image_cfg =
            MinerUImageProcessorConfig::from_path(model_dir.join("preprocessor_config.json"))?;

        if image_cfg.merge_size != cfg.vision_config.spatial_merge_size {
            return Err(OCRError::ConfigError {
                message: format!(
                    "MinerU2.5 merge_size mismatch: preprocessor {} != vision {}",
                    image_cfg.merge_size, cfg.vision_config.spatial_merge_size
                ),
            });
        }
        if image_cfg.patch_size != cfg.vision_config.patch_size {
            return Err(OCRError::ConfigError {
                message: format!(
                    "MinerU2.5 patch_size mismatch: preprocessor {} != vision {}",
                    image_cfg.patch_size, cfg.vision_config.patch_size
                ),
            });
        }

        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| {
            OCRError::ConfigError {
                message: format!("failed to load MinerU2.5 tokenizer.json: {e}"),
            }
        })?;

        let gen_cfg = load_generation_config(model_dir.join("generation_config.json"));
        let repetition_penalty = gen_cfg
            .as_ref()
            .and_then(|cfg| cfg.repetition_penalty)
            .unwrap_or(1.0);
        let no_repeat_ngram_size = gen_cfg
            .as_ref()
            .and_then(|cfg| cfg.no_repeat_ngram_size)
            .unwrap_or(100);
        let do_sample = gen_cfg
            .as_ref()
            .and_then(|cfg| cfg.do_sample)
            .unwrap_or(false);
        let temperature = gen_cfg
            .as_ref()
            .and_then(|cfg| cfg.temperature)
            .unwrap_or(1.0);
        let top_p = gen_cfg.as_ref().and_then(|cfg| cfg.top_p).unwrap_or(1.0);
        let top_k = gen_cfg
            .as_ref()
            .and_then(|cfg| cfg.top_k)
            .map(|v| v as usize)
            .unwrap_or(0);

        if let Some(tok_image_id) = tokenizer.token_to_id("<|image_pad|>")
            && tok_image_id != cfg.image_token_id
        {
            return Err(OCRError::ConfigError {
                message: format!(
                    "MinerU2.5 image_token_id mismatch: tokenizer {tok_image_id} != config {}",
                    cfg.image_token_id
                ),
            });
        }

        let dtype = device.bf16_default_to_f32();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_dir.join("model.safetensors")],
                dtype,
                &device,
            )
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "load model.safetensors", e))?
        };

        let text = MinerUTextModel::load(&cfg, vb.pp("model"))?;
        let vision = MinerUVisionModel::load(&cfg.vision_config, vb.pp("visual"))?;

        let image_token_id = cfg.image_token_id;
        let mut eos_token_ids = vec![cfg.eos_token_id];
        if let Some(gen_cfg) = &gen_cfg {
            if let Some(eos) = &gen_cfg.eos_token_id {
                match eos {
                    MinerUEosTokenId::Single(id) => eos_token_ids.push(*id),
                    MinerUEosTokenId::Multi(ids) => eos_token_ids.extend(ids.iter().copied()),
                }
            }
            if let Some(pad) = gen_cfg.pad_token_id {
                eos_token_ids.push(pad);
            }
        }
        eos_token_ids.sort_unstable();
        eos_token_ids.dedup();

        // Build skip_token_ids set (bos, eos, pad) for filtering before decode
        let mut skip_token_ids: HashSet<u32> = HashSet::new();
        skip_token_ids.insert(cfg.bos_token_id);
        skip_token_ids.extend(eos_token_ids.iter().copied());
        if let Some(pad) = cfg.pad_token_id {
            skip_token_ids.insert(pad);
        }

        let vision_start_token_id = cfg.vision_start_token_id;
        let video_token_id = cfg.video_token_id;
        let spatial_merge_size = cfg.vision_config.spatial_merge_size;

        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(text.token_embedding_weight(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "load lm_head", e))?
        };

        Ok(Self {
            device,
            dtype,
            cfg,
            image_cfg,
            tokenizer,
            text,
            vision,
            lm_head,
            image_token_id,
            eos_token_ids,
            skip_token_ids,
            vision_start_token_id,
            video_token_id,
            spatial_merge_size,
            repetition_penalty,
            no_repeat_ngram_size,
            do_sample,
            temperature,
            top_p,
            top_k,
        })
    }

    /// Generate text for a batch of images and instructions.
    ///
    /// # Arguments
    /// * `images` - Input images to process
    /// * `instructions` - Text instructions/prompts for each image
    /// * `max_new_tokens` - Maximum number of new tokens to generate
    ///
    /// # Returns
    /// Vector of results, one for each input image-instruction pair
    ///
    /// # Limitations
    /// * Batched inference with different sequence lengths has known issues due to
    ///   variable left-padding causing incorrect attention mask computation during
    ///   incremental generation. For reliable results, process samples one at a time
    ///   when sequences have significantly different lengths.
    /// * The current implementation works correctly when all sequences in the batch
    ///   have similar lengths (minimal padding differences).
    pub fn generate(
        &self,
        images: &[RgbImage],
        instructions: &[impl AsRef<str>],
        max_new_tokens: usize,
    ) -> Vec<Result<String, OCRError>> {
        if images.is_empty() {
            return Vec::new();
        }
        if images.len() != instructions.len() {
            return vec![Err(OCRError::InvalidInput {
                message: format!(
                    "MinerU2.5: images count ({}) != instructions count ({})",
                    images.len(),
                    instructions.len()
                ),
            })];
        }

        match self.generate_tokens_internal(images, instructions, max_new_tokens) {
            Ok(results) => results
                .into_iter()
                .map(|tokens| self.decode_generated_tokens(&tokens))
                .collect(),
            Err(e) => {
                let msg = format!("generation failed: {e}");
                (0..images.len())
                    .map(|_| {
                        Err(OCRError::InvalidInput {
                            message: msg.clone(),
                        })
                    })
                    .collect()
            }
        }
    }

    /// Generate raw baseline tokens for oracle-draft / tokenizer round-trip
    /// experiments. Tokens are exactly the ids emitted by the decode loop,
    /// excluding stop tokens, before skip-token filtering and tokenizer decode.
    pub fn generate_tokens(
        &self,
        images: &[RgbImage],
        instructions: &[impl AsRef<str>],
        max_new_tokens: usize,
    ) -> Vec<Result<Vec<u32>, OCRError>> {
        if images.is_empty() {
            return Vec::new();
        }
        if images.len() != instructions.len() {
            return vec![Err(OCRError::InvalidInput {
                message: format!(
                    "MinerU2.5: images count ({}) != instructions count ({})",
                    images.len(),
                    instructions.len()
                ),
            })];
        }

        match self.generate_tokens_internal(images, instructions, max_new_tokens) {
            Ok(results) => results.into_iter().map(Ok).collect(),
            Err(e) => {
                let msg = format!("generation failed: {e}");
                (0..images.len())
                    .map(|_| {
                        Err(OCRError::InvalidInput {
                            message: msg.clone(),
                        })
                    })
                    .collect()
            }
        }
    }

    fn generate_tokens_internal(
        &self,
        images: &[RgbImage],
        instructions: &[impl AsRef<str>],
        max_new_tokens: usize,
    ) -> Result<Vec<Vec<u32>>, OCRError> {
        let batch_size = images.len();

        let image_inputs = preprocess_images(images, &self.image_cfg, &self.device, self.dtype)?;
        let image_token_counts: Vec<usize> = image_inputs
            .image_grid_thw
            .iter()
            .map(|&(t, h, w)| {
                let denom = self.spatial_merge_size * self.spatial_merge_size;
                (t * h * w) / denom
            })
            .collect();

        let mut all_input_ids: Vec<Vec<u32>> = Vec::with_capacity(batch_size);

        for (instruction, &image_token_count) in instructions.iter().zip(image_token_counts.iter())
        {
            let prompt = build_prompt(instruction.as_ref());
            let enc = self
                .tokenizer
                .encode(prompt, false)
                .map_err(|e| OCRError::InvalidInput {
                    message: format!("MinerU2.5: tokenizer encode failed: {e}"),
                })?;

            let input_ids =
                expand_image_tokens(enc.get_ids(), self.image_token_id, &[image_token_count])?;
            all_input_ids.push(input_ids);
        }

        let image_embeds_all = self
            .vision
            .forward(&image_inputs.pixel_values, &image_inputs.image_grid_thw)?;
        let expected_embeds: usize = image_token_counts.iter().sum();
        let actual_embeds = image_embeds_all
            .dim(0)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "image_embeds dim", e))?;
        if actual_embeds != expected_embeds {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "MinerU2.5: image embeds count mismatch: got {actual_embeds}, expected {expected_embeds}"
                ),
            });
        }

        let seq_lens: Vec<usize> = all_input_ids.iter().map(|ids| ids.len()).collect();
        let max_seq_len = *seq_lens.iter().max().unwrap();

        let mut batch_embeds: Vec<Tensor> = Vec::with_capacity(batch_size);
        let mut rope_deltas: Vec<i64> = Vec::with_capacity(batch_size);
        let mut batch_position_ids: Vec<Tensor> = Vec::with_capacity(batch_size);
        let mut embed_offset = 0usize;
        let mut history_tokens: Vec<Vec<u32>> = all_input_ids.clone();

        for (i, input_ids) in all_input_ids.iter().enumerate() {
            let seq_len = input_ids.len();
            let pad_len = max_seq_len - seq_len;
            let image_token_count = image_token_counts[i];

            let image_embeds = image_embeds_all
                .narrow(0, embed_offset, image_token_count)
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "narrow image embeds", e))?;
            embed_offset += image_token_count;

            let input_ids_t = Tensor::new(input_ids.clone(), &self.device)
                .and_then(|t| t.reshape((1, seq_len)))
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "create input_ids", e))?;
            let mut inputs_embeds = self.text.embed(&input_ids_t)?;

            let first_img_pos = input_ids.iter().position(|&id| id == self.image_token_id);
            if let Some(first_pos) = first_img_pos {
                let image_end = first_pos + image_token_count;
                if image_end > seq_len {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "MinerU2.5: image token span out of range: {image_end} > {seq_len}"
                        ),
                    });
                }
                let mut parts: Vec<Tensor> = Vec::with_capacity(3);
                if first_pos > 0 {
                    parts.push(
                        inputs_embeds.narrow(1, 0, first_pos).map_err(|e| {
                            candle_to_ocr_inference("MinerU2.5", "narrow prefix", e)
                        })?,
                    );
                }
                parts.push(
                    image_embeds
                        .unsqueeze(0)
                        .map_err(|e| candle_to_ocr_inference("MinerU2.5", "unsqueeze img", e))?,
                );
                if image_end < seq_len {
                    parts.push(
                        inputs_embeds
                            .narrow(1, image_end, seq_len - image_end)
                            .map_err(|e| {
                                candle_to_ocr_inference("MinerU2.5", "narrow suffix", e)
                            })?,
                    );
                }

                let refs: Vec<&Tensor> = parts.iter().collect();
                inputs_embeds = Tensor::cat(&refs, 1)
                    .map_err(|e| candle_to_ocr_inference("MinerU2.5", "cat embeds", e))?;
            }

            if pad_len > 0 {
                let pad = Tensor::zeros(
                    (1, pad_len, self.cfg.hidden_size),
                    inputs_embeds.dtype(),
                    &self.device,
                )
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "create pad", e))?;
                inputs_embeds = Tensor::cat(&[&pad, &inputs_embeds], 1)
                    .map_err(|e| candle_to_ocr_inference("MinerU2.5", "cat pad", e))?;
            }
            batch_embeds.push(inputs_embeds);

            let (pos_ids, delta) = get_rope_index(
                &self.cfg,
                input_ids,
                &[image_inputs.image_grid_thw[i]],
                self.vision_start_token_id,
                self.video_token_id,
                self.spatial_merge_size,
                &self.device,
            )?;
            rope_deltas.push(delta);

            let pos_ids = if pad_len > 0 {
                let pad_pos = Tensor::zeros((3, 1, pad_len), DType::I64, &self.device)
                    .map_err(|e| candle_to_ocr_inference("MinerU2.5", "create pad pos", e))?;
                Tensor::cat(&[&pad_pos, &pos_ids], 2)
                    .map_err(|e| candle_to_ocr_inference("MinerU2.5", "cat pad pos", e))?
            } else {
                pos_ids
            };
            batch_position_ids.push(pos_ids);
        }

        let batch_refs: Vec<&Tensor> = batch_embeds.iter().collect();
        let inputs_embeds = Tensor::cat(&batch_refs, 0)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "stack embeds", e))?;

        let pos_refs: Vec<&Tensor> = batch_position_ids.iter().collect();
        let position_ids = Tensor::cat(&pos_refs, 1)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "stack pos", e))?;

        let causal = create_causal_mask(max_seq_len, max_seq_len, self.dtype, &self.device)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "create causal", e))?;
        let padding = create_left_padding_mask(&seq_lens, max_seq_len, self.dtype, &self.device)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "create padding", e))?;
        let mask = combine_masks(&causal, &padding)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "combine masks", e))?;

        self.text.clear_kv_cache();
        let hidden = self
            .text
            .forward(&inputs_embeds, &position_ids, Some(&mask))?;

        let mut logits_list: Vec<Tensor> = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let last = hidden
                .i((i, max_seq_len - 1, ..))
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "get last hidden", e))?;
            let logits = self
                .lm_head
                .forward(&last)
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "lm_head", e))?;
            let logits = logits
                .squeeze(0)
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "lm_head squeeze", e))?;
            logits_list.push(logits);
        }

        let mut generated: Vec<Vec<u32>> = vec![Vec::new(); batch_size];
        let mut finished: Vec<bool> = vec![false; batch_size];
        let mut positions: Vec<i64> = seq_lens
            .iter()
            .zip(&rope_deltas)
            .map(|(&len, &d)| (len as i64) + d)
            .collect();

        // Track padding lengths for each sample (for masking during generation)
        // TODO: Batched inference with different padding lengths has issues.
        // Current workaround: process samples one at a time in the example.
        let pad_lens: Vec<usize> = seq_lens.iter().map(|&len| max_seq_len - len).collect();
        // Current KV cache length (grows during generation)
        let mut kv_len = max_seq_len;

        for _ in 0..max_new_tokens {
            if finished.iter().all(|&f| f) {
                break;
            }

            let sampling_params = self.sampling_params();
            let mut next_tokens: Vec<u32> = Vec::with_capacity(batch_size);
            for (i, logits) in logits_list.iter().enumerate() {
                if finished[i] {
                    next_tokens.push(0);
                } else {
                    let tok = select_next_token(logits, &history_tokens[i], &sampling_params)?;
                    if self.eos_token_ids.contains(&tok) {
                        finished[i] = true;
                    } else {
                        generated[i].push(tok);
                        history_tokens[i].push(tok);
                    }
                    next_tokens.push(tok);
                }
            }

            if finished.iter().all(|&f| f) {
                break;
            }

            let tokens = Tensor::new(next_tokens, &self.device)
                .and_then(|t| t.reshape((batch_size, 1)))
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "create tokens", e))?;
            let embeds = self.text.embed(&tokens)?;

            let pos_data: Vec<i64> = positions.iter().flat_map(|&p| [p, p, p]).collect();
            let pos = Tensor::new(pos_data, &self.device)
                .and_then(|t| t.reshape((3, batch_size, 1)))
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "create pos", e))?;

            // Create attention mask for generation step:
            // - Query position can attend to all non-padding positions in KV cache
            // - Padding positions (first pad_len tokens) should be masked
            kv_len += 1;
            let gen_mask = create_generation_mask(&pad_lens, kv_len, self.dtype, &self.device)
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "create gen mask", e))?;

            let hs = self.text.forward(&embeds, &pos, Some(&gen_mask))?;

            logits_list.clear();
            for i in 0..batch_size {
                let h = hs
                    .i((i, 0, ..))
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| candle_to_ocr_inference("MinerU2.5", "get hs", e))?;
                let logits = self
                    .lm_head
                    .forward(&h)
                    .map_err(|e| candle_to_ocr_inference("MinerU2.5", "lm_head step", e))?;
                let logits = logits
                    .squeeze(0)
                    .map_err(|e| candle_to_ocr_inference("MinerU2.5", "lm_head squeeze", e))?;
                logits_list.push(logits);
            }

            for (i, p) in positions.iter_mut().enumerate() {
                if !finished[i] {
                    *p += 1;
                }
            }
        }

        Ok(generated)
    }

    pub fn decode_tokens(&self, tokens: &[u32]) -> Result<String, OCRError> {
        self.decode_generated_tokens(tokens)
    }

    /// Decode tokens in the form the model actually emitted. MinerU2.5's
    /// `decode_tokens` only filters bos/eos/pad before `tokenizer.decode` —
    /// there is no markdown / wrapping / layout post-process at this layer
    /// (layout-aware reordering happens in `two_step_extract`, not here).
    /// This alias exists for API symmetry with PaddleOCR-VL / GLM-OCR.
    pub fn decode_tokens_raw(&self, tokens: &[u32]) -> Result<String, OCRError> {
        self.decode_generated_tokens(tokens)
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn sampling_params(&self) -> SamplingParams {
        SamplingParams {
            repetition_penalty: self.repetition_penalty,
            no_repeat_ngram_size: self.no_repeat_ngram_size,
            do_sample: self.do_sample,
            temperature: self.temperature,
            top_p: self.top_p,
            top_k: self.top_k,
        }
    }

    fn decode_generated_tokens(&self, tokens: &[u32]) -> Result<String, OCRError> {
        // Filter out bos/eos/pad tokens before decoding (matching official implementation).
        let filtered: Vec<u32> = tokens
            .iter()
            .copied()
            .filter(|t| !self.skip_token_ids.contains(t))
            .collect();
        self.tokenizer
            .decode(&filtered, false) // skip_special_tokens=false to preserve special tokens
            .map_err(|e| OCRError::InvalidInput {
                message: format!("decode failed: {e}"),
            })
    }

    /// Hierarchical Speculative Decoding entry for a single image / region.
    ///
    /// MinerU2.5 is naturally per-region (decoupled VLM in the paper's
    /// taxonomy), so the only HSD stage that applies is region-level. The
    /// verifier applies the same repetition / n-gram / sampling logits
    /// processors as the baseline generator before DSV acceptance decisions.
    #[cfg(feature = "hsd")]
    pub fn generate_hsd(
        &self,
        image: &RgbImage,
        instruction: &str,
        drafts: &[String],
        hsd_cfg: &HsdConfig,
    ) -> Result<(String, HsdStats), OCRError> {
        let t_drafter = Instant::now();
        let tokenized = self.tokenize_drafts(drafts)?;
        self.generate_hsd_tokenized(
            image,
            instruction,
            &tokenized,
            hsd_cfg,
            hsd_cfg.max_region_tokens,
            t_drafter.elapsed(),
        )
    }

    /// HSD entry that consumes already-tokenized drafts. This is the oracle
    /// path used by benchmarks to avoid `decode -> encode` tokenizer
    /// round-trips when the draft comes from this backend's own baseline.
    #[cfg(feature = "hsd")]
    pub fn generate_hsd_with_token_drafts(
        &self,
        image: &RgbImage,
        instruction: &str,
        drafts: &[Draft],
        hsd_cfg: &HsdConfig,
    ) -> Result<(String, HsdStats), OCRError> {
        self.generate_hsd_tokenized(
            image,
            instruction,
            drafts,
            hsd_cfg,
            hsd_cfg.max_region_tokens,
            Duration::ZERO,
        )
    }

    #[cfg(feature = "hsd")]
    fn tokenize_drafts(&self, drafts: &[String]) -> Result<Vec<Draft>, OCRError> {
        let mut tokenized: Vec<Draft> = Vec::with_capacity(drafts.len());
        for d in drafts {
            if d.trim().is_empty() {
                continue;
            }
            let enc =
                self.tokenizer
                    .encode(d.as_str(), false)
                    .map_err(|e| OCRError::InvalidInput {
                        message: format!("MinerU2.5 HSD: tokenizer encode failed: {e}"),
                    })?;
            let tokens = enc.get_ids().to_vec();
            if !tokens.is_empty() {
                tokenized.push(Draft::new(tokens));
            }
        }
        Ok(tokenized)
    }

    #[cfg(feature = "hsd")]
    fn generate_hsd_tokenized(
        &self,
        image: &RgbImage,
        instruction: &str,
        tokenized: &[Draft],
        hsd_cfg: &HsdConfig,
        max_new_tokens: usize,
        drafter_elapsed: Duration,
    ) -> Result<(String, HsdStats), OCRError> {
        let mut stats = HsdStats {
            drafter: drafter_elapsed,
            ..Default::default()
        };
        // Stage 2 fields are reused for stat bookkeeping in the single-image path.
        let t_pre = Instant::now();
        let (initial_lp, rope_delta, prompt_tokens) =
            self.hsd_prefill_single(image, instruction)?;
        stats.stage2.vision_prefill = t_pre.elapsed();
        stats.stage2.forward_passes = 1;

        let t_dec = Instant::now();
        let mut backend = MinerUSpecBackend::new(self, rope_delta, prompt_tokens);
        let mut accept = AcceptStats::default();
        let mut dsv = Default::default();
        let generated = spec_decode(
            &mut backend,
            tokenized,
            initial_lp,
            max_new_tokens,
            &hsd_cfg.dsv,
            &mut accept,
            &mut dsv,
        )
        .map_err(|e| candle_to_ocr_inference("MinerU2.5", "spec_decode", e))?;
        stats.stage2.decode = t_dec.elapsed();
        stats.stage2.emitted_tokens = generated.len() as u32;
        stats.stage2.accept = accept;
        stats.stage2.dsv = dsv;
        stats.stage2.forward_passes += backend.forward_passes;

        // Strip the first stop token and anything after it before decoding.
        let stop_pos = generated
            .iter()
            .position(|t| self.eos_token_ids.contains(t))
            .unwrap_or(generated.len());
        let trimmed = &generated[..stop_pos];

        // Match generate_internal: filter bos/eos/pad before decode, preserve
        // other special tokens (skip_special_tokens=false).
        let filtered: Vec<u32> = trimmed
            .iter()
            .copied()
            .filter(|t| !self.skip_token_ids.contains(t))
            .collect();
        let text = self
            .tokenizer
            .decode(&filtered, false)
            .map_err(|e| OCRError::InvalidInput {
                message: format!("MinerU2.5 HSD: tokenizer decode failed: {e}"),
            })?;
        Ok((text, stats))
    }

    /// Run the full two-stage HSD: Stage 1 verifies each layout-detected
    /// region against the layout drafter's text, then Stage 2 (gated by
    /// `hsd_cfg.enable_stage2`) verifies the Stage-1-aggregated markdown on
    /// the full image with `hsd_cfg.max_page_tokens` budget.
    ///
    /// - `enable_stage1 = false`: skip per-region verification; build the
    ///   Stage 2 draft set directly from the layout drafter's per-element
    ///   markdowns (`region_markdowns`). Mirrors the paper's Table 8
    ///   "Page-level Spec. Decoding only" ablation.
    /// - `enable_stage2 = false`: return the Stage-1-only aggregation (lossy
    ///   ablation matching paper Table 8).
    ///
    /// `region_instruction` is used only for Stage 1 crop verification;
    /// `page_instruction` is used for Stage 2 full-page verification.
    ///
    /// **Two-step mode**: when `region_instruction` is empty, Stage 1
    /// dispatches a per-element prompt via [`MinerUTaskPrompt::for_layout`]
    /// (e.g. `\nText Recognition:`, `\nTable Recognition:`,
    /// `\nFormula Recognition:`). This mirrors MinerU's official
    /// `two_step_extract` flow where each layout-detected block is routed to
    /// its matching recognizer. Passing a non-empty `region_instruction` keeps
    /// the legacy "one prompt for all regions" behaviour for ablation.
    #[cfg(feature = "hsd")]
    pub fn generate_hsd_full(
        &self,
        image: &RgbImage,
        elements: &[LayoutElement],
        ignore_labels: &[String],
        page_instruction: &str,
        region_instruction: &str,
        hsd_cfg: &HsdConfig,
    ) -> Result<(String, HsdStats), OCRError> {
        let mut stats = HsdStats::default();
        let mut region_md: Vec<(usize, String)> = Vec::with_capacity(elements.len());
        let two_step_mode = region_instruction.trim().is_empty();

        if hsd_cfg.enable_stage1 {
            for (idx, elem) in elements.iter().enumerate() {
                if let Some(label) = &elem.label
                    && ignore_labels.iter().any(|l| l == label)
                {
                    continue;
                }
                // Visual-only regions have no text to verify.
                if matches!(
                    elem.element_type,
                    LayoutElementType::Image
                        | LayoutElementType::HeaderImage
                        | LayoutElementType::FooterImage
                        | LayoutElementType::Seal
                ) {
                    continue;
                }
                let draft = region_markdown_for(elem, TargetDraftAdapter::MinerU);
                if draft.trim().is_empty() {
                    continue;
                }

                let bbox = bbox_xyxy(&elem.bbox);
                let crop = crop_region_image(image, &bbox)?;
                let drafts = vec![draft];
                // Two-step dispatch: pick the official MinerU per-element
                // prompt based on `LayoutElementType`. The legacy fixed-prompt
                // path is taken only when the caller explicitly passes a
                // non-empty `region_instruction`.
                let effective_region_instruction = if two_step_mode {
                    MinerUTaskPrompt::for_layout(elem.element_type).prompt()
                } else {
                    region_instruction
                };
                let (region_text, region_stats) =
                    self.generate_hsd(&crop, effective_region_instruction, &drafts, hsd_cfg)?;
                stats.drafter += region_stats.drafter;

                let kind = map_layout_kind(elem.element_type);
                stats.stage1_regions.push(RegionStageStats {
                    kind,
                    stats: region_stats.stage2.clone(),
                });
                stats.stage1.add_assign(region_stats.stage2);
                let order = elem.order_index.map(|x| x as usize).unwrap_or(idx);
                region_md.push((order, format_verified_region(&region_text, kind)));
            }
        }

        region_md.sort_by_key(|(order, _)| *order);
        let region_md: Vec<String> = region_md
            .into_iter()
            .map(|(_, text)| text)
            .filter(|s| !s.trim().is_empty())
            .collect();

        // Stage 2 — page-level global verification on the full image. Per
        // paper Eq. 3 the page draft is the *unordered set* `Ỹ^pg = {ŷ^(i)}`,
        // one draft per region. We pass the Vec straight to `spec_decode`
        // instead of pre-joining: `collect_candidates` scans each draft
        // independently (Eqs. 1+2), so per-region n-gram locality is
        // preserved even when full-page transitions don't appear naturally
        // in the target VLM's output. Budget = `max_page_tokens`.
        if hsd_cfg.enable_stage2 {
            let t_drafter = Instant::now();
            let page_drafts: Vec<String> = if !region_md.is_empty() {
                region_md.clone()
            } else {
                region_markdowns_for(elements, ignore_labels, TargetDraftAdapter::MinerU)
            };
            if !page_drafts.is_empty() {
                let tokenized = self.tokenize_drafts(&page_drafts)?;
                let (text, s2_stats) = self.generate_hsd_tokenized(
                    image,
                    page_instruction,
                    &tokenized,
                    hsd_cfg,
                    hsd_cfg.max_page_tokens,
                    t_drafter.elapsed(),
                )?;
                stats.stage2 = s2_stats.stage2;
                stats.drafter += s2_stats.drafter;
                return Ok((text, stats));
            }
        }

        // Stage 2 disabled or no draft to verify — return Stage-1-only join
        // as a human-readable fallback. The `\n\n` separator here is for the
        // *output* (caller-facing), not for any further HSD input.
        Ok((region_md.join("\n\n"), stats))
    }

    /// One-call HSD entry that consumes a `StructureResult` (the output of
    /// the OARStructure / PP-StructureV3 pipeline) directly.
    ///
    /// Backfills table HTML / formula LaTeX via
    /// [`structure_result_to_layout_elements`] then delegates to
    /// [`Self::generate_hsd_full`]. When `region_instruction` is empty the
    /// MinerU two-step mode kicks in and each region uses its canonical
    /// per-type prompt (`MinerUTaskPrompt::for_layout`).
    #[cfg(feature = "hsd")]
    pub fn generate_hsd_with_structure(
        &self,
        image: &RgbImage,
        page_instruction: &str,
        region_instruction: &str,
        structure: &StructureResult,
        ignore_labels: &[String],
        hsd_cfg: &HsdConfig,
    ) -> Result<(String, HsdStats), OCRError> {
        let elements = structure_result_to_layout_elements(structure);
        self.generate_hsd_full(
            image,
            &elements,
            ignore_labels,
            page_instruction,
            region_instruction,
            hsd_cfg,
        )
    }

    /// Run a single-image prefill with the supplied instruction. Returns
    /// the F32 last-position log-probabilities and the MRoPE delta.
    #[cfg(feature = "hsd")]
    fn hsd_prefill_single(
        &self,
        image: &RgbImage,
        instruction: &str,
    ) -> Result<(Tensor, i64, Vec<u32>), OCRError> {
        // Preprocess single image.
        let image_inputs = preprocess_images(
            std::slice::from_ref(image),
            &self.image_cfg,
            &self.device,
            self.dtype,
        )?;
        let (t, h, w) = image_inputs.image_grid_thw[0];
        let image_token_count = (t * h * w) / (self.spatial_merge_size * self.spatial_merge_size);

        // Build prompt and expand image placeholders.
        let prompt = build_prompt(instruction);
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| OCRError::InvalidInput {
                message: format!("MinerU2.5 HSD: tokenizer encode failed: {e}"),
            })?;
        let input_ids =
            expand_image_tokens(enc.get_ids(), self.image_token_id, &[image_token_count])?;
        let seq_len = input_ids.len();

        // Vision features.
        let image_embeds = self
            .vision
            .forward(&image_inputs.pixel_values, &image_inputs.image_grid_thw)?;
        let actual = image_embeds
            .dim(0)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "image_embeds dim", e))?;
        if actual != image_token_count {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "MinerU2.5 HSD: image embeds count mismatch: got {actual}, expected {image_token_count}"
                ),
            });
        }

        // Build embeddings, splice in the image tokens.
        let input_ids_t = Tensor::new(input_ids.clone(), &self.device)
            .and_then(|t| t.reshape((1, seq_len)))
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "create input_ids", e))?;
        let mut inputs_embeds = self.text.embed(&input_ids_t)?;

        if let Some(first_pos) = input_ids.iter().position(|&id| id == self.image_token_id) {
            let image_end = first_pos + image_token_count;
            let mut parts: Vec<Tensor> = Vec::with_capacity(3);
            if first_pos > 0 {
                parts.push(
                    inputs_embeds
                        .narrow(1, 0, first_pos)
                        .map_err(|e| candle_to_ocr_inference("MinerU2.5", "narrow prefix", e))?,
                );
            }
            parts.push(
                image_embeds
                    .unsqueeze(0)
                    .map_err(|e| candle_to_ocr_inference("MinerU2.5", "unsqueeze img", e))?,
            );
            if image_end < seq_len {
                parts.push(
                    inputs_embeds
                        .narrow(1, image_end, seq_len - image_end)
                        .map_err(|e| candle_to_ocr_inference("MinerU2.5", "narrow suffix", e))?,
                );
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            inputs_embeds = Tensor::cat(&refs, 1)
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "cat embeds", e))?;
        }

        // 3-axis MRoPE position ids + delta.
        let (pos_ids, rope_delta) = get_rope_index(
            &self.cfg,
            &input_ids,
            &[image_inputs.image_grid_thw[0]],
            self.vision_start_token_id,
            self.video_token_id,
            self.spatial_merge_size,
            &self.device,
        )?;

        let causal = create_causal_mask(seq_len, seq_len, self.dtype, &self.device)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "create causal", e))?;

        self.text.clear_kv_cache();
        let hidden = self.text.forward(&inputs_embeds, &pos_ids, Some(&causal))?;

        let last = hidden
            .i((0, seq_len - 1, ..))
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "get last hidden", e))?;
        let logits = self
            .lm_head
            .forward(&last)
            .and_then(|t| t.squeeze(0))
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "lm_head prefill", e))?;
        let lp = processed_logprobs_from_logits(
            &logits,
            &input_ids,
            &self.sampling_params(),
            &self.device,
        )
        .map_err(|e| candle_to_ocr_inference("MinerU2.5", "logits processors prefill", e))?;
        Ok((lp, rope_delta, input_ids))
    }
}

/// HSD adapter for MinerU2.5. Same shape as PaddleOCR-VL (3-axis MRoPE,
/// independent lm_head, rope_delta captured at prefill).
#[cfg(feature = "hsd")]
struct MinerUSpecBackend<'a> {
    model: &'a MinerU,
    rope_delta: i64,
    history: Vec<u32>,
    pending_tree: Option<PrefixTree>,
    pre_verify_kv: usize,
    forward_passes: u32,
}

#[cfg(feature = "hsd")]
impl<'a> MinerUSpecBackend<'a> {
    fn new(model: &'a MinerU, rope_delta: i64, prompt_tokens: Vec<u32>) -> Self {
        Self {
            model,
            rope_delta,
            history: prompt_tokens,
            pending_tree: None,
            pre_verify_kv: 0,
            forward_passes: 0,
        }
    }

    fn project_logprobs_2d(&self, hidden_2d: &Tensor, tree: &PrefixTree) -> CandleResult<Tensor> {
        let logits = self.model.lm_head.forward(hidden_2d)?;
        let params = self.model.sampling_params();
        let mut rows: Vec<Tensor> = Vec::with_capacity(tree.num_nodes());
        for node_idx in 0..tree.num_nodes() {
            let mut node_history = self.history.clone();
            node_history.extend(tree.path_tokens(node_idx));
            let row = logits.i(node_idx)?;
            rows.push(processed_logprobs_from_logits(
                &row,
                &node_history,
                &params,
                &self.model.device,
            )?);
        }
        let refs: Vec<&Tensor> = rows.iter().collect();
        Tensor::stack(&refs, 0)
    }

    fn project_logprobs_1d(&self, hidden_1d: &Tensor) -> CandleResult<Tensor> {
        let logits = self
            .model
            .lm_head
            .forward(&hidden_1d.unsqueeze(0)?)?
            .squeeze(0)?;
        processed_logprobs_from_logits(
            &logits,
            &self.history,
            &self.model.sampling_params(),
            &self.model.device,
        )
    }
}

#[cfg(feature = "hsd")]
impl<'a> SpecBackend for MinerUSpecBackend<'a> {
    fn step_one(&mut self, token: u32) -> CandleResult<Tensor> {
        let model = self.model;
        let device = &model.device;
        self.history.push(token);

        let tok_t = Tensor::new(vec![token], device)?.reshape((1usize, 1usize))?;
        let embeds = model
            .text
            .embed(&tok_t)
            .map_err(|e| candle_core::Error::Msg(format!("MinerU2.5 HSD step_one embed: {e}")))?;

        let pos_ids = step_pos_ids(3, model.text.current_kv_len(), self.rope_delta, device)?;

        let hidden = model
            .text
            .forward(&embeds, &pos_ids, None)
            .map_err(|e| candle_core::Error::Msg(format!("MinerU2.5 HSD step_one forward: {e}")))?;
        self.forward_passes += 1;
        let last = hidden.i((0, 0, ..))?;
        self.project_logprobs_1d(&last)
    }

    fn verify_tree(&mut self, tree: &PrefixTree) -> CandleResult<Tensor> {
        let n = tree.num_nodes();
        let model = self.model;
        let device = &model.device;
        let dtype = model.dtype;

        let prefix_kv = model.text.current_kv_len();
        self.pre_verify_kv = prefix_kv;

        let tok_t = Tensor::new(tree.tokens.clone(), device)?.reshape((1usize, n))?;
        let embeds = model.text.embed(&tok_t).map_err(|e| {
            candle_core::Error::Msg(format!("MinerU2.5 HSD verify_tree embed: {e}"))
        })?;

        let pos_ids = tree_pos_ids(3, prefix_kv, self.rope_delta, tree, device)?;
        let mask = create_tree_attention_mask(&tree.parents, prefix_kv, dtype, device)?;

        let hidden = model
            .text
            .forward(&embeds, &pos_ids, Some(&mask))
            .map_err(|e| {
                candle_core::Error::Msg(format!("MinerU2.5 HSD verify_tree forward: {e}"))
            })?;
        self.forward_passes += 1;
        let h2 = hidden.squeeze(0)?;
        self.pending_tree = Some(tree.clone());
        self.project_logprobs_2d(&h2, tree)
    }

    fn commit_verify(&mut self, accepted_path: &[usize]) -> CandleResult<()> {
        let indices = commit_keep_indices(self.pre_verify_kv, accepted_path);
        self.model
            .text
            .keep_kv_indices(&indices)
            .map_err(|e| candle_core::Error::Msg(format!("MinerU2.5 HSD commit_verify: {e}")))?;

        if let Some(tree) = self.pending_tree.take() {
            for &p in accepted_path {
                self.history.push(tree.tokens[p]);
            }
        }
        Ok(())
    }

    fn is_eos(&self, tok: u32) -> bool {
        self.model.eos_token_ids.contains(&tok)
    }
}

/// Create attention mask for generation steps.
///
/// During generation, we need to mask out the left-padding positions in the KV cache.
/// The mask allows the current query position to attend to all non-padding positions.
///
/// # Arguments
/// * `pad_lens` - Number of padding tokens at the start of each sequence
/// * `kv_len` - Current KV cache length
/// * `dtype` - Data type for the mask
/// * `device` - Device for the mask
///
/// # Returns
/// Mask tensor of shape (batch, 1, 1, kv_len) where padding positions are -inf
fn create_generation_mask(
    pad_lens: &[usize],
    kv_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor, candle_core::Error> {
    let batch_size = pad_lens.len();

    on_compute_device(device, |compute_device| {
        // pad_lens as tensor: (batch, 1, 1, 1)
        let pad_lens_tensor = Tensor::from_vec(
            pad_lens.iter().map(|&x| x as u32).collect::<Vec<_>>(),
            (batch_size, 1, 1, 1),
            compute_device,
        )?
        .to_dtype(dtype)?;

        // Position indices: (1, 1, 1, kv_len)
        let pos_tensor = Tensor::arange(0u32, kv_len as u32, compute_device)?
            .reshape((1, 1, 1, kv_len))?
            .to_dtype(dtype)?;

        // Mask condition: pos < pad_len -> masked (large negative value)
        let mask_cond = pos_tensor.broadcast_lt(&pad_lens_tensor)?;

        let zero = Tensor::new(0f32, compute_device)?
            .to_dtype(dtype)?
            .broadcast_as(mask_cond.shape())?;
        // Use large negative value instead of -inf to avoid potential numerical issues
        let mask_value = Tensor::new(-1e9_f32, compute_device)?
            .to_dtype(dtype)?
            .broadcast_as(mask_cond.shape())?;

        mask_cond.where_cond(&mask_value, &zero)
    })
}

fn build_prompt(instruction: &str) -> String {
    let separator = if instruction.starts_with(' ') || instruction.starts_with('\n') {
        ""
    } else {
        " "
    };
    format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{separator}{instruction}<|im_end|>\n<|im_start|>assistant\n"
    )
}

fn load_generation_config(path: impl AsRef<Path>) -> Option<MinerUGenerationConfig> {
    let contents = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&contents).ok()
}

struct SamplingParams {
    repetition_penalty: f32,
    no_repeat_ngram_size: usize,
    do_sample: bool,
    temperature: f32,
    top_p: f32,
    top_k: usize,
}

fn select_next_token(
    logits: &Tensor,
    history: &[u32],
    params: &SamplingParams,
) -> Result<u32, OCRError> {
    let logits = logits
        .to_dtype(DType::F32)
        .map_err(|e| candle_to_ocr_inference("MinerU2.5", "logits cast", e))?
        .to_device(&Device::Cpu)
        .map_err(|e| candle_to_ocr_inference("MinerU2.5", "logits to cpu", e))?;
    let mut logits_vec = logits
        .to_vec1::<f32>()
        .map_err(|e| candle_to_ocr_inference("MinerU2.5", "logits to vec", e))?;

    apply_sampling_processors(&mut logits_vec, history, params);

    if !params.do_sample || params.top_k == 1 {
        return Ok(argmax_token(&logits_vec));
    }

    let probs = softmax(&logits_vec);
    if let Some(idx) = sample_from_probs(&probs) {
        Ok(idx as u32)
    } else {
        Ok(argmax_token(&logits_vec))
    }
}

#[cfg(feature = "hsd")]
fn processed_logprobs_from_logits(
    logits: &Tensor,
    history: &[u32],
    params: &SamplingParams,
    device: &Device,
) -> CandleResult<Tensor> {
    let logits = logits.to_dtype(DType::F32)?.to_device(&Device::Cpu)?;
    let mut logits_vec = logits.to_vec1::<f32>()?;
    apply_sampling_processors(&mut logits_vec, history, params);

    let vocab = logits_vec.len();
    let processed = Tensor::from_vec(logits_vec, vocab, device)?;
    cnn_ops::log_softmax(&processed, D::Minus1)
}

fn apply_sampling_processors(logits: &mut [f32], history: &[u32], params: &SamplingParams) {
    apply_repetition_penalty(logits, history, params.repetition_penalty);
    apply_no_repeat_ngram(logits, history, params.no_repeat_ngram_size);

    if !params.do_sample || params.top_k == 1 {
        return;
    }

    let temp = if params.temperature <= 0.0 {
        1.0
    } else {
        params.temperature
    };
    if (temp - 1.0).abs() > f32::EPSILON {
        for val in logits.iter_mut() {
            *val /= temp;
        }
    }

    apply_top_k(logits, params.top_k);
    apply_top_p(logits, params.top_p);
}

fn argmax_token(logits: &[f32]) -> u32 {
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

fn apply_repetition_penalty(logits: &mut [f32], history: &[u32], penalty: f32) {
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

fn apply_top_k(logits: &mut [f32], top_k: usize) {
    if top_k == 0 || top_k >= logits.len() {
        return;
    }
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(Ordering::Less));
    for &idx in indices.iter().skip(top_k) {
        logits[idx] = f32::NEG_INFINITY;
    }
}

fn apply_top_p(logits: &mut [f32], top_p: f32) {
    if !(0.0..1.0).contains(&top_p) {
        return;
    }
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(Ordering::Less));

    let max = logits[indices[0]];
    let mut exp_vals: Vec<f32> = Vec::with_capacity(indices.len());
    let mut exp_sum = 0.0f32;
    for &idx in &indices {
        let val = logits[idx];
        let exp = if val.is_finite() {
            (val - max).exp()
        } else {
            0.0
        };
        exp_vals.push(exp);
        exp_sum += exp;
    }
    if exp_sum == 0.0 {
        return;
    }

    let mut cumulative = 0.0f32;
    for (rank, _) in indices.iter().enumerate() {
        let prob = exp_vals[rank] / exp_sum;
        cumulative += prob;
        if cumulative > top_p && rank > 0 {
            for &drop in indices.iter().skip(rank) {
                logits[drop] = f32::NEG_INFINITY;
            }
            break;
        }
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let mut max = f32::NEG_INFINITY;
    for &val in logits {
        if val.is_finite() && val > max {
            max = val;
        }
    }
    let mut exps = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    for &val in logits {
        let exp = if val.is_finite() {
            (val - max).exp()
        } else {
            0.0
        };
        exps.push(exp);
        sum += exp;
    }
    if sum == 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.into_iter().map(|v| v / sum).collect()
}

fn sample_from_probs(probs: &[f32]) -> Option<usize> {
    let dist = WeightedIndex::new(probs).ok()?;
    let mut rng = rand::rng();
    Some(dist.sample(&mut rng))
}

fn apply_no_repeat_ngram(logits: &mut [f32], history: &[u32], ngram_size: usize) {
    if ngram_size <= 1 || history.len() < ngram_size {
        return;
    }
    let prefix_len = ngram_size - 1;
    let prefix_start = history.len() - prefix_len;
    let prefix = &history[prefix_start..];
    let mut banned = HashSet::new();
    for i in 0..=history.len() - ngram_size {
        if history[i..i + prefix_len] == *prefix {
            banned.insert(history[i + prefix_len]);
        }
    }
    for token in banned {
        let idx = token as usize;
        if idx < logits.len() {
            logits[idx] = f32::NEG_INFINITY;
        }
    }
}

fn expand_image_tokens(
    input_ids: &[u32],
    image_token_id: u32,
    image_token_counts: &[usize],
) -> Result<Vec<u32>, OCRError> {
    let mut out: Vec<u32> = Vec::new();
    let mut idx = 0usize;
    for &id in input_ids {
        if id == image_token_id {
            let count = image_token_counts
                .get(idx)
                .ok_or_else(|| OCRError::InvalidInput {
                    message: "MinerU2.5: image token count mismatch".to_string(),
                })?;
            out.extend(std::iter::repeat_n(image_token_id, *count));
            idx += 1;
        } else {
            out.push(id);
        }
    }
    if idx != image_token_counts.len() {
        return Err(OCRError::InvalidInput {
            message: "MinerU2.5: image token count mismatch".to_string(),
        });
    }
    Ok(out)
}

fn get_rope_index(
    cfg: &MinerUConfig,
    input_ids: &[u32],
    image_grid_thw: &[(usize, usize, usize)],
    vision_start_token_id: u32,
    video_token_id: u32,
    spatial_merge_size: usize,
    device: &Device,
) -> Result<(Tensor, i64), OCRError> {
    let image_token_id = cfg.image_token_id;
    let mut image_count = 0usize;
    for i in 0..input_ids.len().saturating_sub(1) {
        if input_ids[i] == vision_start_token_id && input_ids[i + 1] == image_token_id {
            image_count += 1;
        }
        if input_ids[i] == vision_start_token_id && input_ids[i + 1] == video_token_id {
            return Err(OCRError::InvalidInput {
                message: "MinerU2.5: video inputs are not supported".to_string(),
            });
        }
    }
    if image_count != image_grid_thw.len() {
        return Err(OCRError::InvalidInput {
            message: format!(
                "MinerU2.5: image count mismatch between prompt ({image_count}) and image_grid_thw ({})",
                image_grid_thw.len()
            ),
        });
    }

    let mut positions: Vec<[i64; 3]> = Vec::with_capacity(input_ids.len());
    let mut st = 0usize;
    let mut current_max: i64 = -1;

    for (image_index, &(t, h, w)) in image_grid_thw.iter().enumerate().take(image_count) {
        let ed = input_ids[st..]
            .iter()
            .position(|&id| id == image_token_id)
            .map(|p| st + p)
            .ok_or_else(|| OCRError::InvalidInput {
                message: format!(
                    "MinerU2.5: expected image token for image[{image_index}] but none found"
                ),
            })?;

        let st_idx = if current_max >= 0 { current_max + 1 } else { 0 };
        let text_len = ed - st;
        for i in 0..text_len {
            let p = st_idx + i as i64;
            positions.push([p, p, p]);
            current_max = current_max.max(p);
        }

        let llm_t = t as i64;
        let llm_h = (h / spatial_merge_size) as i64;
        let llm_w = (w / spatial_merge_size) as i64;
        let vision_base = st_idx + text_len as i64;

        for tt in 0..llm_t {
            for hh in 0..llm_h {
                for ww in 0..llm_w {
                    let t_pos = vision_base + tt;
                    let h_pos = vision_base + hh;
                    let w_pos = vision_base + ww;
                    positions.push([t_pos, h_pos, w_pos]);
                    current_max = current_max.max(t_pos).max(h_pos).max(w_pos);
                }
            }
        }

        st = ed + (llm_t as usize) * (llm_h as usize) * (llm_w as usize);
    }

    let st_idx = if current_max >= 0 { current_max + 1 } else { 0 };
    for i in st..input_ids.len() {
        let p = st_idx + (i - st) as i64;
        positions.push([p, p, p]);
        current_max = p;
    }

    if positions.len() != input_ids.len() {
        return Err(OCRError::InvalidInput {
            message: format!(
                "MinerU2.5: rope position ids length mismatch: got {}, expected {}",
                positions.len(),
                input_ids.len()
            ),
        });
    }

    let mut pos_ids: Vec<i64> = vec![0; 3 * input_ids.len()];
    let len = input_ids.len();
    for (i, v) in positions.iter().enumerate() {
        pos_ids[i] = v[0];
        pos_ids[len + i] = v[1];
        pos_ids[2 * len + i] = v[2];
    }

    let rope_delta = (current_max + 1) - (input_ids.len() as i64);

    let position_ids = Tensor::from_vec(pos_ids, (3usize, 1usize, input_ids.len()), device)
        .map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "MinerU2.5: build position_ids tensor failed",
                e,
            )
        })?;

    Ok((position_ids, rope_delta))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mineru_task_prompt_text_recognition_matches_official() {
        assert_eq!(MinerUTaskPrompt::Text.prompt(), "\nText Recognition:");
    }

    #[test]
    fn mineru_task_prompt_formula_recognition_matches_official() {
        assert_eq!(MinerUTaskPrompt::Formula.prompt(), "\nFormula Recognition:");
    }

    #[test]
    fn mineru_task_prompt_table_recognition_matches_official() {
        assert_eq!(MinerUTaskPrompt::Table.prompt(), "\nTable Recognition:");
    }

    #[test]
    fn mineru_task_prompt_image_analysis_matches_official() {
        assert_eq!(
            MinerUTaskPrompt::ImageAnalysis.prompt(),
            "\nImage Analysis:"
        );
    }

    #[test]
    fn mineru_task_prompt_layout_detection_matches_official() {
        assert_eq!(
            MinerUTaskPrompt::LayoutDetection.prompt(),
            "\nLayout Detection:"
        );
    }

    #[test]
    fn for_layout_routes_table_kinds_to_table() {
        assert_eq!(
            MinerUTaskPrompt::for_layout(LayoutElementType::Table),
            MinerUTaskPrompt::Table
        );
    }

    #[test]
    fn for_layout_routes_formula_kinds_to_formula() {
        assert_eq!(
            MinerUTaskPrompt::for_layout(LayoutElementType::Formula),
            MinerUTaskPrompt::Formula
        );
        assert_eq!(
            MinerUTaskPrompt::for_layout(LayoutElementType::FormulaNumber),
            MinerUTaskPrompt::Formula
        );
    }

    #[test]
    fn for_layout_routes_visual_kinds_to_image_analysis() {
        for ty in [
            LayoutElementType::Image,
            LayoutElementType::Chart,
            LayoutElementType::Seal,
            LayoutElementType::HeaderImage,
            LayoutElementType::FooterImage,
        ] {
            assert_eq!(
                MinerUTaskPrompt::for_layout(ty),
                MinerUTaskPrompt::ImageAnalysis,
                "expected ImageAnalysis for {ty:?}",
            );
        }
    }

    #[test]
    fn for_layout_defaults_text_for_text_like_kinds() {
        for ty in [
            LayoutElementType::Text,
            LayoutElementType::Content,
            LayoutElementType::DocTitle,
            LayoutElementType::ParagraphTitle,
            LayoutElementType::List,
            LayoutElementType::Reference,
            LayoutElementType::Footnote,
            LayoutElementType::Number,
            LayoutElementType::Header,
            LayoutElementType::Footer,
        ] {
            assert_eq!(
                MinerUTaskPrompt::for_layout(ty),
                MinerUTaskPrompt::Text,
                "expected Text for {ty:?}",
            );
        }
    }
}
