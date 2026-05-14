//! PaddleOCR-VL (Vision-Language) model implementation.

use super::config::{PaddleOcrVlConfig, PaddleOcrVlImageProcessorConfig};
use super::ernie::Ernie4_5Model;
use super::processing;
use super::projector::Projector;
use super::vision::VisionModel;
#[cfg(feature = "hsd")]
use crate::attention::create_tree_attention_mask;
use crate::attention::{combine_masks, create_causal_mask, create_left_padding_mask};
#[cfg(feature = "hsd")]
use crate::hsd::backend_util::{commit_keep_indices, step_pos_ids, tree_pos_ids};
#[cfg(feature = "hsd")]
use crate::hsd::drafting::{
    TargetDraftAdapter, bbox_xyxy, crop_region_image, format_verified_region, map_layout_kind,
    region_markdown_for, structure_result_to_layout_elements,
};
#[cfg(feature = "hsd")]
use crate::hsd::prefix_tree::PrefixTree;
#[cfg(feature = "hsd")]
use crate::hsd::types::{AcceptStats, Draft, HsdConfig, HsdStats, RegionStageStats};
#[cfg(feature = "hsd")]
use crate::hsd::verify::{SpecBackend, spec_decode};
use crate::utils::image::pil_resample_to_filter_type;
use crate::utils::{candle_to_ocr_inference, candle_to_ocr_processing};
#[cfg(feature = "hsd")]
use candle_core::Result as CandleResult;
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::Module;
#[cfg(feature = "hsd")]
use candle_nn::ops as cnn_ops;
use image::{RgbImage, imageops::FilterType};
use oar_ocr_core::core::OCRError;
#[cfg(feature = "hsd")]
use oar_ocr_core::domain::structure::{LayoutElement, LayoutElementType, StructureResult};
use std::path::Path;
#[cfg(feature = "hsd")]
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddleOcrVlTask {
    Ocr,
    Table,
    Chart,
    Formula,
    Spotting,
    Seal,
}

impl PaddleOcrVlTask {
    pub fn prompt(self) -> &'static str {
        match self {
            Self::Ocr => "OCR:",
            Self::Table => "Table Recognition:",
            Self::Chart => "Chart Recognition:",
            Self::Formula => "Formula Recognition:",
            Self::Spotting => "Spotting:",
            Self::Seal => "Seal Recognition:",
        }
    }

    pub fn postprocess(self, text: String) -> String {
        let trimmed = text.trim();
        match self {
            Self::Formula => processing::strip_math_wrappers(trimmed).to_string(),
            Self::Table => processing::postprocess_table_output(trimmed),
            Self::Ocr | Self::Chart | Self::Spotting | Self::Seal => trimmed.to_string(),
        }
    }

    fn needs_spotting_preprocess(self) -> bool {
        matches!(self, Self::Spotting)
    }
}

const SPOTTING_UPSCALE_THRESHOLD: u32 = 1500;
const SPOTTING_MAX_LONG_SIDE: u32 = 2048;

pub struct PaddleOcrVl {
    device: Device,
    dtype: DType,
    cfg: PaddleOcrVlConfig,
    image_cfg: PaddleOcrVlImageProcessorConfig,
    tokenizer: Tokenizer,
    llm: Ernie4_5Model,
    lm_head: candle_nn::Linear,
    vision: VisionModel,
    projector: Projector,
    eos_token_id: u32,
    sep_token_id: Option<u32>,
    /// Cached tokenized image placeholder (single token)
    image_placeholder_token_id: u32,
    /// Assistant prefix derived from chat template (e.g., "Assistant: " vs "Assistant:\n")
    assistant_prefix: String,
}

impl PaddleOcrVl {
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, OCRError> {
        let model_dir = model_dir.as_ref();
        let cfg = PaddleOcrVlConfig::from_path(model_dir.join("config.json"))?;
        let image_cfg =
            PaddleOcrVlImageProcessorConfig::from_path(model_dir.join("preprocessor_config.json"))?;

        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| {
            OCRError::ConfigError {
                message: format!("failed to load PaddleOCR-VL tokenizer.json: {e}"),
            }
        })?;

        let eos_token_id = tokenizer
            .token_to_id("</s>")
            .ok_or_else(|| OCRError::ConfigError {
                message: "PaddleOCR-VL: tokenizer is missing </s> token".to_string(),
            })?;
        let sep_token_id = tokenizer.token_to_id("<|end_of_sentence|>");

        let assistant_prefix = if std::fs::read_to_string(model_dir.join("chat_template.jinja"))
            .map(|t| t.contains("Assistant:\\n"))
            .unwrap_or(false)
        {
            "Assistant:\n".to_string()
        } else {
            "Assistant: ".to_string()
        };

        // Pre-tokenize image placeholder to avoid repeated string allocation
        let image_placeholder_token_id = tokenizer
            .token_to_id("<|IMAGE_PLACEHOLDER|>")
            .ok_or_else(|| OCRError::ConfigError {
                message: "PaddleOCR-VL: tokenizer is missing <|IMAGE_PLACEHOLDER|> token"
                    .to_string(),
            })?;

        let dtype = device.bf16_default_to_f32();
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[model_dir.join("model.safetensors")],
                dtype,
                &device,
            )
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load model.safetensors", e))?
        };

        let llm = Ernie4_5Model::load(&cfg, vb.pp("model"))?;
        let lm_head = candle_nn::linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load lm_head", e))?;

        let vision = VisionModel::load(&cfg.vision_config, vb.pp("visual").pp("vision_model"))?;
        let projector = Projector::load(&cfg, &cfg.vision_config, vb.pp("mlp_AR"))?;

        Ok(Self {
            device,
            dtype,
            cfg,
            image_cfg,
            tokenizer,
            llm,
            lm_head,
            vision,
            projector,
            eos_token_id,
            sep_token_id,
            image_placeholder_token_id,
            assistant_prefix,
        })
    }

    /// Generate OCR output for one or more images.
    ///
    /// Supports true GPU batching when multiple images are provided.
    ///
    /// # Arguments
    /// * `images` - Input images
    /// * `tasks` - Task for each image (must match images length)
    /// * `max_new_tokens` - Maximum tokens to generate per image
    ///
    /// # Returns
    /// Vector of results, one per input image.
    pub fn generate(
        &self,
        images: &[RgbImage],
        tasks: &[PaddleOcrVlTask],
        max_new_tokens: usize,
    ) -> Vec<Result<String, OCRError>> {
        self.generate_with_raw(images, tasks, max_new_tokens)
            .into_iter()
            .map(|r| r.map(|(_, processed)| processed))
            .collect()
    }

    /// Generate with both raw and postprocessed output.
    pub fn generate_with_raw(
        &self,
        images: &[RgbImage],
        tasks: &[PaddleOcrVlTask],
        max_new_tokens: usize,
    ) -> Vec<Result<(String, String), OCRError>> {
        if images.is_empty() {
            return Vec::new();
        }
        if images.len() != tasks.len() {
            return vec![Err(OCRError::InvalidInput {
                message: format!(
                    "PaddleOCR-VL: images count ({}) != tasks count ({})",
                    images.len(),
                    tasks.len()
                ),
            })];
        }

        match self.generate_tokens_internal(images, tasks, max_new_tokens) {
            Ok(results) => results
                .into_iter()
                .enumerate()
                .map(|(i, tokens)| self.decode_generated_tokens(&tokens, tasks[i]))
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
    /// excluding EOS / separator stop tokens, before tokenizer decoding or
    /// task postprocessing.
    pub fn generate_tokens(
        &self,
        images: &[RgbImage],
        tasks: &[PaddleOcrVlTask],
        max_new_tokens: usize,
    ) -> Vec<Result<Vec<u32>, OCRError>> {
        if images.is_empty() {
            return Vec::new();
        }
        if images.len() != tasks.len() {
            return vec![Err(OCRError::InvalidInput {
                message: format!(
                    "PaddleOCR-VL: images count ({}) != tasks count ({})",
                    images.len(),
                    tasks.len()
                ),
            })];
        }

        match self.generate_tokens_internal(images, tasks, max_new_tokens) {
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

    /// Internal generation implementation supporting batched inference.
    fn generate_tokens_internal(
        &self,
        images: &[RgbImage],
        tasks: &[PaddleOcrVlTask],
        max_new_tokens: usize,
    ) -> Result<Vec<Vec<u32>>, OCRError> {
        let batch_size = images.len();

        // 1. Preprocess all images
        let needs_spotting = tasks.iter().any(|t| t.needs_spotting_preprocess());
        let resized_images;
        let images_for_preprocess = if needs_spotting {
            let resize_filter = self
                .image_cfg
                .resample
                .and_then(pil_resample_to_filter_type)
                .unwrap_or(FilterType::CatmullRom);
            let mut processed = Vec::with_capacity(images.len());
            for (img, task) in images.iter().zip(tasks.iter()) {
                if task.needs_spotting_preprocess()
                    && img.width() < SPOTTING_UPSCALE_THRESHOLD
                    && img.height() < SPOTTING_UPSCALE_THRESHOLD
                {
                    processed.push(image::imageops::resize(
                        img,
                        img.width().saturating_mul(2),
                        img.height().saturating_mul(2),
                        resize_filter,
                    ));
                } else {
                    processed.push(img.clone());
                }
            }
            resized_images = Some(processed);
            resized_images.as_ref().unwrap().as_slice()
        } else {
            images
        };
        let max_pixels = if needs_spotting {
            let factor = (self.image_cfg.patch_size * self.image_cfg.merge_size) as u32;
            let spotting_max_pixels = SPOTTING_MAX_LONG_SIDE
                .saturating_mul(factor)
                .saturating_mul(factor);
            self.image_cfg.max_pixels.max(spotting_max_pixels)
        } else {
            self.image_cfg.max_pixels
        };
        let image_inputs = processing::preprocess_images_with_max_pixels(
            images_for_preprocess,
            &self.image_cfg,
            &self.device,
            self.dtype,
            max_pixels,
        )?;

        // 2. Build prompts for each sample
        let mut all_input_ids: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
        let mut all_image_token_counts: Vec<usize> = Vec::with_capacity(batch_size);

        for (i, task) in tasks.iter().enumerate() {
            let (t, h, w) = image_inputs.image_grid_thw[i];
            let image_token_count =
                (t * h * w) / (self.image_cfg.merge_size * self.image_cfg.merge_size);
            all_image_token_counts.push(image_token_count);

            let prefix = "<|begin_of_sentence|>User: <|IMAGE_START|>";
            let suffix = format!("<|IMAGE_END|>{}\n{}", task.prompt(), self.assistant_prefix);

            let prefix_enc =
                self.tokenizer
                    .encode(prefix, false)
                    .map_err(|e| OCRError::InvalidInput {
                        message: format!("tokenizer encode failed: {e}"),
                    })?;
            let suffix_enc = self.tokenizer.encode(suffix.as_str(), false).map_err(|e| {
                OCRError::InvalidInput {
                    message: format!("tokenizer encode failed: {e}"),
                }
            })?;

            let mut input_ids =
                Vec::with_capacity(prefix_enc.len() + image_token_count + suffix_enc.len());
            input_ids.extend_from_slice(prefix_enc.get_ids());
            input_ids.extend(std::iter::repeat_n(
                self.image_placeholder_token_id,
                image_token_count,
            ));
            input_ids.extend_from_slice(suffix_enc.get_ids());
            all_input_ids.push(input_ids);
        }

        // 3. Compute vision features
        let vision_feats = self
            .vision
            .forward(&image_inputs.pixel_values, &image_inputs.image_grid_thw)?;
        let image_embeds_all = self
            .projector
            .forward(&vision_feats, &image_inputs.image_grid_thw)?;

        // 4. Build embeddings per sample
        let seq_lens: Vec<usize> = all_input_ids.iter().map(|ids| ids.len()).collect();
        let Some(&max_seq_len) = seq_lens.iter().max() else {
            return Err(OCRError::InvalidInput {
                message: "PaddleOCR-VL: empty batch is not supported".to_string(),
            });
        };

        let mut batch_embeds: Vec<Tensor> = Vec::with_capacity(batch_size);
        let mut rope_deltas: Vec<i64> = Vec::with_capacity(batch_size);
        let mut batch_position_ids: Vec<Tensor> = Vec::with_capacity(batch_size);
        let mut embed_offset = 0usize;

        for (i, input_ids) in all_input_ids.iter().enumerate() {
            let seq_len = input_ids.len();
            let pad_len = max_seq_len - seq_len;
            let image_token_count = all_image_token_counts[i];

            // Get image embeddings for this sample
            let image_embeds = image_embeds_all
                .narrow(0, embed_offset, image_token_count)
                .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "narrow image embeds", e))?;
            embed_offset += image_token_count;

            // Embed tokens
            let input_ids_t = Tensor::new(input_ids.clone(), &self.device)
                .and_then(|t| t.reshape((1, seq_len)))
                .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "create input_ids", e))?;
            let mut inputs_embeds = self.llm.embed(&input_ids_t)?;

            // Fuse image embeddings
            let first_img_pos = input_ids
                .iter()
                .position(|&id| id == self.cfg.image_token_id);
            if let Some(first_pos) = first_img_pos {
                let image_end = first_pos + image_token_count;
                let mut parts: Vec<Tensor> = Vec::with_capacity(3);

                if first_pos > 0 {
                    parts.push(inputs_embeds.narrow(1, 0, first_pos).map_err(|e| {
                        candle_to_ocr_inference("PaddleOCR-VL", "narrow prefix", e)
                    })?);
                }
                parts.push(
                    image_embeds
                        .unsqueeze(0)
                        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "unsqueeze img", e))?,
                );
                if image_end < seq_len {
                    parts.push(
                        inputs_embeds
                            .narrow(1, image_end, seq_len - image_end)
                            .map_err(|e| {
                                candle_to_ocr_inference("PaddleOCR-VL", "narrow suffix", e)
                            })?,
                    );
                }

                let refs: Vec<&Tensor> = parts.iter().collect();
                inputs_embeds = Tensor::cat(&refs, 1)
                    .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "cat embeds", e))?;
            }

            // Left-pad if needed
            if pad_len > 0 {
                let pad = Tensor::zeros(
                    (1, pad_len, self.cfg.hidden_size),
                    inputs_embeds.dtype(),
                    &self.device,
                )
                .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "create pad", e))?;
                inputs_embeds = Tensor::cat(&[&pad, &inputs_embeds], 1)
                    .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "cat pad", e))?;
            }
            batch_embeds.push(inputs_embeds);

            // Position IDs
            let (pos_ids, delta) = get_rope_index(
                &self.cfg,
                input_ids,
                &[image_inputs.image_grid_thw[i]],
                &self.device,
            )?;
            rope_deltas.push(delta);

            let pos_ids = if pad_len > 0 {
                let pad_pos = Tensor::zeros((3, 1, pad_len), DType::I64, &self.device)
                    .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "create pad pos", e))?;
                Tensor::cat(&[&pad_pos, &pos_ids], 2)
                    .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "cat pad pos", e))?
            } else {
                pos_ids
            };
            batch_position_ids.push(pos_ids);
        }

        // 5. Stack batched tensors
        let batch_refs: Vec<&Tensor> = batch_embeds.iter().collect();
        let inputs_embeds = Tensor::cat(&batch_refs, 0)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "stack embeds", e))?;

        let pos_refs: Vec<&Tensor> = batch_position_ids.iter().collect();
        let position_ids = Tensor::cat(&pos_refs, 1)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "stack pos", e))?;

        // 6. Create attention mask
        let causal = create_causal_mask(max_seq_len, max_seq_len, self.dtype, &self.device)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "create causal", e))?;
        let padding = create_left_padding_mask(&seq_lens, max_seq_len, self.dtype, &self.device)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "create padding", e))?;
        let mask = combine_masks(&causal, &padding)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "combine masks", e))?;

        // 7. Prefill
        self.llm.clear_kv_cache();
        let hidden = self
            .llm
            .forward(&inputs_embeds, &position_ids, Some(&mask))?;

        // 8. Get initial logits
        let mut logits_list: Vec<Tensor> = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let last = hidden
                .i((i, max_seq_len - 1, ..))
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "get last hidden", e))?;
            let logits = self
                .lm_head
                .forward(&last)
                .and_then(|t| t.squeeze(0))
                .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "lm_head", e))?;
            logits_list.push(logits);
        }

        // 9. Autoregressive decode
        let mut generated: Vec<Vec<u32>> = vec![Vec::new(); batch_size];
        let mut finished: Vec<bool> = vec![false; batch_size];
        let mut positions: Vec<i64> = seq_lens
            .iter()
            .zip(&rope_deltas)
            .map(|(&len, &d)| (len as i64) + d)
            .collect();

        for _ in 0..max_new_tokens {
            if finished.iter().all(|&f| f) {
                break;
            }

            let mut next_tokens: Vec<u32> = Vec::with_capacity(batch_size);
            for (i, logits) in logits_list.iter().enumerate() {
                if finished[i] {
                    next_tokens.push(0);
                } else {
                    let tok = logits
                        .argmax(D::Minus1)
                        .and_then(|t| t.to_scalar::<u32>())
                        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "argmax", e))?;

                    if tok == self.eos_token_id || self.sep_token_id.is_some_and(|id| id == tok) {
                        finished[i] = true;
                    } else {
                        generated[i].push(tok);
                    }
                    next_tokens.push(tok);
                }
            }

            if finished.iter().all(|&f| f) {
                break;
            }

            // Batch forward
            let tokens = Tensor::new(next_tokens, &self.device)
                .and_then(|t| t.reshape((batch_size, 1)))
                .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "create tokens", e))?;
            let embeds = self.llm.embed(&tokens)?;

            let pos_data: Vec<i64> = positions.iter().flat_map(|&p| [p, p, p]).collect();
            let pos = Tensor::new(pos_data, &self.device)
                .and_then(|t| t.reshape((3, batch_size, 1)))
                .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "create pos", e))?;

            let hs = self.llm.forward(&embeds, &pos, None)?;

            logits_list.clear();
            for i in 0..batch_size {
                let h = hs
                    .i((i, 0, ..))
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "get hs", e))?;
                let logits = self
                    .lm_head
                    .forward(&h)
                    .and_then(|t| t.squeeze(0))
                    .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "lm_head step", e))?;
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

    pub fn decode_tokens(
        &self,
        tokens: &[u32],
        task: PaddleOcrVlTask,
    ) -> Result<(String, String), OCRError> {
        self.decode_generated_tokens(tokens, task)
    }

    /// Decode tokens **without** applying PaddleOCR-VL's task-specific
    /// post-process (OTSL→HTML for tables, `$$..$$` stripping for formulas).
    /// This is the raw pre-postprocess string the model actually emitted —
    /// use this when feeding PaddleOCR-VL output as a draft for another
    /// target VLM. DSV matches at token granularity, so any post-process on
    /// the source side will byte-mismatch the target's natural output.
    pub fn decode_tokens_raw(&self, tokens: &[u32]) -> Result<String, OCRError> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| OCRError::InvalidInput {
                message: format!("decode failed: {e}"),
            })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn decode_generated_tokens(
        &self,
        tokens: &[u32],
        task: PaddleOcrVlTask,
    ) -> Result<(String, String), OCRError> {
        let decoded = self.decode_tokens_raw(tokens)?;
        let processed = task.postprocess(decoded.clone());
        Ok((decoded, processed))
    }

    /// Hierarchical Speculative Decoding entry for a single image / region.
    ///
    /// PaddleOCR-VL is naturally per-region (no full-page mode), so the only
    /// HSD stage that applies is region-level. `task` selects the prompt
    /// prefix (Ocr / Table / Formula / …); `drafts` are the lightweight
    /// pipeline drafter's region-text candidates, tokenized with PaddleOCR-VL's
    /// own tokenizer before being matched in the verifier.
    #[cfg(feature = "hsd")]
    pub fn generate_hsd(
        &self,
        image: &RgbImage,
        task: PaddleOcrVlTask,
        drafts: &[String],
        hsd_cfg: &HsdConfig,
    ) -> Result<(String, HsdStats), OCRError> {
        let t_drafter = Instant::now();

        let mut tokenized: Vec<Draft> = Vec::with_capacity(drafts.len());
        for d in drafts {
            if d.trim().is_empty() {
                continue;
            }
            let enc =
                self.tokenizer
                    .encode(d.as_str(), false)
                    .map_err(|e| OCRError::InvalidInput {
                        message: format!("PaddleOCR-VL HSD: tokenizer encode failed: {e}"),
                    })?;
            let tokens = enc.get_ids().to_vec();
            if !tokens.is_empty() {
                tokenized.push(Draft::new(tokens));
            }
        }
        self.generate_hsd_tokenized(image, task, &tokenized, hsd_cfg, t_drafter.elapsed())
    }

    /// HSD entry that consumes already-tokenized drafts. This is the oracle
    /// path used by benchmarks to avoid `decode -> encode` tokenizer
    /// round-trips when the draft comes from this backend's own baseline.
    #[cfg(feature = "hsd")]
    pub fn generate_hsd_with_token_drafts(
        &self,
        image: &RgbImage,
        task: PaddleOcrVlTask,
        drafts: &[Draft],
        hsd_cfg: &HsdConfig,
    ) -> Result<(String, HsdStats), OCRError> {
        self.generate_hsd_tokenized(image, task, drafts, hsd_cfg, Duration::ZERO)
    }

    #[cfg(feature = "hsd")]
    fn generate_hsd_tokenized(
        &self,
        image: &RgbImage,
        task: PaddleOcrVlTask,
        tokenized: &[Draft],
        hsd_cfg: &HsdConfig,
        drafter_elapsed: Duration,
    ) -> Result<(String, HsdStats), OCRError> {
        if !self.device.is_cuda() {
            return Err(OCRError::ConfigError {
                message: "HSD requires CUDA device".to_string(),
            });
        }

        let mut stats = HsdStats {
            drafter: drafter_elapsed,
            ..Default::default()
        };
        // Stage 2 (page-level) terminology is reused here for stat bookkeeping.
        let t_pre = Instant::now();
        let (initial_lp, rope_delta) = self.hsd_prefill_single(image, task)?;
        stats.stage2.vision_prefill = t_pre.elapsed();
        stats.stage2.forward_passes = 1;

        let t_dec = Instant::now();
        let mut backend = PaddleOcrVlSpecBackend::new(self, rope_delta);
        let mut accept = AcceptStats::default();
        let mut dsv = Default::default();
        let generated = spec_decode(
            &mut backend,
            tokenized,
            initial_lp,
            hsd_cfg.max_region_tokens,
            &hsd_cfg.dsv,
            &mut accept,
            &mut dsv,
        )
        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "spec_decode", e))?;
        stats.stage2.decode = t_dec.elapsed();
        stats.stage2.emitted_tokens = generated.len() as u32;
        stats.stage2.accept = accept;
        stats.stage2.dsv = dsv;
        stats.stage2.forward_passes += backend.forward_passes;

        // Strip the first stop token and anything after it before decoding.
        let stop_pos = generated
            .iter()
            .position(|&t| t == self.eos_token_id || self.sep_token_id.is_some_and(|id| id == t))
            .unwrap_or(generated.len());
        let truncated = &generated[..stop_pos];

        let decoded =
            self.tokenizer
                .decode(truncated, true)
                .map_err(|e| OCRError::InvalidInput {
                    message: format!("PaddleOCR-VL HSD: tokenizer decode failed: {e}"),
                })?;
        let postprocessed = task.postprocess(decoded);
        Ok((postprocessed, stats))
    }

    /// Run HSD per element across an entire layout-detected page, then
    /// aggregate the per-region outputs into a markdown-style document.
    ///
    /// **No Stage 2 page-level verify** — PaddleOCR-VL is element-only by
    /// design (its prompts are task-scoped: "OCR:", "Table Recognition:",
    /// etc.), so there is no native single-prompt page-level inference to
    /// verify against. The HunyuanOCR / GLM-OCR / MinerU `generate_hsd_full`
    /// paths *do* run Stage 2 because their target prompts can describe a
    /// whole page. For full HSD over a PaddleOCR-VL document use the
    /// `doc_parser` flow (layout + per-region HSD) — what this function
    /// already does.
    ///
    /// Element-type → task mapping follows the same heuristic as
    /// `doc_parser::DocParser` (Table → Table, Formula → Formula, etc.).
    /// Visual-only regions (Image / Seal / HeaderImage / FooterImage) are
    /// skipped, mirroring the `task_for_element_type` logic.
    #[cfg(feature = "hsd")]
    pub fn generate_hsd_full(
        &self,
        image: &RgbImage,
        elements: &[LayoutElement],
        ignore_labels: &[String],
        hsd_cfg: &HsdConfig,
    ) -> Result<(String, HsdStats), OCRError> {
        self.generate_hsd_full_impl(image, elements, ignore_labels, hsd_cfg, None)
    }

    /// One-call HSD entry that consumes a `StructureResult` (the output of
    /// the OARStructure / PP-StructureV3 pipeline) directly.
    ///
    /// Backfills table HTML / formula LaTeX via
    /// [`structure_result_to_layout_elements`] then delegates to
    /// [`Self::generate_hsd_full`]. PaddleOCR-VL remains element-level (no
    /// Stage 2) — `page_instruction` / `region_instruction` are not part of
    /// this signature because the per-element prompt is picked from the
    /// layout type by `task_for_element_type` inside `generate_hsd_full_impl`.
    #[cfg(feature = "hsd")]
    pub fn generate_hsd_with_structure(
        &self,
        image: &RgbImage,
        structure: &StructureResult,
        ignore_labels: &[String],
        hsd_cfg: &HsdConfig,
    ) -> Result<(String, HsdStats), OCRError> {
        let elements = structure_result_to_layout_elements(structure);
        self.generate_hsd_full(image, &elements, ignore_labels, hsd_cfg)
    }

    /// Full per-region HSD where some regions already have target-token drafts.
    /// `token_drafts[i]`, when present, is used for `elements[i]` directly and
    /// avoids re-tokenizing decoded baseline text.
    #[cfg(feature = "hsd")]
    pub fn generate_hsd_full_with_token_drafts(
        &self,
        image: &RgbImage,
        elements: &[LayoutElement],
        ignore_labels: &[String],
        token_drafts: &[Option<Vec<u32>>],
        hsd_cfg: &HsdConfig,
    ) -> Result<(String, HsdStats), OCRError> {
        if token_drafts.len() != elements.len() {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "PaddleOCR-VL HSD: token draft count ({}) != element count ({})",
                    token_drafts.len(),
                    elements.len()
                ),
            });
        }
        self.generate_hsd_full_impl(image, elements, ignore_labels, hsd_cfg, Some(token_drafts))
    }

    #[cfg(feature = "hsd")]
    fn generate_hsd_full_impl(
        &self,
        image: &RgbImage,
        elements: &[LayoutElement],
        ignore_labels: &[String],
        hsd_cfg: &HsdConfig,
        token_drafts: Option<&[Option<Vec<u32>>]>,
    ) -> Result<(String, HsdStats), OCRError> {
        let mut stats = HsdStats::default();
        let mut region_md: Vec<(usize, String)> = Vec::with_capacity(elements.len());

        for (idx, elem) in elements.iter().enumerate() {
            if let Some(label) = &elem.label
                && ignore_labels.iter().any(|l| l == label)
            {
                continue;
            }
            let Some(task) = task_for_layout_type(elem.element_type) else {
                continue;
            };
            let token_draft = token_drafts
                .and_then(|drafts| drafts.get(idx))
                .and_then(|d| d.as_ref())
                .filter(|d| !d.is_empty());
            // Format `elem.text` into PaddleOCR-VL's raw (pre-postprocess) form:
            // tables stay OTSL, formulas get `$$..$$` wrapping (post-process
            // strips it), other elements stay plain. This avoids the trap
            // where a layout pipeline's HTML/markdown is byte-incompatible
            // with what the VLM actually emits as logits.
            let text_draft = region_markdown_for(elem, TargetDraftAdapter::PaddleOcrVl);
            let text_draft = text_draft.trim();
            let text_draft = if text_draft.is_empty() {
                None
            } else {
                Some(text_draft)
            };
            if token_draft.is_none() && text_draft.is_none() {
                continue;
            }

            let bbox = bbox_xyxy(&elem.bbox);
            let crop = crop_region_image(image, &bbox)?;
            let (region_text, region_stats) = if let Some(tokens) = token_draft {
                let drafts = vec![Draft::new(tokens.clone())];
                self.generate_hsd_with_token_drafts(&crop, task, &drafts, hsd_cfg)?
            } else {
                let drafts = vec![text_draft.expect("checked above").to_string()];
                self.generate_hsd(&crop, task, &drafts, hsd_cfg)?
            };
            // Accumulate stats (region-level passes are stored under stage1).
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

        region_md.sort_by_key(|(order, _)| *order);
        let merged = region_md
            .into_iter()
            .map(|(_, text)| text)
            .filter(|s| !s.trim().is_empty())
            .collect::<Vec<_>>()
            .join("\n\n");
        Ok((merged, stats))
    }

    /// Run a single-image prefill with the supplied task prompt. Returns
    /// the F32 last-position log-probabilities and the MRoPE delta that
    /// post-image text positions need to add to their token index.
    #[cfg(feature = "hsd")]
    fn hsd_prefill_single(
        &self,
        image: &RgbImage,
        task: PaddleOcrVlTask,
    ) -> Result<(Tensor, i64), OCRError> {
        // Optional spotting upscaling — mirror generate_internal's behaviour.
        let resized;
        let image_for_pp: &RgbImage = if task.needs_spotting_preprocess()
            && image.width() < SPOTTING_UPSCALE_THRESHOLD
            && image.height() < SPOTTING_UPSCALE_THRESHOLD
        {
            let resize_filter = self
                .image_cfg
                .resample
                .and_then(pil_resample_to_filter_type)
                .unwrap_or(FilterType::CatmullRom);
            resized = image::imageops::resize(
                image,
                image.width().saturating_mul(2),
                image.height().saturating_mul(2),
                resize_filter,
            );
            &resized
        } else {
            image
        };
        let max_pixels = if task.needs_spotting_preprocess() {
            let factor = (self.image_cfg.patch_size * self.image_cfg.merge_size) as u32;
            let spotting_max_pixels = SPOTTING_MAX_LONG_SIDE
                .saturating_mul(factor)
                .saturating_mul(factor);
            self.image_cfg.max_pixels.max(spotting_max_pixels)
        } else {
            self.image_cfg.max_pixels
        };

        let image_inputs = processing::preprocess_images_with_max_pixels(
            std::slice::from_ref(image_for_pp),
            &self.image_cfg,
            &self.device,
            self.dtype,
            max_pixels,
        )?;

        let (t, h, w) = image_inputs.image_grid_thw[0];
        let image_token_count =
            (t * h * w) / (self.image_cfg.merge_size * self.image_cfg.merge_size);

        // Build prompt: prefix + N×<|IMAGE_PLACEHOLDER|> + suffix.
        let prefix = "<|begin_of_sentence|>User: <|IMAGE_START|>";
        let suffix = format!("<|IMAGE_END|>{}\n{}", task.prompt(), self.assistant_prefix);
        let prefix_enc =
            self.tokenizer
                .encode(prefix, false)
                .map_err(|e| OCRError::InvalidInput {
                    message: format!("PaddleOCR-VL HSD: tokenizer encode failed: {e}"),
                })?;
        let suffix_enc =
            self.tokenizer
                .encode(suffix.as_str(), false)
                .map_err(|e| OCRError::InvalidInput {
                    message: format!("PaddleOCR-VL HSD: tokenizer encode failed: {e}"),
                })?;
        let mut input_ids =
            Vec::with_capacity(prefix_enc.len() + image_token_count + suffix_enc.len());
        input_ids.extend_from_slice(prefix_enc.get_ids());
        input_ids.extend(std::iter::repeat_n(
            self.image_placeholder_token_id,
            image_token_count,
        ));
        input_ids.extend_from_slice(suffix_enc.get_ids());
        let seq_len = input_ids.len();

        // Vision features → projector → image embeddings.
        let vision_feats = self
            .vision
            .forward(&image_inputs.pixel_values, &image_inputs.image_grid_thw)?;
        let image_embeds = self
            .projector
            .forward(&vision_feats, &image_inputs.image_grid_thw)?;

        // Build token embeddings, then splice in the image embeddings.
        let input_ids_t = Tensor::new(input_ids.clone(), &self.device)
            .and_then(|t| t.reshape((1, seq_len)))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "create input_ids", e))?;
        let mut inputs_embeds = self.llm.embed(&input_ids_t)?;

        if let Some(first_pos) = input_ids
            .iter()
            .position(|&id| id == self.cfg.image_token_id)
        {
            let image_end = first_pos + image_token_count;
            let mut parts: Vec<Tensor> = Vec::with_capacity(3);
            if first_pos > 0 {
                parts.push(
                    inputs_embeds
                        .narrow(1, 0, first_pos)
                        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "narrow prefix", e))?,
                );
            }
            parts.push(
                image_embeds
                    .unsqueeze(0)
                    .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "unsqueeze img", e))?,
            );
            if image_end < seq_len {
                parts.push(
                    inputs_embeds
                        .narrow(1, image_end, seq_len - image_end)
                        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "narrow suffix", e))?,
                );
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            inputs_embeds = Tensor::cat(&refs, 1)
                .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "cat embeds", e))?;
        }

        // Position IDs (3-axis MRoPE) + delta for post-image text positions.
        let (pos_ids, rope_delta) = get_rope_index(
            &self.cfg,
            &input_ids,
            &[image_inputs.image_grid_thw[0]],
            &self.device,
        )?;

        // Causal mask for prefill.
        let causal = create_causal_mask(seq_len, seq_len, self.dtype, &self.device)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "create causal", e))?;

        self.llm.clear_kv_cache();
        let hidden = self.llm.forward(&inputs_embeds, &pos_ids, Some(&causal))?;

        let last = hidden
            .i((0, seq_len - 1, ..))
            .and_then(|t| t.unsqueeze(0))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "get last hidden", e))?;
        // Project via lm_head → log-softmax in F32.
        let logits = self
            .lm_head
            .forward(&last)
            .and_then(|t| t.squeeze(0))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "lm_head prefill", e))?;
        let lp = cnn_ops::log_softmax(
            &logits
                .to_dtype(DType::F32)
                .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "logits to f32", e))?,
            D::Minus1,
        )
        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "log_softmax prefill", e))?;
        Ok((lp, rope_delta))
    }
}

/// Map a layout element type to the PaddleOCR-VL recognition task. Returns
/// `None` for pure-visual regions that have no textual content to verify.
#[cfg(feature = "hsd")]
fn task_for_layout_type(t: LayoutElementType) -> Option<PaddleOcrVlTask> {
    use LayoutElementType::*;
    match t {
        Table => Some(PaddleOcrVlTask::Table),
        Chart => Some(PaddleOcrVlTask::Chart),
        Formula => Some(PaddleOcrVlTask::Formula),
        FormulaNumber => Some(PaddleOcrVlTask::Ocr),
        Image | HeaderImage | FooterImage | Seal => None,
        _ => Some(PaddleOcrVlTask::Ocr),
    }
}

/// HSD adapter for PaddleOCR-VL. The MRoPE position offset (`rope_delta`)
/// is captured during prefill and re-used by every subsequent step / verify.
#[cfg(feature = "hsd")]
struct PaddleOcrVlSpecBackend<'a> {
    model: &'a PaddleOcrVl,
    rope_delta: i64,
    pre_verify_kv: usize,
    forward_passes: u32,
}

#[cfg(feature = "hsd")]
impl<'a> PaddleOcrVlSpecBackend<'a> {
    fn new(model: &'a PaddleOcrVl, rope_delta: i64) -> Self {
        Self {
            model,
            rope_delta,
            pre_verify_kv: 0,
            forward_passes: 0,
        }
    }

    fn project_logprobs_2d(&self, hidden_2d: &Tensor) -> CandleResult<Tensor> {
        // (N, hidden) → (N, vocab) → log-softmax F32.
        let logits = self.model.lm_head.forward(hidden_2d)?;
        cnn_ops::log_softmax(&logits.to_dtype(DType::F32)?, D::Minus1)
    }

    fn project_logprobs_1d(&self, hidden_1d: &Tensor) -> CandleResult<Tensor> {
        // (hidden,) → (1, hidden) → (1, vocab) → (vocab,) → log-softmax F32.
        let logits = self
            .model
            .lm_head
            .forward(&hidden_1d.unsqueeze(0)?)?
            .squeeze(0)?;
        cnn_ops::log_softmax(&logits.to_dtype(DType::F32)?, D::Minus1)
    }
}

#[cfg(feature = "hsd")]
impl<'a> SpecBackend for PaddleOcrVlSpecBackend<'a> {
    fn step_one(&mut self, token: u32) -> CandleResult<Tensor> {
        let model = self.model;
        let device = &model.device;

        let tok_t = Tensor::new(vec![token], device)?.reshape((1usize, 1usize))?;
        let embeds = model.llm.embed(&tok_t).map_err(|e| {
            candle_core::Error::Msg(format!("PaddleOCR-VL HSD step_one embed: {e}"))
        })?;

        let pos_ids = step_pos_ids(3, model.llm.current_kv_len(), self.rope_delta, device)?;

        let hidden = model.llm.forward(&embeds, &pos_ids, None).map_err(|e| {
            candle_core::Error::Msg(format!("PaddleOCR-VL HSD step_one forward: {e}"))
        })?;
        self.forward_passes += 1;
        let last = hidden.i((0, 0, ..))?;
        self.project_logprobs_1d(&last)
    }

    fn verify_tree(&mut self, tree: &PrefixTree) -> CandleResult<Tensor> {
        let n = tree.num_nodes();
        let model = self.model;
        let device = &model.device;
        let dtype = model.dtype;

        let prefix_kv = model.llm.current_kv_len();
        self.pre_verify_kv = prefix_kv;

        let tok_t = Tensor::new(tree.tokens.clone(), device)?.reshape((1usize, n))?;
        let embeds = model.llm.embed(&tok_t).map_err(|e| {
            candle_core::Error::Msg(format!("PaddleOCR-VL HSD verify_tree embed: {e}"))
        })?;

        let pos_ids = tree_pos_ids(3, prefix_kv, self.rope_delta, tree, device)?;
        let mask = create_tree_attention_mask(&tree.parents, prefix_kv, dtype, device)?;

        let hidden = model
            .llm
            .forward(&embeds, &pos_ids, Some(&mask))
            .map_err(|e| {
                candle_core::Error::Msg(format!("PaddleOCR-VL HSD verify_tree forward: {e}"))
            })?;
        self.forward_passes += 1;
        let h2 = hidden.squeeze(0)?;
        self.project_logprobs_2d(&h2)
    }

    fn commit_verify(&mut self, accepted_path: &[usize]) -> CandleResult<()> {
        let indices = commit_keep_indices(self.pre_verify_kv, accepted_path);
        self.model
            .llm
            .keep_kv_indices(&indices)
            .map_err(|e| candle_core::Error::Msg(format!("PaddleOCR-VL HSD commit_verify: {e}")))
    }

    fn is_eos(&self, tok: u32) -> bool {
        tok == self.model.eos_token_id || self.model.sep_token_id.is_some_and(|id| id == tok)
    }
}

fn get_rope_index(
    cfg: &PaddleOcrVlConfig,
    input_ids: &[u32],
    image_grid_thw: &[(usize, usize, usize)],
    device: &Device,
) -> Result<(Tensor, i64), OCRError> {
    let spatial_merge_size = cfg.vision_config.spatial_merge_size;
    if spatial_merge_size == 0 {
        return Err(OCRError::ConfigError {
            message: "PaddleOCR-VL: vision_config.spatial_merge_size must be > 0".to_string(),
        });
    }

    let mut image_count = 0usize;
    for i in 0..input_ids.len().saturating_sub(1) {
        if input_ids[i] == cfg.vision_start_token_id && input_ids[i + 1] == cfg.image_token_id {
            image_count += 1;
        }
        if input_ids[i] == cfg.vision_start_token_id && input_ids[i + 1] == cfg.video_token_id {
            return Err(OCRError::InvalidInput {
                message: "PaddleOCR-VL: video inputs are not supported".to_string(),
            });
        }
    }
    if image_count != image_grid_thw.len() {
        return Err(OCRError::InvalidInput {
            message: format!(
                "PaddleOCR-VL: image count mismatch between prompt ({image_count}) and image_grid_thw ({})",
                image_grid_thw.len()
            ),
        });
    }

    let mut positions: Vec<[i64; 3]> = Vec::with_capacity(input_ids.len());
    let mut st = 0usize;
    let mut current_max: i64 = -1;

    for (image_idx, &(t, h, w)) in image_grid_thw.iter().enumerate() {
        let ed = input_ids[st..]
            .iter()
            .position(|&id| id == cfg.image_token_id)
            .map(|p| st + p)
            .ok_or_else(|| OCRError::InvalidInput {
                message: format!(
                    "PaddleOCR-VL: expected image token for image[{image_idx}] but none found"
                ),
            })?;

        let st_idx = current_max + 1;
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

        for _tt in 0..llm_t {
            for hh in 0..llm_h {
                for ww in 0..llm_w {
                    let t_pos = vision_base;
                    let h_pos = vision_base + hh;
                    let w_pos = vision_base + ww;
                    positions.push([t_pos, h_pos, w_pos]);
                    current_max = current_max.max(t_pos).max(h_pos).max(w_pos);
                }
            }
        }

        st = ed + (llm_t as usize) * (llm_h as usize) * (llm_w as usize);
    }

    let st_idx = current_max + 1;
    for i in st..input_ids.len() {
        let p = st_idx + (i - st) as i64;
        positions.push([p, p, p]);
        current_max = p;
    }

    if positions.len() != input_ids.len() {
        return Err(OCRError::InvalidInput {
            message: format!(
                "PaddleOCR-VL: rope position ids length mismatch: got {}, expected {}",
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

    let rope_deltas = (current_max + 1) - (input_ids.len() as i64);

    let position_ids = Tensor::from_vec(pos_ids, (3usize, 1usize, input_ids.len()), device)
        .map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: build position_ids tensor failed",
                e,
            )
        })?;

    Ok((position_ids, rope_deltas))
}
