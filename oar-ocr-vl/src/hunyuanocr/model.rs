//! HunyuanOCR model wrapper (HunYuanVLForConditionalGeneration).

use super::config::{HunyuanOcrConfig, HunyuanOcrImageProcessorConfig};
use super::llm::HunyuanLlm;
use super::processing::{HunyuanOcrImageInputs, preprocess_image};
use super::vision::HunyuanVisionModel;
use crate::attention::{
    combine_masks, create_causal_mask, create_left_padding_mask, create_tree_attention_mask,
};
use crate::hsd::drafting::{build_region_drafts, crop_region_image, page_markdown};
use crate::hsd::prefix_tree::PrefixTree;
use crate::hsd::types::{AcceptStats, Draft, HsdConfig, HsdStats, RegionDraft, StageStats};
use crate::hsd::verify::{SpecBackend, spec_decode};
use crate::utils::{candle_to_ocr_inference, candle_to_ocr_processing};
use candle_core::{D, DType, Device, IndexOp, Result as CandleResult, Tensor};
use candle_nn::ops as cnn_ops;
use image::RgbImage;
use oar_ocr_core::core::OCRError;
use oar_ocr_core::domain::structure::LayoutElement;
use rayon::prelude::*;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

/// Read `generation_config.json::repetition_penalty`. Returns 1.0 (no-op) if
/// the file is missing, unparseable, or the field is absent — matches
/// HuggingFace's default. Local HunyuanOCR config ships 1.03.
fn load_repetition_penalty(model_dir: &Path) -> f64 {
    let path = model_dir.join("generation_config.json");
    let Ok(contents) = std::fs::read_to_string(&path) else {
        return 1.0;
    };
    let Ok(v) = serde_json::from_str::<serde_json::Value>(&contents) else {
        return 1.0;
    };
    v.get("repetition_penalty").and_then(|x| x.as_f64()).unwrap_or(1.0)
}

/// Apply HuggingFace's `RepetitionPenaltyLogitsProcessor` rule to a 1D logits
/// tensor and return the argmax id. For each token id that appears in
/// `seen`, the rule pushes its logit toward zero **once**:
/// `logit /= penalty` when positive, `logit *= penalty` when non-positive
/// (see `transformers.generation.logits_process.RepetitionPenaltyLogitsProcessor`).
/// HF computes this with `scatter(input_ids, …)`, which collapses duplicate
/// positions in `input_ids` down to a single penalty per unique vocab id —
/// applying the penalty per *occurrence* would compound to `penalty^k` for a
/// token repeated `k` times and quickly suppresses legitimate high-frequency
/// tokens like `<td>` in a structured HTML page. We dedup before applying.
fn argmax_with_repetition_penalty(
    logits: &Tensor,
    seen: &[u32],
    penalty: f32,
) -> Result<u32, OCRError> {
    let mut vec = logits
        .to_dtype(DType::F32)
        .and_then(|t| t.to_vec1::<f32>())
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "rep penalty to_vec1", e))?;
    let vocab = vec.len();
    let mut unique: Vec<u32> = seen.to_vec();
    unique.sort_unstable();
    unique.dedup();
    for &id in &unique {
        let idx = id as usize;
        if idx >= vocab {
            continue;
        }
        let v = vec[idx];
        vec[idx] = if v > 0.0 { v / penalty } else { v * penalty };
    }
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (i, &v) in vec.iter().enumerate() {
        if v > best_val {
            best_val = v;
            best_idx = i;
        }
    }
    Ok(best_idx as u32)
}

pub struct HunyuanOcr {
    model_dir: PathBuf,
    device: Device,
    dtype: DType,
    cfg: HunyuanOcrConfig,
    image_cfg: HunyuanOcrImageProcessorConfig,
    tokenizer: Tokenizer,
    llm: HunyuanLlm,
    vision: HunyuanVisionModel,
    stop_token_ids: Vec<u32>,
    /// `generation_config.json::repetition_penalty`. HuggingFace's
    /// `generate(do_sample=False)` still applies repetition_penalty via the
    /// LogitsProcessor list before the argmax. Without it, large-context chart
    /// inputs can collapse into runaway-repeat loops (e.g. Mermaid node IDs
    /// `A, B, … BZ, BZW, BZWW, BZWWZ …`) that never hit EOS. Default 1.0 means
    /// the value isn't applied. Only the baseline greedy path consumes this;
    /// the HSD verification paths intentionally keep raw logits.
    repetition_penalty: f64,
}

impl HunyuanOcr {
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, OCRError> {
        let model_dir = model_dir.as_ref();

        let cfg = HunyuanOcrConfig::from_path(model_dir.join("config.json"))?;
        let image_cfg =
            HunyuanOcrImageProcessorConfig::from_path(model_dir.join("preprocessor_config.json"))?;

        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| {
            OCRError::ConfigError {
                message: format!("failed to load HunyuanOCR tokenizer.json: {e}"),
            }
        })?;

        let mut stop_token_ids = Vec::new();
        stop_token_ids.push(cfg.eod_token_id);
        stop_token_ids.push(cfg.eos_token_id);
        if let Some(id) = tokenizer.token_to_id("<｜hy_Assistant｜>") {
            stop_token_ids.push(id);
        }
        stop_token_ids.sort_unstable();
        stop_token_ids.dedup();

        let dtype = device.bf16_default_to_f32();

        let weight_files = resolve_safetensors_shards(model_dir)?;
        // SAFETY: from_mmaped_safetensors is unsafe because it memory-maps weight files
        // directly. The caller must ensure the safetensors files are valid and not corrupted.
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(&weight_files, dtype, &device)
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load safetensors shards", e))?
        };

        let llm = HunyuanLlm::load(&cfg, vb.pp("model"))?;
        let vision = HunyuanVisionModel::load(&cfg.vision_config, vb.pp("vit"))?;

        let repetition_penalty = load_repetition_penalty(model_dir);

        Ok(Self {
            model_dir: model_dir.to_path_buf(),
            device,
            dtype,
            cfg,
            image_cfg,
            tokenizer,
            llm,
            vision,
            stop_token_ids,
            repetition_penalty,
        })
    }

    /// Generate OCR output for one or more images with custom instructions.
    ///
    /// Supports true GPU batching when multiple images are provided.
    ///
    /// # Arguments
    /// * `images` - Input images
    /// * `instructions` - Instruction for each image (must match images length)
    /// * `max_new_tokens` - Maximum tokens to generate per image
    ///
    /// # Returns
    /// Vector of results, one per input image.
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
                    "HunyuanOCR: images count ({}) != instructions count ({})",
                    images.len(),
                    instructions.len()
                ),
            })];
        }

        #[cfg(feature = "hunyuan-prefill-debug")]
        if std::env::var_os("OAROCR_PREFILL_INPUTS_FROM").is_some() {
            return match self.generate_from_prefill_env(images.len(), max_new_tokens) {
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
            };
        }

        match self.generate_tokens_internal(images, instructions, max_new_tokens) {
            Ok(results) => results
                .into_iter()
                .map(|tokens| self.decode_generated_tokens(&tokens))
                .collect(),
            Err(e) => {
                // Walk the source chain so the underlying candle / CUDA
                // failure isn't hidden behind the top-level OCRError.
                let mut chain = format!("generation failed: {e}");
                let mut cur: Option<&dyn std::error::Error> = std::error::Error::source(&e);
                while let Some(s) = cur {
                    chain.push_str(&format!("\n  caused by: {s}"));
                    cur = s.source();
                }
                let msg = chain;
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
    /// excluding stop tokens, before tokenizer decoding or trimming.
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
                    "HunyuanOCR: images count ({}) != instructions count ({})",
                    images.len(),
                    instructions.len()
                ),
            })];
        }

        match self.generate_tokens_internal(images, instructions, max_new_tokens) {
            Ok(results) => results.into_iter().map(Ok).collect(),
            Err(e) => {
                let mut chain = format!("generation failed: {e}");
                let mut cur: Option<&dyn std::error::Error> = std::error::Error::source(&e);
                while let Some(s) = cur {
                    chain.push_str(&format!("\n  caused by: {s}"));
                    cur = s.source();
                }
                let msg = chain;
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
        instructions: &[impl AsRef<str>],
        max_new_tokens: usize,
    ) -> Result<Vec<Vec<u32>>, OCRError> {
        let batch_size = images.len();
        #[cfg(feature = "hunyuan-prefill-debug")]
        if let Some(prefill) = self.load_prefill_dump()? {
            if batch_size != 1 {
                return Err(OCRError::InvalidInput {
                    message: "HunyuanOCR: OAROCR_PREFILL_INPUTS_FROM supports batch_size=1 only"
                        .to_string(),
                });
            }
            let _ = prefill;
            let _ = max_new_tokens;
            return Err(OCRError::InvalidInput {
                message: "HunyuanOCR: raw token generation is unavailable with OAROCR_PREFILL_INPUTS_FROM".to_string(),
            });
        }

        // 1. Preprocess all images and build prompts
        let mut all_input_ids: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
        let mut all_image_inputs: Vec<HunyuanOcrImageInputs> = Vec::with_capacity(batch_size);
        let dump_dir = if batch_size == 1 {
            std::env::var("OAROCR_DUMP_DIR").ok()
        } else {
            None
        };

        for (image, instruction) in images.iter().zip(instructions.iter()) {
            let instruction = instruction.as_ref();
            let mut image_inputs = preprocess_image(
                image,
                &self.image_cfg,
                &self.cfg.vision_config,
                &self.device,
                self.dtype,
            )?;
            self.maybe_override_pixel_values(&mut image_inputs)?;

            let prompt = build_prompt(instruction);
            let enc = self
                .tokenizer
                .encode(prompt, false)
                .map_err(|e| OCRError::InvalidInput {
                    message: format!("HunyuanOCR: tokenizer encode failed: {e}"),
                })?;

            let mut input_ids = enc.get_ids().to_vec();
            expand_image_tokens_in_place(&mut input_ids, &self.cfg, &image_inputs)?;
            if let Some(d) = &dump_dir {
                let _ = std::fs::create_dir_all(d);
                let seq_len = input_ids.len();
                let _ = image_inputs
                    .pixel_values
                    .save_safetensors("t", format!("{d}/pixel_values.safetensors"));
                if let Ok(t) = Tensor::new(input_ids.clone(), &self.device)
                    .and_then(|x| x.reshape((1, seq_len)))
                {
                    let _ = t.save_safetensors("t", format!("{d}/input_ids.safetensors"));
                }
            }

            all_input_ids.push(input_ids);
            all_image_inputs.push(image_inputs);
        }

        // 2. Compute vision features and build embeddings for each sample
        let seq_lens: Vec<usize> = all_input_ids.iter().map(|ids| ids.len()).collect();
        let max_seq_len = *seq_lens.iter().max().unwrap();

        let mut batch_embeds: Vec<Tensor> = Vec::with_capacity(batch_size);
        let mut batch_position_ids: Vec<Tensor> = Vec::with_capacity(batch_size);

        for (input_ids, image_inputs) in all_input_ids.iter().zip(all_image_inputs.iter()) {
            let seq_len = input_ids.len();
            let pad_len = max_seq_len - seq_len;

            // Embed tokens
            let input_ids_t = Tensor::new(input_ids.clone(), &self.device)
                .and_then(|t| t.reshape((1, seq_len)))
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "create input_ids", e))?;
            let token_embeds = self.llm.embed(&input_ids_t)?;

            // Get vision features
            let (image_embeds, merged_hw) = self.vision.forward(&image_inputs.pixel_values)?;
            if let Some(d) = &dump_dir {
                let _ = image_embeds.save_safetensors("t", format!("{d}/vit_out.safetensors"));
            }
            if merged_hw
                != (
                    image_inputs.grid_thw_merged.1,
                    image_inputs.grid_thw_merged.2,
                )
            {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "HunyuanOCR: merged grid mismatch: vision={:?} preprocessor={:?}",
                        merged_hw, image_inputs.grid_thw_merged
                    ),
                });
            }

            // Fuse image embeddings — see hsd_prefill_single for the spec.
            let (start_pos, end_pos) = find_image_span(input_ids, &self.cfg)?;
            let inner_len = end_pos.saturating_sub(start_pos + 1);
            let (img_len, _) = image_embeds
                .dims2()
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "image_embeds dims2", e))?;
            if inner_len != img_len {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "HunyuanOCR: image-token run length mismatch: tokens={inner_len} embeds={img_len}"
                    ),
                });
            }

            let token_embeds = token_embeds.squeeze(0).map_err(|e| {
                candle_to_ocr_inference("HunyuanOCR", "squeeze token embeddings", e)
            })?;

            let mut parts: Vec<Tensor> = Vec::with_capacity(3);
            // Prefix incl. image_start (text-embedded).
            parts.push(token_embeds.i((0..=start_pos, ..)).map_err(|e| {
                candle_to_ocr_inference("HunyuanOCR", "slice prefix embeddings", e)
            })?);
            parts.push(image_embeds);
            if end_pos < input_ids.len() {
                // Suffix incl. image_end (text-embedded).
                parts.push(
                    token_embeds
                        .i((end_pos..input_ids.len(), ..))
                        .map_err(|e| {
                            candle_to_ocr_inference("HunyuanOCR", "slice suffix embeddings", e)
                        })?,
                );
            }
            let refs: Vec<&Tensor> = parts.iter().collect();
            let mut inputs_embeds = Tensor::cat(&refs, 0)
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "cat embeds", e))?
                .unsqueeze(0)
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "unsqueeze embeds", e))?;

            // Left-pad if needed
            if pad_len > 0 {
                let hidden_size = inputs_embeds
                    .dim(2)
                    .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "get hidden_size", e))?;
                let pad = Tensor::zeros(
                    (1, pad_len, hidden_size),
                    inputs_embeds.dtype(),
                    &self.device,
                )
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "create pad", e))?;
                inputs_embeds = Tensor::cat(&[&pad, &inputs_embeds], 1)
                    .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "cat pad", e))?;
            }
            batch_embeds.push(inputs_embeds);

            // Build position IDs
            let pos_ids = build_position_ids(input_ids, &self.cfg, image_inputs)?;
            // Left-pad position IDs
            let pos_ids = if pad_len > 0 {
                let pad_pos = Tensor::zeros((4, 1, pad_len), DType::I64, &self.device)
                    .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "create pad pos", e))?;
                Tensor::cat(&[&pad_pos, &pos_ids], 2)
                    .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "cat pad pos", e))?
            } else {
                pos_ids
            };
            batch_position_ids.push(pos_ids);
        }

        // 3. Stack batched tensors
        let batch_refs: Vec<&Tensor> = batch_embeds.iter().collect();
        let inputs_embeds = Tensor::cat(&batch_refs, 0)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "stack embeds", e))?;

        let pos_refs: Vec<&Tensor> = batch_position_ids.iter().collect();
        let position_ids = Tensor::cat(&pos_refs, 1)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "stack pos", e))?;
        if let Some(d) = &dump_dir {
            let _ = inputs_embeds.save_safetensors("t", format!("{d}/inputs_embeds.safetensors"));
            let _ = position_ids.save_safetensors("t", format!("{d}/position_ids.safetensors"));
        }

        // 4. Create attention mask
        let causal = create_causal_mask(max_seq_len, max_seq_len, self.dtype, &self.device)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "create causal", e))?;
        let padding = create_left_padding_mask(&seq_lens, max_seq_len, self.dtype, &self.device)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "create padding", e))?;
        let mask = combine_masks(&causal, &padding)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "combine masks", e))?;

        // 5. Prefill
        self.llm.clear_kv_cache();
        let hidden = self
            .llm
            .forward(&inputs_embeds, &position_ids, Some(&mask))?;

        // 6. Get initial logits per sample
        let mut logits_list: Vec<Tensor> = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let last = hidden
                .i((i, max_seq_len - 1, ..))
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "get last hidden", e))?;
            let logits = self.logits_from_hidden(&last)?;
            if i == 0
                && let Some(d) = &dump_dir
            {
                let _ = logits.save_safetensors("t", format!("{d}/first_token_logits.safetensors"));
            }
            logits_list.push(logits);
        }

        // 7. Autoregressive decode
        let mut generated: Vec<Vec<u32>> = vec![Vec::new(); batch_size];
        let mut finished: Vec<bool> = vec![false; batch_size];
        let mut positions: Vec<i64> = seq_lens.iter().map(|&len| len as i64).collect();

        for _ in 0..max_new_tokens {
            if finished.iter().all(|&f| f) {
                break;
            }

            let mut next_tokens: Vec<u32> = Vec::with_capacity(batch_size);
            for (i, logits) in logits_list.iter().enumerate() {
                if finished[i] {
                    next_tokens.push(0); // Padding token for finished samples
                } else {
                    // Mirror HuggingFace's `generate(do_sample=False)`: even
                    // greedy decoding runs the LogitsProcessor list, so the
                    // `repetition_penalty` from generation_config.json gets
                    // applied to logits before argmax. Without this the model
                    // can spiral into runaway-repeat loops on large-context
                    // inputs (observed on chart_01.jpg, seq≈11584, producing
                    // 33K chars of synthetic Mermaid node IDs `BZ, BZW, …`).
                    // HSD paths intentionally skip this to keep raw log-probs
                    // for the τ-tolerance verification.
                    let tok = if self.repetition_penalty > 1.0 && !generated[i].is_empty() {
                        argmax_with_repetition_penalty(
                            logits,
                            &generated[i],
                            self.repetition_penalty as f32,
                        )?
                    } else {
                        logits
                            .argmax(D::Minus1)
                            .and_then(|t| t.to_scalar::<u32>())
                            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "argmax", e))?
                    };

                    if self.stop_token_ids.contains(&tok) {
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

            // Batch forward for next tokens
            let tokens = Tensor::new(next_tokens, &self.device)
                .and_then(|t| t.reshape((batch_size, 1)))
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "create tokens", e))?;
            let embeds = self.llm.embed(&tokens)?;

            // Build 4-axis position IDs for decode step
            let pos_data: Vec<i64> = positions.iter().flat_map(|&p| [p, p, p, p]).collect();
            let pos = Tensor::new(pos_data, &self.device)
                .and_then(|t| t.reshape((4, batch_size, 1)))
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "create pos", e))?;

            let hs = self.llm.forward(&embeds, &pos, None)?;

            logits_list.clear();
            for i in 0..batch_size {
                let h = hs
                    .i((i, 0, ..))
                    .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "get hs", e))?;
                let logits = self.logits_from_hidden(&h)?;
                logits_list.push(logits);
            }

            for (i, p) in positions.iter_mut().enumerate() {
                if !finished[i] {
                    *p += 1;
                }
            }
        }

        self.llm.clear_kv_cache();
        Ok(generated)
    }

    pub fn decode_tokens(&self, tokens: &[u32]) -> Result<String, OCRError> {
        self.decode_generated_tokens(tokens)
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn decode_generated_tokens(&self, tokens: &[u32]) -> Result<String, OCRError> {
        let decoded = self
            .tokenizer
            .decode(tokens, true)
            .map_err(|e| OCRError::InvalidInput {
                message: format!("HunyuanOCR: tokenizer decode failed: {e}"),
            })?;
        Ok(decoded.trim().to_string())
    }

    #[cfg(feature = "hunyuan-prefill-debug")]
    fn generate_from_prefill_env(
        &self,
        batch_size: usize,
        max_new_tokens: usize,
    ) -> Result<Vec<String>, OCRError> {
        let Some(prefill) = self.load_prefill_dump()? else {
            return Err(OCRError::InvalidInput {
                message: "HunyuanOCR: OAROCR_PREFILL_INPUTS_FROM is set but no dump was loaded"
                    .to_string(),
            });
        };
        if batch_size != 1 {
            return Err(OCRError::InvalidInput {
                message: "HunyuanOCR: OAROCR_PREFILL_INPUTS_FROM supports batch_size=1 only"
                    .to_string(),
            });
        }
        let text = self.generate_from_prefill_dump(prefill, max_new_tokens)?;
        Ok(vec![text])
    }

    /// Generate OCR output for a single image using Hierarchical Speculative
    /// Decoding (currently page-level / Stage-2-style: drafts are matched and
    /// verified against a single full-page forward pass).
    ///
    /// `drafts` are markdown-style strings produced by the lightweight pipeline
    /// drafter (PP-DocLayout + region recognizers). Each is tokenized with the
    /// HunyuanOCR tokenizer (this is mandatory — HSD's prefix matching must
    /// happen in the target VLM's token space).
    ///
    /// Returns `(generated_text, stats)` where `stats.stage2.accept` records
    /// the AAL and step counts needed to compute SR_decode / SR_e2e.
    pub fn generate_hsd(
        &self,
        image: &RgbImage,
        instruction: &str,
        drafts: &[String],
        hsd_cfg: &HsdConfig,
    ) -> Result<(String, HsdStats), OCRError> {
        let t_drafter_prep = Instant::now();

        // Tokenize all drafts up-front with the HunyuanOCR tokenizer.
        let mut tokenized: Vec<Draft> = Vec::with_capacity(drafts.len());
        for d in drafts {
            if d.trim().is_empty() {
                continue;
            }
            let enc =
                self.tokenizer
                    .encode(d.as_str(), false)
                    .map_err(|e| OCRError::InvalidInput {
                        message: format!("HunyuanOCR HSD: tokenizer encode failed: {e}"),
                    })?;
            let tokens = enc.get_ids().to_vec();
            if !tokens.is_empty() {
                tokenized.push(Draft::new(tokens));
            }
        }
        self.generate_hsd_tokenized(
            image,
            instruction,
            &tokenized,
            hsd_cfg,
            t_drafter_prep.elapsed(),
        )
    }

    /// HSD entry that consumes already-tokenized drafts. This is the oracle
    /// path used by benchmarks to avoid `decode -> encode` tokenizer
    /// round-trips when the draft comes from this backend's own baseline.
    pub fn generate_hsd_with_token_drafts(
        &self,
        image: &RgbImage,
        instruction: &str,
        drafts: &[Draft],
        hsd_cfg: &HsdConfig,
    ) -> Result<(String, HsdStats), OCRError> {
        let (tokens, stats) = self.generate_hsd_tokens_tokenized(
            image,
            instruction,
            drafts,
            hsd_cfg,
            Duration::ZERO,
        )?;
        let text = self.decode_tokens(&tokens)?;
        Ok((text.trim().to_string(), stats))
    }

    /// Token-returning HSD entry for diagnostics and exact oracle checks.
    pub fn generate_hsd_tokens_with_token_drafts(
        &self,
        image: &RgbImage,
        instruction: &str,
        drafts: &[Draft],
        hsd_cfg: &HsdConfig,
    ) -> Result<(Vec<u32>, HsdStats), OCRError> {
        self.generate_hsd_tokens_tokenized(image, instruction, drafts, hsd_cfg, Duration::ZERO)
    }

    fn generate_hsd_tokenized(
        &self,
        image: &RgbImage,
        instruction: &str,
        tokenized: &[Draft],
        hsd_cfg: &HsdConfig,
        drafter_elapsed: Duration,
    ) -> Result<(String, HsdStats), OCRError> {
        let (generated, stats) = self.generate_hsd_tokens_tokenized(
            image,
            instruction,
            tokenized,
            hsd_cfg,
            drafter_elapsed,
        )?;
        let text = self
            .tokenizer
            .decode(&generated, true)
            .map_err(|e| OCRError::InvalidInput {
                message: format!("HunyuanOCR HSD: tokenizer decode failed: {e}"),
            })?;
        Ok((text.trim().to_string(), stats))
    }

    fn generate_hsd_tokens_tokenized(
        &self,
        image: &RgbImage,
        instruction: &str,
        tokenized: &[Draft],
        hsd_cfg: &HsdConfig,
        drafter_elapsed: Duration,
    ) -> Result<(Vec<u32>, HsdStats), OCRError> {
        let mut stats = HsdStats::default();
        stats.drafter = drafter_elapsed;
        // Stage 2 (page-level) prefill.
        let t_prefill = Instant::now();
        let initial_lp = self.hsd_prefill_single(image, instruction)?;
        stats.stage2.vision_prefill = t_prefill.elapsed();
        stats.stage2.forward_passes = 1;

        // Drive HSD verification.
        let t_decode = Instant::now();
        let mut backend = HunyuanSpecBackend::new(self);
        let mut accept = AcceptStats::default();
        let mut dsv = Default::default();
        let generated = spec_decode(
            &mut backend,
            &tokenized,
            initial_lp,
            hsd_cfg.max_page_tokens,
            &hsd_cfg.dsv,
            &mut accept,
            &mut dsv,
        )
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "spec_decode", e))?;
        stats.stage2.decode = t_decode.elapsed();
        stats.stage2.emitted_tokens = generated.len() as u32;
        stats.stage2.accept = accept;
        stats.stage2.dsv = dsv;
        stats.stage2.forward_passes += backend.forward_passes;
        self.llm.clear_kv_cache();
        Ok((generated, stats))
    }

    /// Full Hierarchical Speculative Decoding entry: Stage 1 (region-level
    /// local verification) followed by Stage 2 (page-level global verification).
    ///
    /// `elements` is the drafter pipeline's layout output (PP-DocLayout +
    /// recognition backend). Each element's recognized text is treated as the
    /// region draft. Empty / `ignore_labels`-matching elements are skipped.
    ///
    /// `hsd_cfg.enable_stage1` and `hsd_cfg.enable_stage2` independently gate
    /// the two stages (used for ablations — `enable_stage2 = false` reproduces
    /// the lossy "Stage 1 only" line in the paper's Tab. 7).
    pub fn generate_hsd_full(
        &self,
        image: &RgbImage,
        instruction: &str,
        elements: &[LayoutElement],
        ignore_labels: &[String],
        hsd_cfg: &HsdConfig,
    ) -> Result<(String, HsdStats), OCRError> {
        let mut stats = HsdStats::default();

        // 1. Build region drafts using the target VLM's tokenizer.
        let t_drafter = Instant::now();
        let tokenizer = &self.tokenizer;
        let region_drafts = build_region_drafts(elements, ignore_labels, |s: &str| {
            tokenizer
                .encode(s, false)
                .map(|enc| enc.get_ids().to_vec())
                .unwrap_or_default()
        });
        stats.drafter = t_drafter.elapsed();

        // 2. Stage 1 — build independent region work items, then run
        // target-model verification. Each HunyuanOCR instance owns a mutable
        // LLM KV cache, so parallel verification uses separate model workers.
        let mut stage1_outputs: Vec<String> = Vec::with_capacity(region_drafts.len());
        if hsd_cfg.enable_stage1 && !region_drafts.is_empty() {
            let t_prep = Instant::now();
            let stage1_work = build_stage1_work_items(image, &region_drafts)?;
            stats.stage1.draft_prep += t_prep.elapsed();

            let stage1_results = self.run_stage1_work(&stage1_work, instruction, hsd_cfg)?;
            for (text, item_stats) in stage1_results {
                add_stage_stats(&mut stats.stage1, item_stats);
                stage1_outputs.push(text);
            }
        }

        // 3. Stage 2 — page-level global verification.
        if hsd_cfg.enable_stage2 {
            let page_draft_text = if !stage1_outputs.is_empty() {
                stage1_outputs.join("\n\n")
            } else {
                page_markdown(elements, ignore_labels)
            };
            let (text, s2_stats) = self.generate_hsd(
                image,
                instruction,
                std::slice::from_ref(&page_draft_text),
                hsd_cfg,
            )?;
            stats.stage2 = s2_stats.stage2;
            stats.drafter += s2_stats.drafter;
            return Ok((text, stats));
        }

        // 4. Stage 2 disabled — return Stage-1-only aggregation (lossy ablation).
        Ok((stage1_outputs.join("\n\n"), stats))
    }

    fn run_stage1_work(
        &self,
        stage1_work: &[(RegionDraft, RgbImage)],
        instruction: &str,
        hsd_cfg: &HsdConfig,
    ) -> Result<Vec<(String, StageStats)>, OCRError> {
        let workers = stage1_verify_workers().min(stage1_work.len());
        if workers > 1 {
            self.run_stage1_work_parallel(stage1_work, instruction, hsd_cfg, workers)
        } else {
            stage1_work
                .iter()
                .map(|(region, crop)| self.run_stage1_item(region, crop, instruction, hsd_cfg))
                .collect()
        }
    }

    fn run_stage1_work_parallel(
        &self,
        stage1_work: &[(RegionDraft, RgbImage)],
        instruction: &str,
        hsd_cfg: &HsdConfig,
        workers: usize,
    ) -> Result<Vec<(String, StageStats)>, OCRError> {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .map_err(|e| OCRError::InvalidInput {
                message: format!("HunyuanOCR HSD: failed to build stage1 verify pool: {e}"),
            })?;
        let model_dir = self.model_dir.clone();
        let device = self.device.clone();

        let worker_outputs: Vec<Result<Vec<(usize, String, StageStats)>, OCRError>> =
            pool.install(|| {
                (0..workers)
                    .into_par_iter()
                    .map(|worker_id| {
                        let worker = HunyuanOcr::from_dir(&model_dir, device.clone())?;
                        let mut out = Vec::new();
                        for idx in (worker_id..stage1_work.len()).step_by(workers) {
                            let (region, crop) = &stage1_work[idx];
                            let (text, stats) =
                                worker.run_stage1_item(region, crop, instruction, hsd_cfg)?;
                            out.push((idx, text, stats));
                        }
                        Ok(out)
                    })
                    .collect()
            });

        let mut ordered = Vec::with_capacity(stage1_work.len());
        for worker in worker_outputs {
            ordered.extend(worker?);
        }
        ordered.sort_by_key(|(idx, _, _)| *idx);
        Ok(ordered
            .into_iter()
            .map(|(_, text, stats)| (text, stats))
            .collect())
    }

    fn run_stage1_item(
        &self,
        region: &RegionDraft,
        crop: &RgbImage,
        instruction: &str,
        hsd_cfg: &HsdConfig,
    ) -> Result<(String, StageStats), OCRError> {
        let mut stats = StageStats::default();

        let t_pre = Instant::now();
        let init_lp = self.hsd_prefill_single(crop, instruction)?;
        stats.vision_prefill += t_pre.elapsed();
        stats.forward_passes += 1;

        let t_dec = Instant::now();
        let mut backend = HunyuanSpecBackend::new(self);
        let mut accept = AcceptStats::default();
        let mut dsv = Default::default();
        let toks = spec_decode(
            &mut backend,
            std::slice::from_ref(&region.draft),
            init_lp,
            hsd_cfg.max_region_tokens,
            &hsd_cfg.dsv,
            &mut accept,
            &mut dsv,
        )
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "stage1 spec_decode", e))?;
        stats.decode += t_dec.elapsed();
        stats.emitted_tokens += toks.len() as u32;
        stats.forward_passes += backend.forward_passes;
        stats.dsv = dsv;
        stats.accept = accept;
        self.llm.clear_kv_cache();

        let text = self
            .tokenizer
            .decode(&toks, true)
            .map_err(|e| OCRError::InvalidInput {
                message: format!("HunyuanOCR HSD: tokenizer decode failed: {e}"),
            })?;
        Ok((text.trim().to_string(), stats))
    }

    /// Run the prefill forward pass for HSD, leaving the LLM's KV cache
    /// populated. Returns the F32 log-probabilities at the last prompt
    /// position (shape `(vocab,)`).
    fn hsd_prefill_single(&self, image: &RgbImage, instruction: &str) -> Result<Tensor, OCRError> {
        // 1. Preprocess.
        let mut image_inputs = preprocess_image(
            image,
            &self.image_cfg,
            &self.cfg.vision_config,
            &self.device,
            self.dtype,
        )?;
        self.maybe_override_pixel_values(&mut image_inputs)?;
        let prompt = build_prompt(instruction);
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| OCRError::InvalidInput {
                message: format!("HunyuanOCR HSD: tokenizer encode failed: {e}"),
            })?;
        let mut input_ids = enc.get_ids().to_vec();
        expand_image_tokens_in_place(&mut input_ids, &self.cfg, &image_inputs)?;
        let seq_len = input_ids.len();

        let dump_dir = std::env::var("OAROCR_DUMP_DIR").ok();
        if let Some(d) = &dump_dir {
            let _ = std::fs::create_dir_all(d);
            let _ = image_inputs
                .pixel_values
                .save_safetensors("t", format!("{d}/pixel_values.safetensors"));
            // Save input_ids as a U32 tensor.
            if let Ok(t) =
                Tensor::new(input_ids.clone(), &self.device).and_then(|x| x.reshape((1, seq_len)))
            {
                let _ = t.save_safetensors("t", format!("{d}/input_ids.safetensors"));
            }
        }

        // 2. Vision features.
        let (image_embeds, merged_hw) = self.vision.forward(&image_inputs.pixel_values)?;
        if let Some(d) = &dump_dir {
            let _ = image_embeds.save_safetensors("t", format!("{d}/vit_out.safetensors"));
        }
        if merged_hw
            != (
                image_inputs.grid_thw_merged.1,
                image_inputs.grid_thw_merged.2,
            )
        {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "HunyuanOCR: merged grid mismatch: vision={:?} preprocessor={:?}",
                    merged_hw, image_inputs.grid_thw_merged
                ),
            });
        }

        // 3. Build fused embeddings.
        let input_ids_t = Tensor::new(input_ids.clone(), &self.device)
            .and_then(|t| t.reshape((1, seq_len)))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "create input_ids", e))?;
        let token_embeds =
            self.llm.embed(&input_ids_t)?.squeeze(0).map_err(|e| {
                candle_to_ocr_inference("HunyuanOCR", "squeeze token embeddings", e)
            })?;

        // Splice: keep image_start (120118) and image_end (120119) as text
        // tokens (matching the upstream HF processor), replace only the
        // contiguous image_token_id run between them with the vit output.
        let (start_pos, end_pos) = find_image_span(&input_ids, &self.cfg)?;
        let inner_len = end_pos.saturating_sub(start_pos + 1);
        let (img_len, _) = image_embeds
            .dims2()
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "image_embeds dims2", e))?;
        if inner_len != img_len {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "HunyuanOCR: image-token run length mismatch: tokens={inner_len} embeds={img_len}"
                ),
            });
        }
        let mut parts: Vec<Tensor> = Vec::with_capacity(3);
        // Prefix: [0, start_pos] inclusive — keeps image_start as text-embedded.
        parts.push(
            token_embeds
                .i((0..=start_pos, ..))
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "slice prefix embeddings", e))?,
        );
        parts.push(image_embeds);
        if end_pos < input_ids.len() {
            // Suffix: [end_pos, input_ids.len()) — keeps image_end as text-embedded.
            parts.push(
                token_embeds
                    .i((end_pos..input_ids.len(), ..))
                    .map_err(|e| {
                        candle_to_ocr_inference("HunyuanOCR", "slice suffix embeddings", e)
                    })?,
            );
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        let inputs_embeds = Tensor::cat(&refs, 0)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "cat embeds", e))?
            .unsqueeze(0)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "unsqueeze embeds", e))?;

        // 4. Position ids and causal mask.
        let pos_ids = build_position_ids(&input_ids, &self.cfg, &image_inputs)?;
        let causal = create_causal_mask(seq_len, seq_len, self.dtype, &self.device)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "create causal", e))?;

        if let Some(d) = &dump_dir {
            let _ = inputs_embeds.save_safetensors("t", format!("{d}/inputs_embeds.safetensors"));
            let _ = pos_ids.save_safetensors("t", format!("{d}/position_ids.safetensors"));
        }

        // 5. Prefill.
        self.llm.clear_kv_cache();
        let hidden = self.llm.forward(&inputs_embeds, &pos_ids, Some(&causal))?;

        // 6. Last-position log-probabilities, F32.
        let last = hidden
            .i((0, seq_len - 1, ..))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "get last hidden", e))?;
        let logits = self.logits_from_hidden(&last)?; // (vocab,)
        if let Some(d) = &dump_dir {
            let _ = logits.save_safetensors("t", format!("{d}/first_token_logits.safetensors"));
        }
        let lp = cnn_ops::log_softmax(
            &logits
                .to_dtype(DType::F32)
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "logits to f32", e))?,
            D::Minus1,
        )
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "log_softmax prefill", e))?;
        Ok(lp)
    }

    fn logits_from_hidden(&self, hidden: &Tensor) -> Result<Tensor, OCRError> {
        let hidden = hidden.unsqueeze(0).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: unsqueeze hidden failed",
                e,
            )
        })?;
        let w = self.llm.token_embedding_weight();
        let wt = w.transpose(0, 1).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: transpose embedding weight failed",
                e,
            )
        })?;
        hidden
            .matmul(&wt)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "matmul logits", e))?
            .squeeze(0)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "squeeze logits", e))
    }

    fn maybe_override_pixel_values(
        &self,
        image_inputs: &mut HunyuanOcrImageInputs,
    ) -> Result<(), OCRError> {
        // Debug-only alignment hook: lets us feed processor dumps from the
        // Python upstream into the Rust vision stack without changing the
        // normal preprocessor path.
        let Some(path) = std::env::var_os("OAROCR_PIXEL_VALUES_FROM") else {
            return Ok(());
        };
        let path = PathBuf::from(path);
        let mut tensors = candle_core::safetensors::load(&path, &self.device)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load pixel_values override", e))?;
        let tensor = tensors.remove("t").ok_or_else(|| OCRError::InvalidInput {
            message: format!(
                "HunyuanOCR: pixel_values override {} is missing tensor named 't'",
                path.display()
            ),
        })?;
        let expected = image_inputs.pixel_values.shape().clone();
        if tensor.shape() != &expected {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "HunyuanOCR: pixel_values override shape {:?} does not match preprocessed shape {:?}",
                    tensor.shape(),
                    expected
                ),
            });
        }
        image_inputs.pixel_values = tensor
            .to_dtype(self.dtype)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "cast pixel_values override", e))?;
        Ok(())
    }

    #[cfg(feature = "hunyuan-prefill-debug")]
    fn load_prefill_dump(&self) -> Result<Option<PrefillDump>, OCRError> {
        let Some(dir) = std::env::var_os("OAROCR_PREFILL_INPUTS_FROM") else {
            return Ok(None);
        };
        let dir = PathBuf::from(dir);
        let load_t = |name: &str| -> Result<Tensor, OCRError> {
            let path = dir.join(name);
            let mut tensors = candle_core::safetensors::load(&path, &self.device)
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load prefill dump", e))?;
            tensors.remove("t").ok_or_else(|| OCRError::InvalidInput {
                message: format!(
                    "HunyuanOCR: prefill dump {} is missing tensor named 't'",
                    path.display()
                ),
            })
        };

        let input_ids_t = load_t("input_ids.safetensors")?;
        let input_ids = input_ids_t
            .flatten_all()
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "flatten dumped input_ids", e))?
            .to_dtype(DType::U32)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "cast dumped input_ids", e))?
            .to_vec1::<u32>()
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "read dumped input_ids", e))?;

        let inputs_path = dir.join("inputs_embeds.safetensors");
        let inputs_embeds = if inputs_path.exists() {
            load_t("inputs_embeds.safetensors")?
        } else {
            load_t("layer_00_in.safetensors")?
        };
        let inputs_embeds = inputs_embeds
            .to_dtype(self.dtype)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "cast dumped inputs_embeds", e))?;

        let position_ids = normalize_prefill_position_ids(load_t("position_ids.safetensors")?)?;
        let position_ids = position_ids
            .to_dtype(DType::I64)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "cast dumped position_ids", e))?;

        let seq_len = input_ids.len();
        let (_, embed_seq, _) = inputs_embeds
            .dims3()
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "dumped inputs_embeds dims3", e))?;
        let (_, _, pos_seq) = position_ids
            .dims3()
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "dumped position_ids dims3", e))?;
        if embed_seq != seq_len || pos_seq != seq_len {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "HunyuanOCR: prefill dump length mismatch: input_ids={seq_len}, inputs_embeds={embed_seq}, position_ids={pos_seq}"
                ),
            });
        }

        Ok(Some(PrefillDump {
            input_ids,
            inputs_embeds,
            position_ids,
        }))
    }

    #[cfg(feature = "hunyuan-prefill-debug")]
    fn generate_from_prefill_dump(
        &self,
        prefill: PrefillDump,
        max_new_tokens: usize,
    ) -> Result<String, OCRError> {
        let seq_len = prefill.input_ids.len();
        let causal = create_causal_mask(seq_len, seq_len, self.dtype, &self.device)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "create dump causal", e))?;

        if let Ok(d) = std::env::var("OAROCR_DUMP_DIR") {
            let _ = std::fs::create_dir_all(&d);
            let _ = prefill
                .inputs_embeds
                .save_safetensors("t", format!("{d}/inputs_embeds.safetensors"));
            let _ = prefill
                .position_ids
                .save_safetensors("t", format!("{d}/position_ids.safetensors"));
        }

        self.llm.clear_kv_cache();
        let hidden =
            self.llm
                .forward(&prefill.inputs_embeds, &prefill.position_ids, Some(&causal))?;
        let last = hidden
            .i((0, seq_len - 1, ..))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "get dumped last hidden", e))?;
        let mut logits = self.logits_from_hidden(&last)?;
        if let Ok(d) = std::env::var("OAROCR_DUMP_DIR") {
            let _ = logits.save_safetensors("t", format!("{d}/first_token_logits.safetensors"));
        }

        let mut generated: Vec<u32> = Vec::new();
        let mut position = seq_len as i64;
        for _ in 0..max_new_tokens {
            let tok = logits
                .argmax(D::Minus1)
                .and_then(|t| t.to_scalar::<u32>())
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "dump argmax", e))?;
            if self.stop_token_ids.contains(&tok) {
                break;
            }
            generated.push(tok);

            let tokens = Tensor::new(vec![tok], &self.device)
                .and_then(|t| t.reshape((1usize, 1usize)))
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "dump token tensor", e))?;
            let embeds = self.llm.embed(&tokens)?;
            let pos = Tensor::new(vec![position, position, position, position], &self.device)
                .and_then(|t| t.reshape((4usize, 1usize, 1usize)))
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "dump decode pos", e))?;
            let hs = self.llm.forward(&embeds, &pos, None)?;
            let h = hs
                .i((0, 0, ..))
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "dump decode hidden", e))?;
            logits = self.logits_from_hidden(&h)?;
            position += 1;
        }

        self.tokenizer
            .decode(&generated, true)
            .map(|s| s.trim().to_string())
            .map_err(|e| OCRError::InvalidInput {
                message: format!("HunyuanOCR: tokenizer decode dumped generation failed: {e}"),
            })
    }
}

fn build_stage1_work_items(
    image: &RgbImage,
    region_drafts: &[RegionDraft],
) -> Result<Vec<(RegionDraft, RgbImage)>, OCRError> {
    let crop_one = |region: &RegionDraft| {
        crop_region_image(image, &region.bbox).map(|crop| (region.clone(), crop))
    };

    let workers = std::env::var("OAROCR_HSD_STAGE1_CROP_WORKERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(0);
    if workers > 1 && region_drafts.len() > 1 {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .map_err(|e| OCRError::InvalidInput {
                message: format!("HunyuanOCR HSD: failed to build stage1 crop pool: {e}"),
            })?;
        pool.install(|| region_drafts.par_iter().map(crop_one).collect())
    } else {
        region_drafts.iter().map(crop_one).collect()
    }
}

fn stage1_verify_workers() -> usize {
    std::env::var("OAROCR_HSD_STAGE1_VERIFY_WORKERS")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(1)
}

fn add_stage_stats(dst: &mut StageStats, src: StageStats) {
    dst.vision_prefill += src.vision_prefill;
    dst.draft_prep += src.draft_prep;
    dst.decode += src.decode;
    dst.emitted_tokens += src.emitted_tokens;
    dst.forward_passes += src.forward_passes;
    dst.dsv.add_assign(&src.dsv);
    dst.accept
        .per_step_accepted
        .extend(src.accept.per_step_accepted);
    dst.accept.num_steps += src.accept.num_steps;
    dst.accept.num_fallbacks += src.accept.num_fallbacks;
}

#[cfg(feature = "hunyuan-prefill-debug")]
struct PrefillDump {
    input_ids: Vec<u32>,
    inputs_embeds: Tensor,
    position_ids: Tensor,
}

#[cfg(feature = "hunyuan-prefill-debug")]
fn normalize_prefill_position_ids(position_ids: Tensor) -> Result<Tensor, OCRError> {
    let dims = position_ids.dims().to_vec();
    match dims.as_slice() {
        [4, 1, _] => Ok(position_ids),
        [1, 4, _] => position_ids
            .permute((1usize, 0usize, 2usize))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "permute dumped position_ids", e)),
        _ => Err(OCRError::InvalidInput {
            message: format!(
                "HunyuanOCR: expected dumped position_ids shape (4,1,S) or (1,4,S), got {dims:?}"
            ),
        }),
    }
}

/// HSD adapter for HunyuanOCR. Borrows the model and drives the LLM's KV
/// cache through tree-attention verifications and single-token decode steps.
struct HunyuanSpecBackend<'a> {
    model: &'a HunyuanOcr,
    /// KV cache length captured at the start of the most recent
    /// [`SpecBackend::verify_tree`] call. [`SpecBackend::commit_verify`] uses
    /// this to translate path-node indices into absolute KV positions.
    pre_verify_kv: usize,
    /// Number of LLM forward passes (verify_tree + step_one) — populated for
    /// the per-stage StageStats accounting in `generate_hsd`.
    forward_passes: u32,
}

impl<'a> HunyuanSpecBackend<'a> {
    fn new(model: &'a HunyuanOcr) -> Self {
        Self {
            model,
            pre_verify_kv: 0,
            forward_passes: 0,
        }
    }

    fn project_logprobs_2d(&self, hidden_2d: &Tensor) -> CandleResult<Tensor> {
        // hidden_2d: (N, hidden). Project to (N, vocab) and log-softmax in F32.
        let w = self.model.llm.token_embedding_weight();
        let wt = w.transpose(0, 1)?;
        let logits = hidden_2d.matmul(&wt)?;
        cnn_ops::log_softmax(&logits.to_dtype(DType::F32)?, D::Minus1)
    }

    fn project_logprobs_1d(&self, hidden_1d: &Tensor) -> CandleResult<Tensor> {
        // hidden_1d: (hidden,). Project to (vocab,) and log-softmax in F32.
        let w = self.model.llm.token_embedding_weight();
        let wt = w.transpose(0, 1)?;
        // (1, hidden) @ (hidden, vocab) = (1, vocab) -> squeeze -> (vocab,).
        let logits = hidden_1d.unsqueeze(0)?.matmul(&wt)?.squeeze(0)?;
        cnn_ops::log_softmax(&logits.to_dtype(DType::F32)?, D::Minus1)
    }
}

impl<'a> SpecBackend for HunyuanSpecBackend<'a> {
    fn step_one(&mut self, token: u32) -> CandleResult<Tensor> {
        let model = self.model;
        let device = &model.device;

        // (1, 1) token tensor → (1, 1, hidden).
        let tok_t = Tensor::new(vec![token], device)?.reshape((1usize, 1usize))?;
        let embeds = model
            .llm
            .embed(&tok_t)
            .map_err(|e| candle_core::Error::Msg(format!("HunyuanOCR HSD step_one embed: {e}")))?;

        // Position id = current cache length (next slot).
        let pos = model.llm.current_kv_len() as i64;
        let pos_ids = Tensor::from_vec(vec![pos, pos, pos, pos], (4usize, 1usize, 1usize), device)?;

        // Continuation forward — no mask (autoregressive on growing cache).
        let hidden = model.llm.forward(&embeds, &pos_ids, None).map_err(|e| {
            candle_core::Error::Msg(format!("HunyuanOCR HSD step_one forward: {e}"))
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

        // Packed tree tokens: (1, N).
        let tok_t = Tensor::new(tree.tokens.clone(), device)?.reshape((1usize, n))?;
        let embeds = model.llm.embed(&tok_t).map_err(|e| {
            candle_core::Error::Msg(format!("HunyuanOCR HSD verify_tree embed: {e}"))
        })?;

        // Position ids: depth-`d` node represents the d-th newly generated
        // token, so its absolute sequence position is `prefix_kv + d - 1`
        // (depth-1 token sits in the very next cache slot after the prompt,
        // which is at index `prefix_kv`). Mirrors the baseline decode loop's
        // first `pos = seq_len` followed by per-step increments.
        let mut pos_data: Vec<i64> = Vec::with_capacity(4 * n);
        for _axis in 0..4 {
            for d in &tree.depths {
                pos_data.push(prefix_kv as i64 + (*d as i64) - 1);
            }
        }
        let pos_ids = Tensor::from_vec(pos_data, (4usize, 1usize, n), device)?;

        // Tree-ancestry mask — each candidate token sees prefix + its own
        // ancestor chain only (paper Fig. 2c).
        let mask = create_tree_attention_mask(&tree.parents, prefix_kv, dtype, device)?;

        let hidden = model
            .llm
            .forward(&embeds, &pos_ids, Some(&mask))
            .map_err(|e| {
                candle_core::Error::Msg(format!("HunyuanOCR HSD verify_tree forward: {e}"))
            })?;
        self.forward_passes += 1;
        // (1, N, hidden) → (N, hidden) → (N, vocab) log-probs.
        let h2 = hidden.squeeze(0)?;
        self.project_logprobs_2d(&h2)
    }

    fn commit_verify(&mut self, accepted_path: &[usize]) -> CandleResult<()> {
        let prefix_kv = self.pre_verify_kv;
        let mut indices: Vec<u32> = Vec::with_capacity(prefix_kv + accepted_path.len());
        for i in 0..prefix_kv {
            indices.push(i as u32);
        }
        for &p in accepted_path {
            indices.push((prefix_kv + p) as u32);
        }
        self.model.llm.keep_kv_indices(&indices).map_err(|e| {
            let mut chain = format!("HunyuanOCR HSD commit_verify: {e}");
            let mut cur: Option<&dyn std::error::Error> = std::error::Error::source(&e);
            while let Some(s) = cur {
                chain.push_str(&format!("\n      caused by: {s}"));
                cur = s.source();
            }
            candle_core::Error::Msg(chain)
        })
    }

    fn is_eos(&self, tok: u32) -> bool {
        self.model.stop_token_ids.contains(&tok)
    }
}

fn resolve_safetensors_shards(model_dir: &Path) -> Result<Vec<PathBuf>, OCRError> {
    let single = model_dir.join("model.safetensors");
    if single.exists() {
        return Ok(vec![single]);
    }

    let mut shards: Vec<PathBuf> = std::fs::read_dir(model_dir)?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .is_some_and(|s| s.starts_with("model-") && s.ends_with(".safetensors"))
        })
        .collect();
    shards.sort();
    if shards.is_empty() {
        return Err(OCRError::ConfigError {
            message: format!(
                "HunyuanOCR: no safetensors shards found in {}",
                model_dir.display()
            ),
        });
    }
    Ok(shards)
}

fn build_prompt(instruction: &str) -> String {
    // Matches the tokenizer chat_template in the model repo (empty system message).
    // The model expects the generation prompt to end with `<｜hy_User｜>`.
    format!(
        "<｜hy_begin▁of▁sentence｜><｜hy_place▁holder▁no▁3｜>\
<｜hy_place▁holder▁no▁100｜><｜hy_place▁holder▁no▁102｜><｜hy_place▁holder▁no▁101｜>{instruction}\
<｜hy_User｜>"
    )
}

fn expand_image_tokens_in_place(
    input_ids: &mut Vec<u32>,
    cfg: &HunyuanOcrConfig,
    image_inputs: &HunyuanOcrImageInputs,
) -> Result<(), OCRError> {
    let (_, hm, wm) = image_inputs.grid_thw_merged;
    // Match the upstream HuggingFace processor:
    //   transformers/models/hunyuan_vl/processing_hunyuan_vl.py:62
    //   num_image_tokens = patch_h * (patch_w + 1) + 2
    // The `+ 2` accounts for the begin/end markers that the vit's perceive
    // step prepends/appends to the spatial sequence — those positions also
    // get replaced by image embeddings (rather than carrying separate
    // `image_start` / `image_end` text embeddings, which is the scheme an
    // earlier internal Tencent variant used and which this Rust port
    // originally followed). The placeholder run is contiguous and uses
    // `image_token_id` exclusively — no `image_newline_token_id` interleaving.
    let expected_tokens = hm.saturating_mul(wm.saturating_add(1)).saturating_add(2);
    if expected_tokens == 0 {
        return Err(OCRError::InvalidInput {
            message: "HunyuanOCR: empty merged grid".to_string(),
        });
    }

    let mut placeholder = None;
    for i in 0..input_ids.len().saturating_sub(2) {
        if input_ids[i] == cfg.image_start_token_id
            && input_ids[i + 1] == cfg.image_token_id
            && input_ids[i + 2] == cfg.image_end_token_id
        {
            placeholder = Some(i + 1);
            break;
        }
    }
    let Some(pos) = placeholder else {
        return Err(OCRError::InvalidInput {
            message: "HunyuanOCR: prompt is missing image placeholder tokens".to_string(),
        });
    };

    let expanded: Vec<u32> = std::iter::repeat_n(cfg.image_token_id, expected_tokens).collect();
    // Replace the single image_token_id placeholder with the expanded run.
    // image_start_token_id and image_end_token_id stay in input_ids on
    // either side; they receive plain text embeddings via embed_tokens.
    input_ids.splice(pos..pos + 1, expanded);
    Ok(())
}

fn find_image_span(input_ids: &[u32], cfg: &HunyuanOcrConfig) -> Result<(usize, usize), OCRError> {
    let start_pos = input_ids
        .iter()
        .position(|&id| id == cfg.image_start_token_id)
        .ok_or_else(|| OCRError::InvalidInput {
            message: "HunyuanOCR: image_start_token_id not found in input_ids".to_string(),
        })?;
    let end_pos = input_ids[start_pos + 1..]
        .iter()
        .position(|&id| id == cfg.image_end_token_id)
        .map(|p| start_pos + 1 + p)
        .ok_or_else(|| OCRError::InvalidInput {
            message: "HunyuanOCR: image_end_token_id not found after image_start_token_id"
                .to_string(),
        })?;
    Ok((start_pos, end_pos))
}

fn build_position_ids(
    input_ids: &[u32],
    cfg: &HunyuanOcrConfig,
    image_inputs: &HunyuanOcrImageInputs,
) -> Result<Tensor, OCRError> {
    let seq_len = input_ids.len();
    // 4-axis XDRoPE position ids matching the upstream HF processor exactly:
    //   transformers/models/hunyuan_vl/processing_hunyuan_vl.py:74-94.
    //
    // Axis order is `[seq, w, h, t]` (the order `select_rope_sections`
    // expects for `xdrope_section`). For non-image tokens all four axes hold
    // the plain sequence index. For the spatial run inside the image span we
    // overwrite axes w/h/t:
    //   - w cycles `0..(patch_w+1)`, repeated `patch_h` times,
    //   - h is `[h]*(patch_w+1)` for `h` in `0..patch_h`,
    //   - t is 0 across the run.
    // The run starts at `first_image_token + 1` and spans `(patch_w+1)*patch_h`
    // tokens — i.e. the *middle* of the expanded `patch_h*(patch_w+1) + 2`
    // image-token block. The first and last image_tokens (perceive begin/end
    // markers) keep their default arange position.
    //
    // Earlier this port used pure-sequential position ids for all four axes,
    // which made the model produce hallucinated text (e.g. "The text in the
    // image is not complete.") instead of OCR output: the trained weights
    // expect the spatial xdrope rotation to encode 2-D image structure, and
    // collapsing to 1-D destroys the geometry.
    let mut pos: Vec<i64> = vec![0; 4 * seq_len];
    for i in 0..seq_len {
        let p = i as i64;
        pos[i] = p;
        pos[seq_len + i] = p;
        pos[2 * seq_len + i] = p;
        pos[3 * seq_len + i] = p;
    }

    let first_image_pos = input_ids.iter().position(|&id| id == cfg.image_token_id);
    if let Some(first) = first_image_pos {
        let (_, hm, wm) = image_inputs.grid_thw_merged;
        let start = first + 1;
        let replace_num = (wm + 1) * hm;
        if start + replace_num > seq_len {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "HunyuanOCR: image span ({} positions starting at {}) exceeds input length {}",
                    replace_num, start, seq_len
                ),
            });
        }
        for j in 0..replace_num {
            let idx = start + j;
            let row = j / (wm + 1); // 0..hm
            let col = j % (wm + 1); // 0..wm (inclusive of newline column)
            pos[seq_len + idx] = col as i64; // axis 1 = w
            pos[2 * seq_len + idx] = row as i64; // axis 2 = h
            pos[3 * seq_len + idx] = 0; // axis 3 = t
        }
    }

    Tensor::from_vec(
        pos,
        (4usize, 1usize, seq_len),
        image_inputs.pixel_values.device(),
    )
    .map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "HunyuanOCR: build position_ids tensor failed",
            e,
        )
    })
}
