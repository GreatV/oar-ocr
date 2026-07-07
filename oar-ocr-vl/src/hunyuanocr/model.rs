//! HunyuanOCR model wrapper (HunYuanVLForConditionalGeneration).

use super::config::{HunyuanOcrConfig, HunyuanOcrImageProcessorConfig};
use super::llm::HunyuanLlm;
use super::processing::{HunyuanOcrImageInputs, preprocess_image};
use super::vision::HunyuanVisionModel;
use crate::attention::{
    combine_masks, create_causal_mask, create_generation_mask, create_left_padding_mask,
};
use crate::utils::{candle_to_ocr_inference, candle_to_ocr_processing};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use image::RgbImage;
use oar_ocr_core::core::OCRError;
use std::path::Path;
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
    v.get("repetition_penalty")
        .and_then(|x| x.as_f64())
        .unwrap_or(1.0)
}

/// Read `generation_config.json::eos_token_id`. The official config provides
/// a list (e.g. `[120007, 120020]`). Returns `None` if the file is missing
/// or the field is absent.
fn load_generation_eos_ids(model_dir: &Path) -> Option<Vec<u32>> {
    let contents = std::fs::read_to_string(model_dir.join("generation_config.json")).ok()?;
    let v = serde_json::from_str::<serde_json::Value>(&contents).ok()?;
    let eos = v.get("eos_token_id")?;
    if let Some(single) = eos.as_u64() {
        u32::try_from(single).ok().map(|id| vec![id])
    } else {
        eos.as_array().map(|arr| {
            arr.iter()
                .filter_map(|x| x.as_u64().and_then(|v| u32::try_from(v).ok()))
                .collect()
        })
    }
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
    /// the value isn't applied.
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
        // Also include eos_token_ids from generation_config.json — the official
        // config lists [120007, 120020]; missing 120007 can cause the model to
        // overshoot past a valid stop point.
        if let Some(gen_eos) = load_generation_eos_ids(model_dir) {
            stop_token_ids.extend(gen_eos);
        }
        stop_token_ids.sort_unstable();
        stop_token_ids.dedup();

        let dtype = crate::utils::select_dtype(&device);

        let weight_files = crate::utils::collect_safetensors(model_dir, "HunyuanOCR")?;
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

        match self.generate_tokens_internal(images, instructions, max_new_tokens) {
            Ok(results) => results
                .into_iter()
                .map(|tokens| self.decode_generated_tokens(&tokens))
                .collect(),
            Err(e) => {
                let msg = crate::utils::error_chain_message("generation failed", &e);
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
                let msg = crate::utils::error_chain_message("generation failed", &e);
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

        // 1. Preprocess all images and build prompts
        let mut all_input_ids: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
        let mut all_image_inputs: Vec<HunyuanOcrImageInputs> = Vec::with_capacity(batch_size);

        for (image, instruction) in images.iter().zip(instructions.iter()) {
            let instruction = instruction.as_ref();
            let image_inputs = preprocess_image(
                image,
                &self.image_cfg,
                &self.cfg.vision_config,
                &self.device,
                self.dtype,
            )?;

            let prompt = build_prompt(instruction);
            let enc = self
                .tokenizer
                .encode(prompt, false)
                .map_err(|e| OCRError::InvalidInput {
                    message: format!("HunyuanOCR: tokenizer encode failed: {e}"),
                })?;

            let mut input_ids = enc.get_ids().to_vec();
            expand_image_tokens_in_place(&mut input_ids, &self.cfg, &image_inputs)?;

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

            // Fuse image embeddings into the token embedding sequence.
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
            logits_list.push(logits);
        }

        // 7. Autoregressive decode
        let mut generated: Vec<Vec<u32>> = vec![Vec::new(); batch_size];
        let mut finished: Vec<bool> = vec![false; batch_size];
        let mut positions: Vec<i64> = seq_lens.iter().map(|&len| len as i64).collect();

        // Left-padding lengths per row, and current KV-cache length (grows by one
        // each decode step). Used to mask out padding KV during generation so a
        // batch with unequal prompt lengths does not attend to padding positions.
        let pad_lens: Vec<usize> = seq_lens.iter().map(|&len| max_seq_len - len).collect();
        let mut kv_len = max_seq_len;

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

            // Mask out left-padding positions in the KV cache for this step.
            kv_len += 1;
            let gen_mask = create_generation_mask(&pad_lens, kv_len, self.dtype, &self.device)
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "create gen mask", e))?;

            let hs = self.llm.forward(&embeds, &pos, Some(&gen_mask))?;

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

    /// Decode tokens in the form the model actually emitted. HunyuanOCR's
    /// `decode_tokens` only applies `trim()` post-process, so this is
    /// effectively an alias provided for API symmetry with backends that do
    /// have non-trivial post-process (PaddleOCR-VL / GLM-OCR).
    pub fn decode_tokens_raw(&self, tokens: &[u32]) -> Result<String, OCRError> {
        self.tokenizer
            .decode(tokens, true)
            .map_err(|e| OCRError::InvalidInput {
                message: format!("HunyuanOCR: tokenizer decode failed: {e}"),
            })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn decode_generated_tokens(&self, tokens: &[u32]) -> Result<String, OCRError> {
        Ok(self.decode_tokens_raw(tokens)?.trim().to_string())
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
    // Upstream processor: num_image_tokens = patch_h * (patch_w + 1) + 2
    // (processing_hunyuan_vl.py:62). The `+ 2` covers the perceive step's
    // begin/end markers, whose positions are also replaced by image
    // embeddings. The placeholder run is contiguous `image_token_id` only —
    // no `image_newline_token_id` interleaving.
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
    // tokens — the *middle* of the expanded `patch_h*(patch_w+1) + 2` block;
    // the perceive begin/end markers keep their default arange position.
    // Collapsing the spatial axes to plain 1-D sequence ids destroys the 2-D
    // geometry the trained weights expect and yields hallucinated text.
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
