//! HunyuanOCR model wrapper (HunYuanVLForConditionalGeneration).

use super::config::{HunyuanOcrConfig, HunyuanOcrImageProcessorConfig};
use super::llm::HunyuanLlm;
use super::processing::{HunyuanOcrImageInputs, preprocess_image};
use super::vision::HunyuanVisionModel;
use crate::attention::{combine_masks, create_causal_mask, create_left_padding_mask};
use crate::utils::{candle_to_ocr_inference, candle_to_ocr_processing};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use image::RgbImage;
use oar_ocr_core::core::OCRError;
use std::path::{Path, PathBuf};
use tokenizers::Tokenizer;

pub struct HunyuanOcr {
    device: Device,
    dtype: DType,
    cfg: HunyuanOcrConfig,
    image_cfg: HunyuanOcrImageProcessorConfig,
    tokenizer: Tokenizer,
    llm: HunyuanLlm,
    vision: HunyuanVisionModel,
    stop_token_ids: Vec<u32>,
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

        Ok(Self {
            device,
            dtype,
            cfg,
            image_cfg,
            tokenizer,
            llm,
            vision,
            stop_token_ids,
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

        match self.generate_internal(images, instructions, max_new_tokens) {
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
    fn generate_internal(
        &self,
        images: &[RgbImage],
        instructions: &[impl AsRef<str>],
        max_new_tokens: usize,
    ) -> Result<Vec<String>, OCRError> {
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

            // Fuse image embeddings
            let (start_pos, end_pos) = find_image_span(input_ids, &self.cfg)?;
            let region_len = end_pos - start_pos + 1;
            let (img_len, _) = image_embeds
                .dims2()
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "image_embeds dims2", e))?;
            if region_len != img_len {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "HunyuanOCR: image span length mismatch: tokens={region_len} embeds={img_len}"
                    ),
                });
            }

            let token_embeds = token_embeds.squeeze(0).map_err(|e| {
                candle_to_ocr_inference("HunyuanOCR", "squeeze token embeddings", e)
            })?;

            let mut parts: Vec<Tensor> = Vec::with_capacity(3);
            if start_pos > 0 {
                parts.push(token_embeds.i((0..start_pos, ..)).map_err(|e| {
                    candle_to_ocr_inference("HunyuanOCR", "slice prefix embeddings", e)
                })?);
            }
            parts.push(image_embeds);
            if end_pos + 1 < input_ids.len() {
                parts.push(
                    token_embeds
                        .i((end_pos + 1..input_ids.len(), ..))
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

        for _ in 0..max_new_tokens {
            if finished.iter().all(|&f| f) {
                break;
            }

            let mut next_tokens: Vec<u32> = Vec::with_capacity(batch_size);
            for (i, logits) in logits_list.iter().enumerate() {
                if finished[i] {
                    next_tokens.push(0); // Padding token for finished samples
                } else {
                    let tok = logits
                        .argmax(D::Minus1)
                        .and_then(|t| t.to_scalar::<u32>())
                        .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "argmax", e))?;

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

        // 8. Decode results
        let mut results = Vec::with_capacity(batch_size);
        for tokens in generated {
            let decoded =
                self.tokenizer
                    .decode(&tokens, true)
                    .map_err(|e| OCRError::InvalidInput {
                        message: format!("HunyuanOCR: tokenizer decode failed: {e}"),
                    })?;
            results.push(decoded.trim().to_string());
        }

        Ok(results)
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
    let expected_tokens = hm.saturating_mul(wm.saturating_add(1));
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

    let mut expanded: Vec<u32> = Vec::with_capacity(expected_tokens);
    for _r in 0..hm {
        expanded.extend(std::iter::repeat_n(cfg.image_token_id, wm));
        expanded.push(cfg.image_newline_token_id);
    }

    // Replace the single placeholder image_token_id with the expanded sequence.
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
    let mut pos: Vec<i64> = vec![0; 4 * seq_len];
    for i in 0..seq_len {
        let p = i as i64;
        pos[i] = p;
        pos[seq_len + i] = p;
        pos[2 * seq_len + i] = p;
        pos[3 * seq_len + i] = p;
    }

    let (start_pos, _end_pos) = find_image_span(input_ids, cfg)?;
    let vision_start = input_ids[start_pos + 1..]
        .iter()
        .position(|&id| id == cfg.image_token_id)
        .map(|p| start_pos + 1 + p)
        .ok_or_else(|| OCRError::InvalidInput {
            message: "HunyuanOCR: image_token_id not found after image_start_token_id".to_string(),
        })?;

    let (_, hm, wm) = image_inputs.grid_thw_merged;
    let vision_tokens = hm.saturating_mul(wm.saturating_add(1));
    let base = vision_start as i64;

    for j in 0..vision_tokens {
        let idx = vision_start + j;
        if idx >= seq_len {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "HunyuanOCR: vision token span exceeds input length (start={vision_start} count={vision_tokens} len={seq_len})"
                ),
            });
        }
        let row = j / (wm + 1);
        let col = j % (wm + 1);
        let t_pos = base;
        let h_pos = base + row as i64;
        let w_pos = base + col as i64;
        pos[seq_len + idx] = t_pos;
        pos[2 * seq_len + idx] = h_pos;
        pos[3 * seq_len + idx] = w_pos;
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
