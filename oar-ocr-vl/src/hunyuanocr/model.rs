//! HunyuanOCR model wrapper (HunYuanVLForConditionalGeneration).

use super::config::{HunyuanOcrConfig, HunyuanOcrImageProcessorConfig};
use super::llm::{HunyuanLlm, KvCache};
use super::processing::{HunyuanOcrImageInputs, preprocess_image};
use super::vision::HunyuanVisionModel;
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

    /// Generate OCR output for an image with a custom instruction.
    pub fn generate(
        &self,
        image: RgbImage,
        instruction: impl AsRef<str>,
        max_new_tokens: usize,
    ) -> Result<String, OCRError> {
        let instruction = instruction.as_ref();

        let (input_ids, image_inputs, inputs_embeds) =
            self.prepare_inputs_embeds(&image, instruction)?;

        let (mut kv_cache, mut logits) = self.run_prefill(&inputs_embeds, &input_ids, &image_inputs)?;

        let mut generated: Vec<u32> = Vec::new();
        let mut next_pos = input_ids.len() as i64;

        for _ in 0..max_new_tokens {
            let next_token = logits
                .argmax(D::Minus1)
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "argmax", e))?
                .to_scalar::<u32>()
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "argmax to_scalar", e))?;

            if self.stop_token_ids.contains(&next_token) {
                break;
            }
            generated.push(next_token);

            let token_t = Tensor::new(&[next_token], &self.device).map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "HunyuanOCR: create next_token tensor failed",
                    e,
                )
            })?;
            let token_t = token_t.reshape((1usize, 1usize)).map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "HunyuanOCR: reshape next_token tensor failed",
                    e,
                )
            })?;
            let token_embed = self.llm.embed(&token_t)?;

            // For text-only generation, use the same 1D position index for all 4 RoPE dimensions.
            let pos_ids = Tensor::new(&[next_pos, next_pos, next_pos, next_pos], &self.device)
                .map_err(|e| {
                    candle_to_ocr_processing(
                        oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                        "HunyuanOCR: create pos tensor failed",
                        e,
                    )
                })?;
            let pos_ids = pos_ids.reshape((4usize, 1usize, 1usize)).map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "HunyuanOCR: reshape pos tensor failed",
                    e,
                )
            })?;

            let hs = self
                .llm
                .forward(&token_embed, &pos_ids, Some(&mut kv_cache), None)?;
            let hs = hs.squeeze(0).map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "HunyuanOCR: squeeze hs batch failed",
                    e,
                )
            })?;
            let hs = hs.squeeze(0).map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "HunyuanOCR: squeeze hs seq failed",
                    e,
                )
            })?;
            logits = self.logits_from_hidden(&hs)?;
            next_pos += 1;
        }

        let decoded =
            self.tokenizer
                .decode(&generated, true)
                .map_err(|e| OCRError::InvalidInput {
                    message: format!("HunyuanOCR: tokenizer decode failed: {e}"),
                })?;
        Ok(decoded.trim().to_string())
    }

    /// Prepare input embeddings from image and instruction.
    fn prepare_inputs_embeds(
        &self,
        image: &RgbImage,
        instruction: &str,
    ) -> Result<(Vec<u32>, HunyuanOcrImageInputs, Tensor), OCRError> {
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

        let input_ids_t = Tensor::new(input_ids.clone(), &self.device).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: create input_ids tensor failed",
                e,
            )
        })?;
        let input_ids_t = input_ids_t
            .reshape((1usize, input_ids.len()))
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "HunyuanOCR: reshape input_ids tensor failed",
                    e,
                )
            })?;

        let token_embeds = self.llm.embed(&input_ids_t)?;

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

        let (start_pos, end_pos) = find_image_span(&input_ids, &self.cfg)?;
        let region_len = end_pos - start_pos + 1;
        let (img_len, img_dim) = image_embeds.dims2().map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: image_embeds dims2 failed",
                e,
            )
        })?;
        if img_dim != self.cfg.hidden_size {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "HunyuanOCR: image_embeds hidden dim mismatch: got {img_dim}, expected {}",
                    self.cfg.hidden_size
                ),
            });
        }
        if region_len != img_len {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "HunyuanOCR: image span length mismatch: tokens={region_len} embeds={img_len}"
                ),
            });
        }

        let token_embeds = token_embeds.squeeze(0).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: squeeze token embeddings failed",
                e,
            )
        })?;

        let mut parts: Vec<Tensor> = Vec::with_capacity(3);
        if start_pos > 0 {
            parts.push(token_embeds.i((0..start_pos, ..)).map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "HunyuanOCR: slice prefix token embeddings failed",
                    e,
                )
            })?);
        }
        parts.push(image_embeds);
        if end_pos + 1 < input_ids.len() {
            parts.push(
                token_embeds
                    .i((end_pos + 1..input_ids.len(), ..))
                    .map_err(|e| {
                        candle_to_ocr_processing(
                            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                            "HunyuanOCR: slice suffix token embeddings failed",
                            e,
                        )
                    })?,
            );
        }
        let refs: Vec<&Tensor> = parts.iter().collect();
        let inputs_embeds = Tensor::cat(&refs, 0).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: concat input embeddings failed",
                e,
            )
        })?;
        let inputs_embeds = inputs_embeds.unsqueeze(0).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: add batch dim to embeddings failed",
                e,
            )
        })?;

        Ok((input_ids, image_inputs, inputs_embeds))
    }

    /// Run the prefill step: build attention components and populate KV cache.
    fn run_prefill(
        &self,
        inputs_embeds: &Tensor,
        input_ids: &[u32],
        image_inputs: &HunyuanOcrImageInputs,
    ) -> Result<(Vec<KvCache>, Tensor), OCRError> {
        let position_ids = build_position_ids(input_ids, &self.cfg, image_inputs)?;
        let prefill_causal_mask =
            causal_mask(input_ids.len(), &self.device, inputs_embeds.dtype())?;

        let mut kv_cache = vec![KvCache::default(); self.cfg.num_hidden_layers];
        let hidden = self.llm.forward(
            inputs_embeds,
            &position_ids,
            Some(&mut kv_cache),
            Some(&prefill_causal_mask),
        )?;

        let last = hidden.i((0, input_ids.len() - 1, ..)).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: select last hidden state failed",
                e,
            )
        })?;
        let logits = self.logits_from_hidden(&last)?;

        Ok((kv_cache, logits))
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

fn causal_mask(seq_len: usize, device: &Device, dtype: DType) -> Result<Tensor, OCRError> {
    let mut data = Vec::with_capacity(seq_len * seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j <= i {
                data.push(0f32);
            } else {
                data.push(f32::NEG_INFINITY);
            }
        }
    }
    let tensor =
        Tensor::from_vec(data, (1usize, 1usize, seq_len, seq_len), device).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: build causal mask failed",
                e,
            )
        })?;
    tensor.to_dtype(dtype).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "HunyuanOCR: cast causal mask failed",
            e,
        )
    })
}
