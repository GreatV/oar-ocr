use super::config::{EosTokenId, GlmOcrConfig, GlmOcrImageProcessorConfig};
use super::processing::{GlmOcrImageInputs, preprocess_image};
use super::text::GlmOcrTextModel;
use super::vision::GlmOcrVisionModel;
use crate::utils::{
    candle_to_ocr_inference, candle_to_ocr_processing, truncate_repetitive_content,
};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};
use image::RgbImage;
use oar_ocr_core::core::OCRError;
use std::path::Path;
use tokenizers::Tokenizer;

pub struct GlmOcr {
    device: Device,
    dtype: DType,
    cfg: GlmOcrConfig,
    image_cfg: GlmOcrImageProcessorConfig,
    tokenizer: Tokenizer,
    text: GlmOcrTextModel,
    vision: GlmOcrVisionModel,
    lm_head: Linear,
    eos_token_ids: Vec<u32>,
    image_token_id: u32,
}

impl GlmOcr {
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, OCRError> {
        let model_dir = model_dir.as_ref();
        let cfg = GlmOcrConfig::from_path(model_dir.join("config.json"))?;
        let image_cfg =
            GlmOcrImageProcessorConfig::from_path(model_dir.join("preprocessor_config.json"))?;

        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| {
            OCRError::ConfigError {
                message: format!("failed to load GLM-OCR tokenizer.json: {e}"),
            }
        })?;

        if let Some(tok_image_id) = tokenizer.token_to_id("<|image|>")
            && tok_image_id != cfg.image_token_id
        {
            return Err(OCRError::ConfigError {
                message: format!(
                    "GLM-OCR image_token_id mismatch: tokenizer {tok_image_id} != config {}",
                    cfg.image_token_id
                ),
            });
        }

        let dtype = device.bf16_default_to_f32();
        // SAFETY: The mmap'd file must not be modified or deleted while in use.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_dir.join("model.safetensors")],
                dtype,
                &device,
            )
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "load model.safetensors", e))?
        };

        let image_token_id = cfg.image_token_id;
        let text = GlmOcrTextModel::load(&cfg.text_config, vb.pp("model").pp("language_model"))?;
        let vision = GlmOcrVisionModel::load(&cfg.vision_config, vb.pp("model").pp("visual"))?;

        let lm_head = if cfg.text_config.tie_word_embeddings || cfg.tie_word_embeddings {
            Linear::new(text.token_embedding_weight(), None)
        } else {
            linear_no_bias(
                cfg.text_config.hidden_size,
                cfg.text_config.vocab_size,
                vb.pp("lm_head"),
            )
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "load lm_head", e))?
        };

        let eos_token_ids = match &cfg.text_config.eos_token_id {
            EosTokenId::Single(v) => vec![*v],
            EosTokenId::Multiple(v) => v.clone(),
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
            eos_token_ids,
            image_token_id,
        })
    }

    /// Generate OCR output for one or more images with optional instructions.
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
                    "GLM-OCR: images count ({}) != instructions count ({})",
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

    fn generate_internal(
        &self,
        images: &[RgbImage],
        instructions: &[impl AsRef<str>],
        max_new_tokens: usize,
    ) -> Result<Vec<String>, OCRError> {
        let mut results = Vec::with_capacity(images.len());

        for (image, instruction) in images.iter().zip(instructions.iter()) {
            let image_inputs = preprocess_image(
                image,
                &self.image_cfg,
                &self.cfg.vision_config,
                &self.device,
                self.dtype,
            )?;

            let prompt = build_prompt(instruction.as_ref());
            let prompt = expand_image_tokens(&prompt, image_inputs.num_image_tokens)?;

            let enc = self
                .tokenizer
                .encode(prompt, false)
                .map_err(|e| OCRError::InvalidInput {
                    message: format!("GLM-OCR: tokenizer encode failed: {e}"),
                })?;
            let input_ids = enc.get_ids().to_vec();
            if input_ids.is_empty() {
                return Err(OCRError::InvalidInput {
                    message: "GLM-OCR: empty prompt after tokenization".to_string(),
                });
            }

            let seq_len = input_ids.len();
            let inputs_embeds = self.prepare_inputs(&input_ids, &image_inputs)?;
            let (position_ids, max_pos) = build_position_ids(
                &input_ids,
                image_inputs.grid_thw,
                self.cfg.vision_config.spatial_merge_size,
                self.image_token_id,
                &self.device,
            )?;
            let rope_delta = max_pos + 1 - seq_len as i64;

            self.text.clear_kv_cache();
            let hidden = self.text.forward(&inputs_embeds, &position_ids, None)?;
            let last = hidden.i((0, seq_len - 1, ..)).map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: get last hidden",
                    e,
                )
            })?;
            let mut logits = self.logits_from_hidden(&last)?;

            let mut generated: Vec<u32> = Vec::new();
            let mut pos = seq_len as i64;

            for _ in 0..max_new_tokens {
                let tok = logits
                    .argmax(D::Minus1)
                    .and_then(|t| t.to_scalar::<u32>())
                    .map_err(|e| {
                        candle_to_ocr_processing(
                            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                            "GLM-OCR: argmax",
                            e,
                        )
                    })?;
                if self.eos_token_ids.contains(&tok) {
                    break;
                }
                generated.push(tok);

                let token = Tensor::new(&[tok], &self.device)
                    .and_then(|t| t.reshape((1, 1)))
                    .map_err(|e| {
                        candle_to_ocr_processing(
                            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                            "GLM-OCR: create next token",
                            e,
                        )
                    })?;
                let embed = self.text.embed(&token)?;

                let pos_val = pos + rope_delta;
                let pos_ids = Tensor::new(vec![pos_val, pos_val, pos_val], &self.device)
                    .and_then(|t| t.reshape((3, 1, 1)))
                    .map_err(|e| {
                        candle_to_ocr_processing(
                            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                            "GLM-OCR: create position ids",
                            e,
                        )
                    })?;

                let hs = self.text.forward(&embed, &pos_ids, None)?;
                let last = hs.i((0, 0, ..)).map_err(|e| {
                    candle_to_ocr_processing(
                        oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                        "GLM-OCR: get decode hidden",
                        e,
                    )
                })?;
                logits = self.logits_from_hidden(&last)?;

                pos += 1;
            }

            let decoded =
                self.tokenizer
                    .decode(&generated, true)
                    .map_err(|e| OCRError::InvalidInput {
                        message: format!("GLM-OCR: tokenizer decode failed: {e}"),
                    })?;
            let decoded = truncate_repetitive_content(&decoded, 10, 10, 10);
            results.push(decoded.trim().to_string());
        }

        Ok(results)
    }

    fn prepare_inputs(
        &self,
        input_ids: &[u32],
        image_inputs: &GlmOcrImageInputs,
    ) -> Result<Tensor, OCRError> {
        let seq_len = input_ids.len();
        let token_ids =
            Tensor::from_vec(input_ids.to_vec(), (1, seq_len), &self.device).map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: create input_ids tensor",
                    e,
                )
            })?;
        let mut embeds = self.text.embed(&token_ids)?;

        let image_embeds = self
            .vision
            .forward(&image_inputs.pixel_values, image_inputs.grid_thw)?;
        let image_embeds = image_embeds.to_dtype(self.dtype).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: cast image_embeds",
                e,
            )
        })?;

        let indices: Vec<usize> = input_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| {
                if id == self.image_token_id {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        let image_embeds_len = image_embeds.dim(0).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: image_embeds dim",
                e,
            )
        })?;
        if indices.len() != image_embeds_len {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "GLM-OCR: image token count ({}) != image embeds ({})",
                    indices.len(),
                    image_embeds_len
                ),
            });
        }
        if indices.is_empty() {
            return Err(OCRError::InvalidInput {
                message: "GLM-OCR: no image tokens found in prompt".to_string(),
            });
        }
        let start = indices[0];
        let end = *indices.last().unwrap();
        if end + 1 - start != indices.len() {
            return Err(OCRError::InvalidInput {
                message: "GLM-OCR: image tokens are not contiguous in prompt".to_string(),
            });
        }

        let hidden_size = embeds.dim(2).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: embed hidden_size",
                e,
            )
        })?;

        let prefix = if start > 0 {
            embeds.narrow(1, 0, start).map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: embed prefix narrow",
                    e,
                )
            })?
        } else {
            Tensor::zeros((1, 0, hidden_size), embeds.dtype(), embeds.device()).map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: embed prefix zeros",
                    e,
                )
            })?
        };

        let suffix_start = end + 1;
        let suffix = if suffix_start < seq_len {
            embeds
                .narrow(1, suffix_start, seq_len - suffix_start)
                .map_err(|e| {
                    candle_to_ocr_processing(
                        oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                        "GLM-OCR: embed suffix narrow",
                        e,
                    )
                })?
        } else {
            Tensor::zeros((1, 0, hidden_size), embeds.dtype(), embeds.device()).map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: embed suffix zeros",
                    e,
                )
            })?
        };

        let image_embeds = image_embeds.unsqueeze(0).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: image embeds unsqueeze",
                e,
            )
        })?;
        embeds = Tensor::cat(&[&prefix, &image_embeds, &suffix], 1).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: embed cat",
                e,
            )
        })?;

        Ok(embeds)
    }

    fn logits_from_hidden(&self, hidden: &Tensor) -> Result<Tensor, OCRError> {
        let hidden = hidden.unsqueeze(0).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: logits unsqueeze",
                e,
            )
        })?;
        let logits = self
            .lm_head
            .forward(&hidden)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "lm_head forward", e))?;
        logits.squeeze(0).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: logits squeeze",
                e,
            )
        })
    }
}

fn build_prompt(instruction: &str) -> String {
    format!(
        "[gMASK]<sop><|user|>\n<|begin_of_image|><|image|><|end_of_image|>{instruction}<|assistant|>\n"
    )
}

fn expand_image_tokens(prompt: &str, num_image_tokens: usize) -> Result<String, OCRError> {
    if num_image_tokens == 0 {
        return Err(OCRError::InvalidInput {
            message: "GLM-OCR: num_image_tokens is zero".to_string(),
        });
    }
    if !prompt.contains("<|image|>") {
        return Err(OCRError::InvalidInput {
            message: "GLM-OCR: prompt missing <|image|> token".to_string(),
        });
    }
    let placeholder = "<|placeholder|>";
    let repeated = placeholder.repeat(num_image_tokens);
    let expanded = prompt.replacen("<|image|>", &repeated, 1);
    Ok(expanded.replace(placeholder, "<|image|>"))
}

/// Build position IDs tensor and compute max position value.
///
/// Returns `(position_ids, max_pos)` where `max_pos` is used to compute rope_delta.
fn build_position_ids(
    input_ids: &[u32],
    grid_thw: (usize, usize, usize),
    merge_size: usize,
    image_token_id: u32,
    device: &Device,
) -> Result<(Tensor, i64), OCRError> {
    let (mut pos_t, mut pos_h, mut pos_w) = (Vec::new(), Vec::new(), Vec::new());
    let mut max_pos: i64 = -1;

    let mut current = input_ids.first().copied().unwrap_or(image_token_id) == image_token_id;
    let mut start = 0usize;
    let mut groups = Vec::new();
    for (i, &id) in input_ids.iter().enumerate() {
        let is_image = id == image_token_id;
        if i == 0 {
            current = is_image;
            continue;
        }
        if is_image != current {
            groups.push((current, start, i));
            start = i;
            current = is_image;
        }
    }
    groups.push((current, start, input_ids.len()));

    let (grid_t, grid_h, grid_w) = grid_thw;
    let llm_grid_h = grid_h / merge_size;
    let llm_grid_w = grid_w / merge_size;
    let llm_grid_t = grid_t;
    let mut seen_image = false;

    let expected_image_tokens = llm_grid_t * llm_grid_h * llm_grid_w;
    for (is_image, s, e) in groups {
        let st_idx = max_pos + 1;
        if is_image {
            if seen_image {
                return Err(OCRError::InvalidInput {
                    message: "GLM-OCR: multiple image groups in prompt are not supported"
                        .to_string(),
                });
            }
            seen_image = true;
            let group_len = e - s;
            if group_len != expected_image_tokens {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "GLM-OCR: image token count ({group_len}) != expected ({expected_image_tokens})"
                    ),
                });
            }
            for t in 0..llm_grid_t {
                for h in 0..llm_grid_h {
                    for w in 0..llm_grid_w {
                        pos_t.push(t as i64 + st_idx);
                        pos_h.push(h as i64 + st_idx);
                        pos_w.push(w as i64 + st_idx);
                    }
                }
            }
            let group_max = (llm_grid_t.max(llm_grid_h).max(llm_grid_w) as i64) - 1 + st_idx;
            max_pos = group_max;
        } else {
            let len = e - s;
            for i in 0..len {
                let v = i as i64 + st_idx;
                pos_t.push(v);
                pos_h.push(v);
                pos_w.push(v);
            }
            max_pos = st_idx + len as i64 - 1;
        }
    }

    let seq_len = input_ids.len();
    if pos_t.len() != seq_len {
        return Err(OCRError::InvalidInput {
            message: format!(
                "GLM-OCR: position_ids length mismatch: {} vs {}",
                pos_t.len(),
                seq_len
            ),
        });
    }

    let mut data = Vec::with_capacity(3 * seq_len);
    data.extend(pos_t);
    data.extend(pos_h);
    data.extend(pos_w);

    let position_ids = Tensor::from_vec(data, (3, 1, seq_len), device).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: position_ids tensor",
            e,
        )
    })?;
    Ok((position_ids, max_pos))
}
