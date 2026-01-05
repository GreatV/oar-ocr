//! PaddleOCR-VL (Vision-Language) model implementation.

use super::config::{PaddleOcrVlConfig, PaddleOcrVlImageProcessorConfig};
use super::ernie::{Ernie4_5Model, KvCache};
use super::processing;
use super::projector::Projector;
use super::vision::VisionModel;
use crate::core::OCRError;
use crate::vl::utils::{candle_to_ocr_inference, candle_to_ocr_processing};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::Module;
use image::RgbImage;
use std::path::Path;
use tokenizers::Tokenizer;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PaddleOcrVlTask {
    Ocr,
    Table,
    Chart,
    Formula,
}

impl PaddleOcrVlTask {
    pub fn prompt(self) -> &'static str {
        match self {
            Self::Ocr => "OCR:",
            Self::Table => "Table Recognition:",
            Self::Chart => "Chart Recognition:",
            Self::Formula => "Formula Recognition:",
        }
    }

    pub fn postprocess(self, text: String) -> String {
        let trimmed = text.trim();
        match self {
            Self::Formula => processing::strip_math_wrappers(trimmed).to_string(),
            Self::Table => processing::postprocess_table_output(trimmed),
            Self::Ocr | Self::Chart => trimmed.to_string(),
        }
    }
}

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
}

impl PaddleOcrVl {
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, OCRError> {
        let model_dir = model_dir.as_ref();
        let cfg = PaddleOcrVlConfig::from_path(model_dir.join("config.json"))?;
        let image_cfg =
            PaddleOcrVlImageProcessorConfig::from_path(model_dir.join("preprocessor_config.json"))?;

        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| {
            OCRError::config_error(format!("failed to load PaddleOCR-VL tokenizer.json: {e}"))
        })?;

        let eos_token_id = tokenizer.token_to_id("</s>").ok_or_else(|| {
            OCRError::config_error("PaddleOCR-VL: tokenizer is missing </s> token")
        })?;
        let sep_token_id = tokenizer.token_to_id("<|end_of_sentence|>");

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
        })
    }

    pub fn generate(
        &self,
        image: RgbImage,
        task: PaddleOcrVlTask,
        max_new_tokens: usize,
    ) -> Result<String, OCRError> {
        self.generate_with_raw(image, task, max_new_tokens)
            .map(|(_, processed)| processed)
    }

    pub fn generate_with_raw(
        &self,
        image: RgbImage,
        task: PaddleOcrVlTask,
        max_new_tokens: usize,
    ) -> Result<(String, String), OCRError> {
        let image_inputs =
            processing::preprocess_images(&[image], &self.image_cfg, &self.device, self.dtype)?;
        let (t, h, w) = image_inputs.image_grid_thw[0];
        let image_token_repeat =
            (t * h * w) / (self.image_cfg.merge_size * self.image_cfg.merge_size);
        let image_tokens = "<|IMAGE_PLACEHOLDER|>".repeat(image_token_repeat);
        // Note: The chat template includes <|begin_of_sentence|> at the start
        let prompt = format!(
            "<|begin_of_sentence|>User: <|IMAGE_START|>{image_tokens}<|IMAGE_END|>{}\nAssistant: ",
            task.prompt()
        );

        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| OCRError::InvalidInput {
                message: format!("PaddleOCR-VL: tokenizer encode failed: {e}"),
            })?;
        let input_ids = enc.get_ids().to_vec();
        let prompt_len = input_ids.len();

        let input_ids_t = Tensor::new(input_ids.clone(), &self.device).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: create input_ids tensor failed",
                e,
            )
        })?;
        let input_ids_t = input_ids_t.reshape((1usize, prompt_len)).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: reshape input_ids tensor failed",
                e,
            )
        })?;

        let inputs_embeds = self.llm.embed(&input_ids_t)?;

        let vision_feats = self
            .vision
            .forward(&image_inputs.pixel_values, &image_inputs.image_grid_thw)?;

        let image_embeds = self
            .projector
            .forward(&vision_feats, &image_inputs.image_grid_thw)?;

        let image_token_positions: Vec<usize> = input_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| (id == self.cfg.image_token_id).then_some(i))
            .collect();
        if image_token_positions.len() != image_embeds.dims()[0] {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "PaddleOCR-VL: image token count ({}) != image embeds ({})",
                    image_token_positions.len(),
                    image_embeds.dims()[0]
                ),
            });
        }

        let token_embeds = inputs_embeds.squeeze(0).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: squeeze token embeddings failed",
                e,
            )
        })?;

        let mut parts: Vec<Tensor> = Vec::with_capacity(image_token_positions.len() * 2 + 1);
        let mut cursor = 0usize;
        for (img_idx, &pos) in image_token_positions.iter().enumerate() {
            if cursor < pos {
                parts.push(token_embeds.i((cursor..pos, ..)).map_err(|e| {
                    candle_to_ocr_processing(
                        crate::core::errors::ProcessingStage::TensorOperation,
                        "PaddleOCR-VL: slice token embeddings failed",
                        e,
                    )
                })?);
            }
            let img_row = image_embeds.i((img_idx, ..)).map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "PaddleOCR-VL: slice image embeddings failed",
                    e,
                )
            })?;
            let img_row = img_row.unsqueeze(0).map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "PaddleOCR-VL: unsqueeze image embedding failed",
                    e,
                )
            })?;
            parts.push(img_row);
            cursor = pos + 1;
        }
        if cursor < prompt_len {
            parts.push(token_embeds.i((cursor..prompt_len, ..)).map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "PaddleOCR-VL: slice tail token embeddings failed",
                    e,
                )
            })?);
        }
        let refs: Vec<&Tensor> = parts.iter().collect();

        let inputs_embeds = Tensor::cat(&refs, 0).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: concat embeddings failed",
                e,
            )
        })?;

        let inputs_embeds = inputs_embeds.unsqueeze(0).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: unsqueeze batch for embeddings failed",
                e,
            )
        })?;

        let (position_ids, rope_deltas) = get_rope_index(
            &self.cfg,
            &input_ids,
            &image_inputs.image_grid_thw,
            &self.device,
        )?;

        let prefill_causal_mask = causal_mask(prompt_len, &self.device, inputs_embeds.dtype())?;

        let mut kv_cache = vec![KvCache::default(); self.cfg.num_hidden_layers];

        let hidden = self.llm.forward(
            &inputs_embeds,
            &position_ids,
            Some(&mut kv_cache),
            Some(&prefill_causal_mask),
        )?;

        let last = hidden.i((0, prompt_len - 1, ..)).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: select last hidden state failed",
                e,
            )
        })?;
        let last = last.unsqueeze(0).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: unsqueeze last hidden state failed",
                e,
            )
        })?;
        let logits = self
            .lm_head
            .forward(&last)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "lm_head forward", e))?;
        let mut logits = logits
            .squeeze(0)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "lm_head squeeze", e))?;

        let mut generated: Vec<u32> = Vec::new();
        let mut next_pos = (prompt_len as i64) + rope_deltas;
        for _ in 0..max_new_tokens {
            let next_token = logits
                .argmax(D::Minus1)
                .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "argmax", e))?;
            let next_token = next_token
                .to_scalar::<u32>()
                .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "argmax to_scalar", e))?;

            if next_token == self.eos_token_id {
                break;
            }
            if self.sep_token_id.is_some_and(|id| id == next_token) {
                break;
            }

            generated.push(next_token);

            let token_t = Tensor::new(&[next_token], &self.device).map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "PaddleOCR-VL: create next_token tensor failed",
                    e,
                )
            })?;
            let token_t = token_t.reshape((1usize, 1usize)).map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "PaddleOCR-VL: reshape next_token tensor failed",
                    e,
                )
            })?;
            let token_embed = self.llm.embed(&token_t)?;

            // For text-only tokens during generation, all three RoPE dimensions
            // (temporal, height, width) share the same position index.
            let pos_ids =
                Tensor::new(&[next_pos, next_pos, next_pos], &self.device).map_err(|e| {
                    candle_to_ocr_processing(
                        crate::core::errors::ProcessingStage::TensorOperation,
                        "PaddleOCR-VL: create pos tensor failed",
                        e,
                    )
                })?;
            let pos_ids = pos_ids.reshape((3usize, 1usize, 1usize)).map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "PaddleOCR-VL: reshape pos tensor failed",
                    e,
                )
            })?;

            let hs = self
                .llm
                .forward(&token_embed, &pos_ids, Some(&mut kv_cache), None)?;
            let hs = hs.squeeze(0).map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "PaddleOCR-VL: squeeze hs batch failed",
                    e,
                )
            })?;
            // hs is now [1, hidden_size], keep it 2D for lm_head
            let hs_logits = self
                .lm_head
                .forward(&hs)
                .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "lm_head forward step", e))?;
            logits = hs_logits
                .squeeze(0)
                .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "lm_head squeeze step", e))?;

            next_pos += 1;
        }

        let decoded =
            self.tokenizer
                .decode(&generated, true)
                .map_err(|e| OCRError::InvalidInput {
                    message: format!("PaddleOCR-VL: tokenizer decode failed: {e}"),
                })?;
        let processed = task.postprocess(decoded.clone());
        Ok((decoded, processed))
    }
}

fn causal_mask(seq_len: usize, device: &Device, dtype: DType) -> Result<Tensor, OCRError> {
    let mut data = Vec::with_capacity(seq_len * seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j <= i {
                data.push(0f32);
            } else {
                // Use NEG_INFINITY for masked positions - converts correctly to BF16/F16
                data.push(f32::NEG_INFINITY);
            }
        }
    }
    let tensor =
        Tensor::from_vec(data, (1usize, 1usize, seq_len, seq_len), device).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: build causal mask failed",
                e,
            )
        })?;
    let data = tensor.to_dtype(dtype).map_err(|e| {
        candle_to_ocr_processing(
            crate::core::errors::ProcessingStage::TensorOperation,
            "PaddleOCR-VL: cast causal mask failed",
            e,
        )
    })?;
    Ok(data)
}

fn get_rope_index(
    cfg: &PaddleOcrVlConfig,
    input_ids: &[u32],
    image_grid_thw: &[(usize, usize, usize)],
    device: &Device,
) -> Result<(Tensor, i64), OCRError> {
    let spatial_merge_size = cfg.vision_config.spatial_merge_size;
    if spatial_merge_size == 0 {
        return Err(OCRError::config_error(
            "PaddleOCR-VL: vision_config.spatial_merge_size must be > 0",
        ));
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
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: build position_ids tensor failed",
                e,
            )
        })?;

    Ok((position_ids, rope_deltas))
}
