//! PaddleOCR-VL (Vision-Language) model implementation.

use super::config::{PaddleOcrVlConfig, PaddleOcrVlImageProcessorConfig};
use super::ernie::Ernie4_5Model;
use super::processing;
use super::projector::Projector;
use super::vision::VisionModel;
use crate::attention::{combine_masks, create_causal_mask, create_left_padding_mask};
use crate::utils::{candle_to_ocr_inference, candle_to_ocr_processing};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::Module;
use image::RgbImage;
use oar_ocr_core::core::OCRError;
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
    /// Cached tokenized image placeholder (single token)
    image_placeholder_token_id: u32,
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

        match self.generate_internal(images, tasks, max_new_tokens) {
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
        tasks: &[PaddleOcrVlTask],
        max_new_tokens: usize,
    ) -> Result<Vec<(String, String)>, OCRError> {
        let batch_size = images.len();

        // 1. Preprocess all images
        let image_inputs =
            processing::preprocess_images(images, &self.image_cfg, &self.device, self.dtype)?;

        // 2. Build prompts for each sample
        let mut all_input_ids: Vec<Vec<u32>> = Vec::with_capacity(batch_size);
        let mut all_image_token_counts: Vec<usize> = Vec::with_capacity(batch_size);

        for (i, task) in tasks.iter().enumerate() {
            let (t, h, w) = image_inputs.image_grid_thw[i];
            let image_token_count =
                (t * h * w) / (self.image_cfg.merge_size * self.image_cfg.merge_size);
            all_image_token_counts.push(image_token_count);

            let prefix = "<|begin_of_sentence|>User: <|IMAGE_START|>";
            let suffix = format!("<|IMAGE_END|>{}\nAssistant: ", task.prompt());

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
        let max_seq_len = *seq_lens.iter().max().unwrap();

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

        // 10. Decode results
        let mut results = Vec::with_capacity(batch_size);
        for (i, tokens) in generated.into_iter().enumerate() {
            let decoded =
                self.tokenizer
                    .decode(&tokens, true)
                    .map_err(|e| OCRError::InvalidInput {
                        message: format!("decode failed: {e}"),
                    })?;
            let processed = tasks[i].postprocess(decoded.clone());
            results.push((decoded, processed));
        }

        Ok(results)
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
