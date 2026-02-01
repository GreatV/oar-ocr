use super::config::{LightOnOcrConfig, LightOnOcrProcessorConfig};
use super::processing::{LightOnOcrImageInputs, preprocess_image};
use super::text::LightOnOcrTextModel;
use super::vision::{PixtralVisionConfig, PixtralVisionModel};
use crate::attention::{combine_masks, create_causal_mask, create_left_padding_mask};
use crate::utils::{candle_to_ocr_inference, candle_to_ocr_processing};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Activation, Linear, RmsNorm, VarBuilder, linear_b, linear_no_bias, rms_norm};
use image::RgbImage;
use oar_ocr_core::core::OCRError;
use std::path::Path;
use tokenizers::Tokenizer;

pub struct LightOnOcr {
    device: Device,
    dtype: DType,
    cfg: LightOnOcrConfig,
    image_cfg: super::config::LightOnOcrImageProcessorConfig,
    tokenizer: Tokenizer,
    text: LightOnOcrTextModel,
    vision: PixtralVisionModel,
    projector: VisionProjection,
    eos_token_id: u32,
    image_token_id: u32,
}

impl LightOnOcr {
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, OCRError> {
        let model_dir = model_dir.as_ref();
        let cfg = LightOnOcrConfig::from_path(model_dir.join("config.json"))?;
        let processor_cfg =
            LightOnOcrProcessorConfig::from_path(model_dir.join("processor_config.json"))?;
        let image_cfg = processor_cfg.image_processor;

        if let Some(patch_size) = processor_cfg.patch_size
            && patch_size != image_cfg.patch_size
        {
            return Err(OCRError::ConfigError {
                message: format!(
                    "LightOnOCR patch_size mismatch: processor {} != image_processor {}",
                    patch_size, image_cfg.patch_size
                ),
            });
        }
        if let Some(merge_size) = processor_cfg.spatial_merge_size
            && merge_size != cfg.spatial_merge_size
        {
            return Err(OCRError::ConfigError {
                message: format!(
                    "LightOnOCR spatial_merge_size mismatch: processor {} != config {}",
                    merge_size, cfg.spatial_merge_size
                ),
            });
        }
        if image_cfg.patch_size != cfg.vision_config.patch_size {
            return Err(OCRError::ConfigError {
                message: format!(
                    "LightOnOCR patch_size mismatch: image_processor {} != vision_config {}",
                    image_cfg.patch_size, cfg.vision_config.patch_size
                ),
            });
        }

        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| {
            OCRError::ConfigError {
                message: format!("failed to load LightOnOCR tokenizer.json: {e}"),
            }
        })?;
        let tok_image_id =
            tokenizer
                .token_to_id("<|image_pad|>")
                .ok_or_else(|| OCRError::ConfigError {
                    message: "LightOnOCR tokenizer is missing <|image_pad|> token".to_string(),
                })?;
        if tok_image_id != cfg.image_token_id {
            return Err(OCRError::ConfigError {
                message: format!(
                    "LightOnOCR image_token_id mismatch: tokenizer {tok_image_id} != config {}",
                    cfg.image_token_id
                ),
            });
        }
        if let Some(tok_eos_id) = tokenizer.token_to_id("<|im_end|>")
            && tok_eos_id != cfg.eos_token_id
        {
            return Err(OCRError::ConfigError {
                message: format!(
                    "LightOnOCR eos_token_id mismatch: tokenizer {tok_eos_id} != config {}",
                    cfg.eos_token_id
                ),
            });
        }

        let dtype = device.bf16_default_to_f32();
        // SAFETY: The mmap'd file must not be modified or deleted while in use.
        // This is upheld because model files are read-only assets loaded at initialization.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_dir.join("model.safetensors")],
                dtype,
                &device,
            )
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "load model.safetensors", e))?
        };

        let vision_cfg = PixtralVisionConfig::from(&cfg.vision_config);

        let vision = PixtralVisionModel::new(&vision_cfg, vb.pp("model").pp("vision_encoder"))
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "load vision encoder", e))?;

        let text = LightOnOcrTextModel::new(&cfg.text_config, vb.clone())
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "load text model", e))?;

        let projector = VisionProjection::load(&cfg, vb.pp("model").pp("vision_projection"))?;

        let eos_token_id = cfg.eos_token_id;
        let image_token_id = cfg.image_token_id;

        Ok(Self {
            device,
            dtype,
            cfg,
            image_cfg,
            tokenizer,
            text,
            vision,
            projector,
            eos_token_id,
            image_token_id,
        })
    }

    /// Generate OCR output for one or more images with optional instructions.
    ///
    /// Supports true GPU batching when multiple images are provided.
    /// An empty instruction performs plain text extraction. For structured output,
    /// pass a task-specific prompt (e.g. "Parse the table in the image into HTML.").
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
                    "LightOnOCR: images count ({}) != instructions count ({})",
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
        let mut all_image_inputs: Vec<LightOnOcrImageInputs> = Vec::with_capacity(batch_size);
        let mut all_image_token_counts: Vec<usize> = Vec::with_capacity(batch_size);

        for (image, instruction) in images.iter().zip(instructions.iter()) {
            let instruction = instruction.as_ref();
            let image_inputs = preprocess_image(
                image,
                &self.image_cfg,
                self.cfg.spatial_merge_size,
                &self.device,
                self.dtype,
            )?;

            let image_token_count = (image_inputs.grid_h / self.cfg.spatial_merge_size)
                * (image_inputs.grid_w / self.cfg.spatial_merge_size);

            // Build prompt tokens
            let prefix = "<|im_start|>system<|im_end|>\n<|im_start|>user\n";
            let instruction_trimmed = instruction.trim();
            let suffix = if instruction_trimmed.is_empty() {
                "<|im_end|>\n<|im_start|>assistant\n"
            } else {
                "\n"
            };
            let after_instruction = if instruction_trimmed.is_empty() {
                ""
            } else {
                "<|im_end|>\n<|im_start|>assistant\n"
            };

            let prefix_enc =
                self.tokenizer
                    .encode(prefix, false)
                    .map_err(|e| OCRError::InvalidInput {
                        message: format!("LightOnOCR: tokenizer encode prefix failed: {e}"),
                    })?;
            let suffix_enc =
                self.tokenizer
                    .encode(suffix, false)
                    .map_err(|e| OCRError::InvalidInput {
                        message: format!("LightOnOCR: tokenizer encode suffix failed: {e}"),
                    })?;

            let mut input_ids =
                Vec::with_capacity(prefix_enc.len() + image_token_count + suffix_enc.len() + 50);
            input_ids.extend_from_slice(prefix_enc.get_ids());
            input_ids.extend(std::iter::repeat_n(self.image_token_id, image_token_count));
            input_ids.extend_from_slice(suffix_enc.get_ids());

            if !instruction_trimmed.is_empty() {
                let instruction_enc =
                    self.tokenizer
                        .encode(instruction_trimmed, false)
                        .map_err(|e| OCRError::InvalidInput {
                            message: format!(
                                "LightOnOCR: tokenizer encode instruction failed: {e}"
                            ),
                        })?;
                input_ids.extend_from_slice(instruction_enc.get_ids());

                let after_enc = self
                    .tokenizer
                    .encode(after_instruction, false)
                    .map_err(|e| OCRError::InvalidInput {
                        message: format!(
                            "LightOnOCR: tokenizer encode after_instruction failed: {e}"
                        ),
                    })?;
                input_ids.extend_from_slice(after_enc.get_ids());
            }

            all_input_ids.push(input_ids);
            all_image_inputs.push(image_inputs);
            all_image_token_counts.push(image_token_count);
        }

        // 2. Compute vision features and project for each image
        let mut all_image_embeds: Vec<Tensor> = Vec::with_capacity(batch_size);
        for image_inputs in &all_image_inputs {
            let vision_tokens = self
                .vision
                .forward(&image_inputs.pixel_values)
                .map_err(|e| candle_to_ocr_inference("LightOnOCR", "vision forward", e))?;
            let projected =
                self.projector
                    .forward(&vision_tokens, image_inputs.grid_h, image_inputs.grid_w)?;
            all_image_embeds.push(projected);
        }

        // 3. Build embeddings per sample with left-padding
        let seq_lens: Vec<usize> = all_input_ids.iter().map(|ids| ids.len()).collect();
        let max_seq_len = *seq_lens.iter().max().unwrap();

        let mut batch_embeds: Vec<Tensor> = Vec::with_capacity(batch_size);

        for (i, input_ids) in all_input_ids.iter().enumerate() {
            let seq_len = input_ids.len();
            let pad_len = max_seq_len - seq_len;
            let image_token_count = all_image_token_counts[i];

            // Embed tokens
            let input_ids_t = Tensor::new(input_ids.clone(), &self.device)
                .and_then(|t| t.reshape((1, seq_len)))
                .map_err(|e| candle_to_ocr_inference("LightOnOCR", "create input_ids", e))?;
            let mut inputs_embeds = self
                .text
                .embed_tokens(&input_ids_t)
                .map_err(|e| candle_to_ocr_inference("LightOnOCR", "embed tokens", e))?;

            // Insert image embeddings
            let spans = find_image_spans(input_ids, self.image_token_id);
            let total_image_tokens: usize = spans.iter().map(|(s, e)| e - s).sum();
            if total_image_tokens != image_token_count {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "LightOnOCR: image token count mismatch: expected {image_token_count}, got {total_image_tokens}"
                    ),
                });
            }
            inputs_embeds =
                insert_image_embeds(inputs_embeds, all_image_embeds[i].clone(), &spans)?;

            // Left-pad if needed
            if pad_len > 0 {
                let hidden_size = inputs_embeds
                    .dim(2)
                    .map_err(|e| candle_to_ocr_inference("LightOnOCR", "get hidden_size", e))?;
                let pad = Tensor::zeros(
                    (1, pad_len, hidden_size),
                    inputs_embeds.dtype(),
                    &self.device,
                )
                .map_err(|e| candle_to_ocr_inference("LightOnOCR", "create pad", e))?;
                inputs_embeds = Tensor::cat(&[&pad, &inputs_embeds], 1)
                    .map_err(|e| candle_to_ocr_inference("LightOnOCR", "cat pad", e))?;
            }
            batch_embeds.push(inputs_embeds);
        }

        // 4. Stack batched tensors
        let batch_refs: Vec<&Tensor> = batch_embeds.iter().collect();
        let inputs_embeds = Tensor::cat(&batch_refs, 0)
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "stack embeds", e))?;

        // 5. Create attention mask
        let causal = create_causal_mask(max_seq_len, max_seq_len, self.dtype, &self.device)
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "create causal", e))?;
        let padding = create_left_padding_mask(&seq_lens, max_seq_len, self.dtype, &self.device)
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "create padding", e))?;
        let mask = combine_masks(&causal, &padding)
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "combine masks", e))?;

        // 6. Prefill
        self.text.clear_kv_cache();
        // Build seqlen_offsets: for left-padded sequences, offset is the padding length
        let seqlen_offsets: Vec<usize> = seq_lens.iter().map(|&len| max_seq_len - len).collect();
        let logits = self
            .text
            .forward_embeds(inputs_embeds, Some(&mask), &seqlen_offsets)
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "prefill forward", e))?;

        // 7. Get initial logits per sample
        let mut logits_list: Vec<Tensor> = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let sample_logits = logits
                .i(i)
                .map_err(|e| candle_to_ocr_inference("LightOnOCR", "get sample logits", e))?;
            logits_list.push(sample_logits);
        }

        // 8. Autoregressive decode
        let mut generated: Vec<Vec<u32>> = vec![Vec::new(); batch_size];
        let mut finished: Vec<bool> = vec![false; batch_size];
        let mut positions: Vec<usize> = seq_lens.clone();

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
                        .map_err(|e| candle_to_ocr_inference("LightOnOCR", "argmax", e))?;

                    if tok == self.eos_token_id {
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
                .map_err(|e| candle_to_ocr_inference("LightOnOCR", "create tokens", e))?;
            let embeds = self
                .text
                .embed_tokens(&tokens)
                .map_err(|e| candle_to_ocr_inference("LightOnOCR", "embed next tokens", e))?;

            // For decode step, no mask needed, use positions as offsets
            let next_logits = self
                .text
                .forward_embeds(embeds, None, &positions)
                .map_err(|e| candle_to_ocr_inference("LightOnOCR", "decode forward", e))?;

            logits_list.clear();
            for i in 0..batch_size {
                let sample_logits = next_logits
                    .i(i)
                    .map_err(|e| candle_to_ocr_inference("LightOnOCR", "get decode logits", e))?;
                logits_list.push(sample_logits);
            }

            for (i, p) in positions.iter_mut().enumerate() {
                if !finished[i] {
                    *p += 1;
                }
            }
        }

        // 9. Decode results
        let mut results = Vec::with_capacity(batch_size);
        for tokens in generated {
            let text =
                self.tokenizer
                    .decode(&tokens, false)
                    .map_err(|e| OCRError::InvalidInput {
                        message: format!("LightOnOCR: tokenizer decode failed: {e}"),
                    })?;
            results.push(text);
        }

        Ok(results)
    }
}

struct VisionProjection {
    merge_size: usize,
    patch_merger: Linear,
    norm: RmsNorm,
    linear_1: Linear,
    linear_2: Linear,
    act: Activation,
}

impl VisionProjection {
    fn load(cfg: &LightOnOcrConfig, vb: VarBuilder) -> Result<Self, OCRError> {
        let hidden = cfg.vision_config.hidden_size;
        let merge_size = cfg.spatial_merge_size;
        let in_dim = hidden * merge_size * merge_size;

        let patch_merger =
            linear_no_bias(in_dim, hidden, vb.pp("patch_merger").pp("merging_layer"))
                .map_err(|e| candle_to_ocr_inference("LightOnOCR", "load patch_merger", e))?;
        let norm = rms_norm(hidden, cfg.text_config.rms_norm_eps, vb.pp("norm"))
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "load vision_projection norm", e))?;

        let linear_1 = linear_b(
            hidden,
            hidden,
            cfg.multimodal_projector_bias,
            vb.pp("linear_1"),
        )
        .map_err(|e| candle_to_ocr_inference("LightOnOCR", "load vision_projection linear_1", e))?;
        let linear_2 = linear_b(
            hidden,
            hidden,
            cfg.multimodal_projector_bias,
            vb.pp("linear_2"),
        )
        .map_err(|e| candle_to_ocr_inference("LightOnOCR", "load vision_projection linear_2", e))?;

        Ok(Self {
            merge_size,
            patch_merger,
            norm,
            linear_1,
            linear_2,
            act: cfg.projector_hidden_act,
        })
    }

    fn forward(&self, xs: &Tensor, grid_h: usize, grid_w: usize) -> Result<Tensor, OCRError> {
        let normed = xs
            .apply(&self.norm)
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "vision_projection norm", e))?;
        let merged = self.merge_patches(&normed, grid_h, grid_w)?;
        let x = merged
            .apply(&self.linear_1)
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "vision_projection linear_1", e))?;
        let x = x
            .apply(&self.act)
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "vision_projection act", e))?;
        x.apply(&self.linear_2)
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "vision_projection linear_2", e))
    }

    fn merge_patches(&self, xs: &Tensor, grid_h: usize, grid_w: usize) -> Result<Tensor, OCRError> {
        let (b, seq_len, hidden) = xs
            .dims3()
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "merge_patches dims3", e))?;
        if grid_h * grid_w != seq_len {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "LightOnOCR merge_patches: expected grid {}x{} to match seq_len={seq_len}",
                    grid_h, grid_w
                ),
            });
        }
        let merged_h = grid_h / self.merge_size;
        let merged_w = grid_w / self.merge_size;
        if merged_h == 0 || merged_w == 0 {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "LightOnOCR merge_patches: grid {}x{} too small for merge_size={}",
                    grid_h, grid_w, self.merge_size
                ),
            });
        }

        let trim_h = merged_h * self.merge_size;
        let trim_w = merged_w * self.merge_size;

        // Match Mistral3PatchMerger.forward from HuggingFace transformers
        // (models/mistral3/modular_mistral3.py), which uses F.unfold(kernel=merge, stride=merge).
        // Here we implement the equivalent via reshape + permute.
        // 1. (b, seq, hidden) -> (b, h, w, hidden)
        let xs = xs
            .reshape((b, grid_h, grid_w, hidden))
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "merge_patches reshape1", e))?;
        let xs = xs
            .narrow(1, 0, trim_h)
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "merge_patches narrow1", e))?;
        let xs = xs
            .narrow(2, 0, trim_w)
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "merge_patches narrow2", e))?;

        // 2. (b, h, w, hidden) -> (b, h', merge_h, w', merge_w, hidden)
        //    indices:               (0,  1,       2,  3,       4,      5)
        let xs = xs
            .reshape((
                b,
                merged_h,
                self.merge_size,
                merged_w,
                self.merge_size,
                hidden,
            ))
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "merge_patches reshape2", e))?;

        // 3. permute to (b, h', w', hidden, merge_h, merge_w)
        //    unfold order: hidden varies slowest in the flattened output
        //    i.e. unfold gives (h', w', hidden, merge_h, merge_w), view flattens last 3 dims
        let xs = xs
            .permute((0, 1, 3, 5, 2, 4))
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "merge_patches permute", e))?;

        // 4. flatten to (b, h'*w', hidden*merge*merge)
        //    Note: this is hidden*merge*merge, not merge*merge*hidden
        let xs = xs
            .reshape((
                b,
                merged_h * merged_w,
                hidden * self.merge_size * self.merge_size,
            ))
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "merge_patches reshape3", e))?;

        xs.apply(&self.patch_merger)
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "merge_patches patch_merger", e))
    }
}

fn find_image_spans(input_ids: &[u32], image_token_id: u32) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut start = None;
    for (idx, &token) in input_ids.iter().enumerate() {
        if token == image_token_id {
            if start.is_none() {
                start = Some(idx);
            }
        } else if let Some(s) = start.take() {
            spans.push((s, idx));
        }
    }
    if let Some(s) = start {
        spans.push((s, input_ids.len()));
    }
    spans
}

fn insert_image_embeds(
    mut input_embeds: Tensor,
    image_embeds: Tensor,
    spans: &[(usize, usize)],
) -> Result<Tensor, OCRError> {
    let (b, _seq_len, hidden) = input_embeds.dims3().map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "LightOnOCR: input_embeds dims3 failed",
            e,
        )
    })?;
    let image_embeds = image_embeds.to_dtype(input_embeds.dtype()).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "LightOnOCR: cast image_embeds dtype failed",
            e,
        )
    })?;

    let mut offset = 0usize;
    for &(start, end) in spans {
        let len = end - start;
        let chunk = image_embeds.narrow(1, offset, len).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "LightOnOCR: slice image_embeds failed",
                e,
            )
        })?;
        input_embeds = input_embeds
            .slice_assign(&[0..b, start..end, 0..hidden], &chunk)
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "LightOnOCR: slice_assign image_embeds failed",
                    e,
                )
            })?;
        offset += len;
    }
    Ok(input_embeds)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_find_image_spans_single_span() {
        let input_ids = [1, 2, 99, 99, 99, 3, 4];
        let spans = find_image_spans(&input_ids, 99);
        assert_eq!(spans, vec![(2, 5)]);
    }

    #[test]
    fn test_find_image_spans_multiple_spans() {
        let input_ids = [99, 99, 1, 2, 99, 99, 99, 3];
        let spans = find_image_spans(&input_ids, 99);
        assert_eq!(spans, vec![(0, 2), (4, 7)]);
    }

    #[test]
    fn test_find_image_spans_trailing() {
        let input_ids = [1, 2, 99, 99];
        let spans = find_image_spans(&input_ids, 99);
        assert_eq!(spans, vec![(2, 4)]);
    }

    #[test]
    fn test_find_image_spans_none() {
        let input_ids = [1, 2, 3, 4];
        let spans = find_image_spans(&input_ids, 99);
        assert!(spans.is_empty());
    }

    #[test]
    fn test_find_image_spans_all() {
        let input_ids = [99, 99, 99];
        let spans = find_image_spans(&input_ids, 99);
        assert_eq!(spans, vec![(0, 3)]);
    }

    #[test]
    fn test_find_image_spans_single_token_spans() {
        let input_ids = [99, 1, 99, 2, 99];
        let spans = find_image_spans(&input_ids, 99);
        assert_eq!(spans, vec![(0, 1), (2, 3), (4, 5)]);
    }
}
