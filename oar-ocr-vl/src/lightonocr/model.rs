use super::config::{LightOnOcrConfig, LightOnOcrProcessorConfig};
use super::processing::{LightOnOcrImageInputs, preprocess_image};
use super::text::LightOnOcrTextModel;
use super::vision::{PixtralVisionConfig, PixtralVisionModel};
use crate::utils::{candle_to_ocr_inference, candle_to_ocr_processing};
use candle_core::{D, DType, Device, Tensor};
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

    /// Generate OCR output for an image with an optional instruction.
    ///
    /// An empty instruction performs plain text extraction. For structured output,
    /// pass a task-specific prompt (e.g. "Parse the table in the image into HTML.").
    pub fn generate(
        &self,
        image: RgbImage,
        instruction: impl AsRef<str>,
        max_new_tokens: usize,
    ) -> Result<String, OCRError> {
        let instruction = instruction.as_ref();

        let (input_ids, input_embeds) = self.prepare_inputs(&image, instruction)?;

        self.text.clear_kv_cache();
        let mask = build_causal_mask(
            1,
            input_ids.len(),
            0,
            self.text.num_attn_heads,
            self.text.dtype,
            &self.device,
        )?;

        let mut logits = self
            .text
            .forward_embeds(input_embeds, Some(&mask), &[0])
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "prefill forward", e))?;

        tracing::debug!("LightOnOCR generate: logits shape {:?}", logits.dims());

        let mut generated: Vec<u32> = Vec::new();
        let mut next_pos = input_ids.len();

        for step in 0..max_new_tokens {
            let next_token_t = logits
                .argmax(D::Minus1)
                .map_err(|e| candle_to_ocr_inference("LightOnOCR", "argmax", e))?;
            let next_token = if next_token_t.rank() == 0 {
                next_token_t
                    .to_scalar::<u32>()
                    .map_err(|e| candle_to_ocr_inference("LightOnOCR", "argmax to_scalar", e))?
            } else {
                next_token_t
                    .to_vec1::<u32>()
                    .map_err(|e| candle_to_ocr_inference("LightOnOCR", "argmax to_vec1", e))?
                    .into_iter()
                    .next()
                    .ok_or_else(|| OCRError::InvalidInput {
                        message: "LightOnOCR: argmax produced empty tensor".to_string(),
                    })?
            };

            if step < 5 {
                let token_str = self
                    .tokenizer
                    .decode(&[next_token], false)
                    .unwrap_or_default();
                tracing::debug!(
                    "LightOnOCR generate: step {} token {} = {:?}",
                    step,
                    next_token,
                    token_str
                );
            }

            if next_token == self.eos_token_id {
                break;
            }
            generated.push(next_token);

            let token_t = Tensor::new(&[next_token], &self.device).map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "LightOnOCR: create next_token tensor failed",
                    e,
                )
            })?;
            let token_t = token_t.reshape((1usize, 1usize)).map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "LightOnOCR: reshape next_token tensor failed",
                    e,
                )
            })?;
            let token_embed = self
                .text
                .embed_tokens(&token_t)
                .map_err(|e| candle_to_ocr_inference("LightOnOCR", "embed next token", e))?;

            logits = self
                .text
                .forward_embeds(token_embed, None, &[next_pos])
                .map_err(|e| candle_to_ocr_inference("LightOnOCR", "decode forward", e))?;
            next_pos += 1;
        }

        let text =
            self.tokenizer
                .decode(&generated, false)
                .map_err(|e| OCRError::InvalidInput {
                    message: format!("LightOnOCR: tokenizer decode failed: {e}"),
                })?;
        Ok(text)
    }

    fn prepare_inputs(
        &self,
        image: &RgbImage,
        instruction: &str,
    ) -> Result<(Vec<u32>, Tensor), OCRError> {
        let image_inputs = preprocess_image(
            image,
            &self.image_cfg,
            self.cfg.spatial_merge_size,
            &self.device,
            self.dtype,
        )?;

        let image_tokens = image_token_repeat(
            image_inputs.grid_h,
            image_inputs.grid_w,
            self.cfg.spatial_merge_size,
        )?;

        tracing::debug!(
            "LightOnOCR prepare_inputs: grid {}x{}, merge_size={}, expected image tokens={}",
            image_inputs.grid_h,
            image_inputs.grid_w,
            self.cfg.spatial_merge_size,
            (image_inputs.grid_h / self.cfg.spatial_merge_size)
                * (image_inputs.grid_w / self.cfg.spatial_merge_size)
        );

        let prompt = build_prompt(instruction, image_tokens);
        tracing::debug!(
            "LightOnOCR prepare_inputs: prompt length={} chars",
            prompt.len()
        );
        tracing::info!("LightOnOCR prepare_inputs: prompt={:?}", prompt);

        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| OCRError::InvalidInput {
                message: format!("LightOnOCR: tokenizer encode failed: {e}"),
            })?;
        let input_ids = enc.get_ids().to_vec();
        let spans = find_image_spans(&input_ids, self.image_token_id);
        let total_image_tokens: usize = spans.iter().map(|(s, e)| e - s).sum();
        let expected_image_tokens = (image_inputs.grid_h / self.cfg.spatial_merge_size)
            * (image_inputs.grid_w / self.cfg.spatial_merge_size);
        if total_image_tokens != expected_image_tokens {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "LightOnOCR: image token count mismatch: expected {expected_image_tokens}, got {total_image_tokens}"
                ),
            });
        }

        let input_ids_t = Tensor::new(input_ids.clone(), &self.device).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "LightOnOCR: create input_ids tensor failed",
                e,
            )
        })?;
        let input_ids_t = input_ids_t
            .reshape((1usize, input_ids.len()))
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "LightOnOCR: reshape input_ids tensor failed",
                    e,
                )
            })?;

        let mut input_embeds = self
            .text
            .embed_tokens(&input_ids_t)
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "embed prompt tokens", e))?;

        let image_embeds = self.build_image_embeds(&image_inputs)?;
        input_embeds = insert_image_embeds(input_embeds, image_embeds, &spans)?;

        Ok((input_ids, input_embeds))
    }

    fn build_image_embeds(&self, image_inputs: &LightOnOcrImageInputs) -> Result<Tensor, OCRError> {
        let vision_tokens = self
            .vision
            .forward(&image_inputs.pixel_values)
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "vision forward", e))?;

        let projected = self
            .projector
            .forward(&vision_tokens, image_inputs.grid_h, image_inputs.grid_w)
            .map_err(|e| candle_to_ocr_inference("LightOnOCR", "vision projection", e))?;

        tracing::debug!(
            "LightOnOCR build_image_embeds: projected shape {:?}, grid {}x{} -> merged {}x{}",
            projected.dims(),
            image_inputs.grid_h,
            image_inputs.grid_w,
            image_inputs.grid_h / self.cfg.spatial_merge_size,
            image_inputs.grid_w / self.cfg.spatial_merge_size
        );

        Ok(projected)
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

    fn forward(&self, xs: &Tensor, grid_h: usize, grid_w: usize) -> candle_core::Result<Tensor> {
        let normed = xs.apply(&self.norm)?;
        let merged = self.merge_patches(&normed, grid_h, grid_w)?;
        let x = merged.apply(&self.linear_1)?;
        let x = x.apply(&self.act)?;
        x.apply(&self.linear_2)
    }

    fn merge_patches(
        &self,
        xs: &Tensor,
        grid_h: usize,
        grid_w: usize,
    ) -> candle_core::Result<Tensor> {
        let (b, seq_len, hidden) = xs.dims3()?;
        if grid_h * grid_w != seq_len {
            candle_core::bail!(
                "merge_patches: expected grid {}x{} to match seq_len={seq_len}",
                grid_h,
                grid_w
            );
        }
        let merged_h = grid_h / self.merge_size;
        let merged_w = grid_w / self.merge_size;
        if merged_h == 0 || merged_w == 0 {
            candle_core::bail!(
                "merge_patches: grid {}x{} too small for merge_size={}",
                grid_h,
                grid_w,
                self.merge_size
            );
        }

        let trim_h = merged_h * self.merge_size;
        let trim_w = merged_w * self.merge_size;

        // Match Mistral3PatchMerger.forward from HuggingFace transformers
        // (models/mistral3/modular_mistral3.py), which uses F.unfold(kernel=merge, stride=merge).
        // Here we implement the equivalent via reshape + permute.
        // 1. (b, seq, hidden) -> (b, h, w, hidden)
        let xs = xs.reshape((b, grid_h, grid_w, hidden))?;
        let xs = xs.narrow(1, 0, trim_h)?;
        let xs = xs.narrow(2, 0, trim_w)?;

        // 2. (b, h, w, hidden) -> (b, h', merge_h, w', merge_w, hidden)
        //    indices:               (0,  1,       2,  3,       4,      5)
        let xs = xs.reshape((
            b,
            merged_h,
            self.merge_size,
            merged_w,
            self.merge_size,
            hidden,
        ))?;

        // 3. permute to (b, h', w', hidden, merge_h, merge_w)
        //    unfold order: hidden varies slowest in the flattened output
        //    i.e. unfold gives (h', w', hidden, merge_h, merge_w), view flattens last 3 dims
        let xs = xs.permute((0, 1, 3, 5, 2, 4))?;

        // 4. flatten to (b, h'*w', hidden*merge*merge)
        //    Note: this is hidden*merge*merge, not merge*merge*hidden
        let xs = xs.reshape((
            b,
            merged_h * merged_w,
            hidden * self.merge_size * self.merge_size,
        ))?;

        xs.apply(&self.patch_merger)
    }
}

fn build_prompt(instruction: &str, image_tokens: String) -> String {
    let mut prompt = String::new();
    prompt.push_str("<|im_start|>system<|im_end|>\n");
    prompt.push_str("<|im_start|>user\n");
    prompt.push_str(&image_tokens);
    let instruction = instruction.trim();
    if !instruction.is_empty() {
        prompt.push('\n');
        prompt.push_str(instruction);
    }
    prompt.push_str("<|im_end|>\n<|im_start|>assistant\n");
    prompt
}

/// Build the image token sequence.
/// Format matches Pixtral processor:
/// Each row: [IMG][IMG]...[IMG][IMG_BREAK]
/// Generate image placeholder tokens for the prompt.
/// LightOnOCR uses continuous <|image_pad|> tokens - the grid structure is
/// handled internally via image_sizes, not via vision_pad/vision_end tokens.
fn image_token_repeat(grid_h: usize, grid_w: usize, merge_size: usize) -> Result<String, OCRError> {
    if merge_size == 0 {
        return Err(OCRError::ConfigError {
            message: "LightOnOCR merge_size must be > 0".to_string(),
        });
    }
    let merged_h = grid_h / merge_size;
    let merged_w = grid_w / merge_size;
    if merged_h == 0 || merged_w == 0 {
        return Err(OCRError::InvalidInput {
            message: "LightOnOCR image grid is empty".to_string(),
        });
    }

    // Just generate continuous image_pad tokens (no vision_pad breaks)
    let total_tokens = merged_h * merged_w;
    Ok("<|image_pad|>".repeat(total_tokens))
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

fn build_causal_mask(
    batch: usize,
    tgt_len: usize,
    offset: usize,
    num_heads: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor, OCRError> {
    // Use dtype-specific min for numerical stability (matches vision.rs)
    let d_min: f32 = match dtype {
        DType::F32 => f32::MIN,
        DType::F16 => -65504.0,        // half::f16::MIN
        DType::BF16 => -3.3895313e+38, // half::bf16::MIN
        _ => f32::MIN,
    };
    let mut mask = Vec::with_capacity(tgt_len * (tgt_len + offset));
    for i in 0..tgt_len {
        for j in 0..(tgt_len + offset) {
            if j <= i + offset {
                mask.push(0f32);
            } else {
                mask.push(d_min);
            }
        }
    }
    let mask = Tensor::from_vec(mask, (tgt_len, tgt_len + offset), device).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "LightOnOCR: build causal mask tensor failed",
            e,
        )
    })?;
    mask.expand((batch, num_heads, tgt_len, tgt_len + offset))
        .map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "LightOnOCR: expand causal mask failed",
                e,
            )
        })?
        .to_dtype(dtype)
        .map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "LightOnOCR: cast causal mask failed",
                e,
            )
        })
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

    #[test]
    fn test_image_token_repeat_basic() {
        let result = image_token_repeat(4, 6, 2).unwrap();
        // merged_h=2, merged_w=3 -> 6 tokens
        assert_eq!(result, "<|image_pad|>".repeat(6));
    }

    #[test]
    fn test_image_token_repeat_merge_size_1() {
        let result = image_token_repeat(3, 4, 1).unwrap();
        assert_eq!(result, "<|image_pad|>".repeat(12));
    }

    #[test]
    fn test_image_token_repeat_zero_merge_size() {
        let result = image_token_repeat(4, 4, 0);
        assert!(result.is_err());
    }

    #[test]
    fn test_image_token_repeat_empty_grid() {
        // grid_h=1, merge_size=2 -> merged_h=0
        let result = image_token_repeat(1, 4, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_build_prompt_with_instruction() {
        let tokens = "<|image_pad|>".repeat(4);
        let prompt = build_prompt("Describe this image.", tokens.clone());
        let expected = format!(
            "<|im_start|>system<|im_end|>\n<|im_start|>user\n{}\nDescribe this image.<|im_end|>\n<|im_start|>assistant\n",
            tokens
        );
        assert_eq!(prompt, expected);
    }

    #[test]
    fn test_build_prompt_empty_instruction() {
        let tokens = "<|image_pad|>".repeat(2);
        let prompt = build_prompt("", tokens.clone());
        let expected = format!(
            "<|im_start|>system<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            tokens
        );
        assert_eq!(prompt, expected);
    }

    #[test]
    fn test_build_prompt_whitespace_instruction() {
        let tokens = "<|image_pad|>".to_string();
        let prompt = build_prompt("   \n  ", tokens.clone());
        // Whitespace-only instruction is treated as empty
        let expected = format!(
            "<|im_start|>system<|im_end|>\n<|im_start|>user\n{}<|im_end|>\n<|im_start|>assistant\n",
            tokens
        );
        assert_eq!(prompt, expected);
    }
}
