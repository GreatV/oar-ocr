use super::config::{OvisOcr2Config, OvisOcr2ImageProcessorConfig};
use super::processing::{
    OvisOcr2ImageInputs, preprocess_image, validate_processor_vision_compatibility,
};
use super::vision::OvisOcr2VisionModel;
use crate::api::recognition::RecognitionTask;
use crate::backbones::qwen3_5::text::Qwen35TextModel;
use crate::error::Error;
#[cfg(feature = "cuda")]
use crate::runtime::cuda::{ArgmaxFirstBf16, ArgmaxFirstF32};
pub use crate::runtime::trace::GenerationTrace;
use crate::runtime::trace::top3;
use crate::utils::{candle_to_ocr_inference, candle_to_ocr_processing};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use image::RgbImage;
use std::path::Path;
use tokenizers::Tokenizer;

const MODEL_NAME: &str = "OvisOCR2";

/// Official OvisOCR2 full-page document parsing instruction.
pub const DEFAULT_PROMPT: &str = "\nExtract all readable content from the image in natural human reading order and output the result as a single Markdown document. For charts or images, represent them using an HTML image tag: <img src=\"images/bbox_{left}_{top}_{right}_{bottom}.jpg\" />, where left, top, right, bottom are bounding box coordinates scaled to [0, 1000). Format formulas as LaTeX. Format tables as HTML: <table>...</table>. Transcribe all other text as standard Markdown. Preserve the original text without translation or paraphrasing.";

/// Upstream generation limit used by the official OvisOCR2 example.
pub const DEFAULT_MAX_NEW_TOKENS: usize = 16_384;

/// End-to-end OvisOCR2 page parser backed by Qwen3.5-0.8B.
pub struct OvisOcr2 {
    device: Device,
    dtype: DType,
    cfg: OvisOcr2Config,
    image_cfg: OvisOcr2ImageProcessorConfig,
    tokenizer: Tokenizer,
    text: Qwen35TextModel,
    vision: OvisOcr2VisionModel,
    lm_head: Linear,
    stop_token_ids: Vec<u32>,
    image_token_id: u32,
    // Must stay the last field: the captured decode graph's input bundle
    // holds a clone of the tied LM head owned here, so this guard drops
    // last and drains CUDA errors the head's free may stash (see
    // CudaGraphDrainGuard).
    #[cfg(feature = "cuda")]
    _drain_guard: crate::runtime::decoder_graph::CudaGraphDrainGuard,
}

struct TextCacheGuard<'a>(&'a Qwen35TextModel);

impl Drop for TextCacheGuard<'_> {
    fn drop(&mut self) {
        self.0.clear_cache();
    }
}

impl OvisOcr2 {
    /// Load an OvisOCR2 Hugging Face model directory.
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, Error> {
        Self::from_dir_with_runtime(model_dir, crate::RuntimeConfig::new(device))
    }

    pub fn from_dir_with_runtime(
        model_dir: impl AsRef<Path>,
        runtime: crate::RuntimeConfig,
    ) -> Result<Self, Error> {
        let (device, dtype) = runtime.resolve();
        let model_dir = model_dir.as_ref();
        let cfg = OvisOcr2Config::from_path(model_dir.join("config.json"))?;
        let image_cfg =
            OvisOcr2ImageProcessorConfig::from_path(model_dir.join("preprocessor_config.json"))?;
        validate_processor_vision_compatibility(&image_cfg, &cfg.vision_config)?;
        let tokenizer =
            Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| Error::Config {
                message: format!("failed to load OvisOCR2 tokenizer.json: {e}"),
            })?;
        require_token_id(&tokenizer, "<|image_pad|>", Some(cfg.image_token_id))?;
        require_token_id(
            &tokenizer,
            "<|vision_start|>",
            Some(cfg.vision_start_token_id),
        )?;
        require_token_id(&tokenizer, "<|vision_end|>", Some(cfg.vision_end_token_id))?;
        require_token_id(&tokenizer, "<|im_start|>", None)?;
        let tokenizer_eos = require_token_id(&tokenizer, "<|im_end|>", None)?;

        let weight_files = crate::utils::collect_safetensors(model_dir, MODEL_NAME)?;
        // SAFETY: The model files must remain unchanged while their mmap is in use.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&weight_files, dtype, &device)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load safetensors", e))?
        };
        let text = Qwen35TextModel::load(
            &cfg.text_config,
            MODEL_NAME,
            "OAR_OVISOCR2_DISABLE_CUDA_GRAPH",
            vb.pp("model").pp("language_model"),
        )?;
        let vision = OvisOcr2VisionModel::load(&cfg.vision_config, vb.pp("model").pp("visual"))?;
        // OvisOCR2 ties the language-model output projection to token embeddings.
        let lm_head = Linear::new(text.token_embedding_weight(), None);

        // vLLM's OvisOCR2 path stops on both the model-config EOS
        // (`<|endoftext|>`) and the tokenizer EOS (`<|im_end|>`).
        let stop_token_ids = build_stop_token_ids(cfg.text_config.eos_token_id, tokenizer_eos);
        let image_token_id = cfg.image_token_id;
        #[cfg(feature = "cuda")]
        let drain_guard = crate::runtime::decoder_graph::CudaGraphDrainGuard::new(&device);
        Ok(Self {
            device,
            dtype,
            cfg,
            image_cfg,
            tokenizer,
            text,
            vision,
            lm_head,
            stop_token_ids,
            image_token_id,
            #[cfg(feature = "cuda")]
            _drain_guard: drain_guard,
        })
    }

    /// Generate model-native Markdown, retaining visual-region image tags.
    pub fn generate(
        &self,
        images: &[RgbImage],
        max_new_tokens: usize,
    ) -> crate::error::BatchResult<String> {
        Ok(self
            .generate_tokens(images, max_new_tokens)?
            .into_iter()
            .map(|result| result.and_then(|tokens| self.decode_tokens(&tokens)))
            .collect())
    }

    /// Parse pages using the official post-processing, which removes visual
    /// region `<img ...>` blocks by default.
    pub fn parse(
        &self,
        images: &[RgbImage],
        max_new_tokens: usize,
    ) -> crate::error::BatchResult<String> {
        self.parse_with_image_tags(images, max_new_tokens, false)
    }

    /// Parse pages and optionally retain the model's visual-region image tags.
    pub fn parse_with_image_tags(
        &self,
        images: &[RgbImage],
        max_new_tokens: usize,
        keep_image_tags: bool,
    ) -> crate::error::BatchResult<String> {
        Ok(self
            .generate_tokens(images, max_new_tokens)?
            .into_iter()
            .map(|result| {
                result.and_then(|tokens| {
                    let text = self.decode_tokens_raw(&tokens)?;
                    Ok(postprocess_text(text, keep_image_tags))
                })
            })
            .collect())
    }

    /// Generate raw token ids for each input page.
    pub fn generate_tokens(
        &self,
        images: &[RgbImage],
        max_new_tokens: usize,
    ) -> crate::error::BatchResult<Vec<u32>> {
        self.generate_tokens_with_prompt(images, DEFAULT_PROMPT, max_new_tokens)
    }

    pub(crate) fn generate_tokens_with_prompt(
        &self,
        images: &[RgbImage],
        task_prompt: &str,
        max_new_tokens: usize,
    ) -> crate::error::BatchResult<Vec<u32>> {
        Ok(images
            .iter()
            .map(|image| self.generate_one(image, task_prompt, max_new_tokens, None))
            .collect())
    }

    /// Greedy generation with per-step top-3 logits, for numerical
    /// alignment against the upstream reference (see the
    /// `ovisocr2_alignment` test).
    pub fn generate_traced(
        &self,
        image: &RgbImage,
        max_new_tokens: usize,
    ) -> Result<GenerationTrace, Error> {
        self.generate_traced_with_prompt(image, DEFAULT_PROMPT, max_new_tokens)
    }

    pub(crate) fn generate_traced_with_prompt(
        &self,
        image: &RgbImage,
        task_prompt: &str,
        max_new_tokens: usize,
    ) -> Result<GenerationTrace, Error> {
        let mut step_top = Vec::new();
        let tokens = self.generate_one(image, task_prompt, max_new_tokens, Some(&mut step_top))?;
        Ok(GenerationTrace { tokens, step_top })
    }

    fn generate_one(
        &self,
        image: &RgbImage,
        task_prompt: &str,
        max_new_tokens: usize,
        step_top: Option<&mut Vec<[(u32, f32); 3]>>,
    ) -> Result<Vec<u32>, Error> {
        self.generate_request(
            std::slice::from_ref(image),
            task_prompt,
            max_new_tokens,
            step_top,
        )
    }

    pub(crate) fn generate_request(
        &self,
        images: &[RgbImage],
        task_prompt: &str,
        max_new_tokens: usize,
        mut step_top: Option<&mut Vec<[(u32, f32); 3]>>,
    ) -> Result<Vec<u32>, Error> {
        if images.is_empty() {
            return Err(Error::invalid_input(
                "OvisOCR needs at least one image per request",
            ));
        }
        self.text.clear_cache();
        let _cache_guard = TextCacheGuard(&self.text);
        if max_new_tokens == 0 {
            return Ok(Vec::new());
        }
        if max_new_tokens > self.cfg.text_config.max_position_embeddings {
            return Err(Error::InvalidInput {
                message: format!(
                    "OvisOCR2 max_new_tokens {max_new_tokens} exceeds context limit {}",
                    self.cfg.text_config.max_position_embeddings
                ),
            });
        }
        let image_inputs = images
            .iter()
            .map(|image| {
                preprocess_image(
                    image,
                    &self.image_cfg,
                    &self.cfg.vision_config,
                    &self.device,
                    self.dtype,
                )
            })
            .collect::<Result<Vec<_>, _>>()?;
        let prompt = build_request_prompt(
            image_inputs.iter().map(|image| image.num_image_tokens),
            task_prompt,
        );
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| Error::InvalidInput {
                message: format!("OvisOCR2: tokenizer encode failed: {e}"),
            })?;
        let input_ids = encoding.get_ids().to_vec();
        if input_ids.is_empty() {
            return Err(Error::InvalidInput {
                message: "OvisOCR2: prompt tokenization produced no tokens".to_string(),
            });
        }
        validate_generation_length(
            input_ids.len(),
            max_new_tokens,
            self.cfg.text_config.max_position_embeddings,
        )?;

        let inputs_embeds = self.prepare_inputs(&input_ids, &image_inputs)?;
        let grids: Vec<_> = image_inputs.iter().map(|image| image.grid_thw).collect();
        let (position_ids, rope_delta) = build_request_position_ids(
            &input_ids,
            &grids,
            self.cfg.vision_config.spatial_merge_size,
            self.image_token_id,
            &self.device,
        )?;
        let hidden = self.text.forward(&inputs_embeds, &position_ids)?;
        let prompt_len = input_ids.len();
        let last_hidden = hidden
            .i((0, prompt_len - 1, ..))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "select prompt hidden", e))?;
        let mut logits = self.logits_from_hidden(&last_hidden)?;
        let mut generated = Vec::new();
        generated
            .try_reserve_exact(max_new_tokens)
            .map_err(|e| Error::InvalidInput {
                message: format!("OvisOCR2 cannot reserve output for {max_new_tokens} tokens: {e}"),
            })?;

        // Record the decode bucket once the prefill has populated the KV
        // and linear-attention states; the capture itself is lazy and runs
        // at the first decode step, so a page whose first token is a stop
        // token never pays for it. A no-op off CUDA or when ineligible.
        self.text.prepare_decode_graph(prompt_len, max_new_tokens)?;

        for step in 0..max_new_tokens {
            if let Some(step_top) = step_top.as_deref_mut() {
                step_top.push(
                    top3(&logits)
                        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "trace logits", e))?,
                );
            }
            let token = select_greedy_token(&logits)?;
            if self.stop_token_ids.contains(&token) {
                break;
            }
            generated.push(token);
            if step + 1 == max_new_tokens {
                break;
            }

            let token_ids = Tensor::from_vec(vec![token], (1, 1), &self.device).map_err(|e| {
                candle_to_ocr_processing(
                    crate::error::ProcessingStage::TensorOperation,
                    "OvisOCR2: create decode token",
                    e,
                )
            })?;
            let token_embed = self.text.embed(&token_ids)?;
            let position = prompt_len as i64 + step as i64 + rope_delta;
            let position_ids = text_position_ids(position, &self.device)?;
            logits =
                match self
                    .text
                    .decode_step_graph(&token_embed, &position_ids, &self.lm_head)?
                {
                    Some(logits) => logits,
                    None => {
                        let hidden = self.text.forward(&token_embed, &position_ids)?;
                        self.logits_from_hidden(&hidden.i((0, 0, ..)).map_err(|e| {
                            candle_to_ocr_inference(MODEL_NAME, "select decode hidden", e)
                        })?)?
                    }
                };
        }
        Ok(generated)
    }

    fn prepare_inputs(
        &self,
        input_ids: &[u32],
        images: &[OvisOcr2ImageInputs],
    ) -> Result<Tensor, Error> {
        let spans = image_spans(input_ids, self.image_token_id);
        if spans.len() != images.len() {
            return Err(Error::invalid_input(
                "OvisOCR image groups do not match supplied images",
            ));
        }
        let token_ids = Tensor::from_vec(input_ids.to_vec(), (1, input_ids.len()), &self.device)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "prompt ids", e))?;
        let embeds = self.text.embed(&token_ids)?;
        let mut parts = Vec::new();
        let mut cursor = 0;
        for ((start, len), image) in spans.into_iter().zip(images) {
            let visual = self
                .vision
                .forward(&image.pixel_values, image.grid_thw)?
                .to_dtype(self.dtype)
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "image embeddings", e))?;
            if visual
                .dim(1)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "image token count", e))?
                != len
            {
                return Err(Error::invalid_input(
                    "OvisOCR visual token count does not match prompt",
                ));
            }
            if start > cursor {
                parts.push(
                    embeds
                        .narrow(1, cursor, start - cursor)
                        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "prompt text", e))?,
                );
            }
            parts.push(visual);
            cursor = start + len;
        }
        if cursor < input_ids.len() {
            parts.push(
                embeds
                    .narrow(1, cursor, input_ids.len() - cursor)
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "prompt suffix", e))?,
            );
        }
        Tensor::cat(&parts, 1)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "multimodal embeddings", e))
    }

    fn logits_from_hidden(&self, hidden: &Tensor) -> Result<Tensor, Error> {
        self.lm_head
            .forward(
                &hidden
                    .unsqueeze(0)
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "LM head input", e))?,
            )
            .and_then(|logits| logits.squeeze(0))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "language model head", e))
    }

    /// Decode generated token ids and apply the official truncated-repeat cleanup.
    pub fn decode_tokens(&self, tokens: &[u32]) -> Result<String, Error> {
        Ok(clean_truncated_repeats(&self.decode_tokens_raw(tokens)?))
    }

    /// Decode token ids without OvisOCR2 post-processing.
    pub fn decode_tokens_raw(&self, tokens: &[u32]) -> Result<String, Error> {
        self.tokenizer
            .decode(tokens, true)
            .map(|text| text.trim().to_string())
            .map_err(|e| Error::InvalidInput {
                message: format!("OvisOCR2: tokenizer decode failed: {e}"),
            })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn config(&self) -> &OvisOcr2Config {
        &self.cfg
    }

    pub fn image_processor_config(&self) -> &OvisOcr2ImageProcessorConfig {
        &self.image_cfg
    }
}

fn require_token_id(
    tokenizer: &Tokenizer,
    token: &str,
    expected: Option<u32>,
) -> Result<u32, Error> {
    let token_id = tokenizer.token_to_id(token).ok_or_else(|| Error::Config {
        message: format!("OvisOCR2 tokenizer is missing required token {token:?}"),
    })?;
    if let Some(expected) = expected
        && token_id != expected
    {
        return Err(Error::Config {
            message: format!(
                "OvisOCR2 token {token:?} id mismatch: tokenizer {token_id} != config {expected}"
            ),
        });
    }
    Ok(token_id)
}

fn build_stop_token_ids(config_eos: u32, tokenizer_eos: u32) -> Vec<u32> {
    let mut token_ids = vec![config_eos, tokenizer_eos];
    token_ids.sort_unstable();
    token_ids.dedup();
    token_ids
}

fn validate_generation_length(
    prompt_len: usize,
    max_new_tokens: usize,
    context_limit: usize,
) -> Result<(), Error> {
    let requested = prompt_len
        .checked_add(max_new_tokens)
        .ok_or_else(|| Error::InvalidInput {
            message: "OvisOCR2 requested sequence length overflows usize".to_string(),
        })?;
    if requested > context_limit {
        return Err(Error::InvalidInput {
            message: format!(
                "OvisOCR2 prompt ({prompt_len}) plus max_new_tokens ({max_new_tokens}) exceeds context limit {context_limit}"
            ),
        });
    }
    Ok(())
}

fn build_request_prompt(counts: impl IntoIterator<Item = usize>, task_prompt: &str) -> String {
    let mut prompt = String::from("<|im_start|>user\n");
    for count in counts {
        prompt.push_str("<|vision_start|>");
        for _ in 0..count {
            prompt.push_str("<|image_pad|>");
        }
        prompt.push_str("<|vision_end|>");
    }
    prompt.push_str(task_prompt);
    prompt.push_str("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    prompt
}

fn image_spans(ids: &[u32], image_token_id: u32) -> Vec<(usize, usize)> {
    let mut spans = Vec::new();
    let mut cursor = 0;
    while cursor < ids.len() {
        if ids[cursor] != image_token_id {
            cursor += 1;
            continue;
        }
        let start = cursor;
        while cursor < ids.len() && ids[cursor] == image_token_id {
            cursor += 1;
        }
        spans.push((start, cursor - start));
    }
    spans
}

fn build_request_position_ids(
    input_ids: &[u32],
    grids: &[(usize, usize, usize)],
    merge: usize,
    image_token_id: u32,
    device: &Device,
) -> Result<(Tensor, i64), Error> {
    let spans = image_spans(input_ids, image_token_id);
    if merge == 0 || spans.len() != grids.len() || spans.is_empty() {
        return Err(Error::invalid_input(
            "OvisOCR image grids do not match prompt groups",
        ));
    }
    let mut axes = [Vec::new(), Vec::new(), Vec::new()];
    let mut cursor = 0;
    let mut position = 0i64;
    for ((start, len), &(t, h, w)) in spans.into_iter().zip(grids) {
        if !h.is_multiple_of(merge)
            || !w.is_multiple_of(merge)
            || t == 0
            || h == 0
            || w == 0
            || len != t * (h / merge) * (w / merge)
        {
            return Err(Error::invalid_input(
                "OvisOCR prompt image tokens do not match merged grid",
            ));
        }
        for _ in cursor..start {
            for axis in &mut axes {
                axis.push(position);
            }
            position += 1;
        }
        let (h, w) = (h / merge, w / merge);
        for temporal in 0..t {
            for row in 0..h {
                for col in 0..w {
                    axes[0].push(position + temporal as i64);
                    axes[1].push(position + row as i64);
                    axes[2].push(position + col as i64);
                }
            }
        }
        position += t.max(h).max(w) as i64;
        cursor = start + len;
    }
    for _ in cursor..input_ids.len() {
        for axis in &mut axes {
            axis.push(position);
        }
        position += 1;
    }
    let delta = position - input_ids.len() as i64;
    let tensor = Tensor::from_vec(
        axes.into_iter().flatten().collect::<Vec<_>>(),
        (3, 1, input_ids.len()),
        device,
    )
    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "multimodal positions", e))?;
    Ok((tensor, delta))
}

fn text_position_ids(position: i64, device: &Device) -> Result<Tensor, Error> {
    Tensor::from_vec(vec![position; 3], (3, 1, 1), device).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            "OvisOCR2: create decode position ids",
            e,
        )
    })
}

fn select_greedy_token(logits: &Tensor) -> Result<u32, Error> {
    #[cfg(feature = "cuda")]
    if logits.device().is_cuda() && matches!(logits.dtype(), DType::BF16 | DType::F32) {
        let vocab_size = logits.elem_count();
        let logits = logits
            .reshape((1, vocab_size))
            .and_then(|logits| logits.contiguous())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "reshape GPU logits", e))?;
        let tokens = match logits.dtype() {
            DType::BF16 => logits.apply_op1_no_bwd(&ArgmaxFirstBf16),
            DType::F32 => logits.apply_op1_no_bwd(&ArgmaxFirstF32),
            _ => unreachable!("dtype checked above"),
        }
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "stable GPU argmax", e))?;
        return tokens
            .i(0)
            .and_then(|token| token.to_scalar::<u32>())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "copy selected token", e));
    }

    logits
        .argmax(candle_core::D::Minus1)
        .and_then(|token| token.to_scalar::<u32>())
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "OvisOCR2: greedy argmax",
                e,
            )
        })
}

/// Remove model-emitted visual-region image-tag blocks, matching upstream.
pub fn filter_visual_image_tags(text: &str) -> String {
    text.split("\n\n")
        .filter(|block| !block.trim().starts_with("<img src=\"images/bbox_"))
        .collect::<Vec<_>>()
        .join("\n\n")
}

pub(crate) fn postprocess_text(text: String, keep_image_tags: bool) -> String {
    let text = if keep_image_tags {
        text
    } else {
        filter_visual_image_tags(&text)
    };
    clean_truncated_repeats(&text)
}

pub(crate) fn postprocess_recognition_text(text: String, task: RecognitionTask) -> String {
    postprocess_text(text, task == RecognitionTask::Chart)
}

// The truncated-tail cleaner is shared with Xiaomi-OCR-0 (both models'
// official post-processing carries the same heuristic); it lives in the
// render layer and is re-exported here to preserve the public path.
pub use crate::render::text::clean_truncated_repeats;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn joint_image_grids_advance_text_positions_between_images() {
        let ids = [9, 8, 42, 42, 42, 42, 42, 42, 7, 8, 42, 42, 7, 9];
        let (pos, delta) =
            build_request_position_ids(&ids, &[(1, 4, 6), (1, 2, 4)], 2, 42, &Device::Cpu).unwrap();
        let axes = pos.to_vec3::<i64>().unwrap();
        assert_eq!(axes[0][0], [0, 1, 2, 2, 2, 2, 2, 2, 5, 6, 7, 7, 9, 10]);
        assert_eq!(axes[1][0], [0, 1, 2, 2, 2, 3, 3, 3, 5, 6, 7, 7, 9, 10]);
        assert_eq!(axes[2][0], [0, 1, 2, 3, 4, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        assert_eq!(delta, -3);
        assert!(build_request_position_ids(&ids, &[(1, 4, 6)], 2, 42, &Device::Cpu).is_err());
        let prompt = build_request_prompt([6, 2], "table prompt");
        assert_eq!(prompt.matches("<|vision_start|>").count(), 2);
        assert_eq!(prompt.matches("<|image_pad|>").count(), 8);
    }

    #[test]
    fn stop_tokens_include_model_and_tokenizer_eos() {
        assert_eq!(build_stop_token_ids(248_044, 248_046), [248_044, 248_046]);
        assert_eq!(build_stop_token_ids(248_044, 248_044), [248_044]);
    }

    #[test]
    fn greedy_argmax_prefers_the_first_tied_token() {
        let logits = Tensor::from_vec(vec![1f32, 3., 3., 2.], 4, &Device::Cpu).unwrap();
        assert_eq!(select_greedy_token(&logits).unwrap(), 1);
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cuda_greedy_argmax_prefers_the_first_tied_token() -> candle_core::Result<()> {
        let Ok(device) = Device::new_cuda(0) else {
            return Ok(());
        };
        for dtype in [DType::F32, DType::BF16] {
            let logits = Tensor::from_vec(vec![1f32, 3., 3., 2.], 4, &device)?.to_dtype(dtype)?;
            assert_eq!(select_greedy_token(&logits).unwrap(), 1);
        }
        Ok(())
    }

    #[test]
    fn generation_length_is_checked_without_overflow() {
        validate_generation_length(775, 16_384, 262_144).unwrap();
        assert!(validate_generation_length(775, 262_000, 262_144).is_err());
        assert!(validate_generation_length(1, usize::MAX, usize::MAX).is_err());
    }

    #[test]
    fn official_prompt_has_exact_no_think_framing() {
        let prompt = build_request_prompt([2], DEFAULT_PROMPT);
        assert!(prompt.starts_with(
            "<|im_start|>user\n<|vision_start|><|image_pad|><|image_pad|><|vision_end|>\nExtract"
        ));
        assert!(prompt.ends_with("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"));
    }

    #[test]
    fn layout_golden_position_ids_match_upstream() {
        let mut ids = vec![1u32; 775];
        ids[4..652].fill(248_056);
        let (positions, delta) =
            build_request_position_ids(&ids, &[(1, 54, 48)], 2, 248_056, &Device::Cpu).unwrap();
        assert_eq!(delta, -621);
        let positions = positions.to_vec3::<i64>().unwrap();
        assert_eq!(positions[0][0][4], 4);
        assert_eq!(positions[1][0][651], 30);
        assert_eq!(positions[2][0][651], 27);
        assert_eq!(positions[0][0][652], 31);
        assert_eq!(positions[0][0][774], 153);
    }

    #[test]
    fn visual_image_tag_blocks_are_removed() {
        let text = "before\n\n<img src=\"images/bbox_1_2_3_4.jpg\" />\n\nafter";
        assert_eq!(filter_visual_image_tags(text), "before\n\nafter");
    }

    #[test]
    fn chart_recognition_keeps_visual_image_tags() {
        let text = "<img src=\"images/bbox_1_2_3_4.jpg\" />";
        assert_eq!(
            postprocess_recognition_text(text.to_string(), RecognitionTask::Chart),
            text
        );
    }

    #[test]
    fn mixed_recognition_tasks_filter_only_non_charts() {
        let text = "<img src=\"images/bbox_1_2_3_4.jpg\" />";
        let tasks = [
            RecognitionTask::Ocr,
            RecognitionTask::Chart,
            RecognitionTask::Table,
            RecognitionTask::Formula,
        ];
        let outputs = tasks.map(|task| postprocess_recognition_text(text.to_string(), task));

        assert_eq!(outputs, ["", text, "", ""]);
    }
}
