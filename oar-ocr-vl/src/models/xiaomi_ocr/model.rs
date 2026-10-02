use super::config::{XiaomiOcrConfig, XiaomiOcrProcessorConfig};
use super::processing::{XiaomiOcrImageInputs, preprocess_image};
use crate::backbones::qwen3_5::text::Qwen35TextModel;
use crate::backbones::qwen3_vl::Qwen3VlVisionModel;
use crate::error::Error;
use crate::render::table::convert_otsl_to_html;
use crate::render::text::clean_truncated_repeats;
#[cfg(feature = "cuda")]
use crate::runtime::cuda::{ArgmaxFirstBf16, ArgmaxFirstF32};
use crate::utils::{candle_to_ocr_inference, candle_to_ocr_processing};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use image::RgbImage;
use once_cell::sync::Lazy;
use regex::Regex;
use std::path::Path;
use tokenizers::Tokenizer;

const MODEL_NAME: &str = "Xiaomi-OCR-0";

/// Env var that forces the decode-graph path off for this model (the shared
/// `OAR_VL_DISABLE_CUDA_GRAPH` switch applies on top of it).
const GRAPH_DISABLE_ENV: &str = "OAR_XIAOMI_OCR_DISABLE_CUDA_GRAPH";

/// Official Xiaomi-OCR-0 whole-page document parsing instruction.
pub const DEFAULT_PROMPT: &str = "Extract all information from the main body of the document image and represent it in markdown format, ignoring headers and footers. Tables should be expressed in OTSL format, formulas in the document should be represented using LATEX format, and the parsing should be organized according to the reading order.";

/// Official text-region instruction.
pub const TEXT_REGION_PROMPT: &str = "Extract the text in the image.";

/// Official table-region instruction (the output is OTSL; the pipeline's
/// `table_output_is_otsl` capability converts it to HTML).
pub const TABLE_REGION_PROMPT: &str = "Parse the table in the image into OTSL.";

/// Official formula-region instruction.
pub const FORMULA_REGION_PROMPT: &str =
    "Identify the formula in the image and represent it using LATEX format.";

/// Official key-information-extraction instruction prefix; a JSON schema
/// follows it (see [`key_information_extraction_prompt`]).
pub const KIE_PROMPT: &str = "Extract key information in the image";

/// Generation limit used by the official Xiaomi-OCR-0 example.
pub const DEFAULT_MAX_NEW_TOKENS: usize = 4_096;

/// Blank-line separator between Markdown blocks (the official post-processing
/// splits on `\n\s*\n` with capture-preserving splits).
static MARKDOWN_BLOCK_SEP_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"\n\s*\n").expect("static regex"));

/// Build the official key-information-extraction prompt: the instruction
/// followed by the caller's JSON schema.
///
/// The model card pairs the fixed instruction with a schema such as
/// `Please output the key information in JSON format according to the
/// following schema: {...}`.
pub fn key_information_extraction_prompt(schema: &str) -> String {
    format!("{KIE_PROMPT}\n\n{}", schema.trim())
}

/// End-to-end Xiaomi-OCR-0 page parser backed by Qwen3.5-0.8B.
pub struct XiaomiOcr {
    device: Device,
    dtype: DType,
    cfg: XiaomiOcrConfig,
    processor_cfg: XiaomiOcrProcessorConfig,
    tokenizer: Tokenizer,
    text: Qwen35TextModel,
    vision: Qwen3VlVisionModel,
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

impl XiaomiOcr {
    /// Load a Xiaomi-OCR-0 Hugging Face model directory.
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, Error> {
        Self::from_dir_with_runtime(model_dir, crate::RuntimeConfig::new(device))
    }

    pub fn from_dir_with_runtime(
        model_dir: impl AsRef<Path>,
        runtime: crate::RuntimeConfig,
    ) -> Result<Self, Error> {
        let (device, dtype) = runtime.resolve();
        let model_dir = model_dir.as_ref();
        let cfg = XiaomiOcrConfig::from_path(model_dir.join("config.json"))?;
        let processor_cfg =
            XiaomiOcrProcessorConfig::from_path(model_dir.join("processor_config.json"))?;
        processor_cfg.validate_vision_compatibility(&cfg.vision_config)?;
        let tokenizer =
            Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| Error::Config {
                message: format!("failed to load Xiaomi-OCR-0 tokenizer.json: {e}"),
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
            GRAPH_DISABLE_ENV,
            vb.pp("model").pp("language_model"),
        )?;
        let vision = Qwen3VlVisionModel::load(
            &cfg.vision_config.to_qwen3_vl(),
            vb.pp("model").pp("visual"),
        )?;
        // Xiaomi-OCR-0 ties the language-model output projection to token
        // embeddings (the checkpoint additionally carries unused MTP weights
        // that this inference path never reads).
        let lm_head = Linear::new(text.token_embedding_weight(), None);

        // Stop on both the text tower's EOS (`<|endoftext|>`), the top-level
        // model-config EOS (`<|im_end|>`), and the tokenizer's EOS token.
        let stop_token_ids = build_stop_token_ids(
            cfg.text_config.eos_token_id,
            cfg.eos_token_id,
            tokenizer_eos,
        );
        let image_token_id = cfg.image_token_id;
        #[cfg(feature = "cuda")]
        let drain_guard = crate::runtime::decoder_graph::CudaGraphDrainGuard::new(&device);
        Ok(Self {
            device,
            dtype,
            cfg,
            processor_cfg,
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

    /// Generate raw token ids for each input page using the official
    /// whole-page instruction.
    pub fn generate_tokens(
        &self,
        images: &[RgbImage],
        max_new_tokens: usize,
    ) -> crate::error::BatchResult<Vec<u32>> {
        Ok(images
            .iter()
            .map(|image| self.generate_one(image, DEFAULT_PROMPT, max_new_tokens))
            .collect())
    }

    /// Generate text with the official whole-page instruction, applying the
    /// shared truncated-tail cleanup to each decode.
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

    /// Parse pages using the official whole-page post-processing: collapse
    /// degenerate repeated tails and convert OTSL table blocks to HTML.
    pub fn parse(
        &self,
        images: &[RgbImage],
        max_new_tokens: usize,
    ) -> crate::error::BatchResult<String> {
        Ok(self
            .generate_tokens(images, max_new_tokens)?
            .into_iter()
            .map(|result| {
                result.and_then(|tokens| {
                    self.decode_tokens(&tokens)
                        .map(|text| finalize_markdown(&text))
                })
            })
            .collect())
    }

    /// Generate raw token ids for one image under an arbitrary instruction
    /// (the official whole-page prompt or a task-region prompt).
    pub fn generate_tokens_with_prompt(
        &self,
        image: &RgbImage,
        instruction: &str,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>, Error> {
        self.generate_one(image, instruction, max_new_tokens)
    }

    fn generate_one(
        &self,
        image: &RgbImage,
        instruction: &str,
        max_new_tokens: usize,
    ) -> Result<Vec<u32>, Error> {
        self.text.clear_cache();
        let _cache_guard = TextCacheGuard(&self.text);
        if max_new_tokens == 0 {
            return Ok(Vec::new());
        }
        if max_new_tokens > self.cfg.text_config.max_position_embeddings {
            return Err(Error::InvalidInput {
                message: format!(
                    "Xiaomi-OCR-0 max_new_tokens {max_new_tokens} exceeds context limit {}",
                    self.cfg.text_config.max_position_embeddings
                ),
            });
        }
        let image_inputs = preprocess_image(
            image,
            &self.processor_cfg,
            &self.cfg.vision_config,
            &self.device,
            self.dtype,
        )?;
        let prompt = build_prompt(instruction, image_inputs.num_image_tokens);
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| Error::InvalidInput {
                message: format!("Xiaomi-OCR-0: tokenizer encode failed: {e}"),
            })?;
        let input_ids = encoding.get_ids().to_vec();
        if input_ids.is_empty() {
            return Err(Error::InvalidInput {
                message: "Xiaomi-OCR-0: prompt tokenization produced no tokens".to_string(),
            });
        }
        validate_generation_length(
            input_ids.len(),
            max_new_tokens,
            self.cfg.text_config.max_position_embeddings,
        )?;

        let inputs_embeds = self.prepare_inputs(&input_ids, &image_inputs)?;
        let (position_ids, rope_delta) = build_position_ids(
            &input_ids,
            image_inputs.grid_thw,
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
                message: format!(
                    "Xiaomi-OCR-0 cannot reserve output for {max_new_tokens} tokens: {e}"
                ),
            })?;

        // Record the decode bucket once the prefill has populated the KV
        // and linear-attention states; the capture itself is lazy and runs
        // at the first decode step, so a page whose first token is a stop
        // token never pays for it. A no-op off CUDA or when ineligible.
        self.text.prepare_decode_graph(prompt_len, max_new_tokens)?;

        for step in 0..max_new_tokens {
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
                    "Xiaomi-OCR-0: create decode token",
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
        image_inputs: &XiaomiOcrImageInputs,
    ) -> Result<Tensor, Error> {
        let seq_len = input_ids.len();
        let token_ids = Tensor::from_vec(input_ids.to_vec(), (1, seq_len), &self.device)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "create prompt token ids", e))?;
        let embeds = self.text.embed(&token_ids)?;
        let image_embeds = self
            .vision
            .forward(&image_inputs.pixel_values, &[image_inputs.grid_thw])?
            .0
            .to_dtype(self.dtype)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "cast image embeddings", e))?;

        let image_positions: Vec<usize> = input_ids
            .iter()
            .enumerate()
            .filter_map(|(index, &token)| (token == self.image_token_id).then_some(index))
            .collect();
        let image_len = image_embeds
            .dim(0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "image embedding length", e))?;
        if image_positions.len() != image_len || image_positions.is_empty() {
            return Err(Error::InvalidInput {
                message: format!(
                    "Xiaomi-OCR-0: image placeholder count ({}) != image embedding count ({image_len})",
                    image_positions.len()
                ),
            });
        }
        let start = image_positions[0];
        if image_positions
            .iter()
            .enumerate()
            .any(|(offset, &position)| position != start + offset)
        {
            return Err(Error::InvalidInput {
                message: "Xiaomi-OCR-0: image placeholder tokens must be contiguous".to_string(),
            });
        }
        let end = start + image_positions.len();
        let hidden_size = embeds
            .dim(2)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "embedding hidden size", e))?;
        let prefix = if start == 0 {
            Tensor::zeros((1, 0, hidden_size), embeds.dtype(), embeds.device())
        } else {
            embeds.narrow(1, 0, start)
        }
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "embedding prefix", e))?;
        let suffix = if end == seq_len {
            Tensor::zeros((1, 0, hidden_size), embeds.dtype(), embeds.device())
        } else {
            embeds.narrow(1, end, seq_len - end)
        }
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "embedding suffix", e))?;
        let image_embeds = image_embeds
            .unsqueeze(0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "image embedding batch", e))?;
        Tensor::cat(&[&prefix, &image_embeds, &suffix], 1)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "merge multimodal embeddings", e))
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

    /// Decode generated token ids and apply the shared truncated-repeat
    /// cleanup.
    pub fn decode_tokens(&self, tokens: &[u32]) -> Result<String, Error> {
        Ok(clean_truncated_repeats(&self.decode_tokens_raw(tokens)?))
    }

    /// Decode token ids without any post-processing.
    pub fn decode_tokens_raw(&self, tokens: &[u32]) -> Result<String, Error> {
        self.tokenizer
            .decode(tokens, true)
            .map(|text| text.trim().to_string())
            .map_err(|e| Error::InvalidInput {
                message: format!("Xiaomi-OCR-0: tokenizer decode failed: {e}"),
            })
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub fn config(&self) -> &XiaomiOcrConfig {
        &self.cfg
    }

    pub fn processor_config(&self) -> &XiaomiOcrProcessorConfig {
        &self.processor_cfg
    }
}

fn require_token_id(
    tokenizer: &Tokenizer,
    token: &str,
    expected: Option<u32>,
) -> Result<u32, Error> {
    let token_id = tokenizer.token_to_id(token).ok_or_else(|| Error::Config {
        message: format!("Xiaomi-OCR-0 tokenizer is missing required token {token:?}"),
    })?;
    if let Some(expected) = expected
        && token_id != expected
    {
        return Err(Error::Config {
            message: format!(
                "Xiaomi-OCR-0 token {token:?} id mismatch: tokenizer {token_id} != config {expected}"
            ),
        });
    }
    Ok(token_id)
}

fn build_stop_token_ids(
    text_config_eos: u32,
    top_level_eos: Option<u32>,
    tokenizer_eos: u32,
) -> Vec<u32> {
    let mut token_ids = [Some(text_config_eos), top_level_eos, Some(tokenizer_eos)]
        .into_iter()
        .flatten()
        .collect::<Vec<u32>>();
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
            message: "Xiaomi-OCR-0 requested sequence length overflows usize".to_string(),
        })?;
    if requested > context_limit {
        return Err(Error::InvalidInput {
            message: format!(
                "Xiaomi-OCR-0 prompt ({prompt_len}) plus max_new_tokens ({max_new_tokens}) exceeds context limit {context_limit}"
            ),
        });
    }
    Ok(())
}

/// Frame an instruction the way the official chat template does for a
/// non-thinking request: one image span, the instruction text, and the empty
/// think block before the assistant turn.
fn build_prompt(instruction: &str, num_image_tokens: usize) -> String {
    let mut prompt = String::with_capacity(instruction.len() + num_image_tokens * 13 + 128);
    prompt.push_str("<|im_start|>user\n<|vision_start|>");
    for _ in 0..num_image_tokens {
        prompt.push_str("<|image_pad|>");
    }
    prompt.push_str("<|vision_end|>");
    prompt.push_str(instruction);
    prompt.push_str("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n");
    prompt
}

fn build_position_ids(
    input_ids: &[u32],
    grid_thw: (usize, usize, usize),
    spatial_merge_size: usize,
    image_token_id: u32,
    device: &Device,
) -> Result<(Tensor, i64), Error> {
    let image_start = input_ids
        .iter()
        .position(|&token| token == image_token_id)
        .ok_or_else(|| Error::InvalidInput {
            message: "Xiaomi-OCR-0: image token missing from prompt".to_string(),
        })?;
    let image_len = input_ids
        .iter()
        .skip(image_start)
        .take_while(|&&token| token == image_token_id)
        .count();
    if input_ids[image_start + image_len..].contains(&image_token_id) {
        return Err(Error::InvalidInput {
            message: "Xiaomi-OCR-0: non-contiguous image token span".to_string(),
        });
    }
    let (grid_t, grid_h, grid_w) = grid_thw;
    if spatial_merge_size == 0
        || !grid_h.is_multiple_of(spatial_merge_size)
        || !grid_w.is_multiple_of(spatial_merge_size)
    {
        return Err(Error::Config {
            message: format!(
                "Xiaomi-OCR-0: invalid image grid {grid_thw:?} for merge size {spatial_merge_size}"
            ),
        });
    }
    let llm_h = grid_h / spatial_merge_size;
    let llm_w = grid_w / spatial_merge_size;
    if image_len != grid_t * llm_h * llm_w {
        return Err(Error::InvalidInput {
            message: format!(
                "Xiaomi-OCR-0: image token count {image_len} != merged grid token count {}",
                grid_t * llm_h * llm_w
            ),
        });
    }

    let seq_len = input_ids.len();
    let mut axes = [
        Vec::with_capacity(seq_len),
        Vec::with_capacity(seq_len),
        Vec::with_capacity(seq_len),
    ];
    for position in 0..image_start as i64 {
        for axis in &mut axes {
            axis.push(position);
        }
    }
    let vision_start = image_start as i64;
    for temporal in 0..grid_t {
        for row in 0..llm_h {
            for col in 0..llm_w {
                axes[0].push(vision_start + temporal as i64);
                axes[1].push(vision_start + row as i64);
                axes[2].push(vision_start + col as i64);
            }
        }
    }
    let text_start = vision_start + llm_h.max(llm_w) as i64;
    for (offset, _) in (image_start + image_len..seq_len).enumerate() {
        let current = text_start + offset as i64;
        for axis in &mut axes {
            axis.push(current);
        }
    }
    let max_position = axes
        .iter()
        .flat_map(|axis| axis.iter())
        .copied()
        .max()
        .unwrap_or(0);
    let rope_delta = max_position + 1 - seq_len as i64;
    let data: Vec<i64> = axes.into_iter().flatten().collect();
    let tensor = Tensor::from_vec(data, (3, 1, seq_len), device).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            "Xiaomi-OCR-0: create multimodal position ids",
            e,
        )
    })?;
    Ok((tensor, rope_delta))
}

fn text_position_ids(position: i64, device: &Device) -> Result<Tensor, Error> {
    Tensor::from_vec(vec![position; 3], (3, 1, 1), device).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            "Xiaomi-OCR-0: create decode position ids",
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
                "Xiaomi-OCR-0: greedy argmax",
                e,
            )
        })
}

/// Official whole-page post-processing: collapse degenerate repeated tails,
/// then convert OTSL table blocks to HTML tables.
pub fn finalize_markdown(text: &str) -> String {
    convert_markdown_otsl_tables(&clean_truncated_repeats(text))
}

/// Convert OTSL table blocks inside whole-page Markdown to HTML tables,
/// matching the official post-processing: blank-line-separated blocks that
/// contain OTSL tags are converted; fenced code blocks and their fences pass
/// through untouched, and blocks whose conversion comes back empty keep their
/// original text.
pub fn convert_markdown_otsl_tables(markdown: &str) -> String {
    const OTSL_TAGS: [&str; 6] = ["<fcel>", "<ecel>", "<nl>", "<lcel>", "<ucel>", "<xcel>"];
    if markdown.is_empty() || !OTSL_TAGS.iter().any(|tag| markdown.contains(tag)) {
        return markdown.to_string();
    }

    // Split into alternating `[block, separator, block, ...]` slices so the
    // exact blank-line separators survive the round trip.
    let mut parts: Vec<&str> = Vec::new();
    let mut last = 0;
    for separator in MARKDOWN_BLOCK_SEP_RE.find_iter(markdown) {
        parts.push(&markdown[last..separator.start()]);
        parts.push(separator.as_str());
        last = separator.end();
    }
    parts.push(&markdown[last..]);

    let mut output = String::with_capacity(markdown.len());
    let mut in_fence = false;
    for (index, part) in parts.iter().enumerate() {
        if index % 2 == 1 {
            output.push_str(part);
            continue;
        }
        let block = part;
        let fence_count = block.matches("```").count();
        // The fence lines themselves and anything inside an open fence are
        // code, not a stray OTSL table.
        let skip = in_fence || fence_count > 0;
        if fence_count % 2 == 1 {
            in_fence = !in_fence;
        }
        if skip || !OTSL_TAGS.iter().any(|tag| block.contains(tag)) {
            output.push_str(block);
            continue;
        }
        let converted = convert_otsl_to_html(block.trim());
        if converted.trim().is_empty() {
            output.push_str(block);
        } else {
            output.push_str(&converted);
        }
    }
    output
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stop_tokens_union_text_top_level_and_tokenizer_eos() {
        // Released checkpoint: tower EOS 248044 (<|endoftext|>), top-level
        // EOS 248046 (<|im_end|>), tokenizer EOS 248046.
        assert_eq!(
            build_stop_token_ids(248_044, Some(248_046), 248_046),
            [248_044, 248_046]
        );
        assert_eq!(build_stop_token_ids(248_044, None, 248_044), [248_044]);
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
        validate_generation_length(775, 4_096, 262_144).unwrap();
        assert!(validate_generation_length(775, 262_000, 262_144).is_err());
        assert!(validate_generation_length(1, usize::MAX, usize::MAX).is_err());
    }

    #[test]
    fn official_prompts_frame_the_document_instruction() {
        let prompt = build_prompt(DEFAULT_PROMPT, 2);
        assert!(prompt.starts_with(
            "<|im_start|>user\n<|vision_start|><|image_pad|><|image_pad|><|vision_end|>Extract"
        ));
        assert!(prompt.ends_with("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"));
        assert_eq!(TEXT_REGION_PROMPT, "Extract the text in the image.");
        assert_eq!(
            TABLE_REGION_PROMPT,
            "Parse the table in the image into OTSL."
        );
        assert!(FORMULA_REGION_PROMPT.starts_with("Identify the formula"));
        assert_eq!(
            key_information_extraction_prompt(
                "Please output the key information in JSON format according to the following schema:\n{\"date\": \"\"}"
            ),
            "Extract key information in the image\n\nPlease output the key information in JSON format according to the following schema:\n{\"date\": \"\"}"
        );
    }

    #[test]
    fn layout_golden_position_ids_match_upstream() {
        let mut ids = vec![1u32; 775];
        ids[4..652].fill(248_056);
        let (positions, delta) =
            build_position_ids(&ids, (1, 54, 48), 2, 248_056, &Device::Cpu).unwrap();
        assert_eq!(delta, -621);
        let positions = positions.to_vec3::<i64>().unwrap();
        assert_eq!(positions[0][0][4], 4);
        assert_eq!(positions[1][0][651], 30);
        assert_eq!(positions[2][0][651], 27);
        assert_eq!(positions[0][0][652], 31);
        assert_eq!(positions[0][0][774], 153);
    }

    #[test]
    fn otsl_blocks_in_markdown_become_html_tables() {
        let otsl = "<fcel>Item<fcel>Qty<nl><fcel>Tea<fcel>2<nl>";
        let markdown = format!("# Receipt\n\n{otsl}\n\nThat is all.");
        let converted = convert_markdown_otsl_tables(&markdown);
        assert!(converted.contains("<table>"));
        assert!(converted.contains("Tea"));
        assert!(!converted.contains("<fcel>"));
        // The non-table prose survives with its separators.
        assert!(converted.contains("# Receipt\n\n"));
        assert!(converted.ends_with("That is all."));
    }

    #[test]
    fn fenced_otsl_passes_through_untouched() {
        let markdown = "```text\n<fcel>a<nl>\n```\n\n<fcel>b<nl>";
        let converted = convert_markdown_otsl_tables(markdown);
        assert!(converted.contains("```text\n<fcel>a<nl>\n```"));
        assert!(converted.contains("<table>"));
    }

    #[test]
    fn markdown_without_otsl_tags_is_untouched() {
        let markdown = "# Doc\n\n| a | b |\n|---|---|\n\n```rust\nfn main() {}\n```";
        assert_eq!(convert_markdown_otsl_tables(markdown), markdown);
    }

    #[test]
    fn otsl_conversion_keeps_surrounding_prose() {
        let markdown = "intro\n\n<fcel>keep<nl>\n\noutro";
        let converted = convert_markdown_otsl_tables(markdown);
        assert!(converted.contains("<table>"));
        assert!(converted.contains("intro"));
        assert!(converted.contains("outro"));
    }

    #[test]
    fn otsl_conversion_preserves_separator_shapes() {
        let markdown = "<fcel>a<nl>\n\n\n<fcel>b<nl>";
        let converted = convert_markdown_otsl_tables(markdown);
        // Each block becomes its own table; the triple-newline separator
        // survives between them.
        assert_eq!(converted.matches("<table>").count(), 2);
        assert!(converted.contains("\n\n\n"));
    }
}
