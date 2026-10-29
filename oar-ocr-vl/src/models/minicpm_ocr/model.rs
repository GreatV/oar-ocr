use super::config::{GenerationConfig, MiniCpmOcrConfig, MiniCpmOcrImageProcessorConfig};
use super::infer;
use super::processing::{self, MiniCpmOcrImageInputs};
use super::vision::VisionTower;
use crate::backbones::qwen3_5::text::Qwen35TextModel;
use crate::runtime::checkpoint::{collect_safetensors, load_json_config};
#[cfg(feature = "cuda")]
use crate::runtime::cuda::{ArgmaxFirstBf16, ArgmaxFirstF32};
pub use crate::runtime::trace::GenerationTrace;
use crate::runtime::trace::top3;
use crate::{BatchResult, Error, RuntimeConfig};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder};
use image::RgbImage;
use std::path::Path;
use tokenizers::Tokenizer;

const MODEL_NAME: &str = "MiniCPM-OCR";

/// Region prompts from the official `minicpmvocr` SDK prompt map.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum MiniCpmOcrTask {
    /// Text blocks, titles, inline formulas and whole pages.
    Text,
    /// Tables, answered in OTSL.
    Table,
    /// Display formulas, answered in LaTeX.
    Formula,
    /// Charts, answered as an OTSL data table.
    Chart,
    /// Seals, answered as plain text.
    Seal,
}

impl MiniCpmOcrTask {
    pub const fn prompt(self) -> &'static str {
        match self {
            Self::Text => "Text Recognition:",
            Self::Table => "Table Recognition:",
            Self::Formula => "Formula Recognition:",
            Self::Chart => "Chart to Table:",
            Self::Seal => "Seal Recognition:",
        }
    }

    /// The SDK's per-region output budget for this prompt.
    pub const fn default_max_new_tokens(self) -> usize {
        match self {
            Self::Text | Self::Chart => 4096,
            Self::Table => 8192,
            Self::Formula => 2048,
            Self::Seal => 256,
        }
    }
}

/// Greedy generation and slicing controls.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct MiniCpmOcrGenerationOptions {
    pub max_new_tokens: usize,
    /// Upper bound on high-resolution slices next to the overview image;
    /// defaults to the checkpoint's `max_slice_nums` (9).
    pub max_slice_nums: Option<usize>,
}

impl Default for MiniCpmOcrGenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 8192,
            max_slice_nums: None,
        }
    }
}

impl MiniCpmOcrGenerationOptions {
    pub fn with_max_new_tokens(mut self, value: usize) -> Self {
        self.max_new_tokens = value;
        self
    }

    pub fn with_max_slice_nums(mut self, value: usize) -> Self {
        self.max_slice_nums = Some(value);
        self
    }
}

/// Native MiniCPM-OCR: SigLIP vision tower, 2x2 merger and a Qwen3.5 text
/// decoder with tied embeddings.
pub struct MiniCpmOcr {
    device: Device,
    dtype: DType,
    config: MiniCpmOcrConfig,
    image_config: MiniCpmOcrImageProcessorConfig,
    tokenizer: Tokenizer,
    text: Qwen35TextModel,
    vision: VisionTower,
    lm_head: Linear,
    stop_token_ids: Vec<u32>,
    pub(super) generation_options: MiniCpmOcrGenerationOptions,
    // Must stay the last field: the captured decode graph holds a clone of
    // the tied LM head owned here, so this guard drops last and drains CUDA
    // errors the head's free may stash.
    #[cfg(feature = "cuda")]
    _drain_guard: crate::runtime::decoder_graph::CudaGraphDrainGuard,
}

struct TextCacheGuard<'a>(&'a Qwen35TextModel);

impl Drop for TextCacheGuard<'_> {
    fn drop(&mut self) {
        self.0.clear_cache();
    }
}

impl MiniCpmOcr {
    /// Load an `openbmb/MiniCPM-OCR-1.0` model directory.
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, Error> {
        Self::from_dir_with_runtime(model_dir, RuntimeConfig::new(device))
    }

    pub fn from_dir_with_runtime(
        model_dir: impl AsRef<Path>,
        runtime: RuntimeConfig,
    ) -> Result<Self, Error> {
        let dir = model_dir.as_ref();
        let generation: GenerationConfig = load_json_config(
            dir.join("generation_config.json"),
            MODEL_NAME,
            "generation_config.json",
        )?;
        let Some(&first_stop) = generation.eos_token_id.first() else {
            return Err(Error::config(
                "MiniCPM-OCR generation_config.json has no eos_token_id",
            ));
        };
        let config = MiniCpmOcrConfig::from_path(dir.join("config.json"), first_stop)?;
        let image_config =
            MiniCpmOcrImageProcessorConfig::from_path(dir.join("preprocessor_config.json"))?;
        if image_config.patch_size as usize != config.vision_config.patch_size {
            return Err(Error::config(
                "MiniCPM-OCR processor and vision patch sizes differ",
            ));
        }
        if generation
            .eos_token_id
            .iter()
            .any(|&id| id as usize >= config.text_config.vocab_size)
        {
            return Err(Error::config(
                "MiniCPM-OCR generation_config.json has invalid stop IDs",
            ));
        }
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| Error::config(format!("MiniCPM-OCR tokenizer.json: {e}")))?;
        for token in [
            "<|im_start|>",
            "<|im_end|>",
            "<image>",
            "</image>",
            "<slice>",
            "</slice>",
            "<image_id>",
            "</image_id>",
        ] {
            if tokenizer.token_to_id(token).is_none() {
                return Err(Error::config(format!(
                    "MiniCPM-OCR tokenizer is missing {token}"
                )));
            }
        }
        if tokenizer.token_to_id("<|image_pad|>") != Some(config.image_token_id) {
            return Err(Error::config(
                "MiniCPM-OCR <|image_pad|> does not match image_token_id",
            ));
        }

        let (device, dtype) = runtime.resolve();
        let files = collect_safetensors(dir, MODEL_NAME)?;
        // SAFETY: checkpoint files must remain unchanged while the model is alive.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device) }
            .map_err(infer)?;
        let text = Qwen35TextModel::load(
            &config.text_config,
            MODEL_NAME,
            "OAR_VL_DISABLE_CUDA_GRAPH",
            vb.pp("model.language_model"),
        )?;
        let vision = VisionTower::load(
            &config.vision_config,
            config.text_config.hidden_size,
            vb.pp("model"),
        )?;
        let lm_head = Linear::new(text.token_embedding_weight(), None);
        #[cfg(feature = "cuda")]
        let drain_guard = crate::runtime::decoder_graph::CudaGraphDrainGuard::new(&device);
        Ok(Self {
            device,
            dtype,
            config,
            image_config,
            tokenizer,
            text,
            vision,
            lm_head,
            stop_token_ids: generation.eos_token_id,
            generation_options: MiniCpmOcrGenerationOptions::default(),
            #[cfg(feature = "cuda")]
            _drain_guard: drain_guard,
        })
    }

    /// Set defaults for [`generate`](Self::generate) and region recognition.
    ///
    /// Region recognition overrides `max_new_tokens` with the pipeline's
    /// per-request limit.
    pub fn with_generation_options(mut self, options: MiniCpmOcrGenerationOptions) -> Self {
        self.generation_options = options;
        self
    }

    /// Generate raw text for each image/task pair, in order.
    pub fn generate(
        &self,
        images: &[RgbImage],
        tasks: &[MiniCpmOcrTask],
        max_new_tokens: usize,
    ) -> BatchResult<String> {
        if images.len() != tasks.len() {
            return Err(Error::invalid_input(
                "MiniCPM-OCR image and task counts differ",
            ));
        }
        let options = self
            .generation_options
            .clone()
            .with_max_new_tokens(max_new_tokens);
        Ok(images
            .iter()
            .zip(tasks)
            .map(|(image, task)| self.generate_with_prompt(image, task.prompt(), &options))
            .collect())
    }

    /// Generate with a caller-provided instruction, for example the SDK's
    /// key-information-extraction prompt.
    pub fn generate_with_prompt(
        &self,
        image: &RgbImage,
        instruction: &str,
        options: &MiniCpmOcrGenerationOptions,
    ) -> Result<String, Error> {
        let tokens = self.generate_one(image, instruction, options, None)?;
        self.decode_tokens(&tokens)
    }

    /// Greedy generation with per-step top-3 logits for reference alignment.
    pub fn generate_traced(
        &self,
        image: &RgbImage,
        task: MiniCpmOcrTask,
        options: &MiniCpmOcrGenerationOptions,
    ) -> Result<GenerationTrace, Error> {
        let mut step_top = Vec::new();
        let tokens = self.generate_one(image, task.prompt(), options, Some(&mut step_top))?;
        Ok(GenerationTrace { tokens, step_top })
    }

    /// Resize and slice an image without running the vision tower.
    pub fn preprocess_image(
        &self,
        image: &RgbImage,
        options: &MiniCpmOcrGenerationOptions,
    ) -> Result<MiniCpmOcrImageInputs, Error> {
        processing::preprocess(
            image,
            &self.image_config,
            options
                .max_slice_nums
                .unwrap_or(self.image_config.max_slice_nums),
            &self.device,
            self.dtype,
        )
    }

    /// Token IDs of the full chat prompt for one image and instruction.
    pub fn prompt_ids(
        &self,
        inputs: &MiniCpmOcrImageInputs,
        instruction: &str,
    ) -> Result<Vec<u32>, Error> {
        let prompt = processing::build_prompt(
            &inputs.placeholder(self.image_config.use_image_id),
            instruction,
        );
        let encoding = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| Error::invalid_input(format!("MiniCPM-OCR tokenization failed: {e}")))?;
        Ok(encoding.get_ids().to_vec())
    }

    /// Decode generated tokens without task-specific cleanup.
    pub fn decode_tokens(&self, ids: &[u32]) -> Result<String, Error> {
        self.tokenizer
            .decode(ids, true)
            .map(|text| text.trim().to_string())
            .map_err(|e| Error::invalid_input(format!("MiniCPM-OCR decoding failed: {e}")))
    }

    fn generate_one(
        &self,
        image: &RgbImage,
        instruction: &str,
        options: &MiniCpmOcrGenerationOptions,
        mut step_top: Option<&mut Vec<[(u32, f32); 3]>>,
    ) -> Result<Vec<u32>, Error> {
        self.text.clear_cache();
        let _cache_guard = TextCacheGuard(&self.text);
        let max_new_tokens = options.max_new_tokens;
        if max_new_tokens == 0 {
            return Ok(Vec::new());
        }
        let inputs = self.preprocess_image(image, options)?;
        let input_ids = self.prompt_ids(&inputs, instruction)?;
        let prompt_len = input_ids.len();
        let context = self.config.text_config.max_position_embeddings;
        if prompt_len
            .checked_add(max_new_tokens)
            .is_none_or(|len| len > context)
        {
            return Err(Error::invalid_input(format!(
                "MiniCPM-OCR prompt ({prompt_len}) plus max_new_tokens ({max_new_tokens}) exceeds context limit {context}"
            )));
        }
        let embeds = self.prepare_inputs(&input_ids, &inputs)?;
        let positions: Vec<i64> = (0..prompt_len as i64).collect();
        let position_ids = Tensor::from_vec(positions.repeat(3), (3, 1, prompt_len), &self.device)
            .map_err(infer)?;
        let hidden = self.text.forward(&embeds, &position_ids)?;
        let mut logits = self.logits(&hidden.i((0, prompt_len - 1, ..)).map_err(infer)?)?;
        let mut generated = Vec::new();
        // Record the decode bucket after the prefill; the CUDA graph capture
        // itself waits for the first decode step.
        self.text.prepare_decode_graph(prompt_len, max_new_tokens)?;
        for step in 0..max_new_tokens {
            if let Some(step_top) = step_top.as_deref_mut() {
                step_top.push(top3(&logits).map_err(infer)?);
            }
            let token = select_greedy_token(&logits)?;
            if self.stop_token_ids.contains(&token) {
                break;
            }
            generated.push(token);
            if step + 1 == max_new_tokens {
                break;
            }
            let token_embed = Tensor::from_vec(vec![token], (1, 1), &self.device)
                .map_err(infer)
                .and_then(|ids| self.text.embed(&ids))?;
            let position = (prompt_len + step) as i64;
            let position_ids =
                Tensor::from_vec(vec![position; 3], (3, 1, 1), &self.device).map_err(infer)?;
            logits =
                match self
                    .text
                    .decode_step_graph(&token_embed, &position_ids, &self.lm_head)?
                {
                    Some(logits) => logits,
                    None => {
                        let hidden = self.text.forward(&token_embed, &position_ids)?;
                        self.logits(&hidden.i((0, 0, ..)).map_err(infer)?)?
                    }
                };
        }
        Ok(generated)
    }

    /// Text embeddings with every `<|image_pad|>` run replaced, in order, by
    /// the merged visual tokens.
    fn prepare_inputs(
        &self,
        input_ids: &[u32],
        inputs: &MiniCpmOcrImageInputs,
    ) -> Result<Tensor, Error> {
        let ids = Tensor::from_vec(input_ids.to_vec(), (1, input_ids.len()), &self.device)
            .map_err(infer)?;
        let text = self.text.embed(&ids)?.squeeze(0).map_err(infer)?;
        let visual = self
            .vision
            .forward(&inputs.slices)?
            .to_dtype(text.dtype())
            .map_err(infer)?;
        let image_token = self.config.image_token_id;
        let placeholders = input_ids.iter().filter(|&&id| id == image_token).count();
        if placeholders != inputs.num_tokens() || placeholders != visual.dim(0).map_err(infer)? {
            return Err(Error::invalid_input(format!(
                "MiniCPM-OCR image placeholders ({placeholders}) != visual tokens ({})",
                inputs.num_tokens()
            )));
        }
        let mut pieces = Vec::new();
        let (mut start, mut consumed) = (0, 0);
        while start < input_ids.len() {
            let is_image = input_ids[start] == image_token;
            let len = input_ids[start..]
                .iter()
                .take_while(|&&id| (id == image_token) == is_image)
                .count();
            pieces.push(
                if is_image {
                    consumed += len;
                    visual.narrow(0, consumed - len, len)
                } else {
                    text.narrow(0, start, len)
                }
                .map_err(infer)?,
            );
            start += len;
        }
        Tensor::cat(&pieces, 0)
            .and_then(|x| x.unsqueeze(0))
            .map_err(infer)
    }

    fn logits(&self, hidden: &Tensor) -> Result<Tensor, Error> {
        hidden
            .unsqueeze(0)
            .and_then(|x| self.lm_head.forward(&x))
            .and_then(|x| x.squeeze(0))
            .map_err(infer)
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }
}

/// Lowest-index argmax, matching the other greedy decoders on ties.
fn select_greedy_token(logits: &Tensor) -> Result<u32, Error> {
    #[cfg(feature = "cuda")]
    if logits.device().is_cuda() && matches!(logits.dtype(), DType::BF16 | DType::F32) {
        let logits = logits
            .reshape((1, logits.elem_count()))
            .and_then(|logits| logits.contiguous())
            .map_err(infer)?;
        let tokens = match logits.dtype() {
            DType::BF16 => logits.apply_op1_no_bwd(&ArgmaxFirstBf16),
            DType::F32 => logits.apply_op1_no_bwd(&ArgmaxFirstF32),
            _ => unreachable!("dtype checked above"),
        }
        .map_err(infer)?;
        return tokens
            .i(0)
            .and_then(|token| token.to_scalar::<u32>())
            .map_err(infer);
    }
    logits
        .flatten_all()
        .and_then(|x| x.argmax(candle_core::D::Minus1))
        .and_then(|token| token.to_scalar::<u32>())
        .map_err(infer)
}
