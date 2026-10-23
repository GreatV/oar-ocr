use super::config::{Config, GenerationConfig, ImageConfig};
use super::infer;
use super::processing::{self, IMAGE_BEGIN, MERGE_SIZE, OxrImageInputs};
use super::vision::ImageEncoder;
use crate::backbones::sdar::{SdarKvCache, SdarModel};
use crate::runtime::checkpoint::{collect_safetensors, load_json_config};
#[cfg(feature = "cuda")]
use crate::runtime::cuda::{ArgmaxFirstBf16, ArgmaxFirstF32};
use crate::{BatchResult, Error, RuntimeConfig};
use candle_core::{DType, Device, IndexOp, Tensor};
use image::RgbImage;
use std::path::Path;
use tokenizers::Tokenizer;

/// Official image-recognition prompts from OXR's `oxr/model/prompts.py`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[non_exhaustive]
pub enum OxrTask {
    Text,
    Formula,
    Table,
    Code,
    ChemicalStructure,
    WholePage,
}

impl OxrTask {
    pub const fn prompt(self) -> &'static str {
        match self {
            Self::Text => "Recognize the content of the image.",
            Self::Formula => "Recognize the formula of the image into KaTex format.",
            Self::Table => "Recognize the table of the image into html format.",
            Self::Code => "Recognize the code of the image into markdown code format.",
            Self::ChemicalStructure => "Recognize the chemical structure of the image into SMILES.",
            Self::WholePage => "Recognize the whole page content of the image.",
        }
    }
}

/// Greedy generation controls, defaulting to the upstream vLLM processor profile.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct OxrGenerationOptions {
    pub max_new_tokens: usize,
    pub min_pixels: u32,
    pub max_pixels: u32,
    /// Applies only to generated tokens, matching reference generation from
    /// input embeddings. Prompt IDs are not penalized. One disables the penalty;
    /// generation never changes it automatically.
    pub repetition_penalty: f32,
}

impl Default for OxrGenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: 8192,
            min_pixels: processing::MIN_PIXELS,
            max_pixels: processing::MAX_PIXELS,
            repetition_penalty: 1.0,
        }
    }
}

impl OxrGenerationOptions {
    pub fn with_max_new_tokens(mut self, value: usize) -> Self {
        self.max_new_tokens = value;
        self
    }
    pub fn with_min_pixels(mut self, value: u32) -> Self {
        self.min_pixels = value;
        self
    }
    pub fn with_max_pixels(mut self, value: u32) -> Self {
        self.max_pixels = value;
        self
    }
    pub fn with_repetition_penalty(mut self, value: f32) -> Self {
        self.repetition_penalty = value;
        self
    }

    pub(super) fn validate(&self) -> Result<(), Error> {
        if self.min_pixels == 0
            || self.max_pixels < self.min_pixels
            || !self.repetition_penalty.is_finite()
            || self.repetition_penalty <= 0.0
        {
            return Err(Error::invalid_input(
                "OXR needs ordered positive pixel bounds and a finite positive repetition penalty",
            ));
        }
        Ok(())
    }
}

/// Raw greedy tokens and unmodified per-step top-3 logits for reference alignment.
#[derive(Debug, Clone)]
pub struct GenerationTrace {
    pub tokens: Vec<u32>,
    pub step_top: Vec<[(u32, f32); 3]>,
    /// Expanded prompt IDs; negative IDs designate visual embeddings.
    pub prompt_ids: Vec<i64>,
}

/// Native OXR-1.0 visual tokenizer and tied Qwen3 text decoder.
///
/// Weights use SoMark's OXR License; see the module documentation for its terms.
/// This implementation does not execute checkpoint Python code.
pub struct Oxr {
    config: Config,
    image_config: ImageConfig,
    tokenizer: Tokenizer,
    vision: ImageEncoder,
    text: SdarModel,
    stops: Vec<u32>,
    device: Device,
    dtype: DType,
    pub(super) generation_options: OxrGenerationOptions,
}

impl Oxr {
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, Error> {
        Self::from_dir_with_runtime(model_dir, RuntimeConfig::new(device))
    }

    pub fn from_dir_with_runtime(
        model_dir: impl AsRef<Path>,
        runtime: RuntimeConfig,
    ) -> Result<Self, Error> {
        let dir = model_dir.as_ref();
        let config: Config = load_json_config(dir.join("config.json"), "OXR", "config.json")?;
        config.validate()?;
        let image_config: ImageConfig = load_json_config(
            dir.join("preprocessor_config.json"),
            "OXR",
            "preprocessor_config.json",
        )?;
        image_config.validate()?;
        let generation: GenerationConfig = load_json_config(
            dir.join("generation_config.json"),
            "OXR",
            "generation_config.json",
        )?;
        if generation.eos_token_id.is_empty()
            || generation
                .eos_token_id
                .iter()
                .any(|&id| id as usize >= config.llm_config.decoder.vocab_size)
        {
            return Err(Error::config(
                "OXR generation_config.json has invalid stop IDs",
            ));
        }
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| Error::config(format!("OXR tokenizer.json: {e}")))?;
        for token in ["<|im_start|>", "<|im_end|>", "<|endoftext|>"] {
            if tokenizer.token_to_id(token).is_none() {
                return Err(Error::config(format!("OXR tokenizer is missing {token}")));
            }
        }
        let (device, dtype) = runtime.resolve();
        let files = collect_safetensors(dir, "OXR")?;
        // SAFETY: checkpoint files must remain unchanged while the model is alive.
        let vb = unsafe { candle_nn::VarBuilder::from_mmaped_safetensors(&files, dtype, &device) }
            .map_err(infer)?;
        let vision = ImageEncoder::load(&config, vb.clone())?;
        let text = SdarModel::load(&config.llm_config.decoder, vb.pp("llm"), dtype)?;
        Ok(Self {
            config,
            image_config,
            tokenizer,
            vision,
            text,
            stops: generation.eos_token_id,
            device,
            dtype,
            generation_options: OxrGenerationOptions::default(),
        })
    }

    /// Set defaults for [`generate`](Self::generate) and region recognition.
    ///
    /// Their per-request token limit overrides `options.max_new_tokens`.
    /// Methods that take explicit generation options use those instead.
    pub fn with_generation_options(mut self, options: OxrGenerationOptions) -> Self {
        self.generation_options = options;
        self
    }

    /// Generate raw text for each image/task pair, in order.
    pub fn generate(
        &self,
        images: &[RgbImage],
        tasks: &[OxrTask],
        max_new_tokens: usize,
    ) -> BatchResult<String> {
        self.generate_with_options(
            images,
            tasks,
            &self
                .generation_options
                .clone()
                .with_max_new_tokens(max_new_tokens),
        )
    }

    pub fn generate_with_options(
        &self,
        images: &[RgbImage],
        tasks: &[OxrTask],
        options: &OxrGenerationOptions,
    ) -> BatchResult<String> {
        Ok(self
            .generate_tokens(images, tasks, options)?
            .into_iter()
            .map(|result| result.and_then(|ids| self.decode_tokens(&ids)))
            .collect())
    }

    pub fn generate_tokens(
        &self,
        images: &[RgbImage],
        tasks: &[OxrTask],
        options: &OxrGenerationOptions,
    ) -> BatchResult<Vec<u32>> {
        if images.len() != tasks.len() {
            return Err(Error::invalid_input("OXR image and task counts differ"));
        }
        options.validate()?;
        Ok(images
            .iter()
            .zip(tasks)
            .map(|(image, task)| {
                self.generate_one(image, task.prompt(), options, false)
                    .map(|trace| trace.tokens)
            })
            .collect())
    }

    /// Use caller-provided instructions with the same image-first chat template.
    pub fn generate_with_prompts(
        &self,
        images: &[RgbImage],
        prompts: &[impl AsRef<str>],
        options: &OxrGenerationOptions,
    ) -> BatchResult<String> {
        if images.len() != prompts.len() {
            return Err(Error::invalid_input("OXR image and prompt counts differ"));
        }
        options.validate()?;
        Ok(images
            .iter()
            .zip(prompts)
            .map(|(image, prompt)| {
                self.generate_one(image, prompt.as_ref(), options, false)
                    .and_then(|trace| self.decode_tokens(&trace.tokens))
            })
            .collect())
    }

    pub fn generate_traced(
        &self,
        image: &RgbImage,
        task: OxrTask,
        options: &OxrGenerationOptions,
    ) -> Result<GenerationTrace, Error> {
        self.generate_one(image, task.prompt(), options, true)
    }

    /// Prepare image patches without running the vision encoder.
    pub fn preprocess_image(
        &self,
        image: &RgbImage,
        options: &OxrGenerationOptions,
    ) -> Result<OxrImageInputs, Error> {
        options.validate()?;
        processing::preprocess(image, &self.image_config, options, &self.device, self.dtype)
    }

    /// Decode generated tokens without task-specific cleanup.
    pub fn decode_tokens(&self, ids: &[u32]) -> Result<String, Error> {
        self.tokenizer
            .decode(ids, true)
            .map_err(|e| Error::invalid_input(format!("OXR decoding failed: {e}")))
    }

    /// Assemble text spans and image embeddings directly; visual sentinels
    /// are metadata for tracing and never enter the text embedding lookup.
    fn encode_prompt(
        &self,
        image: &RgbImage,
        prompt: &str,
        options: &OxrGenerationOptions,
    ) -> Result<(Tensor, Vec<i64>), Error> {
        let inputs = self.preprocess_image(image, options)?;
        let atoms = inputs.grid_thw[1] * inputs.grid_thw[2] / (MERGE_SIZE * MERGE_SIZE);
        let ids = processing::tokenize_prompt(&self.tokenizer, prompt, atoms)?;
        let requested = ids.len().checked_add(options.max_new_tokens);
        if requested.is_none_or(|len| len > self.config.llm_config.decoder.max_position_embeddings)
        {
            return Err(Error::invalid_input(
                "OXR prompt plus output budget exceeds the text context limit",
            ));
        }
        let text_ids: Vec<u32> = ids
            .iter()
            .filter(|&&id| id >= 0)
            .map(|&id| id as u32)
            .collect();
        if text_ids
            .iter()
            .any(|&id| id as usize >= self.config.llm_config.decoder.vocab_size)
        {
            return Err(Error::invalid_input(
                "OXR prompt contains an out-of-vocabulary token",
            ));
        }
        let token_ids = Tensor::new(text_ids.as_slice(), &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(infer)?;
        let text_embeds = self.text.embed(&token_ids)?;
        let visual = self.vision.encode(&inputs.pixel_values, inputs.grid_thw)?;
        let prefix_len = ids
            .iter()
            .position(|&id| id == IMAGE_BEGIN)
            .expect("expanded image prompt");
        let boundary = |index| {
            visual
                .boundaries
                .narrow(0, index, 1)
                .and_then(|x| x.unsqueeze(0))
                .map_err(infer)
        };
        let embeds = Tensor::cat(
            &[
                text_embeds.narrow(1, 0, prefix_len).map_err(infer)?,
                boundary(0)?,
                visual.tokens.unsqueeze(0).map_err(infer)?,
                boundary(1)?,
                text_embeds
                    .narrow(1, prefix_len, text_ids.len() - prefix_len)
                    .map_err(infer)?,
            ],
            1,
        )
        .map_err(infer)?;
        Ok((embeds, ids))
    }

    pub(super) fn generate_one(
        &self,
        image: &RgbImage,
        prompt: &str,
        options: &OxrGenerationOptions,
        traced: bool,
    ) -> Result<GenerationTrace, Error> {
        options.validate()?;
        let mut trace = GenerationTrace {
            tokens: Vec::new(),
            step_top: Vec::new(),
            prompt_ids: Vec::new(),
        };
        if options.max_new_tokens == 0 {
            return Ok(trace);
        }
        let (embeds, ids) = self.encode_prompt(image, prompt, options)?;
        let mut seen = vec![false; self.config.llm_config.decoder.vocab_size];
        let prompt_len = ids.len();
        trace.prompt_ids = ids;
        let positions: Vec<i64> = (0..prompt_len as i64).collect();
        let mut cache = SdarKvCache::new(self.text.num_layers());
        let hidden = self
            .text
            .forward_causal(&embeds, &positions, &mut cache, true)?;
        let hidden = hidden.narrow(1, prompt_len - 1, 1).map_err(infer)?;
        let mut logits = self.text.lm_logits(&hidden)?.i((0, 0, ..)).map_err(infer)?;
        for step in 0..options.max_new_tokens {
            let token = if !traced && options.repetition_penalty == 1.0 {
                select_greedy_token(&logits)?
            } else {
                let mut scores = logits
                    .to_dtype(DType::F32)
                    .and_then(|t| t.to_vec1::<f32>())
                    .map_err(infer)?;
                if traced {
                    trace.step_top.push(top3(&scores));
                }
                if options.repetition_penalty != 1.0 {
                    for (score, &used) in scores.iter_mut().zip(&seen) {
                        if used {
                            *score = if *score < 0.0 {
                                *score * options.repetition_penalty
                            } else {
                                *score / options.repetition_penalty
                            };
                        }
                    }
                }
                scores
                    .iter()
                    .enumerate()
                    .fold((0, f32::NEG_INFINITY), |best, (id, &value)| {
                        if value > best.1 { (id, value) } else { best }
                    })
                    .0 as u32
            };
            if self.stops.contains(&token) {
                break;
            }
            trace.tokens.push(token);
            seen[token as usize] = true;
            if step + 1 == options.max_new_tokens {
                break;
            }
            let id = Tensor::new(&[[token]], &self.device).map_err(infer)?;
            let embedded = self.text.embed(&id)?;
            let hidden = self.text.forward_causal(
                &embedded,
                &[(prompt_len + step) as i64],
                &mut cache,
                true,
            )?;
            logits = self.text.lm_logits(&hidden)?.i((0, 0, ..)).map_err(infer)?;
        }
        Ok(trace)
    }
}

/// Match the stable lowest-index argmax used by the other greedy decoders.
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
        .argmax(candle_core::D::Minus1)
        .and_then(|token| token.to_scalar::<u32>())
        .map_err(infer)
}

fn top3(scores: &[f32]) -> [(u32, f32); 3] {
    let mut best = [(0, f32::NEG_INFINITY); 3];
    for (id, &value) in scores.iter().enumerate() {
        if let Some(rank) = best.iter().position(|&(_, v)| value > v) {
            for index in (rank + 1..3).rev() {
                best[index] = best[index - 1];
            }
            best[rank] = (id as u32, value);
        }
    }
    best
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn greedy_argmax_and_trace_prefer_the_first_tied_token() {
        let values = [1f32, 3.0, 3.0, 2.0];
        let logits = Tensor::new(&values, &Device::Cpu).unwrap();
        assert_eq!(select_greedy_token(&logits).unwrap(), 1);
        assert_eq!(top3(&values).map(|(id, _)| id), [1, 2, 3]);
    }
}
