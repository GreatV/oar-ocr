use super::config::{MinerUConfig, MinerUImageProcessorConfig};
use super::processing::preprocess_images;
use super::text::MinerUTextModel;
use super::vision::MinerUVisionModel;
use crate::attention::{combine_masks, create_causal_mask, create_left_padding_mask};
use crate::utils::{candle_to_ocr_inference, candle_to_ocr_processing};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};
use image::RgbImage;
use oar_ocr_core::core::OCRError;
use rand::distributions::WeightedIndex;
use rand::prelude::*;
use serde::Deserialize;
use std::cmp::Ordering;
use std::collections::HashSet;
use std::path::Path;
use tokenizers::Tokenizer;

pub struct MinerU {
    device: Device,
    dtype: DType,
    cfg: MinerUConfig,
    image_cfg: MinerUImageProcessorConfig,
    tokenizer: Tokenizer,
    text: MinerUTextModel,
    vision: MinerUVisionModel,
    lm_head: Linear,
    image_token_id: u32,
    eos_token_ids: Vec<u32>,
    skip_token_ids: HashSet<u32>,
    vision_start_token_id: u32,
    video_token_id: u32,
    spatial_merge_size: usize,
    repetition_penalty: f32,
    no_repeat_ngram_size: usize,
    do_sample: bool,
    temperature: f32,
    top_p: f32,
    top_k: usize,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum MinerUEosTokenId {
    Single(u32),
    Multi(Vec<u32>),
}

#[derive(Debug, Deserialize)]
struct MinerUGenerationConfig {
    #[serde(default)]
    do_sample: Option<bool>,
    #[serde(default)]
    temperature: Option<f32>,
    #[serde(default)]
    top_p: Option<f32>,
    #[serde(default)]
    top_k: Option<u32>,
    #[serde(default)]
    repetition_penalty: Option<f32>,
    #[serde(default)]
    no_repeat_ngram_size: Option<usize>,
    #[serde(default)]
    eos_token_id: Option<MinerUEosTokenId>,
    #[serde(default)]
    pad_token_id: Option<u32>,
}

impl MinerU {
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, OCRError> {
        let model_dir = model_dir.as_ref();
        let cfg = MinerUConfig::from_path(model_dir.join("config.json"))?;
        let image_cfg =
            MinerUImageProcessorConfig::from_path(model_dir.join("preprocessor_config.json"))?;

        if image_cfg.merge_size != cfg.vision_config.spatial_merge_size {
            return Err(OCRError::ConfigError {
                message: format!(
                    "MinerU2.5 merge_size mismatch: preprocessor {} != vision {}",
                    image_cfg.merge_size, cfg.vision_config.spatial_merge_size
                ),
            });
        }
        if image_cfg.patch_size != cfg.vision_config.patch_size {
            return Err(OCRError::ConfigError {
                message: format!(
                    "MinerU2.5 patch_size mismatch: preprocessor {} != vision {}",
                    image_cfg.patch_size, cfg.vision_config.patch_size
                ),
            });
        }

        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| {
            OCRError::ConfigError {
                message: format!("failed to load MinerU2.5 tokenizer.json: {e}"),
            }
        })?;

        let gen_cfg = load_generation_config(model_dir.join("generation_config.json"));
        let repetition_penalty = gen_cfg
            .as_ref()
            .and_then(|cfg| cfg.repetition_penalty)
            .unwrap_or(1.0);
        let no_repeat_ngram_size = gen_cfg
            .as_ref()
            .and_then(|cfg| cfg.no_repeat_ngram_size)
            .unwrap_or(100);
        let do_sample = gen_cfg
            .as_ref()
            .and_then(|cfg| cfg.do_sample)
            .unwrap_or(false);
        let temperature = gen_cfg
            .as_ref()
            .and_then(|cfg| cfg.temperature)
            .unwrap_or(1.0);
        let top_p = gen_cfg.as_ref().and_then(|cfg| cfg.top_p).unwrap_or(1.0);
        let top_k = gen_cfg
            .as_ref()
            .and_then(|cfg| cfg.top_k)
            .map(|v| v as usize)
            .unwrap_or(0);

        if let Some(tok_image_id) = tokenizer.token_to_id("<|image_pad|>")
            && tok_image_id != cfg.image_token_id
        {
            return Err(OCRError::ConfigError {
                message: format!(
                    "MinerU2.5 image_token_id mismatch: tokenizer {tok_image_id} != config {}",
                    cfg.image_token_id
                ),
            });
        }

        let dtype = device.bf16_default_to_f32();
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_dir.join("model.safetensors")],
                dtype,
                &device,
            )
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "load model.safetensors", e))?
        };

        let text = MinerUTextModel::load(&cfg, vb.pp("model"))?;
        let vision = MinerUVisionModel::load(&cfg.vision_config, vb.pp("visual"))?;

        let image_token_id = cfg.image_token_id;
        let mut eos_token_ids = vec![cfg.eos_token_id];
        if let Some(gen_cfg) = &gen_cfg {
            if let Some(eos) = &gen_cfg.eos_token_id {
                match eos {
                    MinerUEosTokenId::Single(id) => eos_token_ids.push(*id),
                    MinerUEosTokenId::Multi(ids) => eos_token_ids.extend(ids.iter().copied()),
                }
            }
            if let Some(pad) = gen_cfg.pad_token_id {
                eos_token_ids.push(pad);
            }
        }
        eos_token_ids.sort_unstable();
        eos_token_ids.dedup();

        // Build skip_token_ids set (bos, eos, pad) for filtering before decode
        let mut skip_token_ids: HashSet<u32> = HashSet::new();
        skip_token_ids.insert(cfg.bos_token_id);
        skip_token_ids.extend(eos_token_ids.iter().copied());
        if let Some(pad) = cfg.pad_token_id {
            skip_token_ids.insert(pad);
        }

        let vision_start_token_id = cfg.vision_start_token_id;
        let video_token_id = cfg.video_token_id;
        let spatial_merge_size = cfg.vision_config.spatial_merge_size;

        let lm_head = if cfg.tie_word_embeddings {
            Linear::new(text.token_embedding_weight(), None)
        } else {
            linear_no_bias(cfg.hidden_size, cfg.vocab_size, vb.pp("lm_head"))
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "load lm_head", e))?
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
            image_token_id,
            eos_token_ids,
            skip_token_ids,
            vision_start_token_id,
            video_token_id,
            spatial_merge_size,
            repetition_penalty,
            no_repeat_ngram_size,
            do_sample,
            temperature,
            top_p,
            top_k,
        })
    }

    /// Generate text for a batch of images and instructions.
    ///
    /// # Arguments
    /// * `images` - Input images to process
    /// * `instructions` - Text instructions/prompts for each image
    /// * `max_new_tokens` - Maximum number of new tokens to generate
    ///
    /// # Returns
    /// Vector of results, one for each input image-instruction pair
    ///
    /// # Limitations
    /// * Batched inference with different sequence lengths has known issues due to
    ///   variable left-padding causing incorrect attention mask computation during
    ///   incremental generation. For reliable results, process samples one at a time
    ///   when sequences have significantly different lengths.
    /// * The current implementation works correctly when all sequences in the batch
    ///   have similar lengths (minimal padding differences).
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
                    "MinerU2.5: images count ({}) != instructions count ({})",
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
        let batch_size = images.len();

        let image_inputs = preprocess_images(images, &self.image_cfg, &self.device, self.dtype)?;
        let image_token_counts: Vec<usize> = image_inputs
            .image_grid_thw
            .iter()
            .map(|&(t, h, w)| {
                let denom = self.spatial_merge_size * self.spatial_merge_size;
                (t * h * w) / denom
            })
            .collect();

        let mut all_input_ids: Vec<Vec<u32>> = Vec::with_capacity(batch_size);

        for (instruction, &image_token_count) in instructions.iter().zip(image_token_counts.iter())
        {
            let prompt = build_prompt(instruction.as_ref());
            let enc = self
                .tokenizer
                .encode(prompt, false)
                .map_err(|e| OCRError::InvalidInput {
                    message: format!("MinerU2.5: tokenizer encode failed: {e}"),
                })?;

            let input_ids =
                expand_image_tokens(enc.get_ids(), self.image_token_id, &[image_token_count])?;
            all_input_ids.push(input_ids);
        }

        let image_embeds_all = self
            .vision
            .forward(&image_inputs.pixel_values, &image_inputs.image_grid_thw)?;
        let expected_embeds: usize = image_token_counts.iter().sum();
        let actual_embeds = image_embeds_all
            .dim(0)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "image_embeds dim", e))?;
        if actual_embeds != expected_embeds {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "MinerU2.5: image embeds count mismatch: got {actual_embeds}, expected {expected_embeds}"
                ),
            });
        }

        let seq_lens: Vec<usize> = all_input_ids.iter().map(|ids| ids.len()).collect();
        let max_seq_len = *seq_lens.iter().max().unwrap();

        let mut batch_embeds: Vec<Tensor> = Vec::with_capacity(batch_size);
        let mut rope_deltas: Vec<i64> = Vec::with_capacity(batch_size);
        let mut batch_position_ids: Vec<Tensor> = Vec::with_capacity(batch_size);
        let mut embed_offset = 0usize;
        let mut history_tokens: Vec<Vec<u32>> = all_input_ids.clone();

        for (i, input_ids) in all_input_ids.iter().enumerate() {
            let seq_len = input_ids.len();
            let pad_len = max_seq_len - seq_len;
            let image_token_count = image_token_counts[i];

            let image_embeds = image_embeds_all
                .narrow(0, embed_offset, image_token_count)
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "narrow image embeds", e))?;
            embed_offset += image_token_count;

            let input_ids_t = Tensor::new(input_ids.clone(), &self.device)
                .and_then(|t| t.reshape((1, seq_len)))
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "create input_ids", e))?;
            let mut inputs_embeds = self.text.embed(&input_ids_t)?;

            let first_img_pos = input_ids.iter().position(|&id| id == self.image_token_id);
            if let Some(first_pos) = first_img_pos {
                let image_end = first_pos + image_token_count;
                if image_end > seq_len {
                    return Err(OCRError::InvalidInput {
                        message: format!(
                            "MinerU2.5: image token span out of range: {image_end} > {seq_len}"
                        ),
                    });
                }
                let mut parts: Vec<Tensor> = Vec::with_capacity(3);
                if first_pos > 0 {
                    parts.push(
                        inputs_embeds.narrow(1, 0, first_pos).map_err(|e| {
                            candle_to_ocr_inference("MinerU2.5", "narrow prefix", e)
                        })?,
                    );
                }
                parts.push(
                    image_embeds
                        .unsqueeze(0)
                        .map_err(|e| candle_to_ocr_inference("MinerU2.5", "unsqueeze img", e))?,
                );
                if image_end < seq_len {
                    parts.push(
                        inputs_embeds
                            .narrow(1, image_end, seq_len - image_end)
                            .map_err(|e| {
                                candle_to_ocr_inference("MinerU2.5", "narrow suffix", e)
                            })?,
                    );
                }

                let refs: Vec<&Tensor> = parts.iter().collect();
                inputs_embeds = Tensor::cat(&refs, 1)
                    .map_err(|e| candle_to_ocr_inference("MinerU2.5", "cat embeds", e))?;
            }

            if pad_len > 0 {
                let pad = Tensor::zeros(
                    (1, pad_len, self.cfg.hidden_size),
                    inputs_embeds.dtype(),
                    &self.device,
                )
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "create pad", e))?;
                inputs_embeds = Tensor::cat(&[&pad, &inputs_embeds], 1)
                    .map_err(|e| candle_to_ocr_inference("MinerU2.5", "cat pad", e))?;
            }
            batch_embeds.push(inputs_embeds);

            let (pos_ids, delta) = get_rope_index(
                &self.cfg,
                input_ids,
                &[image_inputs.image_grid_thw[i]],
                self.vision_start_token_id,
                self.video_token_id,
                self.spatial_merge_size,
                &self.device,
            )?;
            rope_deltas.push(delta);

            let pos_ids = if pad_len > 0 {
                let pad_pos = Tensor::zeros((3, 1, pad_len), DType::I64, &self.device)
                    .map_err(|e| candle_to_ocr_inference("MinerU2.5", "create pad pos", e))?;
                Tensor::cat(&[&pad_pos, &pos_ids], 2)
                    .map_err(|e| candle_to_ocr_inference("MinerU2.5", "cat pad pos", e))?
            } else {
                pos_ids
            };
            batch_position_ids.push(pos_ids);
        }

        let batch_refs: Vec<&Tensor> = batch_embeds.iter().collect();
        let inputs_embeds = Tensor::cat(&batch_refs, 0)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "stack embeds", e))?;

        let pos_refs: Vec<&Tensor> = batch_position_ids.iter().collect();
        let position_ids = Tensor::cat(&pos_refs, 1)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "stack pos", e))?;

        let causal = create_causal_mask(max_seq_len, max_seq_len, self.dtype, &self.device)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "create causal", e))?;
        let padding = create_left_padding_mask(&seq_lens, max_seq_len, self.dtype, &self.device)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "create padding", e))?;
        let mask = combine_masks(&causal, &padding)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "combine masks", e))?;

        self.text.clear_kv_cache();
        let hidden = self
            .text
            .forward(&inputs_embeds, &position_ids, Some(&mask))?;

        let mut logits_list: Vec<Tensor> = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            let last = hidden
                .i((i, max_seq_len - 1, ..))
                .and_then(|t| t.unsqueeze(0))
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "get last hidden", e))?;
            let logits = self
                .lm_head
                .forward(&last)
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "lm_head", e))?;
            let logits = logits
                .squeeze(0)
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "lm_head squeeze", e))?;
            logits_list.push(logits);
        }

        let mut generated: Vec<Vec<u32>> = vec![Vec::new(); batch_size];
        let mut finished: Vec<bool> = vec![false; batch_size];
        let mut positions: Vec<i64> = seq_lens
            .iter()
            .zip(&rope_deltas)
            .map(|(&len, &d)| (len as i64) + d)
            .collect();

        // Track padding lengths for each sample (for masking during generation)
        // TODO: Batched inference with different padding lengths has issues.
        // Current workaround: process samples one at a time in the example.
        let pad_lens: Vec<usize> = seq_lens.iter().map(|&len| max_seq_len - len).collect();
        // Current KV cache length (grows during generation)
        let mut kv_len = max_seq_len;

        for _ in 0..max_new_tokens {
            if finished.iter().all(|&f| f) {
                break;
            }

            let sampling_params = SamplingParams {
                repetition_penalty: self.repetition_penalty,
                no_repeat_ngram_size: self.no_repeat_ngram_size,
                do_sample: self.do_sample,
                temperature: self.temperature,
                top_p: self.top_p,
                top_k: self.top_k,
            };
            let mut next_tokens: Vec<u32> = Vec::with_capacity(batch_size);
            for (i, logits) in logits_list.iter().enumerate() {
                if finished[i] {
                    next_tokens.push(0);
                } else {
                    let tok = select_next_token(logits, &history_tokens[i], &sampling_params)?;
                    if self.eos_token_ids.contains(&tok) {
                        finished[i] = true;
                    } else {
                        generated[i].push(tok);
                        history_tokens[i].push(tok);
                    }
                    next_tokens.push(tok);
                }
            }

            if finished.iter().all(|&f| f) {
                break;
            }

            let tokens = Tensor::new(next_tokens, &self.device)
                .and_then(|t| t.reshape((batch_size, 1)))
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "create tokens", e))?;
            let embeds = self.text.embed(&tokens)?;

            let pos_data: Vec<i64> = positions.iter().flat_map(|&p| [p, p, p]).collect();
            let pos = Tensor::new(pos_data, &self.device)
                .and_then(|t| t.reshape((3, batch_size, 1)))
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "create pos", e))?;

            // Create attention mask for generation step:
            // - Query position can attend to all non-padding positions in KV cache
            // - Padding positions (first pad_len tokens) should be masked
            kv_len += 1;
            let gen_mask = create_generation_mask(&pad_lens, kv_len, self.dtype, &self.device)
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "create gen mask", e))?;

            let hs = self.text.forward(&embeds, &pos, Some(&gen_mask))?;

            logits_list.clear();
            for i in 0..batch_size {
                let h = hs
                    .i((i, 0, ..))
                    .and_then(|t| t.unsqueeze(0))
                    .map_err(|e| candle_to_ocr_inference("MinerU2.5", "get hs", e))?;
                let logits = self
                    .lm_head
                    .forward(&h)
                    .map_err(|e| candle_to_ocr_inference("MinerU2.5", "lm_head step", e))?;
                let logits = logits
                    .squeeze(0)
                    .map_err(|e| candle_to_ocr_inference("MinerU2.5", "lm_head squeeze", e))?;
                logits_list.push(logits);
            }

            for (i, p) in positions.iter_mut().enumerate() {
                if !finished[i] {
                    *p += 1;
                }
            }
        }

        let mut results = Vec::with_capacity(batch_size);
        for tokens in generated.into_iter() {
            // Filter out bos/eos/pad tokens before decoding (matching official implementation)
            let filtered: Vec<u32> = tokens
                .into_iter()
                .filter(|t| !self.skip_token_ids.contains(t))
                .collect();
            let decoded = self
                .tokenizer
                .decode(&filtered, false) // skip_special_tokens=false to preserve special tokens
                .map_err(|e| OCRError::InvalidInput {
                    message: format!("decode failed: {e}"),
                })?;
            results.push(decoded);
        }

        Ok(results)
    }
}

/// Create attention mask for generation steps.
///
/// During generation, we need to mask out the left-padding positions in the KV cache.
/// The mask allows the current query position to attend to all non-padding positions.
///
/// # Arguments
/// * `pad_lens` - Number of padding tokens at the start of each sequence
/// * `kv_len` - Current KV cache length
/// * `dtype` - Data type for the mask
/// * `device` - Device for the mask
///
/// # Returns
/// Mask tensor of shape (batch, 1, 1, kv_len) where padding positions are -inf
fn create_generation_mask(
    pad_lens: &[usize],
    kv_len: usize,
    dtype: DType,
    device: &Device,
) -> Result<Tensor, candle_core::Error> {
    let batch_size = pad_lens.len();

    // For Metal, create mask on CPU first then transfer to device
    let compute_device = match device {
        Device::Metal(_) => &Device::Cpu,
        _ => device,
    };

    // pad_lens as tensor: (batch, 1, 1, 1)
    let pad_lens_tensor = Tensor::from_vec(
        pad_lens.iter().map(|&x| x as u32).collect::<Vec<_>>(),
        (batch_size, 1, 1, 1),
        compute_device,
    )?
    .to_dtype(dtype)?;

    // Position indices: (1, 1, 1, kv_len)
    let pos_tensor = Tensor::arange(0u32, kv_len as u32, compute_device)?
        .reshape((1, 1, 1, kv_len))?
        .to_dtype(dtype)?;

    // Mask condition: pos < pad_len -> masked (large negative value)
    let mask_cond = pos_tensor.broadcast_lt(&pad_lens_tensor)?;

    let zero = Tensor::new(0f32, compute_device)?
        .to_dtype(dtype)?
        .broadcast_as(mask_cond.shape())?;
    // Use large negative value instead of -inf to avoid potential numerical issues
    let mask_value = Tensor::new(-1e9_f32, compute_device)?
        .to_dtype(dtype)?
        .broadcast_as(mask_cond.shape())?;

    let mask = mask_cond.where_cond(&mask_value, &zero)?;

    // Transfer to target device if needed
    if matches!(device, Device::Metal(_)) {
        mask.to_device(device)
    } else {
        Ok(mask)
    }
}

fn build_prompt(instruction: &str) -> String {
    let separator = if instruction.starts_with(' ') || instruction.starts_with('\n') {
        ""
    } else {
        " "
    };
    format!(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n<|vision_start|><|image_pad|><|vision_end|>{separator}{instruction}<|im_end|>\n<|im_start|>assistant\n"
    )
}

fn load_generation_config(path: impl AsRef<Path>) -> Option<MinerUGenerationConfig> {
    let contents = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&contents).ok()
}

struct SamplingParams {
    repetition_penalty: f32,
    no_repeat_ngram_size: usize,
    do_sample: bool,
    temperature: f32,
    top_p: f32,
    top_k: usize,
}

fn select_next_token(
    logits: &Tensor,
    history: &[u32],
    params: &SamplingParams,
) -> Result<u32, OCRError> {
    let logits = logits
        .to_dtype(DType::F32)
        .map_err(|e| candle_to_ocr_inference("MinerU2.5", "logits cast", e))?
        .to_device(&Device::Cpu)
        .map_err(|e| candle_to_ocr_inference("MinerU2.5", "logits to cpu", e))?;
    let mut logits_vec = logits
        .to_vec1::<f32>()
        .map_err(|e| candle_to_ocr_inference("MinerU2.5", "logits to vec", e))?;

    apply_repetition_penalty(&mut logits_vec, history, params.repetition_penalty);
    apply_no_repeat_ngram(&mut logits_vec, history, params.no_repeat_ngram_size);

    if !params.do_sample || params.top_k == 1 {
        return Ok(argmax_token(&logits_vec));
    }

    let temp = if params.temperature <= 0.0 {
        1.0
    } else {
        params.temperature
    };
    if (temp - 1.0).abs() > f32::EPSILON {
        for val in logits_vec.iter_mut() {
            *val /= temp;
        }
    }

    apply_top_k(&mut logits_vec, params.top_k);
    apply_top_p(&mut logits_vec, params.top_p);

    let probs = softmax(&logits_vec);
    if let Some(idx) = sample_from_probs(&probs) {
        Ok(idx as u32)
    } else {
        Ok(argmax_token(&logits_vec))
    }
}

fn argmax_token(logits: &[f32]) -> u32 {
    let mut best_idx = 0usize;
    let mut best_val = f32::NEG_INFINITY;
    for (idx, &val) in logits.iter().enumerate() {
        if val.is_nan() {
            continue;
        }
        if val > best_val {
            best_val = val;
            best_idx = idx;
        }
    }
    best_idx as u32
}

fn apply_repetition_penalty(logits: &mut [f32], history: &[u32], penalty: f32) {
    if penalty <= 1.0 {
        return;
    }
    let mut seen = HashSet::new();
    for &token in history {
        if !seen.insert(token) {
            continue;
        }
        let idx = token as usize;
        if idx >= logits.len() {
            continue;
        }
        let val = logits[idx];
        logits[idx] = if val < 0.0 {
            val * penalty
        } else {
            val / penalty
        };
    }
}

fn apply_top_k(logits: &mut [f32], top_k: usize) {
    if top_k == 0 || top_k >= logits.len() {
        return;
    }
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(Ordering::Less));
    for &idx in indices.iter().skip(top_k) {
        logits[idx] = f32::NEG_INFINITY;
    }
}

fn apply_top_p(logits: &mut [f32], top_p: f32) {
    if !(0.0..1.0).contains(&top_p) {
        return;
    }
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.sort_by(|&a, &b| logits[b].partial_cmp(&logits[a]).unwrap_or(Ordering::Less));

    let max = logits[indices[0]];
    let mut exp_vals: Vec<f32> = Vec::with_capacity(indices.len());
    let mut exp_sum = 0.0f32;
    for &idx in &indices {
        let val = logits[idx];
        let exp = if val.is_finite() {
            (val - max).exp()
        } else {
            0.0
        };
        exp_vals.push(exp);
        exp_sum += exp;
    }
    if exp_sum == 0.0 {
        return;
    }

    let mut cumulative = 0.0f32;
    for (rank, _) in indices.iter().enumerate() {
        let prob = exp_vals[rank] / exp_sum;
        cumulative += prob;
        if cumulative > top_p && rank > 0 {
            for &drop in indices.iter().skip(rank) {
                logits[drop] = f32::NEG_INFINITY;
            }
            break;
        }
    }
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let mut max = f32::NEG_INFINITY;
    for &val in logits {
        if val.is_finite() && val > max {
            max = val;
        }
    }
    let mut exps = Vec::with_capacity(logits.len());
    let mut sum = 0.0f32;
    for &val in logits {
        let exp = if val.is_finite() {
            (val - max).exp()
        } else {
            0.0
        };
        exps.push(exp);
        sum += exp;
    }
    if sum == 0.0 {
        return vec![0.0; logits.len()];
    }
    exps.into_iter().map(|v| v / sum).collect()
}

fn sample_from_probs(probs: &[f32]) -> Option<usize> {
    let dist = WeightedIndex::new(probs).ok()?;
    let mut rng = thread_rng();
    Some(dist.sample(&mut rng))
}

fn apply_no_repeat_ngram(logits: &mut [f32], history: &[u32], ngram_size: usize) {
    if ngram_size <= 1 || history.len() < ngram_size {
        return;
    }
    let prefix_len = ngram_size - 1;
    let prefix_start = history.len() - prefix_len;
    let prefix = &history[prefix_start..];
    let mut banned = HashSet::new();
    for i in 0..=history.len() - ngram_size {
        if history[i..i + prefix_len] == *prefix {
            banned.insert(history[i + prefix_len]);
        }
    }
    for token in banned {
        let idx = token as usize;
        if idx < logits.len() {
            logits[idx] = f32::NEG_INFINITY;
        }
    }
}

fn expand_image_tokens(
    input_ids: &[u32],
    image_token_id: u32,
    image_token_counts: &[usize],
) -> Result<Vec<u32>, OCRError> {
    let mut out: Vec<u32> = Vec::new();
    let mut idx = 0usize;
    for &id in input_ids {
        if id == image_token_id {
            let count = image_token_counts
                .get(idx)
                .ok_or_else(|| OCRError::InvalidInput {
                    message: "MinerU2.5: image token count mismatch".to_string(),
                })?;
            out.extend(std::iter::repeat_n(image_token_id, *count));
            idx += 1;
        } else {
            out.push(id);
        }
    }
    if idx != image_token_counts.len() {
        return Err(OCRError::InvalidInput {
            message: "MinerU2.5: image token count mismatch".to_string(),
        });
    }
    Ok(out)
}

fn get_rope_index(
    cfg: &MinerUConfig,
    input_ids: &[u32],
    image_grid_thw: &[(usize, usize, usize)],
    vision_start_token_id: u32,
    video_token_id: u32,
    spatial_merge_size: usize,
    device: &Device,
) -> Result<(Tensor, i64), OCRError> {
    let image_token_id = cfg.image_token_id;
    let mut image_count = 0usize;
    for i in 0..input_ids.len().saturating_sub(1) {
        if input_ids[i] == vision_start_token_id && input_ids[i + 1] == image_token_id {
            image_count += 1;
        }
        if input_ids[i] == vision_start_token_id && input_ids[i + 1] == video_token_id {
            return Err(OCRError::InvalidInput {
                message: "MinerU2.5: video inputs are not supported".to_string(),
            });
        }
    }
    if image_count != image_grid_thw.len() {
        return Err(OCRError::InvalidInput {
            message: format!(
                "MinerU2.5: image count mismatch between prompt ({image_count}) and image_grid_thw ({})",
                image_grid_thw.len()
            ),
        });
    }

    let mut positions: Vec<[i64; 3]> = Vec::with_capacity(input_ids.len());
    let mut st = 0usize;
    let mut current_max: i64 = -1;

    for (image_index, &(t, h, w)) in image_grid_thw.iter().enumerate().take(image_count) {
        let ed = input_ids[st..]
            .iter()
            .position(|&id| id == image_token_id)
            .map(|p| st + p)
            .ok_or_else(|| OCRError::InvalidInput {
                message: format!(
                    "MinerU2.5: expected image token for image[{image_index}] but none found"
                ),
            })?;

        let st_idx = if current_max >= 0 { current_max + 1 } else { 0 };
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

        for tt in 0..llm_t {
            for hh in 0..llm_h {
                for ww in 0..llm_w {
                    let t_pos = vision_base + tt;
                    let h_pos = vision_base + hh;
                    let w_pos = vision_base + ww;
                    positions.push([t_pos, h_pos, w_pos]);
                    current_max = current_max.max(t_pos).max(h_pos).max(w_pos);
                }
            }
        }

        st = ed + (llm_t as usize) * (llm_h as usize) * (llm_w as usize);
    }

    let st_idx = if current_max >= 0 { current_max + 1 } else { 0 };
    for i in st..input_ids.len() {
        let p = st_idx + (i - st) as i64;
        positions.push([p, p, p]);
        current_max = p;
    }

    if positions.len() != input_ids.len() {
        return Err(OCRError::InvalidInput {
            message: format!(
                "MinerU2.5: rope position ids length mismatch: got {}, expected {}",
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

    let rope_delta = (current_max + 1) - (input_ids.len() as i64);

    let position_ids = Tensor::from_vec(pos_ids, (3usize, 1usize, input_ids.len()), device)
        .map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "MinerU2.5: build position_ids tensor failed",
                e,
            )
        })?;

    Ok((position_ids, rope_delta))
}
