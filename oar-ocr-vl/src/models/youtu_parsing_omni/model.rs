use super::config::Config;
use super::processing::{ImageConfig, ImagePlan, plan, preprocess, prompt_tokens};
use super::text::TextModel;
use super::vision::VisionModel;
use super::{DEFAULT_PROMPT, YoutuParsingOmniParseOptions};
use crate::runtime::checkpoint::{collect_safetensors, load_optional_json_config};
use crate::runtime::errors::candle_to_ocr_inference;
use crate::{BatchResult, Error, RuntimeConfig};
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};
use image::RgbImage;
use std::path::Path;
use tokenizers::Tokenizer;

const MIN_NEW_TOKENS: usize = 16;

fn greedy_token(logits: &Tensor) -> candle_core::Result<u32> {
    #[cfg(feature = "cuda")]
    if logits.device().is_cuda() {
        use crate::runtime::cuda::{ArgmaxFirstBf16, ArgmaxFirstF32};
        let picked = if logits.dtype() == DType::BF16 {
            logits.contiguous()?.apply_op1_no_bwd(&ArgmaxFirstBf16)?
        } else {
            logits
                .to_dtype(DType::F32)?
                .contiguous()?
                .apply_op1_no_bwd(&ArgmaxFirstF32)?
        };
        return picked.squeeze(0)?.to_scalar::<u32>();
    }
    logits.squeeze(0)?.argmax(0)?.to_scalar::<u32>()
}

fn sample_token(logits: &Tensor, stop_mask: &Tensor, generated: usize) -> candle_core::Result<u32> {
    if generated < MIN_NEW_TOKENS {
        // Match the official request's min_tokens: suppress stops before sampling.
        greedy_token(&(logits + stop_mask)?)
    } else {
        greedy_token(logits)
    }
}

/// Image-to-OmniSchema parser for `tencent/Youtu-Parsing-Omni`.
pub struct YoutuParsingOmni {
    cfg: Config,
    image_cfg: ImageConfig,
    tokenizer: Tokenizer,
    text: TextModel,
    vision: VisionModel,
    head: Linear,
    stop_ids: Vec<u32>,
    stop_mask: Tensor,
    device: Device,
    dtype: DType,
}

struct CacheGuard<'a>(&'a TextModel);

impl Drop for CacheGuard<'_> {
    fn drop(&mut self) {
        self.0.clear_cache();
    }
}

impl YoutuParsingOmni {
    /// Load the checkpoint with an automatically selected compute dtype.
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, Error> {
        Self::from_dir_with_runtime(model_dir, RuntimeConfig::new(device))
    }

    /// Load the checkpoint with an explicit device and dtype policy.
    pub fn from_dir_with_runtime(
        model_dir: impl AsRef<Path>,
        runtime: RuntimeConfig,
    ) -> Result<Self, Error> {
        let (device, dtype) = runtime.resolve();
        let dir = model_dir.as_ref();
        let cfg = Config::load(&dir.join("config.json"))?;
        let image_cfg = ImageConfig::load(&dir.join("processor_config.json"), &cfg.omni_config)?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json")).map_err(|e| {
            Error::config(format!("failed to load Youtu-Parsing-Omni tokenizer: {e}"))
        })?;
        for token in [
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "<|end_of_text|>",
        ] {
            if tokenizer.token_to_id(token).is_none() {
                return Err(Error::config(format!(
                    "Youtu-Parsing-Omni tokenizer lacks {token}"
                )));
            }
        }
        let files = collect_safetensors(dir, "Youtu-Parsing-Omni")?;
        // SAFETY: Checkpoint files must remain unchanged while the model is loaded.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device) }
            .map_err(|e| candle_to_ocr_inference("Youtu-Parsing-Omni", "load weights", e))?;
        let text = TextModel::load(&cfg.text_config, vb.pp("model.language_model"))?;
        let vision = VisionModel::load(&cfg.omni_config, vb.pp("model.omni_model"))?;
        let head = if cfg.tie_word_embeddings {
            Linear::new(text.embedding.embeddings().clone(), None)
        } else {
            linear_no_bias(
                cfg.text_config.hidden_size,
                cfg.text_config.vocab_size,
                vb.pp("lm_head"),
            )?
        };
        let generation: Option<serde_json::Value> = load_optional_json_config(
            dir.join("generation_config.json"),
            "Youtu-Parsing-Omni",
            "generation_config.json",
        )?;
        let mut stop_ids = vec![
            tokenizer
                .token_to_id("<|image_pad|>")
                .expect("validated token"),
            tokenizer
                .token_to_id("<|end_of_text|>")
                .expect("validated token"),
        ];
        if let Some(eos) = generation.as_ref().and_then(|g| g.get("eos_token_id")) {
            let ids = if let Some(ids) = eos.as_array() {
                ids.clone()
            } else {
                vec![eos.clone()]
            };
            for id in ids {
                let id = id
                    .as_u64()
                    .filter(|id| *id < cfg.text_config.vocab_size as u64)
                    .ok_or_else(|| Error::config("invalid Youtu-Parsing-Omni EOS token ID"))?;
                stop_ids.push(id as u32);
            }
        }
        stop_ids.sort_unstable();
        stop_ids.dedup();
        let stop_mask = Tensor::from_vec(
            (0..cfg.text_config.vocab_size)
                .map(|id| {
                    if stop_ids.contains(&(id as u32)) {
                        f32::NEG_INFINITY
                    } else {
                        0.0
                    }
                })
                .collect::<Vec<_>>(),
            (1, cfg.text_config.vocab_size),
            &device,
        )?
        .to_dtype(dtype)?;
        Ok(Self {
            cfg,
            image_cfg,
            tokenizer,
            text,
            vision,
            head,
            stop_ids,
            stop_mask,
            device,
            dtype,
        })
    }

    /// Generate native JSON for each image, retaining per-image failures.
    pub fn generate(
        &self,
        images: &[RgbImage],
        options: &YoutuParsingOmniParseOptions,
    ) -> BatchResult<String> {
        Ok(images
            .iter()
            .map(|image| {
                self.generate_one(image, options)
                    .and_then(|(tokens, _)| self.decode_tokens(&tokens))
            })
            .collect())
    }

    /// Generate native token IDs for each image.
    pub fn generate_tokens(
        &self,
        images: &[RgbImage],
        options: &YoutuParsingOmniParseOptions,
    ) -> BatchResult<Vec<u32>> {
        Ok(images
            .iter()
            .map(|image| self.generate_one(image, options).map(|(tokens, _)| tokens))
            .collect())
    }

    /// Decode tokens without stripping box coordinates or table markers.
    pub fn decode_tokens(&self, tokens: &[u32]) -> Result<String, Error> {
        self.tokenizer
            .decode(tokens, false)
            .map_err(|e| Error::config(format!("Youtu-Parsing-Omni decoding failed: {e}")))
    }

    /// The checkpoint tokenizer, including OmniSchema structural tokens.
    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    pub(super) fn image_plan(
        &self,
        image: &RgbImage,
        options: &YoutuParsingOmniParseOptions,
    ) -> Result<ImagePlan, Error> {
        plan(
            image.width(),
            image.height(),
            &self.image_cfg,
            options.max_pixels,
        )
    }

    pub(super) fn generate_one(
        &self,
        image: &RgbImage,
        options: &YoutuParsingOmniParseOptions,
    ) -> Result<(Vec<u32>, bool), Error> {
        self.text.clear_cache();
        let _cache = CacheGuard(&self.text);
        let plan = self.image_plan(image, options)?;
        let (ids, indices) = prompt_tokens(
            &self.tokenizer,
            plan,
            self.image_cfg.patch_size * self.image_cfg.spatial_merge_size,
            options.prompt.as_deref().unwrap_or(DEFAULT_PROMPT),
        )?;
        let limit = ids
            .len()
            .checked_add(options.max_new_tokens)
            .ok_or_else(|| Error::invalid_input("Youtu-Parsing-Omni generation length overflow"))?;
        if limit > self.cfg.text_config.max_position_embeddings {
            return Err(Error::invalid_input(
                "Youtu-Parsing-Omni prompt and generation budget exceed the context length",
            ));
        }
        if options.max_new_tokens == 0 {
            return Ok((Vec::new(), false));
        }
        let pixels = preprocess(image, &self.image_cfg, plan, self.dtype, &self.device)?;
        let visual = self.vision.forward(
            &pixels,
            plan.height as usize / self.image_cfg.patch_size,
            plan.width as usize / self.image_cfg.patch_size,
        )?;
        let input = Tensor::from_vec(ids.clone(), (1, ids.len()), &self.device)?;
        let embeddings = self.text.embedding.forward(&input)?;
        let mut segments = Vec::new();
        let mut previous = 0;
        let cols =
            plan.width as usize / (self.image_cfg.patch_size * self.image_cfg.spatial_merge_size);
        for (row, positions) in indices.chunks(cols).enumerate() {
            let start = positions[0];
            if start > previous {
                segments.push(embeddings.narrow(1, previous, start - previous)?);
            }
            segments.push(visual.narrow(0, row * cols, cols)?.unsqueeze(0)?);
            previous = start + cols;
        }
        if previous < ids.len() {
            segments.push(embeddings.narrow(1, previous, ids.len() - previous)?);
        }
        let mut embeddings = Tensor::cat(&segments, 1)?;
        let mut offset = 0;
        let mut tokens = Vec::new();
        for _ in 0..options.max_new_tokens {
            let len = embeddings.dim(1)?;
            let hidden = self.text.forward(&embeddings, offset)?;
            let logits = self.head.forward(&hidden.i((.., len - 1, ..))?)?;
            let token = sample_token(&logits, &self.stop_mask, tokens.len())?;
            if self.stop_ids.contains(&token) {
                return Ok((tokens, true));
            }
            tokens.push(token);
            offset += len;
            embeddings = self
                .text
                .embedding
                .forward(&Tensor::new(&[[token]], &self.device)?)?;
        }
        Ok((tokens, false))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tokenizers::models::wordlevel::WordLevel;

    #[test]
    fn loading_without_generation_config_keeps_stops_and_minimum_length() {
        let dir = tempfile::tempdir().unwrap();
        let reference: serde_json::Value =
            serde_json::from_str(include_str!("fixtures/reference.json")).unwrap();
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_vec(&reference["config"]).unwrap(),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("model.safetensors"),
            include_bytes!("fixtures/tiny.safetensors"),
        )
        .unwrap();
        std::fs::write(
            dir.path().join("processor_config.json"),
            r#"{"image_processor":{"patch_size":2,"spatial_merge_size":2,"temporal_patch_size":1,"image_min_num_tokens":1,"image_max_num_tokens":32,"mean":[0.5,0.5,0.5],"std":[0.5,0.5,0.5]}}"#,
        )
        .unwrap();
        let vocab = [
            "<unk>",
            "<|vision_start|>",
            "<|vision_end|>",
            "<|image_pad|>",
            "<|end_of_text|>",
        ]
        .into_iter()
        .enumerate()
        .map(|(id, token)| (token.to_string(), id as u32))
        .collect();
        let tokenizer = Tokenizer::new(
            WordLevel::builder()
                .vocab(vocab)
                .unk_token("<unk>".into())
                .build()
                .unwrap(),
        );
        tokenizer
            .save(dir.path().join("tokenizer.json"), false)
            .unwrap();

        let model = YoutuParsingOmni::from_dir(dir.path(), Device::Cpu).unwrap();
        for token in ["<|image_pad|>", "<|end_of_text|>"] {
            assert!(
                model
                    .stop_ids
                    .contains(&tokenizer.token_to_id(token).unwrap()),
                "missing protocol stop token {token}"
            );
        }
        let mut logits = vec![0.0f32; 32];
        let regular = tokenizer.token_to_id("<|vision_end|>").unwrap();
        let stop = tokenizer.token_to_id("<|image_pad|>").unwrap();
        logits[regular as usize] = 2.0;
        logits[stop as usize] = 10.0;
        logits[tokenizer.token_to_id("<|end_of_text|>").unwrap() as usize] = 9.0;
        let logits = Tensor::from_vec(logits, (1, 32), &Device::Cpu).unwrap();
        for generated in 0..16 {
            assert_eq!(
                sample_token(&logits, &model.stop_mask, generated).unwrap(),
                regular
            );
        }
        assert_eq!(sample_token(&logits, &model.stop_mask, 16).unwrap(), stop);
    }
}
