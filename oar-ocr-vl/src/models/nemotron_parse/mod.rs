//! Native single-pass page parser for Nemotron-3.5-Parse and OpenThai2.0-OCR-Markdown.
mod decoder;
mod parser;
mod position;
mod processing;
mod vision;

pub(crate) use parser::source_boxes;

use crate::{
    AnyPageParserOptions, Error, PageDocument, PageParser,
    runtime::{
        checkpoint::{collect_safetensors, load_json_config},
        device::select_dtype,
        sampling::{RepetitionHistory, apply_unique_repetition_penalty, argmax_token},
    },
};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use image::RgbImage;
use serde::Deserialize;
use std::path::Path;
use tokenizers::{Decoder as _, Model as _, Tokenizer};

#[cfg(feature = "auto-download")]
pub(crate) const INFERENCE_FILES: &[&str] = &[
    "config.json",
    "model.safetensors",
    "generation_config.json",
    "preprocessor_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "processor_config.json",
    "chat_template.jinja",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "TOKENIZER_ATTRIBUTION.md",
];

const PROMPT: &str = "</s><s><predict_bbox><predict_classes><output_markdown><predict_text_in_pic>";

#[derive(Deserialize)]
struct ModelConfig {
    legacy_causal_vision_prefix: bool,
    max_sequence_length: usize,
}

#[derive(Deserialize)]
struct Generation {
    max_new_tokens: usize,
    repetition_penalty: f32,
    eos_token_id: u32,
}

fn infer(error: candle_core::Error) -> Error {
    crate::runtime::errors::candle_to_ocr_inference("Nemotron Parse", "page inference", error)
}

/// A page encoder and prefix-causal decoder loaded from a Nemotron Parse family checkpoint.
pub struct NemotronParse {
    device: Device,
    dtype: DType,
    model_config: ModelConfig,
    processor: processing::ProcessorConfig,
    encoder: vision::Encoder,
    decoder: decoder::Decoder,
    tokenizer: Tokenizer,
    prompt: Vec<u32>,
    generation: Generation,
}

impl NemotronParse {
    /// Load the visual encoder, tied decoder and tokenizer from a local checkpoint directory.
    pub fn from_dir(dir: impl AsRef<Path>, device: Device) -> Result<Self, Error> {
        let dir = dir.as_ref();
        let dtype = if device.is_cpu() {
            DType::F32
        } else {
            select_dtype(&device)
        };
        let model_config: ModelConfig =
            load_json_config(dir.join("config.json"), "Nemotron Parse", "config.json")?;
        if !model_config.legacy_causal_vision_prefix {
            return Err(Error::config(
                "Nemotron Parse requires a causal visual prefix",
            ));
        }
        let processor = load_json_config(
            dir.join("preprocessor_config.json"),
            "Nemotron Parse",
            "preprocessor_config.json",
        )?;
        let paths = collect_safetensors(dir, "Nemotron Parse")?;
        // SAFETY: The checkpoint files must remain unchanged while the parser is alive.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&paths, dtype, &device)? };
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| Error::config(e.to_string()))?;
        let prompt = tokenizer
            .encode(PROMPT, false)
            .map_err(|e| Error::config(e.to_string()))?
            .get_ids()
            .to_vec();
        let generation = load_json_config(
            dir.join("generation_config.json"),
            "Nemotron Parse",
            "generation_config.json",
        )?;
        Ok(Self {
            model_config,
            processor,
            encoder: vision::Encoder::load(vb.pp("encoder"))?,
            decoder: decoder::Decoder::load(vb.pp("decoder"))?,
            device,
            dtype,
            tokenizer,
            prompt,
            generation,
        })
    }

    fn decode(&self, ids: &[u32]) -> Result<String, Error> {
        // The checkpoint decodes added tokens using their original spelling (e.g. Thai sara am).
        // Normalization of those tokens applies when matching the input prompt.
        let added = self
            .tokenizer
            .get_added_vocabulary()
            .get_added_tokens_decoder();
        let tokens = ids
            .iter()
            .filter_map(|id| {
                added
                    .get(id)
                    .map(|t| t.content.clone())
                    .or_else(|| self.tokenizer.get_model().id_to_token(*id))
            })
            .collect();
        self.tokenizer
            .get_decoder()
            .ok_or_else(|| Error::config("Nemotron Parse tokenizer has no decoder"))?
            .decode(tokens)
            .map_err(|e| Error::config(e.to_string()))
    }

    fn next_token(&self, logits: &Tensor, seen: &[u32]) -> Result<u32, Error> {
        let penalty = self.generation.repetition_penalty;
        #[cfg(feature = "cuda")]
        if self.device.is_cuda() {
            return crate::runtime::sampling::select_penalized_greedy_token_cuda(
                "Nemotron Parse",
                logits,
                seen,
                penalty,
            );
        }
        let mut scores = logits.to_dtype(DType::F32)?.to_vec1::<f32>()?;
        apply_unique_repetition_penalty(&mut scores, seen, penalty);
        Ok(argmax_token(&scores))
    }

    fn generate(&self, image: &RgbImage, max_tokens: usize) -> Result<Vec<u32>, Error> {
        let page = processing::Page::prepare(image, &self.processor, self.dtype, &self.device)?;
        let prefix = self.encoder.forward(&page)?;
        let prefix_len = prefix.dim(1)? + self.prompt.len();
        if max_tokens == 0 || prefix_len > self.model_config.max_sequence_length {
            return Err(Error::invalid_input(
                "Nemotron Parse requires room for the visual prefix and completion",
            ));
        }
        let capacity = prefix_len
            .saturating_add(max_tokens)
            .min(self.model_config.max_sequence_length);
        let mut cache = self.decoder.cache(capacity);
        let mut history = self.prompt.clone();
        let ids = Tensor::new(history.as_slice(), &self.device)?.unsqueeze(0)?;
        let mut logits = self.decoder.forward(&ids, Some(&prefix), &mut cache)?;
        let mut seen = RepetitionHistory::from_tokens(logits.elem_count(), &history);
        // The checkpoint forces EOS at the token limit; the single-pass API rejects a length stop.
        for _ in 1..max_tokens {
            let token = self.next_token(&logits, &seen.unique)?;
            seen.insert(token);
            history.push(token);
            if token == self.generation.eos_token_id {
                return Ok(history);
            }
            if cache[0].current_seq_len() == capacity {
                return Err(Error::invalid_input(
                    "Nemotron Parse reached its context limit before completing the page",
                ));
            }
            let ids = Tensor::new(&[[token]], &self.device)?;
            logits = self.decoder.forward(&ids, None, &mut cache)?;
        }
        Err(Error::invalid_input(
            "Nemotron Parse reached the generation limit before completing the page",
        ))
    }
}

impl PageParser for NemotronParse {
    type Options = AnyPageParserOptions;
    fn parse_page(&self, image: &RgbImage, options: &Self::Options) -> Result<PageDocument, Error> {
        let ids = self.generate(
            image,
            options
                .max_new_tokens
                .unwrap_or(self.generation.max_new_tokens),
        )?;
        let raw = self.decode(&ids)?;
        Ok(parser::parse(&raw))
    }
}
