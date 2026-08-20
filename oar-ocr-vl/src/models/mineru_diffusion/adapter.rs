//! Region-recognition adapter for MinerU-Diffusion.

use super::{DiffusionGenerationConfig, MinerUDiffusion};
use crate::api::error::Error;
use crate::api::recognition::{BackendCapabilities, RecognitionBackend, RecognitionTask};
use image::RgbImage;

impl RecognitionBackend for MinerUDiffusion {
    fn recognize(
        &self,
        image: RgbImage,
        task: RecognitionTask,
        max_tokens: usize,
    ) -> Result<String, Error> {
        if max_tokens == 0 {
            return Ok(String::new());
        }
        let prompt = match task {
            RecognitionTask::Ocr => "\nText Recognition:",
            RecognitionTask::Table => "\nTable Recognition:",
            RecognitionTask::Formula => "\nFormula Recognition:",
            RecognitionTask::Chart => "\nDocument Parsing:",
        };
        let mut generation = DiffusionGenerationConfig::default();
        generation.gen_length = max_tokens
            .checked_next_multiple_of(generation.block_length)
            .ok_or_else(|| {
                Error::invalid_input(format!(
                    "MinerU-Diffusion max_tokens {max_tokens} is too large"
                ))
            })?;
        let mut tokens = self.generate_token_ids(&image, prompt, &generation)?;
        tokens.truncate(max_tokens);
        self.decode_tokens(&tokens, true)
    }

    fn capabilities(&self) -> BackendCapabilities {
        BackendCapabilities {
            table_output_is_otsl: true,
            truncate_repetitive_output: true,
            ..BackendCapabilities::default()
        }
    }
}
