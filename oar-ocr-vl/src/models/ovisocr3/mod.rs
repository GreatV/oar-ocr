//! OvisOCR3 Markdown, structured-page, and cropped-region recognition.
//!
//! Select this version explicitly through [`OvisOcr3`] or the repository ID
//! `ATH-MaaS/OvisOCR3`; checkpoint configuration never selects the protocol.
//! The OvisOCR2 engine supplies identical image processing, Qwen3.5 decoding,
//! and no-thinking chat framing. Its decoder does not load or use MTP layers,
//! so OvisOCR3's zero MTP depth requires no speculative-decoding path.

mod adapter;
mod parser;
pub mod prompts;

use crate::ovisocr2::{OvisOcr2, OvisOcr2Config, OvisOcr2ImageProcessorConfig, postprocess_text};
use crate::{BatchResult, Error, RuntimeConfig};
use candle_core::Device;
use image::RgbImage;
use std::path::Path;
use tokenizers::Tokenizer;

pub use crate::ovisocr2::{DEFAULT_MAX_NEW_TOKENS, GenerationTrace};
pub use parser::{OvisOcr3ParseMode, OvisOcr3ParseOptions};
pub use prompts::DOCUMENT_MARKDOWN as DEFAULT_PROMPT;

/// OvisOCR3 protocols over the shared OvisOCR2 inference engine.
pub struct OvisOcr3 {
    engine: OvisOcr2,
}

impl OvisOcr3 {
    /// Load an explicitly selected OvisOCR3 checkpoint directory.
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, Error> {
        Self::from_dir_with_runtime(model_dir, RuntimeConfig::new(device))
    }

    /// Load OvisOCR3 with an explicit device and compute dtype policy.
    pub fn from_dir_with_runtime(
        model_dir: impl AsRef<Path>,
        runtime: RuntimeConfig,
    ) -> Result<Self, Error> {
        Ok(Self {
            engine: OvisOcr2::from_dir_with_runtime(model_dir, runtime)?,
        })
    }

    /// Generate Markdown, retaining image tags and cleaning repeated tails.
    pub fn generate(&self, images: &[RgbImage], max_new_tokens: usize) -> BatchResult<String> {
        self.parse_with_image_tags(images, max_new_tokens, true)
    }

    /// Parse Markdown with image-tag filtering and repeated-tail cleanup.
    pub fn parse(&self, images: &[RgbImage], max_new_tokens: usize) -> BatchResult<String> {
        self.parse_with_image_tags(images, max_new_tokens, false)
    }

    /// Strip whitespace, optionally filter image-tag blocks, then clean repeats.
    pub fn parse_with_image_tags(
        &self,
        images: &[RgbImage],
        max_new_tokens: usize,
        keep_image_tags: bool,
    ) -> BatchResult<String> {
        Ok(self
            .generate_with_prompt(images, DEFAULT_PROMPT, max_new_tokens)?
            .into_iter()
            .map(|result| result.map(|text| postprocess_text(text, keep_image_tags)))
            .collect())
    }

    /// Generate with a task prompt, trimming only outer whitespace.
    ///
    /// Each image is a separate request. Fill template placeholders with
    /// literal replacement before passing a prompt from [`prompts`]. JSON
    /// responses are returned without repair or Markdown post-processing.
    pub fn generate_with_prompt(
        &self,
        images: &[RgbImage],
        prompt: &str,
        max_new_tokens: usize,
    ) -> BatchResult<String> {
        Ok(self
            .engine
            .generate_tokens_with_prompt(images, prompt, max_new_tokens)?
            .into_iter()
            .map(|result| result.and_then(|tokens| self.decode_tokens_raw(&tokens)))
            .collect())
    }

    /// Generate token IDs with the verbatim v3 Markdown prompt.
    pub fn generate_tokens(
        &self,
        images: &[RgbImage],
        max_new_tokens: usize,
    ) -> BatchResult<Vec<u32>> {
        self.engine
            .generate_tokens_with_prompt(images, DEFAULT_PROMPT, max_new_tokens)
    }

    /// Generate Markdown tokens and per-step top-three logits for alignment.
    pub fn generate_traced(
        &self,
        image: &RgbImage,
        max_new_tokens: usize,
    ) -> Result<GenerationTrace, Error> {
        self.engine
            .generate_traced_with_prompt(image, DEFAULT_PROMPT, max_new_tokens)
    }

    /// Decode tokens with Markdown repeated-tail cleanup, keeping image tags.
    pub fn decode_tokens(&self, tokens: &[u32]) -> Result<String, Error> {
        self.engine.decode_tokens(tokens)
    }

    /// Decode tokens, trimming only outer whitespace as in OvisOCR2.
    pub fn decode_tokens_raw(&self, tokens: &[u32]) -> Result<String, Error> {
        self.engine.decode_tokens_raw(tokens)
    }

    /// Return the checkpoint tokenizer used by the shared engine.
    pub fn tokenizer(&self) -> &Tokenizer {
        self.engine.tokenizer()
    }

    /// Return the model configuration in the shared OvisOCR2 schema.
    pub fn config(&self) -> &OvisOcr2Config {
        self.engine.config()
    }

    /// Return the image processor configuration shared with OvisOCR2.
    pub fn image_processor_config(&self) -> &OvisOcr2ImageProcessorConfig {
        self.engine.image_processor_config()
    }

    fn generate_one(
        &self,
        image: &RgbImage,
        prompt: &str,
        max_new_tokens: usize,
    ) -> Result<String, Error> {
        self.generate_with_prompt(std::slice::from_ref(image), prompt, max_new_tokens)?
            .pop()
            .ok_or_else(|| Error::invalid_input("OvisOCR3 returned no result"))?
    }
}
