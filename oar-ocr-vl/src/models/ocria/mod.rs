//! OCRia-1 page-to-Markdown model support.
//!
//! [`Ocria`] (llamaindex/OCRia-1) is a Qwen3.5-4B fine-tune that reads one page image and writes Markdown: tables as HTML, charts as HTML data tables, formulas as LaTeX and diagrams as Mermaid. Select it explicitly through [`Ocria`] or the repository ID `llamaindex/OCRia-1`.
//!
//! The checkpoint is a plain `Qwen3_5ForConditionalGeneration`, so the Xiaomi-OCR-0 engine supplies image processing, decoding and the no-thinking chat framing. The protocol is what differs: the model's own system prompt (`prompt.txt` in the model directory, read at load and never embedded here), an image-only user turn, and an answer wrapped in a Markdown code fence.
//!
//! The weights and `prompt.txt` are under the OCRia-1 Model and Code License, which restricts commercial use. The layout-box code shipped with the model is not ported.

mod parser;

use crate::xiaomi_ocr::{XiaomiOcr, XiaomiOcrConfig, XiaomiOcrProcessorConfig};
use crate::{BatchResult, Error, RuntimeConfig};
use candle_core::Device;
use image::RgbImage;
use once_cell::sync::Lazy;
use regex::Regex;
use std::path::Path;
use tokenizers::Tokenizer;

pub use crate::xiaomi_ocr::GenerationTrace;
pub use parser::OcriaParseOptions;

/// Generation limit of the official examples; dense pages can need more than 20,000 tokens.
pub const DEFAULT_MAX_NEW_TOKENS: usize = 24_576;

/// Page image budget behind the published benchmark numbers, about 210 dpi for a letter page.
pub const DEFAULT_MAX_PIXELS: u64 = 4_100_000;

/// File in the model directory that holds the system prompt.
const PROMPT_FILE: &str = "prompt.txt";

/// The files of the model repository that inference reads, with its license.
#[cfg(feature = "auto-download")]
pub(crate) const INFERENCE_FILES: &[&str] = &[
    "config.json",
    "model.safetensors",
    "processor_config.json",
    "tokenizer.json",
    PROMPT_FILE,
    "LICENSE",
];

static FENCE_OPEN_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"^```(?:markdown)?\s*\n").expect("static regex"));
static FENCE_CLOSE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"(?:^|\n)```\s*$").expect("static regex"));

/// OCRia-1 over the shared Qwen3.5 inference engine.
pub struct Ocria {
    engine: XiaomiOcr,
    system_prompt: String,
}

impl Ocria {
    /// Set whether to use CUDA graphs for decoding. Enabled by default.
    pub fn with_cuda_graphs(mut self, enabled: bool) -> Self {
        self.engine = self.engine.with_cuda_graphs(enabled);
        self
    }

    /// Load an OCRia-1 Hugging Face model directory.
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, Error> {
        Self::from_dir_with_runtime(model_dir, RuntimeConfig::new(device))
    }

    /// Load OCRia-1 with an explicit device and compute dtype policy.
    pub fn from_dir_with_runtime(
        model_dir: impl AsRef<Path>,
        runtime: RuntimeConfig,
    ) -> Result<Self, Error> {
        let model_dir = model_dir.as_ref();
        let prompt_path = model_dir.join(PROMPT_FILE);
        let system_prompt = std::fs::read_to_string(&prompt_path).map_err(|e| Error::Config {
            message: format!(
                "OCRia-1 needs its system prompt at {}: {e}",
                prompt_path.display()
            ),
        })?;
        if system_prompt.trim().is_empty() {
            return Err(Error::Config {
                message: format!("OCRia-1 system prompt {} is empty", prompt_path.display()),
            });
        }
        Ok(Self {
            engine: XiaomiOcr::from_dir_with_runtime(model_dir, runtime)?,
            system_prompt,
        })
    }

    /// Generate the answer's token ids for each page.
    ///
    /// Each image is a separate request and reaches the model at the size the checkpoint's image processor gives it; the page budget of [`OcriaParseOptions`] applies only to page parsing.
    pub fn generate_tokens(
        &self,
        images: &[RgbImage],
        max_new_tokens: usize,
    ) -> BatchResult<Vec<u32>> {
        Ok(images
            .iter()
            .map(|image| self.request(image, max_new_tokens, None))
            .collect())
    }

    /// Generate the raw answer for each page, still inside its Markdown fence.
    pub fn generate(&self, images: &[RgbImage], max_new_tokens: usize) -> BatchResult<String> {
        Ok(self
            .generate_tokens(images, max_new_tokens)?
            .into_iter()
            .map(|result| result.and_then(|tokens| self.decode_tokens(&tokens)))
            .collect())
    }

    /// Generate each page's Markdown, with the answer's fence removed.
    pub fn parse(&self, images: &[RgbImage], max_new_tokens: usize) -> BatchResult<String> {
        Ok(self
            .generate(images, max_new_tokens)?
            .into_iter()
            .map(|result| result.map(|answer| page_markdown(&answer)))
            .collect())
    }

    /// Greedy generation with per-step top-three logits, for numerical alignment against the reference.
    pub fn generate_traced(
        &self,
        image: &RgbImage,
        max_new_tokens: usize,
    ) -> Result<GenerationTrace, Error> {
        let mut step_top = Vec::new();
        let tokens = self.request(image, max_new_tokens, Some(&mut step_top))?;
        Ok(GenerationTrace { tokens, step_top })
    }

    /// Decode token ids into the raw answer, trimming only outer whitespace.
    pub fn decode_tokens(&self, tokens: &[u32]) -> Result<String, Error> {
        self.engine.decode_tokens_raw(tokens)
    }

    /// The system prompt read from the model directory.
    pub fn system_prompt(&self) -> &str {
        &self.system_prompt
    }

    /// Return the checkpoint tokenizer.
    pub fn tokenizer(&self) -> &Tokenizer {
        self.engine.tokenizer()
    }

    /// Return the model configuration in the shared Qwen3.5 schema.
    pub fn config(&self) -> &XiaomiOcrConfig {
        self.engine.config()
    }

    /// Return the processor configuration in the shared Qwen3.5 schema.
    pub fn processor_config(&self) -> &XiaomiOcrProcessorConfig {
        self.engine.processor_config()
    }

    fn request(
        &self,
        image: &RgbImage,
        max_new_tokens: usize,
        step_top: Option<&mut Vec<[(u32, f32); 3]>>,
    ) -> Result<Vec<u32>, Error> {
        self.engine.generate_request(
            image,
            Some(self.system_prompt.as_str()),
            "",
            max_new_tokens,
            step_top,
        )
    }
}

/// Markdown of a page from the model's raw answer: the surrounding ```` ```markdown ```` fence is removed, and an answer cut off before its closing fence keeps everything after the opening one. An answer without the opening fence is returned as it is.
///
/// Only the outer fence is touched, so a page that ends with a fenced block of its own (a Mermaid diagram, code) keeps that block whole.
pub fn page_markdown(answer: &str) -> String {
    let answer = answer.trim();
    let Some(open) = FENCE_OPEN_RE.find(answer) else {
        return answer.to_string();
    };
    FENCE_CLOSE_RE
        .replace(&answer[open.end()..], "")
        .trim()
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn page_markdown_removes_only_the_outer_fence() {
        assert_eq!(
            page_markdown("```markdown\n# Title\n\nBody.\n```"),
            "# Title\n\nBody."
        );
        // A truncated answer has no closing fence.
        assert_eq!(page_markdown("```markdown\n# Title\n\nBo"), "# Title\n\nBo");
        // The first inner closing fence is not the end of the page.
        let page = "A\n\n```mermaid\ngraph TD\n```\n\nB";
        assert_eq!(page_markdown(&format!("```markdown\n{page}\n```\n")), page);
        // Without the wrapper, a closing fence belongs to the page.
        assert_eq!(page_markdown("A\n```"), "A\n```");
        assert_eq!(page_markdown("```markdown\n\n```"), "");
    }
}
