use super::config::UnlimitedOcrConfig;
use super::dash::DashDecoder;
use super::infer;
use super::processing::{self, UnlimitedOcrPage, UnlimitedOcrViewMode};
use crate::backbones::deep_encoder::DeepEncoder;
use crate::runtime::checkpoint::collect_safetensors;
pub use crate::runtime::trace::GenerationTrace;
use crate::runtime::trace::top3;
use crate::{Error, RuntimeConfig};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use image::RgbImage;
use std::path::Path;
use tokenizers::Tokenizer;

const MODEL_NAME: &str = "Unlimited-OCR";

/// Reference context length; the default output budget is what remains
/// of it after the prompt.
const MAX_LENGTH: usize = 131_072;

/// Generation controls. Decoding is greedy, like the reference.
#[derive(Debug, Clone)]
#[non_exhaustive]
pub struct UnlimitedOcrGenerationOptions {
    /// Output budget; `None` uses the rest of the 131,072-token context.
    pub max_new_tokens: Option<usize>,
    /// Per-page output cap of the page-count guard: after this many tokens
    /// on one page the next `<PAGE>` (or the end) is forced.
    pub max_page_tokens: usize,
    pub view_mode: UnlimitedOcrViewMode,
    /// No-repeat n-gram length over the generated tokens; `0` disables it.
    /// The default 35 follows the official serving and benchmark setup.
    pub no_repeat_ngram_size: usize,
    /// Tokens the n-gram guard looks back over; `None` uses the official
    /// 128 for one page and 1024 for several.
    pub ngram_window: Option<usize>,
}

impl Default for UnlimitedOcrGenerationOptions {
    fn default() -> Self {
        Self {
            max_new_tokens: None,
            max_page_tokens: 20_000,
            view_mode: UnlimitedOcrViewMode::Auto,
            no_repeat_ngram_size: 35,
            ngram_window: None,
        }
    }
}

impl UnlimitedOcrGenerationOptions {
    pub fn with_max_new_tokens(mut self, value: usize) -> Self {
        self.max_new_tokens = Some(value);
        self
    }

    pub fn with_max_page_tokens(mut self, value: usize) -> Self {
        self.max_page_tokens = value;
        self
    }

    pub fn with_view_mode(mut self, value: UnlimitedOcrViewMode) -> Self {
        self.view_mode = value;
        self
    }

    pub fn with_no_repeat_ngram_size(mut self, value: usize) -> Self {
        self.no_repeat_ngram_size = value;
        self
    }

    pub fn with_ngram_window(mut self, value: usize) -> Self {
        self.ngram_window = Some(value);
        self
    }
}

/// Native Unlimited-OCR-2: the DeepSeek-OCR DeepEncoder and a DeepSeek-V2
/// MoE decoder with DASH block-sparse attention, parsing one or many pages
/// per request.
pub struct UnlimitedOcr {
    config: UnlimitedOcrConfig,
    tokenizer: Tokenizer,
    vision: DeepEncoder,
    image_newline: Tensor,
    view_separator: Tensor,
    decoder: DashDecoder,
    device: Device,
    dtype: DType,
}

impl UnlimitedOcr {
    /// Load a `baidu/Unlimited-OCR-2` model directory.
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, Error> {
        Self::from_dir_with_runtime(model_dir, RuntimeConfig::new(device))
    }

    pub fn from_dir_with_runtime(
        model_dir: impl AsRef<Path>,
        runtime: RuntimeConfig,
    ) -> Result<Self, Error> {
        let dir = model_dir.as_ref();
        let config = UnlimitedOcrConfig::from_path(dir.join("config.json"))?;
        let tokenizer = Tokenizer::from_file(dir.join("tokenizer.json"))
            .map_err(|e| Error::config(format!("Unlimited-OCR tokenizer.json: {e}")))?;
        let sparse = &config.sparse_attention_config;
        if tokenizer.token_to_id("<PAGE>") != Some(sparse.page_token_id)
            || tokenizer.token_to_id("<image>") != Some(sparse.image_token_id)
        {
            return Err(Error::config(
                "Unlimited-OCR tokenizer <PAGE>/<image> ids do not match the config",
            ));
        }
        let (device, dtype) = runtime.resolve();
        let files = collect_safetensors(dir, MODEL_NAME)?;
        // SAFETY: checkpoint files must remain unchanged while the model is alive.
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&files, dtype, &device) }
            .map_err(infer)?;
        let (sam, clip) = config.vision_configs()?;
        let hidden = config.text.hidden_size;
        let vision = DeepEncoder::load(&sam, &clip, hidden, vb.pp("model"))?;
        let image_newline = vb.pp("model").get(hidden, "image_newline").map_err(infer)?;
        let view_separator = vb
            .pp("model")
            .get(hidden, "view_seperator")
            .map_err(infer)?;
        let decoder = DashDecoder::load(&config, vb)?;
        Ok(Self {
            config,
            tokenizer,
            vision,
            image_newline,
            view_separator,
            decoder,
            device,
            dtype,
        })
    }

    /// Parse pages in order with the default "Multi page parsing." prompt;
    /// returns the raw output, one `<PAGE>` per input page.
    pub fn generate(
        &self,
        pages: &[RgbImage],
        options: &UnlimitedOcrGenerationOptions,
    ) -> Result<String, Error> {
        self.generate_with_prompt(pages, processing::DEFAULT_PROMPT, options)
    }

    /// Parse pages with a custom instruction placed after the page images.
    pub fn generate_with_prompt(
        &self,
        pages: &[RgbImage],
        instruction: &str,
        options: &UnlimitedOcrGenerationOptions,
    ) -> Result<String, Error> {
        let tokens = self.generate_tokens(pages, instruction, options, None)?;
        self.decode_tokens(&tokens)
    }

    /// Greedy generation with per-step top-3 logits for reference alignment.
    pub fn generate_traced(
        &self,
        pages: &[RgbImage],
        options: &UnlimitedOcrGenerationOptions,
    ) -> Result<GenerationTrace, Error> {
        let mut step_top = Vec::new();
        let tokens = self.generate_tokens(
            pages,
            processing::DEFAULT_PROMPT,
            options,
            Some(&mut step_top),
        )?;
        Ok(GenerationTrace { tokens, step_top })
    }

    /// Preprocess pages under `mode` without running the encoder.
    pub fn preprocess_pages(
        &self,
        pages: &[RgbImage],
        mode: UnlimitedOcrViewMode,
    ) -> Result<Vec<UnlimitedOcrPage>, Error> {
        let tiled = match mode {
            UnlimitedOcrViewMode::Auto => pages.len() == 1,
            UnlimitedOcrViewMode::Tiled => true,
            UnlimitedOcrViewMode::Global => false,
        };
        pages
            .iter()
            .map(|page| processing::preprocess_page(page, tiled, &self.device, self.dtype))
            .collect()
    }

    /// Prompt token IDs for preprocessed pages and an instruction.
    pub fn prompt_ids(
        &self,
        pages: &[UnlimitedOcrPage],
        instruction: &str,
    ) -> Result<Vec<u32>, Error> {
        let sparse = &self.config.sparse_attention_config;
        processing::prompt_ids(
            &self.tokenizer,
            pages,
            instruction,
            self.config.text.bos_token_id,
            sparse.page_token_id,
            sparse.image_token_id,
        )
    }

    /// Decode output ids keeping `<PAGE>` and grounding markers, without the
    /// sentence markers.
    pub fn decode_tokens(&self, ids: &[u32]) -> Result<String, Error> {
        self.tokenizer
            .decode(ids, false)
            .map_err(|e| Error::invalid_input(format!("Unlimited-OCR decoding failed: {e}")))
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    /// Visual features of one page in placeholder order: tile rows (each
    /// ending in the newline embedding), global rows, view separator.
    fn page_features(&self, page: &UnlimitedOcrPage) -> Result<Tensor, Error> {
        let newline = self.image_newline.unsqueeze(0).map_err(infer)?;
        let rows = |features: Tensor, rows: usize, cols: usize| -> Result<Tensor, Error> {
            let dim = features.dim(candle_core::D::Minus1).map_err(infer)?;
            let features = features.reshape((rows, cols, dim)).map_err(infer)?;
            let newlines = newline
                .unsqueeze(0)
                .and_then(|n| n.broadcast_as((rows, 1, dim)))
                .map_err(infer)?;
            Tensor::cat(&[&features, &newlines], 1)
                .and_then(|x| x.reshape((rows * (cols + 1), dim)))
                .map_err(infer)
        };
        let global = self.vision.forward(&page.global)?;
        let side = global.dim(1).map_err(infer)?.isqrt();
        let global = rows(global.squeeze(0).map_err(infer)?, side, side)?;
        let mut parts = Vec::with_capacity(3);
        if let Some(tiles) = &page.tiles {
            let (columns, tile_rows) = page.grid;
            let local = self.vision.forward(tiles)?;
            let (_, per_tile, dim) = local.dims3().map_err(infer)?;
            let q = per_tile.isqrt();
            let mosaic = local
                .reshape((tile_rows, columns, q, q, dim))
                .and_then(|x| x.permute((0, 2, 1, 3, 4)))
                .and_then(|x| x.contiguous())
                .and_then(|x| x.reshape((tile_rows * q * columns * q, dim)))
                .map_err(infer)?;
            parts.push(rows(mosaic, tile_rows * q, columns * q)?);
        }
        parts.push(global);
        parts.push(self.view_separator.unsqueeze(0).map_err(infer)?);
        Tensor::cat(&parts, 0).map_err(infer)
    }

    fn embeddings(&self, ids: &[u32], pages: &[UnlimitedOcrPage]) -> Result<Tensor, Error> {
        let image_token = self.config.sparse_attention_config.image_token_id;
        let ids_tensor = Tensor::new(ids, &self.device)
            .and_then(|t| t.unsqueeze(0))
            .map_err(infer)?;
        let text = self.decoder.embed(&ids_tensor)?.squeeze(0).map_err(infer)?;
        let visual = pages
            .iter()
            .map(|page| self.page_features(page))
            .collect::<Result<Vec<_>, _>>()?;
        let visual = Tensor::cat(&visual, 0)
            .and_then(|v| v.to_dtype(text.dtype()))
            .map_err(infer)?;
        let placeholders = ids.iter().filter(|&&id| id == image_token).count();
        if placeholders != visual.dim(0).map_err(infer)? {
            return Err(Error::invalid_input(format!(
                "Unlimited-OCR image placeholders ({placeholders}) != visual features ({})",
                visual.dim(0).map_err(infer)?
            )));
        }
        let mut pieces = Vec::new();
        let (mut start, mut consumed) = (0, 0);
        while start < ids.len() {
            let is_image = ids[start] == image_token;
            let len = ids[start..]
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

    fn generate_tokens(
        &self,
        pages: &[RgbImage],
        instruction: &str,
        options: &UnlimitedOcrGenerationOptions,
        mut step_top: Option<&mut Vec<[(u32, f32); 3]>>,
    ) -> Result<Vec<u32>, Error> {
        if pages.is_empty() {
            return Err(Error::invalid_input(
                "Unlimited-OCR needs at least one page",
            ));
        }
        let inputs = self.preprocess_pages(pages, options.view_mode)?;
        let ids = self.prompt_ids(&inputs, instruction)?;
        let max_new_tokens = options
            .max_new_tokens
            .unwrap_or_else(|| MAX_LENGTH.saturating_sub(ids.len()).max(1));
        let embeds = self.embeddings(&ids, &inputs)?;
        drop(inputs);

        self.decoder.clear_cache();
        let result = (|| {
            let (mut logits, mut state) = self.decoder.prefill(&ids, &embeds)?;
            let page_token = self.config.sparse_attention_config.page_token_id;
            let eos = self.config.text.eos_token_id;
            let mut guard = PageCountGuard::new(pages.len(), options.max_page_tokens);
            let ngram_window =
                options
                    .ngram_window
                    .unwrap_or(if pages.len() == 1 { 128 } else { 1024 });
            let mut generated = Vec::new();
            for step in 0..max_new_tokens {
                if let Some(step_top) = step_top.as_deref_mut() {
                    step_top.push(top3(&logits).map_err(infer)?);
                }
                let mut scores = logits.to_vec1::<f32>().map_err(infer)?;
                ban_repeated_ngrams(
                    &mut scores,
                    &generated,
                    options.no_repeat_ngram_size,
                    ngram_window,
                );
                guard.apply(&mut scores, &generated, page_token, eos);
                let token = argmax(&scores);
                if token == eos {
                    break;
                }
                generated.push(token);
                let position = ids.len() + step;
                if step + 1 == max_new_tokens || position + 1 >= MAX_LENGTH {
                    break;
                }
                logits = self.decoder.decode(token, position, &mut state)?;
            }
            Ok(generated)
        })();
        self.decoder.clear_cache();
        result
    }
}

/// First index of the largest score, like `torch.argmax`.
fn argmax(scores: &[f32]) -> u32 {
    scores
        .iter()
        .enumerate()
        .fold((0, f32::NEG_INFINITY), |best, (index, &value)| {
            if value > best.1 { (index, value) } else { best }
        })
        .0 as u32
}

/// The reference no-repeat n-gram guard (vLLM/SGLang DeepSeek-OCR
/// semantics): ban every token that would complete an n-gram already seen
/// among the last `window` generated tokens.
fn ban_repeated_ngrams(scores: &mut [f32], generated: &[u32], size: usize, window: usize) {
    if size == 0 || generated.len() < size {
        return;
    }
    let prefix = &generated[generated.len() + 1 - size..];
    let start = generated.len().saturating_sub(window);
    for index in start..=generated.len() - size {
        let ngram = &generated[index..index + size];
        if ngram[..size - 1] == *prefix
            && let Some(score) = scores.get_mut(ngram[size - 1] as usize)
        {
            *score = f32::NEG_INFINITY;
        }
    }
}

/// The reference page-count guard: no end before every page has its
/// `<PAGE>`, a forced end after too many, and a forced page break after
/// `max_page_tokens` tokens on one page.
struct PageCountGuard {
    expected: usize,
    max_page_tokens: usize,
    markers: usize,
    page_len: usize,
    scanned: usize,
}

impl PageCountGuard {
    fn new(expected: usize, max_page_tokens: usize) -> Self {
        Self {
            expected,
            max_page_tokens,
            markers: 0,
            page_len: 0,
            scanned: 0,
        }
    }

    fn apply(&mut self, scores: &mut [f32], generated: &[u32], page: u32, eos: u32) {
        for &token in &generated[self.scanned..] {
            if token == page {
                self.markers += 1;
                self.page_len = 0;
            } else {
                self.page_len += 1;
            }
        }
        self.scanned = generated.len();
        // A forced token stays selectable even when the n-gram guard has
        // banned it; the reference would fall back to token 0 there.
        let keep_only = |scores: &mut [f32], token: u32| {
            let kept = scores[token as usize];
            scores.fill(f32::NEG_INFINITY);
            scores[token as usize] = if kept.is_finite() { kept } else { 0.0 };
        };
        if self.markers > self.expected {
            keep_only(scores, eos);
        } else if self.max_page_tokens > 0 && self.page_len >= self.max_page_tokens {
            keep_only(
                scores,
                if self.markers < self.expected {
                    page
                } else {
                    eos
                },
            );
        } else if self.markers < self.expected {
            scores[eos as usize] = f32::NEG_INFINITY;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn guards_follow_the_reference() {
        let (page, eos) = (2u32, 1u32);
        let mut guard = PageCountGuard::new(2, 3);
        let mut scores = vec![0.0f32; 4];
        guard.apply(&mut scores, &[page], page, eos);
        assert_eq!(
            scores[eos as usize],
            f32::NEG_INFINITY,
            "no end before page two"
        );
        let mut scores = vec![0.0f32; 4];
        guard.apply(&mut scores, &[page, 3, 3, 3], page, eos);
        assert_eq!(
            scores,
            vec![f32::NEG_INFINITY, f32::NEG_INFINITY, 0.0, f32::NEG_INFINITY]
        );
        // The page cap still forces `<PAGE>` after the n-gram guard banned it.
        let mut guard = PageCountGuard::new(2, 1);
        let mut scores = vec![0.0f32; 4];
        scores[page as usize] = f32::NEG_INFINITY;
        guard.apply(&mut scores, &[page, 3], page, eos);
        assert_eq!(argmax(&scores), page);

        // `5 6` was followed by 7 inside the window, so 7 is banned; a
        // window that no longer holds that n-gram bans nothing.
        let mut scores = vec![0.0f32; 8];
        ban_repeated_ngrams(&mut scores, &[5, 6, 7, 5, 6], 3, 128);
        assert_eq!(scores[7], f32::NEG_INFINITY);
        assert_eq!(scores.iter().filter(|s| s.is_infinite()).count(), 1);
        let mut scores = vec![0.0f32; 8];
        ban_repeated_ngrams(&mut scores, &[5, 6, 7, 5, 6], 3, 2);
        assert!(scores.iter().all(|s| s.is_finite()));
    }
}
