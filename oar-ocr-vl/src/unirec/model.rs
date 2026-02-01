//! UniRec Vision-Language model for unified text, formula, and table recognition.

use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module};
use image::RgbImage;
use once_cell::sync::Lazy;
use regex::Regex;
use std::path::Path;
use tokenizers::Tokenizer;
use tokenizers::decoders::byte_level::ByteLevel;

use super::config::UniRecConfig;
use super::decoder::M2M100Decoder;
use super::encoder::FocalSVTR;
use crate::utils::{candle_to_ocr_inference, image::image_to_chw};
use oar_ocr_core::core::OCRError;

// Static regexes for postprocessing (compiled once)
static UNDERSCORE_RE: Lazy<Regex> =
    Lazy::new(|| Regex::new(r"_{4,}").expect("static underscore regex"));
static DOTS_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"\.{4,}").expect("static dots regex"));
static SPACES_RE: Lazy<Regex> = Lazy::new(|| Regex::new(r"[ ]{2,}").expect("static spaces regex"));

/// UniRec model for unified text, formula, and table recognition.
pub struct UniRec {
    device: Device,
    dtype: DType,
    cfg: UniRecConfig,
    tokenizer: Tokenizer,
    encoder: FocalSVTR,
    decoder: M2M100Decoder,
    lm_head: Linear,
}

impl UniRec {
    /// Load UniRec model from a directory containing model weights and config.
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, OCRError> {
        let model_dir = model_dir.as_ref();

        // Load config
        let cfg = UniRecConfig::from_path(model_dir.join("config.json"))?;

        // Load tokenizer with ByteLevel decoder for proper BPE decoding
        let mut tokenizer =
            Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| {
                OCRError::ConfigError {
                    message: format!("Failed to load UniRec tokenizer: {}", e),
                }
            })?;
        tokenizer.with_decoder(Some(ByteLevel::default()));

        // Determine dtype
        let dtype = device.bf16_default_to_f32();

        // Load model weights
        let vb = unsafe {
            candle_nn::VarBuilder::from_mmaped_safetensors(
                &[model_dir.join("model.safetensors")],
                dtype,
                &device,
            )
            .map_err(|e| candle_to_ocr_inference("UniRec", "load model.safetensors", e))?
        };

        // Load encoder
        let encoder = FocalSVTR::load(&cfg, vb.pp("model.encoder"))?;

        // Load decoder
        let decoder = M2M100Decoder::load(&cfg, vb.pp("model.decoder"))?;

        // Load language model head
        let lm_head = candle_nn::linear_no_bias(cfg.d_model, cfg.vocab_size, vb.pp("lm_head"))
            .map_err(|e| candle_to_ocr_inference("UniRec", "load lm_head", e))?;

        Ok(Self {
            device,
            dtype,
            cfg,
            tokenizer,
            encoder,
            decoder,
            lm_head,
        })
    }

    /// Compute target dimensions for an image preserving aspect ratio.
    fn compute_target_size(&self, image: &RgbImage) -> (usize, usize) {
        let (max_w, max_h) = (self.cfg.input_width, self.cfg.input_height);
        let (orig_w, orig_h) = (image.width() as usize, image.height() as usize);

        // Compute resize dimensions preserving aspect ratio
        let (new_w, new_h) = if orig_w > max_w || orig_h > max_h {
            let aspect_ratio = orig_w as f64 / orig_h as f64;
            if (max_w as f64 / max_h as f64) >= aspect_ratio {
                // Height limited
                let new_h = max_h;
                let new_w = (new_h as f64 * aspect_ratio) as usize;
                (new_w, new_h)
            } else {
                // Width limited
                let new_w = max_w;
                let new_h = (new_w as f64 / aspect_ratio) as usize;
                (new_w, new_h)
            }
        } else {
            (orig_w, orig_h)
        };

        // Round to multiples of 64 (minimum 64)
        let target_h = ((new_h / 64) * 64).max(64);
        let target_w = ((new_w / 64) * 64).max(64);

        (target_w, target_h)
    }

    /// Preprocess an image with specific target dimensions (for batching).
    fn preprocess_image_with_size(
        &self,
        image: &RgbImage,
        target_w: usize,
        target_h: usize,
    ) -> Result<Tensor, OCRError> {
        // First resize to natural dimensions preserving aspect ratio
        let (natural_w, natural_h) = self.compute_target_size(image);

        let resized = image::imageops::resize(
            image,
            natural_w as u32,
            natural_h as u32,
            image::imageops::FilterType::CatmullRom,
        );

        // Convert to tensor with normalization and padding to target size
        let chw_data = image_to_chw(
            &resized,
            &[0.5, 0.5, 0.5],
            &[0.5, 0.5, 0.5],
            Some(1.0 / 255.0),
        );

        let mut data = vec![0f32; 3 * target_h * target_w];
        let natural_area = natural_h * natural_w;
        let target_area = target_h * target_w;

        // Copy into padded buffer respecting stride
        for c in 0..3 {
            let src_channel_offset = c * natural_area;
            let dst_channel_offset = c * target_area;
            for y in 0..natural_h {
                let src_row_start = src_channel_offset + y * natural_w;
                let dst_row_start = dst_channel_offset + y * target_w;
                data[dst_row_start..dst_row_start + natural_w]
                    .copy_from_slice(&chw_data[src_row_start..src_row_start + natural_w]);
            }
        }

        Tensor::from_vec(data, (1, 3, target_h, target_w), &self.device)
            .map_err(|e| candle_to_ocr_inference("UniRec", "create padded image tensor", e))?
            .to_dtype(self.dtype)
            .map_err(|e| candle_to_ocr_inference("UniRec", "cast padded image dtype", e))
    }

    /// Generate text from one or more images using greedy decoding.
    ///
    /// Supports true GPU batching when multiple images are provided.
    ///
    /// # Arguments
    /// * `images` - Input images
    /// * `max_new_tokens` - Maximum tokens to generate per image
    ///
    /// # Returns
    /// Vector of results, one per input image.
    pub fn generate(
        &self,
        images: &[RgbImage],
        max_new_tokens: usize,
    ) -> Vec<Result<String, OCRError>> {
        if images.is_empty() {
            return Vec::new();
        }

        match self.generate_internal(images, max_new_tokens) {
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

    /// Internal generation implementation supporting batched inference.
    fn generate_internal(
        &self,
        images: &[RgbImage],
        max_new_tokens: usize,
    ) -> Result<Vec<String>, OCRError> {
        let batch_size = images.len();

        // 1. Compute max dimensions across all images
        let mut max_w = 0usize;
        let mut max_h = 0usize;
        for image in images {
            let (w, h) = self.compute_target_size(image);
            max_w = max_w.max(w);
            max_h = max_h.max(h);
        }

        // 2. Preprocess all images with padding to max size
        let mut pixel_tensors: Vec<Tensor> = Vec::with_capacity(batch_size);
        for image in images {
            let tensor = self.preprocess_image_with_size(image, max_w, max_h)?;
            pixel_tensors.push(tensor);
        }

        // 3. Stack into batch tensor
        let refs: Vec<&Tensor> = pixel_tensors.iter().collect();
        let pixel_values = Tensor::cat(&refs, 0)
            .map_err(|e| candle_to_ocr_inference("UniRec", "stack pixel values", e))?;

        // 4. Encode all images
        let encoder_hidden_states = self.encoder.forward(&pixel_values)?;

        // 5. Clear KV cache before generation
        self.decoder.clear_kv_cache();

        // 6. Initialize generation state
        let mut generated_tokens: Vec<Vec<u32>> = vec![vec![self.cfg.bos_token_id]; batch_size];
        let mut finished: Vec<bool> = vec![false; batch_size];
        let mut position_offset = 0usize;

        for _ in 0..max_new_tokens {
            if finished.iter().all(|&f| f) {
                break;
            }

            // Create input tensor
            let input_ids = if position_offset == 0 {
                // First step - use BOS token for all
                Tensor::new(vec![self.cfg.bos_token_id; batch_size], &self.device)
                    .map_err(|e| candle_to_ocr_inference("UniRec", "create input_ids", e))?
                    .reshape((batch_size, 1))
                    .map_err(|e| candle_to_ocr_inference("UniRec", "reshape input_ids", e))?
            } else {
                // Subsequent steps - use last token for each sample
                let last_tokens: Vec<u32> = generated_tokens
                    .iter()
                    .map(|tokens| *tokens.last().unwrap_or(&self.cfg.bos_token_id))
                    .collect();
                Tensor::new(last_tokens, &self.device)
                    .map_err(|e| candle_to_ocr_inference("UniRec", "create input_ids", e))?
                    .reshape((batch_size, 1))
                    .map_err(|e| candle_to_ocr_inference("UniRec", "reshape input_ids", e))?
            };

            // No causal mask needed for single token decode
            let self_attn_mask = None;

            // Decoder forward
            let hidden_states = self.decoder.forward(
                &input_ids,
                &encoder_hidden_states,
                position_offset,
                self_attn_mask,
            )?;

            // Get logits for last position
            let (_, seq_len, _) = hidden_states
                .dims3()
                .map_err(|e| candle_to_ocr_inference("UniRec", "get hidden dims", e))?;
            let last_hidden = hidden_states
                .i((.., seq_len - 1, ..))
                .map_err(|e| candle_to_ocr_inference("UniRec", "select last hidden", e))?;

            let logits = self
                .lm_head
                .forward(&last_hidden)
                .map_err(|e| candle_to_ocr_inference("UniRec", "lm_head forward", e))?;

            // Greedy decoding for each sample
            let next_tokens = logits
                .argmax(D::Minus1)
                .map_err(|e| candle_to_ocr_inference("UniRec", "argmax", e))?
                .to_vec1::<u32>()
                .map_err(|e| candle_to_ocr_inference("UniRec", "to_vec1", e))?;

            for (i, &token) in next_tokens.iter().enumerate() {
                if finished[i] {
                    continue;
                }
                if token == self.cfg.eos_token_id {
                    finished[i] = true;
                } else {
                    generated_tokens[i].push(token);
                }
            }

            // Update position offset
            position_offset += 1;
        }

        // 7. Decode tokens for each sample
        let mut results = Vec::with_capacity(batch_size);
        for tokens in generated_tokens {
            // Skip BOS token
            let tokens_to_decode: Vec<u32> = if tokens.len() > 1 {
                tokens[1..].to_vec()
            } else {
                vec![]
            };

            let raw_output = self
                .tokenizer
                .decode(&tokens_to_decode, false)
                .map_err(|e| OCRError::InvalidInput {
                    message: format!("UniRec: tokenizer decode failed: {}", e),
                })?;

            results.push(postprocess_unirec_output(&raw_output));
        }

        Ok(results)
    }

    /// Get model configuration.
    pub fn config(&self) -> &UniRecConfig {
        &self.cfg
    }

    /// Get the device the model is running on.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// Postprocess UniRec output to clean up special tokens.
///
/// Matches the Python `clean_special_tokens` function from OpenOCR.
/// Note: ByteLevel decoder handles Ġ→space and Ċ→newline conversion automatically.
fn postprocess_unirec_output(text: &str) -> String {
    // Clean up special tokens (ByteLevel decoder handles Ġ and Ċ)
    let result = text
        .replace("<|bos|>", "")
        .replace("<|eos|>", "")
        .replace("<|pad|>", "")
        .replace("<|unk|>", "")
        .replace("-<|sn|>", "")
        .replace("<|sn|>", " ")
        .replace("<s>", "")
        .replace("</s>", "")
        .replace('\u{FFFF}', "");

    // Match OpenOCR's extra cleanup rules (using static regexes for efficiency).
    let result = UNDERSCORE_RE.replace_all(&result, "___");
    let result = DOTS_RE.replace_all(&result, "...");

    // Collapse repeated spaces introduced during token cleanup.
    let result = SPACES_RE.replace_all(&result, " ");

    // Trim leading/trailing whitespace
    result.trim().to_string()
}
