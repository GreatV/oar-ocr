//! UniRec Vision-Language model for unified text, formula, and table recognition.

use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Linear, Module};
use image::RgbImage;
use regex::Regex;
use std::path::Path;
use tokenizers::Tokenizer;
use tokenizers::decoders::byte_level::ByteLevel;

use super::config::UniRecConfig;
use super::decoder::{KvCache, M2M100Decoder, create_causal_mask};
use super::encoder::FocalSVTR;
use crate::utils::candle_to_ocr_inference;
use oar_ocr_core::core::OCRError;

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

    /// Preprocess an image for the model.
    fn preprocess_image(&self, image: &RgbImage) -> Result<Tensor, OCRError> {
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

        // Resize image.
        //
        // OpenOCR uses torchvision/PIL BICUBIC interpolation for `NaSizeResize`.
        // `image::imageops::FilterType::CatmullRom` is the closest equivalent.
        let resized = image::imageops::resize(
            image,
            target_w as u32,
            target_h as u32,
            image::imageops::FilterType::CatmullRom,
        );

        // Convert to tensor with normalization: (pixel / 255 - 0.5) / 0.5
        let mut data = Vec::with_capacity(3 * target_h * target_w);

        // CHW format
        for c in 0..3 {
            for y in 0..target_h {
                for x in 0..target_w {
                    let pixel = resized.get_pixel(x as u32, y as u32);
                    let val = pixel[c] as f32 / 255.0;
                    let normalized = (val - 0.5) / 0.5;
                    data.push(normalized);
                }
            }
        }

        Tensor::from_vec(data, (1, 3, target_h, target_w), &self.device)
            .map_err(|e| candle_to_ocr_inference("UniRec", "create image tensor", e))?
            .to_dtype(self.dtype)
            .map_err(|e| candle_to_ocr_inference("UniRec", "cast image tensor dtype", e))
    }

    /// Generate text from an image using greedy decoding.
    pub fn generate(&self, image: RgbImage, max_new_tokens: usize) -> Result<String, OCRError> {
        // Preprocess image
        let pixel_values = self.preprocess_image(&image)?;

        // Encode image
        let encoder_hidden_states = self.encoder.forward(&pixel_values)?;

        // Initialize KV cache
        let mut kv_cache: Vec<KvCache> = (0..self.cfg.decoder_layers)
            .map(|_| KvCache::default())
            .collect();

        // Start with BOS token
        let mut generated_tokens: Vec<u32> = vec![self.cfg.bos_token_id];
        let mut position_offset = 0usize;

        for _ in 0..max_new_tokens {
            let current_len = generated_tokens.len();

            // Create input tensor (only last token for cached inference)
            let input_ids = if position_offset == 0 {
                // First step - use all tokens
                Tensor::new(&generated_tokens[..], &self.device)
                    .map_err(|e| candle_to_ocr_inference("UniRec", "create input_ids", e))?
                    .unsqueeze(0)
                    .map_err(|e| candle_to_ocr_inference("UniRec", "unsqueeze input_ids", e))?
            } else {
                // Subsequent steps - only use last token
                let last_token =
                    generated_tokens
                        .last()
                        .copied()
                        .ok_or_else(|| OCRError::InvalidInput {
                            message: "UniRec: generated_tokens is empty".to_string(),
                        })?;
                Tensor::new(&[last_token], &self.device)
                    .map_err(|e| candle_to_ocr_inference("UniRec", "create input_id", e))?
                    .unsqueeze(0)
                    .map_err(|e| candle_to_ocr_inference("UniRec", "unsqueeze input_id", e))?
            };

            // Create causal mask (only for first step with multiple tokens)
            let self_attn_mask = if position_offset == 0 && current_len > 1 {
                Some(create_causal_mask(current_len, &self.device, self.dtype)?)
            } else {
                None
            };

            // Decoder forward
            let hidden_states = self.decoder.forward(
                &input_ids,
                &encoder_hidden_states,
                position_offset,
                self_attn_mask.as_ref(),
                Some(&mut kv_cache),
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

            // Greedy decoding - select argmax
            let argmax_result = logits
                .argmax(D::Minus1)
                .map_err(|e| candle_to_ocr_inference("UniRec", "argmax", e))?;

            // Extract next token - flatten_all() handles all shapes uniformly
            let next_token = argmax_result
                .flatten_all()
                .map_err(|e| candle_to_ocr_inference("UniRec", "flatten argmax", e))?
                .get(0)
                .map_err(|e| candle_to_ocr_inference("UniRec", "get first token", e))?
                .to_scalar::<u32>()
                .map_err(|e| candle_to_ocr_inference("UniRec", "to_scalar", e))?;

            // Check for EOS
            if next_token == self.cfg.eos_token_id {
                break;
            }

            generated_tokens.push(next_token);

            // Update position offset
            if position_offset == 0 {
                position_offset = current_len;
            } else {
                position_offset += 1;
            }
        }

        // Decode tokens (skip BOS token)
        let tokens_to_decode: Vec<u32> = if generated_tokens.len() > 1 {
            generated_tokens[1..].to_vec()
        } else {
            vec![]
        };

        // Decode with skip_special_tokens=false to preserve LaTeX tokens
        // The tokenizer treats many LaTeX commands as "special" tokens
        let raw_output = self
            .tokenizer
            .decode(&tokens_to_decode, false)
            .map_err(|e| OCRError::InvalidInput {
                message: format!("UniRec: tokenizer decode failed: {}", e),
            })?;

        // Postprocess: clean up special tokens and GPT-style markers
        Ok(postprocess_unirec_output(&raw_output))
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

    // Match OpenOCR's extra cleanup rules.
    let underscore_re = Regex::new(r"_{4,}").expect("static underscore regex must compile");
    let dots_re = Regex::new(r"\.{4,}").expect("static dots regex must compile");
    let result = underscore_re.replace_all(&result, "___");
    let result = dots_re.replace_all(&result, "...");

    // Collapse repeated spaces introduced during token cleanup.
    let spaces_re = Regex::new(r"[ ]{2,}").expect("static spaces regex must compile");
    let result = spaces_re.replace_all(&result, " ");

    // Trim leading/trailing whitespace
    result.trim().to_string()
}
