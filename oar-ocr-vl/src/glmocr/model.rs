use super::config::{EosTokenId, GlmOcrConfig, GlmOcrImageProcessorConfig};
use super::processing::{GlmOcrImageInputs, preprocess_image};
use super::text::GlmOcrTextModel;
use super::vision::GlmOcrVisionModel;
#[cfg(feature = "hsd")]
use crate::attention::create_tree_attention_mask;
#[cfg(feature = "hsd")]
use crate::hsd::backend_util::{commit_keep_indices, step_pos_ids, tree_pos_ids};
#[cfg(feature = "hsd")]
use crate::hsd::drafting::{
    TargetDraftAdapter, bbox_xyxy, crop_region_image, format_verified_region, map_layout_kind,
    region_markdown_for, region_markdowns_for, structure_result_to_layout_elements,
};
#[cfg(feature = "hsd")]
use crate::hsd::prefix_tree::PrefixTree;
#[cfg(feature = "hsd")]
use crate::hsd::types::{AcceptStats, Draft, HsdConfig, HsdStats, RegionStageStats};
#[cfg(feature = "hsd")]
use crate::hsd::verify::{SpecBackend, spec_decode};
use crate::utils::{
    candle_to_ocr_inference, candle_to_ocr_processing, truncate_repetitive_content,
};
#[cfg(feature = "hsd")]
use candle_core::Result as CandleResult;
use candle_core::{D, DType, Device, IndexOp, Tensor};
#[cfg(feature = "hsd")]
use candle_nn::ops as cnn_ops;
use candle_nn::{Linear, Module, VarBuilder, linear_no_bias};
use image::RgbImage;
use oar_ocr_core::core::OCRError;
#[cfg(feature = "hsd")]
use oar_ocr_core::domain::structure::{LayoutElement, LayoutElementType, StructureResult};
use std::path::Path;
#[cfg(feature = "hsd")]
use std::time::{Duration, Instant};
use tokenizers::Tokenizer;

pub struct GlmOcr {
    device: Device,
    dtype: DType,
    cfg: GlmOcrConfig,
    image_cfg: GlmOcrImageProcessorConfig,
    tokenizer: Tokenizer,
    text: GlmOcrTextModel,
    vision: GlmOcrVisionModel,
    lm_head: Linear,
    eos_token_ids: Vec<u32>,
    image_token_id: u32,
}

impl GlmOcr {
    pub fn from_dir(model_dir: impl AsRef<Path>, device: Device) -> Result<Self, OCRError> {
        let model_dir = model_dir.as_ref();
        let cfg = GlmOcrConfig::from_path(model_dir.join("config.json"))?;
        let image_cfg =
            GlmOcrImageProcessorConfig::from_path(model_dir.join("preprocessor_config.json"))?;

        let tokenizer = Tokenizer::from_file(model_dir.join("tokenizer.json")).map_err(|e| {
            OCRError::ConfigError {
                message: format!("failed to load GLM-OCR tokenizer.json: {e}"),
            }
        })?;

        if let Some(tok_image_id) = tokenizer.token_to_id("<|image|>")
            && tok_image_id != cfg.image_token_id
        {
            return Err(OCRError::ConfigError {
                message: format!(
                    "GLM-OCR image_token_id mismatch: tokenizer {tok_image_id} != config {}",
                    cfg.image_token_id
                ),
            });
        }

        let dtype = device.bf16_default_to_f32();
        // SAFETY: The mmap'd file must not be modified or deleted while in use.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(
                &[model_dir.join("model.safetensors")],
                dtype,
                &device,
            )
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "load model.safetensors", e))?
        };

        let image_token_id = cfg.image_token_id;
        let text = GlmOcrTextModel::load(&cfg.text_config, vb.pp("model").pp("language_model"))?;
        let vision = GlmOcrVisionModel::load(&cfg.vision_config, vb.pp("model").pp("visual"))?;

        let lm_head = if cfg.text_config.tie_word_embeddings || cfg.tie_word_embeddings {
            Linear::new(text.token_embedding_weight(), None)
        } else {
            linear_no_bias(
                cfg.text_config.hidden_size,
                cfg.text_config.vocab_size,
                vb.pp("lm_head"),
            )
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "load lm_head", e))?
        };

        let eos_token_ids = match &cfg.text_config.eos_token_id {
            EosTokenId::Single(v) => vec![*v],
            EosTokenId::Multiple(v) => v.clone(),
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
            eos_token_ids,
            image_token_id,
        })
    }

    /// Generate OCR output for one or more images with optional instructions.
    ///
    /// # Arguments
    /// * `images` - Input images
    /// * `instructions` - Instruction for each image (must match images length)
    /// * `max_new_tokens` - Maximum tokens to generate per image
    ///
    /// # Returns
    /// Vector of results, one per input image.
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
                    "GLM-OCR: images count ({}) != instructions count ({})",
                    images.len(),
                    instructions.len()
                ),
            })];
        }

        match self.generate_tokens_internal(images, instructions, max_new_tokens) {
            Ok(results) => results
                .into_iter()
                .map(|tokens| self.decode_generated_tokens(&tokens))
                .collect(),
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

    /// Generate raw baseline tokens for oracle-draft / tokenizer round-trip
    /// experiments. Tokens are exactly the ids emitted by the decode loop,
    /// excluding stop tokens, before tokenizer decoding or repetition
    /// truncation.
    pub fn generate_tokens(
        &self,
        images: &[RgbImage],
        instructions: &[impl AsRef<str>],
        max_new_tokens: usize,
    ) -> Vec<Result<Vec<u32>, OCRError>> {
        if images.is_empty() {
            return Vec::new();
        }
        if images.len() != instructions.len() {
            return vec![Err(OCRError::InvalidInput {
                message: format!(
                    "GLM-OCR: images count ({}) != instructions count ({})",
                    images.len(),
                    instructions.len()
                ),
            })];
        }

        match self.generate_tokens_internal(images, instructions, max_new_tokens) {
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

    fn generate_tokens_internal(
        &self,
        images: &[RgbImage],
        instructions: &[impl AsRef<str>],
        max_new_tokens: usize,
    ) -> Result<Vec<Vec<u32>>, OCRError> {
        let mut results = Vec::with_capacity(images.len());

        for (image, instruction) in images.iter().zip(instructions.iter()) {
            let image_inputs = preprocess_image(
                image,
                &self.image_cfg,
                &self.cfg.vision_config,
                &self.device,
                self.dtype,
            )?;

            let prompt = build_prompt(instruction.as_ref());
            let prompt = expand_image_tokens(&prompt, image_inputs.num_image_tokens)?;

            let enc = self
                .tokenizer
                .encode(prompt, false)
                .map_err(|e| OCRError::InvalidInput {
                    message: format!("GLM-OCR: tokenizer encode failed: {e}"),
                })?;
            let input_ids = enc.get_ids().to_vec();
            if input_ids.is_empty() {
                return Err(OCRError::InvalidInput {
                    message: "GLM-OCR: empty prompt after tokenization".to_string(),
                });
            }

            let seq_len = input_ids.len();
            let inputs_embeds = self.prepare_inputs(&input_ids, &image_inputs)?;
            let (position_ids, max_pos) = build_position_ids(
                &input_ids,
                image_inputs.grid_thw,
                self.cfg.vision_config.spatial_merge_size,
                self.image_token_id,
                &self.device,
            )?;
            let rope_delta = max_pos + 1 - seq_len as i64;

            self.text.clear_kv_cache();
            let hidden = self.text.forward(&inputs_embeds, &position_ids, None)?;
            let last = hidden.i((0, seq_len - 1, ..)).map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: get last hidden",
                    e,
                )
            })?;
            let mut logits = self.logits_from_hidden(&last)?;

            let mut generated: Vec<u32> = Vec::new();

            for (pos, _) in (seq_len as i64..).zip(0..max_new_tokens) {
                let tok = logits
                    .argmax(D::Minus1)
                    .and_then(|t| t.to_scalar::<u32>())
                    .map_err(|e| {
                        candle_to_ocr_processing(
                            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                            "GLM-OCR: argmax",
                            e,
                        )
                    })?;
                if self.eos_token_ids.contains(&tok) {
                    break;
                }
                generated.push(tok);

                let token = Tensor::new(&[tok], &self.device)
                    .and_then(|t| t.reshape((1, 1)))
                    .map_err(|e| {
                        candle_to_ocr_processing(
                            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                            "GLM-OCR: create next token",
                            e,
                        )
                    })?;
                let embed = self.text.embed(&token)?;

                let pos_val = pos + rope_delta;
                let pos_ids = Tensor::new(vec![pos_val, pos_val, pos_val], &self.device)
                    .and_then(|t| t.reshape((3, 1, 1)))
                    .map_err(|e| {
                        candle_to_ocr_processing(
                            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                            "GLM-OCR: create position ids",
                            e,
                        )
                    })?;

                let hs = self.text.forward(&embed, &pos_ids, None)?;
                let last = hs.i((0, 0, ..)).map_err(|e| {
                    candle_to_ocr_processing(
                        oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                        "GLM-OCR: get decode hidden",
                        e,
                    )
                })?;
                logits = self.logits_from_hidden(&last)?;
            }

            results.push(generated);
        }

        Ok(results)
    }

    pub fn decode_tokens(&self, tokens: &[u32]) -> Result<String, OCRError> {
        self.decode_generated_tokens(tokens)
    }

    /// Decode tokens **without** applying GLM-OCR's repetition-collapse
    /// post-process. Use this when feeding GLM-OCR output as a draft to
    /// another target VLM — DSV matches at token granularity, and any
    /// repetition collapse on the source side will byte-mismatch the target's
    /// natural output, destroying acceptance length.
    pub fn decode_tokens_raw(&self, tokens: &[u32]) -> Result<String, OCRError> {
        let decoded = self
            .tokenizer
            .decode(tokens, true)
            .map_err(|e| OCRError::InvalidInput {
                message: format!("GLM-OCR: tokenizer decode failed: {e}"),
            })?;
        Ok(decoded.trim().to_string())
    }

    pub fn tokenizer(&self) -> &Tokenizer {
        &self.tokenizer
    }

    fn decode_generated_tokens(&self, tokens: &[u32]) -> Result<String, OCRError> {
        let raw = self.decode_tokens_raw(tokens)?;
        let truncated = truncate_repetitive_content(&raw, 10, 10, 10);
        Ok(truncated.trim().to_string())
    }

    fn prepare_inputs(
        &self,
        input_ids: &[u32],
        image_inputs: &GlmOcrImageInputs,
    ) -> Result<Tensor, OCRError> {
        let seq_len = input_ids.len();
        let token_ids =
            Tensor::from_vec(input_ids.to_vec(), (1, seq_len), &self.device).map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: create input_ids tensor",
                    e,
                )
            })?;
        let mut embeds = self.text.embed(&token_ids)?;

        let image_embeds = self
            .vision
            .forward(&image_inputs.pixel_values, image_inputs.grid_thw)?;
        let image_embeds = image_embeds.to_dtype(self.dtype).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: cast image_embeds",
                e,
            )
        })?;

        let indices: Vec<usize> = input_ids
            .iter()
            .enumerate()
            .filter_map(|(i, &id)| {
                if id == self.image_token_id {
                    Some(i)
                } else {
                    None
                }
            })
            .collect();
        let image_embeds_len = image_embeds.dim(0).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: image_embeds dim",
                e,
            )
        })?;
        if indices.len() != image_embeds_len {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "GLM-OCR: image token count ({}) != image embeds ({})",
                    indices.len(),
                    image_embeds_len
                ),
            });
        }
        if indices.is_empty() {
            return Err(OCRError::InvalidInput {
                message: "GLM-OCR: no image tokens found in prompt".to_string(),
            });
        }
        let start = indices[0];
        let end = *indices.last().unwrap();
        if end + 1 - start != indices.len() {
            return Err(OCRError::InvalidInput {
                message: "GLM-OCR: image tokens are not contiguous in prompt".to_string(),
            });
        }

        let hidden_size = embeds.dim(2).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: embed hidden_size",
                e,
            )
        })?;

        let prefix = if start > 0 {
            embeds.narrow(1, 0, start).map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: embed prefix narrow",
                    e,
                )
            })?
        } else {
            Tensor::zeros((1, 0, hidden_size), embeds.dtype(), embeds.device()).map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: embed prefix zeros",
                    e,
                )
            })?
        };

        let suffix_start = end + 1;
        let suffix = if suffix_start < seq_len {
            embeds
                .narrow(1, suffix_start, seq_len - suffix_start)
                .map_err(|e| {
                    candle_to_ocr_processing(
                        oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                        "GLM-OCR: embed suffix narrow",
                        e,
                    )
                })?
        } else {
            Tensor::zeros((1, 0, hidden_size), embeds.dtype(), embeds.device()).map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: embed suffix zeros",
                    e,
                )
            })?
        };

        let image_embeds = image_embeds.unsqueeze(0).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: image embeds unsqueeze",
                e,
            )
        })?;
        embeds = Tensor::cat(&[&prefix, &image_embeds, &suffix], 1).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: embed cat",
                e,
            )
        })?;

        Ok(embeds)
    }

    /// Hierarchical Speculative Decoding entry for a single image / region.
    ///
    /// Use `generate_hsd_full` for the two-stage document flow.
    #[cfg(feature = "hsd")]
    pub fn generate_hsd(
        &self,
        image: &RgbImage,
        instruction: &str,
        drafts: &[String],
        hsd_cfg: &HsdConfig,
    ) -> Result<(String, HsdStats), OCRError> {
        let t_drafter = Instant::now();
        let tokenized = self.tokenize_drafts(drafts)?;
        self.generate_hsd_tokenized(
            image,
            instruction,
            &tokenized,
            hsd_cfg,
            hsd_cfg.max_region_tokens,
            t_drafter.elapsed(),
        )
    }

    /// HSD entry that consumes already-tokenized drafts. This is the oracle
    /// path used by benchmarks to avoid `decode -> encode` tokenizer
    /// round-trips when the draft comes from this backend's own baseline.
    #[cfg(feature = "hsd")]
    pub fn generate_hsd_with_token_drafts(
        &self,
        image: &RgbImage,
        instruction: &str,
        drafts: &[Draft],
        hsd_cfg: &HsdConfig,
    ) -> Result<(String, HsdStats), OCRError> {
        self.generate_hsd_tokenized(
            image,
            instruction,
            drafts,
            hsd_cfg,
            hsd_cfg.max_region_tokens,
            Duration::ZERO,
        )
    }

    #[cfg(feature = "hsd")]
    fn tokenize_drafts(&self, drafts: &[String]) -> Result<Vec<Draft>, OCRError> {
        let mut tokenized: Vec<Draft> = Vec::with_capacity(drafts.len());
        for d in drafts {
            if d.trim().is_empty() {
                continue;
            }
            let enc =
                self.tokenizer
                    .encode(d.as_str(), false)
                    .map_err(|e| OCRError::InvalidInput {
                        message: format!("GLM-OCR HSD: tokenizer encode failed: {e}"),
                    })?;
            let tokens = enc.get_ids().to_vec();
            if !tokens.is_empty() {
                tokenized.push(Draft::new(tokens));
            }
        }
        Ok(tokenized)
    }

    #[cfg(feature = "hsd")]
    fn generate_hsd_tokenized(
        &self,
        image: &RgbImage,
        instruction: &str,
        tokenized: &[Draft],
        hsd_cfg: &HsdConfig,
        max_new_tokens: usize,
        drafter_elapsed: Duration,
    ) -> Result<(String, HsdStats), OCRError> {
        if !self.device.is_cuda() {
            return Err(OCRError::ConfigError {
                message: "HSD requires CUDA device".to_string(),
            });
        }

        let mut stats = HsdStats {
            drafter: drafter_elapsed,
            ..Default::default()
        };
        let t_pre = Instant::now();
        let (initial_lp, rope_delta) = self.hsd_prefill_single(image, instruction)?;
        stats.stage2.vision_prefill = t_pre.elapsed();
        stats.stage2.forward_passes = 1;

        let t_dec = Instant::now();
        let mut backend = GlmOcrSpecBackend::new(self, rope_delta);
        let mut accept = AcceptStats::default();
        let mut dsv = Default::default();
        let generated = spec_decode(
            &mut backend,
            tokenized,
            initial_lp,
            max_new_tokens,
            &hsd_cfg.dsv,
            &mut accept,
            &mut dsv,
        )
        .map_err(|e| candle_to_ocr_inference("GLM-OCR", "spec_decode", e))?;
        stats.stage2.decode = t_dec.elapsed();
        stats.stage2.emitted_tokens = generated.len() as u32;
        stats.stage2.accept = accept;
        stats.stage2.dsv = dsv;
        stats.stage2.forward_passes += backend.forward_passes;

        // Strip the first stop token and anything after it before decoding.
        let stop_pos = generated
            .iter()
            .position(|t| self.eos_token_ids.contains(t))
            .unwrap_or(generated.len());
        let trimmed = &generated[..stop_pos];

        let decoded = self
            .tokenizer
            .decode(trimmed, true)
            .map_err(|e| OCRError::InvalidInput {
                message: format!("GLM-OCR HSD: tokenizer decode failed: {e}"),
            })?;
        let decoded = truncate_repetitive_content(&decoded, 10, 10, 10);
        Ok((decoded.trim().to_string(), stats))
    }

    /// Run the full two-stage HSD: Stage 1 verifies each layout-detected
    /// region against the layout drafter's text, then Stage 2 (gated by
    /// `hsd_cfg.enable_stage2`) verifies the Stage-1-aggregated markdown on
    /// the full image with `hsd_cfg.max_page_tokens` budget.
    ///
    /// - `enable_stage1 = false`: skip per-region verification; build the
    ///   Stage 2 draft set directly from the layout drafter's per-element
    ///   markdowns (`region_markdowns`). Mirrors the paper's Table 8
    ///   "Page-level Spec. Decoding only" ablation.
    /// - `enable_stage2 = false`: return the Stage-1-only aggregation (lossy
    ///   ablation matching paper Table 8).
    ///
    /// `region_instruction` is used only for Stage 1 crop verification;
    /// `page_instruction` is used for Stage 2 full-page verification.
    #[cfg(feature = "hsd")]
    pub fn generate_hsd_full(
        &self,
        image: &RgbImage,
        elements: &[LayoutElement],
        ignore_labels: &[String],
        page_instruction: &str,
        region_instruction: &str,
        hsd_cfg: &HsdConfig,
    ) -> Result<(String, HsdStats), OCRError> {
        let mut stats = HsdStats::default();
        let mut region_md: Vec<(usize, String)> = Vec::with_capacity(elements.len());

        if hsd_cfg.enable_stage1 {
            for (idx, elem) in elements.iter().enumerate() {
                if let Some(label) = &elem.label
                    && ignore_labels.iter().any(|l| l == label)
                {
                    continue;
                }
                if matches!(
                    elem.element_type,
                    LayoutElementType::Image
                        | LayoutElementType::HeaderImage
                        | LayoutElementType::FooterImage
                        | LayoutElementType::Seal
                ) {
                    continue;
                }
                let draft = region_markdown_for(elem, TargetDraftAdapter::GlmOcr);
                if draft.trim().is_empty() {
                    continue;
                }

                let bbox = bbox_xyxy(&elem.bbox);
                let crop = crop_region_image(image, &bbox)?;
                let drafts = vec![draft];
                let (region_text, region_stats) =
                    self.generate_hsd(&crop, region_instruction, &drafts, hsd_cfg)?;
                stats.drafter += region_stats.drafter;

                let kind = map_layout_kind(elem.element_type);
                stats.stage1_regions.push(RegionStageStats {
                    kind,
                    stats: region_stats.stage2.clone(),
                });
                stats.stage1.add_assign(region_stats.stage2);
                let order = elem.order_index.map(|x| x as usize).unwrap_or(idx);
                region_md.push((order, format_verified_region(&region_text, kind)));
            }
        }

        region_md.sort_by_key(|(order, _)| *order);
        let region_md: Vec<String> = region_md
            .into_iter()
            .map(|(_, text)| text)
            .filter(|s| !s.trim().is_empty())
            .collect();

        // Stage 2 — page-level global verification on the full image. Per
        // paper Eq. 3 the page draft is the *unordered set* `Ỹ^pg = {ŷ^(i)}`,
        // one draft per region. We pass the Vec straight to `spec_decode`
        // instead of pre-joining: `collect_candidates` scans each draft
        // independently (Eqs. 1+2), so per-region n-gram locality is
        // preserved even when full-page transitions don't appear naturally
        // in the target VLM's output. Budget = `max_page_tokens`.
        if hsd_cfg.enable_stage2 {
            let t_drafter = Instant::now();
            let page_drafts: Vec<String> = if !region_md.is_empty() {
                region_md.clone()
            } else {
                region_markdowns_for(elements, ignore_labels, TargetDraftAdapter::GlmOcr)
            };
            if !page_drafts.is_empty() {
                let tokenized = self.tokenize_drafts(&page_drafts)?;
                let (text, s2_stats) = self.generate_hsd_tokenized(
                    image,
                    page_instruction,
                    &tokenized,
                    hsd_cfg,
                    hsd_cfg.max_page_tokens,
                    t_drafter.elapsed(),
                )?;
                stats.stage2 = s2_stats.stage2;
                stats.drafter += s2_stats.drafter;
                return Ok((text, stats));
            }
        }

        // Stage 2 disabled or no draft to verify — return Stage-1-only join
        // as a human-readable fallback. The `\n\n` separator here is for the
        // *output* (caller-facing), not for any further HSD input.
        Ok((region_md.join("\n\n"), stats))
    }

    /// One-call HSD entry that consumes a `StructureResult` (the output of
    /// the OARStructure / PP-StructureV3 pipeline) directly.
    ///
    /// Backfills table HTML / formula LaTeX via
    /// [`structure_result_to_layout_elements`] then delegates to
    /// [`Self::generate_hsd_full`]. See the HunyuanOCR sibling for the
    /// full design discussion — GLM-OCR keeps single-draft-per-region semantics
    /// (its public Recognition prompts emit one canonical output).
    #[cfg(feature = "hsd")]
    pub fn generate_hsd_with_structure(
        &self,
        image: &RgbImage,
        page_instruction: &str,
        region_instruction: &str,
        structure: &StructureResult,
        ignore_labels: &[String],
        hsd_cfg: &HsdConfig,
    ) -> Result<(String, HsdStats), OCRError> {
        let elements = structure_result_to_layout_elements(structure);
        self.generate_hsd_full(
            image,
            &elements,
            ignore_labels,
            page_instruction,
            region_instruction,
            hsd_cfg,
        )
    }

    /// Run a single-image prefill with the supplied instruction. Returns
    /// the F32 last-position log-probabilities and the MRoPE delta.
    #[cfg(feature = "hsd")]
    fn hsd_prefill_single(
        &self,
        image: &RgbImage,
        instruction: &str,
    ) -> Result<(Tensor, i64), OCRError> {
        let image_inputs = preprocess_image(
            image,
            &self.image_cfg,
            &self.cfg.vision_config,
            &self.device,
            self.dtype,
        )?;

        let prompt = build_prompt(instruction);
        let prompt = expand_image_tokens(&prompt, image_inputs.num_image_tokens)?;
        let enc = self
            .tokenizer
            .encode(prompt, false)
            .map_err(|e| OCRError::InvalidInput {
                message: format!("GLM-OCR HSD: tokenizer encode failed: {e}"),
            })?;
        let input_ids = enc.get_ids().to_vec();
        if input_ids.is_empty() {
            return Err(OCRError::InvalidInput {
                message: "GLM-OCR HSD: empty prompt after tokenization".to_string(),
            });
        }
        let seq_len = input_ids.len();

        let inputs_embeds = self.prepare_inputs(&input_ids, &image_inputs)?;
        let (position_ids, max_pos) = build_position_ids(
            &input_ids,
            image_inputs.grid_thw,
            self.cfg.vision_config.spatial_merge_size,
            self.image_token_id,
            &self.device,
        )?;
        let rope_delta = max_pos + 1 - seq_len as i64;

        // GLM-OCR's existing prefill calls `text.forward(..., None)` — the
        // implementation builds its own internal causal mask via the rotary
        // path. Match that behaviour here.
        self.text.clear_kv_cache();
        let hidden = self.text.forward(&inputs_embeds, &position_ids, None)?;
        let last = hidden
            .i((0, seq_len - 1, ..))
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "get last hidden", e))?;
        let logits = self.logits_from_hidden(&last)?;
        let lp = cnn_ops::log_softmax(
            &logits
                .to_dtype(DType::F32)
                .map_err(|e| candle_to_ocr_inference("GLM-OCR", "logits to f32", e))?,
            D::Minus1,
        )
        .map_err(|e| candle_to_ocr_inference("GLM-OCR", "log_softmax prefill", e))?;
        Ok((lp, rope_delta))
    }

    fn logits_from_hidden(&self, hidden: &Tensor) -> Result<Tensor, OCRError> {
        let hidden = hidden.unsqueeze(0).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: logits unsqueeze",
                e,
            )
        })?;
        let logits = self
            .lm_head
            .forward(&hidden)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "lm_head forward", e))?;
        logits.squeeze(0).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: logits squeeze",
                e,
            )
        })
    }
}

/// HSD adapter for GLM-OCR. 3-axis MRoPE, independent lm_head, rope_delta
/// captured at prefill (same shape as MinerU / PaddleOCR-VL).
#[cfg(feature = "hsd")]
struct GlmOcrSpecBackend<'a> {
    model: &'a GlmOcr,
    rope_delta: i64,
    pre_verify_kv: usize,
    forward_passes: u32,
}

#[cfg(feature = "hsd")]
impl<'a> GlmOcrSpecBackend<'a> {
    fn new(model: &'a GlmOcr, rope_delta: i64) -> Self {
        Self {
            model,
            rope_delta,
            pre_verify_kv: 0,
            forward_passes: 0,
        }
    }

    fn project_logprobs_2d(&self, hidden_2d: &Tensor) -> CandleResult<Tensor> {
        // (N, hidden) → (N, vocab) → log-softmax F32.
        // GLM-OCR's lm_head expects shape (..., hidden); 2D works directly.
        let logits = self.model.lm_head.forward(hidden_2d)?;
        cnn_ops::log_softmax(&logits.to_dtype(DType::F32)?, D::Minus1)
    }

    fn project_logprobs_1d(&self, hidden_1d: &Tensor) -> CandleResult<Tensor> {
        // lm_head is a Linear that requires ≥ 2-D input.
        let logits = self
            .model
            .lm_head
            .forward(&hidden_1d.unsqueeze(0)?)?
            .squeeze(0)?;
        cnn_ops::log_softmax(&logits.to_dtype(DType::F32)?, D::Minus1)
    }
}

#[cfg(feature = "hsd")]
impl<'a> SpecBackend for GlmOcrSpecBackend<'a> {
    fn step_one(&mut self, token: u32) -> CandleResult<Tensor> {
        let model = self.model;
        let device = &model.device;

        let tok_t = Tensor::new(vec![token], device)?.reshape((1usize, 1usize))?;
        let embeds = model
            .text
            .embed(&tok_t)
            .map_err(|e| candle_core::Error::Msg(format!("GLM-OCR HSD step_one embed: {e}")))?;

        let pos_ids = step_pos_ids(3, model.text.current_kv_len(), self.rope_delta, device)?;

        let hidden = model
            .text
            .forward(&embeds, &pos_ids, None)
            .map_err(|e| candle_core::Error::Msg(format!("GLM-OCR HSD step_one forward: {e}")))?;
        self.forward_passes += 1;
        let last = hidden.i((0, 0, ..))?;
        self.project_logprobs_1d(&last)
    }

    fn verify_tree(&mut self, tree: &PrefixTree) -> CandleResult<Tensor> {
        let n = tree.num_nodes();
        let model = self.model;
        let device = &model.device;
        let dtype = model.dtype;

        let prefix_kv = model.text.current_kv_len();
        self.pre_verify_kv = prefix_kv;

        let tok_t = Tensor::new(tree.tokens.clone(), device)?.reshape((1usize, n))?;
        let embeds = model
            .text
            .embed(&tok_t)
            .map_err(|e| candle_core::Error::Msg(format!("GLM-OCR HSD verify_tree embed: {e}")))?;

        let pos_ids = tree_pos_ids(3, prefix_kv, self.rope_delta, tree, device)?;
        let mask = create_tree_attention_mask(&tree.parents, prefix_kv, dtype, device)?;

        let hidden = model
            .text
            .forward(&embeds, &pos_ids, Some(&mask))
            .map_err(|e| {
                candle_core::Error::Msg(format!("GLM-OCR HSD verify_tree forward: {e}"))
            })?;
        self.forward_passes += 1;
        let h2 = hidden.squeeze(0)?;
        self.project_logprobs_2d(&h2)
    }

    fn commit_verify(&mut self, accepted_path: &[usize]) -> CandleResult<()> {
        let indices = commit_keep_indices(self.pre_verify_kv, accepted_path);
        self.model
            .text
            .keep_kv_indices(&indices)
            .map_err(|e| candle_core::Error::Msg(format!("GLM-OCR HSD commit_verify: {e}")))
    }

    fn is_eos(&self, tok: u32) -> bool {
        self.model.eos_token_ids.contains(&tok)
    }
}

fn build_prompt(instruction: &str) -> String {
    format!(
        "[gMASK]<sop><|user|>\n<|begin_of_image|><|image|><|end_of_image|>{instruction}<|assistant|>\n"
    )
}

fn expand_image_tokens(prompt: &str, num_image_tokens: usize) -> Result<String, OCRError> {
    if num_image_tokens == 0 {
        return Err(OCRError::InvalidInput {
            message: "GLM-OCR: num_image_tokens is zero".to_string(),
        });
    }
    if !prompt.contains("<|image|>") {
        return Err(OCRError::InvalidInput {
            message: "GLM-OCR: prompt missing <|image|> token".to_string(),
        });
    }
    let placeholder = "<|placeholder|>";
    let repeated = placeholder.repeat(num_image_tokens);
    let expanded = prompt.replacen("<|image|>", &repeated, 1);
    Ok(expanded.replace(placeholder, "<|image|>"))
}

/// Build position IDs tensor and compute max position value.
///
/// Returns `(position_ids, max_pos)` where `max_pos` is used to compute rope_delta.
fn build_position_ids(
    input_ids: &[u32],
    grid_thw: (usize, usize, usize),
    merge_size: usize,
    image_token_id: u32,
    device: &Device,
) -> Result<(Tensor, i64), OCRError> {
    let (mut pos_t, mut pos_h, mut pos_w) = (Vec::new(), Vec::new(), Vec::new());
    let mut max_pos: i64 = -1;

    let mut current = input_ids.first().copied().unwrap_or(image_token_id) == image_token_id;
    let mut start = 0usize;
    let mut groups = Vec::new();
    for (i, &id) in input_ids.iter().enumerate() {
        let is_image = id == image_token_id;
        if i == 0 {
            current = is_image;
            continue;
        }
        if is_image != current {
            groups.push((current, start, i));
            start = i;
            current = is_image;
        }
    }
    groups.push((current, start, input_ids.len()));

    let (grid_t, grid_h, grid_w) = grid_thw;
    let llm_grid_h = grid_h / merge_size;
    let llm_grid_w = grid_w / merge_size;
    let llm_grid_t = grid_t;
    let mut seen_image = false;

    let expected_image_tokens = llm_grid_t * llm_grid_h * llm_grid_w;
    for (is_image, s, e) in groups {
        let st_idx = max_pos + 1;
        if is_image {
            if seen_image {
                return Err(OCRError::InvalidInput {
                    message: "GLM-OCR: multiple image groups in prompt are not supported"
                        .to_string(),
                });
            }
            seen_image = true;
            let group_len = e - s;
            if group_len != expected_image_tokens {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "GLM-OCR: image token count ({group_len}) != expected ({expected_image_tokens})"
                    ),
                });
            }
            for t in 0..llm_grid_t {
                for h in 0..llm_grid_h {
                    for w in 0..llm_grid_w {
                        pos_t.push(t as i64 + st_idx);
                        pos_h.push(h as i64 + st_idx);
                        pos_w.push(w as i64 + st_idx);
                    }
                }
            }
            let group_max = (llm_grid_t.max(llm_grid_h).max(llm_grid_w) as i64) - 1 + st_idx;
            max_pos = group_max;
        } else {
            let len = e - s;
            for i in 0..len {
                let v = i as i64 + st_idx;
                pos_t.push(v);
                pos_h.push(v);
                pos_w.push(v);
            }
            max_pos = st_idx + len as i64 - 1;
        }
    }

    let seq_len = input_ids.len();
    if pos_t.len() != seq_len {
        return Err(OCRError::InvalidInput {
            message: format!(
                "GLM-OCR: position_ids length mismatch: {} vs {}",
                pos_t.len(),
                seq_len
            ),
        });
    }

    let mut data = Vec::with_capacity(3 * seq_len);
    data.extend(pos_t);
    data.extend(pos_h);
    data.extend(pos_w);

    let position_ids = Tensor::from_vec(data, (3, 1, seq_len), device).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: position_ids tensor",
            e,
        )
    })?;
    Ok((position_ids, max_pos))
}
