use super::config::HunyuanOcrVisionConfig;
use crate::utils::{candle_to_ocr_inference, candle_to_ocr_processing};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, LayerNormConfig, Linear, Module};
use oar_ocr_core::core::OCRError;
use std::collections::BTreeSet;

#[derive(Debug, Clone)]
struct VisionEmbeddings {
    patch_embedding: Conv2d,
    position_embedding: candle_nn::Embedding,
}

impl VisionEmbeddings {
    fn load(cfg: &HunyuanOcrVisionConfig, vb: candle_nn::VarBuilder) -> Result<Self, OCRError> {
        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            padding: 0,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };

        let patch_embedding = if cfg.add_patchemb_bias {
            candle_nn::conv2d(
                cfg.num_channels,
                cfg.hidden_size,
                cfg.patch_size,
                conv_cfg,
                vb.pp("patch_embedding"),
            )
        } else {
            candle_nn::conv2d_no_bias(
                cfg.num_channels,
                cfg.hidden_size,
                cfg.patch_size,
                conv_cfg,
                vb.pp("patch_embedding"),
            )
        }
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load vit patch_embedding", e))?;

        let num_positions = cfg.max_vit_seq_len + cfg.cat_extra_token;
        let position_embedding =
            candle_nn::embedding(num_positions, cfg.hidden_size, vb.pp("position_embedding"))
                .map_err(|e| {
                    candle_to_ocr_inference("HunyuanOCR", "load vit position_embedding", e)
                })?;

        Ok(Self {
            patch_embedding,
            position_embedding,
        })
    }

    fn interpolate_patch_pos(
        &self,
        height: usize,
        width: usize,
        device: &Device,
        dtype: DType,
    ) -> Result<Tensor, OCRError> {
        if height == 0 || width == 0 {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "HunyuanOCR: vit interpolate_patch_pos requires height/width > 0, got {height}x{width}"
                ),
            });
        }

        let pos_w = self.position_embedding.embeddings();
        let (num_positions, dim) = pos_w.dims2().map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: vit position_embedding dims2 failed",
                e,
            )
        })?;
        if num_positions < 2 {
            return Err(OCRError::ConfigError {
                message: format!(
                    "HunyuanOCR: vit position_embedding is too small: {num_positions}"
                ),
            });
        }

        let patch_pos = pos_w.i((1..num_positions, ..)).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: vit slice patch position embedding failed",
                e,
            )
        })?;
        let (num_patch_positions, _) = patch_pos.dims2().map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: vit patch position dims2 failed",
                e,
            )
        })?;

        let grid = (num_patch_positions as f64).sqrt() as usize;
        if grid * grid != num_patch_positions {
            return Err(OCRError::ConfigError {
                message: format!(
                    "HunyuanOCR: vit patch position grid is not square: num={num_patch_positions}"
                ),
            });
        }

        let base = patch_pos
            .to_dtype(DType::F32)
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "HunyuanOCR: vit patch position cast to f32 failed",
                    e,
                )
            })?
            .flatten_all()
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "HunyuanOCR: vit patch position flatten failed",
                    e,
                )
            })?
            .to_vec1::<f32>()
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "HunyuanOCR: vit patch position to_vec failed",
                    e,
                )
            })?;

        let out = interpolate_bilinear_align_corners_false(&base, grid, grid, height, width, dim);
        Tensor::from_vec(out, (height * width, dim), device)
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "HunyuanOCR: vit interpolated patch pos tensor failed",
                    e,
                )
            })?
            .to_dtype(dtype)
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "HunyuanOCR: vit interpolated patch pos cast failed",
                    e,
                )
            })
    }

    #[allow(dead_code)]
    fn extra_pos(&self) -> Result<Tensor, OCRError> {
        self.position_embedding
            .embeddings()
            .i((0, ..))
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "HunyuanOCR: vit slice extra position embedding failed",
                    e,
                )
            })
    }
}

#[derive(Debug, Clone)]
struct VisionAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scaling: f64,
    /// Layer index. Used to gate the F32 attention path on late vit layers
    /// where BF16 Q·K accumulation drift causes attention "sink" positions
    /// to diverge from upstream (cross-implementation cosine drops from
    /// ~0.999 at layer 11 to ~0.95 at layer 26 with BF16 attention).
    layer_idx: usize,
}

impl VisionAttention {
    fn load(
        cfg: &HunyuanOcrVisionConfig,
        layer_idx: usize,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self, OCRError> {
        if !cfg.hidden_size.is_multiple_of(cfg.num_attention_heads) {
            return Err(OCRError::ConfigError {
                message: format!(
                    "HunyuanOCR: vit hidden_size {} not divisible by num_attention_heads {}",
                    cfg.hidden_size, cfg.num_attention_heads
                ),
            });
        }
        let q_proj = candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("q_proj"))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load vit q_proj", e))?;
        let k_proj = candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("k_proj"))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load vit k_proj", e))?;
        let v_proj = candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("v_proj"))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load vit v_proj", e))?;
        let o_proj = candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("o_proj"))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load vit o_proj", e))?;

        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: cfg.num_attention_heads,
            head_dim,
            scaling: (head_dim as f64).powf(-0.5),
            layer_idx,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor, OCRError> {
        let (b, seq_len, _) = hidden_states.dims3().map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: vit attn hidden_states dims3 failed",
                e,
            )
        })?;

        let q_proj = self
            .q_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn q_proj", e))?;
        let k_proj = self
            .k_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn k_proj", e))?;
        let v_proj = self
            .v_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn v_proj", e))?;

        let debug_dir = std::env::var("OAROCR_DUMP_DIR").ok();
        let debug_layers = parse_layer_set_from_env("OAROCR_VIT_DEBUG_LAYERS", 0, None);
        if debug_layers.contains(&self.layer_idx)
            && let Some(d) = &debug_dir
        {
            let _ = q_proj.save_safetensors(
                "t",
                format!("{d}/vit_layer_{:02}_q_proj.safetensors", self.layer_idx),
            );
            let _ = k_proj.save_safetensors(
                "t",
                format!("{d}/vit_layer_{:02}_k_proj.safetensors", self.layer_idx),
            );
            let _ = v_proj.save_safetensors(
                "t",
                format!("{d}/vit_layer_{:02}_v_proj.safetensors", self.layer_idx),
            );
        }

        let q = q_proj
            .reshape((b, seq_len, self.num_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn q reshape", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn q transpose", e))?;

        let k = k_proj
            .reshape((b, seq_len, self.num_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn k reshape", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn k transpose", e))?;

        let v = v_proj
            .reshape((b, seq_len, self.num_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn v reshape", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn v transpose", e))?;

        let q = q
            .contiguous()
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn q contiguous", e))?;
        let k = k
            .contiguous()
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn k contiguous", e))?;
        let v = v
            .contiguous()
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn v contiguous", e))?;

        // Chunked attention over the query dimension. Without chunking the
        // (B, H, N, N) attention matrix at N=4320 (the vit's full patch
        // sequence) needs ~4 GB just for the BF16 buffer, OOM'ing the 4090.
        //
        // Late layers are the current drift hotspot: tiny q/k/v differences
        // can redirect attention to different sink tokens and then get
        // amplified by the final MLPs. Keep the default late-F32 path as the
        // most stable observed compromise, with OAROCR_VIT_ATTN_MODE available
        // for A/B drift experiments (`bf16`, `upstream`, `f32`, `late-f32`).
        const VIT_ATTN_QUERY_CHUNK: usize = 1024;
        const LATE_F32_THRESHOLD: usize = 20;
        let attn_mode =
            std::env::var("OAROCR_VIT_ATTN_MODE").unwrap_or_else(|_| "late-f32".to_string());
        let use_f32 = match attn_mode.as_str() {
            "bf16" | "upstream" => false,
            "f32" => true,
            _ => self.layer_idx >= LATE_F32_THRESHOLD,
        };
        let v_dtype = v.dtype();
        let kt = k
            .transpose(2, 3)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn k t23", e))?
            .contiguous()
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn k t23 contiguous", e))?;
        let (kt_attn, v_attn) = if use_f32 {
            let kt_f32 = kt
                .to_dtype(DType::F32)
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn kt to f32", e))?;
            let v_f32 = v
                .to_dtype(DType::F32)
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn v to f32", e))?;
            (kt_f32, v_f32)
        } else {
            (kt, v.clone())
        };

        let attend_chunk = |q_in: &Tensor| -> Result<Tensor, OCRError> {
            let q_use = if use_f32 {
                q_in.to_dtype(DType::F32)
                    .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn q to f32", e))?
            } else {
                q_in.clone()
            };
            let attn = q_use
                .matmul(&kt_attn)
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn qk matmul", e))?
                .affine(self.scaling, 0.0)
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn scaling", e))?;
            let attn = if use_f32 {
                candle_nn::ops::softmax_last_dim(&attn)
                    .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn softmax", e))?
            } else {
                let attn_f32 = attn
                    .to_dtype(DType::F32)
                    .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn cast f32", e))?;
                let attn_f32 = candle_nn::ops::softmax_last_dim(&attn_f32)
                    .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn softmax", e))?;
                attn_f32
                    .to_dtype(v_dtype)
                    .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn cast back", e))?
            };
            let out = attn
                .matmul(&v_attn)
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn av matmul", e))?;
            if use_f32 {
                out.to_dtype(v_dtype)
                    .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn out to bf16", e))
            } else {
                Ok(out)
            }
        };

        let attn_output = if seq_len <= VIT_ATTN_QUERY_CHUNK {
            attend_chunk(&q)?
        } else {
            let mut chunks: Vec<Tensor> =
                Vec::with_capacity(seq_len.div_ceil(VIT_ATTN_QUERY_CHUNK));
            let mut start = 0;
            while start < seq_len {
                let len = (seq_len - start).min(VIT_ATTN_QUERY_CHUNK);
                let q_chunk = q.narrow(2, start, len).map_err(|e| {
                    candle_to_ocr_inference("HunyuanOCR", "vit attn chunked q narrow", e)
                })?;
                chunks.push(attend_chunk(&q_chunk)?);
                start += len;
            }
            let chunk_refs: Vec<&Tensor> = chunks.iter().collect();
            Tensor::cat(&chunk_refs, 2)
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn chunked cat", e))?
        };

        let attn_output = attn_output
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn out transpose", e))?
            .reshape((b, seq_len, self.num_heads * self.head_dim))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn out reshape", e))?;

        self.o_proj
            .forward(&attn_output)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn o_proj", e))
    }
}

#[derive(Debug, Clone)]
struct VisionMlp {
    fc1: Linear,
    fc2: Linear,
}

impl VisionMlp {
    fn load(cfg: &HunyuanOcrVisionConfig, vb: candle_nn::VarBuilder) -> Result<Self, OCRError> {
        let fc1 = candle_nn::linear(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("dense_h_to_4h"),
        )
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load vit mlp fc1", e))?;
        let fc2 = candle_nn::linear(
            cfg.intermediate_size,
            cfg.hidden_size,
            vb.pp("dense_4h_to_h"),
        )
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load vit mlp fc2", e))?;
        Ok(Self { fc1, fc2 })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, OCRError> {
        let hidden = self
            .fc1
            .forward(xs)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit mlp fc1", e))?;
        // Match PyTorch `nn.GELU()` (exact erf formula). candle's `.gelu()`
        // uses the tanh approximation
        // (`0.5 * v * (1 + tanh(sqrt(2/π)*(v + 0.044715*v³)))`), which
        // diverges by up to ~0.001 per element from the erf formula. Across
        // 27 vit MLPs that drift compounds enough to swap which positions
        // become attention sinks in late layers (max-abs delta jumped from
        // 1.4 at layer 1 to 13419 at layer 26).
        let hidden = hidden
            .gelu_erf()
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit mlp gelu_erf", e))?;
        self.fc2
            .forward(&hidden)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit mlp fc2", e))
    }
}

#[derive(Debug, Clone)]
struct VisionEncoderLayer {
    layer_idx: usize,
    self_attn: VisionAttention,
    mlp: VisionMlp,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
}

impl VisionEncoderLayer {
    fn load(
        cfg: &HunyuanOcrVisionConfig,
        layer_idx: usize,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self, OCRError> {
        let self_attn = VisionAttention::load(cfg, layer_idx, vb.pp("self_attn"))?;
        let mlp = VisionMlp::load(cfg, vb.pp("mlp"))?;

        let ln_cfg = LayerNormConfig {
            eps: cfg.rms_norm_eps,
            remove_mean: true,
            affine: true,
        };
        let input_layernorm =
            candle_nn::layer_norm(cfg.hidden_size, ln_cfg, vb.pp("input_layernorm")).map_err(
                |e| candle_to_ocr_inference("HunyuanOCR", "load vit input_layernorm", e),
            )?;
        let post_attention_layernorm =
            candle_nn::layer_norm(cfg.hidden_size, ln_cfg, vb.pp("post_attention_layernorm"))
                .map_err(|e| {
                    candle_to_ocr_inference("HunyuanOCR", "load vit post_attention_layernorm", e)
                })?;

        Ok(Self {
            layer_idx,
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, OCRError> {
        let debug_dir = std::env::var("OAROCR_DUMP_DIR").ok();
        let debug_layers = parse_layer_set_from_env("OAROCR_VIT_DEBUG_LAYERS", 0, None);
        let debug = debug_layers.contains(&self.layer_idx);
        let residual = xs;
        let hidden = self
            .input_layernorm
            .forward(xs)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit input_layernorm forward", e))?;
        if debug && let Some(d) = &debug_dir {
            let _ = hidden.save_safetensors(
                "t",
                format!("{d}/vit_layer_{:02}_ln1.safetensors", self.layer_idx),
            );
        }
        let attn_out = self.self_attn.forward(&hidden)?;
        if debug && let Some(d) = &debug_dir {
            let _ = attn_out.save_safetensors(
                "t",
                format!("{d}/vit_layer_{:02}_attn.safetensors", self.layer_idx),
            );
        }
        let hidden = (residual + &attn_out)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attention residual add", e))?;

        let residual = hidden.clone();
        let hidden = self
            .post_attention_layernorm
            .forward(&hidden)
            .map_err(|e| {
                candle_to_ocr_inference("HunyuanOCR", "vit post_attention_layernorm forward", e)
            })?;
        if debug && let Some(d) = &debug_dir {
            let _ = hidden.save_safetensors(
                "t",
                format!("{d}/vit_layer_{:02}_ln2.safetensors", self.layer_idx),
            );
        }
        let mlp_out = self.mlp.forward(&hidden)?;
        if debug && let Some(d) = &debug_dir {
            let _ = mlp_out.save_safetensors(
                "t",
                format!("{d}/vit_layer_{:02}_mlp.safetensors", self.layer_idx),
            );
        }
        (&residual + &mlp_out)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit mlp residual add", e))
    }
}

#[derive(Debug, Clone)]
struct VisionPerceive {
    hidden_size: usize,
    before_rms: candle_nn::RmsNorm,
    proj_0: Conv2d,
    proj_2: Conv2d,
    mlp: Linear,
    after_rms: candle_nn::RmsNorm,
    image_begin: Tensor,
    image_end: Tensor,
    image_newline: Tensor,
}

impl VisionPerceive {
    fn load(cfg: &HunyuanOcrVisionConfig, vb: candle_nn::VarBuilder) -> Result<Self, OCRError> {
        let before_rms =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("before_rms")).map_err(
                |e| candle_to_ocr_inference("HunyuanOCR", "load perceive before_rms", e),
            )?;

        let conv_cfg_0 = Conv2dConfig {
            stride: cfg.spatial_merge_size,
            padding: 0,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let proj_0 = candle_nn::conv2d(
            cfg.hidden_size,
            2304usize,
            cfg.spatial_merge_size,
            conv_cfg_0,
            vb.pp("proj.0"),
        )
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load perceive proj.0", e))?;

        let conv_cfg_2 = Conv2dConfig {
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let proj_2 =
            candle_nn::conv2d(2304usize, 4608usize, 1usize, conv_cfg_2, vb.pp("proj.2"))
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load perceive proj.2", e))?;

        let mlp = candle_nn::linear(4608usize, 1024usize, vb.pp("mlp"))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load perceive mlp", e))?;

        let after_rms = candle_nn::rms_norm(1024usize, cfg.rms_norm_eps, vb.pp("after_rms"))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load perceive after_rms", e))?;

        let image_begin = vb
            .get(1024usize, "image_begin")
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load perceive image_begin", e))?;
        let image_end = vb
            .get(1024usize, "image_end")
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load perceive image_end", e))?;
        // `image_sep` exists in the trained weights but is *never* used by
        // upstream's `HunYuanVisionPatchMerger.forward` — see
        // `transformers/models/hunyuan_vl/modeling_hunyuan_vl.py:189-206`. We
        // skip loading it.
        let image_newline = vb
            .get(4608usize, "image_newline")
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load perceive image_newline", e))?;

        Ok(Self {
            hidden_size: cfg.hidden_size,
            before_rms,
            proj_0,
            proj_2,
            mlp,
            after_rms,
            image_begin,
            image_end,
            image_newline,
        })
    }

    fn forward(&self, patch_tokens: &Tensor, h: usize, w: usize) -> Result<Tensor, OCRError> {
        let patch_tokens = self
            .before_rms
            .forward(patch_tokens)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "perceive before_rms forward", e))?;

        let d = patch_tokens.dim(D::Minus1).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: perceive patch_tokens dim failed",
                e,
            )
        })?;
        if d != self.hidden_size {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "HunyuanOCR: unexpected vit hidden dim {d}, expected {}",
                    self.hidden_size
                ),
            });
        }

        let feat_map = patch_tokens
            .reshape((h, w, d))
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "HunyuanOCR: perceive reshape hwd failed",
                    e,
                )
            })?
            .permute((2, 0, 1))
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "HunyuanOCR: perceive permute to chw failed",
                    e,
                )
            })?
            .unsqueeze(0)
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "HunyuanOCR: perceive add batch dim failed",
                    e,
                )
            })?;

        let feat = self
            .proj_0
            .forward(&feat_map)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "perceive proj.0 forward", e))?;
        // Match PyTorch `nn.GELU()` exact erf formula here too — see
        // `VisionMlp::forward` for the rationale.
        let feat = feat
            .gelu_erf()
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "perceive proj.0 gelu_erf", e))?;
        let feat = self
            .proj_2
            .forward(&feat)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "perceive proj.2 forward", e))?;

        let (_b, c, h2, w2) = feat.dims4().map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: perceive feat dims4 failed",
                e,
            )
        })?;
        if c != 4608 {
            return Err(OCRError::InvalidInput {
                message: format!("HunyuanOCR: unexpected perceive channel dim {c}, expected 4608"),
            });
        }

        // Append an extra newline token per row (extra column).
        let newline = self
            .image_newline
            .reshape((1usize, 4608usize, 1usize, 1usize))
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "HunyuanOCR: reshape image_newline failed",
                    e,
                )
            })?
            .expand((1usize, 4608usize, h2, 1usize))
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "HunyuanOCR: expand image_newline column failed",
                    e,
                )
            })?;
        let feat = Tensor::cat(&[&feat, &newline], 3).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: concat newline column failed",
                e,
            )
        })?;

        // Flatten as row-major: (H2, W2+1) tokens, each with 4608 dims.
        let tokens = feat
            .permute((0, 2, 3, 1))
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "HunyuanOCR: permute perceive tokens failed",
                    e,
                )
            })?
            .reshape((h2 * (w2 + 1), 4608usize))
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "HunyuanOCR: reshape perceive tokens failed",
                    e,
                )
            })?;

        let tokens = self
            .mlp
            .forward(&tokens)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "perceive mlp forward", e))?;

        // Match upstream HF (`HunYuanVisionPatchMerger.forward` in
        // modeling_hunyuan_vl.py:189-206) exactly:
        //   1. mlp(x)
        //   2. cat([image_begin, mlp_out, image_end])
        //   3. after_rms(cat)
        //
        // We previously (a) applied `after_rms` to the mlp output BEFORE
        // concatenating the begin/end markers and (b) broadcast-added an
        // unused `image_sep` parameter to every token. Both wrong:
        // upstream's `after_rms` runs once over the full begin+tokens+end
        // sequence, which lifts the image_begin / image_end embedding
        // magnitudes from their stored norm (~0.9) up to the post-RMSNorm
        // scale (~22), matching the surrounding patch tokens. Our pre-fix
        // perceive output had image_begin / image_end at norm ~0.9 — 25×
        // smaller than upstream — so the LLM saw those marker positions as
        // near-zero vectors and the prefill's last-position logits diverged
        // (cos 0.69 vs upstream → wrong argmax → hallucinated
        // continuations like "The presence of factors…" instead of OCR text).
        // The `image_sep` Parameter is declared in upstream weights but
        // *never used in the forward path*; we now drop it on the floor too.
        let begin = self.image_begin.reshape((1usize, 1024usize)).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: reshape image_begin failed",
                e,
            )
        })?;
        let end = self.image_end.reshape((1usize, 1024usize)).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: reshape image_end failed",
                e,
            )
        })?;
        let begin = begin.to_dtype(tokens.dtype()).map_err(|e| {
            candle_to_ocr_inference("HunyuanOCR", "perceive image_begin to dtype", e)
        })?;
        let end = end
            .to_dtype(tokens.dtype())
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "perceive image_end to dtype", e))?;
        let cat = Tensor::cat(&[&begin, &tokens, &end], 0).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: concat begin/tokens/end failed",
                e,
            )
        })?;
        self.after_rms
            .forward(&cat)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "perceive after_rms forward", e))
    }
}

#[derive(Debug, Clone)]
pub struct HunyuanVisionModel {
    cfg: HunyuanOcrVisionConfig,
    embeddings: VisionEmbeddings,
    layers: Vec<VisionEncoderLayer>,
    perceive: VisionPerceive,
}

fn parse_layer_set(spec: &str, n_layers: usize) -> BTreeSet<usize> {
    let mut out = BTreeSet::new();
    for part in spec.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        if part == "last" {
            out.insert(n_layers.saturating_sub(1));
        } else if let Some((lo, hi)) = part.split_once('-') {
            if let (Ok(lo), Ok(hi)) = (lo.parse::<usize>(), hi.parse::<usize>()) {
                for idx in lo..=hi {
                    if n_layers == 0 || idx < n_layers {
                        out.insert(idx);
                    }
                }
            }
        } else if let Ok(idx) = part.parse::<usize>()
            && (n_layers == 0 || idx < n_layers)
        {
            out.insert(idx);
        }
    }
    out
}

fn parse_layer_set_from_env(
    env_name: &str,
    n_layers: usize,
    default: Option<BTreeSet<usize>>,
) -> BTreeSet<usize> {
    let fallback = default.unwrap_or_default();
    let Ok(spec) = std::env::var(env_name) else {
        return fallback;
    };
    if spec.trim().is_empty() {
        return fallback;
    }
    let out = parse_layer_set(&spec, n_layers);
    if out.is_empty() { fallback } else { out }
}

fn parse_vit_dump_layers(n_layers: usize) -> BTreeSet<usize> {
    parse_layer_set_from_env(
        "OAROCR_VIT_DUMP_LAYERS",
        n_layers,
        Some(BTreeSet::from([0, 1, 11, n_layers.saturating_sub(1)])),
    )
}

impl HunyuanVisionModel {
    pub fn load(cfg: &HunyuanOcrVisionConfig, vb: candle_nn::VarBuilder) -> Result<Self, OCRError> {
        let embeddings = VisionEmbeddings::load(cfg, vb.pp("embeddings"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(VisionEncoderLayer::load(
                cfg,
                i,
                vb.pp(format!("layers.{i}")),
            )?);
        }
        let perceive = VisionPerceive::load(cfg, vb.pp("perceive"))?;
        Ok(Self {
            cfg: cfg.clone(),
            embeddings,
            layers,
            perceive,
        })
    }

    /// Produce image token embeddings suitable for replacing the image-token span in the LLM input.
    ///
    /// Returned shape: (1 + Hm*(Wm+1) + 1, out_hidden=1024) where Hm/Wm are merged-grid sizes.
    pub fn forward(&self, pixel_values: &Tensor) -> Result<(Tensor, (usize, usize)), OCRError> {
        let (_b, _c, rh, rw) = pixel_values.dims4().map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: pixel_values dims4 failed",
                e,
            )
        })?;
        if rh % self.cfg.patch_size != 0 || rw % self.cfg.patch_size != 0 {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "HunyuanOCR: pixel_values {rw}x{rh} not divisible by patch_size={}",
                    self.cfg.patch_size
                ),
            });
        }
        let h = rh / self.cfg.patch_size;
        let w = rw / self.cfg.patch_size;

        let patches = self
            .embeddings
            .patch_embedding
            .forward(pixel_values)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit patch_embedding forward", e))?
            .contiguous()
            .map_err(|e| {
                candle_to_ocr_inference("HunyuanOCR", "vit patch_embedding contiguous", e)
            })?;
        let (_b2, _c2, ph, pw) = patches.dims4().map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: vit patches dims4 failed",
                e,
            )
        })?;
        if ph != h || pw != w {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "HunyuanOCR: vit patch grid mismatch: expected {h}x{w} got {ph}x{pw}"
                ),
            });
        }

        let patches = patches.squeeze(0).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: squeeze batch from vit patch embeddings failed",
                e,
            )
        })?;

        let patches = patches
            .permute((1, 2, 0))
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "HunyuanOCR: permute patches to hwd failed",
                    e,
                )
            })?
            .reshape((h * w, self.cfg.hidden_size))
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "HunyuanOCR: reshape patches to seq failed",
                    e,
                )
            })?;

        let patch_pos =
            self.embeddings
                .interpolate_patch_pos(h, w, pixel_values.device(), patches.dtype())?;
        let dump_dir = std::env::var("OAROCR_DUMP_DIR").ok();
        if let Some(d) = &dump_dir {
            let _ = patches
                .unsqueeze(0)
                .and_then(|t| t.save_safetensors("t", format!("{d}/vit_patch_tokens.safetensors")));
            let _ = patch_pos
                .unsqueeze(0)
                .and_then(|t| t.save_safetensors("t", format!("{d}/vit_patch_pos.safetensors")));
        }
        let patch_tokens = patches.broadcast_add(&patch_pos).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: add vit patch pos embedding failed",
                e,
            )
        })?;

        // No extra/cls token. The upstream HunYuanVisionPatchEmbed
        // (`transformers/models/hunyuan_vl/modeling_hunyuan_vl.py`) declares
        // `num_positions = max_num_patches + 1` but the runtime path uses
        // `position_embedding.weight[1:, :]` and feeds only the patch tokens
        // through the encoder — the slot-0 entry is a vestigial cls token
        // present in the trained weights for compatibility but never
        // propagated. An earlier internal Tencent variant (which this Rust
        // port originally followed) prepended a mean-pooled extra token plus
        // a learned `extra_pos`, which contributed unwanted attention scores
        // to every patch and accumulated noise across 27 encoder layers.
        // Removing that prepend halves the residual vit_out drift vs upstream.
        let hidden = patch_tokens.unsqueeze(0).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: add batch dim to vit tokens failed",
                e,
            )
        })?;

        // Optional vit-internal dump for cross-impl drift localisation.
        let dump_layers = parse_vit_dump_layers(self.layers.len());
        if let Some(d) = &dump_dir {
            let _ = hidden.save_safetensors("t", format!("{d}/vit_pre_layers.safetensors"));
        }

        let mut hidden_states = hidden;
        for (i, layer) in self.layers.iter().enumerate() {
            if let Some(d) = &dump_dir
                && dump_layers.contains(&i)
            {
                let _ = hidden_states
                    .save_safetensors("t", format!("{d}/vit_layer_{i:02}_in.safetensors"));
            }
            hidden_states = layer
                .forward(&hidden_states)
                .map_err(|e| OCRError::Inference {
                    model_name: "HunyuanOCR".to_string(),
                    context: format!("vit encoder layer {i} forward failed"),
                    source: Box::new(e),
                })?;
            if let Some(d) = &dump_dir
                && dump_layers.contains(&i)
            {
                let _ = hidden_states
                    .save_safetensors("t", format!("{d}/vit_layer_{i:02}_out.safetensors"));
            }
        }
        if let Some(d) = &dump_dir {
            let _ = hidden_states.save_safetensors("t", format!("{d}/vit_post_layers.safetensors"));
        }

        // No extra token to drop now (see above). Just unbatch.
        let patch_out = hidden_states.squeeze(0).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: squeeze vit patch outputs failed",
                e,
            )
        })?;

        if !h.is_multiple_of(self.cfg.spatial_merge_size)
            || !w.is_multiple_of(self.cfg.spatial_merge_size)
        {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "HunyuanOCR: vit grid {h}x{w} not divisible by spatial_merge_size={}",
                    self.cfg.spatial_merge_size
                ),
            });
        }

        let image_embeds = self.perceive.forward(&patch_out, h, w)?;
        let merged_hw = (
            h / self.cfg.spatial_merge_size,
            w / self.cfg.spatial_merge_size,
        );
        Ok((image_embeds, merged_hw))
    }
}

// A small, dependency-free bilinear interpolation that matches PyTorch's
// align_corners=False semantics for position embeddings.
fn interpolate_bilinear_align_corners_false(
    base: &[f32],
    in_h: usize,
    in_w: usize,
    out_h: usize,
    out_w: usize,
    dim: usize,
) -> Vec<f32> {
    let mut out = vec![0f32; out_h * out_w * dim];
    if in_h == 0 || in_w == 0 || out_h == 0 || out_w == 0 {
        return out;
    }

    // Match upstream HF (`HunYuanVisionPatchEmbed.forward` in
    // modeling_hunyuan_vl.py:143-148): the sample stride is computed from
    // `(out_h + 0.1) / in_h` (a deliberate `+0.1` to "avoid floating point
    // error in the interpolation" — see the comment + facebookresearch/dino#8).
    // PyTorch's `interpolate(scale_factor)` then derives the source coord as
    // `(out_x + 0.5) / scale_factor - 0.5`, which is *not* the same as
    // `(out_x + 0.5) * (in / out) - 0.5` we used before.
    let scale_factor_y = (out_h as f32 + 0.1) / in_h as f32;
    let scale_factor_x = (out_w as f32 + 0.1) / in_w as f32;
    let inv_scale_y = 1.0 / scale_factor_y;
    let inv_scale_x = 1.0 / scale_factor_x;

    for oy in 0..out_h {
        let fy = ((oy as f32 + 0.5) * inv_scale_y - 0.5).max(0.0);
        let y0 = fy.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(in_h - 1);
        let wy = fy - y0 as f32;

        for ox in 0..out_w {
            let fx = ((ox as f32 + 0.5) * inv_scale_x - 0.5).max(0.0);
            let x0 = fx.floor().max(0.0) as usize;
            let x1 = (x0 + 1).min(in_w - 1);
            let wx = fx - x0 as f32;

            let out_base = (oy * out_w + ox) * dim;
            let i00 = (y0 * in_w + x0) * dim;
            let i01 = (y0 * in_w + x1) * dim;
            let i10 = (y1 * in_w + x0) * dim;
            let i11 = (y1 * in_w + x1) * dim;

            for c in 0..dim {
                let v00 = base[i00 + c];
                let v01 = base[i01 + c];
                let v10 = base[i10 + c];
                let v11 = base[i11 + c];
                let v0 = v00 + (v01 - v00) * wx;
                let v1 = v10 + (v11 - v10) * wx;
                out[out_base + c] = v0 + (v1 - v0) * wy;
            }
        }
    }
    out
}
