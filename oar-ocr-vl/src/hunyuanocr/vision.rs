use super::config::HunyuanOcrVisionConfig;
use crate::utils::{candle_to_ocr_inference, candle_to_ocr_processing};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{Conv2d, Conv2dConfig, LayerNorm, LayerNormConfig, Linear, Module};
use oar_ocr_core::core::OCRError;

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
}

impl VisionAttention {
    fn load(cfg: &HunyuanOcrVisionConfig, vb: candle_nn::VarBuilder) -> Result<Self, OCRError> {
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

        let q = self
            .q_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn q_proj", e))?
            .reshape((b, seq_len, self.num_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn q reshape", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn q transpose", e))?;

        let k = self
            .k_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn k_proj", e))?
            .reshape((b, seq_len, self.num_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn k reshape", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn k transpose", e))?;

        let v = self
            .v_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn v_proj", e))?
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

        let attn_weights = q
            .matmul(
                &k.transpose(2, 3)
                    .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn k t23", e))?
                    .contiguous()
                    .map_err(|e| {
                        candle_to_ocr_inference("HunyuanOCR", "vit attn k t23 contiguous", e)
                    })?,
            )
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn qk matmul", e))?
            .affine(self.scaling, 0.0)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn scaling", e))?;

        let attn_weights = candle_nn::ops::softmax_last_dim(
            &attn_weights
                .to_dtype(DType::F32)
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn cast f32", e))?,
        )
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn softmax", e))?
        .to_dtype(v.dtype())
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn cast back", e))?;

        let attn_output = attn_weights
            .matmul(&v)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attn av matmul", e))?
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
        let hidden = hidden
            .gelu()
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit mlp gelu", e))?;
        self.fc2
            .forward(&hidden)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit mlp fc2", e))
    }
}

#[derive(Debug, Clone)]
struct VisionEncoderLayer {
    self_attn: VisionAttention,
    mlp: VisionMlp,
    input_layernorm: LayerNorm,
    post_attention_layernorm: LayerNorm,
}

impl VisionEncoderLayer {
    fn load(cfg: &HunyuanOcrVisionConfig, vb: candle_nn::VarBuilder) -> Result<Self, OCRError> {
        let self_attn = VisionAttention::load(cfg, vb.pp("self_attn"))?;
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
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, OCRError> {
        let residual = xs;
        let hidden = self
            .input_layernorm
            .forward(xs)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit input_layernorm forward", e))?;
        let attn_out = self.self_attn.forward(&hidden)?;
        let hidden = (residual + &attn_out)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit attention residual add", e))?;

        let residual = hidden.clone();
        let hidden = self
            .post_attention_layernorm
            .forward(&hidden)
            .map_err(|e| {
                candle_to_ocr_inference("HunyuanOCR", "vit post_attention_layernorm forward", e)
            })?;
        let mlp_out = self.mlp.forward(&hidden)?;
        (&residual + &mlp_out)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "vit mlp residual add", e))
    }
}

#[derive(Debug, Clone)]
struct VisionPerceive {
    before_rms: candle_nn::RmsNorm,
    proj_0: Conv2d,
    proj_2: Conv2d,
    mlp: Linear,
    after_rms: candle_nn::RmsNorm,
    image_begin: Tensor,
    image_end: Tensor,
    image_sep: Tensor,
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
        let image_sep = vb
            .get(1024usize, "image_sep")
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load perceive image_sep", e))?;
        let image_newline = vb
            .get(4608usize, "image_newline")
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "load perceive image_newline", e))?;

        Ok(Self {
            before_rms,
            proj_0,
            proj_2,
            mlp,
            after_rms,
            image_begin,
            image_end,
            image_sep,
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
        if d != 1152 {
            return Err(OCRError::InvalidInput {
                message: format!("HunyuanOCR: unexpected vit hidden dim {d}, expected 1152"),
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
        let feat = feat
            .gelu()
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "perceive proj.0 gelu", e))?;
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
        let tokens = self
            .after_rms
            .forward(&tokens)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR", "perceive after_rms forward", e))?;

        let sep = self.image_sep.reshape((1usize, 1024usize)).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: reshape image_sep failed",
                e,
            )
        })?;
        let tokens = tokens.broadcast_add(&sep).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: add image_sep failed",
                e,
            )
        })?;

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
        Tensor::cat(&[&begin, &tokens, &end], 0).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: concat begin/tokens/end failed",
                e,
            )
        })
    }
}

#[derive(Debug, Clone)]
pub struct HunyuanVisionModel {
    cfg: HunyuanOcrVisionConfig,
    embeddings: VisionEmbeddings,
    layers: Vec<VisionEncoderLayer>,
    perceive: VisionPerceive,
}

impl HunyuanVisionModel {
    pub fn load(cfg: &HunyuanOcrVisionConfig, vb: candle_nn::VarBuilder) -> Result<Self, OCRError> {
        let embeddings = VisionEmbeddings::load(cfg, vb.pp("embeddings"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(VisionEncoderLayer::load(cfg, vb.pp(format!("layers.{i}")))?);
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
        let patch_tokens = patches.broadcast_add(&patch_pos).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: add vit patch pos embedding failed",
                e,
            )
        })?;

        // Add a lightweight extra token (mean-pooled) to match the original position embedding layout.
        let extra_pos = self.embeddings.extra_pos()?.unsqueeze(0).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: unsqueeze extra_pos failed",
                e,
            )
        })?;
        let extra = patch_tokens.mean_keepdim(0).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: mean_keepdim vit extra token failed",
                e,
            )
        })?;
        let extra = extra.broadcast_add(&extra_pos).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: add extra position embedding failed",
                e,
            )
        })?;
        let extra = extra.unsqueeze(0).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: unsqueeze extra token failed",
                e,
            )
        })?;

        let patch_tokens = patch_tokens.unsqueeze(0).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: add batch dim to vit tokens failed",
                e,
            )
        })?;
        let hidden = Tensor::cat(&[&extra, &patch_tokens], 1).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: concat vit extra token failed",
                e,
            )
        })?;

        let mut hidden_states = hidden;
        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer
                .forward(&hidden_states)
                .map_err(|e| OCRError::Inference {
                    model_name: "HunyuanOCR".to_string(),
                    context: format!("vit encoder layer {i} forward failed"),
                    source: Box::new(e),
                })?;
        }

        // Drop the extra token before spatial perceiver merge.
        let patch_out = hidden_states.i((.., 1.., ..)).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "HunyuanOCR: slice vit patch outputs failed",
                e,
            )
        })?;
        let patch_out = patch_out.squeeze(0).map_err(|e| {
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

    let scale_y = in_h as f32 / out_h as f32;
    let scale_x = in_w as f32 / out_w as f32;

    for oy in 0..out_h {
        let fy = (oy as f32 + 0.5) * scale_y - 0.5;
        let y0 = fy.floor().max(0.0) as usize;
        let y1 = (y0 + 1).min(in_h - 1);
        let wy = fy - y0 as f32;

        for ox in 0..out_w {
            let fx = (ox as f32 + 0.5) * scale_x - 0.5;
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
