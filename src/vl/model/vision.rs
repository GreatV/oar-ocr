use super::utils::{candle_to_ocr_inference, candle_to_ocr_processing, rotate_half};
use crate::core::OCRError;
use crate::vl::config::PaddleOcrVlVisionConfig;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::Module;

// =======================================================================
// Vision Model Implementation
// =======================================================================

/// SigLIP-style 2D rotary embedding for vision encoder
#[derive(Debug, Clone)]
struct SigLIPRotaryEmbedding {
    inv_freq: Tensor,
}

impl SigLIPRotaryEmbedding {
    fn new(dim: usize, theta: f64, device: &Device) -> Result<Self, OCRError> {
        let mut inv_freq = Vec::with_capacity(dim / 2);
        for i in (0..dim).step_by(2) {
            let v = 1f64 / theta.powf(i as f64 / dim as f64);
            inv_freq.push(v as f32);
        }
        let inv_freq = Tensor::from_vec(inv_freq, (dim / 2,), device).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: failed to create vision inv_freq tensor",
                e,
            )
        })?;
        Ok(Self { inv_freq })
    }

    /// Compute freqs for positions 0..seqlen
    fn forward(&self, seqlen: usize, device: &Device) -> Result<Tensor, OCRError> {
        let seq: Vec<f32> = (0..seqlen).map(|i| i as f32).collect();
        let seq = Tensor::from_vec(seq, (seqlen,), device).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: vision rope seq tensor failed",
                e,
            )
        })?;
        // outer product: (seqlen, dim/2)
        let inv_freq = self.inv_freq.to_dtype(DType::F32).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: vision inv_freq cast failed",
                e,
            )
        })?;
        // seq: (seqlen,), inv_freq: (dim/2,) -> freqs: (seqlen, dim/2)
        let freqs = seq
            .unsqueeze(1)
            .map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "PaddleOCR-VL: vision seq unsqueeze failed",
                    e,
                )
            })?
            .broadcast_mul(&inv_freq.unsqueeze(0).map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "PaddleOCR-VL: vision inv_freq unsqueeze failed",
                    e,
                )
            })?)
            .map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "PaddleOCR-VL: vision rope outer product failed",
                    e,
                )
            })?;
        Ok(freqs)
    }
}

fn rotate_half_vision(x: &Tensor) -> Result<Tensor, OCRError> {
    rotate_half(x)
}

/// Apply rotary position embedding for vision (2D)
fn apply_rotary_pos_emb_vision(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor), OCRError> {
    // q, k: (batch, seq, num_heads, head_dim)
    // cos, sin: (seq, head_dim) -> need to unsqueeze for broadcasting
    let orig_dtype = q.dtype();

    let q = q.to_dtype(DType::F32).map_err(|e| {
        candle_to_ocr_processing(
            crate::core::errors::ProcessingStage::TensorOperation,
            "PaddleOCR-VL: vision rope q cast failed",
            e,
        )
    })?;
    let k = k.to_dtype(DType::F32).map_err(|e| {
        candle_to_ocr_processing(
            crate::core::errors::ProcessingStage::TensorOperation,
            "PaddleOCR-VL: vision rope k cast failed",
            e,
        )
    })?;

    // cos, sin: (seq, head_dim) -> (1, seq, 1, head_dim)
    let cos = cos
        .unsqueeze(0)
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: vision cos unsqueeze0 failed",
                e,
            )
        })?
        .unsqueeze(2)
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: vision cos unsqueeze2 failed",
                e,
            )
        })?
        .to_dtype(DType::F32)
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: vision cos cast failed",
                e,
            )
        })?;

    let sin = sin
        .unsqueeze(0)
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: vision sin unsqueeze0 failed",
                e,
            )
        })?
        .unsqueeze(2)
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: vision sin unsqueeze2 failed",
                e,
            )
        })?
        .to_dtype(DType::F32)
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: vision sin cast failed",
                e,
            )
        })?;

    // q_embed = q * cos + rotate_half(q) * sin
    let q_rot = rotate_half_vision(&q)?;
    let q_embed = q
        .broadcast_mul(&cos)
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: vision q*cos failed",
                e,
            )
        })?
        .broadcast_add(&q_rot.broadcast_mul(&sin).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: vision rotate_half(q)*sin failed",
                e,
            )
        })?)
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: vision q rope add failed",
                e,
            )
        })?;

    let k_rot = rotate_half_vision(&k)?;
    let k_embed = k
        .broadcast_mul(&cos)
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: vision k*cos failed",
                e,
            )
        })?
        .broadcast_add(&k_rot.broadcast_mul(&sin).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: vision rotate_half(k)*sin failed",
                e,
            )
        })?)
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: vision k rope add failed",
                e,
            )
        })?;

    let q_embed = q_embed.to_dtype(orig_dtype).map_err(|e| {
        candle_to_ocr_processing(
            crate::core::errors::ProcessingStage::TensorOperation,
            "PaddleOCR-VL: vision q_embed cast back failed",
            e,
        )
    })?;
    let k_embed = k_embed.to_dtype(orig_dtype).map_err(|e| {
        candle_to_ocr_processing(
            crate::core::errors::ProcessingStage::TensorOperation,
            "PaddleOCR-VL: vision k_embed cast back failed",
            e,
        )
    })?;

    Ok((q_embed, k_embed))
}

#[derive(Debug, Clone)]
struct VisionAttention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    out_proj: candle_nn::Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl VisionAttention {
    fn load(cfg: &PaddleOcrVlVisionConfig, vb: candle_nn::VarBuilder) -> Result<Self, OCRError> {
        let q_proj = candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("q_proj"))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load vision q_proj", e))?;
        let k_proj = candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("k_proj"))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load vision k_proj", e))?;
        let v_proj = candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("v_proj"))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load vision v_proj", e))?;
        let out_proj = candle_nn::linear(cfg.hidden_size, cfg.hidden_size, vb.pp("out_proj"))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load vision out_proj", e))?;
        if !cfg.hidden_size.is_multiple_of(cfg.num_attention_heads) {
            return Err(OCRError::ConfigError {
                message: format!(
                    "PaddleOCR-VL vision: hidden_size ({}) must be divisible by num_attention_heads ({})",
                    cfg.hidden_size, cfg.num_attention_heads
                ),
            });
        }
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads: cfg.num_attention_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        rope_emb: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor, OCRError> {
        let (b, seq, embed_dim) = hidden_states
            .dims3()
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision attn dims3", e))?;

        let q = self
            .q_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision q_proj", e))?
            .reshape((b, seq, self.num_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision q reshape", e))?;

        let k = self
            .k_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision k_proj", e))?
            .reshape((b, seq, self.num_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision k reshape", e))?;

        let v = self
            .v_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision v_proj", e))?
            .reshape((b, seq, self.num_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision v reshape", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision v transpose", e))?
            .contiguous()
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision v contiguous", e))?;

        // Apply rotary embeddings if provided
        let (q, k) = if let Some((cos, sin)) = rope_emb {
            let (q_rot, k_rot) = apply_rotary_pos_emb_vision(&q, &k, cos, sin)?;
            (
                q_rot
                    .transpose(1, 2)
                    .map_err(|e| {
                        candle_to_ocr_inference("PaddleOCR-VL", "vision q_rot transpose", e)
                    })?
                    .contiguous()
                    .map_err(|e| {
                        candle_to_ocr_inference("PaddleOCR-VL", "vision q_rot contiguous", e)
                    })?,
                k_rot
                    .transpose(1, 2)
                    .map_err(|e| {
                        candle_to_ocr_inference("PaddleOCR-VL", "vision k_rot transpose", e)
                    })?
                    .contiguous()
                    .map_err(|e| {
                        candle_to_ocr_inference("PaddleOCR-VL", "vision k_rot contiguous", e)
                    })?,
            )
        } else {
            (
                q.transpose(1, 2)
                    .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision q transpose", e))?
                    .contiguous()
                    .map_err(|e| {
                        candle_to_ocr_inference("PaddleOCR-VL", "vision q contiguous", e)
                    })?,
                k.transpose(1, 2)
                    .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision k transpose", e))?
                    .contiguous()
                    .map_err(|e| {
                        candle_to_ocr_inference("PaddleOCR-VL", "vision k contiguous", e)
                    })?,
            )
        };

        let attn_weights = q
            .matmul(
                &k.transpose(2, 3)
                    .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision k t23", e))?
                    .contiguous()
                    .map_err(|e| {
                        candle_to_ocr_inference("PaddleOCR-VL", "vision k t23 contiguous", e)
                    })?,
            )
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision qk matmul", e))?
            .affine(self.scale, 0.0)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision scaling", e))?;

        let attn_weights = candle_nn::ops::softmax_last_dim(
            &attn_weights
                .to_dtype(DType::F32)
                .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision attn cast f32", e))?,
        )
        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision attn softmax", e))?
        .to_dtype(v.dtype())
        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision attn cast back", e))?
        .contiguous()
        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision attn contiguous", e))?;

        let attn_output = attn_weights
            .matmul(&v)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision av matmul", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision out transpose", e))?
            .reshape((b, seq, embed_dim))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision out reshape", e))?;

        self.out_proj
            .forward(&attn_output)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision out_proj", e))
    }
}

#[derive(Debug, Clone)]
struct VisionMlp {
    fc1: candle_nn::Linear,
    fc2: candle_nn::Linear,
    act: candle_nn::Activation,
}

impl VisionMlp {
    fn load(cfg: &PaddleOcrVlVisionConfig, vb: candle_nn::VarBuilder) -> Result<Self, OCRError> {
        let fc1 = candle_nn::linear(cfg.hidden_size, cfg.intermediate_size, vb.pp("fc1"))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load vision fc1", e))?;
        let fc2 = candle_nn::linear(cfg.intermediate_size, cfg.hidden_size, vb.pp("fc2"))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load vision fc2", e))?;
        let act = serde_json::from_str::<candle_nn::Activation>(&format!("\"{}\"", cfg.hidden_act))
            .unwrap_or(candle_nn::Activation::GeluPytorchTanh);
        Ok(Self { fc1, fc2, act })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, OCRError> {
        let xs = self
            .fc1
            .forward(xs)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision mlp fc1", e))?;
        let xs = self
            .act
            .forward(&xs)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision mlp act", e))?;
        self.fc2
            .forward(&xs)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision mlp fc2", e))
    }
}

#[derive(Debug, Clone)]
struct VisionEncoderLayer {
    layer_norm1: candle_nn::LayerNorm,
    self_attn: VisionAttention,
    layer_norm2: candle_nn::LayerNorm,
    mlp: VisionMlp,
}

impl VisionEncoderLayer {
    fn load(cfg: &PaddleOcrVlVisionConfig, vb: candle_nn::VarBuilder) -> Result<Self, OCRError> {
        let ln_cfg = candle_nn::LayerNormConfig {
            eps: cfg.layer_norm_eps,
            remove_mean: true,
            affine: true,
        };
        let layer_norm1 = candle_nn::layer_norm(cfg.hidden_size, ln_cfg, vb.pp("layer_norm1"))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load vision layer_norm1", e))?;
        let self_attn = VisionAttention::load(cfg, vb.pp("self_attn"))?;
        let layer_norm2 = candle_nn::layer_norm(cfg.hidden_size, ln_cfg, vb.pp("layer_norm2"))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load vision layer_norm2", e))?;
        let mlp = VisionMlp::load(cfg, vb.pp("mlp"))?;
        Ok(Self {
            layer_norm1,
            self_attn,
            layer_norm2,
            mlp,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        rope_emb: Option<(&Tensor, &Tensor)>,
    ) -> Result<Tensor, OCRError> {
        let residual = hidden_states.clone();
        let hidden_states = self
            .layer_norm1
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision ln1", e))?;
        let attn = self.self_attn.forward(&hidden_states, rope_emb)?;
        let hidden_states = (&residual + &attn)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision attn residual add", e))?;

        let residual = hidden_states.clone();
        let hidden_states = self
            .layer_norm2
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision ln2", e))?;
        let mlp = self.mlp.forward(&hidden_states)?;
        (&residual + &mlp)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision mlp residual add", e))
    }
}

#[derive(Debug, Clone)]
struct VisionEmbeddings {
    patch_embedding: candle_nn::Conv2d,
    packing_position_embedding: candle_nn::Embedding,
}

impl VisionEmbeddings {
    fn load(cfg: &PaddleOcrVlVisionConfig, vb: candle_nn::VarBuilder) -> Result<Self, OCRError> {
        let conv_cfg = candle_nn::Conv2dConfig {
            padding: 0,
            stride: cfg.patch_size,
            dilation: 1,
            groups: 1,
            cudnn_fwd_algo: None,
        };
        let patch_embedding = candle_nn::conv2d(
            3,
            cfg.hidden_size,
            cfg.patch_size,
            conv_cfg,
            vb.pp("patch_embedding"),
        )
        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load vision patch_embedding", e))?;
        let packing_position_embedding =
            candle_nn::embedding(32768, cfg.hidden_size, vb.pp("packing_position_embedding"))
                .map_err(|e| {
                    candle_to_ocr_inference(
                        "PaddleOCR-VL",
                        "load vision packing_position_embedding",
                        e,
                    )
                })?;
        Ok(Self {
            patch_embedding,
            packing_position_embedding,
        })
    }
}

#[derive(Debug, Clone)]
pub struct VisionModel {
    embeddings: VisionEmbeddings,
    layers: Vec<VisionEncoderLayer>,
    post_layernorm: candle_nn::LayerNorm,
    rotary_pos_emb: SigLIPRotaryEmbedding,
}

impl VisionModel {
    pub fn load(
        cfg: &PaddleOcrVlVisionConfig,
        vb: candle_nn::VarBuilder,
    ) -> Result<Self, OCRError> {
        let embeddings = VisionEmbeddings::load(cfg, vb.pp("embeddings"))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(VisionEncoderLayer::load(
                cfg,
                vb.pp(format!("encoder.layers.{i}")),
            )?);
        }
        let ln_cfg = candle_nn::LayerNormConfig {
            eps: cfg.layer_norm_eps,
            remove_mean: true,
            affine: true,
        };
        let post_layernorm =
            candle_nn::layer_norm(cfg.hidden_size, ln_cfg, vb.pp("post_layernorm")).map_err(
                |e| candle_to_ocr_inference("PaddleOCR-VL", "load vision post_layernorm", e),
            )?;

        // Create rotary embedding for vision
        // Python: SigLIPRotaryEmbedding(head_dim // 2)
        let head_dim = cfg.hidden_size / cfg.num_attention_heads;
        let rotary_pos_emb = SigLIPRotaryEmbedding::new(head_dim / 2, 10000.0, vb.device())?;

        Ok(Self {
            embeddings,
            layers,
            post_layernorm,
            rotary_pos_emb,
        })
    }

    pub fn forward(
        &self,
        pixel_values: &Tensor,
        image_grid_thw: &[(usize, usize, usize)],
    ) -> Result<Vec<Tensor>, OCRError> {
        let device = pixel_values.device();
        let mut position_ids: Vec<u32> = Vec::with_capacity(pixel_values.dims()[0]);

        // Compute height/width position IDs for 2D rope
        let mut height_position_ids: Vec<i64> = Vec::new();
        let mut width_position_ids: Vec<i64> = Vec::new();

        for &(t, h, w) in image_grid_thw {
            let hw = (h * w) as u32;
            let numel = t * h * w;
            for idx in 0..numel as u32 {
                position_ids.push(idx % hw);
                // height_id = patch_idx // w, width_id = patch_idx % w
                let patch_idx = idx % hw;
                height_position_ids.push((patch_idx / w as u32) as i64);
                width_position_ids.push((patch_idx % w as u32) as i64);
            }
        }

        let position_ids = Tensor::new(position_ids, device).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: failed to create vision position_ids tensor",
                e,
            )
        })?;

        // Build rope embeddings
        // pids = stack([height_ids, width_ids], dim=-1)  -> (num_patches, 2)
        // rope_emb_max_grid = rotary_pos_emb(max_grid_size) -> (max_grid_size, dim/2)
        // rope_emb = rope_emb_max_grid[pids].flatten(1).repeat(1, 2) -> (num_patches, head_dim)
        let max_h = height_position_ids.iter().copied().max().unwrap_or(0) as usize + 1;
        let max_w = width_position_ids.iter().copied().max().unwrap_or(0) as usize + 1;
        let max_grid_size = max_h.max(max_w);

        // Compute freqs for all positions 0..max_grid_size
        let freqs = self.rotary_pos_emb.forward(max_grid_size, device)?;

        // Gather freqs for height and width positions
        let h_ids = Tensor::new(height_position_ids.clone(), device)
            .map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "PaddleOCR-VL: vision height_ids tensor failed",
                    e,
                )
            })?
            .to_dtype(DType::U32)
            .map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "PaddleOCR-VL: vision height_ids cast failed",
                    e,
                )
            })?;

        let w_ids = Tensor::new(width_position_ids, device)
            .map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "PaddleOCR-VL: vision width_ids tensor failed",
                    e,
                )
            })?
            .to_dtype(DType::U32)
            .map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "PaddleOCR-VL: vision width_ids cast failed",
                    e,
                )
            })?;

        // freqs_h = freqs[height_ids], freqs_w = freqs[width_ids]
        let freqs_h = freqs.index_select(&h_ids, 0).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: vision freqs_h gather failed",
                e,
            )
        })?;
        let freqs_w = freqs.index_select(&w_ids, 0).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: vision freqs_w gather failed",
                e,
            )
        })?;

        // Concatenate to get (num_patches, head_dim/2) then repeat to get (num_patches, head_dim)
        let rope_emb = Tensor::cat(&[&freqs_h, &freqs_w], 1).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: vision rope_emb cat failed",
                e,
            )
        })?;
        // Repeat to double the dimension for full head_dim
        let rope_emb = Tensor::cat(&[&rope_emb, &rope_emb], 1).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: vision rope_emb repeat failed",
                e,
            )
        })?;

        let cos = rope_emb.cos().map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: vision rope cos failed",
                e,
            )
        })?;
        let sin = rope_emb.sin().map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: vision rope sin failed",
                e,
            )
        })?;

        let patch = self
            .embeddings
            .patch_embedding
            .forward(pixel_values)
            .map_err(|e| {
                candle_to_ocr_inference("PaddleOCR-VL", "vision patch_embedding forward", e)
            })?;

        let patch = patch
            .flatten_from(2)
            .map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "PaddleOCR-VL: vision patch flatten failed",
                    e,
                )
            })?
            .squeeze(2)
            .map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "PaddleOCR-VL: vision patch squeeze failed",
                    e,
                )
            })?;

        let pos = self
            .embeddings
            .packing_position_embedding
            .forward(&position_ids)
            .map_err(|e| {
                candle_to_ocr_inference(
                    "PaddleOCR-VL",
                    "vision packing_position_embedding forward",
                    e,
                )
            })?;

        let mut hidden = patch
            .broadcast_add(&pos)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "vision add pos", e))?
            .unsqueeze(0)
            .map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "PaddleOCR-VL: vision add batch dim failed",
                    e,
                )
            })?;

        // Pass rope embeddings to each layer
        let rope_emb = Some((&cos, &sin));
        for layer in self.layers.iter() {
            hidden = layer.forward(&hidden, rope_emb)?;
        }

        hidden = self.post_layernorm.forward(&hidden).map_err(|e| {
            candle_to_ocr_inference("PaddleOCR-VL", "vision post_layernorm forward", e)
        })?;

        let hidden = hidden.squeeze(0).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: vision squeeze batch failed",
                e,
            )
        })?;

        let mut out = Vec::with_capacity(image_grid_thw.len());
        let mut start = 0usize;
        for &(t, h, w) in image_grid_thw {
            let len = t * h * w;
            let end = start + len;
            let slice = hidden.i((start..end, ..)).map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "PaddleOCR-VL: vision slice failed",
                    e,
                )
            })?;
            out.push(slice);
            start = end;
        }
        Ok(out)
    }
}
