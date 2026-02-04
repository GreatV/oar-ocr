use super::config::GlmOcrTextConfig;
use crate::attention::{repeat_kv, scaled_dot_product_attention};
use crate::utils::{candle_to_ocr_inference, candle_to_ocr_processing};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::{
    Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, kv_cache::KvCache, linear_no_bias,
    rms_norm,
};
use oar_ocr_core::core::OCRError;
use std::cell::RefCell;

fn rotate_half_interleaved(x: &Tensor) -> Result<Tensor, OCRError> {
    let (b, h, s, d) = x.dims4().map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: rotate_half dims4",
            e,
        )
    })?;
    if d % 2 != 0 {
        return Err(OCRError::ConfigError {
            message: format!("GLM-OCR: head_dim must be even, got {d}"),
        });
    }
    let half = d / 2;
    let x = x.reshape((b, h, s, half, 2)).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: rotate_half reshape",
            e,
        )
    })?;
    let x_even = x.i((.., .., .., .., 0)).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: rotate_half even slice",
            e,
        )
    })?;
    let x_odd = x.i((.., .., .., .., 1)).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: rotate_half odd slice",
            e,
        )
    })?;
    let x_odd = x_odd.neg().map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: rotate_half neg",
            e,
        )
    })?;
    let stacked = Tensor::stack(&[&x_odd, &x_even], D::Minus1).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: rotate_half stack",
            e,
        )
    })?;
    stacked.flatten_from(D::Minus2).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: rotate_half flatten",
            e,
        )
    })
}

fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor), OCRError> {
    let cos = cos.unsqueeze(1).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: cos unsqueeze",
            e,
        )
    })?;
    let sin = sin.unsqueeze(1).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: sin unsqueeze",
            e,
        )
    })?;

    let (b, h, s, rot_dim) = cos.dims4().map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: cos dims4",
            e,
        )
    })?;
    let half = rot_dim / 2;
    let cos_half = cos.narrow(D::Minus1, 0, half).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: cos narrow",
            e,
        )
    })?;
    let sin_half = sin.narrow(D::Minus1, 0, half).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: sin narrow",
            e,
        )
    })?;
    let cos = Tensor::stack(&[&cos_half, &cos_half], D::Minus1)
        .map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: cos stack",
                e,
            )
        })?
        .reshape((b, h, s, rot_dim))
        .map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: cos reshape",
                e,
            )
        })?;
    let sin = Tensor::stack(&[&sin_half, &sin_half], D::Minus1)
        .map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: sin stack",
                e,
            )
        })?
        .reshape((b, h, s, rot_dim))
        .map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: sin reshape",
                e,
            )
        })?;

    let head_dim = q.dim(D::Minus1).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: q head_dim",
            e,
        )
    })?;

    let q_rot = q.narrow(D::Minus1, 0, rot_dim).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: q narrow",
            e,
        )
    })?;
    let k_rot = k.narrow(D::Minus1, 0, rot_dim).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: k narrow",
            e,
        )
    })?;

    let q_rotated = rotate_half_interleaved(&q_rot)?;
    let k_rotated = rotate_half_interleaved(&k_rot)?;

    let q_mul = q_rot.broadcast_mul(&cos).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: q*cos",
            e,
        )
    })?;
    let q_rot_mul = q_rotated.broadcast_mul(&sin).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: rotate_half(q)*sin",
            e,
        )
    })?;
    let mut q_out = (&q_mul + &q_rot_mul).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: q apply rope",
            e,
        )
    })?;

    let k_mul = k_rot.broadcast_mul(&cos).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: k*cos",
            e,
        )
    })?;
    let k_rot_mul = k_rotated.broadcast_mul(&sin).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: rotate_half(k)*sin",
            e,
        )
    })?;
    let mut k_out = (&k_mul + &k_rot_mul).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: k apply rope",
            e,
        )
    })?;

    if rot_dim < head_dim {
        let q_pass = q
            .narrow(D::Minus1, rot_dim, head_dim - rot_dim)
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: q pass narrow",
                    e,
                )
            })?;
        let k_pass = k
            .narrow(D::Minus1, rot_dim, head_dim - rot_dim)
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: k pass narrow",
                    e,
                )
            })?;
        q_out = Tensor::cat(&[&q_out, &q_pass], D::Minus1).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: q cat pass",
                e,
            )
        })?;
        k_out = Tensor::cat(&[&k_out, &k_pass], D::Minus1).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: k cat pass",
                e,
            )
        })?;
    }

    Ok((q_out, k_out))
}

fn apply_mrope(freqs: &Tensor, mrope_section: &[usize]) -> Result<Tensor, OCRError> {
    if mrope_section.is_empty() {
        return Err(OCRError::ConfigError {
            message: "GLM-OCR: mrope_section is empty".to_string(),
        });
    }
    let dims = freqs.dims();
    if dims.len() != 4 || dims[0] != 3 {
        return Err(OCRError::InvalidInput {
            message: format!("GLM-OCR: freqs dims mismatch, got {:?}", dims),
        });
    }
    let mut offset = 0usize;
    let mut chunks = Vec::with_capacity(mrope_section.len());
    for (i, &sec) in mrope_section.iter().enumerate() {
        let seg = freqs.narrow(D::Minus1, offset, sec).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: mrope narrow",
                e,
            )
        })?;
        let picked = seg.i((i % 3, .., .., ..)).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: mrope pick",
                e,
            )
        })?;
        chunks.push(picked);
        offset += sec;
    }
    let refs: Vec<&Tensor> = chunks.iter().collect();
    Tensor::cat(&refs, D::Minus1).map_err(|e| {
        candle_to_ocr_processing(
            oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
            "GLM-OCR: mrope cat",
            e,
        )
    })
}

#[derive(Debug, Clone)]
struct GlmOcrTextRotaryEmbedding {
    inv_freq: Tensor,
    mrope_section: Vec<usize>,
}

impl GlmOcrTextRotaryEmbedding {
    fn new(cfg: &GlmOcrTextConfig, device: &Device) -> Result<Self, OCRError> {
        if cfg.rope_parameters.rope_type != "default" {
            return Err(OCRError::ConfigError {
                message: format!(
                    "GLM-OCR: unsupported rope_type '{}'",
                    cfg.rope_parameters.rope_type
                ),
            });
        }
        let head_dim = if cfg.head_dim > 0 {
            cfg.head_dim
        } else {
            cfg.hidden_size / cfg.num_attention_heads
        };
        let dim = (head_dim as f64 * cfg.rope_parameters.partial_rotary_factor).floor() as usize;
        if !dim.is_multiple_of(2) {
            return Err(OCRError::ConfigError {
                message: format!("GLM-OCR: rotary dim must be even, got {dim}"),
            });
        }
        let rope_theta = cfg.rope_parameters.rope_theta;
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| (1.0f64 / rope_theta.powf(i as f64 / dim as f64)) as f32)
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (dim / 2,), device).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: create inv_freq",
                e,
            )
        })?;

        let section_sum: usize = cfg.rope_parameters.mrope_section.iter().sum();
        if section_sum != dim / 2 {
            return Err(OCRError::ConfigError {
                message: format!(
                    "GLM-OCR: mrope_section sum ({section_sum}) != dim/2 ({})",
                    dim / 2
                ),
            });
        }

        Ok(Self {
            inv_freq,
            mrope_section: cfg.rope_parameters.mrope_section.clone(),
        })
    }

    fn forward(&self, x: &Tensor, position_ids: &Tensor) -> Result<(Tensor, Tensor), OCRError> {
        let dtype = x.dtype();
        let dims = position_ids.dims();
        if dims.len() != 3 || dims[0] != 3 {
            return Err(OCRError::InvalidInput {
                message: format!("GLM-OCR: position_ids must be (3, B, S), got {:?}", dims),
            });
        }

        let position_ids = position_ids.to_dtype(DType::F32).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: position_ids cast",
                e,
            )
        })?;

        let inv_len = self.inv_freq.dims1().map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: inv_freq dims1",
                e,
            )
        })?;
        let inv = self.inv_freq.reshape((1, 1, 1, inv_len)).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: inv_freq reshape",
                e,
            )
        })?;
        let freqs = position_ids
            .unsqueeze(3)
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: position_ids unsqueeze",
                    e,
                )
            })?
            .broadcast_mul(&inv)
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: freqs multiply",
                    e,
                )
            })?;

        let freqs = apply_mrope(&freqs, &self.mrope_section)?;
        let emb = Tensor::cat(&[&freqs, &freqs], D::Minus1).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: freqs cat",
                e,
            )
        })?;
        let cos = emb.cos().map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: cos",
                e,
            )
        })?;
        let sin = emb.sin().map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: sin",
                e,
            )
        })?;

        let cos = cos.to_dtype(dtype).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: cos cast",
                e,
            )
        })?;
        let sin = sin.to_dtype(dtype).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: sin cast",
                e,
            )
        })?;
        Ok((cos, sin))
    }
}

#[derive(Debug, Clone)]
struct GlmOcrTextMLP {
    gate_up_proj: Linear,
    down_proj: Linear,
    activation_fn: candle_nn::Activation,
}

impl GlmOcrTextMLP {
    fn load(cfg: &GlmOcrTextConfig, vb: VarBuilder) -> Result<Self, OCRError> {
        let gate_up_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.intermediate_size * 2,
            vb.pp("mlp").pp("gate_up_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text gate_up_proj", e))?;
        let down_proj = linear_no_bias(
            cfg.intermediate_size,
            cfg.hidden_size,
            vb.pp("mlp").pp("down_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text down_proj", e))?;
        Ok(Self {
            gate_up_proj,
            down_proj,
            activation_fn: cfg.hidden_act,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor, OCRError> {
        let up_states = self
            .gate_up_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text gate_up_proj forward", e))?;
        let mut parts = up_states.chunk(2, D::Minus1).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: gate_up_proj chunk",
                e,
            )
        })?;
        if parts.len() != 2 {
            return Err(OCRError::InvalidInput {
                message: format!(
                    "GLM-OCR: expected 2 chunks from gate_up_proj, got {}",
                    parts.len()
                ),
            });
        }
        let up_states = parts.pop().unwrap();
        let gate = parts.pop().unwrap();
        let gate = gate
            .apply(&self.activation_fn)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text gate act", e))?;
        let up_states = (up_states * gate)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text gate*up", e))?;
        self.down_proj
            .forward(&up_states)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text down_proj forward", e))
    }
}

#[derive(Debug, Clone)]
struct GlmOcrTextAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
    kv_cache: RefCell<KvCache>,
}

impl GlmOcrTextAttention {
    fn load(cfg: &GlmOcrTextConfig, vb: VarBuilder) -> Result<Self, OCRError> {
        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            return Err(OCRError::ConfigError {
                message: format!(
                    "GLM-OCR: num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                    cfg.num_attention_heads, cfg.num_key_value_heads
                ),
            });
        }

        let q_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.head_dim,
            vb.pp("self_attn").pp("q_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text q_proj", e))?;
        let k_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            vb.pp("self_attn").pp("k_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text k_proj", e))?;
        let v_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            vb.pp("self_attn").pp("v_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text v_proj", e))?;
        let o_proj = linear_no_bias(
            cfg.num_attention_heads * cfg.head_dim,
            cfg.hidden_size,
            vb.pp("self_attn").pp("o_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text o_proj", e))?;

        let cache_cap = cfg.max_position_embeddings.min(16384);
        let kv_cache = KvCache::new(2, cache_cap);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            scaling: 1.0 / (cfg.head_dim as f64).sqrt(),
            kv_cache: RefCell::new(kv_cache),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, OCRError> {
        let (b, seq_len, _) = hidden_states.dims3().map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: attn hidden dims3",
                e,
            )
        })?;

        let mut q = self
            .q_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text q_proj forward", e))?;
        let mut k = self
            .k_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text k_proj forward", e))?;
        let mut v = self
            .v_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text v_proj forward", e))?;

        q = q
            .reshape((b, seq_len, self.num_heads, self.head_dim))
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: q reshape",
                    e,
                )
            })?
            .transpose(1, 2)
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: q transpose",
                    e,
                )
            })?
            .contiguous()
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: q contiguous",
                    e,
                )
            })?;
        k = k
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: k reshape",
                    e,
                )
            })?
            .transpose(1, 2)
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: k transpose",
                    e,
                )
            })?
            .contiguous()
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: k contiguous",
                    e,
                )
            })?;
        v = v
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: v reshape",
                    e,
                )
            })?
            .transpose(1, 2)
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: v transpose",
                    e,
                )
            })?
            .contiguous()
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: v contiguous",
                    e,
                )
            })?;

        (q, k) = apply_rotary_pos_emb(&q, &k, cos, sin)?;
        q = q.contiguous().map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: q contiguous post-rope",
                e,
            )
        })?;
        k = k.contiguous().map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: k contiguous post-rope",
                e,
            )
        })?;

        let (k, v) = self.kv_cache.borrow_mut().append(&k, &v).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: kv_cache append",
                e,
            )
        })?;
        let k = repeat_kv(&k, self.num_kv_groups).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: repeat_kv k",
                e,
            )
        })?;
        let v = repeat_kv(&v, self.num_kv_groups).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: repeat_kv v",
                e,
            )
        })?;

        let attn = scaled_dot_product_attention(&q, &k, &v, attention_mask, self.scaling, true)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text attention", e))?;

        let attn = attn
            .transpose(1, 2)
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: attn transpose",
                    e,
                )
            })?
            .reshape((b, seq_len, self.num_heads * self.head_dim))
            .map_err(|e| {
                candle_to_ocr_processing(
                    oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                    "GLM-OCR: attn reshape",
                    e,
                )
            })?;

        self.o_proj
            .forward(&attn)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text o_proj forward", e))
    }

    fn clear_kv_cache(&self) {
        self.kv_cache.borrow_mut().reset();
    }
}

#[derive(Debug, Clone)]
struct GlmOcrTextDecoderLayer {
    self_attn: GlmOcrTextAttention,
    mlp: GlmOcrTextMLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
    post_self_attn_layernorm: RmsNorm,
    post_mlp_layernorm: RmsNorm,
}

impl GlmOcrTextDecoderLayer {
    fn load(cfg: &GlmOcrTextConfig, vb: VarBuilder) -> Result<Self, OCRError> {
        let self_attn = GlmOcrTextAttention::load(cfg, vb.clone())?;
        let mlp = GlmOcrTextMLP::load(cfg, vb.clone())?;
        let input_layernorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text input_layernorm", e))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )
        .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text post_attention_layernorm", e))?;
        let post_self_attn_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_self_attn_layernorm"),
        )
        .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text post_self_attn_layernorm", e))?;
        let post_mlp_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_mlp_layernorm"),
        )
        .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text post_mlp_layernorm", e))?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
            post_self_attn_layernorm,
            post_mlp_layernorm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, OCRError> {
        let residual = hidden_states.clone();
        let hidden = self
            .input_layernorm
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text input_layernorm forward", e))?;
        let hidden = self.self_attn.forward(&hidden, cos, sin, attention_mask)?;
        let hidden = self
            .post_self_attn_layernorm
            .forward(&hidden)
            .map_err(|e| {
                candle_to_ocr_inference("GLM-OCR", "text post_self_attn_layernorm forward", e)
            })?;
        let hidden = (&residual + &hidden).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: text residual add",
                e,
            )
        })?;

        let residual = hidden.clone();
        let hidden = self
            .post_attention_layernorm
            .forward(&hidden)
            .map_err(|e| {
                candle_to_ocr_inference("GLM-OCR", "text post_attention_layernorm forward", e)
            })?;
        let hidden = self.mlp.forward(&hidden)?;
        let hidden = self.post_mlp_layernorm.forward(&hidden).map_err(|e| {
            candle_to_ocr_inference("GLM-OCR", "text post_mlp_layernorm forward", e)
        })?;
        (&residual + &hidden).map_err(|e| {
            candle_to_ocr_processing(
                oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
                "GLM-OCR: text residual add mlp",
                e,
            )
        })
    }

    fn clear_kv_cache(&self) {
        self.self_attn.clear_kv_cache();
    }
}

#[derive(Debug, Clone)]
pub struct GlmOcrTextModel {
    embed_tokens: Embedding,
    layers: Vec<GlmOcrTextDecoderLayer>,
    norm: RmsNorm,
    rotary_emb: GlmOcrTextRotaryEmbedding,
}

impl GlmOcrTextModel {
    pub fn load(cfg: &GlmOcrTextConfig, vb: VarBuilder) -> Result<Self, OCRError> {
        let embed_tokens = embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text embed_tokens", e))?;
        let rotary_emb = GlmOcrTextRotaryEmbedding::new(cfg, vb.device())?;

        let vb_layers = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for idx in 0..cfg.num_hidden_layers {
            layers.push(GlmOcrTextDecoderLayer::load(cfg, vb_layers.pp(idx))?);
        }
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text norm", e))?;

        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary_emb,
        })
    }

    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor, OCRError> {
        self.embed_tokens
            .forward(input_ids)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text embed forward", e))
    }

    pub fn token_embedding_weight(&self) -> Tensor {
        self.embed_tokens.embeddings().clone()
    }

    pub fn forward(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor, OCRError> {
        let (cos, sin) = self.rotary_emb.forward(inputs_embeds, position_ids)?;
        let mut hidden_states = inputs_embeds.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, &cos, &sin, attention_mask)?;
        }
        self.norm
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference("GLM-OCR", "text norm forward", e))
    }

    pub fn clear_kv_cache(&self) {
        for layer in &self.layers {
            layer.clear_kv_cache();
        }
    }
}
