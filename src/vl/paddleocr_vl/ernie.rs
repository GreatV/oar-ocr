use super::config::PaddleOcrVlConfig;
use crate::core::OCRError;
use crate::vl::utils::{candle_to_ocr_inference, candle_to_ocr_processing, rotate_half};
use candle_core::{D, DType, Device, IndexOp, Tensor};
use candle_nn::Module;

#[derive(Debug, Default, Clone)]
pub struct KvCache {
    pub key: Option<Tensor>,
    pub value: Option<Tensor>,
}

#[derive(Debug, Clone)]
pub struct RotaryEmbedding {
    pub inv_freq: Tensor,
}

impl RotaryEmbedding {
    pub fn new(head_dim: usize, rope_theta: f64, device: &Device) -> Result<Self, OCRError> {
        if !head_dim.is_multiple_of(2) {
            return Err(OCRError::config_error(format!(
                "RotaryEmbedding: head_dim must be even, got {head_dim}"
            )));
        }
        let half = head_dim / 2;
        let mut inv_freq = Vec::with_capacity(half);
        for i in (0..head_dim).step_by(2) {
            let v = 1f64 / rope_theta.powf(i as f64 / head_dim as f64);
            inv_freq.push(v as f32);
        }
        let inv_freq = Tensor::from_vec(inv_freq, (half,), device).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "RotaryEmbedding: failed to create inv_freq tensor",
                e,
            )
        })?;
        Ok(Self { inv_freq })
    }

    pub fn forward(
        &self,
        position_ids: &Tensor,
        dtype: DType,
    ) -> Result<(Tensor, Tensor), OCRError> {
        // position_ids: (3, batch, seq)
        let position_ids = position_ids.to_dtype(DType::F32).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "RotaryEmbedding: position_ids cast to f32 failed",
                e,
            )
        })?;
        let inv_len = self.inv_freq.dims1().map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "RotaryEmbedding: inv_freq dims1 failed",
                e,
            )
        })?;
        let inv = self
            .inv_freq
            .reshape((1usize, 1usize, 1usize, inv_len))
            .map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "RotaryEmbedding: inv_freq reshape failed",
                    e,
                )
            })?;
        let freqs = position_ids
            .unsqueeze(3)
            .map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "RotaryEmbedding: position_ids unsqueeze failed",
                    e,
                )
            })?
            .broadcast_mul(&inv)
            .map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "RotaryEmbedding: rotary freqs multiply failed",
                    e,
                )
            })?;
        let emb = Tensor::cat(&[&freqs, &freqs], candle_core::D::Minus1).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "RotaryEmbedding: rotary emb cat failed",
                e,
            )
        })?;
        let cos = emb
            .cos()
            .map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "RotaryEmbedding: rotary cos failed",
                    e,
                )
            })?
            .to_dtype(dtype)
            .map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "RotaryEmbedding: rotary cos cast failed",
                    e,
                )
            })?;
        let sin = emb
            .sin()
            .map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "RotaryEmbedding: rotary sin failed",
                    e,
                )
            })?
            .to_dtype(dtype)
            .map_err(|e| {
                candle_to_ocr_processing(
                    crate::core::errors::ProcessingStage::TensorOperation,
                    "RotaryEmbedding: rotary sin cast failed",
                    e,
                )
            })?;
        Ok((cos, sin))
    }
}

fn select_mrope(cos_or_sin: &Tensor, mrope_section: &[usize]) -> Result<Tensor, OCRError> {
    // Input: (3, batch, seq, head_dim) -> Output: (batch, 1, seq, head_dim)
    if mrope_section.is_empty() {
        return Err(OCRError::config_error(
            "PaddleOCR-VL: rope_scaling.mrope_section is empty",
        ));
    }

    // Validate that mrope_section sums to half of head_dim (since we double it)
    let dims = cos_or_sin.dims();
    let head_dim = dims.get(3).copied().unwrap_or(0);
    let section_sum: usize = mrope_section.iter().sum();
    if section_sum * 2 != head_dim {
        return Err(OCRError::config_error(format!(
            "PaddleOCR-VL: mrope_section sum ({}) * 2 != head_dim ({})",
            section_sum, head_dim
        )));
    }

    // Duplicate the list like Python: mrope_section * 2
    let doubled_sections: Vec<usize> = mrope_section
        .iter()
        .chain(mrope_section.iter())
        .copied()
        .collect();

    let mut offset = 0usize;
    let mut chunks: Vec<Tensor> = Vec::with_capacity(doubled_sections.len());
    for (i, &sec) in doubled_sections.iter().enumerate() {
        let next = offset + sec;
        let seg = cos_or_sin.i((.., .., .., offset..next)).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                format!(
                    "PaddleOCR-VL: mrope slice failed at chunk {} (offset {}..{})",
                    i, offset, next
                ),
                e,
            )
        })?;
        let picked = seg.i((i % 3, .., .., ..)).map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                format!(
                    "PaddleOCR-VL: mrope pick failed at chunk {} (dim {})",
                    i,
                    i % 3
                ),
                e,
            )
        })?;
        chunks.push(picked);
        offset = next;
    }
    let refs: Vec<&Tensor> = chunks.iter().collect();
    let cat = Tensor::cat(&refs, D::Minus1).map_err(|e| {
        candle_to_ocr_processing(
            crate::core::errors::ProcessingStage::TensorOperation,
            "PaddleOCR-VL: mrope cat failed",
            e,
        )
    })?;
    cat.unsqueeze(1).map_err(|e| {
        candle_to_ocr_processing(
            crate::core::errors::ProcessingStage::TensorOperation,
            "PaddleOCR-VL: mrope unsqueeze failed",
            e,
        )
    })
}

fn apply_multimodal_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
    mrope_section: &[usize],
) -> Result<(Tensor, Tensor), OCRError> {
    let cos = select_mrope(cos, mrope_section)?;
    let sin = select_mrope(sin, mrope_section)?;
    let q_mul = q.broadcast_mul(&cos).map_err(|e| {
        candle_to_ocr_processing(
            crate::core::errors::ProcessingStage::TensorOperation,
            "PaddleOCR-VL: mrope q*cos failed",
            e,
        )
    })?;
    let q_half = rotate_half(q)?;
    let q_half_mul = q_half.broadcast_mul(&sin).map_err(|e| {
        candle_to_ocr_processing(
            crate::core::errors::ProcessingStage::TensorOperation,
            "PaddleOCR-VL: mrope rotate_half(q)*sin failed",
            e,
        )
    })?;
    let q_rot = (&q_mul + &q_half_mul).map_err(|e| {
        candle_to_ocr_processing(
            crate::core::errors::ProcessingStage::TensorOperation,
            "PaddleOCR-VL: mrope apply on q failed",
            e,
        )
    })?;

    let k_mul = k.broadcast_mul(&cos).map_err(|e| {
        candle_to_ocr_processing(
            crate::core::errors::ProcessingStage::TensorOperation,
            "PaddleOCR-VL: mrope k*cos failed",
            e,
        )
    })?;
    let k_half = rotate_half(k)?;
    let k_half_mul = k_half.broadcast_mul(&sin).map_err(|e| {
        candle_to_ocr_processing(
            crate::core::errors::ProcessingStage::TensorOperation,
            "PaddleOCR-VL: mrope rotate_half(k)*sin failed",
            e,
        )
    })?;
    let k_rot = (&k_mul + &k_half_mul).map_err(|e| {
        candle_to_ocr_processing(
            crate::core::errors::ProcessingStage::TensorOperation,
            "PaddleOCR-VL: mrope apply on k failed",
            e,
        )
    })?;
    Ok((q_rot, k_rot))
}

fn repeat_kv(hidden_states: &Tensor, n_rep: usize) -> Result<Tensor, OCRError> {
    if n_rep == 1 {
        return Ok(hidden_states.clone());
    }
    let (b, kv_heads, slen, head_dim) = hidden_states.dims4().map_err(|e| {
        candle_to_ocr_processing(
            crate::core::errors::ProcessingStage::TensorOperation,
            "PaddleOCR-VL: repeat_kv dims failed",
            e,
        )
    })?;
    hidden_states
        .unsqueeze(2)
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: repeat_kv unsqueeze failed",
                e,
            )
        })?
        .expand((b, kv_heads, n_rep, slen, head_dim))
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: repeat_kv expand failed",
                e,
            )
        })?
        .reshape((b, kv_heads * n_rep, slen, head_dim))
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::core::errors::ProcessingStage::TensorOperation,
                "PaddleOCR-VL: repeat_kv reshape failed",
                e,
            )
        })
}

#[derive(Debug, Clone)]
struct Ernie4_5Mlp {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
}

impl Ernie4_5Mlp {
    fn load(cfg: &PaddleOcrVlConfig, vb: candle_nn::VarBuilder) -> Result<Self, OCRError> {
        let gate_proj = candle_nn::linear_b(
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.use_bias,
            vb.pp("gate_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load gate_proj", e))?;
        let up_proj = candle_nn::linear_b(
            cfg.hidden_size,
            cfg.intermediate_size,
            cfg.use_bias,
            vb.pp("up_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load up_proj", e))?;
        let down_proj = candle_nn::linear_b(
            cfg.intermediate_size,
            cfg.hidden_size,
            cfg.use_bias,
            vb.pp("down_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load down_proj", e))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, OCRError> {
        let gate = self
            .gate_proj
            .forward(xs)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "mlp gate_proj", e))?;
        let gate = candle_nn::ops::silu(&gate)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "mlp silu", e))?;
        let up = self
            .up_proj
            .forward(xs)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "mlp up_proj", e))?;
        let prod =
            (&gate * &up).map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "mlp gate*up", e))?;
        self.down_proj
            .forward(&prod)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "mlp down_proj", e))
    }
}

#[derive(Debug, Clone)]
struct Ernie4_5Attention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scaling: f64,
    mrope_section: Vec<usize>,
}

impl Ernie4_5Attention {
    fn load(cfg: &PaddleOcrVlConfig, vb: candle_nn::VarBuilder) -> Result<Self, OCRError> {
        let q_proj = candle_nn::linear_b(
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.head_dim,
            cfg.use_bias,
            vb.pp("q_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load q_proj", e))?;
        let k_proj = candle_nn::linear_b(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            cfg.use_bias,
            vb.pp("k_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load k_proj", e))?;
        let v_proj = candle_nn::linear_b(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            cfg.use_bias,
            vb.pp("v_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load v_proj", e))?;
        let o_proj = candle_nn::linear_b(
            cfg.num_attention_heads * cfg.head_dim,
            cfg.hidden_size,
            cfg.use_bias,
            vb.pp("o_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load o_proj", e))?;

        if !cfg
            .num_attention_heads
            .is_multiple_of(cfg.num_key_value_heads)
        {
            return Err(OCRError::config_error(format!(
                "PaddleOCR-VL: num_attention_heads ({}) must be divisible by num_key_value_heads ({})",
                cfg.num_attention_heads, cfg.num_key_value_heads
            )));
        }

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            scaling: (cfg.head_dim as f64).powf(-0.5),
            mrope_section: cfg.rope_scaling.mrope_section.clone(),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cache: Option<&mut KvCache>,
        causal_mask: Option<&Tensor>,
    ) -> Result<Tensor, OCRError> {
        let (b, seq_len, _) = hidden_states
            .dims3()
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn hidden_states dims3", e))?;

        let q = self
            .q_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn q_proj", e))?
            .reshape((b, seq_len, self.num_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn q reshape", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn q transpose", e))?;

        let k = self
            .k_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn k_proj", e))?
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn k reshape", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn k transpose", e))?;

        let v = self
            .v_proj
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn v_proj", e))?
            .reshape((b, seq_len, self.num_kv_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn v reshape", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn v transpose", e))?;

        let (q, k) = apply_multimodal_rotary_pos_emb(&q, &k, cos, sin, &self.mrope_section)?;

        let (k_all, v_all) = match cache {
            None => (k, v),
            Some(cache) => {
                let (k_all, v_all) = match (&cache.key, &cache.value) {
                    (Some(k_prev), Some(v_prev)) => {
                        let k_all = Tensor::cat(&[k_prev, &k], 2).map_err(|e| {
                            candle_to_ocr_inference("PaddleOCR-VL", "attn cat k cache", e)
                        })?;
                        let v_all = Tensor::cat(&[v_prev, &v], 2).map_err(|e| {
                            candle_to_ocr_inference("PaddleOCR-VL", "attn cat v cache", e)
                        })?;
                        (k_all, v_all)
                    }
                    _ => (k.clone(), v.clone()),
                };
                cache.key = Some(k_all.clone());
                cache.value = Some(v_all.clone());
                (k_all, v_all)
            }
        };

        let key_states = repeat_kv(&k_all, self.num_kv_groups)?;
        let value_states = repeat_kv(&v_all, self.num_kv_groups)?;

        // Make tensors contiguous for matmul
        let q = q
            .contiguous()
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn q contiguous", e))?;
        let key_states = key_states.contiguous().map_err(|e| {
            candle_to_ocr_inference("PaddleOCR-VL", "attn key_states contiguous", e)
        })?;
        let value_states = value_states.contiguous().map_err(|e| {
            candle_to_ocr_inference("PaddleOCR-VL", "attn value_states contiguous", e)
        })?;

        let attn_weights = q
            .matmul(
                &key_states
                    .transpose(2, 3)
                    .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn k transpose23", e))?
                    .contiguous()
                    .map_err(|e| {
                        candle_to_ocr_inference("PaddleOCR-VL", "attn k t23 contiguous", e)
                    })?,
            )
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn qk matmul", e))?
            .affine(self.scaling, 0.0)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn scaling", e))?;

        let attn_weights = match causal_mask {
            None => attn_weights,
            Some(mask) => attn_weights
                .broadcast_add(mask)
                .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn add causal mask", e))?,
        };

        let attn_weights =
            candle_nn::ops::softmax_last_dim(&attn_weights.to_dtype(DType::F32).map_err(|e| {
                candle_to_ocr_inference("PaddleOCR-VL", "attn weights cast f32", e)
            })?)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn softmax", e))?
            .to_dtype(value_states.dtype())
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn weights cast back", e))?
            .contiguous()
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn weights contiguous", e))?;

        let attn_output = attn_weights
            .matmul(&value_states)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn av matmul", e))?
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn out transpose", e))?
            .reshape((b, seq_len, self.num_heads * self.head_dim))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn out reshape", e))?;

        self.o_proj
            .forward(&attn_output)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "attn o_proj", e))
    }
}

#[derive(Debug, Clone)]
struct Ernie4_5DecoderLayer {
    self_attn: Ernie4_5Attention,
    mlp: Ernie4_5Mlp,
    input_layernorm: candle_nn::RmsNorm,
    post_attention_layernorm: candle_nn::RmsNorm,
}

impl Ernie4_5DecoderLayer {
    fn load(cfg: &PaddleOcrVlConfig, vb: candle_nn::VarBuilder) -> Result<Self, OCRError> {
        let self_attn = Ernie4_5Attention::load(cfg, vb.pp("self_attn"))?;
        let mlp = Ernie4_5Mlp::load(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))
                .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load input_layernorm", e))?;
        let post_attention_layernorm = candle_nn::rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )
        .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load post_attention_layernorm", e))?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cache: Option<&mut KvCache>,
        causal_mask: Option<&Tensor>,
    ) -> Result<Tensor, OCRError> {
        let residual = hidden_states.clone();
        let hidden_states = self
            .input_layernorm
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "llm input_layernorm", e))?;
        let attn_out = self
            .self_attn
            .forward(&hidden_states, cos, sin, cache, causal_mask)?;
        let hidden_states = (&residual + &attn_out)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "llm attn residual add", e))?;

        let residual = hidden_states.clone();
        let hidden_states = self
            .post_attention_layernorm
            .forward(&hidden_states)
            .map_err(|e| {
                candle_to_ocr_inference("PaddleOCR-VL", "llm post_attention_layernorm", e)
            })?;
        let mlp_out = self.mlp.forward(&hidden_states)?;
        (&residual + &mlp_out)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "llm mlp residual add", e))
    }
}

#[derive(Debug, Clone)]
pub struct Ernie4_5Model {
    embed_tokens: candle_nn::Embedding,
    layers: Vec<Ernie4_5DecoderLayer>,
    norm: candle_nn::RmsNorm,
    rotary: RotaryEmbedding,
}

impl Ernie4_5Model {
    pub fn load(cfg: &PaddleOcrVlConfig, vb: candle_nn::VarBuilder) -> Result<Self, OCRError> {
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb.pp("embed_tokens"))
                .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load embed_tokens", e))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(Ernie4_5DecoderLayer::load(
                cfg,
                vb.pp(format!("layers.{i}")),
            )?);
        }
        let norm = candle_nn::rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "load final norm", e))?;
        let rotary = RotaryEmbedding::new(cfg.head_dim, cfg.rope_theta, vb.device())?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            rotary,
        })
    }

    pub fn embed(&self, input_ids: &Tensor) -> Result<Tensor, OCRError> {
        self.embed_tokens
            .forward(input_ids)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "embed tokens", e))
    }

    pub fn forward(
        &self,
        inputs_embeds: &Tensor,
        position_ids: &Tensor,
        kv_cache: Option<&mut [KvCache]>,
        causal_mask: Option<&Tensor>,
    ) -> Result<Tensor, OCRError> {
        let (cos, sin) = self.rotary.forward(position_ids, inputs_embeds.dtype())?;

        let mut hidden_states = inputs_embeds.clone();
        let mut kv_cache = kv_cache;
        for (idx, layer) in self.layers.iter().enumerate() {
            let cache = kv_cache.as_deref_mut().and_then(|c| c.get_mut(idx));
            hidden_states = layer.forward(&hidden_states, &cos, &sin, cache, causal_mask)?;
        }
        let hidden_states = self
            .norm
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference("PaddleOCR-VL", "llm final norm", e))?;
        Ok(hidden_states)
    }
}
