//! DFlash parallel draft model for HunyuanOCR 1.5.
//!
//! The draft consumes intermediate features from the target decoder as cached
//! context K/V. Its queries are one target-produced bonus token followed by a
//! block of mask tokens. All mask positions are predicted in one non-causal
//! pass and are then verified by the target model in one causal pass.

use crate::attention::{RotaryEmbedding, repeat_kv, scaled_dot_product_attention};
use crate::utils::{candle_to_ocr_inference, candle_to_ocr_processing, rotate_half};
use candle_core::{DType, Device, Tensor};
use candle_nn::{Linear, Module, RmsNorm, VarBuilder, linear_no_bias, rms_norm};
use oar_ocr_core::core::OCRError;
use serde::Deserialize;
use std::cell::RefCell;
use std::path::Path;

fn tensor_err(message: &'static str, error: candle_core::Error) -> OCRError {
    candle_to_ocr_processing(
        oar_ocr_core::core::errors::ProcessingStage::TensorOperation,
        message,
        error,
    )
}

#[derive(Debug, Clone, Deserialize)]
pub struct DFlashTargetConfig {
    pub mask_token_id: u32,
    pub target_layer_ids: Vec<usize>,
}

/// Configuration stored in `dflash/config.json`.
#[derive(Debug, Clone, Deserialize)]
pub struct DFlashConfig {
    pub block_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_attention_heads: usize,
    pub num_hidden_layers: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
    pub dflash_config: DFlashTargetConfig,
}

impl DFlashConfig {
    pub fn from_path(path: impl AsRef<Path>) -> Result<Self, OCRError> {
        crate::utils::load_json_config(path, "HunyuanOCR DFlash", "config.json")
    }

    fn validate(&self) -> Result<(), OCRError> {
        if self.block_size == 0
            || self.num_hidden_layers == 0
            || self.dflash_config.target_layer_ids.is_empty()
        {
            return Err(OCRError::ConfigError {
                message:
                    "HunyuanOCR DFlash: block size, layer count, and target layers must be non-zero"
                        .to_string(),
            });
        }
        if !self
            .num_attention_heads
            .is_multiple_of(self.num_key_value_heads)
        {
            return Err(OCRError::ConfigError {
                message: format!(
                    "HunyuanOCR DFlash: {} attention heads is not divisible by {} KV heads",
                    self.num_attention_heads, self.num_key_value_heads
                ),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Default)]
struct ContextKv {
    k: Option<Tensor>,
    v: Option<Tensor>,
    len: usize,
}

impl ContextKv {
    fn reset(&mut self) {
        self.k = None;
        self.v = None;
        self.len = 0;
    }

    fn append(&mut self, k: &Tensor, v: &Tensor) -> Result<(), OCRError> {
        let added = k
            .dim(2)
            .map_err(|e| tensor_err("HunyuanOCR DFlash: context K length", e))?;
        let (new_k, new_v) = match (&self.k, &self.v) {
            (Some(old_k), Some(old_v)) => (
                Tensor::cat(&[old_k, k], 2)
                    .and_then(|x| x.contiguous())
                    .map_err(|e| tensor_err("HunyuanOCR DFlash: append context K", e))?,
                Tensor::cat(&[old_v, v], 2)
                    .and_then(|x| x.contiguous())
                    .map_err(|e| tensor_err("HunyuanOCR DFlash: append context V", e))?,
            ),
            _ => (k.clone(), v.clone()),
        };
        self.k = Some(new_k);
        self.v = Some(new_v);
        self.len += added;
        Ok(())
    }
}

fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor, OCRError> {
    let a = x
        .broadcast_mul(cos)
        .map_err(|e| tensor_err("HunyuanOCR DFlash: rope x*cos", e))?;
    let b = rotate_half(x)?
        .broadcast_mul(sin)
        .map_err(|e| tensor_err("HunyuanOCR DFlash: rope rotate_half*sin", e))?;
    (a + b).map_err(|e| tensor_err("HunyuanOCR DFlash: rope sum", e))
}

#[derive(Debug)]
struct DFlashAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    q_norm: RmsNorm,
    k_norm: RmsNorm,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    scale: f64,
}

impl DFlashAttention {
    fn load(cfg: &DFlashConfig, vb: VarBuilder) -> Result<Self, OCRError> {
        let q_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_attention_heads * cfg.head_dim,
            vb.pp("q_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "load q_proj", e))?;
        let k_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            vb.pp("k_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "load k_proj", e))?;
        let v_proj = linear_no_bias(
            cfg.hidden_size,
            cfg.num_key_value_heads * cfg.head_dim,
            vb.pp("v_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "load v_proj", e))?;
        let o_proj = linear_no_bias(
            cfg.num_attention_heads * cfg.head_dim,
            cfg.hidden_size,
            vb.pp("o_proj"),
        )
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "load o_proj", e))?;
        let q_norm = rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("q_norm"))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "load q_norm", e))?;
        let k_norm = rms_norm(cfg.head_dim, cfg.rms_norm_eps, vb.pp("k_norm"))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "load k_norm", e))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            q_norm,
            k_norm,
            num_heads: cfg.num_attention_heads,
            num_kv_heads: cfg.num_key_value_heads,
            num_kv_groups: cfg.num_attention_heads / cfg.num_key_value_heads,
            head_dim: cfg.head_dim,
            scale: (cfg.head_dim as f64).powf(-0.5),
        })
    }

    fn project(
        &self,
        projection: &Linear,
        input: &Tensor,
        heads: usize,
        norm: Option<&RmsNorm>,
    ) -> Result<Tensor, OCRError> {
        let (batch, seq_len, _) = input
            .dims3()
            .map_err(|e| tensor_err("HunyuanOCR DFlash: projection input dims", e))?;
        let projected = projection
            .forward(input)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "attention projection", e))?
            .reshape((batch, seq_len, heads, self.head_dim))
            .map_err(|e| tensor_err("HunyuanOCR DFlash: projection reshape", e))?;
        let projected = match norm {
            Some(norm) => norm
                .forward(&projected)
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "Q/K norm", e))?,
            None => projected,
        };
        projected
            .transpose(1, 2)
            .and_then(|x| x.contiguous())
            .map_err(|e| tensor_err("HunyuanOCR DFlash: projection transpose", e))
    }

    fn append_context(
        &self,
        target_hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cache: &mut ContextKv,
    ) -> Result<(), OCRError> {
        let k = self.project(
            &self.k_proj,
            target_hidden,
            self.num_kv_heads,
            Some(&self.k_norm),
        )?;
        let k = apply_rope(&k, cos, sin)?;
        let v = self.project(&self.v_proj, target_hidden, self.num_kv_heads, None)?;
        cache.append(&k, &v)
    }

    fn forward(
        &self,
        hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cache: &ContextKv,
    ) -> Result<Tensor, OCRError> {
        let q = self.project(&self.q_proj, hidden, self.num_heads, Some(&self.q_norm))?;
        let q = apply_rope(&q, cos, sin)?;
        let query_k = self.project(&self.k_proj, hidden, self.num_kv_heads, Some(&self.k_norm))?;
        let query_k = apply_rope(&query_k, cos, sin)?;
        let query_v = self.project(&self.v_proj, hidden, self.num_kv_heads, None)?;

        let (Some(context_k), Some(context_v)) = (&cache.k, &cache.v) else {
            return Err(OCRError::InvalidInput {
                message: "HunyuanOCR DFlash: context cache is empty".to_string(),
            });
        };
        let k = Tensor::cat(&[context_k, &query_k], 2)
            .and_then(|x| x.contiguous())
            .map_err(|e| tensor_err("HunyuanOCR DFlash: concatenate attention K", e))?;
        let v = Tensor::cat(&[context_v, &query_v], 2)
            .and_then(|x| x.contiguous())
            .map_err(|e| tensor_err("HunyuanOCR DFlash: concatenate attention V", e))?;
        let k = repeat_kv(&k, self.num_kv_groups)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "repeat K", e))?;
        let v = repeat_kv(&v, self.num_kv_groups)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "repeat V", e))?;
        let output = scaled_dot_product_attention(&q, &k, &v, None, self.scale, false)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "attention", e))?;
        let (batch, _, query_len, _) = output
            .dims4()
            .map_err(|e| tensor_err("HunyuanOCR DFlash: attention output dims", e))?;
        let output = output
            .transpose(1, 2)
            .and_then(|x| x.reshape((batch, query_len, self.num_heads * self.head_dim)))
            .map_err(|e| tensor_err("HunyuanOCR DFlash: attention output reshape", e))?;
        self.o_proj
            .forward(&output)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "o_proj", e))
    }
}

#[derive(Debug)]
struct DFlashMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
}

impl DFlashMlp {
    fn load(cfg: &DFlashConfig, vb: VarBuilder) -> Result<Self, OCRError> {
        let gate_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("gate_proj"))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "load gate_proj", e))?;
        let up_proj = linear_no_bias(cfg.hidden_size, cfg.intermediate_size, vb.pp("up_proj"))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "load up_proj", e))?;
        let down_proj = linear_no_bias(cfg.intermediate_size, cfg.hidden_size, vb.pp("down_proj"))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "load down_proj", e))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
        })
    }

    fn forward(&self, input: &Tensor) -> Result<Tensor, OCRError> {
        let gate = self
            .gate_proj
            .forward(input)
            .and_then(|x| candle_nn::ops::silu(&x))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "MLP gate", e))?;
        let up = self
            .up_proj
            .forward(input)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "MLP up", e))?;
        let hidden = (gate * up).map_err(|e| tensor_err("HunyuanOCR DFlash: MLP multiply", e))?;
        self.down_proj
            .forward(&hidden)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "MLP down", e))
    }
}

#[derive(Debug)]
struct DFlashLayer {
    input_layernorm: RmsNorm,
    self_attn: DFlashAttention,
    post_attention_layernorm: RmsNorm,
    mlp: DFlashMlp,
}

impl DFlashLayer {
    fn load(cfg: &DFlashConfig, vb: VarBuilder) -> Result<Self, OCRError> {
        let input_layernorm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "load input_layernorm", e))?;
        let post_attention_layernorm = rms_norm(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )
        .map_err(|e| {
            candle_to_ocr_inference("HunyuanOCR DFlash", "load post_attention_layernorm", e)
        })?;
        Ok(Self {
            input_layernorm,
            self_attn: DFlashAttention::load(cfg, vb.pp("self_attn"))?,
            post_attention_layernorm,
            mlp: DFlashMlp::load(cfg, vb.pp("mlp"))?,
        })
    }

    fn forward(
        &self,
        hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        cache: &ContextKv,
    ) -> Result<Tensor, OCRError> {
        let normed = self
            .input_layernorm
            .forward(hidden)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "input norm", e))?;
        let attention = self.self_attn.forward(&normed, cos, sin, cache)?;
        let hidden = (hidden + attention)
            .map_err(|e| tensor_err("HunyuanOCR DFlash: attention residual", e))?;
        let normed = self
            .post_attention_layernorm
            .forward(&hidden)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "post-attention norm", e))?;
        let mlp = self.mlp.forward(&normed)?;
        (hidden + mlp).map_err(|e| tensor_err("HunyuanOCR DFlash: MLP residual", e))
    }
}

/// Loaded DFlash draft and its incremental target-context K/V caches.
pub(crate) struct DFlashModel {
    cfg: DFlashConfig,
    fc: Linear,
    hidden_norm: RmsNorm,
    layers: Vec<DFlashLayer>,
    norm: RmsNorm,
    rotary: RotaryEmbedding,
    caches: RefCell<Vec<ContextKv>>,
    dtype: DType,
    device: Device,
}

impl DFlashModel {
    pub(crate) fn from_dir(
        model_dir: impl AsRef<Path>,
        dtype: DType,
        device: &Device,
    ) -> Result<Self, OCRError> {
        let model_dir = model_dir.as_ref();
        let cfg = DFlashConfig::from_path(model_dir.join("config.json"))?;
        cfg.validate()?;
        let files = crate::utils::collect_safetensors(model_dir, "HunyuanOCR DFlash")?;
        // SAFETY: the checkpoint files remain mapped for the lifetime of the tensors.
        let vb = unsafe {
            VarBuilder::from_mmaped_safetensors(&files, dtype, device)
                .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "load safetensors", e))?
        };
        let fc = linear_no_bias(
            cfg.hidden_size * cfg.dflash_config.target_layer_ids.len(),
            cfg.hidden_size,
            vb.pp("fc"),
        )
        .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "load fc", e))?;
        let hidden_norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("hidden_norm"))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "load hidden_norm", e))?;
        let norm = rms_norm(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("norm"))
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "load norm", e))?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for index in 0..cfg.num_hidden_layers {
            layers.push(DFlashLayer::load(&cfg, vb.pp(format!("layers.{index}")))?);
        }
        let rotary = RotaryEmbedding::new_dynamic(cfg.head_dim, cfg.rope_theta, device)?;
        let caches = (0..cfg.num_hidden_layers)
            .map(|_| ContextKv::default())
            .collect();
        Ok(Self {
            cfg,
            fc,
            hidden_norm,
            layers,
            norm,
            rotary,
            caches: RefCell::new(caches),
            dtype,
            device: device.clone(),
        })
    }

    pub(crate) fn config(&self) -> &DFlashConfig {
        &self.cfg
    }

    fn rope(&self, start: usize, len: usize) -> Result<(Tensor, Tensor), OCRError> {
        let positions: Vec<i64> = (start..start + len).map(|x| x as i64).collect();
        let positions = Tensor::from_vec(positions, (1, 1, len), &self.device)
            .map_err(|e| tensor_err("HunyuanOCR DFlash: position tensor", e))?;
        self.rotary.forward_multi_axis(&positions, self.dtype)
    }

    fn transform_target(&self, aux_hidden: &Tensor) -> Result<Tensor, OCRError> {
        let hidden = self
            .fc
            .forward(aux_hidden)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "target fc", e))?;
        self.hidden_norm
            .forward(&hidden)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "target hidden norm", e))
    }

    pub(crate) fn reset_context(&self, aux_hidden: &Tensor) -> Result<(), OCRError> {
        let len = aux_hidden
            .dim(1)
            .map_err(|e| tensor_err("HunyuanOCR DFlash: target context length", e))?;
        let target = self.transform_target(aux_hidden)?;
        let (cos, sin) = self.rope(0, len)?;
        let mut caches = self.caches.borrow_mut();
        for cache in caches.iter_mut() {
            cache.reset();
        }
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            layer.self_attn.append_context(&target, &cos, &sin, cache)?;
        }
        Ok(())
    }

    pub(crate) fn append_context(&self, aux_hidden: &Tensor) -> Result<(), OCRError> {
        let added = aux_hidden
            .dim(1)
            .map_err(|e| tensor_err("HunyuanOCR DFlash: appended context length", e))?;
        if added == 0 {
            return Ok(());
        }
        let start = self.context_len();
        let target = self.transform_target(aux_hidden)?;
        let (cos, sin) = self.rope(start, added)?;
        let mut caches = self.caches.borrow_mut();
        for (layer, cache) in self.layers.iter().zip(caches.iter_mut()) {
            layer.self_attn.append_context(&target, &cos, &sin, cache)?;
        }
        Ok(())
    }

    pub(crate) fn context_len(&self) -> usize {
        self.caches.borrow().first().map_or(0, |cache| cache.len)
    }

    /// Run the bonus+mask query block. Returns post-norm hidden states for all
    /// query positions; the caller samples only rows `1..` (the mask rows).
    pub(crate) fn forward_queries(&self, query_embeds: &Tensor) -> Result<Tensor, OCRError> {
        let query_len = query_embeds
            .dim(1)
            .map_err(|e| tensor_err("HunyuanOCR DFlash: query length", e))?;
        let context_len = self.context_len();
        let (cos, sin) = self.rope(context_len, query_len)?;
        let caches = self.caches.borrow();
        let mut hidden = query_embeds.clone();
        for (layer, cache) in self.layers.iter().zip(caches.iter()) {
            hidden = layer.forward(&hidden, &cos, &sin, cache)?;
        }
        self.norm
            .forward(&hidden)
            .map_err(|e| candle_to_ocr_inference("HunyuanOCR DFlash", "final norm", e))
    }

    pub(crate) fn clear_context(&self) {
        for cache in self.caches.borrow_mut().iter_mut() {
            cache.reset();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::DFlashConfig;

    #[test]
    fn parses_hunyuan_dflash_config() {
        let cfg: DFlashConfig = serde_json::from_str(
            r#"{
                "block_size": 16,
                "hidden_size": 1024,
                "intermediate_size": 3584,
                "num_attention_heads": 16,
                "num_hidden_layers": 5,
                "num_key_value_heads": 8,
                "head_dim": 128,
                "vocab_size": 120818,
                "rms_norm_eps": 0.00001,
                "rope_theta": 10000.0,
                "dflash_config": {
                    "mask_token_id": 120817,
                    "target_layer_ids": [1, 8, 15, 22]
                }
            }"#,
        )
        .unwrap();
        assert_eq!(cfg.block_size, 16);
        assert_eq!(cfg.dflash_config.target_layer_ids, [1, 8, 15, 22]);
        cfg.validate().unwrap();
    }
}
