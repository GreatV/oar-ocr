//! M2M100-style decoder implementation for UniRec.
//!
//! Based on the M2M100 (Multilingual Translation) decoder architecture
//! with learned positional embeddings and scaled word embeddings.

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{Dropout, Embedding, LayerNorm, Linear, Module, VarBuilder};

use super::config::UniRecConfig;
use crate::core::OCRError;
use crate::vl::utils::candle_to_ocr_inference;

/// KV cache for decoder attention layers.
#[derive(Debug, Default, Clone)]
pub struct KvCache {
    pub self_key: Option<Tensor>,
    pub self_value: Option<Tensor>,
    pub cross_key: Option<Tensor>,
    pub cross_value: Option<Tensor>,
}

/// Scaled word embedding for M2M100.
#[derive(Debug, Clone)]
pub struct ScaledWordEmbedding {
    embedding: Embedding,
    embed_scale: f64,
}

impl ScaledWordEmbedding {
    fn load(vocab_size: usize, embed_dim: usize, scale: bool, vb: VarBuilder) -> Result<Self> {
        let embedding = candle_nn::embedding(vocab_size, embed_dim, vb)?;
        let embed_scale = if scale {
            (embed_dim as f64).sqrt()
        } else {
            1.0
        };
        Ok(Self {
            embedding,
            embed_scale,
        })
    }
}

impl Module for ScaledWordEmbedding {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let emb = self.embedding.forward(x)?;
        if self.embed_scale != 1.0 {
            &emb * self.embed_scale
        } else {
            Ok(emb)
        }
    }
}

/// Sinusoidal positional embedding (computed, not learned).
#[derive(Debug, Clone)]
struct SinusoidalPositionalEmbedding {
    embed_dim: usize,
    offset: usize,
}

impl SinusoidalPositionalEmbedding {
    fn new(embed_dim: usize) -> Self {
        // M2M100 uses offset=2 for padding
        Self {
            embed_dim,
            offset: 2,
        }
    }

    fn forward(&self, position_ids: &Tensor, device: &Device, dtype: DType) -> Result<Tensor> {
        // Generate sinusoidal embeddings on the fly
        let positions = position_ids.to_vec2::<u32>()?;
        let (batch_size, seq_len) = (positions.len(), positions[0].len());

        let half_dim = self.embed_dim / 2;
        let emb_scale = -(10000f64.ln()) / (half_dim as f64);

        let mut embeddings = Vec::with_capacity(batch_size * seq_len * self.embed_dim);
        for batch in &positions {
            for &pos in batch {
                let pos_with_offset = pos as f64 + self.offset as f64;
                for i in 0..half_dim {
                    let freq = (i as f64 * emb_scale).exp();
                    let angle = pos_with_offset * freq;
                    embeddings.push(angle.sin() as f32);
                }
                for i in 0..half_dim {
                    let freq = (i as f64 * emb_scale).exp();
                    let angle = pos_with_offset * freq;
                    embeddings.push(angle.cos() as f32);
                }
            }
        }

        Tensor::from_vec(embeddings, (batch_size, seq_len, self.embed_dim), device)?.to_dtype(dtype)
    }
}

/// Multi-head attention for decoder.
#[derive(Debug)]
struct M2M100Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    out_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
    #[allow(dead_code)]
    is_cross_attention: bool,
}

impl M2M100Attention {
    fn load(
        embed_dim: usize,
        num_heads: usize,
        is_cross_attention: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = embed_dim / num_heads;
        let q_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("out_proj"))?;

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
            is_cross_attention,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        key_value_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        past_key: Option<&Tensor>,
        past_value: Option<&Tensor>,
    ) -> Result<(Tensor, Tensor, Tensor)> {
        // Returns: (output, new_key, new_value)
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        // Query projection
        let query_states = self.q_proj.forward(hidden_states)?;
        let query_states = query_states
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?; // (B, num_heads, seq_len, head_dim)

        // Key/Value projection
        let kv_source = key_value_states.unwrap_or(hidden_states);
        let (_, kv_len, _) = kv_source.dims3()?;

        let k = self.k_proj.forward(kv_source)?;
        let v = self.v_proj.forward(kv_source)?;
        let k = k
            .reshape((batch_size, kv_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;
        let v = v
            .reshape((batch_size, kv_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?;

        // Concatenate with past KV if provided
        let (key_states, value_states) = match (past_key, past_value) {
            (Some(pk), Some(pv)) => (Tensor::cat(&[pk, &k], 2)?, Tensor::cat(&[pv, &v], 2)?),
            _ => (k.clone(), v.clone()),
        };

        // Attention scores - ensure contiguous for matmul
        let key_t = key_states.transpose(2, 3)?.contiguous()?;
        let attn_weights = query_states.matmul(&key_t)?;
        let attn_weights = (&attn_weights * self.scale)?;

        // Apply attention mask
        let attn_weights = if let Some(mask) = attention_mask {
            attn_weights.broadcast_add(mask)?
        } else {
            attn_weights
        };

        // Softmax and apply to values
        let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
        let attn_output = attn_weights.matmul(&value_states)?;

        // Reshape back
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        let output = self.out_proj.forward(&attn_output)?;
        Ok((output, key_states, value_states))
    }
}

/// M2M100 decoder layer.
#[derive(Debug)]
struct M2M100DecoderLayer {
    self_attn: M2M100Attention,
    self_attn_layer_norm: LayerNorm,
    encoder_attn: M2M100Attention,
    encoder_attn_layer_norm: LayerNorm,
    fc1: Linear,
    fc2: Linear,
    final_layer_norm: LayerNorm,
    dropout: Dropout,
}

impl M2M100DecoderLayer {
    fn load(cfg: &UniRecConfig, vb: VarBuilder) -> Result<Self> {
        let self_attn = M2M100Attention::load(
            cfg.d_model,
            cfg.decoder_attention_heads,
            false,
            vb.pp("self_attn"),
        )?;
        let self_attn_layer_norm =
            candle_nn::layer_norm(cfg.d_model, 1e-5, vb.pp("self_attn_layer_norm"))?;

        let encoder_attn = M2M100Attention::load(
            cfg.d_model,
            cfg.decoder_attention_heads,
            true,
            vb.pp("encoder_attn"),
        )?;
        let encoder_attn_layer_norm =
            candle_nn::layer_norm(cfg.d_model, 1e-5, vb.pp("encoder_attn_layer_norm"))?;

        let fc1 = candle_nn::linear(cfg.d_model, cfg.decoder_ffn_dim, vb.pp("fc1"))?;
        let fc2 = candle_nn::linear(cfg.decoder_ffn_dim, cfg.d_model, vb.pp("fc2"))?;
        let final_layer_norm = candle_nn::layer_norm(cfg.d_model, 1e-5, vb.pp("final_layer_norm"))?;

        let dropout = Dropout::new(cfg.dropout as f32);

        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            encoder_attn,
            encoder_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
            dropout,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        self_attn_mask: Option<&Tensor>,
        cross_attn_mask: Option<&Tensor>,
        mut kv_cache: Option<&mut KvCache>,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();

        // Self attention
        let hidden_states = self.self_attn_layer_norm.forward(hidden_states)?;
        let (past_self_key, past_self_value) = if let Some(cache) = kv_cache.as_ref() {
            (cache.self_key.as_ref(), cache.self_value.as_ref())
        } else {
            (None, None)
        };
        let (hidden_states, new_self_key, new_self_value) = self.self_attn.forward(
            &hidden_states,
            None,
            self_attn_mask,
            past_self_key,
            past_self_value,
        )?;
        // Update self-attention cache
        if let Some(cache) = kv_cache.as_mut() {
            cache.self_key = Some(new_self_key);
            cache.self_value = Some(new_self_value);
        }
        let hidden_states = self.dropout.forward(&hidden_states, false)?;
        let hidden_states = (&residual + &hidden_states)?;

        // Cross attention
        let residual = hidden_states.clone();
        let hidden_states = self.encoder_attn_layer_norm.forward(&hidden_states)?;
        let (past_cross_key, past_cross_value) = if let Some(cache) = kv_cache.as_ref() {
            (cache.cross_key.as_ref(), cache.cross_value.as_ref())
        } else {
            (None, None)
        };
        let (hidden_states, new_cross_key, new_cross_value) = self.encoder_attn.forward(
            &hidden_states,
            Some(encoder_hidden_states),
            cross_attn_mask,
            past_cross_key,
            past_cross_value,
        )?;
        // Update cross-attention cache (only first step computes new KV)
        if let Some(cache) = kv_cache.as_mut()
            && cache.cross_key.is_none()
        {
            cache.cross_key = Some(new_cross_key);
            cache.cross_value = Some(new_cross_value);
        }
        let hidden_states = self.dropout.forward(&hidden_states, false)?;
        let hidden_states = (&residual + &hidden_states)?;

        // FFN
        let residual = hidden_states.clone();
        let hidden_states = self.final_layer_norm.forward(&hidden_states)?;
        let hidden_states = self.fc1.forward(&hidden_states)?;
        let hidden_states = hidden_states.relu()?;
        let hidden_states = self.dropout.forward(&hidden_states, false)?;
        let hidden_states = self.fc2.forward(&hidden_states)?;
        let hidden_states = self.dropout.forward(&hidden_states, false)?;
        &residual + &hidden_states
    }
}

/// M2M100 Decoder.
#[derive(Debug)]
pub struct M2M100Decoder {
    embed_tokens: ScaledWordEmbedding,
    embed_positions: SinusoidalPositionalEmbedding,
    layers: Vec<M2M100DecoderLayer>,
    layer_norm: LayerNorm,
    dropout: Dropout,
}

impl M2M100Decoder {
    /// Load M2M100 decoder from weights.
    pub fn load(cfg: &UniRecConfig, vb: VarBuilder) -> std::result::Result<Self, OCRError> {
        let embed_tokens = ScaledWordEmbedding::load(
            cfg.vocab_size,
            cfg.d_model,
            cfg.scale_embedding,
            vb.pp("embed_tokens"),
        )
        .map_err(|e| candle_to_ocr_inference("M2M100Decoder", "load embed_tokens", e))?;

        // Use sinusoidal positional embeddings (computed, not learned)
        let embed_positions = SinusoidalPositionalEmbedding::new(cfg.d_model);

        let mut layers = Vec::new();
        for i in 0..cfg.decoder_layers {
            let layer =
                M2M100DecoderLayer::load(cfg, vb.pp(format!("layers.{}", i))).map_err(|e| {
                    candle_to_ocr_inference("M2M100Decoder", format!("load layer.{}", i), e)
                })?;
            layers.push(layer);
        }

        let layer_norm = candle_nn::layer_norm(cfg.d_model, 1e-5, vb.pp("layer_norm"))
            .map_err(|e| candle_to_ocr_inference("M2M100Decoder", "load layer_norm", e))?;

        let dropout = Dropout::new(cfg.dropout as f32);

        Ok(Self {
            embed_tokens,
            embed_positions,
            layers,
            layer_norm,
            dropout,
        })
    }

    /// Forward pass through the decoder.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        encoder_hidden_states: &Tensor,
        position_offset: usize,
        self_attn_mask: Option<&Tensor>,
        mut kv_cache: Option<&mut Vec<KvCache>>,
    ) -> std::result::Result<Tensor, OCRError> {
        let (batch_size, seq_len) = input_ids
            .dims2()
            .map_err(|e| candle_to_ocr_inference("M2M100Decoder", "get input dims", e))?;
        let device = input_ids.device();
        let dtype = encoder_hidden_states.dtype();

        // Token embeddings
        let inputs_embeds = self
            .embed_tokens
            .forward(input_ids)
            .map_err(|e| candle_to_ocr_inference("M2M100Decoder", "embed_tokens forward", e))?;

        // Position ids
        let position_ids: Vec<u32> = (position_offset..(position_offset + seq_len))
            .map(|p| p as u32)
            .collect();
        let position_ids = Tensor::new(&position_ids[..], device)
            .map_err(|e| candle_to_ocr_inference("M2M100Decoder", "create position_ids", e))?
            .unsqueeze(0)
            .map_err(|e| candle_to_ocr_inference("M2M100Decoder", "unsqueeze position_ids", e))?
            .broadcast_as((batch_size, seq_len))
            .map_err(|e| candle_to_ocr_inference("M2M100Decoder", "broadcast position_ids", e))?;

        let positions = self
            .embed_positions
            .forward(&position_ids, device, dtype)
            .map_err(|e| candle_to_ocr_inference("M2M100Decoder", "embed_positions forward", e))?;

        // Combine embeddings
        let hidden_states = (&inputs_embeds + &positions)
            .map_err(|e| candle_to_ocr_inference("M2M100Decoder", "add embeddings", e))?;
        let mut hidden_states = self
            .dropout
            .forward(&hidden_states, false)
            .map_err(|e| candle_to_ocr_inference("M2M100Decoder", "dropout", e))?;

        // Process through layers
        for (i, layer) in self.layers.iter().enumerate() {
            let layer_cache = kv_cache.as_mut().map(|c| &mut c[i]);
            hidden_states = layer
                .forward(
                    &hidden_states,
                    encoder_hidden_states,
                    self_attn_mask,
                    None, // cross_attn_mask - encoder hidden states don't need masking
                    layer_cache,
                )
                .map_err(|e| {
                    candle_to_ocr_inference("M2M100Decoder", format!("layer.{} forward", i), e)
                })?;
        }

        // Final layer norm
        self.layer_norm
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference("M2M100Decoder", "layer_norm forward", e))
    }
}

/// Create causal attention mask for decoder.
pub fn create_causal_mask(
    seq_len: usize,
    device: &Device,
    dtype: DType,
) -> std::result::Result<Tensor, OCRError> {
    let mut data = Vec::with_capacity(seq_len * seq_len);
    for i in 0..seq_len {
        for j in 0..seq_len {
            if j <= i {
                data.push(0f32);
            } else {
                data.push(f32::NEG_INFINITY);
            }
        }
    }
    Tensor::from_vec(data, (1, 1, seq_len, seq_len), device)
        .map_err(|e| candle_to_ocr_inference("create_causal_mask", "build tensor", e))?
        .to_dtype(dtype)
        .map_err(|e| candle_to_ocr_inference("create_causal_mask", "cast dtype", e))
}
