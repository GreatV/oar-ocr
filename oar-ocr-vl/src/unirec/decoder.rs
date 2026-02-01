//! M2M100-style decoder implementation for UniRec.
//!
//! Based on the M2M100 (Multilingual Translation) decoder architecture
//! with learned positional embeddings and scaled word embeddings.

use candle_core::{D, DType, Device, Result, Tensor};
use candle_nn::{Embedding, LayerNorm, Linear, Module, VarBuilder, kv_cache::KvCache};
use std::cell::RefCell;

use super::config::UniRecConfig;
use crate::attention::scaled_dot_product_attention;
use crate::utils::candle_to_ocr_inference;
use oar_ocr_core::core::OCRError;

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
        // Generate sinusoidal embeddings entirely on GPU using tensor operations
        let half_dim = self.embed_dim / 2;
        let emb_scale = -(10000f64.ln()) / (half_dim as f64);

        // Create frequency tensor: exp(-i * log(10000) / half_dim) for i in [0, half_dim)
        // Shape: [half_dim]
        let freq_indices = Tensor::arange(0u32, half_dim as u32, device)?.to_dtype(DType::F32)?;
        let freqs = (&freq_indices * emb_scale)?.exp()?;

        // Add offset to positions: [batch, seq_len]
        let positions = position_ids
            .to_dtype(DType::F32)?
            .broadcast_add(&Tensor::new(self.offset as f32, device)?)?;

        // Compute angles: positions [batch, seq_len, 1] * freqs [1, 1, half_dim]
        // Result shape: [batch, seq_len, half_dim]
        let positions_expanded = positions.unsqueeze(D::Minus1)?;
        let freqs_expanded = freqs.reshape((1, 1, half_dim))?;
        let angles = positions_expanded.broadcast_mul(&freqs_expanded)?;

        // Compute sin and cos, then concatenate: [batch, seq_len, embed_dim]
        let sin_emb = angles.sin()?;
        let cos_emb = angles.cos()?;
        Tensor::cat(&[&sin_emb, &cos_emb], D::Minus1)?.to_dtype(dtype)
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
    kv_cache: RefCell<KvCache>,
}

impl M2M100Attention {
    fn load(
        embed_dim: usize,
        num_heads: usize,
        _is_cross_attention: bool,
        vb: VarBuilder,
    ) -> Result<Self> {
        let head_dim = embed_dim / num_heads;
        let q_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("v_proj"))?;
        let out_proj = candle_nn::linear(embed_dim, embed_dim, vb.pp("out_proj"))?;

        // Create KvCache with dim=2 for seq_len dimension
        // Pre-allocate 8192 to avoid reallocation (double of typical 4096 max_tokens)
        let kv_cache = KvCache::new(2, 8192);

        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            out_proj,
            num_heads,
            head_dim,
            scale: (head_dim as f64).powf(-0.5),
            kv_cache: RefCell::new(kv_cache),
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        key_value_states: Option<&Tensor>,
        attention_mask: Option<&Tensor>,
        is_cross_attention: bool,
    ) -> Result<Tensor> {
        let (batch_size, seq_len, _) = hidden_states.dims3()?;

        // Query projection
        let query_states = self.q_proj.forward(hidden_states)?;
        let query_states = query_states
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?
            .contiguous()?; // (B, num_heads, seq_len, head_dim)

        // For cross-attention, check if we already have cached KV
        let use_cached = if is_cross_attention {
            self.kv_cache.borrow().current_seq_len() > 0
        } else {
            false
        };

        let (key_states, value_states) = if use_cached {
            // Cross-attention: reuse cached encoder KV
            let cache = self.kv_cache.borrow();
            match (cache.k()?, cache.v()?) {
                (Some(k), Some(v)) => (k.clone(), v.clone()),
                _ => return Err(candle_core::Error::Msg("kv cache is empty".into())),
            }
        } else {
            // Self-attention or first cross-attention step: compute new KV
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

            // Append to cache
            self.kv_cache.borrow_mut().append(&k, &v)?
        };

        // Use unified scaled dot-product attention
        let attn_output = scaled_dot_product_attention(
            &query_states,
            &key_states,
            &value_states,
            attention_mask,
            self.scale,
            false, // is_causal=false, mask is passed explicitly
        )?;

        // Reshape back
        let attn_output = attn_output.transpose(1, 2)?.contiguous()?.reshape((
            batch_size,
            seq_len,
            self.num_heads * self.head_dim,
        ))?;

        self.out_proj.forward(&attn_output)
    }

    fn clear_cache(&self) {
        self.kv_cache.borrow_mut().reset();
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

        Ok(Self {
            self_attn,
            self_attn_layer_norm,
            encoder_attn,
            encoder_attn_layer_norm,
            fc1,
            fc2,
            final_layer_norm,
        })
    }

    fn forward(
        &self,
        hidden_states: &Tensor,
        encoder_hidden_states: &Tensor,
        self_attn_mask: Option<&Tensor>,
        cross_attn_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let residual = hidden_states.clone();

        // Self attention
        let hidden_states = self.self_attn_layer_norm.forward(hidden_states)?;
        let hidden_states = self.self_attn.forward(
            &hidden_states,
            None,
            self_attn_mask,
            false, // is_cross_attention = false for self-attention
        )?;
        let hidden_states = (&residual + &hidden_states)?;

        // Cross attention
        let residual = hidden_states.clone();
        let hidden_states = self.encoder_attn_layer_norm.forward(&hidden_states)?;
        let hidden_states = self.encoder_attn.forward(
            &hidden_states,
            Some(encoder_hidden_states),
            cross_attn_mask,
            true, // is_cross_attention = true for cross-attention
        )?;
        let hidden_states = (&residual + &hidden_states)?;

        // FFN
        let residual = hidden_states.clone();
        let hidden_states = self.final_layer_norm.forward(&hidden_states)?;
        let hidden_states = self.fc1.forward(&hidden_states)?;
        let hidden_states = hidden_states.relu()?;
        let hidden_states = self.fc2.forward(&hidden_states)?;
        &residual + &hidden_states
    }

    fn clear_kv_cache(&self) {
        self.self_attn.clear_cache();
        self.encoder_attn.clear_cache();
    }
}

/// M2M100 Decoder.
#[derive(Debug)]
pub struct M2M100Decoder {
    embed_tokens: ScaledWordEmbedding,
    embed_positions: SinusoidalPositionalEmbedding,
    layers: Vec<M2M100DecoderLayer>,
    layer_norm: LayerNorm,
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

        Ok(Self {
            embed_tokens,
            embed_positions,
            layers,
            layer_norm,
        })
    }

    /// Forward pass through the decoder.
    pub fn forward(
        &self,
        input_ids: &Tensor,
        encoder_hidden_states: &Tensor,
        position_offset: usize,
        self_attn_mask: Option<&Tensor>,
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
        let mut hidden_states = (&inputs_embeds + &positions)
            .map_err(|e| candle_to_ocr_inference("M2M100Decoder", "add embeddings", e))?;

        // Process through layers
        for (i, layer) in self.layers.iter().enumerate() {
            hidden_states = layer
                .forward(
                    &hidden_states,
                    encoder_hidden_states,
                    self_attn_mask,
                    None, // cross_attn_mask - encoder hidden states don't need masking
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

    pub fn clear_kv_cache(&self) {
        for layer in &self.layers {
            layer.clear_kv_cache();
        }
    }
}
