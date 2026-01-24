//! Pixtral Vision Model implementation for LightOnOCR.
//!
//! This is a custom implementation that matches HuggingFace Transformers'
//! `modeling_pixtral.py` more closely than the Candle version, particularly
//! in the RoPE (Rotary Position Embedding) computation.

use candle_core::{D, DType, Device, Module, Result, Tensor};
use candle_nn::{
    Conv2d, Conv2dConfig, Linear, RmsNorm, VarBuilder, conv2d_no_bias, linear_b, rms_norm,
};
use tracing::debug;

use super::config::LightOnOcrVisionConfig;

/// Pixtral Vision Model configuration.
#[derive(Debug, Clone)]
pub struct PixtralVisionConfig {
    pub hidden_size: usize,
    pub num_channels: usize,
    pub image_size: usize,
    pub patch_size: usize,
    pub rope_theta: f64,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub head_dim: usize,
    pub hidden_act: candle_nn::Activation,
}

impl From<&LightOnOcrVisionConfig> for PixtralVisionConfig {
    fn from(cfg: &LightOnOcrVisionConfig) -> Self {
        Self {
            hidden_size: cfg.hidden_size,
            num_channels: cfg.num_channels,
            image_size: cfg.image_size,
            patch_size: cfg.patch_size,
            rope_theta: cfg.rope_theta,
            intermediate_size: cfg.intermediate_size,
            num_hidden_layers: cfg.num_hidden_layers,
            num_attention_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim,
            hidden_act: cfg.hidden_act,
        }
    }
}

/// Compute 2D position IDs in meshgrid format.
///
/// This matches `position_ids_in_meshgrid` from HuggingFace Transformers.
fn position_ids_in_meshgrid(
    num_patches_h: usize,
    num_patches_w: usize,
    max_width: usize,
    device: &Device,
) -> Result<Tensor> {
    let h = Tensor::arange(0u32, num_patches_h as u32, device)?;
    let w = Tensor::arange(0u32, num_patches_w as u32, device)?;

    // meshgrid with indexing="ij"
    let h_grid = h
        .unsqueeze(1)?
        .broadcast_as((num_patches_h, num_patches_w))?;
    let w_grid = w
        .unsqueeze(0)?
        .broadcast_as((num_patches_h, num_patches_w))?;

    // ids = h_grid * max_width + w_grid
    let ids = (h_grid.to_dtype(DType::U32)? * (max_width as f64))?
        .add(&w_grid.to_dtype(DType::U32)?)?
        .flatten_all()?;

    Ok(ids)
}

/// Pixtral Rotary Embedding.
///
/// The key difference from standard RoPE is that Pixtral uses 2D position encoding
/// where half the frequencies are for height and half for width.
#[derive(Debug, Clone)]
pub struct PixtralRotaryEmbedding {
    /// Precomputed inverse frequencies, shape: (max_patches^2, head_dim)
    inv_freq: Tensor,
    max_patches_per_side: usize,
}

impl PixtralRotaryEmbedding {
    pub fn new(cfg: &PixtralVisionConfig, device: &Device) -> Result<Self> {
        let dim = cfg.head_dim;
        let base = cfg.rope_theta as f32;
        let max_patches_per_side = cfg.image_size / cfg.patch_size;

        debug!(
            "PixtralRotaryEmbedding: dim={}, base={}, max_patches={}",
            dim, base, max_patches_per_side
        );

        // Compute base frequencies: 1 / (theta ^ (2i/dim)) for i in 0..dim/2
        let freqs: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / base.powf(i as f32 / dim as f32))
            .collect();

        debug!(
            "PixtralRotaryEmbedding: freqs len={}, first 4: {:?}",
            freqs.len(),
            &freqs[..4.min(freqs.len())]
        );

        // Split frequencies: even indices for height, odd indices for width
        let freqs_h: Vec<f32> = freqs.iter().step_by(2).copied().collect();
        let freqs_w: Vec<f32> = freqs.iter().skip(1).step_by(2).copied().collect();

        debug!(
            "PixtralRotaryEmbedding: freqs_h len={}, first 4: {:?}",
            freqs_h.len(),
            &freqs_h[..4.min(freqs_h.len())]
        );
        debug!(
            "PixtralRotaryEmbedding: freqs_w len={}, first 4: {:?}",
            freqs_w.len(),
            &freqs_w[..4.min(freqs_w.len())]
        );

        let freqs_h = Tensor::new(freqs_h, device)?;
        let freqs_w = Tensor::new(freqs_w, device)?;

        // Position indices
        let h = Tensor::arange(0u32, max_patches_per_side as u32, device)?.to_dtype(DType::F32)?;
        let w = Tensor::arange(0u32, max_patches_per_side as u32, device)?.to_dtype(DType::F32)?;

        // Compute outer products: (max_patches, dim/4)
        let freqs_h = h.unsqueeze(1)?.matmul(&freqs_h.unsqueeze(0)?)?;
        let freqs_w = w.unsqueeze(1)?.matmul(&freqs_w.unsqueeze(0)?)?;

        debug!(
            "PixtralRotaryEmbedding: freqs_h outer shape {:?}, freqs_w outer shape {:?}",
            freqs_h.dims(),
            freqs_w.dims()
        );

        // Build the full inv_freq tensor:
        // freqs_h: (max_patches, 1, dim/4) repeated along width
        // freqs_w: (1, max_patches, dim/4) repeated along height
        // concat along last dim -> (max_patches, max_patches, dim/2)
        // reshape -> (max_patches^2, dim/2)
        let freqs_h = freqs_h
            .unsqueeze(1)?
            .broadcast_as((max_patches_per_side, max_patches_per_side, freqs_h.dim(1)?))?
            .contiguous()?;
        let freqs_w = freqs_w
            .unsqueeze(0)?
            .broadcast_as((max_patches_per_side, max_patches_per_side, freqs_w.dim(1)?))?
            .contiguous()?;

        debug!(
            "PixtralRotaryEmbedding: broadcast freqs_h {:?}, freqs_w {:?}",
            freqs_h.dims(),
            freqs_w.dims()
        );

        let inv_freq = Tensor::cat(&[freqs_h, freqs_w], D::Minus1)?
            .reshape((max_patches_per_side * max_patches_per_side, dim / 2))?;

        debug!(
            "PixtralRotaryEmbedding: inv_freq after concat and reshape {:?}",
            inv_freq.dims()
        );

        // CRITICAL: Duplicate inv_freq to full dimension (dim/2 -> dim)
        // This matches: inv_freq = torch.cat((inv_freq, inv_freq), dim=-1)
        let inv_freq = Tensor::cat(&[&inv_freq, &inv_freq], D::Minus1)?;

        debug!(
            "PixtralRotaryEmbedding: final inv_freq shape {:?}",
            inv_freq.dims()
        );

        // Debug: print inv_freq values at key positions for verification
        if let Ok(inv_freq_vec) = inv_freq.to_vec2::<f32>() {
            // Position 0 (h=0, w=0) - should be all zeros
            debug!(
                "PixtralRotaryEmbedding: inv_freq[0, :8]: {:?}",
                &inv_freq_vec[0][..8]
            );
            // Position 1 (h=0, w=1) - should have zeros in first 16 (freqs_h[0]=0), and freqs_w[1] in next 16
            debug!(
                "PixtralRotaryEmbedding: inv_freq[1, :8]: {:?}",
                &inv_freq_vec[1][..8]
            );
            // Position 110 (h=1, w=0) - should have freqs_h[1] in first 16, and zeros in next 16
            if inv_freq_vec.len() > 110 {
                debug!(
                    "PixtralRotaryEmbedding: inv_freq[110, :8]: {:?}",
                    &inv_freq_vec[110][..8]
                );
                debug!(
                    "PixtralRotaryEmbedding: inv_freq[110, 16:24]: {:?}",
                    &inv_freq_vec[110][16..24]
                );
            }
        }

        Ok(Self {
            inv_freq,
            max_patches_per_side,
        })
    }

    /// Compute cos and sin embeddings for given position IDs.
    pub fn forward(&self, position_ids: &Tensor, dtype: DType) -> Result<(Tensor, Tensor)> {
        // Select frequencies for the given positions
        let freqs = self.inv_freq.index_select(position_ids, 0)?;

        debug!(
            "PixtralRotaryEmbedding forward: position_ids shape {:?}, freqs shape {:?}",
            position_ids.dims(),
            freqs.dims()
        );

        let cos = freqs.cos()?.to_dtype(dtype)?;
        let sin = freqs.sin()?.to_dtype(dtype)?;

        Ok((cos, sin))
    }

    pub fn max_patches_per_side(&self) -> usize {
        self.max_patches_per_side
    }
}

/// Rotate half of the hidden dims.
fn rotate_half(x: &Tensor) -> Result<Tensor> {
    let last_dim = x.dim(D::Minus1)?;
    let x1 = x.narrow(D::Minus1, 0, last_dim / 2)?;
    let x2 = x.narrow(D::Minus1, last_dim / 2, last_dim / 2)?;
    Tensor::cat(&[&x2.neg()?, &x1], D::Minus1)
}

/// Apply rotary position embedding to query and key tensors.
///
/// Args:
///     q: Query tensor of shape (batch, heads, seq_len, head_dim)
///     k: Key tensor of shape (batch, heads, seq_len, head_dim)
///     cos: Cosine embedding of shape (seq_len, head_dim)
///     sin: Sine embedding of shape (seq_len, head_dim)
fn apply_rotary_pos_emb(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor)> {
    // unsqueeze_dim=0 for Pixtral: (seq_len, head_dim) -> (1, seq_len, head_dim)
    // This broadcasts with (batch, heads, seq_len, head_dim)
    let cos = cos.unsqueeze(0)?;
    let sin = sin.unsqueeze(0)?;

    let q_embed = q
        .broadcast_mul(&cos)?
        .add(&rotate_half(q)?.broadcast_mul(&sin)?)?;
    let k_embed = k
        .broadcast_mul(&cos)?
        .add(&rotate_half(k)?.broadcast_mul(&sin)?)?;

    Ok((q_embed, k_embed))
}

/// Pixtral MLP layer.
#[derive(Debug, Clone)]
struct PixtralMlp {
    gate_proj: Linear,
    up_proj: Linear,
    down_proj: Linear,
    act_fn: candle_nn::Activation,
}

impl PixtralMlp {
    fn new(cfg: &PixtralVisionConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let gate_proj = linear_b(h, i, false, vb.pp("gate_proj"))?;
        let up_proj = linear_b(h, i, false, vb.pp("up_proj"))?;
        let down_proj = linear_b(i, h, false, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for PixtralMlp {
    fn forward(&self, xs: &Tensor) -> Result<Tensor> {
        let gate = self.gate_proj.forward(xs)?.apply(&self.act_fn)?;
        let up = self.up_proj.forward(xs)?;
        self.down_proj.forward(&(gate * up)?)
    }
}

/// Pixtral Attention layer.
#[derive(Debug, Clone)]
struct PixtralAttention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl PixtralAttention {
    fn new(cfg: &PixtralVisionConfig, vb: VarBuilder) -> Result<Self> {
        let h = cfg.hidden_size;
        let q_proj = linear_b(h, h, false, vb.pp("q_proj"))?;
        let k_proj = linear_b(h, h, false, vb.pp("k_proj"))?;
        let v_proj = linear_b(h, h, false, vb.pp("v_proj"))?;
        let o_proj = linear_b(h, h, false, vb.pp("o_proj"))?;
        let scale = (cfg.head_dim as f64).powf(-0.5);
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim,
            scale,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (batch, patches, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        // Reshape to (batch, heads, patches, head_dim)
        let shape = (batch, patches, self.num_heads, self.head_dim);
        let q = q.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape(shape)?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape(shape)?.transpose(1, 2)?.contiguous()?;

        // Apply rotary embeddings
        let (q, k) = apply_rotary_pos_emb(&q, &k, cos, sin)?;

        // Scaled dot-product attention
        let attn_weights = (q.matmul(&k.transpose(D::Minus2, D::Minus1)?)? * self.scale)?;

        let attn_weights = match attention_mask {
            Some(mask) => attn_weights.broadcast_add(mask)?,
            None => attn_weights,
        };

        // Compute softmax in float32 for numerical stability, then cast back
        let input_dtype = attn_weights.dtype();
        let attn_weights = attn_weights.to_dtype(DType::F32)?;
        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_weights = attn_weights.to_dtype(input_dtype)?;
        let attn_output = attn_weights.matmul(&v)?;

        // Reshape back to (batch, patches, hidden)
        let attn_output = attn_output.transpose(1, 2)?.reshape((batch, patches, ()))?;

        self.o_proj.forward(&attn_output)
    }
}

/// Pixtral Attention Layer (attention + MLP with residual connections).
#[derive(Debug, Clone)]
struct PixtralAttentionLayer {
    attention_norm: RmsNorm,
    attention: PixtralAttention,
    ffn_norm: RmsNorm,
    feed_forward: PixtralMlp,
}

impl PixtralAttentionLayer {
    fn new(cfg: &PixtralVisionConfig, vb: VarBuilder) -> Result<Self> {
        let attention_norm = rms_norm(cfg.hidden_size, 1e-5, vb.pp("attention_norm"))?;
        let attention = PixtralAttention::new(cfg, vb.pp("attention"))?;
        let ffn_norm = rms_norm(cfg.hidden_size, 1e-5, vb.pp("ffn_norm"))?;
        let feed_forward = PixtralMlp::new(cfg, vb.pp("feed_forward"))?;
        Ok(Self {
            attention_norm,
            attention,
            ffn_norm,
            feed_forward,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        // Pre-norm attention with residual
        let residual = xs;
        let xs = self.attention_norm.forward(xs)?;
        let xs = self.attention.forward(&xs, cos, sin, attention_mask)?;
        let xs = (residual + xs)?;

        // Pre-norm FFN with residual
        let residual = &xs;
        let xs = self.ffn_norm.forward(&xs)?;
        let xs = self.feed_forward.forward(&xs)?;
        residual + xs
    }
}

/// Pixtral Transformer.
#[derive(Debug, Clone)]
struct PixtralTransformer {
    layers: Vec<PixtralAttentionLayer>,
}

impl PixtralTransformer {
    fn new(cfg: &PixtralVisionConfig, vb: VarBuilder) -> Result<Self> {
        let vb_layers = vb.pp("layers");
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            layers.push(PixtralAttentionLayer::new(cfg, vb_layers.pp(i))?);
        }
        Ok(Self { layers })
    }

    fn forward(
        &self,
        xs: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut hidden_states = xs.clone();
        for layer in &self.layers {
            hidden_states = layer.forward(&hidden_states, cos, sin, attention_mask)?;
        }
        Ok(hidden_states)
    }
}

/// Generate block attention mask for multiple images.
///
/// Each image's patches can only attend to patches from the same image.
fn generate_block_attention_mask(
    patch_counts: &[usize],
    dtype: DType,
    device: &Device,
) -> Result<Tensor> {
    let seq_len: usize = patch_counts.iter().sum();
    // Use dtype.min for numerical stability (matches HuggingFace)
    // BF16/F16 min is ~-3.39e+38, which is more stable than NEG_INFINITY
    let d_min: f32 = match dtype {
        DType::F32 => -3.4028235e+38, // f32::MIN
        DType::F16 => -65504.0,       // f16::MIN
        DType::BF16 => -3.39e+38,     // bf16::MIN (approximate)
        _ => -3.4028235e+38,
    };

    // Start with all d_min (no attention)
    let mut mask_data = vec![d_min; seq_len * seq_len];

    // For each block, allow full attention within the block
    let mut offset = 0usize;
    for &count in patch_counts {
        for i in 0..count {
            for j in 0..count {
                mask_data[(offset + i) * seq_len + (offset + j)] = 0.0;
            }
        }
        offset += count;
    }

    let mask = Tensor::from_vec(mask_data, (seq_len, seq_len), device)?;
    // Expand to (1, 1, seq_len, seq_len) for broadcasting
    mask.unsqueeze(0)?.unsqueeze(0)?.to_dtype(dtype)
}

/// Pixtral Vision Model.
#[derive(Debug, Clone)]
pub struct PixtralVisionModel {
    patch_conv: Conv2d,
    ln_pre: RmsNorm,
    transformer: PixtralTransformer,
    rotary_emb: PixtralRotaryEmbedding,
}

impl PixtralVisionModel {
    pub fn new(cfg: &PixtralVisionConfig, vb: VarBuilder) -> Result<Self> {
        let conv_cfg = Conv2dConfig {
            stride: cfg.patch_size,
            ..Default::default()
        };
        let patch_conv = conv2d_no_bias(
            cfg.num_channels,
            cfg.hidden_size,
            cfg.patch_size,
            conv_cfg,
            vb.pp("patch_conv"),
        )?;

        let ln_pre = rms_norm(cfg.hidden_size, 1e-5, vb.pp("ln_pre"))?;
        let transformer = PixtralTransformer::new(cfg, vb.pp("transformer"))?;
        let rotary_emb = PixtralRotaryEmbedding::new(cfg, vb.device())?;

        Ok(Self {
            patch_conv,
            ln_pre,
            transformer,
            rotary_emb,
        })
    }

    /// Forward pass for a single image.
    ///
    /// Args:
    ///     pixel_values: Image tensor of shape (1, C, H, W)
    ///
    /// Returns:
    ///     Hidden states of shape (1, num_patches, hidden_size)
    pub fn forward(&self, pixel_values: &Tensor) -> Result<Tensor> {
        let dtype = pixel_values.dtype();

        // Apply patch convolution: (1, C, H, W) -> (1, hidden, grid_h, grid_w)
        let patch_embeds = self.patch_conv.forward(pixel_values)?;
        let (_, _, grid_h, grid_w) = patch_embeds.dims4()?;

        // Flatten and transpose: (1, hidden, grid_h * grid_w) -> (1, grid_h * grid_w, hidden)
        let patch_embeds = patch_embeds.flatten_from(2)?.transpose(1, 2)?;

        // Apply pre-norm
        let patch_embeds = self.ln_pre.forward(&patch_embeds)?;

        // Compute position IDs and rotary embeddings
        let position_ids = position_ids_in_meshgrid(
            grid_h,
            grid_w,
            self.rotary_emb.max_patches_per_side(),
            patch_embeds.device(),
        )?;
        let (cos, sin) = self.rotary_emb.forward(&position_ids, dtype)?;

        // For single image, use block attention mask (though it's effectively full attention)
        let patch_count = grid_h * grid_w;
        let attention_mask =
            generate_block_attention_mask(&[patch_count], dtype, patch_embeds.device())?;

        // Run transformer
        self.transformer
            .forward(&patch_embeds, &cos, &sin, Some(&attention_mask))
    }
}
