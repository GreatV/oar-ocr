//! Qwen3-VL vision tower.
//!
//! Ported from `modeling_qwen3_vl.py` (transformers 4.57). Differences from
//! the in-repo Qwen2-VL/2.5-VL towers: full attention in every block (no
//! windowing), LayerNorm block norms, a learnable `pos_embed` bilinearly
//! interpolated to the patch grid, and **DeepStack** — the block outputs at
//! `deepstack_visual_indexes` each pass through an extra post-shuffle merger
//! and are returned alongside the merged embeddings for the text model to
//! inject into its first decoder layers.

use crate::error::Error;
use crate::runtime::errors::{candle_to_ocr_inference, candle_to_ocr_processing};
use crate::runtime::tensor::rotate_half;
use candle_core::{DType, Device, IndexOp, Tensor};
use candle_nn::{
    Activation, LayerNorm, LayerNormConfig, Linear, Module, VarBuilder, layer_norm, linear,
};

const MODEL_NAME: &str = "Qwen3-VL";

fn default_in_channels() -> usize {
    3
}

/// Vision-tower configuration shared by Qwen3-VL checkpoints.
#[derive(Debug, Clone, serde::Deserialize)]
pub struct Qwen3VlVisionConfig {
    pub model_type: String,
    pub depth: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_heads: usize,
    #[serde(default = "default_in_channels")]
    pub in_channels: usize,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    pub temporal_patch_size: usize,
    pub out_hidden_size: usize,
    pub num_position_embeddings: usize,
    pub hidden_act: Activation,
    #[serde(default)]
    pub deepstack_visual_indexes: Vec<usize>,
}

impl Qwen3VlVisionConfig {
    pub fn head_dim(&self) -> Result<usize, Error> {
        if self.num_heads == 0 || !self.hidden_size.is_multiple_of(self.num_heads) {
            return Err(Error::Config {
                message: format!(
                    "{MODEL_NAME} vision hidden_size {} must be divisible by num_heads {}",
                    self.hidden_size, self.num_heads
                ),
            });
        }
        Ok(self.hidden_size / self.num_heads)
    }

    /// Side length of the square learned-position grid
    /// (`sqrt(num_position_embeddings)`).
    pub fn position_grid_size(&self) -> Result<usize, Error> {
        let side = (self.num_position_embeddings as f64).sqrt() as usize;
        if side == 0 || side * side != self.num_position_embeddings {
            return Err(Error::Config {
                message: format!(
                    "{MODEL_NAME} num_position_embeddings must be a non-zero square, got {}",
                    self.num_position_embeddings
                ),
            });
        }
        Ok(side)
    }

    pub fn validate(&self) -> Result<(), Error> {
        if self.model_type != "qwen3_vl" {
            return Err(Error::Config {
                message: format!(
                    "{MODEL_NAME} expected vision model_type 'qwen3_vl', got '{}'",
                    self.model_type
                ),
            });
        }
        if self.hidden_size == 0
            || self.intermediate_size == 0
            || self.in_channels == 0
            || self.patch_size == 0
            || self.spatial_merge_size == 0
            || self.temporal_patch_size == 0
            || self.out_hidden_size == 0
            || self.num_position_embeddings == 0
        {
            return Err(Error::Config {
                message: format!("{MODEL_NAME} vision dimensions must be non-zero"),
            });
        }
        let head_dim = self.head_dim()?;
        if !head_dim.is_multiple_of(4) {
            return Err(Error::Config {
                message: format!(
                    "{MODEL_NAME} vision head_dim must be divisible by 4 for 2D RoPE, got {head_dim}"
                ),
            });
        }
        self.position_grid_size()?;
        for &index in &self.deepstack_visual_indexes {
            if index >= self.depth {
                return Err(Error::Config {
                    message: format!(
                        "{MODEL_NAME} deepstack_visual_indexes entry {index} exceeds tower depth {}",
                        self.depth
                    ),
                });
            }
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
struct PatchEmbed {
    weight: Tensor,
    bias: Tensor,
}

impl PatchEmbed {
    fn load(cfg: &Qwen3VlVisionConfig, vb: VarBuilder) -> Result<Self, Error> {
        let vb = vb.pp("patch_embed").pp("proj");
        let patch_dim = cfg.in_channels * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
        let weight = vb
            .get(
                (
                    cfg.hidden_size,
                    cfg.in_channels,
                    cfg.temporal_patch_size,
                    cfg.patch_size,
                    cfg.patch_size,
                ),
                "weight",
            )
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load patch_embed weight", e))?
            .reshape((cfg.hidden_size, patch_dim))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "reshape patch_embed weight", e))?;
        let bias = vb
            .get(cfg.hidden_size, "bias")
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load patch_embed bias", e))?;
        Ok(Self { weight, bias })
    }

    fn forward(&self, patches: &Tensor) -> Result<Tensor, Error> {
        let patches = patches
            .to_dtype(self.weight.dtype())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "cast patch input", e))?;
        let weight_t = self
            .weight
            .transpose(0, 1)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "transpose patch weight", e))?;
        patches
            .matmul(&weight_t)
            .and_then(|output| output.broadcast_add(&self.bias))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "patch embedding", e))
    }
}

/// One-dimensional inverse frequencies for the 2D vision RoPE
/// (`Qwen3VLVisionRotaryEmbedding(head_dim // 2)`, theta 10000).
#[derive(Debug, Clone)]
struct VisionRotaryEmbedding {
    inv_freq: Tensor,
}

impl VisionRotaryEmbedding {
    fn new(dim: usize, device: &Device) -> Result<Self, Error> {
        let inv_freq = crate::runtime::tensor::vision_inv_freq(dim, 10_000.0, MODEL_NAME, device)?;
        Ok(Self { inv_freq })
    }

    fn forward(&self, sequence_length: usize, device: &Device) -> Result<Tensor, Error> {
        crate::runtime::attention::on_compute_device(device, |compute_device| {
            let positions = Tensor::arange(0u32, sequence_length as u32, compute_device)?
                .to_dtype(DType::F32)?;
            let inv_freq = self
                .inv_freq
                .to_device(compute_device)?
                .to_dtype(DType::F32)?;
            positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)
        })
        .map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                format!("{MODEL_NAME}: build vision rotary table"),
                e,
            )
        })
    }
}

fn apply_rotary_pos_emb_vision(
    q: &Tensor,
    k: &Tensor,
    cos: &Tensor,
    sin: &Tensor,
) -> Result<(Tensor, Tensor), Error> {
    let q_dtype = q.dtype();
    let k_dtype = k.dtype();
    let q = q
        .to_dtype(DType::F32)
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "cast vision query", e))?;
    let k = k
        .to_dtype(DType::F32)
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "cast vision key", e))?;
    let cos = cos
        .unsqueeze(1)
        .and_then(|value| value.to_dtype(DType::F32))
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "prepare vision cos", e))?;
    let sin = sin
        .unsqueeze(1)
        .and_then(|value| value.to_dtype(DType::F32))
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "prepare vision sin", e))?;
    let q_rot = rotate_half(&q)?;
    let k_rot = rotate_half(&k)?;
    let q = q
        .broadcast_mul(&cos)
        .and_then(|value| value.broadcast_add(&q_rot.broadcast_mul(&sin)?))
        .and_then(|value| value.to_dtype(q_dtype))
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "apply vision query rotary", e))?;
    let k = k
        .broadcast_mul(&cos)
        .and_then(|value| value.broadcast_add(&k_rot.broadcast_mul(&sin)?))
        .and_then(|value| value.to_dtype(k_dtype))
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "apply vision key rotary", e))?;
    Ok((q, k))
}

#[derive(Debug, Clone)]
struct VisionAttention {
    qkv: Linear,
    proj: Linear,
    num_heads: usize,
    head_dim: usize,
    scale: f64,
}

impl VisionAttention {
    fn load(cfg: &Qwen3VlVisionConfig, vb: VarBuilder) -> Result<Self, Error> {
        let head_dim = cfg.head_dim()?;
        let qkv = linear(
            cfg.hidden_size,
            cfg.hidden_size * 3,
            vb.pp("attn").pp("qkv"),
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load vision qkv", e))?;
        let proj = linear(cfg.hidden_size, cfg.hidden_size, vb.pp("attn").pp("proj"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load vision proj", e))?;
        Ok(Self {
            qkv,
            proj,
            num_heads: cfg.num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    /// Non-causal attention over one image's patches.
    fn forward(&self, hidden_states: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor, Error> {
        use crate::runtime::attention::{
            VISION_CHUNKED_ATTN_CHUNK_SIZE, VISION_CHUNKED_ATTN_SEQ_THRESHOLD,
            chunked_vision_attention, flash_attention, scaled_dot_product_attention,
        };
        let sequence_length = hidden_states
            .dim(0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "vision sequence length", e))?;
        let qkv = self
            .qkv
            .forward(hidden_states)
            .and_then(|qkv| qkv.reshape((sequence_length, 3, self.num_heads, self.head_dim)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "vision qkv projection", e))?;
        let q = qkv
            .i((.., 0, .., ..))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "slice vision query", e))?;
        let k = qkv
            .i((.., 1, .., ..))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "slice vision key", e))?;
        let v = qkv
            .i((.., 2, .., ..))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "slice vision value", e))?;
        let (q, k) = apply_rotary_pos_emb_vision(&q, &k, cos, sin)?;

        let q = q
            .transpose(0, 1)
            .and_then(|value| value.unsqueeze(0))
            .and_then(|value| value.contiguous())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "layout vision query", e))?;
        let k = k
            .transpose(0, 1)
            .and_then(|value| value.unsqueeze(0))
            .and_then(|value| value.contiguous())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "layout vision key", e))?;
        let v = v
            .transpose(0, 1)
            .and_then(|value| value.unsqueeze(0))
            .and_then(|value| value.contiguous())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "layout vision value", e))?;

        let attention = match flash_attention(&q, &k, &v, self.scale, false)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "vision flash attention", e))?
        {
            Some(output) => output,
            None if sequence_length > VISION_CHUNKED_ATTN_SEQ_THRESHOLD => {
                chunked_vision_attention(&q, &k, &v, self.scale, VISION_CHUNKED_ATTN_CHUNK_SIZE)
                    .map_err(|e| {
                        candle_to_ocr_inference(MODEL_NAME, "chunked vision attention", e)
                    })?
            }
            None => scaled_dot_product_attention(&q, &k, &v, None, self.scale, false)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "vision attention", e))?,
        };
        let attention = attention
            .transpose(1, 2)
            .and_then(|value| value.reshape((sequence_length, self.num_heads * self.head_dim)))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "reshape vision attention", e))?;
        self.proj
            .forward(&attention)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "vision output projection", e))
    }
}

#[derive(Debug, Clone)]
struct VisionMlp {
    linear_fc1: Linear,
    linear_fc2: Linear,
    activation: Activation,
}

impl VisionMlp {
    fn load(cfg: &Qwen3VlVisionConfig, vb: VarBuilder) -> Result<Self, Error> {
        let linear_fc1 = linear(
            cfg.hidden_size,
            cfg.intermediate_size,
            vb.pp("mlp").pp("linear_fc1"),
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load vision MLP fc1", e))?;
        let linear_fc2 = linear(
            cfg.intermediate_size,
            cfg.hidden_size,
            vb.pp("mlp").pp("linear_fc2"),
        )
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load vision MLP fc2", e))?;
        Ok(Self {
            linear_fc1,
            linear_fc2,
            activation: cfg.hidden_act,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor, Error> {
        let hidden_states = self
            .linear_fc1
            .forward(hidden_states)
            .and_then(|value| self.activation.forward(&value))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "vision MLP fc1", e))?;
        self.linear_fc2
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "vision MLP fc2", e))
    }
}

#[derive(Debug, Clone)]
struct VisionBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attention: VisionAttention,
    mlp: VisionMlp,
}

impl VisionBlock {
    fn load(cfg: &Qwen3VlVisionConfig, vb: VarBuilder) -> Result<Self, Error> {
        let norm_cfg = LayerNormConfig {
            eps: 1e-6,
            ..Default::default()
        };
        let norm1 = layer_norm(cfg.hidden_size, norm_cfg, vb.pp("norm1"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load vision norm1", e))?;
        let norm2 = layer_norm(cfg.hidden_size, norm_cfg, vb.pp("norm2"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load vision norm2", e))?;
        let attention = VisionAttention::load(cfg, vb.clone())?;
        let mlp = VisionMlp::load(cfg, vb)?;
        Ok(Self {
            norm1,
            norm2,
            attention,
            mlp,
        })
    }

    fn forward(&self, hidden_states: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor, Error> {
        let normed = self
            .norm1
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "vision norm1", e))?;
        let attention = self.attention.forward(&normed, cos, sin)?;
        let hidden_states = (hidden_states + attention).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                format!("{MODEL_NAME}: vision attention residual"),
                e,
            )
        })?;
        let normed = self
            .norm2
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "vision norm2", e))?;
        let mlp = self.mlp.forward(&normed)?;
        (hidden_states + mlp).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                format!("{MODEL_NAME}: vision MLP residual"),
                e,
            )
        })
    }
}

/// Spatial merger applied either to the main tower output (LayerNorm over
/// `hidden_size` before the merge reshape) or, for DeepStack taps, to the
/// reshaped merge blocks (`use_postshuffle_norm`, LayerNorm over
/// `hidden_size * merge²`).
#[derive(Debug, Clone)]
struct VisionPatchMerger {
    norm: LayerNorm,
    linear_fc1: Linear,
    linear_fc2: Linear,
    merged_hidden_size: usize,
    merge_group: usize,
    postshuffle_norm: bool,
}

impl VisionPatchMerger {
    /// `vb` must point at the merger directory itself: `merger` for the main
    /// head, `deepstack_merger_list.<i>` for a DeepStack tap (their key
    /// layouts match from there).
    fn load(
        cfg: &Qwen3VlVisionConfig,
        vb: VarBuilder,
        postshuffle_norm: bool,
    ) -> Result<Self, Error> {
        let merge_group = cfg.spatial_merge_size * cfg.spatial_merge_size;
        let merged_hidden_size = cfg.hidden_size * merge_group;
        let norm_cfg = LayerNormConfig {
            eps: 1e-6,
            ..Default::default()
        };
        let norm_size = if postshuffle_norm {
            merged_hidden_size
        } else {
            cfg.hidden_size
        };
        let norm = layer_norm(norm_size, norm_cfg, vb.pp("norm"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load vision merger norm", e))?;
        let linear_fc1 = linear(merged_hidden_size, merged_hidden_size, vb.pp("linear_fc1"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load vision merger fc1", e))?;
        let linear_fc2 = linear(merged_hidden_size, cfg.out_hidden_size, vb.pp("linear_fc2"))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load vision merger fc2", e))?;
        Ok(Self {
            norm,
            linear_fc1,
            linear_fc2,
            merged_hidden_size,
            merge_group,
            postshuffle_norm,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor, Error> {
        let num_patches = hidden_states
            .dim(0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "vision merger patch count", e))?;
        if !num_patches.is_multiple_of(self.merge_group) {
            return Err(Error::InvalidInput {
                message: format!(
                    "{MODEL_NAME} vision merger expected patch count divisible by {}, got {num_patches}",
                    self.merge_group
                ),
            });
        }
        let merged_view = hidden_states
            .reshape((num_patches / self.merge_group, self.merged_hidden_size))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "vision merger reshape", e))?;
        // Pre-shuffle norm runs over raw patches and is reshaped afterwards;
        // post-shuffle norm runs over the concatenated merge blocks.
        let hidden_states = if self.postshuffle_norm {
            self.norm
                .forward(&merged_view)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "vision merger norm", e))?
        } else {
            self.norm
                .forward(hidden_states)
                .and_then(|normed| {
                    normed.reshape((num_patches / self.merge_group, self.merged_hidden_size))
                })
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "vision merger norm", e))?
        };
        let hidden_states = self
            .linear_fc1
            .forward(&hidden_states)
            .and_then(|value| value.gelu_erf())
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "vision merger fc1", e))?;
        self.linear_fc2
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "vision merger fc2", e))
    }
}

pub struct Qwen3VlVisionModel {
    patch_embed: PatchEmbed,
    position_embedding: Tensor,
    position_grid_size: usize,
    blocks: Vec<VisionBlock>,
    merger: VisionPatchMerger,
    deepstack_mergers: Vec<VisionPatchMerger>,
    deepstack_visual_indexes: Vec<usize>,
    rotary_embedding: VisionRotaryEmbedding,
    spatial_merge_size: usize,
}

impl Qwen3VlVisionModel {
    pub fn load(cfg: &Qwen3VlVisionConfig, vb: VarBuilder) -> Result<Self, Error> {
        cfg.validate()?;
        let patch_embed = PatchEmbed::load(cfg, vb.clone())?;
        let position_embedding = vb
            .get(
                (cfg.num_position_embeddings, cfg.hidden_size),
                "pos_embed.weight",
            )
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "load position embedding", e))?;
        let position_grid_size = cfg.position_grid_size()?;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for index in 0..cfg.depth {
            blocks.push(VisionBlock::load(cfg, vb.pp("blocks").pp(index))?);
        }
        let merger = VisionPatchMerger::load(cfg, vb.pp("merger"), false)?;
        let mut deepstack_mergers = Vec::with_capacity(cfg.deepstack_visual_indexes.len());
        for index in 0..cfg.deepstack_visual_indexes.len() {
            deepstack_mergers.push(VisionPatchMerger::load(
                cfg,
                vb.pp("deepstack_merger_list").pp(index),
                true,
            )?);
        }
        let rotary_embedding = VisionRotaryEmbedding::new(cfg.head_dim()? / 2, vb.device())?;
        Ok(Self {
            patch_embed,
            position_embedding,
            position_grid_size,
            blocks,
            merger,
            deepstack_mergers,
            deepstack_visual_indexes: cfg.deepstack_visual_indexes.clone(),
            rotary_embedding,
            spatial_merge_size: cfg.spatial_merge_size,
        })
    }

    /// Run the tower over one batch of images.
    ///
    /// Returns `(merged_embeddings, deepstack_features)`: one row per merged
    /// token in raster order, plus one tensor per configured deepstack index
    /// (empty when `deepstack_visual_indexes` is empty).
    pub fn forward(
        &self,
        pixel_values: &Tensor,
        grid_thw: &[(usize, usize, usize)],
    ) -> Result<(Tensor, Vec<Tensor>), Error> {
        let device = pixel_values.device();
        let max_grid = grid_thw
            .iter()
            .map(|(_, h, w)| (*h).max(*w))
            .max()
            .unwrap_or(0);
        let frequency_table = self.rotary_embedding.forward(max_grid, device)?;

        let mut outputs: Vec<Tensor> = Vec::with_capacity(grid_thw.len());
        let mut deepstack_outputs: Vec<Vec<Tensor>> = (0..self.deepstack_mergers.len())
            .map(|_| Vec::new())
            .collect();
        let mut offset = 0usize;
        for &(t, h, w) in grid_thw {
            let num_patches = t * h * w;
            let patches = pixel_values
                .narrow(0, offset, num_patches)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "narrow patches", e))?;
            offset += num_patches;

            let mut hidden_states = self.patch_embed.forward(&patches)?;
            let pos_embeds = interpolate_position_embedding(
                &self.position_embedding,
                self.position_grid_size,
                (t, h, w),
                self.spatial_merge_size,
            )?;
            hidden_states = hidden_states
                .broadcast_add(&pos_embeds)
                .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "add position embedding", e))?;
            let (cos, sin) = build_vision_rotary_embeddings(
                &frequency_table,
                (t, h, w),
                self.spatial_merge_size,
                device,
            )?;

            for (layer_index, block) in self.blocks.iter().enumerate() {
                hidden_states = block.forward(&hidden_states, &cos, &sin)?;
                if let Some(tap) = self
                    .deepstack_visual_indexes
                    .iter()
                    .position(|&index| index == layer_index)
                {
                    let feature = self.deepstack_mergers[tap].forward(&hidden_states)?;
                    deepstack_outputs[tap].push(feature);
                }
            }
            outputs.push(self.merger.forward(&hidden_states)?);
        }

        let refs: Vec<&Tensor> = outputs.iter().collect();
        let merged = Tensor::cat(&refs, 0)
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "cat merged outputs", e))?;
        let deepstack = deepstack_outputs
            .into_iter()
            .map(|features| {
                let refs: Vec<&Tensor> = features.iter().collect();
                Tensor::cat(&refs, 0)
                    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "cat deepstack features", e))
            })
            .collect::<Result<Vec<_>, _>>()?;
        Ok((merged, deepstack))
    }
}

/// Merge-grouped patch coordinates: block-major raster
/// `(hb, wb)` followed by the intra-block `(h_inner, w_inner)` offsets, the
/// patchifier's output order.
fn merge_grouped_spatial_coordinates(
    grid_thw: (usize, usize, usize),
    merge_size: usize,
) -> Result<Vec<(usize, usize)>, Error> {
    let (grid_t, grid_h, grid_w) = grid_thw;
    if grid_t == 0
        || grid_h == 0
        || grid_w == 0
        || merge_size == 0
        || !grid_h.is_multiple_of(merge_size)
        || !grid_w.is_multiple_of(merge_size)
    {
        return Err(Error::InvalidInput {
            message: format!(
                "{MODEL_NAME} invalid merge-grouped grid: grid={grid_thw:?}, merge={merge_size}"
            ),
        });
    }
    let num_patches = grid_t
        .checked_mul(grid_h)
        .and_then(|value| value.checked_mul(grid_w))
        .ok_or_else(|| Error::InvalidInput {
            message: format!("{MODEL_NAME} merge-grouped patch count overflow"),
        })?;
    let mut coordinates = Vec::with_capacity(num_patches);
    for _ in 0..grid_t {
        for height_block in 0..(grid_h / merge_size) {
            for width_block in 0..(grid_w / merge_size) {
                for height_inner in 0..merge_size {
                    for width_inner in 0..merge_size {
                        coordinates.push((
                            height_block * merge_size + height_inner,
                            width_block * merge_size + width_inner,
                        ));
                    }
                }
            }
        }
    }
    Ok(coordinates)
}

/// Bilinearly interpolate the learned `(side², hidden)` position grid to the
/// patch grid, mirroring `fast_pos_embed_interpolate`: per-axis indices are
/// `linspace(0, side - 1, grid)` floored, with the fractional part as weight.
fn interpolate_position_embedding(
    position_embedding: &Tensor,
    base_grid_size: usize,
    grid_thw: (usize, usize, usize),
    merge_size: usize,
) -> Result<Tensor, Error> {
    let (_, grid_h, grid_w) = grid_thw;
    if base_grid_size == 0 {
        return Err(Error::InvalidInput {
            message: format!(
                "{MODEL_NAME} invalid learned-position grid: base={base_grid_size}, grid={grid_thw:?}"
            ),
        });
    }
    let (num_positions, hidden_size) = position_embedding
        .dims2()
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "position embedding shape", e))?;
    if num_positions != base_grid_size * base_grid_size {
        return Err(Error::Config {
            message: format!(
                "{MODEL_NAME} position embedding rows ({num_positions}) do not match base grid {base_grid_size}x{base_grid_size}"
            ),
        });
    }

    let coordinates = merge_grouped_spatial_coordinates(grid_thw, merge_size)?;
    let num_patches = coordinates.len();
    let mut index00 = Vec::with_capacity(num_patches);
    let mut index01 = Vec::with_capacity(num_patches);
    let mut index10 = Vec::with_capacity(num_patches);
    let mut index11 = Vec::with_capacity(num_patches);
    let mut weight00 = Vec::with_capacity(num_patches);
    let mut weight01 = Vec::with_capacity(num_patches);
    let mut weight10 = Vec::with_capacity(num_patches);
    let mut weight11 = Vec::with_capacity(num_patches);

    // torch.linspace(0, side - 1, steps) computes `start + i * step` in the
    // output dtype; with start = 0 this is `i * step`.
    let h_step = if grid_h == 1 {
        0.0f32
    } else {
        (base_grid_size - 1) as f32 / (grid_h - 1) as f32
    };
    let w_step = if grid_w == 1 {
        0.0f32
    } else {
        (base_grid_size - 1) as f32 / (grid_w - 1) as f32
    };
    let last = base_grid_size - 1;
    for (height, width) in coordinates {
        let source_h = height as f32 * h_step;
        let source_w = width as f32 * w_step;
        let h0 = (source_h as usize).min(last);
        let w0 = (source_w as usize).min(last);
        let h1 = (h0 + 1).min(last);
        let w1 = (w0 + 1).min(last);
        let dh = source_h - h0 as f32;
        let dw = source_w - w0 as f32;

        index00.push((h0 * base_grid_size + w0) as u32);
        index01.push((h0 * base_grid_size + w1) as u32);
        index10.push((h1 * base_grid_size + w0) as u32);
        index11.push((h1 * base_grid_size + w1) as u32);
        weight00.push((1.0 - dh) * (1.0 - dw));
        weight01.push((1.0 - dh) * dw);
        weight10.push(dh * (1.0 - dw));
        weight11.push(dh * dw);
    }

    let weighted = |indices: Vec<u32>, weights: Vec<f32>| -> Result<Tensor, Error> {
        let indices =
            Tensor::from_vec(indices, num_patches, position_embedding.device()).map_err(|e| {
                candle_to_ocr_inference(MODEL_NAME, "position interpolation indices", e)
            })?;
        let weights = Tensor::from_vec(weights, (num_patches, 1), position_embedding.device())
            .and_then(|weights| weights.to_dtype(position_embedding.dtype()))
            .and_then(|weights| weights.broadcast_as((num_patches, hidden_size)))
            .map_err(|e| {
                candle_to_ocr_inference(MODEL_NAME, "position interpolation weights", e)
            })?;
        position_embedding
            .index_select(&indices, 0)
            .and_then(|selected| selected.broadcast_mul(&weights))
            .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "interpolate position embedding", e))
    };

    let output00 = weighted(index00, weight00)?;
    let output01 = weighted(index01, weight01)?;
    let output10 = weighted(index10, weight10)?;
    let output11 = weighted(index11, weight11)?;
    ((&output00 + &output01)
        .and_then(|output| &output + &output10)
        .and_then(|output| &output + &output11))
    .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "sum interpolated position embedding", e))
}

fn build_vision_rotary_embeddings(
    frequency_table: &Tensor,
    grid_thw: (usize, usize, usize),
    merge_size: usize,
    device: &Device,
) -> Result<(Tensor, Tensor), Error> {
    let coordinates = merge_grouped_spatial_coordinates(grid_thw, merge_size)?;
    let num_patches = coordinates.len();
    let (height_ids, width_ids): (Vec<u32>, Vec<u32>) = coordinates
        .into_iter()
        .map(|(height, width)| (height as u32, width as u32))
        .unzip();
    let height_ids = Tensor::from_vec(height_ids, num_patches, device)
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "vision height positions", e))?;
    let width_ids = Tensor::from_vec(width_ids, num_patches, device)
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "vision width positions", e))?;
    let height_frequencies = frequency_table
        .index_select(&height_ids, 0)
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "gather vision height frequencies", e))?;
    let width_frequencies = frequency_table
        .index_select(&width_ids, 0)
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "gather vision width frequencies", e))?;
    let rotary = Tensor::cat(&[&height_frequencies, &width_frequencies], 1)
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "join vision rotary frequencies", e))?;
    let embedding = Tensor::cat(&[&rotary, &rotary], 1)
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "expand vision rotary frequencies", e))?;
    let cos = embedding
        .cos()
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "vision rotary cos", e))?;
    let sin = embedding
        .sin()
        .map_err(|e| candle_to_ocr_inference(MODEL_NAME, "vision rotary sin", e))?;
    Ok((cos, sin))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn tiny_config() -> Qwen3VlVisionConfig {
        Qwen3VlVisionConfig {
            model_type: "qwen3_vl".to_string(),
            depth: 2,
            hidden_size: 4,
            intermediate_size: 8,
            num_heads: 1,
            in_channels: 3,
            patch_size: 1,
            spatial_merge_size: 1,
            temporal_patch_size: 1,
            out_hidden_size: 4,
            num_position_embeddings: 4,
            hidden_act: Activation::GeluPytorchTanh,
            deepstack_visual_indexes: vec![0],
        }
    }

    fn test_tensors(cfg: &Qwen3VlVisionConfig, device: &Device) -> HashMap<String, Tensor> {
        let mut tensors = HashMap::new();
        let zeros_2d =
            |rows: usize, cols: usize| Tensor::zeros((rows, cols), DType::F32, device).unwrap();
        let ones = |n: usize| Tensor::ones(n, DType::F32, device).unwrap();
        let zeros = |n: usize| Tensor::zeros(n, DType::F32, device).unwrap();
        tensors.insert(
            "patch_embed.proj.weight".to_string(),
            Tensor::zeros((cfg.hidden_size, 3, 1, 1, 1), DType::F32, device).unwrap(),
        );
        tensors.insert("patch_embed.proj.bias".to_string(), zeros(cfg.hidden_size));
        tensors.insert(
            "pos_embed.weight".to_string(),
            zeros_2d(cfg.num_position_embeddings, cfg.hidden_size),
        );
        for prefix in [
            "merger",
            "deepstack_merger_list.0",
            "deepstack_merger_list.1",
        ] {
            // With spatial_merge_size 1 both norm variants cover
            // `hidden_size` elements.
            let norm_size = cfg.hidden_size;
            tensors.insert(format!("{prefix}.norm.weight"), ones(norm_size));
            tensors.insert(format!("{prefix}.norm.bias"), zeros(norm_size));
            tensors.insert(format!("{prefix}.linear_fc1.weight"), zeros_2d(4, 4));
            tensors.insert(format!("{prefix}.linear_fc1.bias"), zeros(4));
            tensors.insert(format!("{prefix}.linear_fc2.weight"), zeros_2d(4, 4));
            tensors.insert(format!("{prefix}.linear_fc2.bias"), zeros(4));
        }
        for layer in 0..cfg.depth {
            let prefix = format!("blocks.{layer}");
            tensors.insert(format!("{prefix}.norm1.weight"), ones(4));
            tensors.insert(format!("{prefix}.norm1.bias"), zeros(4));
            tensors.insert(format!("{prefix}.norm2.weight"), ones(4));
            tensors.insert(format!("{prefix}.norm2.bias"), zeros(4));
            tensors.insert(format!("{prefix}.attn.qkv.weight"), zeros_2d(12, 4));
            tensors.insert(format!("{prefix}.attn.qkv.bias"), zeros(12));
            tensors.insert(format!("{prefix}.attn.proj.weight"), zeros_2d(4, 4));
            tensors.insert(format!("{prefix}.attn.proj.bias"), zeros(4));
            tensors.insert(format!("{prefix}.mlp.linear_fc1.weight"), zeros_2d(8, 4));
            tensors.insert(format!("{prefix}.mlp.linear_fc1.bias"), zeros(8));
            tensors.insert(format!("{prefix}.mlp.linear_fc2.weight"), zeros_2d(4, 8));
            tensors.insert(format!("{prefix}.mlp.linear_fc2.bias"), zeros(4));
        }
        tensors
    }

    #[test]
    fn loads_weight_names_and_forwards_with_deepstack() -> Result<(), Error> {
        let cfg = tiny_config();
        let device = Device::Cpu;
        let vb = VarBuilder::from_tensors(test_tensors(&cfg, &device), DType::F32, &device);
        let model = Qwen3VlVisionModel::load(&cfg, vb)?;
        let (merged, deepstack) =
            model.forward(&Tensor::zeros((4, 3), DType::F32, &device)?, &[(1, 2, 2)])?;
        assert_eq!(merged.dims(), &[4, 4]);
        // One deepstack tap at layer 0, one tensor over all 4 merged tokens.
        assert_eq!(deepstack.len(), 1);
        assert_eq!(deepstack[0].dims(), &[4, 4]);
        Ok(())
    }

    #[test]
    fn rejects_out_of_range_deepstack_index() {
        let mut cfg = tiny_config();
        cfg.deepstack_visual_indexes = vec![5];
        assert!(cfg.validate().is_err());
    }

    #[test]
    fn learned_position_interpolation_matches_bilinear_reference() -> Result<(), Error> {
        let weights = Tensor::from_vec(vec![0.0f32, 1.0, 2.0, 3.0], (4, 1), &Device::Cpu)?;
        let output = interpolate_position_embedding(&weights, 2, (1, 3, 3), 1)?;
        let values = output.flatten_all()?.to_vec1::<f32>()?;
        let expected = [0.0, 0.5, 1.0, 1.0, 1.5, 2.0, 2.0, 2.5, 3.0];
        for (actual, expected) in values.iter().zip(expected) {
            assert!((actual - expected).abs() < 1e-6, "{actual} != {expected}");
        }
        Ok(())
    }

    #[test]
    fn learned_positions_follow_spatial_merge_group_order() -> Result<(), Error> {
        let weights = Tensor::from_vec(
            (0..16).map(|value| value as f32).collect::<Vec<_>>(),
            (16, 1),
            &Device::Cpu,
        )?;
        let output = interpolate_position_embedding(&weights, 4, (1, 4, 4), 2)?;
        let values = output.flatten_all()?.to_vec1::<f32>()?;
        assert_eq!(
            values,
            [
                0.0, 1.0, 4.0, 5.0, 2.0, 3.0, 6.0, 7.0, 8.0, 9.0, 12.0, 13.0, 10.0, 11.0, 14.0,
                15.0,
            ]
        );
        Ok(())
    }

    #[test]
    fn merge_grouped_coordinates_match_patchifier_order() {
        let coords = merge_grouped_spatial_coordinates((1, 4, 6), 2).unwrap();
        // 2x3 merge blocks, each block listing its 4 inner patches.
        let expected = [
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (0, 4),
            (0, 5),
            (1, 4),
            (1, 5),
            (2, 0),
            (2, 1),
            (3, 0),
            (3, 1),
            (2, 2),
            (2, 3),
            (3, 2),
            (3, 3),
            (2, 4),
            (2, 5),
            (3, 4),
            (3, 5),
        ];
        assert_eq!(coords, expected);
    }

    #[test]
    fn rejects_invalid_grid() {
        assert!(merge_grouped_spatial_coordinates((1, 3, 4), 2).is_err());
        assert!(merge_grouped_spatial_coordinates((0, 4, 4), 2).is_err());
    }
}
