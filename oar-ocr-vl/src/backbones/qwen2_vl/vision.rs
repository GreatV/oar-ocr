use crate::attention::{
    VISION_CHUNKED_ATTN_CHUNK_SIZE, VISION_CHUNKED_ATTN_SEQ_THRESHOLD, chunked_vision_attention,
    scaled_dot_product_attention,
};
use crate::backbones::patch_merger::{MergerKeys, SpatialPatchMerger};
use crate::error::Error;
use crate::runtime::errors::{candle_to_ocr_inference, candle_to_ocr_processing};
use crate::runtime::tensor::{
    VisionRotaryEmbedding, apply_rotary_f32, merge_grouped_positions, vision_rotary_cos_sin,
};
use candle_core::{IndexOp, Tensor};
use candle_nn::{LayerNorm, LayerNormConfig, Linear, Module, VarBuilder, layer_norm, linear};
use serde::Deserialize;

fn default_hidden_act() -> String {
    "quick_gelu".to_string()
}

/// Shared Qwen2-VL vision-tower configuration.
#[derive(Debug, Clone, Deserialize)]
pub struct Qwen2VlVisionConfig {
    pub depth: usize,
    pub embed_dim: usize,
    pub hidden_size: usize,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: String,
    pub mlp_ratio: f64,
    pub num_heads: usize,
    #[serde(default, rename = "in_chans")]
    in_chans: Option<usize>,
    #[serde(default, rename = "in_channels")]
    in_channels_alias: Option<usize>,
    pub patch_size: usize,
    pub spatial_merge_size: usize,
    #[serde(default)]
    pub spatial_patch_size: Option<usize>,
    pub temporal_patch_size: usize,
}

impl Qwen2VlVisionConfig {
    pub fn mlp_hidden_dim(&self) -> usize {
        (self.embed_dim as f64 * self.mlp_ratio).round() as usize
    }

    /// Input channel count, accepting both checkpoint spellings.
    pub fn in_channels(&self) -> usize {
        self.in_chans.or(self.in_channels_alias).unwrap_or(3)
    }
}

fn quick_gelu(xs: &Tensor) -> Result<Tensor, Error> {
    let scaled = (xs * 1.702).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            "MinerU2.5: quick_gelu scale failed",
            e,
        )
    })?;
    let sig = candle_nn::ops::sigmoid(&scaled).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            "MinerU2.5: quick_gelu sigmoid failed",
            e,
        )
    })?;
    (xs * sig).map_err(|e| {
        candle_to_ocr_processing(
            crate::error::ProcessingStage::TensorOperation,
            "MinerU2.5: quick_gelu multiply failed",
            e,
        )
    })
}

#[derive(Debug, Clone, Copy)]
enum VisionAct {
    QuickGelu,
    Gelu,
    Silu,
}

impl VisionAct {
    fn from_str(name: &str) -> Result<Self, Error> {
        match name {
            "quick_gelu" => Ok(Self::QuickGelu),
            "gelu" | "gelu_new" | "gelu_pytorch_tanh" => Ok(Self::Gelu),
            "silu" => Ok(Self::Silu),
            _ => Err(Error::Config {
                message: format!("MinerU2.5: unsupported vision hidden_act '{name}'"),
            }),
        }
    }

    fn forward(self, xs: &Tensor) -> Result<Tensor, Error> {
        match self {
            Self::QuickGelu => quick_gelu(xs),
            Self::Gelu => xs.gelu_erf().map_err(|e| {
                candle_to_ocr_processing(
                    crate::error::ProcessingStage::TensorOperation,
                    "MinerU2.5: vision gelu failed",
                    e,
                )
            }),
            Self::Silu => candle_nn::ops::silu(xs).map_err(|e| {
                candle_to_ocr_processing(
                    crate::error::ProcessingStage::TensorOperation,
                    "MinerU2.5: vision silu failed",
                    e,
                )
            }),
        }
    }
}

#[derive(Debug, Clone)]
struct PatchEmbed {
    weight: Tensor,
}

impl PatchEmbed {
    fn load(cfg: &Qwen2VlVisionConfig, vb: VarBuilder) -> Result<Self, Error> {
        let patch_dim =
            cfg.in_channels() * cfg.temporal_patch_size * cfg.patch_size * cfg.patch_size;
        let weight = match vb.get((cfg.embed_dim, patch_dim), "patch_embed.proj.weight") {
            Ok(weight) => weight,
            Err(_) => {
                let weight = vb
                    .get(
                        (
                            cfg.embed_dim,
                            cfg.in_channels(),
                            cfg.temporal_patch_size,
                            cfg.patch_size,
                            cfg.patch_size,
                        ),
                        "patch_embed.proj.weight",
                    )
                    .map_err(|e| candle_to_ocr_inference("MinerU2.5", "load patch_embed", e))?;
                weight
                    .reshape((cfg.embed_dim, patch_dim))
                    .map_err(|e| candle_to_ocr_inference("MinerU2.5", "reshape patch_embed", e))?
            }
        };
        Ok(Self { weight })
    }

    fn forward(&self, patches: &Tensor) -> Result<Tensor, Error> {
        let weight_t = self.weight.transpose(0, 1).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "MinerU2.5: patch_embed weight transpose failed",
                e,
            )
        })?;
        let patches = patches.to_dtype(self.weight.dtype()).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "MinerU2.5: patch_embed input cast failed",
                e,
            )
        })?;
        patches
            .matmul(&weight_t)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "patch_embed matmul", e))
    }
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
    fn load(cfg: &Qwen2VlVisionConfig, vb: VarBuilder) -> Result<Self, Error> {
        let qkv = linear(cfg.embed_dim, cfg.embed_dim * 3, vb.pp("attn.qkv"))
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "load vision qkv", e))?;
        let proj = linear(cfg.embed_dim, cfg.embed_dim, vb.pp("attn.proj"))
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "load vision proj", e))?;

        if !cfg.embed_dim.is_multiple_of(cfg.num_heads) {
            return Err(Error::Config {
                message: format!(
                    "MinerU2.5: vision embed_dim {} not divisible by num_heads {}",
                    cfg.embed_dim, cfg.num_heads
                ),
            });
        }

        let head_dim = cfg.embed_dim / cfg.num_heads;
        Ok(Self {
            qkv,
            proj,
            num_heads: cfg.num_heads,
            head_dim,
            scale: 1.0 / (head_dim as f64).sqrt(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor, Error> {
        let seq_len = hidden_states
            .dim(0)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision hidden_states dim", e))?;
        let qkv = self
            .qkv
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision qkv", e))?;
        let qkv = qkv
            .reshape((seq_len, 3, self.num_heads, self.head_dim))
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision qkv reshape", e))?;
        let q = qkv
            .i((.., 0, .., ..))
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision q slice", e))?;
        let k = qkv
            .i((.., 1, .., ..))
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision k slice", e))?;
        let v = qkv
            .i((.., 2, .., ..))
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision v slice", e))?;

        let cos = cos
            .unsqueeze(1)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision cos layout", e))?;
        let sin = sin
            .unsqueeze(1)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision sin layout", e))?;
        let (q, k) = apply_rotary_f32(&q, &k, &cos, &sin, "MinerU2.5")?;

        let q = q
            .transpose(0, 1)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision q transpose", e))?
            .unsqueeze(0)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision q unsqueeze", e))?
            .contiguous()
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision q contiguous", e))?;
        let k = k
            .transpose(0, 1)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision k transpose", e))?
            .unsqueeze(0)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision k unsqueeze", e))?
            .contiguous()
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision k contiguous", e))?;
        let v = v
            .transpose(0, 1)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision v transpose", e))?
            .unsqueeze(0)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision v unsqueeze", e))?
            .contiguous()
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision v contiguous", e))?;

        let seq_len = q
            .dim(2)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision q dim", e))?;
        let attn = if seq_len > VISION_CHUNKED_ATTN_SEQ_THRESHOLD {
            chunked_vision_attention(&q, &k, &v, self.scale, VISION_CHUNKED_ATTN_CHUNK_SIZE)
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision attention", e))?
        } else {
            scaled_dot_product_attention(&q, &k, &v, None, self.scale, false)
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision attention", e))?
        };
        let attn = attn
            .transpose(1, 2)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision attn transpose", e))?
            .reshape((seq_len, self.num_heads * self.head_dim))
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision attn reshape", e))?;
        self.proj
            .forward(&attn)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision proj", e))
    }
}

#[derive(Debug, Clone)]
struct VisionMlp {
    fc1: Linear,
    fc2: Linear,
    act: VisionAct,
}

impl VisionMlp {
    fn load(cfg: &Qwen2VlVisionConfig, vb: VarBuilder) -> Result<Self, Error> {
        let hidden_dim = cfg.mlp_hidden_dim();
        let fc1 = linear(cfg.embed_dim, hidden_dim, vb.pp("mlp.fc1"))
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "load vision fc1", e))?;
        let fc2 = linear(hidden_dim, cfg.embed_dim, vb.pp("mlp.fc2"))
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "load vision fc2", e))?;
        let act = VisionAct::from_str(cfg.hidden_act.as_str())?;
        Ok(Self { fc1, fc2, act })
    }

    fn forward(&self, xs: &Tensor) -> Result<Tensor, Error> {
        let x = self
            .fc1
            .forward(xs)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision fc1", e))?;
        let x = self.act.forward(&x)?;
        self.fc2
            .forward(&x)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision fc2", e))
    }
}

#[derive(Debug, Clone)]
struct VisionBlock {
    norm1: LayerNorm,
    norm2: LayerNorm,
    attn: VisionAttention,
    mlp: VisionMlp,
}

impl VisionBlock {
    fn load(cfg: &Qwen2VlVisionConfig, vb: VarBuilder) -> Result<Self, Error> {
        let norm_cfg = LayerNormConfig {
            eps: 1e-6,
            ..Default::default()
        };
        let norm1 = layer_norm(cfg.embed_dim, norm_cfg, vb.pp("norm1"))
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "load vision norm1", e))?;
        let norm2 = layer_norm(cfg.embed_dim, norm_cfg, vb.pp("norm2"))
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "load vision norm2", e))?;
        let attn = VisionAttention::load(cfg, vb.clone())?;
        let mlp = VisionMlp::load(cfg, vb)?;
        Ok(Self {
            norm1,
            norm2,
            attn,
            mlp,
        })
    }

    fn forward(&self, hidden_states: &Tensor, cos: &Tensor, sin: &Tensor) -> Result<Tensor, Error> {
        let normed = self
            .norm1
            .forward(hidden_states)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision norm1 forward", e))?;
        let attn_out = self.attn.forward(&normed, cos, sin)?;
        let hidden_states = (hidden_states + attn_out).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "MinerU2.5: vision attn residual failed",
                e,
            )
        })?;

        let normed = self
            .norm2
            .forward(&hidden_states)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision norm2 forward", e))?;
        let mlp_out = self.mlp.forward(&normed)?;
        (hidden_states + mlp_out).map_err(|e| {
            candle_to_ocr_processing(
                crate::error::ProcessingStage::TensorOperation,
                "MinerU2.5: vision mlp residual failed",
                e,
            )
        })
    }
}

pub struct Qwen2VlVisionModel {
    patch_embed: PatchEmbed,
    blocks: Vec<VisionBlock>,
    merger: Option<SpatialPatchMerger>,
    rotary_pos_emb: VisionRotaryEmbedding,
    spatial_merge_size: usize,
}

#[allow(dead_code)]
impl Qwen2VlVisionModel {
    pub fn load(cfg: &Qwen2VlVisionConfig, vb: VarBuilder) -> Result<Self, Error> {
        Self::load_inner(cfg, vb, true)
    }

    /// Load only the patch-embedding + transformer backbone, skipping the
    /// `merger` head. Used by checkpoints (e.g. MinerU-Diffusion) that set the
    /// vision merger to identity and project the raw per-patch hidden states
    /// with an external abstractor instead. Pair with [`forward_tokens`].
    ///
    /// [`forward_tokens`]: Qwen2VlVisionModel::forward_tokens
    pub(crate) fn load_backbone(cfg: &Qwen2VlVisionConfig, vb: VarBuilder) -> Result<Self, Error> {
        Self::load_inner(cfg, vb, false)
    }

    fn load_inner(
        cfg: &Qwen2VlVisionConfig,
        vb: VarBuilder,
        with_merger: bool,
    ) -> Result<Self, Error> {
        let patch_embed = PatchEmbed::load(cfg, vb.clone())?;
        let mut blocks = Vec::with_capacity(cfg.depth);
        for i in 0..cfg.depth {
            let block_vb = vb.pp(format!("blocks.{i}"));
            blocks.push(VisionBlock::load(cfg, block_vb)?);
        }
        let merger = if with_merger {
            Some(SpatialPatchMerger::load(
                vb.pp("merger"),
                MergerKeys::QWEN2_VL,
                cfg.embed_dim,
                cfg.spatial_merge_size,
                cfg.hidden_size,
                false,
                "MinerU2.5",
            )?)
        } else {
            None
        };
        let head_dim = cfg.embed_dim / cfg.num_heads;
        if !head_dim.is_multiple_of(2) {
            return Err(Error::Config {
                message: format!(
                    "MinerU2.5: head_dim {} must be even for rotary embeddings",
                    head_dim
                ),
            });
        }
        let rotary_pos_emb =
            VisionRotaryEmbedding::new(head_dim / 2, 10000.0, "MinerU2.5", vb.device())?;
        Ok(Self {
            patch_embed,
            blocks,
            merger,
            rotary_pos_emb,
            spatial_merge_size: cfg.spatial_merge_size,
        })
    }

    /// Run the patch embedding + transformer blocks per image and return the
    /// per-patch hidden states (one row per patch, dim `embed_dim`), without
    /// applying the spatial merger. Attention is isolated per image.
    pub(crate) fn forward_tokens(
        &self,
        pixel_values: &Tensor,
        grid_thw: &[(usize, usize, usize)],
    ) -> Result<Tensor, Error> {
        let max_grid = grid_thw
            .iter()
            .map(|(_, h, w)| (*h).max(*w))
            .max()
            .unwrap_or(0);
        let rotary_full = self
            .rotary_pos_emb
            .forward(max_grid, pixel_values.device())?;

        let mut outputs: Vec<Tensor> = Vec::with_capacity(grid_thw.len());
        let mut offset = 0usize;
        for &(t, h, w) in grid_thw {
            let num_patches = t * h * w;
            let patches = pixel_values
                .narrow(0, offset, num_patches)
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision narrow patches", e))?;
            offset += num_patches;

            let mut hidden = self.patch_embed.forward(&patches)?;
            let (hpos, wpos) =
                merge_grouped_positions((t, h, w), self.spatial_merge_size, "MinerU2.5")?;
            let (cos, sin) = vision_rotary_cos_sin(
                &rotary_full,
                &hpos,
                &wpos,
                "MinerU2.5",
                pixel_values.device(),
            )?;
            for block in &self.blocks {
                hidden = block.forward(&hidden, &cos, &sin)?;
            }
            outputs.push(hidden);
        }

        let refs: Vec<&Tensor> = outputs.iter().collect();
        Tensor::cat(&refs, 0)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision tokens cat", e))
    }

    pub fn forward(
        &self,
        pixel_values: &Tensor,
        grid_thw: &[(usize, usize, usize)],
    ) -> Result<Tensor, Error> {
        let merger = self.merger.as_ref().ok_or_else(|| Error::Config {
            message: "MinerU2.5: vision merger not loaded (backbone-only model)".to_string(),
        })?;
        let mut outputs: Vec<Tensor> = Vec::with_capacity(grid_thw.len());

        let max_grid = grid_thw
            .iter()
            .map(|(_, h, w)| (*h).max(*w))
            .max()
            .unwrap_or(0);
        let rotary_full = self
            .rotary_pos_emb
            .forward(max_grid, pixel_values.device())?;

        let mut offset = 0usize;
        for &(t, h, w) in grid_thw {
            let num_patches = t * h * w;
            let patches = pixel_values
                .narrow(0, offset, num_patches)
                .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision narrow patches", e))?;
            offset += num_patches;

            let mut hidden = self.patch_embed.forward(&patches)?;

            let (hpos, wpos) =
                merge_grouped_positions((t, h, w), self.spatial_merge_size, "MinerU2.5")?;
            let (cos, sin) = vision_rotary_cos_sin(
                &rotary_full,
                &hpos,
                &wpos,
                "MinerU2.5",
                pixel_values.device(),
            )?;

            for block in &self.blocks {
                hidden = block.forward(&hidden, &cos, &sin)?;
            }

            let merged = merger.forward(&hidden)?;
            outputs.push(merged);
        }

        let refs: Vec<&Tensor> = outputs.iter().collect();
        Tensor::cat(&refs, 0)
            .map_err(|e| candle_to_ocr_inference("MinerU2.5", "vision outputs cat", e))
    }
}
