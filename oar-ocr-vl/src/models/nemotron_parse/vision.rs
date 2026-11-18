//! Global ViT, learned summary tokens and the horizontal patch neck from the OpenMDW checkpoint.
use super::{position, processing::Page};
use crate::{backbones::vit::mlp_residual, runtime::attention::global_vision_attention};
use candle_core::{DType, Result, Tensor};
use candle_nn::{
    Activation, LayerNorm, Linear, Module, VarBuilder, layer_norm, linear, linear_no_bias,
};

const WIDTH: usize = 1152;
const HEADS: usize = 16;

struct Block {
    norm1: LayerNorm,
    qkv: Linear,
    out: Linear,
    norm2: LayerNorm,
    up: Linear,
    down: Linear,
}
impl Block {
    fn load(vb: VarBuilder) -> Result<Self> {
        Ok(Self {
            norm1: layer_norm(WIDTH, 1e-6, vb.pp("norm1"))?,
            qkv: linear(WIDTH, 3 * WIDTH, vb.pp("attn.qkv"))?,
            out: linear(WIDTH, WIDTH, vb.pp("attn.proj"))?,
            norm2: layer_norm(WIDTH, 1e-6, vb.pp("norm2"))?,
            up: linear(WIDTH, 4304, vb.pp("mlp.fc1"))?,
            down: linear(4304, WIDTH, vb.pp("mlp.fc2"))?,
        })
    }
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let (batch, n, _) = x.dims3()?;
        let qkv =
            x.apply(&self.norm1)?
                .apply(&self.qkv)?
                .reshape((batch, n, 3, HEADS, WIDTH / HEADS))?;
        let heads = |i| {
            qkv.narrow(2, i, 1)?
                .squeeze(2)?
                .transpose(1, 2)?
                .contiguous()
        };
        let context = global_vision_attention(&heads(0)?, &heads(1)?, &heads(2)?)?
            .transpose(1, 2)?
            .reshape((batch, n, WIDTH))?;
        mlp_residual(
            &(x + context.apply(&self.out)?)?,
            &self.norm2,
            &self.up,
            Activation::Gelu,
            &self.down,
        )
    }
}
pub(super) struct Encoder {
    embed: Linear,
    positions: Tensor,
    tokens: Tensor,
    blocks: Vec<Block>,
    summary_idxs: Tensor,
    spatial: Linear,
    spatial_norm: LayerNorm,
    merge: Linear,
    merge_norm: LayerNorm,
    summary: Linear,
    summary_norm: LayerNorm,
}
impl Encoder {
    pub fn load(vb: VarBuilder) -> Result<Self> {
        let radio = vb.pp("model_encoder.radio_model");
        let tower = radio.pp("model");
        let pg = tower.pp("patch_generator");
        // A [1,4] convolution is a linear map of four channel-major horizontal patches.
        let merge = vb
            .get((1024, 1024, 1, 4), "conv2.weight")?
            .reshape((1024, 4096))?;
        Ok(Self {
            embed: linear_no_bias(768, WIDTH, pg.pp("embedder"))?,
            positions: pg.get((1, 16384, WIDTH), "pos_embed")?,
            tokens: pg.get((10, WIDTH), "cls_token.token")?.unsqueeze(0)?,
            blocks: (0..27)
                .map(|i| Block::load(tower.pp("blocks").pp(i)))
                .collect::<Result<_>>()?,
            summary_idxs: radio.get_with_hints_dtype(
                2,
                "summary_idxs",
                Default::default(),
                DType::U32,
            )?,
            spatial: Linear::new(
                vb.get((1024, WIDTH, 1), "conv1.weight")?.squeeze(2)?,
                Some(vb.get(1024, "conv1.bias")?),
            ),
            spatial_norm: layer_norm(1024, 1e-6, vb.pp("layer_norm1"))?,
            merge: Linear::new(merge, None),
            merge_norm: layer_norm(1024, 1e-6, vb.pp("layer_norm2"))?,
            summary: linear(2304, 1024, vb.pp("sum_proj"))?,
            summary_norm: layer_norm(1024, 1e-6, vb.pp("layer_norm3"))?,
        })
    }
    pub fn forward(&self, page: &Page) -> Result<Tensor> {
        let patches = self.embed.forward(&page.patches)?.unsqueeze(0)?;
        let positioned = (patches + position::positions(&self.positions, page)?)?;
        let mut hidden = Tensor::cat(&[&self.tokens, &positioned], 1)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }
        let summary = hidden
            .index_select(&self.summary_idxs, 1)?
            .reshape((1, 1, 2304))?
            .apply(&self.summary)?
            .apply(&self.summary_norm)?;
        let spatial = hidden
            .narrow(1, 10, page.rows * page.cols)?
            .apply(&self.spatial)?
            .apply(&self.spatial_norm)?;
        let merged = spatial
            .reshape((page.rows, page.cols / 4, 4, 1024))?
            .transpose(2, 3)?
            .contiguous()?
            .reshape((1, page.rows * page.cols / 4, 4096))?
            .apply(&self.merge)?
            .apply(&self.merge_norm)?;
        Tensor::cat(&[merged, summary], 1)
    }
}
