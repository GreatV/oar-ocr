//! DeepSeek-V2 decoder with Unlimited-OCR-2's DASH block-sparse attention.
//!
//! Every query attends its last `local_window` keys plainly, plus the
//! remote 32-token key blocks a per-layer router selects: block summary keys
//! (`head_cls` attention pooling of the router keys) are scored against the
//! router query, the top 16 go through 1.5-entmax, and the first 12 are read
//! with a mean-centred `log(prior)` logit bias. Remote candidates are the
//! image blocks of the query's own page. Queries and keys use a two-axis
//! (`tok`, `page`) rotary embedding. The math follows the checkpoint's
//! `modeling_deepseekv2.py` inference path (`_infer_prefill_attend`,
//! `_dash_decode_forward`, `_infer_route_select`).

use super::config::UnlimitedOcrConfig;
use super::infer;
use crate::Error;
use crate::backbones::deepseek_v2::FeedForward;
use crate::runtime::cache::TrimmableKvCache;
use candle_core::{D, DType, Device, Tensor};
use candle_nn::{
    Embedding, Linear, Module, RmsNorm, VarBuilder, embedding, linear_no_bias, rms_norm,
};
use std::cell::RefCell;

const ACT_CLAMP: f64 = 1.0e4;
const ENTMAX_NEG: f32 = -1.0e4;
/// Router window: the top-M block scores 1.5-entmax normalizes over.
const ROUTER_TOP_M: usize = 16;
/// Queries per chunk of the dense prefill attention.
const PREFILL_QUERY_CHUNK: usize = 256;

/// Router settings shared by all layers, plus each layer's scale and floor.
#[derive(Debug, Clone)]
struct RouterParams {
    block_size: usize,
    local_blocks: usize,
    budget: usize,
    sigma: f32,
}

/// Two-axis RoPE inverse frequencies: `n_tok` token pairs resampled over
/// the full spectrum, then the page pairs; the page coordinate is folded
/// modulo `page_mod`.
#[derive(Debug, Clone)]
struct PageRope {
    inv_freq: Vec<f32>,
    n_tok: usize,
    page_mod: i64,
}

impl PageRope {
    fn new(config: &UnlimitedOcrConfig, head_dim: usize) -> Self {
        let sparse = &config.sparse_attention_config;
        let half = head_dim / 2;
        let page_pairs = sparse.sparse_rope_page_pairs;
        let n_tok = half - page_pairs;
        let mut inv_freq: Vec<f32> = (0..n_tok)
            .map(|i| 1.0 / sparse.sparse_rope_tok_theta.powf(i as f32 / n_tok as f32))
            .collect();
        inv_freq.extend((0..page_pairs).map(|i| {
            1.0 / sparse
                .sparse_rope_page_theta
                .powf(i as f32 / page_pairs as f32)
        }));
        Self {
            inv_freq,
            n_tok,
            page_mod: sparse.sparse_rope_page_mod as i64,
        }
    }

    /// `(cos, sin)` of shape `(len, head_dim / 2)` for `(tok, page)` pairs,
    /// computed in f32 and cast to `dtype` like the reference.
    fn tables(
        &self,
        coords: &[(i64, i64)],
        device: &Device,
        dtype: DType,
    ) -> Result<(Tensor, Tensor), Error> {
        let half = self.inv_freq.len();
        let mut angles = Vec::with_capacity(coords.len() * half);
        for &(tok, page) in coords {
            let page = if self.page_mod > 0 {
                page.rem_euclid(self.page_mod)
            } else {
                page
            };
            for (index, &freq) in self.inv_freq.iter().enumerate() {
                let position = if index < self.n_tok { tok } else { page };
                angles.push(position as f32 * freq);
            }
        }
        let cos: Vec<f32> = angles.iter().map(|a| a.cos()).collect();
        let sin: Vec<f32> = angles.iter().map(|a| a.sin()).collect();
        let table = |values: Vec<f32>| {
            Tensor::from_vec(values, (coords.len(), half), device)
                .and_then(|t| t.to_dtype(dtype))
                .map_err(infer)
        };
        Ok((table(cos)?, table(sin)?))
    }
}

/// Page bookkeeping of one request: the RoPE coordinates and the decode
/// candidate domain, advanced by every decoded token.
#[derive(Debug, Clone)]
pub(super) struct PageState {
    /// RoPE `(tok, page)` of the last processed position.
    page: i64,
    tok: i64,
    /// Prefill tokens per RoPE page bucket.
    prefill_counts: Vec<i64>,
    /// Image page range `[min, max]` of each prefill block (`-1` = none).
    block_pages: Vec<(i64, i64)>,
    /// `<PAGE>` markers up to the last prompt image token.
    prompt_pages: i64,
    /// `<PAGE>` markers seen so far, prompt included.
    markers: i64,
}

/// Per-position page metadata of a prompt.
struct PromptPages {
    /// RoPE coordinates of each prompt token.
    coords: Vec<(i64, i64)>,
    /// Image-page id of each token (`-1` for non-image tokens).
    image_pages: Vec<i64>,
    /// Candidate page of each query (`-1` = unconstrained).
    query_pages: Vec<i64>,
    state: PageState,
}

fn prompt_pages(
    ids: &[u32],
    image_token_id: u32,
    page_token_id: u32,
    block_size: usize,
) -> Result<PromptPages, Error> {
    let last_image = ids
        .iter()
        .rposition(|&id| id == image_token_id)
        .ok_or_else(|| Error::invalid_input("Unlimited-OCR prompt has no image token"))?;
    let mut cum = Vec::with_capacity(ids.len());
    let mut count = 0i64;
    for &id in ids {
        count += i64::from(id == page_token_id);
        cum.push(count);
    }
    let prompt_pages = cum[last_image];
    if prompt_pages == 0 {
        return Err(Error::invalid_input(
            "Unlimited-OCR prompt has no <PAGE> marker before its images",
        ));
    }
    // RoPE buckets: prefill rows take the 1-based marker count, the rest
    // `count - prompt_pages`; 0 is the no-page bucket.
    let rope_pages: Vec<i64> = cum
        .iter()
        .enumerate()
        .map(|(index, &c)| {
            if index <= last_image {
                c
            } else {
                (c - prompt_pages).max(0)
            }
        })
        .collect();
    let buckets = *rope_pages.iter().max().unwrap_or(&0) as usize + 1;
    let mut running = vec![0i64; buckets];
    let coords: Vec<(i64, i64)> = rope_pages
        .iter()
        .map(|&page| {
            let tok = running[page as usize];
            running[page as usize] += 1;
            (tok, page)
        })
        .collect();
    // Prompt tokens all count as prefill for `pf_cnt` (the reference's
    // `in_prefill` only gates the page numbering).
    let mut prefill_counts = vec![0i64; buckets];
    for (index, &page) in rope_pages.iter().enumerate() {
        if index <= last_image {
            prefill_counts[page as usize] += 1;
        }
    }
    let image_pages: Vec<i64> = ids
        .iter()
        .zip(&cum)
        .map(|(&id, &c)| if id == image_token_id { c - 1 } else { -1 })
        .collect();
    let query_pages: Vec<i64> = cum
        .iter()
        .enumerate()
        .map(|(index, &c)| {
            let page = if index <= last_image {
                c - 1
            } else {
                c - 1 - prompt_pages
            };
            page.max(-1)
        })
        .collect();
    let block_pages = image_pages
        .chunks(block_size)
        .map(|block| {
            let pages = block.iter().copied().filter(|&page| page >= 0);
            match (pages.clone().min(), pages.max()) {
                (Some(min), Some(max)) => (min, max),
                _ => (-1, -1),
            }
        })
        .collect();
    let &(tok, page) = coords.last().expect("non-empty prompt");
    Ok(PromptPages {
        coords,
        image_pages,
        query_pages,
        state: PageState {
            page,
            tok,
            prefill_counts,
            block_pages,
            prompt_pages,
            markers: count,
        },
    })
}

impl PageState {
    /// Advance by the decoded input token; returns its RoPE coordinates
    /// and the query's page (`-1` = unconstrained).
    fn advance(&mut self, token: u32, page_token_id: u32) -> ((i64, i64), i64) {
        if token == page_token_id {
            let buckets = self.prefill_counts.len() as i64;
            self.page = (self.page + 1).min(buckets - 1);
            self.tok = self.prefill_counts[self.page.max(0) as usize];
            self.markers += 1;
        } else {
            self.tok += 1;
        }
        let query_page = self.markers - 1 - self.prompt_pages;
        ((self.tok, self.page), query_page.max(-1))
    }

    /// Decode candidate blocks: prefill blocks holding the query page's
    /// image tokens, or every block when the page is unknown or absent.
    fn decode_candidates(&self, query_page: i64, blocks: usize) -> Vec<bool> {
        let same: Vec<bool> = self
            .block_pages
            .iter()
            .map(|&(min, max)| {
                query_page >= 0 && min >= 0 && min <= query_page && query_page <= max
            })
            .collect();
        if query_page < 0 || !same.iter().any(|&s| s) {
            return vec![true; blocks];
        }
        (0..blocks)
            .map(|c| same.get(c).copied().unwrap_or(false))
            .collect()
    }
}

/// `entmax15` over one row, the reference's sort-based threshold in f32.
fn entmax15(values: &[f32]) -> Vec<f32> {
    let max = values.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let x: Vec<f32> = values.iter().map(|&v| (v - max) / 2.0).collect();
    let mut sorted = x.clone();
    sorted.sort_by(|a, b| b.total_cmp(a));
    let (mut sum, mut sum_sq, mut support, mut taus) = (0f32, 0f32, 0usize, Vec::new());
    for (index, &value) in sorted.iter().enumerate() {
        let rho = (index + 1) as f32;
        sum += value;
        sum_sq += value * value;
        let mean = sum / rho;
        let mean_sq = sum_sq / rho;
        let ss = rho * (mean_sq - mean * mean);
        let delta = ((1.0 - ss) / rho).max(0.0);
        let tau = mean - delta.sqrt();
        support += usize::from(tau <= value);
        taus.push(tau);
    }
    let tau = taus[support.max(1) - 1];
    x.iter().map(|&v| (v - tau).max(0.0).powi(2)).collect()
}

/// `_infer_route_select` for one query: returns `(block, bias)` pairs of the
/// blocks it reads remotely. `scores[c]` is `None` for non-remote blocks.
fn route(
    scores: &[Option<f32>],
    candidates: impl Fn(usize) -> bool,
    idx_scale: f32,
    floor: f32,
    params: &RouterParams,
) -> Vec<(usize, f32)> {
    let blocks = scores.len();
    let mut cand: Vec<bool> = (0..blocks)
        .map(|c| scores[c].is_some() && candidates(c))
        .collect();
    if !cand.iter().any(|&c| c) {
        cand = scores.iter().map(Option::is_some).collect();
    }
    let x: Vec<f32> = (0..blocks)
        .map(|c| match scores[c] {
            Some(score) if cand[c] && (score * idx_scale).is_finite() => score * idx_scale,
            _ => ENTMAX_NEG,
        })
        .collect();
    let m = ROUTER_TOP_M.min(blocks).max(1);
    // Descending score, then ascending block: a total order, so selecting
    // the top `m` before sorting them picks what a full sort would.
    let by_score = |a: &usize, b: &usize| x[*b].total_cmp(&x[*a]).then(a.cmp(b));
    let mut order: Vec<usize> = (0..blocks).collect();
    if m < blocks {
        order.select_nth_unstable_by(m - 1, by_score);
        order.truncate(m);
    }
    order.sort_by(by_score);
    let prior = entmax15(&order.iter().map(|&c| x[c]).collect::<Vec<_>>());
    let k = params.budget.min(m).max(1);
    let picked: Vec<(usize, f32)> = order[..k]
        .iter()
        .zip(&prior[..k])
        .filter(|&(&c, _)| cand[c])
        .map(|(&c, &p)| (c, p.max(floor).ln()))
        .collect();
    if picked.is_empty() {
        return picked;
    }
    let mean = picked.iter().map(|&(_, logp)| logp).sum::<f32>() / picked.len() as f32;
    picked
        .into_iter()
        .map(|(c, logp)| (c, (logp - mean) / params.sigma))
        .collect()
}

/// Gemma-style router RMSNorm: f32 normalize, ±1e4 clamps, `(1 + w)`
/// scale, cast back to the input dtype.
fn router_norm(x: &Tensor, scale: &Tensor, eps: f64) -> candle_core::Result<Tensor> {
    let dtype = x.dtype();
    let x = x.to_dtype(DType::F32)?;
    let rms = x.sqr()?.mean_keepdim(D::Minus1)?.affine(1.0, eps)?.sqrt()?;
    x.broadcast_div(&rms)?
        .clamp(-ACT_CLAMP, ACT_CLAMP)?
        .broadcast_mul(scale)?
        .clamp(-ACT_CLAMP, ACT_CLAMP)?
        .to_dtype(dtype)
}

/// `x * cos + rotate_half(x) * sin` with `(len, dim / 2)` tables over
/// `(..., len, dim)` inputs.
fn apply_rope(x: &Tensor, cos: &Tensor, sin: &Tensor) -> candle_core::Result<Tensor> {
    candle_nn::rotary_emb::rope(&x.contiguous()?, cos, sin)
}

struct Router {
    q_proj: Linear,
    k_proj: Linear,
    q_scale: Tensor,
    k_scale: Tensor,
    head_cls: Tensor,
    eps: f64,
    idx_scale: f32,
    floor: f32,
}

impl Router {
    fn load(
        config: &UnlimitedOcrConfig,
        layer: usize,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let text = &config.text;
        let sparse = &config.sparse_attention_config;
        let dim = sparse.sparse_index_dim;
        let scale = |name: &str| -> candle_core::Result<Tensor> {
            vb.get(dim, &format!("{name}.weight"))?
                .to_dtype(DType::F32)?
                .affine(1.0, 1.0)
        };
        Ok(Self {
            q_proj: linear_no_bias(text.hidden_size, dim, vb.pp("q_proj"))?,
            k_proj: linear_no_bias(text.hidden_size, dim, vb.pp("k_proj"))?,
            q_scale: scale("q_norm")?,
            k_scale: scale("k_norm")?,
            head_cls: vb.get(dim, "head_cls")?.to_dtype(DType::F32)?,
            eps: text.rms_norm_eps,
            idx_scale: sparse.dash_scale_per_layer[layer] / (dim as f32).sqrt(),
            floor: sparse.dash_floor_per_layer[layer],
        })
    }

    /// Router `(q, k)`, `(len, dim)` f32, after norm and RoPE in the model dtype.
    fn qk(
        &self,
        hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let project = |proj: &Linear, scale: &Tensor| -> candle_core::Result<Tensor> {
            let x = router_norm(&proj.forward(hidden)?, scale, self.eps)?;
            // (1, len, dim) -> (1, 1, len, dim) for the rotary kernel.
            apply_rope(&x.unsqueeze(1)?, cos, sin)?
                .squeeze(1)?
                .squeeze(0)?
                .to_dtype(DType::F32)
        };
        Ok((
            project(&self.q_proj, &self.q_scale)?,
            project(&self.k_proj, &self.k_scale)?,
        ))
    }

    /// `head_cls` summary key of every block, `(blocks, dim)` f32.
    fn block_keys(&self, keys: &Tensor, block_size: usize) -> candle_core::Result<Tensor> {
        let (len, dim) = keys.dims2()?;
        let blocks = len.div_ceil(block_size);
        let padded = blocks * block_size;
        let keys = if padded > len {
            Tensor::cat(
                &[
                    keys,
                    &Tensor::zeros((padded - len, dim), DType::F32, keys.device())?,
                ],
                0,
            )?
        } else {
            keys.clone()
        };
        let keys = keys.reshape((blocks, block_size, dim))?;
        let scale = 1.0 / (dim as f64).sqrt();
        let logits = keys
            .broadcast_mul(&self.head_cls.reshape((1, 1, dim))?)?
            .sum(D::Minus1)?
            .affine(scale, 0.0)?
            .clamp(-ACT_CLAMP, ACT_CLAMP)?;
        let real: Vec<f32> = (0..padded).map(|i| f32::from(u8::from(i < len))).collect();
        let real = Tensor::from_vec(real, (blocks, block_size), keys.device())?;
        // Padded slots never win the row max: they carry the f32 minimum.
        let masked = logits.broadcast_mul(&real)?.broadcast_add(
            &real
                .affine(-f64::from(f32::MAX), f64::from(f32::MAX))?
                .neg()?,
        )?;
        let row_max = masked.max_keepdim(D::Minus1)?;
        let weights = logits.broadcast_sub(&row_max)?.exp()?.mul(&real)?;
        let weights =
            weights.broadcast_div(&weights.sum_keepdim(D::Minus1)?.clamp(1e-6, f64::MAX)?)?;
        weights.unsqueeze(D::Minus1)?.broadcast_mul(&keys)?.sum(1)
    }

    /// Scores `(queries, blocks)` of router queries against block keys.
    fn scores(&self, queries: &Tensor, block_keys: &Tensor) -> candle_core::Result<Vec<Vec<f32>>> {
        let dim = queries.dim(D::Minus1)?;
        queries
            .matmul(&block_keys.t()?)?
            .affine(1.0 / (dim as f64).sqrt(), 0.0)?
            .clamp(-ACT_CLAMP, ACT_CLAMP)?
            .to_vec2::<f32>()
    }
}

/// Decode state of one layer: the K/V cache, the summary keys of complete
/// blocks, and the router keys of the trailing partial block, so a step
/// only pools the block it extends.
struct LayerCache {
    kv: TrimmableKvCache,
    block_keys: Option<Tensor>,
    pending_router_keys: Option<Tensor>,
}

impl Default for LayerCache {
    fn default() -> Self {
        Self {
            kv: TrimmableKvCache::new(2, 0),
            block_keys: None,
            pending_router_keys: None,
        }
    }
}

struct Attention {
    q_proj: Linear,
    k_proj: Linear,
    v_proj: Linear,
    o_proj: Linear,
    router: Router,
    heads: usize,
    head_dim: usize,
    cache: RefCell<LayerCache>,
}

impl Attention {
    fn load(config: &UnlimitedOcrConfig, layer: usize, vb: VarBuilder) -> Result<Self, Error> {
        let text = &config.text;
        let heads = text.num_attention_heads;
        let head_dim = text.head_dim()?;
        let width = heads * head_dim;
        let proj = |name: &str, i, o| linear_no_bias(i, o, vb.pp(name)).map_err(infer);
        Ok(Self {
            q_proj: proj("q_proj", text.hidden_size, width)?,
            k_proj: proj("k_proj", text.hidden_size, width)?,
            v_proj: proj("v_proj", text.hidden_size, width)?,
            o_proj: proj("o_proj", width, text.hidden_size)?,
            router: Router::load(config, layer, vb.pp("dash_router")).map_err(infer)?,
            heads,
            head_dim,
            cache: RefCell::new(LayerCache::default()),
        })
    }

    /// `(1, len, hidden)` -> `(1, heads, len, head_dim)` with RoPE.
    fn heads(
        &self,
        proj: &Linear,
        hidden: &Tensor,
        rope: Option<(&Tensor, &Tensor)>,
    ) -> candle_core::Result<Tensor> {
        let (_, len, _) = hidden.dims3()?;
        let x = proj
            .forward(hidden)?
            .reshape((1, len, self.heads, self.head_dim))?
            .transpose(1, 2)?;
        match rope {
            Some((cos, sin)) => apply_rope(&x, cos, sin),
            None => x.contiguous(),
        }
    }

    /// Prefill: dense attention with the routed block-bias mask.
    fn prefill(
        &self,
        hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        pages: &PromptPages,
        params: &RouterParams,
    ) -> Result<Tensor, Error> {
        let len = hidden.dim(1).map_err(infer)?;
        let device = hidden.device();
        let q = self
            .heads(&self.q_proj, hidden, Some((cos, sin)))
            .map_err(infer)?;
        let k = self
            .heads(&self.k_proj, hidden, Some((cos, sin)))
            .map_err(infer)?;
        let v = self.heads(&self.v_proj, hidden, None).map_err(infer)?;
        let (router_q, router_k) = self.router.qk(hidden, cos, sin).map_err(infer)?;
        let block_size = params.block_size;
        let blocks = len.div_ceil(block_size);
        let block_keys = self
            .router
            .block_keys(&router_k, block_size)
            .map_err(infer)?;
        let scores = self.router.scores(&router_q, &block_keys).map_err(infer)?;

        // Blocks holding each page's image tokens, for page-scoped queries.
        let pages_per_block: Vec<Vec<i64>> = pages
            .image_pages
            .chunks(block_size)
            .map(|block| {
                let mut pages: Vec<i64> = block.iter().copied().filter(|&p| p >= 0).collect();
                pages.dedup();
                pages
            })
            .collect();
        let mut bias = vec![f32::NEG_INFINITY; len * blocks];
        for (query, row) in scores.iter().enumerate() {
            let remote_limit = (query / block_size) as i64 - params.local_blocks as i64;
            let row: Vec<Option<f32>> = row
                .iter()
                .enumerate()
                .map(|(c, &s)| (c as i64 <= remote_limit).then_some(s))
                .collect();
            let page = pages.query_pages[query];
            let any_same = page >= 0 && pages_per_block.iter().any(|p| p.contains(&page));
            let candidate = |c: usize| !any_same || pages_per_block[c].contains(&page);
            for (c, b) in route(
                &row,
                candidate,
                self.router.idx_scale,
                self.router.floor,
                params,
            ) {
                bias[query * blocks + c] = b;
            }
        }
        // Per query chunk, (rows, blocks) -> (rows, keys): each block bias
        // spans its 32 keys; the causal local window reads unbiased and wins
        // on overlap. Chunking keeps the mask at `rows * len`.
        let local_window = (params.local_blocks * block_size) as f64;
        let cols = Tensor::arange(0i64, len as i64, device)
            .and_then(|c| c.reshape((1, len)))
            .map_err(infer)?;
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let mut outputs = Vec::with_capacity(len.div_ceil(PREFILL_QUERY_CHUNK));
        let kt = k
            .transpose(2, 3)
            .map_err(infer)?
            .contiguous()
            .map_err(infer)?;
        for start in (0..len).step_by(PREFILL_QUERY_CHUNK) {
            let rows = PREFILL_QUERY_CHUNK.min(len - start);
            let mask = (|| {
                let bias = Tensor::from_slice(
                    &bias[start * blocks..(start + rows) * blocks],
                    (rows, blocks, 1),
                    device,
                )?
                .broadcast_as((rows, blocks, block_size))?
                .reshape((rows, blocks * block_size))?
                .narrow(1, 0, len)?;
                let queries = Tensor::arange(start as i64, (start + rows) as i64, device)?
                    .reshape((rows, 1))?;
                let causal = cols.broadcast_le(&queries)?;
                let window = cols.broadcast_gt(&queries.affine(1.0, -local_window)?)?;
                causal
                    .mul(&window)?
                    .where_cond(&Tensor::zeros((rows, len), DType::F32, device)?, &bias)
            })()
            .map_err(infer)?;
            let logits = q
                .narrow(2, start, rows)
                .and_then(|q| q.matmul(&kt))
                .and_then(|l| l.to_dtype(DType::F32))
                .and_then(|l| l.affine(scale, 0.0))
                .and_then(|l| l.broadcast_add(&mask.unsqueeze(0)?.unsqueeze(0)?))
                .map_err(infer)?;
            let probs = candle_nn::ops::softmax_last_dim(&logits)
                .and_then(|p| p.to_dtype(v.dtype()))
                .map_err(infer)?;
            outputs.push(probs.matmul(&v).map_err(infer)?);
        }
        let context = Tensor::cat(&outputs, 2).map_err(infer)?;
        let mut cache = LayerCache::default();
        cache.kv.append(&k, &v).map_err(infer)?;
        let complete = len / block_size;
        let partial = len - complete * block_size;
        cache.block_keys = (complete > 0)
            .then(|| block_keys.narrow(0, 0, complete))
            .transpose()
            .map_err(infer)?;
        cache.pending_router_keys = (partial > 0)
            .then(|| router_k.narrow(0, complete * block_size, partial))
            .transpose()
            .map_err(infer)?;
        *self.cache.borrow_mut() = cache;
        self.output(&context, len)
    }

    /// One decode step at absolute `position` over the whole cache, in f32.
    #[allow(clippy::too_many_arguments)]
    fn decode(
        &self,
        hidden: &Tensor,
        cos: &Tensor,
        sin: &Tensor,
        position: usize,
        state: &PageState,
        query_page: i64,
        params: &RouterParams,
    ) -> Result<Tensor, Error> {
        let device = hidden.device();
        let q = self
            .heads(&self.q_proj, hidden, Some((cos, sin)))
            .map_err(infer)?;
        let k = self
            .heads(&self.k_proj, hidden, Some((cos, sin)))
            .map_err(infer)?;
        let v = self.heads(&self.v_proj, hidden, None).map_err(infer)?;
        let (router_q, router_k) = self.router.qk(hidden, cos, sin).map_err(infer)?;
        let mut cache = self.cache.borrow_mut();
        cache.kv.append(&k, &v).map_err(infer)?;
        // Gather from the contiguous backing storage: every index lies below
        // the logical length, and `index_select` rejects the narrowed view.
        let (keys, values) = cache.kv.storage().expect("appended above");
        let block_size = params.block_size;
        let pending = match &cache.pending_router_keys {
            Some(pending) => Tensor::cat(&[pending, &router_k], 0).map_err(infer)?,
            None => router_k,
        };
        let pending_key = self
            .router
            .block_keys(&pending, block_size)
            .map_err(infer)?;
        let all_block_keys = match &cache.block_keys {
            Some(done) => Tensor::cat(&[done, &pending_key], 0).map_err(infer)?,
            None => pending_key,
        };
        let scores = self
            .router
            .scores(&router_q, &all_block_keys)
            .map_err(infer)?;
        if pending.dim(0).map_err(infer)? == block_size {
            cache.block_keys = Some(all_block_keys);
            cache.pending_router_keys = None;
        } else {
            cache.pending_router_keys = Some(pending);
        }
        drop(cache);

        let len = position + 1;
        let blocks = len.div_ceil(block_size);
        let remote_limit = (position / block_size) as i64 - params.local_blocks as i64;
        let row: Vec<Option<f32>> = scores[0]
            .iter()
            .enumerate()
            .map(|(c, &s)| (c as i64 <= remote_limit).then_some(s))
            .collect();
        let candidates = state.decode_candidates(query_page, blocks);
        let picked = route(
            &row,
            |c| candidates[c],
            self.router.idx_scale,
            self.router.floor,
            params,
        );
        // Gather the routed blocks (biased) and the local window (unbiased,
        // winning where they overlap): the keys a dense masked softmax
        // would read, at a cost that does not grow with the context.
        let local_start = len.saturating_sub(params.local_blocks * block_size);
        let mut index = Vec::with_capacity(picked.len() * block_size + len - local_start);
        let mut bias = Vec::with_capacity(index.capacity());
        for (c, b) in picked {
            let end = ((c + 1) * block_size).min(local_start);
            for key in c * block_size..end {
                index.push(key as u32);
                bias.push(b);
            }
        }
        index.extend(local_start as u32..len as u32);
        bias.resize(index.len(), 0.0);
        let read = index.len();
        let index = Tensor::from_vec(index, read, device).map_err(infer)?;
        let bias = Tensor::from_vec(bias, (1, 1, 1, read), device).map_err(infer)?;
        let gather = |cache: &Tensor| {
            cache
                .index_select(&index, 2)
                .and_then(|x| x.to_dtype(DType::F32))
        };
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let logits = q
            .to_dtype(DType::F32)
            .and_then(|q| q.matmul(&gather(&keys)?.transpose(2, 3)?))
            .and_then(|l| l.affine(scale, 0.0)?.broadcast_add(&bias))
            .map_err(infer)?;
        let context = candle_nn::ops::softmax_last_dim(&logits)
            .and_then(|p| p.matmul(&gather(&values)?))
            .and_then(|c| c.to_dtype(hidden.dtype()))
            .map_err(infer)?;
        self.output(&context, 1)
    }

    fn output(&self, context: &Tensor, len: usize) -> Result<Tensor, Error> {
        context
            .transpose(1, 2)
            .and_then(|c| c.reshape((1, len, self.heads * self.head_dim)))
            .and_then(|c| self.o_proj.forward(&c))
            .map_err(infer)
    }
}

struct Layer {
    input_layernorm: RmsNorm,
    attention: Attention,
    post_attention_layernorm: RmsNorm,
    mlp: FeedForward,
}

/// The DASH decoder with its embeddings and LM head.
pub(super) struct DashDecoder {
    embed_tokens: Embedding,
    layers: Vec<Layer>,
    norm: RmsNorm,
    lm_head: Linear,
    rope: PageRope,
    params: RouterParams,
    image_token_id: u32,
    page_token_id: u32,
    dtype: DType,
}

impl DashDecoder {
    /// `vb` is the checkpoint root.
    pub fn load(config: &UnlimitedOcrConfig, vb: VarBuilder) -> Result<Self, Error> {
        let text = &config.text;
        let sparse = &config.sparse_attention_config;
        let model = vb.pp("model");
        let norm =
            |vb: VarBuilder| rms_norm(text.hidden_size, text.rms_norm_eps, vb).map_err(infer);
        let layers = (0..text.num_hidden_layers)
            .map(|index| {
                let vb = model.pp(format!("layers.{index}"));
                Ok(Layer {
                    input_layernorm: norm(vb.pp("input_layernorm"))?,
                    attention: Attention::load(config, index, vb.pp("self_attn"))?,
                    post_attention_layernorm: norm(vb.pp("post_attention_layernorm"))?,
                    mlp: FeedForward::load(text, index, vb)?,
                })
            })
            .collect::<Result<Vec<_>, Error>>()?;
        Ok(Self {
            embed_tokens: embedding(text.vocab_size, text.hidden_size, model.pp("embed_tokens"))
                .map_err(infer)?,
            layers,
            norm: norm(model.pp("norm"))?,
            lm_head: linear_no_bias(text.hidden_size, text.vocab_size, vb.pp("lm_head"))
                .map_err(infer)?,
            rope: PageRope::new(config, text.head_dim()?),
            params: RouterParams {
                block_size: sparse.sparse_block_size,
                local_blocks: sparse.sparse_local_block,
                budget: sparse.dash_topk_blocks,
                sigma: sparse.dash_sigma,
            },
            image_token_id: sparse.image_token_id,
            page_token_id: sparse.page_token_id,
            dtype: vb.dtype(),
        })
    }

    pub fn embed(&self, ids: &Tensor) -> Result<Tensor, Error> {
        self.embed_tokens.forward(ids).map_err(infer)
    }

    pub fn clear_cache(&self) {
        for layer in &self.layers {
            *layer.attention.cache.borrow_mut() = LayerCache::default();
        }
    }

    fn logits(&self, hidden: &Tensor) -> Result<Tensor, Error> {
        let len = hidden.dim(1).map_err(infer)?;
        hidden
            .narrow(1, len - 1, 1)
            .and_then(|h| self.norm.forward(&h))
            .and_then(|h| self.lm_head.forward(&h))
            .and_then(|l| l.flatten_all())
            .and_then(|l| l.to_dtype(DType::F32))
            .map_err(infer)
    }

    fn layer_forward(
        &self,
        layer: &Layer,
        hidden: &Tensor,
        attend: impl FnOnce(&Attention, &Tensor) -> Result<Tensor, Error>,
    ) -> Result<Tensor, Error> {
        let normed = layer.input_layernorm.forward(hidden).map_err(infer)?;
        let hidden = (hidden + attend(&layer.attention, &normed)?).map_err(infer)?;
        let normed = layer
            .post_attention_layernorm
            .forward(&hidden)
            .map_err(infer)?;
        (&hidden + layer.mlp.forward(&normed)?).map_err(infer)
    }

    /// Prefill the prompt; returns the last-position logits and the page
    /// state that later decode steps advance.
    pub fn prefill(&self, ids: &[u32], embeds: &Tensor) -> Result<(Tensor, PageState), Error> {
        let pages = prompt_pages(
            ids,
            self.image_token_id,
            self.page_token_id,
            self.params.block_size,
        )?;
        let (cos, sin) = self
            .rope
            .tables(&pages.coords, embeds.device(), self.dtype)?;
        let mut hidden = embeds.clone();
        for layer in &self.layers {
            hidden = self.layer_forward(layer, &hidden, |attention, normed| {
                attention.prefill(normed, &cos, &sin, &pages, &self.params)
            })?;
        }
        Ok((self.logits(&hidden)?, pages.state))
    }

    /// Decode `token` at absolute `position`; returns the next logits.
    pub fn decode(
        &self,
        token: u32,
        position: usize,
        state: &mut PageState,
    ) -> Result<Tensor, Error> {
        let device = self.lm_head.weight().device();
        let (coords, query_page) = state.advance(token, self.page_token_id);
        let (cos, sin) = self.rope.tables(&[coords], device, self.dtype)?;
        let mut hidden = Tensor::new(&[[token]], device)
            .map_err(infer)
            .and_then(|ids| self.embed(&ids))?;
        for layer in &self.layers {
            hidden = self.layer_forward(layer, &hidden, |attention, normed| {
                attention.decode(
                    normed,
                    &cos,
                    &sin,
                    position,
                    state,
                    query_page,
                    &self.params,
                )
            })?;
        }
        self.logits(&hidden)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn entmax15_matches_the_closed_form_threshold() {
        // Two equal scores share the mass; a far-off score gets none.
        let p = entmax15(&[1.0, 1.0, -10.0]);
        assert!(
            (p[0] - 0.5).abs() < 1e-6 && (p[1] - 0.5).abs() < 1e-6 && p[2] == 0.0,
            "{p:?}"
        );
        // Scores (2, 0) halve to (1, 0): tau = 0, so the prior is (1, 0).
        assert_eq!(entmax15(&[2.0, 0.0]), vec![1.0, 0.0]);
    }

    #[test]
    fn prompt_pages_follow_the_reference_numbering() {
        // [BOS, <PAGE>, img, img, <PAGE>, img, txt, txt] with image 7, page 9.
        let ids = [0, 9, 7, 7, 9, 7, 5, 5];
        let pages = prompt_pages(&ids, 7, 9, 2).unwrap();
        assert_eq!(
            pages.coords,
            vec![
                (0, 0),
                (0, 1),
                (1, 1),
                (2, 1),
                (0, 2),
                (1, 2),
                (1, 0),
                (2, 0)
            ]
        );
        assert_eq!(pages.image_pages, vec![-1, -1, 0, 0, -1, 1, -1, -1]);
        assert_eq!(pages.query_pages, vec![-1, 0, 0, 0, 1, 1, -1, -1]);
        assert_eq!(
            pages.state.block_pages,
            vec![(-1, -1), (0, 0), (1, 1), (-1, -1)]
        );
        assert_eq!(pages.state.prefill_counts, vec![1, 3, 2]);
        // Decoding the first response <PAGE> moves to page 1 after its prefill run.
        let mut state = pages.state;
        assert_eq!(state.advance(9, 9), ((3, 1), 0));
        assert_eq!(state.advance(5, 9), ((4, 1), 0));
        assert_eq!(
            state.decode_candidates(0, 5),
            vec![false, true, false, false, false]
        );
    }
}
