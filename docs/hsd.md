# Hierarchical Speculative Decoding (HSD)

HSD is an optional CUDA acceleration path for document VLM decoding. It leaves the target model unchanged. A cheaper document pipeline — the paper uses PP-StructureV3 (layout analysis + element recognition) — prepares draft text once per page. The VLM then verifies those drafts with tree-batched speculative decoding and commits only accepted tokens.

Reference: Liao et al., *"HSD: Training-Free Acceleration for Document Parsing Vision-Language Model with Hierarchical Speculative Decoding"* (arXiv:2602.12957). Section references below cite that paper.

## When to use it

HSD helps when the draft is close to what the VLM would generate on its own. That is common on text-heavy pages, tables with regular structure, and repeated document boilerplate. A good draft lets one verify pass accept several tokens.

It is not a general CPU speedup. The implementation is intended for CUDA, where a wider tree-attention verify pass is cheap compared with repeated single-token decoding. On CPU or Metal, the verify work is effectively serialized and the benefit usually disappears.

The paper defines the acceptance threshold on the open interval $\tau \in (0, 1)$ (§3.2). Lower values accept more near-tie tokens, which can improve speed but may change the output. This implementation also accepts `tau = 1.0` as a degenerate boundary: at $\tau = 1.0$ the acceptance test collapses to "child equals the unrestricted argmax", so HSD follows the target model's greedy path. That extension is not part of the paper.

## Document flow

The document-level path has two stages (§3.1):

- **Stage 1: region-level local verification.** For each region $r_i \in \mathcal{R}$, the target VLM verifies the region draft set $\tilde{\mathcal{Y}}^{(i)}$ on the cropped image $z_i = x|_{r_i}$:
   $$\hat{y}^{(i)} = \mathrm{SpecDecode}(p_\theta, z_i, \tilde{\mathcal{Y}}^{(i)}).$$
- **Stage 2: page-level global verification.** Stage 1 outputs are aggregated into an unordered page-level draft set
   $$\tilde{\mathcal{Y}}^{\mathrm{pg}} = \{\hat{y}^{(i)} \mid r_i \in \mathcal{R}\},$$
   which the target VLM then verifies in a single full-page pass: $\hat{y}^{\mathrm{pg}} = \mathrm{SpecDecode}(p_\theta, x, \tilde{\mathcal{Y}}^{\mathrm{pg}})$. Because the matcher scans each $\hat{y}^{(i)}$ independently, draft order has no semantic effect; the target model resolves the final reading order during verification.

Backends that implement the full document path can turn either stage off through `HsdConfig`. PaddleOCR-VL is not evaluated in the paper; in this implementation it stays element-oriented by model design and uses only the region path.

## One SpecDecode step

For the accepted prefix $\hat{y}_{1:t}$ and a draft set $\tilde{\mathcal{Y}}$ (§3.2):

1. **Draft-target matching.** Let the reference window be the most recent $n$ accepted tokens, $w = \hat{y}_{t-n+1:t}$. For each draft $\tilde{y} \in \tilde{\mathcal{Y}}$, record every start index $j$ with $\tilde{y}_{j:j+n-1} = w$. Collect the suffixes that follow each match:
   $$\mathcal{C} = \big\{\, \tilde{y}_{j+n:|\tilde{y}|} \,\big|\, \tilde{y} \in \tilde{\mathcal{Y}},\; j \in \mathcal{J}(\tilde{y}),\; j + n \le |\tilde{y}|\,\big\}.$$
2. **Prefix-tree batching.** Merge $\mathcal{C}$ into a prefix tree $\mathcal{T}$ whose root represents the empty prefix and whose every root-to-leaf path is one element of $\mathcal{C}$. For a node $v$, $\pi(v)$ is the token sequence on the path root → $v$, and $\mathrm{Next}(v) = \{c_{|\pi(v)|+1} \mid c \in \mathcal{C},\; c_{1:|\pi(v)|} = \pi(v)\}$ is the set of distinct next tokens shared by candidates that pass through $v$.
3. **One tree-batched forward.** Linearize $\mathcal{T}$ into a packed sequence and run the target VLM under a tree-ancestry attention mask: a token at node $v$ attends only to $\hat{y}_{1:t}$ and to the tokens on $v$'s ancestor path. This produces $p_\theta(\cdot \mid z, \hat{y}_{1:t} \oplus \pi(v))$ at every node in one pass.
4. **Greedy traversal with the $\tau$-test.** Start at the root $s$. At each step, select the best child token in the tree's local candidate set and compare it with the unrestricted argmax over the full vocabulary $\mathcal{V}$:
   $$u^\star = \arg\max_{u \in \mathrm{Next}(s)} p_\theta(u \mid z, \hat{y}_{1:t} \oplus \pi(s)), \qquad \hat{u} = \arg\max_{u \in \mathcal{V}} p_\theta(u \mid z, \hat{y}_{1:t} \oplus \pi(s)).$$
   Accept $u^\star$ and descend to its child node iff
   $$\log p_\theta(u^\star \mid z, \hat{y}_{1:t} \oplus \pi(s)) - \log p_\theta(\hat{u} \mid z, \hat{y}_{1:t} \oplus \pi(s)) \ge \log \tau.$$
   Stop when the test fails, when $\mathrm{Next}(s) = \emptyset$, or when $s$ is a leaf.
5. **Bonus target token.** At the terminal node $s$, append the unrestricted argmax $\hat{u}$ to extend the accepted sequence by one extra target token:
   $$\hat{y}_{1:t_\mathrm{new}} = \hat{y}_{1:t} \oplus \pi(s) \oplus \hat{u}.$$
6. **Commit KV state.** Gather the KV cache so it keeps only the accepted prefix and the path through $s$, then continue decoding from $\hat{u}$.

If $\mathcal{C} = \emptyset$ (no draft matches the current window), $\mathcal{T}$ contains only the root, $\mathrm{Next}(\mathrm{root}) = \emptyset$, the traversal stops immediately, and step 5 falls back to a single greedy token — the paper's algorithm with no special case.

## Correctness at `tau = 1.0`

The paper proves training-free, near-lossless acceleration over its stated domain $\tau \in (0, 1)$. This implementation also exposes $\tau = 1.0$ as a degenerate boundary: with $\log \tau = 0$, the acceptance test in step 4 reduces to $u^\star = \hat{u}$, so a child token is accepted only when it coincides with the unrestricted argmax. The committed sequence is then independent of the drafter and identical to the target model's greedy decode.

By default this is enforced via a strict replay path (`strict_at_tau_one = true`, see Configuration). That replay path is an OAR-side correctness oracle, not part of the paper. Set `strict_at_tau_one = false` to keep $\tau = 1.0$ on the same tree-batched verify path the paper describes.

The demo harness runs this oracle check and compares HSD output with baseline output byte-for-byte.

## Reading AAL

The main debug metric is **Average Acceptance Length (AAL)** (§4.2). At verification step $k$, let $\alpha_k$ be the number of consecutive draft tokens accepted before the first mismatch ($\alpha_k = 0$ on a full rejection). Over $N$ verification steps:

$$\mathrm{AAL} = \frac{1}{N} \sum_{k=1}^{N} \alpha_k.$$

The bonus target token appended at step 5 is not counted. Larger AAL means more decoding steps are saved per verify pass; the realized end-to-end speedup also depends on per-step verify overhead and parallel efficiency.

For reference, the paper reports overall AAL on OmniDocBench v1.5 (Tab. 1) in the **2.5 to 4.6** range across the evaluated backbones (HunyuanOCR 4.55, dots.ocr 3.98, Qwen3-VL-8B 3.98, Qwen3-VL-2B 3.33, Qwen2.5-VL-7B 3.56, Qwen2.5-VL-3B 2.52). The ranges below are operational rules of thumb observed on this implementation, not paper numbers; use AAL as a draft-quality signal, not as a correctness metric:

- `AAL` around `1`: the draft is doing little work. Check tokenization, window length, reading order, and whether the drafter output resembles the target output.
- `AAL` from `3` to `6`: a normal range for many text-heavy pages with OCR drafts.
- `AAL` from `8` to `15`: strong alignment, often from tables, lists, or repeated layout.
- `AAL > 20`: usually a long exact span. Inspect output quality as well as speed.

## Configuration

`HsdConfig` controls the two-stage document path. Its `dsv` field contains the per-step speculative decoding knobs. The first two fields (`window_len`, `tau`) follow the paper's defaults (§4.3); the rest are OAR-side engineering knobs not present in the paper.

| Field | Default | Source | Notes |
|-------|---------|--------|-------|
| `window_len` | `3` | paper §4.3 | Reference-window length $n$. Longer windows reduce false matches on repetitive text but also reduce draft hits. |
| `tau` | `0.75` | paper §4.3 | Acceptance threshold. Paper domain is $\tau \in (0, 1)$; lower accepts more borderline tokens. `1.0` is a boundary extension that recovers greedy decoding. |
| `max_candidates_per_step` | `32` | OAR addition | Bounds the number of candidate suffixes used to build each tree. The paper's ablations use uncapped trees. |
| `max_suffix_len` | `256` | OAR addition | Bounds candidate depth so long drafts do not create oversized trees. |
| `cold_start_full_draft` | `true` | OAR addition | Seeds the first step from draft prefixes before any accepted window exists. The paper's matcher has no cold-start fallback. |
| `strict_at_tau_one` | `true` | OAR addition | When `true` and $\tau \ge 1.0$, route through a strict replay oracle. Set `false` to keep $\tau = 1.0$ on the paper's tree-batched verify path. |

The candidate caps are guardrails for long tables, formulas, and repeated boilerplate. To reproduce a paper-faithful matcher, set both caps to `usize::MAX`, `cold_start_full_draft = false`, and `strict_at_tau_one = false`.

## Running it

Build the VLM crate with the `hsd` feature. It enables CUDA transitively:

```bash
cargo run -p oar-ocr-vl --release --features hsd,download-binaries \
    --example hsd_demo -- \
    --backend hunyuanocr \
    --model-dir models/HunyuanOCR \
    --device cuda \
    --image document.jpg
```

The supported backbones expose `generate_hsd*` methods next to their baseline `generate` methods: `PaddleOcrVl`, `HunyuanOcr`, `GlmOcr`, and `MinerU`.
