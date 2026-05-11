//! Shared data types for Hierarchical Speculative Decoding.

use std::time::Duration;

/// A single draft sequence: the tokenized output of the pipeline drafter for one
/// region (Stage 1) or the aggregated page (Stage 2).
///
/// Tokens are encoded with the **target VLM's** tokenizer so the verification path
/// can match prefixes at token granularity without re-tokenization.
#[derive(Debug, Clone)]
pub struct Draft {
    /// Token id sequence as produced by the target VLM's tokenizer.
    pub tokens: Vec<u32>,
}

impl Draft {
    pub fn new(tokens: Vec<u32>) -> Self {
        Self { tokens }
    }

    pub fn len(&self) -> usize {
        self.tokens.len()
    }

    pub fn is_empty(&self) -> bool {
        self.tokens.is_empty()
    }
}

/// Stage-1 input: a draft plus the region geometry needed to crop the page image.
///
/// The crop itself is materialised by the caller (we keep this struct image-agnostic
/// so the algorithm layer stays free of `image` / `candle` dependencies).
#[derive(Debug, Clone)]
pub struct RegionDraft {
    /// Region bounding box `[x0, y0, x1, y1]` in original image pixel coordinates.
    pub bbox: [f32; 4],
    /// Tokenized draft for this region.
    pub draft: Draft,
    /// Optional reading-order index assigned by the drafter.
    pub reading_order: Option<usize>,
    /// Drafter-side category label (paragraph / table / formula / figure / ...).
    pub kind: RegionKind,
}

/// Coarse semantic kind reported by the drafter. Mirrors the layout categories used
/// by `oar_ocr_core::domain::structure::LayoutElementType` but stays decoupled so the
/// HSD module compiles without pulling layout dependencies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RegionKind {
    Text,
    Title,
    List,
    Table,
    Formula,
    Figure,
    Header,
    Footer,
    Other,
}

impl Default for RegionKind {
    fn default() -> Self {
        RegionKind::Other
    }
}

/// Stage-2 input: an unordered collection of token sequences treated as page-level
/// drafts (each entry is one of the Stage-1 verified outputs).
#[derive(Debug, Clone, Default)]
pub struct PageDraft {
    pub drafts: Vec<Draft>,
}

impl PageDraft {
    pub fn new(drafts: Vec<Draft>) -> Self {
        Self { drafts }
    }

    pub fn push(&mut self, d: Draft) {
        self.drafts.push(d);
    }

    pub fn len(&self) -> usize {
        self.drafts.len()
    }

    pub fn is_empty(&self) -> bool {
        self.drafts.is_empty()
    }
}

/// Configuration for Decoupled Speculative Verification.
///
/// Defaults follow the paper's experimental setup (Section 4.3): n=3, τ=0.75.
#[derive(Debug, Clone, Copy)]
pub struct DsvConfig {
    /// Reference-window length `n`. Token sequences are matched by sliding a window
    /// of this size over each draft.
    pub window_len: usize,
    /// Acceptance threshold `τ ∈ (0, 1]`. At each tree node, the best child
    /// token is accepted iff its log-probability is within `log τ` of the
    /// model's unrestricted argmax token.
    pub tau: f32,
    /// Cap on the number of candidate paths kept per verification step. Protects
    /// against pathological prefix trees on very long drafts.
    pub max_candidates_per_step: usize,
    /// Cap on the depth of any candidate suffix considered. Drafts longer than this
    /// are truncated in the suffix extraction step.
    pub max_suffix_len: usize,
}

impl Default for DsvConfig {
    fn default() -> Self {
        Self {
            window_len: 3,
            tau: 0.75,
            max_candidates_per_step: 32,
            max_suffix_len: 64,
        }
    }
}

/// Top-level HSD configuration covering both stages.
#[derive(Debug, Clone)]
pub struct HsdConfig {
    pub dsv: DsvConfig,
    /// If false, Stage 1 is skipped and only the page-level pass runs.
    ///
    /// Currently only HunyuanOCR's full HSD path honors the stage gates; the
    /// GLM-OCR, MinerU, and PaddleOCR-VL helpers run a single page-level pass.
    pub enable_stage1: bool,
    /// If false, Stage 2 is skipped in backends that implement the full
    /// two-stage path.
    pub enable_stage2: bool,
    /// Hard cap on `max_new_tokens` for the page-level pass.
    pub max_page_tokens: usize,
    /// Hard cap on `max_new_tokens` for any single region pass.
    pub max_region_tokens: usize,
}

impl Default for HsdConfig {
    fn default() -> Self {
        Self {
            dsv: DsvConfig::default(),
            enable_stage1: true,
            enable_stage2: true,
            max_page_tokens: 16384,
            max_region_tokens: 4096,
        }
    }
}

/// Per-step acceptance bookkeeping. Used to compute Average Acceptance Length (AAL)
/// in the spirit of Leviathan et al. 2023: the number of *draft* tokens accepted at
/// each verification step (excludes the bonus token sampled by the target).
#[derive(Debug, Clone, Default)]
pub struct AcceptStats {
    /// Per-step accepted draft-token counts (`α_k` in the paper's notation).
    pub per_step_accepted: Vec<u32>,
    /// Number of verification steps (`N`).
    pub num_steps: u32,
    /// Number of fallback steps where the prefix tree was empty / fully rejected.
    pub num_fallbacks: u32,
}

impl AcceptStats {
    /// Average Acceptance Length over recorded steps.
    pub fn aal(&self) -> f32 {
        if self.num_steps == 0 {
            0.0
        } else {
            let sum: u32 = self.per_step_accepted.iter().sum();
            sum as f32 / self.num_steps as f32
        }
    }

    pub fn record(&mut self, accepted: u32) {
        self.per_step_accepted.push(accepted);
        self.num_steps += 1;
    }

    pub fn record_fallback(&mut self) {
        self.per_step_accepted.push(0);
        self.num_steps += 1;
        self.num_fallbacks += 1;
    }
}

/// Internal timing/counter breakdown for the shared speculative decoder.
#[derive(Debug, Clone, Default)]
pub struct SpecDecodeStats {
    /// Sliding-window candidate collection plus prefix-tree construction.
    pub candidate_build: Duration,
    /// Packed target-model verify-tree forward calls.
    pub verify_tree: Duration,
    /// Host-side greedy tree traversal and acceptance test.
    pub traverse: Duration,
    /// KV-cache trim/gather after verification.
    pub commit: Duration,
    /// Single-token decode steps after bonus/fallback tokens or strict replay.
    pub step_one: Duration,
    /// Device argmax used by empty-tree fallback.
    pub fallback_argmax: Duration,
    pub verify_tree_calls: u32,
    pub step_one_calls: u32,
    pub fallback_argmax_calls: u32,
    pub candidate_steps: u32,
    pub candidates_total: u64,
    pub candidates_max: u32,
    pub empty_tree_calls: u32,
    pub rejected_tree_calls: u32,
    pub accepted_tree_calls: u32,
    pub tree_nodes_total: u64,
    pub tree_nodes_max: u32,
}

impl SpecDecodeStats {
    pub fn add_assign(&mut self, other: &Self) {
        self.candidate_build += other.candidate_build;
        self.verify_tree += other.verify_tree;
        self.traverse += other.traverse;
        self.commit += other.commit;
        self.step_one += other.step_one;
        self.fallback_argmax += other.fallback_argmax;
        self.verify_tree_calls += other.verify_tree_calls;
        self.step_one_calls += other.step_one_calls;
        self.fallback_argmax_calls += other.fallback_argmax_calls;
        self.candidate_steps += other.candidate_steps;
        self.candidates_total += other.candidates_total;
        self.candidates_max = self.candidates_max.max(other.candidates_max);
        self.empty_tree_calls += other.empty_tree_calls;
        self.rejected_tree_calls += other.rejected_tree_calls;
        self.accepted_tree_calls += other.accepted_tree_calls;
        self.tree_nodes_total += other.tree_nodes_total;
        self.tree_nodes_max = self.tree_nodes_max.max(other.tree_nodes_max);
    }

    pub fn avg_candidates(&self) -> f32 {
        if self.candidate_steps == 0 {
            0.0
        } else {
            self.candidates_total as f32 / self.candidate_steps as f32
        }
    }

    pub fn avg_tree_nodes(&self) -> f32 {
        if self.verify_tree_calls == 0 {
            0.0
        } else {
            self.tree_nodes_total as f32 / self.verify_tree_calls as f32
        }
    }
}

/// Wall-clock and counter stats for one HSD stage (Stage 1 or Stage 2).
#[derive(Debug, Clone, Default)]
pub struct StageStats {
    pub vision_prefill: Duration,
    pub draft_prep: Duration,
    pub decode: Duration,
    pub accept: AcceptStats,
    /// Total *target* tokens emitted by this stage.
    pub emitted_tokens: u32,
    /// Number of forward passes through the LLM (prefill counted as 1).
    pub forward_passes: u32,
    /// Shared HSD driver internals for profiler attribution.
    pub dsv: SpecDecodeStats,
}

/// End-to-end HSD timing breakdown for one page, plus per-stage details.
#[derive(Debug, Clone, Default)]
pub struct HsdStats {
    pub stage1: StageStats,
    pub stage2: StageStats,
    /// Drafter pipeline wall-clock (layout + region recognition).
    pub drafter: Duration,
}

impl HsdStats {
    /// Total wall-clock from page-image input to final parsing result.
    pub fn total(&self) -> Duration {
        self.drafter
            + self.stage1.vision_prefill
            + self.stage1.draft_prep
            + self.stage1.decode
            + self.stage2.vision_prefill
            + self.stage2.draft_prep
            + self.stage2.decode
    }

    /// Combined AAL across both stages (weighted by step count).
    pub fn overall_aal(&self) -> f32 {
        let total_steps = self.stage1.accept.num_steps + self.stage2.accept.num_steps;
        if total_steps == 0 {
            0.0
        } else {
            let sum: u32 = self
                .stage1
                .accept
                .per_step_accepted
                .iter()
                .chain(self.stage2.accept.per_step_accepted.iter())
                .sum();
            sum as f32 / total_steps as f32
        }
    }
}
