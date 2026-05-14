//! Hierarchical Speculative Decoding (HSD) for VLM-based document parsers.
//!
//! Reference: Liao et al., "HSD: Training-Free Acceleration for Document Parsing
//! Vision-Language Model with Hierarchical Speculative Decoding" (arXiv:2602.12957).
//!
//! ## High-level flow
//!
//! 1. A lightweight pipeline drafter (e.g., PP-DocLayout + PP-OCRv5) emits
//!    a region partition `R = {r_i}` and one or more coarse text candidates
//!    for each region.
//! 2. Each region candidate is tokenized with the **target VLM's** tokenizer, yielding
//!    [`RegionDraft`]. Stage 2 keeps page-level drafts as an unordered collection
//!    of [`Draft`] values, one entry per region/output.
//! 3. Stage 1 (region-level): for each `r_i`, the target VLM runs `SpecDecode` on the
//!    cropped region image and the corresponding candidate set, in parallel across regions.
//! 4. Stage 2 (page-level): the Stage-1 outputs are passed as the paper's unordered
//!    set `Y^pg = {y^(i)}`. Each output remains a separate [`Draft`], so the
//!    sliding-window matcher scans each draft independently before the full-page
//!    verification restores global coherence.
//!
//! ## Module layout
//!
//! - [`types`]    — shared data structures (drafts, configs, stats).
//! - [`drafting`] — layout-output to HSD draft conversion helpers.
//! - [`kv_trim`]  — append/trim/gather KV-cache wrapper used by backends.
//! - [`matching`] — draft-target matching with a sliding reference window.
//! - [`prefix_tree`] — prefix-tree construction over candidate suffixes.
//! - [`verify`]   — DSV `SpecDecode` operator (model-side hooks live here).
//! - [`backend_util`] — mechanical helpers (pos-id / keep-index construction)
//!   shared by every `SpecBackend` impl.

pub mod backend_util;
pub mod drafting;
pub mod matching;
pub mod prefix_tree;
pub mod types;
pub mod verify;

// `kv_trim` is declared at crate root (see `lib.rs`) so the cache type stays
// available without the `hsd` feature; re-export the type here so
// HSD-internal callers keep using `crate::hsd::TrimmableKvCache`.
pub use crate::kv_trim::TrimmableKvCache;
pub use types::{
    AcceptStats, Draft, DsvConfig, HsdConfig, HsdStats, RegionDraft, RegionStageStats, StageStats,
};
