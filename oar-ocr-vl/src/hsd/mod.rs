//! Hierarchical Speculative Decoding (HSD) for VLM-based document parsers.
//!
//! Reference: Liao et al., "HSD: Training-Free Acceleration for Document Parsing
//! Vision-Language Model with Hierarchical Speculative Decoding" (arXiv:2602.12957).
//!
//! ## High-level flow
//!
//! 1. A lightweight pipeline drafter (e.g., PP-DocLayout + PP-OCRv5) emits
//!    a region partition `R = {r_i}` and a coarse text draft for each region.
//! 2. Each region draft is tokenized with the **target VLM's** tokenizer, yielding
//!    [`RegionDraft`]. The aggregated reading-order draft becomes [`PageDraft`].
//! 3. Stage 1 (region-level): for each `r_i`, the target VLM runs `SpecDecode` on the
//!    cropped region image and the corresponding region draft, in parallel across regions.
//! 4. Stage 2 (page-level): the Stage-1 outputs are concatenated into a single page-level
//!    draft, and `SpecDecode` runs once more on the full-page image to restore global
//!    coherence (reading order, cross-region references).
//!
//! ## Module layout
//!
//! - [`types`]    — shared data structures (drafts, configs, stats).
//! - [`drafting`] — layout-output to HSD draft conversion helpers.
//! - [`kv_trim`]  — append/trim/gather KV-cache wrapper used by backends.
//! - [`matching`] — draft-target matching with a sliding reference window.
//! - [`nvtx`]     — optional NVTX profiling markers.
//! - [`prefix_tree`] — prefix-tree construction over candidate suffixes.
//! - [`verify`]   — DSV `SpecDecode` operator (model-side hooks live here).

pub mod drafting;
pub mod kv_trim;
pub mod matching;
pub mod nvtx;
pub mod prefix_tree;
pub mod types;
pub mod verify;

pub use kv_trim::TrimmableKvCache;
pub use types::{
    AcceptStats, Draft, DsvConfig, HsdConfig, HsdStats, PageDraft, RegionDraft, StageStats,
};
