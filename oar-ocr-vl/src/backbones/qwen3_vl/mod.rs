//! Qwen3-VL backbone shared by checkpoints with the `qwen3_vl` architecture.
//!
//! Split into a full-attention vision tower ([`vision`], with DeepStack
//! auxiliary mergers and bilinearly interpolated learned position embeddings)
//! and the Qwen3 text decoder ([`text`], per-head q/k RMSNorm, GQA, and
//! interleaved MRoPE). Model crates own tokenization, prompt assembly,
//! preprocessing, and generation; the backbone only maps tensors to tensors.

pub(crate) mod text;
pub(crate) mod vision;

// Crate-internal aliases; models re-export the types from `text`/`vision`
// directly to preserve their public visibility.
pub(crate) use text::{DeepstackVisualEmbeds, Qwen3VlTextModel};
pub(crate) use vision::Qwen3VlVisionModel;
