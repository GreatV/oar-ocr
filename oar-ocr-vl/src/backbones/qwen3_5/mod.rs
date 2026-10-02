//! Qwen3.5 backbone shared by checkpoints with the `qwen3_5` architecture.
//!
//! The text decoder alternates three Gated DeltaNet linear-attention layers
//! with one full-attention layer ([`text`], zero-centred decoder RMSNorm,
//! gated attention output, interleaved MRoPE) and recurs through the Gated
//! Delta Rule kernel ([`gated_delta`]). Model crates own tokenization, prompt
//! assembly, preprocessing, and generation; the backbone only maps tensors to
//! tensors.

pub(crate) mod gated_delta;
pub(crate) mod text;
