//! Qwen2-VL-family backbones shared across models.
//!
//! [`vision`] holds the vision tower (used by MinerU2.5 and
//! MinerU-Diffusion); [`text`] holds the Qwen2 text decoder shared by
//! MinerU2.5 and NaviDC-OCR, parameterised by [`Qwen2VlTextConfig`].

mod text;
mod vision;

pub use text::{Qwen2VlTextConfig, Qwen2VlTextModel};
pub use vision::{Qwen2VlVisionConfig, Qwen2VlVisionModel};
