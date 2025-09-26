//! Image resizing helpers for different OCR stages.
//!
//! - [`ocr`] contains resizing logic tailored for recognition inputs.
//! - [`detection`] offers resizing strategies for detection models.

pub mod detection;
pub mod ocr;

pub use detection::*;
pub use ocr::*;
