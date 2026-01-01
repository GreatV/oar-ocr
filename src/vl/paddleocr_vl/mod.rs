//! PaddleOCR-VL (Vision-Language) model.
//!
//! A ~2B parameter VLM for document understanding tasks including:
//! - OCR (text recognition)
//! - Table recognition (outputs HTML)
//! - Formula recognition (outputs LaTeX)
//! - Chart recognition

mod config;
mod ernie;
mod model;
mod processing;
mod projector;
mod vision;

pub use config::{
    PaddleOcrVlConfig, PaddleOcrVlImageProcessorConfig, PaddleOcrVlRopeScaling,
    PaddleOcrVlVisionConfig,
};
pub use model::{PaddleOcrVl, PaddleOcrVlTask};
pub use processing::{
    PaddleOcrVlImageInputs, postprocess_table_output, preprocess_images, smart_resize,
    strip_math_wrappers,
};
