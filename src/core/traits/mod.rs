//! Trait definitions for the OCR pipeline.
//!
//! This module groups the foundational predictor traits (`standard`) and the
//! component-level, composable traits (`granular`). Use `standard` for the
//! high-level predictor interfaces implemented across the crate, and reach for
//! `granular` when you need to assemble predictors from interchangeable image
//! readers, preprocessors, inference engines, and postprocessors.

pub mod granular;
pub mod standard;

pub use granular::{
    ImageReader as GranularImageReader, InferenceEngine, ModularPredictor, Postprocessor,
    Preprocessor,
};
pub use standard::{
    BasePredictor, ImageReader, PredictorBuilder, PredictorConfig, Sampler, StandardPredictor,
};
