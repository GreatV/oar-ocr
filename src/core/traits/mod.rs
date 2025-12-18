//! Trait definitions for the OCR pipeline.
//!
//! This module groups the foundational predictor traits (`standard`) and the
//! component-level, composable traits (`granular`). Use `standard` for the
//! high-level predictor interfaces implemented across the crate, and reach for
//! `granular` when you need to assemble predictors from interchangeable image
//! readers, preprocessors, inference engines, and postprocessors.
//!
//! The new `task` and `adapter` modules provide a more flexible architecture
//! for defining OCR tasks and adapting models to conform to task interfaces.

pub mod adapter;
pub mod granular;
pub mod standard;
pub mod task;

pub use adapter::{AdapterBuilder, AdapterInfo, AdapterTask, ModelAdapter};
pub use granular::{
    ImageReader as GranularImageReader, InferenceEngine, ModularPredictor, Postprocessor,
    Preprocessor,
};
pub use standard::{
    BasePredictor, ImageReader, PredictorBuilder, PredictorConfig, Sampler, StandardPredictor,
};
pub use task::{ImageTaskInput, Task, TaskRunner, TaskSchema, TaskType};
