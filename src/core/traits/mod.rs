//! Trait definitions for the OCR pipeline.
//!
//! This module groups the foundational predictor traits (`standard`) and the
//! component-level, composable traits (`granular`). Use `standard` for the
//! high-level predictor interfaces implemented across the crate, and reach for
//! `granular` when you need to assemble predictors from interchangeable image
//! readers, preprocessors, inference engines, and postprocessors.
//!
//! The `task` and `adapter` modules provide the primary architecture
//! for defining OCR tasks and adapting models to conform to task interfaces.
//!
//! The `task_def` module provides the `TaskDefinition` trait for compile-time
//! task registration, allowing each task module to define its own metadata.

pub mod adapter;
pub mod granular;
pub mod standard;
pub mod task;
pub mod task_def;

pub use adapter::{AdapterBuilder, AdapterInfo, AdapterTask, ModelAdapter, OrtConfigurable};
pub use granular::{
    ImageReader as GranularImageReader, InferenceEngine, Postprocessor, Preprocessor,
};
pub use standard::{ImageReader, Sampler};
pub use task::{ImageTaskInput, Task, TaskRunner, TaskSchema, TaskType};
pub use task_def::TaskDefinition;
