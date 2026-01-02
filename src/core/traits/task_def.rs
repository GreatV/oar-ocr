//! Trait for compile-time task registration.
//!
//! This module provides the `TaskDefinition` trait that each task output type
//! implements to provide metadata for the task system. This decentralizes
//! task metadata from the central macro to individual task modules.

use std::fmt::Debug;

/// Trait for compile-time task registration.
///
/// Implemented by each task's output type to provide metadata for enum generation
/// and dynamic dispatch. The trait constants are used by the `define_task_system!`
/// macro to generate documentation and helper methods.
///
/// # Example
///
/// ```rust,ignore
/// use oar_ocr::core::traits::TaskDefinition;
///
/// #[derive(Debug, Clone)]
/// pub struct MyTaskOutput {
///     pub results: Vec<String>,
/// }
///
/// impl TaskDefinition for MyTaskOutput {
///     const TASK_NAME: &'static str = "my_task";
///     const TASK_DOC: &'static str = "My custom task for processing data";
///
///     fn empty() -> Self {
///         Self { results: Vec::new() }
///     }
/// }
/// ```
pub trait TaskDefinition: Send + Sync + Clone + Debug + 'static {
    /// Snake_case name for the task (e.g., "text_detection").
    ///
    /// Used by `TaskType::name()` to return a human-readable identifier.
    const TASK_NAME: &'static str;

    /// Human-readable documentation for the task.
    ///
    /// Used to generate doc comments on enum variants.
    const TASK_DOC: &'static str;

    /// Creates an empty output instance.
    ///
    /// Used for default values, testing, and by `DynTaskOutput::empty_for()`.
    fn empty() -> Self;
}
