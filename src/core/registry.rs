//! Dynamic adapter types for runtime adapter management.
//!
//! This module provides type-erased wrapper types for working with model adapters
//! at runtime, enabling flexible task graph construction and execution.
//!
//! # Type System Design
//!
//! This module uses enum-based dispatch for both inputs/outputs and adapters:
//!
//! - **Input/Output**: Enumerated types (`DynTaskInput`, `DynTaskOutput`) with
//!   pattern matching for type-safe conversions.
//! - **Adapters**: Enumerated type (`TaskAdapter`) with direct variant matching.
//!   This avoids runtime downcast and provides compile-time exhaustiveness checking.
//!
//! The `DynModelAdapter` trait is retained for custom/mock adapters in testing scenarios.
//!
//! # Note on Imports
//!
//! Adapter and Output types are NOT imported here - the `with_task_registry!` macro
//! uses fully qualified `$crate::` paths, so macro-generated code includes full paths.
//! This means adding a new task type requires NO changes to this file's imports.

use crate::core::OCRError;
use crate::core::traits::{
    adapter::{AdapterInfo, ModelAdapter},
    task::{ImageTaskInput, TaskType},
};
use std::fmt::Debug;

/// Type-erased input for dynamic adapter execution.
///
/// All task inputs are image-based. This enum provides a uniform wrapper
/// for dynamic dispatch while maintaining type safety.
#[derive(Debug, Clone)]
pub enum DynTaskInput {
    /// Image-based input (used by all tasks)
    Image(ImageTaskInput),
}

impl DynTaskInput {
    /// Creates a DynTaskInput from ImageTaskInput.
    pub fn from_images(input: ImageTaskInput) -> Self {
        Self::Image(input)
    }
}

// Generate DynTaskOutput enum from the central task registry
crate::with_task_registry!(crate::impl_dyn_task_output);

/// A type-erased model adapter that can be stored in the registry.
///
/// This trait extends ModelAdapter to support dynamic dispatch and execution.
pub trait DynModelAdapter: Send + Sync + Debug {
    /// Returns information about this adapter.
    fn info(&self) -> AdapterInfo;

    /// Returns the task type this adapter handles.
    fn task_type(&self) -> TaskType;

    /// Returns whether this adapter supports batching.
    fn supports_batching(&self) -> bool;

    /// Returns the recommended batch size.
    fn recommended_batch_size(&self) -> usize;

    /// Executes the adapter with type-erased input and returns type-erased output.
    ///
    /// This method enables dynamic execution of adapters without knowing their
    /// concrete types at compile time.
    ///
    /// # Arguments
    ///
    /// * `input` - Type-erased input matching the adapter's task type
    ///
    /// # Returns
    ///
    /// Type-erased output from the adapter execution
    ///
    /// # Errors
    ///
    /// Returns an error if:
    /// - The input type doesn't match the adapter's expected input
    /// - The adapter execution fails
    fn execute_dyn(&self, input: DynTaskInput) -> Result<DynTaskOutput, OCRError>;
}

// Generate TaskAdapter enum and DynModelAdapter impl from the central task registry.
// All adapters are uniformly boxed to enable macro generation.
crate::with_task_registry!(crate::impl_task_adapter);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dyn_task_output_has_variant_and_conversion_for_each_task_type() {
        use crate::core::traits::task::TaskType::*;

        let all = [
            TextDetection,
            TextRecognition,
            DocumentOrientation,
            TextLineOrientation,
            DocumentRectification,
            LayoutDetection,
            TableCellDetection,
            FormulaRecognition,
            SealTextDetection,
            TableClassification,
            TableStructureRecognition,
        ];

        for task_type in all {
            let output = DynTaskOutput::empty_for(task_type);
            assert_eq!(output.task_type(), task_type);

            // Ensure the corresponding conversion method is wired and returns Ok.
            match task_type {
                TextDetection => {
                    output.clone().into_text_detection().unwrap();
                }
                TextRecognition => {
                    output.clone().into_text_recognition().unwrap();
                }
                DocumentOrientation => {
                    output.clone().into_document_orientation().unwrap();
                }
                TextLineOrientation => {
                    output.clone().into_text_line_orientation().unwrap();
                }
                DocumentRectification => {
                    output.clone().into_document_rectification().unwrap();
                }
                LayoutDetection => {
                    output.clone().into_layout_detection().unwrap();
                }
                TableCellDetection => {
                    output.clone().into_table_cell_detection().unwrap();
                }
                FormulaRecognition => {
                    output.clone().into_formula_recognition().unwrap();
                }
                SealTextDetection => {
                    output.clone().into_seal_text_detection().unwrap();
                }
                TableClassification => {
                    output.clone().into_table_classification().unwrap();
                }
                TableStructureRecognition => {
                    output.clone().into_table_structure_recognition().unwrap();
                }
            }
        }
    }
}
