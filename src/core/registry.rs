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

use crate::core::OCRError;
use crate::core::traits::{
    adapter::{AdapterInfo, ModelAdapter},
    task::{ImageTaskInput, TaskType},
};
use crate::domain::adapters::{
    DocumentOrientationAdapter, FormulaRecognitionAdapter, LayoutDetectionAdapter,
    SealTextDetectionAdapter, TableCellDetectionAdapter, TableClassificationAdapter,
    TableStructureRecognitionAdapter, TextDetectionAdapter, TextLineOrientationAdapter,
    TextRecognitionAdapter, UVDocRectifierAdapter,
};
use crate::domain::tasks::{
    DocumentOrientationOutput, DocumentRectificationOutput, FormulaRecognitionOutput,
    LayoutDetectionOutput, SealTextDetectionOutput, TableCellDetectionOutput,
    TableClassificationOutput, TableStructureRecognitionOutput, TextDetectionOutput,
    TextLineOrientationOutput, TextRecognitionInput, TextRecognitionOutput,
};
use std::fmt::Debug;

/// Type-erased input for dynamic adapter execution.
///
/// This enum wraps all possible task input types to enable dynamic dispatch.
#[derive(Debug, Clone)]
pub enum DynTaskInput {
    /// Image-based input (used by most tasks)
    Image(ImageTaskInput),
    /// Text recognition input (cropped text images)
    TextRecognition(TextRecognitionInput),
}

impl DynTaskInput {
    /// Creates a DynTaskInput from ImageTaskInput.
    pub fn from_images(input: ImageTaskInput) -> Self {
        Self::Image(input)
    }

    /// Creates a DynTaskInput from TextRecognitionInput.
    pub fn from_text_recognition(input: TextRecognitionInput) -> Self {
        Self::TextRecognition(input)
    }
}

/// Type-erased output from dynamic adapter execution.
///
/// This enum wraps all possible task output types to enable dynamic dispatch.
#[derive(Debug, Clone)]
pub enum DynTaskOutput {
    /// Text detection output
    TextDetection(TextDetectionOutput),
    /// Text recognition output
    TextRecognition(TextRecognitionOutput),
    /// Document orientation output
    DocumentOrientation(DocumentOrientationOutput),
    /// Text line orientation output
    TextLineOrientation(TextLineOrientationOutput),
    /// Document rectification output
    DocumentRectification(DocumentRectificationOutput),
    /// Layout detection output
    LayoutDetection(LayoutDetectionOutput),
    /// Table cell detection output
    TableCellDetection(TableCellDetectionOutput),
    /// Formula recognition output
    FormulaRecognition(FormulaRecognitionOutput),
    /// Seal text detection output
    SealTextDetection(SealTextDetectionOutput),
    /// Table classification output
    TableClassification(TableClassificationOutput),
    /// Table structure recognition output
    TableStructureRecognition(TableStructureRecognitionOutput),
}

/// Macro to generate conversion methods for DynTaskOutput variants
macro_rules! impl_dyn_output_conversions {
    ($($variant:ident => $method:ident, $output_type:ty);* $(;)?) => {
        impl DynTaskOutput {
            $(
                #[doc = concat!("Extracts ", stringify!($output_type), " if this is a ", stringify!($variant), " variant.")]
                pub fn $method(self) -> Result<$output_type, OCRError> {
                    match self {
                        Self::$variant(output) => Ok(output),
                        _ => Err(OCRError::InvalidInput {
                            message: format!(
                                concat!("Expected ", stringify!($variant), " output, got {:?}"),
                                std::mem::discriminant(&self)
                            ),
                        }),
                    }
                }
            )*
        }
    };
}

impl_dyn_output_conversions! {
    TextDetection => into_text_detection, TextDetectionOutput;
    TextRecognition => into_text_recognition, TextRecognitionOutput;
    DocumentOrientation => into_document_orientation, DocumentOrientationOutput;
    TextLineOrientation => into_text_line_orientation, TextLineOrientationOutput;
    DocumentRectification => into_document_rectification, DocumentRectificationOutput;
    LayoutDetection => into_layout_detection, LayoutDetectionOutput;
    SealTextDetection => into_seal_text_detection, SealTextDetectionOutput;
    TableCellDetection => into_table_cell_detection, TableCellDetectionOutput;
    FormulaRecognition => into_formula_recognition, FormulaRecognitionOutput;
    TableClassification => into_table_classification, TableClassificationOutput;
    TableStructureRecognition => into_table_structure_recognition, TableStructureRecognitionOutput;
}

impl DynTaskOutput {
    /// Returns the underlying task type for this output.
    pub fn task_type(&self) -> TaskType {
        match self {
            DynTaskOutput::TextDetection(_) => TaskType::TextDetection,
            DynTaskOutput::TextRecognition(_) => TaskType::TextRecognition,
            DynTaskOutput::DocumentOrientation(_) => TaskType::DocumentOrientation,
            DynTaskOutput::TextLineOrientation(_) => TaskType::TextLineOrientation,
            DynTaskOutput::DocumentRectification(_) => TaskType::DocumentRectification,
            DynTaskOutput::LayoutDetection(_) => TaskType::LayoutDetection,
            DynTaskOutput::TableCellDetection(_) => TaskType::TableCellDetection,
            DynTaskOutput::FormulaRecognition(_) => TaskType::FormulaRecognition,
            DynTaskOutput::SealTextDetection(_) => TaskType::SealTextDetection,
            DynTaskOutput::TableClassification(_) => TaskType::TableClassification,
            DynTaskOutput::TableStructureRecognition(_) => TaskType::TableStructureRecognition,
        }
    }

    /// Creates an empty `DynTaskOutput` variant for the given task type.
    ///
    /// This is intended for registry wiring completeness checks and test scaffolding.
    pub fn empty_for(task_type: TaskType) -> Self {
        match task_type {
            TaskType::TextDetection => DynTaskOutput::TextDetection(TextDetectionOutput::empty()),
            TaskType::TextRecognition => {
                DynTaskOutput::TextRecognition(TextRecognitionOutput::empty())
            }
            TaskType::DocumentOrientation => {
                DynTaskOutput::DocumentOrientation(DocumentOrientationOutput::empty())
            }
            TaskType::TextLineOrientation => {
                DynTaskOutput::TextLineOrientation(TextLineOrientationOutput::empty())
            }
            TaskType::DocumentRectification => {
                DynTaskOutput::DocumentRectification(DocumentRectificationOutput::empty())
            }
            TaskType::LayoutDetection => {
                DynTaskOutput::LayoutDetection(LayoutDetectionOutput::empty())
            }
            TaskType::TableCellDetection => {
                DynTaskOutput::TableCellDetection(TableCellDetectionOutput::empty())
            }
            TaskType::FormulaRecognition => {
                DynTaskOutput::FormulaRecognition(FormulaRecognitionOutput::empty())
            }
            TaskType::SealTextDetection => {
                DynTaskOutput::SealTextDetection(SealTextDetectionOutput::empty())
            }
            TaskType::TableClassification => {
                DynTaskOutput::TableClassification(TableClassificationOutput::empty())
            }
            TaskType::TableStructureRecognition => {
                DynTaskOutput::TableStructureRecognition(TableStructureRecognitionOutput::empty())
            }
        }
    }
}

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

/// Task-specific adapter enum that replaces fake dynamic dispatch with honest enum-based dispatch.
///
/// This enum directly holds concrete adapter types, avoiding the downcast pattern.
/// Each variant corresponds to a specific task type and its adapter.
///
/// # Benefits
///
/// - **No runtime downcast**: Direct pattern matching on enum variants
/// - **Compile-time exhaustiveness**: Adding a new task type requires handling it explicitly
/// - **Type safety**: Each variant holds the exact adapter type for that task
///
/// # Custom Adapters
///
/// For testing or extension, use the `Custom` variant which wraps a `Box<dyn DynModelAdapter>`.
/// This preserves backward compatibility with mock adapters in tests.
#[derive(Debug)]
pub enum TaskAdapter {
    /// Text detection adapter
    TextDetection(TextDetectionAdapter),
    /// Text recognition adapter
    TextRecognition(TextRecognitionAdapter),
    /// Document orientation classification adapter
    DocumentOrientation(DocumentOrientationAdapter),
    /// Text line orientation classification adapter
    TextLineOrientation(TextLineOrientationAdapter),
    /// Document rectification adapter (UVDoc)
    DocumentRectification(UVDocRectifierAdapter),
    /// Layout detection adapter
    LayoutDetection(LayoutDetectionAdapter),
    /// Table cell detection adapter
    TableCellDetection(TableCellDetectionAdapter),
    /// Formula recognition adapter (boxed due to large size)
    FormulaRecognition(Box<FormulaRecognitionAdapter>),
    /// Seal text detection adapter
    SealTextDetection(SealTextDetectionAdapter),
    /// Table classification adapter
    TableClassification(TableClassificationAdapter),
    /// Table structure recognition adapter
    TableStructureRecognition(TableStructureRecognitionAdapter),
    /// Custom adapter for testing or extension (wraps DynModelAdapter)
    Custom(Box<dyn DynModelAdapter>),
}

impl TaskAdapter {
    /// Creates a TaskAdapter from a TextDetectionAdapter.
    pub fn text_detection(adapter: TextDetectionAdapter) -> Self {
        Self::TextDetection(adapter)
    }

    /// Creates a TaskAdapter from a TextRecognitionAdapter.
    pub fn text_recognition(adapter: TextRecognitionAdapter) -> Self {
        Self::TextRecognition(adapter)
    }

    /// Creates a TaskAdapter from a DocumentOrientationAdapter.
    pub fn document_orientation(adapter: DocumentOrientationAdapter) -> Self {
        Self::DocumentOrientation(adapter)
    }

    /// Creates a TaskAdapter from a TextLineOrientationAdapter.
    pub fn text_line_orientation(adapter: TextLineOrientationAdapter) -> Self {
        Self::TextLineOrientation(adapter)
    }

    /// Creates a TaskAdapter from a UVDocRectifierAdapter.
    pub fn document_rectification(adapter: UVDocRectifierAdapter) -> Self {
        Self::DocumentRectification(adapter)
    }

    /// Creates a TaskAdapter from a LayoutDetectionAdapter.
    pub fn layout_detection(adapter: LayoutDetectionAdapter) -> Self {
        Self::LayoutDetection(adapter)
    }

    /// Creates a TaskAdapter from a TableCellDetectionAdapter.
    pub fn table_cell_detection(adapter: TableCellDetectionAdapter) -> Self {
        Self::TableCellDetection(adapter)
    }

    /// Creates a TaskAdapter from a FormulaRecognitionAdapter.
    pub fn formula_recognition(adapter: FormulaRecognitionAdapter) -> Self {
        Self::FormulaRecognition(Box::new(adapter))
    }

    /// Creates a TaskAdapter from a SealTextDetectionAdapter.
    pub fn seal_text_detection(adapter: SealTextDetectionAdapter) -> Self {
        Self::SealTextDetection(adapter)
    }

    /// Creates a TaskAdapter from a TableClassificationAdapter.
    pub fn table_classification(adapter: TableClassificationAdapter) -> Self {
        Self::TableClassification(adapter)
    }

    /// Creates a TaskAdapter from a TableStructureRecognitionAdapter.
    pub fn table_structure_recognition(adapter: TableStructureRecognitionAdapter) -> Self {
        Self::TableStructureRecognition(adapter)
    }

    /// Creates a TaskAdapter from a custom DynModelAdapter.
    pub fn custom<A: DynModelAdapter + 'static>(adapter: A) -> Self {
        Self::Custom(Box::new(adapter))
    }
}

/// Helper macro to extract ImageTaskInput from DynTaskInput
macro_rules! extract_image_input {
    ($input:expr, $task_name:expr) => {
        match $input {
            DynTaskInput::Image(img) => img,
            _ => {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        "Expected ImageTaskInput for {}, got {:?}",
                        $task_name,
                        std::mem::discriminant(&$input)
                    ),
                });
            }
        }
    };
}

impl DynModelAdapter for TaskAdapter {
    fn info(&self) -> AdapterInfo {
        match self {
            Self::TextDetection(a) => a.info(),
            Self::TextRecognition(a) => a.info(),
            Self::DocumentOrientation(a) => a.info(),
            Self::TextLineOrientation(a) => a.info(),
            Self::DocumentRectification(a) => a.info(),
            Self::LayoutDetection(a) => a.info(),
            Self::TableCellDetection(a) => a.info(),
            Self::FormulaRecognition(a) => a.info(),
            Self::SealTextDetection(a) => a.info(),
            Self::TableClassification(a) => a.info(),
            Self::TableStructureRecognition(a) => a.info(),
            Self::Custom(a) => a.info(),
        }
    }

    fn task_type(&self) -> TaskType {
        match self {
            Self::TextDetection(_) => TaskType::TextDetection,
            Self::TextRecognition(_) => TaskType::TextRecognition,
            Self::DocumentOrientation(_) => TaskType::DocumentOrientation,
            Self::TextLineOrientation(_) => TaskType::TextLineOrientation,
            Self::DocumentRectification(_) => TaskType::DocumentRectification,
            Self::LayoutDetection(_) => TaskType::LayoutDetection,
            Self::TableCellDetection(_) => TaskType::TableCellDetection,
            Self::FormulaRecognition(_) => TaskType::FormulaRecognition,
            Self::SealTextDetection(_) => TaskType::SealTextDetection,
            Self::TableClassification(_) => TaskType::TableClassification,
            Self::TableStructureRecognition(_) => TaskType::TableStructureRecognition,
            Self::Custom(a) => a.task_type(),
        }
    }

    fn supports_batching(&self) -> bool {
        match self {
            Self::TextDetection(a) => a.supports_batching(),
            Self::TextRecognition(a) => a.supports_batching(),
            Self::DocumentOrientation(a) => a.supports_batching(),
            Self::TextLineOrientation(a) => a.supports_batching(),
            Self::DocumentRectification(a) => a.supports_batching(),
            Self::LayoutDetection(a) => a.supports_batching(),
            Self::TableCellDetection(a) => a.supports_batching(),
            Self::FormulaRecognition(a) => a.supports_batching(),
            Self::SealTextDetection(a) => a.supports_batching(),
            Self::TableClassification(a) => a.supports_batching(),
            Self::TableStructureRecognition(a) => a.supports_batching(),
            Self::Custom(a) => a.supports_batching(),
        }
    }

    fn recommended_batch_size(&self) -> usize {
        match self {
            Self::TextDetection(a) => a.recommended_batch_size(),
            Self::TextRecognition(a) => a.recommended_batch_size(),
            Self::DocumentOrientation(a) => a.recommended_batch_size(),
            Self::TextLineOrientation(a) => a.recommended_batch_size(),
            Self::DocumentRectification(a) => a.recommended_batch_size(),
            Self::LayoutDetection(a) => a.recommended_batch_size(),
            Self::TableCellDetection(a) => a.recommended_batch_size(),
            Self::FormulaRecognition(a) => a.recommended_batch_size(),
            Self::SealTextDetection(a) => a.recommended_batch_size(),
            Self::TableClassification(a) => a.recommended_batch_size(),
            Self::TableStructureRecognition(a) => a.recommended_batch_size(),
            Self::Custom(a) => a.recommended_batch_size(),
        }
    }

    fn execute_dyn(&self, input: DynTaskInput) -> Result<DynTaskOutput, OCRError> {
        match self {
            Self::TextDetection(adapter) => {
                let image_input = extract_image_input!(input, "TextDetection");
                let output = adapter.execute(image_input, None)?;
                Ok(DynTaskOutput::TextDetection(output))
            }
            Self::TextRecognition(adapter) => {
                let rec_input = match input {
                    DynTaskInput::TextRecognition(rec) => rec,
                    _ => {
                        return Err(OCRError::InvalidInput {
                            message: format!(
                                "Expected TextRecognitionInput for TextRecognition, got {:?}",
                                std::mem::discriminant(&input)
                            ),
                        });
                    }
                };
                let output = adapter.execute(rec_input, None)?;
                Ok(DynTaskOutput::TextRecognition(output))
            }
            Self::DocumentOrientation(adapter) => {
                let image_input = extract_image_input!(input, "DocumentOrientation");
                let output = adapter.execute(image_input, None)?;
                Ok(DynTaskOutput::DocumentOrientation(output))
            }
            Self::TextLineOrientation(adapter) => {
                let image_input = extract_image_input!(input, "TextLineOrientation");
                let output = adapter.execute(image_input, None)?;
                Ok(DynTaskOutput::TextLineOrientation(output))
            }
            Self::DocumentRectification(adapter) => {
                let image_input = extract_image_input!(input, "DocumentRectification");
                let output = adapter.execute(image_input, None)?;
                Ok(DynTaskOutput::DocumentRectification(output))
            }
            Self::LayoutDetection(adapter) => {
                let image_input = extract_image_input!(input, "LayoutDetection");
                let output = adapter.execute(image_input, None)?;
                Ok(DynTaskOutput::LayoutDetection(output))
            }
            Self::TableCellDetection(adapter) => {
                let image_input = extract_image_input!(input, "TableCellDetection");
                let output = adapter.execute(image_input, None)?;
                Ok(DynTaskOutput::TableCellDetection(output))
            }
            Self::FormulaRecognition(adapter) => {
                let image_input = extract_image_input!(input, "FormulaRecognition");
                let output = adapter.execute(image_input, None)?;
                Ok(DynTaskOutput::FormulaRecognition(output))
            }
            Self::SealTextDetection(adapter) => {
                let image_input = extract_image_input!(input, "SealTextDetection");
                let output = adapter.execute(image_input, None)?;
                Ok(DynTaskOutput::SealTextDetection(output))
            }
            Self::TableClassification(adapter) => {
                let image_input = extract_image_input!(input, "TableClassification");
                let output = adapter.execute(image_input, None)?;
                Ok(DynTaskOutput::TableClassification(output))
            }
            Self::TableStructureRecognition(adapter) => {
                let image_input = extract_image_input!(input, "TableStructureRecognition");
                let output = adapter.execute(image_input, None)?;
                Ok(DynTaskOutput::TableStructureRecognition(output))
            }
            Self::Custom(adapter) => adapter.execute_dyn(input),
        }
    }
}

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
