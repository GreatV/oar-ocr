//! Dynamic adapter types for runtime adapter management.
//!
//! This module provides type-erased wrapper types for working with model adapters
//! at runtime, enabling flexible task graph construction and execution.
//!
//! # Type System Design
//!
//! This module uses a hybrid approach for type safety and flexibility:
//!
//! - **Input/Output**: Enumerated types (`DynTaskInput`, `DynTaskOutput`) with
//!   pattern matching for type-safe conversions. This avoids downcast on data paths.
//! - **Adapters**: Trait objects (`DynModelAdapter`) with runtime downcast for execution.
//!   This enables dynamic execution without knowing concrete types at compile time.
//!
//! This design balances compile-time safety (for data) with runtime flexibility
//! (for adapter execution). Adapter downcast occurs once per execution and
//! includes proper error handling.

use crate::core::OCRError;
use crate::core::traits::{
    adapter::{AdapterInfo, ModelAdapter},
    task::{ImageTaskInput, TaskType},
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

/// Wrapper to make ModelAdapter trait object-safe.
#[derive(Debug)]
pub struct AdapterWrapper<A: ModelAdapter> {
    adapter: A,
}

impl<A: ModelAdapter> AdapterWrapper<A> {
    pub fn new(adapter: A) -> Self {
        Self { adapter }
    }
}

/// Macro to execute adapters that use ImageTaskInput
macro_rules! execute_image_adapter {
    ($self:expr, $input:expr, $task_type:expr, $adapter_type:ty, $output_variant:ident) => {{
        let image_input = match $input {
            DynTaskInput::Image(img) => img,
            _ => {
                return Err(OCRError::InvalidInput {
                    message: format!(
                        concat!(
                            "Expected ImageTaskInput for ",
                            stringify!($task_type),
                            ", got {:?}"
                        ),
                        std::mem::discriminant(&$input)
                    ),
                });
            }
        };

        let adapter = (&$self.adapter as &dyn std::any::Any)
            .downcast_ref::<$adapter_type>()
            .ok_or_else(|| OCRError::ConfigError {
                message: concat!("Failed to downcast to ", stringify!($adapter_type)).to_string(),
            })?;

        let output = adapter.execute(image_input, None)?;
        Ok(DynTaskOutput::$output_variant(output))
    }};
}

impl<A: ModelAdapter + 'static> DynModelAdapter for AdapterWrapper<A> {
    fn info(&self) -> AdapterInfo {
        self.adapter.info()
    }

    fn task_type(&self) -> TaskType {
        self.adapter.info().task_type
    }

    fn supports_batching(&self) -> bool {
        self.adapter.supports_batching()
    }

    fn recommended_batch_size(&self) -> usize {
        self.adapter.recommended_batch_size()
    }

    fn execute_dyn(&self, input: DynTaskInput) -> Result<DynTaskOutput, OCRError> {
        use crate::domain::adapters::*;

        match self.adapter.info().task_type {
            TaskType::TextDetection => execute_image_adapter!(
                self,
                input,
                TextDetection,
                TextDetectionAdapter,
                TextDetection
            ),
            TaskType::TextRecognition => {
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
                let adapter = (&self.adapter as &dyn std::any::Any)
                    .downcast_ref::<TextRecognitionAdapter>()
                    .ok_or_else(|| OCRError::ConfigError {
                        message: "Failed to downcast to TextRecognitionAdapter".to_string(),
                    })?;
                let output = adapter.execute(rec_input, None)?;
                Ok(DynTaskOutput::TextRecognition(output))
            }
            TaskType::DocumentOrientation => execute_image_adapter!(
                self,
                input,
                DocumentOrientation,
                DocumentOrientationAdapter,
                DocumentOrientation
            ),
            TaskType::TextLineOrientation => execute_image_adapter!(
                self,
                input,
                TextLineOrientation,
                TextLineOrientationAdapter,
                TextLineOrientation
            ),
            TaskType::DocumentRectification => execute_image_adapter!(
                self,
                input,
                DocumentRectification,
                UVDocRectifierAdapter,
                DocumentRectification
            ),
            TaskType::LayoutDetection => execute_image_adapter!(
                self,
                input,
                LayoutDetection,
                LayoutDetectionAdapter,
                LayoutDetection
            ),
            TaskType::TableCellDetection => execute_image_adapter!(
                self,
                input,
                TableCellDetection,
                TableCellDetectionAdapter,
                TableCellDetection
            ),
            TaskType::FormulaRecognition => execute_image_adapter!(
                self,
                input,
                FormulaRecognition,
                FormulaRecognitionAdapter,
                FormulaRecognition
            ),
            TaskType::SealTextDetection => execute_image_adapter!(
                self,
                input,
                SealTextDetection,
                SealTextDetectionAdapter,
                SealTextDetection
            ),
            TaskType::TableClassification => execute_image_adapter!(
                self,
                input,
                TableClassification,
                TableClassificationAdapter,
                TableClassification
            ),
            TaskType::TableStructureRecognition => execute_image_adapter!(
                self,
                input,
                TableStructureRecognition,
                TableStructureRecognitionAdapter,
                TableStructureRecognition
            ),
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
