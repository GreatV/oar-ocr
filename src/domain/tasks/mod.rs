//! Task module exports.
//!
//! This module contains concrete implementations of OCR tasks.

pub mod document_orientation;
pub mod document_rectification;
pub mod formula_recognition;
pub mod layout_detection;
pub mod text_detection;
pub mod text_line_orientation;
pub mod text_recognition;

pub use document_orientation::{
    DocumentOrientationConfig, DocumentOrientationOutput, DocumentOrientationTask,
};
pub use document_rectification::{
    DocumentRectificationConfig, DocumentRectificationOutput, DocumentRectificationTask,
};
pub use formula_recognition::{
    FormulaRecognitionConfig, FormulaRecognitionOutput, FormulaRecognitionTask,
};
pub use layout_detection::{
    LayoutDetectionConfig, LayoutDetectionOutput, LayoutDetectionTask, LayoutElement,
    LayoutElementType,
};
pub use text_detection::{TextDetectionConfig, TextDetectionOutput, TextDetectionTask};
pub use text_line_orientation::{
    TextLineOrientationConfig, TextLineOrientationOutput, TextLineOrientationTask,
};
pub use text_recognition::{
    TextRecognitionConfig, TextRecognitionInput, TextRecognitionOutput, TextRecognitionTask,
};
