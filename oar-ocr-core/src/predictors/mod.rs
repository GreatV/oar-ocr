//! Predictors module
//!
//! This module provides high-level predictor APIs for various OCR tasks.
//! Each predictor encapsulates the core functionality for a specific task,
//! providing a simple interface for model loading, configuration, and prediction.

#[macro_use]
mod builder;
mod core;

use std::path::{Path, PathBuf};

use crate::core::traits::adapter::{AdapterBuilder, ModelAdapter};

pub use core::TaskPredictorCore;

pub(crate) fn resolve_asset_path(path: &Path) -> crate::core::OcrResult<PathBuf> {
    #[cfg(feature = "auto-download")]
    {
        crate::core::download::resolve_path(path)
    }
    #[cfg(not(feature = "auto-download"))]
    {
        Ok(path.to_path_buf())
    }
}

pub(crate) fn build_adapter<B>(
    builder: B,
    model_path: &Path,
) -> crate::core::OcrResult<Box<B::Adapter>>
where
    B: AdapterBuilder,
    B::Adapter: ModelAdapter + 'static,
{
    let model_path = resolve_asset_path(model_path)?;
    Ok(Box::new(builder.build(&model_path)?))
}

pub mod document_orientation;
pub mod document_rectification;
pub mod formula_recognition;
pub mod layout_detection;
pub mod seal_text_detection;
pub mod table_cell_detection;
pub mod table_classification;
pub mod table_structure_recognition;
pub mod text_detection;
pub mod text_line_orientation;
pub mod text_recognition;

pub use document_orientation::DocumentOrientationPredictor;
pub use document_rectification::DocumentRectificationPredictor;
pub use formula_recognition::FormulaRecognitionPredictor;
pub use layout_detection::LayoutDetectionPredictor;
pub use seal_text_detection::SealTextDetectionPredictor;
pub use table_cell_detection::{TableCellDetectionPredictor, TableCellModelVariant};
pub use table_classification::TableClassificationPredictor;
pub use table_structure_recognition::TableStructureRecognitionPredictor;
pub use text_detection::TextDetectionPredictor;
pub use text_line_orientation::TextLineOrientationPredictor;
pub use text_recognition::TextRecognitionPredictor;
