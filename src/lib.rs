//! # OAR OCR
//!
//! A Rust OCR library that extracts text from document images using ONNX models.
//! Supports text detection, recognition, document orientation, and rectification.
//!
//! ## Features
//!
//! - Complete OCR pipeline from image to text
//! - High-level builder APIs for easy pipeline configuration
//! - Model adapter system for easy model swapping
//! - Batch processing support
//! - ONNX Runtime integration for fast inference
//!
//! ## Components
//!
//! - **Text Detection**: Find text regions in images
//! - **Text Recognition**: Convert text regions to readable text
//! - **Layout Detection**: Identify document structure elements (text blocks, titles, tables, figures)
//! - **Document Orientation**: Detect document rotation (0°, 90°, 180°, 270°)
//! - **Document Rectification**: Fix perspective distortion
//! - **Text Line Classification**: Detect text line orientation
//! - **Seal Text Detection**: Detect text in circular seals
//! - **Formula Recognition**: Recognize mathematical formulas
//!
//! ## Modules
//!
//! * [`core`] - Core traits, error handling, and batch processing
//! * [`domain`] - Domain types like orientation helpers and prediction models
//! * [`models`] - Model adapters for different OCR tasks
//! * [`oarocr`] - High-level OCR pipeline builders
//! * [`processors`] - Image processing utilities
//! * [`utils`] - Utility functions for images and tensors
//! * [`predictors`] - Task-specific predictor interfaces
//!
//! ## Quick Start
//!
//! ### OCR Pipeline
//!
//! ```rust,no_run
//! use oar_ocr::oarocr::{OAROCRBuilder, OAROCR};
//! use oar_ocr::utils::load_image;
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create OCR pipeline with required components
//! let ocr = OAROCRBuilder::new(
//!     "models/text_detection.onnx",
//!     "models/text_recognition.onnx",
//!     "models/character_dict.txt"
//! )
//! .with_document_image_orientation_classification("models/doc_orient.onnx")
//! .with_text_line_orientation_classification("models/line_orient.onnx")
//! .image_batch_size(4)
//! .region_batch_size(32)
//! .build()?;
//!
//! // Process images
//! let image = load_image(Path::new("document.jpg"))?;
//! let results = ocr.predict(vec![image])?;
//!
//! for result in results {
//!     for region in result.text_regions {
//!         if let Some(text) = region.text {
//!             println!("Text: {}", text);
//!         }
//!     }
//! }
//! # Ok(())
//! # }
//! ```
//!
//! ### Document Structure Analysis
//!
//! ```rust,no_run
//! use oar_ocr::oarocr::{OARStructureBuilder, OARStructure};
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create structure analysis pipeline
//! let structure = OARStructureBuilder::new("models/layout_detection.onnx")
//!     .with_table_classification("models/table_classification.onnx")
//!     .with_table_cell_detection("models/table_cell_detection.onnx", "wired")
//!     .with_table_structure_recognition("models/table_structure.onnx", "wired")
//!     .table_structure_dict_path("models/table_structure_dict.txt")
//!     .with_formula_recognition(
//!         "models/formula_recognition.onnx",
//!         "models/tokenizer.json",
//!         "pp_formulanet"
//!     )
//!     .build()?;
//!
//! // Analyze document structure
//! let result = structure.predict("document.jpg")?;
//!
//! println!("Layout elements: {}", result.layout_elements.len());
//! println!("Tables: {}", result.tables.len());
//! println!("Formulas: {}", result.formulas.len());
//! # Ok(())
//! # }
//! ```

// Re-export core modules from oar-ocr-core
pub mod core {
    pub use oar_ocr_core::core::*;
}

pub mod domain {
    pub use oar_ocr_core::domain::*;
}

pub mod models {
    pub use oar_ocr_core::models::*;
}

pub mod processors {
    pub use oar_ocr_core::processors::*;
}

pub mod predictors {
    pub use oar_ocr_core::predictors::*;
}

// Utils module with re-exports from core and OCR-specific visualization
pub mod utils;

// High-level OCR API (remains in main crate)
pub mod oarocr;

// Re-export derive macros for convenient use
pub use oar_ocr_derive::{ConfigValidator, TaskPredictorBuilder};

/// Prelude module for convenient imports.
///
///  Bring the essentials into scope with a single use statement:
///
/// ```rust
/// use oar_ocr::prelude::*;
/// ```
///
/// Included items focus on the most common tasks:
/// - Builder APIs (`OAROCRBuilder`, `OARStructureBuilder`)
/// - Edge processors (`EdgeProcessorConfig`)
/// - Results (`OAROCRResult`, `TextRegion`)
/// - Essential error and result types (`OCRError`, `OcrResult`)
/// - Basic image loading (`load_image`, `load_images`)
///
/// For advanced customization (model adapters, traits),
/// import directly from the respective modules (e.g., `oar_ocr::models`, `oar_ocr::core::traits`).
pub mod prelude {
    // High-level builder APIs
    pub use crate::oarocr::{
        EdgeProcessorConfig, OAROCR, OAROCRBuilder, OAROCRResult, OARStructure,
        OARStructureBuilder, TextRegion,
    };

    // Error Handling
    pub use oar_ocr_core::core::{OCRError, OcrResult};

    // Image Utilities
    pub use oar_ocr_core::utils::{load_image, load_images};

    // Predictors (high-level API)
    pub use oar_ocr_core::predictors::*;
}
