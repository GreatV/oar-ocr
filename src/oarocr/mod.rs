//! The OCR pipeline module.
//!
//! This module provides high-level builder APIs for constructing OCR and document
//! structure analysis pipelines. The builders simplify the process of configuring
//! text detection, recognition, and optional preprocessing components.
//!
//! # Main APIs
//!
//! - [`OAROCRBuilder`] - For text detection and recognition
//! - [`OARStructureBuilder`] - For document structure analysis (layout, tables, formulas)

pub mod ocr;
pub(crate) mod preprocess;
pub mod processors;
pub mod result;
pub mod stitching;
pub mod structure;
pub(crate) mod table_analyzer;

pub use ocr::*;
pub use processors::*;
pub use result::*;
pub use stitching::*;
pub use structure::*;
