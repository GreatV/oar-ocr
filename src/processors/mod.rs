//! Image processing utilities for OCR systems.
//!
//! This module provides a collection of image processing functions and utilities
//! specifically designed for OCR (Optical Character Recognition) systems. It includes
//! functionality for image resizing, normalization, geometric operations, text decoding,
//! and post-processing of OCR results.
//!
//! # Modules
//!
//! * `aspect_ratio_bucketing` - Aspect ratio bucketing for efficient batch processing
//! * `decode` - Text decoding utilities for converting model predictions to readable text
//! * `geometry` - Geometric primitives and algorithms for OCR processing
//! * `normalization` - Image normalization utilities for preparing images for OCR models
//! * `db_postprocess` - DB detection post-processing
//! * `uvdoc_postprocess` - UVDoc rectification post-processing
//! * `formula_preprocess` - Formula recognition preprocessing
//! * `layout_postprocess` - Layout detection post-processing
//! * `resize_detection` - Resizing for detection models
//! * `resize_recognition` - Resizing for recognition models
//! * `types` - Type definitions used across the processors module

mod aspect_ratio_bucketing;
pub mod db_postprocess;
mod decode;
pub mod formula_preprocess;
mod geometry;
pub mod layout_postprocess;
pub mod layout_sorting;
pub mod layout_utils;
mod normalization;
pub mod resize_detection;
pub mod resize_recognition;
mod sorting;
pub mod table_structure_decode;
pub mod types;
pub mod unimernet_preprocess;
pub mod uvdoc_postprocess;

pub use crate::utils::{Crop, Topk, TopkResult};
pub use aspect_ratio_bucketing::*;
pub use db_postprocess::*;
pub use decode::*;
pub use formula_preprocess::{FormulaPreprocessParams, FormulaPreprocessor, normalize_latex};
pub use geometry::*;
pub use layout_postprocess::*;
pub use layout_utils::{
    LayoutElement, LayoutOCRAssociation, OverlapRemovalResult, associate_ocr_with_layout,
    get_overlap_boxes_idx, get_overlap_removal_indices, reconcile_table_cells,
    remove_overlap_blocks, reprocess_table_cells_with_ocr, sort_layout_boxes,
};
pub use normalization::*;
pub use resize_detection::*;
pub use resize_recognition::*;
pub use sorting::{
    RegionBlock, SortDirection, assign_elements_to_regions, calculate_iou, calculate_overlap_ratio,
    sort_boxes_xycut, sort_by_xycut, sort_elements_with_regions, sort_poly_boxes, sort_quad_boxes,
    sort_regions, sort_with_region_hierarchy,
};
pub use table_structure_decode::{
    CellGridInfo, TableStructureDecode, TableStructureDecodeOutput, parse_cell_grid_info,
    wrap_table_html, wrap_table_html_with_content,
};
pub use types::*;
pub use unimernet_preprocess::{UniMERNetPreprocessParams, UniMERNetPreprocessor};
pub use uvdoc_postprocess::*;
