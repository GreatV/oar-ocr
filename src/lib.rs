//! # OAR OCR
//!
//! A Rust OCR library that extracts text from document images using ONNX models.
//! Supports text detection, recognition, document orientation, and rectification.
//!
//! ## Features
//!
//! - Complete OCR pipeline from image to text
//! - Task graph architecture for flexible pipeline configuration
//! - Model adapter system for easy model swapping
//! - Edge processors for data transformation between tasks
//! - Batch processing support
//! - ONNX Runtime integration for fast inference
//!
//! ## Components
//!
//! - **Text Detection**: Find text regions in images
//! - **Text Recognition**: Convert text regions to readable text
//! - **Document Orientation**: Detect document rotation (0°, 90°, 180°, 270°)
//! - **Document Rectification**: Fix perspective distortion
//! - **Text Line Classification**: Detect text line orientation
//!
//! ## Modules
//!
//! * [`core`] - Core traits, error handling, and batch processing
//! * [`domain`] - Domain types like orientation helpers and prediction models
//! * [`models`] - Model adapters for different OCR tasks
//! * [`pipeline`] - Task graph-based OCR pipeline
//! * [`processors`] - Image processing utilities
//! * [`utils`] - Utility functions for images and tensors
//!
//! ## Quick Start
//!
//! ### Task Graph-Based OCR Pipeline
//!
//! ```rust,no_run
//! use oar_ocr::prelude::*;
//! use oar_ocr::core::traits::TaskType;
//! use std::path::Path;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Create task graph configuration
//! let config = TaskGraphConfig::new()
//!     .add_model_binding("detection", ModelBinding::new(
//!         "DB",
//!         "models/detection.onnx",
//!         TaskType::TextDetection,
//!     ))
//!     .add_model_binding("recognition", ModelBinding::new(
//!         "CRNN",
//!         "models/recognition.onnx",
//!         TaskType::TextRecognition,
//!     ))
//!     .add_task_node(TaskNode::new(
//!         "text_detection",
//!         TaskType::TextDetection,
//!         "detection",
//!     ))
//!     .add_task_node(TaskNode::new(
//!         "text_recognition",
//!         TaskType::TextRecognition,
//!         "recognition",
//!     )
//!     .with_dependency("text_detection")
//!     .with_edge_processor(
//!         "text_detection",
//!         EdgeProcessorConfig::TextCropping { handle_rotation: true }
//!     ))
//!     .with_character_dict("models/dict.txt");
//!
//! // Build and execute pipeline
//! let builder = TaskGraphBuilder::new(config);
//! let adapters = builder.build_adapters()?;
//!
//! // Process images
//! let image = load_image(Path::new("document.jpg"))?;
//! // Execute tasks using adapters...
//! # Ok(())
//! # }
//! ```
//!
//! ### JSON Configuration
//!
//! ```rust,no_run
//! use oar_ocr::prelude::*;
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! // Load configuration from JSON
//! let config: TaskGraphConfig = serde_json::from_str(r#"
//! {
//!   "model_bindings": {
//!     "detection": {
//!       "model_name": "DB",
//!       "model_path": "models/detection.onnx",
//!       "task_type": "TextDetection"
//!     },
//!     "recognition": {
//!       "model_name": "CRNN",
//!       "model_path": "models/recognition.onnx",
//!       "task_type": "TextRecognition"
//!     }
//!   },
//!   "task_nodes": [
//!     {
//!       "id": "text_detection",
//!       "task_type": "TextDetection",
//!       "model_binding": "detection"
//!     },
//!     {
//!       "id": "text_recognition",
//!       "task_type": "TextRecognition",
//!       "model_binding": "recognition",
//!       "dependencies": ["text_detection"],
//!       "edge_processors": {
//!         "text_detection": {
//!           "type": "TextCropping",
//!           "handle_rotation": true
//!         }
//!       }
//!     }
//!   ]
//! }
//! "#)?;
//!
//! let builder = TaskGraphBuilder::new(config);
//! # Ok(())
//! # }
//! ```

// Core modules
pub mod core;
pub mod domain;
pub mod models;

pub mod pipeline;
pub mod processors;
pub mod utils;

/// Prelude module for convenient imports.
///
/// Bring the essentials into scope with a single use statement:
///
/// ```rust
/// use oar_ocr::prelude::*;
/// ```
///
/// Included items focus on the most common tasks:
/// - Task graph pipeline (`TaskGraphBuilder`, `TaskGraphConfig`, `TaskNode`, `ModelBinding`)
/// - Edge processors (`EdgeProcessorConfig`)
/// - Results (`OAROCRResult`, `TextRegion`)
/// - Essential error and result types (`OCRError`, `OcrResult`)
/// - Basic image loading (`load_image`)
///
/// For advanced customization (model adapters, traits, validation),
/// import directly from the respective modules (e.g., `oar_ocr::models`, `oar_ocr::core::traits`,
/// `oar_ocr::pipeline`).
pub mod prelude {
    // Task Graph Pipeline (essential)
    pub use crate::pipeline::{
        EdgeProcessorConfig, ModelBinding, OAROCRResult, TaskGraphBuilder, TaskGraphConfig,
        TaskNode, TextRegion,
    };

    // Error Handling (essential)
    pub use crate::core::{OCRError, OcrResult};

    // Image Utility (minimal)
    pub use crate::utils::{load_image, load_images};
}
