//! Dynamic batching for OCR pipeline components.
//!
//! This module provides functionality for dynamically batching images based on
//! shape compatibility and performance requirements.
//! It supports both same-image batching (multiple images) and cross-image
//! batching (text regions from multiple images).
//!
//! # Features
//!
//! - **Shape Compatibility**: Group images by exact dimensions, aspect ratio, or custom strategies
//! - **Flexible Batching**: Support for detection and recognition batching
//! - **Performance Metrics**: Track batching performance and efficiency
//! - **Cross-Image Batching**: Batch text regions from multiple images together
//!
//! # Example
//!
//! ```rust,no_run
//! use oar_ocr::core::{DynamicBatchConfig, DefaultDynamicBatcher, DynamicBatcher};
//! use image::RgbImage;
//!
//! let config = DynamicBatchConfig::default();
//! let batcher = DefaultDynamicBatcher::new();
//! let images = vec![(0, RgbImage::new(100, 100)), (1, RgbImage::new(100, 100))];
//!
//! let batches = batcher.group_images_by_compatibility(images, &config).unwrap();
//! ```

mod config;
mod processor;
mod types;

// Re-export public types
pub use config::{DynamicBatchConfig, PaddingStrategy, ShapeCompatibilityStrategy};
pub use processor::{DefaultDynamicBatcher, DynamicBatcher};
pub use types::{
    BatchPerformanceMetrics, CompatibleBatch, CrossImageBatch, CrossImageItem, DynamicBatchResult,
};
