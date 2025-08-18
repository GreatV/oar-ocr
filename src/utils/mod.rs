//! Utility functions for the OCR pipeline.
//!
//! This module provides various utility functions used throughout the OCR pipeline,
//! including image processing utilities, tensor conversion functions, and logging setup.

pub mod image;
pub mod tensor;
pub mod transform;
#[cfg(feature = "visualization")]
pub mod visualization;

// Re-export image processing functions
pub use image::{
    OCRResizePadConfig, PaddingStrategy, ResizePadConfig, create_rgb_image, dynamic_to_gray,
    dynamic_to_rgb, load_image, load_images, load_images_batch_with_policy,
    load_images_batch_with_threshold, ocr_resize_and_pad, resize_and_pad, resize_images_batch,
    resize_images_batch_to_dynamic,
};

// Re-export tensor utility functions
pub use tensor::*;

// Re-export transform utility functions
pub use transform::{Point2f, get_rotate_crop_image};

/// Initializes the tracing subscriber for logging.
///
/// This function sets up the tracing subscriber with environment filter and formatting layer.
/// It's typically called at the start of an application to enable logging.
pub fn init_tracing() {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    tracing_subscriber::registry()
        .with(tracing_subscriber::EnvFilter::from_default_env())
        .with(tracing_subscriber::fmt::layer())
        .init();
}
