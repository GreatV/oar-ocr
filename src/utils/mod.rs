//! Utility functions for the OCR pipeline.
//!
//! This module provides various utility functions used throughout the OCR pipeline,
//! including image processing helpers, tensor conversion functions, cropping helpers,
//! classification utilities, and logging setup.

pub mod bbox_crop;
pub mod cow;
pub mod crop;
pub mod image;
pub mod tensor;
pub mod topk;
pub mod transform;
pub mod validation;
#[cfg(feature = "visualization")]
pub mod visualization;

// Re-export image processing functions
pub use image::{
    OCRResizePadConfig, PaddingStrategy, ResizePadConfig, calculate_center_crop_coords,
    check_image_size, create_rgb_image, dynamic_to_gray, dynamic_to_rgb, load_image, load_images,
    load_images_batch_with_policy, load_images_batch_with_threshold, mask_region, mask_regions,
    ocr_resize_and_pad, pad_image, resize_and_pad, resize_gray_image, resize_image,
    resize_images_batch, resize_images_batch_to_dynamic, rgb_to_grayscale, slice_gray_image,
    slice_image, validate_crop_bounds,
};

// Re-export tensor utility functions
pub use tensor::*;

// Re-export transform utility functions
pub use transform::{Point2f, get_rotate_crop_image};

// Re-export shared processors-style utilities
pub use bbox_crop::BBoxCrop;
pub use crop::Crop;
pub use topk::{Topk, TopkResult};

// Re-export validation utilities
pub use validation::{
    ScoreValidator, validate_length_match, validate_max_value, validate_positive_dimensions,
};

/// Initializes the tracing subscriber for logging.
///
/// This function sets up the tracing subscriber with environment filter and formatting layer.
/// It's typically called at the start of an application to enable logging.
pub fn init_tracing() {
    use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

    // Default to info-level logging if RUST_LOG is not configured by the caller.
    let env_filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));

    tracing_subscriber::registry()
        .with(env_filter)
        .with(tracing_subscriber::fmt::layer())
        .init();
}
