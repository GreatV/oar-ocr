//! Utility functions for the OCR pipeline.
//!
//! This module provides various utility functions used throughout the OCR pipeline,
//! including image processing utilities and tensor conversion functions.

pub mod image;
pub mod tensor;
pub mod transform;
#[cfg(feature = "visualization")]
pub mod visualization;

// Re-export image processing functions
pub use image::{
    create_rgb_image, dynamic_to_gray, dynamic_to_rgb, load_image, load_images_batch,
    load_images_batch_with_threshold,
};

// Re-export tensor utility functions
pub use tensor::*;

// Re-export transform utility functions
pub use transform::{Point2f, get_rotate_crop_image};
