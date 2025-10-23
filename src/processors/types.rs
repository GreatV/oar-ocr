//! Types used in image processing operations
//!
//! This module defines various enums that represent different options and configurations
//! for image processing operations in the OCR pipeline.
use std::str::FromStr;

use crate::core::errors::ImageProcessError;

/// Specifies how to crop an image when the aspect ratios don't match
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CropMode {
    /// Crop from the center of the image
    Center,
    /// Crop from the top-left corner of the image
    TopLeft,
    /// Crop from the top-right corner of the image
    TopRight,
    /// Crop from the bottom-left corner of the image
    BottomLeft,
    /// Crop from the bottom-right corner of the image
    BottomRight,
    /// Crop from custom coordinates
    Custom { x: u32, y: u32 },
}

/// Implementation of FromStr trait for CropMode to parse crop mode from string
impl FromStr for CropMode {
    type Err = ImageProcessError;

    /// Parses a string into a CropMode variant
    ///
    /// # Arguments
    /// * `mode` - A string slice that should contain either "C" for Center or "TL" for TopLeft
    ///
    /// # Returns
    /// * `Ok(CropMode)` - If the string matches a valid crop mode
    /// * `Err(ImageProcessError::UnsupportedMode)` - If the string doesn't match any valid crop mode
    fn from_str(mode: &str) -> Result<Self, Self::Err> {
        match mode {
            "C" => Ok(CropMode::Center),
            "TL" => Ok(CropMode::TopLeft),
            _ => Err(ImageProcessError::UnsupportedMode),
        }
    }
}

/// Specifies how to limit the size of an image during resizing operations
#[derive(Debug, Clone, PartialEq, serde::Serialize, serde::Deserialize)]
pub enum LimitType {
    /// Limit the smaller dimension of the image
    Min,
    /// Limit the larger dimension of the image
    Max,
    /// Resize the long dimension to a specific size while maintaining aspect ratio
    ResizeLong,
}

/// Specifies the order of channels in an image tensor
#[derive(Debug, Clone)]
pub enum ChannelOrder {
    /// Channel, Height, Width order (common in PyTorch)
    CHW,
    /// Height, Width, Channel order (common in TensorFlow)
    HWC,
}

/// Specifies the type of bounding box used for text detection
#[derive(Debug, Clone, Copy)]
pub enum BoxType {
    /// Quadrilateral bounding box (4 points)
    Quad,
    /// Polygonal bounding box (variable number of points)
    Poly,
}

/// Specifies the mode for calculating scores in text detection/recognition
#[derive(Debug, Clone, Copy)]
pub enum ScoreMode {
    /// Fast scoring algorithm (less accurate but faster)
    Fast,
    /// Slow scoring algorithm (more accurate but slower)
    Slow,
}

/// Specifies different strategies for resizing images
#[derive(Debug)]
pub enum ResizeType {
    /// Type 0 resize (implementation specific)
    Type0,
    /// Type 1 resize with specific image shape and ratio preservation option
    Type1 {
        /// Target image shape (height, width)
        image_shape: (u32, u32),
        /// Whether to maintain the aspect ratio of the original image
        keep_ratio: bool,
    },
    /// Type 2 resize that resizes the long dimension to a specific size
    Type2 {
        /// Target size for the long dimension
        resize_long: u32,
    },
    /// Type 3 resize to a specific input shape
    Type3 {
        /// Target input shape (channels, height, width)
        input_shape: (u32, u32, u32),
    },
}
