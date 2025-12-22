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
#[serde(rename_all = "lowercase")]
pub enum LimitType {
    /// Limit the smaller dimension of the image
    Min,
    /// Limit the larger dimension of the image
    Max,
    /// Resize the long dimension to a specific size while maintaining aspect ratio
    #[serde(
        rename = "resize_long",
        alias = "resizelong",
        alias = "resizeLong",
        alias = "resize-long"
    )]
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

/// Specifies the color channel order in an image
#[derive(Debug, Clone, Copy, Default)]
pub enum ColorOrder {
    /// Red, Green, Blue order (default for most image libraries like PIL, image-rs)
    #[default]
    RGB,
    /// Blue, Green, Red order (used by OpenCV and PaddlePaddle models)
    BGR,
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

/// Information about image scaling during preprocessing
///
/// This struct captures the original dimensions and scaling ratios applied
/// during image resizing operations. It replaces the opaque `[f32; 4]` tuple
/// format with self-documenting named fields.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ImageScaleInfo {
    /// Original image height before resizing
    pub src_h: f32,
    /// Original image width before resizing
    pub src_w: f32,
    /// Height scaling ratio (resized_height / original_height)
    pub ratio_h: f32,
    /// Width scaling ratio (resized_width / original_width)
    pub ratio_w: f32,
}

impl ImageScaleInfo {
    /// Creates a new `ImageScaleInfo` from original dimensions and ratios
    pub fn new(src_h: f32, src_w: f32, ratio_h: f32, ratio_w: f32) -> Self {
        Self {
            src_h,
            src_w,
            ratio_h,
            ratio_w,
        }
    }

    /// Creates an `ImageScaleInfo` from a `[f32; 4]` array
    ///
    /// This is provided for backward compatibility during migration.
    /// The array format is `[src_h, src_w, ratio_h, ratio_w]`.
    pub fn from_array(arr: [f32; 4]) -> Self {
        Self {
            src_h: arr[0],
            src_w: arr[1],
            ratio_h: arr[2],
            ratio_w: arr[3],
        }
    }

    /// Converts this `ImageScaleInfo` to a `[f32; 4]` array
    ///
    /// This is provided for backward compatibility during migration.
    /// The array format is `[src_h, src_w, ratio_h, ratio_w]`.
    pub fn to_array(&self) -> [f32; 4] {
        [self.src_h, self.src_w, self.ratio_h, self.ratio_w]
    }
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
