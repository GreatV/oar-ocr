//! Model input configuration for preprocessing.
//!
//! This module provides configuration types for model input shapes and preprocessing parameters.
//! Input shapes can be parsed from ONNX models, where -1 indicates dynamic dimensions.
//!
//! # Input Shape Representation
//!
//! ONNX models define input shapes as `[batch, channels, height, width]` where:
//! - Positive values indicate fixed dimensions
//! - -1 (or 0) indicates dynamic dimensions
//!
//! Examples:
//! - `[1, 3, 512, 512]` - Fixed batch=1, channels=3, height=512, width=512
//! - `[-1, 3, 512, 512]` - Dynamic batch, fixed spatial dimensions
//! - `[1, 3, -1, -1]` - Fixed batch/channels, dynamic spatial dimensions
//! - `[-1, -1, -1, -1]` - Fully dynamic

/// Represents a dimension that can be fixed or dynamic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dim {
    /// Fixed dimension with a specific value
    Fixed(i64),
    /// Dynamic dimension (represented as -1 in ONNX)
    Dynamic,
}

impl Dim {
    /// Returns true if this dimension is dynamic.
    pub fn is_dynamic(&self) -> bool {
        matches!(self, Dim::Dynamic)
    }

    /// Returns true if this dimension is fixed.
    pub fn is_fixed(&self) -> bool {
        matches!(self, Dim::Fixed(_))
    }

    /// Returns the fixed value if this dimension is fixed, None otherwise.
    pub fn value(&self) -> Option<i64> {
        match self {
            Dim::Fixed(v) => Some(*v),
            Dim::Dynamic => None,
        }
    }

    /// Returns the fixed value or a default if dynamic.
    pub fn value_or(&self, default: i64) -> i64 {
        self.value().unwrap_or(default)
    }
}

impl From<i64> for Dim {
    fn from(value: i64) -> Self {
        if value < 0 {
            Dim::Dynamic
        } else {
            Dim::Fixed(value)
        }
    }
}

impl std::fmt::Display for Dim {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Dim::Fixed(v) => write!(f, "{}", v),
            Dim::Dynamic => write!(f, "-1"),
        }
    }
}

/// Input shape specification for an ONNX model.
/// Represents `[batch, channels, height, width]` with support for dynamic dimensions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InputShape {
    /// Batch dimension
    pub batch: Dim,
    /// Channel dimension
    pub channels: Dim,
    /// Height dimension
    pub height: Dim,
    /// Width dimension
    pub width: Dim,
}

impl InputShape {
    /// Creates a new input shape from individual dimensions.
    pub fn new(batch: Dim, channels: Dim, height: Dim, width: Dim) -> Self {
        Self {
            batch,
            channels,
            height,
            width,
        }
    }

    /// Creates an input shape from an array of i64 values.
    /// Values <= 0 are treated as dynamic dimensions.
    pub fn from_array(shape: [i64; 4]) -> Self {
        Self {
            batch: shape[0].into(),
            channels: shape[1].into(),
            height: shape[2].into(),
            width: shape[3].into(),
        }
    }

    /// Creates a fully fixed input shape.
    pub fn fixed(batch: i64, channels: i64, height: i64, width: i64) -> Self {
        Self {
            batch: Dim::Fixed(batch),
            channels: Dim::Fixed(channels),
            height: Dim::Fixed(height),
            width: Dim::Fixed(width),
        }
    }

    /// Creates a shape with dynamic batch and fixed spatial dimensions.
    /// Common pattern: `[-1, 3, H, W]`
    pub fn dynamic_batch(channels: i64, height: i64, width: i64) -> Self {
        Self {
            batch: Dim::Dynamic,
            channels: Dim::Fixed(channels),
            height: Dim::Fixed(height),
            width: Dim::Fixed(width),
        }
    }

    /// Creates a shape with fixed batch/channels and dynamic spatial dimensions.
    /// Common pattern: `[1, 3, -1, -1]`
    pub fn dynamic_spatial(batch: i64, channels: i64) -> Self {
        Self {
            batch: Dim::Fixed(batch),
            channels: Dim::Fixed(channels),
            height: Dim::Dynamic,
            width: Dim::Dynamic,
        }
    }

    /// Creates a fully dynamic shape: `[-1, -1, -1, -1]`
    pub fn fully_dynamic() -> Self {
        Self {
            batch: Dim::Dynamic,
            channels: Dim::Dynamic,
            height: Dim::Dynamic,
            width: Dim::Dynamic,
        }
    }

    /// Returns the spatial dimensions as (height, width) with defaults for dynamic dims.
    pub fn spatial_size_or(&self, default_h: u32, default_w: u32) -> (u32, u32) {
        (
            self.height.value_or(default_h as i64) as u32,
            self.width.value_or(default_w as i64) as u32,
        )
    }

    /// Returns whether the spatial dimensions (height, width) are fixed.
    pub fn has_fixed_spatial(&self) -> bool {
        self.height.is_fixed() && self.width.is_fixed()
    }

    /// Returns whether any dimension is dynamic.
    pub fn has_dynamic(&self) -> bool {
        self.batch.is_dynamic()
            || self.channels.is_dynamic()
            || self.height.is_dynamic()
            || self.width.is_dynamic()
    }

    /// Converts to array representation where dynamic = -1.
    pub fn to_array(&self) -> [i64; 4] {
        [
            self.batch.value().unwrap_or(-1),
            self.channels.value().unwrap_or(-1),
            self.height.value().unwrap_or(-1),
            self.width.value().unwrap_or(-1),
        ]
    }

    /// Parses input shape from ONNX model input dimensions.
    ///
    /// Handles various dimension representations:
    /// - Positive values: fixed dimensions
    /// - Negative values or 0: dynamic dimensions
    ///
    /// # Arguments
    /// * `dims` - Dimensions from ONNX model (typically 4 elements: [batch, channels, height, width])
    ///
    /// # Returns
    /// * `Some(InputShape)` if dims has exactly 4 elements
    /// * `None` if dims has wrong number of elements
    pub fn from_onnx_dims(dims: &[i64]) -> Option<Self> {
        if dims.len() != 4 {
            return None;
        }
        Some(Self::from_array([dims[0], dims[1], dims[2], dims[3]]))
    }
}

impl Default for InputShape {
    fn default() -> Self {
        // Default: dynamic batch, 3 channels, dynamic spatial
        Self::dynamic_spatial(1, 3)
    }
}

impl std::fmt::Display for InputShape {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "[{}, {}, {}, {}]",
            self.batch, self.channels, self.height, self.width
        )
    }
}

/// Color channel order for image input.
///
/// Specifies whether the model expects RGB or BGR channel ordering.
/// Most image libraries (PIL, image-rs) use RGB, while OpenCV and
/// PaddlePaddle models typically expect BGR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ColorOrder {
    /// Red, Green, Blue order (default for most image libraries)
    #[default]
    RGB,
    /// Blue, Green, Red order (used by OpenCV/PaddlePaddle models)
    BGR,
}

/// Normalization parameters for image preprocessing.
#[derive(Debug, Clone)]
pub struct NormalizationConfig {
    /// Scale factor applied before normalization (e.g., 1/255)
    pub scale: f32,
    /// Mean values per channel (in the model's expected channel order)
    pub mean: [f32; 3],
    /// Standard deviation values per channel (in the model's expected channel order)
    pub std: [f32; 3],
}

impl NormalizationConfig {
    /// ImageNet normalization in RGB order.
    pub const IMAGENET_RGB: Self = Self {
        scale: 1.0 / 255.0,
        mean: [0.485, 0.456, 0.406],
        std: [0.229, 0.224, 0.225],
    };

    /// ImageNet normalization in BGR order.
    pub const IMAGENET_BGR: Self = Self {
        scale: 1.0 / 255.0,
        mean: [0.406, 0.456, 0.485],
        std: [0.225, 0.224, 0.229],
    };

    /// No normalization (just scale to [0, 1]).
    pub const SCALE_ONLY: Self = Self {
        scale: 1.0 / 255.0,
        mean: [0.0, 0.0, 0.0],
        std: [1.0, 1.0, 1.0],
    };

    /// Creates a new normalization config.
    pub fn new(scale: f32, mean: [f32; 3], std: [f32; 3]) -> Self {
        Self { scale, mean, std }
    }
}

impl Default for NormalizationConfig {
    fn default() -> Self {
        Self::IMAGENET_RGB
    }
}

/// Complete model input configuration.
#[derive(Debug, Clone, Default)]
pub struct ModelInputConfig {
    /// Input shape specification (parsed from ONNX or configured)
    pub input_shape: InputShape,
    /// Expected color channel order
    pub color_order: ColorOrder,
    /// Normalization parameters
    pub normalization: NormalizationConfig,
}

impl ModelInputConfig {
    /// Creates a new model input configuration.
    pub fn new(
        input_shape: InputShape,
        color_order: ColorOrder,
        normalization: NormalizationConfig,
    ) -> Self {
        Self {
            input_shape,
            color_order,
            normalization,
        }
    }

    /// Creates a configuration for fixed-size BGR input with ImageNet normalization.
    pub fn fixed_bgr_imagenet(height: i64, width: i64) -> Self {
        Self {
            input_shape: InputShape::dynamic_batch(3, height, width),
            color_order: ColorOrder::BGR,
            normalization: NormalizationConfig::IMAGENET_BGR,
        }
    }

    /// Creates a configuration for fixed-size RGB input with ImageNet normalization.
    pub fn fixed_rgb_imagenet(height: i64, width: i64) -> Self {
        Self {
            input_shape: InputShape::dynamic_batch(3, height, width),
            color_order: ColorOrder::RGB,
            normalization: NormalizationConfig::IMAGENET_RGB,
        }
    }

    /// Returns the spatial dimensions as (height, width) with defaults for dynamic dims.
    pub fn spatial_size_or(&self, default_h: u32, default_w: u32) -> (u32, u32) {
        self.input_shape.spatial_size_or(default_h, default_w)
    }

    /// Returns whether the model has fixed spatial dimensions.
    pub fn has_fixed_spatial(&self) -> bool {
        self.input_shape.has_fixed_spatial()
    }
}

/// Predefined configurations for known models.
pub mod presets {
    use super::*;

    /// SLANeXt Wired table structure recognition model.
    /// Input: [-1, 3, 512, 512], BGR, ImageNet normalization.
    pub const SLANEXT_WIRED: ModelInputConfig = ModelInputConfig {
        input_shape: InputShape {
            batch: Dim::Dynamic,
            channels: Dim::Fixed(3),
            height: Dim::Fixed(512),
            width: Dim::Fixed(512),
        },
        color_order: ColorOrder::BGR,
        normalization: NormalizationConfig::IMAGENET_BGR,
    };

    /// SLANet Plus (wireless) table structure recognition model.
    /// Input: [-1, 3, 488, 488], BGR, ImageNet normalization.
    pub const SLANET_PLUS: ModelInputConfig = ModelInputConfig {
        input_shape: InputShape {
            batch: Dim::Dynamic,
            channels: Dim::Fixed(3),
            height: Dim::Fixed(488),
            width: Dim::Fixed(488),
        },
        color_order: ColorOrder::BGR,
        normalization: NormalizationConfig::IMAGENET_BGR,
    };

    /// SLANeXt Wireless table structure recognition model.
    /// Input: [-1, 3, 488, 488], BGR, ImageNet normalization.
    pub const SLANEXT_WIRELESS: ModelInputConfig = ModelInputConfig {
        input_shape: InputShape {
            batch: Dim::Dynamic,
            channels: Dim::Fixed(3),
            height: Dim::Fixed(488),
            width: Dim::Fixed(488),
        },
        color_order: ColorOrder::BGR,
        normalization: NormalizationConfig::IMAGENET_BGR,
    };

    /// SLANet (original) table structure recognition model.
    /// Input: [-1, 3, 488, 488], BGR, ImageNet normalization.
    pub const SLANET: ModelInputConfig = ModelInputConfig {
        input_shape: InputShape {
            batch: Dim::Dynamic,
            channels: Dim::Fixed(3),
            height: Dim::Fixed(488),
            width: Dim::Fixed(488),
        },
        color_order: ColorOrder::BGR,
        normalization: NormalizationConfig::IMAGENET_BGR,
    };

    /// PP-LCNet document orientation classification model.
    /// Input: [-1, 3, 224, 224], BGR, ImageNet normalization.
    pub const PP_LCNET_DOC_ORI: ModelInputConfig = ModelInputConfig {
        input_shape: InputShape {
            batch: Dim::Dynamic,
            channels: Dim::Fixed(3),
            height: Dim::Fixed(224),
            width: Dim::Fixed(224),
        },
        color_order: ColorOrder::BGR,
        normalization: NormalizationConfig::IMAGENET_BGR,
    };

    /// PP-LCNet table classification model.
    /// Input: [-1, 3, 224, 224], BGR, ImageNet normalization.
    pub const PP_LCNET_TABLE_CLS: ModelInputConfig = ModelInputConfig {
        input_shape: InputShape {
            batch: Dim::Dynamic,
            channels: Dim::Fixed(3),
            height: Dim::Fixed(224),
            width: Dim::Fixed(224),
        },
        color_order: ColorOrder::BGR,
        normalization: NormalizationConfig::IMAGENET_BGR,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dim_from_i64() {
        assert_eq!(Dim::from(512), Dim::Fixed(512));
        assert_eq!(Dim::from(-1), Dim::Dynamic);
        assert_eq!(Dim::from(0), Dim::Fixed(0));
    }

    #[test]
    fn test_input_shape_from_array() {
        let shape = InputShape::from_array([1, 3, 512, 512]);
        assert_eq!(shape.batch, Dim::Fixed(1));
        assert_eq!(shape.channels, Dim::Fixed(3));
        assert_eq!(shape.height, Dim::Fixed(512));
        assert_eq!(shape.width, Dim::Fixed(512));
        assert!(shape.has_fixed_spatial());
        assert!(!shape.has_dynamic());
    }

    #[test]
    fn test_input_shape_dynamic_batch() {
        let shape = InputShape::from_array([-1, 3, 488, 488]);
        assert!(shape.batch.is_dynamic());
        assert!(shape.has_fixed_spatial());
        assert!(shape.has_dynamic());
        assert_eq!(shape.to_array(), [-1, 3, 488, 488]);
    }

    #[test]
    fn test_input_shape_dynamic_spatial() {
        let shape = InputShape::from_array([1, 3, -1, -1]);
        assert!(shape.batch.is_fixed());
        assert!(!shape.has_fixed_spatial());
        assert_eq!(shape.spatial_size_or(640, 640), (640, 640));
    }

    #[test]
    fn test_input_shape_display() {
        let shape = InputShape::from_array([-1, 3, 512, 512]);
        assert_eq!(format!("{}", shape), "[-1, 3, 512, 512]");
    }

    #[test]
    fn test_normalization_imagenet() {
        let rgb = NormalizationConfig::IMAGENET_RGB;
        let bgr = NormalizationConfig::IMAGENET_BGR;

        // RGB and BGR should have swapped mean/std
        assert_eq!(rgb.mean[0], bgr.mean[2]);
        assert_eq!(rgb.mean[2], bgr.mean[0]);
        assert_eq!(rgb.std[0], bgr.std[2]);
        assert_eq!(rgb.std[2], bgr.std[0]);
    }

    #[test]
    fn test_presets() {
        assert_eq!(presets::SLANEXT_WIRED.spatial_size_or(0, 0), (512, 512));
        assert_eq!(presets::SLANET_PLUS.spatial_size_or(0, 0), (488, 488));
        assert_eq!(presets::SLANEXT_WIRELESS.spatial_size_or(0, 0), (488, 488));
    }
}
