//! Configuration types for dynamic batching

use serde::{Deserialize, Serialize};

/// Strategy for determining shape compatibility
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum ShapeCompatibilityStrategy {
    /// Exact shape matching - images must have identical dimensions
    Exact,
    /// Aspect ratio bucketing - images are grouped by aspect ratio ranges
    AspectRatio {
        /// Tolerance for aspect ratio matching (e.g., 0.1 means ±10%)
        tolerance: f32,
    },
    /// Maximum dimension bucketing - images are grouped by maximum dimension ranges
    MaxDimension {
        /// Size of each dimension bucket
        bucket_size: u32,
    },
    /// Custom bucketing with predefined dimension targets
    Custom {
        /// List of target dimensions (height, width)
        targets: Vec<(u32, u32)>,
        /// Tolerance for matching to targets
        tolerance: f32,
    },
}

impl Default for ShapeCompatibilityStrategy {
    fn default() -> Self {
        Self::AspectRatio { tolerance: 0.1 }
    }
}

/// Strategy for padding images to uniform size
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum PaddingStrategy {
    /// Zero padding (fill with zeros)
    Zero,
    /// Center padding (center image in padded area)
    Center {
        /// RGB color to use for padding
        fill_color: [u8; 3],
    },
    /// Edge padding (repeat edge pixels)
    Edge,
    /// Smart padding (content-aware padding)
    Smart,
}

impl Default for PaddingStrategy {
    fn default() -> Self {
        Self::Center {
            fill_color: [0, 0, 0],
        }
    }
}

/// Configuration for dynamic batching
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DynamicBatchConfig {
    /// Maximum batch size for detection
    pub max_detection_batch_size: usize,
    /// Maximum batch size for recognition
    pub max_recognition_batch_size: usize,
    /// Minimum batch size (smaller batches are processed individually)
    pub min_batch_size: usize,
    /// Shape compatibility strategy
    pub shape_compatibility: ShapeCompatibilityStrategy,
    /// Padding strategy for uniform batch sizes
    pub padding_strategy: PaddingStrategy,
}

impl Default for DynamicBatchConfig {
    fn default() -> Self {
        Self {
            max_detection_batch_size: 8,
            max_recognition_batch_size: 16,
            min_batch_size: 2,
            shape_compatibility: ShapeCompatibilityStrategy::default(),
            padding_strategy: PaddingStrategy::default(),
        }
    }
}

impl DynamicBatchConfig {
    /// Creates a new configuration with default values
    pub fn new() -> Self {
        Self::default()
    }

    /// Sets the maximum detection batch size
    pub fn with_max_detection_batch_size(mut self, size: usize) -> Self {
        self.max_detection_batch_size = size;
        self
    }

    /// Sets the maximum recognition batch size
    pub fn with_max_recognition_batch_size(mut self, size: usize) -> Self {
        self.max_recognition_batch_size = size;
        self
    }

    /// Sets the minimum batch size
    pub fn with_min_batch_size(mut self, size: usize) -> Self {
        self.min_batch_size = size;
        self
    }

    /// Sets the shape compatibility strategy
    pub fn with_shape_compatibility(mut self, strategy: ShapeCompatibilityStrategy) -> Self {
        self.shape_compatibility = strategy;
        self
    }

    /// Sets the padding strategy
    pub fn with_padding_strategy(mut self, strategy: PaddingStrategy) -> Self {
        self.padding_strategy = strategy;
        self
    }
}
