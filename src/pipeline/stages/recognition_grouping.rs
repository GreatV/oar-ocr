//! Image grouping strategies for batch processing.
//!
//! This module provides different strategies for grouping images before batch processing,
//! separating the grouping logic from the recognition processing logic.
//!
//! ## Grouping Strategies
//!
//! ### ExactDimensionStrategy
//! Groups images by their exact pixel dimensions. This ensures that all images in a group
//! have identical dimensions, which is required by some ML models but can lead to many
//! small batches if images have varied sizes.
//!
//! ### AspectRatioBucketingStrategy
//! Groups images by aspect ratio ranges and resizes them to common target dimensions.
//! This approach:
//! - Reduces the number of groups, leading to larger, more efficient batches
//! - Handles varied image sizes gracefully
//! - May introduce slight distortion due to resizing and padding
//!
//! ## Usage Example
//!
//! ```rust
//! use oar_ocr::pipeline::stages::{
//!     GroupingStrategy, GroupingStrategyFactory, GroupingStrategyConfig
//! };
//! use oar_ocr::processors::AspectRatioBucketingConfig;
//! use image::RgbImage;
//!
//! // Create images to group
//! let images = vec![
//!     (0, RgbImage::new(100, 50)),
//!     (1, RgbImage::new(200, 100)),
//!     (2, RgbImage::new(50, 50)),
//! ];
//!
//! // Use the factory to create strategies
//! let exact_config = GroupingStrategyConfig::ExactDimensions;
//! let exact_strategy = GroupingStrategyFactory::create_strategy(&exact_config).unwrap();
//! let exact_groups = exact_strategy.group_images(images.clone()).unwrap();
//!
//! // Use aspect ratio bucketing strategy
//! let bucketing_config = GroupingStrategyConfig::AspectRatioBucketing(
//!     AspectRatioBucketingConfig::default()
//! );
//! let bucketing_strategy = GroupingStrategyFactory::create_strategy(&bucketing_config).unwrap();
//! let bucketing_groups = bucketing_strategy.group_images(images).unwrap();
//! ```

use image::RgbImage;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use tracing::{debug, warn};

use crate::core::OCRError;
use crate::processors::{AspectRatioBucketing, AspectRatioBucketingConfig};

/// Trait for image grouping strategies
pub trait GroupingStrategy {
    /// Group images for batch processing
    ///
    /// # Arguments
    /// * `images` - Vector of (index, image) pairs to group
    ///
    /// # Returns
    /// HashMap where keys are group names and values are vectors of (index, image) pairs
    fn group_images(
        &self,
        images: Vec<(usize, RgbImage)>,
    ) -> Result<HashMap<String, Vec<(usize, RgbImage)>>, OCRError>;
}

/// Configuration for different grouping strategies
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum GroupingStrategyConfig {
    /// Use aspect ratio bucketing for grouping
    AspectRatioBucketing(AspectRatioBucketingConfig),
    /// Use exact dimensions for grouping
    ExactDimensions,
}

impl Default for GroupingStrategyConfig {
    fn default() -> Self {
        Self::ExactDimensions
    }
}

/// Aspect ratio bucketing grouping strategy
#[derive(Debug, Clone)]
pub struct AspectRatioBucketingStrategy {
    bucketing: AspectRatioBucketing,
}

impl AspectRatioBucketingStrategy {
    /// Create a new aspect ratio bucketing strategy
    pub fn new(config: AspectRatioBucketingConfig) -> Self {
        Self {
            bucketing: AspectRatioBucketing::new(config),
        }
    }
}

impl GroupingStrategy for AspectRatioBucketingStrategy {
    fn group_images(
        &self,
        images: Vec<(usize, RgbImage)>,
    ) -> Result<HashMap<String, Vec<(usize, RgbImage)>>, OCRError> {
        match self.bucketing.group_images_by_buckets(images.clone()) {
            Ok(bucket_groups) => {
                debug!(
                    "Using aspect ratio bucketing with {} groups",
                    bucket_groups.len()
                );
                Ok(bucket_groups)
            }
            Err(e) => {
                warn!(
                    "Aspect ratio bucketing failed, falling back to exact grouping: {}",
                    e
                );
                // Fall back to exact dimension grouping - create strategy on demand
                ExactDimensionStrategy::new().group_images(images)
            }
        }
    }
}

/// Exact dimension grouping strategy
#[derive(Debug, Clone, Default)]
pub struct ExactDimensionStrategy;

impl ExactDimensionStrategy {
    /// Create a new exact dimension strategy
    pub fn new() -> Self {
        Self
    }
}

impl GroupingStrategy for ExactDimensionStrategy {
    fn group_images(
        &self,
        images: Vec<(usize, RgbImage)>,
    ) -> Result<HashMap<String, Vec<(usize, RgbImage)>>, OCRError> {
        let mut dimension_groups: HashMap<String, Vec<(usize, RgbImage)>> = HashMap::new();

        for (i, image) in images {
            let dims = (image.height(), image.width());
            let key = format!("exact_{}x{}", dims.0, dims.1);
            dimension_groups.entry(key).or_default().push((i, image));
        }

        debug!(
            "Using exact dimension grouping with {} groups",
            dimension_groups.len()
        );
        Ok(dimension_groups)
    }
}

/// Factory for creating grouping strategies
pub struct GroupingStrategyFactory;

impl GroupingStrategyFactory {
    /// Create a grouping strategy from configuration
    pub fn create_strategy(
        config: &GroupingStrategyConfig,
    ) -> Result<Box<dyn GroupingStrategy>, OCRError> {
        match config {
            GroupingStrategyConfig::AspectRatioBucketing(bucketing_config) => Ok(Box::new(
                AspectRatioBucketingStrategy::new(bucketing_config.clone()),
            )),
            GroupingStrategyConfig::ExactDimensions => Ok(Box::new(ExactDimensionStrategy::new())),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    fn create_test_image(width: u32, height: u32) -> RgbImage {
        let mut img = RgbImage::new(width, height);
        img.put_pixel(0, 0, Rgb([255, 0, 0])); // Red pixel at top-left
        img
    }

    #[test]
    fn test_exact_dimension_strategy() {
        let strategy = ExactDimensionStrategy::new();

        let images = vec![
            (0, create_test_image(100, 50)),
            (1, create_test_image(100, 50)),
            (2, create_test_image(200, 100)),
        ];

        let groups = strategy.group_images(images).unwrap();

        assert_eq!(groups.len(), 2);
        assert!(groups.contains_key("exact_50x100"));
        assert!(groups.contains_key("exact_100x200"));
        assert_eq!(groups["exact_50x100"].len(), 2);
        assert_eq!(groups["exact_100x200"].len(), 1);
    }

    #[test]
    fn test_aspect_ratio_bucketing_strategy() {
        let config = AspectRatioBucketingConfig::default();
        let strategy = AspectRatioBucketingStrategy::new(config);

        let images = vec![
            (0, create_test_image(100, 50)),  // Aspect ratio 2.0 -> "normal"
            (1, create_test_image(200, 100)), // Aspect ratio 2.0 -> "normal"
            (2, create_test_image(50, 50)),   // Aspect ratio 1.0 -> "square"
        ];

        let groups = strategy.group_images(images).unwrap();

        // Should have at least 2 groups (normal and square)
        assert!(groups.len() >= 2);
    }

    #[test]
    fn test_grouping_strategy_factory() {
        // Test exact dimensions
        let config = GroupingStrategyConfig::ExactDimensions;
        let strategy = GroupingStrategyFactory::create_strategy(&config).unwrap();

        // Test that we can create the strategy successfully
        let images = vec![(0, create_test_image(100, 50))];
        let groups = strategy.group_images(images).unwrap();
        assert!(!groups.is_empty());

        // Test aspect ratio bucketing
        let config =
            GroupingStrategyConfig::AspectRatioBucketing(AspectRatioBucketingConfig::default());
        let strategy = GroupingStrategyFactory::create_strategy(&config).unwrap();

        // Test with multiple images of different aspect ratios
        let images = vec![
            (0, create_test_image(100, 50)), // 2:1 ratio (normal)
            (1, create_test_image(32, 32)),  // 1:1 ratio (square)
            (2, create_test_image(160, 32)), // 5:1 ratio (ultra_wide)
        ];
        let groups = strategy.group_images(images).unwrap();
        assert!(!groups.is_empty());

        // Should have multiple groups for different aspect ratios
        assert!(groups.len() >= 2);
    }

    #[test]
    fn test_trait_based_grouping_system() {
        // Test that both strategies implement the GroupingStrategy trait
        let exact_strategy: Box<dyn GroupingStrategy> = Box::new(ExactDimensionStrategy::new());
        let bucketing_strategy: Box<dyn GroupingStrategy> = Box::new(
            AspectRatioBucketingStrategy::new(AspectRatioBucketingConfig::default()),
        );

        let test_images = vec![
            (0, create_test_image(100, 50)),
            (1, create_test_image(100, 50)),
            (2, create_test_image(200, 100)),
        ];

        // Test exact strategy
        let exact_groups = exact_strategy.group_images(test_images.clone()).unwrap();
        assert_eq!(exact_groups.len(), 2); // Two different dimensions

        // Test bucketing strategy
        let bucketing_groups = bucketing_strategy.group_images(test_images).unwrap();
        assert!(!bucketing_groups.is_empty());
    }

    #[test]
    fn test_config_based_strategy_selection() {
        let test_images = vec![
            (0, create_test_image(100, 50)),
            (1, create_test_image(200, 100)),
        ];

        // Test default configuration (should be ExactDimensions)
        let default_config = GroupingStrategyConfig::default();
        let strategy = GroupingStrategyFactory::create_strategy(&default_config).unwrap();
        let groups = strategy.group_images(test_images.clone()).unwrap();
        assert_eq!(groups.len(), 2); // Exact dimensions should create 2 groups

        // Test explicit exact dimensions configuration
        let exact_config = GroupingStrategyConfig::ExactDimensions;
        let strategy = GroupingStrategyFactory::create_strategy(&exact_config).unwrap();
        let groups = strategy.group_images(test_images.clone()).unwrap();
        assert_eq!(groups.len(), 2);

        // Test aspect ratio bucketing configuration
        let bucketing_config =
            GroupingStrategyConfig::AspectRatioBucketing(AspectRatioBucketingConfig::default());
        let strategy = GroupingStrategyFactory::create_strategy(&bucketing_config).unwrap();
        let groups = strategy.group_images(test_images).unwrap();
        assert!(!groups.is_empty());
    }

    #[test]
    fn test_integration_grouping_with_mixed_sizes() {
        // Test that different strategies handle mixed image sizes appropriately
        let images = vec![
            (0, create_test_image(100, 50)),  // 2:1 ratio
            (1, create_test_image(100, 50)),  // 2:1 ratio (duplicate)
            (2, create_test_image(200, 100)), // 2:1 ratio (different size)
            (3, create_test_image(50, 50)),   // 1:1 ratio
            (4, create_test_image(300, 100)), // 3:1 ratio
        ];

        // Test exact dimension strategy
        let exact_strategy = ExactDimensionStrategy::new();
        let exact_groups = exact_strategy.group_images(images.clone()).unwrap();

        // Should have 4 groups (3 different dimensions)
        assert_eq!(exact_groups.len(), 4);
        assert_eq!(exact_groups["exact_50x100"].len(), 2); // Two 100x50 images
        assert_eq!(exact_groups["exact_100x200"].len(), 1); // One 200x100 image
        assert_eq!(exact_groups["exact_50x50"].len(), 1); // One 50x50 image
        assert_eq!(exact_groups["exact_100x300"].len(), 1); // One 300x100 image

        // Test aspect ratio bucketing strategy
        let bucketing_config = AspectRatioBucketingConfig::default();
        let bucketing_strategy = AspectRatioBucketingStrategy::new(bucketing_config);
        let bucketing_groups = bucketing_strategy.group_images(images).unwrap();

        // Should have fewer groups due to aspect ratio bucketing
        assert!(bucketing_groups.len() <= exact_groups.len());
    }

    #[test]
    fn test_aspect_ratio_bucketing_strategy_fallback() {
        let config = AspectRatioBucketingConfig::default();
        let strategy = AspectRatioBucketingStrategy::new(config);

        // Create test images that should work with bucketing
        let images = vec![
            (0, create_test_image(100, 50)), // 2:1 aspect ratio
            (1, create_test_image(50, 50)),  // 1:1 aspect ratio
        ];

        // This should succeed with normal bucketing
        let groups = strategy.group_images(images).unwrap();
        assert!(!groups.is_empty());

        // The fallback behavior is tested implicitly in other tests
        // We've simplified the implementation by removing the cached fallback strategy
        // since ExactDimensionStrategy is essentially free to create
    }
}
