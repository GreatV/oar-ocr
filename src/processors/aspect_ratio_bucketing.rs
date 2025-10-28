//! Aspect Ratio Bucketing for OCR Recognition
//!
//! This module provides functionality for grouping images by aspect ratio ranges
//! instead of exact dimensions, improving batch efficiency in OCR recognition.
//! Images are resized and padded to standardized bucket dimensions.

use crate::core::OCRError;
use crate::utils::{PaddingStrategy, ResizePadConfig, resize_and_pad};
use image::RgbImage;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Aspect ratio bucket definition
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AspectRatioBucket {
    /// Minimum aspect ratio (inclusive)
    pub min_ratio: f32,
    /// Maximum aspect ratio (exclusive, except for the last bucket)
    pub max_ratio: f32,
    /// Target dimensions for this bucket (height, width)
    pub target_dims: (u32, u32),
    /// Bucket identifier
    pub name: String,
}

/// Configuration for aspect ratio bucketing
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AspectRatioBucketingConfig {
    /// List of aspect ratio buckets
    pub buckets: Vec<AspectRatioBucket>,
    /// Padding color for resized images (default: black) [R, G, B]
    pub padding_color: [u8; 3],
    /// Whether to fall back to exact dimension grouping for edge cases
    pub fallback_to_exact: bool,
    /// Maximum number of images per bucket (0 = unlimited)
    pub max_images_per_bucket: usize,
}

impl Default for AspectRatioBucketingConfig {
    fn default() -> Self {
        Self {
            buckets: vec![
                AspectRatioBucket {
                    min_ratio: 0.0,
                    max_ratio: 0.8,
                    target_dims: (64, 32),
                    name: "tall".to_string(),
                },
                AspectRatioBucket {
                    min_ratio: 0.8,
                    max_ratio: 1.2,
                    target_dims: (32, 32),
                    name: "square".to_string(),
                },
                AspectRatioBucket {
                    min_ratio: 1.2,
                    max_ratio: 2.5,
                    target_dims: (32, 80),
                    name: "normal".to_string(),
                },
                AspectRatioBucket {
                    min_ratio: 2.5,
                    max_ratio: 4.5,
                    target_dims: (32, 160),
                    name: "wide".to_string(),
                },
                AspectRatioBucket {
                    min_ratio: 4.5,
                    max_ratio: f32::MAX, // Use f32::MAX instead of infinity for JSON compatibility
                    target_dims: (32, 320),
                    name: "ultra_wide".to_string(),
                },
            ],
            padding_color: [0, 0, 0], // Black padding
            fallback_to_exact: false,
            max_images_per_bucket: 0, // Unlimited
        }
    }
}

/// Aspect ratio bucketing processor
#[derive(Debug, Clone)]
pub struct AspectRatioBucketing {
    config: AspectRatioBucketingConfig,
    /// Cached ResizePadConfig for each bucket to avoid repeated creation
    resize_configs: HashMap<String, ResizePadConfig>,
}

impl Default for AspectRatioBucketing {
    fn default() -> Self {
        Self::new(AspectRatioBucketingConfig::default())
    }
}

impl AspectRatioBucketing {
    /// Create a new aspect ratio bucketing processor
    pub fn new(config: AspectRatioBucketingConfig) -> Self {
        // Pre-compute ResizePadConfig for each bucket to avoid repeated creation
        let mut resize_configs = HashMap::new();
        for bucket in &config.buckets {
            let (target_height, target_width) = bucket.target_dims;
            let resize_config = ResizePadConfig::new((target_width, target_height))
                .with_padding_strategy(PaddingStrategy::SolidColor(config.padding_color));
            resize_configs.insert(bucket.name.clone(), resize_config);
        }

        Self {
            config,
            resize_configs,
        }
    }

    /// Calculate aspect ratio of an image
    pub fn calculate_aspect_ratio(&self, image: &RgbImage) -> f32 {
        let (width, height) = image.dimensions();
        width as f32 / height as f32
    }

    /// Find the appropriate bucket for an aspect ratio
    pub fn find_bucket(&self, aspect_ratio: f32) -> Option<&AspectRatioBucket> {
        self.config
            .buckets
            .iter()
            .find(|bucket| aspect_ratio >= bucket.min_ratio && aspect_ratio < bucket.max_ratio)
    }

    /// Resize and pad an image to fit bucket dimensions
    pub fn resize_and_pad_to_bucket(
        &self,
        image: &RgbImage,
        bucket: &AspectRatioBucket,
    ) -> Result<RgbImage, OCRError> {
        // Use cached ResizePadConfig to avoid repeated creation
        let config =
            self.resize_configs
                .get(&bucket.name)
                .ok_or_else(|| OCRError::ConfigError {
                    message: format!("No cached resize config found for bucket: {}", bucket.name),
                })?;

        let padded = resize_and_pad(image, config);

        Ok(padded)
    }

    /// Group images by aspect ratio buckets
    pub fn group_images_by_buckets(
        &self,
        images: Vec<(usize, RgbImage)>,
    ) -> Result<HashMap<String, Vec<(usize, RgbImage)>>, OCRError> {
        let mut bucket_groups: HashMap<String, Vec<(usize, RgbImage)>> = HashMap::new();
        let mut exact_groups: HashMap<(u32, u32), Vec<(usize, RgbImage)>> = HashMap::new();

        for (index, image) in images {
            let aspect_ratio = self.calculate_aspect_ratio(&image);

            if let Some(bucket) = self.find_bucket(aspect_ratio) {
                // Resize and pad image to bucket dimensions
                let processed_image = self.resize_and_pad_to_bucket(&image, bucket)?;

                // Check bucket size limit
                let bucket_group = bucket_groups.entry(bucket.name.clone()).or_default();
                if self.config.max_images_per_bucket == 0
                    || bucket_group.len() < self.config.max_images_per_bucket
                {
                    bucket_group.push((index, processed_image));
                } else if self.config.fallback_to_exact {
                    // Fall back to exact grouping if bucket is full
                    let dims = (image.height(), image.width());
                    exact_groups.entry(dims).or_default().push((index, image));
                } else {
                    // Force into bucket even if over limit
                    bucket_group.push((index, processed_image));
                }
            } else if self.config.fallback_to_exact {
                // No bucket found, use exact grouping
                let dims = (image.height(), image.width());
                exact_groups.entry(dims).or_default().push((index, image));
            } else {
                return Err(OCRError::ConfigError {
                    message: format!(
                        "No bucket found for aspect ratio {:.2} and fallback disabled",
                        aspect_ratio
                    ),
                });
            }
        }

        // Add exact groups with dimension-based names
        for ((h, w), group) in exact_groups {
            let exact_key = format!("exact_{}x{}", h, w);
            bucket_groups.insert(exact_key, group);
        }

        Ok(bucket_groups)
    }

    /// Get bucket statistics for debugging
    pub fn get_bucket_stats(&self, images: &[(usize, RgbImage)]) -> HashMap<String, usize> {
        let mut stats = HashMap::new();

        for (_index, image) in images {
            let aspect_ratio = self.calculate_aspect_ratio(image);
            if let Some(bucket) = self.find_bucket(aspect_ratio) {
                *stats.entry(bucket.name.clone()).or_insert(0) += 1;
            } else {
                *stats.entry("no_bucket".to_string()).or_insert(0) += 1;
            }
        }

        stats
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    fn create_test_image(width: u32, height: u32) -> RgbImage {
        ImageBuffer::from_pixel(width, height, Rgb([255, 255, 255]))
    }

    #[test]
    fn test_aspect_ratio_calculation() {
        let bucketing = AspectRatioBucketing::default();
        let image = create_test_image(100, 50);
        let ratio = bucketing.calculate_aspect_ratio(&image);
        assert_eq!(ratio, 2.0);
    }

    #[test]
    fn test_bucket_finding() {
        let bucketing = AspectRatioBucketing::default();

        // Test different aspect ratios
        assert_eq!(bucketing.find_bucket(0.5).unwrap().name, "tall");
        assert_eq!(bucketing.find_bucket(1.0).unwrap().name, "square");
        assert_eq!(bucketing.find_bucket(2.0).unwrap().name, "normal");
        assert_eq!(bucketing.find_bucket(3.0).unwrap().name, "wide");
        assert_eq!(bucketing.find_bucket(5.0).unwrap().name, "ultra_wide");

        // Test very large aspect ratios (should still go to ultra_wide)
        assert_eq!(bucketing.find_bucket(100.0).unwrap().name, "ultra_wide");
        assert_eq!(bucketing.find_bucket(5000.0).unwrap().name, "ultra_wide");
    }

    #[test]
    fn test_resize_and_pad() {
        let bucketing = AspectRatioBucketing::default();
        let image = create_test_image(100, 50); // 2:1 aspect ratio
        let bucket = bucketing.find_bucket(2.0).unwrap();

        let result = bucketing.resize_and_pad_to_bucket(&image, bucket).unwrap();
        let (width, height) = result.dimensions();
        assert_eq!((height, width), bucket.target_dims);
    }

    #[test]
    fn test_group_images_by_buckets() {
        let bucketing = AspectRatioBucketing::default();

        // Create images with different aspect ratios
        let images = vec![
            (0, create_test_image(100, 50)),  // 2:1 ratio -> normal bucket
            (1, create_test_image(200, 100)), // 2:1 ratio -> normal bucket
            (2, create_test_image(50, 100)),  // 0.5:1 ratio -> tall bucket
            (3, create_test_image(100, 100)), // 1:1 ratio -> square bucket
            (4, create_test_image(300, 60)),  // 5:1 ratio -> ultra_wide bucket
        ];

        let groups = bucketing.group_images_by_buckets(images).unwrap();

        // Should have 4 different bucket groups
        assert!(groups.len() >= 4);

        // Check that images with similar aspect ratios are grouped together
        assert!(groups.contains_key("normal"));
        assert!(groups.contains_key("tall"));
        assert!(groups.contains_key("square"));
        assert!(groups.contains_key("ultra_wide"));

        // Normal bucket should have 2 images
        assert_eq!(groups.get("normal").unwrap().len(), 2);

        // Other buckets should have 1 image each
        assert_eq!(groups.get("tall").unwrap().len(), 1);
        assert_eq!(groups.get("square").unwrap().len(), 1);
        assert_eq!(groups.get("ultra_wide").unwrap().len(), 1);
    }

    #[test]
    fn test_bucket_efficiency_comparison() {
        let bucketing = AspectRatioBucketing::default();

        // Create many images with slightly different dimensions but similar aspect ratios
        let mut images = Vec::new();
        for i in 0..20 {
            // Create images with aspect ratios around 2:1 but different exact dimensions
            let width = 100 + i * 2; // 100, 102, 104, ... 138
            let height = 50 + i; // 50, 51, 52, ... 69
            images.push((i as usize, create_test_image(width, height)));
        }

        // With aspect ratio bucketing, these should mostly go into one or two buckets
        let bucket_groups = bucketing.group_images_by_buckets(images.clone()).unwrap();

        // With exact dimension grouping, each image would be in its own group
        let mut exact_groups = HashMap::new();
        for (i, image) in images {
            let dims = (image.height(), image.width());
            exact_groups
                .entry(dims)
                .or_insert_with(Vec::new)
                .push((i, image));
        }

        // Aspect ratio bucketing should create fewer groups (better batch efficiency)
        assert!(bucket_groups.len() < exact_groups.len());

        // Most images should be in the same bucket
        let largest_bucket_size = bucket_groups.values().map(|v| v.len()).max().unwrap();
        assert!(largest_bucket_size > 10); // Most images should be grouped together

        // Exact grouping should have mostly single-image groups
        let exact_single_groups = exact_groups.values().filter(|v| v.len() == 1).count();
        assert!(exact_single_groups > 15); // Most groups should have only one image
    }

    #[test]
    fn test_json_serialization_roundtrip() {
        let config = AspectRatioBucketingConfig::default();

        // Test that JSON serialization and deserialization work correctly
        let json_str = serde_json::to_string(&config).expect("Should serialize to JSON");
        let deserialized: AspectRatioBucketingConfig =
            serde_json::from_str(&json_str).expect("Should deserialize from JSON");

        // Verify that the deserialized config matches the original
        assert_eq!(config.buckets.len(), deserialized.buckets.len());
        assert_eq!(config.padding_color, deserialized.padding_color);
        assert_eq!(config.fallback_to_exact, deserialized.fallback_to_exact);
        assert_eq!(
            config.max_images_per_bucket,
            deserialized.max_images_per_bucket
        );

        // Verify that the last bucket still works for very large aspect ratios
        let bucketing = AspectRatioBucketing::new(deserialized);
        let bucket_name = bucketing
            .find_bucket(5000.0)
            .map(|bucket| bucket.name.as_str());
        assert_eq!(bucket_name, Some("ultra_wide"));
    }

    #[test]
    fn test_resize_config_caching_optimization() {
        let bucketing = AspectRatioBucketing::default();

        // Verify that resize configs are pre-computed for all buckets
        assert_eq!(
            bucketing.resize_configs.len(),
            bucketing.config.buckets.len()
        );

        // Verify that each bucket has a corresponding cached config
        for bucket in &bucketing.config.buckets {
            assert!(bucketing.resize_configs.contains_key(&bucket.name));

            // Verify the cached config has correct dimensions and padding
            let cached_config = &bucketing.resize_configs[&bucket.name];
            let (target_height, target_width) = bucket.target_dims;
            assert_eq!(cached_config.target_dims, (target_width, target_height));

            let PaddingStrategy::SolidColor(color) = cached_config.padding_strategy else {
                assert!(
                    matches!(
                        cached_config.padding_strategy,
                        PaddingStrategy::SolidColor(_)
                    ),
                    "Expected SolidColor padding strategy"
                );
                continue;
            };
            assert_eq!(color, bucketing.config.padding_color);
        }

        // Test that resize_and_pad_to_bucket works correctly with cached configs
        let test_image = create_test_image(100, 50); // 2:1 aspect ratio
        let maybe_bucket = bucketing.find_bucket(2.0);
        let Some(bucket) = maybe_bucket else {
            assert!(
                maybe_bucket.is_some(),
                "aspect ratio 2.0 should map to an existing bucket"
            );
            return;
        };
        let result = match bucketing.resize_and_pad_to_bucket(&test_image, bucket) {
            Ok(image) => image,
            Err(err) => {
                panic!("expected resize_and_pad_to_bucket to succeed, got {err:?}");
            }
        };

        let (target_height, target_width) = bucket.target_dims;
        assert_eq!(result.dimensions(), (target_width, target_height));
    }
}
