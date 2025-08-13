//! Text line orientation correction component.
//!
//! This module provides functionality for applying text line orientation corrections
//! to images, separating this concern from the recognition processing logic.
//!
//! ## Overview
//!
//! The `OrientationCorrector` component handles the application of text line orientation
//! corrections to images before they are processed by the text recognition model.
//! This separation allows for:
//!
//! - Clear separation of concerns
//! - Configurable orientation correction behavior
//! - Easier testing and maintenance
//! - Reusable orientation correction logic
//!
//! ## Usage Example
//!
//! ```rust
//! use oar_ocr::pipeline::stages::{
//!     OrientationCorrector, OrientationCorrectionConfig
//! };
//! use image::RgbImage;
//!
//! // Create corrector with orientation correction enabled
//! let config = OrientationCorrectionConfig { enabled: true };
//! let corrector = OrientationCorrector::new(config);
//!
//! // Apply corrections to a batch of images
//! let mut images = vec![RgbImage::new(100, 50)];
//! let indices = vec![0];
//! let orientations = vec![Some(180.0)]; // 180-degree rotation needed
//!
//! let corrections_applied = corrector.apply_corrections(
//!     &mut images,
//!     &indices,
//!     Some(&orientations)
//! );
//!
//! println!("Applied {} corrections", corrections_applied);
//! ```

use image::RgbImage;
use serde::{Deserialize, Serialize};
use tracing::debug;

use crate::core::apply_text_line_orientation;

/// Configuration for orientation correction
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct OrientationCorrectionConfig {
    /// Whether to apply text line orientation corrections
    #[serde(default)]
    pub enabled: bool,
}

/// Component responsible for applying text line orientation corrections
#[derive(Debug, Clone)]
pub struct OrientationCorrector {
    config: OrientationCorrectionConfig,
}

impl OrientationCorrector {
    /// Create a new orientation corrector
    pub fn new(config: OrientationCorrectionConfig) -> Self {
        Self { config }
    }

    /// Apply orientation corrections to a batch of images
    ///
    /// # Arguments
    /// * `images` - Mutable reference to vector of images to correct
    /// * `indices` - Original indices of the images in the batch
    /// * `orientations` - Optional slice of orientation angles for each original image
    ///
    /// # Returns
    /// Number of corrections applied
    pub fn apply_corrections(
        &self,
        images: &mut [RgbImage],
        indices: &[usize],
        orientations: Option<&[Option<f32>]>,
    ) -> usize {
        if !self.config.enabled {
            return 0;
        }

        let Some(orientations) = orientations else {
            return 0;
        };

        let mut corrections_applied = 0;

        for (i, image) in images.iter_mut().enumerate() {
            // Early-continue style for clarity and to avoid nested if-let chains
            let Some(&original_idx) = indices.get(i) else {
                continue;
            };

            let Some(angle) = orientations
                .get(original_idx) // Option<&Option<f32>>
                .copied() // Option<Option<f32>>
                .flatten()
            // Option<f32>
            else {
                continue;
            };

            if angle == 0.0 {
                continue;
            }

            *image = apply_text_line_orientation(image.clone(), angle);
            corrections_applied += 1;
            debug!(
                "Applied text line orientation correction: {} degrees for image {}",
                angle, original_idx
            );
        }

        if corrections_applied > 0 {
            debug!(
                "Applied {} text line orientation corrections to batch",
                corrections_applied
            );
        }

        corrections_applied
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
    fn test_orientation_corrector_disabled() {
        let config = OrientationCorrectionConfig { enabled: false };
        let corrector = OrientationCorrector::new(config);

        let mut images = vec![create_test_image(100, 50)];
        let indices = vec![0];
        let orientations = vec![Some(180.0)];

        let corrections = corrector.apply_corrections(&mut images, &indices, Some(&orientations));
        assert_eq!(corrections, 0);
    }

    #[test]
    fn test_orientation_corrector_enabled() {
        let config = OrientationCorrectionConfig { enabled: true };
        let corrector = OrientationCorrector::new(config);

        let mut images = vec![create_test_image(2, 2)];
        let indices = vec![0];
        let orientations = vec![Some(180.0)];

        let corrections = corrector.apply_corrections(&mut images, &indices, Some(&orientations));
        assert_eq!(corrections, 1);
    }

    #[test]
    fn test_orientation_corrector_no_orientations() {
        let config = OrientationCorrectionConfig { enabled: true };
        let corrector = OrientationCorrector::new(config);

        let mut images = vec![create_test_image(100, 50)];
        let indices = vec![0];

        let corrections = corrector.apply_corrections(&mut images, &indices, None);
        assert_eq!(corrections, 0);
    }

    #[test]
    fn test_orientation_corrector_zero_angle() {
        let config = OrientationCorrectionConfig { enabled: true };
        let corrector = OrientationCorrector::new(config);

        let mut images = vec![create_test_image(100, 50)];
        let indices = vec![0];
        let orientations = vec![Some(0.0)];

        let corrections = corrector.apply_corrections(&mut images, &indices, Some(&orientations));
        assert_eq!(corrections, 0);
    }

    #[test]
    fn test_default_config() {
        let config = OrientationCorrectionConfig::default();
        assert!(!config.enabled);
    }
}
