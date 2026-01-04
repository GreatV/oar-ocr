//! # Stage Definition: Document Preprocessing
//!
//! This service is considered "Done" when it fulfills the following contract:
//!
//! - **Inputs**: Single `Arc<image::RgbImage>`.
//! - **Outputs**: `PreprocessResult` containing the (potentially rotated/rectified) image,
//!   detected orientation angle, and optional `OrientationCorrection` for coordinate back-mapping.
//! - **Logging**: Traces orientation corrections (angle) and rectification application.
//! - **Invariants**:
//!     - If rectification is applied, `rotation` metadata is `None` (back-mapping is not supported for warped images).
//!     - Output image is always in RGB format.
//!     - Corrected images are rotated to upright (0°) orientation.

use crate::core::OCRError;
use crate::core::traits::adapter::ModelAdapter;
use crate::core::traits::task::ImageTaskInput;
use crate::domain::adapters::{DocumentOrientationAdapter, UVDocRectifierAdapter};
use std::sync::Arc;

/// Orientation correction metadata for a single image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) struct OrientationCorrection {
    /// Detected orientation angle in degrees (0/90/180/270).
    pub angle: f32,
    /// Width of the corrected (rotated) image.
    pub rotated_width: u32,
    /// Height of the corrected (rotated) image.
    pub rotated_height: u32,
}

/// Result of preprocessing an image.
#[derive(Debug)]
pub(crate) struct PreprocessResult {
    /// The preprocessed image (potentially rotated/rectified), wrapped in Arc for zero-copy sharing.
    pub image: Arc<image::RgbImage>,
    pub orientation_angle: Option<f32>,
    /// Bounding boxes should only be mapped back when rectification is not applied.
    pub rotation: Option<OrientationCorrection>,
    pub rectified_img: Option<Arc<image::RgbImage>>,
}

/// Shared document preprocessor (optional orientation + optional rectification).
#[derive(Debug, Clone)]
pub(crate) struct DocumentPreprocessor<'a> {
    orientation_adapter: Option<&'a DocumentOrientationAdapter>,
    rectification_adapter: Option<&'a UVDocRectifierAdapter>,
}

impl<'a> DocumentPreprocessor<'a> {
    pub(crate) fn new(
        orientation_adapter: Option<&'a DocumentOrientationAdapter>,
        rectification_adapter: Option<&'a UVDocRectifierAdapter>,
    ) -> Self {
        Self {
            orientation_adapter,
            rectification_adapter,
        }
    }

    pub(crate) fn preprocess(
        &self,
        image: Arc<image::RgbImage>,
    ) -> Result<PreprocessResult, OCRError> {
        let (mut current_image, orientation_angle, rotation) =
            if let Some(orientation_adapter) = self.orientation_adapter {
                let (rotated, rotation) =
                    correct_image_orientation(Arc::clone(&image), orientation_adapter)?;
                (rotated, rotation.map(|r| r.angle), rotation)
            } else {
                (image, None, None)
            };

        let mut rectified_img: Option<Arc<image::RgbImage>> = None;

        if let Some(rectification_adapter) = self.rectification_adapter {
            // Adapter boundary: must clone to transfer ownership
            let input = ImageTaskInput::new(vec![(*current_image).clone()]);
            let rect_output = rectification_adapter.execute(input, None)?;

            if let Some(rectified) = rect_output.rectified_images.first() {
                current_image = Arc::new(rectified.clone());
                rectified_img = Some(Arc::clone(&current_image));
            }
        }

        // UVDoc rectification can't be inverted precisely; keep results in rectified space.
        let rotation = if rectified_img.is_none() {
            rotation
        } else {
            None
        };

        Ok(PreprocessResult {
            image: current_image,
            orientation_angle,
            rotation,
            rectified_img,
        })
    }
}

/// Applies the shared orientation policy to an image using the provided adapter.
///
/// Returns the corrected image and optional correction metadata. If the adapter
/// fails to produce a classification, the original image is returned with `None`.
pub(crate) fn correct_image_orientation(
    image: Arc<image::RgbImage>,
    orientation_adapter: &DocumentOrientationAdapter,
) -> Result<(Arc<image::RgbImage>, Option<OrientationCorrection>), OCRError> {
    // Adapter boundary: must clone to transfer ownership
    let input = ImageTaskInput::new(vec![(*image).clone()]);
    let output = orientation_adapter.execute(input, None)?;

    let class_id = output
        .classifications
        .first()
        .and_then(|c| c.first())
        .map(|c| c.class_id);

    let Some(class_id) = class_id else {
        return Ok((image, None));
    };

    let angle = (class_id as f32) * 90.0;

    // Shared correction policy (same as OCR/structure document correction):
    // class_id: 0=0°, 1=90°, 2=180°, 3=270°.
    // To correct the image, rotate by the inverse transform:
    //  - 90° -> rotate 90° CCW (rotate270)
    //  - 180° -> rotate 180°
    //  - 270° -> rotate 90° CW (rotate90)
    // For unknown class_ids, no rotation is applied but metadata is preserved
    // to allow downstream processing to handle new model outputs.
    let rotated = match class_id {
        1 => Arc::new(image::imageops::rotate270(&*image)),
        2 => Arc::new(image::imageops::rotate180(&*image)),
        3 => Arc::new(image::imageops::rotate90(&*image)),
        _ => image,
    };

    let correction = OrientationCorrection {
        angle,
        rotated_width: rotated.width(),
        rotated_height: rotated.height(),
    };

    Ok((rotated, Some(correction)))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Creates a test image with specified dimensions.
    fn create_test_image(width: u32, height: u32) -> image::RgbImage {
        image::RgbImage::new(width, height)
    }

    #[test]
    fn test_orientation_correction_creation() {
        let correction = OrientationCorrection {
            angle: 90.0,
            rotated_width: 200,
            rotated_height: 100,
        };

        assert_eq!(correction.angle, 90.0);
        assert_eq!(correction.rotated_width, 200);
        assert_eq!(correction.rotated_height, 100);
    }

    #[test]
    fn test_orientation_correction_equality() {
        let c1 = OrientationCorrection {
            angle: 180.0,
            rotated_width: 100,
            rotated_height: 200,
        };
        let c2 = OrientationCorrection {
            angle: 180.0,
            rotated_width: 100,
            rotated_height: 200,
        };
        let c3 = OrientationCorrection {
            angle: 90.0,
            rotated_width: 100,
            rotated_height: 200,
        };

        assert_eq!(c1, c2);
        assert_ne!(c1, c3);
    }

    #[test]
    fn test_orientation_correction_copy() {
        let c1 = OrientationCorrection {
            angle: 270.0,
            rotated_width: 150,
            rotated_height: 300,
        };
        let c2 = c1; // Copy
        assert_eq!(c1, c2);
    }

    #[test]
    fn test_preprocess_result_creation() {
        let image = Arc::new(create_test_image(100, 200));
        let result = PreprocessResult {
            image: Arc::clone(&image),
            orientation_angle: Some(90.0),
            rotation: Some(OrientationCorrection {
                angle: 90.0,
                rotated_width: 200,
                rotated_height: 100,
            }),
            rectified_img: None,
        };

        assert_eq!(result.image.width(), 100);
        assert_eq!(result.image.height(), 200);
        assert_eq!(result.orientation_angle, Some(90.0));
        assert!(result.rotation.is_some());
        assert!(result.rectified_img.is_none());
    }

    #[test]
    fn test_preprocess_result_with_rectified_image() {
        let rectified = Arc::new(create_test_image(120, 220));
        let result = PreprocessResult {
            image: Arc::clone(&rectified),
            orientation_angle: None,
            rotation: None, // Should be None when rectified
            rectified_img: Some(rectified),
        };

        assert!(result.orientation_angle.is_none());
        assert!(result.rotation.is_none());
        assert!(result.rectified_img.is_some());
    }

    #[test]
    fn test_document_preprocessor_no_adapters() {
        let preprocessor = DocumentPreprocessor::new(None, None);
        let image = Arc::new(create_test_image(100, 200));

        let result = preprocessor
            .preprocess(Arc::clone(&image))
            .expect("preprocess should succeed");

        // Without adapters, the image should be unchanged
        assert_eq!(result.image.width(), 100);
        assert_eq!(result.image.height(), 200);
        assert!(result.orientation_angle.is_none());
        assert!(result.rotation.is_none());
        assert!(result.rectified_img.is_none());
    }

    #[test]
    fn test_document_preprocessor_clone() {
        let preprocessor = DocumentPreprocessor::<'static>::new(None, None);
        let cloned = preprocessor.clone();

        // Both should work identically
        let image = Arc::new(create_test_image(50, 50));
        let r1 = preprocessor
            .preprocess(Arc::clone(&image))
            .expect("preprocess should succeed");
        let r2 = cloned
            .preprocess(Arc::clone(&image))
            .expect("preprocess should succeed");

        assert_eq!(r1.image.width(), r2.image.width());
        assert_eq!(r1.image.height(), r2.image.height());
    }

    #[test]
    fn test_rotation_logic_0_degrees() {
        // class_id 0 -> 0° -> no rotation
        let original = create_test_image(100, 200);
        let class_id = 0u32;

        // Simulate the rotation logic from correct_image_orientation
        let rotated = match class_id {
            1 => image::imageops::rotate270(&original),
            2 => image::imageops::rotate180(&original),
            3 => image::imageops::rotate90(&original),
            _ => original.clone(),
        };

        // No rotation - dimensions unchanged
        assert_eq!(rotated.width(), 100);
        assert_eq!(rotated.height(), 200);
    }

    #[test]
    fn test_rotation_logic_90_degrees() {
        // class_id 1 -> 90° -> rotate270 (CCW) to correct
        let original = create_test_image(100, 200);
        let class_id = 1u32;

        let rotated = match class_id {
            1 => image::imageops::rotate270(&original),
            2 => image::imageops::rotate180(&original),
            3 => image::imageops::rotate90(&original),
            _ => original.clone(),
        };

        // 90° rotation swaps dimensions
        assert_eq!(rotated.width(), 200);
        assert_eq!(rotated.height(), 100);
    }

    #[test]
    fn test_rotation_logic_180_degrees() {
        // class_id 2 -> 180° -> rotate180 to correct
        let original = create_test_image(100, 200);
        let class_id = 2u32;

        let rotated = match class_id {
            1 => image::imageops::rotate270(&original),
            2 => image::imageops::rotate180(&original),
            3 => image::imageops::rotate90(&original),
            _ => original.clone(),
        };

        // 180° rotation keeps dimensions
        assert_eq!(rotated.width(), 100);
        assert_eq!(rotated.height(), 200);
    }

    #[test]
    fn test_rotation_logic_270_degrees() {
        // class_id 3 -> 270° -> rotate90 (CW) to correct
        let original = create_test_image(100, 200);
        let class_id = 3u32;

        let rotated = match class_id {
            1 => image::imageops::rotate270(&original),
            2 => image::imageops::rotate180(&original),
            3 => image::imageops::rotate90(&original),
            _ => original.clone(),
        };

        // 270° rotation swaps dimensions
        assert_eq!(rotated.width(), 200);
        assert_eq!(rotated.height(), 100);
    }

    #[test]
    fn test_rotation_logic_unknown_class_id() {
        // Unknown class_id -> no rotation
        let original = create_test_image(100, 200);
        let class_id = 99u32;

        let rotated = match class_id {
            1 => image::imageops::rotate270(&original),
            2 => image::imageops::rotate180(&original),
            3 => image::imageops::rotate90(&original),
            _ => original.clone(),
        };

        // No rotation - dimensions unchanged
        assert_eq!(rotated.width(), 100);
        assert_eq!(rotated.height(), 200);
    }

    #[test]
    fn test_angle_calculation_from_class_id() {
        // Verify angle = class_id * 90.0
        assert_eq!(0_f32 * 90.0, 0.0);
        assert_eq!(1_f32 * 90.0, 90.0);
        assert_eq!(2_f32 * 90.0, 180.0);
        assert_eq!(3_f32 * 90.0, 270.0);
    }

    #[test]
    fn test_arc_sharing_without_clone() {
        // Verify Arc allows zero-copy sharing
        let image = Arc::new(create_test_image(100, 200));
        assert_eq!(Arc::strong_count(&image), 1);

        let shared = Arc::clone(&image);
        assert_eq!(Arc::strong_count(&image), 2);
        assert_eq!(Arc::strong_count(&shared), 2);

        // Both point to the same image
        assert!(Arc::ptr_eq(&image, &shared));
    }
}
