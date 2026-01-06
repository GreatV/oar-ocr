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

use oar_ocr_core::core::OCRError;
use oar_ocr_core::core::traits::adapter::ModelAdapter;
use oar_ocr_core::core::traits::task::ImageTaskInput;
use oar_ocr_core::domain::adapters::{DocumentOrientationAdapter, UVDocRectifierAdapter};
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

/// Applies orientation correction to an image based on the detected class ID.
///
/// This is the core rotation logic extracted for testability. The class_id corresponds to:
/// - 0: 0° (no rotation needed)
/// - 1: 90° (rotate 270° CCW to correct)
/// - 2: 180° (rotate 180° to correct)
/// - 3: 270° (rotate 90° CW to correct)
///
/// Returns the corrected image and correction metadata.
fn apply_orientation_from_class_id(
    image: Arc<image::RgbImage>,
    class_id: Option<usize>,
) -> (Arc<image::RgbImage>, Option<OrientationCorrection>) {
    let Some(class_id) = class_id else {
        return (image, None);
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

    (rotated, Some(correction))
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

    Ok(apply_orientation_from_class_id(image, class_id))
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
    fn test_apply_orientation_none_class_id_returns_original_image() {
        // When no classification is available, return original image with None metadata
        let image = Arc::new(create_test_image(100, 200));
        let (result, correction) = apply_orientation_from_class_id(Arc::clone(&image), None);

        // Image should be unchanged
        assert!(Arc::ptr_eq(&image, &result));
        assert!(correction.is_none());
    }

    #[test]
    fn test_apply_orientation_class_id_0_no_rotation() {
        // class_id 0 -> 0° -> no rotation needed
        let image = Arc::new(create_test_image(100, 200));
        let (result, correction) = apply_orientation_from_class_id(Arc::clone(&image), Some(0));

        // No rotation - dimensions unchanged, same image pointer
        assert!(Arc::ptr_eq(&image, &result));
        assert_eq!(result.width(), 100);
        assert_eq!(result.height(), 200);

        // Correction metadata should still be present
        let correction = correction.expect("correction should be Some");
        assert_eq!(correction.angle, 0.0);
        assert_eq!(correction.rotated_width, 100);
        assert_eq!(correction.rotated_height, 200);
    }

    #[test]
    fn test_apply_orientation_class_id_1_rotates_270_ccw() {
        // class_id 1 -> 90° detected -> rotate 270° CCW (rotate270) to correct
        let image = Arc::new(create_test_image(100, 200));
        let (result, correction) = apply_orientation_from_class_id(Arc::clone(&image), Some(1));

        // 90° rotation swaps dimensions
        assert_eq!(result.width(), 200);
        assert_eq!(result.height(), 100);
        // Should be a new image, not the original
        assert!(!Arc::ptr_eq(&image, &result));

        let correction = correction.expect("correction should be Some");
        assert_eq!(correction.angle, 90.0);
        assert_eq!(correction.rotated_width, 200);
        assert_eq!(correction.rotated_height, 100);
    }

    #[test]
    fn test_apply_orientation_class_id_2_rotates_180() {
        // class_id 2 -> 180° detected -> rotate 180° to correct
        let image = Arc::new(create_test_image(100, 200));
        let (result, correction) = apply_orientation_from_class_id(Arc::clone(&image), Some(2));

        // 180° rotation keeps dimensions
        assert_eq!(result.width(), 100);
        assert_eq!(result.height(), 200);
        // Should be a new image
        assert!(!Arc::ptr_eq(&image, &result));

        let correction = correction.expect("correction should be Some");
        assert_eq!(correction.angle, 180.0);
        assert_eq!(correction.rotated_width, 100);
        assert_eq!(correction.rotated_height, 200);
    }

    #[test]
    fn test_apply_orientation_class_id_3_rotates_90_cw() {
        // class_id 3 -> 270° detected -> rotate 90° CW (rotate90) to correct
        let image = Arc::new(create_test_image(100, 200));
        let (result, correction) = apply_orientation_from_class_id(Arc::clone(&image), Some(3));

        // 270° rotation swaps dimensions
        assert_eq!(result.width(), 200);
        assert_eq!(result.height(), 100);
        assert!(!Arc::ptr_eq(&image, &result));

        let correction = correction.expect("correction should be Some");
        assert_eq!(correction.angle, 270.0);
        assert_eq!(correction.rotated_width, 200);
        assert_eq!(correction.rotated_height, 100);
    }

    #[test]
    fn test_apply_orientation_unknown_class_id_preserves_metadata() {
        // Unknown class_id (e.g., future model outputs) -> no rotation but metadata preserved
        let image = Arc::new(create_test_image(100, 200));
        let (result, correction) = apply_orientation_from_class_id(Arc::clone(&image), Some(99));

        // No rotation - same image
        assert!(Arc::ptr_eq(&image, &result));
        assert_eq!(result.width(), 100);
        assert_eq!(result.height(), 200);

        // Correction metadata preserves the unknown angle
        let correction = correction.expect("correction should be Some");
        assert_eq!(correction.angle, 8910.0); // 99 * 90.0
        assert_eq!(correction.rotated_width, 100);
        assert_eq!(correction.rotated_height, 200);
    }

    #[test]
    fn test_apply_orientation_square_image_all_rotations() {
        // Square images should maintain dimensions for all rotations
        let image = Arc::new(create_test_image(150, 150));

        for class_id in 0..4 {
            let (result, correction) =
                apply_orientation_from_class_id(Arc::clone(&image), Some(class_id));

            assert_eq!(result.width(), 150);
            assert_eq!(result.height(), 150);

            let correction = correction.expect("correction should be Some");
            assert_eq!(correction.angle, (class_id as f32) * 90.0);
        }
    }

    #[test]
    fn test_angle_calculation_from_class_id() {
        // Verify angle = class_id * 90.0 for standard orientations
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

    #[test]
    fn test_preprocess_result_invariant_rotation_none_when_rectified() {
        // Invariant: When rectification is applied, rotation metadata is None
        // because back-mapping is not supported for warped images
        let _image = Arc::new(create_test_image(100, 200));
        let rectified = Arc::new(create_test_image(110, 210));

        let result = PreprocessResult {
            image: Arc::clone(&rectified),
            orientation_angle: Some(90.0), // Orientation was detected
            rotation: None,                // But rotation metadata is cleared
            rectified_img: Some(rectified),
        };

        // This models the behavior in DocumentPreprocessor::preprocess
        // where rotation is set to None when rectified_img is Some
        assert!(result.rotation.is_none());
        assert!(result.rectified_img.is_some());
    }

    #[test]
    fn test_preprocess_result_rotation_preserved_without_rectification() {
        // When no rectification is applied, rotation metadata is preserved
        let (rotated_image, correction) =
            apply_orientation_from_class_id(Arc::new(create_test_image(100, 200)), Some(1));

        let result = PreprocessResult {
            image: rotated_image,
            orientation_angle: Some(90.0),
            rotation: correction,
            rectified_img: None,
        };

        // Rotation metadata preserved for coordinate back-mapping
        assert!(result.rotation.is_some());
        assert!(result.rectified_img.is_none());
        assert_eq!(result.rotation.unwrap().angle, 90.0);
    }

    /// Helper to simulate the class_id extraction logic from DocumentOrientationOutput
    fn extract_class_id_from_classifications(
        classifications: &[Vec<(usize, String, f32)>],
    ) -> Option<usize> {
        classifications
            .first()
            .and_then(|c| c.first())
            .map(|(class_id, _, _)| *class_id)
    }

    #[test]
    fn test_classification_extraction_with_valid_result() {
        // Simulates: adapter returns [[Classification { class_id: 1, ... }]]
        let classifications = vec![vec![(1_usize, "90".to_string(), 0.95_f32)]];

        let class_id = extract_class_id_from_classifications(&classifications);
        assert_eq!(class_id, Some(1));
    }

    #[test]
    fn test_classification_extraction_with_multiple_topk() {
        // Simulates: adapter returns top-k results, we take the first (highest confidence)
        let classifications = vec![vec![
            (2_usize, "180".to_string(), 0.85_f32),
            (0_usize, "0".to_string(), 0.10_f32),
            (1_usize, "90".to_string(), 0.05_f32),
        ]];

        let class_id = extract_class_id_from_classifications(&classifications);
        assert_eq!(class_id, Some(2)); // Takes first (highest confidence)
    }

    #[test]
    fn test_classification_extraction_with_empty_inner_vec() {
        // Simulates: adapter returns classification but with empty inner vec
        let classifications: Vec<Vec<(usize, String, f32)>> = vec![vec![]];

        let class_id = extract_class_id_from_classifications(&classifications);
        assert_eq!(class_id, None);
    }

    #[test]
    fn test_classification_extraction_with_empty_outer_vec() {
        // Simulates: adapter returns empty classifications (no images processed)
        let classifications: Vec<Vec<(usize, String, f32)>> = vec![];

        let class_id = extract_class_id_from_classifications(&classifications);
        assert_eq!(class_id, None);
    }

    #[test]
    fn test_classification_extraction_for_batch_input() {
        // Simulates: batch input with multiple images, we take the first image's result
        let classifications = vec![
            vec![(1_usize, "90".to_string(), 0.95_f32)],
            vec![(2_usize, "180".to_string(), 0.90_f32)],
        ];

        // correct_image_orientation only processes single images
        let class_id = extract_class_id_from_classifications(&classifications);
        assert_eq!(class_id, Some(1)); // First image's classification
    }

    #[test]
    fn test_orientation_correction_flow_with_90_degree_detection() {
        // Complete flow: 90° detected -> rotate 270° CCW to correct
        let image = Arc::new(create_test_image(100, 200));

        // Simulate what correct_image_orientation does after adapter call
        let class_id = Some(1_usize); // 90° orientation detected
        let (result, correction) = apply_orientation_from_class_id(Arc::clone(&image), class_id);

        // Verify the complete flow result
        assert_eq!(result.width(), 200); // Swapped
        assert_eq!(result.height(), 100);

        let correction = correction.expect("should have correction");
        assert_eq!(correction.angle, 90.0);
        assert_eq!(correction.rotated_width, 200);
        assert_eq!(correction.rotated_height, 100);
    }

    #[test]
    fn test_orientation_correction_flow_no_classification_available() {
        // When adapter returns no classification, image is unchanged
        let image = Arc::new(create_test_image(100, 200));

        let class_id = None; // No classification available
        let (result, correction) = apply_orientation_from_class_id(Arc::clone(&image), class_id);

        // Image unchanged, no correction metadata
        assert!(Arc::ptr_eq(&image, &result));
        assert!(correction.is_none());
    }

    #[test]
    fn test_preprocessor_builds_correct_result_structure() {
        // Without adapters, preprocessor returns pass-through result
        let preprocessor = DocumentPreprocessor::new(None, None);
        let image = Arc::new(create_test_image(100, 200));

        let result = preprocessor.preprocess(Arc::clone(&image)).unwrap();

        // Verify result structure
        assert_eq!(result.image.width(), 100);
        assert_eq!(result.image.height(), 200);
        assert!(result.orientation_angle.is_none());
        assert!(result.rotation.is_none());
        assert!(result.rectified_img.is_none());
    }
}
