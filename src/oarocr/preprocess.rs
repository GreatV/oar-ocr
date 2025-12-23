//! # Stage Definition: Document Preprocessing
//!
//! This service is considered "Done" when it fulfills the following contract:
//!
//! - **Inputs**: Single `image::RgbImage`.
//! - **Outputs**: `PreprocessResult` containing the (potentially rotated/rectified) image,
//!   detected orientation angle, and optional `OrientationCorrection` for coordinate back-mapping.
//! - **Logging**: Traces orientation corrections (angle) and rectification application.
//! - **Invariants**:
//!     - If rectification is applied, `rotation` metadata is `None` (back-mapping is not supported for warped images).
//!     - Output image is always in RGB format.
//!     - Corrected images are rotated to upright (0°) orientation.

use crate::core::OCRError;
use crate::core::registry::{DynModelAdapter, DynTaskInput};
use crate::core::traits::task::ImageTaskInput;
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
    pub image: image::RgbImage,
    pub orientation_angle: Option<f32>,
    /// Bounding boxes should only be mapped back when rectification is not applied.
    pub rotation: Option<OrientationCorrection>,
    pub rectified_img: Option<Arc<image::RgbImage>>,
}

/// Shared document preprocessor (optional orientation + optional rectification).
#[derive(Debug, Default, Clone)]
pub(crate) struct DocumentPreprocessor {
    orientation_adapter: Option<Arc<dyn DynModelAdapter>>,
    rectification_adapter: Option<Arc<dyn DynModelAdapter>>,
}

impl DocumentPreprocessor {
    pub(crate) fn new(
        orientation_adapter: Option<Arc<dyn DynModelAdapter>>,
        rectification_adapter: Option<Arc<dyn DynModelAdapter>>,
    ) -> Self {
        Self {
            orientation_adapter,
            rectification_adapter,
        }
    }

    pub(crate) fn preprocess(&self, image: image::RgbImage) -> Result<PreprocessResult, OCRError> {
        let (mut current_image, orientation_angle, rotation) =
            if let Some(ref orientation_adapter) = self.orientation_adapter {
                let (rotated, rotation) = correct_image_orientation(image, orientation_adapter)?;
                (rotated, rotation.map(|r| r.angle), rotation)
            } else {
                (image, None, None)
            };

        let mut rectified_img: Option<Arc<image::RgbImage>> = None;

        if let Some(ref rectification_adapter) = self.rectification_adapter {
            let input = DynTaskInput::from_images(ImageTaskInput::new(vec![current_image.clone()]));
            let output = rectification_adapter.execute_dyn(input)?;

            if let Ok(rect_output) = output.into_document_rectification()
                && let Some(rectified) = rect_output.rectified_images.first()
            {
                current_image = rectified.clone();
                rectified_img = Some(Arc::new(current_image.clone()));
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
    image: image::RgbImage,
    orientation_adapter: &Arc<dyn DynModelAdapter>,
) -> Result<(image::RgbImage, Option<OrientationCorrection>), OCRError> {
    let input = DynTaskInput::from_images(ImageTaskInput::new(vec![image.clone()]));
    let output = orientation_adapter.execute_dyn(input)?;

    let class_id = output.into_document_orientation().ok().and_then(|o| {
        o.classifications
            .first()
            .and_then(|c| c.first())
            .map(|c| c.class_id)
    });

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
        1 => image::imageops::rotate270(&image),
        2 => image::imageops::rotate180(&image),
        3 => image::imageops::rotate90(&image),
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
    use crate::core::registry::DynTaskOutput;
    use crate::core::{AdapterInfo, TaskType};
    use crate::domain::tasks::document_orientation::{Classification, DocumentOrientationOutput};

    /// Creates a test image with specified dimensions.
    fn create_test_image(width: u32, height: u32) -> image::RgbImage {
        image::RgbImage::new(width, height)
    }

    /// Mock adapter that returns a specific class_id for orientation detection.
    #[derive(Debug)]
    struct MockOrientationAdapter {
        class_id: usize,
    }

    impl MockOrientationAdapter {
        fn new(class_id: usize) -> Self {
            Self { class_id }
        }
    }

    impl DynModelAdapter for MockOrientationAdapter {
        fn info(&self) -> AdapterInfo {
            AdapterInfo {
                model_name: "MockOrientationAdapter".to_string(),
                version: "1.0".to_string(),
                task_type: TaskType::DocumentOrientation,
                description: "Mock adapter for testing".to_string(),
            }
        }

        fn task_type(&self) -> TaskType {
            TaskType::DocumentOrientation
        }

        fn supports_batching(&self) -> bool {
            false
        }

        fn recommended_batch_size(&self) -> usize {
            1
        }

        fn execute_dyn(&self, _input: DynTaskInput) -> Result<DynTaskOutput, OCRError> {
            let classification =
                Classification::new(self.class_id, format!("{}deg", self.class_id * 90), 0.99);
            let output = DocumentOrientationOutput {
                classifications: vec![vec![classification]],
            };
            Ok(DynTaskOutput::DocumentOrientation(output))
        }
    }

    /// Mock adapter that returns empty classification (simulates failure to classify).
    #[derive(Debug)]
    struct MockEmptyOrientationAdapter;

    impl DynModelAdapter for MockEmptyOrientationAdapter {
        fn info(&self) -> AdapterInfo {
            AdapterInfo {
                model_name: "MockEmptyOrientationAdapter".to_string(),
                version: "1.0".to_string(),
                task_type: TaskType::DocumentOrientation,
                description: "Mock adapter for testing".to_string(),
            }
        }

        fn task_type(&self) -> TaskType {
            TaskType::DocumentOrientation
        }

        fn supports_batching(&self) -> bool {
            false
        }

        fn recommended_batch_size(&self) -> usize {
            1
        }

        fn execute_dyn(&self, _input: DynTaskInput) -> Result<DynTaskOutput, OCRError> {
            let output = DocumentOrientationOutput::empty();
            Ok(DynTaskOutput::DocumentOrientation(output))
        }
    }

    /// Mock adapter that returns an error.
    #[derive(Debug)]
    struct MockFailingAdapter;

    impl DynModelAdapter for MockFailingAdapter {
        fn info(&self) -> AdapterInfo {
            AdapterInfo {
                model_name: "MockFailingAdapter".to_string(),
                version: "1.0".to_string(),
                task_type: TaskType::DocumentOrientation,
                description: "Mock adapter for testing".to_string(),
            }
        }

        fn task_type(&self) -> TaskType {
            TaskType::DocumentOrientation
        }

        fn supports_batching(&self) -> bool {
            false
        }

        fn recommended_batch_size(&self) -> usize {
            1
        }

        fn execute_dyn(&self, _input: DynTaskInput) -> Result<DynTaskOutput, OCRError> {
            Err(OCRError::invalid_input("Mock adapter failure"))
        }
    }

    #[test]
    fn test_correct_image_orientation_class_0_no_rotation() {
        let image = create_test_image(100, 200);
        let adapter: Arc<dyn DynModelAdapter> = Arc::new(MockOrientationAdapter::new(0));

        let (rotated, correction) =
            correct_image_orientation(image.clone(), &adapter).expect("should succeed");

        // class_id 0 = 0° - no rotation needed
        assert_eq!(rotated.width(), 100);
        assert_eq!(rotated.height(), 200);

        let correction = correction.expect("should have correction metadata");
        assert_eq!(correction.angle, 0.0);
        assert_eq!(correction.rotated_width, 100);
        assert_eq!(correction.rotated_height, 200);
    }

    #[test]
    fn test_correct_image_orientation_class_1_rotate_90() {
        let image = create_test_image(100, 200);
        let adapter: Arc<dyn DynModelAdapter> = Arc::new(MockOrientationAdapter::new(1));

        let (rotated, correction) =
            correct_image_orientation(image.clone(), &adapter).expect("should succeed");

        // class_id 1 = 90° - rotate270 to correct (swaps dimensions)
        assert_eq!(rotated.width(), 200);
        assert_eq!(rotated.height(), 100);

        let correction = correction.expect("should have correction metadata");
        assert_eq!(correction.angle, 90.0);
        assert_eq!(correction.rotated_width, 200);
        assert_eq!(correction.rotated_height, 100);
    }

    #[test]
    fn test_correct_image_orientation_class_2_rotate_180() {
        let image = create_test_image(100, 200);
        let adapter: Arc<dyn DynModelAdapter> = Arc::new(MockOrientationAdapter::new(2));

        let (rotated, correction) =
            correct_image_orientation(image.clone(), &adapter).expect("should succeed");

        // class_id 2 = 180° - rotate180 to correct (dimensions unchanged)
        assert_eq!(rotated.width(), 100);
        assert_eq!(rotated.height(), 200);

        let correction = correction.expect("should have correction metadata");
        assert_eq!(correction.angle, 180.0);
        assert_eq!(correction.rotated_width, 100);
        assert_eq!(correction.rotated_height, 200);
    }

    #[test]
    fn test_correct_image_orientation_class_3_rotate_270() {
        let image = create_test_image(100, 200);
        let adapter: Arc<dyn DynModelAdapter> = Arc::new(MockOrientationAdapter::new(3));

        let (rotated, correction) =
            correct_image_orientation(image.clone(), &adapter).expect("should succeed");

        // class_id 3 = 270° - rotate90 to correct (swaps dimensions)
        assert_eq!(rotated.width(), 200);
        assert_eq!(rotated.height(), 100);

        let correction = correction.expect("should have correction metadata");
        assert_eq!(correction.angle, 270.0);
        assert_eq!(correction.rotated_width, 200);
        assert_eq!(correction.rotated_height, 100);
    }

    #[test]
    fn test_correct_image_orientation_empty_classification_returns_original() {
        let image = create_test_image(100, 200);
        let adapter: Arc<dyn DynModelAdapter> = Arc::new(MockEmptyOrientationAdapter);

        let (rotated, correction) =
            correct_image_orientation(image.clone(), &adapter).expect("should succeed");

        // Empty classification should return original image unchanged
        assert_eq!(rotated.width(), 100);
        assert_eq!(rotated.height(), 200);
        assert!(correction.is_none());
    }

    #[test]
    fn test_correct_image_orientation_adapter_failure_propagates_error() {
        let image = create_test_image(100, 200);
        let adapter: Arc<dyn DynModelAdapter> = Arc::new(MockFailingAdapter);

        let result = correct_image_orientation(image, &adapter);

        assert!(result.is_err());
    }

    #[test]
    fn test_orientation_correction_metadata_accuracy() {
        // Test with a non-square image to verify dimension tracking
        let image = create_test_image(640, 480);
        let adapter: Arc<dyn DynModelAdapter> = Arc::new(MockOrientationAdapter::new(1));

        let (rotated, correction) =
            correct_image_orientation(image, &adapter).expect("should succeed");

        let correction = correction.expect("should have correction");

        // After 90° correction (rotate270), 640x480 becomes 480x640
        assert_eq!(rotated.width(), 480);
        assert_eq!(rotated.height(), 640);
        assert_eq!(correction.rotated_width, 480);
        assert_eq!(correction.rotated_height, 640);
        assert_eq!(correction.angle, 90.0);
    }

    #[test]
    fn test_document_preprocessor_no_adapters() {
        let preprocessor = DocumentPreprocessor::new(None, None);
        let image = create_test_image(100, 200);

        let result = preprocessor.preprocess(image).expect("should succeed");

        assert_eq!(result.image.width(), 100);
        assert_eq!(result.image.height(), 200);
        assert!(result.orientation_angle.is_none());
        assert!(result.rotation.is_none());
        assert!(result.rectified_img.is_none());
    }

    #[test]
    fn test_document_preprocessor_with_orientation_adapter() {
        let orientation_adapter: Arc<dyn DynModelAdapter> =
            Arc::new(MockOrientationAdapter::new(1));
        let preprocessor = DocumentPreprocessor::new(Some(orientation_adapter), None);
        let image = create_test_image(100, 200);

        let result = preprocessor.preprocess(image).expect("should succeed");

        // Should be rotated (90° correction)
        assert_eq!(result.image.width(), 200);
        assert_eq!(result.image.height(), 100);
        assert_eq!(result.orientation_angle, Some(90.0));
        assert!(result.rotation.is_some());
        assert!(result.rectified_img.is_none());

        let rotation = result.rotation.unwrap();
        assert_eq!(rotation.angle, 90.0);
        assert_eq!(rotation.rotated_width, 200);
        assert_eq!(rotation.rotated_height, 100);
    }

    #[test]
    fn test_orientation_correction_struct_equality() {
        let c1 = OrientationCorrection {
            angle: 90.0,
            rotated_width: 200,
            rotated_height: 100,
        };
        let c2 = OrientationCorrection {
            angle: 90.0,
            rotated_width: 200,
            rotated_height: 100,
        };
        let c3 = OrientationCorrection {
            angle: 180.0,
            rotated_width: 200,
            rotated_height: 100,
        };

        assert_eq!(c1, c2);
        assert_ne!(c1, c3);
    }

    #[test]
    fn test_orientation_correction_copy_semantics() {
        let original = OrientationCorrection {
            angle: 90.0,
            rotated_width: 200,
            rotated_height: 100,
        };
        let copied = original;

        // Both should be usable after copy
        assert_eq!(original.angle, 90.0);
        assert_eq!(copied.angle, 90.0);
    }

    #[test]
    fn test_angle_calculation_formula() {
        // Verify that angle = class_id * 90.0
        for class_id in 0..4 {
            let expected_angle = (class_id as f32) * 90.0;
            let image = create_test_image(100, 100);
            let adapter: Arc<dyn DynModelAdapter> = Arc::new(MockOrientationAdapter::new(class_id));

            let (_, correction) =
                correct_image_orientation(image, &adapter).expect("should succeed");

            let correction = correction.expect("should have correction");
            assert_eq!(
                correction.angle, expected_angle,
                "class_id {} should produce angle {}",
                class_id, expected_angle
            );
        }
    }

    #[test]
    fn test_square_image_rotation_dimensions() {
        // Square images should have same dimensions after any rotation
        let image = create_test_image(256, 256);

        for class_id in 0..4 {
            let adapter: Arc<dyn DynModelAdapter> = Arc::new(MockOrientationAdapter::new(class_id));
            let (rotated, _) =
                correct_image_orientation(image.clone(), &adapter).expect("should succeed");

            assert_eq!(rotated.width(), 256, "class_id {} width mismatch", class_id);
            assert_eq!(
                rotated.height(),
                256,
                "class_id {} height mismatch",
                class_id
            );
        }
    }
}
