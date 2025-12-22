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
