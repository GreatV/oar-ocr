//! Bounding box based image cropping utilities.

use crate::core::OCRError;
use crate::processors::BoundingBox;
use crate::utils::transform::get_rotate_crop_image;
use image::{RgbImage, imageops};

/// Bounding box based image cropping utilities.
pub struct BBoxCrop;

impl BBoxCrop {
    /// Crops an image based on a bounding box.
    ///
    /// This function calculates the bounding rectangle of a polygonal bounding box
    /// and crops the image to that region. It handles edge cases like empty bounding
    /// boxes and ensures the crop region is within the image boundaries.
    ///
    /// # Arguments
    ///
    /// * `image` - The source image
    /// * `bbox` - The bounding box defining the crop region
    ///
    /// # Returns
    ///
    /// A Result containing the cropped image or an OCRError
    pub fn crop_bounding_box(image: &RgbImage, bbox: &BoundingBox) -> Result<RgbImage, OCRError> {
        // Check if the bounding box is empty
        if bbox.points.is_empty() {
            return Err(OCRError::image_processing_error("Empty bounding box"));
        }

        // Calculate the bounding rectangle of the polygon
        let min_x = bbox
            .points
            .iter()
            .map(|p| p.x)
            .fold(f32::INFINITY, f32::min)
            .max(0.0);
        let max_x = bbox
            .points
            .iter()
            .map(|p| p.x)
            .fold(f32::NEG_INFINITY, f32::max);
        let min_y = bbox
            .points
            .iter()
            .map(|p| p.y)
            .fold(f32::INFINITY, f32::min)
            .max(0.0);
        let max_y = bbox
            .points
            .iter()
            .map(|p| p.y)
            .fold(f32::NEG_INFINITY, f32::max);

        // Convert to integer coordinates, ensuring they're within image bounds
        let x1 = (min_x as u32).min(image.width().saturating_sub(1));
        let y1 = (min_y as u32).min(image.height().saturating_sub(1));
        let x2 = (max_x as u32).min(image.width());
        let y2 = (max_y as u32).min(image.height());

        // Validate the crop region
        if x2 <= x1 || y2 <= y1 {
            return Err(OCRError::image_processing_error(format!(
                "Invalid crop region: ({x1}, {y1}) to ({x2}, {y2})"
            )));
        }

        let coords = (x1, y1, x2, y2);
        Ok(Self::slice_rgb_image(image, coords))
    }

    /// Slices an RGB image based on coordinates.
    ///
    /// This function creates a new image by copying pixels from a rectangular
    /// region of the source image. It performs bounds checking to ensure
    /// that only valid pixels are copied.
    ///
    /// # Arguments
    ///
    /// * `img` - The source image
    /// * `coords` - The coordinates as (x1, y1, x2, y2)
    ///
    /// # Returns
    ///
    /// The sliced image
    fn slice_rgb_image(img: &RgbImage, coords: (u32, u32, u32, u32)) -> RgbImage {
        let (x1, y1, x2, y2) = coords;
        let width = x2 - x1;
        let height = y2 - y1;
        // Use library-provided immutable crop (zero-copy view) and then materialize
        imageops::crop_imm(img, x1, y1, width, height).to_image()
    }

    /// Crops multiple bounding boxes from the same source image.
    ///
    /// Processes all bounding boxes for batch cropping operations, such as extracting
    /// multiple text regions from a document image.
    ///
    /// # Arguments
    ///
    /// * `image` - The source image
    /// * `bboxes` - A slice of bounding boxes to crop
    ///
    /// # Returns
    ///
    /// A vector of Results, each containing either a cropped image or an OCRError.
    /// The order corresponds to the input bounding boxes.
    pub fn batch_crop_bounding_boxes(
        image: &RgbImage,
        bboxes: &[BoundingBox],
    ) -> Vec<Result<RgbImage, OCRError>> {
        bboxes
            .iter()
            .map(|bbox| Self::crop_bounding_box(image, bbox))
            .collect()
    }

    /// Crops multiple rotated bounding boxes from the same source image.
    ///
    /// Processes batch cropping operations with perspective correction.
    ///
    /// # Arguments
    ///
    /// * `image` - The source image
    /// * `bboxes` - A slice of bounding boxes to crop with rotation
    ///
    /// # Returns
    ///
    /// A vector of Results, each containing either a cropped image or an OCRError.
    /// The order corresponds to the input bounding boxes.
    pub fn batch_crop_rotated_bounding_boxes(
        image: &RgbImage,
        bboxes: &[BoundingBox],
    ) -> Vec<Result<RgbImage, OCRError>> {
        bboxes
            .iter()
            .map(|bbox| get_rotate_crop_image(image, &bbox.points))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::processors::Point;
    use image::{ImageBuffer, Rgb};

    fn create_test_image(width: u32, height: u32) -> RgbImage {
        let mut img = ImageBuffer::new(width, height);
        for y in 0..height {
            for x in 0..width {
                // Create a pattern for testing
                let r = (x * 255 / width.max(1)) as u8;
                let g = (y * 255 / height.max(1)) as u8;
                let b = 128;
                img.put_pixel(x, y, Rgb([r, g, b]));
            }
        }
        img
    }

    #[test]
    fn test_crop_bounding_box_valid_rectangle() {
        let img = create_test_image(100, 100);
        let bbox = BoundingBox {
            points: vec![
                Point { x: 10.0, y: 10.0 },
                Point { x: 50.0, y: 10.0 },
                Point { x: 50.0, y: 40.0 },
                Point { x: 10.0, y: 40.0 },
            ],
        };

        let result = BBoxCrop::crop_bounding_box(&img, &bbox);
        assert!(result.is_ok());

        let cropped = match result {
            Ok(cropped) => cropped,
            Err(err) => panic!("expected crop to succeed: {err}"),
        };
        assert_eq!(cropped.width(), 40); // 50 - 10
        assert_eq!(cropped.height(), 30); // 40 - 10
    }

    #[test]
    fn test_crop_bounding_box_empty_points() {
        let img = create_test_image(100, 100);
        let bbox = BoundingBox { points: vec![] };

        let result = BBoxCrop::crop_bounding_box(&img, &bbox);
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Empty bounding box"));
    }

    #[test]
    fn test_crop_bounding_box_single_point() {
        let img = create_test_image(100, 100);
        let bbox = BoundingBox {
            points: vec![Point { x: 50.0, y: 50.0 }],
        };

        let result = BBoxCrop::crop_bounding_box(&img, &bbox);
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Invalid crop region"));
    }

    #[test]
    fn test_crop_bounding_box_negative_coordinates() {
        let img = create_test_image(100, 100);
        let bbox = BoundingBox {
            points: vec![
                Point { x: -10.0, y: -5.0 },
                Point { x: 30.0, y: -5.0 },
                Point { x: 30.0, y: 25.0 },
                Point { x: -10.0, y: 25.0 },
            ],
        };

        let result = BBoxCrop::crop_bounding_box(&img, &bbox);
        assert!(result.is_ok());

        let cropped = match result {
            Ok(cropped) => cropped,
            Err(err) => panic!("expected crop to succeed: {err}"),
        };
        // Should clamp negative coordinates to 0
        assert_eq!(cropped.width(), 30); // 30 - 0 (clamped from -10)
        assert_eq!(cropped.height(), 25); // 25 - 0 (clamped from -5)
    }

    #[test]
    fn test_crop_bounding_box_out_of_bounds() {
        let img = create_test_image(100, 100);
        let bbox = BoundingBox {
            points: vec![
                Point { x: 80.0, y: 80.0 },
                Point { x: 150.0, y: 80.0 },  // Beyond image width
                Point { x: 150.0, y: 120.0 }, // Beyond image height
                Point { x: 80.0, y: 120.0 },
            ],
        };

        let result = BBoxCrop::crop_bounding_box(&img, &bbox);
        assert!(result.is_ok());

        let cropped = match result {
            Ok(cropped) => cropped,
            Err(err) => panic!("expected crop to succeed: {err}"),
        };
        // Should clamp to image boundaries
        assert_eq!(cropped.width(), 20); // 100 - 80
        assert_eq!(cropped.height(), 20); // 100 - 80
    }

    #[test]
    fn test_crop_bounding_box_irregular_polygon() {
        let img = create_test_image(100, 100);
        let bbox = BoundingBox {
            points: vec![
                Point { x: 20.0, y: 30.0 },
                Point { x: 60.0, y: 10.0 },
                Point { x: 80.0, y: 50.0 },
                Point { x: 40.0, y: 70.0 },
                Point { x: 10.0, y: 40.0 },
            ],
        };

        let result = BBoxCrop::crop_bounding_box(&img, &bbox);
        assert!(result.is_ok());

        let cropped = match result {
            Ok(cropped) => cropped,
            Err(err) => panic!("expected crop to succeed: {err}"),
        };
        // Should use bounding rectangle of the polygon
        assert_eq!(cropped.width(), 70); // 80 - 10
        assert_eq!(cropped.height(), 60); // 70 - 10
    }

    #[test]
    fn test_batch_crop_rotated_bounding_boxes_valid() {
        let img = create_test_image(100, 100);
        let bbox = BoundingBox {
            points: vec![
                Point { x: 20.0, y: 20.0 },
                Point { x: 60.0, y: 20.0 },
                Point { x: 60.0, y: 40.0 },
                Point { x: 20.0, y: 40.0 },
            ],
        };

        let mut results = BBoxCrop::batch_crop_rotated_bounding_boxes(&img, &[bbox]);
        assert_eq!(results.len(), 1);
        let result = results.remove(0);

        let cropped = match result {
            Ok(cropped) => cropped,
            Err(err) => panic!("expected crop to succeed: {err}"),
        };
        assert!(cropped.width() > 0);
        assert!(cropped.height() > 0);
    }

    #[test]
    fn test_batch_crop_rotated_bounding_boxes_wrong_point_count() {
        let img = create_test_image(100, 100);
        let bbox = BoundingBox {
            points: vec![
                Point { x: 20.0, y: 20.0 },
                Point { x: 60.0, y: 20.0 },
                Point { x: 60.0, y: 40.0 },
            ], // Only 3 points instead of 4
        };

        let mut results = BBoxCrop::batch_crop_rotated_bounding_boxes(&img, &[bbox]);
        assert_eq!(results.len(), 1);
        let result = results.remove(0);
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("Box must contain exactly 4 points"));
    }

    #[test]
    fn test_get_rotate_crop_image_axis_aligned_wide_box() {
        let img = create_test_image(100, 100);
        // Define an axis-aligned rectangle with 4 points
        let bbox = BoundingBox {
            points: vec![
                Point { x: 10.0, y: 20.0 },
                Point { x: 60.0, y: 20.0 },
                Point { x: 60.0, y: 50.0 },
                Point { x: 10.0, y: 50.0 },
            ],
        };
        let cropped_fast = match get_rotate_crop_image(&img, &bbox.points) {
            Ok(cropped_fast) => cropped_fast,
            Err(err) => panic!("expected rotated crop to succeed: {err}"),
        };
        assert_eq!(cropped_fast.dimensions(), (50, 30));
    }

    #[test]
    fn test_get_rotate_crop_image_axis_aligned_tall_box_rotates() {
        let img = create_test_image(100, 100);
        let bbox = BoundingBox {
            points: vec![
                Point { x: 10.0, y: 20.0 },
                Point { x: 30.0, y: 20.0 },
                Point { x: 30.0, y: 80.0 },
                Point { x: 10.0, y: 80.0 },
            ],
        };

        let cropped = match get_rotate_crop_image(&img, &bbox.points) {
            Ok(cropped) => cropped,
            Err(err) => panic!("expected rotated crop to succeed: {err}"),
        };
        assert_eq!(cropped.dimensions(), (60, 20));
    }

    #[test]
    fn test_slice_rgb_image() {
        let img = create_test_image(100, 100);
        let coords = (10, 20, 50, 60);

        let sliced = BBoxCrop::slice_rgb_image(&img, coords);
        assert_eq!(sliced.width(), 40); // 50 - 10
        assert_eq!(sliced.height(), 40); // 60 - 20

        // Check that the pixel values are correctly copied
        let original_pixel = img.get_pixel(10, 20);
        let sliced_pixel = sliced.get_pixel(0, 0);
        assert_eq!(original_pixel, sliced_pixel);
    }
}
