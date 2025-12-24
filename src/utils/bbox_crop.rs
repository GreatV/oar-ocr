//! Bounding box based image cropping utilities.

use crate::core::OCRError;
use crate::processors::BoundingBox;
use crate::utils::transform::{Point2f, get_rotate_crop_image};
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
            .map(|bbox| Self::crop_rotated_bounding_box(image, bbox))
            .collect()
    }

    /// Crops and rectifies an image region using rotated crop with perspective transformation.
    ///
    /// This function implements the same functionality as OpenCV's GetRotateCropImage.
    /// It takes a bounding box (quadrilateral) and applies perspective transformation
    /// to rectify it into a rectangular image. This is particularly useful for text
    /// regions that may be rotated or have perspective distortion.
    ///
    /// # Arguments
    ///
    /// * `image` - The source image
    /// * `bbox` - The bounding box defining the quadrilateral region
    ///
    /// # Returns
    ///
    /// A Result containing the rotated and cropped image or an OCRError
    pub fn crop_rotated_bounding_box(
        image: &RgbImage,
        bbox: &BoundingBox,
    ) -> Result<RgbImage, OCRError> {
        // Check if the bounding box has exactly 4 points
        if bbox.points.len() != 4 {
            return Err(OCRError::image_processing_error(format!(
                "Bounding box must have exactly 4 points, got {}",
                bbox.points.len()
            )));
        }

        // Convert BoundingBox points to Point2f
        let box_points: Vec<Point2f> = bbox.points.iter().map(|p| Point2f::new(p.x, p.y)).collect();

        // Fast path: if the quadrilateral is axis-aligned rectangle, use simple crop
        if let [p0, p1, p2, p3] = &box_points[..] {
            let is_axis_aligned = (p0.y == p1.y && p2.y == p3.y && p0.x == p3.x && p1.x == p2.x)
                || (p0.x == p1.x && p2.x == p3.x && p0.y == p3.y && p1.y == p2.y);
            if is_axis_aligned {
                let min_x = p0.x.min(p1.x).min(p2.x).min(p3.x).max(0.0) as u32;
                let min_y = p0.y.min(p1.y).min(p2.y).min(p3.y).max(0.0) as u32;
                let max_x = p0.x.max(p1.x).max(p2.x).max(p3.x).min(image.width() as f32) as u32;
                let max_y =
                    p0.y.max(p1.y)
                        .max(p2.y)
                        .max(p3.y)
                        .min(image.height() as f32) as u32;
                if max_x > min_x && max_y > min_y {
                    use image::imageops;
                    let w = max_x - min_x;
                    let h = max_y - min_y;
                    return Ok(imageops::crop_imm(image, min_x, min_y, w, h).to_image());
                }
            }
        }

        // Apply rotated crop transformation
        get_rotate_crop_image(image, &box_points)
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

        let cropped = result.unwrap();
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

        let cropped = result.unwrap();
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

        let cropped = result.unwrap();
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

        let cropped = result.unwrap();
        // Should use bounding rectangle of the polygon
        assert_eq!(cropped.width(), 70); // 80 - 10
        assert_eq!(cropped.height(), 60); // 70 - 10
    }

    #[test]
    fn test_crop_rotated_bounding_box_valid() {
        let img = create_test_image(100, 100);
        let bbox = BoundingBox {
            points: vec![
                Point { x: 20.0, y: 20.0 },
                Point { x: 60.0, y: 20.0 },
                Point { x: 60.0, y: 40.0 },
                Point { x: 20.0, y: 40.0 },
            ],
        };

        let result = BBoxCrop::crop_rotated_bounding_box(&img, &bbox);
        assert!(result.is_ok());

        let cropped = result.unwrap();
        assert!(cropped.width() > 0);
        assert!(cropped.height() > 0);
    }

    #[test]
    fn test_crop_rotated_bounding_box_wrong_point_count() {
        let img = create_test_image(100, 100);
        let bbox = BoundingBox {
            points: vec![
                Point { x: 20.0, y: 20.0 },
                Point { x: 60.0, y: 20.0 },
                Point { x: 60.0, y: 40.0 },
            ], // Only 3 points instead of 4
        };

        let result = BBoxCrop::crop_rotated_bounding_box(&img, &bbox);
        assert!(result.is_err());

        let error_msg = result.unwrap_err().to_string();
        assert!(error_msg.contains("must have exactly 4 points"));
    }

    #[test]
    fn test_crop_rotated_bounding_box_axis_aligned_fast_path() {
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
        let cropped_fast = BBoxCrop::crop_rotated_bounding_box(&img, &bbox).unwrap();
        // Expected via simple crop
        let expected = imageops::crop_imm(&img, 10, 20, 50, 30).to_image();
        assert_eq!(cropped_fast.dimensions(), expected.dimensions());
        // Sample a couple of pixels to ensure identical content
        assert_eq!(cropped_fast.get_pixel(0, 0), expected.get_pixel(0, 0));
        assert_eq!(cropped_fast.get_pixel(49, 29), expected.get_pixel(49, 29));
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
