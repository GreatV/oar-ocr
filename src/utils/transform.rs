//! Image transformation utilities for OCR processing.
//!
//! This module provides functions for perspective transformation and image warping,
//! which are essential for correcting skewed text regions in images.

use crate::core::OCRError;
use image::{Rgb, RgbImage, imageops};
use nalgebra::{Matrix3, Vector3};
use rayon::prelude::*;
use tracing::debug;

/// A 2D point with floating-point coordinates.
#[derive(Debug, Clone, Copy)]
pub struct Point2f {
    /// X coordinate of the point.
    pub x: f32,
    /// Y coordinate of the point.
    pub y: f32,
}

impl Point2f {
    /// Creates a new Point2f with the given coordinates.
    ///
    /// # Arguments
    ///
    /// * `x` - X coordinate
    /// * `y` - Y coordinate
    ///
    /// # Returns
    ///
    /// A new Point2f instance.
    pub fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }
}

/// Calculates the Euclidean distance between two points.
///
/// # Arguments
///
/// * `p1` - First point
/// * `p2` - Second point
///
/// # Returns
///
/// The distance between the two points.
fn distance(p1: &Point2f, p2: &Point2f) -> f32 {
    (p1.x - p2.x).hypot(p1.y - p2.y)
}

/// Extracts a rotated and cropped image from a source image based on bounding box points.
///
/// This function takes a source image and a set of four points that define a quadrilateral
/// region in the image. It crops the image to the bounding box of these points, then applies
/// a perspective transformation to produce a rectified image of the region. If the resulting
/// image has an aspect ratio that suggests it's rotated, it will be automatically rotated.
///
/// # Arguments
///
/// * `src_image` - The source image to crop from
/// * `box_points` - Array of exactly 4 points defining the quadrilateral region
///
/// # Returns
///
/// A Result containing the cropped and transformed image, or an OCRError if the operation fails.
///
/// # Errors
///
/// Returns an OCRError if:
/// * The box_points array doesn't contain exactly 4 points
/// * The calculated crop region is invalid
/// * The calculated crop dimensions are zero
/// * The perspective transformation fails
pub fn get_rotate_crop_image(
    src_image: &RgbImage,
    box_points: &[Point2f],
) -> Result<RgbImage, OCRError> {
    // Validate input
    if box_points.len() != 4 {
        return Err(OCRError::InvalidInput {
            message: "Box must contain exactly 4 points".to_string(),
        });
    }

    // Find bounding box of the points
    let mut min_x = f32::INFINITY;
    let mut max_x = f32::NEG_INFINITY;
    let mut min_y = f32::INFINITY;
    let mut max_y = f32::NEG_INFINITY;

    for p in box_points {
        min_x = min_x.min(p.x);
        max_x = max_x.max(p.x);
        min_y = min_y.min(p.y);
        max_y = max_y.max(p.y);
    }

    // Calculate crop boundaries, clamping to image dimensions
    let left = min_x.max(0.0) as u32;
    let top = min_y.max(0.0) as u32;
    let right = max_x.min(src_image.width() as f32) as u32;
    let bottom = max_y.min(src_image.height() as f32) as u32;

    // Validate crop region
    if right <= left || bottom <= top {
        return Err(OCRError::InvalidInput {
            message: "Invalid crop region".to_string(),
        });
    }

    // Perform initial crop
    let crop_width = right - left;
    let crop_height = bottom - top;
    let img_crop = imageops::crop_imm(src_image, left, top, crop_width, crop_height).to_image();

    // Adjust points relative to the cropped image
    let points: Vec<Point2f> = box_points
        .iter()
        .map(|p| Point2f::new(p.x - left as f32, p.y - top as f32))
        .collect();

    // Reorder points to (top-left, top-right, bottom-right, bottom-left)
    // to keep width/height estimation stable when point order varies.
    let mut sorted = points.clone();
    sorted.sort_by(|a, b| a.x.partial_cmp(&b.x).unwrap_or(std::cmp::Ordering::Equal));
    let (mut index_a, mut index_d) = (0usize, 1usize);
    if sorted[1].y < sorted[0].y {
        index_a = 1;
        index_d = 0;
    }
    let (mut index_b, mut index_c) = (2usize, 3usize);
    if sorted[3].y < sorted[2].y {
        index_b = 3;
        index_c = 2;
    }
    let ordered = [
        sorted[index_a],
        sorted[index_b],
        sorted[index_c],
        sorted[index_d],
    ];

    // Calculate target image dimensions based on the max opposite-edge lengths
    let width1 = distance(&ordered[0], &ordered[1]);
    let width2 = distance(&ordered[2], &ordered[3]);
    let img_crop_width = width1.max(width2).round() as u32;

    let height1 = distance(&ordered[0], &ordered[3]);
    let height2 = distance(&ordered[1], &ordered[2]);
    let img_crop_height = height1.max(height2).round() as u32;

    // Validate target dimensions
    if img_crop_width == 0 || img_crop_height == 0 {
        return Err(OCRError::InvalidInput {
            message: "Invalid crop dimensions".to_string(),
        });
    }

    // Define standard points for the target rectangle
    let pts_std = [
        Point2f::new(0.0, 0.0),
        Point2f::new(img_crop_width as f32, 0.0),
        Point2f::new(img_crop_width as f32, img_crop_height as f32),
        Point2f::new(0.0, img_crop_height as f32),
    ];

    // Calculate perspective transformation matrix
    let transform_matrix = get_perspective_transform(&ordered, &pts_std)?;

    // Apply perspective transformation
    let dst_img = warp_perspective(
        &img_crop,
        &transform_matrix,
        img_crop_width,
        img_crop_height,
    )?;

    // Automatically rotate if the aspect ratio suggests the text is vertical
    if dst_img.height() as f32 >= dst_img.width() as f32 * 1.5 {
        debug!(
            "Rotating image due to aspect ratio: {}x{}",
            dst_img.width(),
            dst_img.height()
        );

        Ok(imageops::rotate270(&dst_img))
    } else {
        Ok(dst_img)
    }
}

/// Calculates the perspective transformation matrix that maps source points to destination points.
///
/// This function solves the linear system of equations to find the perspective transformation
/// matrix that maps four source points to four destination points.
///
/// # Arguments
///
/// * `src_points` - Array of exactly 4 source points
/// * `dst_points` - Array of exactly 4 destination points
///
/// # Returns
///
/// A Result containing the 3x3 transformation matrix, or an OCRError if the operation fails.
///
/// # Errors
///
/// Returns an OCRError if:
/// * Either array doesn't contain exactly 4 points
/// * The linear system cannot be solved
fn get_perspective_transform(
    src_points: &[Point2f],
    dst_points: &[Point2f],
) -> Result<Matrix3<f32>, OCRError> {
    // Validate input
    if src_points.len() != 4 || dst_points.len() != 4 {
        return Err(OCRError::InvalidInput {
            message: "Need exactly 4 points for perspective transformation".to_string(),
        });
    }

    // Set up the linear system of equations
    let mut a = nalgebra::DMatrix::<f32>::zeros(8, 8);
    let mut b = nalgebra::DVector::<f32>::zeros(8);

    // Fill the matrix A and vector b with the equations for perspective transformation
    for i in 0..4 {
        let src = &src_points[i];
        let dst = &dst_points[i];

        // First equation for x coordinate transformation
        a.set_row(
            i * 2,
            &nalgebra::RowDVector::from_row_slice(&[
                src.x,
                src.y,
                1.0,
                0.0,
                0.0,
                0.0,
                -src.x * dst.x,
                -src.y * dst.x,
            ]),
        );
        b[i * 2] = dst.x;

        // Second equation for y coordinate transformation
        a.set_row(
            i * 2 + 1,
            &nalgebra::RowDVector::from_row_slice(&[
                0.0,
                0.0,
                0.0,
                src.x,
                src.y,
                1.0,
                -src.x * dst.y,
                -src.y * dst.y,
            ]),
        );
        b[i * 2 + 1] = dst.y;
    }

    // Solve the linear system to find the transformation parameters
    let decomp = a.lu();
    let solution = decomp.solve(&b).ok_or_else(|| OCRError::InvalidInput {
        message: "Cannot solve perspective transformation".to_string(),
    })?;

    // Construct the 3x3 transformation matrix
    Ok(Matrix3::new(
        solution[0],
        solution[1],
        solution[2],
        solution[3],
        solution[4],
        solution[5],
        solution[6],
        solution[7],
        1.0,
    ))
}

/// Applies a perspective transformation to an image.
///
/// This function transforms an image using a given perspective transformation matrix.
/// It uses inverse mapping with bilinear interpolation to produce the output image.
///
/// # Arguments
///
/// * `src_image` - The source image to transform
/// * `transform_matrix` - The 3x3 perspective transformation matrix
/// * `dst_width` - Width of the output image
/// * `dst_height` - Height of the output image
///
/// # Returns
///
/// A Result containing the transformed image, or an OCRError if the operation fails.
///
/// # Errors
///
/// Returns an OCRError if:
/// * The transformation matrix cannot be inverted
fn warp_perspective(
    src_image: &RgbImage,
    transform_matrix: &Matrix3<f32>,
    dst_width: u32,
    dst_height: u32,
) -> Result<RgbImage, OCRError> {
    // Calculate the inverse transformation matrix for inverse mapping
    let inv_matrix = transform_matrix
        .try_inverse()
        .ok_or_else(|| OCRError::InvalidInput {
            message: "Cannot invert transformation matrix".to_string(),
        })?;

    // Create the destination image
    let mut dst_image = RgbImage::new(dst_width, dst_height);
    let buffer: &mut [u8] = dst_image.as_mut();

    // Process rows with a small-image sequential fast path to avoid rayon overhead
    // Use bicubic interpolation with border replication (matches cv2.warpPerspective
    // with flags=INTER_CUBIC and borderMode=BORDER_REPLICATE)
    if dst_height <= 1 {
        let row_buffer = &mut buffer[0..(dst_width * 3) as usize];
        let dst_y = 0u32;
        for dst_x in 0..dst_width {
            let dst_point = Vector3::new(dst_x as f32, dst_y as f32, 1.0);
            let src_point = inv_matrix * dst_point;
            let final_pixel = if src_point.z.abs() > f32::EPSILON {
                let src_x = src_point.x / src_point.z;
                let src_y = src_point.y / src_point.z;
                // bicubic_interpolate handles out-of-bounds via border replication
                bicubic_interpolate(src_image, src_x, src_y)
            } else {
                // Degenerate case: replicate top-left corner pixel
                *src_image.get_pixel(0, 0)
            };
            let index = (dst_x * 3) as usize;
            row_buffer[index..index + 3].copy_from_slice(&final_pixel.0);
        }
    } else {
        buffer
            .par_chunks_mut((dst_width * 3) as usize)
            .enumerate()
            .for_each(|(dst_y, row_buffer)| {
                for dst_x in 0..dst_width {
                    let dst_point = Vector3::new(dst_x as f32, dst_y as f32, 1.0);
                    let src_point = inv_matrix * dst_point;
                    let final_pixel = if src_point.z.abs() > f32::EPSILON {
                        let src_x = src_point.x / src_point.z;
                        let src_y = src_point.y / src_point.z;
                        // bicubic_interpolate handles out-of-bounds via border replication
                        bicubic_interpolate(src_image, src_x, src_y)
                    } else {
                        // Degenerate case: replicate top-left corner pixel
                        *src_image.get_pixel(0, 0)
                    };
                    let index = (dst_x * 3) as usize;
                    row_buffer[index..index + 3].copy_from_slice(&final_pixel.0);
                }
            });
    }

    Ok(dst_image)
}

/// Gets a pixel value with border replication for out-of-bounds coordinates.
///
/// This function implements OpenCV's BORDER_REPLICATE behavior:
/// when coordinates are outside the image, the nearest edge pixel is used.
///
/// # Arguments
///
/// * `image` - The source image
/// * `x` - X coordinate (can be negative or >= width)
/// * `y` - Y coordinate (can be negative or >= height)
///
/// # Returns
///
/// The pixel value at the clamped coordinates.
#[inline]
fn get_pixel_replicate(image: &RgbImage, x: i32, y: i32) -> Rgb<u8> {
    let clamped_x = x.clamp(0, image.width() as i32 - 1) as u32;
    let clamped_y = y.clamp(0, image.height() as i32 - 1) as u32;
    *image.get_pixel(clamped_x, clamped_y)
}

/// Cubic interpolation kernel function.
///
/// This implements the standard cubic convolution kernel used in bicubic interpolation.
/// The kernel is defined as:
/// - For |t| <= 1: (a+2)|t|³ - (a+3)|t|² + 1
/// - For 1 < |t| < 2: a|t|³ - 5a|t|² + 8a|t| - 4a
/// - Otherwise: 0
///
/// Where a = -0.5 (Catmull-Rom spline, same as OpenCV's default)
#[inline]
fn cubic_kernel(t: f32) -> f32 {
    const A: f32 = -0.5; // Catmull-Rom spline coefficient (OpenCV default)
    let t_abs = t.abs();

    if t_abs <= 1.0 {
        (A + 2.0) * t_abs * t_abs * t_abs - (A + 3.0) * t_abs * t_abs + 1.0
    } else if t_abs < 2.0 {
        A * t_abs * t_abs * t_abs - 5.0 * A * t_abs * t_abs + 8.0 * A * t_abs - 4.0 * A
    } else {
        0.0
    }
}

/// Performs bicubic interpolation to get a pixel value at non-integer coordinates.
///
/// This function calculates the pixel value at a fractional (x, y) coordinate
/// by interpolating using a 4x4 neighborhood of pixels with cubic convolution.
/// Uses border replication for edge handling (same as OpenCV's BORDER_REPLICATE).
///
/// # Arguments
///
/// * `image` - The source image
/// * `x` - X coordinate (can be fractional)
/// * `y` - Y coordinate (can be fractional)
///
/// # Returns
///
/// The interpolated pixel value.
fn bicubic_interpolate(image: &RgbImage, x: f32, y: f32) -> Rgb<u8> {
    let x_int = x.floor() as i32;
    let y_int = y.floor() as i32;
    let dx = x - x_int as f32;
    let dy = y - y_int as f32;

    // Calculate x-direction weights
    let wx = [
        cubic_kernel(dx + 1.0),
        cubic_kernel(dx),
        cubic_kernel(dx - 1.0),
        cubic_kernel(dx - 2.0),
    ];

    // Calculate y-direction weights
    let wy = [
        cubic_kernel(dy + 1.0),
        cubic_kernel(dy),
        cubic_kernel(dy - 1.0),
        cubic_kernel(dy - 2.0),
    ];

    let mut result = [0.0f32; 3];

    // Sample 4x4 neighborhood
    for (j, &weight_y) in wy.iter().enumerate() {
        let sample_y = y_int - 1 + j as i32;

        for (i, &weight_x) in wx.iter().enumerate() {
            let sample_x = x_int - 1 + i as i32;
            let weight = weight_x * weight_y;

            // Use border replication for out-of-bounds pixels
            let pixel = get_pixel_replicate(image, sample_x, sample_y);

            for (c, result_c) in result.iter_mut().enumerate().take(3) {
                *result_c += weight * pixel.0[c] as f32;
            }
        }
    }

    // Clamp and convert to u8
    Rgb([
        result[0].round().clamp(0.0, 255.0) as u8,
        result[1].round().clamp(0.0, 255.0) as u8,
        result[2].round().clamp(0.0, 255.0) as u8,
    ])
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Performs bilinear interpolation to get a pixel value at non-integer coordinates.
    ///
    /// This function calculates the pixel value at a fractional (x, y) coordinate
    /// by interpolating between the four nearest pixels.
    /// Uses border replication for edge handling (same as OpenCV's BORDER_REPLICATE).
    fn bilinear_interpolate(image: &RgbImage, x: f32, y: f32) -> Rgb<u8> {
        let x_int = x.floor() as i32;
        let y_int = y.floor() as i32;

        // Calculate the fractional parts
        let dx = x - x_int as f32;
        let dy = y - y_int as f32;

        // Get the four neighboring pixels with border replication
        let p11 = get_pixel_replicate(image, x_int, y_int);
        let p12 = get_pixel_replicate(image, x_int, y_int + 1);
        let p21 = get_pixel_replicate(image, x_int + 1, y_int);
        let p22 = get_pixel_replicate(image, x_int + 1, y_int + 1);

        // Interpolate each color channel
        let mut result = [0u8; 3];
        for (i, result_channel) in result.iter_mut().enumerate() {
            let val = (1.0 - dx) * (1.0 - dy) * p11.0[i] as f32
                + dx * (1.0 - dy) * p21.0[i] as f32
                + (1.0 - dx) * dy * p12.0[i] as f32
                + dx * dy * p22.0[i] as f32;
            *result_channel = val.round().clamp(0.0, 255.0) as u8;
        }

        Rgb(result)
    }

    #[test]
    fn test_distance() {
        let p1 = Point2f::new(0.0, 0.0);
        let p2 = Point2f::new(3.0, 4.0);
        let dist = distance(&p1, &p2);
        assert_eq!(dist, 5.0);
    }

    #[test]
    fn test_get_perspective_transform() {
        // Define a simple square in source and destination
        let src_points = [
            Point2f::new(0.0, 0.0),
            Point2f::new(1.0, 0.0),
            Point2f::new(1.0, 1.0),
            Point2f::new(0.0, 1.0),
        ];

        let dst_points = [
            Point2f::new(0.0, 0.0),
            Point2f::new(2.0, 0.0),
            Point2f::new(2.0, 2.0),
            Point2f::new(0.0, 2.0),
        ];

        let transform = get_perspective_transform(&src_points, &dst_points).unwrap();

        // Check that the transformation matrix is valid (all elements are finite)
        assert!(transform.iter().all(|&x| x.is_finite()));
    }

    #[test]
    fn test_get_perspective_transform_invalid_input() {
        // Test with wrong number of points
        let src_points = [Point2f::new(0.0, 0.0), Point2f::new(1.0, 0.0)];

        let dst_points = [
            Point2f::new(0.0, 0.0),
            Point2f::new(2.0, 0.0),
            Point2f::new(2.0, 2.0),
            Point2f::new(0.0, 2.0),
        ];

        let result = get_perspective_transform(&src_points, &dst_points);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_rotate_crop_image_invalid_points() {
        // Create a simple 4x4 image
        let image = RgbImage::new(4, 4);

        // Test with wrong number of points
        let points = vec![Point2f::new(0.0, 0.0), Point2f::new(1.0, 0.0)];

        let result = get_rotate_crop_image(&image, &points);
        assert!(result.is_err());
    }

    #[test]
    fn test_get_rotate_crop_image_success() {
        // Create a simple 4x4 image with distinct colors
        let mut image = RgbImage::new(4, 4);
        for y in 0..4 {
            for x in 0..4 {
                // Create a gradient
                let r = (x * 64) as u8;
                let g = (y * 64) as u8;
                let b = ((x + y) * 32) as u8;
                image.put_pixel(x, y, Rgb([r, g, b]));
            }
        }

        // Define a simple square region
        let points = vec![
            Point2f::new(1.0, 1.0),
            Point2f::new(3.0, 1.0),
            Point2f::new(3.0, 3.0),
            Point2f::new(1.0, 3.0),
        ];

        let result = get_rotate_crop_image(&image, &points);
        assert!(result.is_ok());

        let cropped_image = result.unwrap();
        // Check that we got an image back
        assert!(cropped_image.width() > 0);
        assert!(cropped_image.height() > 0);
    }

    #[test]
    fn test_warp_perspective_invalid_matrix() {
        // Create a simple 2x2 image
        let image = RgbImage::new(2, 2);

        // Create a singular matrix (non-invertible)
        let matrix = Matrix3::new(1.0, 1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 1.0);

        let result = warp_perspective(&image, &matrix, 2, 2);
        assert!(result.is_err());
    }

    #[test]
    fn test_bilinear_interpolate() {
        // Create a simple 2x2 image with distinct colors
        let mut image = RgbImage::new(2, 2);
        image.put_pixel(0, 0, Rgb([255, 0, 0])); // Red
        image.put_pixel(1, 0, Rgb([0, 255, 0])); // Green
        image.put_pixel(0, 1, Rgb([0, 0, 255])); // Blue
        image.put_pixel(1, 1, Rgb([255, 255, 0])); // Yellow

        // Test interpolation at the center
        let pixel = bilinear_interpolate(&image, 0.5, 0.5);
        // Expected: average of all four colors
        // Red + Green + Blue + Yellow = (255, 0, 0) + (0, 255, 0) + (0, 0, 255) + (255, 255, 0)
        // = (510, 510, 255) / 4 = (127.5, 127.5, 63.75) ≈ (128, 128, 64)
        assert_eq!(pixel.0[0], 128);
        assert_eq!(pixel.0[1], 128);
        assert_eq!(pixel.0[2], 64);
    }
}
