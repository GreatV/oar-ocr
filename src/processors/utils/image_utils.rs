//! Utility functions for basic image operations.

use crate::processors::types::ImageProcessError;
use image::{GrayImage, RgbImage};

/// Checks if the given image size is valid (non-zero dimensions).
///
/// # Arguments
///
/// * `size` - A reference to an array containing width and height values.
///
/// # Returns
///
/// * `Ok(())` if both dimensions are greater than zero.
/// * `Err(ImageProcessError::InvalidCropSize)` if either dimension is zero.
pub fn check_image_size(size: &[u32; 2]) -> Result<(), ImageProcessError> {
    if size[0] == 0 || size[1] == 0 {
        return Err(ImageProcessError::InvalidCropSize);
    }
    Ok(())
}

/// Slices a region from an image buffer.
///
/// Extracts a rectangular region from the input image based on the provided coordinates.
///
/// # Arguments
///
/// * `img` - Reference to the source image buffer.
/// * `coords` - Tuple containing (x1, y1, x2, y2) coordinates defining the crop region.
///   (x1, y1) is the top-left corner, (x2, y2) is the bottom-right corner.
///
/// # Returns
///
/// * `Ok(RgbImage)` - The cropped image region.
/// * `Err(ImageProcessError)` - If the coordinates are invalid or out of bounds.
///
/// # Examples
///
/// ```rust,no_run
/// use image::RgbImage;
/// use oar_ocr::processors::utils::image_utils::slice_image;
///
/// let img = RgbImage::new(100, 100);
/// let cropped = slice_image(&img, (10, 10, 50, 50)).unwrap();
/// assert_eq!(cropped.dimensions(), (40, 40));
/// ```
pub fn slice_image(
    img: &RgbImage,
    coords: (u32, u32, u32, u32),
) -> Result<RgbImage, ImageProcessError> {
    let (x1, y1, x2, y2) = coords;
    let (img_width, img_height) = img.dimensions();

    // Validate coordinates
    if x1 >= x2 || y1 >= y2 {
        return Err(ImageProcessError::InvalidCropCoordinates);
    }

    if x2 > img_width || y2 > img_height {
        return Err(ImageProcessError::CropOutOfBounds);
    }

    let crop_width = x2 - x1;
    let crop_height = y2 - y1;

    // Create a new image buffer for the cropped region
    let mut cropped = RgbImage::new(crop_width, crop_height);

    // Copy pixels from the source image to the cropped image
    for y in 0..crop_height {
        for x in 0..crop_width {
            let src_x = x1 + x;
            let src_y = y1 + y;
            let pixel = img.get_pixel(src_x, src_y);
            cropped.put_pixel(x, y, *pixel);
        }
    }

    Ok(cropped)
}

/// Slices a region from a grayscale image buffer.
///
/// Similar to `slice_image` but for grayscale images.
///
/// # Arguments
///
/// * `img` - Reference to the source grayscale image buffer.
/// * `coords` - Tuple containing (x1, y1, x2, y2) coordinates defining the crop region.
///
/// # Returns
///
/// * `Ok(GrayImage)` - The cropped grayscale image region.
/// * `Err(ImageProcessError)` - If the coordinates are invalid or out of bounds.
pub fn slice_gray_image(
    img: &GrayImage,
    coords: (u32, u32, u32, u32),
) -> Result<GrayImage, ImageProcessError> {
    let (x1, y1, x2, y2) = coords;
    let (img_width, img_height) = img.dimensions();

    // Validate coordinates
    if x1 >= x2 || y1 >= y2 {
        return Err(ImageProcessError::InvalidCropCoordinates);
    }

    if x2 > img_width || y2 > img_height {
        return Err(ImageProcessError::CropOutOfBounds);
    }

    let crop_width = x2 - x1;
    let crop_height = y2 - y1;

    // Create a new image buffer for the cropped region
    let mut cropped = GrayImage::new(crop_width, crop_height);

    // Copy pixels from the source image to the cropped image
    for y in 0..crop_height {
        for x in 0..crop_width {
            let src_x = x1 + x;
            let src_y = y1 + y;
            let pixel = img.get_pixel(src_x, src_y);
            cropped.put_pixel(x, y, *pixel);
        }
    }

    Ok(cropped)
}

/// Calculates the center coordinates for a crop operation.
///
/// Given image dimensions and desired crop size, calculates the top-left
/// coordinates for a center crop.
///
/// # Arguments
///
/// * `img_width` - Width of the source image.
/// * `img_height` - Height of the source image.
/// * `crop_width` - Desired width of the crop.
/// * `crop_height` - Desired height of the crop.
///
/// # Returns
///
/// * `Ok((x, y))` - Top-left coordinates for the center crop.
/// * `Err(ImageProcessError)` - If the crop size is larger than the image.
pub fn calculate_center_crop_coords(
    img_width: u32,
    img_height: u32,
    crop_width: u32,
    crop_height: u32,
) -> Result<(u32, u32), ImageProcessError> {
    if crop_width > img_width || crop_height > img_height {
        return Err(ImageProcessError::CropSizeTooLarge);
    }

    let x = (img_width - crop_width) / 2;
    let y = (img_height - crop_height) / 2;

    Ok((x, y))
}

/// Validates that crop coordinates are within image bounds.
///
/// # Arguments
///
/// * `img_width` - Width of the source image.
/// * `img_height` - Height of the source image.
/// * `x` - X coordinate of the crop region.
/// * `y` - Y coordinate of the crop region.
/// * `crop_width` - Width of the crop region.
/// * `crop_height` - Height of the crop region.
///
/// # Returns
///
/// * `Ok(())` - If the crop region is valid.
/// * `Err(ImageProcessError)` - If the crop region is out of bounds.
pub fn validate_crop_bounds(
    img_width: u32,
    img_height: u32,
    x: u32,
    y: u32,
    crop_width: u32,
    crop_height: u32,
) -> Result<(), ImageProcessError> {
    if x + crop_width > img_width || y + crop_height > img_height {
        return Err(ImageProcessError::CropOutOfBounds);
    }
    Ok(())
}

/// Resizes an image to the specified dimensions.
///
/// # Arguments
///
/// * `img` - Reference to the source image.
/// * `width` - Target width.
/// * `height` - Target height.
///
/// # Returns
///
/// * `RgbImage` - The resized image.
pub fn resize_image(img: &RgbImage, width: u32, height: u32) -> RgbImage {
    image::imageops::resize(img, width, height, image::imageops::FilterType::Lanczos3)
}

/// Resizes a grayscale image to the specified dimensions.
///
/// # Arguments
///
/// * `img` - Reference to the source grayscale image.
/// * `width` - Target width.
/// * `height` - Target height.
///
/// # Returns
///
/// * `GrayImage` - The resized grayscale image.
pub fn resize_gray_image(img: &GrayImage, width: u32, height: u32) -> GrayImage {
    image::imageops::resize(img, width, height, image::imageops::FilterType::Lanczos3)
}

/// Converts an RGB image to grayscale.
///
/// # Arguments
///
/// * `img` - Reference to the source RGB image.
///
/// # Returns
///
/// * `GrayImage` - The converted grayscale image.
pub fn rgb_to_grayscale(img: &RgbImage) -> GrayImage {
    image::imageops::grayscale(img)
}

/// Pads an image to the specified dimensions with a given color.
///
/// # Arguments
///
/// * `img` - Reference to the source image.
/// * `target_width` - Target width after padding.
/// * `target_height` - Target height after padding.
/// * `fill_color` - RGB color to use for padding.
///
/// # Returns
///
/// * `Ok(RgbImage)` - The padded image.
/// * `Err(ImageProcessError)` - If the target dimensions are smaller than the source.
pub fn pad_image(
    img: &RgbImage,
    target_width: u32,
    target_height: u32,
    fill_color: [u8; 3],
) -> Result<RgbImage, ImageProcessError> {
    let (src_width, src_height) = img.dimensions();

    if target_width < src_width || target_height < src_height {
        return Err(ImageProcessError::InvalidCropSize);
    }

    if target_width == src_width && target_height == src_height {
        return Ok(img.clone());
    }

    let mut padded = RgbImage::from_pixel(target_width, target_height, image::Rgb(fill_color));

    // Calculate center position
    let x_offset = (target_width - src_width) / 2;
    let y_offset = (target_height - src_height) / 2;

    // Copy the original image to the center of the padded image
    image::imageops::overlay(&mut padded, img, x_offset as i64, y_offset as i64);

    Ok(padded)
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{Rgb, RgbImage};

    #[test]
    fn test_check_image_size() {
        assert!(check_image_size(&[100, 100]).is_ok());
        assert!(check_image_size(&[0, 100]).is_err());
        assert!(check_image_size(&[100, 0]).is_err());
        assert!(check_image_size(&[0, 0]).is_err());
    }

    #[test]
    fn test_slice_image() {
        let img = RgbImage::from_pixel(100, 100, Rgb([255, 0, 0]));
        let cropped = slice_image(&img, (10, 10, 50, 50)).unwrap();
        assert_eq!(cropped.dimensions(), (40, 40));

        // Test invalid coordinates
        assert!(slice_image(&img, (50, 50, 10, 10)).is_err()); // x1 >= x2
        assert!(slice_image(&img, (10, 10, 200, 50)).is_err()); // out of bounds
    }

    #[test]
    fn test_calculate_center_crop_coords() {
        let (x, y) = calculate_center_crop_coords(100, 100, 50, 50).unwrap();
        assert_eq!((x, y), (25, 25));

        // Test crop larger than image
        assert!(calculate_center_crop_coords(100, 100, 200, 50).is_err());
    }

    #[test]
    fn test_validate_crop_bounds() {
        assert!(validate_crop_bounds(100, 100, 10, 10, 50, 50).is_ok());
        assert!(validate_crop_bounds(100, 100, 60, 60, 50, 50).is_err()); // out of bounds
    }

    #[test]
    fn test_pad_image() {
        let img = RgbImage::from_pixel(50, 50, Rgb([255, 0, 0]));
        let padded = pad_image(&img, 100, 100, [0, 0, 0]).unwrap();
        assert_eq!(padded.dimensions(), (100, 100));

        // Test invalid target size
        assert!(pad_image(&img, 25, 25, [0, 0, 0]).is_err());
    }
}
