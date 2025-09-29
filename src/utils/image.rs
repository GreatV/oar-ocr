//! Utility functions for image processing.
//!
//! This module provides functions for loading, converting, and manipulating images
//! in the OCR pipeline. It includes functions for converting between different
//! image formats, loading single or batch images from files, creating images
//! from raw data, and resize-and-pad operations.

use crate::core::OCRError;
use crate::core::errors::ImageProcessError;
use image::{DynamicImage, GrayImage, ImageBuffer, RgbImage};

/// Converts a DynamicImage to an RgbImage.
///
/// This function takes a DynamicImage (which can be in any format) and converts
/// it to an RgbImage (8-bit RGB format).
///
/// # Arguments
///
/// * `img` - The DynamicImage to convert
///
/// # Returns
///
/// * `RgbImage` - The converted RGB image
pub fn dynamic_to_rgb(img: DynamicImage) -> RgbImage {
    img.to_rgb8()
}

/// Converts a DynamicImage to a GrayImage.
///
/// This function takes a DynamicImage (which can be in any format) and converts
/// it to a GrayImage (8-bit grayscale format).
///
/// # Arguments
///
/// * `img` - The DynamicImage to convert
///
/// # Returns
///
/// * `GrayImage` - The converted grayscale image
pub fn dynamic_to_gray(img: DynamicImage) -> GrayImage {
    img.to_luma8()
}

/// Loads an image from the given bytes and converts it to RgbImage.
///
/// This function decodes an image from a byte slice and converts it
/// to an RgbImage. It handles any image format supported by the image crate.
///
/// # Arguments
///
/// * `bytes` - A byte slice containing the encoded image data
///
/// # Returns
///
/// * `Ok(RgbImage)` - The decoded and converted RGB image
/// * `Err(OCRError)` - An error if the image could not be decoded or converted
///
/// # Errors
///
/// This function will return an `OCRError::ImageLoad` error if the image cannot
/// be decoded from the provided bytes, or if there is an error during conversion.
pub fn load_image_from_memory(bytes: &[u8]) -> Result<RgbImage, OCRError> {
    let img = image::load_from_memory(bytes).map_err(OCRError::ImageLoad)?;
    Ok(dynamic_to_rgb(img))
}

/// Loads an image from a file path and converts it to RgbImage.
///
/// This function opens an image from the specified file path and converts it
/// to an RgbImage. It handles any image format supported by the image crate.
///
/// # Arguments
///
/// * `path` - A reference to the path of the image file to load
///
/// # Returns
///
/// * `Ok(RgbImage)` - The loaded and converted RGB image
/// * `Err(OCRError)` - An error if the image could not be loaded or converted
///
/// # Errors
///
/// This function will return an `OCRError::ImageLoad` error if the image cannot
/// be loaded from the specified path, or if there is an error during conversion.
pub fn load_image(path: &std::path::Path) -> Result<RgbImage, OCRError> {
    let img = image::open(path).map_err(OCRError::ImageLoad)?;
    Ok(dynamic_to_rgb(img))
}

/// Creates an RgbImage from raw pixel data.
///
/// This function creates an RgbImage from raw pixel data. The data must be
/// in RGB format (3 bytes per pixel) and the length must match the specified
/// width and height.
///
/// # Arguments
///
/// * `width` - The width of the image in pixels
/// * `height` - The height of the image in pixels
/// * `data` - A vector containing the raw pixel data (RGB format)
///
/// # Returns
///
/// * `Some(RgbImage)` - The created RGB image if the data is valid
/// * `None` - If the data length doesn't match the specified dimensions
pub fn create_rgb_image(width: u32, height: u32, data: Vec<u8>) -> Option<RgbImage> {
    if data.len() != (width * height * 3) as usize {
        return None;
    }

    ImageBuffer::from_raw(width, height, data)
}

/// Checks if the given image size is valid (non-zero dimensions).
pub fn check_image_size(size: &[u32; 2]) -> Result<(), ImageProcessError> {
    if size[0] == 0 || size[1] == 0 {
        return Err(ImageProcessError::InvalidCropSize);
    }
    Ok(())
}

/// Extracts a rectangular region from an RGB image.
pub fn slice_image(
    img: &RgbImage,
    coords: (u32, u32, u32, u32),
) -> Result<RgbImage, ImageProcessError> {
    let (x1, y1, x2, y2) = coords;
    let (img_width, img_height) = img.dimensions();

    if x1 >= x2 || y1 >= y2 {
        return Err(ImageProcessError::InvalidCropCoordinates);
    }

    if x2 > img_width || y2 > img_height {
        return Err(ImageProcessError::CropOutOfBounds);
    }

    let crop_width = x2 - x1;
    let crop_height = y2 - y1;

    let mut cropped = RgbImage::new(crop_width, crop_height);
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

/// Extracts a rectangular region from a grayscale image.
pub fn slice_gray_image(
    img: &GrayImage,
    coords: (u32, u32, u32, u32),
) -> Result<GrayImage, ImageProcessError> {
    let (x1, y1, x2, y2) = coords;
    let (img_width, img_height) = img.dimensions();

    if x1 >= x2 || y1 >= y2 {
        return Err(ImageProcessError::InvalidCropCoordinates);
    }

    if x2 > img_width || y2 > img_height {
        return Err(ImageProcessError::CropOutOfBounds);
    }

    let crop_width = x2 - x1;
    let crop_height = y2 - y1;

    let mut cropped = GrayImage::new(crop_width, crop_height);
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

/// Calculates centered crop coordinates for a target size.
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

/// Validates that crop coordinates stay within image bounds.
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

/// Resizes an RGB image to the target dimensions using Lanczos3 filtering.
pub fn resize_image(img: &RgbImage, width: u32, height: u32) -> RgbImage {
    image::imageops::resize(img, width, height, image::imageops::FilterType::Lanczos3)
}

/// Resizes a grayscale image to the target dimensions using Lanczos3 filtering.
pub fn resize_gray_image(img: &GrayImage, width: u32, height: u32) -> GrayImage {
    image::imageops::resize(img, width, height, image::imageops::FilterType::Lanczos3)
}

/// Converts an RGB image to grayscale.
pub fn rgb_to_grayscale(img: &RgbImage) -> GrayImage {
    image::imageops::grayscale(img)
}

/// Pads an image to the specified dimensions with a fill color.
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
    let x_offset = (target_width - src_width) / 2;
    let y_offset = (target_height - src_height) / 2;
    image::imageops::overlay(&mut padded, img, x_offset as i64, y_offset as i64);

    Ok(padded)
}

/// Loads a batch of images from file paths.
///
/// This function loads multiple images from the specified file paths and
/// converts them to RgbImages. It uses parallel processing when the number
/// of images exceeds the default parallel threshold.
///
/// # Arguments
///
/// * `paths` - A slice of paths to the image files to load
///
/// # Returns
///
/// * `Ok(Vec<RgbImage>)` - A vector of loaded RGB images
/// * `Err(OCRError)` - An error if any image could not be loaded
///
/// # Errors
///
/// This function will return an `OCRError` if any image cannot be loaded
/// from its specified path.
pub fn load_images<P: AsRef<std::path::Path> + Send + Sync>(
    paths: &[P],
) -> Result<Vec<RgbImage>, OCRError> {
    load_images_batch_with_threshold(paths, None)
}

/// Loads a batch of images from file paths with a custom parallel threshold.
///
/// This function loads multiple images from the specified file paths and
/// converts them to RgbImages. It uses parallel processing when the number
/// of images exceeds the specified threshold, or the default threshold if
/// none is provided.
///
/// # Arguments
///
/// * `paths` - A slice of paths to the image files to load
/// * `parallel_threshold` - An optional threshold for parallel processing.
///   If `None`, the default threshold from `DEFAULT_PARALLEL_THRESHOLD` is used.
///
/// # Returns
///
/// * `Ok(Vec<RgbImage>)` - A vector of loaded RGB images
/// * `Err(OCRError)` - An error if any image could not be loaded
///
/// # Errors
///
/// This function will return an `OCRError` if any image cannot be loaded
/// from its specified path.
pub fn load_images_batch_with_threshold<P: AsRef<std::path::Path> + Send + Sync>(
    paths: &[P],
    parallel_threshold: Option<usize>,
) -> Result<Vec<RgbImage>, OCRError> {
    use crate::core::constants::DEFAULT_PARALLEL_THRESHOLD;

    let threshold = parallel_threshold.unwrap_or(DEFAULT_PARALLEL_THRESHOLD);

    if paths.len() > threshold {
        use rayon::prelude::*;
        paths.par_iter().map(|p| load_image(p.as_ref())).collect()
    } else {
        paths.iter().map(|p| load_image(p.as_ref())).collect()
    }
}

/// Load multiple images from file paths using centralized parallel policy.
///
/// This function loads images from the provided file paths using the utility threshold
/// from the centralized ParallelPolicy. If the number of paths exceeds the threshold,
/// the loading is performed in parallel using rayon. Otherwise, images are loaded
/// sequentially.
///
/// # Arguments
///
/// * `paths` - A slice of paths to image files
/// * `policy` - The parallel policy containing the utility threshold
///
/// # Returns
///
/// A Result containing a vector of loaded RgbImages, or an OCRError if any image fails to load.
///
/// # Errors
///
/// This function will return an `OCRError` if any image cannot be loaded
/// from its specified path.
pub fn load_images_batch_with_policy<P: AsRef<std::path::Path> + Send + Sync>(
    paths: &[P],
    policy: &crate::core::config::ParallelPolicy,
) -> Result<Vec<RgbImage>, OCRError> {
    if paths.len() > policy.utility_threshold {
        use rayon::prelude::*;
        paths.par_iter().map(|p| load_image(p.as_ref())).collect()
    } else {
        paths.iter().map(|p| load_image(p.as_ref())).collect()
    }
}

/// Padding strategy for resize-and-pad operations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum PaddingStrategy {
    /// Pad with a solid color
    SolidColor([u8; 3]),
    /// Pad with black (equivalent to SolidColor([0, 0, 0]))
    Black,
    /// Left-align the resized image (no centering)
    LeftAlign([u8; 3]),
}

impl Default for PaddingStrategy {
    fn default() -> Self {
        Self::Black
    }
}

/// Configuration for resize-and-pad operations.
#[derive(Debug, Clone)]
pub struct ResizePadConfig {
    /// Target dimensions (width, height)
    pub target_dims: (u32, u32),
    /// Padding strategy to use
    pub padding_strategy: PaddingStrategy,
    /// Filter type for resizing
    pub filter_type: image::imageops::FilterType,
}

impl ResizePadConfig {
    /// Create a new resize-pad configuration.
    pub fn new(target_dims: (u32, u32)) -> Self {
        Self {
            target_dims,
            padding_strategy: PaddingStrategy::default(),
            filter_type: image::imageops::FilterType::Triangle,
        }
    }

    /// Set the padding strategy.
    pub fn with_padding_strategy(mut self, strategy: PaddingStrategy) -> Self {
        self.padding_strategy = strategy;
        self
    }

    /// Set the filter type for resizing.
    pub fn with_filter_type(mut self, filter_type: image::imageops::FilterType) -> Self {
        self.filter_type = filter_type;
        self
    }
}

/// Resize an image to fit within target dimensions while maintaining aspect ratio,
/// then pad to exact target dimensions.
///
/// This function provides a unified approach to resize-and-pad operations that
/// can replace the duplicated logic found in various processors.
///
/// # Arguments
///
/// * `image` - The input RGB image to resize and pad
/// * `config` - Configuration for the resize-and-pad operation
///
/// # Returns
///
/// A resized and padded RGB image with exact target dimensions.
pub fn resize_and_pad(image: &RgbImage, config: &ResizePadConfig) -> RgbImage {
    let (target_width, target_height) = config.target_dims;
    let (orig_width, orig_height) = image.dimensions();

    // Calculate scaling factor to fit within target dimensions while maintaining aspect ratio
    let scale_w = target_width as f32 / orig_width as f32;
    let scale_h = target_height as f32 / orig_height as f32;
    let scale = scale_w.min(scale_h);

    // Calculate new dimensions
    let new_width = (orig_width as f32 * scale) as u32;
    let new_height = (orig_height as f32 * scale) as u32;

    // Resize the image
    let resized = image::imageops::resize(image, new_width, new_height, config.filter_type);

    // Create padded image with target dimensions
    let padding_color = match config.padding_strategy {
        PaddingStrategy::SolidColor(color) => color,
        PaddingStrategy::Black => [0, 0, 0],
        PaddingStrategy::LeftAlign(color) => color,
    };
    let padding_rgb = image::Rgb(padding_color);
    let mut padded = ImageBuffer::from_pixel(target_width, target_height, padding_rgb);

    // Calculate padding offsets
    let (pad_x, pad_y) = match config.padding_strategy {
        PaddingStrategy::LeftAlign(_) => (0, 0),
        _ => {
            // Center the image
            let pad_x = (target_width - new_width) / 2;
            let pad_y = (target_height - new_height) / 2;
            (pad_x, pad_y)
        }
    };

    // Copy resized image to padded image using efficient overlay
    image::imageops::overlay(&mut padded, &resized, pad_x as i64, pad_y as i64);

    padded
}

/// Configuration for OCR-style resize-and-pad operations with width constraints.
#[derive(Debug, Clone)]
pub struct OCRResizePadConfig {
    /// Target height
    pub target_height: u32,
    /// Maximum allowed width
    pub max_width: u32,
    /// Padding strategy to use
    pub padding_strategy: PaddingStrategy,
    /// Filter type for resizing
    pub filter_type: image::imageops::FilterType,
}

impl OCRResizePadConfig {
    /// Create a new OCR resize-pad configuration.
    pub fn new(target_height: u32, max_width: u32) -> Self {
        Self {
            target_height,
            max_width,
            padding_strategy: PaddingStrategy::default(),
            filter_type: image::imageops::FilterType::Triangle,
        }
    }

    /// Set the padding strategy.
    pub fn with_padding_strategy(mut self, strategy: PaddingStrategy) -> Self {
        self.padding_strategy = strategy;
        self
    }

    /// Set the filter type for resizing.
    pub fn with_filter_type(mut self, filter_type: image::imageops::FilterType) -> Self {
        self.filter_type = filter_type;
        self
    }
}

/// Resize an image for OCR processing with width constraints and padding.
///
/// This function handles the specific resize-and-pad logic used in OCR processing,
/// where images are resized to a fixed height while maintaining aspect ratio,
/// with a maximum width constraint, and then padded to a target width.
///
/// # Arguments
///
/// * `image` - The input RGB image to resize and pad
/// * `config` - Configuration for the OCR resize-and-pad operation
/// * `target_width_ratio` - Optional ratio to calculate target width from height.
///   If None, uses the image's original aspect ratio.
///
/// # Returns
///
/// A tuple containing:
/// - The resized and padded RGB image
/// - The actual width used for the padded image
pub fn ocr_resize_and_pad(
    image: &RgbImage,
    config: &OCRResizePadConfig,
    target_width_ratio: Option<f32>,
) -> (RgbImage, u32) {
    let (original_w, original_h) = image.dimensions();
    let original_ratio = original_w as f32 / original_h as f32;

    // Calculate target width based on ratio or original aspect ratio
    let mut target_w = if let Some(ratio) = target_width_ratio {
        (config.target_height as f32 * ratio) as u32
    } else {
        (config.target_height as f32 * original_ratio).ceil() as u32
    };

    // Apply maximum width constraint
    let resized_w = if target_w > config.max_width {
        target_w = config.max_width;
        config.max_width
    } else {
        // Calculate actual resized width based on aspect ratio
        let ratio = original_w as f32 / original_h as f32;
        if (config.target_height as f32 * ratio).ceil() as u32 > target_w {
            target_w
        } else {
            (config.target_height as f32 * ratio).ceil() as u32
        }
    };

    // Resize the image
    let resized_image =
        image::imageops::resize(image, resized_w, config.target_height, config.filter_type);

    // Create padded image with target dimensions
    let padding_color = match config.padding_strategy {
        PaddingStrategy::SolidColor(color) => color,
        PaddingStrategy::Black => [0, 0, 0],
        PaddingStrategy::LeftAlign(color) => color,
    };
    let padding_rgb = image::Rgb(padding_color);
    let mut padded_image = ImageBuffer::from_pixel(target_w, config.target_height, padding_rgb);

    // Copy resized image to padded image (left-aligned for OCR)
    image::imageops::overlay(&mut padded_image, &resized_image, 0, 0);

    (padded_image, target_w)
}

/// Resizes a batch of images to the specified dimensions.
///
/// This function provides a unified approach to batch image resizing that can replace
/// duplicated resize loops found in various predictors. It supports both functional
/// and imperative styles and can optionally apply post-processing operations.
///
/// # Arguments
///
/// * `images` - A slice of RGB images to resize
/// * `target_width` - Target width for all images
/// * `target_height` - Target height for all images
/// * `filter_type` - The filter type to use for resizing (defaults to Lanczos3 if None)
///
/// # Returns
///
/// A vector of resized RGB images.
///
/// # Example
///
/// ```rust,no_run
/// use oar_ocr::utils::resize_images_batch;
/// use image::RgbImage;
///
/// let images = vec![RgbImage::new(100, 100), RgbImage::new(200, 150)];
/// let resized = resize_images_batch(&images, 224, 224, None);
/// assert_eq!(resized.len(), 2);
/// assert_eq!(resized[0].dimensions(), (224, 224));
/// ```
pub fn resize_images_batch(
    images: &[RgbImage],
    target_width: u32,
    target_height: u32,
    filter_type: Option<image::imageops::FilterType>,
) -> Vec<RgbImage> {
    let filter = filter_type.unwrap_or(image::imageops::FilterType::Lanczos3);

    images
        .iter()
        .map(|img| image::imageops::resize(img, target_width, target_height, filter))
        .collect()
}

/// Resizes a batch of images and converts them to DynamicImage format.
///
/// This function combines batch resizing with conversion to DynamicImage format,
/// which is commonly needed in OCR preprocessing pipelines.
///
/// # Arguments
///
/// * `images` - A slice of RGB images to resize
/// * `target_width` - Target width for all images
/// * `target_height` - Target height for all images
/// * `filter_type` - The filter type to use for resizing (defaults to Lanczos3 if None)
///
/// # Returns
///
/// A vector of resized images as DynamicImage instances.
pub fn resize_images_batch_to_dynamic(
    images: &[RgbImage],
    target_width: u32,
    target_height: u32,
    filter_type: Option<image::imageops::FilterType>,
) -> Vec<DynamicImage> {
    let filter = filter_type.unwrap_or(image::imageops::FilterType::Lanczos3);

    images
        .iter()
        .map(|img| {
            let resized = image::imageops::resize(img, target_width, target_height, filter);
            DynamicImage::ImageRgb8(resized)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::image::{GenericImageView, GrayImage, ImageBuffer, Rgb, RgbImage};

    fn create_test_image(width: u32, height: u32, color: [u8; 3]) -> RgbImage {
        ImageBuffer::from_pixel(width, height, Rgb(color))
    }

    #[test]
    fn basic_size_checks() {
        assert!(check_image_size(&[100, 100]).is_ok());
        assert!(check_image_size(&[0, 50]).is_err());
    }

    #[test]
    fn slice_rgb_image_region() {
        let img = RgbImage::from_pixel(10, 10, Rgb([255, 0, 0]));
        let cropped = slice_image(&img, (2, 2, 6, 6)).unwrap();
        assert_eq!(cropped.dimensions(), (4, 4));
        assert!(slice_image(&img, (6, 6, 2, 2)).is_err());
    }

    #[test]
    fn slice_gray_image_region() {
        let img = GrayImage::from_pixel(10, 10, image::Luma([128]));
        let cropped = slice_gray_image(&img, (1, 1, 5, 5)).unwrap();
        assert_eq!(cropped.dimensions(), (4, 4));
    }

    #[test]
    fn center_crop_coordinates() {
        let coords = calculate_center_crop_coords(100, 60, 40, 20).unwrap();
        assert_eq!(coords, (30, 20));
        assert!(calculate_center_crop_coords(20, 20, 40, 10).is_err());
    }

    #[test]
    fn crop_bounds_validation() {
        assert!(validate_crop_bounds(100, 80, 10, 10, 40, 40).is_ok());
        assert!(validate_crop_bounds(100, 80, 70, 10, 40, 40).is_err());
    }

    #[test]
    fn pad_image_to_target() {
        let img = RgbImage::from_pixel(20, 20, Rgb([10, 20, 30]));
        let padded = pad_image(&img, 40, 40, [0, 0, 0]).unwrap();
        assert_eq!(padded.dimensions(), (40, 40));
        assert!(pad_image(&img, 10, 10, [0, 0, 0]).is_err());
    }

    #[test]
    fn test_resize_and_pad_with_custom_padding() {
        let image = create_test_image(50, 100, [255, 0, 0]); // 1:2 aspect ratio (tall)
        let config = ResizePadConfig::new((80, 80))
            .with_padding_strategy(PaddingStrategy::SolidColor([0, 255, 0])); // Green padding

        let result = resize_and_pad(&image, &config);

        assert_eq!(result.dimensions(), (80, 80));

        // The resized image should be 40x80 (maintaining 1:2 ratio), centered in 80x80
        // So there should be 20 pixels of padding on left and right
        let center_pixel = result.get_pixel(40, 40); // Center of image
        assert_eq!(*center_pixel, Rgb([255, 0, 0])); // Should be red (original image)

        let left_padding = result.get_pixel(10, 40); // Left padding area
        assert_eq!(*left_padding, Rgb([0, 255, 0])); // Should be green (custom padding)
    }

    #[test]
    fn test_resize_and_pad_left_align() {
        let image = create_test_image(50, 100, [0, 0, 255]); // 1:2 aspect ratio (tall)
        let config = ResizePadConfig::new((80, 80))
            .with_padding_strategy(PaddingStrategy::LeftAlign([255, 255, 0])); // Yellow padding, left-aligned

        let result = resize_and_pad(&image, &config);

        assert_eq!(result.dimensions(), (80, 80));

        // The resized image should be 40x80, left-aligned in 80x80
        let left_edge_pixel = result.get_pixel(20, 40); // Should be in the resized image
        assert_eq!(*left_edge_pixel, Rgb([0, 0, 255])); // Should be blue (original image)

        let right_padding = result.get_pixel(60, 40); // Right padding area
        assert_eq!(*right_padding, Rgb([255, 255, 0])); // Should be yellow (padding)
    }

    #[test]
    fn test_resize_images_batch() {
        // Create test images with different sizes
        let img1 = create_test_image(100, 50, [255, 0, 0]); // Red
        let img2 = create_test_image(200, 100, [0, 255, 0]); // Green
        let images = vec![img1, img2];

        // Resize batch to 64x64
        let resized = resize_images_batch(&images, 64, 64, None);

        assert_eq!(resized.len(), 2);
        assert_eq!(resized[0].dimensions(), (64, 64));
        assert_eq!(resized[1].dimensions(), (64, 64));

        // Check that the colors are preserved (approximately)
        let pixel1 = resized[0].get_pixel(32, 32);
        let pixel2 = resized[1].get_pixel(32, 32);

        // Red image should still be predominantly red
        assert!(pixel1[0] > pixel1[1] && pixel1[0] > pixel1[2]);
        // Green image should still be predominantly green
        assert!(pixel2[1] > pixel2[0] && pixel2[1] > pixel2[2]);
    }

    #[test]
    fn test_resize_images_batch_to_dynamic() {
        // Create test images
        let img1 = create_test_image(100, 50, [255, 0, 0]);
        let img2 = create_test_image(200, 100, [0, 255, 0]);
        let images = vec![img1, img2];

        // Resize batch to 32x32 and convert to DynamicImage
        let resized = resize_images_batch_to_dynamic(&images, 32, 32, None);

        assert_eq!(resized.len(), 2);

        // Check that they are DynamicImage::ImageRgb8 variants
        for dynamic_img in &resized {
            assert_eq!(dynamic_img.dimensions(), (32, 32));
            match dynamic_img {
                DynamicImage::ImageRgb8(_) => {} // Expected
                _ => panic!("Expected ImageRgb8 variant"),
            }
        }
    }

    #[test]
    fn test_resize_images_batch_empty() {
        let images: Vec<RgbImage> = vec![];
        let resized = resize_images_batch(&images, 64, 64, None);
        assert!(resized.is_empty());
    }

    #[test]
    fn test_resize_images_batch_custom_filter() {
        let img = create_test_image(100, 100, [128, 128, 128]);
        let images = vec![img];

        // Test with different filter types
        let resized_lanczos =
            resize_images_batch(&images, 50, 50, Some(image::imageops::FilterType::Lanczos3));
        let resized_nearest =
            resize_images_batch(&images, 50, 50, Some(image::imageops::FilterType::Nearest));

        assert_eq!(resized_lanczos.len(), 1);
        assert_eq!(resized_nearest.len(), 1);
        assert_eq!(resized_lanczos[0].dimensions(), (50, 50));
        assert_eq!(resized_nearest[0].dimensions(), (50, 50));
    }

    #[test]
    fn test_ocr_resize_and_pad_with_max_width_constraint() {
        let image = create_test_image(400, 100, [200, 100, 50]); // 4:1 aspect ratio
        let config = OCRResizePadConfig::new(32, 100); // Height 32, max width 100

        let (result, actual_width) = ocr_resize_and_pad(&image, &config, None);

        assert_eq!(result.height(), 32);
        assert_eq!(actual_width, 100); // Should be constrained to max width
        assert_eq!(result.width(), 100);

        // Check that the image is left-aligned
        let left_pixel = result.get_pixel(0, 16); // Left edge, middle height
        assert_eq!(*left_pixel, Rgb([200, 100, 50])); // Should be original color
    }

    #[test]
    fn test_ocr_resize_and_pad_with_target_ratio() {
        let image = create_test_image(100, 50, [255, 128, 64]); // 2:1 aspect ratio
        let config = OCRResizePadConfig::new(32, 200); // Height 32, max width 200
        let target_ratio = 3.0; // Force 3:1 ratio

        let (result, actual_width) = ocr_resize_and_pad(&image, &config, Some(target_ratio));

        assert_eq!(result.height(), 32);
        assert_eq!(actual_width, 96); // 32 * 3.0 = 96
        assert_eq!(result.width(), 96);
    }

    #[test]
    fn test_resize_pad_config_builder() {
        let config = ResizePadConfig::new((100, 50))
            .with_padding_strategy(PaddingStrategy::SolidColor([255, 0, 0]))
            .with_filter_type(image::imageops::FilterType::Lanczos3);

        assert_eq!(config.target_dims, (100, 50));
        assert_eq!(
            config.padding_strategy,
            PaddingStrategy::SolidColor([255, 0, 0])
        );
        assert_eq!(config.filter_type, image::imageops::FilterType::Lanczos3);
    }

    #[test]
    fn test_ocr_resize_pad_config_builder() {
        let config = OCRResizePadConfig::new(64, 320)
            .with_padding_strategy(PaddingStrategy::SolidColor([100, 100, 100]))
            .with_filter_type(image::imageops::FilterType::Nearest);

        assert_eq!(config.target_height, 64);
        assert_eq!(config.max_width, 320);
        assert_eq!(
            config.padding_strategy,
            PaddingStrategy::SolidColor([100, 100, 100])
        );
        assert_eq!(config.filter_type, image::imageops::FilterType::Nearest);
    }
}
