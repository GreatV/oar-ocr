//! Image cropping functionality with different modes.

use crate::core::errors::ImageProcessError;
use crate::processors::types::CropMode;
use crate::utils::image;
use ::image::RgbImage;

/// A processor for cropping images with different positioning modes.
///
/// The `Crop` struct provides functionality to crop images to a specified size
/// using different positioning strategies (center, top-left, etc.).
#[derive(Debug)]
pub struct Crop {
    /// The dimensions [width, height] for the crop operation.
    crop_size: [u32; 2],
    /// The mode determining how the crop region is positioned.
    crop_mode: CropMode,
}

impl Crop {
    /// Creates a new Crop instance with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `crop_size` - Array containing [width, height] for the crop operation.
    /// * `crop_mode` - The positioning mode for the crop operation.
    ///
    /// # Returns
    ///
    /// * `Ok(Crop)` - A new Crop instance.
    /// * `Err(ImageProcessError::InvalidCropSize)` - If either dimension is zero.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use oar_ocr_core::utils::crop::Crop;
    /// use oar_ocr_core::processors::types::CropMode;
    ///
    /// # fn main() -> Result<(), oar_ocr_core::core::errors::ImageProcessError> {
    /// let crop = Crop::new([224, 224], CropMode::Center)?;
    /// # let _ = crop;
    /// # Ok(())
    /// # }
    /// ```
    pub fn new(crop_size: [u32; 2], crop_mode: CropMode) -> Result<Self, ImageProcessError> {
        image::check_image_size(&crop_size)?;
        Ok(Self {
            crop_size,
            crop_mode,
        })
    }

    /// Gets the crop size.
    ///
    /// # Returns
    ///
    /// The crop size as [width, height].
    pub fn crop_size(&self) -> [u32; 2] {
        self.crop_size
    }

    /// Gets the crop mode.
    ///
    /// # Returns
    ///
    /// The crop mode.
    pub fn crop_mode(&self) -> &CropMode {
        &self.crop_mode
    }

    /// Processes an image by cropping it according to the configured parameters.
    ///
    /// # Arguments
    ///
    /// * `img` - Reference to the input image to be cropped.
    ///
    /// # Returns
    ///
    /// * `Ok(RgbImage)` - The cropped image.
    /// * `Err(ImageProcessError)` - If the crop operation fails.
    ///
    /// # Examples
    ///
    /// ```rust,no_run
    /// use image::RgbImage;
    /// use oar_ocr_core::utils::crop::Crop;
    /// use oar_ocr_core::processors::types::CropMode;
    ///
    /// # fn main() -> Result<(), oar_ocr_core::core::errors::ImageProcessError> {
    /// let crop = Crop::new([100, 100], CropMode::Center)?;
    /// let img = RgbImage::new(200, 200);
    /// let cropped = crop.process(&img)?;
    /// assert_eq!(cropped.dimensions(), (100, 100));
    /// # Ok(())
    /// # }
    /// ```
    pub fn process(&self, img: &RgbImage) -> Result<RgbImage, ImageProcessError> {
        let (img_width, img_height) = img.dimensions();
        let [crop_width, crop_height] = self.crop_size;

        // Check if crop size is larger than image
        if crop_width > img_width || crop_height > img_height {
            return Err(ImageProcessError::CropSizeTooLarge);
        }

        // If the image is already the desired size, return a clone
        if crop_width == img_width && crop_height == img_height {
            return Ok(img.clone());
        }

        let (x, y) = self.calculate_crop_position(img_width, img_height)?;

        // Validate the calculated position
        image::validate_crop_bounds(img_width, img_height, x, y, crop_width, crop_height)?;

        // Perform the crop
        let coords = (x, y, x + crop_width, y + crop_height);
        image::slice_image(img, coords)
    }

    /// Calculates the top-left position for the crop based on the crop mode.
    ///
    /// # Arguments
    ///
    /// * `img_width` - Width of the source image.
    /// * `img_height` - Height of the source image.
    ///
    /// # Returns
    ///
    /// * `Ok((x, y))` - Top-left coordinates for the crop.
    /// * `Err(ImageProcessError)` - If the calculation fails.
    fn calculate_crop_position(
        &self,
        img_width: u32,
        img_height: u32,
    ) -> Result<(u32, u32), ImageProcessError> {
        let [crop_width, crop_height] = self.crop_size;

        match self.crop_mode {
            CropMode::Center => {
                image::calculate_center_crop_coords(img_width, img_height, crop_width, crop_height)
            }
            CropMode::TopLeft => Ok((0, 0)),
            CropMode::TopRight => {
                if crop_width > img_width {
                    return Err(ImageProcessError::CropSizeTooLarge);
                }
                Ok((img_width - crop_width, 0))
            }
            CropMode::BottomLeft => {
                if crop_height > img_height {
                    return Err(ImageProcessError::CropSizeTooLarge);
                }
                Ok((0, img_height - crop_height))
            }
            CropMode::BottomRight => {
                if crop_width > img_width || crop_height > img_height {
                    return Err(ImageProcessError::CropSizeTooLarge);
                }
                Ok((img_width - crop_width, img_height - crop_height))
            }
            CropMode::Custom { x, y } => {
                // Validate custom position
                if x + crop_width > img_width || y + crop_height > img_height {
                    return Err(ImageProcessError::CropOutOfBounds);
                }
                Ok((x, y))
            }
        }
    }

    /// Processes multiple images in batch.
    ///
    /// # Arguments
    ///
    /// * `images` - Vector of images to be cropped.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<RgbImage>)` - Vector of cropped images.
    /// * `Err(ImageProcessError)` - If any crop operation fails.
    pub fn process_batch(&self, images: &[RgbImage]) -> Result<Vec<RgbImage>, ImageProcessError> {
        images.iter().map(|img| self.process(img)).collect()
    }

    /// Updates the crop size.
    ///
    /// # Arguments
    ///
    /// * `crop_size` - New crop size as [width, height].
    ///
    /// # Returns
    ///
    /// * `Ok(())` - If the size is valid.
    /// * `Err(ImageProcessError::InvalidCropSize)` - If either dimension is zero.
    pub fn set_crop_size(&mut self, crop_size: [u32; 2]) -> Result<(), ImageProcessError> {
        image::check_image_size(&crop_size)?;
        self.crop_size = crop_size;
        Ok(())
    }

    /// Updates the crop mode.
    ///
    /// # Arguments
    ///
    /// * `crop_mode` - New crop mode.
    pub fn set_crop_mode(&mut self, crop_mode: CropMode) {
        self.crop_mode = crop_mode;
    }

    /// Checks if the crop can be applied to an image with the given dimensions.
    ///
    /// # Arguments
    ///
    /// * `img_width` - Width of the target image.
    /// * `img_height` - Height of the target image.
    ///
    /// # Returns
    ///
    /// * `true` - If the crop can be applied.
    /// * `false` - If the crop size is too large for the image.
    pub fn can_crop(&self, img_width: u32, img_height: u32) -> bool {
        let [crop_width, crop_height] = self.crop_size;
        crop_width <= img_width && crop_height <= img_height
    }

    /// Gets the aspect ratio of the crop.
    ///
    /// # Returns
    ///
    /// The aspect ratio (width / height) of the crop.
    pub fn aspect_ratio(&self) -> f32 {
        let [crop_width, crop_height] = self.crop_size;
        crop_width as f32 / crop_height as f32
    }
}

impl Default for Crop {
    /// Creates a default Crop instance with 224x224 center crop.
    fn default() -> Self {
        Self {
            crop_size: [224, 224],
            crop_mode: CropMode::Center,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ::image::{Rgb, RgbImage};

    fn create_test_image(width: u32, height: u32) -> RgbImage {
        RgbImage::from_pixel(width, height, Rgb([255, 0, 0]))
    }

    #[test]
    fn test_crop_center() -> Result<(), ImageProcessError> {
        let crop = Crop::new([100, 100], CropMode::Center)?;
        let img = create_test_image(200, 200);
        let cropped = crop.process(&img)?;
        assert_eq!(cropped.dimensions(), (100, 100));
        Ok(())
    }

    #[test]
    fn test_crop_top_left() -> Result<(), ImageProcessError> {
        let crop = Crop::new([100, 100], CropMode::TopLeft)?;
        let img = create_test_image(200, 200);
        let cropped = crop.process(&img)?;
        assert_eq!(cropped.dimensions(), (100, 100));
        Ok(())
    }

    #[test]
    fn test_crop_custom() -> Result<(), ImageProcessError> {
        let crop = Crop::new([100, 100], CropMode::Custom { x: 50, y: 50 })?;
        let img = create_test_image(200, 200);
        let cropped = crop.process(&img)?;
        assert_eq!(cropped.dimensions(), (100, 100));

        // Test out of bounds custom position
        let crop_oob = Crop::new([100, 100], CropMode::Custom { x: 150, y: 150 })?;
        assert!(crop_oob.process(&img).is_err());
        Ok(())
    }

    #[test]
    fn test_crop_size_too_large() -> Result<(), ImageProcessError> {
        let crop = Crop::new([300, 300], CropMode::Center)?;
        let img = create_test_image(200, 200);
        assert!(crop.process(&img).is_err());
        Ok(())
    }

    #[test]
    fn test_crop_same_size() -> Result<(), ImageProcessError> {
        let crop = Crop::new([200, 200], CropMode::Center)?;
        let img = create_test_image(200, 200);
        let cropped = crop.process(&img)?;
        assert_eq!(cropped.dimensions(), (200, 200));
        Ok(())
    }

    #[test]
    fn test_can_crop() -> Result<(), ImageProcessError> {
        let crop = Crop::new([100, 100], CropMode::Center)?;
        assert!(crop.can_crop(200, 200));
        assert!(crop.can_crop(100, 100));
        assert!(!crop.can_crop(50, 50));
        Ok(())
    }

    #[test]
    fn test_aspect_ratio() -> Result<(), ImageProcessError> {
        let crop = Crop::new([200, 100], CropMode::Center)?;
        assert_eq!(crop.aspect_ratio(), 2.0);
        Ok(())
    }

    #[test]
    fn test_process_batch() -> Result<(), ImageProcessError> {
        let crop = Crop::new([100, 100], CropMode::Center)?;
        let images = vec![create_test_image(200, 200), create_test_image(300, 300)];
        let cropped = crop.process_batch(&images)?;
        assert_eq!(cropped.len(), 2);
        assert_eq!(cropped[0].dimensions(), (100, 100));
        assert_eq!(cropped[1].dimensions(), (100, 100));
        Ok(())
    }
}
