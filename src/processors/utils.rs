//! Utility functions for image processing operations.
//!
//! This module provides various utility functions for image processing, including:
//! - Image cropping operations with different modes
//! - Top-k classification result processing
//! - Document transformation post-processing
//!
//! The module is organized into several components:
//! - `image_utils`: Helper functions for basic image operations
//! - `Crop`: Struct for handling image cropping with different modes
//! - `Topk`: Struct for processing classification results
//! - `DocTrPostProcess`: Struct for document transformation post-processing

use crate::processors::types::{CropMode, ImageProcessError};
use image::{GrayImage, RgbImage};
use std::collections::HashMap;
use std::str::FromStr;

/// Utility functions for basic image operations.
pub mod image_utils {
    use super::ImageProcessError;

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
    /// A new image buffer containing the cropped region.
    ///
    /// # Type Parameters
    ///
    /// * `P` - Pixel type that implements the `image::Pixel` trait.
    /// * `Container` - Container type for pixel data.
    pub fn slice_image<P, Container>(
        img: &image::ImageBuffer<P, Container>,
        coords: (u32, u32, u32, u32),
    ) -> image::ImageBuffer<P, Vec<P::Subpixel>>
    where
        P: image::Pixel + 'static,
        P::Subpixel: 'static + Clone,
        Container: std::ops::Deref<Target = [P::Subpixel]>,
    {
        let (x1, y1, x2, y2) = coords;
        let width = x2 - x1;
        let height = y2 - y1;

        let mut cropped = image::ImageBuffer::new(width, height);

        // Copy pixels from the source image to the cropped image
        for y in 0..height {
            for x in 0..width {
                let src_x = x1 + x;
                let src_y = y1 + y;
                // Ensure we don't access pixels outside the source image bounds
                if src_x < img.width() && src_y < img.height() {
                    let pixel = img.get_pixel(src_x, src_y);
                    cropped.put_pixel(x, y, *pixel);
                }
            }
        }

        cropped
    }

    /// Slices a region from an RGB image.
    ///
    /// A convenience function that wraps `slice_image` for RGB images.
    ///
    /// # Arguments
    ///
    /// * `img` - Reference to the source RGB image.
    /// * `coords` - Tuple containing (x1, y1, x2, y2) coordinates defining the crop region.
    ///
    /// # Returns
    ///
    /// A new RGB image containing the cropped region.
    pub fn slice_rgb_image(img: &image::RgbImage, coords: (u32, u32, u32, u32)) -> image::RgbImage {
        slice_image(img, coords)
    }

    /// Slices a region from a grayscale image.
    ///
    /// A convenience function that wraps `slice_image` for grayscale images.
    ///
    /// # Arguments
    ///
    /// * `img` - Reference to the source grayscale image.
    /// * `coords` - Tuple containing (x1, y1, x2, y2) coordinates defining the crop region.
    ///
    /// # Returns
    ///
    /// A new grayscale image containing the cropped region.
    pub fn slice_gray_image(
        img: &image::GrayImage,
        coords: (u32, u32, u32, u32),
    ) -> image::GrayImage {
        slice_image(img, coords)
    }
}

/// Configuration for image cropping operations.
///
/// This struct holds the parameters needed for cropping images, including
/// the desired crop size and the cropping mode (center or top-left).
#[derive(Debug)]
pub struct Crop {
    /// The dimensions [width, height] for the crop operation.
    crop_size: [u32; 2],
    /// The mode determining how the crop region is positioned.
    mode: CropMode,
}

impl Crop {
    /// Creates a new Crop instance with the specified parameters.
    ///
    /// # Arguments
    ///
    /// * `crop_size` - Slice containing the crop dimensions. Can be:
    ///   - 1 element: square crop of size crop_size[0] x crop_size[0]
    ///   - 2 elements: rectangular crop of size crop_size[0] x crop_size[1]
    /// * `mode` - String representation of the crop mode ("center" or "topleft").
    ///
    /// # Returns
    ///
    /// * `Ok(Crop)` if the parameters are valid.
    /// * `Err(ImageProcessError)` if the parameters are invalid.
    pub fn new(crop_size: &[u32], mode: &str) -> Result<Self, ImageProcessError> {
        let crop_size = match crop_size.len() {
            1 => [crop_size[0], crop_size[0]],
            2 => [crop_size[0], crop_size[1]],
            _ => return Err(ImageProcessError::InvalidCropSize),
        };

        image_utils::check_image_size(&crop_size)?;
        let mode = CropMode::from_str(mode)?;

        Ok(Self { crop_size, mode })
    }

    /// Applies cropping to a batch of RGB images.
    ///
    /// # Arguments
    ///
    /// * `imgs` - Slice of RGB images to crop.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<RgbImage>)` containing the cropped images.
    /// * `Err(ImageProcessError)` if any image is too small for the crop size.
    pub fn apply_rgb(&self, imgs: &[RgbImage]) -> Result<Vec<RgbImage>, ImageProcessError> {
        imgs.iter().map(|img| self.crop_rgb(img)).collect()
    }

    /// Applies cropping to a batch of grayscale images.
    ///
    /// # Arguments
    ///
    /// * `imgs` - Slice of grayscale images to crop.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<GrayImage>)` containing the cropped images.
    /// * `Err(ImageProcessError)` if any image is too small for the crop size.
    pub fn apply_gray(&self, imgs: &[GrayImage]) -> Result<Vec<GrayImage>, ImageProcessError> {
        imgs.iter().map(|img| self.crop_gray(img)).collect()
    }

    /// Crops a single RGB image according to the configured parameters.
    ///
    /// # Arguments
    ///
    /// * `img` - Reference to the RGB image to crop.
    ///
    /// # Returns
    ///
    /// * `Ok(RgbImage)` containing the cropped image.
    /// * `Err(ImageProcessError::ImageTooSmall)` if the image is smaller than the crop size.
    pub fn crop_rgb(&self, img: &RgbImage) -> Result<RgbImage, ImageProcessError> {
        let (w, h) = (img.width(), img.height());
        let [cw, ch] = self.crop_size;

        // Check if the image is large enough for the crop
        if w < cw || h < ch {
            return Err(ImageProcessError::ImageTooSmall {
                image_size: (w, h),
                crop_size: (cw, ch),
            });
        }

        // Calculate crop coordinates based on the mode
        let (x1, y1) = match self.mode {
            CropMode::Center => {
                let x1 = (w.saturating_sub(cw)) / 2;
                let y1 = (h.saturating_sub(ch)) / 2;
                (x1, y1)
            }
            CropMode::TopLeft => (0, 0),
        };

        // Ensure coordinates don't exceed image boundaries
        let x2 = (x1 + cw).min(w);
        let y2 = (y1 + ch).min(h);
        let coords = (x1, y1, x2, y2);

        Ok(image_utils::slice_rgb_image(img, coords))
    }

    /// Crops a single grayscale image according to the configured parameters.
    ///
    /// # Arguments
    ///
    /// * `img` - Reference to the grayscale image to crop.
    ///
    /// # Returns
    ///
    /// * `Ok(GrayImage)` containing the cropped image.
    /// * `Err(ImageProcessError::ImageTooSmall)` if the image is smaller than the crop size.
    pub fn crop_gray(&self, img: &GrayImage) -> Result<GrayImage, ImageProcessError> {
        let (w, h) = (img.width(), img.height());
        let [cw, ch] = self.crop_size;

        // Check if the image is large enough for the crop
        if w < cw || h < ch {
            return Err(ImageProcessError::ImageTooSmall {
                image_size: (w, h),
                crop_size: (cw, ch),
            });
        }

        // Calculate crop coordinates based on the mode
        let (x1, y1) = match self.mode {
            CropMode::Center => {
                let x1 = (w.saturating_sub(cw)) / 2;
                let y1 = (h.saturating_sub(ch)) / 2;
                (x1, y1)
            }
            CropMode::TopLeft => (0, 0),
        };

        // Ensure coordinates don't exceed image boundaries
        let x2 = (x1 + cw).min(w);
        let y2 = (y1 + ch).min(h);
        let coords = (x1, y1, x2, y2);

        Ok(image_utils::slice_gray_image(img, coords))
    }
}

/// Results from a top-k classification operation.
///
/// Contains the top-k predictions for one or more classification tasks,
/// including class indexes, confidence scores, and optionally class names.
#[derive(Debug, Clone)]
pub struct TopkResult {
    /// Vector of vectors containing the class indexes for each prediction.
    pub indexes: Vec<Vec<usize>>,
    /// Vector of vectors containing the confidence scores for each prediction.
    pub scores: Vec<Vec<f32>>,
    /// Vector of vectors containing the class names for each prediction (if available).
    pub label_names: Vec<Vec<String>>,
}

/// Processes classification predictions to extract top-k results.
///
/// This struct handles the post-processing of classification model outputs
/// to extract the top-k most confident predictions for each sample.
#[derive(Debug)]
pub struct Topk {
    /// Optional mapping from class IDs to class names.
    class_id_map: Option<HashMap<usize, String>>,
}

impl Topk {
    /// Creates a new Topk processor with optional class name mapping.
    ///
    /// # Arguments
    ///
    /// * `class_ids` - Optional slice of strings representing class names.
    ///   If provided, the index in this slice corresponds to the class ID.
    ///
    /// # Returns
    ///
    /// A new Topk instance.
    pub fn new(class_ids: Option<&[String]>) -> Self {
        let class_id_map = Self::parse_class_id_map(class_ids);
        Self { class_id_map }
    }

    /// Parses class ID mappings from a slice of class names.
    ///
    /// # Arguments
    ///
    /// * `class_ids` - Optional slice of strings representing class names.
    ///
    /// # Returns
    ///
    /// * `Some(HashMap<usize, String>)` mapping class IDs to names if input is provided.
    /// * `None` if no class names are provided.
    fn parse_class_id_map(class_ids: Option<&[String]>) -> Option<HashMap<usize, String>> {
        class_ids.map(|ids| {
            ids.iter()
                .enumerate()
                .map(|(id, label)| (id, label.clone()))
                .collect()
        })
    }

    /// Applies top-k processing to a batch of predictions.
    ///
    /// # Arguments
    ///
    /// * `preds` - 2D tensor of predictions where each row represents a sample.
    /// * `topk` - Number of top predictions to extract for each sample.
    ///
    /// # Returns
    ///
    /// A TopkResult containing the top-k predictions for each sample.
    pub fn apply(&self, preds: &crate::core::Tensor2D, topk: usize) -> TopkResult {
        let mut indexes = Vec::new();
        let mut scores = Vec::new();
        let mut label_names = Vec::new();

        // Process each prediction sample
        for pred in preds.outer_iter() {
            // Create indexed pairs of (class_id, score)
            let mut indexed_preds: Vec<(usize, f32)> = pred
                .iter()
                .enumerate()
                .map(|(i, &score)| (i, score))
                .collect();

            // Sort by score in descending order
            indexed_preds
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Extract top-k indexes and scores
            let topk_indices: Vec<usize> = indexed_preds
                .iter()
                .take(topk)
                .map(|(idx, _)| *idx)
                .collect();

            let topk_scores: Vec<f32> = indexed_preds
                .iter()
                .take(topk)
                .map(|(_, score)| *score)
                .collect();

            // Map indexes to class names if available
            let topk_labels: Vec<String> = if let Some(ref class_map) = self.class_id_map {
                topk_indices
                    .iter()
                    .map(|&idx| {
                        class_map
                            .get(&idx)
                            .cloned()
                            .unwrap_or_else(|| idx.to_string())
                    })
                    .collect()
            } else {
                topk_indices.iter().map(|idx| idx.to_string()).collect()
            };

            indexes.push(topk_indices);
            scores.push(topk_scores);
            label_names.push(topk_labels);
        }

        TopkResult {
            indexes,
            scores,
            label_names,
        }
    }

    /// Applies top-k processing to a single prediction vector.
    ///
    /// # Arguments
    ///
    /// * `pred` - Slice of f32 values representing the prediction scores.
    /// * `topk` - Number of top predictions to extract.
    ///
    /// # Returns
    ///
    /// * `Ok(TopkResult)` containing the top-k predictions.
    /// * `Err(ImageProcessError::InvalidInput)` if the input cannot be converted to a tensor.
    pub fn apply_single(
        &self,
        pred: &[f32],
        topk: usize,
    ) -> Result<TopkResult, crate::processors::ImageProcessError> {
        let tensor = ndarray::Array2::from_shape_vec((1, pred.len()), pred.to_vec())
            .map_err(|_| crate::processors::ImageProcessError::InvalidInput)?;
        Ok(self.apply(&tensor, topk))
    }
}

/// Default scale factor for document transformation post-processing.
const DEFAULT_DOCTR_SCALE: f32 = 255.0;

/// Post-processor for document transformation model outputs.
///
/// This struct handles the conversion of document transformation model outputs
/// (typically normalized tensors) back to RGB images.
#[derive(Debug)]
pub struct DocTrPostProcess {
    /// Scale factor to convert normalized values back to pixel values.
    pub scale: f32,
}

impl DocTrPostProcess {
    /// Creates a new DocTrPostProcess instance.
    ///
    /// # Arguments
    ///
    /// * `scale` - Optional scale factor. If None, uses DEFAULT_DOCTR_SCALE.
    ///
    /// # Returns
    ///
    /// A new DocTrPostProcess instance.
    pub fn new(scale: Option<f32>) -> Self {
        Self {
            scale: scale.unwrap_or(DEFAULT_DOCTR_SCALE),
        }
    }

    /// Applies document transformation post-processing to a batch of predictions.
    ///
    /// # Arguments
    ///
    /// * `batch_preds` - 4D tensor containing batched predictions.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<RgbImage>)` containing the processed images.
    /// * `Err(ImageProcessError)` if any prediction is invalid.
    pub fn apply_batch(
        &self,
        batch_preds: &ndarray::Array4<f32>,
    ) -> Result<Vec<RgbImage>, ImageProcessError> {
        batch_preds
            .outer_iter()
            .map(|pred_view| {
                let tensor = pred_view.to_owned();
                self.doctr(&tensor)
            })
            .collect()
    }

    /// Converts a document transformation prediction tensor to an RGB image.
    ///
    /// # Arguments
    ///
    /// * `pred` - 3D tensor representing the document transformation output.
    ///   Expected shape: (channels, height, width) with 3 channels (RGB).
    ///
    /// # Returns
    ///
    /// * `Ok(RgbImage)` containing the processed image.
    /// * `Err(ImageProcessError::InvalidInput)` if the tensor is invalid.
    pub fn doctr(&self, pred: &crate::core::Tensor3D) -> Result<RgbImage, ImageProcessError> {
        // Validate tensor dimensions
        if pred.is_empty() || pred.shape()[0] == 0 {
            return Err(ImageProcessError::InvalidInput);
        }

        let channels = pred.shape()[0];
        let height = pred.shape()[1];
        let width = pred.shape()[2];

        // Ensure we have RGB channels
        if channels != 3 {
            return Err(ImageProcessError::InvalidInput);
        }
        if height == 0 || width == 0 {
            return Err(ImageProcessError::InvalidInput);
        }

        // Create output image
        let mut output_img = RgbImage::new(width as u32, height as u32);

        // Convert tensor values to RGB pixels
        for y in 0..height {
            for x in 0..width {
                // Apply scale factor and clamp values to valid range
                let b = pred[[0, y, x]] * self.scale;
                let g = pred[[1, y, x]] * self.scale;
                let r = pred[[2, y, x]] * self.scale;

                let pixel = image::Rgb([
                    r.clamp(0.0, 255.0) as u8,
                    g.clamp(0.0, 255.0) as u8,
                    b.clamp(0.0, 255.0) as u8,
                ]);

                output_img.put_pixel(x as u32, y as u32, pixel);
            }
        }

        Ok(output_img)
    }

    /// Applies document transformation post-processing to multiple predictions.
    ///
    /// # Arguments
    ///
    /// * `imgs` - Slice of 3D tensors representing document transformation outputs.
    ///
    /// # Returns
    ///
    /// * `Ok(Vec<RgbImage>)` containing the processed images.
    /// * `Err(ImageProcessError)` if any prediction is invalid.
    pub fn apply(
        &self,
        imgs: &[crate::core::Tensor3D],
    ) -> Result<Vec<RgbImage>, ImageProcessError> {
        imgs.iter().map(|img| self.doctr(img)).collect()
    }

    /// Converts a document transformation prediction tensor to an RGB image.
    ///
    /// This is an alias for the `doctr` method.
    ///
    /// # Arguments
    ///
    /// * `pred` - 3D tensor representing the document transformation output.
    ///
    /// # Returns
    ///
    /// * `Ok(RgbImage)` containing the processed image.
    /// * `Err(ImageProcessError::InvalidInput)` if the tensor is invalid.
    pub fn doctr_tuple(&self, pred: &crate::core::Tensor3D) -> Result<RgbImage, ImageProcessError> {
        if pred.is_empty() {
            return Err(ImageProcessError::InvalidInput);
        }

        self.doctr(pred)
    }
}
