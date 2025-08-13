//! Dynamic batch processing logic

use super::config::{DynamicBatchConfig, PaddingStrategy, ShapeCompatibilityStrategy};
use super::types::{CompatibleBatch, CrossImageBatch, CrossImageItem};
use crate::core::OCRError;
use crate::core::traits::StandardPredictor;
use image::{ImageBuffer, Rgb, RgbImage};
use std::collections::HashMap;
use std::time::Instant;

/// Enhanced trait for dynamic batching functionality
pub trait DynamicBatcher {
    /// Group images by compatible shapes for batching
    fn group_images_by_compatibility(
        &self,
        images: Vec<(usize, RgbImage)>,
        config: &DynamicBatchConfig,
    ) -> Result<Vec<CompatibleBatch>, OCRError>;

    /// Group cross-image items (e.g., text regions from multiple images)
    fn group_cross_image_items(
        &self,
        items: Vec<(usize, usize, RgbImage)>, // (source_image_idx, item_idx, image)
        config: &DynamicBatchConfig,
    ) -> Result<Vec<CrossImageBatch>, OCRError>;

    /// Batch predict with a predictor
    fn batch_predict<P>(
        &self,
        predictor: &P,
        images: Vec<RgbImage>,
        config: Option<P::Config>,
    ) -> Result<Vec<P::Result>, OCRError>
    where
        P: StandardPredictor;
}

/// Default implementation of dynamic batcher
#[derive(Debug)]
pub struct DefaultDynamicBatcher;

impl DefaultDynamicBatcher {
    /// Create a new default dynamic batcher
    pub fn new() -> Self {
        Self
    }

    /// Calculate aspect ratio of an image
    fn calculate_aspect_ratio(image: &RgbImage) -> f32 {
        let (width, height) = image.dimensions();
        width as f32 / height as f32
    }

    /// Check if two images are compatible based on strategy
    fn are_images_compatible(
        img1: &RgbImage,
        img2: &RgbImage,
        strategy: &ShapeCompatibilityStrategy,
    ) -> bool {
        match strategy {
            ShapeCompatibilityStrategy::Exact => img1.dimensions() == img2.dimensions(),
            ShapeCompatibilityStrategy::AspectRatio { tolerance } => {
                let ratio1 = Self::calculate_aspect_ratio(img1);
                let ratio2 = Self::calculate_aspect_ratio(img2);
                (ratio1 - ratio2).abs() <= *tolerance
            }
            ShapeCompatibilityStrategy::MaxDimension { bucket_size } => {
                let (w1, h1) = img1.dimensions();
                let (w2, h2) = img2.dimensions();
                let max1 = w1.max(h1);
                let max2 = w2.max(h2);
                max1 / bucket_size == max2 / bucket_size
            }
            ShapeCompatibilityStrategy::Custom { targets, tolerance } => {
                // Find the best target for each image and check if they match
                let target1 = Self::find_best_target(img1, targets, *tolerance);
                let target2 = Self::find_best_target(img2, targets, *tolerance);
                target1 == target2
            }
        }
    }

    /// Find the best target dimensions for an image
    fn find_best_target(
        image: &RgbImage,
        targets: &[(u32, u32)],
        tolerance: f32,
    ) -> Option<(u32, u32)> {
        let (width, height) = image.dimensions();
        let aspect_ratio = width as f32 / height as f32;

        targets
            .iter()
            .find(|(target_w, target_h)| {
                let target_ratio = *target_w as f32 / *target_h as f32;
                (aspect_ratio - target_ratio).abs() <= tolerance
            })
            .copied()
    }

    /// Calculate target dimensions for a batch
    fn calculate_target_dimensions(
        images: &[RgbImage],
        strategy: &ShapeCompatibilityStrategy,
    ) -> (u32, u32) {
        match strategy {
            ShapeCompatibilityStrategy::Exact => {
                // All images should have the same dimensions
                images.first().map(|img| img.dimensions()).unwrap_or((0, 0))
            }
            _ => {
                // Calculate the maximum dimensions
                let max_width = images.iter().map(|img| img.width()).max().unwrap_or(0);
                let max_height = images.iter().map(|img| img.height()).max().unwrap_or(0);
                (max_width, max_height)
            }
        }
    }

    /// Pad an image to target dimensions
    fn pad_image(
        image: &RgbImage,
        target_dims: (u32, u32),
        strategy: &PaddingStrategy,
    ) -> Result<RgbImage, OCRError> {
        let (current_width, current_height) = image.dimensions();
        let (target_width, target_height) = target_dims;

        if current_width == target_width && current_height == target_height {
            return Ok(image.clone());
        }

        if current_width > target_width || current_height > target_height {
            return Err(OCRError::Processing {
                kind: crate::core::ProcessingStage::ImageProcessing,
                context: format!(
                    "Image dimensions ({}, {}) exceed target dimensions ({}, {})",
                    current_width, current_height, target_width, target_height
                ),
                source: Box::new(crate::core::errors::SimpleError::new("Image too large")),
            });
        }

        let mut padded = ImageBuffer::new(target_width, target_height);

        match strategy {
            PaddingStrategy::Zero => {
                // Fill with zeros (black)
                for pixel in padded.pixels_mut() {
                    *pixel = Rgb([0, 0, 0]);
                }
            }
            PaddingStrategy::Center { fill_color } => {
                // Fill with specified color
                for pixel in padded.pixels_mut() {
                    *pixel = Rgb(*fill_color);
                }
            }
            PaddingStrategy::Edge => {
                // Fill with edge pixels (not implemented in this simple version)
                for pixel in padded.pixels_mut() {
                    *pixel = Rgb([128, 128, 128]); // Gray as placeholder
                }
            }
            PaddingStrategy::Smart => {
                // Smart padding (not implemented in this simple version)
                for pixel in padded.pixels_mut() {
                    *pixel = Rgb([64, 64, 64]); // Dark gray as placeholder
                }
            }
        }

        // Copy the original image to the center of the padded image
        let x_offset = (target_width - current_width) / 2;
        let y_offset = (target_height - current_height) / 2;

        for y in 0..current_height {
            for x in 0..current_width {
                let pixel = image.get_pixel(x, y);
                padded.put_pixel(x + x_offset, y + y_offset, *pixel);
            }
        }

        Ok(padded)
    }

    /// Generate a batch ID based on target dimensions
    fn generate_batch_id(target_dims: (u32, u32), batch_index: usize) -> String {
        format!("{}x{}_{}", target_dims.0, target_dims.1, batch_index)
    }
}

impl Default for DefaultDynamicBatcher {
    fn default() -> Self {
        Self::new()
    }
}

impl DynamicBatcher for DefaultDynamicBatcher {
    fn group_images_by_compatibility(
        &self,
        images: Vec<(usize, RgbImage)>,
        config: &DynamicBatchConfig,
    ) -> Result<Vec<CompatibleBatch>, OCRError> {
        let _start_time = Instant::now();
        let mut batches = Vec::new();
        let mut batch_counter = 0;

        // Group images by compatibility
        let mut compatibility_groups: HashMap<String, Vec<(usize, RgbImage)>> = HashMap::new();

        for (index, image) in images {
            let mut target_group_key = None;

            // Try to find a compatible group
            for (group_key, group_images) in compatibility_groups.iter() {
                if let Some((_, first_image)) = group_images.first()
                    && Self::are_images_compatible(&image, first_image, &config.shape_compatibility)
                {
                    target_group_key = Some(group_key.clone());
                    break;
                }
            }

            // Add to the compatible group or create a new one
            if let Some(group_key) = target_group_key {
                compatibility_groups
                    .get_mut(&group_key)
                    .unwrap()
                    .push((index, image));
            } else {
                let group_key = format!("group_{}", compatibility_groups.len());
                compatibility_groups.insert(group_key, vec![(index, image)]);
            }
        }

        // Convert groups to batches
        for (_, group_images) in compatibility_groups {
            if group_images.len() < config.min_batch_size {
                // Process small groups as individual batches
                for (index, image) in group_images {
                    let target_dims = image.dimensions();
                    let batch_id = Self::generate_batch_id(target_dims, batch_counter);
                    let mut batch = CompatibleBatch::new(batch_id, target_dims);
                    batch.add_image(image, index);
                    batches.push(batch);
                    batch_counter += 1;
                }
            } else {
                // Split large groups into appropriately sized batches
                let max_batch_size = config.max_detection_batch_size;
                let images_vec: Vec<RgbImage> =
                    group_images.iter().map(|(_, img)| img.clone()).collect();
                let target_dims =
                    Self::calculate_target_dimensions(&images_vec, &config.shape_compatibility);

                for chunk in group_images.chunks(max_batch_size) {
                    let batch_id = Self::generate_batch_id(target_dims, batch_counter);
                    let mut batch = CompatibleBatch::new(batch_id, target_dims);

                    for (index, image) in chunk {
                        // Pad image to target dimensions if needed
                        let padded_image =
                            Self::pad_image(image, target_dims, &config.padding_strategy)?;
                        batch.add_image(padded_image, *index);
                    }

                    batches.push(batch);
                    batch_counter += 1;
                }
            }
        }

        Ok(batches)
    }

    fn group_cross_image_items(
        &self,
        items: Vec<(usize, usize, RgbImage)>,
        config: &DynamicBatchConfig,
    ) -> Result<Vec<CrossImageBatch>, OCRError> {
        let mut batches = Vec::new();
        let mut batch_counter = 0;

        // Convert to CrossImageItem
        let cross_items: Vec<CrossImageItem> = items
            .into_iter()
            .map(|(source_idx, item_idx, image)| CrossImageItem::new(source_idx, item_idx, image))
            .collect();

        // Group by compatibility
        let mut compatibility_groups: HashMap<String, Vec<CrossImageItem>> = HashMap::new();

        for item in cross_items {
            let mut target_group_key = None;

            // Try to find a compatible group
            for (group_key, group_items) in compatibility_groups.iter() {
                if let Some(first_item) = group_items.first()
                    && Self::are_images_compatible(
                        &item.image,
                        &first_item.image,
                        &config.shape_compatibility,
                    )
                {
                    target_group_key = Some(group_key.clone());
                    break;
                }
            }

            // Add to the compatible group or create a new one
            if let Some(group_key) = target_group_key {
                compatibility_groups.get_mut(&group_key).unwrap().push(item);
            } else {
                let group_key = format!("cross_group_{}", compatibility_groups.len());
                compatibility_groups.insert(group_key, vec![item]);
            }
        }

        // Convert groups to batches
        for (_, group_items) in compatibility_groups {
            if group_items.len() < config.min_batch_size {
                // Process small groups individually
                for item in group_items {
                    let target_dims = item.dimensions();
                    let batch_id = Self::generate_batch_id(target_dims, batch_counter);
                    let mut batch = CrossImageBatch::new(batch_id, target_dims);
                    batch.add_item(item);
                    batches.push(batch);
                    batch_counter += 1;
                }
            } else {
                // Split large groups into appropriately sized batches
                let max_batch_size = config.max_recognition_batch_size;
                let images_vec: Vec<RgbImage> =
                    group_items.iter().map(|item| item.image.clone()).collect();
                let target_dims =
                    Self::calculate_target_dimensions(&images_vec, &config.shape_compatibility);

                for chunk in group_items.chunks(max_batch_size) {
                    let batch_id = Self::generate_batch_id(target_dims, batch_counter);
                    let mut batch = CrossImageBatch::new(batch_id, target_dims);

                    for item in chunk {
                        // Pad image to target dimensions if needed
                        let padded_image =
                            Self::pad_image(&item.image, target_dims, &config.padding_strategy)?;
                        let mut padded_item = item.clone();
                        padded_item.image = padded_image;
                        batch.add_item(padded_item);
                    }

                    batches.push(batch);
                    batch_counter += 1;
                }
            }
        }

        Ok(batches)
    }

    fn batch_predict<P>(
        &self,
        predictor: &P,
        images: Vec<RgbImage>,
        config: Option<P::Config>,
    ) -> Result<Vec<P::Result>, OCRError>
    where
        P: StandardPredictor,
    {
        // For now, just call the predictor directly and wrap the result in a Vec
        // In a more sophisticated implementation, this could handle
        // batching logic, memory management, etc.
        let result = predictor.predict(images, config)?;
        Ok(vec![result])
    }
}
