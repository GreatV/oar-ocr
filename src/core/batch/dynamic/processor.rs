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

        // Calculate offsets for centering the original image
        let x_offset = (target_width - current_width) / 2;
        let y_offset = (target_height - current_height) / 2;

        match strategy {
            PaddingStrategy::Zero => {
                // Fill with zeros (black)
                for pixel in padded.pixels_mut() {
                    *pixel = Rgb([0, 0, 0]);
                }
                // Copy the original image to the center
                Self::copy_centered_image(&mut padded, image, x_offset, y_offset);
            }
            PaddingStrategy::Center { fill_color } => {
                // Fill with specified color
                for pixel in padded.pixels_mut() {
                    *pixel = Rgb(*fill_color);
                }
                // Copy the original image to the center
                Self::copy_centered_image(&mut padded, image, x_offset, y_offset);
            }
            PaddingStrategy::Edge => {
                // Edge padding: directly compute all pixels with edge replication
                Self::apply_optimized_edge_padding(&mut padded, image, x_offset, y_offset);
            }
            PaddingStrategy::Smart => {
                // Smart padding: content-aware padding based on image analysis
                let smart_color = Self::calculate_smart_padding_color(image);
                for pixel in padded.pixels_mut() {
                    *pixel = smart_color;
                }
                // Copy the original image to the center
                Self::copy_centered_image(&mut padded, image, x_offset, y_offset);
            }
        }

        Ok(padded)
    }

    /// Copy the original image to the center of the padded image
    fn copy_centered_image(
        padded: &mut RgbImage,
        original: &RgbImage,
        x_offset: u32,
        y_offset: u32,
    ) {
        let (orig_width, orig_height) = original.dimensions();
        for y in 0..orig_height {
            for x in 0..orig_width {
                let pixel = original.get_pixel(x, y);
                padded.put_pixel(x + x_offset, y + y_offset, *pixel);
            }
        }
    }

    /// Apply optimized edge padding by directly computing all pixels
    fn apply_optimized_edge_padding(
        padded: &mut RgbImage,
        original: &RgbImage,
        x_offset: u32,
        y_offset: u32,
    ) {
        let (padded_width, padded_height) = padded.dimensions();
        let (orig_width, orig_height) = original.dimensions();

        // Fill the entire padded image with edge pixel replication
        for y in 0..padded_height {
            for x in 0..padded_width {
                // Determine source coordinates with edge replication
                let source_x = if x < x_offset {
                    // Left padding area - use leftmost column
                    0
                } else if x >= x_offset + orig_width {
                    // Right padding area - use rightmost column
                    orig_width - 1
                } else {
                    // Within original image bounds
                    x - x_offset
                };

                let source_y = if y < y_offset {
                    // Top padding area - use topmost row
                    0
                } else if y >= y_offset + orig_height {
                    // Bottom padding area - use bottommost row
                    orig_height - 1
                } else {
                    // Within original image bounds
                    y - y_offset
                };

                let pixel = original.get_pixel(source_x, source_y);
                padded.put_pixel(x, y, *pixel);
            }
        }
    }

    /// Calculate smart padding color based on image content analysis
    fn calculate_smart_padding_color(image: &RgbImage) -> Rgb<u8> {
        let (width, height) = image.dimensions();

        if width == 0 || height == 0 {
            return Rgb([0, 0, 0]); // Default to black for empty images
        }

        // Sample edge pixels to determine the most appropriate padding color
        let mut edge_pixels = Vec::new();

        // Sample top and bottom edges
        for x in 0..width {
            edge_pixels.push(*image.get_pixel(x, 0)); // Top edge
            if height > 1 {
                edge_pixels.push(*image.get_pixel(x, height - 1)); // Bottom edge
            }
        }

        // Sample left and right edges (excluding corners to avoid double counting)
        for y in 1..height.saturating_sub(1) {
            edge_pixels.push(*image.get_pixel(0, y)); // Left edge
            if width > 1 {
                edge_pixels.push(*image.get_pixel(width - 1, y)); // Right edge
            }
        }

        if edge_pixels.is_empty() {
            return Rgb([0, 0, 0]);
        }

        // Calculate the median color of edge pixels for robustness against outliers
        let mut r_values: Vec<u8> = edge_pixels.iter().map(|p| p.0[0]).collect();
        let mut g_values: Vec<u8> = edge_pixels.iter().map(|p| p.0[1]).collect();
        let mut b_values: Vec<u8> = edge_pixels.iter().map(|p| p.0[2]).collect();

        r_values.sort_unstable();
        g_values.sort_unstable();
        b_values.sort_unstable();

        let len = r_values.len();
        let median_r = r_values[len / 2];
        let median_g = g_values[len / 2];
        let median_b = b_values[len / 2];

        // Apply some heuristics to improve the padding color choice
        // If the median color is very bright, slightly darken it to avoid harsh contrast
        // If the median color is very dark, slightly brighten it for better visibility
        let adjusted_r = Self::adjust_padding_component(median_r);
        let adjusted_g = Self::adjust_padding_component(median_g);
        let adjusted_b = Self::adjust_padding_component(median_b);

        Rgb([adjusted_r, adjusted_g, adjusted_b])
    }

    /// Adjust a color component for better padding appearance
    fn adjust_padding_component(component: u8) -> u8 {
        match component {
            // Very dark colors (0-63): brighten slightly
            0..=63 => (component as u16 + 16).min(255) as u8,
            // Very bright colors (192-255): darken slightly
            192..=255 => (component as i16 - 16).max(0) as u8,
            // Medium colors (64-191): use as-is
            _ => component,
        }
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
                if let Some(group) = compatibility_groups.get_mut(&group_key) {
                    group.push((index, image));
                } else {
                    // This should not happen, but handle it defensively
                    let group_key = format!("group_{}", compatibility_groups.len());
                    compatibility_groups.insert(group_key, vec![(index, image)]);
                }
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
                if let Some(group) = compatibility_groups.get_mut(&group_key) {
                    group.push(item);
                } else {
                    // This should not happen, but handle it defensively
                    let group_key = format!("cross_group_{}", compatibility_groups.len());
                    compatibility_groups.insert(group_key, vec![item]);
                }
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

#[cfg(test)]
mod tests {
    use super::*;
    use image::{ImageBuffer, Rgb};

    /// Helper function to create a test image with a specific pattern
    fn create_test_image(width: u32, height: u32, pattern: &str) -> RgbImage {
        let mut image = ImageBuffer::new(width, height);

        match pattern {
            "solid_red" => {
                for pixel in image.pixels_mut() {
                    *pixel = Rgb([255, 0, 0]);
                }
            }
            "gradient" => {
                for (x, y, pixel) in image.enumerate_pixels_mut() {
                    let r = (x * 255 / width.max(1)) as u8;
                    let g = (y * 255 / height.max(1)) as u8;
                    *pixel = Rgb([r, g, 128]);
                }
            }
            "border" => {
                // Create an image with distinct border colors
                for (x, y, pixel) in image.enumerate_pixels_mut() {
                    if x == 0 {
                        *pixel = Rgb([255, 0, 0]); // Red left edge
                    } else if x == width - 1 {
                        *pixel = Rgb([0, 255, 0]); // Green right edge
                    } else if y == 0 {
                        *pixel = Rgb([0, 0, 255]); // Blue top edge
                    } else if y == height - 1 {
                        *pixel = Rgb([255, 255, 0]); // Yellow bottom edge
                    } else {
                        *pixel = Rgb([128, 128, 128]); // Gray center
                    }
                }
            }
            _ => {
                // Default: black image
                for pixel in image.pixels_mut() {
                    *pixel = Rgb([0, 0, 0]);
                }
            }
        }

        image
    }

    #[test]
    fn test_pad_image_zero_strategy() {
        let image = create_test_image(10, 10, "solid_red");
        let strategy = PaddingStrategy::Zero;
        let result = DefaultDynamicBatcher::pad_image(&image, (20, 20), &strategy).unwrap();

        assert_eq!(result.dimensions(), (20, 20));

        // Check that padding areas are black (zero)
        assert_eq!(*result.get_pixel(0, 0), Rgb([0, 0, 0])); // Top-left corner
        assert_eq!(*result.get_pixel(19, 19), Rgb([0, 0, 0])); // Bottom-right corner

        // Check that the original image is centered
        assert_eq!(*result.get_pixel(10, 10), Rgb([255, 0, 0])); // Center of original
    }

    #[test]
    fn test_pad_image_center_strategy() {
        let image = create_test_image(10, 10, "solid_red");
        let strategy = PaddingStrategy::Center {
            fill_color: [0, 255, 0],
        }; // Green padding
        let result = DefaultDynamicBatcher::pad_image(&image, (20, 20), &strategy).unwrap();

        assert_eq!(result.dimensions(), (20, 20));

        // Check that padding areas are green
        assert_eq!(*result.get_pixel(0, 0), Rgb([0, 255, 0])); // Top-left corner
        assert_eq!(*result.get_pixel(19, 19), Rgb([0, 255, 0])); // Bottom-right corner

        // Check that the original image is centered
        assert_eq!(*result.get_pixel(10, 10), Rgb([255, 0, 0])); // Center of original
    }

    #[test]
    fn test_pad_image_edge_strategy() {
        let image = create_test_image(6, 6, "border");
        let strategy = PaddingStrategy::Edge;
        let result = DefaultDynamicBatcher::pad_image(&image, (12, 12), &strategy).unwrap();

        assert_eq!(result.dimensions(), (12, 12));

        // Check edge replication
        // Left padding should replicate the left edge (red)
        assert_eq!(*result.get_pixel(0, 6), Rgb([255, 0, 0])); // Left edge replication

        // Right padding should replicate the right edge (green)
        assert_eq!(*result.get_pixel(11, 6), Rgb([0, 255, 0])); // Right edge replication

        // Top padding should replicate the top edge (blue)
        assert_eq!(*result.get_pixel(6, 0), Rgb([0, 0, 255])); // Top edge replication

        // Bottom padding should replicate the bottom edge (yellow)
        assert_eq!(*result.get_pixel(6, 11), Rgb([255, 255, 0])); // Bottom edge replication

        // Check that the original image content is preserved
        assert_eq!(*result.get_pixel(6, 6), Rgb([128, 128, 128])); // Center of original
    }

    #[test]
    fn test_pad_image_smart_strategy() {
        let image = create_test_image(10, 10, "border");
        let strategy = PaddingStrategy::Smart;
        let result = DefaultDynamicBatcher::pad_image(&image, (20, 20), &strategy).unwrap();

        assert_eq!(result.dimensions(), (20, 20));

        // The smart strategy should calculate a color based on edge analysis
        // We can't predict the exact color, but we can verify it's not the default placeholder
        let padding_pixel = *result.get_pixel(0, 0);
        assert_ne!(padding_pixel, Rgb([64, 64, 64])); // Should not be the old placeholder

        // Check that the original image is centered and preserved
        // The original image is 10x10, centered in 20x20, so it starts at (5,5)
        assert_eq!(*result.get_pixel(10, 10), Rgb([128, 128, 128])); // Center of original (5+5, 5+5)
    }

    #[test]
    fn test_pad_image_no_padding_needed() {
        let image = create_test_image(10, 10, "solid_red");
        let strategy = PaddingStrategy::Zero;
        let result = DefaultDynamicBatcher::pad_image(&image, (10, 10), &strategy).unwrap();

        // Should return a clone of the original image
        assert_eq!(result.dimensions(), (10, 10));
        assert_eq!(*result.get_pixel(5, 5), Rgb([255, 0, 0]));
    }

    #[test]
    fn test_pad_image_error_on_oversized_image() {
        let image = create_test_image(20, 20, "solid_red");
        let strategy = PaddingStrategy::Zero;
        let result = DefaultDynamicBatcher::pad_image(&image, (10, 10), &strategy);

        // Should return an error when trying to pad to smaller dimensions
        assert!(result.is_err());
    }

    #[test]
    fn test_calculate_smart_padding_color() {
        // Test with a uniform color image
        let uniform_image = create_test_image(10, 10, "solid_red");
        let smart_color = DefaultDynamicBatcher::calculate_smart_padding_color(&uniform_image);

        // For a uniform red image, the smart color should be close to red but adjusted
        assert!(smart_color.0[0] > 200); // Should still be predominantly red
        assert!(smart_color.0[1] < 50); // Should have low green
        assert!(smart_color.0[2] < 50); // Should have low blue

        // Test with a gradient image
        let gradient_image = create_test_image(10, 10, "gradient");
        let gradient_smart_color =
            DefaultDynamicBatcher::calculate_smart_padding_color(&gradient_image);

        // Should return a reasonable color (not extreme values)
        assert!(gradient_smart_color.0[0] < 255);
        assert!(gradient_smart_color.0[1] < 255);
        assert!(gradient_smart_color.0[2] < 255);
    }

    #[test]
    fn test_adjust_padding_component() {
        // Test dark color adjustment (should brighten)
        assert!(DefaultDynamicBatcher::adjust_padding_component(30) > 30);

        // Test bright color adjustment (should darken)
        assert!(DefaultDynamicBatcher::adjust_padding_component(220) < 220);

        // Test medium color (should remain unchanged)
        assert_eq!(DefaultDynamicBatcher::adjust_padding_component(128), 128);

        // Test edge cases
        assert_eq!(DefaultDynamicBatcher::adjust_padding_component(0), 16);
        assert_eq!(DefaultDynamicBatcher::adjust_padding_component(255), 239);
    }
}
