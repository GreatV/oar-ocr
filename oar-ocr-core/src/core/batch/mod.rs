//! Batch processing utilities for the OCR pipeline.
//!
//! This module provides structures and functions for handling batched data
//! in the OCR pipeline, including batching of input data, sampling, and
//! tensor operations for batched processing.

pub mod dynamic;

use crate::core::traits::Sampler;
use std::sync::Arc;

/// A 2-dimensional tensor represented as a 2D array of f32 values.
pub type Tensor2D = ndarray::Array2<f32>;

/// A 3-dimensional tensor represented as a 3D array of f32 values.
pub type Tensor3D = ndarray::Array3<f32>;

/// A 4-dimensional tensor represented as a 4D array of f32 values.
pub type Tensor4D = ndarray::Array4<f32>;

/// A 1-dimensional tensor represented as a dynamic-dimensional array of f32 values.
pub type Tensor1D = ndarray::ArrayD<f32>;

/// Data structure for holding batched input data.
///
/// This struct contains the instances, input paths, and indexes for a batch of data.
/// It's used in the OCR pipeline to process multiple inputs together for efficiency.
pub struct BatchData {
    /// The instances in the batch, stored as `Arc<str>` for efficient sharing.
    pub instances: Vec<Arc<str>>,
    /// The input paths for the instances in the batch, stored as `Arc<str>` for efficient sharing.
    pub input_paths: Vec<Arc<str>>,
    /// The indexes of the instances in the original data set.
    pub indexes: Vec<usize>,
}

impl BatchData {
    /// Creates a new BatchData instance from shared `Arc<str>` paths and indexes.
    ///
    /// # Arguments
    ///
    /// * `paths` - A vector of `Arc<str>` representing the paths to the instances.
    /// * `indexes` - A vector of usize representing the indexes of the instances in the original data set.
    ///
    /// # Returns
    ///
    /// A new BatchData instance.
    pub fn from_shared_arc_paths(paths: Vec<Arc<str>>, indexes: Vec<usize>) -> Self {
        let input_paths = paths.clone();
        Self {
            instances: paths,
            input_paths,
            indexes,
        }
    }

    /// Returns the number of instances in the batch.
    ///
    /// # Returns
    ///
    /// The number of instances in the batch.
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    /// Checks if the batch is empty.
    ///
    /// # Returns
    ///
    /// True if the batch is empty, false otherwise.
    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// Returns an iterator over the instances as string slices.
    ///
    /// # Returns
    ///
    /// An iterator over the instances as string slices.
    pub fn instances_as_str(&self) -> impl Iterator<Item = &str> + '_ {
        self.instances.iter().map(|arc| arc.as_ref())
    }

    /// Returns an iterator over the input paths as string slices.
    ///
    /// # Returns
    ///
    /// An iterator over the input paths as string slices.
    pub fn input_paths_as_str(&self) -> impl Iterator<Item = &str> + '_ {
        self.input_paths.iter().map(|arc| arc.as_ref())
    }
}

/// A sampler that creates batches of data with a specified batch size.
///
/// This struct is used to divide data into batches for processing in the OCR pipeline.
/// It implements the Sampler trait for String data.
#[derive(Debug)]
pub struct BatchSampler {
    /// The size of each batch.
    batch_size: usize,
}

impl BatchSampler {
    /// Creates a new BatchSampler with the specified batch size.
    ///
    /// # Arguments
    ///
    /// * `batch_size` - The size of each batch.
    ///
    /// # Returns
    ///
    /// A new BatchSampler instance.
    pub fn new(batch_size: usize) -> Self {
        Self { batch_size }
    }

    /// Returns the batch size.
    ///
    /// # Returns
    ///
    /// The batch size.
    pub fn batch_size(&self) -> usize {
        self.batch_size
    }

    /// Creates an iterator over batches of data.
    ///
    /// # Arguments
    ///
    /// * `data` - A slice of data to be batched.
    ///
    /// # Returns
    ///
    /// An iterator over batches of data.
    pub fn batches<'a, T>(&self, data: &'a [T]) -> impl Iterator<Item = &'a [T]> {
        if self.batch_size == 0 {
            data.chunks(1).take(0)
        } else {
            data.chunks(self.batch_size).take(usize::MAX)
        }
    }

    /// Creates an iterator over batches of data with their indexes.
    ///
    /// # Arguments
    ///
    /// * `data` - A slice of data to be batched.
    ///
    /// # Returns
    ///
    /// An iterator over tuples containing batches of data and their indexes.
    pub fn batches_with_indexes<'a, T>(
        &self,
        data: &'a [T],
    ) -> impl Iterator<Item = (&'a [T], Vec<usize>)> {
        let batch_size = if self.batch_size == 0 {
            1
        } else {
            self.batch_size
        };
        let take_count = if self.batch_size == 0 { 0 } else { usize::MAX };

        data.chunks(batch_size)
            .take(take_count)
            .enumerate()
            .map(move |(batch_idx, chunk)| {
                let start_idx = batch_idx * self.batch_size;
                let indexes: Vec<usize> = (0..chunk.len()).map(|i| start_idx + i).collect();
                (chunk, indexes)
            })
    }

    /// Samples batches of data from a vector of strings.
    ///
    /// # Arguments
    ///
    /// * `data` - A vector of strings to be batched.
    ///
    /// # Returns
    ///
    /// A vector of BatchData instances.
    pub fn sample_batch(&self, data: Vec<String>) -> Vec<BatchData> {
        if self.batch_size == 0 {
            return Vec::new();
        }

        data.chunks(self.batch_size)
            .enumerate()
            .map(|(batch_idx, chunk)| {
                let start_idx = batch_idx * self.batch_size;
                let indexes: Vec<usize> = (0..chunk.len()).map(|i| start_idx + i).collect();

                BatchData::from_shared_arc_paths(
                    chunk.iter().map(|s| Arc::from(s.as_str())).collect(),
                    indexes,
                )
            })
            .collect()
    }
}

impl Sampler<String> for BatchSampler {
    type BatchData = BatchData;

    /// Samples batches of data from a vector of strings.
    ///
    /// This method implements the Sampler trait for String data.
    ///
    /// # Arguments
    ///
    /// * `data` - A vector of strings to be batched.
    ///
    /// # Returns
    ///
    /// A vector of BatchData instances.
    fn sample(&self, data: Vec<String>) -> Vec<Self::BatchData> {
        self.sample_batch(data)
    }
}

/// A struct for converting image data into batched tensor format.
///
/// This struct provides methods for validating input data and converting
/// images into a batched tensor format suitable for processing in the OCR pipeline.
#[derive(Debug, Default)]
pub struct ToBatch;

impl ToBatch {
    /// Creates a new ToBatch instance.
    ///
    /// # Returns
    ///
    /// A new ToBatch instance.
    pub fn new() -> Self {
        ToBatch
    }

    /// Validates the input images and their shapes.
    ///
    /// This method checks that the images and shapes arrays have the same length,
    /// that all images have the correct number of elements for their shapes,
    /// and that all dimensions are greater than zero.
    ///
    /// # Arguments
    ///
    /// * `imgs` - A slice of vectors of f32 values representing the images.
    /// * `shapes` - A slice of tuples representing the shapes of the images (channels, height, width).
    ///
    /// # Returns
    ///
    /// A Result indicating success or an OCRError if validation fails.
    pub fn validate_inputs(
        &self,
        imgs: &[Vec<f32>],
        shapes: &[(usize, usize, usize)],
    ) -> Result<(), crate::core::OCRError> {
        if imgs.is_empty() && shapes.is_empty() {
            return Ok(());
        }

        if imgs.is_empty() {
            return Err(crate::core::OCRError::InvalidInput {
                message: "Images array is empty but shapes array is not".to_string(),
            });
        }

        if shapes.is_empty() {
            return Err(crate::core::OCRError::InvalidInput {
                message: "Shapes array is empty but images array is not".to_string(),
            });
        }

        if imgs.len() != shapes.len() {
            return Err(crate::core::OCRError::InvalidInput {
                message: format!(
                    "Images and shapes must have the same length: got {} images and {} shapes",
                    imgs.len(),
                    shapes.len()
                ),
            });
        }

        for (i, (img, &(c, h, w))) in imgs.iter().zip(shapes.iter()).enumerate() {
            let expected_len = c * h * w;
            if img.len() != expected_len {
                return Err(crate::core::OCRError::InvalidInput {
                    message: format!(
                        "Image {} has {} elements but shape ({}, {}, {}) requires {}",
                        i,
                        img.len(),
                        c,
                        h,
                        w,
                        expected_len
                    ),
                });
            }

            if c == 0 || h == 0 || w == 0 {
                return Err(crate::core::OCRError::InvalidInput {
                    message: format!(
                        "Image {i} has invalid shape dimensions ({c}, {h}, {w}): all must be greater than 0"
                    ),
                });
            }

            if expected_len > crate::core::constants::MAX_TENSOR_SIZE {
                return Err(crate::core::OCRError::InvalidInput {
                    message: format!(
                        "Image {} tensor size {} exceeds maximum allowed size {}",
                        i,
                        expected_len,
                        crate::core::constants::MAX_TENSOR_SIZE
                    ),
                });
            }
        }

        Ok(())
    }

    /// Applies the batch conversion to the input images and shapes.
    ///
    /// This method validates the inputs, then converts the images into a batched tensor format.
    /// If all images have the same dimensions, it uses a more efficient contiguous copying method.
    /// Otherwise, it uses a method that handles mixed dimensions.
    ///
    /// # Arguments
    ///
    /// * `imgs` - A slice of vectors of f32 values representing the images.
    /// * `shapes` - A slice of tuples representing the shapes of the images (channels, height, width).
    ///
    /// # Returns
    ///
    /// A Result containing a vector of f32 values representing the batched tensor,
    /// or an OCRError if the operation fails.
    pub fn apply(
        &self,
        imgs: &[Vec<f32>],
        shapes: &[(usize, usize, usize)],
    ) -> Result<Vec<f32>, crate::core::OCRError> {
        self.validate_inputs(imgs, shapes)?;

        if imgs.is_empty() {
            return Ok(Vec::new());
        }

        let batch_size = imgs.len();
        let first_shape = shapes.first().copied().unwrap_or((0, 0, 0));
        let channels = first_shape.0;
        let mut max_height = first_shape.1;
        let mut max_width = first_shape.2;
        let mut all_same_dimensions = true;

        for (i, &(c, h, w)) in shapes.iter().enumerate() {
            if c != channels {
                return Err(crate::core::OCRError::InvalidInput {
                    message: format!(
                        "All images must have the same channel count: image 0 has {channels} channels, image {i} has {c} channels"
                    ),
                });
            }

            if h > max_height {
                max_height = h;
            }
            if w > max_width {
                max_width = w;
            }
            if all_same_dimensions && (h != first_shape.1 || w != first_shape.2) {
                all_same_dimensions = false;
            }
        }

        let tensor_size = batch_size * channels * max_height * max_width;
        let mut batch_tensor = vec![0.0; tensor_size];

        if all_same_dimensions {
            self.apply_contiguous(imgs, &mut batch_tensor, channels, max_height, max_width);
        } else {
            self.apply_mixed_dimensions(
                imgs,
                shapes,
                &mut batch_tensor,
                channels,
                max_height,
                max_width,
            );
        }

        Ok(batch_tensor)
    }

    /// Applies contiguous copying for images with the same dimensions.
    ///
    /// This method is used when all images in the batch have the same dimensions,
    /// allowing for more efficient copying.
    ///
    /// # Arguments
    ///
    /// * `imgs` - A slice of vectors of f32 values representing the images.
    /// * `batch_tensor` - A mutable slice of f32 values representing the batched tensor.
    /// * `channels` - The number of channels in the images.
    /// * `height` - The height of the images.
    /// * `width` - The width of the images.
    fn apply_contiguous(
        &self,
        imgs: &[Vec<f32>],
        batch_tensor: &mut [f32],
        channels: usize,
        height: usize,
        width: usize,
    ) {
        let img_size = channels * height * width;

        for (batch_idx, img) in imgs.iter().enumerate() {
            let batch_offset = batch_idx * img_size;
            let dst_slice = &mut batch_tensor[batch_offset..batch_offset + img.len()];

            dst_slice.copy_from_slice(img);
        }
    }

    /// Applies copying for images with mixed dimensions.
    ///
    /// This method is used when images in the batch have different dimensions,
    /// requiring padding to the maximum dimensions.
    ///
    /// # Arguments
    ///
    /// * `imgs` - A slice of vectors of f32 values representing the images.
    /// * `shapes` - A slice of tuples representing the shapes of the images (channels, height, width).
    /// * `batch_tensor` - A mutable slice of f32 values representing the batched tensor.
    /// * `channels` - The number of channels in the images.
    /// * `max_height` - The maximum height among all images in the batch.
    /// * `max_width` - The maximum width among all images in the batch.
    fn apply_mixed_dimensions(
        &self,
        imgs: &[Vec<f32>],
        shapes: &[(usize, usize, usize)],
        batch_tensor: &mut [f32],
        channels: usize,
        max_height: usize,
        max_width: usize,
    ) {
        for (batch_idx, (img, &(c, h, w))) in imgs.iter().zip(shapes.iter()).enumerate() {
            let batch_base = batch_idx * channels * max_height * max_width;

            for ch in 0..c {
                let src_channel_start = ch * h * w;
                let dst_channel_start = batch_base + ch * max_height * max_width;

                for y in 0..h {
                    let src_row_start = src_channel_start + y * w;
                    let dst_row_start = dst_channel_start + y * max_width;

                    let src_row = &img[src_row_start..src_row_start + w];
                    let dst_row = &mut batch_tensor[dst_row_start..dst_row_start + w];
                    dst_row.copy_from_slice(src_row);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::core::OCRError;

    #[test]
    fn test_to_batch_apply_contiguous() -> Result<(), OCRError> {
        let to_batch = ToBatch::new();

        // Create test images with same dimensions
        let img1 = vec![1.0, 2.0, 3.0, 4.0]; // 1x2x2 image
        let img2 = vec![5.0, 6.0, 7.0, 8.0]; // 1x2x2 image
        let imgs = vec![img1, img2];
        let shapes = vec![(1, 2, 2), (1, 2, 2)]; // Same shapes

        let result = to_batch.apply(&imgs, &shapes)?;

        // Expected: batch_size=2, channels=1, height=2, width=2
        // Total size: 2 * 1 * 2 * 2 = 8
        assert_eq!(result.len(), 8);

        // First image should be at positions 0-3
        assert_eq!(result[0], 1.0);
        assert_eq!(result[1], 2.0);
        assert_eq!(result[2], 3.0);
        assert_eq!(result[3], 4.0);

        // Second image should be at positions 4-7
        assert_eq!(result[4], 5.0);
        assert_eq!(result[5], 6.0);
        assert_eq!(result[6], 7.0);
        assert_eq!(result[7], 8.0);
        Ok(())
    }

    #[test]
    fn test_to_batch_apply_mixed_dimensions() -> Result<(), OCRError> {
        let to_batch = ToBatch::new();

        // Create test images with different dimensions
        let img1 = vec![1.0, 2.0]; // 1x1x2 image
        let img2 = vec![3.0, 4.0, 5.0, 6.0]; // 1x2x2 image
        let imgs = vec![img1, img2];
        let shapes = vec![(1, 1, 2), (1, 2, 2)]; // Different shapes

        let result = to_batch.apply(&imgs, &shapes)?;

        // Expected: batch_size=2, channels=1, max_height=2, max_width=2
        // Total size: 2 * 1 * 2 * 2 = 8
        assert_eq!(result.len(), 8);

        // First image (1x2) should be padded to (2x2)
        // Second image (2x2) should fit exactly
        // The exact layout depends on the mixed dimensions implementation
        assert!(result.contains(&1.0));
        assert!(result.contains(&2.0));
        assert!(result.contains(&3.0));
        assert!(result.contains(&4.0));
        assert!(result.contains(&5.0));
        assert!(result.contains(&6.0));
        Ok(())
    }
}
